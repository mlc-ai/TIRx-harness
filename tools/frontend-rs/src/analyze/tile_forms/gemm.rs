//! gemm (mma.sync.m16n8k*) and gemm_async.

use tvm::analysis::Analyzer;
use tvm::ir::PrimExpr;
use tvm::tirx::{TileLayout, TilePrimitiveCallObj};
use tvm::tvm_ffi::ObjectRefCast;

use super::super::layout::{expr_any, int_any, op_binary};
use super::super::util::{ffi_error, oref, repr_text, unsupported, AResult, Failure};
use super::super::Ctx;
use super::coordinates::{
    axes_are, physical_axes, physical_get, physical_layout_coordinates, prove_equal, simplify_val,
    vexpr, vint, vsub,
};
use super::parse::{
    any_value, check_common, destination, literal_bool, literal_float, literal_int, literal_string,
    operand, require_arity, sorted_strings, unmodeled_tile_form, AnyValue, CommonFacts, BOOL,
    ELEMENTWISE_EXEC_SCOPES, EXEC_SCOPES,
};
use super::tcgen_layout::{
    matches_ws_packed_tmem_layout, validate_block_scale_layout, validate_tcgen_smem_layout,
    validate_tcgen_tmem_a_layout, validate_ws_batched_tmem_layout,
};
use super::{ParsedTileCall, TileAttr, TileOpKind, TileOperand, TileRegion, TileScalar};
use crate::emit::tcgen_descriptor::{
    encode_block_scaled_instr_descriptor_fields, encode_dense_instr_descriptor_fields,
    validate_tcgen05_instruction_shape,
};
use crate::tables::is_integer_dtype;

fn m16n8_abi_coordinates(role: &str, row: i64, col: i64, mma_k: i64) -> (i64, i64, (i64, i64)) {
    match role {
        "D" | "C" => (
            4 * (row % 8) + (col % 8) / 2,
            2 * ((row % 16) / 8) + col % 2,
            (row / 16, col / 8),
        ),
        "A" => (
            4 * (row % 8) + (col % 8) / 2,
            4 * ((col % mma_k) / 8) + 2 * ((row % 16) / 8) + col % 2,
            (row / 16, col / mma_k),
        ),
        "B" => (
            4 * (col % 8) + (row % 8) / 2,
            2 * ((row % mma_k) / 8) + row % 2,
            (row / mma_k, col / 8),
        ),
        _ => unreachable!("unknown m16n8 fragment role {role}"),
    }
}

fn validate_m16n8_fragment_layout(
    region: &TileRegion,
    role: &str,
    rows: i64,
    cols: i64,
    mma_k: i64,
    transpose_storage: bool,
) -> AResult<()> {
    let analyzer = Analyzer::new()?;
    let Some(layout) = region.buffer.buffer_type().layout.clone() else {
        return Err(Failure::Ffi(ffi_error(
            "'NoneType' object has no attribute 'canonicalize'",
        )));
    };
    let canonical = layout.canonicalize()?;
    let Ok(canonical) = canonical.try_cast::<TileLayout>() else {
        return Err(Failure::Ffi(ffi_error(
            "'ComposeLayout' object has no attribute 'replica'",
        )));
    };
    if !canonical.replica()?.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): {role} fragment layout has replica axes, which mma.sync.m16n8k{mma_k} does not support"
        ));
    }
    let mut register_bases: Vec<((i64, i64), PrimExpr)> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let (storage_row, storage_col) = if transpose_storage {
                (col, row)
            } else {
                (row, col)
            };
            let physical = physical_layout_coordinates(
                region,
                &vint(storage_row),
                &vint(storage_col),
                "gemm",
            )?;
            if !axes_are(&physical, &["laneid", "m"]) {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm): {role} fragment layout must map exactly to laneid and m for mma.sync.m16n8k{mma_k}, got {:?}",
                    &physical_axes(&physical)));
            }
            let (expected_lane, expected_slot, tile) = m16n8_abi_coordinates(role, row, col, mma_k);
            let lane = physical_get(&physical, "laneid").expect("laneid axis");
            if !prove_equal(&analyzer, &vexpr(lane), &vint(expected_lane))? {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm): {role} fragment layout does not match fixed mma.sync.m16n8k{mma_k} ABI at logical ({row}, {col}): expected laneid={expected_lane}, got {}",
                    repr_text(&oref(lane.clone()))?
                ));
            }
            let slot = physical_get(&physical, "m").expect("m axis");
            let element_base = simplify_val(&analyzer, &vsub(&vexpr(slot), &vint(expected_slot))?)?;
            let register_base = match register_bases.iter().find(|(key, _)| *key == tile) {
                Some((_, base)) => base.clone(),
                None => {
                    register_bases.push((tile, element_base.clone()));
                    element_base.clone()
                }
            };
            if !analyzer.can_prove_equal(&element_base, &register_base)? {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm): {role} fragment layout does not match fixed mma.sync.m16n8k{mma_k} ABI at logical ({row}, {col}): register slot {} is not ABI slot {expected_slot} relative to one fragment base",
                    repr_text(&oref(slot.clone()))?
                ));
            }
        }
    }
    let slots_per_tile = match role {
        "D" | "C" => 4,
        "A" => mma_k / 2,
        "B" => mma_k / 4,
        _ => unreachable!("unknown m16n8 fragment role {role}"),
    };
    for (index, (tile, base)) in register_bases.iter().enumerate() {
        for (other_tile, other_base) in &register_bases[index + 1..] {
            let disjoint = analyzer.can_prove(&op_binary(
                "_OpLE",
                expr_any(&op_binary(
                    "_OpAdd",
                    expr_any(base),
                    int_any(slots_per_tile),
                )?),
                expr_any(other_base),
            )?)? || analyzer.can_prove(&op_binary(
                "_OpLE",
                expr_any(&op_binary(
                    "_OpAdd",
                    expr_any(other_base),
                    int_any(slots_per_tile),
                )?),
                expr_any(base),
            )?)?;
            if !disjoint {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm): {role} instruction tiles ({}, {}) and ({}, {}) may alias the same physical registers for mma.sync.m16n8k{mma_k}",
                    tile.0, tile.1, other_tile.0, other_tile.1
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn resolve_gemm(ctx: &Ctx, call: &TilePrimitiveCallObj) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "gemm";
    require_arity(call, op_name, 8)?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    if !["warp", "warpgroup", "cta"].contains(&facts.scope.as_str()) {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): requires warp, warpgroup, or CTA scope, got {}",
            facts.scope
        ));
    }
    if !matches!(facts.dispatch.as_deref(), None | Some("mma.m16n8k*")) {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unknown dispatch {:?}",
            facts.dispatch.as_deref().expect("dispatch")
        ));
    }
    if !facts.config.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported config keys {:?}",
            &sorted_strings(&facts.keys())
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let left = operand(analyzer, &call.args.get(1)?, "gemm.A")?;
    let right = operand(analyzer, &call.args.get(2)?, "gemm.B")?;
    let accumulator = operand(analyzer, &call.args.get(3)?, "gemm.C")?;
    let (Some(left), Some(right), Some(accumulator)) =
        (left.region(), right.region(), accumulator.region())
    else {
        return unsupported("TilePrimitiveCall(gemm): A, B, and C must be buffer regions");
    };
    let regions = [&destination, left, right, accumulator];
    if regions.iter().any(|region| region.memory_scope != "local") {
        return unsupported("TilePrimitiveCall(gemm): mma.m16n8k* requires local D, A, B, and C");
    }
    let dtypes: Vec<String> = regions.iter().map(|region| region.dtype.clone()).collect();
    let dtype_refs: Vec<&str> = dtypes.iter().map(String::as_str).collect();
    if dtype_refs != ["float32", "float16", "float16", "float32"]
        && dtype_refs != ["float32", "bfloat16", "bfloat16", "float32"]
    {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): mma.m16n8k* requires matching float16 or bfloat16 A/B with float32 C/D, got D/A/B/C={:?}",
            &dtypes));
    }
    let trans_a = literal_bool(&call.args.get(4)?, "gemm.transpose_A")?;
    let trans_b = literal_bool(&call.args.get(5)?, "gemm.transpose_B")?;
    let alpha = literal_float(&call.args.get(6)?, "gemm.alpha")?;
    let beta = literal_float(&call.args.get(7)?, "gemm.beta")?;
    if alpha != 1.0 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): mma.m16n8k* requires alpha=1, got {:?}",
            alpha
        ));
    }
    if beta != 0.0 && beta != 1.0 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): mma.m16n8k* requires beta in {{0, 1}}, got {:?}",
            beta
        ));
    }
    let d_shape = destination.logical_shape();
    let a_storage_shape = left.logical_shape();
    let b_storage_shape = right.logical_shape();
    let c_shape = accumulator.logical_shape();
    if [&d_shape, &a_storage_shape, &b_storage_shape, &c_shape]
        .iter()
        .any(|shape| shape.len() != 2)
    {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): mma.m16n8k* requires rank-2 D/A/B/C regions, got {:?}",
            &[&d_shape, &a_storage_shape, &b_storage_shape, &c_shape,]
        ));
    }
    let (m, k) = if trans_a {
        (a_storage_shape[1], a_storage_shape[0])
    } else {
        (a_storage_shape[0], a_storage_shape[1])
    };
    let (b_k, n) = if trans_b {
        (b_storage_shape[1], b_storage_shape[0])
    } else {
        (b_storage_shape[0], b_storage_shape[1])
    };
    if b_k != k || d_shape != [m, n] || c_shape != [m, n] {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): inconsistent logical matrix shapes after transpose normalization: D={:?}, A={:?}, B={:?}, C={:?}, transpose_A={:?}, transpose_B={:?}",
            &d_shape,
            &a_storage_shape,
            &b_storage_shape,
            &c_shape,
            trans_a,
            trans_b));
    }
    if m % 16 != 0 || n % 8 != 0 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm): mma.m16n8k* requires M divisible by 16 and N divisible by 8, got M={m}, N={n}"
        ));
    }
    let mut failures: Vec<String> = Vec::new();
    let mut mma_k: Option<i64> = None;
    for candidate_k in [16, 8] {
        if k % candidate_k != 0 {
            continue;
        }
        let checks: [(&str, &TileRegion, i64, i64, bool); 4] = [
            ("D", &destination, m, n, false),
            ("A", left, m, k, trans_a),
            ("B", right, k, n, trans_b),
            ("C", accumulator, m, n, false),
        ];
        let mut failed = false;
        for (role, region, rows, cols, transpose_storage) in checks {
            match validate_m16n8_fragment_layout(
                region,
                role,
                rows,
                cols,
                candidate_k,
                transpose_storage,
            ) {
                Ok(()) => {}
                Err(Failure::Unsupported { message, .. }) => {
                    failures.push(message);
                    failed = true;
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        if failed {
            continue;
        }
        mma_k = Some(candidate_k);
        break;
    }
    let Some(mma_k) = mma_k else {
        let detail = match failures.first() {
            Some(first) => format!("; first rejected candidate: {first}"),
            None => String::new(),
        };
        return unsupported(format!(
            "TilePrimitiveCall(gemm): no mma.sync.m16n8k{{16,8}} instruction matches M={m}, N={n}, K={k}, D/A/B/C={:?}{detail}",
            &dtypes));
    };
    let input_dtype = left.dtype.clone();
    Ok(ParsedTileCall::new(
        TileOpKind::Gemm,
        &facts.scope,
        destination,
        vec![
            TileOperand::Region(left.clone()),
            TileOperand::Region(right.clone()),
            TileOperand::Region(accumulator.clone()),
        ],
        vec![
            ("m", TileAttr::Int(m)),
            ("n", TileAttr::Int(n)),
            ("k", TileAttr::Int(k)),
            ("mma_k", TileAttr::Int(mma_k)),
            ("input_dtype", TileAttr::Str(input_dtype)),
            ("trans_a", TileAttr::Bool(trans_a)),
            ("trans_b", TileAttr::Bool(trans_b)),
            ("beta", TileAttr::Int(beta as i64)),
        ],
    ))
}

// ----------------------------------------------------------------------
// gemm_async.
// ----------------------------------------------------------------------

fn tcgen_instruction_tile(m: i64, n: i64, cta_group: i64, minimum_n: i64) -> AResult<(i64, i64)> {
    let total_m = m * cta_group;
    let valid_m: [i64; 2] = if cta_group == 1 {
        [128, 64]
    } else {
        [256, 128]
    };
    let Some(descriptor_m) = valid_m
        .iter()
        .copied()
        .find(|candidate| total_m % candidate == 0)
    else {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): total M cannot be tiled by a legal TCGEN descriptor: M={m}, cta_group={cta_group}, total={total_m}"
        ));
    };
    let descriptor_n = if n <= 256 && n % minimum_n == 0 {
        Some(n)
    } else {
        let mut found = None;
        let mut candidate = 256;
        while candidate >= minimum_n {
            if n % candidate == 0 {
                found = Some(candidate);
                break;
            }
            candidate -= minimum_n;
        }
        found
    };
    let Some(descriptor_n) = descriptor_n else {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): N cannot be tiled by a legal TCGEN descriptor: N={n}, minimum={minimum_n}"
        ));
    };
    Ok((descriptor_m, descriptor_n))
}

pub struct GemmAsyncMappingForm {
    pub ws_batched: bool,
    pub cta2_banked_a: bool,
    pub variant_marker: Option<&'static str>,
}

const GEMM_ASYNC_MAPPING_FORMS: [GemmAsyncMappingForm; 3] = [
    GemmAsyncMappingForm {
        ws_batched: false,
        cta2_banked_a: false,
        variant_marker: None,
    },
    GemmAsyncMappingForm {
        ws_batched: true,
        cta2_banked_a: false,
        variant_marker: Some("v2::tile::variant::MappedWsBatched"),
    },
    GemmAsyncMappingForm {
        ws_batched: false,
        cta2_banked_a: true,
        variant_marker: Some("v2::tile::variant::MappedCta2BankedA"),
    },
];

fn is_predicate_dtype(dtype: &str) -> bool {
    is_integer_dtype(dtype) || dtype == BOOL
}

/// The metadata and config values of one gemm_async call.
struct GemmAsyncConfig {
    facts: CommonFacts,
    instruction_descriptor: Option<TileScalar>,
    predicate: Option<TileScalar>,
    is_ab_tf32: bool,
    weight_stationary: bool,
}

fn gemm_async_config(
    analyzer: &Analyzer,
    call: &TilePrimitiveCallObj,
    op_name: &str,
) -> AResult<GemmAsyncConfig> {
    if call.args.len() != 6 && call.args.len() != 8 {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): expected 6 dense or 8 block-scaled args, got {}",
            call.args.len()
        ));
    }
    let facts = check_common(call, op_name, &EXEC_SCOPES)?;
    if facts.scope != "thread" && facts.scope != "warp" {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): requires thread or warp scope, got {}",
            facts.scope
        ));
    }
    if !matches!(facts.dispatch.as_deref(), None | Some("tcgen05")) {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unknown dispatch {:?}",
            facts.dispatch.as_deref().expect("dispatch")
        ));
    }
    let unknown = facts.unknown_keys(&[
        "cta_group",
        "descI",
        "is_AB_tf32",
        "mma_m",
        "mma_n",
        "pred",
        "smem_desc",
        "weight_stationary",
    ]);
    if !unknown.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported config keys {:?}",
            &unknown
        ));
    }
    let mut instruction_descriptor: Option<TileScalar> = None;
    if let Some(value) = facts.get("descI") {
        match operand(analyzer, value, "gemm_async.descI")? {
            TileOperand::Scalar(scalar) if scalar.dtype == "uint32" => {
                instruction_descriptor = Some(scalar)
            }
            _ => {
                return unsupported("TilePrimitiveCall(gemm_async): descI must be a uint32 scalar")
            }
        }
    }
    let mut predicate: Option<TileScalar> = None;
    if let Some(value) = facts.get("pred") {
        match operand(analyzer, value, "gemm_async.pred")? {
            TileOperand::Scalar(scalar) if is_predicate_dtype(&scalar.dtype) => {
                predicate = Some(scalar)
            }
            _ => {
                return unsupported(
                    "TilePrimitiveCall(gemm_async): pred must be a bool or integer scalar",
                )
            }
        }
    }
    let is_ab_tf32 = match facts.get("is_AB_tf32") {
        Some(value) => literal_bool(value, "gemm_async.is_AB_tf32")?,
        None => false,
    };
    let weight_stationary = match facts.get("weight_stationary") {
        Some(value) => literal_bool(value, "gemm_async.weight_stationary")?,
        None => false,
    };
    if let Some(value) = facts.get("smem_desc") {
        let smem_desc = literal_string(value, "gemm_async.smem_desc")?;
        if !["encode", "hoist", "local_hoist", "recompute"].contains(&smem_desc.as_str()) {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): unsupported smem_desc {:?}",
                &smem_desc
            ));
        }
    }
    Ok(GemmAsyncConfig {
        facts,
        instruction_descriptor,
        predicate,
        is_ab_tf32,
        weight_stationary,
    })
}

/// The positional arguments of one gemm_async call.
struct GemmAsyncArgs {
    destination: TileRegion,
    left: TileRegion,
    right: TileRegion,
    scale_regions: Vec<TileRegion>,
    trans_a: bool,
    trans_b: bool,
    accum: TileScalar,
}

fn gemm_async_args(
    analyzer: &Analyzer,
    call: &TilePrimitiveCallObj,
    op_name: &str,
) -> AResult<GemmAsyncArgs> {
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let left = operand(analyzer, &call.args.get(1)?, "gemm_async.A")?;
    let right = operand(analyzer, &call.args.get(2)?, "gemm_async.B")?;
    let (Some(left), Some(right)) = (left.region(), right.region()) else {
        return unsupported("TilePrimitiveCall(gemm_async): A and B must be buffer regions");
    };
    if destination.dtype != "float32" || destination.memory_scope != "tmem" {
        return unsupported("TilePrimitiveCall(gemm_async): C must be float32 TMEM");
    }
    if left.dtype != right.dtype {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): A/B dtype mismatch {} != {}",
            left.dtype, right.dtype
        ));
    }
    if right.memory_scope != "shared" || !["shared", "tmem"].contains(&left.memory_scope.as_str()) {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): expected A in SMEM/TMEM and B in SMEM, got {}/{}",
            left.memory_scope, right.memory_scope
        ));
    }

    let mut scale_regions: Vec<TileRegion> = Vec::new();
    let trans_a_index = if call.args.len() == 8 {
        let scale_a = operand(analyzer, &call.args.get(3)?, "gemm_async.SFA")?;
        let scale_b = operand(analyzer, &call.args.get(4)?, "gemm_async.SFB")?;
        let (Some(scale_a), Some(scale_b)) = (scale_a.region(), scale_b.region()) else {
            return unsupported(
                "TilePrimitiveCall(gemm_async): scale factors must be buffer regions",
            );
        };
        scale_regions = vec![scale_a.clone(), scale_b.clone()];
        5
    } else {
        3
    };
    let trans_a = literal_bool(&call.args.get(trans_a_index)?, "gemm_async.transA")?;
    let trans_b = literal_bool(&call.args.get(trans_a_index + 1)?, "gemm_async.transB")?;
    let accum_arg = call.args.get(trans_a_index + 2)?;
    let accum = if matches!(any_value(&accum_arg)?, AnyValue::Bool(_)) {
        TileOperand::Scalar(TileScalar {
            expr: accum_arg.clone(),
            dtype: BOOL.to_owned(),
        })
    } else {
        operand(analyzer, &accum_arg, "gemm_async.accum")?
    };
    let accum = match accum {
        TileOperand::Scalar(scalar) if is_predicate_dtype(&scalar.dtype) => scalar,
        _ => {
            return unsupported(
                "TilePrimitiveCall(gemm_async): accum must be a bool or integer scalar",
            )
        }
    };
    Ok(GemmAsyncArgs {
        destination,
        left: left.clone(),
        right: right.clone(),
        scale_regions,
        trans_a,
        trans_b,
        accum,
    })
}

/// The logical matrix geometry of one gemm_async call.
#[derive(Clone, Copy)]
struct GemmAsyncGeometry {
    cta_group: i64,
    mapping_form: &'static GemmAsyncMappingForm,
    m: i64,
    k_left: i64,
    n: i64,
    output_n: i64,
    weight_stationary: bool,
}

fn gemm_async_geometry(
    config: &GemmAsyncConfig,
    args: &GemmAsyncArgs,
) -> AResult<GemmAsyncGeometry> {
    let facts = &config.facts;
    let mut weight_stationary = config.weight_stationary;
    let (destination, left, right) = (&args.destination, &args.left, &args.right);
    let (trans_a, trans_b) = (args.trans_a, args.trans_b);
    let left_shape = left.logical_shape();
    let right_shape = right.logical_shape();
    let output_shape = destination.logical_shape();
    let cta_group = match facts.get("cta_group") {
        Some(value) => literal_int(value, "gemm_async.cta_group")?,
        None => 1,
    };
    if cta_group != 1 && cta_group != 2 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): cta_group must be 1 or 2, got {cta_group}"
        ));
    }
    let ws_batched = output_shape.len() == 3
        && output_shape[0] == 2
        && left_shape.len() == 3
        && left_shape[0] == 2
        && right_shape.len() == 2;
    let cta2_banked_a = cta_group == 2
        && output_shape.len() == 2
        && left_shape.len() == 3
        && left_shape[0] == 2
        && right_shape.len() == 2;
    let mapping_form = GEMM_ASYNC_MAPPING_FORMS
        .iter()
        .find(|form| form.ws_batched == ws_batched && form.cta2_banked_a == cta2_banked_a)
        .expect("mapping form");
    let m;
    let k_left;
    let n;
    let k_right;
    let output_n;
    if ws_batched {
        if cta_group != 1 || left.memory_scope != "tmem" || trans_a {
            return unsupported(
                "TilePrimitiveCall(gemm_async): the batched [2,M,*] form requires cta_group=1, TMEM A, and transA=False",
            );
        }
        m = left_shape[1];
        k_left = left_shape[2];
        if trans_b {
            n = right_shape[1];
            k_right = right_shape[0];
        } else {
            n = right_shape[0];
            k_right = right_shape[1];
        }
        if m != 64 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): batched weight-stationary M must be 64, got {m}"
            ));
        }
        if output_shape != [2, m, n / 2] || n % 2 != 0 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): batched weight-stationary C must be [2,M,N/2], got C={:?}, A={:?}, B={:?}",
                &output_shape,
                &left_shape,
                &right_shape));
        }
        output_n = n;
        weight_stationary = true;
    } else if cta2_banked_a {
        if left.memory_scope != "tmem" || trans_a {
            return unsupported(
                "TilePrimitiveCall(gemm_async): the CTA2 bank-batched A form requires TMEM A and transA=False",
            );
        }
        m = left_shape[1];
        k_left = left_shape[2];
        if trans_b {
            n = right_shape[1];
            k_right = right_shape[0];
        } else {
            n = right_shape[0];
            k_right = right_shape[1];
        }
        output_n = n * 2;
        if output_shape != [m, output_n] {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): CTA2 bank-batched A requires C=[M,2N], got C={:?}, A={:?}, B={:?}",
                &output_shape,
                &left_shape,
                &right_shape));
        }
    } else if left_shape.len() != 2 || right_shape.len() != 2 || output_shape.len() != 2 {
        return unsupported(
            "TilePrimitiveCall(gemm_async): C, A, and B must be logical rank-2 regions, except for the supported weight-stationary [2,M,*] fold",
        );
    } else {
        if trans_a {
            m = left_shape[1];
            k_left = left_shape[0];
        } else {
            m = left_shape[0];
            k_left = left_shape[1];
        }
        if trans_b {
            n = right_shape[1];
            k_right = right_shape[0];
        } else {
            n = right_shape[0];
            k_right = right_shape[1];
        }
        output_n = n * cta_group;
        if output_shape != [m, output_n] {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): C shape {:?} does not match {:?} for cta_group={cta_group}",
                &output_shape,
                &[m, output_n]));
        }
        let packed_ws_output =
            cta_group == 1 && matches_ws_packed_tmem_layout(destination, m, output_n)?;
        if packed_ws_output {
            if facts.has("weight_stationary") && !weight_stationary {
                return unsupported(
                    "TilePrimitiveCall(gemm_async): packed Layout-E C implies weight_stationary=True, but the config explicitly disables it",
                );
            }
            weight_stationary = true;
        }
    }
    if k_left != k_right {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): K mismatch {k_left} != {k_right}"
        ));
    }
    if weight_stationary && cta_group != 1 {
        return unsupported("TilePrimitiveCall(gemm_async): tcgen05.mma.ws requires cta_group=1");
    }
    if m != 64 && m != 128 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): typed TCGEN ABI supports M=64 or M=128, got {m}"
        ));
    }
    if left.memory_scope == "tmem" && trans_a {
        return unsupported(
            "TilePrimitiveCall(gemm_async): transA must be False when A resides in TMEM",
        );
    }
    Ok(GemmAsyncGeometry {
        cta_group,
        mapping_form,
        m,
        k_left,
        n,
        output_n,
        weight_stationary,
    })
}

/// The tcgen05 instruction one gemm_async call lowers to.
struct GemmAsyncInstruction {
    mma_k: i64,
    descriptor_m: i64,
    descriptor_n: i64,
    expected_instruction_descriptor: i64,
    scale_numbers: Option<GemmAsyncScaleNumbers>,
}

/// The block-scaling numbers of one gemm_async call.
#[derive(Clone, Copy)]
pub struct GemmAsyncScaleNumbers {
    pub vector: i64,
    pub values_per_mma: i64,
    pub a_elements_per_ki: i64,
    pub b_elements_per_ki: i64,
}

/// The resolved gemm_async call: the operands, geometry and instruction the
/// lowering reads, instead of re-deriving them from
/// the generic operand list and attributes.
#[derive(Clone)]
pub struct GemmAsyncFacts {
    pub left: TileRegion,
    pub right: TileRegion,
    pub accumulate: TileScalar,
    /// The scale operands and numbers of a block-scaled call.
    pub scales: Option<(TileRegion, TileRegion, GemmAsyncScaleNumbers)>,
    pub instruction_descriptor: Option<TileScalar>,
    pub predicate: Option<TileScalar>,
    pub mapping_form: &'static GemmAsyncMappingForm,
    pub instruction_m: i64,
    pub instruction_n: i64,
    pub instruction_k: i64,
    pub expected_instruction_descriptor: i64,
    pub is_ab_tf32: bool,
    pub trans_a: bool,
    pub trans_b: bool,
    pub weight_stationary: bool,
    pub cta_group: i64,
    pub m: i64,
    /// The destination N; `source_n` is the B operand's N.
    pub n: i64,
    pub source_n: i64,
    pub k: i64,
}

impl GemmAsyncFacts {
    /// The generic operand list these facts stand for: the matrices, the block
    /// scales and the optional scalars, in tcgen05 argument order.
    fn operands(&self) -> Vec<TileOperand> {
        let mut operands = vec![
            TileOperand::Region(self.left.clone()),
            TileOperand::Region(self.right.clone()),
        ];
        if let Some((scale_a, scale_b, _)) = &self.scales {
            operands.push(TileOperand::Region(scale_a.clone()));
            operands.push(TileOperand::Region(scale_b.clone()));
        }
        operands.push(TileOperand::Scalar(self.accumulate.clone()));
        for scalar in [&self.instruction_descriptor, &self.predicate]
            .into_iter()
            .flatten()
        {
            operands.push(TileOperand::Scalar(scalar.clone()));
        }
        operands
    }
}

fn dense_gemm_async_instruction(
    config: &GemmAsyncConfig,
    args: &GemmAsyncArgs,
    geometry: &GemmAsyncGeometry,
    op_name: &str,
) -> AResult<GemmAsyncInstruction> {
    let (left, right) = (&args.left, &args.right);
    let (trans_a, trans_b) = (args.trans_a, args.trans_b);
    let GemmAsyncGeometry {
        cta_group,
        m,
        k_left,
        output_n,
        ..
    } = *geometry;
    let instruction_kind: &'static str;
    let mma_k: i64;
    let descriptor_m: i64;
    let descriptor_n: i64;
    let expected_instruction_descriptor: i64;
    if config.is_ab_tf32 {
        mma_k = 8;
        if cta_group != 1 {
            return unsupported(
                "TilePrimitiveCall(gemm_async): TF32 lowering currently requires cta_group=1",
            );
        }
        if left.dtype != "float32" {
            return unsupported(
                "TilePrimitiveCall(gemm_async): is_AB_tf32 requires float32 A/B operands",
            );
        }
        if k_left % 8 != 0 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): TF32 K={k_left} is not divisible by MMA_K=8"
            ));
        }
        (descriptor_m, descriptor_n) = tcgen_instruction_tile(m, output_n, cta_group, 8)?;
        validate_tcgen05_instruction_shape(
            "tf32",
            cta_group,
            descriptor_m,
            descriptor_n,
            mma_k,
            false,
        )?;
        expected_instruction_descriptor = encode_dense_instr_descriptor_fields(
            "float32",
            "tf32",
            "tf32",
            descriptor_m,
            descriptor_n,
            mma_k,
            trans_a,
            trans_b,
            cta_group,
            false,
            false,
            false,
            false,
        )?;
    } else if !["float16", "bfloat16", "float8_e4m3fn"].contains(&left.dtype.as_str()) {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall(gemm_async): dense dtype {} is not implemented",
                left.dtype
            ),
        );
    } else {
        (instruction_kind, mma_k) = if left.dtype == "float8_e4m3fn" {
            ("f8f6f4", 32)
        } else {
            ("f16", 16)
        };
        if k_left % mma_k != 0 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): dense K={k_left} is not divisible by MMA_K={mma_k}"
            ));
        }
        (descriptor_m, descriptor_n) =
            tcgen_instruction_tile(m, output_n, cta_group, if cta_group == 1 { 8 } else { 16 })?;
        validate_tcgen05_instruction_shape(
            instruction_kind,
            cta_group,
            descriptor_m,
            descriptor_n,
            mma_k,
            false,
        )?;
        expected_instruction_descriptor = encode_dense_instr_descriptor_fields(
            "float32",
            &left.dtype,
            &right.dtype,
            descriptor_m,
            descriptor_n,
            mma_k,
            trans_a,
            trans_b,
            cta_group,
            false,
            false,
            false,
            false,
        )?;
    }
    Ok(GemmAsyncInstruction {
        mma_k,
        descriptor_m,
        descriptor_n,
        expected_instruction_descriptor,
        scale_numbers: None,
    })
}

/// Without an explicit descI, the typed tcgen05 descriptor carries one scale ID
/// sequence for both SFA and SFB.
fn validate_shared_scale_ids(
    scale_a_ids: &[PrimExpr],
    scale_b_ids: &[PrimExpr],
    scale_a_elements_per_ki: i64,
    sf_per_mma: i64,
) -> AResult<()> {
    let scale_analyzer = Analyzer::new()?;
    let mut mismatch = scale_a_ids.len() != scale_b_ids.len();
    if !mismatch {
        for (scale_a_id, scale_b_id) in scale_a_ids.iter().zip(scale_b_ids.iter()) {
            if !scale_analyzer.can_prove_equal(scale_a_id, scale_b_id)? {
                mismatch = true;
                break;
            }
        }
    }
    if mismatch {
        let render = |ids: &[PrimExpr]| -> AResult<String> {
            let items = ids
                .iter()
                .map(|id| repr_text(&oref(id.clone())))
                .collect::<AResult<Vec<_>>>()?;
            Ok(format!("{:?}", &items))
        };
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): SFA/SFB layouts require different scale IDs; the typed tcgen05 descriptor carries one shared ID sequence ({} != {})",
            render(scale_a_ids)?,
            render(scale_b_ids)?
        ));
    }
    if scale_a_elements_per_ki == 0 || sf_per_mma == 4 {
        for scale_id in scale_a_ids {
            if !prove_equal(&scale_analyzer, &vexpr(scale_id), &vint(0))? {
                return unsupported(
                    "TilePrimitiveCall(gemm_async): block-scale layout requires a nonzero scale ID where the typed tcgen05 lowering leaves the descriptor ID at zero",
                );
            }
        }
    }
    Ok(())
}

fn block_scaled_gemm_async_instruction(
    config: &GemmAsyncConfig,
    args: &GemmAsyncArgs,
    geometry: &GemmAsyncGeometry,
    op_name: &str,
) -> AResult<GemmAsyncInstruction> {
    let (left, right) = (&args.left, &args.right);
    let (trans_a, trans_b) = (args.trans_a, args.trans_b);
    let GemmAsyncGeometry {
        cta_group,
        m,
        k_left,
        output_n,
        ..
    } = *geometry;
    let scale_a = &args.scale_regions[0];
    let scale_b = &args.scale_regions[1];
    if left.dtype != "float8_e4m3fn" && left.dtype != "float4_e2m1fn" {
        return unmodeled_tile_form(
            op_name,
            format!(
                "TilePrimitiveCall(gemm_async): block-scaled lowering currently requires float8_e4m3fn or float4_e2m1fn A/B, got {}",
                left.dtype
            ),
        );
    }
    let allowed_scale_dtypes: Vec<String> = if left.dtype == "float8_e4m3fn" {
        vec!["float8_e8m0fnu".to_owned()]
    } else {
        vec!["float8_e4m3fn".to_owned(), "float8_e8m0fnu".to_owned()]
    };
    if !allowed_scale_dtypes.contains(&scale_a.dtype)
        || !allowed_scale_dtypes.contains(&scale_b.dtype)
        || scale_a.dtype != scale_b.dtype
    {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {} block scaling requires matching SFA/SFB in {:?}, got {}/{}",
            left.dtype,
            &sorted_strings(&allowed_scale_dtypes),
            scale_a.dtype,
            scale_b.dtype
        ));
    }
    if scale_a.memory_scope != "tmem" || scale_b.memory_scope != "tmem" {
        return unsupported(
            "TilePrimitiveCall(gemm_async): block scale factors must reside in TMEM",
        );
    }
    let scale_a_shape = scale_a.logical_shape();
    let scale_b_shape = scale_b.logical_shape();
    if scale_a_shape.len() != 2 || scale_b_shape.len() != 2 {
        return unsupported(
            "TilePrimitiveCall(gemm_async): SFA/SFB must be logical rank-2 regions",
        );
    }
    if scale_a_shape[0] != m || scale_b_shape[0] < output_n {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): SFA rows must equal M and SFB rows must cover N: SFA={:?}, SFB={:?}, A rows={m}, B rows={output_n}",
            &scale_a_shape,
            &scale_b_shape));
    }
    let mma_k: i64;
    let sf_per_mma: i64;
    let instruction_kind: &'static str;
    if left.dtype == "float8_e4m3fn" {
        mma_k = 32;
        sf_per_mma = 1;
        instruction_kind = "mxf8f6f4";
    } else {
        mma_k = 64;
        sf_per_mma = if scale_a.dtype == "float8_e8m0fnu" {
            2
        } else {
            4
        };
        instruction_kind = if scale_a.dtype == "float8_e8m0fnu" {
            "mxf4"
        } else {
            "mxf4nvf4"
        };
    }
    let scale_vector = mma_k / sf_per_mma;
    if k_left % mma_k != 0 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): block-scaled K={k_left} is not divisible by MMA_K={mma_k}"
        ));
    }
    let (descriptor_m, descriptor_n) =
        tcgen_instruction_tile(m, output_n, cta_group, if cta_group == 1 { 8 } else { 16 })?;
    validate_tcgen05_instruction_shape(
        instruction_kind,
        cta_group,
        descriptor_m,
        descriptor_n,
        mma_k,
        false,
    )?;
    let k_iters = k_left / mma_k;
    let (scale_a_elements_per_ki, scale_a_ids) =
        validate_block_scale_layout(scale_a, "SFA", m, k_iters, sf_per_mma)?;
    let (scale_b_elements_per_ki, scale_b_ids) =
        validate_block_scale_layout(scale_b, "SFB", output_n, k_iters, sf_per_mma)?;
    if config.instruction_descriptor.is_none() {
        validate_shared_scale_ids(
            &scale_a_ids,
            &scale_b_ids,
            scale_a_elements_per_ki,
            sf_per_mma,
        )?;
    }
    let expected_instruction_descriptor = encode_block_scaled_instr_descriptor_fields(
        "float32",
        &left.dtype,
        &right.dtype,
        &scale_a.dtype,
        &scale_b.dtype,
        descriptor_m,
        descriptor_n,
        mma_k,
        trans_a,
        trans_b,
        cta_group,
        false,
        false,
        false,
    )?;
    Ok(GemmAsyncInstruction {
        mma_k,
        descriptor_m,
        descriptor_n,
        expected_instruction_descriptor,
        scale_numbers: Some(GemmAsyncScaleNumbers {
            vector: scale_vector,
            values_per_mma: sf_per_mma,
            a_elements_per_ki: scale_a_elements_per_ki,
            b_elements_per_ki: scale_b_elements_per_ki,
        }),
    })
}

pub(super) fn resolve_gemm_async(
    ctx: &Ctx,
    call: &TilePrimitiveCallObj,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "gemm_async";
    let config = gemm_async_config(analyzer, call, op_name)?;
    let args = gemm_async_args(analyzer, call, op_name)?;
    let geometry = gemm_async_geometry(&config, &args)?;
    let GemmAsyncGeometry {
        cta_group,
        mapping_form,
        m,
        k_left,
        n,
        output_n,
        weight_stationary,
    } = geometry;
    let (ws_batched, cta2_banked_a) = (mapping_form.ws_batched, mapping_form.cta2_banked_a);
    let (destination, left, right) = (&args.destination, &args.left, &args.right);
    let layouts = (|| -> AResult<()> {
        if ws_batched {
            validate_ws_batched_tmem_layout(destination, m, output_n / 2, "C")?;
            validate_ws_batched_tmem_layout(left, m, k_left, "A")?;
        } else if cta2_banked_a {
            validate_ws_batched_tmem_layout(left, m, k_left, "A")?;
        } else if left.memory_scope == "shared" {
            validate_tcgen_smem_layout(ctx, left, "A")?;
        } else {
            validate_tcgen_tmem_a_layout(left, m, k_left)?;
        }
        validate_tcgen_smem_layout(ctx, right, "B")
    })();
    match layouts {
        Ok(()) => {}
        Err(Failure::Unsupported { message, .. }) => {
            return unmodeled_tile_form(op_name, message);
        }
        Err(error) => return Err(error),
    }

    let block_scaled = !args.scale_regions.is_empty();
    if (ws_batched || cta2_banked_a) && block_scaled {
        return unsupported(
            "TilePrimitiveCall(gemm_async): batched TMEM-A block scaling is unsupported",
        );
    }
    if block_scaled && config.is_ab_tf32 {
        return unsupported(
            "TilePrimitiveCall(gemm_async): is_AB_tf32 is incompatible with block scaling",
        );
    }
    let GemmAsyncInstruction {
        mma_k,
        descriptor_m,
        descriptor_n,
        expected_instruction_descriptor,
        scale_numbers,
    } = if !block_scaled {
        dense_gemm_async_instruction(&config, &args, &geometry, op_name)?
    } else {
        block_scaled_gemm_async_instruction(&config, &args, &geometry, op_name)?
    };

    for (key, expected) in [("mma_m", descriptor_m), ("mma_n", descriptor_n)] {
        if let Some(value) = config.facts.get(key) {
            let declared = literal_int(value, &format!("gemm_async.{key}"))?;
            if declared != expected {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm_async): {key}={declared} disagrees with the descriptor geometry inferred from the operands ({expected})"
                ));
            }
        }
    }

    let scales = match scale_numbers {
        Some(numbers) => Some((
            args.scale_regions[0].clone(),
            args.scale_regions[1].clone(),
            numbers,
        )),
        None => None,
    };
    let facts = GemmAsyncFacts {
        left: args.left.clone(),
        right: args.right.clone(),
        accumulate: args.accum.clone(),
        scales,
        instruction_descriptor: config.instruction_descriptor.clone(),
        predicate: config.predicate.clone(),
        mapping_form,
        instruction_m: descriptor_m,
        instruction_n: descriptor_n,
        instruction_k: mma_k,
        expected_instruction_descriptor,
        is_ab_tf32: config.is_ab_tf32,
        trans_a: args.trans_a,
        trans_b: args.trans_b,
        weight_stationary,
        cta_group,
        m,
        n: output_n,
        source_n: n,
        k: k_left,
    };
    let operands = facts.operands();
    Ok(ParsedTileCall::new(
        TileOpKind::GemmAsync,
        &config.facts.scope,
        args.destination,
        operands,
        Vec::new(),
    )
    .with_gemm_async(facts))
}
