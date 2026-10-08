//! copy / permute_layout / copy_async.

use crate::tvm_compat::int_value;
use tvm::ir::IntImmObj;
use tvm::tirx::TilePrimitiveCallObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, Array, Map, ObjectRefCore};

use super::super::util::{unsupported, AResult};
use super::super::Ctx;
use super::parse::{
    any_value, check_common, destination, is_float_dtype, literal_bool, literal_int,
    literal_string, operand, require_arity, same_logical_shape, unmodeled_tile_form, AnyValue,
    CommonFacts, ELEMENTWISE_EXEC_SCOPES, EXEC_SCOPES,
};
use super::tcgen_layout::{validate_tcgen_cp_layout, validate_tcgen_ldst_layout};
use super::{ParsedTileCall, TileAttr, TileOpKind, TileOperand, TileRegion};

const SNAPSHOT_DTYPES: [&str; 16] = [
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "float8_e4m3fn",
    "float8_e8m0fnu",
    "float4_e2m1fn",
];
const TMA_REDUCTION_DTYPES: [(&str, &[&str]); 8] = [
    (
        "add",
        &[
            "uint32", "int32", "uint64", "float32", "float16", "bfloat16",
        ],
    ),
    (
        "min",
        &["uint32", "int32", "uint64", "int64", "float16", "bfloat16"],
    ),
    (
        "max",
        &["uint32", "int32", "uint64", "int64", "float16", "bfloat16"],
    ),
    ("inc", &["uint32"]),
    ("dec", &["uint32"]),
    (
        "and",
        &["uint32", "int32", "float32", "uint64", "int64", "float64"],
    ),
    (
        "or",
        &["uint32", "int32", "float32", "uint64", "int64", "float64"],
    ),
    (
        "xor",
        &["uint32", "int32", "float32", "uint64", "int64", "float64"],
    ),
];
const TMA_CACHE_HINTS: [&str; 5] = [
    "",
    "evict_first",
    "evict_last",
    "evict_last_use",
    "evict_normal",
];
pub(crate) const TYPED_TMA_ELEMENT_DTYPES: [&str; 12] = [
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "int32",
    "int64",
    "int8",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
];

pub(crate) fn typed_tma_dtype_rejection(dtype: &str) -> String {
    let reason = match dtype {
        "bool" => "the production TensorMap encoder has no bool element ABI",
        "float4_e2m1fn" => {
            "typed TMA lowering has no exact CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B shared-layout \
             model; use the packed-uint8 idiom instead: declare the global and SMEM buffers as \
             uint8 with K//2 columns, TMA the bytes, and reinterpret the SMEM view as \
             float4_e2m1fn via decl_buffer (elem_offset*2), as the canonical NVFP4 GEMM does"
        }
        "float8_e8m0fnu" => "the production TensorMap encoder has no float8_e8m0fnu element ABI",
        "int16" => "the production TensorMap encoder has no signed int16 element ABI",
        "uint32x2" => "the production TensorMap encoder requires a scalar dtype with lanes=1",
        _ => "there is no reviewed production TensorMap element ABI",
    };
    format!(
        "TilePrimitiveCall(copy_async, dispatch=tma): unsupported dtype {:?}: {reason}",
        dtype
    )
}

fn copy_owner_axes(ctx: &Ctx, region: &TileRegion) -> AResult<Vec<String>> {
    let info = ctx.inspect_layout(&region.buffer, &Map::new())?;
    Ok(info
        .physical_axes
        .iter()
        .filter(|axis| ctx.schema.register_owner_axes.contains(*axis))
        .cloned()
        .collect())
}

fn validate_copy_owner_transport(
    ctx: &Ctx,
    destination: &TileRegion,
    source: &TileRegion,
) -> AResult<()> {
    let destination_axes = copy_owner_axes(ctx, destination)?;
    let source_axes = copy_owner_axes(ctx, source)?;
    if destination.memory_scope == "local"
        && source.memory_scope == "local"
        && destination_axes.is_empty() != source_axes.is_empty()
    {
        return unsupported(
            "TilePrimitiveCall(copy): canonical copy between a per-lane local buffer and a distributed local layout is not implemented",
        );
    }
    Ok(())
}

pub(super) fn resolve_copy(ctx: &Ctx, call: &TilePrimitiveCallObj) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "copy";
    require_arity(call, op_name, 2)?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    let unknown =
        facts.unknown_keys(&["cache", "l1_evict", "l2_evict", "prefetch_size", "vec_len"]);
    if !unknown.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall(copy): unsupported config keys {:?}",
            &unknown
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let TileOperand::Region(source) = operand(analyzer, &call.args.get(1)?, "copy.src")? else {
        return unsupported("TilePrimitiveCall(copy): source must be a buffer");
    };
    same_logical_shape(op_name, &destination, &[&source])?;
    if destination.dtype != source.dtype {
        return unsupported(format!(
            "TilePrimitiveCall(copy): dtype mismatch {} != {}",
            destination.dtype, source.dtype
        ));
    }
    let mut cache: Option<String> = None;
    if let Some(value) = facts.get("cache") {
        let text = literal_string(value, "copy.cache")?;
        if text != "nc" {
            return unsupported(format!(
                "TilePrimitiveCall(copy): cache must be 'nc', got {:?}",
                &text
            ));
        }
        cache = Some(text);
    }
    let cache_hint_values: [(&str, &[&str]); 3] = [
        ("l1_evict", &["L1::evict_normal", "L1::no_allocate"]),
        ("l2_evict", &["L2::evict_normal", "L2::evict_first"]),
        ("prefetch_size", &["L2::256B"]),
    ];
    for (key, allowed) in cache_hint_values {
        let Some(value) = facts.get(key) else {
            continue;
        };
        let text = literal_string(value, &format!("copy.{key}"))?;
        if !allowed.contains(&text.as_str()) {
            let mut sorted: Vec<String> = allowed.iter().map(|item| (*item).to_owned()).collect();
            sorted.sort();
            return unsupported(format!(
                "TilePrimitiveCall(copy): unsupported {key} {:?}; expected one of {:?}",
                &text, &sorted
            ));
        }
    }
    let has_cache_hint = cache_hint_values.iter().any(|(key, _)| facts.has(key));
    if (cache.is_some() || has_cache_hint) && source.memory_scope != "global" {
        return unsupported(
            "TilePrimitiveCall(copy): load cache hints require a global-memory source",
        );
    }
    if !SNAPSHOT_DTYPES.contains(&source.dtype.as_str()) {
        return unmodeled_tile_form(
            op_name,
            format!(
                "Snapshot copy lowering is not implemented for {}",
                source.dtype
            ),
        );
    }
    let allowed_spaces = ["global", "local", "shared"];
    if !allowed_spaces.contains(&destination.memory_scope.as_str())
        || !allowed_spaces.contains(&source.memory_scope.as_str())
    {
        return unsupported(format!(
            "TilePrimitiveCall(copy): unsupported memory pair {}->{}",
            source.memory_scope, destination.memory_scope
        ));
    }
    validate_copy_owner_transport(ctx, &destination, &source)?;
    Ok(ParsedTileCall::new(
        TileOpKind::Copy,
        &facts.scope,
        destination,
        vec![TileOperand::Region(source)],
        vec![("snapshot_source", TileAttr::Bool(true))],
    ))
}

pub(super) fn resolve_permute_layout(
    ctx: &Ctx,
    call: &TilePrimitiveCallObj,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "permute_layout";
    require_arity(call, op_name, 2)?;
    let facts = check_common(call, op_name, &EXEC_SCOPES)?;
    if facts.scope != "warp" {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): requires warp scope, got {}",
            facts.scope
        ));
    }
    if !facts.config.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): config is not implemented"
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let TileOperand::Region(source) =
        operand(analyzer, &call.args.get(1)?, &format!("{op_name}.src"))?
    else {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): source must be a buffer region"
        ));
    };
    same_logical_shape(op_name, &destination, &[&source])?;
    if destination.dtype != source.dtype {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): dtype mismatch {} != {}",
            destination.dtype, source.dtype
        ));
    }
    if !SNAPSHOT_DTYPES.contains(&source.dtype.as_str()) {
        return unmodeled_tile_form(
            op_name,
            format!(
                "Snapshot copy lowering is not implemented for {}",
                source.dtype
            ),
        );
    }
    let allowed_spaces = ["global", "local", "shared"];
    if !allowed_spaces.contains(&destination.memory_scope.as_str())
        || !allowed_spaces.contains(&source.memory_scope.as_str())
    {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}): unsupported memory pair {}->{}",
            source.memory_scope, destination.memory_scope
        ));
    }
    validate_copy_owner_transport(ctx, &destination, &source)?;
    let zero_fill = source.memory_scope == "shared";
    Ok(ParsedTileCall::new(
        TileOpKind::PermuteLayout,
        &facts.scope,
        destination,
        vec![TileOperand::Region(source)],
        vec![
            ("snapshot_source", TileAttr::Bool(true)),
            ("zero_fill_invalid_source", TileAttr::Bool(zero_fill)),
        ],
    ))
}

// ----------------------------------------------------------------------
// copy_async.
// ----------------------------------------------------------------------

/// The legality set of one (source, destination) pair.
struct CopyAsyncTransport {
    selects: &'static [(Option<&'static str>, &'static str)],
    unknown: &'static str,
}

const TMA_SELECTS: [(Option<&str>, &str); 3] = [
    (Some("tma"), "tma"),
    (Some("tma_auto"), "tma"),
    (Some("tma_explicit"), "tma"),
];

fn copy_async_transport(source: &str, destination: &str) -> Option<CopyAsyncTransport> {
    const GLOBAL_SHARED: [(Option<&str>, &str); 5] = [
        (None, "ldgsts"),
        (Some("ldgsts"), "ldgsts"),
        TMA_SELECTS[0],
        TMA_SELECTS[1],
        TMA_SELECTS[2],
    ];
    const SHARED_GLOBAL: [(Option<&str>, &str); 4] = [
        (None, "tma"),
        TMA_SELECTS[0],
        TMA_SELECTS[1],
        TMA_SELECTS[2],
    ];
    const SHARED_SHARED: [(Option<&str>, &str); 2] = [(None, "dsmem"), (Some("dsmem"), "dsmem")];
    const SHARED_TMEM: [(Option<&str>, &str); 2] =
        [(None, "tcgen05_cp"), (Some("smem->tmem"), "tcgen05_cp")];
    const TMEM_LOCAL: [(Option<&str>, &str); 2] = [
        (None, "tcgen05_ldst"),
        (Some("tmem<->local"), "tcgen05_ldst"),
    ];
    match (source, destination) {
        ("global", "shared") => Some(CopyAsyncTransport {
            selects: &GLOBAL_SHARED,
            unknown: "TilePrimitiveCall(copy_async): unknown global->shared dispatch {dispatch!r}",
        }),
        ("shared", "global") => Some(CopyAsyncTransport {
            selects: &SHARED_GLOBAL,
            unknown: "TilePrimitiveCall(copy_async): unknown shared->global dispatch {dispatch!r}",
        }),
        ("shared", "shared") => Some(CopyAsyncTransport {
            selects: &SHARED_SHARED,
            unknown: "TilePrimitiveCall(copy_async): unknown shared->shared dispatch {dispatch!r}",
        }),
        ("shared", "tmem") => Some(CopyAsyncTransport {
            selects: &SHARED_TMEM,
            unknown: "TilePrimitiveCall(copy_async): unknown shared->tmem dispatch {dispatch!r}",
        }),
        ("tmem", "local") | ("local", "tmem") => Some(CopyAsyncTransport {
            selects: &TMEM_LOCAL,
            unknown: "TilePrimitiveCall(copy_async): unknown TMEM/local dispatch {dispatch!r}",
        }),
        _ => None,
    }
}

fn copy_async_dispatch_scope(dispatch: &str) -> Option<&'static str> {
    match dispatch {
        "tma" => Some("thread"),
        "dsmem" => Some("thread"),
        "tcgen05_cp" => Some("thread"),
        "tcgen05_ldst" => Some("warpgroup"),
        _ => None,
    }
}

fn multicast_modifier_enabled(cta_mask: &Any) -> AResult<bool> {
    if let AnyValue::Object(node) = any_value(cta_mask)? {
        if let Some(imm) = node.as_node::<IntImmObj>() {
            return Ok(int_value(imm)?.count_ones() > 1);
        }
    }
    Ok(true)
}

/// `isinstance(value, (int, IntImm)) and not isinstance(value, bool)`.
fn is_int_like(value: &Any) -> AResult<bool> {
    Ok(match any_value(value)? {
        AnyValue::Int(_) => true,
        AnyValue::Object(node) => node.as_node::<IntImmObj>().is_some(),
        _ => false,
    })
}

fn copy_async_gather4(facts: &CommonFacts) -> AResult<Vec<Any>> {
    let mut gather4: Vec<Any> = Vec::new();
    if let Some(raw_gather4) = facts.get("gather4") {
        let rows = match any_value(raw_gather4)? {
            AnyValue::Object(node) => Array::<Any>::try_from(Any::from(node)).ok(),
            _ => None,
        };
        let Some(rows) = rows else {
            return unsupported(
                "TilePrimitiveCall(copy_async): gather4 must contain four row coordinates",
            );
        };
        gather4 = rows.iter().collect();
        if gather4.len() != 4 {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): gather4 requires four rows, got {}",
                gather4.len()
            ));
        }
    }
    Ok(gather4)
}

/// The dispatch the (source, destination) memory pair selects, checked against
/// the operand shapes and dtypes.
fn select_copy_async_dispatch(
    facts: &CommonFacts,
    pair: (&str, &str),
    source: &TileRegion,
    destination: &TileRegion,
    gather4: &[Any],
    op_name: &str,
) -> AResult<&'static str> {
    if pair == ("shared", "tmem") {
        if source.element_count() != destination.element_count() {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): tcgen05.cp element count mismatch {} != {}",
                destination.element_count(),
                source.element_count()
            ));
        }
    } else if !gather4.is_empty() {
        if facts.dispatch.as_deref() != Some("tma_explicit") || pair != ("global", "shared") {
            return unsupported(
                "TilePrimitiveCall(copy_async): gather4 requires tma_explicit global->shared",
            );
        }
        if source.extents.len() != 2
            || destination.extents.len() != 2
            || source.extents[0] != 1
            || destination.extents[0] != 4
            || source.extents[1] != destination.extents[1]
        {
            return unsupported(
                "TilePrimitiveCall(copy_async): gather4 requires a rank-2 one-row source and four-row destination with equal row width",
            );
        }
    } else {
        same_logical_shape(op_name, destination, &[source])?;
    }

    let Some(transport) = copy_async_transport(pair.0, pair.1) else {
        return unsupported(format!(
            "TilePrimitiveCall(copy_async): typed GMEM/SMEM lowering does not implement {}->{}",
            source.memory_scope, destination.memory_scope
        ));
    };
    let dispatch = facts.dispatch.as_deref();
    let Some((_, selected_dispatch)) = transport
        .selects
        .iter()
        .find(|(candidate, _)| *candidate == dispatch)
    else {
        let rendered = match dispatch {
            Some(text) => format!("{:?}", text),
            None => "None".to_owned(),
        };
        return unsupported(transport.unknown.replace("{dispatch!r}", &rendered));
    };
    let selected_dispatch = *selected_dispatch;

    if selected_dispatch == "tcgen05_cp" {
        if source.element_count() != destination.element_count() {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): tcgen05.cp physical element count mismatch {} != {}",
                destination.element_count(),
                source.element_count()
            ));
        }
    } else if gather4.is_empty() {
        same_logical_shape(op_name, destination, &[source])?;
    }

    if destination.dtype != source.dtype {
        let scale_bitcast = selected_dispatch == "tcgen05_cp"
            && source.dtype == "uint8"
            && ["float8_e4m3fn", "float8_e8m0fnu"].contains(&destination.dtype.as_str());
        let integer_bitcast =
            selected_dispatch == "tma" && source.dtype == "int64" && destination.dtype == "uint64";
        if !(scale_bitcast || integer_bitcast) {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): physical copy requires matching dtypes, got {}->{}",
                source.dtype, destination.dtype
            ));
        }
    }
    if selected_dispatch == "tma" && !TYPED_TMA_ELEMENT_DTYPES.contains(&destination.dtype.as_str())
    {
        return unmodeled_tile_form("copy_async", typed_tma_dtype_rejection(&destination.dtype));
    }
    Ok(selected_dispatch)
}

/// The exec scope and config keys the selected dispatch accepts.
fn validate_copy_async_config(
    facts: &CommonFacts,
    selected_dispatch: &str,
    pair: (&str, &str),
) -> AResult<()> {
    if let Some(required_scope) = copy_async_dispatch_scope(selected_dispatch) {
        if facts.scope != required_scope {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): {selected_dispatch} requires {required_scope} scope, got {}",
                facts.scope
            ));
        }
    }
    if selected_dispatch == "tma" && pair == ("global", "shared") && !facts.has("mbar") {
        return unsupported("TilePrimitiveCall(copy_async): TMA global->shared requires mbar");
    }
    if selected_dispatch == "dsmem" {
        let missing: Vec<String> = ["mbar", "remote_cta_id"]
            .iter()
            .filter(|key| !facts.has(key))
            .map(|key| (*key).to_owned())
            .collect();
        if !missing.is_empty() {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): dsmem requires config keys {:?}",
                &missing
            ));
        }
    }
    if facts.has("cta_mask") && !(selected_dispatch == "tma" && pair == ("global", "shared")) {
        return unsupported(
            "TilePrimitiveCall(copy_async): cta_mask is only valid for TMA global->shared",
        );
    }
    if facts.has("remote_cta_id") && selected_dispatch != "dsmem" {
        return unsupported("TilePrimitiveCall(copy_async): remote_cta_id is only valid for dsmem");
    }
    if facts.has("mbar") && !["tma", "dsmem"].contains(&selected_dispatch) {
        return unsupported(
            "TilePrimitiveCall(copy_async): mbar is not an argument to TCGEN transfers",
        );
    }
    if let Some(value) = facts.get("mbarrier_addr") {
        if selected_dispatch != "tma" || pair != ("global", "shared") {
            return unsupported(
                "TilePrimitiveCall(copy_async): mbarrier_addr is only valid for global->shared TMA",
            );
        }
        literal_bool(value, "copy_async.mbarrier_addr")?;
    }
    for key in ["shape", "multicast"] {
        if facts.has(key) && selected_dispatch != "tcgen05_cp" {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): {key} is only valid for tcgen05.cp"
            ));
        }
    }
    Ok(())
}

/// The `reduction`, `oob` and `tma_dtype` attributes of a TMA copy.
fn copy_async_tma_attrs(
    facts: &CommonFacts,
    selected_dispatch: &str,
    pair: (&str, &str),
    source: &TileRegion,
    destination: &TileRegion,
) -> AResult<Vec<(&'static str, TileAttr)>> {
    let mut attrs_oob: Option<String> = None;
    if let Some(value) = facts.get("oob") {
        if selected_dispatch != "tma" || pair != ("global", "shared") {
            return unsupported(
                "TilePrimitiveCall(copy_async): oob is only valid for global->shared TMA",
            );
        }
        let oob = literal_string(value, "copy_async.oob")?;
        if oob != "zero" && oob != "nan" {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): unsupported TMA oob mode {:?}",
                &oob
            ));
        }
        if oob == "nan" && !is_float_dtype(&source.dtype) {
            return unsupported(
                "TilePrimitiveCall(copy_async): TMA oob='nan' requires a floating-point dtype",
            );
        }
        attrs_oob = Some(oob);
    }
    let mut reduction: Option<String> = None;
    if let Some(value) = facts.get("use_tma_reduce") {
        if selected_dispatch != "tma" || pair != ("shared", "global") {
            return unsupported(
                "TilePrimitiveCall(copy_async): use_tma_reduce is only valid for shared->global TMA stores",
            );
        }
        let name = literal_string(value, "copy_async.use_tma_reduce")?;
        let Some((_, allowed_dtypes)) = TMA_REDUCTION_DTYPES
            .iter()
            .find(|(candidate, _)| *candidate == name)
        else {
            let mut names: Vec<String> = TMA_REDUCTION_DTYPES
                .iter()
                .map(|(candidate, _)| (*candidate).to_owned())
                .collect();
            names.sort();
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): unsupported TMA reduction {:?}; expected one of {:?}",
                &name,
                &names));
        };
        if !allowed_dtypes.contains(&destination.dtype.as_str()) {
            let mut allowed: Vec<String> = allowed_dtypes
                .iter()
                .map(|dtype| (*dtype).to_owned())
                .collect();
            allowed.sort();
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): TMA reduction {:?} is invalid for dtype {:?}; expected one of {:?}",
                &name,
                &destination.dtype,
                &allowed));
        }
        reduction = Some(name);
    }
    let mut tma_dtype: Option<String> = None;
    if let Some(value) = facts.get("tma_dtype") {
        let name = literal_string(value, "copy_async.tma_dtype")?;
        if !(selected_dispatch == "tma"
            && pair == ("global", "shared")
            && source.dtype == "float32"
            && destination.dtype == "float32"
            && name == "tf32")
        {
            return unsupported(
                "TilePrimitiveCall(copy_async): tma_dtype is only implemented as 'tf32' for float32 TMA global->shared copies",
            );
        }
        tma_dtype = Some(name);
    }

    let mut attrs: Vec<(&'static str, TileAttr)> = Vec::new();
    if let Some(reduction) = reduction {
        attrs.push(("reduction", TileAttr::Str(reduction)));
    }
    if let Some(oob) = attrs_oob {
        attrs.push(("oob", TileAttr::Str(oob)));
    }
    if let Some(tma_dtype) = tma_dtype {
        attrs.push(("tma_dtype", TileAttr::Str(tma_dtype)));
    }
    Ok(attrs)
}

/// The `cache_hint`, `l2_promotion` and `tensormap_l2_promotion` attributes.
fn copy_async_cache_attrs(
    facts: &CommonFacts,
    selected_dispatch: &str,
) -> AResult<Vec<(&'static str, TileAttr)>> {
    let mut attrs: Vec<(&'static str, TileAttr)> = Vec::new();
    if let Some(raw_cache_hint) = facts.get("cache_hint") {
        if is_int_like(raw_cache_hint)? {
            let cache_hint = literal_int(raw_cache_hint, "copy_async.cache_hint")?;
            if cache_hint < 0 {
                return unsupported(
                    "TilePrimitiveCall(copy_async): integer cache_hint must fit uint64",
                );
            }
            attrs.push(("cache_hint", TileAttr::Int(cache_hint)));
        } else {
            let cache_hint = literal_string(raw_cache_hint, "copy_async.cache_hint")?;
            if !TMA_CACHE_HINTS.contains(&cache_hint.as_str()) {
                let mut hints: Vec<String> = TMA_CACHE_HINTS
                    .iter()
                    .map(|hint| (*hint).to_owned())
                    .collect();
                hints.sort();
                return unsupported(format!(
                    "TilePrimitiveCall(copy_async): unsupported TMA cache_hint {:?}; expected one of {:?}",
                    &cache_hint,
                    &hints));
            }
            attrs.push(("cache_hint", TileAttr::Str(cache_hint)));
        }
    }
    if let Some(value) = facts.get("l2_promotion") {
        if selected_dispatch != "tma" {
            return unsupported(
                "TilePrimitiveCall(copy_async): l2_promotion is only valid for TMA copies",
            );
        }
        let l2_promotion = literal_int(value, "copy_async.l2_promotion")?;
        if ![64, 128, 256].contains(&l2_promotion) {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): l2_promotion must be 64, 128, or 256 bytes, got {l2_promotion}"
            ));
        }
        attrs.push(("l2_promotion", TileAttr::Int(l2_promotion)));
    }
    if let Some(value) = facts.get("tensormap_l2_promotion") {
        if selected_dispatch != "tma" {
            return unsupported(
                "TilePrimitiveCall(copy_async): tensormap_l2_promotion is only valid for TMA copies",
            );
        }
        let promotion = if is_int_like(value)? {
            let promotion = literal_int(value, "copy_async.tensormap_l2_promotion")?;
            if !(0..=3).contains(&promotion) {
                return unsupported(format!(
                    "TilePrimitiveCall(copy_async): tensormap_l2_promotion integer must be in [0, 3], got {promotion}"
                ));
            }
            promotion
        } else {
            let name = literal_string(value, "copy_async.tensormap_l2_promotion")?;
            match name.as_str() {
                "none" | "L2::none" => 0,
                "L2::64B" => 1,
                "L2::128B" => 2,
                "L2::256B" => 3,
                _ => {
                    return unsupported(format!(
                        "TilePrimitiveCall(copy_async): unsupported tensormap_l2_promotion {:?}",
                        &name
                    ))
                }
            }
        };
        attrs.push(("tensormap_l2_promotion", TileAttr::Int(promotion)));
    }
    Ok(attrs)
}

pub(super) fn resolve_copy_async(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &TilePrimitiveCallObj,
) -> AResult<ParsedTileCall> {
    let analyzer = &ctx.analyzer;
    let op_name = "copy_async";
    require_arity(call, op_name, 2)?;
    let facts = check_common(call, op_name, &ELEMENTWISE_EXEC_SCOPES)?;
    let unknown = facts.unknown_keys(&[
        "cache_hint",
        "cta_group",
        "cta_mask",
        "gather4",
        "l2_promotion",
        "mbar",
        "mbarrier_addr",
        "oob",
        "prefetch_tensormap",
        "remote_cta_id",
        "shape",
        "multicast",
        "tensormap_l2_promotion",
        "tma_dtype",
        "use_tma_reduce",
    ]);
    if !unknown.is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall(copy_async): unsupported config keys {:?}",
            &unknown
        ));
    }
    let destination = destination(analyzer, &call.args.get(0)?, op_name)?;
    let TileOperand::Region(source) = operand(analyzer, &call.args.get(1)?, "copy_async.src")?
    else {
        return unsupported("TilePrimitiveCall(copy_async): source must be a buffer");
    };

    let gather4 = copy_async_gather4(&facts)?;
    let pair = (
        source.memory_scope.as_str(),
        destination.memory_scope.as_str(),
    );
    let selected_dispatch =
        select_copy_async_dispatch(&facts, pair, &source, &destination, &gather4, op_name)?;
    validate_copy_async_config(&facts, selected_dispatch, pair)?;

    let mut attrs: Vec<(&'static str, TileAttr)> = vec![
        ("dispatch", TileAttr::Str(selected_dispatch.to_owned())),
        ("source_scope", TileAttr::Str(source.memory_scope.clone())),
        (
            "destination_scope",
            TileAttr::Str(destination.memory_scope.clone()),
        ),
    ];
    attrs.extend(copy_async_tma_attrs(
        &facts,
        selected_dispatch,
        pair,
        &source,
        &destination,
    )?);
    if let Some(value) = facts.get("cta_group") {
        let cta_group = literal_int(value, "copy_async.cta_group")?;
        if cta_group != 1 && cta_group != 2 {
            return unsupported(format!(
                "TilePrimitiveCall(copy_async): cta_group must be 1 or 2, got {cta_group}"
            ));
        }
        attrs.push(("cta_group", TileAttr::Int(cta_group)));
    }
    for key in ["mbar", "cta_mask", "remote_cta_id"] {
        if let Some(value) = facts.get(key) {
            attrs.push((key, TileAttr::Value(value.clone())));
        }
    }
    if let Some(value) = facts.get("cta_mask") {
        attrs.push((
            "multicast",
            TileAttr::Bool(multicast_modifier_enabled(value)?),
        ));
    }
    attrs.extend(copy_async_cache_attrs(&facts, selected_dispatch)?);
    if let Some(value) = facts.get("prefetch_tensormap") {
        attrs.push(("prefetch_tensormap", TileAttr::Value(value.clone())));
    }
    if !gather4.is_empty() {
        attrs.push(("gather4", TileAttr::Values(gather4)));
    }
    if selected_dispatch == "tcgen05_cp" {
        validate_tcgen_cp_layout(node, &source, &destination)?;
    } else if selected_dispatch == "tcgen05_ldst" {
        let (tmem, local) = if destination.memory_scope == "tmem" {
            (&destination, &source)
        } else {
            (&source, &destination)
        };
        let validation = validate_tcgen_ldst_layout(ctx, tmem, local)?;
        attrs.push((
            "tmem_row_mode",
            TileAttr::Str(validation.row_mode.to_owned()),
        ));
        attrs.push(("tcgen_shape", TileAttr::Str(validation.shape.to_owned())));
        attrs.push(("tcgen_num", TileAttr::Int(validation.num)));
    }
    Ok(ParsedTileCall::new(
        TileOpKind::CopyAsync,
        &facts.scope,
        destination,
        vec![TileOperand::Region(source)],
        attrs,
    ))
}
