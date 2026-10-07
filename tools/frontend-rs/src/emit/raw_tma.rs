//! Validation and emission of the raw_tma instruction family.

use tvm::tirx::TensorMapTypeObj;

pub mod host_prelude;

use crate::analyze::buffers::call_op_name;
use crate::analyze::frontend::KernelPlan;
use crate::analyze::util::json_object;
use crate::analyze::util::{
    as_var, dtype_of, ffi_error, ffi_text, int_imm, not_covered, oref, repr_text, same,
    sorted_unique, unsupported, upper_first, AResult, Failure, Json,
};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::atomic_bulk::copy_report_pattern;
use crate::emit::memory_support::v2_memory_space_rust;
use crate::emit::memory_support::{shared_pointer_source, SharedAddressForms};
use crate::emit::{abi, ControlProvenance, Emitter, RustValue, Uniformity, Variables};
use crate::tables::is_integer_dtype;
use crate::tables::{json_string, phase_binding_name};
use tvm::ir::{Call, CallObj, Expr, Op, PointerTypeObj, PrimType, Var};
use tvm::prim::SelectObj;
use tvm::tirx::{BufferVar, PrimFunc};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore, String as FfiString};

pub struct TensorMapCode {
    pub variable: Var,
    pub name: String,
    pub binding_name: String,
    pub field: String,
}

pub struct State {
    pub uses_registry: bool,
    pub tensor_maps: Vec<TensorMapCode>,
}

impl State {
    pub fn new(
        plan: &KernelPlan,
        params: &[Var],
        kernel_index: i64,
        kernel_count: i64,
    ) -> AResult<Self> {
        let mut tensor_maps = Vec::new();
        for (index, entry) in plan.tensor_maps.iter().enumerate() {
            let Some(variable) = usize::try_from(entry.parameter_index)
                .ok()
                .and_then(|at| params.get(at))
            else {
                return not_covered("manifest parameter index is out of range");
            };
            tensor_maps.push(TensorMapCode {
                variable: variable.clone(),
                binding_name: phase_binding_name(kernel_index, &entry.name, kernel_count),
                name: entry.name.clone(),
                field: format!("tensor_map_{index}"),
            });
        }
        Ok(Self {
            uses_registry: plan.uses_raw_tensor_map_registry,
            tensor_maps,
        })
    }
}

impl Emitter<'_> {
    pub fn tensor_map_registry_ref(&self) -> AResult<String> {
        if !self.raw_tma.uses_registry {
            return unsupported("raw TensorMap registry was not declared by the semantic manifest");
        }
        Ok("buffers.tensor_map_registry".to_owned())
    }
}

pub const G2S_CLUSTER_CALLS: [&str; 3] = [
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast16",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast32",
];
/// `TMA_REPORT_BASE`.
const TMA_REPORT_BASE: [(&str, &str); 2] = [
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_report",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_report",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
];
/// `TMA_IM2COL_HINT_BASE`.
const TMA_IM2COL_HINT_BASE: [(&str, &str); 6] = [
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_im2col",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_address_im2col",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
];
/// `TMA_IM2COL_TRANSFER_BASE`.
const TMA_IM2COL_TRANSFER_BASE: [(&str, &str); 8] = [
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_im2col_no_offs_w",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_address_im2col_no_offs_w",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_im2col_no_offs_w",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_address_im2col_no_offs_w",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
];

/// `TMA_OVERRIDE_BASE`: instruction-local overrides share the corresponding
/// ordinary transfer/cache path, including the `_override_address` im2col
/// spellings of `TMA_IM2COL_HINT_BASE` and `TMA_IM2COL_TRANSFER_BASE`.
pub const TMA_OVERRIDE_BASE: [(&str, &str); 42] = [
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b8",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_b16",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b8",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_global_dim_stride_b16",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b8",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_b16",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b8",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_global_dim_stride_b16",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_address",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b8",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_b16",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b8",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_global_dim_stride_b16",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_address",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b8",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_b16",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b8",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_global_dim_stride_b16",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_b8",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_b16",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_b8",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_b16",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_evict_last_b8",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_evict_last_b16",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_evict_last_b8",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_global_dim_stride_evict_last_b16",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_address",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_b8",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_b16",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_stride_b8",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_global_dim_stride_b16",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_prefetch",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    ),
    (
        "tirx.ptx.applypriority_async_bulk_tensor_override_address_im2col",
        "tirx.ptx.applypriority_async_bulk_tensor",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    ),
    (
        "tirx.ptx.cp_async_bulk_tensor_s2g_override_address_im2col_no_offs_w",
        "tirx.ptx.cp_async_bulk_tensor_s2g",
    ),
    (
        "tirx.ptx.cp_reduce_async_bulk_tensor_override_address_im2col_no_offs_w",
        "tirx.ptx.cp_reduce_async_bulk_tensor",
    ),
];

fn table_base(table: &[(&str, &'static str)], op_name: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(name, _)| *name == op_name)
        .map(|(_, base)| *base)
}

/// `TMA_OVERRIDE_BASE.get(op_name)`.
pub fn override_base(op_name: &str) -> Option<&'static str> {
    table_base(&TMA_OVERRIDE_BASE, op_name)
}

/// `op_name in TMA_IM2COL_HINT_BASE`.
fn is_im2col_hint_call(op_name: &str) -> bool {
    table_base(&TMA_IM2COL_HINT_BASE, op_name).is_some()
}

pub fn tma_base(op_name: &str) -> &str {
    override_base(op_name)
        .or_else(|| table_base(&TMA_IM2COL_TRANSFER_BASE, op_name))
        .or_else(|| table_base(&TMA_IM2COL_HINT_BASE, op_name))
        .or_else(|| table_base(&TMA_REPORT_BASE, op_name))
        .unwrap_or(op_name)
}

pub const PTX_BULK_CACHE_HINT_CALLS: [&str; 3] = [
    "tirx.ptx.cp_async_bulk_prefetch",
    "tirx.ptx.cp_async_bulk_prefetch_evict_last",
    "tirx.ptx.applypriority_async_bulk",
];
const TENSOR_CACHE_HINT_BASES: [&str; 3] = [
    "tirx.ptx.cp_async_bulk_tensor_prefetch",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    "tirx.ptx.applypriority_async_bulk_tensor",
];

/// `op_name in PTX_TENSOR_CACHE_HINT_CALLS`: the bases, their im2col
/// spellings and their overrides.
pub fn is_tensor_cache_hint_call(name: &str) -> bool {
    TENSOR_CACHE_HINT_BASES.contains(&tma_base(name))
}

fn is_g2s(name: &str) -> bool {
    matches!(
        tma_base(name),
        "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"
            | "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast16"
            | "tirx.ptx.cp_async_bulk_tensor_g2s_cluster_multicast32"
            | "tirx.ptx.cp_async_bulk_tensor_g2s_cta"
    )
}

fn is_s2g(name: &str) -> bool {
    matches!(
        tma_base(name),
        "tirx.ptx.cp_async_bulk_tensor_s2g" | "tirx.ptx.cp_reduce_async_bulk_tensor"
    )
}

fn is_typed_tensor_map_var(variable: &Var) -> bool {
    variable
        .ty
        .as_node::<PointerTypeObj>()
        .is_some_and(|pointer| pointer.element_type.as_node::<TensorMapTypeObj>().is_some())
}

fn call_args(call: &CallObj) -> Vec<ObjectRef> {
    call.args.iter().map(oref).collect()
}

/// Branches of a typed descriptor selector, validated by the scalar emitter's rule.
pub fn tensor_map_selector_branches(
    ctx: &Ctx,
    expression: &ObjectRef,
) -> AResult<Option<(ObjectRef, ObjectRef, ObjectRef)>> {
    if let Some(select) = expression.as_node::<SelectObj>() {
        return Ok(Some((
            oref(select.condition.clone()),
            oref(select.true_value.clone()),
            oref(select.false_value.clone()),
        )));
    }
    let Some(call) = expression.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("prim.if_then_else") {
        return Ok(None);
    }
    match super::pure::prim_if_then_else_signature(ctx, expression, call) {
        Ok(_) => {
            let args = call_args(call);
            Ok(Some((args[0].clone(), args[1].clone(), args[2].clone())))
        }
        Err(Failure::Unsupported { .. } | Failure::Unmodeled { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

/// `tensor_map_parameters`: exact typed TensorMap parameters retained by an
/// address expression.
pub fn tensor_map_parameters(
    ctx: &Ctx,
    expression: &ObjectRef,
    params: Option<&[Var]>,
) -> AResult<Vec<Var>> {
    fn leaf(variable: &ObjectRef, params: Option<&[Var]>) -> Vec<Var> {
        let Some(variable) = as_var(variable) else {
            return Vec::new();
        };
        if !is_typed_tensor_map_var(&variable) {
            return Vec::new();
        }
        if let Some(params) = params {
            if !params.iter().any(|parameter| same(&variable, parameter)) {
                return Vec::new();
            }
        }
        vec![variable]
    }

    fn walk(ctx: &Ctx, node: &ObjectRef, params: Option<&[Var]>) -> AResult<Vec<Var>> {
        if let Some(call) = node.as_node::<CallObj>() {
            let name = call_op_name(call)?.unwrap_or_default();
            let signature = match name.as_str() {
                "tirx.address_of" => super::pure::address_of_signature(ctx, node, call),
                "tirx.reinterpret" => super::pure::reinterpret_signature(ctx, node, call),
                "prim.if_then_else" => super::pure::prim_if_then_else_signature(ctx, node, call),
                _ => return Ok(Vec::new()),
            };
            match signature {
                Ok(_) => {
                    let args = call_args(call);
                    if name == "tirx.address_of" {
                        return Ok(leaf(&args[0], params));
                    }
                    if name == "tirx.reinterpret" {
                        return walk(ctx, &args[0], params);
                    }
                }
                Err(Failure::Unsupported { .. } | Failure::Unmodeled { .. }) => {
                    return Ok(Vec::new())
                }
                Err(error) => return Err(error),
            }
        }
        let Some((_condition, true_value, false_value)) = tensor_map_selector_branches(ctx, node)?
        else {
            return Ok(Vec::new());
        };
        let true_parameters = walk(ctx, &true_value, params)?;
        let false_parameters = walk(ctx, &false_value, params)?;
        if true_parameters.is_empty() || false_parameters.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique: Vec<Var> = Vec::new();
        for parameter in true_parameters.iter().chain(false_parameters.iter()) {
            if !unique.iter().any(|existing| same(parameter, existing)) {
                unique.push(parameter.clone());
            }
        }
        Ok(unique)
    }

    walk(ctx, expression, params)
}

/// The family-wide validation of one decoded call.
fn validate(decoded: &DecodedPtx) -> AResult<()> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    if let Some(predicate) = &decoded.predicate {
        let dtype = dtype_of(predicate)?;
        if !(is_integer_dtype(&dtype) || dtype == "bool") {
            return unsupported(format!("{op_name} predicate must lower to bool or integer"));
        }
    }
    Ok(())
}

pub fn rank(decoded: &DecodedPtx) -> AResult<i64> {
    let op_name = decoded.op_name.as_str();
    let dim = decoded.modifier("dim")?;
    let chars: Vec<char> = dim.chars().collect();
    if chars.len() != 2 || chars[1] != 'd' || !chars[0].is_ascii_digit() {
        return unsupported(format!("{op_name}.dim has malformed token {:?}", dim));
    }
    let rank = chars[0].to_digit(10).expect("digit") as i64;
    if !(1..=5).contains(&rank) {
        return unsupported(format!("{op_name}.dim must be in 1d..5d, got {:?}", dim));
    }
    Ok(rank)
}

pub fn load_mode(decoded: &DecodedPtx) -> AResult<String> {
    let token = decoded.modifier("load_mode")?;
    Ok(if token.is_empty() {
        "tile".to_owned()
    } else {
        token.to_owned()
    })
}

fn present_operands(decoded: &DecodedPtx, name: &str) -> AResult<Vec<ObjectRef>> {
    let mut values = Vec::new();
    for value in decoded.operand(name)? {
        match value {
            Some(value) => values.push(value.clone()),
            None => {
                return Err(Failure::Ffi(crate::analyze::util::ffi_error(&format!(
                    "{} operand {name:?} has a sunk lane",
                    decoded.op_name
                ))))
            }
        }
    }
    Ok(values)
}

/// The table-owned literal of one immediate slot (`decoded.operand(name)` on
/// a literal slot is the literal text).
fn literal_operand<'d>(decoded: &'d DecodedPtx, name: &str) -> Option<&'d str> {
    decoded
        .operands
        .iter()
        .find(|slot| slot.name == name)
        .filter(|slot| slot.values.is_empty())
        .and_then(|slot| slot.literal.as_deref())
}

pub fn coordinates(decoded: &DecodedPtx) -> AResult<Vec<ObjectRef>> {
    let op_name = decoded.op_name.as_str();
    let coordinates = present_operands(decoded, "coords")?;
    let rank = rank(decoded)?;
    let load_mode = load_mode(decoded)?;
    let expected = if load_mode == "tile::gather4" || load_mode == "tile::scatter4" {
        5
    } else {
        rank
    };
    if coordinates.len() as i64 != expected {
        return unsupported(format!(
            "{op_name} {load_mode} expects {expected} coordinates, got {}",
            coordinates.len()
        ));
    }
    for (axis, coordinate) in coordinates.iter().enumerate() {
        let coordinate_dtype = dtype_of(coordinate)?;
        if matches!(
            tma_base(op_name),
            "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last"
                | "tirx.ptx.applypriority_async_bulk_tensor"
        ) {
            if coordinate_dtype != "int32" {
                return unsupported(format!(
                    "{op_name}.coords[{axis}] must be int32, got {:?}",
                    &coordinate_dtype
                ));
            }
        } else if !is_integer_dtype(&coordinate_dtype) {
            return unsupported(format!(
                "{op_name}.coords[{axis}] must be integer, got {:?}",
                &coordinate_dtype
            ));
        }
    }
    Ok(coordinates)
}

pub fn cache_hint(decoded: &DecodedPtx) -> AResult<bool> {
    let op_name = decoded.op_name.as_str();
    let enabled = !decoded.modifier("cache")?.is_empty();
    let operands = present_operands(decoded, "cache_policy")?;
    if operands.len() != usize::from(enabled) {
        return unsupported(format!(
            "{op_name}.cache modifier and cache_policy operand disagree"
        ));
    }
    if let Some(policy) = operands.first() {
        let dtype = dtype_of(policy)?;
        if dtype != "uint64" {
            return unsupported(format!(
                "{op_name}.cache_policy must be uint64, got {:?}",
                &dtype
            ));
        }
    }
    Ok(enabled)
}

pub fn tensor_map_expression(decoded: &DecodedPtx) -> AResult<ObjectRef> {
    let name = if decoded.op_name == "tirx.ptx.prefetch" {
        "addr"
    } else {
        "tmap"
    };
    decoded.scalar_operand(name)
}

fn decoded_tensor_map_parameters(
    ctx: &Ctx,
    decoded: &DecodedPtx,
    params: Option<&[Var]>,
) -> AResult<Vec<Var>> {
    let expression = tensor_map_expression(decoded)?;
    tensor_map_parameters(ctx, &expression, params)
}

/// `(parameters, "typed" | "raw")`.
pub fn tensor_map_transport(
    ctx: &Ctx,
    decoded: &DecodedPtx,
    params: Option<&[Var]>,
    allow_raw: bool,
) -> AResult<(Vec<Var>, &'static str)> {
    let parameters = decoded_tensor_map_parameters(ctx, decoded, params)?;
    if !parameters.is_empty() {
        return Ok((parameters, "typed"));
    }
    let expression = tensor_map_expression(decoded)?;
    if !allow_raw || dtype_of(&expression)? != "handle" {
        return unsupported(format!(
            "{} TensorMap operand must retain typed PrimFunc parameter identity{}, got {}",
            decoded.op_name,
            if allow_raw {
                " or a physical descriptor pointer"
            } else {
                ""
            },
            repr_text(&expression)?
        ));
    }
    Ok((Vec::new(), "raw"))
}

fn cache_hint_pointer_source(decoded: &DecodedPtx, operand_name: &str) -> AResult<ObjectRef> {
    let source = decoded.scalar_operand(operand_name)?;
    if dtype_of(&source)? != "handle" {
        return unsupported(format!(
            "{}.{operand_name} must retain a concrete source pointer",
            decoded.op_name
        ));
    }
    Ok(source)
}

/// `(source, size)`.
pub fn address_cache_hint_facts(
    ctx: &Ctx,
    decoded: &DecodedPtx,
) -> AResult<(ObjectRef, Option<i64>)> {
    validate(decoded)?;
    let op_name = decoded.op_name.as_str();
    let source = if op_name == "tirx.ptx.prefetch" {
        shared_pointer_source(
            &decoded.scalar_operand("addr")?,
            &format!("{op_name}.addr"),
            SharedAddressForms::tma(&ctx.analyzer, false, false),
        )?
        .0
    } else {
        cache_hint_pointer_source(decoded, "addr")?
    };
    let mut size = None;
    if op_name == "tirx.ptx.applypriority" {
        if literal_operand(decoded, "size") != Some("128") {
            return unsupported(format!("{op_name}.size must be the immediate 128"));
        }
        size = Some(128);
    }
    Ok((source, size))
}

/// `(source, size, cache_policy)`.
pub fn bulk_cache_hint_facts(
    ctx: &Ctx,
    decoded: &DecodedPtx,
) -> AResult<(ObjectRef, ObjectRef, Vec<ObjectRef>)> {
    validate(decoded)?;
    let op_name = decoded.op_name.as_str();
    let address_name = if op_name == "tirx.ptx.applypriority_async_bulk" {
        "addr"
    } else {
        "src_mem"
    };
    let source = if op_name == "tirx.ptx.cp_async_bulk_prefetch" {
        shared_pointer_source(
            &decoded.scalar_operand(address_name)?,
            &format!("{op_name}.{address_name}"),
            SharedAddressForms::tma(&ctx.analyzer, false, false),
        )?
        .0
    } else {
        cache_hint_pointer_source(decoded, address_name)?
    };
    let size = decoded.scalar_operand("size")?;
    let size_dtype = dtype_of(&size)?;
    if size_dtype != "uint32" {
        return unsupported(format!(
            "{op_name}.size must be uint32, got {:?}",
            &size_dtype
        ));
    }
    let mut cache_policy = Vec::new();
    if op_name == "tirx.ptx.cp_async_bulk_prefetch" {
        cache_hint(decoded)?;
        cache_policy = present_operands(decoded, "cache_policy")?;
    }
    Ok((source, size, cache_policy))
}

/// The tensor load modes of the gather transfers and cache hints.
fn is_load_mode(load_mode: &str) -> bool {
    matches!(
        load_mode,
        "tile" | "tile::gather4" | "im2col" | "im2col::w" | "im2col::w::128"
    )
}

/// The TensorMap operands of one tensor transfer or tensor cache hint.
pub struct TensorOperands {
    pub rank: i64,
    pub load_mode: String,
    pub coordinates: Vec<ObjectRef>,
    /// The exact TensorMap parameters; empty for a raw descriptor pointer.
    pub parameters: Vec<Var>,
}

pub struct G2sFacts {
    pub operands: TensorOperands,
    /// The `dst_mem` shared data pointer source.
    pub shared_source: ObjectRef,
    /// The `mbar` shared pointer source.
    pub barrier_source: ObjectRef,
    pub cta_group: i64,
    pub multicast: bool,
    pub report_pattern: i64,
}

pub fn g2s_facts(ctx: &Ctx, decoded: &DecodedPtx, params: Option<&[Var]>) -> AResult<G2sFacts> {
    let op_name = decoded.op_name.as_str();
    let rank = rank(decoded)?;
    let load_mode = load_mode(decoded)?;
    if load_mode == "tile::gather4" && rank != 2 {
        return unsupported(format!("{op_name} tile::gather4 requires dim=2d"));
    }
    if !is_load_mode(&load_mode) {
        return unsupported(format!(
            "{op_name} load mode {:?} is not modeled",
            &load_mode
        ));
    }
    if load_mode.starts_with("im2col") {
        for value in decoded.operand("im2col_info")?.iter().flatten() {
            if !is_integer_dtype(&dtype_of(value)?) {
                return unsupported(format!("{op_name} im2col operands must be integer"));
            }
        }
    }
    let coordinates = coordinates(decoded)?;
    let (parameters, _) = tensor_map_transport(ctx, decoded, params, true)?;
    let (shared_source, _) = shared_pointer_source(
        &decoded.scalar_operand("dst_mem")?,
        &format!("{op_name}.dst_mem"),
        SharedAddressForms::tma(&ctx.analyzer, true, false),
    )?;
    let (barrier_source, barrier_pair_base) = shared_pointer_source(
        &decoded.scalar_operand("mbar")?,
        &format!("{op_name}.mbar"),
        SharedAddressForms::tma(&ctx.analyzer, true, true),
    )?;
    let cta_group = decoded.cta_group(Some(1))?;
    if barrier_pair_base && cta_group != 2 {
        return unsupported(format!(
            "{op_name} SM100 two-SM mbarrier address requires cta_group::2"
        ));
    }
    let cluster = G2S_CLUSTER_CALLS.contains(&tma_base(op_name));
    let multicast = cluster && !decoded.modifier("multicast")?.is_empty();
    if cluster {
        let masks = present_operands(decoded, "cta_mask")?;
        if masks.len() != usize::from(multicast) {
            return unsupported(format!(
                "{op_name}.multicast modifier and cta_mask operand disagree"
            ));
        }
        if let Some(mask) = masks.first() {
            let dtype = dtype_of(mask)?;
            if !is_integer_dtype(&dtype) {
                return unsupported(format!(
                    "{op_name}.cta_mask must be integer, got {:?}",
                    &dtype
                ));
            }
        }
    }
    cache_hint(decoded)?;
    Ok(G2sFacts {
        operands: TensorOperands {
            rank,
            load_mode,
            coordinates,
            parameters,
        },
        shared_source,
        barrier_source,
        cta_group,
        multicast,
        report_pattern: copy_report_pattern(decoded)?,
    })
}

/// The parsed call of one TensorMap copy or tensor cache-hint instruction.
pub fn ptx_tma_call(
    ctx: &Ctx,
    decoded: &DecodedPtx,
    params: Option<&[Var]>,
) -> AResult<RawTmaCall> {
    validate(decoded)?;
    let op_name = decoded.op_name.as_str();
    override_facts(decoded)?;
    if is_g2s(op_name) {
        return Ok(RawTmaCall::G2s(g2s_facts(ctx, decoded, params)?));
    }
    if is_s2g(op_name) {
        let rank = rank(decoded)?;
        let load_mode = load_mode(decoded)?;
        if !matches!(
            load_mode.as_str(),
            "tile" | "im2col_no_offs" | "im2col_no_offs::w"
        ) {
            return unsupported(format!(
                "{op_name} load mode {:?} is not modeled",
                &load_mode
            ));
        }
        let coordinates = coordinates(decoded)?;
        let (parameters, _) = tensor_map_transport(ctx, decoded, params, true)?;
        let (shared_source, _) = shared_pointer_source(
            &decoded.scalar_operand("src_mem")?,
            &format!("{op_name}.src_mem"),
            SharedAddressForms::tma(&ctx.analyzer, true, false),
        )?;
        cache_hint(decoded)?;
        return Ok(RawTmaCall::S2g(
            TensorOperands {
                rank,
                load_mode,
                coordinates,
                parameters,
            },
            shared_source,
        ));
    }
    if is_tensor_cache_hint_call(op_name) {
        let rank = rank(decoded)?;
        let load_mode = load_mode(decoded)?;
        if load_mode == "tile::gather4" && rank != 2 {
            return unsupported(format!("{op_name} tile::gather4 requires dim=2d"));
        }
        if !is_load_mode(&load_mode) {
            return unsupported(format!(
                "{op_name} load mode {:?} is not modeled",
                &load_mode
            ));
        }
        let coordinates = coordinates(decoded)?;
        if is_im2col_hint_call(op_name) {
            // The schema owns im2col operand widths/counts. Values affect cache
            // residency only; applypriority still participates in the bulk group.
            for value in decoded.operand("im2col_info")?.iter().flatten() {
                if !is_integer_dtype(&dtype_of(value)?) {
                    return unsupported(format!("{op_name} im2col operand must be integer"));
                }
            }
        }
        let (parameters, _) = tensor_map_transport(ctx, decoded, params, true)?;
        if tma_base(op_name) == "tirx.ptx.cp_async_bulk_tensor_prefetch" {
            cache_hint(decoded)?;
        }
        return Ok(RawTmaCall::TensorCacheHint(TensorOperands {
            rank,
            load_mode,
            coordinates,
            parameters,
        }));
    }
    Err(Failure::Ffi(crate::analyze::util::ffi_error(&format!(
        "raw TMA resolution received {op_name}"
    ))))
}

fn override_facts(decoded: &DecodedPtx) -> AResult<Option<String>> {
    let op_name = decoded.op_name.as_str();
    if override_base(op_name).is_none() {
        return Ok(None);
    }
    if dtype_of(&decoded.scalar_operand("global_address")?)? != "uint64" {
        return unsupported(format!("{op_name} global address must be uint64"));
    }
    let attribute = decoded.modifier_or_empty("override_attribute");
    if !attribute.is_empty() && load_mode(decoded)? != "tile" {
        return unsupported(format!("{op_name} attribute override requires tile mode"));
    }
    Ok(Some(if attribute.is_empty() {
        "override::global_address".to_owned()
    } else {
        attribute.to_owned()
    }))
}

/// The parsed call of one cache-hint instruction.
pub fn ptx_cache_hint_call(
    ctx: &Ctx,
    decoded: &DecodedPtx,
    params: Option<&[Var]>,
) -> AResult<RawTmaCall> {
    validate(decoded)?;
    let op_name = decoded.op_name.as_str();
    if is_tensor_cache_hint_call(op_name) {
        return ptx_tma_call(ctx, decoded, params);
    }
    if PTX_BULK_CACHE_HINT_CALLS.contains(&op_name) {
        let (source, size, cache_policy) = bulk_cache_hint_facts(ctx, decoded)?;
        return Ok(RawTmaCall::BulkCacheHint(source, size, cache_policy));
    }
    if op_name == "tirx.ptx.prefetch" && !decoded.modifier("tensormap")?.is_empty() {
        let (parameters, _) = tensor_map_transport(ctx, decoded, params, false)?;
        return Ok(RawTmaCall::TensorMapPrefetch(parameters));
    }
    let (source, _) = address_cache_hint_facts(ctx, decoded)?;
    if !matches!(
        op_name,
        "tirx.ptx.prefetch"
            | "tirx.ptx.prefetch_valid_addr"
            | "tirx.ptx.prefetchu"
            | "tirx.ptx.applypriority"
    ) {
        return Err(Failure::Ffi(crate::analyze::util::ffi_error(&format!(
            "cache-hint resolution received {op_name}"
        ))));
    }
    Ok(RawTmaCall::AddressCacheHint(source))
}

pub struct TensorMapReplaceFacts {
    pub field: String,
    pub index: Option<i64>,
}

/// The replacement fields whose new value is an immediate.
fn is_immediate_replace_field(field: &str) -> bool {
    matches!(
        field,
        "elemtype" | "fill_mode" | "swizzle_mode" | "interleave_layout"
    )
}

/// `(field, ordinal)`.
pub fn tensor_map_replace_facts(decoded: &DecodedPtx) -> AResult<TensorMapReplaceFacts> {
    validate(decoded)?;
    crate::emit::register_call::require_register_call(decoded, false)?;
    let op_name = decoded.op_name.as_str();
    // The target table owns the entry's field domain. Carry the decoded field
    // through to the existing image update, including box_dim/element_stride.
    let field = decoded.modifier("field")?;
    let expected_type = if field == "global_address" || field == "global_stride" {
        "b64"
    } else {
        "b32"
    };
    let space = decoded.modifier("space")?;
    if decoded.modifier("mode")? != "tile"
        || !matches!(space, "" | "global" | "shared::cta")
        || decoded.modifier("width")? != "b1024"
        || decoded.modifier("type")? != expected_type
    {
        return unsupported(format!(
            "{op_name} requires tiled global/shared-CTA b1024.{expected_type} replacement"
        ));
    }
    let descriptor = decoded.scalar_operand("addr")?;
    let descriptor_dtype = dtype_of(&descriptor)?;
    if descriptor_dtype != "handle" && !(space == "shared::cta" && descriptor_dtype == "uint32") {
        return unsupported(format!(
            "{op_name}.addr must be a physical descriptor pointer"
        ));
    }
    let value = decoded.scalar_operand("new_val")?;
    if is_immediate_replace_field(field) && int_imm(&value).is_none() {
        return unsupported(format!("{op_name}.new_val must be an immediate"));
    }
    let mut index = None;
    if op_name == "tirx.ptx.tensormap_replace_dim" || op_name == "tirx.ptx.tensormap_replace_stride"
    {
        let ordinal = decoded.scalar_operand("ord")?;
        let Some(ordinal) = int_imm(&ordinal) else {
            return unsupported(format!("{op_name}.ord must be an immediate"));
        };
        let upper = if op_name == "tirx.ptx.tensormap_replace_dim" {
            4
        } else {
            3
        };
        if ordinal < 0 || ordinal > upper {
            return unsupported(format!(
                "{op_name}.ord must be in [0, {upper}], got {ordinal}"
            ));
        }
        index = Some(ordinal);
    }
    Ok(TensorMapReplaceFacts {
        field: field.to_owned(),
        index,
    })
}

/// The parsed call of one TensorMap replace instruction.
pub fn ptx_tensor_map_replace_call(decoded: &DecodedPtx) -> AResult<RawTmaCall> {
    Ok(RawTmaCall::TensorMapReplace(tensor_map_replace_facts(
        decoded,
    )?))
}

/// The fence operation and the shared data pointer
/// source of a copy-release.
pub fn tensor_map_fence_facts(
    ctx: &Ctx,
    decoded: &DecodedPtx,
) -> AResult<(&'static str, Option<ObjectRef>)> {
    validate(decoded)?;
    let op_name = decoded.op_name.as_str();
    if !matches!(
        decoded.modifier("scope")?,
        "cta" | "cluster" | "gpu" | "sys"
    ) {
        return unsupported(format!("{op_name} has invalid scope"));
    }
    if op_name == "tirx.ptx.tensormap_cp_fenceproxy" {
        for (name, value) in [
            ("dst", "global"),
            ("src", "shared::cta"),
            ("proxy", "tensormap::generic"),
            ("sem", "release"),
            ("sync", "sync"),
            ("aligned", "aligned"),
        ] {
            if decoded.modifier(name)? != value {
                return unsupported(
                    "tensormap.cp_fenceproxy requires global.shared::cta release.sync.aligned",
                );
            }
        }
        if dtype_of(&decoded.scalar_operand("dst_mem")?)? != "handle" {
            return unsupported("tensormap.cp_fenceproxy destination must be a physical pointer");
        }
        let (source, _) = shared_pointer_source(
            &decoded.scalar_operand("src_mem")?,
            "tensormap.cp_fenceproxy source",
            SharedAddressForms::tma(&ctx.analyzer, true, false),
        )?;
        if literal_operand(decoded, "size") != Some("128") {
            return unsupported("tensormap.cp_fenceproxy size must be 128");
        }
        return Ok(("copy_release", Some(source)));
    }
    let operation = if op_name == "tirx.ptx.fence_proxy_tensormap_acquire" {
        "acquire"
    } else {
        "release"
    };
    for (name, value) in [
        ("proxy", "proxy"),
        ("proxykind", "tensormap::generic"),
        ("sem", operation),
    ] {
        let actual = decoded.modifier(name)?;
        if actual != value {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                value, actual
            ));
        }
    }
    if op_name == "tirx.ptx.fence_proxy_tensormap_acquire" {
        if dtype_of(&decoded.scalar_operand("addr")?)? != "handle" {
            return unsupported(format!(
                "{op_name}.addr must be a physical descriptor pointer"
            ));
        }
        if literal_operand(decoded, "size") != Some("128") {
            return unsupported(format!("{op_name}.size must be the immediate 128"));
        }
    }
    Ok((operation, None))
}

/// The parsed call of one TensorMap fence instruction.
pub fn ptx_tensor_map_fence_call(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<RawTmaCall> {
    let (_, copy_source) = tensor_map_fence_facts(ctx, decoded)?;
    Ok(RawTmaCall::TensorMapFence(copy_source))
}

/// The parsed call of one raw TensorMap or cache-hint instruction.

/// A validated raw TMA call and the operand facts its resolution
/// parsed; its lowering and TensorMap bookkeeping read them.
pub enum RawTmaCall {
    G2s(G2sFacts),
    /// A scatter transfer and its `src_mem` shared data pointer source.
    S2g(TensorOperands, ObjectRef),
    TensorCacheHint(TensorOperands),
    /// `(source, size, cache_policy)` of a bulk prefetch or applypriority.
    BulkCacheHint(ObjectRef, ObjectRef, Vec<ObjectRef>),
    /// The source pointer of an address cache hint.
    AddressCacheHint(ObjectRef),
    /// The exact TensorMap parameters of a TensorMap `prefetch`.
    TensorMapPrefetch(Vec<Var>),
    TensorMapReplace(TensorMapReplaceFacts),
    /// The shared data pointer source of a copy-release fence.
    TensorMapFence(Option<ObjectRef>),
}

/// The contextual `tirx.address_of` of a typed TensorMap.

/// The contextual `tirx.address_of` path resolved for one call.

// ----------------------------------------------------------------------
// `analyze_primfunc` TensorMap bookkeeping.
// ----------------------------------------------------------------------

/// The per-call `raw_tma` bookkeeping of `analyze_primfunc.record`.
pub struct RawTmaCallRecord {
    /// `tensor_map_indices` additions (exact PrimFunc parameter positions).
    pub parameter_indices: Vec<i64>,
    pub uses_registry: bool,
}

/// Record the `raw_tma` facts of one resolved table call.

/// Record the `raw_tma` facts of one contextual TensorMap `address_of`; its
/// resolution with `params` decided that it addresses a TensorMap parameter.
pub fn record_address_call(
    ctx: &Ctx,
    node: &ObjectRef,
    params: &[Var],
) -> AResult<RawTmaCallRecord> {
    let parameters = tensor_map_parameters(ctx, node, Some(params))?;
    Ok(RawTmaCallRecord {
        parameter_indices: parameter_indices(&parameters, params),
        uses_registry: true,
    })
}

/// The PrimFunc positions of exact TensorMap parameters.
fn parameter_indices(parameters: &[Var], params: &[Var]) -> Vec<i64> {
    parameters
        .iter()
        .map(|parameter| {
            params
                .iter()
                .position(|candidate| same(candidate, parameter))
                .expect("exact PrimFunc parameter") as i64
        })
        .collect()
}

const IMPLICIT_TENSOR_MAP_ATTR: &str = "numsim.implicit_tensor_maps";
const IMPLICIT_METADATA_FIELDS: [&str; 14] = [
    "name",
    "base_buffer",
    "base_byte_offset",
    "dtype",
    "tma_dtype",
    "fp4_shared_layout",
    "global_shape",
    "global_strides",
    "box_shape",
    "element_strides",
    "interleave",
    "swizzle",
    "l2_promotion",
    "fill_mode",
];
const EXPRESSION_FIELDS: [&str; 5] = [
    "base_byte_offset",
    "global_shape",
    "global_strides",
    "box_shape",
    "element_strides",
];

/// Read and validate metadata produced by native host-prelude normalization.
pub fn implicit_tensor_map_metadata(func: &PrimFunc) -> AResult<Vec<Json>> {
    let Some(attribute) = func
        .attrs
        .dict
        .get(&FfiString::from(IMPLICIT_TENSOR_MAP_ATTR))?
    else {
        return Ok(Vec::new());
    };
    let Ok(raw) = FfiString::try_from(attribute) else {
        return unsupported("NumSim implicit TensorMap metadata is malformed");
    };
    let Ok(payload) = serde_json::from_str::<Json>(&ffi_text(&raw)) else {
        return unsupported("NumSim implicit TensorMap metadata is malformed");
    };
    let Json::Array(items) = payload else {
        return unsupported("NumSim implicit TensorMap metadata must be a list of objects");
    };
    if !items.iter().all(|item| matches!(item, Json::Object(_))) {
        return unsupported("NumSim implicit TensorMap metadata must be a list of objects");
    }
    for item in &items {
        let Json::Object(fields) = item else {
            unreachable!()
        };
        let mut keys: Vec<&str> = fields.iter().map(|(key, _)| key.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        let mut required: Vec<&str> = IMPLICIT_METADATA_FIELDS.to_vec();
        required.sort_unstable();
        if keys != required {
            return unsupported("NumSim implicit TensorMap metadata has missing or unknown fields");
        }
        for field in EXPRESSION_FIELDS {
            let value = item.get(field).expect("required field");
            let serialized = if field == "base_byte_offset" {
                matches!(value, Json::String(_))
            } else {
                value.as_array().is_some_and(|values| {
                    values.iter().all(|value| matches!(value, Json::String(_)))
                })
            };
            if !serialized {
                return unsupported(format!(
                    "NumSim implicit TensorMap metadata field {field} must contain serialized TIR expressions"
                ));
            }
        }
    }
    Ok(items)
}

/// A TensorMap PrimFunc parameter as emission binds it.
pub struct TensorMapParameter {
    pub name: String,
    pub parameter_index: i64,
    /// Whether its metadata names a `base_buffer` (an implicit TensorMap).
    pub implicit: bool,
}

/// `TensorMapSpec` manifest entries for `sorted(tensor_map_indices)`.
pub fn tensor_map_specs(
    metadata: &[Json],
    params: &[Var],
    indices: &[i64],
) -> (Vec<Json>, Vec<TensorMapParameter>) {
    let mut sorted: Vec<i64> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut specs = Vec::new();
    let mut parameters = Vec::new();
    for index in sorted {
        let name = ffi_text(&params[index as usize].name);
        let implicit = metadata
            .iter()
            .find(|item| item.get("name").and_then(Json::as_str) == Some(name.as_str()));
        let field = |key: &str, default: Json| -> Json {
            implicit
                .and_then(|item| item.get(key).cloned())
                .unwrap_or(default)
        };
        let base_buffer = field("base_buffer", Json::Null);
        parameters.push(TensorMapParameter {
            name: name.clone(),
            parameter_index: index,
            implicit: !matches!(base_buffer, Json::Null),
        });
        specs.push(json_object(vec![
            ("name", Json::String(name.clone())),
            ("parameter_index", Json::from(index)),
            ("base_buffer", base_buffer),
            ("base_byte_offset", field("base_byte_offset", Json::from(0))),
            ("dtype", field("dtype", Json::Null)),
            ("tma_dtype", field("tma_dtype", Json::Null)),
            ("fp4_shared_layout", field("fp4_shared_layout", Json::Null)),
            (
                "global_shape",
                field("global_shape", Json::Array(Vec::new())),
            ),
            (
                "global_strides",
                field("global_strides", Json::Array(Vec::new())),
            ),
            ("box_shape", field("box_shape", Json::Array(Vec::new()))),
            (
                "element_strides",
                field("element_strides", Json::Array(Vec::new())),
            ),
            ("interleave", field("interleave", Json::Null)),
            ("swizzle", field("swizzle", Json::Null)),
            ("l2_promotion", field("l2_promotion", Json::Null)),
            ("fill_mode", field("fill_mode", Json::Null)),
        ]));
    }
    (specs, parameters)
}

/// The scalar PrimFunc parameters (`not is_buffer_var and not PointerType`).
pub fn scalar_parameters(func: &PrimFunc) -> Vec<Var> {
    func.params
        .iter()
        .filter(|parameter| {
            BufferVar::try_from(parameter).is_err()
                && parameter.ty.as_node::<PointerTypeObj>().is_none()
        })
        .collect()
}

/// Deserialize an integer expression and rebind its parameters. The caller
/// applies `validate_integer_expression` with its own variable policy.
pub fn deserialize_integer_expr(
    func: &PrimFunc,
    payload: &Json,
    field: &str,
) -> AResult<ObjectRef> {
    let Json::String(text) = payload else {
        return unsupported(format!(
            "host TensorMap {field} must contain serialized TIR, got {}",
            match payload {
                Json::Null => "NoneType",
                Json::Bool(_) => "bool",
                Json::Number(value) if value.is_i64() || value.is_u64() => "int",
                Json::Number(_) => "float",
                Json::Array(_) => "list",
                Json::Object(_) => "dict",
                Json::String(_) => unreachable!(),
            }
        ));
    };
    let loaded: Result<tvm::tvm_ffi::Any, _> =
        tvm::tvm_ffi::cached_global_func!("ffi.FromJSONGraphString")
            .call_tuple((FfiString::from(text.as_str()),));
    let expression = match loaded.and_then(ObjectRef::try_from) {
        Ok(expression) => expression,
        Err(_) => {
            return unsupported(format!(
                "host TensorMap {field} contains malformed serialized TIR"
            ))
        }
    };
    let scalars = scalar_parameters(func);
    let mut replacements: Vec<(Var, tvm::ir::Expr)> = Vec::new();
    for node in crate::post_order_nodes(expression.clone())?.iter() {
        let Some(variable) = as_var(&node) else {
            continue;
        };
        let name = ffi_text(&variable.name);
        let dtype = crate::analyze::util::expr_type(&node)
            .and_then(|ty| crate::analyze::util::prim_dtype(&ty))
            .map(crate::analyze::util::dtype_text)
            .unwrap_or_default();
        let matches: Vec<&Var> = scalars
            .iter()
            .filter(|parameter| {
                ffi_text(&parameter.name) == name
                    && crate::analyze::util::prim_dtype(&parameter.ty)
                        .map(crate::analyze::util::dtype_text)
                        .unwrap_or_default()
                        == dtype
            })
            .collect();
        if matches.len() != 1 {
            return unsupported(format!(
                "host TensorMap {field} scalar {:?} with dtype {:?} resolves to {} PrimFunc parameters",
                &name,
                &dtype,
                matches.len()
            ));
        }
        replacements.push((variable, matches[0].clone().into()));
    }
    Ok(crate::substitute(
        expression,
        replacements.into_iter().collect(),
    )?)
}

/// `host_abi.HostAbiContract.implicit_tensor_maps`: the base host binding of
/// one implicit TensorMap from its kernel's parameters.
pub fn implicit_base_canonical_name(
    plan: &crate::analyze::frontend::KernelPlan,
    kernel_index: i64,
    kernel_count: i64,
    base_buffer: &str,
) -> AResult<String> {
    let mut targets: Vec<String> = Vec::new();
    // A buffer parameter's manifest `parameter` is its name.
    for name in &plan.buffer_names {
        if name == base_buffer {
            targets.push(crate::tables::phase_binding_name(
                kernel_index,
                name,
                kernel_count,
            ));
        }
    }
    for pointer in &plan.pointers {
        let name = pointer.name.as_str();
        if name == base_buffer {
            targets.push(crate::tables::phase_binding_name(
                kernel_index,
                name,
                kernel_count,
            ));
        }
    }
    let targets = sorted_unique(targets);
    if targets.len() != 1 {
        return unsupported(format!(
            "implicit TensorMap base {:?} resolves to {} host buffers, expected one",
            base_buffer,
            targets.len()
        ));
    }
    Ok(targets[0].clone())
}

fn reduction_marker(reduction: &str) -> Option<&'static str> {
    Some(match reduction {
        "add" => "v2::async_copy::variant::ReduceAdd",
        "min" => "v2::async_copy::variant::ReduceMin",
        "max" => "v2::async_copy::variant::ReduceMax",
        "inc" => "v2::async_copy::variant::ReduceInc",
        "dec" => "v2::async_copy::variant::ReduceDec",
        "and" => "v2::async_copy::variant::ReduceAnd",
        "or" => "v2::async_copy::variant::ReduceOr",
        "xor" => "v2::async_copy::variant::ReduceXor",
        _ => return None,
    })
}

fn require_physical_pointer(value: &RustValue, message: &str) -> AResult<()> {
    if value.rust_type != "PhysicalPtr" {
        return unsupported(message);
    }
    Ok(())
}

fn shared_address(value: &RustValue) -> String {
    abi::address("v2::Shared", &abi::cloned(&value.code), None)
}

/// The rendered warp call of one statement site.
fn stateful(
    function: &str,
    site: &str,
    arguments: &str,
    variant: Option<&str>,
    context: Option<&str>,
) -> String {
    abi::warp_call(
        function,
        site,
        &[arguments.to_owned()],
        variant,
        context,
        false,
        true,
    )
}

/// `tvm.tirx.reinterpret(dtype, value)`: the value itself when its type
/// already matches, otherwise a fresh reinterpret call node.
pub(super) fn reinterpret(dtype: &str, value: &ObjectRef) -> AResult<ObjectRef> {
    if dtype_of(value)? == dtype {
        return Ok(value.clone());
    }
    let call = Call::new(
        PrimType::new(dtype)?,
        Op::get("tirx.reinterpret")?,
        vec![Expr::try_from(Any::from(value.clone()))?],
    );
    Ok(oref(call))
}

/// The snapshotted override address and the
/// per-lane registers of each override operand group.
struct TensorMapOverride {
    address: String,
    tensor_size: Vec<String>,
    lower_stride: Vec<String>,
    coords: Vec<String>,
    upper_stride: Vec<String>,
}

impl<'a> Emitter<'a> {
    fn emit_raw_tma_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &str,
        variant: Option<&str>,
        context: Option<&str>,
    ) {
        let site = self.v2_site(Some(source_op_id));
        let invocation = stateful(function, &site, arguments, variant, context);
        self.emit_line(&format!("{invocation};"));
    }

    fn raw_tma_global_pointer(&mut self, expression: &ObjectRef) -> AResult<RustValue> {
        let value = self.emit_expr(expression)?;
        if value.rust_type != "PhysicalPtr" {
            return self.emit_raw_generic_pointer(value, "ctx.active_mask()");
        }
        Ok(value)
    }

    fn raw_tma_condition_code(
        &mut self,
        expression: &ObjectRef,
        label: &str,
        active_mask: &str,
        require_uniform: bool,
    ) -> AResult<String> {
        let mut value = self.emit_expr(expression)?;
        if value.rust_type != "bool" {
            return unsupported(format!("{label} must be boolean"));
        }
        if value.uniformity == Uniformity::Uniform {
            return Ok(value.code);
        }
        if !value.is_mask {
            let lane = self.boolean_lane(&value)?;
            value = self.emit_varying_mask(&lane, ControlProvenance::None);
        }
        if require_uniform {
            let selected = self.control_name("raw_tma_selector_selected");
            let result = self.control_name("raw_tma_selector_condition");
            self.emit_line(&format!("let {selected} = {} & {active_mask};", value.code));
            self.emit_line(&format!(
                "if !{selected}.is_empty() && {selected} != {active_mask} {{"
            ));
            self.indent += 1;
            self.emit_line(&format!(
                "return Err(EngineError::message(\"{label} must be uniform across issuing lanes\"));"
            ));
            self.indent -= 1;
            self.emit_line("}");
            self.emit_line(&format!("let {result} = !{selected}.is_empty();"));
            return Ok(result);
        }
        let first = self.control_name("raw_tma_selector_lane");
        let result = self.control_name("raw_tma_selector_condition");
        self.emit_line(&format!(
            "let {first} = {active_mask}.first_active().ok_or_else(|| EngineError::message(\"{label} has no active issuing lane\"))?;"
        ));
        self.emit_line(&format!("let {result} = {}.contains({first});", value.code));
        Ok(result)
    }

    fn raw_tma_tensor_map_ref(&self, variable: &Var) -> AResult<String> {
        for tensor_map in &self.raw_tma.tensor_maps {
            if same(&tensor_map.variable, variable) {
                return Ok(format!("buffers.{}", tensor_map.field));
            }
        }
        unsupported(format!(
            "TensorMap parameter {:?} is missing from the semantic manifest by exact parameter identity",
            &ffi_text(&variable.name)))
    }

    #[allow(clippy::too_many_arguments)]
    fn raw_tma_tensor_map(
        &mut self,
        expression: &ObjectRef,
        expected_parameters: &[Var],
        expected_rank: Option<i64>,
        active_mask: &str,
        require_uniform_selector: bool,
        source_op_id: i64,
    ) -> AResult<String> {
        if let Some(call) = expression.as_node::<CallObj>() {
            if call_op_name(call)?.as_deref() == Some("tirx.reinterpret") && call.args.len() == 1 {
                let inner = oref(call.args.get(0)?);
                return self.raw_tma_tensor_map(
                    &inner,
                    expected_parameters,
                    expected_rank,
                    active_mask,
                    require_uniform_selector,
                    source_op_id,
                );
            }
        }
        if let Some((condition_expression, true_expression, false_expression)) =
            tensor_map_selector_branches(self.ctx, expression)?
        {
            if expected_parameters.is_empty() {
                return unsupported("raw TensorMap descriptor selectors are not modeled");
            }
            let condition = self.raw_tma_condition_code(
                &condition_expression,
                "raw TMA TensorMap selector",
                active_mask,
                require_uniform_selector,
            )?;
            let true_map = self.raw_tma_tensor_map(
                &true_expression,
                expected_parameters,
                expected_rank,
                active_mask,
                require_uniform_selector,
                source_op_id,
            )?;
            let false_map = self.raw_tma_tensor_map(
                &false_expression,
                expected_parameters,
                expected_rank,
                active_mask,
                require_uniform_selector,
                source_op_id,
            )?;
            let result = self.control_name("raw_tma_tensor_map");
            self.emit_line(&format!(
                "let {result} = if {condition} {{ {true_map} }} else {{ {false_map} }};"
            ));
            return Ok(result);
        }
        if !expected_parameters.is_empty() && expression.as_node::<CallObj>().is_some() {
            let parameters = tensor_map_parameters(self.ctx, expression, Some(&self.params))?;
            if parameters.len() != 1
                || !expected_parameters
                    .iter()
                    .any(|expected| same(&parameters[0], expected))
            {
                return unsupported(
                    "raw TMA descriptor address lost its classified TensorMap parameter identity",
                );
            }
            return Ok(format!(
                "v2_tensor_map({}.clone())",
                self.raw_tma_tensor_map_ref(&parameters[0])?
            ));
        }
        if !expected_parameters.is_empty() {
            return unsupported(format!(
                "raw TMA TensorMap selector is not implemented for {}",
                crate::analyze::util::kind(expression).unwrap_or("Object")
            ));
        }
        let Some(expected_rank) = expected_rank else {
            return unsupported("raw TensorMap descriptor lookup requires an instruction rank");
        };
        let mut pointer = self.emit_expr(expression)?;
        if pointer.rust_type != "PhysicalPtr" {
            pointer = self.emit_raw_generic_pointer(pointer, "ctx.active_mask()")?;
        }
        require_physical_pointer(
            &pointer,
            "raw TensorMap descriptor did not resolve to a physical address",
        )?;
        let descriptor = self.control_name("raw_tma_descriptor_pointer");
        self.emit_line(&format!("let {descriptor} = ({}).clone();", pointer.code));
        self.raw_tma_lookup_tensor_map(&descriptor, expected_rank, active_mask, source_op_id)
    }

    fn raw_tma_lookup_tensor_map(
        &mut self,
        descriptor: &str,
        rank: i64,
        active_mask: &str,
        source_op_id: i64,
    ) -> AResult<String> {
        let resolved = self.control_name("raw_tma_resolved_tensor_map");
        let registry = self.tensor_map_registry_ref()?;
        self.emit_line(&format!(
            "let {resolved} = {registry}.lookup(warp, v2_context(ctx.with_active_mask({active_mask})), {}, &{descriptor}, {rank}_usize)?;",
            abi::site(source_op_id as u64)
        ));
        Ok(format!("v2_tensor_map({resolved})"))
    }

    fn raw_tma_addressed_tensor_map(
        &mut self,
        decoded: &DecodedPtx,
        parameters: &[Var],
        active_mask: &str,
        source_op_id: i64,
    ) -> AResult<String> {
        let expression = tensor_map_expression(decoded)?;
        let tensor_map = self.raw_tma_tensor_map(
            &expression,
            parameters,
            None,
            active_mask,
            false,
            source_op_id,
        )?;
        let addressed = self.control_name("raw_tma_addressed_tensor_map");
        self.emit_line(&format!("let {addressed} = {tensor_map};"));
        Ok(addressed)
    }

    fn emit_raw_cache_hint(
        &mut self,
        decoded: &DecodedPtx,
        call: &RawTmaCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_cache_hint",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let context = region.context.clone();
        let result = (|| -> AResult<()> {
            if let RawTmaCall::BulkCacheHint(source, size, cache_policy) = call {
                let pointer = self.raw_tma_global_pointer(source)?;
                require_physical_pointer(
                    &pointer,
                    &format!("{op_name} address did not resolve to a physical address"),
                )?;
                let size_value = self.emit_expr(size)?;
                let size_value = self.as_warp_value(size_value);
                if size_value.rust_type != "u32" {
                    return unsupported(format!(
                        "{op_name} size lowered to {}, expected u32",
                        size_value.rust_type
                    ));
                }
                for operand in cache_policy {
                    let value = self.emit_expr(operand)?;
                    self.emit_line(&format!("let _ = {};", value.code));
                }
                let arguments = format!(
                    "({}, {})",
                    abi::address("v2::Global", &abi::cloned(&pointer.code), None),
                    abi::register(&size_value.code)
                );
                if op_name == "tirx.ptx.applypriority_async_bulk" {
                    self.emit_raw_tma_stateful(
                        "async_copy::applypriority",
                        source_op_id,
                        &arguments,
                        Some("v2::async_copy::variant::BulkApplyPriority"),
                        context.as_deref(),
                    );
                } else {
                    self.emit_raw_tma_stateful(
                        "async_copy::cp_async_bulk_prefetch",
                        source_op_id,
                        &arguments,
                        None,
                        context.as_deref(),
                    );
                }
                return Ok(());
            }
            let RawTmaCall::AddressCacheHint(source) = call else {
                return Err(Failure::Ffi(ffi_error(&format!(
                    "cache-hint emitter received {op_name}"
                ))));
            };
            let pointer = self.raw_tma_global_pointer(source)?;
            require_physical_pointer(
                &pointer,
                &format!("{op_name} address did not resolve to a physical address"),
            )?;
            if op_name == "tirx.ptx.prefetchu" {
                let marker = self.control_name("ptx_prefetch_address");
                self.emit_line(&format!("let {marker} = ({}).clone();", pointer.code));
                self.emit_line(&format!("let _ = {marker};"));
                return Ok(());
            }
            let argument = abi::address("v2::Global", &abi::cloned(&pointer.code), None);
            if op_name == "tirx.ptx.prefetch_valid_addr" {
                self.emit_raw_tma_stateful(
                    "async_copy::prefetch_valid_address",
                    source_op_id,
                    &argument,
                    None,
                    context.as_deref(),
                );
            } else {
                self.emit_raw_tma_stateful(
                    "async_copy::applypriority",
                    source_op_id,
                    &argument,
                    Some("v2::async_copy::variant::ApplyPriority"),
                    context.as_deref(),
                );
            }
            Ok(())
        })();
        self.close_predicated_region(region);
        result
    }

    /// Prefetch of a TensorMap.
    fn emit_raw_tensor_map_prefetch(
        &mut self,
        decoded: &DecodedPtx,
        parameters: &[Var],
        source_op_id: i64,
    ) -> AResult<()> {
        let addressed = self.raw_tma_addressed_tensor_map(
            decoded,
            parameters,
            "ctx.active_mask()",
            source_op_id,
        )?;
        self.emit_raw_tma_stateful(
            "async_copy::prefetch_tensormap",
            source_op_id,
            &addressed,
            None,
            None,
        );
        Ok(())
    }

    /// Prefetch of an address.
    fn emit_raw_prefetch(&mut self, decoded: &DecodedPtx, source: &ObjectRef) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let space = decoded.modifier("space")?.to_owned();
        let pointer = self.emit_address_pointer(source, &space, None, "ctx.active_mask()")?;
        require_physical_pointer(
            &pointer,
            &format!("{op_name} address did not resolve to a physical address"),
        )?;
        let marker = self.control_name("ptx_prefetch_address");
        self.emit_line(&format!("let {marker} = ({}).clone();", pointer.code));
        self.emit_line(&format!("let _ = {marker};"));
        Ok(())
    }

    fn emit_raw_tensor_cache_hint(
        &mut self,
        decoded: &DecodedPtx,
        operands: &TensorOperands,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let base = tma_base(&op_name);
        let marker = match base {
            "tirx.ptx.cp_async_bulk_tensor_prefetch" => "TensorPrefetch",
            "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last" => "TensorPrefetchEvictLast",
            "tirx.ptx.applypriority_async_bulk_tensor" => "TensorApplyPriority",
            _ => {
                return Err(Failure::Ffi(ffi_error(&format!(
                    "tensor cache-hint emitter received {op_name}"
                ))))
            }
        };
        let rank = operands.rank;
        let specialization = if operands.load_mode == "tile::gather4" {
            format!("v2::async_copy::variant::{marker}Gather4")
        } else {
            format!("v2::async_copy::variant::{marker}<{rank}>")
        };
        let instruction = if base == "tirx.ptx.applypriority_async_bulk_tensor" {
            "applypriority"
        } else {
            "cp_async_bulk_prefetch_tensor"
        };
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_tensor_cache_hint",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let context = region.context.clone();
        let issue_mask = region.mask.clone();
        let result = (|| -> AResult<()> {
            self.raw_tma_cache_hint_operands(decoded)?;
            let coordinates = self.raw_tma_coordinate_code(&operands.coordinates)?;
            self.raw_tma_tensor_map_regions(
                decoded,
                &operands.parameters,
                rank,
                context.as_deref(),
                &issue_mask,
                source_op_id,
                |emitter, map_context, addressed| {
                    emitter.emit_raw_tma_stateful(
                        &format!("async_copy::{instruction}"),
                        source_op_id,
                        &format!("({addressed}, {coordinates})"),
                        Some(&specialization),
                        map_context,
                    );
                    Ok(())
                },
            )
        })();
        self.close_predicated_region(region);
        result
    }

    /// Evaluate runtime register inputs; they
    /// select cache regions, not numerical memory accesses. There is no
    /// additional engine state/ABI.
    fn raw_tma_cache_hint_operands(&mut self, decoded: &DecodedPtx) -> AResult<()> {
        for field in ["im2col_info", "cache_policy"] {
            if !decoded.has_operand(field) {
                continue;
            }
            for operand in decoded.operand(field)?.iter().flatten() {
                let value = self.emit_expr(operand)?;
                self.emit_line(&format!("let _ = {};", value.code));
            }
        }
        Ok(())
    }

    /// `emit_raw_generic_pointer` on an emitted expression, then
    /// `require_physical_pointer`, bound to a fresh control name.
    fn raw_tma_descriptor_binding(
        &mut self,
        expression: &ObjectRef,
        message: &str,
        binding: &str,
    ) -> AResult<String> {
        let mut value = self.emit_expr(expression)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_generic_pointer(value, "ctx.active_mask()")?;
        }
        require_physical_pointer(&value, message)?;
        let name = self.control_name(binding);
        self.emit_line(&format!("let {name} = ({}).clone();", value.code));
        Ok(name)
    }

    fn emit_raw_tensor_map_replace(
        &mut self,
        decoded: &DecodedPtx,
        facts: &TensorMapReplaceFacts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let registry = self.tensor_map_registry_ref()?;

        let space_token = decoded.modifier("space")?.to_owned();
        let descriptor_value = self.emit_address_pointer(
            &decoded.scalar_operand("addr")?,
            &space_token,
            None,
            "ctx.active_mask()",
        )?;
        require_physical_pointer(
            &descriptor_value,
            &format!("{op_name} descriptor did not resolve to a physical address"),
        )?;
        let descriptor = self.control_name("tensor_map_replace_descriptor");
        self.emit_line(&format!(
            "let {descriptor} = ({}).clone();",
            descriptor_value.code
        ));
        let space = v2_memory_space_rust(&space_token)?;
        let destination = abi::address(space, &descriptor, None);

        if op_name == "tirx.ptx.tensormap_replace_address" {
            let operand = reinterpret("uint64", &decoded.scalar_operand("new_val")?)?;
            let address_value = self.raw_tma_global_pointer(&operand)?;

            let address = self.control_name("tensor_map_replace_global_address");
            self.emit_line(&format!(
                "let {address} = ({}).clone();",
                address_value.code
            ));
            self.emit_raw_tma_stateful(
                "sync::tensormap_replace",
                source_op_id,
                &format!(
                    "({destination}, {registry}.clone(), {})",
                    abi::address("v2::Global", &address, None)
                ),
                Some(&format!("v2::sync::variant::TensorMapAddress<{space}>")),
                None,
            );
            return Ok(());
        }

        // PTX .b32/.b64 carriers encode bits, not numeric float values.
        let carrier = format!("uint{}", decoded.modifier("type")?.get(1..).unwrap_or(""));
        let operand = reinterpret(&carrier, &decoded.scalar_operand("new_val")?)?;
        let value = self.v2_register_operand(&operand, "u64", "tensor_map_replace_value")?;
        let ordinal = match facts.index {
            None => "None".to_owned(),
            Some(index) => format!("Some({index}_usize)"),
        };
        self.emit_raw_tma_stateful(
            "sync::tensormap_replace",
            source_op_id,
            &format!(
                "({destination}, {registry}.clone(), {}, {ordinal}, {value})",
                json_string(&facts.field)
            ),
            Some(&format!("v2::sync::variant::TensorMapField<{space}>")),
            None,
        );
        Ok(())
    }

    fn emit_raw_tensor_map_fence(
        &mut self,
        decoded: &DecodedPtx,
        copy_source: Option<&ObjectRef>,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let registry = self.tensor_map_registry_ref()?;
        let scope_name = upper_first(decoded.modifier("scope")?);
        let scope = format!("MemoryScope::{scope_name}");
        if op_name == "tirx.ptx.tensormap_cp_fenceproxy" {
            let mut addresses = Vec::new();
            for (name, space) in [("dst_mem", "Global"), ("src_mem", "SharedCta")] {
                let expression = if name == "src_mem" {
                    copy_source.expect("validated copy-release source").clone()
                } else {
                    decoded.scalar_operand(name)?
                };
                let value = self.emit_address_pointer(
                    &expression,
                    if name == "src_mem" {
                        "shared::cta"
                    } else {
                        "global"
                    },
                    None,
                    "ctx.active_mask()",
                )?;
                require_physical_pointer(
                    &value,
                    &format!("tensormap.cp_fenceproxy {name} lost pointer provenance"),
                )?;
                addresses.push(abi::address(
                    &format!("v2::{space}"),
                    &abi::cloned(&value.code),
                    None,
                ));
            }
            let args = format!("({}, {registry}.clone())", addresses.join(", "));
            self.emit_raw_tma_stateful(
                "sync::tensormap_cp_fenceproxy",
                source_op_id,
                &args,
                Some(&format!(
                    "v2::sync::variant::TensorMapCopy<v2::sync::variant::{scope_name}>"
                )),
                None,
            );
            return Ok(());
        }
        let site = abi::site(source_op_id as u64);
        if op_name == "tirx.ptx.fence_proxy_tensormap_acquire" {
            let descriptor = self.raw_tma_descriptor_binding(
                &decoded.scalar_operand("addr")?,
                &format!("{op_name} descriptor did not resolve to a physical address"),
                "tensor_map_acquire_descriptor",
            )?;
            self.emit_line(&format!(
                "{registry}.acquire(warp, v2_context(ctx), {site}, &{descriptor}, {scope})?;"
            ));
            return Ok(());
        }
        self.emit_line(&format!(
            "{registry}.release(warp, v2_context(ctx), {site}, {scope})?;"
        ));
        Ok(())
    }

    /// Snapshot the source selector tree before any
    /// transfer can write memory.
    fn raw_tma_descriptor_selections(
        &mut self,
        expression: &ObjectRef,
        active_mask: &str,
    ) -> AResult<Vec<(String, ObjectRef)>> {
        if let Some(call) = expression.as_node::<CallObj>() {
            if call_op_name(call)?.as_deref() == Some("tirx.reinterpret") {
                let inner = oref(call.args.get(0)?);
                return self.raw_tma_descriptor_selections(&inner, active_mask);
            }
        }
        let Some((condition, true_expression, false_expression)) =
            tensor_map_selector_branches(self.ctx, expression)?
        else {
            return Ok(vec![(active_mask.to_owned(), expression.clone())]);
        };
        let selected = self.control_name("tma_selected_lanes");
        let rejected = self.control_name("tma_other_lanes");
        self.emit_line(&format!("let mut {selected} = WarpMask::EMPTY;"));
        self.emit_line(&format!("if !{active_mask}.is_empty() {{"));
        self.indent += 1;
        self.emit_line(&format!("let ctx = ctx.with_active_mask({active_mask});"));
        let mask = self.instruction_predicate_mask(
            Some(&condition),
            "tma_selector",
            "TensorMap selector must be boolean or integer",
        )?;
        self.emit_line(&format!("{selected} = {mask};"));
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line(&format!(
            "let {rejected} = WarpMask::from_bits({active_mask}.bits() & !{selected}.bits());"
        ));
        let mut selections = self.raw_tma_descriptor_selections(&true_expression, &selected)?;
        selections.extend(self.raw_tma_descriptor_selections(&false_expression, &rejected)?);
        Ok(selections)
    }

    /// Lower `body` once per descriptor region, with
    /// the region's context and resolved TensorMap.
    #[allow(clippy::too_many_arguments)]
    fn raw_tma_tensor_map_regions<F>(
        &mut self,
        decoded: &DecodedPtx,
        parameters: &[Var],
        rank: i64,
        context: Option<&str>,
        active_mask: &str,
        source_op_id: i64,
        mut body: F,
    ) -> AResult<()>
    where
        F: FnMut(&mut Self, Option<&str>, &str) -> AResult<()>,
    {
        let overrides = self.raw_tma_snapshot_tensor_map_override(decoded)?;
        let expression = tensor_map_expression(decoded)?;
        let selections = self.raw_tma_descriptor_selections(&expression, active_mask)?;
        for (mask, expression) in selections {
            let split = mask != active_mask;
            if split {
                self.emit_line(&format!("if !{mask}.is_empty() {{"));
                self.indent += 1;
                self.emit_line(&format!("let ctx = ctx.with_active_mask({mask});"));
            }
            let result = self.raw_tma_tensor_map_region(
                &expression,
                parameters,
                rank,
                &mask,
                if split { None } else { context },
                split,
                overrides.as_ref(),
                source_op_id,
                &mut body,
            );
            if split {
                self.indent -= 1;
                self.emit_line("}");
            }
            result?;
        }
        Ok(())
    }

    /// One selection of `raw_tma_tensor_map_regions`.
    #[allow(clippy::too_many_arguments)]
    fn raw_tma_tensor_map_region<F>(
        &mut self,
        expression: &ObjectRef,
        parameters: &[Var],
        rank: i64,
        mask: &str,
        context: Option<&str>,
        split: bool,
        overrides: Option<&TensorMapOverride>,
        source_op_id: i64,
        body: &mut F,
    ) -> AResult<()>
    where
        F: FnMut(&mut Self, Option<&str>, &str) -> AResult<()>,
    {
        let lane_context = abi::context("ctx");
        if !parameters.is_empty() && overrides.is_none() {
            let descriptor = self.raw_tma_tensor_map(
                expression,
                parameters,
                Some(rank),
                mask,
                true,
                source_op_id,
            )?;
            let map_context = if split {
                Some(lane_context.as_str())
            } else {
                context
            };
            return body(self, map_context, &descriptor);
        }
        let mut descriptor = String::new();
        let mut address = String::new();
        if !parameters.is_empty() {
            descriptor = self.raw_tma_tensor_map(
                expression,
                parameters,
                Some(rank),
                mask,
                true,
                source_op_id,
            )?;
        } else {
            let pointer = self.emit_address_pointer(expression, "generic", None, mask)?;
            address = self.control_name("tma_descriptor_addresses");
            self.emit_line(&format!("let {address} = ({}).clone();", pointer.code));
        }
        let lane = self.control_name("tma_descriptor_lane");
        let lane_mask = self.control_name("tma_descriptor_mask");
        self.emit_line(&format!("for {lane} in {mask}.iter() {{"));
        self.indent += 1;
        self.emit_line(&format!(
            "let {lane_mask} = WarpMask::from_bits(1_u32 << {lane});"
        ));
        self.emit_line(&format!("let ctx = ctx.with_active_mask({lane_mask});"));
        if parameters.is_empty() {
            descriptor =
                self.raw_tma_lookup_tensor_map(&address, rank, &lane_mask, source_op_id)?;
        }
        if let Some(prepared) = overrides {
            descriptor = self.raw_tma_override_tensor_map(
                descriptor,
                &lane_context,
                prepared,
                &lane,
                source_op_id,
            );
        }
        body(self, Some(&lane_context), &descriptor)?;
        self.indent -= 1;
        self.emit_line("}");
        Ok(())
    }

    /// The per-lane registers of one override operand group.
    fn raw_tma_override_operands(
        &mut self,
        decoded: &DecodedPtx,
        field: &str,
    ) -> AResult<Vec<String>> {
        let mut names = Vec::new();
        if !decoded.has_operand(field) {
            return Ok(names);
        }
        for expression in decoded.operand(field)?.iter().flatten() {
            let name = self.control_name(&format!("tma_override_{field}"));
            let value = self.raw_tma_lane_i64(expression)?;
            self.emit_line(&format!("let {name} = {value};"));
            names.push(name);
        }
        Ok(names)
    }

    fn raw_tma_snapshot_tensor_map_override(
        &mut self,
        decoded: &DecodedPtx,
    ) -> AResult<Option<TensorMapOverride>> {
        let op_name = decoded.op_name.clone();
        if override_base(&op_name).is_none() {
            return Ok(None);
        }
        let mut address = self.emit_expr(&decoded.scalar_operand("global_address")?)?;
        if address.rust_type != "PhysicalPtr" {
            address = self.emit_raw_generic_pointer(address, "ctx.active_mask()")?;
        }
        require_physical_pointer(
            &address,
            &format!("{op_name} override address did not resolve to a physical address"),
        )?;
        let address_name = self.control_name("tma_override_addresses");
        self.emit_line(&format!("let {address_name} = ({}).clone();", address.code));
        Ok(Some(TensorMapOverride {
            address: address_name,
            tensor_size: self.raw_tma_override_operands(decoded, "tensor_size")?,
            lower_stride: self.raw_tma_override_operands(decoded, "lower_stride")?,
            coords: self.raw_tma_override_operands(decoded, "coords")?,
            upper_stride: self.raw_tma_override_operands(decoded, "upper_stride")?,
        }))
    }

    fn raw_tma_override_tensor_map(
        &mut self,
        tensor_map: String,
        context: &str,
        prepared: &TensorMapOverride,
        lane: &str,
        source_op_id: i64,
    ) -> String {
        let address_code = abi::address("v2::Global", &abi::cloned(&prepared.address), None);
        let values = |names: &[String]| -> String {
            let rendered: Vec<String> = names
                .iter()
                .map(|value| format!("{value}[{lane}]"))
                .collect();
            format!("&[{}]", rendered.join(", "))
        };
        let sizes = values(&prepared.tensor_size);
        let strides = values(&prepared.lower_stride);
        let coordinates = values(&prepared.coords);
        let upper = match prepared.upper_stride.first() {
            Some(value) => format!("{value}[{lane}]"),
            None => "0_i64".to_owned(),
        };
        let result = self.control_name("overridden_tensor_map");
        let expression = abi::call(
            "async_copy::override_tensor_map",
            &[
                abi::WARP.to_owned(),
                context.to_owned(),
                abi::site(source_op_id as u64),
                tensor_map,
                address_code,
                sizes,
                strides,
                upper,
                coordinates,
            ],
            &[],
            true,
            false,
        );
        self.emit_line(&format!("let {result} = {expression};"));
        result
    }

    /// The shared data pointer of the data operands.
    fn raw_tma_shared_data_pointer(
        &mut self,
        decoded: &DecodedPtx,
        shared_source: &ObjectRef,
    ) -> AResult<RustValue> {
        let op_name = decoded.op_name.clone();
        let pointer = if dtype_of(shared_source)? == "uint32" {
            self.emit_raw_shared_pointer(shared_source, None, "ctx.active_mask()")?
        } else {
            self.shared_pointer(shared_source, &format!("{op_name} shared pointer"))?
        };
        let pointer_name = self.control_name("raw_tma_shared_pointer");
        self.emit_line(&format!("let {pointer_name} = ({}).clone();", pointer.code));
        Ok(RustValue::new(
            pointer_name,
            "PhysicalPtr",
            pointer.uniformity,
        ))
    }

    fn raw_tma_lane_i64(&mut self, expression: &ObjectRef) -> AResult<String> {
        let value = self.emit_expr(expression)?;
        let value = self.as_i64(value)?;
        let value = self.as_warp_value(value);
        Ok(abi::register(&value.code))
    }

    fn raw_tma_coordinate_code(&mut self, coordinates: &[ObjectRef]) -> AResult<String> {
        let mut rendered = Vec::new();
        for value in coordinates {
            rendered.push(self.raw_tma_lane_i64(value)?);
        }
        Ok(format!("[{}]", rendered.join(", ")))
    }

    fn emit_raw_g2s(
        &mut self,
        decoded: &DecodedPtx,
        facts: &G2sFacts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let operands = &facts.operands;
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_tma_g2s_issue",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let context = region.context.clone();
        let issue_mask = region.mask.clone();
        let result = (|| -> AResult<()> {
            let shared = self.raw_tma_shared_data_pointer(decoded, &facts.shared_source)?;
            let barrier_source = &facts.barrier_source;
            let barrier = if dtype_of(barrier_source)? == "uint32" {
                self.emit_raw_shared_pointer(barrier_source, None, "ctx.active_mask()")?
            } else {
                self.shared_pointer(barrier_source, &format!("{op_name} mbarrier pointer"))?
            };
            let barrier_name = self.control_name("raw_tma_mbarrier_pointer");
            self.emit_line(&format!("let {barrier_name} = ({}).clone();", barrier.code));
            let barrier = RustValue::new(barrier_name, "PhysicalPtr", barrier.uniformity);

            let mut cta_mask_code = String::new();
            if facts.multicast {
                let cta_mask = decoded.scalar_operand("cta_mask")?;
                cta_mask_code = format!(", {}", self.raw_tma_lane_i64(&cta_mask)?);
            }
            // The manifest topology
            // is `extract_topology` of this same PrimFunc.
            let ctas_per_cluster = self.plan.topology.ctas_per_cluster;
            if facts.cta_group == 2 && ctas_per_cluster < 2 {
                return unsupported(format!("{op_name} cta_group=2 requires a two-CTA cluster"));
            }
            if facts.cta_group == 2 && ctas_per_cluster % 2 != 0 {
                return unsupported(format!("{op_name} cta_group=2 requires complete CTA pairs"));
            }

            let cluster = G2S_CLUSTER_CALLS.contains(&tma_base(&op_name));
            let im2col = operands.load_mode.starts_with("im2col");
            let specialization = if im2col {
                let mode = ["im2col", "im2col::w", "im2col::w::128"]
                    .iter()
                    .position(|candidate| *candidate == operands.load_mode)
                    .expect("validated im2col load mode");
                format!(
                    "TensorIm2col<{}, {mode}, {}, {}, {}>",
                    operands.rank, facts.cta_group, facts.multicast, facts.report_pattern
                )
            } else if operands.load_mode == "tile::gather4" {
                if operands.coordinates.len() != 5 {
                    return unsupported(format!(
                        "{op_name} lowering requires one column and four row coordinates"
                    ));
                }
                let marker = if facts.multicast {
                    "TensorGather4ClusterMulticast"
                } else if cluster {
                    "TensorGather4Cluster"
                } else {
                    "TensorGather4Cta"
                };
                match facts.report_pattern {
                    0 => format!("{marker}<{}>", facts.cta_group),
                    pattern => format!("{marker}<{}, {pattern}>", facts.cta_group),
                }
            } else {
                let marker = if facts.multicast {
                    "TensorG2sClusterMulticast"
                } else if cluster {
                    "TensorG2sCluster"
                } else {
                    "TensorG2sCta"
                };
                match facts.report_pattern {
                    0 => format!("{marker}<{}, {}>", operands.rank, facts.cta_group),
                    pattern => format!(
                        "{marker}<{}, {}, {pattern}>",
                        operands.rank, facts.cta_group
                    ),
                }
            };
            let coordinate_code = self.raw_tma_coordinate_code(&operands.coordinates)?;
            if im2col {
                let mut info = Vec::new();
                for value in decoded.operand("im2col_info")?.iter().flatten() {
                    info.push(self.raw_tma_lane_i64(value)?);
                }
                while info.len() < 3 {
                    info.push(abi::splat("0_i64"));
                }
                let mask = if cta_mask_code.is_empty() {
                    format!(", {}", abi::splat("0_i64"))
                } else {
                    cta_mask_code.clone()
                };
                cta_mask_code = format!(", [{}]{mask}", info.join(", "));
            }
            let shared_code = shared_address(&shared);
            let barrier_code = shared_address(&barrier);
            let variant = format!("v2::async_copy::variant::{specialization}");
            self.raw_tma_tensor_map_regions(
                decoded,
                &operands.parameters,
                operands.rank,
                context.as_deref(),
                &issue_mask,
                source_op_id,
                |emitter, map_context, tensor_map| {
                    let args = format!(
                        "({shared_code}, {tensor_map}, {coordinate_code}, {barrier_code}{cta_mask_code})"
                    );
                    emitter.emit_raw_tma_stateful(
                        "async_copy::cp_async_bulk_tensor",
                        source_op_id,
                        &args,
                        Some(&variant),
                        map_context,
                    );
                    Ok(())
                },
            )
        })();
        self.close_predicated_region(region);
        result
    }

    fn emit_raw_s2g(
        &mut self,
        decoded: &DecodedPtx,
        operands: &TensorOperands,
        shared_source: &ObjectRef,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let rank = operands.rank;
        let mode = ["tile", "im2col_no_offs", "im2col_no_offs::w"]
            .iter()
            .position(|candidate| *candidate == operands.load_mode)
            .expect("validated s2g load mode");
        let mode_arg = if mode != 0 {
            format!(", {mode}")
        } else {
            String::new()
        };
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_tma_s2g_issue",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let context = region.context.clone();
        let issue_mask = region.mask.clone();
        let result = (|| -> AResult<()> {
            let shared = self.raw_tma_shared_data_pointer(decoded, shared_source)?;
            let (instruction, specialization) =
                if tma_base(&op_name) == "tirx.ptx.cp_reduce_async_bulk_tensor" {
                    let reduction = decoded.modifier("redop")?;
                    let Some(reduce_marker) = reduction_marker(reduction) else {
                        return unsupported(format!(
                            "{op_name} reduction {:?} has no NumSim engine ABI",
                            reduction
                        ));
                    };
                    (
                        "cp_reduce_async_bulk_tensor",
                        format!("TensorS2gReduce<{rank}, {reduce_marker}{mode_arg}>"),
                    )
                } else {
                    (
                        "cp_async_bulk_tensor",
                        format!("TensorS2g<{rank}{mode_arg}>"),
                    )
                };
            let coordinate_code = self.raw_tma_coordinate_code(&operands.coordinates)?;
            self.emit_line("{");
            self.indent += 1;
            let shared_code = shared_address(&shared);
            let function = format!("async_copy::{instruction}");
            let variant = format!("v2::async_copy::variant::{specialization}");
            self.raw_tma_tensor_map_regions(
                decoded,
                &operands.parameters,
                rank,
                context.as_deref(),
                &issue_mask,
                source_op_id,
                |emitter, map_context, tensor_map| {
                    let args = format!("({shared_code}, {tensor_map}, {coordinate_code})");
                    let site = emitter.v2_site(Some(source_op_id));
                    let rendered_call = format!(
                        "{};",
                        stateful(&function, &site, &args, Some(&variant), map_context)
                    );
                    emitter.emit_line(&rendered_call);
                    Ok(())
                },
            )?;
            self.indent -= 1;
            self.emit_line("}");
            Ok(())
        })();
        self.close_predicated_region(region);
        result?;
        if !operands.parameters.is_empty() && override_base(&op_name).is_none() {
            // The recorded store only feeds the scaffold checkpoint count.
            self.recorded_store_count += 1;
        }
        Ok(())
    }

    /// Run one lowering inside `predicated_instruction_region(...,
    /// shadow_context=True)`.
    fn raw_tma_shadow_region(
        &mut self,
        decoded: &DecodedPtx,
        prefix: &str,
        lower: impl FnOnce(&mut Self) -> AResult<()>,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            prefix,
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let result = lower(self);
        self.close_predicated_region(region);
        result
    }

    /// The `raw_tma`-kind statement lowering.
    pub fn emit_raw_tma_statement(
        &mut self,
        decoded: &DecodedPtx,
        call: &RawTmaCall,
        source_op_id: i64,
    ) -> AResult<()> {
        match call {
            RawTmaCall::G2s(facts) => self.emit_raw_g2s(decoded, facts, source_op_id),
            RawTmaCall::S2g(operands, shared_source) => {
                self.emit_raw_s2g(decoded, operands, shared_source, source_op_id)
            }
            RawTmaCall::TensorCacheHint(operands) => {
                self.emit_raw_tensor_cache_hint(decoded, operands, source_op_id)
            }
            RawTmaCall::TensorMapPrefetch(parameters) => {
                self.raw_tma_shadow_region(decoded, "raw_prefetch", |emitter| {
                    emitter.emit_raw_tensor_map_prefetch(decoded, parameters, source_op_id)
                })
            }
            RawTmaCall::AddressCacheHint(source) if decoded.op_name == "tirx.ptx.prefetch" => self
                .raw_tma_shadow_region(decoded, "raw_prefetch", |emitter| {
                    emitter.emit_raw_prefetch(decoded, source)
                }),
            RawTmaCall::AddressCacheHint(_) | RawTmaCall::BulkCacheHint(..) => {
                self.emit_raw_cache_hint(decoded, call, source_op_id)
            }
            RawTmaCall::TensorMapReplace(facts) => {
                self.raw_tma_shadow_region(decoded, "tensor_map_replace", |emitter| {
                    emitter.emit_raw_tensor_map_replace(decoded, facts, source_op_id)
                })
            }
            RawTmaCall::TensorMapFence(copy_source) => {
                self.raw_tma_shadow_region(decoded, "tensor_map_fence", |emitter| {
                    emitter.emit_raw_tensor_map_fence(decoded, copy_source.as_ref(), source_op_id)
                })
            }
        }
    }

    /// One typed TensorMap `address_of` lowered to its opaque descriptor token.
    pub fn emit_tensor_map_address(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        let parameters = tensor_map_parameters(self.ctx, expr, Some(&self.params))?;
        if parameters.len() != 1 {
            return unsupported(
                "typed TensorMap address_of must name exactly one PrimFunc parameter",
            );
        }
        let parameter = &parameters[0];
        let Some(binding_name) = self
            .raw_tma
            .tensor_maps
            .iter()
            .find(|candidate| same(&candidate.variable, parameter))
            .map(|tensor_map| tensor_map.binding_name.clone())
        else {
            return unsupported(format!(
                "TensorMap parameter {:?} is missing from the semantic manifest by exact parameter identity",
                &ffi_text(&parameter.name)));
        };
        let pointer = self.control_name("tensor_map_parameter_address");
        let registry = self.tensor_map_registry_ref()?;
        self.emit_line(&format!(
            "let {pointer} = {registry}.parameter_address({})?;",
            json_string(&binding_name)
        ));
        let address = self.control_name("tensor_map_parameter_address_bits");
        self.emit_line(&format!(
            "let {address} = ({pointer}).generic_addresses_u64(&ctx, ctx.active_mask())?;"
        ));
        Ok(RustValue::new(address, "u64", Uniformity::Varying))
    }

    fn raw_tma_append_converted_host_integer(
        &mut self,
        setup: &mut Vec<String>,
        host_variables: &Variables,
        name: &str,
        expression: &ObjectRef,
        field: &str,
        target_type: &str,
    ) -> AResult<()> {
        let (lines, value) = self.emit_host_integer(expression, host_variables, field)?;
        setup.extend(lines.iter().map(|line| format!("        {line}")));
        setup.push(format!(
            "        let {name}_value: {} = {};\n        let {name} = {target_type}::try_from({name}_value).map_err(|_| {{\n            PyValueError::new_err(format!(\n                {},\n                {name}_value,\n            ))\n        }})?;",
            value.rust_type,
            value.code,
            json_string(&format!("{field} is negative or too large: {{}}"))
        ));
        Ok(())
    }

    /// Emit host setup for an implicit TensorMap from normalized metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_tensor_map_setup(
        &mut self,
        setup: &mut Vec<String>,
        host_variables: &Variables,
        name: &str,
        binding_name: &str,
        field: &str,
        kernel_index: usize,
        kernel_count: usize,
    ) -> AResult<()> {
        let plan = self.plan;
        let Some(descriptor) = plan
            .implicit_tensor_maps
            .iter()
            .find(|item| item.get("name").and_then(Json::as_str) == Some(name))
        else {
            return unsupported(format!(
                "implicit TensorMap {:?} is missing its host expression or ABI metadata",
                binding_name
            ));
        };
        let base_buffer = descriptor
            .get("base_buffer")
            .and_then(Json::as_str)
            .unwrap_or("");
        let base_canonical_name = implicit_base_canonical_name(
            plan,
            kernel_index as i64,
            kernel_count as i64,
            base_buffer,
        )?;
        // Rebind serialized expressions to this PrimFunc's scalar parameters.
        let scalars = scalar_parameters(self.func);
        let deserialize = |payload: &Json, expression_field: &str| -> AResult<ObjectRef> {
            let expression = deserialize_integer_expr(self.func, payload, expression_field)?;
            crate::emit::validate_integer_expression(
                &expression,
                &|variable| scalars.iter().any(|parameter| same(variable, parameter)),
                &format!("host TensorMap {expression_field}"),
            )?;
            Ok(expression)
        };
        let base_byte_offset = deserialize(
            descriptor.get("base_byte_offset").expect("validated field"),
            &format!("{name}.base_byte_offset"),
        )?;
        let expression_fields = [
            "global_shape",
            "global_strides",
            "box_shape",
            "element_strides",
        ];
        let mut expressions: Vec<Vec<ObjectRef>> = Vec::new();
        for field_name in expression_fields {
            let mut values = Vec::new();
            for (axis, payload) in descriptor
                .get(field_name)
                .and_then(Json::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                values.push(deserialize(
                    payload,
                    &format!("{name}.{field_name}[{axis}]"),
                )?);
            }
            expressions.push(values);
        }

        let prefix = format!("implicit_tensor_map_{kernel_index}_{field}");
        let offset_name = format!("{prefix}_base_byte_offset");
        self.raw_tma_append_converted_host_integer(
            setup,
            host_variables,
            &offset_name,
            &base_byte_offset,
            &format!("implicit TensorMap {binding_name}.base_byte_offset"),
            "i128",
        )?;
        let mut rendered_fields: Vec<Vec<String>> = Vec::new();
        for (field_name, values) in expression_fields.iter().zip(expressions.iter()) {
            let mut names = Vec::new();
            for (axis, expression) in values.iter().enumerate() {
                let axis_name = format!("{prefix}_{field_name}_{axis}");
                self.raw_tma_append_converted_host_integer(
                    setup,
                    host_variables,
                    &axis_name,
                    expression,
                    &format!("implicit TensorMap {binding_name}.{field_name}[{axis}]"),
                    "usize",
                )?;
                names.push(axis_name);
            }
            rendered_fields.push(names);
        }
        let rust_option = |value: Option<&Json>| -> String {
            match value {
                None | Some(Json::Null) => "None".to_owned(),
                Some(Json::String(text)) => format!("Some({})", json_string(text)),
                Some(other) => format!("Some({})", json_string(&other.to_string())),
            }
        };
        let Some(dtype) = descriptor.get("dtype").and_then(Json::as_str) else {
            return unsupported(format!(
                "implicit TensorMap {:?} has no static dtype",
                binding_name
            ));
        };
        setup.push(format!(
            "        let {field} = extract_or_build_implicit_tensor_map(\n            inputs,\n            \"{binding_name}\",\n            \"{base_canonical_name}\",\n            {offset_name},\n            {},\n            {},\n            {},\n            &[{}],\n            &[{}],\n            &[{}],\n            &[{}],\n            {},\n            {},\n            {},\n            physical.global(),\n            allocation_ids,\n        )?;",
            json_string(dtype),
            rust_option(descriptor.get("tma_dtype")),
            rust_option(descriptor.get("fp4_shared_layout")),
            rendered_fields[0].join(", "),
            rendered_fields[1].join(", "),
            rendered_fields[2].join(", "),
            rendered_fields[3].join(", "),
            rust_option(descriptor.get("interleave")),
            rust_option(descriptor.get("swizzle")),
            rust_option(descriptor.get("fill_mode")),
        ));
        Ok(())
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = ptx_tma_call(emitter.ctx, decoded, call.params)?;
    if let RawTmaCall::S2g(operands, _) = &parts {
        if decoded.has_operand("global_address") {
            emitter.record_pointer_write(&decoded.scalar_operand("global_address")?, "global")?;
        }
        if operands.parameters.is_empty() {
            // A descriptor loaded from arbitrary memory has no static base.
            emitter.written_global_buffers = None;
        }
        for parameter in &operands.parameters {
            let index = emitter.params.iter()
                .position(|p| same(p, parameter)).expect("TensorMap parameter");
            emitter.written_tensor_maps.insert(index);
        }
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_raw_tma_statement(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_cache_hint(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = ptx_cache_hint_call(emitter.ctx, decoded, call.params)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_raw_tma_statement(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_replace(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = ptx_tensor_map_replace_call(decoded)?;
    emitter.record_pointer_write(&decoded.scalar_operand("addr")?, decoded.modifier("space")?)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_raw_tma_statement(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_fence(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = ptx_tensor_map_fence_call(emitter.ctx, decoded)?;
    if decoded.op_name == "tirx.ptx.tensormap_cp_fenceproxy" {
        // The copied descriptor can contain an arbitrary global base.
        emitter.written_global_buffers = None;
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_raw_tma_statement(decoded, &parts, source_op_id)?;
    Ok(None)
}

fn is_tensor_map_parameter(node: &ObjectRef, params: Option<&[Var]>) -> bool {
    let Some(variable) = as_var(node) else {
        return false;
    };
    let Some(pointer) = variable.ty.as_node::<PointerTypeObj>() else {
        return false;
    };
    if pointer.element_type.as_node::<TensorMapTypeObj>().is_none() {
        return false;
    }
    match params {
        None => true,
        Some(params) => params.iter().any(|parameter| same(parameter, &variable)),
    }
}

pub fn is_tensor_map_address_call(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &CallObj,
    params: Option<&[Var]>,
) -> AResult<bool> {
    if call_op_name(call)?.as_deref() != Some("tirx.address_of") {
        return Ok(false);
    }
    match super::pure::address_of_signature(ctx, node, call) {
        Ok(_) => {}
        Err(Failure::Unsupported { .. }) | Err(Failure::Unmodeled { .. }) => return Ok(false),
        Err(error) => return Err(error),
    }
    let argument = oref(call.args.get(0)?);
    // Selector branches (Select / if_then_else) never reach here: address_of
    // takes its parameter directly.
    Ok(is_tensor_map_parameter(&argument, params))
}

pub fn emit_address(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    emitter
        .with_call_expr(call.node, |emitter| {
            emitter.emit_tensor_map_address(call.node)
        })
        .map(Some)
}

/// TensorMap bindings and registry storage needed before the body is emitted.
#[derive(Default)]
pub struct TensorMapFacts {
    pub parameter_indices: Vec<i64>,
    pub uses_registry: bool,
}

pub fn scan_tensor_maps(ctx: &Ctx, nodes: &[ObjectRef], params: &[Var]) -> AResult<TensorMapFacts> {
    let mut facts = TensorMapFacts::default();
    for node in nodes {
        if let Some(call) = node.as_node::<CallObj>() {
            if is_tensor_map_address_call(ctx, node, call, Some(params))? {
                let record = record_address_call(ctx, node, params)?;
                facts.parameter_indices.extend(record.parameter_indices);
                facts.uses_registry |= record.uses_registry;
            }
        }
        let decoded = match ctx.decoding.ptx_decode(node) {
            Ok(Some(crate::decode::ptx::PtxDecode::Decoded(decoded))) => decoded,
            Ok(_) | Err(Failure::Unsupported { .. }) => continue,
            Err(error) => return Err(error),
        };
        let name = decoded.op_name.as_str();
        if name.starts_with("tirx.ptx.tensormap_replace")
            || matches!(
                name,
                "tirx.ptx.tensormap_cp_fenceproxy"
                    | "tirx.ptx.fence_proxy_tensormap_release"
                    | "tirx.ptx.fence_proxy_tensormap_acquire"
            )
        {
            facts.uses_registry = true;
        }
        let prefetch =
            name == "tirx.ptx.prefetch" && !decoded.modifier_or_empty("tensormap").is_empty();
        if decoded.has_operand("tmap") || prefetch {
            let parameters = match decoded_tensor_map_parameters(ctx, &decoded, Some(params)) {
                Ok(parameters) => parameters,
                Err(Failure::Unsupported { .. }) => continue,
                Err(error) => return Err(error),
            };
            facts
                .parameter_indices
                .extend(parameter_indices(&parameters, params));
            facts.uses_registry |= parameters.is_empty() && !prefetch;
        }
    }
    Ok(facts)
}
