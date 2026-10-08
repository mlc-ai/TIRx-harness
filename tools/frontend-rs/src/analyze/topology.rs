//! Launch topology extraction and the AttrStmt/For validators.

use crate::analyze::util::json_object;
use crate::tvm_compat::int_value;
use tvm::ir::{FloatImmObj, IntImmObj, VarObj};
use tvm::prim::{CastObj, NotObj, SelectObj};
use tvm::tirx::{
    AttrStmtObj, BindObj, ForKind, ForObj, IterVar, ScopeIdDefStmtObj, SeqStmtObj, Stmt,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore, TypeIndex};

use super::util::{dtype_of, ffi_text, kind, oref, repr_text, unsupported, AResult, IdMap, Json};
use super::{topology_operands, Ctx};

const WARP_SIZE: i64 = 32;
const WARPS_PER_WARPGROUP: i64 = 4;
const THREADS_PER_WARPGROUP: i64 = WARP_SIZE * WARPS_PER_WARPGROUP;

#[derive(Clone, Copy, Debug)]
pub enum StaticValue {
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl StaticValue {
    fn truthy(self) -> bool {
        match self {
            StaticValue::Bool(value) => value,
            StaticValue::Int(value) => value != 0,
            StaticValue::Float(value) => value != 0.0,
        }
    }

    fn as_int(self) -> i128 {
        match self {
            StaticValue::Bool(value) => i128::from(value),
            StaticValue::Int(value) => value,
            StaticValue::Float(value) => value.trunc() as i128,
        }
    }

    fn as_float(self) -> f64 {
        match self {
            StaticValue::Bool(value) => f64::from(value),
            StaticValue::Int(value) => value as f64,
            StaticValue::Float(value) => value,
        }
    }
}

fn cast_static_scalar(ctx: &Ctx, value: StaticValue, dtype: &str) -> AResult<StaticValue> {
    if dtype == "bool" {
        return Ok(StaticValue::Bool(value.truthy()));
    }
    if dtype == "float32" || dtype == "float64" {
        return Ok(StaticValue::Float(value.as_float()));
    }
    let Some((bits, signed)) = ctx.schema.integer_dtype_bits(dtype) else {
        return unsupported(format!(
            "NumSim static scalar Cast is not implemented for {dtype}"
        ));
    };
    let mask: i128 = (1i128 << bits) - 1;
    let mut truncated = value.as_int() & mask;
    if signed && truncated >= (1i128 << (bits - 1)) {
        truncated -= 1i128 << bits;
    }
    Ok(StaticValue::Int(truncated))
}

fn trunc_div(lhs: i128, rhs: i128) -> AResult<i128> {
    if rhs == 0 {
        return super::util::not_covered("integer division by zero in a static launch extent");
    }
    let magnitude = lhs.abs() / rhs.abs();
    Ok(if (lhs < 0) != (rhs < 0) {
        -magnitude
    } else {
        magnitude
    })
}

fn floor_div(lhs: i128, rhs: i128) -> AResult<i128> {
    if rhs == 0 {
        return super::util::not_covered("integer division by zero in a static launch extent");
    }
    Ok(lhs.div_euclid(rhs)
        - if rhs < 0 && lhs.rem_euclid(rhs) != 0 {
            1
        } else {
            0
        })
}

fn floor_mod(lhs: i128, rhs: i128) -> AResult<i128> {
    Ok(lhs - floor_div(lhs, rhs)? * rhs)
}

fn evaluate_integer_binary(kind: &str, lhs: i128, rhs: i128) -> AResult<i128> {
    Ok(match kind {
        "Add" => lhs + rhs,
        "Sub" => lhs - rhs,
        "Mul" => lhs * rhs,
        "Div" => trunc_div(lhs, rhs)?,
        "Mod" => lhs - trunc_div(lhs, rhs)? * rhs,
        "FloorDiv" => floor_div(lhs, rhs)?,
        "FloorMod" => floor_mod(lhs, rhs)?,
        "Min" => lhs.min(rhs),
        "Max" => lhs.max(rhs),
        _ => {
            return unsupported(format!(
                "integer expression operation {kind} is not registered"
            ))
        }
    })
}

pub fn static_value(
    ctx: &Ctx,
    node: &ObjectRef,
    environment: &IdMap<StaticValue>,
) -> AResult<StaticValue> {
    let node_kind = kind(node).unwrap_or("");
    if let Some(imm) = node.as_node::<IntImmObj>() {
        return cast_static_scalar(
            ctx,
            StaticValue::Int(i128::from(int_value(imm)?)),
            &dtype_of(node)?,
        );
    }
    if let Some(imm) = node.as_node::<FloatImmObj>() {
        return cast_static_scalar(ctx, StaticValue::Float(imm.value), &dtype_of(node)?);
    }
    if node.as_node::<VarObj>().is_some() {
        if let Some(value) = environment.get(node) {
            return Ok(*value);
        }
    }
    if let Some((name, operands)) = super::util::bitwise_expr(node) {
        let a = static_value(ctx, &operands[0], environment)?;
        let dtype = dtype_of(node)?;
        if name == "bitwise_not" {
            let value = if dtype == "bool" {
                StaticValue::Bool(!a.truthy())
            } else {
                StaticValue::Int(!a.as_int())
            };
            return cast_static_scalar(ctx, value, &dtype);
        }
        let b = static_value(ctx, &operands[1], environment)?;
        let bits = super::util::prim_dtype(&super::util::expr_type(node).expect("primitive node"))
            .expect("primitive dtype")
            .bits;
        let shift = (b.as_int() as u32) % u32::from(bits);
        let value = match name {
            "bitwise_and" => a.as_int() & b.as_int(),
            "bitwise_or" => a.as_int() | b.as_int(),
            "bitwise_xor" => a.as_int() ^ b.as_int(),
            "shift_left" => a.as_int().wrapping_shl(shift),
            "shift_right" => a.as_int().wrapping_shr(shift),
            _ => unreachable!(),
        };
        return cast_static_scalar(ctx, StaticValue::Int(value), &dtype);
    }
    if ctx.schema.integer_binary_node_kinds.contains(node_kind) {
        let (a, b) = topology_operands(node).expect("binary node");
        let lhs = static_value(ctx, &a, environment)?;
        let rhs = static_value(ctx, &b, environment)?;
        let dtype = dtype_of(node)?;
        let value = if dtype == "float32" || dtype == "float64" {
            let result = match node_kind {
                "Add" => lhs.as_float() + rhs.as_float(),
                "Sub" => lhs.as_float() - rhs.as_float(),
                "Mul" => lhs.as_float() * rhs.as_float(),
                "Div" => {
                    if rhs.as_float() == 0.0 {
                        return super::util::not_covered(
                            "float division by zero in a static launch extent",
                        );
                    }
                    lhs.as_float() / rhs.as_float()
                }
                "Min" => lhs.as_float().min(rhs.as_float()),
                "Max" => lhs.as_float().max(rhs.as_float()),
                _ => {
                    return unsupported(format!(
                        "NumSim static scalar {node_kind} is not implemented for {dtype}"
                    ))
                }
            };
            StaticValue::Float(result)
        } else {
            StaticValue::Int(evaluate_integer_binary(
                node_kind,
                lhs.as_int(),
                rhs.as_int(),
            )?)
        };
        return cast_static_scalar(ctx, value, &dtype);
    }
    if ["LT", "LE", "GT", "GE", "EQ", "NE"].contains(&node_kind) {
        let (a, b) = topology_operands(node).expect("comparison node");
        let lhs = static_value(ctx, &a, environment)?;
        let rhs = static_value(ctx, &b, environment)?;
        let (l, r) = (lhs.as_float(), rhs.as_float());
        let both_int = matches!(lhs, StaticValue::Int(_) | StaticValue::Bool(_))
            && matches!(rhs, StaticValue::Int(_) | StaticValue::Bool(_));
        let (li, ri) = (lhs.as_int(), rhs.as_int());
        let result = match node_kind {
            "LT" => {
                if both_int {
                    li < ri
                } else {
                    l < r
                }
            }
            "LE" => {
                if both_int {
                    li <= ri
                } else {
                    l <= r
                }
            }
            "GT" => {
                if both_int {
                    li > ri
                } else {
                    l > r
                }
            }
            "GE" => {
                if both_int {
                    li >= ri
                } else {
                    l >= r
                }
            }
            "EQ" => {
                if both_int {
                    li == ri
                } else {
                    l == r
                }
            }
            _ => {
                if both_int {
                    li != ri
                } else {
                    l != r
                }
            }
        };
        return Ok(StaticValue::Bool(result));
    }
    if node_kind == "And" || node_kind == "Or" {
        let (a, b) = topology_operands(node).expect("logical node");
        let lhs = static_value(ctx, &a, environment)?.truthy();
        let rhs = static_value(ctx, &b, environment)?.truthy();
        return Ok(StaticValue::Bool(if node_kind == "And" {
            lhs && rhs
        } else {
            lhs || rhs
        }));
    }
    if let Some(not) = node.as_node::<NotObj>() {
        return Ok(StaticValue::Bool(
            !static_value(ctx, &oref(not.a.clone()), environment)?.truthy(),
        ));
    }
    let branches = if let Some(select) = node.as_node::<SelectObj>() {
        Some((
            oref(select.condition.clone()),
            oref(select.true_value.clone()),
            oref(select.false_value.clone()),
        ))
    } else {
        crate::emit::pure::conditional_operands(node)?
    };
    if let Some((condition, true_value, false_value)) = branches {
        let branch = if static_value(ctx, &condition, environment)?.truthy() {
            true_value
        } else {
            false_value
        };
        return static_value(ctx, &branch, environment);
    }
    if let Some(cast) = node.as_node::<CastObj>() {
        let value = static_value(ctx, &oref(cast.value.clone()), environment)?;
        return cast_static_scalar(ctx, value, &dtype_of(node)?);
    }
    unsupported(format!(
        "NumSim launch extent is not statically known: {}",
        repr_text(node)?
    ))
}

pub fn static_int(ctx: &Ctx, node: &ObjectRef, environment: &IdMap<StaticValue>) -> AResult<i128> {
    match static_value(ctx, node, environment)? {
        StaticValue::Int(value) => Ok(value),
        _ => unsupported(format!(
            "NumSim launch extent is not an integer: {}",
            repr_text(node)?
        )),
    }
}

pub struct LaunchTopology {
    pub clusters: i64,
    pub ctas_per_cluster: i64,
    pub warps_per_cta: i64,
    pub warps_per_warpgroup: i64,
    /// `tirx.launch_bounds_min_blocks_per_sm`, or 1. Emission caps the initial
    /// setmaxnreg count with it; it stays out of `json()` so the manifest is
    /// unchanged.
    pub min_blocks_per_sm: i64,
}

impl LaunchTopology {
    pub fn json(&self) -> Json {
        json_object(vec![
            ("clusters", Json::from(self.clusters)),
            ("ctas_per_cluster", Json::from(self.ctas_per_cluster)),
            ("warps_per_cta", Json::from(self.warps_per_cta)),
            ("warps_per_warpgroup", Json::from(self.warps_per_warpgroup)),
        ])
    }
}

struct Constraints {
    entries: Vec<(&'static str, i64)>,
}

impl Constraints {
    fn get(&self, name: &str) -> Option<i64> {
        self.entries
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }

    fn constrain(&mut self, name: &'static str, value: i64) -> AResult<()> {
        if value <= 0 {
            return unsupported(format!(
                "launch constraint {name} must be positive, got {value}"
            ));
        }
        match self.get(name) {
            None => {
                self.entries.push((name, value));
                Ok(())
            }
            Some(previous) if previous == value => Ok(()),
            Some(previous) => unsupported(format!(
                "conflicting launch constraints for {name}: {previous} versus {value}"
            )),
        }
    }
}

pub fn iter_var_of(value: &Any) -> Option<IterVar> {
    IterVar::try_from(value.clone()).ok()
}

fn visit(
    ctx: &Ctx,
    node: &Stmt,
    constraints: &mut Constraints,
    environment: &mut IdMap<StaticValue>,
) -> AResult<()> {
    if let Some(sequence) = node.as_node::<SeqStmtObj>() {
        for child in sequence.seq.iter() {
            visit(ctx, &child, constraints, environment)?;
        }
        return Ok(());
    }
    if let Some(attr) = node.as_node::<AttrStmtObj>() {
        if ffi_text(&attr.attr_key) == "thread_extent" {
            let thread_tag = match iter_var_of(&attr.node) {
                Some(iter_var) => ffi_text(&iter_var.thread_tag()?),
                None => String::new(),
            };
            let extent = static_int(ctx, &oref(attr.value.clone()), environment)?;
            let constraint = match thread_tag.as_str() {
                "blockIdx.x" => "global CTAs",
                "clusterCtaIdx.x" => "CTAs per cluster",
                "threadIdx.x" => "threads per CTA",
                _ => {
                    return unsupported(format!(
                        "NumSim does not support direct launch thread tag {:?}",
                        &thread_tag
                    ))
                }
            };
            constraints.constrain(constraint, i64::try_from(extent).unwrap_or(i64::MAX))?;
        } else if ffi_text(&attr.attr_key) == "tirx.launch_bounds_min_blocks_per_sm" {
            let blocks = static_int(ctx, &oref(attr.value.clone()), environment)?;
            constraints.constrain("min blocks per sm", i64::try_from(blocks).unwrap_or(i64::MAX))?;
        }
        return visit(ctx, &attr.body, constraints, environment);
    }
    if let Some(bind) = node.as_node::<BindObj>() {
        match static_value(ctx, &oref(bind.value.clone()), environment) {
            Ok(value) => {
                let key = oref(bind.var.clone());
                if environment.contains(&key) {
                    *environment.get_mut(&key).expect("bound") = value;
                } else {
                    environment.insert(key, value);
                }
            }
            Err(super::util::Failure::Unsupported { .. }) => {}
            Err(error) => return Err(error),
        }
        return Ok(());
    }
    let Some(definition_stmt) = node.as_node::<ScopeIdDefStmtObj>() else {
        return Ok(());
    };
    let definition = &definition_stmt.def;
    let scope = i64::from(definition.scope.as_raw());
    if !(0..10).contains(&scope) {
        return unsupported(format!("unsupported launch scope id {scope}"));
    }
    let Some(extents) = &definition.extents else {
        return Ok(());
    };
    if extents.is_empty() {
        return Ok(());
    }
    let mut product: i128 = 1;
    for extent in extents.iter() {
        product *= static_int(ctx, &oref(extent), environment)?;
    }
    let extent = i64::try_from(product).unwrap_or(i64::MAX);
    match scope {
        0 => constraints.constrain("clusters", extent)?,
        1 => constraints.constrain("global CTAs", extent)?,
        2 => constraints.constrain("CTAs per cluster", extent)?,
        9 => {
            if extent != 2 {
                return unsupported(format!(
                    "cluster CTA-pair result extent must be 2, got {extent}"
                ));
            }
        }
        3 => constraints.constrain("warpgroups per CTA", extent)?,
        4 => constraints.constrain("warps per CTA", extent)?,
        5 => constraints.constrain("warps per warpgroup", extent)?,
        7 => constraints.constrain("threads per CTA", extent)?,
        8 => constraints.constrain("threads per warpgroup", extent)?,
        6 => constraints.constrain("threads per warp", extent)?,
        _ => {}
    }
    Ok(())
}

/// `_extract_topology(func, environment)`: `environment` holds concrete
/// scalar parameter values.
pub(crate) fn extract_topology(
    ctx: &Ctx,
    body: &Stmt,
    mut environment: IdMap<StaticValue>,
) -> AResult<LaunchTopology> {
    let mut constraints = Constraints {
        entries: Vec::new(),
    };
    visit(ctx, body, &mut constraints, &mut environment)?;
    let threads_per_warp = constraints.get("threads per warp").unwrap_or(WARP_SIZE);
    if threads_per_warp != WARP_SIZE {
        return unsupported(format!(
            "warp thread extent must be {WARP_SIZE}, got {threads_per_warp}"
        ));
    }
    let warps_per_warpgroup = constraints
        .get("warps per warpgroup")
        .unwrap_or(WARPS_PER_WARPGROUP);
    if warps_per_warpgroup != WARPS_PER_WARPGROUP {
        return unsupported(format!(
            "warpgroup warp extent must be {WARPS_PER_WARPGROUP}, got {warps_per_warpgroup}"
        ));
    }
    let threads_per_warpgroup = constraints
        .get("threads per warpgroup")
        .unwrap_or(THREADS_PER_WARPGROUP);
    if threads_per_warpgroup != THREADS_PER_WARPGROUP {
        return unsupported(format!(
            "warpgroup thread extent must be {THREADS_PER_WARPGROUP}, got {threads_per_warpgroup}"
        ));
    }
    let explicit_clusters = constraints.get("clusters");
    let global_ctas = constraints.get("global CTAs");
    let explicit_ctas_per_cluster = constraints.get("CTAs per cluster");
    let (clusters, ctas_per_cluster) = match (
        explicit_clusters,
        global_ctas,
        explicit_ctas_per_cluster,
    ) {
        (Some(clusters), global, Some(per_cluster)) => {
            let derived = clusters * per_cluster;
            if let Some(global) = global {
                if global != derived {
                    return unsupported(format!(
                        "global CTA count {global} disagrees with clusters {clusters} * CTAs per cluster {per_cluster}"
                    ));
                }
            }
            (clusters, per_cluster)
        }
        (Some(clusters), Some(global), None) => {
            if global % clusters != 0 {
                return unsupported(format!(
                    "global CTA count {global} is not divisible by cluster count {clusters}"
                ));
            }
            (clusters, global / clusters)
        }
        (None, Some(global), Some(per_cluster)) => {
            if global % per_cluster != 0 {
                return unsupported(format!(
                    "global CTA count {global} is not divisible by cluster size {per_cluster}"
                ));
            }
            (global / per_cluster, per_cluster)
        }
        (Some(clusters), None, None) => (clusters, 1),
        (None, Some(global), None) => (global, 1),
        (None, None, Some(per_cluster)) => (1, per_cluster),
        (None, None, None) => (1, 1),
    };
    let mut warp_candidates: Vec<(&str, i64)> = Vec::new();
    if let Some(direct_warps) = constraints.get("warps per CTA") {
        warp_candidates.push(("CTA warp extent", direct_warps));
    }
    if let Some(direct_threads) = constraints.get("threads per CTA") {
        if direct_threads % WARP_SIZE != 0 {
            return unsupported(format!(
                "CTA thread extent {direct_threads} is not a whole number of warps"
            ));
        }
        warp_candidates.push(("CTA thread extent", direct_threads / WARP_SIZE));
    }
    if let Some(warpgroup_count) = constraints.get("warpgroups per CTA") {
        warp_candidates.push(("warpgroup extent", warpgroup_count * warps_per_warpgroup));
    } else if warp_candidates.is_empty()
        && (constraints.get("warps per warpgroup").is_some()
            || constraints.get("threads per warpgroup").is_some())
    {
        warp_candidates.push(("implicit warpgroup extent", warps_per_warpgroup));
    }
    let warps_per_cta = if let Some((source, warps)) = warp_candidates.first().copied() {
        for (candidate_source, candidate_value) in warp_candidates.iter().skip(1) {
            if *candidate_value != warps {
                return unsupported(format!(
                    "{source} implies {warps} warps per CTA, but {candidate_source} implies {candidate_value}"
                ));
            }
        }
        warps
    } else {
        1
    };
    Ok(LaunchTopology {
        clusters,
        ctas_per_cluster,
        warps_per_cta,
        warps_per_warpgroup,
        min_blocks_per_sm: constraints.get("min blocks per sm").unwrap_or(1),
    })
}

/// `None` when the attribute is acceptable.
pub fn validate_attr_stmt(ctx: &Ctx, attr: &AttrStmtObj) -> AResult<Option<String>> {
    let attr_key = ffi_text(&attr.attr_key);
    if !ctx.schema.numerically_irrelevant_attrs.contains(&attr_key)
        && !ctx.schema.semantic_attrs.contains(&attr_key)
    {
        return Ok(Some(format!(
            "AttrStmt(attr_key={:?}) has unsupported numerical semantics",
            &attr_key
        )));
    }
    let value_node = oref(attr.value.clone());
    let value_int = value_node
        .as_node::<IntImmObj>()
        .and_then(|imm| int_value(imm).ok());
    let node_object = ObjectRef::try_from(attr.node.clone()).ok();
    let zero_node = match &node_object {
        None => i64::try_from(attr.node.clone()).ok() == Some(0),
        Some(object) => object
            .as_node::<IntImmObj>()
            .is_some_and(|imm| int_value(imm).ok() == Some(0)),
    };
    let message = match attr_key.as_str() {
        "tirx.device_entry" => {
            if !zero_node
                || value_int.is_none()
                || dtype_of(&value_node)? != "bool"
                || value_int != Some(1)
            {
                Some(
                    "AttrStmt(tirx.device_entry) must carry the canonical zero/true markers"
                        .to_owned(),
                )
            } else {
                None
            }
        }
        "tirx.max_registers" => {
            if !zero_node || value_int.is_none_or(|value| value <= 0) {
                Some("AttrStmt(tirx.max_registers) must be a positive IntImm".to_owned())
            } else {
                None
            }
        }
        "thread_extent" => {
            let iter_var = node_object.as_ref().and_then(|_| iter_var_of(&attr.node));
            let thread_tag = match &iter_var {
                Some(iter_var) => ffi_text(&iter_var.thread_tag()?),
                None => String::new(),
            };
            if !["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x"].contains(&thread_tag.as_str()) {
                Some(format!(
                    "AttrStmt(thread_extent) has unsupported thread tag {:?}",
                    &thread_tag
                ))
            } else {
                let var_dtype = match &iter_var {
                    Some(iter_var) => super::util::dtype_text(iter_var.var()?.dtype()),
                    None => String::new(),
                };
                if iter_var.is_none()
                    || value_int.is_none_or(|value| value <= 0)
                    || var_dtype != "int32"
                {
                    Some(
                        "AttrStmt(thread_extent) must bind an int32 IterVar to a positive IntImm"
                            .to_owned(),
                    )
                } else {
                    None
                }
            }
        }
        "tirx.launch_bounds_min_blocks_per_sm" | "tirx.launch_bounds_max_blocks_per_cluster" => {
            if !zero_node || value_int.is_none_or(|value| value <= 0) {
                Some(format!("AttrStmt({attr_key}) must be a positive IntImm"))
            } else {
                None
            }
        }
        "tirx.dyn_smem_bytes" => {
            if !zero_node || value_int.is_none_or(|value| value < 0) {
                Some(
                    "AttrStmt(tirx.dyn_smem_bytes) must carry zero and a non-negative IntImm"
                        .to_owned(),
                )
            } else {
                None
            }
        }
        // Memory planning validates pool capacity and resolves its buffer root.
        _ => None,
    };
    Ok(message)
}

pub fn validate_for(ctx: &Ctx, loop_stmt: &ForObj) -> AResult<Option<String>> {
    let kind_value = loop_stmt.kind.as_raw();
    if kind_value != ForKind::kSerial.as_raw() && kind_value != ForKind::kUnrolled.as_raw() {
        let kind_name = match kind_value {
            1 => "PARALLEL".to_owned(),
            2 => "VECTORIZED".to_owned(),
            4 => "THREAD_BINDING".to_owned(),
            other => other.to_string(),
        };
        return Ok(Some(format!("For(kind={kind_name}) is unsupported")));
    }
    if loop_stmt.thread_binding.is_some() {
        return Ok(Some("For(thread_binding=...) is unsupported".to_owned()));
    }
    let mut annotations: Vec<(String, Any)> = loop_stmt
        .annotations
        .iter()
        .map(|(key, value)| (ffi_text(&key), value))
        .collect();
    annotations.sort_by(|left, right| left.0.cmp(&right.0));
    let names: Vec<String> = annotations
        .iter()
        .map(|(key, _)| (key).to_string())
        .collect();
    if kind_value == ForKind::kUnrolled.as_raw() {
        if !annotations.is_empty() {
            return Ok(Some(format!(
                "For(kind=UNROLLED) has unsupported annotations {:?}",
                &names
            )));
        }
        return Ok(None);
    }
    let unknown: Vec<String> = annotations
        .iter()
        .filter(|(key, _)| !ctx.schema.serial_loop_annotations.contains(key))
        .map(|(key, _)| (key).to_string())
        .collect();
    if !unknown.is_empty() {
        return Ok(Some(format!(
            "For(kind=SERIAL) has unsupported annotations {:?}",
            &unknown
        )));
    }
    let keys: std::collections::HashSet<String> =
        annotations.iter().map(|(key, _)| key.clone()).collect();
    if keys == ctx.schema.serial_loop_annotations {
        return Ok(Some(
            "For(kind=SERIAL) cannot request and disable unrolling simultaneously".to_owned(),
        ));
    }
    for (key, value) in &annotations {
        let is_true = value.type_index() == TypeIndex::kTVMFFIBool as i32
            && bool::try_from(value.clone()).unwrap_or(false);
        if key == "pragma_unroll" {
            let positive_int = value.type_index() == TypeIndex::kTVMFFIInt as i32
                && i64::try_from(value.clone()).is_ok_and(|factor| factor > 0);
            if !(is_true || positive_int) {
                return Ok(Some(
                    "For(kind=SERIAL) annotation 'pragma_unroll' must be true or a positive integer factor".to_owned(),
                ));
            }
        } else if !is_true {
            return Ok(Some(format!(
                "For(kind=SERIAL) annotation {:?} must be the boolean true",
                key
            )));
        }
    }
    Ok(None)
}
