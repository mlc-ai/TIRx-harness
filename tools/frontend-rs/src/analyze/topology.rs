//! Launch topology extraction and the launch region and For validators.

use crate::analyze::util::json_object;
use crate::tvm_compat::int_value;
use tvm_tirx::ir::prim::{CastObj, NotObj, SelectObj};
use tvm_tirx::ir::{Expr, FloatImmObj, IntImmObj, VarObj};
use tvm_tirx::ir::{ForKind, ForObj, RegionStmtObj, SeqStmtObj, Stmt};
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::{Any, Array, Map, ObjectRefCore, String as FfiString, TypeIndex};

use super::util::{
    dtype_of, ffi_text, int_imm, kind, oref, repr_text, scalar_bind, static_string, unsupported,
    AResult, IdMap, Json,
};
use super::{topology_operands, Ctx};

const WARP_SIZE: i64 = 32;
const WARPS_PER_WARPGROUP: i64 = 4;

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

pub const DEVICE_ENTRY: &str = "tirx.device_entry";
/// The launch region `TileDispatch` makes of a device entry.
pub const DEVICE_SCOPE: &str = "tirx.device_scope";
pub const LAUNCH_THREAD: &str = "tirx.launch_thread";

/// Whether `region` launches a kernel with its `cuda.launch_fields` operands.
pub fn is_launch_entry(region: &RegionStmtObj) -> bool {
    matches!(region.op.name.as_str(), DEVICE_ENTRY | DEVICE_SCOPE)
}
/// Launch thread tags NumSim executes, with the constraint each extent sets.
const LAUNCH_THREAD_TAGS: &[(&str, &str)] = &[
    ("blockIdx.x", "global CTAs"),
    ("clusterCtaIdx.x", "CTAs per cluster"),
    ("threadIdx.x", "threads per CTA"),
];

/// The `(field, value)` operands a `tirx.device_entry` region's
/// `cuda.launch_fields` names.
pub fn launch_values(region: &RegionStmtObj) -> AResult<Vec<(String, Expr)>> {
    let Some(fields) = region.attrs.dict.get(&FfiString::from("cuda.launch_fields"))? else {
        return Ok(Vec::new());
    };
    let fields: Array<FfiString> = fields.try_into()?;
    if fields.len() != region.args.len() {
        return unsupported("device_entry launch fields must describe every launch operand");
    }
    Ok(fields.iter().map(|field| ffi_text(&field)).zip(region.args.iter()).collect())
}

/// A `tirx.device_entry` region's `cuda.kernel_attrs`.
pub fn kernel_attrs(region: &RegionStmtObj) -> AResult<Vec<(String, i64)>> {
    let Some(attrs) = region.attrs.dict.get(&FfiString::from("cuda.kernel_attrs"))? else {
        return Ok(Vec::new());
    };
    let attrs: Map<FfiString, i64> = attrs.try_into()?;
    Ok(attrs.iter().map(|(key, value)| (ffi_text(&key), value)).collect())
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
    if let Some(region) = node.as_node::<RegionStmtObj>() {
        match ffi_text(&region.op.name).as_str() {
            DEVICE_ENTRY | DEVICE_SCOPE => {
                for (field, value) in launch_values(region)? {
                    let constraint = match field.as_str() {
                        "grid.x" => "global CTAs",
                        "block.x" => "threads per CTA",
                        "cluster.x" => "CTAs per cluster",
                        _ if field.ends_with(".y") || field.ends_with(".z") => {
                            let extent = static_int(ctx, &oref(value), environment)?;
                            if extent != 1 {
                                return unsupported(format!(
                                    "NumSim launches are one-dimensional, but {field} is {extent}"
                                ));
                            }
                            continue;
                        }
                        _ => continue,
                    };
                    let extent = static_int(ctx, &oref(value), environment)?;
                    constraints.constrain(constraint, i64::try_from(extent).unwrap_or(i64::MAX))?;
                }
                for (key, value) in kernel_attrs(region)? {
                    if key == "min_blocks_per_sm" {
                        constraints.constrain("min blocks per sm", value)?;
                    }
                }
            }
            LAUNCH_THREAD => {
                let tag = static_string(&oref(region.args.get(0)?), "launch_thread tag")?;
                let Some((_, constraint)) = LAUNCH_THREAD_TAGS.iter().find(|(name, _)| *name == tag) else {
                    return unsupported(format!(
                        "NumSim does not support direct launch thread tag {:?}",
                        &tag
                    ));
                };
                let extent = static_int(ctx, &oref(region.args.get(1)?), environment)?;
                constraints.constrain(constraint, i64::try_from(extent).unwrap_or(i64::MAX))?;
            }
            _ => {}
        }
        return visit(ctx, &Stmt::from(region.body.clone()), constraints, environment);
    }
    if let Some(bind) = scalar_bind(node) {
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
    let (clusters, ctas_per_cluster) = match (
        constraints.get("global CTAs"),
        constraints.get("CTAs per cluster"),
    ) {
        (Some(global), Some(per_cluster)) => {
            if global % per_cluster != 0 {
                return unsupported(format!(
                    "global CTA count {global} is not divisible by cluster size {per_cluster}"
                ));
            }
            (global / per_cluster, per_cluster)
        }
        (Some(global), None) => (global, 1),
        (None, Some(per_cluster)) => (1, per_cluster),
        (None, None) => (1, 1),
    };
    let warps_per_cta = match constraints.get("threads per CTA") {
        Some(threads) if threads % WARP_SIZE != 0 => {
            return unsupported(format!(
                "CTA thread extent {threads} is not a whole number of warps"
            ))
        }
        Some(threads) => threads / WARP_SIZE,
        None => 1,
    };
    Ok(LaunchTopology {
        clusters,
        ctas_per_cluster,
        warps_per_cta,
        warps_per_warpgroup: WARPS_PER_WARPGROUP,
        min_blocks_per_sm: constraints.get("min blocks per sm").unwrap_or(1),
    })
}

/// `tirx.device_entry` attributes: the launch operand names, the kernel
/// attributes, the shared-memory pool requirement and the compile settings.
const DEVICE_ENTRY_ATTRS: &[&str] = &[
    "cuda.launch_fields",
    "cuda.kernel_attrs",
    "cuda.smem_required",
    "backend_config",
];
/// Launch operands besides the grid, block and cluster shapes. The stream and
/// priority only schedule the launch.
const SCALAR_LAUNCH_FIELDS: &[&str] = &["dynamic_smem_bytes", "stream", "priority"];
/// Kernel attributes constrain occupancy and launch validity only; NumSim
/// executes the launch shape directly.
const KERNEL_ATTRS: &[&str] = &[
    "min_blocks_per_sm",
    "max_blocks_per_cluster",
    "max_registers_per_thread",
    "required_block_size",
    "required_block_x",
    "required_block_y",
    "required_block_z",
    "required_cluster_x",
    "required_cluster_y",
    "required_cluster_z",
];

/// `None` when the launch region is acceptable.
pub fn validate_region(region: &RegionStmtObj) -> AResult<Option<String>> {
    let op_name = ffi_text(&region.op.name);
    if op_name == LAUNCH_THREAD {
        let tag = region.args.get(0).ok().and_then(|tag| static_string(&oref(tag), "tag").ok());
        if !tag.as_deref().is_some_and(|tag| LAUNCH_THREAD_TAGS.iter().any(|(name, _)| *name == tag)) {
            return Ok(Some(format!(
                "RegionStmt(tirx.launch_thread) has unsupported thread tag {:?}",
                tag.unwrap_or_default()
            )));
        }
        let extent = region.args.get(1).ok().and_then(|extent| int_imm(&oref(extent)));
        let int32_var = region.body_params.len() == 1
            && dtype_of(&oref(region.body_params.get(0)?))? == "int32";
        if extent.is_none_or(|value| value <= 0) || !int32_var {
            return Ok(Some(
                "RegionStmt(tirx.launch_thread) must bind an int32 variable to a positive IntImm"
                    .to_owned(),
            ));
        }
        return Ok(None);
    }
    if !is_launch_entry(region) {
        return Ok(Some(format!("RegionStmt(op={op_name:?}) is unsupported")));
    }
    for (key, _) in region.attrs.dict.iter() {
        if !DEVICE_ENTRY_ATTRS.contains(&key.as_str()) {
            return Ok(Some(format!(
                "RegionStmt({op_name}) attribute {:?} has unsupported numerical semantics",
                key.as_str()
            )));
        }
    }
    let launch = launch_values(region)?;
    for (field, value) in &launch {
        let shape = ["grid.", "block.", "cluster.", "preferred_cluster."]
            .iter()
            .any(|prefix| field.starts_with(prefix));
        if !shape && !SCALAR_LAUNCH_FIELDS.contains(&field.as_str()) {
            return Ok(Some(format!(
                "RegionStmt({op_name}) launch field {field:?} has unsupported numerical semantics"
            )));
        }
        if field == "dynamic_smem_bytes" && int_imm(&oref(value.clone())).is_none_or(|bytes| bytes < 0) {
            return Ok(Some(format!(
                "RegionStmt({op_name}) dynamic_smem_bytes must be a non-negative IntImm"
            )));
        }
    }
    let shape = |prefix: &str| -> Vec<Option<i64>> {
        ["x", "y", "z"]
            .iter()
            .map(|axis| {
                launch
                    .iter()
                    .find(|(field, _)| *field == format!("{prefix}.{axis}"))
                    .map_or(Some(1), |(_, value)| int_imm(&oref(value.clone())))
            })
            .collect()
    };
    let preferred = launch.iter().any(|(field, _)| field.starts_with("preferred_cluster."));
    if preferred && shape("preferred_cluster") != shape("cluster") {
        return Ok(Some(format!(
            "RegionStmt({op_name}) preferred_cluster must equal cluster"
        )));
    }
    for (key, value) in kernel_attrs(region)? {
        if !KERNEL_ATTRS.contains(&key.as_str()) || value <= 0 {
            return Ok(Some(format!(
                "RegionStmt({op_name}) kernel attribute {key}={value} is unsupported"
            )));
        }
    }
    Ok(None)
}

pub fn validate_for(ctx: &Ctx, loop_stmt: &ForObj) -> AResult<Option<String>> {
    let kind_value = loop_stmt.kind;
    if kind_value != ForKind::Default as i32 && kind_value != ForKind::Unrolled as i32 {
        let kind_name = match kind_value {
            1 => "PARALLEL".to_owned(),
            2 => "VECTORIZED".to_owned(),
            other => other.to_string(),
        };
        return Ok(Some(format!("For(kind={kind_name}) is unsupported")));
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
    if kind_value == ForKind::Unrolled as i32 {
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
        if key == "tirx.pragma_unroll" {
            let positive_int = value.type_index() == TypeIndex::kTVMFFIInt as i32
                && i64::try_from(value.clone()).is_ok_and(|factor| factor > 0);
            if !(is_true || positive_int) {
                return Ok(Some(
                    "For(kind=SERIAL) annotation 'tirx.pragma_unroll' must be true or a positive integer factor".to_owned(),
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
