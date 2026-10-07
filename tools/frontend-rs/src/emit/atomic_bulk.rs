//! Validation and emission of the atomic_bulk instruction family.

use crate::analyze::util::{capitalize, dtype_of, ffi_error, oref, unsupported, AResult, Failure};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::{shared_pointer_source, SharedAddressForms};
use crate::emit::pure::reinterpret_atom;
use crate::emit::raw_tma::reinterpret;
use crate::emit::register_call::{marker, table_marker, PTX_TYPE_MARKERS};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use crate::tables::is_integer_dtype;
use crate::tables::{expr_rust_type, v2_memory_type_rust};
use tvm::ir::{CallObj, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

pub const BULK_G2S_CLUSTER_CALLS: [&str; 4] = [
    "tirx.ptx.cp_async_bulk_g2s_cluster",
    "tirx.ptx.cp_async_bulk_g2s_cluster_report",
    "tirx.ptx.cp_async_bulk_g2s_cluster_multicast16",
    "tirx.ptx.cp_async_bulk_g2s_cluster_multicast32",
];
pub const BULK_S2C_CALLS: [&str; 2] = [
    "tirx.ptx.cp_async_bulk_s2c",
    "tirx.ptx.cp_reduce_async_bulk_s2c",
];
pub const BULK_REDUCE_S2G_CALLS: [&str; 2] = [
    "tirx.ptx.cp_reduce_async_bulk_s2g",
    "tirx.ptx.cp_reduce_async_bulk_s2g_f32_noftz",
];

pub const PTX_REDUCTIONS: [&str; 3] = [
    "tirx.ptx.red",
    "tirx.ptx.red_half",
    "tirx.ptx.red_f32_noftz",
];
pub const PTX_VECTOR_REDUCTIONS: [&str; 3] = [
    "tirx.ptx.red_vec_f32",
    "tirx.ptx.red_vec_half",
    "tirx.ptx.red_vec_f32_noftz",
];
pub const PTX_VECTOR_ATOMIC_CALLS: [&str; 9] = [
    "tirx.ptx.atom_vec_f32",
    "tirx.ptx.atom_vec_f32_bitbucket",
    "tirx.ptx.red_vec_f32",
    "tirx.ptx.atom_vec_half",
    "tirx.ptx.atom_vec_half_bitbucket",
    "tirx.ptx.red_vec_half",
    "tirx.ptx.atom_vec_f32_noftz",
    "tirx.ptx.atom_vec_f32_noftz_bitbucket",
    "tirx.ptx.red_vec_f32_noftz",
];
pub const PTX_CAS_CALLS: [&str; 2] = ["tirx.ptx.atom_cas", "tirx.ptx.atom_cas_bitbucket"];
pub const PTX_SCALAR_ATOMIC_CALLS: [&str; 11] = [
    "tirx.ptx.atom",
    "tirx.ptx.atom_exch",
    "tirx.ptx.atom_half",
    "tirx.ptx.atom_bitbucket",
    "tirx.ptx.atom_exch_bitbucket",
    "tirx.ptx.atom_half_bitbucket",
    "tirx.ptx.red",
    "tirx.ptx.red_half",
    "tirx.ptx.atom_f32_noftz",
    "tirx.ptx.atom_f32_noftz_bitbucket",
    "tirx.ptx.red_f32_noftz",
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    And,
    Or,
    Xor,
    Exchange,
    Increment,
    Decrement,
    Minimum,
    Maximum,
}

impl AtomicOp {
    pub fn marker(self) -> String {
        let name = match self {
            AtomicOp::Add => "Add",
            AtomicOp::And => "BitAnd",
            AtomicOp::Or => "BitOr",
            AtomicOp::Xor => "BitXor",
            AtomicOp::Exchange => "Exchange",
            AtomicOp::Increment => "Increment",
            AtomicOp::Decrement => "Decrement",
            AtomicOp::Minimum => "Minimum",
            AtomicOp::Maximum => "Maximum",
        };
        format!("v2::mem::variant::{name}")
    }
}

fn bitwise_atomic_type(ptx_type: &str) -> Option<&'static str> {
    match ptx_type {
        "b32" => Some("uint32"),
        "b64" => Some("uint64"),
        _ => None,
    }
}

fn ordered_atomic_type(ptx_type: &str) -> Option<&'static str> {
    match ptx_type {
        "u32" => Some("uint32"),
        "s32" => Some("int32"),
        "u64" => Some("uint64"),
        "s64" => Some("int64"),
        _ => None,
    }
}

/// `ptx_atomic_signature`.
pub fn ptx_atomic_signature(operation: &str, ptx_type: &str) -> Option<(AtomicOp, &'static str)> {
    let (kind, dtype) = match operation {
        "and" => (AtomicOp::And, bitwise_atomic_type(ptx_type)),
        "or" => (AtomicOp::Or, bitwise_atomic_type(ptx_type)),
        "xor" => (AtomicOp::Xor, bitwise_atomic_type(ptx_type)),
        "exch" => (
            AtomicOp::Exchange,
            if ptx_type == "b128" {
                Some("uint128")
            } else {
                bitwise_atomic_type(ptx_type)
            },
        ),
        "add" => (
            AtomicOp::Add,
            match ptx_type {
                "u32" => Some("uint32"),
                "s32" => Some("int32"),
                "u64" => Some("uint64"),
                "f16" => Some("float16"),
                "bf16" => Some("bfloat16"),
                "f16x2" => Some("float16x2"),
                "bf16x2" => Some("bfloat16x2"),
                "f32" => Some("float32"),
                "f64" => Some("float64"),
                _ => None,
            },
        ),
        "inc" => (
            AtomicOp::Increment,
            if ptx_type == "u32" {
                Some("uint32")
            } else {
                None
            },
        ),
        "dec" => (
            AtomicOp::Decrement,
            if ptx_type == "u32" {
                Some("uint32")
            } else {
                None
            },
        ),
        "min" => (AtomicOp::Minimum, ordered_atomic_type(ptx_type)),
        "max" => (AtomicOp::Maximum, ordered_atomic_type(ptx_type)),
        _ => return None,
    };
    dtype.map(|dtype| (kind, dtype))
}

/// `ptx_atomic_value_dtypes`.
pub fn ptx_atomic_value_dtypes(ptx_type: &str) -> &'static [&'static str] {
    match ptx_type {
        "b16" => &["uint16", "int16", "float16", "bfloat16"],
        "b128" => &["uint128", "int128"],
        "b32" => &["uint32", "int32", "float32"],
        "u32" => &["uint32", "int32"],
        "s32" => &["uint32", "int32"],
        "f32" => &["float32"],
        "f16" => &["uint16", "int16"],
        "bf16" => &["uint16", "int16"],
        "f16x2" => &["uint32", "int32", "float32"],
        "bf16x2" => &["uint32", "int32", "float32"],
        "b64" => &["uint64", "int64", "float64"],
        "u64" => &["uint64", "int64"],
        "s64" => &["uint64", "int64"],
        "f64" => &["float64"],
        _ => &[],
    }
}

/// The bit-size types whose destinations accept any
/// compatible register.
fn atom_cas_type(ptx_type: &str) -> Option<&'static str> {
    match ptx_type {
        "b128" => Some("uint128"),
        "b16" => Some("uint16"),
        "b32" => Some("uint32"),
        "b64" => Some("uint64"),
        _ => None,
    }
}

/// Bit-size instructions accept compatible
/// registers, preserving their bits; other destinations retain the current TVM
/// schema's exact dtype contract. `Some(None)` is a sunk destination lane.
fn check_atomic_destinations(
    decoded: &DecodedPtx,
    destination: Option<&Option<ObjectRef>>,
    result_dtype: &str,
) -> AResult<()> {
    let Some(destination) = destination else {
        return Ok(());
    };
    let ptx_type = decoded.modifier("type")?;
    let destination_dtype = match result_dtype {
        "float16" | "bfloat16" => "uint16",
        "float16x2" | "bfloat16x2" => "uint32",
        other => other,
    };
    let compatible = match destination {
        Some(node) if node.as_node::<TensorLoadObj>().is_some() => {
            let dtype = dtype_of(node)?;
            if atom_cas_type(ptx_type).is_some() {
                ptx_atomic_value_dtypes(ptx_type).contains(&dtype.as_str())
            } else {
                dtype == destination_dtype
            }
        }
        _ => false,
    };
    if !compatible {
        return unsupported(format!(
            "{}.d must be TensorLoad lvalues compatible with {ptx_type}",
            decoded.op_name
        ));
    }
    let node = destination.as_ref().expect("validated destination");
    // `C_BINDING[dtype].carrier != C_BINDING[dtype].c_type`.
    if decoded.preserve_dst
        && matches!(
            dtype_of(node)?.as_str(),
            "uint8" | "int8" | "float16" | "bfloat16"
        )
    {
        return unsupported(format!(
            "{}.d cannot preserve a destination that binds through a TVM carrier",
            decoded.op_name
        ));
    }
    Ok(())
}

pub fn v2_atomic_space_rust(space: Option<&str>) -> AResult<&'static str> {
    Ok(match space {
        None | Some("") | Some("generic") => "v2::Generic",
        Some("global") => "v2::Global",
        Some("shared") => "v2::Shared",
        Some("shared::cta") => "v2::SharedCta",
        Some("shared::cluster") => "v2::SharedCluster",
        Some(other) => {
            return unsupported(format!("no v2 atom/red state-space marker for {:?}", other))
        }
    })
}

/// `v2_atomic_type_rust`.
pub fn v2_atomic_type_rust(ctx: &Ctx, dtype: &str, compare_exchange: bool) -> AResult<String> {
    if !compare_exchange {
        let marker = match dtype {
            "float16" => Some("F16"),
            "bfloat16" => Some("Bf16"),
            "float16x2" => Some("F16x2"),
            "bfloat16x2" => Some("Bf16x2"),
            "float32x2" => Some("F32x2"),
            "float32x4" => Some("F32x4"),
            _ => None,
        };
        if let Some(marker) = marker {
            return Ok(format!("v2::mem::variant::{marker}"));
        }
    }
    v2_memory_type_rust(ctx.schema, dtype)
}

// ----------------------------------------------------------------------
// CUDA `atomic_add` / `atomic_cas` expressions.
// ----------------------------------------------------------------------

pub struct CudaAtomicCall {
    pub op_name: String,
    pub address: ObjectRef,
    pub value: ObjectRef,
    pub result_dtype: String,
    pub compare: Option<ObjectRef>,
    pub operation: Option<AtomicOp>,
}

const CUDA_ATOMIC_ADD_DTYPES: [&str; 11] = [
    "int32",
    "uint32",
    "uint64",
    "float16",
    "float16x2",
    "bfloat16",
    "bfloat16x2",
    "float32",
    "float32x2",
    "float32x4",
    "float64",
];
const CUDA_ATOMIC_CAS_DTYPES: [&str; 5] = ["int32", "uint16", "uint32", "uint64", "uint64x2"];

pub fn resolve_cuda_atomic_call(
    ctx: &Ctx,
    node: &ObjectRef,
    op_name: &str,
) -> AResult<CudaAtomicCall> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("atomic lowering expects a TIRx Call"),
        ));
    };
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    let expected = if op_name == "tirx.cuda.atomic_add" {
        2
    } else {
        3
    };
    if args.len() != expected {
        return unsupported(format!(
            "{op_name} expects {expected} arguments, got {}",
            args.len()
        ));
    }
    let address = args[0].clone();
    let address_dtype = dtype_of(&address)?;
    if address_dtype != "handle" {
        return unsupported(format!(
            "{op_name}.address must be a handle, got {:?}",
            &address_dtype
        ));
    }
    let result_dtype = dtype_of(node)?;
    if op_name == "tirx.cuda.atomic_add" {
        let value = args[1].clone();
        let value_dtype = dtype_of(&value)?;
        if result_dtype != value_dtype || !CUDA_ATOMIC_ADD_DTYPES.contains(&result_dtype.as_str()) {
            return unsupported(format!(
                "{op_name} requires matching supported operands, got result={:?}, value={:?}",
                &result_dtype, &value_dtype
            ));
        }
        return Ok(CudaAtomicCall {
            op_name: op_name.to_owned(),
            address,
            value,
            result_dtype,
            compare: None,
            operation: Some(AtomicOp::Add),
        });
    }
    let (compare, value) = (args[1].clone(), args[2].clone());
    let vector_abi = ctx.schema.vector_dtype_abi(&result_dtype);
    let supported = CUDA_ATOMIC_CAS_DTYPES.contains(&result_dtype.as_str())
        || vector_abi.is_some_and(|(_, _, _, total_bits)| total_bits == 128);
    if !supported || dtype_of(&compare)? != result_dtype || dtype_of(&value)? != result_dtype {
        return unsupported(format!(
            "{op_name} requires matching supported scalar or 128-bit vector operands"
        ));
    }
    Ok(CudaAtomicCall {
        op_name: op_name.to_owned(),
        address,
        value,
        result_dtype,
        compare: Some(compare),
        operation: None,
    })
}

pub fn cuda_atomic_variant_rust(
    ctx: &Ctx,
    call: &CudaAtomicCall,
    instruction: &str,
) -> AResult<String> {
    let scalar = v2_atomic_type_rust(ctx, &call.result_dtype, instruction == "AtomCas")?;
    let space = v2_atomic_space_rust(None)?;
    let semantics = "v2::mem::variant::Relaxed<v2::mem::variant::Gpu>";
    if instruction == "AtomCas" {
        return Ok(format!(
            "v2::mem::variant::AtomCas<{scalar}, {space}, {semantics}>"
        ));
    }
    let operation = call.operation.expect("atomic operation").marker();
    Ok(format!(
        "v2::mem::variant::Atom<{scalar}, {space}, {operation}, {semantics}>"
    ))
}

// ----------------------------------------------------------------------
// Decoded PTX statements.
// ----------------------------------------------------------------------

fn integer_operand(decoded: &DecodedPtx, name: &str, value: &ObjectRef) -> AResult<()> {
    let dtype = dtype_of(value)?;
    if !is_integer_dtype(&dtype) {
        return unsupported(format!(
            "{}.{name} must be an integer, got {:?}",
            decoded.op_name, &dtype
        ));
    }
    Ok(())
}

fn decoded_atomic_ordering(decoded: &DecodedPtx, reduction: bool) -> AResult<(String, String)> {
    let semantics = match decoded.modifier("sem")? {
        "" => "relaxed".to_owned(),
        other => other.to_owned(),
    };
    let scope = match decoded.modifier("scope")? {
        "" => "gpu".to_owned(),
        other => other.to_owned(),
    };
    if !["relaxed", "acquire", "release", "acq_rel"].contains(&semantics.as_str()) {
        return unsupported(format!(
            "{} has unsupported memory semantic {:?}",
            decoded.op_name, &semantics
        ));
    }
    if reduction && (semantics == "acquire" || semantics == "acq_rel") {
        return unsupported(format!(
            "{} reduction cannot be an acquire endpoint",
            decoded.op_name
        ));
    }
    if !["cta", "cluster", "gpu", "sys"].contains(&scope.as_str()) {
        return unsupported(format!(
            "{} has unsupported scope {:?}",
            decoded.op_name, &scope
        ));
    }
    Ok((semantics, scope))
}

fn decoded_atomic_cache_policy(decoded: &DecodedPtx, space: &str) -> AResult<Option<ObjectRef>> {
    let cache_policy = decoded.cache_policy_operand()?;
    if cache_policy.is_some() && space != "global" {
        return unsupported(format!(
            "{} cache policy is valid only for global memory",
            decoded.op_name
        ));
    }
    Ok(cache_policy)
}

pub fn f32_noftz(decoded: &DecodedPtx) -> AResult<bool> {
    Ok(decoded.modifier("type")? == "f32" && decoded.modifier_or_empty("noftz") == "noftz")
}

fn space_or_generic(space: &str) -> &str {
    if space.is_empty() {
        "generic"
    } else {
        space
    }
}

fn require_atomic_address(decoded: &DecodedPtx, address: &ObjectRef, space: &str) -> AResult<()> {
    let address_dtype = dtype_of(address)?;
    let allowed = if space.starts_with("shared") {
        address_dtype == "handle" || address_dtype == "uint32"
    } else {
        address_dtype == "handle"
    };
    if !allowed {
        return unsupported(format!(
            "{}.addr has invalid {} address dtype {:?}",
            decoded.op_name,
            space_or_generic(space),
            &address_dtype
        ));
    }
    Ok(())
}

const ATOMIC_SPACES: [&str; 5] = ["", "global", "shared", "shared::cta", "shared::cluster"];

pub struct AtomCasParts {
    pub destination: Option<ObjectRef>,
    pub address: ObjectRef,
    pub compare: ObjectRef,
    pub value: ObjectRef,
    pub result_dtype: String,
    pub space: String,
    pub semantics: String,
    pub scope: String,
}

pub fn decoded_atom_cas_parts(decoded: &DecodedPtx) -> AResult<AtomCasParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let destination = if op_name.ends_with("_bitbucket") {
        None
    } else {
        Some(decoded.scalar_lane("d")?)
    };
    let address = decoded.scalar_operand("addr")?;
    let compare = decoded.scalar_operand("compare")?;
    let value = decoded.scalar_operand("value")?;
    let (semantics, scope) = decoded_atomic_ordering(decoded, false)?;
    let space = decoded.modifier("space")?.to_owned();
    if !ATOMIC_SPACES.contains(&space.as_str()) {
        return unsupported(format!(
            "{op_name} has unsupported state space {:?}",
            &space
        ));
    }
    let ptx_type = decoded.modifier("type")?.to_owned();
    let Some(result_dtype) = atom_cas_type(&ptx_type) else {
        return unsupported(format!("{op_name} has unsupported type .{ptx_type}"));
    };
    check_atomic_destinations(decoded, destination.as_ref(), result_dtype)?;
    let allowed = ptx_atomic_value_dtypes(&ptx_type);
    for (name, operand) in [("compare", &compare), ("value", &value)] {
        let operand_dtype = dtype_of(operand)?;
        if !allowed.contains(&operand_dtype.as_str()) {
            return unsupported(format!(
                "{op_name} {ptx_type} {name} operand cannot consume {:?}",
                &operand_dtype
            ));
        }
    }
    require_atomic_address(decoded, &address, &space)?;
    Ok(AtomCasParts {
        destination: destination.flatten(),
        address,
        compare,
        value,
        result_dtype: result_dtype.to_owned(),
        space,
        semantics,
        scope,
    })
}

pub struct AtomicParts {
    pub destination: Option<ObjectRef>,
    pub address: ObjectRef,
    pub value: ObjectRef,
    pub operation: AtomicOp,
    pub result_dtype: String,
    pub space: String,
    pub semantics: String,
    pub scope: String,
}

pub fn decoded_atomic_parts(decoded: &DecodedPtx) -> AResult<AtomicParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let reduction = PTX_REDUCTIONS.contains(&op_name);
    let destination = if reduction || op_name.ends_with("_bitbucket") {
        None
    } else {
        Some(decoded.scalar_lane("d")?)
    };
    let address = decoded.scalar_operand("addr")?;
    let value = decoded.scalar_operand("value")?;
    let (semantics, scope) = decoded_atomic_ordering(decoded, reduction)?;
    let space = decoded.modifier("space")?.to_owned();
    if op_name.contains("_half") && decoded.modifier("noftz")? != "noftz" {
        return unsupported(format!("{op_name} requires noftz half-precision addition"));
    }
    if !ATOMIC_SPACES.contains(&space.as_str()) {
        return unsupported(format!(
            "{op_name} has unsupported state space {:?}",
            &space
        ));
    }
    let op_modifier = decoded.modifier("op")?.to_owned();
    let type_modifier = decoded.modifier("type")?.to_owned();
    let Some((operation, result_dtype)) = ptx_atomic_signature(&op_modifier, &type_modifier) else {
        return unsupported(format!(
            "{op_name} has unsupported {op_modifier}.{type_modifier} signature"
        ));
    };
    let value_dtype = dtype_of(&value)?;
    if !ptx_atomic_value_dtypes(&type_modifier).contains(&value_dtype.as_str()) {
        return unsupported(format!(
            "{op_name} {type_modifier} operand cannot consume {:?}",
            &value_dtype
        ));
    }
    check_atomic_destinations(decoded, destination.as_ref(), result_dtype)?;
    require_atomic_address(decoded, &address, &space)?;
    decoded_atomic_cache_policy(decoded, &space)?;
    Ok(AtomicParts {
        destination: destination.flatten(),
        address,
        value,
        operation,
        result_dtype: result_dtype.to_owned(),
        space,
        semantics,
        scope,
    })
}

pub struct VectorAtomicParts {
    pub destinations: Vec<ObjectRef>,
    pub address: ObjectRef,
    pub values: Vec<ObjectRef>,
    pub operation: AtomicOp,
    pub result_dtype: String,
    pub space: String,
    pub semantics: String,
    pub scope: String,
}

pub fn decoded_vector_atomic_parts(decoded: &DecodedPtx) -> AResult<VectorAtomicParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let (semantics, scope) =
        decoded_atomic_ordering(decoded, PTX_VECTOR_REDUCTIONS.contains(&op_name))?;
    let space = decoded.modifier("space")?.to_owned();
    if !(space.is_empty() || space == "global") {
        return unsupported(format!(
            "{op_name} has unsupported state space {:?}",
            &space
        ));
    }
    let ptx_type = decoded.modifier("type")?.to_owned();
    let op = decoded.modifier("op")?.to_owned();
    let operation = match op.as_str() {
        "add" => Some(AtomicOp::Add),
        "min" if ptx_type != "f32" => Some(AtomicOp::Minimum),
        "max" if ptx_type != "f32" => Some(AtomicOp::Maximum),
        _ => None,
    };
    let Some(operation) = operation
        .filter(|_| ["f32", "f16", "bf16", "f16x2", "bf16x2"].contains(&ptx_type.as_str()))
    else {
        return unsupported(format!("{op_name} has unsupported vector operation/type"));
    };
    let half = ptx_type != "f32";
    if half && decoded.modifier("noftz")? != "noftz" {
        return unsupported(format!("{op_name} requires noftz half-vector arithmetic"));
    }
    let packed = ptx_type.ends_with("x2");
    let carrier = match (half, packed) {
        (false, _) => "float32",
        (true, true) => "uint32",
        (true, false) => "uint16",
    };
    let vec_modifier = decoded.modifier("vec")?.to_owned();
    let width: usize = match vec_modifier.as_str() {
        "v2" => 2,
        "v4" => 4,
        "v8" if half && !packed => 8,
        _ => {
            return unsupported(format!(
                "{op_name} has unsupported vector width {:?}",
                &vec_modifier
            ))
        }
    };
    let discard = op_name.ends_with("_bitbucket") || PTX_VECTOR_REDUCTIONS.contains(&op_name);
    let destination_lanes: Vec<Option<ObjectRef>> = if discard {
        Vec::new()
    } else {
        decoded.operand("d")?.to_vec()
    };
    let value_lanes: Vec<Option<ObjectRef>> = decoded.operand("value")?.to_vec();
    if (!discard && destination_lanes.len() != width) || value_lanes.len() != width {
        return unsupported(format!(
            "{op_name} {vec_modifier} requires {width} destination/value lanes"
        ));
    }
    let mut destinations = Vec::new();
    let mut destinations_valid = true;
    for lane in &destination_lanes {
        match lane {
            Some(node) if node.as_node::<TensorLoadObj>().is_some() => {
                if dtype_of(node)? != carrier {
                    destinations_valid = false;
                }
                destinations.push(node.clone());
            }
            _ => destinations_valid = false,
        }
    }
    if !destinations_valid {
        return unsupported(format!(
            "{op_name}.d lanes must be {carrier} TensorLoad lvalues"
        ));
    }
    let mut values = Vec::new();
    for lane in &value_lanes {
        match lane {
            Some(node) => {
                if dtype_of(node)? != carrier {
                    return unsupported(format!("{op_name}.value lanes must be {carrier}"));
                }
                values.push(node.clone());
            }
            None => {
                return Err(crate::analyze::util::Failure::Ffi(
                    crate::analyze::util::ffi_error("sunk lane in a vector atomic value"),
                ))
            }
        }
    }
    let address = decoded.scalar_operand("addr")?;
    let address_dtype = dtype_of(&address)?;
    if address_dtype != "handle" {
        return unsupported(format!(
            "{op_name}.addr has invalid {} address dtype {:?}",
            space_or_generic(&space),
            &address_dtype
        ));
    }
    decoded_atomic_cache_policy(decoded, &space)?;
    let result_dtype = if half {
        format!(
            "{}x{}",
            if ptx_type.starts_with("bf16") {
                "bfloat16"
            } else {
                "float16"
            },
            width * if packed { 2 } else { 1 }
        )
    } else {
        format!("float32x{width}")
    };
    Ok(VectorAtomicParts {
        destinations,
        address,
        values,
        operation,
        result_dtype,
        space,
        semantics,
        scope,
    })
}

pub fn decoded_atomic_semantics_rust(semantics: &str, scope: &str) -> String {
    let order = match semantics {
        "relaxed" => "Relaxed",
        "acquire" => "Acquire",
        "release" => "Release",
        _ => "AcqRel",
    };
    let scope_marker = crate::tables::scope_marker(scope);
    format!("v2::mem::variant::{order}<v2::mem::variant::{scope_marker}>")
}

#[allow(clippy::too_many_arguments)]
pub fn decoded_atomic_variant(
    ctx: &Ctx,
    instruction: &str,
    operation: AtomicOp,
    result_dtype: &str,
    space: &str,
    semantics: &str,
    scope: &str,
    noftz: bool,
    vector_half: bool,
) -> AResult<String> {
    let scalar = if vector_half {
        let (dtype, width) = result_dtype
            .rsplit_once('x')
            .expect("packed half vector dtype");
        format!(
            "v2::mem::variant::HalfVector<{}, {width}>",
            v2_atomic_type_rust(ctx, dtype, false)?
        )
    } else {
        v2_atomic_type_rust(ctx, result_dtype, false)?
    };
    let mut operation_marker = operation.marker();
    if noftz {
        operation_marker.push_str("<true>");
    }
    Ok(format!(
        "v2::mem::variant::{instruction}<{scalar}, {}, {operation_marker}, {}>",
        v2_atomic_space_rust(Some(space))?,
        decoded_atomic_semantics_rust(semantics, scope)
    ))
}

pub struct StBulkParts {
    pub address: ObjectRef,
    pub num_bytes: ObjectRef,
    pub space: String,
}

pub fn decoded_st_bulk_parts(decoded: &DecodedPtx) -> AResult<StBulkParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let weak = decoded.modifier("weak")?.to_owned();
    if weak != "" && weak != "weak" {
        return unsupported(format!(
            "{op_name} has unsupported weak modifier {:?}",
            &weak
        ));
    }
    let space = decoded.modifier("space")?.to_owned();
    if space != "" && space != "shared::cta" {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    let address = decoded.scalar_operand("addr")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let address_dtype = dtype_of(&address)?;
    let allowed = if space == "shared::cta" {
        address_dtype == "handle" || address_dtype == "uint32"
    } else {
        address_dtype == "handle"
    };
    if !allowed {
        return unsupported(format!(
            "{op_name}.addr has invalid {} address dtype {:?}",
            space_or_generic(&space),
            &address_dtype
        ));
    }
    let size_dtype = dtype_of(&num_bytes)?;
    if !["uint32", "int32", "uint64", "int64"].contains(&size_dtype.as_str()) {
        return unsupported(format!(
            "{op_name}.size must be a 32- or 64-bit integer, got {:?}",
            &size_dtype
        ));
    }
    Ok(StBulkParts {
        address,
        num_bytes,
        space,
    })
}

pub fn copy_report_pattern(decoded: &DecodedPtx) -> AResult<i64> {
    let report = decoded.modifier_or_empty("report");
    if report.is_empty() || report == "mbarrier::report::disabled" {
        return Ok(0);
    }
    let digits = report.rsplit("::").next().unwrap_or(report);
    i64::from_str_radix(digits, 16).map_err(|_| {
        crate::analyze::util::Failure::Ffi(crate::analyze::util::ffi_error(&format!(
            "invalid literal for int() with base 16: {:?}",
            digits
        )))
    })
}

/// `None` for weak copies, the scope of relaxed b128 copies.
fn bulk_copy_scope(decoded: &DecodedPtx) -> AResult<Option<String>> {
    let semantics = match decoded.modifier("sem")? {
        "" => "weak",
        other => other,
    };
    let scope = decoded.modifier("scope")?;
    let element_type = decoded.modifier("type")?;
    if semantics == "weak" && scope.is_empty() && element_type.is_empty() {
        return Ok(None);
    }
    let scopes: &[&str] = if decoded.op_name == "tirx.ptx.cp_async_bulk_s2c" {
        &["cta", "cluster"]
    } else {
        &["cta", "cluster", "gpu", "sys"]
    };
    if semantics == "relaxed" && scopes.contains(&scope) && element_type == "b128" {
        return Ok(Some(scope.to_owned()));
    }
    unsupported(format!(
        "{} requires weak or relaxed.scope.b128 semantics",
        decoded.op_name
    ))
}

pub struct BulkG2sCtaParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub num_bytes: ObjectRef,
    pub barrier: ObjectRef,
    pub ignore_left: Option<ObjectRef>,
    pub ignore_right: Option<ObjectRef>,
    pub report_pattern: i64,
    pub scope: Option<String>,
}

pub fn decoded_bulk_g2s_cta_parts(decoded: &DecodedPtx) -> AResult<BulkG2sCtaParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let report_pattern = copy_report_pattern(decoded)?;
    let scope = bulk_copy_scope(decoded)?;
    decoded.require_modifiers(&[
        ("api", "async"),
        ("kind", "bulk"),
        ("dst", "shared::cta"),
        ("src", "global"),
        ("completion", "mbarrier::complete_tx::bytes"),
    ])?;
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let barrier = decoded.scalar_operand("mbar")?;
    let ignore_left = decoded.optional_scalar_operand("ignore_bytes_left")?;
    let ignore_right = decoded.optional_scalar_operand("ignore_bytes_right")?;
    let cache_policy = decoded.optional_scalar_operand("cache_policy")?;
    integer_operand(decoded, "size", &num_bytes)?;
    let ignore_oob = decoded.modifier("ignore_oob")? == "ignore_oob";
    if ignore_oob != (ignore_left.is_some() && ignore_right.is_some()) {
        return unsupported(format!(
            "{op_name} ignore_oob modifier and ignored-byte operands disagree"
        ));
    }
    for (name, value) in [
        ("ignore_bytes_left", &ignore_left),
        ("ignore_bytes_right", &ignore_right),
    ] {
        if let Some(value) = value {
            integer_operand(decoded, name, value)?;
        }
    }
    let has_cache_hint = decoded.modifier("cache")? == "L2::cache_hint";
    decoded.require_cache_policy(cache_policy.as_ref(), has_cache_hint)?;
    Ok(BulkG2sCtaParts {
        destination,
        source,
        num_bytes,
        barrier,
        ignore_left,
        ignore_right,
        report_pattern,
        scope,
    })
}

pub struct BulkG2sClusterParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub num_bytes: ObjectRef,
    pub barrier: ObjectRef,
    pub cta_mask: Option<ObjectRef>,
    pub report_pattern: i64,
    pub scope: Option<String>,
}

pub fn decoded_bulk_g2s_cluster_parts(decoded: &DecodedPtx) -> AResult<BulkG2sClusterParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let report_pattern = copy_report_pattern(decoded)?;
    let scope = bulk_copy_scope(decoded)?;
    decoded.require_modifiers(&[
        ("api", "async"),
        ("kind", "bulk"),
        ("dst", "shared::cluster"),
        ("src", "global"),
        ("completion", "mbarrier::complete_tx::bytes"),
    ])?;
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let barrier = decoded.scalar_operand("mbar")?;
    let cta_mask = decoded.optional_scalar_operand("cta_mask")?;
    let cache_policy = decoded.optional_scalar_operand("cache_policy")?;
    integer_operand(decoded, "size", &num_bytes)?;
    let multicast = !decoded.modifier("multicast")?.is_empty();
    if multicast != cta_mask.is_some() {
        return unsupported(format!(
            "{op_name} multicast modifier and CTA-mask operand disagree"
        ));
    }
    if let Some(mask) = &cta_mask {
        integer_operand(decoded, "cta_mask", mask)?;
    }
    let has_cache_hint = decoded.modifier("cache")? == "L2::cache_hint";
    decoded.require_cache_policy(cache_policy.as_ref(), has_cache_hint)?;
    Ok(BulkG2sClusterParts {
        destination,
        source,
        num_bytes,
        barrier,
        cta_mask,
        report_pattern,
        scope,
    })
}

pub struct BulkS2cParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub num_bytes: ObjectRef,
    pub barrier: ObjectRef,
    /// The reduction marker of `cp.reduce`.
    pub reduction: Option<String>,
    pub scope: Option<String>,
}

pub fn decoded_bulk_s2c_parts(decoded: &DecodedPtx) -> AResult<BulkS2cParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let scope = if op_name == "tirx.ptx.cp_async_bulk_s2c" {
        bulk_copy_scope(decoded)?
    } else {
        None
    };
    decoded.require_modifiers(&[
        ("api", "async"),
        ("kind", "bulk"),
        ("dst", "shared::cluster"),
        ("src", "shared::cta"),
        ("completion", "mbarrier::complete_tx::bytes"),
    ])?;
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let barrier = decoded.scalar_operand("mbar")?;
    integer_operand(decoded, "size", &num_bytes)?;
    for (name, value) in [
        ("dst_mem", &destination),
        ("src_mem", &source),
        ("mbar", &barrier),
    ] {
        let dtype = dtype_of(value)?;
        if dtype != "handle" && dtype != "uint32" {
            return unsupported(format!(
                "{op_name}.{name} must be a shared pointer or uint32 shared address, got {:?}",
                &dtype
            ));
        }
    }
    let reduction = bulk_s2c_reduction_variant(decoded)?;
    Ok(BulkS2cParts {
        destination,
        source,
        num_bytes,
        barrier,
        reduction,
        scope,
    })
}

fn bulk_s2c_reduction_variant(decoded: &DecodedPtx) -> AResult<Option<String>> {
    if decoded.op_name != "tirx.ptx.cp_reduce_async_bulk_s2c" {
        return Ok(None);
    }
    let signature = ptx_atomic_signature(decoded.modifier("redop")?, decoded.modifier("type")?);
    let Some((operation, scalar)) = signature.and_then(|(operation, dtype)| {
        let scalar = match dtype {
            "uint32" => "u32",
            "int32" => "i32",
            "uint64" => "u64",
            _ => return None,
        };
        Some((operation, scalar))
    }) else {
        return unsupported(format!(
            "{} has unsupported scalar reduction",
            decoded.op_name
        ));
    };
    let scope = match decoded.modifier("scope")? {
        "" => "sys",
        other => other,
    };
    Ok(Some(format!(
        "BulkS2cReduce<{scalar}, {}, v2::mem::variant::{}>",
        operation.marker(),
        capitalize(scope)
    )))
}

pub struct BulkS2gParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub num_bytes: ObjectRef,
    pub byte_mask: Option<ObjectRef>,
    pub scope: Option<String>,
}

pub fn decoded_bulk_s2g_parts(decoded: &DecodedPtx) -> AResult<BulkS2gParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    decoded.require_modifiers(&[
        ("api", "async"),
        ("kind", "bulk"),
        ("dst", "global"),
        ("src", "shared::cta"),
        ("completion", "bulk_group"),
    ])?;
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let cache_policy = decoded.optional_scalar_operand("cache_policy")?;
    let byte_mask = decoded.optional_scalar_operand("byte_mask")?;
    let scope = bulk_copy_scope(decoded)?;
    integer_operand(decoded, "size", &num_bytes)?;
    let has_cache_hint = decoded.modifier("cache")? == "L2::cache_hint";
    decoded.require_cache_policy(cache_policy.as_ref(), has_cache_hint)?;
    let masked = decoded.modifier("cp_mask")? == "cp_mask";
    if masked != byte_mask.is_some() {
        return unsupported(format!(
            "{op_name} cp_mask modifier and byte-mask operand disagree"
        ));
    }
    if let Some(mask) = &byte_mask {
        integer_operand(decoded, "byte_mask", mask)?;
    }
    Ok(BulkS2gParts {
        destination,
        source,
        num_bytes,
        byte_mask,
        scope,
    })
}

pub struct BulkReduceParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub num_bytes: ObjectRef,
    /// PTX non-tensor reductions default to relaxed.sys.
    pub scope: String,
    pub redop: String,
    pub ptx_type: String,
}

pub fn decoded_bulk_reduce_s2g_parts(decoded: &DecodedPtx) -> AResult<BulkReduceParts> {
    decoded.require_void()?;
    decoded.require_modifiers(&[
        ("op", "reduce"),
        ("api", "async"),
        ("kind", "bulk"),
        ("dst", "global"),
        ("src", "shared::cta"),
        ("completion", "bulk_group"),
    ])?;
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let num_bytes = decoded.scalar_operand("size")?;
    let cache_policy = decoded.optional_scalar_operand("cache_policy")?;
    integer_operand(decoded, "size", &num_bytes)?;
    let has_cache_hint = decoded.modifier("cache")? == "L2::cache_hint";
    decoded.require_cache_policy(cache_policy.as_ref(), has_cache_hint)?;
    let scope = match decoded.modifier("scope")? {
        "" => "sys",
        other => other,
    };
    Ok(BulkReduceParts {
        destination,
        source,
        num_bytes,
        scope: scope.to_owned(),
        redop: decoded.modifier("redop")?.to_owned(),
        ptx_type: decoded.modifier("type")?.to_owned(),
    })
}

pub struct MbarrierArriveParts {
    pub address: ObjectRef,
    /// The shared source `address` retains.
    pub address_source: ObjectRef,
    pub noinc: bool,
}

pub fn decoded_cp_async_mbarrier_arrive_parts(
    decoded: &DecodedPtx,
) -> AResult<MbarrierArriveParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    decoded.require_modifiers(&[
        ("api", "async"),
        ("target", "mbarrier"),
        ("action", "arrive"),
        ("type", "b64"),
    ])?;
    let space = decoded.modifier("space")?.to_owned();
    if !["", "shared", "shared::cta"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    let address = decoded.scalar_operand("addr")?;
    let (address_source, _) = shared_pointer_source(
        &address,
        &format!("{op_name}.addr"),
        SharedAddressForms::MEMORY,
    )?;
    Ok(MbarrierArriveParts {
        address,
        address_source,
        noinc: decoded.modifier("noinc")? == "noinc",
    })
}

/// The parsed parts of one decoded atom/red or bulk-memory call.
pub enum AtomicBulkParts {
    VectorAtomic(VectorAtomicParts),
    AtomCas(AtomCasParts),
    Atomic(AtomicParts),
    StBulk(StBulkParts),
    BulkG2sCluster(BulkG2sClusterParts),
    BulkS2c(BulkS2cParts),
    BulkS2g(BulkS2gParts),
    BulkReduce(BulkReduceParts),
    MbarrierArrive(MbarrierArriveParts),
    BulkG2sCta(BulkG2sCtaParts),
}

/// The parsed parts.
fn atomic_parts(decoded: &DecodedPtx) -> AResult<AtomicBulkParts> {
    let op_name = decoded.op_name.as_str();
    let parts = if PTX_VECTOR_ATOMIC_CALLS.contains(&op_name) {
        AtomicBulkParts::VectorAtomic(decoded_vector_atomic_parts(decoded)?)
    } else if PTX_CAS_CALLS.contains(&op_name) {
        AtomicBulkParts::AtomCas(decoded_atom_cas_parts(decoded)?)
    } else if PTX_SCALAR_ATOMIC_CALLS.contains(&op_name) {
        AtomicBulkParts::Atomic(decoded_atomic_parts(decoded)?)
    } else if op_name == "tirx.ptx.st_bulk" {
        AtomicBulkParts::StBulk(decoded_st_bulk_parts(decoded)?)
    } else if BULK_G2S_CLUSTER_CALLS.contains(&op_name) {
        AtomicBulkParts::BulkG2sCluster(decoded_bulk_g2s_cluster_parts(decoded)?)
    } else if BULK_S2C_CALLS.contains(&op_name) {
        AtomicBulkParts::BulkS2c(decoded_bulk_s2c_parts(decoded)?)
    } else if op_name == "tirx.ptx.cp_async_bulk_s2g" {
        AtomicBulkParts::BulkS2g(decoded_bulk_s2g_parts(decoded)?)
    } else if BULK_REDUCE_S2G_CALLS.contains(&op_name) {
        AtomicBulkParts::BulkReduce(decoded_bulk_reduce_s2g_parts(decoded)?)
    } else if op_name == "tirx.ptx.cp_async_mbarrier_arrive" {
        AtomicBulkParts::MbarrierArrive(decoded_cp_async_mbarrier_arrive_parts(decoded)?)
    } else {
        AtomicBulkParts::BulkG2sCta(decoded_bulk_g2s_cta_parts(decoded)?)
    };
    Ok(parts)
}

enum BulkPath {
    G2sCluster,
    G2sCta,
    SharedToCluster,
}

impl<'a> Emitter<'a> {
    fn cuda_atomic_pointer(&mut self, call: &CudaAtomicCall) -> AResult<RustValue> {
        let pointer = self.emit_pointer_handle(&call.address)?;
        if pointer.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{} address must resolve to a physical address",
                call.op_name
            ));
        }
        Ok(pointer)
    }

    fn emit_cuda_atomic_cas(
        &mut self,
        expr: &ObjectRef,
        call: &CudaAtomicCall,
    ) -> AResult<RustValue> {
        let compare_expr = call.compare.as_ref().expect("cas compare");
        let pointer = self.cuda_atomic_pointer(call)?;
        let compare = self.emit_expr(compare_expr)?;
        let compare = self.as_warp_value(compare);
        let value = self.emit_expr(&call.value)?;
        let value = self.as_warp_value(value);
        let rust_type = expr_rust_type(self.ctx.schema, &call.result_dtype)?;
        if compare.rust_type != rust_type || value.rust_type != rust_type {
            return unsupported(format!(
                "{} compare/value must lower to {rust_type}",
                call.op_name
            ));
        }
        let result = self.temp("atomic_cas");
        let site = self.lowered_instruction_site(expr);
        let address = abi::address("v2::Generic", &abi::cloned(&pointer.code), None);
        let invocation = abi::warp_call(
            "mem::atom",
            &abi::site(site as u64),
            &[format!(
                "({address}, {}, {})",
                abi::register(&compare.code),
                abi::register(&value.code)
            )],
            Some(&cuda_atomic_variant_rust(self.ctx, call, "AtomCas")?),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = {invocation};"));
        self.emit_line(&format!("let {result} = v2_register_out({result});"));
        Ok(RustValue::new(result, rust_type, Uniformity::Varying))
    }

    fn emit_cuda_scalar_atomic(
        &mut self,
        expr: &ObjectRef,
        call: &CudaAtomicCall,
    ) -> AResult<RustValue> {
        let pointer = self.cuda_atomic_pointer(call)?;
        let value = self.emit_expr(&call.value)?;
        let mut value = self.as_warp_value(value);
        let rust_type = expr_rust_type(self.ctx.schema, &call.result_dtype)?;
        if value.rust_type != rust_type {
            let reinterpreted = self.temp("atomic_operand_bits");
            let atom = reinterpret_atom(
                &format!("{}[lane]", value.code),
                &value.rust_type,
                &rust_type,
            );
            self.emit_line(&format!(
                "let {reinterpreted} = WarpValue::from_fn(|lane| {atom});"
            ));
            value = RustValue::new(reinterpreted, rust_type.clone(), Uniformity::Varying);
        }
        let result = self.temp("atomic_scalar");
        let site = self.lowered_instruction_site(expr);
        let address = abi::address("v2::Generic", &abi::cloned(&pointer.code), None);
        let invocation = abi::warp_call(
            "mem::atom",
            &abi::site(site as u64),
            &[format!("({address}, {})", abi::register(&value.code))],
            Some(&cuda_atomic_variant_rust(self.ctx, call, "Atom")?),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = {invocation};"));
        self.emit_line(&format!("let {result} = v2_register_out({result});"));
        Ok(RustValue::new(result, rust_type, Uniformity::Varying))
    }

    fn decoded_atomic_pointer(
        &mut self,
        address_expression: &ObjectRef,
        op_name: &str,
        space: &str,
        active_mask: &str,
        result_dtype: Option<&str>,
    ) -> AResult<RustValue> {
        let (pointer_source, _) = shared_pointer_source(
            address_expression,
            &format!("{op_name}.addr"),
            SharedAddressForms::MEMORY,
        )?;
        let mut pointer = self.emit_expr(&pointer_source)?;
        if pointer.rust_type != "PhysicalPtr" {
            pointer =
                self.emit_address_pointer(address_expression, space, Some(pointer), active_mask)?;
        }
        if pointer.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{op_name} address must resolve to a physical address"
            ));
        }
        if result_dtype == Some("uint128") {
            pointer = RustValue::new(
                format!("({}).with_pointee_itemsize(16_usize)", pointer.code),
                pointer.rust_type,
                pointer.uniformity,
            );
        }
        Ok(pointer)
    }

    fn decoded_shared_pointer(
        &mut self,
        expression: &ObjectRef,
        field: &str,
    ) -> AResult<RustValue> {
        let (pointer_source, _) =
            shared_pointer_source(expression, field, SharedAddressForms::MEMORY)?;
        self.shared_source_pointer(expression, &pointer_source, field)
    }

    /// `decoded_shared_pointer` of an address whose shared source is known.
    fn shared_source_pointer(
        &mut self,
        expression: &ObjectRef,
        pointer_source: &ObjectRef,
        field: &str,
    ) -> AResult<RustValue> {
        let mut pointer = self.emit_expr(pointer_source)?;
        if pointer.rust_type != "PhysicalPtr" {
            pointer = self.emit_address_pointer(
                expression,
                "shared",
                Some(pointer),
                "ctx.active_mask()",
            )?;
        }
        if pointer.rust_type != "PhysicalPtr" {
            return unsupported(format!("{field} must resolve to a physical address"));
        }
        Ok(pointer)
    }

    /// A statement-level warp call.
    fn emit_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: String,
        variant: &str,
        context: Option<&str>,
        await_result: bool,
    ) {
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            function,
            &site,
            &[arguments],
            Some(variant),
            context,
            await_result,
            true,
        );
        if await_result {
            self.emit_suspend_line(&format!("{invocation};"));
        } else {
            self.emit_line(&format!("{invocation};"));
        }
    }

    fn atomic_register_bits(&mut self, value: RustValue, rust_type: &str) -> RustValue {
        let value = self.as_warp_value(value);
        if value.rust_type == rust_type {
            return value;
        }
        let name = self.control_name("atomic_register_bits");
        let atom = reinterpret_atom(
            &format!("{}[lane]", value.code),
            &value.rust_type,
            rust_type,
        );
        self.emit_line(&format!("let {name} = WarpValue::from_fn(|lane| {atom});"));
        RustValue::new(name, rust_type, Uniformity::Varying)
    }

    /// b16 CAS can bind a half register. Store its bits
    /// directly: the numerical half carrier is f32, whose bit pattern is not a
    /// 16-bit register encoding.
    fn atomic_result_store(
        &mut self,
        destination: &ObjectRef,
        value: RustValue,
        source_op_id: i64,
        instruction_mask: &str,
    ) -> AResult<()> {
        let destination_dtype = dtype_of(destination)?;
        let raw_half = destination_dtype == "float16" || destination_dtype == "bfloat16";
        let dtype = if raw_half {
            "uint16"
        } else {
            destination_dtype.as_str()
        };
        let rust_type = expr_rust_type(self.ctx.schema, dtype)?;
        let value = self.atomic_register_bits(value, &rust_type);
        self.emit_explicit_buffer_store(
            destination,
            value,
            source_op_id,
            None,
            Some(instruction_mask),
            if raw_half { Some("uint16") } else { None },
        )
    }

    fn emit_decoded_atom_cas(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AtomCasParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "decoded_atomic",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        // A sink still executes atom (not red), preserving its ordering semantics.
        let pointer = self.decoded_atomic_pointer(
            &parts.address,
            &op_name,
            &parts.space,
            "ctx.active_mask()",
            Some(&parts.result_dtype),
        )?;
        let rust_type = expr_rust_type(self.ctx.schema, &parts.result_dtype)?;
        let compare_expression =
            if matches!(dtype_of(&parts.compare)?.as_str(), "float16" | "bfloat16") {
                reinterpret("uint16", &parts.compare)?
            } else {
                parts.compare.clone()
            };
        let value_expression = if matches!(dtype_of(&parts.value)?.as_str(), "float16" | "bfloat16")
        {
            reinterpret("uint16", &parts.value)?
        } else {
            parts.value.clone()
        };
        let compare = self.emit_expr(&compare_expression)?;
        let compare = self.atomic_register_bits(compare, &rust_type);
        let value = self.emit_expr(&value_expression)?;
        let value = self.atomic_register_bits(value, &rust_type);
        let space_rust = v2_atomic_space_rust(Some(&parts.space))?;
        let address = abi::address(space_rust, &abi::cloned(&pointer.code), None);
        let variant = format!(
            "v2::mem::variant::AtomCas<{}, {space_rust}, {}>",
            v2_atomic_type_rust(self.ctx, &parts.result_dtype, true)?,
            decoded_atomic_semantics_rust(&parts.semantics, &parts.scope)
        );
        let arguments = format!(
            "({address}, {}, {})",
            abi::register(&compare.code),
            abi::register(&value.code)
        );
        let loaded = self.control_name("decoded_atom_cas_result");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            "mem::atom",
            &site,
            &[arguments],
            Some(&variant),
            region.context.as_deref(),
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {loaded} = {invocation};"));
        if let Some(destination) = parts.destination.as_ref() {
            self.emit_line(&format!("let {loaded} = v2_register_out({loaded});"));
            self.atomic_result_store(
                destination,
                RustValue::new(loaded, rust_type, Uniformity::Varying),
                source_op_id,
                &region.mask,
            )?;
        }
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_decoded_atomic(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AtomicParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "decoded_atomic",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let pointer = self.decoded_atomic_pointer(
            &parts.address,
            &op_name,
            &parts.space,
            &region.mask,
            Some(&parts.result_dtype),
        )?;
        let half = parts.result_dtype == "float16" || parts.result_dtype == "bfloat16";
        let value_expression = if half {
            reinterpret(&parts.result_dtype, &parts.value)?
        } else {
            parts.value.clone()
        };
        let mut rust_type = expr_rust_type(self.ctx.schema, &parts.result_dtype)?;
        let value = self.emit_expr(&value_expression)?;
        let value = self.atomic_register_bits(value, &rust_type);
        let address = abi::address(
            v2_atomic_space_rust(Some(&parts.space))?,
            &abi::cloned(&pointer.code),
            None,
        );
        let reduction = PTX_REDUCTIONS.contains(&op_name.as_str());
        let variant = decoded_atomic_variant(
            self.ctx,
            if reduction { "Red" } else { "Atom" },
            parts.operation,
            &parts.result_dtype,
            &parts.space,
            &parts.semantics,
            &parts.scope,
            f32_noftz(&decoded)?,
            false,
        )?;
        let arguments = format!("({address}, {})", abi::register(&value.code));
        if reduction {
            self.emit_stateful(
                "mem::red",
                source_op_id,
                arguments,
                &variant,
                region.context.as_deref(),
                true,
            );
            self.close_predicated_region(region);
            return Ok(());
        }
        let loaded = self.control_name("decoded_atomic_result");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            "mem::atom",
            &site,
            &[arguments],
            Some(&variant),
            region.context.as_deref(),
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {loaded} = {invocation};"));
        if let Some(destination) = parts.destination.as_ref() {
            self.emit_line(&format!("let {loaded} = v2_register_out({loaded});"));
            if half {
                let codec = if parts.result_dtype == "float16" {
                    "decoded_fp16_to_bits"
                } else {
                    "decoded_bf16_to_bits"
                };
                self.emit_line(&format!(
                    "let {loaded} = WarpValue::from_fn(|lane| {codec}({loaded}[lane]));"
                ));
                rust_type = "u16".to_owned();
            }
            self.atomic_result_store(
                destination,
                RustValue::new(loaded, rust_type, Uniformity::Varying),
                source_op_id,
                &region.mask,
            )?;
        }
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_decoded_vector_atomic(
        &mut self,
        decoded: &DecodedPtx,
        parts: &VectorAtomicParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "decoded_vector_atomic",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let pointer = self.decoded_atomic_pointer(
            &parts.address,
            &op_name,
            &parts.space,
            "ctx.active_mask()",
            None,
        )?;
        let width = parts.values.len();
        let half = parts.result_dtype.starts_with("float16x")
            || parts.result_dtype.starts_with("bfloat16x");
        let mut values = Vec::new();
        for value in &parts.values {
            let value = self.emit_expr(value)?;
            values.push(self.as_warp_value(value));
        }
        let register_bits: usize = if !half || decoded.modifier("type")?.ends_with("x2") {
            32
        } else {
            16
        };
        let carrier = if half {
            format!("u{register_bits}")
        } else {
            "f32".to_owned()
        };
        if values.iter().any(|value| value.rust_type != carrier) {
            return unsupported(format!("{op_name}.value lanes must lower to {carrier}"));
        }
        let operand = self.control_name("decoded_vector_atomic_operand");
        let rust_type;
        let per_word = 64 / register_bits;
        if half {
            let total_bits = width * register_bits;
            let word_type = format!("u{}", total_bits.min(64));
            let words: Vec<String> = values
                .chunks(per_word)
                .map(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            format!(
                                "({word_type}::from({}[lane]) << {})",
                                value.code,
                                index * register_bits
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect();
            let packed = if words.len() == 1 {
                words[0].clone()
            } else {
                format!("[{}]", words.join(", "))
            };
            self.emit_line(&format!(
                "let {operand} = WarpValue::from_fn(|lane| {packed});"
            ));
            rust_type = if words.len() == 1 {
                word_type
            } else {
                "U64x2".to_owned()
            };
        } else if width == 2 {
            self.emit_line(&format!(
                "let {operand} = WarpValue::from_fn(|lane| u64::from({}[lane].to_bits()) | (u64::from({}[lane].to_bits()) << 32));",
                values[0].code, values[1].code
            ));
            rust_type = "u64".to_owned();
        } else {
            let components: Vec<String> = values
                .iter()
                .map(|value| format!("{}[lane]", value.code))
                .collect();
            self.emit_line(&format!(
                "let {operand} = WarpValue::from_fn(|lane| [{}]);",
                components.join(", ")
            ));
            rust_type = "F32x4".to_owned();
        }
        let vector_pointer = self.control_name("decoded_vector_atomic_pointer");
        self.emit_line(&format!(
            "let {vector_pointer} = {}.with_pointee_itemsize({}_usize);",
            pointer.code,
            width * register_bits / 8
        ));
        let address = abi::address(
            v2_atomic_space_rust(Some(&parts.space))?,
            &abi::cloned(&vector_pointer),
            None,
        );
        let reduction = PTX_VECTOR_REDUCTIONS.contains(&op_name.as_str());
        let variant = decoded_atomic_variant(
            self.ctx,
            if reduction { "Red" } else { "Atom" },
            parts.operation,
            &parts.result_dtype,
            &parts.space,
            &parts.semantics,
            &parts.scope,
            f32_noftz(&decoded)?,
            half,
        )?;
        let context = region.context.clone();
        if reduction {
            self.emit_stateful(
                "mem::red",
                source_op_id,
                format!("({address}, {})", abi::register(&operand)),
                &variant,
                context.as_deref(),
                true,
            );
            self.close_predicated_region(region);
            return Ok(());
        }
        let loaded = self.control_name("decoded_vector_atomic_result");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            "mem::atom",
            &site,
            &[format!("({address}, {})", abi::register(&operand))],
            Some(&variant),
            context.as_deref(),
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {loaded} = {invocation};"));
        self.emit_line(&format!("let {loaded} = v2_register_out({loaded});"));
        for (component, destination) in parts.destinations.iter().enumerate() {
            let component_value = self.control_name("decoded_vector_atomic_component");
            if half {
                let mut word = format!("{loaded}[lane]");
                if rust_type == "U64x2" {
                    word.push_str(&format!("[{}]", component / per_word));
                }
                let shift = (component % per_word) * register_bits;
                self.emit_line(&format!(
                    "let {component_value} = WarpValue::from_fn(|lane| ({word} >> {shift}) as {carrier});"
                ));
            } else if rust_type == "u64" {
                self.emit_line(&format!(
                    "let {component_value} = WarpValue::from_fn(|lane| f32::from_bits(({loaded}[lane] >> {}) as u32));",
                    component * 32
                ));
            } else {
                self.emit_line(&format!(
                    "let {component_value} = WarpValue::from_fn(|lane| {loaded}[lane][{component}]);"
                ));
            }
            self.emit_explicit_buffer_store(
                destination,
                RustValue::new(component_value, carrier.clone(), Uniformity::Varying),
                source_op_id,
                None,
                Some(&region.mask),
                None,
            )?;
        }
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_st_bulk(
        &mut self,
        op_name: &str,
        address_expression: &ObjectRef,
        num_bytes_expression: &ObjectRef,
        space: &str,
        source_op_id: i64,
        context: Option<&str>,
    ) -> AResult<()> {
        let (pointer_source, _) = shared_pointer_source(
            address_expression,
            &format!("{op_name}.addr"),
            SharedAddressForms::MEMORY,
        )?;
        let mut pointer = self.emit_expr(&pointer_source)?;
        if pointer.rust_type != "PhysicalPtr" {
            pointer = self.emit_address_pointer(
                address_expression,
                space,
                Some(pointer),
                "ctx.active_mask()",
            )?;
        }
        if pointer.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{op_name} address must resolve to a physical address"
            ));
        }
        let num_bytes = self.emit_expr(num_bytes_expression)?;
        let num_bytes = self.as_i64(num_bytes)?;
        let num_bytes = self.as_warp_value(num_bytes);
        let variant = match space {
            "" => "StBulkZeroGeneric",
            "shared" | "shared::cta" => "StBulkZeroShared",
            other => {
                return unsupported(format!(
                    "{op_name} state space {:?} has no st.bulk variant",
                    other
                ))
            }
        };
        let rust_space = if space.is_empty() {
            "v2::Generic"
        } else {
            "v2::Shared"
        };
        self.emit_stateful(
            "mem::st_bulk",
            source_op_id,
            format!(
                "({}, {})",
                abi::address(rust_space, &abi::cloned(&pointer.code), None),
                abi::register(&num_bytes.code)
            ),
            &format!("v2::mem::variant::{variant}"),
            context,
            false,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_bulk_to_shared(
        &mut self,
        op_name: &str,
        destination_expression: &ObjectRef,
        source_expression: &ObjectRef,
        barrier_expression: &ObjectRef,
        num_bytes_expression: &ObjectRef,
        source_op_id: i64,
        context: Option<&str>,
        path: BulkPath,
        cta_mask_expression: Option<&ObjectRef>,
        ignore_left_expression: Option<&ObjectRef>,
        ignore_right_expression: Option<&ObjectRef>,
        reduction: Option<&str>,
        report_pattern: i64,
        scope: Option<&str>,
    ) -> AResult<()> {
        let destination =
            self.decoded_shared_pointer(destination_expression, &format!("{op_name}.dst_mem"))?;
        let barrier_pointer =
            self.decoded_shared_pointer(barrier_expression, &format!("{op_name}.mbar"))?;
        let source = if matches!(path, BulkPath::SharedToCluster) {
            self.decoded_shared_pointer(source_expression, &format!("{op_name}.src_mem"))?
        } else {
            let source = self.emit_expr(source_expression)?;
            let source_space = if matches!(path, BulkPath::G2sCluster) {
                ""
            } else {
                "global"
            };
            self.emit_address_pointer(
                source_expression,
                source_space,
                Some(source),
                "ctx.active_mask()",
            )?
        };
        if [&destination, &source, &barrier_pointer]
            .iter()
            .any(|value| value.rust_type != "PhysicalPtr")
        {
            return unsupported(format!(
                "{op_name} pointers must resolve to physical addresses"
            ));
        }
        let num_bytes = self.emit_expr(num_bytes_expression)?;
        let num_bytes = self.as_i64(num_bytes)?;
        let num_bytes = self.as_warp_value(num_bytes);
        let source_space = if matches!(path, BulkPath::SharedToCluster) {
            "v2::Shared"
        } else {
            "v2::Global"
        };
        let mut operands = vec![
            abi::address("v2::Shared", &abi::cloned(&destination.code), None),
            abi::address(source_space, &abi::cloned(&source.code), None),
            abi::register(&num_bytes.code),
            abi::address("v2::Shared", &abi::cloned(&barrier_pointer.code), None),
        ];
        let mut marker = match path {
            BulkPath::G2sCluster => {
                let mut marker = "BulkG2sCluster".to_owned();
                if let Some(cta_mask_expression) = cta_mask_expression {
                    let cta_mask = self.emit_expr(cta_mask_expression)?;
                    let cta_mask = self.as_i64(cta_mask)?;
                    let cta_mask = self.as_warp_value(cta_mask);
                    operands.push(abi::register(&cta_mask.code));
                    marker.push_str("Multicast");
                }
                marker
            }
            BulkPath::G2sCta => {
                let ignore_oob = ignore_left_expression.is_some();
                if let (Some(left), Some(right)) = (ignore_left_expression, ignore_right_expression)
                {
                    let ignore_left = self.emit_expr(left)?;
                    let ignore_left = self.as_i64(ignore_left)?;
                    let ignore_left = self.as_warp_value(ignore_left);
                    let ignore_right = self.emit_expr(right)?;
                    let ignore_right = self.as_i64(ignore_right)?;
                    let ignore_right = self.as_warp_value(ignore_right);
                    operands.push(abi::register(&ignore_left.code));
                    operands.push(abi::register(&ignore_right.code));
                }
                if ignore_oob {
                    "BulkG2sCtaIgnoreOob".to_owned()
                } else {
                    "BulkG2sCta".to_owned()
                }
            }
            BulkPath::SharedToCluster => reduction.unwrap_or("BulkSharedToCluster").to_owned(),
        };
        if let Some(scope) = scope {
            let semantics = format!(
                "v2::mem::variant::Relaxed<v2::mem::variant::{}>",
                capitalize(scope)
            );
            let parameters =
                if matches!(path, BulkPath::SharedToCluster) || marker == "BulkG2sCtaIgnoreOob" {
                    semantics
                } else {
                    format!("{report_pattern}, {semantics}")
                };
            marker.push_str(&format!("<{parameters}>"));
        } else if report_pattern != 0 {
            marker.push_str(&format!("<{report_pattern}>"));
        }
        self.emit_stateful(
            if reduction.is_some() {
                "async_copy::cp_reduce_async_bulk"
            } else {
                "async_copy::cp_async_bulk"
            },
            source_op_id,
            format!("({})", operands.join(", ")),
            &format!("v2::async_copy::variant::{marker}"),
            context,
            false,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_bulk_s2g(
        &mut self,
        op_name: &str,
        scope: Option<&str>,
        destination_expression: &ObjectRef,
        source_expression: &ObjectRef,
        num_bytes_expression: &ObjectRef,
        byte_mask_expression: Option<&ObjectRef>,
        source_op_id: i64,
        context: Option<&str>,
    ) -> AResult<()> {
        let mut destination = self.emit_expr(destination_expression)?;
        if destination.rust_type != "PhysicalPtr" {
            destination = self.emit_address_pointer(
                destination_expression,
                "global",
                Some(destination),
                "ctx.active_mask()",
            )?;
        }
        let source =
            self.decoded_shared_pointer(source_expression, &format!("{op_name}.src_mem"))?;
        if destination.rust_type != "PhysicalPtr" || source.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{op_name} pointers must resolve to physical addresses"
            ));
        }
        let num_bytes = self.emit_expr(num_bytes_expression)?;
        let num_bytes = self.as_i64(num_bytes)?;
        let num_bytes = self.as_warp_value(num_bytes);
        let mut operands = vec![
            abi::address("v2::Global", &abi::cloned(&destination.code), None),
            abi::address("v2::Shared", &abi::cloned(&source.code), None),
            abi::register(&num_bytes.code),
        ];
        let mut marker = "BulkS2g".to_owned();
        if let Some(byte_mask_expression) = byte_mask_expression {
            let byte_mask = self.emit_expr(byte_mask_expression)?;
            let byte_mask = self.as_i64(byte_mask)?;
            let byte_mask = self.as_warp_value(byte_mask);
            operands.push(abi::register(&byte_mask.code));
            marker.push_str("Masked");
        }
        if let Some(scope) = scope {
            marker.push_str(&format!(
                "<v2::mem::variant::Relaxed<v2::mem::variant::{}>>",
                capitalize(scope)
            ));
        }
        self.emit_stateful(
            "async_copy::cp_async_bulk",
            source_op_id,
            format!("({})", operands.join(", ")),
            &format!("v2::async_copy::variant::{marker}"),
            context,
            false,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_bulk_reduce_s2g(
        &mut self,
        op_name: &str,
        destination_expression: &ObjectRef,
        source_expression: &ObjectRef,
        num_bytes_expression: &ObjectRef,
        source_op_id: i64,
        context: Option<&str>,
        scope: &str,
        redop: &str,
        ptx_type: &str,
    ) -> AResult<()> {
        let mut destination = self.emit_expr(destination_expression)?;
        if destination.rust_type != "PhysicalPtr" {
            destination = self.emit_address_pointer(
                destination_expression,
                "global",
                Some(destination),
                "ctx.active_mask()",
            )?;
        }
        let source =
            self.decoded_shared_pointer(source_expression, &format!("{op_name}.src_mem"))?;
        if destination.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{op_name} destination must resolve to a physical address"
            ));
        }
        let num_bytes = self.emit_expr(num_bytes_expression)?;
        let num_bytes = self.as_i64(num_bytes)?;
        let num_bytes = self.as_warp_value(num_bytes);
        let arguments = format!(
            "({}, {}, {})",
            abi::address("v2::Global", &abi::cloned(&destination.code), None),
            abi::address("v2::Shared", &abi::cloned(&source.code), None),
            abi::register(&num_bytes.code)
        );
        let Some(type_marker) = table_marker(PTX_TYPE_MARKERS, ptx_type) else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "PTX type {ptx_type} has no register marker"
            ))));
        };
        let variant = format!(
            "v2::async_copy::variant::BulkS2gReduce<{}, v2::async_copy::variant::Reduce{}, v2::mem::variant::{}>",
            marker(type_marker),
            capitalize(redop),
            capitalize(scope)
        );
        self.emit_stateful(
            "async_copy::cp_reduce_async_bulk",
            source_op_id,
            arguments,
            &variant,
            context,
            false,
        );
        Ok(())
    }

    fn emit_atomic_statement(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AtomicBulkParts,
        source_op_id: i64,
    ) -> AResult<()> {
        match parts {
            AtomicBulkParts::VectorAtomic(parts) => {
                return self.emit_decoded_vector_atomic(decoded, parts, source_op_id)
            }
            AtomicBulkParts::AtomCas(parts) => {
                return self.emit_decoded_atom_cas(decoded, parts, source_op_id)
            }
            AtomicBulkParts::Atomic(parts) => {
                return self.emit_decoded_atomic(decoded, parts, source_op_id)
            }
            _ => {}
        }
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "decoded_bulk",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let context = region.context.clone();
        self.emit_decoded_bulk(decoded, parts, source_op_id, context.as_deref())?;
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_decoded_bulk(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AtomicBulkParts,
        source_op_id: i64,
        context: Option<&str>,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        match parts {
            AtomicBulkParts::VectorAtomic(_)
            | AtomicBulkParts::AtomCas(_)
            | AtomicBulkParts::Atomic(_) => unreachable!("atomics are lowered above"),
            AtomicBulkParts::StBulk(parts) => self.emit_st_bulk(
                &op_name,
                &parts.address,
                &parts.num_bytes,
                &parts.space,
                source_op_id,
                context,
            ),
            AtomicBulkParts::BulkG2sCluster(parts) => {
                let (destination, _) = shared_pointer_source(
                    &parts.destination,
                    &format!("{op_name}.dst_mem"),
                    SharedAddressForms::MEMORY,
                )?;
                let (barrier, _) = shared_pointer_source(
                    &parts.barrier,
                    &format!("{op_name}.mbar"),
                    SharedAddressForms::MEMORY,
                )?;
                self.emit_bulk_to_shared(
                    &op_name,
                    &destination,
                    &parts.source,
                    &barrier,
                    &parts.num_bytes,
                    source_op_id,
                    context,
                    BulkPath::G2sCluster,
                    parts.cta_mask.as_ref(),
                    None,
                    None,
                    None,
                    parts.report_pattern,
                    parts.scope.as_deref(),
                )
            }
            AtomicBulkParts::BulkS2c(parts) => self.emit_bulk_to_shared(
                &op_name,
                &parts.destination,
                &parts.source,
                &parts.barrier,
                &parts.num_bytes,
                source_op_id,
                context,
                BulkPath::SharedToCluster,
                None,
                None,
                None,
                parts.reduction.as_deref(),
                0,
                parts.scope.as_deref(),
            ),
            AtomicBulkParts::BulkS2g(parts) => self.emit_bulk_s2g(
                &op_name,
                parts.scope.as_deref(),
                &parts.destination,
                &parts.source,
                &parts.num_bytes,
                parts.byte_mask.as_ref(),
                source_op_id,
                context,
            ),
            AtomicBulkParts::BulkReduce(parts) => self.emit_bulk_reduce_s2g(
                &op_name,
                &parts.destination,
                &parts.source,
                &parts.num_bytes,
                source_op_id,
                context,
                &parts.scope,
                &parts.redop,
                &parts.ptx_type,
            ),
            AtomicBulkParts::MbarrierArrive(parts) => {
                let barrier = self.shared_source_pointer(
                    &parts.address,
                    &parts.address_source,
                    &format!("{op_name}.addr"),
                )?;
                let marker = if parts.noinc {
                    "CpAsyncMbarrierArriveNoInc"
                } else {
                    "CpAsyncMbarrierArrive"
                };
                self.emit_stateful(
                    "async_copy::cp_async_mbarrier_arrive",
                    source_op_id,
                    abi::address("v2::Shared", &abi::cloned(&barrier.code), None),
                    &format!("v2::async_copy::variant::{marker}"),
                    context,
                    false,
                );
                Ok(())
            }
            AtomicBulkParts::BulkG2sCta(parts) => {
                let (destination, _) = shared_pointer_source(
                    &parts.destination,
                    &format!("{op_name}.dst_mem"),
                    SharedAddressForms::MEMORY,
                )?;
                let (barrier, _) = shared_pointer_source(
                    &parts.barrier,
                    &format!("{op_name}.mbar"),
                    SharedAddressForms::MEMORY,
                )?;
                self.emit_bulk_to_shared(
                    &op_name,
                    &destination,
                    &parts.source,
                    &barrier,
                    &parts.num_bytes,
                    source_op_id,
                    context,
                    BulkPath::G2sCta,
                    None,
                    parts.ignore_left.as_ref(),
                    parts.ignore_right.as_ref(),
                    None,
                    parts.report_pattern,
                    parts.scope.as_deref(),
                )
            }
        }
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = atomic_parts(decoded)?;
    match &parts {
        AtomicBulkParts::Atomic(p) => emitter.record_pointer_write(&p.address, &p.space)?,
        AtomicBulkParts::VectorAtomic(p) => emitter.record_pointer_write(&p.address, &p.space)?,
        AtomicBulkParts::AtomCas(p) => emitter.record_pointer_write(&p.address, &p.space)?,
        AtomicBulkParts::StBulk(p) => emitter.record_pointer_write(&p.address, &p.space)?,
        AtomicBulkParts::BulkS2g(p) => emitter.record_pointer_write(&p.destination, "global")?,
        AtomicBulkParts::BulkReduce(p) => emitter.record_pointer_write(&p.destination, "global")?,
        _ => {} // The remaining variants write only shared memory.
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_atomic_statement(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_legacy(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = resolve_cuda_atomic_call(emitter.ctx, call.node, &call.op_name)?;
    emitter.record_pointer_write(&parts.address, "")?;
    emitter
        .with_call_expr(call.node, |emitter| {
            if call.op_name == "tirx.cuda.atomic_cas" {
                emitter.emit_cuda_atomic_cas(call.node, &parts)
            } else {
                emitter.emit_cuda_scalar_atomic(call.node, &parts)
            }
        })
        .map(Some)
}
