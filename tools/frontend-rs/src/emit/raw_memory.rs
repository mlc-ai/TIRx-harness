//! Validation and emission of the raw_memory instruction family.

use crate::analyze::buffers::call_op_name;
use crate::analyze::memory::MemorySpace;
use crate::analyze::util::{
    as_buffer, dtype_of, oref, prim, static_int, static_string, unsupported, AResult,
};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::{v2_memory_space_rust, v2_memory_type_rust_ptx};
use crate::emit::{abi, Emitter, NestedLoadSite, RustValue, Uniformity};
use crate::tables::is_integer_dtype;
use crate::tables::{dtype_byte_len, expr_rust_type, scope_marker};
use tvm::ir::{CallObj, PrimExpr, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

pub const LDMATRIX_CALLS: [&str; 4] = [
    "tirx.ptx.ldmatrix",
    "tirx.ptx.ldmatrix_m16n16_b8",
    "tirx.ptx.ldmatrix_b8fmt",
    "tirx.ptx.ldmatrix_s8_s4",
];
pub const VECTOR_LOAD_CALLS: [&str; 3] =
    ["tirx.ptx.ld_vec", "tirx.ptx.ld_vec256", "tirx.ptx.ldu_vec"];
pub const STORE_CALLS: [&str; 3] = ["tirx.ptx.st", "tirx.ptx.st_vec", "tirx.ptx.st_vec256"];
// ----------------------------------------------------------------------
// Legacy non-table calls.
// ----------------------------------------------------------------------

pub struct LegacyMemoryCall {
    pub op_name: String,
    pub address: Option<ObjectRef>,
    pub destination: Option<ObjectRef>,
    pub value: Option<ObjectRef>,
    pub byte_offset: Option<ObjectRef>,
    pub destination_offset: Option<ObjectRef>,
    pub source_offset: Option<ObjectRef>,
    pub predicate: Option<ObjectRef>,
    pub target_dtype: Option<String>,
    pub result_dtype: Option<String>,
    pub ptx_type: Option<String>,
    pub space: Option<String>,
    pub vector_width: i64,
    pub matrix_count: i64,
    pub transpose: bool,
}

impl LegacyMemoryCall {
    fn new(op_name: &str) -> Self {
        Self {
            op_name: op_name.to_owned(),
            address: None,
            destination: None,
            value: None,
            byte_offset: None,
            destination_offset: None,
            source_offset: None,
            predicate: None,
            target_dtype: None,
            result_dtype: None,
            ptx_type: None,
            space: None,
            vector_width: 1,
            matrix_count: 0,
            transpose: false,
        }
    }
}

fn require_arity(call: &CallObj, expected: usize) -> AResult<Vec<ObjectRef>> {
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    if args.len() != expected {
        let name = call_op_name(call)?.unwrap_or_default();
        return unsupported(format!(
            "{name} expects {expected} arguments, got {}",
            args.len()
        ));
    }
    Ok(args)
}

fn static_flag(ctx: &Ctx, value: &ObjectRef, field: &str) -> AResult<bool> {
    let integer = static_int(
        &ctx.analyzer,
        &prim(value)?,
        field,
        "must be a static integer",
    )?;
    if integer != 0 && integer != 1 {
        return unsupported(format!("{field} must be 0 or 1, got {integer}"));
    }
    Ok(integer != 0)
}

pub fn is_type_annotation(value: &ObjectRef) -> AResult<bool> {
    let Some(call) = value.as_node::<CallObj>() else {
        return Ok(false);
    };
    Ok(call_op_name(call)?.as_deref() == Some("tirx.type_annotation") && call.args.is_empty())
}

pub fn resolve_legacy_memory_call(
    ctx: &Ctx,
    node: &ObjectRef,
    op_name: &str,
) -> AResult<LegacyMemoryCall> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("raw memory lowering expects a TIRx Call"),
        ));
    };
    match op_name {
        "tirx.type_annotation" => {
            require_arity(call, 0)?;
            let target_dtype = dtype_of(node)?;
            dtype_byte_len(ctx.schema, &target_dtype)?;
            let mut parsed = LegacyMemoryCall::new(op_name);
            parsed.target_dtype = Some(target_dtype.clone());
            parsed.result_dtype = Some(target_dtype);
            Ok(parsed)
        }
        "tirx.ptr_byte_offset" => {
            let args = require_arity(call, 3)?;
            let (source, byte_offset, annotation) = (&args[0], &args[1], &args[2]);
            if dtype_of(node)? != "handle" || dtype_of(source)? != "handle" {
                return unsupported(format!("{op_name} requires handle source and result"));
            }
            if !is_integer_dtype(&dtype_of(byte_offset)?) {
                return unsupported(format!("{op_name} byte offset must be an integer"));
            }
            if !is_type_annotation(annotation)? {
                return unsupported(format!(
                    "{op_name} target must be a zero-argument tirx.type_annotation"
                ));
            }
            let target_dtype = dtype_of(annotation)?;
            dtype_byte_len(ctx.schema, &target_dtype)?;
            let mut parsed = LegacyMemoryCall::new(op_name);
            parsed.address = Some(source.clone());
            parsed.byte_offset = Some(byte_offset.clone());
            parsed.target_dtype = Some(target_dtype);
            Ok(parsed)
        }
        "tirx.cuda.cvta_generic_to_shared" => {
            let args = require_arity(call, 1)?;
            if dtype_of(node)? != "uint32" || dtype_of(&args[0])? != "handle" {
                return unsupported(format!("{op_name} requires handle input and uint32 result"));
            }
            let mut parsed = LegacyMemoryCall::new(op_name);
            parsed.address = Some(args[0].clone());
            Ok(parsed)
        }
        "tirx.ptx_legacy.ldmatrix" => {
            let args = require_arity(call, 7)?;
            let (trans, num, dtype, destination, destination_offset, source, source_offset) = (
                &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &args[6],
            );
            let target_dtype = dtype_of(node)?;
            let target_itemsize = dtype_byte_len(ctx.schema, &target_dtype)?;
            let transpose = static_flag(ctx, trans, &format!("{op_name}.trans"))?;
            let matrix_count = static_int(
                &ctx.analyzer,
                &prim(num)?,
                &format!("{op_name}.num"),
                "must be a static integer",
            )?;
            let matrix_dtype = static_string(dtype, &format!("{op_name}.dtype"))?;
            let matrix_dtype = matrix_dtype.trim_start_matches('.');
            if ![1, 2, 4].contains(&matrix_count) || matrix_dtype != "b16" {
                return unsupported(format!(
                    "{op_name} supports b16 with matrix count 1, 2, or 4"
                ));
            }
            if target_itemsize == 1 && transpose && matrix_count != 4 {
                return unsupported(format!(
                    "{op_name} 8-bit transpose uses the TIRx manual gather fallback and requires num=4"
                ));
            }
            if dtype_of(destination)? != "handle" || dtype_of(source)? != "handle" {
                return unsupported(format!("{op_name} source and destination must be handles"));
            }
            if !is_integer_dtype(&dtype_of(destination_offset)?)
                || !is_integer_dtype(&dtype_of(source_offset)?)
            {
                return unsupported(format!(
                    "{op_name} source/destination offsets must be integers"
                ));
            }
            let mut parsed = LegacyMemoryCall::new(op_name);
            parsed.address = Some(source.clone());
            parsed.destination = Some(destination.clone());
            parsed.destination_offset = Some(destination_offset.clone());
            parsed.source_offset = Some(source_offset.clone());
            parsed.target_dtype = Some(target_dtype);
            parsed.result_dtype = Some("uint32".to_owned());
            parsed.ptx_type = Some("b16".to_owned());
            parsed.space = Some("shared".to_owned());
            parsed.vector_width = matrix_count;
            parsed.matrix_count = matrix_count;
            parsed.transpose = transpose;
            Ok(parsed)
        }
        "tirx.s_tir.ldg32" => {
            let args = require_arity(call, 4)?;
            let (destination, predicate, source, destination_offset) =
                (&args[0], &args[1], &args[2], &args[3]);
            if dtype_of(destination)? != "handle" || source.as_node::<TensorLoadObj>().is_none() {
                return unsupported(format!(
                    "{op_name} requires local data handle and global TensorLoad"
                ));
            }
            let predicate_dtype = dtype_of(predicate)?;
            if dtype_of(source)? != "float32"
                || !(is_integer_dtype(&predicate_dtype) || predicate_dtype == "bool")
            {
                return unsupported(format!(
                    "{op_name} requires float32 source and bool/integer guard"
                ));
            }
            if !is_integer_dtype(&dtype_of(destination_offset)?) {
                return unsupported(format!("{op_name} local address must be integer"));
            }
            let mut parsed = LegacyMemoryCall::new(op_name);
            parsed.destination = Some(destination.clone());
            parsed.value = Some(source.clone());
            parsed.destination_offset = Some(destination_offset.clone());
            parsed.predicate = Some(predicate.clone());
            parsed.result_dtype = Some("float32".to_owned());
            parsed.ptx_type = Some("f32".to_owned());
            parsed.space = Some("global".to_owned());
            Ok(parsed)
        }
        other => Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error(&format!("unsupported legacy memory op {other}")),
        )),
    }
}

/// The engine function (below `v2::`) the `discard` lowering calls.
pub const DISCARD: &str = "mem::discard";

// ----------------------------------------------------------------------
// Decoded table instructions.
// ----------------------------------------------------------------------

/// TVM owns legal register classes, including its
/// vector toolchain limits.
fn memory_carrier_dtype(decoded: &DecodedPtx, operand: &str) -> AResult<String> {
    let Some(slot) = decoded.operands.iter().find(|slot| slot.name == operand) else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error(&format!(
                "{} has no operand slot named {operand:?}",
                decoded.op_name
            )),
        ));
    };
    let mut dtypes = Vec::new();
    for value in slot.values.iter().flatten() {
        dtypes.push(dtype_of(value)?);
    }
    let mut unique = dtypes.clone();
    unique.sort();
    unique.dedup();
    if dtypes.is_empty() || unique.len() != 1 || !slot.dtypes.contains(&dtypes[0]) {
        return unsupported(format!(
            "{}.{operand} PTX type {:?} cannot use {:?}",
            decoded.op_name,
            decoded.modifier("type")?,
            &dtypes
        ));
    }
    Ok(dtypes.swap_remove(0))
}

const SPACES: [&str; 6] = [
    "",
    "global",
    "shared",
    "shared::cta",
    "shared::cluster",
    "local",
];
const SCOPES: [&str; 4] = ["cta", "cluster", "gpu", "sys"];

fn require_address_dtype(decoded: &DecodedPtx, address: &ObjectRef, space: &str) -> AResult<()> {
    let address_dtype = dtype_of(address)?;
    if address_dtype != "handle"
        && !((address_dtype == "uint32" || address_dtype == "uint64")
            && space.starts_with("shared"))
    {
        return unsupported(format!(
            "{}.addr must lower to a resolved pointer or integer shared address",
            decoded.op_name
        ));
    }
    Ok(())
}

pub struct LoadParts {
    pub destination: ObjectRef,
    pub address: ObjectRef,
    pub result_dtype: String,
    pub ptx_type: String,
    pub space: String,
    pub sem: String,
    pub scope: String,
    pub mmio: bool,
}

/// Space and ordering slots shared by scalar/vector loads and stores.
/// Loads permit omitted ordering slots; stores require the table slots.
fn memory_modifiers(decoded: &DecodedPtx, store: bool) -> AResult<(String, String, String)> {
    let op_name = decoded.op_name.as_str();
    let space = decoded.modifier("space")?.to_owned();
    if !SPACES.contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    let (sem, scope) = if store {
        (decoded.modifier("sem")?, decoded.modifier("scope")?)
    } else {
        (
            decoded.modifier_or_empty("sem"),
            decoded.modifier_or_empty("scope"),
        )
    };
    let ordered = if store { "release" } else { "acquire" };
    if (sem == ordered || sem == "relaxed") && !SCOPES.contains(&scope) {
        return unsupported(format!(
            "{op_name} semantic {:?} requires a supported scope",
            sem
        ));
    }
    if !["", "weak", ordered, "relaxed", "volatile"].contains(&sem) {
        return unsupported(format!("{op_name} has unsupported semantic {:?}", sem));
    }
    Ok((space, sem.to_owned(), scope.to_owned()))
}

pub fn decoded_load_parts(decoded: &DecodedPtx) -> AResult<LoadParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let destination = decoded.scalar_operand("d")?;
    if destination.as_node::<TensorLoadObj>().is_none() {
        return unsupported(format!("{op_name}.d must be a TensorLoad lvalue"));
    }
    let result_dtype = memory_carrier_dtype(decoded, "d")?;
    let ptx_type = decoded.modifier("type")?.to_owned();
    let (mut space, mut sem, scope) = memory_modifiers(decoded, false)?;
    if op_name == "tirx.ptx.ld_proxy_readonly" {
        if decoded.modifier("proxy")? != "proxy::readonly"
            || !(space.is_empty() || space == "global")
        {
            return unsupported("readonly-proxy loads require a global address");
        }
        space = "global".to_owned();
        sem = "readonly".to_owned();
    }
    let mmio = decoded.modifier_or_empty("mmio") == "mmio";
    if mmio && (sem != "relaxed" || scope != "sys" || !(space.is_empty() || space == "global")) {
        return unsupported(format!(
            "{op_name} MMIO loads require relaxed.sys with global or generic addressing"
        ));
    }
    let address = decoded.scalar_operand("addr")?;
    require_address_dtype(decoded, &address, &space)?;
    decoded.cache_policy_operand()?;
    Ok(LoadParts {
        destination,
        address,
        result_dtype,
        ptx_type,
        space,
        sem,
        scope,
        mmio,
    })
}

pub struct VectorLoadParts {
    pub destinations: Vec<Option<ObjectRef>>,
    pub address: ObjectRef,
    pub result_dtype: String,
    pub ptx_type: String,
    pub space: String,
    pub sem: String,
    pub scope: String,
    pub vector_width: i64,
}

fn vector_width_of(token: &str) -> i64 {
    match token {
        "v2" => 2,
        "v4" => 4,
        "v8" => 8,
        _ => 0,
    }
}

pub fn decoded_vector_load_parts(decoded: &DecodedPtx) -> AResult<VectorLoadParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let destinations: Vec<Option<ObjectRef>> = decoded.operand("d")?.to_vec();
    let vector_token = decoded.modifier("vec")?.to_owned();
    let vector_width = vector_width_of(&vector_token);
    if vector_width == 0 || destinations.len() as i64 != vector_width {
        return unsupported(format!(
            "{op_name} has invalid decoded vector width {:?}/{}",
            &vector_token,
            destinations.len()
        ));
    }
    let concrete: Vec<&ObjectRef> = destinations.iter().flatten().collect();
    if concrete.is_empty() {
        return unsupported(format!("{op_name} cannot discard every destination lane"));
    }
    if concrete
        .iter()
        .any(|value| value.as_node::<TensorLoadObj>().is_none())
    {
        return unsupported(format!(
            "{op_name} destinations must be TensorLoad lvalues or explicit sinks"
        ));
    }
    let result_dtype = memory_carrier_dtype(decoded, "d")?;
    let ptx_type = decoded.modifier("type")?.to_owned();
    let (space, sem, scope) = memory_modifiers(decoded, false)?;
    let address = decoded.scalar_operand("addr")?;
    require_address_dtype(decoded, &address, &space)?;
    decoded.cache_policy_operand()?;
    Ok(VectorLoadParts {
        destinations,
        address,
        result_dtype,
        ptx_type,
        space,
        sem,
        scope,
        vector_width,
    })
}

pub struct LdmatrixParts {
    pub destinations: Vec<ObjectRef>,
    pub address: ObjectRef,
    pub transpose: bool,
    pub source_bits: i64,
}

pub fn decoded_ldmatrix_parts(decoded: &DecodedPtx) -> AResult<LdmatrixParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let mut expected: Vec<(&str, String)> = vec![
        ("sync", "sync".to_owned()),
        ("aligned", "aligned".to_owned()),
    ];
    let shape = decoded.modifier("shape")?.to_owned();
    let transpose = decoded.modifier_or_empty("trans") == "trans";
    let source_bits;
    if op_name == "tirx.ptx.ldmatrix" {
        expected.push(("shape", "m8n8".to_owned()));
        expected.push(("type", "b16".to_owned()));
        source_bits = 16;
    } else if op_name == "tirx.ptx.ldmatrix_s8_s4" {
        expected.push(("shape", "m8n16".to_owned()));
        expected.push(("dtype", "s8".to_owned()));
        expected.push(("ctype", "s4".to_owned()));
        source_bits = 4;
    } else if op_name == "tirx.ptx.ldmatrix_m16n16_b8" {
        expected.push(("shape", "m16n16".to_owned()));
        expected.push(("type", "b8".to_owned()));
        expected.push(("trans", "trans".to_owned()));
        source_bits = 8;
    } else {
        expected.push(("dst_fmt", "b8x16".to_owned()));
        source_bits = match decoded.modifier("src_fmt")? {
            "b6x16_p32" => 6,
            "b4x16_p64" => 4,
            _ => 0,
        };
        if source_bits == 0 || !(shape == "m8n16" || shape == "m16n16") {
            return unsupported(format!("{op_name} has unsupported shape/source format"));
        }
        if transpose != (shape == "m16n16") {
            return unsupported(format!("{op_name} requires trans exactly for m16n16"));
        }
    }
    for (name, required) in &expected {
        let actual = decoded.modifier(name)?;
        if actual != required {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                required, actual
            ));
        }
    }
    let num = decoded.modifier("num")?.to_owned();
    let count = match num.as_str() {
        "x1" => 1,
        "x2" => 2,
        "x4" => 4,
        _ => 0,
    };
    let destinations: Vec<Option<ObjectRef>> = decoded.operand("r")?.to_vec();
    let registers = count * if shape == "m16n16" { 2 } else { 1 };
    if count == 0 || registers > 4 || destinations.len() as i64 != registers {
        return unsupported(format!(
            "{op_name} has invalid matrix count {:?}/{}",
            &num,
            destinations.len()
        ));
    }
    let mut concrete = Vec::new();
    for destination in &destinations {
        let Some(destination) = destination else {
            return unsupported(format!(
                "{op_name} destinations must be uint32 TensorLoad lvalues"
            ));
        };
        if destination.as_node::<TensorLoadObj>().is_none() || dtype_of(destination)? != "uint32" {
            return unsupported(format!(
                "{op_name} destinations must be uint32 TensorLoad lvalues"
            ));
        }
        concrete.push(destination.clone());
    }
    let address = decoded.scalar_operand("p")?;
    let space = decoded.modifier("space")?.to_owned();
    if !["", "shared", "shared::cta"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    let address_dtype = dtype_of(&address)?;
    if address_dtype != "handle" && address_dtype != "uint32" {
        return unsupported(format!(
            "{op_name}.p must lower to a resolved pointer or integer shared address"
        ));
    }
    Ok(LdmatrixParts {
        destinations: concrete,
        address,
        transpose,
        source_bits,
    })
}

pub struct StoreParts {
    pub address: ObjectRef,
    /// Value lanes in PTX order; `None` is a 256-bit source sink.
    pub values: Vec<Option<ObjectRef>>,
    pub value_dtype: String,
    pub ptx_type: String,
    pub space: String,
    pub sem: String,
    pub scope: String,
    pub mmio: bool,
    pub vector_width: i64,
}

pub fn decoded_store_parts(decoded: &DecodedPtx) -> AResult<StoreParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    if let Some(predicate) = &decoded.predicate {
        let dtype = dtype_of(predicate)?;
        if !(is_integer_dtype(&dtype) || dtype == "bool") {
            return unsupported(format!("{op_name} predicate must lower to bool or integer"));
        }
    }
    let address = decoded.scalar_operand("addr")?;
    let lanes: Vec<Option<ObjectRef>> = decoded.operand("value")?.to_vec();
    let mut vector_width = 1;
    if op_name != "tirx.ptx.st" {
        let vector_token = decoded.modifier("vec")?.to_owned();
        vector_width = vector_width_of(&vector_token);
        if vector_width == 0 {
            return unsupported(format!(
                "{op_name} has unsupported vector modifier {:?}",
                &vector_token
            ));
        }
    }
    if lanes.len() as i64 != vector_width {
        return unsupported(format!(
            "{op_name} {vector_width}-lane store decoded {} values",
            lanes.len()
        ));
    }
    let concrete = lanes.iter().flatten().count() as i64;
    if concrete == 0 || (concrete != vector_width && op_name != "tirx.ptx.st_vec256") {
        return unsupported(format!(
            "{op_name} sinks require a 256-bit vector with at least one real source"
        ));
    }
    let value_dtype = memory_carrier_dtype(decoded, "value")?;
    let ptx_type = decoded.modifier("type")?.to_owned();
    let (space, sem, scope) = memory_modifiers(decoded, true)?;
    let mmio = decoded.modifier_or_empty("mmio") == "mmio";
    if mmio && (sem != "relaxed" || scope != "sys" || !(space.is_empty() || space == "global")) {
        return unsupported(format!(
            "{op_name} MMIO stores require relaxed.sys with global or generic addressing"
        ));
    }
    require_address_dtype(decoded, &address, &space)?;
    let cache_policy = decoded.cache_policy_operand()?;
    if cache_policy.is_some() && space != "global" {
        return unsupported(format!(
            "{op_name} cache policy is valid only for global memory"
        ));
    }
    Ok(StoreParts {
        address,
        values: lanes,
        value_dtype,
        ptx_type,
        space,
        sem,
        scope,
        mmio,
        vector_width,
    })
}

/// The parsed parts of one raw PTX memory call.
pub enum RawMemoryParts {
    Discard,
    Ldmatrix(LdmatrixParts),
    VectorLoad(VectorLoadParts),
    Store(StoreParts),
    Load(LoadParts),
}

/// The parsed parts.
fn memory_parts(decoded: &DecodedPtx) -> AResult<RawMemoryParts> {
    let op_name = decoded.op_name.as_str();
    let parts = if op_name == "tirx.ptx.discard" {
        if !decoded.result_type.is_empty() {
            return unsupported("discard must return void");
        }
        RawMemoryParts::Discard
    } else if LDMATRIX_CALLS.contains(&op_name) {
        RawMemoryParts::Ldmatrix(decoded_ldmatrix_parts(decoded)?)
    } else if VECTOR_LOAD_CALLS.contains(&op_name) {
        RawMemoryParts::VectorLoad(decoded_vector_load_parts(decoded)?)
    } else if STORE_CALLS.contains(&op_name) {
        RawMemoryParts::Store(decoded_store_parts(decoded)?)
    } else {
        RawMemoryParts::Load(decoded_load_parts(decoded)?)
    };
    Ok(parts)
}

/// The bit set of sunk value lanes.
pub fn store_sinks(values: &[Option<ObjectRef>]) -> i64 {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_none())
        .map(|(index, _)| 1i64 << index)
        .sum()
}

pub(super) fn decoded_load_semantics(sem: &str, scope: &str, mmio: bool) -> String {
    if sem == "readonly" {
        return "v2::mem::variant::Readonly".to_owned();
    }
    if mmio {
        return "v2::mem::variant::MmioRelaxed".to_owned();
    }
    if sem.is_empty() || sem == "weak" {
        return "v2::mem::variant::Plain".to_owned();
    }
    if sem == "volatile" {
        return "v2::mem::variant::Volatile".to_owned();
    }
    let order = if sem == "acquire" {
        "Acquire"
    } else {
        "Relaxed"
    };
    format!(
        "v2::mem::variant::{order}<v2::mem::variant::{}>",
        scope_marker(scope)
    )
}

fn decoded_store_semantics(sem: &str, scope: &str, mmio: bool) -> String {
    if mmio {
        return "v2::mem::variant::MmioRelaxed".to_owned();
    }
    if sem.is_empty() || sem == "weak" {
        return "v2::mem::variant::Plain".to_owned();
    }
    if sem == "volatile" {
        return "v2::mem::variant::Volatile".to_owned();
    }
    let order = if sem == "release" {
        "Release"
    } else {
        "Relaxed"
    };
    format!(
        "v2::mem::variant::{order}<v2::mem::variant::{}>",
        scope_marker(scope)
    )
}

/// ld/st move bits. Signed extension belongs to the
/// register result, not to the storage width or a separate floating-point
/// memory operation.
fn memory_storage_dtype(ptx_type: &str) -> AResult<String> {
    if !(ptx_type.is_empty() || "busf".contains(&ptx_type[..1]))
        || !["8", "16", "32", "64", "128"].contains(&ptx_type.get(1..).unwrap_or(""))
    {
        return unsupported(format!("unmodeled memory instruction type {:?}", ptx_type));
    }
    Ok(format!("uint{}", &ptx_type[1..]))
}

fn decoded_memory_register_type(emitter: &Emitter, ptx_type: &str) -> AResult<String> {
    if &ptx_type[1..] == "8" {
        return Ok("u32".to_owned());
    }
    expr_rust_type(emitter.ctx.schema, &memory_storage_dtype(ptx_type)?)
}

/// The PTX-sized memory carrier of a load or store type.
fn memory_type(emitter: &Emitter, ptx_type: &str) -> AResult<String> {
    v2_memory_type_rust_ptx(
        emitter.ctx.schema,
        &memory_storage_dtype(ptx_type)?,
        Some(&format!("b{}", &ptx_type[1..])),
    )
}

fn decoded_load_variant(
    emitter: &Emitter,
    ptx_type: &str,
    space: &str,
    sem: &str,
    scope: &str,
    mmio: bool,
) -> AResult<String> {
    Ok(format!(
        "v2::mem::variant::Ld<{}, {}, {}>",
        memory_type(emitter, ptx_type)?,
        v2_memory_space_rust(space)?,
        decoded_load_semantics(sem, scope, mmio)
    ))
}

#[allow(clippy::too_many_arguments)]
fn decoded_store_variant(
    emitter: &Emitter,
    ptx_type: &str,
    space: &str,
    sem: &str,
    scope: &str,
    mmio: bool,
    vector_width: i64,
    sinks: i64,
) -> AResult<String> {
    let common = format!(
        "{}, {}, ",
        memory_type(emitter, ptx_type)?,
        v2_memory_space_rust(space)?
    );
    if vector_width != 1 {
        let sinks = if sinks != 0 {
            format!(", {sinks}")
        } else {
            String::new()
        };
        return Ok(format!(
            "v2::mem::variant::StVec<{common}{vector_width}, {}{sinks}>",
            decoded_store_semantics(sem, scope, mmio)
        ));
    }
    Ok(format!(
        "v2::mem::variant::St<{common}{}>",
        decoded_store_semantics(sem, scope, mmio)
    ))
}

/// `shared_address_decode_lines`.
fn shared_address_decode_lines(pointer: &str, result: &str) -> [String; 2] {
    [
        format!(
            "let {result} = WarpValue::from_fn(|lane| decode_generic_shared_address({pointer}[lane]).unwrap_or(u32::MAX));"
        ),
        format!(
            "if ctx.active_mask().into_iter().any(|lane| {result}[lane] == u32::MAX) {{ return Err(EngineError::message(\"cvta.to.shared requires a generic shared address\")); }}"
        ),
    ]
}

fn ptx_bits(ptx_type: &str) -> i64 {
    ptx_type[1..].parse().expect("validated PTX type width")
}

impl<'a> Emitter<'a> {
    /// The logical buffer of an exact address-of.
    pub(super) fn address_logical_buffer(&mut self, address: &ObjectRef) -> AResult<Option<String>> {
        let mut address = address.clone();
        if let Some(call) = address.as_node::<CallObj>() {
            if call_op_name(call)?.as_deref() == Some("tirx.cuda.cvta_generic_to_shared")
                && call.args.len() == 1
            {
                address = oref(call.args.get(0)?);
            }
        }
        let Some(call) = address.as_node::<CallObj>() else {
            return Ok(None);
        };
        if call_op_name(call)?.as_deref() != Some("tirx.address_of") || call.args.len() != 1 {
            return Ok(None);
        }
        let argument = oref(call.args.get(0)?);
        let Some(load) = argument.as_node::<TensorLoadObj>() else {
            return Ok(None);
        };
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return Ok(None);
        };
        Ok(Some(self.logical_buffer_name(&source)?))
    }

    fn emit_ldu_address_check(&mut self, pointer: &RustValue, byte_count: i64) {
        let address = self.control_name("ldu_uniform_address");
        self.emit_line("if !ctx.active_mask().is_empty() {");
        self.emit_line(&format!(
            "let {address} = ({}).resolve_uniform(&ctx, ctx.active_mask())?;",
            pointer.code
        ));
        self.emit_line(&format!(
            "if {address}.byte_offset() % {byte_count}_usize != 0 {{"
        ));
        self.emit_line(&format!(
            "return Err(EngineError::message(\"ldu requires {byte_count}-byte alignment\"));"
        ));
        self.emit_line("}");
        self.emit_line("}");
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_decoded_ordered_b128_load(
        &mut self,
        pointer: &RustValue,
        source_op_id: i64,
        space: &str,
        sem: &str,
        scope: &str,
        logical_buffer: Option<&str>,
        context: Option<&str>,
        active_mask: &str,
    ) -> AResult<String> {
        let source_base = self.control_name("decoded_b128_load_source");
        let physical_offset = self.control_name("decoded_b128_physical_offset");
        self.emit_line(&format!("let {source_base} = ({}).clone();", pointer.code));
        self.emit_line(&format!("for b128_lane in {active_mask} {{"));
        self.emit_line(&format!(
            "    let _ = {source_base}.lane_read_byte_offset(b128_lane, 16_usize)?;"
        ));
        self.emit_line(&format!(
            "    let {physical_offset} = {source_base}.lane_physical_byte_offset(b128_lane, 16_usize)?;"
        ));
        self.emit_line(&format!("    if {physical_offset} % 16_usize != 0 {{"));
        self.emit_line(
            "        return Err(EngineError::message(format!(\"load b128 requires 16-byte alignment on lane {b128_lane}\")));",
        );
        self.emit_line("    }");
        self.emit_line("}");
        let mut loaded_halves = Vec::new();
        let variant = decoded_load_variant(self, "b64", space, sem, scope, false)?;
        for byte_offset in [0, 8] {
            let offsets = self.control_name("decoded_b128_load_offsets");
            let half_pointer = self.control_name("decoded_b128_load_pointer");
            let loaded = self.control_name("decoded_b128_load_half");
            self.emit_line(&format!(
                "let {offsets} = WarpValue::splat({byte_offset}_i64);"
            ));
            self.emit_line(&format!(
                "let {half_pointer} = physical_ptr_byte_offset(&{source_base}, &{offsets}, 8_usize, {active_mask})?;"
            ));
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                "mem::ld",
                &site,
                &[abi::address(
                    v2_memory_space_rust(space)?,
                    &half_pointer,
                    logical_buffer,
                )],
                Some(&variant),
                context,
                false,
                true,
            );
            self.emit_line(&format!("let {loaded} = {call};"));
            self.emit_line(&format!("let {loaded} = v2_register_out({loaded});"));
            loaded_halves.push(loaded);
        }
        let combined = self.control_name("decoded_b128_load_value");
        self.emit_line(&format!(
            "let {combined} = WarpValue::from_fn(|lane| [{}[lane], {}[lane]]);",
            loaded_halves[0], loaded_halves[1]
        ));
        Ok(combined)
    }

    /// `emit_ptx_raw_memory`: evaluate predicated memory operands and effects
    /// under one lane mask.
    fn emit_ptx_raw_memory(
        &mut self,
        decoded: &DecodedPtx,
        parts: &RawMemoryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        if decoded.predicate.is_none() || matches!(parts, RawMemoryParts::Discard) {
            return self.emit_raw_memory_body(decoded, parts, source_op_id);
        }
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_memory",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        self.emit_raw_memory_body(decoded, parts, source_op_id)?;
        let mask = region.mask.clone();
        self.close_predicated_region(region);
        match parts {
            RawMemoryParts::Load(parts) => self.finish_predicated_destinations(
                decoded,
                &[Some(parts.destination.clone())],
                &parts.result_dtype,
                &mask,
                source_op_id,
                false,
            ),
            RawMemoryParts::VectorLoad(parts) => self.finish_predicated_destinations(
                decoded,
                &parts.destinations,
                &parts.result_dtype,
                &mask,
                source_op_id,
                false,
            ),
            RawMemoryParts::Ldmatrix(parts) => {
                let destinations: Vec<Option<ObjectRef>> =
                    parts.destinations.iter().cloned().map(Some).collect();
                self.finish_predicated_destinations(
                    decoded,
                    &destinations,
                    "uint32",
                    &mask,
                    source_op_id,
                    false,
                )
            }
            RawMemoryParts::Store(_) | RawMemoryParts::Discard => Ok(()),
        }
    }

    fn memory_load_value(
        &mut self,
        value: RustValue,
        dtype: &str,
        ptx_type: &str,
    ) -> AResult<(RustValue, Option<&'static str>)> {
        self.emit_extended_register_value(
            value,
            dtype,
            ptx_bits(ptx_type),
            ptx_type.starts_with('s'),
            "PTX memory load",
            "memory_register_extend",
        )
    }

    /// One decoded target-table PTX raw-memory statement.
    fn emit_raw_memory_body(
        &mut self,
        decoded: &DecodedPtx,
        parts: &RawMemoryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        if let RawMemoryParts::Discard = parts {
            let region = self.open_shadow_predicated_region(
                decoded.predicate.as_ref(),
                "discard",
                "discard predicate must be bool or integer",
            )?;
            let pointer = self.emit_address_pointer(
                &decoded.scalar_operand("addr")?,
                "global",
                None,
                "ctx.active_mask()",
            )?;
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                DISCARD,
                &site,
                &[abi::address(
                    "v2::Global",
                    &abi::cloned(&pointer.code),
                    None,
                )],
                None,
                region.context.as_deref(),
                false,
                true,
            );
            self.emit_line(&format!("{call};"));
            self.close_predicated_region(region);
            return Ok(());
        }
        if let RawMemoryParts::Ldmatrix(parts) = parts {
            let logical_buffer = self.address_logical_buffer(&parts.address)?;
            let pointer =
                self.emit_address_pointer(&parts.address, "shared", None, "ctx.active_mask()")?;
            let loaded = self.control_name("decoded_ldmatrix_fragments");
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                "mem::ldmatrix",
                &site,
                &[abi::address(
                    "v2::Shared",
                    &abi::cloned(&pointer.code),
                    logical_buffer.as_deref(),
                )],
                Some(&format!(
                    "v2::mem::variant::Ldmatrix<{}, {}, {}, {}>",
                    parts.destinations.len(),
                    parts.transpose,
                    parts.source_bits,
                    op_name == "tirx.ptx.ldmatrix_s8_s4"
                )),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {loaded} = {call};"));
            for (index, destination) in parts.destinations.iter().enumerate() {
                let fragment = self.control_name("decoded_ldmatrix_fragment");
                self.emit_line(&format!(
                    "let {fragment} = {loaded}.clone().map(|_lane, fragments| fragments[{index}]);"
                ));
                self.emit_line(&format!("let {fragment} = v2_register_out({fragment});"));
                self.emit_explicit_buffer_store(
                    destination,
                    RustValue::new(fragment, "u32", Uniformity::Varying),
                    source_op_id,
                    None,
                    None,
                    None,
                )?;
            }
            return Ok(());
        }
        if let RawMemoryParts::VectorLoad(parts) = parts {
            let mut space = parts.space.clone();
            let logical_buffer = self.address_logical_buffer(&parts.address)?;
            let pointer =
                self.emit_address_pointer(&parts.address, &space, None, "ctx.active_mask()")?;
            let source_base = self.control_name("decoded_vector_load_source");
            self.emit_line(&format!("let {source_base} = ({}).clone();", pointer.code));
            // Memory stride is defined by PTX .type, not its wider register carrier.
            let itemsize = ptx_bits(&parts.ptx_type) / 8;
            if op_name == "tirx.ptx.ldu_vec" {
                self.emit_ldu_address_check(&pointer, itemsize * parts.vector_width);
                space = "global".to_owned();
            }
            let engine_rust_type = decoded_memory_register_type(self, &parts.ptx_type)?;
            let mut loaded_values = Vec::new();
            for index in 0..parts.vector_width {
                let offsets = self.control_name("decoded_vector_load_offsets");
                let element_pointer = self.control_name("decoded_vector_load_pointer");
                let loaded = self.control_name("decoded_vector_load_value");
                self.emit_line(&format!(
                    "let {offsets} = WarpValue::splat({}_i64);",
                    index * itemsize
                ));
                self.emit_line(&format!(
                    "let {element_pointer} = physical_ptr_byte_offset(&{source_base}, &{offsets}, {itemsize}_usize, ctx.active_mask())?;"
                ));
                let variant = decoded_load_variant(
                    self,
                    &parts.ptx_type,
                    &space,
                    &parts.sem,
                    &parts.scope,
                    false,
                )?;
                let site = self.v2_site(Some(source_op_id));
                let call = abi::warp_call(
                    "mem::ld",
                    &site,
                    &[abi::address(
                        v2_memory_space_rust(&space)?,
                        &element_pointer,
                        logical_buffer.as_deref(),
                    )],
                    Some(&variant),
                    None,
                    false,
                    true,
                );
                self.emit_line(&format!("let {loaded} = {call};"));
                self.emit_line(&format!("let {loaded} = v2_register_out({loaded});"));
                let Some(destination) = &parts.destinations[index as usize] else {
                    continue;
                };
                let loaded_value =
                    RustValue::new(loaded, engine_rust_type.clone(), Uniformity::Varying);
                let (loaded_value, storage_dtype) =
                    self.memory_load_value(loaded_value, &parts.result_dtype, &parts.ptx_type)?;
                loaded_values.push((destination, loaded_value, storage_dtype));
            }
            for (destination, loaded_value, storage_dtype) in loaded_values {
                self.emit_explicit_buffer_store(
                    destination,
                    loaded_value,
                    source_op_id,
                    None,
                    Some("ctx.active_mask()"),
                    storage_dtype,
                )?;
            }
            return Ok(());
        }
        if let RawMemoryParts::Store(parts) = parts {
            let logical_buffer = self.address_logical_buffer(&parts.address)?;
            let pointer =
                self.emit_address_pointer(&parts.address, &parts.space, None, "ctx.active_mask()")?;
            let carrier_width = dtype_byte_len(self.ctx.schema, &parts.value_dtype)? * 8;
            let mut raw_values = Vec::new();
            // The store owns the loads its value reads: an uninitialized read
            // reports this call, not the TensorLoad the value happens to be.
            self.with_load_site(Some(NestedLoadSite::Site(Some(source_op_id))), |emitter| {
                for value_expression in &parts.values {
                    raw_values.push(match value_expression {
                        None => None,
                        Some(value_expression)
                            if parts.value_dtype == format!("uint{carrier_width}") =>
                        {
                            Some(emitter.emit_expr(value_expression)?)
                        }
                        Some(value_expression) => Some(emitter.emit_as_unsigned_bits(
                            value_expression,
                            carrier_width,
                            &op_name,
                            "memory_store_bits",
                            None,
                        )?),
                    });
                }
                Ok(())
            })?;
            let engine_rust_type = decoded_memory_register_type(self, &parts.ptx_type)?;
            // Sinks have no source evaluation and are excluded from writes/footprints.
            let mut values = Vec::new();
            for value in raw_values {
                let value = match value {
                    Some(value) => self.observe_pointer_bits(value),
                    None => RustValue::new(
                        Emitter::zero_literal(&engine_rust_type)?,
                        engine_rust_type.clone(),
                        Uniformity::Uniform,
                    ),
                };
                values.push(self.as_warp_value(value));
            }
            let mut engine_values = Vec::new();
            for mut value in values {
                if value.rust_type == "U64x2" && engine_rust_type != "U64x2" {
                    let low = self.control_name("memory_store_low_half");
                    self.emit_line(&format!(
                        "let {low} = WarpValue::from_fn(|lane| {}[lane][0]);",
                        value.code
                    ));
                    value = RustValue::new(low, "u64", Uniformity::Varying);
                }
                engine_values.push(self.coerce_value(
                    value,
                    &engine_rust_type,
                    "memory_store_low_bits",
                )?);
            }
            let address = abi::address(
                v2_memory_space_rust(&parts.space)?,
                &abi::cloned(&pointer.code),
                logical_buffer.as_deref(),
            );
            let registers: Vec<String> = engine_values
                .iter()
                .map(|value| abi::register(&value.code))
                .collect();
            let value_argument = if parts.vector_width == 1 {
                registers[0].clone()
            } else {
                format!("[{}]", registers.join(", "))
            };
            let variant = decoded_store_variant(
                self,
                &parts.ptx_type,
                &parts.space,
                &parts.sem,
                &parts.scope,
                parts.mmio,
                parts.vector_width,
                store_sinks(&parts.values),
            )?;
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                "mem::st",
                &site,
                &[format!("({address}, {value_argument})")],
                Some(&variant),
                None,
                false,
                true,
            );
            self.emit_line(&format!("{call};"));
            return Ok(());
        }
        let RawMemoryParts::Load(parts) = parts else {
            unreachable!("discard, ldmatrix, vector load and store parts are lowered above");
        };
        let mut space = parts.space.clone();
        let logical_buffer = self.address_logical_buffer(&parts.address)?;
        let pointer =
            self.emit_address_pointer(&parts.address, &space, None, "ctx.active_mask()")?;
        if op_name == "tirx.ptx.ldu" {
            self.emit_ldu_address_check(&pointer, ptx_bits(&parts.ptx_type) / 8);
            space = "global".to_owned();
        }
        let loaded;
        if parts.ptx_type == "b128"
            && (parts.sem == "acquire" || parts.sem == "relaxed")
            && !parts.mmio
        {
            loaded = self.emit_decoded_ordered_b128_load(
                &pointer,
                source_op_id,
                &space,
                &parts.sem,
                &parts.scope,
                logical_buffer.as_deref(),
                None,
                "ctx.active_mask()",
            )?;
        } else {
            loaded = self.control_name("decoded_raw_load");
            let variant = decoded_load_variant(
                self,
                &parts.ptx_type,
                &space,
                &parts.sem,
                &parts.scope,
                parts.mmio,
            )?;
            let site = self.v2_site(Some(source_op_id));
            let loaded_call = abi::warp_call(
                "mem::ld",
                &site,
                &[abi::address(
                    v2_memory_space_rust(&space)?,
                    &abi::cloned(&pointer.code),
                    logical_buffer.as_deref(),
                )],
                Some(&variant),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {loaded} = v2_register_out({loaded_call});"));
        }
        let engine_rust_type = decoded_memory_register_type(self, &parts.ptx_type)?;
        let loaded_value = RustValue::new(loaded, engine_rust_type, Uniformity::Varying);
        let (loaded_value, storage_dtype) =
            self.memory_load_value(loaded_value, &parts.result_dtype, &parts.ptx_type)?;
        self.emit_explicit_buffer_store(
            &parts.destination,
            loaded_value,
            source_op_id,
            None,
            Some("ctx.active_mask()"),
            storage_dtype,
        )
    }

    /// `emit_raw_ldmatrix_legacy`.
    fn emit_raw_ldmatrix_legacy(
        &mut self,
        call: &LegacyMemoryCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let address = call.address.as_ref().expect("address");
        let destination_expr = call.destination.as_ref().expect("destination");
        let source_offset_expr = call.source_offset.as_ref().expect("source offset");
        let destination_offset_expr = call
            .destination_offset
            .as_ref()
            .expect("destination offset");
        let target_dtype = call.target_dtype.as_ref().expect("target dtype");
        let source = self.emit_pointer_handle(address)?;
        let destination = self.emit_pointer_handle(destination_expr)?;
        if source.rust_type != "PhysicalPtr" || destination.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{} source/destination did not resolve to physical addresses",
                call.op_name
            ));
        }
        let source_offsets = self.emit_expr(source_offset_expr)?;
        let source_offsets = self.as_i64(source_offsets)?;
        let source_offsets = self.as_warp_value(source_offsets);
        let destination_offsets = self.emit_expr(destination_offset_expr)?;
        let destination_offsets = self.as_i64(destination_offsets)?;
        let destination_offsets = self.as_warp_value(destination_offsets);
        let itemsize = dtype_byte_len(self.ctx.schema, target_dtype)?;
        if itemsize == 1 && call.transpose {
            self.emit_legacy_ldmatrix_i8_gather(
                &source.code,
                &destination.code,
                &source_offsets.code,
                &destination_offsets.code,
                source_op_id,
            );
            return Ok(());
        }
        let source_byte_offsets = self.control_name("legacy_ldmatrix_source_byte_offsets");
        let lane = self.control_name("legacy_ldmatrix_source_lane");
        self.emit_line(&format!(
            "let mut {source_byte_offsets} = WarpValue::splat(0_i64);"
        ));
        self.emit_line(&format!("for {lane} in ctx.active_mask() {{"));
        self.emit_line(&format!(
            "    {source_byte_offsets}[{lane}] = {}[{lane}].checked_mul({itemsize}_i64).ok_or_else(|| EngineError::message(\"legacy ldmatrix source byte offset overflow\"))?;",
            source_offsets.code
        ));
        self.emit_line("}");
        let source_pointer = self.control_name("legacy_ldmatrix_source_pointer");
        self.emit_line(&format!(
            "let {source_pointer} = physical_ptr_byte_offset(&{}, &{source_byte_offsets}, {itemsize}_usize, ctx.active_mask())?;",
            source.code
        ));
        let loaded = self.control_name("legacy_ldmatrix_fragments");
        let site = self.v2_site(Some(source_op_id));
        let load = abi::warp_call(
            "mem::ldmatrix",
            &site,
            &[abi::address("v2::Shared", &source_pointer, None)],
            Some(&format!(
                "v2::mem::variant::Ldmatrix<{}, {}>",
                call.matrix_count, call.transpose
            )),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {loaded} = {load};"));
        for index in 0..call.matrix_count {
            let fragment = self.control_name(&format!("legacy_ldmatrix_fragment_{index}"));
            let destination_byte_offsets =
                self.control_name(&format!("legacy_ldmatrix_destination_byte_offsets_{index}"));
            let destination_lane =
                self.control_name(&format!("legacy_ldmatrix_destination_lane_{index}"));
            let destination_pointer =
                self.control_name(&format!("legacy_ldmatrix_destination_pointer_{index}"));
            self.emit_line(&format!(
                "let {fragment} = {loaded}.clone().map(|_lane, fragments| fragments[{index}]);"
            ));
            self.emit_line(&format!(
                "let mut {destination_byte_offsets} = WarpValue::splat(0_i64);"
            ));
            self.emit_line(&format!("for {destination_lane} in ctx.active_mask() {{"));
            self.emit_line(&format!(
                "    {destination_byte_offsets}[{destination_lane}] = {}[{destination_lane}].checked_mul({itemsize}_i64).and_then(|value| value.checked_add({}_i64)).ok_or_else(|| EngineError::message(\"legacy ldmatrix destination byte offset overflow\"))?;",
                destination_offsets.code,
                index * 4
            ));
            self.emit_line("}");
            self.emit_line(&format!(
                "let {destination_pointer} = physical_ptr_byte_offset(&{}, &{destination_byte_offsets}, 4_usize, ctx.active_mask())?;",
                destination.code
            ));
            self.emit_line(&format!(
                "write_frontend_register_result::<u32>(&physical, {}, &{destination_pointer}, {fragment})?;",
                abi::context("ctx")
            ));
        }
        Ok(())
    }

    fn emit_legacy_ldmatrix_i8_gather(
        &mut self,
        source: &str,
        destination: &str,
        source_strides: &str,
        destination_offsets: &str,
        source_op_id: i64,
    ) {
        for element in 0..16 {
            let source_byte_offsets =
                self.control_name(&format!("legacy_ldmatrix_gather_source_{element}"));
            let lane = self.control_name(&format!("legacy_ldmatrix_gather_lane_{element}"));
            let thread = self.control_name(&format!("legacy_ldmatrix_gather_thread_{element}"));
            let stride = self.control_name(&format!("legacy_ldmatrix_gather_stride_{element}"));
            let source_element =
                self.control_name(&format!("legacy_ldmatrix_gather_element_{element}"));
            self.emit_line(&format!(
                "let mut {source_byte_offsets} = WarpValue::splat(0_i64);"
            ));
            self.emit_line(&format!("for {lane} in ctx.active_mask() {{"));
            self.emit_line(&format!(
                "    let {thread} = ctx.warp_id_in_cta().checked_mul(WARP_SIZE).and_then(|value| value.checked_add({lane})).ok_or_else(|| EngineError::message(\"legacy ldmatrix thread index overflow\"))?;"
            ));
            self.emit_line(&format!(
                "    let {stride} = usize::try_from({source_strides}[{lane}]).map_err(|_| EngineError::message(\"negative legacy ldmatrix shared stride\"))?;"
            ));
            self.emit_line(&format!(
                "    let {source_element} = {}_usize.checked_mul({stride}).and_then(|value| value.checked_mul(16_usize)).and_then(|value| value.checked_add(({thread} % 4_usize) * 4_usize * {stride})).and_then(|value| value.checked_add({}_usize * {stride})).and_then(|value| value.checked_add({thread} / 4_usize)).and_then(|value| value.checked_add({}_usize)).ok_or_else(|| EngineError::message(\"legacy ldmatrix source offset overflow\"))?;",
                (element % 8) / 4,
                element % 4,
                element / 8 * 8
            ));
            self.emit_line(&format!(
                "    {source_byte_offsets}[{lane}] = i64::try_from({source_element}).map_err(|_| EngineError::message(\"legacy ldmatrix source offset exceeds i64\"))?;"
            ));
            self.emit_line("}");
            let source_pointer =
                self.control_name(&format!("legacy_ldmatrix_gather_source_pointer_{element}"));
            let loaded = self.control_name(&format!("legacy_ldmatrix_gather_value_{element}"));
            self.emit_line(&format!(
                "let {source_pointer} = physical_ptr_byte_offset(&{source}, &{source_byte_offsets}, 1_usize, ctx.active_mask())?;"
            ));
            let site = self.v2_site(Some(source_op_id));
            let load = abi::warp_call(
                "mem::ld",
                &site,
                &[abi::address("v2::Shared", &source_pointer, None)],
                Some("v2::mem::variant::Ld<v2::reg::variant::U8, v2::Shared>"),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {loaded} = {load};"));
            let destination_byte_offsets =
                self.control_name(&format!("legacy_ldmatrix_gather_destination_{element}"));
            let destination_lane = self.control_name(&format!(
                "legacy_ldmatrix_gather_destination_lane_{element}"
            ));
            self.emit_line(&format!(
                "let mut {destination_byte_offsets} = ({destination_offsets}).clone();"
            ));
            self.emit_line(&format!("for {destination_lane} in ctx.active_mask() {{"));
            self.emit_line(&format!(
                "    {destination_byte_offsets}[{destination_lane}] = {destination_byte_offsets}[{destination_lane}].checked_add({element}_i64).ok_or_else(|| EngineError::message(\"legacy ldmatrix destination offset overflow\"))?;"
            ));
            self.emit_line("}");
            let destination_pointer = self.control_name(&format!(
                "legacy_ldmatrix_gather_destination_pointer_{element}"
            ));
            self.emit_line(&format!(
                "let {destination_pointer} = physical_ptr_byte_offset(&{destination}, &{destination_byte_offsets}, 1_usize, ctx.active_mask())?;"
            ));
            self.emit_line(&format!(
                "write_frontend_register_result::<u8>(&physical, {}, &{destination_pointer}, {loaded})?;",
                abi::context("ctx")
            ));
        }
    }

    /// `emit_ldg32`.
    fn emit_ldg32(&mut self, call: &LegacyMemoryCall, source_op_id: i64) -> AResult<()> {
        let destination_expr = call.destination.as_ref().expect("destination");
        let destination_offset_expr = call.destination_offset.as_ref().expect("offset");
        let source_load = call.value.as_ref().expect("source load");
        let predicate_expr = call.predicate.as_ref().expect("predicate");
        let destination = self.emit_pointer_handle(destination_expr)?;
        if destination.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{} destination did not resolve to a physical address",
                call.op_name
            ));
        }
        let guard = self.emit_expr(predicate_expr)?;
        let guard = self.coerce_value(guard, "u32", "ldg32_guard")?;
        let guard = self.as_warp_value(guard);
        let predicate = self.control_name("ldg32_predicate");
        let predicate_values = self.control_name("ldg32_predicate_values");
        let mask = self.control_name("ldg32_guard_mask");
        let site = self.v2_site(Some(source_op_id));
        let setp = abi::lane_call_context(
            "reg::setp",
            &site,
            &[format!(
                "({}, {})",
                abi::register(&guard.code),
                abi::splat("0_u32")
            )],
            Some("v2::reg::variant::Setp<v2::reg::variant::U32, v2::reg::variant::Ne>"),
            None,
        );
        self.emit_line(&format!("let {predicate} = {setp};"));
        self.emit_line(&format!(
            "let {predicate_values} = v2_register_out({predicate});"
        ));
        self.emit_line(&format!(
            "let {mask} = {predicate_values}.to_mask(|_, value| *value) & ctx.active_mask();"
        ));
        let load = source_load
            .as_node::<TensorLoadObj>()
            .expect("validated TensorLoad");
        let Some(source_buffer) = as_buffer(&oref(load.source.clone())) else {
            return crate::analyze::util::not_covered("ldg32 source is not a typed buffer");
        };
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let source_code = self.buffer_code(&source_buffer)?;
        if self.plan_of(&source_code).space != MemorySpace::Global {
            return unsupported(format!("{} source must be global memory", call.op_name));
        }
        let source_index = self.physical_index(&source_buffer, &indices)?;
        let source_mask = self.physical_access_mask(&source_buffer, &indices, &mask)?;
        let source_context = self.control_name("ldg32_source_context");
        let loaded = self.control_name("ldg32_loaded");
        let buffer_ref = self.buffer_ref(&source_buffer)?;
        self.emit_line(&format!(
            "let {source_context} = ctx.with_active_mask({source_mask});"
        ));
        let logical_name = self.logical_buffer_name(&source_buffer)?;
        let site = self.v2_site(Some(source_op_id));
        let load_call = abi::warp_call(
            "mem::ld",
            &site,
            &[abi::buffer_address(
                "v2::Global",
                &buffer_ref,
                &source_index.code,
                4,
                &logical_name,
            )],
            Some(
                "v2::mem::variant::Ld<v2::reg::variant::F32, v2::Global, v2::mem::variant::Plain>",
            ),
            Some(&abi::context(&source_context)),
            false,
            true,
        );
        self.emit_line(&format!("let {loaded} = v2_register_out({load_call});"));
        let result = self.control_name("ldg32_result");
        let zero = self.control_name("ldg32_zero");
        let site = self.v2_site(Some(source_op_id));
        let mov = abi::lane_call_context(
            "reg::mov",
            &site,
            &[abi::splat("0.0_f32")],
            Some("v2::reg::variant::F32"),
            Some(&abi::context(&format!(
                "ctx.with_active_mask(ctx.active_mask() - {mask})"
            ))),
        );
        self.emit_line(&format!("let {zero} = {mov};"));
        self.emit_line(&format!("let mut {result} = v2_register_out({zero});"));
        self.emit_line(&format!("{result}.masked_assign({mask}, &{loaded});"));
        let offsets = self.emit_expr(destination_offset_expr)?;
        let offsets = self.as_i64(offsets)?;
        let offsets = self.as_warp_value(offsets);
        let extents = self.control_name("ldg32_destination_extent");
        let pointer = self.control_name("ldg32_destination");
        self.emit_line(&format!("let {extents} = WarpValue::splat(1_i64);"));
        self.emit_line(&format!(
            "let {pointer} = {}.with_element_offset_extent(&{}, &{extents}, 4_usize, ctx.active_mask(), 2_u8, \"{}\")?;",
            destination.code, offsets.code, call.op_name
        ));
        self.emit_line(&format!(
            "write_frontend_register_result::<f32>(&physical, {}, &{pointer}, v2_register({result}))?;",
            abi::context("ctx")
        ));
        Ok(())
    }

    fn emit_ptr_byte_offset(&mut self, call: &LegacyMemoryCall) -> AResult<RustValue> {
        let mut pointer = self.emit_expr(call.address.as_ref().expect("address"))?;
        if pointer.rust_type == "PhysicalPtr" {
            let address = self.temp("ptr_byte_offset_address");
            self.emit_line(&format!(
                "let {address} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
                pointer.code
            ));
            pointer = RustValue::new(address, "u64", Uniformity::Varying);
        }
        let pointer = self.as_warp_value(pointer);
        if pointer.rust_type != "u64" {
            return unsupported(format!(
                "ptr_byte_offset source lowered to {}, expected u64",
                pointer.rust_type
            ));
        }
        let offset = self.emit_expr(call.byte_offset.as_ref().expect("byte offset"))?;
        let offset = self.as_i64(offset)?;
        let offset = self.as_warp_value(offset);
        let result = self.temp("ptr_byte_offset");
        self.emit_line(&format!(
            "let {result} = WarpValue::from_fn(|lane| {}[lane].wrapping_add({}[lane] as u64));",
            pointer.code, offset.code
        ));
        Ok(RustValue::new(result, "u64", Uniformity::Varying))
    }

    fn emit_cvta_shared(&mut self, call: &LegacyMemoryCall) -> AResult<RustValue> {
        let mut pointer = self.emit_expr(call.address.as_ref().expect("address"))?;
        let result = self.temp("shared_address");
        if pointer.rust_type == "PhysicalPtr" {
            self.emit_line(&format!(
                "let {result} = ({}).shared_byte_addresses_u32(&ctx, ctx.active_mask())?;",
                pointer.code
            ));
        } else {
            pointer = self.as_warp_value(pointer);
            if pointer.rust_type != "u64" {
                return unsupported(format!(
                    "cvta_generic_to_shared source lowered to {}, expected u64",
                    pointer.rust_type
                ));
            }
            if self.use_typed_helpers && self.selected_lane.is_none() {
                self.emit_line(&format!(
                    "let {result} = numsim_decode_shared_address(&({}), &physical, ctx)?;",
                    pointer.code
                ));
            } else {
                for line in shared_address_decode_lines(&pointer.code, &result) {
                    self.emit_line(&line);
                }
            }
        }
        let output = RustValue::new(result, "u32", Uniformity::Varying);
        let Some(call_expr) = self.call_expr_stack.last().cloned() else {
            return Ok(output);
        };
        self.instrument(
            &call_expr,
            "Call:cvta_generic_to_shared",
            &[&pointer],
            output,
        )
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = memory_parts(decoded)?;
    if let RawMemoryParts::Store(store) = &parts {
        emitter.record_pointer_write(&store.address, &store.space)?;
    } else if matches!(parts, RawMemoryParts::Discard) {
        emitter.record_pointer_write(&decoded.scalar_operand("addr")?, "global")?;
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_raw_memory(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_legacy(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = resolve_legacy_memory_call(emitter.ctx, call.node, &call.op_name)?;
    match call.op_name.as_str() {
        "tirx.ptx_legacy.ldmatrix" | "tirx.s_tir.ldg32" => {
            let source_op_id = call.source_op_id(emitter)?;
            if call.op_name == "tirx.ptx_legacy.ldmatrix" {
                emitter.emit_raw_ldmatrix_legacy(&parts, source_op_id)?;
            } else {
                emitter.emit_ldg32(&parts, source_op_id)?;
            }
            Ok(None)
        }
        _ => emitter
            .with_call_expr(call.node, |emitter| match call.op_name.as_str() {
                "tirx.type_annotation" => {
                    Ok(RustValue::new("()", "TypeAnnotation", Uniformity::Uniform))
                }
                "tirx.ptr_byte_offset" => emitter.emit_ptr_byte_offset(&parts),
                "tirx.cuda.cvta_generic_to_shared" => emitter.emit_cvta_shared(&parts),
                other => crate::analyze::util::not_covered(format!(
                    "raw memory expression {other} has no lowering"
                )),
            })
            .map(Some),
    }
}

pub fn uses_readonly_proxy(nodes: &[ObjectRef]) -> AResult<bool> {
    for node in nodes {
        if crate::decode::call_name(node)?.as_deref() == Some("tirx.ptx.ld_proxy_readonly") {
            return Ok(true);
        }
    }
    Ok(false)
}
