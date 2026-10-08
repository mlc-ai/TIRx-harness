//! Validation and emission of the memory_ops instruction family.

use crate::analyze::buffers::call_op_name;
use crate::analyze::util::{
    as_var, dtype_of, expr_type, ffi_text, is_pointer_type, oref, prim, repr_text, static_int,
    unmodeled, unsupported, AResult,
};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::{shared_pointer_source, SharedAddressForms};
use crate::emit::raw_memory::is_type_annotation;
use crate::emit::register_call::require_register_call;
use crate::emit::{abi, join_uniformity, Emitter, RustValue, Uniformity};
use crate::tables::is_integer_dtype;
use crate::tables::{
    dtype_by_rust_type, dtype_byte_len, expr_rust_type, json_string, v2_memory_type_rust, MEM_LD,
};
use tvm::ir::CallObj;
use tvm::ir::StringImmObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

/// The engine functions (below `v2::`) this family's lowerings call.
pub const CP_ASYNC: &str = "async_copy::cp_async";
pub const STMATRIX: &str = "mem::stmatrix";

const CP_ASYNC_CA_CALLS: [&str; 3] = [
    "tirx.ptx.cp_async_ca",
    "tirx.ptx.cp_async_ca_ignore_src",
    "tirx.ptx.cp_async_ca_src_size",
];
const CP_ASYNC_IGNORE_SRC_CALLS: [&str; 2] = [
    "tirx.ptx.cp_async_ca_ignore_src",
    "tirx.ptx.cp_async_cg_ignore_src",
];
const CP_ASYNC_SRC_SIZE_CALLS: [&str; 2] = [
    "tirx.ptx.cp_async_ca_src_size",
    "tirx.ptx.cp_async_cg_src_size",
];

fn call_of(node: &ObjectRef) -> AResult<&CallObj> {
    match node.as_node::<CallObj>() {
        Some(call) => Ok(call),
        None => Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("memory lowerer requires a TIRx Call"),
        )),
    }
}

pub struct AccessPtrParts {
    pub target_dtype: String,
    pub address: ObjectRef,
    pub element_offset: ObjectRef,
    pub element_extent: ObjectRef,
    pub access_mask: i64,
}

pub fn access_ptr_parts(ctx: &Ctx, node: &ObjectRef) -> AResult<AccessPtrParts> {
    let call = call_of(node)?;
    let op_name = "tirx.tvm_access_ptr";
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    if args.len() != 5 || dtype_of(node)? != "handle" {
        return unsupported(format!(
            "{op_name} requires five arguments and handle result"
        ));
    }
    let (annotation, address, element_offset, element_extent, access_mask) =
        (&args[0], &args[1], &args[2], &args[3], &args[4]);
    if !is_type_annotation(annotation)? {
        return unsupported(format!(
            "{op_name} requires a zero-argument type annotation"
        ));
    }
    let target_dtype = dtype_of(annotation)?;
    dtype_byte_len(ctx.schema, &target_dtype)?;
    if dtype_of(address)? != "handle" {
        return unsupported(format!("{op_name} base address must be handle"));
    }
    if !is_integer_dtype(&dtype_of(element_offset)?)
        || !is_integer_dtype(&dtype_of(element_extent)?)
    {
        return unsupported(format!("{op_name} offset and extent must be integers"));
    }
    let mask = static_int(
        &ctx.analyzer,
        &prim(access_mask)?,
        &format!("{op_name}.access_mask"),
        "must be static",
    )?;
    if ![1, 2, 3].contains(&mask) {
        return unsupported(format!(
            "{op_name} access mask must be read=1, write=2, or both=3"
        ));
    }
    Ok(AccessPtrParts {
        target_dtype,
        address: address.clone(),
        element_offset: element_offset.clone(),
        element_extent: element_extent.clone(),
        access_mask: mask,
    })
}

/// The validated parts.

fn is_cuda_ldg_dtype(ctx: &Ctx, dtype: &str) -> bool {
    if ctx.schema.scalar_dtype_bits.contains_key(dtype) {
        return is_integer_dtype(dtype)
            || matches!(dtype, "float16" | "bfloat16" | "float32" | "float64");
    }
    let Some((element, lanes, _, total_bits)) = ctx.schema.vector_dtype_abi(dtype) else {
        return false;
    };
    if element == "bool" {
        return false;
    }
    if element.starts_with("float8_") {
        return total_bits == 64 || total_bits == 128;
    }
    if element == "float16" || element == "bfloat16" {
        return lanes == 2 || lanes == 8;
    }
    is_integer_dtype(&element) || element == "float32" || element == "float64"
}

pub fn cuda_ldg_parts(ctx: &Ctx, node: &ObjectRef) -> AResult<(String, ObjectRef)> {
    let call = call_of(node)?;
    let op_name = "tirx.cuda.ldg";
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    let result_dtype = dtype_of(node)?;
    if args.len() != 2 || dtype_of(&args[0])? != "handle" {
        return unsupported(format!("{op_name} requires (handle, dtype_name)"));
    }
    let dtype_name = match args[1].as_node::<StringImmObj>() {
        Some(imm) => ffi_text(&imm.value),
        None => {
            return unsupported(format!(
                "{op_name}.dtype must be a static string, got {}",
                repr_text(&args[1])?
            ))
        }
    };
    if result_dtype == dtype_name && (result_dtype == "boolx2" || result_dtype == "boolx4") {
        return unmodeled(
            format!("call:{op_name}"),
            format!(
                "{op_name} compile-valid dtype {:?} has no exact NumSim packed-bool ABI",
                &result_dtype
            ),
        );
    }
    if result_dtype != dtype_name || !is_cuda_ldg_dtype(ctx, &result_dtype) {
        return unsupported(format!(
            "{op_name} dtype attribute {:?} does not match supported result dtype {:?}",
            &dtype_name, &result_dtype
        ));
    }
    let address = args[0].clone();
    if let Some(inner) = address.as_node::<CallObj>() {
        if call_op_name(inner)?.as_deref() == Some("tirx.address_of") && inner.args.len() == 1 {
            let pointee_dtype = dtype_of(&oref(inner.args.get(0)?))?;
            if pointee_dtype != result_dtype {
                return unsupported(format!(
                    "{op_name} direct pointer dtype {:?} does not match loaded dtype {:?}",
                    &pointee_dtype, &result_dtype
                ));
            }
        }
    }
    Ok((result_dtype, address))
}

/// (v2 variant, Rust storage type, optional decoder).
pub fn cuda_ldg_variant(
    ctx: &Ctx,
    result_dtype: &str,
) -> AResult<(String, String, Option<&'static str>)> {
    let (storage_type, decoder): (String, Option<&'static str>) = match result_dtype {
        "float16" => ("u16".to_owned(), Some("fp16_bits_to_f32")),
        "bfloat16" => ("u16".to_owned(), Some("bf16_bits_to_f32")),
        _ => (expr_rust_type(ctx.schema, result_dtype)?, None),
    };
    let Some(storage_dtype) = dtype_by_rust_type(&storage_type) else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error(&format!(
                "cuda.ldg storage type {storage_type} has no dtype"
            )),
        ));
    };
    Ok((
        format!(
            "v2::mem::variant::Ld<{}, v2::Global>",
            v2_memory_type_rust(ctx.schema, storage_dtype)?
        ),
        storage_type,
        decoder,
    ))
}

/// A validated `cuda.ldg` load: its result dtype and address.
pub struct CudaLdg {
    pub result_dtype: String,
    pub address: ObjectRef,
}

/// The validated parts.

pub fn cp_async_variant(byte_count: i64, fill_mode: &str) -> String {
    let fill_variant = match fill_mode {
        "zero" => "ZeroFill",
        "size" => "SourceSize",
        _ => "NoFill",
    };
    format!(
        "v2::async_copy::variant::CpAsync<{byte_count}, v2::async_copy::variant::{fill_variant}>"
    )
}

pub struct CpAsyncParts {
    /// The shared source `dst_mem` retains.
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub byte_count: i64,
    pub source_size: Option<ObjectRef>,
    pub ignore_source: Option<ObjectRef>,
    pub cache_policy: Option<ObjectRef>,
    pub predicate: Option<ObjectRef>,
}

pub fn cp_async_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<CpAsyncParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let expected_cop = if CP_ASYNC_CA_CALLS.contains(&op_name) {
        "ca"
    } else {
        "cg"
    };
    for (name, required) in [("api", "async"), ("cop", expected_cop), ("src", "global")] {
        let actual = decoded.modifier(name)?;
        if actual != required {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                required, actual
            ));
        }
    }
    let destination_space = decoded.modifier("dst")?;
    if destination_space != "shared" && destination_space != "shared::cta" {
        return unsupported(format!(
            "{op_name} has unsupported destination space {:?}",
            destination_space
        ));
    }
    let destination = decoded.scalar_operand("dst_mem")?;
    let source = decoded.scalar_operand("src_mem")?;
    let byte_count = static_int(
        &ctx.analyzer,
        &prim(&decoded.scalar_operand("cp_size")?)?,
        &format!("{op_name}.cp_size"),
        "must be static",
    )?;
    if ![4, 8, 16].contains(&byte_count) {
        return unsupported(format!("{op_name}.cp_size must be 4, 8, or 16"));
    }
    let source_size = if CP_ASYNC_SRC_SIZE_CALLS.contains(&op_name) {
        Some(decoded.scalar_operand("src_size")?)
    } else {
        None
    };
    if let Some(size) = &source_size {
        let dtype = dtype_of(size)?;
        if !is_integer_dtype(&dtype) {
            return unsupported(format!(
                "{op_name}.src_size must be an integer, got {:?}",
                &dtype
            ));
        }
    }
    let ignore_source = if CP_ASYNC_IGNORE_SRC_CALLS.contains(&op_name) {
        Some(decoded.scalar_operand("ignore_src")?)
    } else {
        None
    };
    if let Some(ignore) = &ignore_source {
        let dtype = dtype_of(ignore)?;
        if !(is_integer_dtype(&dtype) || dtype == "bool") {
            return unsupported(format!(
                "{op_name}.ignore_src must be bool or integer, got {:?}",
                &dtype
            ));
        }
    }
    let cache_policy = decoded.optional_scalar_operand("cache_policy")?;
    let has_cache = decoded.modifier("cache")? == "L2::cache_hint";
    decoded.require_cache_policy(cache_policy.as_ref(), has_cache)?;
    let predicate = decoded.predicate.clone();
    if let Some(predicate) = &predicate {
        let dtype = dtype_of(predicate)?;
        if !(is_integer_dtype(&dtype) || dtype == "bool") {
            return unsupported(format!(
                "{op_name} predicate must be bool or integer, got {:?}",
                &dtype
            ));
        }
    }
    let (destination, _) = shared_pointer_source(
        &destination,
        &format!("{op_name}.dst_mem"),
        SharedAddressForms::MEMORY,
    )?;
    Ok(CpAsyncParts {
        destination,
        source,
        byte_count,
        source_size,
        ignore_source,
        cache_policy,
        predicate,
    })
}

/// The validated parts.

pub struct StmatrixParts {
    pub shape: String,
    pub count: i64,
    pub transpose: bool,
    pub dtype: String,
    pub space: String,
    pub destination: ObjectRef,
    pub sources: Vec<ObjectRef>,
}

pub fn stmatrix_parts(decoded: &DecodedPtx) -> AResult<StmatrixParts> {
    let op_name = decoded.op_name.as_str();
    require_register_call(decoded, false)?;
    for (modifier, expected) in [("sync", "sync"), ("aligned", "aligned")] {
        let actual = decoded.modifier(modifier)?;
        if actual != expected {
            return unsupported(format!(
                "{op_name} requires {modifier}={:?}, got {:?}",
                expected, actual
            ));
        }
    }
    let shape = decoded.modifier("shape")?.to_owned();
    let dtype = decoded.modifier("type")?.to_owned();
    let transpose_token = decoded.modifier("trans")?.to_owned();
    if transpose_token != "" && transpose_token != "trans" {
        return unsupported(format!(
            "{op_name} has unsupported trans modifier {:?}",
            &transpose_token
        ));
    }
    let transpose = transpose_token == "trans";
    if op_name == "tirx.ptx.stmatrix" {
        if (shape.as_str(), dtype.as_str()) != ("m8n8", "b16") {
            return unsupported(format!(
                "{op_name} requires shape/type m8n8.b16, got {shape}.{dtype}"
            ));
        }
    } else if (shape.as_str(), dtype.as_str(), transpose) != ("m16n8", "b8", true) {
        return unsupported(format!(
            "{op_name} requires shape/type/trans m16n8.b8.trans, got {shape}.{dtype} trans={}",
            if transpose { "True" } else { "False" }
        ));
    }
    let num = decoded.modifier("num")?.to_owned();
    let count = match num.as_str() {
        "x1" => 1,
        "x2" => 2,
        "x4" => 4,
        _ => return unsupported(format!("{op_name} has unsupported num modifier {:?}", &num)),
    };
    let space = decoded.modifier("space")?.to_owned();
    if !["", "shared", "shared::cta"].contains(&space.as_str()) {
        return unsupported(format!(
            "{op_name} state space must be generic, shared or shared::cta, got {:?}",
            &space
        ));
    }
    let destination = decoded.scalar_operand("p")?;
    if dtype_of(&destination)? != "uint32"
        && expr_type(&destination).is_none_or(|ty| !is_pointer_type(&ty))
    {
        return unsupported(format!(
            "{op_name}.p must retain a physical pointer or uint32 shared address"
        ));
    }
    let lanes = decoded.operand("r")?;
    let mut sources = Vec::new();
    for lane in lanes {
        match lane {
            Some(value) => sources.push(value.clone()),
            None => {
                return Err(crate::analyze::util::Failure::Ffi(
                    crate::analyze::util::ffi_error("sunk lane in a stmatrix source"),
                ))
            }
        }
    }
    Ok(StmatrixParts {
        shape,
        count,
        transpose,
        dtype,
        space,
        destination,
        sources,
    })
}

/// The state space of one `stmatrix` destination.
pub fn stmatrix_space(parts: &StmatrixParts) -> &'static str {
    if parts.space == "shared::cta" {
        "v2::SharedCta"
    } else {
        "v2::Shared"
    }
}

pub fn stmatrix_variant(parts: &StmatrixParts) -> String {
    let rust_space = stmatrix_space(parts);
    if (parts.shape.as_str(), parts.dtype.as_str()) == ("m8n8", "b16") {
        return format!(
            "v2::mem::variant::StmatrixM8N8B16<{rust_space}, {}, {}>",
            parts.count, parts.transpose
        );
    }
    format!(
        "v2::mem::variant::StmatrixM16N8B8<{rust_space}, {}>",
        parts.count
    )
}

/// The validated parts.

impl<'a> Emitter<'a> {
    fn emit_access_ptr(&mut self, parts: &AccessPtrParts) -> AResult<RustValue> {
        let address = &parts.address;
        let mut base = if as_var(address).is_some() && self.variables.contains(address) {
            self.emit_expr(address)?
        } else if as_var(address).is_some() {
            self.emit_buffer_data_pointer(address)?
        } else {
            self.emit_expr(address)?
        };
        if base.rust_type != "PhysicalPtr" {
            base = self.emit_raw_generic_pointer(base, "ctx.active_mask()")?;
        }
        if base.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{} base must resolve to a physical address",
                "tirx.tvm_access_ptr"
            ));
        }
        let offset = self.emit_expr(&parts.element_offset)?;
        let offset = self.as_i64(offset)?;
        let offset = self.as_warp_value(offset);
        let extent = self.emit_expr(&parts.element_extent)?;
        let extent = self.as_i64(extent)?;
        let extent = self.as_warp_value(extent);
        let result = self.temp("access_ptr");
        self.emit_line(&format!(
            "let {result} = physical_ptr_access_view(&{}, &{}, &{}, {}_usize, ctx.active_mask(), {}_u8, {})?;",
            base.code,
            offset.code,
            extent.code,
            dtype_byte_len(self.ctx.schema, &parts.target_dtype)?,
            parts.access_mask,
            json_string("tirx.tvm_access_ptr")
        ));
        Ok(RustValue::new(
            result,
            "PhysicalPtr",
            join_uniformity([base.uniformity, offset.uniformity, extent.uniformity]),
        ))
    }

    fn emit_cuda_ldg(&mut self, expr: &ObjectRef, ldg: &CudaLdg) -> AResult<RustValue> {
        let (result_dtype, address) = (&ldg.result_dtype, &ldg.address);
        let mut pointer = self.emit_expr(&address)?;
        if pointer.rust_type != "PhysicalPtr" {
            pointer = self.emit_raw_generic_pointer(pointer, "ctx.active_mask()")?;
        }
        if pointer.rust_type != "PhysicalPtr" {
            return unsupported(format!(
                "{} address must resolve to a physical address",
                "tirx.cuda.ldg"
            ));
        }
        let (variant, storage_type, decoder) = cuda_ldg_variant(self.ctx, &result_dtype)?;
        let raw_result = self.temp("cuda_ldg_raw");
        let site = self.lowered_instruction_site(expr);
        let abi_call = abi::warp_call(
            MEM_LD,
            &abi::site(site as u64),
            &[abi::address(
                "v2::Global",
                &abi::cloned(&pointer.code),
                None,
            )],
            Some(&variant),
            None,
            false,
            true,
        );
        self.emit_line(&format!("let {raw_result} = {abi_call};"));
        self.emit_line(&format!(
            "let {raw_result} = v2_register_out({raw_result});"
        ));
        let Some(decoder) = decoder else {
            return Ok(RustValue::new(
                raw_result,
                storage_type,
                Uniformity::Varying,
            ));
        };
        let result = self.temp("cuda_ldg");
        self.emit_line(&format!(
            "let {result} = WarpValue::from_fn(|lane| {decoder}({raw_result}[lane]));"
        ));
        Ok(RustValue {
            quantized_dtype: Some(result_dtype.clone()),
            ..RustValue::new(result, "f32", Uniformity::Varying)
        })
    }

    fn cp_async_predicate_mask(&mut self, predicate: Option<&ObjectRef>) -> AResult<String> {
        self.instruction_predicate_mask(
            predicate,
            "cp_async_predicate_mask",
            "cp.async predicate must lower to bool or integer",
        )
    }

    /// `emit_ptx_cp_async`.
    fn emit_ptx_cp_async(&mut self, parts: &CpAsyncParts, source_op_id: i64) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            parts.predicate.as_ref(),
            "cp_async",
            "cp.async predicate must lower to bool or integer",
        )?;
        self.emit_cp_async_body(parts, source_op_id)?;
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_cp_async_body(&mut self, parts: &CpAsyncParts, source_op_id: i64) -> AResult<()> {
        let destination =
            self.emit_raw_shared_pointer(&parts.destination, None, "ctx.active_mask()")?;
        if let Some(cache_policy) = &parts.cache_policy {
            self.emit_expr(cache_policy)?;
        }
        let mut source_access_mask = "ctx.active_mask()".to_owned();
        let mut fill: Option<String> = None;
        let variant;
        if let Some(source_size) = &parts.source_size {
            let sizes = self.emit_expr(source_size)?;
            let sizes = self.as_warp_value(sizes);
            fill = Some(format!(
                "v2_register({}).map(|_, size| size as u32)",
                sizes.code
            ));
            source_access_mask = self.control_name("cp_async_source_access_mask");
            self.emit_line(&format!(
                "let {source_access_mask} = ctx.active_mask() & {}.to_mask(|_, size| *size != 0);",
                sizes.code
            ));
            variant = cp_async_variant(parts.byte_count, "size");
        } else if parts.ignore_source.is_some() {
            let ignore_mask = self.cp_async_predicate_mask(parts.ignore_source.as_ref())?;
            let copy = self.control_name("cp_async_source_not_ignored");
            self.emit_line(&format!(
                "let {copy} = WarpValue::from_fn(|lane| !{ignore_mask}.contains(lane));"
            ));
            source_access_mask = self.control_name("cp_async_source_access_mask");
            self.emit_line(&format!(
                "let {source_access_mask} = ctx.active_mask() - {ignore_mask};"
            ));
            fill = Some(format!("v2_register({copy})"));
            variant = cp_async_variant(parts.byte_count, "zero");
        } else {
            variant = cp_async_variant(parts.byte_count, "");
        }
        let source = self.emit_expr(&parts.source)?;
        let source = self.emit_raw_generic_pointer(source, &source_access_mask)?;
        let arguments = format!(
            "({}, {}{})",
            abi::address("v2::Shared", &abi::cloned(&destination.code), None),
            abi::address("v2::Global", &abi::cloned(&source.code), None),
            fill.map_or_else(String::new, |fill| format!(", {fill}"))
        );
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            CP_ASYNC,
            &site,
            &[arguments],
            Some(&variant),
            None,
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        Ok(())
    }

    /// `emit_ptx_stmatrix`.
    fn emit_ptx_stmatrix(
        &mut self,
        decoded: &DecodedPtx,
        parts: &StmatrixParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "stmatrix",
            "stmatrix predicate must be bool or integer",
        )?;
        let destination = match parts.destination.as_node::<CallObj>() {
            Some(call)
                if call_op_name(call)?.as_deref() == Some("tirx.cuda.cvta_generic_to_shared")
                    && call.args.len() == 1 =>
            {
                oref(call.args.get(0)?)
            }
            _ => parts.destination.clone(),
        };
        let destination_pointer = self.emit_address_pointer(
            &destination,
            &parts.space,
            None,
            &format!(
                "ctx.active_mask() & WarpMask::from_bits({}_u32)",
                (1i64 << (parts.count * 8)) - 1
            ),
        )?;
        let mut source_values = Vec::new();
        for (index, source) in parts.sources.iter().enumerate() {
            let value = self.emit_as_unsigned_bits(
                source,
                32,
                &op_name,
                &format!("stmatrix_source_{index}"),
                None,
            )?;
            source_values.push(value.code);
        }
        let registers = self.control_name("stmatrix_registers");
        let register_fields: Vec<String> = source_values
            .iter()
            .map(|value| format!("{value}[lane]"))
            .collect();
        self.emit_line(&format!(
            "let {registers} = WarpValue::from_fn(|lane| [{}]);",
            register_fields.join(", ")
        ));
        let rust_space = stmatrix_space(parts);
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            STMATRIX,
            &site,
            &[format!(
                "({}, v2_register({registers}))",
                abi::address(rust_space, &abi::cloned(&destination_pointer.code), None)
            )],
            Some(&stmatrix_variant(&parts)),
            region.context.as_deref(),
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        self.close_predicated_region(region);
        Ok(())
    }
}

pub fn emit_stmatrix(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = stmatrix_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_stmatrix(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_access_ptr(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = access_ptr_parts(emitter.ctx, call.node)?;
    emitter
        .with_call_expr(call.node, |emitter| emitter.emit_access_ptr(&parts))
        .map(Some)
}

pub fn emit_cuda_ldg(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let (result_dtype, address) = cuda_ldg_parts(emitter.ctx, call.node)?;
    cuda_ldg_variant(emitter.ctx, &result_dtype)?;
    let parts = CudaLdg {
        result_dtype,
        address,
    };
    emitter
        .with_call_expr(call.node, |emitter| {
            emitter.emit_cuda_ldg(call.node, &parts)
        })
        .map(Some)
}

pub fn emit_cp_async(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = cp_async_parts(emitter.ctx, call.table()?)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_cp_async(&parts, source_op_id)?;
    Ok(None)
}
