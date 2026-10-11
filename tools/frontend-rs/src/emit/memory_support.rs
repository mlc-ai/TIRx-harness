//! Emitter services shared by the memory families: the explicit register
//! destination store, pointer recovery helpers, the PTX instruction-predicate
//! regions and the memory carrier spellings that take a PTX type.

use tvm::analysis::Analyzer;
use tvm::ir::{CallObj, PrimExpr, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

use super::super::analyze::buffers::call_op_name;
use super::super::analyze::memory::MemorySpace;
use super::super::analyze::util::{
    as_buffer, as_var, buffer_dtype, dtype_of, int_imm_expr, oref, prim, repr_text, simplify,
    unsupported, AResult,
};
use super::abi;
use super::{ControlProvenance, Emitter, RustValue, Uniformity};
use crate::tables::{
    dtype_itemsize, expr_rust_type, is_integer_rust_type, stmt_rust_scalar_by_dtype,
    v2_memory_type_rust, vector_rust_type,
};

#[derive(Default)]
pub struct CandidateState {
    pub shared: bool,
    pub generic: bool,
}

pub fn v2_memory_type_rust_ptx(
    schema: &crate::schema::Schema,
    dtype: &str,
    ptx_type: Option<&str>,
) -> AResult<String> {
    let carrier = match (ptx_type.unwrap_or(""), dtype) {
        ("b8", "uint32")
        | ("u8", "uint32")
        | ("b8", "uint8")
        | ("b8", "uint16")
        | ("u8", "uint8") => Some("v2::mem::variant::U8AsU32"),
        ("s8", "int32") | ("b8", "int8") | ("s8", "int8") => Some("v2::mem::variant::S8AsI32"),
        ("b16", "uint32") | ("u16", "uint32") => Some("v2::mem::variant::U16AsU32"),
        ("s16", "int32") => Some("v2::mem::variant::S16AsI32"),
        _ => None,
    };
    if let Some(carrier) = carrier {
        return Ok(carrier.to_owned());
    }
    v2_memory_type_rust(schema, dtype)
}

pub fn v2_memory_space_rust(space: &str) -> AResult<&'static str> {
    Ok(match space {
        "" => "v2::Generic",
        "global" => "v2::Global",
        "shared" => "v2::Shared",
        "shared::cta" => "v2::SharedCta",
        "shared::cluster" => "v2::SharedCluster",
        "local" => "v2::Local",
        other => return unsupported(format!("no v2 memory state-space marker for {:?}", other)),
    })
}

pub fn stmt_rust_scalar_type(schema: &crate::schema::Schema, dtype: &str) -> Option<String> {
    stmt_rust_scalar_by_dtype(dtype)
        .map(|native| crate::tables::precision_scalar_type(schema, dtype, native).to_owned())
        .or_else(|| vector_rust_type(schema, dtype))
}

/// An open predicated instruction region.
pub struct PredicatedRegion {
    /// `v2_context(<context>)` inside the region; `None` without a predicate.
    pub context: Option<String>,
    /// The instruction mask (`ctx.active_mask()` without a predicate).
    pub mask: String,
    open: bool,
}

impl<'a> Emitter<'a> {
    /// Reuse the resolved physical view at the point its store is lowered.
    pub fn record_global_write(&mut self, buffer: &tvm::tirx::BufferVar) -> AResult<()> {
        if self.written_global_buffers.is_none() {
            return Ok(());
        }
        let code = self.buffer_code(buffer)?;
        let plan = self.plan_of(code);
        if plan.space != MemorySpace::Global {
            return Ok(());
        }
        match code.storage_index {
            Some(index) => {
                self.written_global_buffers.as_mut().unwrap().insert(index);
            }
            None => {
                let data = oref(plan.dynamic_data_var.as_ref().expect("dynamic view").clone());
                self.record_pointer_write(&data, "global")?;
            }
        }
        Ok(())
    }

    /// Record the may-target set before execution, including reads preceding
    /// the first write and host aliases of a syntactically read-only buffer.
    pub fn record_pointer_write(&mut self, address: &ObjectRef, space: &str) -> AResult<()> {
        use crate::analyze::pointer_targets::{PointerTargets, Target};
        if self.written_global_buffers.is_none() || !matches!(space, "" | "generic" | "global") {
            return Ok(());
        }
        if self.pointer_targets.is_none() {
            self.pointer_targets = Some(PointerTargets::new(self.ctx, self.plan)?);
        }
        let Some(targets) = self.pointer_targets.as_ref().unwrap().resolve(address)? else {
            self.written_global_buffers = None;
            return Ok(());
        };
        for target in targets {
            match target {
                Target::Backing(backing) => {
                    if self.memory_plan.backings[backing].space != MemorySpace::Global {
                        continue;
                    }
                    let slot = self.buffers.iter().find_map(|(_, code)| {
                        (self.plan_of(code).backing_index == Some(backing))
                            .then_some(code.storage_index).flatten()
                    });
                    if let Some(slot) = slot {
                        self.written_global_buffers.as_mut().unwrap().insert(slot);
                    } else {
                        self.written_global_buffers = None;
                        return Ok(());
                    }
                }
                Target::Pointer(index) => {
                    self.written_global_parameters.insert(index);
                }
            }
        }
        Ok(())
    }

    /// `ops.predicate.instruction_predicate_mask`.
    pub fn instruction_predicate_mask(
        &mut self,
        predicate: Option<&ObjectRef>,
        prefix: &str,
        invalid_message: &str,
    ) -> AResult<String> {
        let Some(predicate) = predicate else {
            return Ok("ctx.active_mask()".to_owned());
        };
        let value = self.emit_expr(predicate)?;
        if !(is_integer_rust_type(&value.rust_type) || value.rust_type == "bool") {
            return unsupported(invalid_message);
        }
        let name = self.control_name(prefix);
        if value.uniformity == Uniformity::Uniform {
            let condition = if value.rust_type == "bool" {
                value.code.clone()
            } else {
                format!("{} != 0", value.code)
            };
            self.emit_line(&format!(
                "let {name} = if {condition} {{ ctx.active_mask() }} else {{ WarpMask::EMPTY }};"
            ));
            return Ok(name);
        }
        if value.rust_type == "bool" {
            let value = if value.is_mask {
                value
            } else {
                let lane = self.boolean_lane(&value)?;
                self.emit_varying_mask(&lane, ControlProvenance::None)
            };
            self.emit_line(&format!("let {name} = {} & ctx.active_mask();", value.code));
        } else {
            let lanes = self.as_warp_value(value);
            self.emit_line(&format!(
                "let {name} = {}.to_mask(|_, value| *value != 0) & ctx.active_mask();",
                lanes.code
            ));
        }
        Ok(name)
    }

    /// The region
    /// shadows `ctx`, so the predicate also gates operand evaluation.
    pub fn open_shadow_predicated_region(
        &mut self,
        predicate: Option<&ObjectRef>,
        prefix: &str,
        invalid_message: &str,
    ) -> AResult<PredicatedRegion> {
        if predicate.is_none() {
            return Ok(PredicatedRegion {
                context: None,
                mask: "ctx.active_mask()".to_owned(),
                open: false,
            });
        }
        let mask =
            self.instruction_predicate_mask(predicate, &format!("{prefix}_mask"), invalid_message)?;
        self.emit_line(&format!("if !{mask}.is_empty() {{"));
        self.indent += 1;
        self.emit_line(&format!("let mut ctx = ctx.with_active_mask({mask});"));
        Ok(PredicatedRegion {
            context: Some(abi::context("ctx")),
            mask,
            open: true,
        })
    }

    /// `finish_predicated_destinations`: finish inactive outputs using TVM's
    /// register-carrier contract.
    pub fn finish_predicated_destinations(
        &mut self,
        decoded: &crate::decode::ptx::DecodedPtx,
        destinations: &[Option<ObjectRef>],
        result_dtype: &str,
        mask: &str,
        source_op_id: i64,
        predicate_output: bool,
    ) -> AResult<()> {
        if decoded.predicate.is_none() || (decoded.preserve_dst && !predicate_output) {
            return Ok(());
        }
        // Zero is a simulator representative, not a promised inactive GPU value.
        let rust_type = expr_rust_type(self.ctx.schema, result_dtype)?;
        let inactive = self.control_name("instruction_inactive");
        self.emit_line(&format!("let {inactive} = ctx.active_mask() - {mask};"));
        if decoded.preserve_dst {
            // TVM's .pred bridge converts the incoming carrier with setp.ne and
            // writes back through selp.b32, even on non-issuing lanes. Preserve
            // the logical predicate, not arbitrary nonzero carrier bits.
            self.emit_line(&format!("if !{inactive}.is_empty() {{"));
            self.indent += 1;
            self.emit_line(&format!("let mut ctx = ctx.with_active_mask({inactive});"));
            for destination in destinations.iter().flatten() {
                let value = self.emit_expr(destination)?;
                let value = self.coerce_value(value, "bool", "kept_predicate")?;
                let value = self.coerce_dtype(value, result_dtype, "kept_predicate_carrier")?;
                self.emit_explicit_buffer_store(
                    destination,
                    value,
                    source_op_id,
                    None,
                    None,
                    None,
                )?;
            }
            self.indent -= 1;
            self.emit_line("}");
            return Ok(());
        }
        let zero = self.control_name("instruction_undefined_zero");
        self.emit_line(&format!(
            "let {zero} = WarpValue::splat({});",
            Self::zero_literal(&rust_type)?
        ));
        for destination in destinations.iter().flatten() {
            self.emit_explicit_buffer_store(
                destination,
                RustValue::new(zero.clone(), rust_type.clone(), Uniformity::Varying),
                source_op_id,
                None,
                Some(&inactive),
                None,
            )?;
        }
        Ok(())
    }

    /// Close a predicated instruction region.
    pub fn close_predicated_region(&mut self, region: PredicatedRegion) {
        if region.open {
            self.indent -= 1;
            self.emit_line("}");
        }
    }

    pub fn emit_pointer_handle(&mut self, value: &ObjectRef) -> AResult<RustValue> {
        let emitted = if as_var(value).is_some() && !self.variables.contains(value) {
            self.emit_buffer_data_pointer(value)?
        } else {
            self.emit_expr(value)?
        };
        self.emit_raw_generic_pointer(emitted, "ctx.active_mask()")
    }

    pub fn emit_address_pointer(
        &mut self,
        address: &ObjectRef,
        space: &str,
        value: Option<RustValue>,
        active_mask: &str,
    ) -> AResult<RustValue> {
        let value = match value {
            Some(value) => value,
            None => self.emit_expr(address)?,
        };
        if value.rust_type == "PhysicalPtr" {
            return Ok(value);
        }
        if space.starts_with("shared") {
            return self.emit_raw_shared_pointer(address, Some(value), active_mask);
        }
        if space == "local" {
            let bits = self.as_warp_value(value);
            return Ok(RustValue::new(
                format!(
                    "PhysicalPtr::integer(({}).clone().map(|_, value| value as u64))",
                    bits.code
                ),
                "PhysicalPtr",
                Uniformity::Varying,
            ));
        }
        self.emit_raw_generic_pointer(value, active_mask)
    }

    /// A resolved shared-memory pointer.
    pub fn shared_pointer(&mut self, expr: &ObjectRef, label: &str) -> AResult<RustValue> {
        let mut value = self.emit_expr(expr)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_shared_pointer(expr, Some(value), "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} must be a physical SMEM pointer"));
        }
        Ok(value)
    }

    pub fn emit_explicit_buffer_store(
        &mut self,
        destination: &ObjectRef,
        value: RustValue,
        source_op_id: i64,
        predicate: Option<&ObjectRef>,
        instruction_mask: Option<&str>,
        storage_dtype_override: Option<&str>,
    ) -> AResult<()> {
        if predicate.is_some() && instruction_mask.is_some() {
            return Err(super::super::analyze::util::Failure::Ffi(
                super::super::analyze::util::ffi_error(
                    "explicit store requires one predicate authority",
                ),
            ));
        }
        let Some(load) = destination.as_node::<TensorLoadObj>() else {
            return unsupported(
                "explicit instruction destination must be a concrete TensorLoad lvalue",
            );
        };
        let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
            return super::super::analyze::util::not_covered(
                "explicit destination source is not a typed buffer",
            );
        };
        self.record_global_write(&buffer)?;
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let code = self.buffer_code(&buffer)?;
        let plan = self.plan_of(&code);
        let plan_name = plan.name.clone();
        let plan_space = plan.space;
        let plan_itemsize = plan.layout.itemsize;
        let plan_dynamic = plan.dynamic_data_var.is_some();
        let dtype = buffer_dtype(&buffer);
        if plan_space == MemorySpace::Tmem {
            return unsupported(format!(
                "buffer:{plan_name}:explicit instruction destinations do not support TMEM"
            ));
        }
        if self.ctx.schema.vector_dtype_abi(&dtype).is_some() {
            return unsupported(format!(
                "buffer:{plan_name}:explicit instruction destination must be scalar"
            ));
        }
        let storage_dtype = storage_dtype_override.unwrap_or(dtype.as_str()).to_owned();
        if let Some(override_dtype) = storage_dtype_override {
            if !matches!(
                (dtype.as_str(), override_dtype),
                ("float16", "uint16") | ("bfloat16", "uint16")
            ) {
                return unsupported(format!(
                    "buffer:{plan_name}:raw storage override {dtype}->{override_dtype} is not implemented"
                ));
            }
            let storage_size = dtype_itemsize(self.ctx.schema, override_dtype);
            if storage_size != Some(plan_itemsize) {
                return unsupported(format!(
                    "buffer:{plan_name}:raw storage override {override_dtype} has size {}, expected {plan_itemsize}",
                    storage_size.map_or("None".to_owned(), |size| size.to_string())
                ));
            }
        }
        let Some(rust_type) = stmt_rust_scalar_type(self.ctx.schema, &storage_dtype) else {
            return unsupported(format!(
                "buffer:{plan_name}:explicit store is not implemented for {storage_dtype}"
            ));
        };
        let stored = if storage_dtype_override.is_some() {
            value
        } else {
            self.coerce_dtype(value, &dtype, "explicit_store_cast")?
        };
        let stored = self.as_warp_value(stored);
        if stored.rust_type != rust_type {
            return unsupported(format!(
                "buffer:{plan_name}:explicit store produced {}, expected {rust_type}",
                stored.rust_type
            ));
        }
        if matches!(
            dtype.as_str(),
            "float8_e4m3fn" | "float8_e8m0fnu" | "float4_e2m1fn"
        ) {
            return unsupported(format!(
                "buffer:{plan_name}:explicit store does not support packed {dtype}"
            ));
        }
        let space = if plan_dynamic {
            "v2::Generic"
        } else {
            match plan_space {
                MemorySpace::Global => "v2::Global",
                MemorySpace::Shared => "v2::Shared",
                MemorySpace::Local | MemorySpace::Register => "v2::Local",
                MemorySpace::Tmem => unreachable!("rejected above"),
            }
        };
        let index = self.physical_index(&buffer, &indices)?;
        let mut access_mask = self.physical_access_mask(&buffer, &indices, "ctx.active_mask()")?;
        if predicate.is_some() {
            let predicate_mask = self.instruction_predicate_mask(
                predicate,
                "explicit_store_predicate",
                "explicit instruction predicate must lower to bool or integer",
            )?;
            access_mask = format!("({access_mask}) & {predicate_mask}");
        }
        if let Some(instruction_mask) = instruction_mask {
            access_mask = format!("({access_mask}) & {instruction_mask}");
        }
        let buffer_ref = self.buffer_ref(&buffer)?;
        let logical_name = self.logical_buffer_name(&buffer)?;
        let address = abi::buffer_address(
            space,
            &buffer_ref,
            &format!("({})", index.code),
            plan_itemsize,
            &logical_name,
        );
        let memory_dtype = if storage_dtype == "uint128" || storage_dtype == "int128" {
            "uint64x2".to_owned()
        } else {
            storage_dtype.clone()
        };
        let marker = v2_memory_type_rust(self.ctx.schema, &memory_dtype)?;
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            "mem::st",
            &site,
            &[format!(
                "({address}, v2_register(({}).clone()))",
                stored.code
            )],
            Some(&format!("v2::mem::variant::St<{marker}, {space}>")),
            Some(&abi::context(&format!(
                "ctx.with_active_mask({access_mask})"
            ))),
            false,
            false,
        );
        let invocation = match shared_local_rust_type(&memory_dtype).filter(|_| self.use_typed_helpers && space == "v2::Local") {
            Some(helper_type) => {
                let site = self.v2_site(Some(source_op_id));
                local_memory_call(
                    "st",
                    helper_type,
                    &format!("ctx.with_active_mask({access_mask})"),
                    &site,
                    &buffer_ref,
                    &index.code,
                    plan_itemsize,
                    &logical_name,
                    Some(&stored.code),
                )
            }
            None => invocation,
        };
        self.emit_write_call(&invocation);
        Ok(())
    }
}

/// The dtypes with a local-memory helper.
pub fn shared_local_rust_type(dtype: &str) -> Option<&'static str> {
    match dtype {
        "int32" => Some("i32"),
        "float32" => Some("f32"),
        "uint64" => Some("u64"),
        "uint32" => Some("u32"),
        "int64" => Some("i64"),
        "uint16" => Some("u16"),
        _ => None,
    }
}

/// `memory_helpers.local_memory_call`.
#[allow(clippy::too_many_arguments)]
pub fn local_memory_call(
    operation: &str,
    rust_type: &str,
    context: &str,
    site: &str,
    buffer: &str,
    index: &str,
    itemsize: i64,
    label: &str,
    value: Option<&str>,
) -> String {
    let mut arguments = vec![
        "warp".to_owned(),
        context.to_owned(),
        site.to_owned(),
        format!("&({buffer})"),
        format!("&({index})"),
        format!("{itemsize}_usize"),
        crate::tables::json_string(label),
    ];
    if let Some(value) = value {
        arguments.push(format!("&({value})"));
    }
    format!("numsim_local_{operation}_{rust_type}({})", arguments.join(", "))
}

/// The spellings of a shared-memory address operand one instruction family
/// accepts.
#[derive(Clone, Copy)]
pub struct SharedAddressForms<'a> {
    /// Integer dtypes accepted unchanged as a raw shared address; `None`
    /// accepts any expression unchanged.
    pub integers: Option<&'a [&'a str]>,
    /// Unwrap `cvta_generic_to_shared` of a pointer.
    pub cvta: bool,
    /// Unwrap `sm100_2sm_leader_smem_addr` of a pointer, the SM100 two-SM
    /// pair base.
    pub leader: bool,
    /// Unwrap the pair base `cvta_generic_to_shared(pointer) & 0xFEFF_FFFF`,
    /// simplifying the mask with this analyzer.
    pub pair_mask: Option<&'a Analyzer>,
    /// Accept a pair-base address.
    pub allow_pair_base: bool,
}

impl SharedAddressForms<'static> {
    /// Memory and bulk operands: any address, unwrapping
    /// `cvta_generic_to_shared`.
    pub const MEMORY: Self = Self {
        integers: None,
        cvta: true,
        leader: false,
        pair_mask: None,
        allow_pair_base: false,
    };
    /// Synchronization operands: a pointer or a 32/64-bit shared address, as
    /// written.
    pub const SYNC: Self = Self {
        integers: Some(&["uint32", "uint64"]),
        cvta: false,
        leader: false,
        pair_mask: None,
        allow_pair_base: false,
    };
    /// TCGEN05 operands: a pointer, one of its public shared conversions, or a
    /// 32-bit shared address.
    pub const TCGEN: Self = Self {
        integers: Some(&["uint32"]),
        cvta: true,
        leader: true,
        pair_mask: None,
        allow_pair_base: true,
    };
}

impl<'a> SharedAddressForms<'a> {
    /// TMA operands: a pointer, one of its public shared conversions including
    /// the masked pair base, or (with `raw_address`) a 32-bit shared address.
    pub fn tma(analyzer: &'a Analyzer, raw_address: bool, allow_pair_base: bool) -> Self {
        Self {
            integers: Some(if raw_address { &["uint32"] } else { &[] }),
            cvta: true,
            leader: true,
            pair_mask: Some(analyzer),
            allow_pair_base,
        }
    }
}

/// The source a shared-memory address operand retains under `forms`, and
/// whether it is the SM100 two-SM pair base.
pub fn shared_pointer_source(
    expression: &ObjectRef,
    field: &str,
    forms: SharedAddressForms,
) -> AResult<(ObjectRef, bool)> {
    let dtype = dtype_of(expression)?;
    if dtype == "handle" {
        return Ok((expression.clone(), false));
    }
    if let Some((source, pair_base)) = shared_conversion(expression, &forms)? {
        if pair_base && !forms.allow_pair_base {
            return unsupported(format!(
                "{field} cannot use an SM100 two-SM mbarrier address"
            ));
        }
        return Ok((source, pair_base));
    }
    match forms.integers {
        Some(integers) if !integers.contains(&dtype.as_str()) => unsupported(format!(
            "{field} must retain a concrete shared-memory source pointer, got {}",
            repr_text(expression)?
        )),
        _ => Ok((expression.clone(), false)),
    }
}

/// `(source, pair_base)` of a public shared-address conversion `forms`
/// unwraps.
fn shared_conversion(
    expression: &ObjectRef,
    forms: &SharedAddressForms,
) -> AResult<Option<(ObjectRef, bool)>> {
    if let Some(call) = expression.as_node::<CallObj>() {
        let op_name = call_op_name(call)?.unwrap_or_default();
        let arguments: Vec<ObjectRef> = call.args.iter().map(oref).collect();
        let leader = op_name == "tirx.cuda.sm100_2sm_leader_smem_addr";
        if (forms.cvta && op_name == "tirx.cuda.cvta_generic_to_shared") || (forms.leader && leader)
        {
            if arguments.len() == 1 && dtype_of(&arguments[0])? == "handle" {
                return Ok(Some((arguments[0].clone(), leader)));
            }
        }
    }
    let Some(analyzer) = forms.pair_mask else {
        return Ok(None);
    };
    let Some(("bitwise_and", arguments)) = crate::analyze::util::bitwise_expr(expression) else {
        return Ok(None);
    };
    for (converted, mask) in [
        (&arguments[0], &arguments[1]),
        (&arguments[1], &arguments[0]),
    ] {
        let simplified_mask = simplify(analyzer, &prim(mask)?)?;
        if int_imm_expr(&simplified_mask) != Some(0xFEFF_FFFF) {
            continue;
        }
        if let Some((source, false)) = shared_conversion(converted, forms)? {
            return Ok(Some((source, true)));
        }
    }
    Ok(None)
}

/// Concrete local-memory call boundaries for ordinary numerical artifacts.
pub(super) fn local_memory_helpers() -> String {
    let mut declarations = vec![
        "#[inline(never)]\nfn numsim_decode_shared_address(\n    pointer: &WarpValue<u64>, physical: &PhysicalMemory, ctx: WarpContext,\n) -> Result<WarpValue<u32>, EngineError> {\n    let result = WarpValue::from_fn(|lane| decode_generic_shared_address(pointer[lane]).unwrap_or(u32::MAX));\nif ctx.active_mask().into_iter().any(|lane| result[lane] == u32::MAX) { return Err(EngineError::message(\"cvta.to.shared requires a generic shared address\")); }\n    Ok(result)\n}".to_owned(),
    ];
    for (rust_type, marker) in [
        ("f32", "F32"),
        ("i32", "I32"),
        ("i64", "I64"),
        ("u16", "U16"),
        ("u32", "U32"),
        ("u64", "U64"),
    ] {
        let parameters = "warp: &mut NumSimWarpEngine, ctx: WarpContext, site: v2::SiteId,\n    buffer: &RuntimeBuffer, index: &WarpValue<i64>, itemsize: usize, label: &'static str";
        let address = "v2_buffer_address::<v2::Local>(buffer, index, itemsize, label)";
        let load = abi::warp_call(
            "mem::ld",
            "site",
            &[address.to_owned()],
            Some(&format!(
                "v2::mem::variant::Ld<v2::reg::variant::{marker}, v2::Local>"
            )),
            None,
            false,
            true,
        );
        let store = abi::warp_call(
            "mem::st",
            "site",
            &[format!("({address}, v2_register(value.clone()))")],
            Some(&format!(
                "v2::mem::variant::St<v2::reg::variant::{marker}, v2::Local>"
            )),
            None,
            false,
            true,
        );
        declarations.push(format!(
            "#[inline(never)]\nfn numsim_local_ld_{rust_type}({parameters}) -> Result<WarpValue<{rust_type}>, EngineError> {{\n    Ok(v2_register_out({load}))\n}}\n#[inline(never)]\nfn numsim_local_st_{rust_type}({parameters}, value: &WarpValue<{rust_type}>) -> Result<(), EngineError> {{\n    {store};\n    Ok(())\n}}"
        ));
    }
    declarations.join("\n\n")
}
