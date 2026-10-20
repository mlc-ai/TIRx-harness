//! Statement and buffer-access emission.

use tvm::ir::{IntImmObj, PointerTypeObj, PrimExpr};
use tvm::prim::RampObj;
use tvm::tirx::{
    AssertStmtObj, AttrStmtObj, BindObj, BreakObj, BufferStoreObj, BufferVar, ContinueObj,
    DeclBufferObj, EvaluateObj, ForObj, IfThenElseObj, ReturnObj, ScopeIdDefStmtObj, SeqStmtObj,
    Stmt, TilePrimitiveCallObj, WhileObj,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore};

use super::super::analyze::memory::{BufferPlan, MemorySpace};
use super::super::analyze::topology::iter_var_of;
use super::super::analyze::util::{
    buffer_dtype, buffer_name, dtype_text, expr_type, ffi_text, int_imm, kind_or_bail, not_covered,
    oref, prim_dtype, repr_text, same, unsupported, AResult, Failure,
};
use super::super::analyze::vector::classify_contiguous_ramp;
use super::abi;
use super::{ControlProvenance, Emitter, RustValue, Uniformity};
use crate::decode::projected_buffer;
use crate::tables::{
    dtype_byte_len, is_integer_rust_type, json_string, stmt_rust_scalar_by_dtype,
    v2_memory_type_rust, vector_rust_type,
};

enum WarpLoadBackend {
    RuntimeScalar,
    RuntimeFloat4,
    Tmem,
}

struct WarpLoadPlan {
    backend: WarpLoadBackend,
    decoder: Option<&'static str>,
}

fn warp_load_plan(dtype: &str, space: MemorySpace) -> AResult<WarpLoadPlan> {
    if space == MemorySpace::Tmem {
        if dtype == "float4_e2m1fn" {
            return unsupported("float4 TMEM scalar loads are not implemented");
        }
        return Ok(WarpLoadPlan {
            backend: WarpLoadBackend::Tmem,
            decoder: None,
        });
    }
    if dtype == "float4_e2m1fn" {
        return Ok(WarpLoadPlan {
            backend: WarpLoadBackend::RuntimeFloat4,
            decoder: None,
        });
    }
    let decoder = match dtype {
        "float16" => Some("fp16_bits_to_f32"),
        "bfloat16" => Some("bf16_bits_to_f32"),
        "float8_e4m3fn" => Some("float8_e4m3fn_bits_to_f32"),
        "float8_e8m0fnu" => Some("float8_e8m0fnu_bits_to_f32"),
        _ => None,
    };
    Ok(WarpLoadPlan {
        backend: WarpLoadBackend::RuntimeScalar,
        decoder,
    })
}

/// The pointee dtype of an expression's declared type.
pub(super) fn pointer_pointee_dtype(node: &ObjectRef) -> Option<String> {
    let ty = expr_type(node)?;
    let pointer = ty.as_node::<PointerTypeObj>()?;
    prim_dtype(&pointer.element_type).map(dtype_text)
}

pub fn is_int_imm_value(node: &ObjectRef, expected: i64) -> bool {
    int_imm(node) == Some(expected)
}

impl<'a> Emitter<'a> {
    pub fn rust_scalar_type(&self, dtype: &str) -> Option<String> {
        stmt_rust_scalar_by_dtype(dtype)
            .map(str::to_owned)
            .or_else(|| vector_rust_type(self.ctx.schema, dtype))
    }

    // ------------------------------------------------------------------
    // Buffer references.
    // ------------------------------------------------------------------

    pub fn buffer_ref(&mut self, buffer: &BufferVar) -> AResult<String> {
        for (candidate, reference) in self.dynamic_buffers.clone().iter().rev() {
            if same(candidate.as_var(), buffer.as_var()) {
                self.record_dynamic_buffer_use(reference);
                return Ok(reference.clone());
            }
        }
        let code = self.buffer_code(buffer)?;
        let plan = self.plan_of(code);
        if plan.dynamic_data_var.is_some() {
            return unsupported(format!(
                "buffer:{}:dynamic DeclBuffer view is used before declaration",
                plan.name
            ));
        }
        Ok(format!(
            "buffers.buffers[{}]",
            code.storage_index.expect("storage index")
        ))
    }

    fn same_static_byte_interval(lhs: &BufferPlan, rhs: &BufferPlan) -> AResult<bool> {
        let left = &lhs.layout;
        let right = &rhs.layout;
        if left.dynamic_elem_offset.is_some()
            || right.dynamic_elem_offset.is_some()
            || left.dynamic_layout_elem_offset.is_some()
            || right.dynamic_layout_elem_offset.is_some()
            || left.element_count.is_none()
            || right.element_count.is_none()
        {
            return Ok(false);
        }
        Ok(
            left.elem_offset * left.itemsize == right.elem_offset * right.itemsize
                && left.byte_end()? == right.byte_end()?,
        )
    }

    fn ordinary_logical_view_root(&self, buffer: &BufferVar) -> AResult<BufferVar> {
        let mut current = buffer.clone();
        let mut seen: Vec<BufferVar> = Vec::new();
        while !seen
            .iter()
            .any(|candidate| same(candidate.as_var(), current.as_var()))
        {
            seen.push(current.clone());
            let data = self.buffer_bindings.declared_data(&current);
            let parent = match data {
                Some(data) => projected_buffer(&oref(data))?,
                None => None,
            };
            let Some(parent) = parent else {
                break;
            };
            let (child_plan, parent_plan) = match (
                self.memory_plan.resolve(&current),
                self.memory_plan.resolve(&parent),
            ) {
                (Ok(child), Ok(parent)) => (child, parent),
                (Err(Failure::Unsupported { .. }), _) | (_, Err(Failure::Unsupported { .. })) => {
                    break
                }
                (Err(error), _) | (_, Err(error)) => return Err(error),
            };
            if child_plan.backing_index != parent_plan.backing_index
                || !Self::same_static_byte_interval(child_plan, parent_plan)?
                || !(child_plan.view_geometry_changed(parent_plan)?
                    || child_plan.name.is_empty()
                    || parent_plan.name.is_empty())
            {
                break;
            }
            current = parent;
        }
        Ok(current)
    }

    /// `logical_buffer_name`: one checker identity for a physical view.
    pub fn logical_buffer_name(&mut self, buffer: &BufferVar) -> AResult<String> {
        for (candidate, name) in &self.logical_buffer_name_cache {
            if same(candidate.as_var(), buffer.as_var()) {
                return Ok(name.clone());
            }
        }
        let plan = self.memory_plan.resolve(buffer)?;
        if plan.space == MemorySpace::Tmem {
            return self.tmem_logical_buffer_name(buffer);
        }
        let root = self.ordinary_logical_view_root(buffer)?;
        let mut name = self.memory_plan.resolve(&root)?.name.clone();
        if name.is_empty() {
            name = format!("anonymous_{}", self.buffer_code(buffer)?.field);
        }
        self.logical_buffer_name_cache
            .push((buffer.clone(), name.clone()));
        Ok(name)
    }

    /// The runtime pointer behind a decoded buffer projection.
    pub fn emit_buffer_data_pointer(&mut self, expression: &ObjectRef) -> AResult<RustValue> {
        let source = projected_buffer(expression)?;
        let source_plan = match &source {
            Some(source) => Some(self.memory_plan.resolve(source)?),
            None => None,
        };
        if let (Some(source), Some(plan)) = (&source, source_plan) {
            if plan.dynamic_data_var.is_some() {
                let reference = self.buffer_ref(source)?;
                return Ok(RustValue::new(
                    format!("PhysicalPtr::new({reference}.clone(), WarpValue::splat(0_i64), 1)"),
                    "PhysicalPtr",
                    Uniformity::Uniform,
                ));
            }
        }
        let backing_index = source_plan.and_then(|plan| plan.backing_index);
        let mut candidates: Vec<usize> = Vec::new();
        for (position, (buffer, code)) in self.buffers.iter().enumerate() {
            let plan = self.plan_of(code);
            if plan.dynamic_data_var.is_some() {
                continue;
            }
            let matches = match backing_index {
                Some(index) => plan.backing_index == Some(index),
                None => same(&self.buffer_bindings.storage_key(buffer)?, expression),
            };
            if matches {
                candidates.push(position);
            }
        }
        if candidates.is_empty() {
            return unsupported(
                "No physical buffer backing owns the requested buffer-data projection",
            );
        }
        let mut backing_indices: Vec<Option<usize>> = candidates
            .iter()
            .map(|position| self.plan_of(&self.buffers[*position].1).backing_index)
            .collect();
        backing_indices.sort();
        backing_indices.dedup();
        if backing_indices.len() != 1 {
            return unsupported("Buffer-data projection spans multiple physical backings");
        }
        let roots: Vec<usize> = candidates
            .into_iter()
            .filter(|position| self.plan_of(&self.buffers[*position].1).layout.elem_offset == 0)
            .collect();
        if roots.is_empty() {
            return unsupported("Buffer-data projection has no zero-offset physical root");
        }
        let byte_len = |position: usize| -> i64 {
            let layout = &self.plan_of(&self.buffers[position].1).layout;
            match layout.element_count {
                None => -1,
                Some(count) => count * layout.itemsize,
            }
        };
        // Python `max` keeps the first maximal candidate.
        let mut best = roots[0];
        for position in &roots[1..] {
            if byte_len(*position) > byte_len(best) {
                best = *position;
            }
        }
        let root_buffer = self.buffers[best].0.clone();
        let reference = self.buffer_ref(&root_buffer)?;
        Ok(RustValue::new(
            format!("PhysicalPtr::new({reference}.clone(), WarpValue::splat(0_i64), 1)"),
            "PhysicalPtr",
            Uniformity::Uniform,
        ))
    }

    // ------------------------------------------------------------------
    // Raw address candidates.
    // ------------------------------------------------------------------

    pub fn raw_shared_candidate_storage_indices(&self) -> AResult<Vec<usize>> {
        let mut selected: Vec<((usize, i64, i64), usize)> = Vec::new();
        for (position, (_, code)) in self.buffers.iter().enumerate() {
            let plan = self.plan_of(code);
            if plan.space != MemorySpace::Shared || plan.dynamic_data_var.is_some() {
                continue;
            }
            let (Some(backing), Some(count)) = (plan.backing_index, plan.layout.element_count)
            else {
                continue;
            };
            let key = (
                backing,
                plan.layout.elem_offset * plan.layout.itemsize,
                count * plan.layout.itemsize,
            );
            if !selected.iter().any(|(existing, _)| *existing == key) {
                selected.push((key, position));
            }
        }
        if selected.is_empty() {
            return unsupported(
                "raw shared address requires a declared physical shared-memory view",
            );
        }
        selected.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(selected
            .iter()
            .map(|(_, position)| {
                self.buffers[*position]
                    .1
                    .storage_index
                    .expect("storage index")
            })
            .collect())
    }

    fn raw_shared_candidates(&mut self) -> AResult<String> {
        self.raw_shared_candidate_storage_indices()?;
        self.memory_candidates.shared = true;
        Ok("&buffers.raw_shared_candidates".to_owned())
    }

    pub fn raw_generic_candidate_storage_indices(&self) -> AResult<Vec<usize>> {
        let mut global_roots: Vec<(usize, usize)> = Vec::new();
        let mut shared_views: Vec<((usize, i64, i64), usize)> = Vec::new();
        for (position, (_, code)) in self.buffers.iter().enumerate() {
            let plan = self.plan_of(code);
            if plan.dynamic_data_var.is_some() {
                continue;
            }
            let Some(backing) = plan.backing_index else {
                continue;
            };
            if plan.space == MemorySpace::Global {
                if plan.is_parameter && !global_roots.iter().any(|(key, _)| *key == backing) {
                    global_roots.push((backing, position));
                }
                continue;
            }
            if !matches!(
                plan.space,
                MemorySpace::Shared | MemorySpace::Local | MemorySpace::Register
            ) {
                continue;
            }
            let Some(count) = plan.layout.element_count else {
                continue;
            };
            let key = (
                backing,
                plan.layout.elem_offset * plan.layout.itemsize,
                count * plan.layout.itemsize,
            );
            if !shared_views.iter().any(|(existing, _)| *existing == key) {
                shared_views.push((key, position));
            }
        }
        global_roots.sort_by(|left, right| left.0.cmp(&right.0));
        shared_views.sort_by(|left, right| left.0.cmp(&right.0));
        let selected: Vec<usize> = global_roots
            .iter()
            .map(|(_, position)| *position)
            .chain(shared_views.iter().map(|(_, position)| *position))
            .collect();
        if selected.is_empty()
            && self.pointers.is_empty()
            && !(self.raw_tma.uses_registry && !self.raw_tma.tensor_maps.is_empty())
        {
            return unsupported(
                "integer pointer resolution requires a declared global or shared-memory view",
            );
        }
        Ok(selected
            .iter()
            .map(|position| {
                self.buffers[*position]
                    .1
                    .storage_index
                    .expect("storage index")
            })
            .collect())
    }

    fn raw_generic_candidates(&mut self) -> AResult<String> {
        self.raw_generic_candidate_storage_indices()?;
        self.memory_candidates.generic = true;
        Ok("&buffers.raw_generic_candidates".to_owned())
    }

    pub fn emit_raw_shared_pointer(
        &mut self,
        address: &ObjectRef,
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
        let mut value = self.as_warp_value(value);
        if value.rust_type == "u64" {
            let narrowed = self.control_name("raw_shared_address_u32");
            self.emit_line(&format!(
                "if ({active_mask}).into_iter().any(|lane| decode_generic_shared_address({}[lane]).is_none() && {}[lane] > u64::from(u32::MAX)) {{ return Err(EngineError::message(\"shared instruction received an address outside shared memory\")); }}",
                value.code, value.code
            ));
            self.emit_line(&format!(
                "let {narrowed} = WarpValue::from_fn(|lane| decode_generic_shared_address({}[lane]).unwrap_or({}[lane] as u32));",
                value.code, value.code
            ));
            value = RustValue::new(narrowed, "u32", Uniformity::Varying);
        } else if value.rust_type != "u32" {
            return unsupported(format!(
                "raw shared address lowered to {}, expected u32 or u64",
                value.rust_type
            ));
        }
        let pointer = self.control_name("raw_shared_pointer");
        let candidates = self.raw_shared_candidates()?;
        self.emit_line(&format!(
            "let {pointer} = physical_ptr_from_shared_addresses_u32({candidates}, &{}, &ctx, ({active_mask}))?;",
            value.code
        ));
        Ok(RustValue::new(pointer, "PhysicalPtr", Uniformity::Varying))
    }

    pub fn emit_raw_generic_pointer(
        &mut self,
        value: RustValue,
        active_mask: &str,
    ) -> AResult<RustValue> {
        self.emit_raw_address_pointer(value, active_mask, "physical_ptr_from_generic_addresses_u64")
    }

    /// A view's integer data address also accepts shared offsets.
    fn emit_raw_address_pointer(
        &mut self,
        value: RustValue,
        active_mask: &str,
        resolver: &str,
    ) -> AResult<RustValue> {
        if value.rust_type == "PhysicalPtr" {
            return Ok(value);
        }
        let value = self.as_warp_value(value);
        if value.rust_type != "u64" {
            return unsupported(format!(
                "generic address lowered to {}, expected u64",
                value.rust_type
            ));
        }
        let pointer = self.control_name("raw_generic_pointer");
        let candidates = self.raw_generic_candidates()?;
        self.emit_line(&format!(
            "let {pointer} = {resolver}(&physical, {candidates}, &{}, &ctx, ({active_mask}))?;",
            value.code
        ));
        Ok(RustValue::new(pointer, "PhysicalPtr", Uniformity::Varying))
    }

    // ------------------------------------------------------------------
    // Physical indexing and loads.
    // ------------------------------------------------------------------

    pub fn physical_index(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<RustValue> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        if !info.physical_axes.iter().any(|axis| axis == "m") {
            return unsupported(format!(
                "buffer:{}:does not have a linear physical index",
                buffer_name(buffer)
            ));
        }
        let offset = self.physical_element_offset(buffer, indices)?;
        let value = self.emit_expr(&oref(offset))?;
        let value = self.as_i64(value)?;
        Ok(self.as_warp_value(value))
    }

    pub fn physical_access_mask(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        base_mask: &str,
    ) -> AResult<String> {
        let owners = self.physical_owner_coordinates(buffer, indices)?;
        if owners.is_empty() {
            return Ok(base_mask.to_owned());
        }
        let mut predicates = Vec::new();
        for (axis, expression) in owners {
            let value = self.emit_expr(&oref(expression))?;
            let value = self.as_i64(value)?;
            let coordinate = self.control_name(&format!("layout_owner_{axis}"));
            self.emit_line(&format!("let {coordinate} = {};", value.code));
            let target = if value.uniformity == Uniformity::Uniform {
                coordinate
            } else {
                format!("{coordinate}[lane]")
            };
            let warps = self.warps_per_warpgroup;
            let current = match axis.as_str() {
                "laneid" => "lane as i64".to_owned(),
                "wid_in_wg" => format!("(ctx.warp_id_in_cta() % {warps}) as i64"),
                "tid_in_wg" => {
                    format!("((ctx.warp_id_in_cta() % {warps}) * WARP_SIZE + lane) as i64")
                }
                other => return not_covered(format!("owner axis {other} without a coordinate")),
            };
            predicates.push(format!("({target}) == ({current})"));
        }
        let name = self.control_name("layout_owner_mask");
        self.emit_line(&format!(
            "let {name} = ({base_mask}) & WarpMask::from_predicate(|lane| {});",
            predicates.join(" && ")
        ));
        Ok(name)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_buffer_load(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        access_mask: Option<String>,
        register_access_mask: Option<String>,
        zero_fill_invalid: bool,
        result_dtype: Option<String>,
        source_node: Option<&ObjectRef>,
        source_op_id: Option<i64>,
    ) -> AResult<RustValue> {
        let mut source_op_id = source_op_id;
        if let Some(node) = source_node {
            if source_op_id.is_some() {
                return unsupported("TensorLoad cannot provide both source_node and source_op_id");
            }
            source_op_id = Some(self.lowered_instruction_site(node));
        }
        let buffer_dtype_name = buffer_dtype(buffer);
        let dtype = result_dtype
            .clone()
            .unwrap_or_else(|| buffer_dtype_name.clone());
        let mut access_mask = access_mask;
        let (space, plan_itemsize, plan_is_dynamic) = {
            let code = self.buffer_code(buffer)?;
            let plan = self.plan_of(code);
            (
                plan.space,
                plan.layout.itemsize,
                plan.dynamic_data_var.is_some(),
            )
        };
        if let Some(register_mask) = register_access_mask {
            if access_mask.is_some() {
                return unsupported(
                    "TensorLoad cannot combine general and register-only access masks",
                );
            }
            if space == MemorySpace::Local || space == MemorySpace::Register {
                access_mask = Some(register_mask);
            }
        }
        if let Some(result_dtype) = &result_dtype {
            if *result_dtype != buffer_dtype_name {
                return self.emit_contiguous_vector_load(
                    buffer,
                    indices,
                    result_dtype,
                    access_mask,
                    zero_fill_invalid,
                    source_op_id,
                );
            }
        }
        if self.ctx.schema.vector_dtype_abi(&dtype).is_some() {
            if space == MemorySpace::Tmem {
                return unsupported(format!("{dtype} element loads do not support TMEM storage"));
            }
            if indices
                .iter()
                .any(|index| oref(index.clone()).as_node::<RampObj>().is_some())
            {
                return unsupported(format!("{dtype} element load requires scalar indices"));
            }
        }
        let Some(rust_type) = self.rust_scalar_type(&dtype) else {
            return unsupported(format!(
                "TensorLoad scalar lowering is not implemented for {dtype}"
            ));
        };
        let name = self.control_name("load");
        let buffer_ref = self.buffer_ref(buffer)?;
        if zero_fill_invalid && space != MemorySpace::Shared {
            return unsupported("Zero-filled padding reads are implemented only for shared memory");
        }
        let load_plan = warp_load_plan(&dtype, space)?;
        let requested_mask = access_mask
            .clone()
            .unwrap_or_else(|| "ctx.active_mask()".to_owned());
        if matches!(load_plan.backend, WarpLoadBackend::Tmem) {
            return self.emit_tmem_buffer_load(
                buffer,
                indices,
                &name,
                &buffer_ref,
                &requested_mask,
                &dtype,
                &rust_type,
                source_op_id,
            );
        }
        let index = self.physical_index(buffer, indices)?;
        let access_mask = self.physical_access_mask(buffer, indices, &requested_mask)?;
        let space_marker = if plan_is_dynamic {
            Some("v2::Generic")
        } else {
            match space {
                MemorySpace::Global => Some("v2::Global"),
                MemorySpace::Shared => Some("v2::Shared"),
                MemorySpace::Local | MemorySpace::Register => Some("v2::Local"),
                MemorySpace::Tmem => None,
            }
        };
        let Some(space_marker) = space_marker else {
            return unsupported(format!(
                "TensorLoad has no PTX ld state-space for {}",
                space.value()
            ));
        };
        let access_context = self.control_name("load_context");
        self.emit_line(&format!(
            "let {access_context} = ctx.with_active_mask({access_mask});"
        ));
        let mut access_index = index.code.clone();
        let mut memory_dtype = if dtype == "uint128" || dtype == "int128" {
            "uint64x2".to_owned()
        } else {
            dtype.clone()
        };
        let mut decoder = load_plan.decoder;
        if dtype == "float16" || dtype == "bfloat16" {
            decoder = None;
        }
        if matches!(load_plan.backend, WarpLoadBackend::RuntimeFloat4) {
            let byte_index = self.control_name("float4_byte_index");
            self.emit_line(&format!(
                "let {byte_index} = WarpValue::from_fn(|lane| {}[lane].div_euclid(2_i64));",
                index.code
            ));
            access_index = byte_index;
            memory_dtype = "uint8".to_owned();
            decoder = None;
        } else if dtype == "float8_e4m3fn" || dtype == "float8_e8m0fnu" {
            memory_dtype = "uint8".to_owned();
        }
        let marker = v2_memory_type_rust(self.ctx.schema, &memory_dtype)?;
        let logical_name = self.logical_buffer_name(buffer)?;
        let address = abi::buffer_address(
            space_marker,
            &buffer_ref,
            &format!("({access_index})"),
            plan_itemsize,
            &logical_name,
        );
        let raw_name = if decoder.is_none() && dtype != "float4_e2m1fn" {
            name.clone()
        } else {
            self.control_name("load_bits")
        };
        let site = self.v2_site(source_op_id);
        match super::memory_support::shared_local_rust_type(&memory_dtype)
            .filter(|_| self.use_typed_helpers && space_marker == "v2::Local")
        {
            Some(helper_type) => {
                let invocation = super::memory_support::local_memory_call(
                    "ld",
                    helper_type,
                    &access_context,
                    &site,
                    &buffer_ref,
                    &access_index,
                    plan_itemsize,
                    &logical_name,
                    None,
                );
                self.emit_line(&format!("let {raw_name} = {invocation}?;"));
            }
            None => {
                let call = abi::warp_call(
                    "mem::ld",
                    &site,
                    &[address],
                    Some(&format!("v2::mem::variant::Ld<{marker}, {space_marker}>")),
                    Some(&abi::context(&access_context)),
                    false,
                    true,
                );
                self.emit_line(&format!("let {raw_name} = v2_register_out({call});"));
            }
        }
        if dtype == "float4_e2m1fn" {
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| {{ let byte = {raw_name}[lane]; let bits = if {}[lane].rem_euclid(2_i64) == 0 {{ byte & 0x0f_u8 }} else {{ byte >> 4 }}; float4_e2m1fn_bits_to_f32(bits) }});",
                index.code
            ));
        } else if let Some(decoder) = decoder {
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| {decoder}({raw_name}[lane]));"
            ));
        }
        if dtype == "bool" {
            let mask = self.control_name("load_mask");
            self.emit_line(&format!(
                "let {mask} = {name}.to_mask(|_, value| *value) & {requested_mask};"
            ));
            return Ok(RustValue::mask(mask));
        }
        Ok(RustValue::new(name, rust_type, Uniformity::Varying))
    }

    fn emit_contiguous_vector_load(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        result_dtype: &str,
        access_mask: Option<String>,
        zero_fill_invalid: bool,
        source_op_id: Option<i64>,
    ) -> AResult<RustValue> {
        let (element_dtype, lanes, required_space) = match result_dtype {
            "uint32x2" => ("uint32", 2, MemorySpace::Global),
            "float32x4" => ("float32", 4, MemorySpace::Shared),
            _ => {
                return unsupported(format!(
                    "contiguous vector load dtype {result_dtype} is not implemented"
                ))
            }
        };
        if buffer_dtype(buffer) != element_dtype {
            return unsupported(format!(
                "{result_dtype} load requires {element_dtype} source elements"
            ));
        }
        let last = indices.last().map(|index| oref(index.clone()));
        let Some(last) = last.filter(|node| node.as_node::<RampObj>().is_some()) else {
            return unsupported(format!("{result_dtype} load requires a final Ramp index"));
        };
        if indices[..indices.len() - 1]
            .iter()
            .any(|index| oref(index.clone()).as_node::<RampObj>().is_some())
        {
            return unsupported(format!(
                "{result_dtype} load permits a Ramp only on the final axis"
            ));
        }
        let ramp = last.as_node::<RampObj>().expect("ramp");
        let (base, _) = classify_contiguous_ramp(self.ctx, &last, ramp, Some(lanes))?;
        let space = {
            let code = self.buffer_code(buffer)?;
            self.plan_of(code).space
        };
        if space != required_space {
            return unsupported(format!(
                "{result_dtype} load requires {} storage, got {}",
                required_space.value(),
                space.value()
            ));
        }
        let base: PrimExpr = PrimExpr::try_from(Any::from(base))?;
        let mut components = Vec::new();
        for component in 0..lanes {
            let component_index = if component == 0 {
                base.clone()
            } else {
                super::super::analyze::layout::op_binary(
                    "_OpAdd",
                    super::super::analyze::layout::expr_any(&base),
                    super::super::analyze::layout::int_any(component),
                )?
            };
            let mut component_indices: Vec<PrimExpr> = indices[..indices.len() - 1].to_vec();
            component_indices.push(component_index);
            components.push(self.emit_buffer_load(
                buffer,
                &component_indices,
                access_mask.clone(),
                None,
                zero_fill_invalid,
                None,
                None,
                source_op_id,
            )?);
        }
        let expected = stmt_rust_scalar_by_dtype(element_dtype).expect("element type");
        if components.iter().any(|value| value.rust_type != expected) {
            return unsupported(format!(
                "{result_dtype} component load produced an invalid type"
            ));
        }
        let name = self.control_name(if result_dtype == "uint32x2" {
            "packed_u32x2"
        } else {
            "loaded_f32x4"
        });
        if result_dtype == "uint32x2" {
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| ({}[lane] as u64) | ((({}[lane]) as u64) << 32));",
                components[0].code, components[1].code
            ));
            return Ok(RustValue::new(name, "u64", Uniformity::Varying));
        }
        let component_values: Vec<String> = components
            .iter()
            .map(|value| format!("{}[lane]", value.code))
            .collect();
        self.emit_line(&format!(
            "let {name} = WarpValue::from_fn(|lane| [{}]);",
            component_values.join(", ")
        ));
        Ok(RustValue::new(name, "F32x4", Uniformity::Varying))
    }

    // ------------------------------------------------------------------
    // Statements.
    // ------------------------------------------------------------------

    pub fn contains_current_loop_transfer(&mut self, stmt: &Stmt) -> AResult<bool> {
        let key = oref(stmt.clone());
        if let Some(cached) = self.loop_transfer_cache.get(&key) {
            return Ok(*cached);
        }
        let kind = kind_or_bail(&key)?;
        let result = if stmt.as_node::<BreakObj>().is_some()
            || stmt.as_node::<ContinueObj>().is_some()
        {
            true
        } else if let Some(evaluate) = stmt.as_node::<EvaluateObj>() {
            let value = oref(evaluate.value.clone());
            if value.as_node::<tvm::ir::CallObj>().is_some() && projected_buffer(&value)?.is_none()
            {
                super::control::is_loop_transfer(&value)?
            } else {
                false
            }
        } else if let Some(sequence) = stmt.as_node::<SeqStmtObj>() {
            let mut found = false;
            for child in sequence.seq.iter() {
                if self.contains_current_loop_transfer(&child)? {
                    found = true;
                    break;
                }
            }
            found
        } else if let Some(attr) = stmt.as_node::<AttrStmtObj>() {
            self.contains_current_loop_transfer(&attr.body)?
        } else if let Some(branch) = stmt.as_node::<IfThenElseObj>() {
            self.contains_current_loop_transfer(&branch.then_case)?
                || match &branch.else_case {
                    Some(else_case) => self.contains_current_loop_transfer(else_case)?,
                    None => false,
                }
        } else if matches!(
            kind,
            "For"
                | "While"
                | "AssertStmt"
                | "AllocBuffer"
                | "DeclBuffer"
                | "Bind"
                | "BufferStore"
                | "ScopeIdDefStmt"
                | "TilePrimitiveCall"
        ) {
            false
        } else {
            return unsupported(format!(
                "loop-control traversal is not implemented for statement {kind}"
            ));
        };
        self.loop_transfer_cache.insert(key, result);
        Ok(result)
    }

    pub fn emit_loop_sequence_items(
        &mut self,
        statements: &[Stmt],
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let mut previous = previous_sequence_stmt.cloned();
        for (index, child) in statements.iter().enumerate() {
            self.emit_stmt(child, previous.as_ref())?;
            previous = Some(child.clone());
            if !self.contains_current_loop_transfer(child)? {
                continue;
            }
            let remainder = &statements[index + 1..];
            if !remainder.is_empty() {
                self.emit_line("if !ctx.active_mask().is_empty() {");
                self.indent += 1;
                self.emit_loop_sequence_items(remainder, previous.as_ref())?;
                self.indent -= 1;
                self.emit_line("}");
            }
            return Ok(());
        }
        Ok(())
    }

    pub(super) fn emit_stmt_inner(
        &mut self,
        stmt: &Stmt,
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let node = oref(stmt.clone());
        let kind = kind_or_bail(&node)?;
        if let Some(attr) = stmt.as_node::<AttrStmtObj>() {
            if ffi_text(&attr.attr_key) == "thread_extent" {
                return self.emit_thread_extent(attr, previous_sequence_stmt);
            }
            return self.emit_stmt(&attr.body, previous_sequence_stmt);
        }
        if let Some(sequence) = stmt.as_node::<SeqStmtObj>() {
            let statements: Vec<Stmt> = sequence.seq.iter().collect();
            if self.control_depth == 0 {
                return self.emit_root_sequence(&statements, previous_sequence_stmt);
            }
            if self.split_arm_depth > 0 && self.control_depth >= 1 {
                return self.emit_inner_sequence(&statements, previous_sequence_stmt);
            }
            if !self.loop_live_masks.is_empty() && self.contains_current_loop_transfer(stmt)? {
                let snapshot = self.scope_snapshot();
                let result = self.emit_loop_sequence_items(&statements, previous_sequence_stmt);
                self.restore_scope(snapshot);
                return result;
            }
            let snapshot = self.scope_snapshot();
            let mut previous = previous_sequence_stmt.cloned();
            let mut result = Ok(());
            for child in &statements {
                result = self.emit_stmt(child, previous.as_ref());
                if result.is_err() {
                    break;
                }
                previous = Some(child.clone());
            }
            self.restore_scope(snapshot);
            return result;
        }
        if let Some(bind) = stmt.as_node::<BindObj>() {
            let value = self.emit_expr(&oref(bind.value.clone()))?;
            let mut value = self.materialize(value, "bound")?;
            let var_ref = oref(bind.var.clone());
            let pointer_dtype = pointer_pointee_dtype(&var_ref);
            if value.rust_type == "PhysicalPtr" {
                let mut code = format!("({}).clone()", value.code);
                if let Some(dtype) = pointer_dtype.filter(|dtype| !dtype.is_empty()) {
                    code = format!(
                        "{code}.with_pointee_itemsize({}_usize)",
                        dtype_byte_len(self.ctx.schema, &dtype)?
                    );
                }
                let binding = self.control_name("pointer_binding");
                self.emit_line(&format!("let {binding} = {code};"));
                value = RustValue {
                    code: binding,
                    requires_statement: false,
                    ..value
                };
            }
            self.variables.set(var_ref, value);
            return Ok(());
        }
        if kind == "AllocBuffer" {
            return Ok(());
        }
        if let Some(declaration) = stmt.as_node::<DeclBufferObj>() {
            return self.emit_decl_buffer(declaration);
        }
        if let Some(definition) = stmt.as_node::<ScopeIdDefStmtObj>() {
            return self.emit_scope_id(definition);
        }
        if stmt.as_node::<TilePrimitiveCallObj>().is_some() {
            let source_op_id = self.static_op_id(&node)?;
            return self.emit_tile_call(&node, source_op_id);
        }
        if let Some(evaluate) = stmt.as_node::<EvaluateObj>() {
            let value = oref(evaluate.value.clone());
            if value.as_node::<IntImmObj>().is_some() {
                return Ok(());
            }
            if value.as_node::<tvm::ir::CallObj>().is_some() {
                if projected_buffer(&value)?.is_some() {
                    return Ok(());
                }
                let Some(result) = self.emit_call(&value)? else {
                    return Ok(());
                };
                let name = self.control_name("evaluate");
                let code = if result.rust_type == "PhysicalPtr" {
                    format!("&({})", result.code)
                } else {
                    result.code.clone()
                };
                self.emit_line(&format!("let {name} = {code};"));
                self.emit_line(&format!("let _ = {name};"));
                return Ok(());
            }
            return Ok(());
        }
        if let Some(branch) = stmt.as_node::<IfThenElseObj>() {
            return self.emit_if(stmt, branch, previous_sequence_stmt);
        }
        if let Some(assertion) = stmt.as_node::<AssertStmtObj>() {
            return self.emit_assert(&node, assertion);
        }
        if let Some(loop_stmt) = stmt.as_node::<ForObj>() {
            if let Some(helper) = self.try_capture_inner_for_statement_split(stmt, loop_stmt)? {
                self.emit_split_call(&helper, None);
                return Ok(());
            }
            return self.emit_for(stmt, loop_stmt);
        }
        if let Some(loop_stmt) = stmt.as_node::<WhileObj>() {
            return self.emit_while(&node, loop_stmt);
        }
        if kind == "Break" {
            return self.emit_break();
        }
        if kind == "Continue" {
            return self.emit_continue();
        }
        if let Some(ret) = stmt.as_node::<ReturnObj>() {
            return self.emit_return(ret);
        }
        if let Some(store) = stmt.as_node::<BufferStoreObj>() {
            return self.emit_store(&node, store);
        }
        unsupported(format!(
            "Rust statement codegen is not implemented for {kind}"
        ))
    }

    fn emit_decl_buffer(&mut self, stmt: &DeclBufferObj) -> AResult<()> {
        let (name, dynamic_data_var, referenced, elem_offset, itemsize, element_count) = {
            let plan = self.memory_plan.resolve(&stmt.buffer)?;
            (
                plan.name.clone(),
                plan.dynamic_data_var.clone(),
                plan.referenced,
                plan.layout.elem_offset,
                plan.layout.itemsize,
                plan.layout.element_count,
            )
        };
        let Some(dynamic_data_var) = dynamic_data_var else {
            return Ok(());
        };
        let address = oref(dynamic_data_var);
        let pointer_value = self.emit_expr(&address)?;
        let numeric_address = pointer_value.rust_type != "PhysicalPtr";
        if !referenced && numeric_address {
            return Ok(());
        }
        let pointer = if numeric_address && pointer_value.rust_type == "u32" {
            self.emit_raw_shared_pointer(&address, Some(pointer_value), "ctx.active_mask()")?
        } else if numeric_address && pointer_value.rust_type == "u64" {
            self.emit_raw_address_pointer(
                pointer_value,
                "ctx.active_mask()",
                "physical_ptr_from_view_addresses_u64",
            )?
        } else if numeric_address {
            return unsupported(format!(
                "buffer:{name}:integer data address lowered to {}, expected u32 or u64",
                pointer_value.rust_type
            ));
        } else {
            pointer_value
        };
        let Some(element_count) = element_count else {
            return unsupported(format!(
                "buffer:{name}:pointer-derived view requires a static element count"
            ));
        };
        let reference = self.control_name("pointer_view");
        let byte_offset = elem_offset * itemsize;
        let byte_len = element_count * itemsize;
        let mut pointer_code = pointer.code.clone();
        let pointer_dtype = pointer_pointee_dtype(&oref(stmt.data.clone()));
        let byte_typed = matches!(
            pointer_dtype.as_deref(),
            Some("") | Some("int8") | Some("uint8")
        );
        if numeric_address || (byte_typed && itemsize != 1) {
            pointer_code = format!("{}.with_pointee_itemsize({itemsize}_usize)", pointer.code);
        }
        let pointer_space = format!(
            "({}).pointer_space_for_mask(ctx.active_mask())?",
            pointer.code
        );
        self.emit_line(&format!(
            "let {reference} = {pointer_code}.runtime_view(&physical, {pointer_space}, {byte_offset}_usize, {byte_len}_usize, {itemsize}_usize, &ctx, ctx.active_mask())?;"
        ));
        self.dynamic_buffers.push((stmt.buffer.clone(), reference));
        Ok(())
    }

    fn emit_assert(&mut self, node: &ObjectRef, stmt: &AssertStmtObj) -> AResult<()> {
        let condition = self.emit_expr(&oref(stmt.condition.clone()))?;
        if condition.rust_type != "bool" {
            return unsupported("AssertStmt condition must be boolean");
        }
        let message = json_string(&repr_text(node)?);
        if condition.uniformity == Uniformity::Uniform {
            self.emit_line(&format!("if !({}) {{", condition.code));
            self.emit_line(&format!("    return Err(EngineError::message({message}));"));
            self.emit_line("}");
            return Ok(());
        }
        let condition_mask = if condition.is_mask {
            condition
        } else {
            let predicate = self.boolean_lane(&condition)?;
            self.emit_varying_mask(&predicate, ControlProvenance::None)
        };
        let failed = self.control_name("assert_failed");
        self.emit_line(&format!(
            "let {failed} = ctx.active_mask() - {};",
            condition_mask.code
        ));
        self.emit_line(&format!("if !{failed}.is_empty() {{"));
        self.emit_line(&format!(
            "    return Err(EngineError::message(format!(\"{{}}; failed lanes={{:?}}\", {message}, {failed}.iter().collect::<Vec<_>>() )));"
        ));
        self.emit_line("}");
        Ok(())
    }

    fn emit_scope_id(&mut self, stmt: &ScopeIdDefStmtObj) -> AResult<()> {
        let definition = &stmt.def;
        let variables: Vec<tvm::ir::Var> = definition
            .def_ids
            .iter()
            .map(|variable| variable.as_var().clone())
            .collect();
        let mut extents: Vec<Option<i64>> = Vec::new();
        if variables.len() == 1 {
            extents.push(None);
        } else if let Some(declared) = &definition.extents {
            for extent in declared.iter() {
                let Some(value) = int_imm(&oref(extent.clone())) else {
                    return unsupported("Scope ID extents must be static integers");
                };
                extents.push(Some(value));
            }
        } else {
            if variables.is_empty() {
                return unsupported("Extent-free scope IDs require at least one coordinate");
            }
            return unsupported("Multi-coordinate scope IDs require explicit static extents");
        }
        if variables.len() != extents.len() || variables.is_empty() {
            return unsupported("Scope ID variables/extents are inconsistent");
        }
        let scope = i64::from(definition.scope.as_raw());
        let warps = self.warps_per_warpgroup;
        let (flat_code, uniformity) = match scope {
            0 => ("ctx.kernel_cluster_id() as i64".to_owned(), Uniformity::Uniform),
            1 => ("ctx.kernel_cta_id() as i64".to_owned(), Uniformity::Uniform),
            2 => (
                "ctx.cta_id_in_cluster() as i64".to_owned(),
                Uniformity::Uniform,
            ),
            9 => (
                "(ctx.cta_id_in_cluster() % 2) as i64".to_owned(),
                Uniformity::Uniform,
            ),
            3 => (
                format!("(ctx.warp_id_in_cta() / {warps}) as i64"),
                Uniformity::Uniform,
            ),
            4 => (
                "ctx.warp_id_in_cta() as i64".to_owned(),
                Uniformity::Uniform,
            ),
            5 => (
                format!("(ctx.warp_id_in_cta() % {warps}) as i64"),
                Uniformity::Uniform,
            ),
            6 => ("lane as i64".to_owned(), Uniformity::Varying),
            7 => (
                "(ctx.warp_id_in_cta() * WARP_SIZE + lane) as i64".to_owned(),
                Uniformity::Varying,
            ),
            8 => (
                format!("((ctx.warp_id_in_cta() % {warps}) * WARP_SIZE + lane) as i64"),
                Uniformity::Varying,
            ),
            other => {
                return unsupported(format!(
                    "Bootstrap scope binding {other} is not implemented"
                ))
            }
        };
        let mut divisor: i64 = 1;
        for (variable, extent) in variables.iter().zip(extents.iter()) {
            let name = ffi_text(&variable.name);
            let dtype = prim_dtype(&variable.ty).map(dtype_text).unwrap_or_default();
            let rust_type = stmt_rust_scalar_by_dtype(&dtype);
            if !rust_type.is_some_and(is_integer_rust_type) {
                return unsupported(format!(
                    "Scope ID variable {name} has unsupported dtype {dtype}"
                ));
            }
            let rust_type = rust_type.unwrap();
            let coordinate_i64 = match extent {
                None => format!("({flat_code})"),
                Some(extent) => format!("(({flat_code}) / {divisor}_i64) % {extent}_i64"),
            };
            let coordinate = if rust_type == "i64" {
                coordinate_i64
            } else {
                format!("({coordinate_i64}) as {rust_type}")
            };
            let value = if uniformity == Uniformity::Uniform {
                RustValue::new(coordinate, rust_type, Uniformity::Uniform)
            } else {
                let prefix = if name.is_empty() {
                    "scope".to_owned()
                } else {
                    format!("scope_{name}")
                };
                let rust_name = self.control_name(&prefix);
                self.emit_line(&format!(
                    "let {rust_name} = WarpValue::from_fn(|lane| {coordinate});"
                ));
                RustValue::new(rust_name, rust_type, Uniformity::Varying)
            };
            self.variables.set(oref(variable.clone()), value);
            if let Some(extent) = extent {
                divisor *= extent;
            }
        }
        Ok(())
    }

    fn emit_thread_extent(
        &mut self,
        stmt: &AttrStmtObj,
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let Some(iteration) = iter_var_of(&stmt.node) else {
            return not_covered("thread_extent without an IterVar node");
        };
        let thread_tag = ffi_text(&iteration.thread_tag()?);
        let variable = iteration.var()?.as_var().clone();
        let dtype = prim_dtype(&variable.ty).map(dtype_text).unwrap_or_default();
        if dtype != "int32" {
            return unsupported(format!(
                "direct launch coordinate {:?} must have int32 dtype",
                &thread_tag
            ));
        }
        let value = match thread_tag.as_str() {
            "blockIdx.x" => {
                RustValue::new("ctx.kernel_cta_id() as i32", "i32", Uniformity::Uniform)
            }
            "clusterCtaIdx.x" => {
                RustValue::new("ctx.cta_id_in_cluster() as i32", "i32", Uniformity::Uniform)
            }
            "threadIdx.x" => {
                let name = self.control_name("direct_thread_idx");
                self.emit_line(&format!(
                    "let {name} = WarpValue::from_fn(|lane| (ctx.warp_id_in_cta() * WARP_SIZE + lane) as i32);"
                ));
                RustValue::new(name, "i32", Uniformity::Varying)
            }
            _ => {
                return unsupported(format!(
                    "direct launch thread tag {:?} is not implemented",
                    &thread_tag
                ))
            }
        };
        let snapshot = self.scope_snapshot();
        self.variables.set(oref(variable), value);
        let result = self.emit_stmt(&stmt.body, previous_sequence_stmt);
        self.restore_scope(snapshot);
        result
    }

    fn emit_if(
        &mut self,
        stmt: &Stmt,
        branch: &IfThenElseObj,
        initial_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        let mut condition = self.emit_expr(&oref(branch.condition.clone()))?;
        if condition.rust_type != "bool" {
            if !is_integer_rust_type(&condition.rust_type) {
                return unsupported(format!(
                    "If condition must be bool or integer, got {}",
                    condition.rust_type
                ));
            }
            if condition.uniformity == Uniformity::Uniform {
                condition = RustValue {
                    control_provenance: condition.control_provenance,
                    ..RustValue::new(
                        format!("({}) != 0", condition.code),
                        "bool",
                        Uniformity::Uniform,
                    )
                };
            } else {
                let condition_mask = self.control_name("integer_condition_mask");
                self.emit_line(&format!(
                    "let {condition_mask} = {}.to_mask(|_, value| *value != 0) & ctx.active_mask();",
                    condition.code
                ));
                condition = RustValue {
                    control_provenance: condition.control_provenance,
                    ..RustValue::mask(condition_mask)
                };
            }
        } else if condition.uniformity == Uniformity::Varying && !condition.is_mask {
            let condition_mask = self.control_name("boolean_condition_mask");
            self.emit_line(&format!(
                "let {condition_mask} = {}.to_mask(|_, value| *value) & ctx.active_mask();",
                condition.code
            ));
            condition = RustValue {
                control_provenance: condition.control_provenance,
                ..RustValue::mask(condition_mask)
            };
        }
        if condition.uniformity == Uniformity::Uniform {
            if self.try_emit_root_uniform_if_split(branch, &condition, initial_stmt)? {
                return Ok(());
            }
            if self.try_emit_inner_uniform_if_split(stmt, branch, &condition, initial_stmt)? {
                return Ok(());
            }
            self.emit_line(&format!("if {} {{", condition.code));
            self.indent += 1;
            self.emit_scoped(&branch.then_case, initial_stmt)?;
            self.indent -= 1;
            let Some(else_case) = &branch.else_case else {
                self.emit_line("}");
                return Ok(());
            };
            self.emit_line("} else {");
            self.indent += 1;
            self.emit_scoped(else_case, initial_stmt)?;
            self.indent -= 1;
            self.emit_line("}");
            return Ok(());
        }
        if !condition.is_mask {
            return unsupported("Varying control condition did not lower to WarpMask");
        }
        let split_helpers =
            self.try_capture_inner_varying_if_split(branch, &condition, initial_stmt)?;
        let split_call_codes = match &split_helpers {
            None => None,
            Some((then_helper, else_helper)) => {
                let then_codes = self.snapshot_split_call_arguments(then_helper);
                let else_codes = else_helper
                    .as_ref()
                    .map(|helper| self.snapshot_split_call_arguments(helper));
                Some((then_codes, else_codes))
            }
        };
        let suffix = self.control_name("branch");
        let parent = format!("parent_mask_{suffix}");
        let then_mask = format!("then_mask_{suffix}");
        let else_mask = format!("else_mask_{suffix}");
        let join_mask = format!("join_mask_{suffix}");
        let parent_context = format!("parent_context_{suffix}");
        let elect_control = condition.control_provenance == ControlProvenance::ElectSync;
        self.emit_line(&format!("let {parent} = ctx.active_mask();"));
        self.emit_line(&format!("let {parent_context} = ctx;"));
        self.emit_line(&format!("let {then_mask} = {parent} & {};", condition.code));
        self.emit_line(&format!("let {else_mask} = {parent} - {};", condition.code));
        let initial_join = if branch.else_case.is_none() {
            else_mask.clone()
        } else {
            "WarpMask::EMPTY".to_owned()
        };
        self.emit_line(&format!("let mut {join_mask} = {initial_join};"));
        self.emit_line(&format!("if !{then_mask}.is_empty() {{"));
        self.indent += 1;
        let branch_variant = if elect_control {
            "ElectSync"
        } else {
            "Ordinary"
        };
        let branch_context = |mask: &str, variant: &str| {
            format!(
                "ctx = v2_context_out({});",
                abi::call(
                    "control::branch_context",
                    &[abi::context(&parent_context), format!("v2_mask({mask})")],
                    &[format!("v2::control::{variant}")],
                    true,
                    false,
                )
            )
        };
        self.emit_line(&branch_context(&then_mask, branch_variant));
        match &split_helpers {
            None => self.emit_scoped(&branch.then_case, initial_stmt)?,
            Some((then_helper, _)) => {
                let codes = split_call_codes.as_ref().map(|codes| codes.0.clone());
                self.emit_split_call(then_helper, codes);
            }
        }
        self.emit_line(&format!("{join_mask} |= ctx.active_mask();"));
        self.indent -= 1;
        self.emit_line("}");
        if let Some(else_case) = &branch.else_case {
            self.emit_line(&format!("if !{else_mask}.is_empty() {{"));
            self.indent += 1;
            self.emit_line(&branch_context(&else_mask, branch_variant));
            match &split_helpers {
                None => self.emit_scoped(else_case, initial_stmt)?,
                Some((_, else_helper)) => {
                    let helper = else_helper.as_ref().expect("else helper");
                    let codes = split_call_codes.as_ref().and_then(|codes| codes.1.clone());
                    self.emit_split_call(helper, codes);
                }
            }
            self.emit_line(&format!("{join_mask} |= ctx.active_mask();"));
            self.indent -= 1;
            self.emit_line("}");
        }
        self.emit_line(&branch_context(&join_mask, "Ordinary"));
        Ok(())
    }

    pub fn emit_scoped(
        &mut self,
        stmt: &Stmt,
        previous_sequence_stmt: Option<&Stmt>,
    ) -> AResult<()> {
        self.control_depth += 1;
        let snapshot = self.scope_snapshot();
        let result = self.emit_stmt(stmt, previous_sequence_stmt);
        self.restore_scope(snapshot);
        self.control_depth -= 1;
        result
    }

    fn emit_native_loop_enter(
        &mut self,
        kind: &str,
        site_id: i64,
        live: &str,
        iteration: Option<&str>,
    ) {
        if kind == "for" {
            let call = abi::call(
                "control::for_enter",
                &[
                    abi::WARP.to_owned(),
                    abi::site(site_id as u64),
                    iteration.expect("iteration").to_owned(),
                ],
                &[],
                true,
                false,
            );
            self.emit_line(&format!("{call};"));
            return;
        }
        let call = abi::call(
            &format!("control::{kind}_enter"),
            &[
                abi::WARP.to_owned(),
                abi::site(site_id as u64),
                abi::lane_mask(live),
            ],
            &[],
            false,
            false,
        );
        self.emit_suspend_line(&format!("{call}.await?;"));
    }

    fn emit_native_loop_exit(
        &mut self,
        kind: &str,
        site_id: i64,
        next_live: &str,
        next_iteration: Option<&str>,
    ) {
        let mut arguments = vec![abi::WARP.to_owned(), abi::site(site_id as u64)];
        if kind == "for" {
            arguments.push(next_iteration.expect("next iteration").to_owned());
        }
        arguments.push(abi::lane_mask(next_live));
        let call = abi::call(
            &format!("control::{kind}_exit"),
            &arguments,
            &[],
            true,
            true,
        );
        self.emit_suspend_line(&format!("{call};"));
    }

    fn emit_for_body(
        &mut self,
        stmt: &Stmt,
        loop_stmt: &ForObj,
        allow_live_mutation: bool,
        allow_nested_for_split: bool,
    ) -> AResult<()> {
        self.outer_loop_live_split_permissions
            .push(allow_live_mutation);
        self.nested_for_statement_split_permissions
            .push(allow_nested_for_split);
        let result = (|| -> AResult<()> {
            if let Some(helper) = self.try_capture_inner_for_body_split(stmt, loop_stmt)? {
                self.emit_split_call(&helper, None);
                return Ok(());
            }
            self.emit_scoped(&loop_stmt.body, None)
        })();
        self.nested_for_statement_split_permissions.pop();
        self.outer_loop_live_split_permissions.pop();
        result
    }

    fn emit_for(&mut self, stmt: &Stmt, loop_stmt: &ForObj) -> AResult<()> {
        let loop_site_id = self.static_op_id(&oref(stmt.clone()))?;
        let extent_node = oref(loop_stmt.extent.clone());
        let static_extent = match int_imm(&extent_node) {
            Some(value) if value == 0 || value == 1 => Some(value),
            _ => None,
        };
        let has_loop_transfer = self.contains_current_loop_transfer(&loop_stmt.body)?;
        let bounded =
            super::scaffold::allows_bounded_transfer_split(loop_stmt, &self.split_thresholds);
        let allow_live_mutation = has_loop_transfer && bounded;
        let allow_nested_for_split = has_loop_transfer && bounded;
        let suffix = self.control_name("loop");
        let parent = format!("parent_mask_{suffix}");
        let live = format!("live_mask_{suffix}");
        let iteration = format!("iteration_{suffix}");
        self.emit_line(&format!("let {parent} = ctx.active_mask();"));
        self.emit_line(&format!("if !{parent}.is_empty() {{"));
        self.indent += 1;
        let minimum = self.emit_expr(&oref(loop_stmt.min.clone()))?;
        let minimum = self.as_i64(minimum)?;
        let extent = self.emit_expr(&extent_node)?;
        let extent = self.as_i64(extent)?;
        let mut step = RustValue::new("1_i64", "i64", Uniformity::Uniform);
        if let Some(step_expr) = &loop_stmt.step {
            let value = self.emit_expr(&oref(step_expr.clone()))?;
            step = self.as_i64(value)?;
        }
        let loop_var = loop_stmt.loop_var.as_var().clone();
        let loop_var_ref = oref(loop_var.clone());
        let loop_dtype = prim_dtype(&loop_var.ty).map(dtype_text).unwrap_or_default();
        let loop_rust_type = stmt_rust_scalar_by_dtype(&loop_dtype);
        if !loop_rust_type.is_some_and(is_integer_rust_type) {
            return unsupported(format!(
                "For loop variable dtype {loop_dtype} is not a supported integer"
            ));
        }
        let loop_rust_type = loop_rust_type.unwrap();
        let previous = self.variables.get(&loop_var_ref).cloned();
        let rust_var = format!("loop_var_{suffix}");
        let offset = format!("loop_offset_{suffix}");
        let all_uniform = [&minimum, &extent, &step]
            .iter()
            .all(|value| value.uniformity == Uniformity::Uniform);
        if static_extent == Some(0) {
            // An empty static loop emits nothing.
        } else if static_extent == Some(1) {
            self.emit_line(&format!("let mut {live} = {parent};"));
            if step.uniformity == Uniformity::Uniform {
                self.emit_line(&format!("if ({}) <= 0_i64 {{", step.code));
                self.emit_line(
                    "    return Err(EngineError::message(\"For step must be positive\"));",
                );
                self.emit_line("}");
            } else {
                let step_lanes = self.as_warp_value(step.clone());
                let invalid_step = format!("invalid_step_{suffix}");
                self.emit_line(&format!(
                    "let {invalid_step} = {}.to_mask(|lane, value| {parent}.contains(lane) && *value <= 0_i64);",
                    step_lanes.code
                ));
                self.emit_line(&format!("if !{invalid_step}.is_empty() {{"));
                self.emit_line(&format!(
                    "    return Err(EngineError::message(format!(\"For step must be positive; failed lanes={{:?}}\", {invalid_step}.iter().collect::<Vec<_>>() )));"
                ));
                self.emit_line("}");
            }
            self.emit_line(&format!("ctx.set_active_mask({live});"));
            self.emit_native_loop_enter("for", loop_site_id, &live, Some("0_i64"));
            let loop_value = if minimum.uniformity == Uniformity::Uniform {
                self.emit_line(&format!(
                    "let {rust_var}: {loop_rust_type} = ({}) as {loop_rust_type};",
                    minimum.code
                ));
                RustValue::new(rust_var.clone(), loop_rust_type, Uniformity::Uniform)
            } else {
                let minimum_lanes = self.as_warp_value(minimum.clone());
                self.emit_line(&format!(
                    "let {rust_var}: WarpValue<{loop_rust_type}> = WarpValue::from_fn(|lane| ({}[lane]) as {loop_rust_type});",
                    minimum_lanes.code
                ));
                RustValue::new(rust_var.clone(), loop_rust_type, Uniformity::Varying)
            };
            self.variables.set(loop_var_ref.clone(), loop_value);
            self.loop_live_masks.push(live.clone());
            self.loop_depth += 1;
            self.emit_for_body(stmt, loop_stmt, allow_live_mutation, allow_nested_for_split)?;
            self.emit_native_loop_exit("for", loop_site_id, "WarpMask::EMPTY", Some("1_i64"));
            self.loop_depth -= 1;
            self.loop_live_masks.pop();
        } else if all_uniform {
            self.emit_line(&format!("let mut {live} = {parent};"));
            self.emit_line(&format!(
                "if ({}) > 0_i64 && ({}) <= 0_i64 {{",
                extent.code, step.code
            ));
            self.emit_line("    return Err(EngineError::message(\"For step must be positive\"));");
            self.emit_line("}");
            self.emit_line(&format!("let mut {offset} = 0_i64;"));
            self.emit_line(&format!("let mut {iteration} = 0_i64;"));
            self.emit_line(&format!("while {offset} < ({}) {{", extent.code));
            self.indent += 1;
            self.emit_line(&format!("if {live}.is_empty() {{ break; }}"));
            self.emit_line(&format!("ctx.set_active_mask({live});"));
            self.emit_native_loop_enter("for", loop_site_id, &live, Some(&iteration));
            self.emit_line(&format!(
                "let {rust_var}: {loop_rust_type} = (({}).wrapping_add({offset})) as {loop_rust_type};",
                minimum.code
            ));
            self.variables.set(
                loop_var_ref.clone(),
                RustValue::new(rust_var.clone(), loop_rust_type, Uniformity::Uniform),
            );
            self.loop_live_masks.push(live.clone());
            self.loop_depth += 1;
            self.emit_for_body(stmt, loop_stmt, allow_live_mutation, allow_nested_for_split)?;
            self.loop_depth -= 1;
            self.loop_live_masks.pop();
            self.emit_line(&format!(
                "{offset} = {offset}.checked_add({}).ok_or_else(|| EngineError::message(\"For loop offset overflow\"))?;",
                step.code
            ));
            self.emit_line(&format!(
                "{iteration} = {iteration}.checked_add(1_i64).ok_or_else(|| EngineError::message(\"For loop iteration overflow\"))?;"
            ));
            let next_iteration_mask = format!("next_iteration_mask_{suffix}");
            self.emit_line(&format!(
                "let {next_iteration_mask} = if !{live}.is_empty() && {offset} < ({}) {{ {live} }} else {{ WarpMask::EMPTY }};",
                extent.code
            ));
            self.emit_native_loop_exit("for", loop_site_id, &next_iteration_mask, Some(&iteration));
            self.indent -= 1;
            self.emit_line("}");
        } else {
            self.emit_line(&format!("let mut {live} = {parent};"));
            let minimum_lanes = self.as_warp_value(minimum.clone());
            let extent_lanes = self.as_warp_value(extent.clone());
            let step_lanes = self.as_warp_value(step.clone());
            let invalid_step = format!("invalid_step_{suffix}");
            let iteration_mask = format!("iteration_mask_{suffix}");
            self.emit_line(&format!(
                "let {invalid_step} = {}.to_mask(|lane, value| {parent}.contains(lane) && {}[lane] > 0_i64 && *value <= 0_i64);",
                step_lanes.code, extent_lanes.code
            ));
            self.emit_line(&format!("if !{invalid_step}.is_empty() {{"));
            self.emit_line(&format!(
                "    return Err(EngineError::message(format!(\"For step must be positive; failed lanes={{:?}}\", {invalid_step}.iter().collect::<Vec<_>>() )));"
            ));
            self.emit_line("}");
            self.emit_line(&format!("let mut {offset} = WarpValue::splat(0_i64);"));
            self.emit_line(&format!("let mut {iteration} = 0_i64;"));
            self.emit_line("loop {");
            self.indent += 1;
            self.emit_line(&format!(
                "let {iteration_mask} = {offset}.to_mask(|lane, value| {live}.contains(lane) && *value < {}[lane]);",
                extent_lanes.code
            ));
            self.emit_line(&format!("{live} = {iteration_mask};"));
            self.emit_line(&format!("if {live}.is_empty() {{ break; }}"));
            self.emit_line(&format!("ctx.set_active_mask({live});"));
            self.emit_native_loop_enter("for", loop_site_id, &live, Some(&iteration));
            self.emit_line(&format!(
                "let {rust_var}: WarpValue<{loop_rust_type}> = WarpValue::from_fn(|lane| ({}[lane].wrapping_add({offset}[lane])) as {loop_rust_type});",
                minimum_lanes.code
            ));
            self.variables.set(
                loop_var_ref.clone(),
                RustValue::new(rust_var.clone(), loop_rust_type, Uniformity::Varying),
            );
            self.loop_live_masks.push(live.clone());
            self.loop_depth += 1;
            self.emit_for_body(stmt, loop_stmt, allow_live_mutation, allow_nested_for_split)?;
            self.loop_depth -= 1;
            self.loop_live_masks.pop();
            self.emit_line(&format!("for lane in {live} {{"));
            self.emit_line(&format!(
                "    {offset}[lane] = {offset}[lane].checked_add({}[lane]).ok_or_else(|| EngineError::message(\"For loop offset overflow\"))?;",
                step_lanes.code
            ));
            self.emit_line("}");
            self.emit_line(&format!(
                "{iteration} = {iteration}.checked_add(1_i64).ok_or_else(|| EngineError::message(\"For loop iteration overflow\"))?;"
            ));
            let next_iteration_mask = format!("next_iteration_mask_{suffix}");
            self.emit_line(&format!(
                "let {next_iteration_mask} = {offset}.to_mask(|lane, value| {live}.contains(lane) && *value < {}[lane]);",
                extent_lanes.code
            ));
            self.emit_native_loop_exit("for", loop_site_id, &next_iteration_mask, Some(&iteration));
            self.indent -= 1;
            self.emit_line("}");
        }
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line(&format!("ctx.set_active_mask({parent});"));
        match previous {
            None => {
                self.variables.remove(&loop_var_ref);
            }
            Some(previous) => self.variables.set(loop_var_ref, previous),
        }
        Ok(())
    }

    pub fn emit_break(&mut self) -> AResult<()> {
        let Some(live) = self.loop_live_masks.last().cloned() else {
            return unsupported("Break outside a generated loop");
        };
        let live_lvalue = self.split_loop_live_mask_lvalue(&live);
        self.emit_line(&format!(
            "{live_lvalue} = {live_lvalue} - ctx.active_mask();"
        ));
        self.emit_line("ctx.set_active_mask(WarpMask::EMPTY);");
        Ok(())
    }

    pub fn emit_continue(&mut self) -> AResult<()> {
        if self.loop_live_masks.is_empty() {
            return unsupported("Continue outside a generated loop");
        }
        self.emit_line("ctx.set_active_mask(WarpMask::EMPTY);");
        Ok(())
    }

    fn emit_return(&mut self, stmt: &ReturnObj) -> AResult<()> {
        if !is_int_imm_value(&oref(stmt.value.clone()), 0) {
            return unsupported("device kernels may only contain a successful return 0");
        }
        if !self.loop_live_masks.is_empty() {
            return unsupported("Return from inside a generated loop is not implemented");
        }
        self.emit_line("if ctx.active_mask() != WarpMask::FULL {");
        self.emit_line(
            "    return Err(EngineError::message(\"kernel return requires a full warp\"));",
        );
        self.emit_line("}");
        self.emit_line("ctx.set_active_mask(WarpMask::EMPTY);");
        Ok(())
    }

    fn emit_while(&mut self, node: &ObjectRef, stmt: &WhileObj) -> AResult<()> {
        let loop_site_id = self.static_op_id(node)?;
        let suffix = self.control_name("while");
        let parent = format!("parent_mask_{suffix}");
        let live = format!("live_mask_{suffix}");
        let iterations = format!("iterations_{suffix}");
        self.emit_line(&format!("let {parent} = ctx.active_mask();"));
        self.emit_line(&format!("let mut {live} = {parent};"));
        self.emit_line(&format!("let mut {iterations}: usize = 0;"));
        self.emit_line("loop {");
        self.indent += 1;
        self.emit_line(&format!("if {live}.is_empty() {{ break; }}"));
        self.emit_line(&format!("ctx.set_active_mask({live});"));
        let condition = self.emit_expr(&oref(stmt.condition.clone()))?;
        let condition = self.coerce_value(condition, "bool", "while_condition")?;
        if condition.uniformity == Uniformity::Uniform {
            self.emit_line(&format!("if !({}) {{", condition.code));
            self.emit_native_loop_enter("while", loop_site_id, "WarpMask::EMPTY", None);
            self.emit_line("    break;");
            self.emit_line("}");
        } else if condition.is_mask {
            let iteration_mask = format!("iteration_mask_{suffix}");
            self.emit_line(&format!(
                "let {iteration_mask} = {live} & {};",
                condition.code
            ));
            self.emit_line(&format!("if {iteration_mask}.is_empty() {{"));
            self.emit_native_loop_enter("while", loop_site_id, "WarpMask::EMPTY", None);
            self.emit_line("    break;");
            self.emit_line("}");
            self.emit_line(&format!("{live} = {iteration_mask};"));
            self.emit_line(&format!("ctx.set_active_mask({iteration_mask});"));
        } else {
            return unsupported("Varying while condition did not lower to WarpMask");
        }
        self.emit_native_loop_enter("while", loop_site_id, &live, None);
        self.loop_live_masks.push(live.clone());
        self.loop_depth += 1;
        // Split helpers borrow the live mask so a break survives the call;
        // continue is carried by the returned context's active mask.
        self.outer_loop_live_split_permissions.push(true);
        self.nested_for_statement_split_permissions.push(true);
        let result = self.emit_scoped(&stmt.body, None);
        self.nested_for_statement_split_permissions.pop();
        self.outer_loop_live_split_permissions.pop();
        self.loop_depth -= 1;
        self.loop_live_masks.pop();
        result?;
        self.emit_line(&format!("{iterations} += 1;"));
        self.emit_native_loop_exit("while", loop_site_id, &live, None);
        self.emit_line(&format!("if {live}.is_empty() {{"));
        self.emit_line("    break;");
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line(&format!("ctx.set_active_mask({parent});"));
        Ok(())
    }

    fn emit_store(&mut self, node: &ObjectRef, stmt: &BufferStoreObj) -> AResult<()> {
        self.record_global_write(&stmt.buffer)?;
        let op_id = self.static_op_id(node)?;
        let dtype = buffer_dtype(&stmt.buffer);
        let (space, itemsize, dynamic) = {
            let code = self.buffer_code(&stmt.buffer)?;
            let plan = self.plan_of(code);
            (
                plan.space,
                plan.layout.itemsize,
                plan.dynamic_data_var.is_some(),
            )
        };
        let mut raw_value = self.emit_expr(&oref(stmt.value.clone()))?;
        if raw_value.rust_type == "PhysicalPtr" {
            let address = self.control_name("stored_generic_address");
            self.emit_line(&format!(
                "let {address} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
                raw_value.code
            ));
            raw_value = RustValue::new(address, "u64", Uniformity::Varying);
        }
        let Some(rust_type) = self.rust_scalar_type(&dtype) else {
            return unsupported(format!(
                "BufferStore scalar lowering is not implemented for {dtype}"
            ));
        };
        if self.ctx.schema.vector_dtype_abi(&dtype).is_some() && space == MemorySpace::Tmem {
            return unsupported(format!(
                "BufferStore for {dtype} does not support TMEM storage"
            ));
        }
        let value = self.coerce_dtype(raw_value, &dtype, "store_cast")?;
        let value = self.as_warp_value(value);
        if value.rust_type != rust_type {
            return unsupported(format!(
                "BufferStore for {dtype} produced Rust value {}",
                value.rust_type
            ));
        }
        let buffer_ref = self.buffer_ref(&stmt.buffer)?;
        if space == MemorySpace::Tmem {
            return self.emit_tmem_store(stmt, op_id, &dtype, &value, &buffer_ref);
        }
        if dtype == "float4_e2m1fn" {
            return unsupported(
                "float4 BufferStore requires an explicit packed-byte frontend lowering",
            );
        }
        let indices: Vec<PrimExpr> = stmt.indices.iter().collect();
        let index = self.physical_index(&stmt.buffer, &indices)?;
        let access_mask = self.physical_access_mask(&stmt.buffer, &indices, "ctx.active_mask()")?;
        let space_marker = if dynamic {
            Some("v2::Generic")
        } else {
            match space {
                MemorySpace::Global => Some("v2::Global"),
                MemorySpace::Shared => Some("v2::Shared"),
                MemorySpace::Local | MemorySpace::Register => Some("v2::Local"),
                MemorySpace::Tmem => None,
            }
        };
        let Some(space_marker) = space_marker else {
            return unsupported(format!(
                "BufferStore has no PTX st state-space for {}",
                space.value()
            ));
        };
        let mut stored_value = value.code.clone();
        let mut memory_dtype = dtype.clone();
        if dtype == "float8_e4m3fn" || dtype == "float8_e8m0fnu" {
            memory_dtype = "uint8".to_owned();
            stored_value = self.control_name("float8_store_bits");
            let encoder = if dtype == "float8_e4m3fn" {
                "f32_to_float8_e4m3fn_bits"
            } else {
                "f32_to_float8_e8m0fnu_bits"
            };
            self.emit_line(&format!(
                "let {stored_value} = WarpValue::from_fn(|lane| {encoder}({}[lane]));",
                value.code
            ));
        }
        let marker = v2_memory_type_rust(self.ctx.schema, &memory_dtype)?;
        let logical_name = self.logical_buffer_name(&stmt.buffer)?;
        let address = abi::buffer_address(
            space_marker,
            &buffer_ref,
            &format!("({})", index.code),
            itemsize,
            &logical_name,
        );
        let site = self.v2_site(Some(op_id));
        let invocation = abi::warp_call(
            "mem::st",
            &site,
            &[format!(
                "({address}, v2_register(({stored_value}).clone()))"
            )],
            Some(&format!("v2::mem::variant::St<{marker}, {space_marker}>")),
            Some(&abi::context(&format!(
                "ctx.with_active_mask({access_mask})"
            ))),
            false,
            false,
        );
        let invocation = match super::memory_support::shared_local_rust_type(&memory_dtype)
            .filter(|_| self.use_typed_helpers && space_marker == "v2::Local")
        {
            Some(helper_type) => {
                let site = self.v2_site(Some(op_id));
                super::memory_support::local_memory_call(
                    "st",
                    helper_type,
                    &format!("ctx.with_active_mask({access_mask})"),
                    &site,
                    &buffer_ref,
                    &index.code,
                    itemsize,
                    &logical_name,
                    Some(&stored_value),
                )
            }
            None => invocation,
        };
        self.emit_write_call(&invocation);
        self.recorded_store_count += 1;
        Ok(())
    }
}
