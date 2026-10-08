//! The TMEM paths of the kernel emitter: `tmem_coordinates`,
//! `tmem_access_mode`, the TMEM branches of `emit_buffer_load` /
//! `emit_store` / `logical_buffer_name`, and `physical_tmem_coordinates`.

use tvm::analysis::Analyzer;
use tvm::ir::{Expr, IntImm, PrimExpr, Range, Var};
use tvm::tirx::{BufferStoreObj, BufferVar, ForObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Array, Map, ObjectRefCore, String as FfiString};

use super::super::analyze::expression_bindings::ExpressionBindings;
use super::super::analyze::frontend::KernelPlan;
use super::super::analyze::layout::{expr_any, int_any, layout_signature, op_binary};
use super::super::analyze::memory::{BufferPlan, MemorySpace};
use super::super::analyze::util::{
    buffer_name, ffi_error, ffi_text, not_covered, oref, same, unsupported, AResult, Failure,
};
use super::abi;
use super::abi::named_buffer;
use super::{Emitter, RustValue, Uniformity};
use crate::tables::v2_memory_type_rust;

pub struct State<'a> {
    pub implicit: bool,
    pub dynamic_lifecycle: bool,
    naming: Option<TmemNaming<'a>>,
}

impl State<'_> {
    pub fn new(plan: &KernelPlan) -> Self {
        Self {
            implicit: plan.requires_implicit_tmem,
            dynamic_lifecycle: plan.uses_dynamic_tmem_lifecycle,
            naming: None,
        }
    }
}

/// The whole-function facts `logical_buffer_name` consults for TMEM views
/// (`self.expression_bindings` and the logical-name loop ranges).
struct TmemNaming<'a> {
    bindings: &'a ExpressionBindings,
    substitutions: Map<Var, Expr>,
    loop_ranges: Vec<(Var, PrimExpr, PrimExpr)>,
}

impl<'a> TmemNaming<'a> {
    fn build(emitter: &Emitter<'a>) -> AResult<Self> {
        let bindings = emitter.expression_bindings;
        let substitutions = bindings.shape_expressions.substitutions();
        let mut loop_ranges = Vec::new();
        for statement in &emitter.plan.statements {
            if let Some(loop_stmt) = statement.as_node::<ForObj>() {
                loop_ranges.push((
                    loop_stmt.loop_var.as_var().clone(),
                    bindings.resolve_expression_with(&loop_stmt.min, &substitutions)?,
                    bindings.resolve_expression_with(&loop_stmt.extent, &substitutions)?,
                ));
            }
        }
        Ok(Self {
            bindings,
            substitutions,
            loop_ranges,
        })
    }

    /// `(start, end, width)` in TMEM bit units.
    fn interval_bits(&self, plan: &BufferPlan) -> AResult<Option<(PrimExpr, PrimExpr, i64)>> {
        let info = &plan.layout;
        let (Some(allocated_addr), Some(tcol_base), Some(tcol_span)) = (
            &info.allocated_addr,
            info.tmem_tcol_base_static,
            info.tmem_tcol_span_elements,
        ) else {
            return Ok(None);
        };
        let base = info.elem_offset + tcol_base;
        let high = info.elem_offset + tcol_span;
        if high < base {
            return Ok(None);
        }
        let address = self
            .bindings
            .resolve_expression_with(allocated_addr, &self.substitutions)?;
        let scaled = op_binary("_OpMul", expr_any(&address), int_any(32))?;
        let start = op_binary(
            "_OpAdd",
            expr_any(&scaled),
            int_any(base * info.itemsize * 8),
        )?;
        let end = op_binary(
            "_OpAdd",
            expr_any(&scaled),
            int_any(high * info.itemsize * 8),
        )?;
        Ok(Some((start, end, (high - base) * info.itemsize * 8)))
    }

    fn prove_contains(&self, outer: &BufferPlan, inner: &BufferPlan) -> AResult<bool> {
        let (Some(outer_interval), Some(inner_interval)) =
            (self.interval_bits(outer)?, self.interval_bits(inner)?)
        else {
            return Ok(false);
        };
        if outer.layout.tmem_lane_span.unwrap_or(0) < inner.layout.tmem_lane_span.unwrap_or(0) {
            return Ok(false);
        }
        let analyzer = Analyzer::new()?;
        for (variable, minimum, extent) in &self.loop_ranges {
            let range = Range::from_complete_fields(minimum.clone(), extent.clone(), None);
            if analyzer.bind(variable, &range).is_err() {
                return Ok(false);
            }
        }
        let starts = op_binary(
            "_OpLE",
            expr_any(&outer_interval.0),
            expr_any(&inner_interval.0),
        )?;
        let ends = op_binary(
            "_OpLE",
            expr_any(&inner_interval.1),
            expr_any(&outer_interval.1),
        )?;
        Ok(analyzer.can_prove(&starts)? && analyzer.can_prove(&ends)?)
    }
}

impl<'a> Emitter<'a> {
    /// The TMEM branch of `logical_buffer_name` (the cache miss is already
    /// established by the caller).
    pub fn tmem_logical_buffer_name(&mut self, buffer: &BufferVar) -> AResult<String> {
        // Whole-function facts,
        // computed once per kernel.
        if self.tmem.naming.is_none() {
            self.tmem.naming = Some(TmemNaming::build(self)?);
        }
        let naming = self.tmem.naming.take().expect("tmem naming");
        let name = {
            let buffers = &self.memory_plan.buffers;
            let Some(plan_index) = buffers
                .iter()
                .position(|candidate| same(candidate.buffer.as_var(), buffer.as_var()))
            else {
                return Err(Failure::Ffi(ffi_error("TMEM view has no memory plan")));
            };
            let plan = &buffers[plan_index];
            let mut containing: Vec<usize> = Vec::new();
            for (index, candidate) in buffers.iter().enumerate() {
                if index > plan_index
                    || candidate.space != MemorySpace::Tmem
                    || candidate.backing_index != plan.backing_index
                    || candidate.name.is_empty()
                    || !naming.prove_contains(candidate, plan)?
                {
                    continue;
                }
                let same_interval = naming.prove_contains(plan, candidate)?;
                let mut anonymous_view_bridge = false;
                let bridges: &[BufferPlan] = if index + 1 <= plan_index {
                    &buffers[index + 1..plan_index]
                } else {
                    &[]
                };
                for bridge in bridges {
                    if bridge.name.is_empty()
                        && bridge.space == MemorySpace::Tmem
                        && bridge.backing_index == plan.backing_index
                        && naming.prove_contains(bridge, plan)?
                        && naming.prove_contains(plan, bridge)?
                    {
                        anonymous_view_bridge = true;
                        break;
                    }
                }
                // Distinct named declarations with identical geometry are
                // separate logical lifetimes even if they intentionally reuse
                // the same physical TMEM cells.  A containing subview, or an
                // exact-range representation change emitted through anonymous
                // view nodes, keeps the earlier root's identity.
                if same_interval
                    && index != plan_index
                    && candidate.name != plan.name
                    && !candidate.view_geometry_changed(plan)?
                    && !anonymous_view_bridge
                {
                    continue;
                }
                containing.push(index);
            }
            match containing.iter().min() {
                Some(first) => buffers[*first].name.clone(),
                None => plan.name.clone(),
            }
        };
        self.tmem.naming = Some(naming);
        let name = if name.is_empty() {
            format!("anonymous_{}", self.buffer_code(buffer)?.field)
        } else {
            name
        };
        self.logical_buffer_name_cache
            .push((buffer.clone(), name.clone()));
        Ok(name)
    }

    /// Layout-derived `(TLane, TCol)` coordinates in element units.
    pub fn physical_tmem_coordinates(
        &self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<(PrimExpr, PrimExpr)> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        let name = buffer_name(buffer);
        if info.physical_axes != ["TCol", "TLane"] {
            return unsupported(format!("buffer:{name}:is not a TMEM TLane/TCol view"));
        }
        if indices.len() != info.shape.len() {
            return unsupported(format!(
                "buffer:{name}:expected {} indices, got {}",
                info.shape.len(),
                indices.len()
            ));
        }
        let Some(layout) = buffer.buffer_type().layout.clone() else {
            return not_covered("buffer without a layout");
        };
        let coordinates = Array::new(indices.to_vec());
        let mapped: Map<FfiString, PrimExpr> = layout
            .canonicalize()?
            .apply_with_shape(&coordinates, &buffer.buffer_type().shape)?;
        let mut axes: Vec<(String, PrimExpr)> = mapped
            .iter()
            .map(|(axis, value)| (ffi_text(&axis), value))
            .collect();
        axes.sort_by(|left, right| left.0.cmp(&right.0));
        if axes
            .iter()
            .any(|(axis, _)| axis != "TLane" && axis != "TCol")
        {
            let rendered: Vec<String> = axes.iter().map(|(axis, _)| (axis).to_string()).collect();
            return unsupported(format!(
                "buffer:{name}:TMEM layout maps to unsupported axes {:?}: {}",
                &rendered,
                layout_signature(&layout)?
            ));
        }
        let zero: PrimExpr = IntImm::new("int32", 0)?.into();
        let axis = |wanted: &str| -> PrimExpr {
            axes.iter()
                .find(|(axis, _)| axis == wanted)
                .map_or_else(|| zero.clone(), |(_, value)| value.clone())
        };
        Ok((axis("TLane"), axis("TCol")))
    }

    /// Every TensorLoad nested in `expr` lowers with `source_op_id` as its
    /// site instead of its own source identity.
    fn emit_expr_with_load_site(
        &mut self,
        expr: &ObjectRef,
        source_op_id: Option<i64>,
    ) -> AResult<RustValue> {
        let previous = self
            .nested_load_site
            .replace(super::NestedLoadSite::Site(source_op_id));
        let result = self.emit_expr(expr);
        self.nested_load_site = previous;
        result
    }

    /// `tmem_coordinates`: `(lane, tcol, allocated_addr)` as warp `i64` values.
    pub fn tmem_coordinates(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        source_op_id: Option<i64>,
    ) -> AResult<(RustValue, RustValue, RustValue)> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        let Some(allocated_addr) = info.allocated_addr.clone() else {
            return unsupported(format!(
                "buffer:{}:TMEM view has no allocated_addr",
                buffer_name(buffer)
            ));
        };
        let (lane, tcol) = self.physical_tmem_coordinates(buffer, indices)?;
        let lane_value = self.emit_expr(&oref(lane))?;
        let lane_value = self.as_i64(lane_value)?;
        let lane_value = self.as_warp_value(lane_value);
        let tcol_value = self.emit_expr(&oref(tcol))?;
        let tcol_value = self.as_i64(tcol_value)?;
        let tcol_value = self.as_warp_value(tcol_value);
        let allocated = self.emit_expr_with_load_site(&oref(allocated_addr), source_op_id)?;
        let allocated = self.as_i64(allocated)?;
        let allocated = self.as_warp_value(allocated);
        Ok((lane_value, tcol_value, allocated))
    }

    /// `tmem_access_mode`.
    pub fn tmem_access_mode(&self, buffer: &BufferVar) -> AResult<String> {
        let plan = self.memory_plan.resolve(buffer)?;
        if plan.space != MemorySpace::Tmem {
            return unsupported(format!(
                "buffer:{}:does not have a TMEM access mode",
                plan.name
            ));
        }
        let mode = if plan.layout.allocated_addr_static.is_some() {
            "Static"
        } else {
            "Dynamic"
        };
        Ok(format!("TmemAccessMode::{mode}"))
    }

    pub(super) fn tmem_access_marker(&self, buffer: &BufferVar) -> AResult<&'static str> {
        Ok(if self.tmem_access_mode(buffer)?.ends_with("::Static") {
            "v2::tcgen05::variant::StaticTmem"
        } else {
            "v2::tcgen05::variant::DynamicTmem"
        })
    }

    /// The TMEM branch of `emit_buffer_load`.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_tmem_buffer_load(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        name: &str,
        buffer_ref: &str,
        requested_mask: &str,
        dtype: &str,
        rust_type: &str,
        source_op_id: Option<i64>,
    ) -> AResult<RustValue> {
        let (lane, tcol, allocated_addr) = self.tmem_coordinates(buffer, indices, source_op_id)?;
        let access_marker = self.tmem_access_marker(buffer)?;
        let memory_dtype = if dtype == "float8_e4m3fn" || dtype == "float8_e8m0fnu" {
            "uint8"
        } else {
            dtype
        };
        let marker = v2_memory_type_rust(self.ctx.schema, memory_dtype)?;
        let decoder = match dtype {
            "float8_e4m3fn" => Some("float8_e4m3fn_bits_to_f32"),
            "float8_e8m0fnu" => Some("float8_e8m0fnu_bits_to_f32"),
            _ => None,
        };
        let raw_name = if decoder.is_none() {
            name.to_owned()
        } else {
            self.control_name("tmem_load_bits")
        };
        let site = self.v2_site(source_op_id);
        let logical_name = self.logical_buffer_name(buffer)?;
        let call = abi::warp_call(
            "tmem::read",
            &site,
            &[format!(
                "({}, v2_register(({}).clone()), v2_register(({}).clone()), v2_register(({}).clone()))",
                named_buffer("v2::Tmem", buffer_ref, &logical_name),
                lane.code,
                tcol.code,
                allocated_addr.code
            )],
            Some(&format!(
                "v2::tmem::variant::Access<{marker}, {access_marker}>"
            )),
            Some(&abi::context(&format!(
                "ctx.with_active_mask({requested_mask})"
            ))),
            false,
            true,
        );
        self.emit_line(&format!("let {raw_name} = v2_register_out({call});"));
        if let Some(decoder) = decoder {
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

    /// The `MemorySpace.TMEM` branch of `emit_store`.
    pub fn emit_tmem_store(
        &mut self,
        stmt: &BufferStoreObj,
        op_id: i64,
        dtype: &str,
        value: &RustValue,
        buffer_ref: &str,
    ) -> AResult<()> {
        let indices: Vec<PrimExpr> = stmt.indices.iter().collect();
        let (lane, tcol, allocated_addr) =
            self.tmem_coordinates(&stmt.buffer, &indices, Some(op_id))?;
        let access_marker = self.tmem_access_marker(&stmt.buffer)?;
        let memory_dtype = if dtype == "float8_e4m3fn" || dtype == "float8_e8m0fnu" {
            "uint8"
        } else {
            dtype
        };
        let mut stored_value = value.code.clone();
        if dtype == "float8_e4m3fn" {
            stored_value = self.control_name("tmem_float8_store_bits");
            self.emit_line(&format!(
                "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e4m3fn_bits({}[lane]));",
                value.code
            ));
        } else if dtype == "float8_e8m0fnu" {
            stored_value = self.control_name("tmem_float8_store_bits");
            self.emit_line(&format!(
                "let {stored_value} = WarpValue::from_fn(|lane| f32_to_float8_e8m0fnu_bits({}[lane]));",
                value.code
            ));
        }
        let marker = v2_memory_type_rust(self.ctx.schema, memory_dtype)?;
        let site = self.v2_site(Some(op_id));
        let logical_name = self.logical_buffer_name(&stmt.buffer)?;
        let invocation = abi::warp_call(
            "tmem::write",
            &site,
            &[format!(
                "({}, v2_register(({}).clone()), v2_register(({}).clone()), v2_register(({}).clone()), v2_register(({stored_value}).clone()))",
                named_buffer("v2::Tmem", buffer_ref, &logical_name),
                lane.code,
                tcol.code,
                allocated_addr.code
            )],
            Some(&format!(
                "v2::tmem::variant::Access<{marker}, {access_marker}>"
            )),
            None,
            false,
            false,
        );
        self.emit_write_call(&invocation);
        self.recorded_store_count += 1;
        Ok(())
    }
}
