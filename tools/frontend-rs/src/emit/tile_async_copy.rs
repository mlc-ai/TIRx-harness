//! Typed asynchronous copies through the whole-tile v2 instruction ABI, plus
//! the lane-selected lowering helpers and pure-closure capture the tile
//! emitters share.

use std::mem;

use tvm::analysis::Analyzer;
use tvm::ir::{CallObj, IntImm, PrimExpr, PrimType, TensorLoadObj};
use tvm::prim::{And, Cast, CastObj, LetObj, NotObj, Select, SelectObj};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore};

use super::super::analyze::layout::{expr_any, op_binary};
use super::super::analyze::memory::MemorySpace;
use super::super::analyze::tile_forms::{
    any_value, typed_tma_dtype_rejection, AnyValue, ParsedTileCall, TileAttr, TileOpKind,
    TileRegion, TYPED_TMA_ELEMENT_DTYPES,
};
use super::super::analyze::topology_operands;
use super::super::analyze::util::{
    as_buffer, buffer_name, dtype_text, int_imm_expr, kind, not_covered, oref, prim, unsupported,
    AResult, Failure,
};
use super::super::analyze::vector::classify_vector_buffer_load;
use super::abi;
use super::abi::{element_ref, mapped_view, table_state, FN_MAP_STATE};
use super::tile_common::{
    int64_imm, linear_coordinates, offset_index, tir_var, v2_tile_scope, Capture,
};
use super::NestedLoadSite;
use super::{Emitter, RustValue, SplitArgument, Uniformity};
use crate::tables::is_integer_dtype;

fn v2_tile_element(dtype: &str) -> Option<&'static str> {
    match dtype {
        "bool" | "float8_e4m3fn" | "float8_e8m0fnu" => Some("v2::reg::variant::U8"),
        _ => super::tile_common::v2_tile_element(dtype),
    }
}

/// `(space, mapper)`.
fn v2_space(scope: &str) -> Option<(&'static str, &'static str)> {
    Some(match scope {
        "global" => ("v2::Global", "NumSimGlobalMap"),
        "shared" => ("v2::Shared", "NumSimSharedMap"),
        "local" => ("v2::Local", "NumSimLocalMap"),
        "register" => ("v2::Register", "NumSimRegisterMap"),
        _ => return None,
    })
}

fn v2_reduction(name: &str) -> Option<&'static str> {
    Some(match name {
        "add" => "v2::tile::variant::ReduceAdd",
        "min" => "v2::tile::variant::ReduceMin",
        "max" => "v2::tile::variant::ReduceMax",
        "inc" => "v2::tile::variant::ReduceInc",
        "dec" => "v2::tile::variant::ReduceDec",
        "and" => "v2::tile::variant::ReduceAnd",
        "or" => "v2::tile::variant::ReduceOr",
        "xor" => "v2::tile::variant::ReduceXor",
        _ => return None,
    })
}

// ----------------------------------------------------------------------
// Lane-selected helpers.
// ----------------------------------------------------------------------

impl<'a> Emitter<'a> {
    pub fn implicit_source_op_id(
        &self,
        buffer: &BufferVar,
        source_op_id: i64,
    ) -> AResult<Option<i64>> {
        let space = self.memory_plan.resolve(buffer)?.space;
        match space {
            MemorySpace::Local | MemorySpace::Register => Ok(None),
            MemorySpace::Global | MemorySpace::Shared | MemorySpace::Tmem => Ok(Some(source_op_id)),
        }
    }

    /// Run `body` with the buffer loader swapped to `site`.
    pub(super) fn with_load_site<R>(
        &mut self,
        site: Option<NestedLoadSite>,
        body: impl FnOnce(&mut Self) -> AResult<R>,
    ) -> AResult<R> {
        let previous = mem::replace(&mut self.nested_load_site, site);
        let result = body(self);
        self.nested_load_site = previous;
        result
    }

    fn can_select_lane(&mut self, expr: &ObjectRef) -> AResult<bool> {
        let Some(node_kind) = kind(expr) else {
            return Ok(false);
        };
        match node_kind {
            "IntImm" | "FloatImm" | "Var" | "StringImm" => return Ok(true),
            "Ramp" | "Shuffle" => return Ok(false),
            _ => {}
        }
        if let Some(load) = expr.as_node::<TensorLoadObj>() {
            let Some(source) = as_buffer(&oref(load.source.clone())) else {
                return not_covered("TensorLoad source is not a typed buffer");
            };
            if classify_vector_buffer_load(self.ctx, expr, load, &source)?.is_some() {
                return Ok(false);
            }
            for index in load.indices.iter() {
                if !self.can_select_lane(&oref(index))? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if let Some((_, operands)) = crate::analyze::util::bitwise_expr(expr) {
            for operand in operands {
                if !self.can_select_lane(&operand)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if let Some(cast) = expr.as_node::<CastObj>() {
            return self.can_select_lane(&oref(cast.value.clone()));
        }
        if let Some(let_expr) = expr.as_node::<LetObj>() {
            return Ok(self.can_select_lane(&oref(let_expr.value.clone()))?
                && self.can_select_lane(&oref(let_expr.body.clone()))?);
        }
        if self
            .ctx
            .schema
            .integer_binary_node_kinds
            .contains(node_kind)
            || ["LT", "LE", "GT", "GE", "EQ", "NE", "And", "Or"].contains(&node_kind)
        {
            let Some((lhs, rhs)) = topology_operands(expr) else {
                return Ok(false);
            };
            return Ok(self.can_select_lane(&lhs)? && self.can_select_lane(&rhs)?);
        }
        if let Some(not) = expr.as_node::<NotObj>() {
            return self.can_select_lane(&oref(not.a.clone()));
        }
        if let Some(select) = expr.as_node::<SelectObj>() {
            for value in [
                oref(select.condition.clone()),
                oref(select.true_value.clone()),
                oref(select.false_value.clone()),
            ] {
                if !self.can_select_lane(&value)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let Some(call) = expr.as_node::<CallObj>() else {
            return Ok(false);
        };
        let name = crate::decode::call_name(expr)?.unwrap_or_default();
        const SCALAR_CALLS: &[&str] = &[
            "prim.if_then_else",
            "tirx.address_of",
            "tirx.cuda.__activemask",
            "tirx.cuda.__shfl_down_sync",
            "tirx.cuda.__shfl_sync",
            "tirx.cuda.__shfl_up_sync",
            "tirx.cuda.__shfl_xor_sync",
            "tirx.cuda.any_sync",
            "tirx.cuda.ballot_sync",
            "tirx.cuda.bfloat1622float2",
            "tirx.cuda.bfloat162float",
            "tirx.cuda.clock64",
            "tirx.cuda.cta_reduce",
            "tirx.cuda.elect_sync",
            "tirx.cuda.fadd2_rn",
            "tirx.cuda.fdividef",
            "tirx.cuda.ffs_u32",
            "tirx.cuda.float22bfloat162_rn",
            "tirx.cuda.float22bfloat162_rn_from_float2",
            "tirx.cuda.float2_x",
            "tirx.cuda.float2_y",
            "tirx.cuda.float_as_uint",
            "tirx.cuda.fmul2_rn",
            "tirx.cuda.fp8x4_e4m3_from_float4",
            "tirx.cuda.get_tmem_addr",
            "tirx.cuda.half2float",
            "tirx.cuda.hmax2",
            "tirx.cuda.hmin2",
            "tirx.cuda.iket_mark",
            "tirx.cuda.iket_official_event",
            "tirx.cuda.iket_range_end",
            "tirx.cuda.iket_range_pop",
            "tirx.cuda.iket_range_push",
            "tirx.cuda.iket_range_start",
            "tirx.cuda.iket_sentinel_token",
            "tirx.cuda.make_float2",
            "tirx.cuda.mov_sreg",
            "tirx.cuda.reduce_add_sync_u32",
            "tirx.cuda.reduce_min_sync_u32",
            "tirx.cuda.sm100_2sm_leader_smem_addr",
            "tirx.cuda.smem_addr_from_uint64",
            "tirx.cuda.syncthreads_and",
            "tirx.cuda.syncthreads_or",
            "tirx.cuda.thread_rank",
            "tirx.cuda.uint_as_float",
            "tirx.cuda.warp_reduce",
            "tirx.exp",
            "tirx.fabs",
            "tirx.fma",
            "tirx.isnullptr",
            "tirx.log",
            "tirx.log1p",
            "prim.log2",
            "tirx.popcount",
            "tirx.reinterpret",
            "tirx.rsqrt",
            "tirx.sigmoid",
            "tirx.timer_end_cuda",
            "tirx.timer_finalize_cuda",
            "tirx.timer_init_cuda",
            "tirx.timer_start_cuda",
            "tirx.tvm_warp_activemask",
            "tirx.tvm_warp_shuffle",
            "tirx.tvm_warp_shuffle_down",
            "tirx.tvm_warp_shuffle_up",
            "tirx.tvm_warp_shuffle_xor",
        ];
        const COLLECTIVE_CALLS: &[&str] = &[
            "tirx.cuda.__shfl_down_sync",
            "tirx.cuda.__shfl_sync",
            "tirx.cuda.__shfl_up_sync",
            "tirx.cuda.__shfl_xor_sync",
            "tirx.cuda.any_sync",
            "tirx.cuda.ballot_sync",
            "tirx.cuda.cta_reduce",
            "tirx.cuda.reduce_add_sync_u32",
            "tirx.cuda.reduce_min_sync_u32",
            "tirx.cuda.syncthreads_and",
            "tirx.cuda.syncthreads_or",
            "tirx.cuda.warp_reduce",
            "tirx.tvm_warp_shuffle",
            "tirx.tvm_warp_shuffle_down",
            "tirx.tvm_warp_shuffle_up",
            "tirx.tvm_warp_shuffle_xor",
        ];
        if !SCALAR_CALLS.contains(&name.as_str())
            || name == "tirx.address_of"
            || (COLLECTIVE_CALLS.contains(&name.as_str()) && !call.args.is_empty())
        {
            return Ok(false);
        }
        if name == "tirx.reinterpret"
            && (crate::analyze::util::dtype_of(expr)? == "handle"
                || call
                    .args
                    .iter()
                    .map(oref)
                    .map(|arg| crate::analyze::util::dtype_of(&arg))
                    .collect::<AResult<Vec<_>>>()?
                    .iter()
                    .any(|dtype| dtype == "handle"))
        {
            return Ok(false);
        }
        for argument in call.args.iter() {
            let argument = oref(argument);
            if kind(&argument) != Some("StringImm") && !self.can_select_lane(&argument)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn expression_at_lane(
        &mut self,
        expression: &ObjectRef,
        lane: &str,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<RustValue> {
        let previous_load = self.nested_load_site.clone();
        let previous_lane = self.selected_lane.clone();
        if self.can_select_lane(expression)? {
            self.nested_load_site =
                Some(buffer_load.unwrap_or_else(|| NestedLoadSite::AtLane(lane.to_owned())));
            self.selected_lane = Some(lane.to_owned());
        } else {
            self.nested_load_site = Some(buffer_load.unwrap_or(NestedLoadSite::Exact));
            self.selected_lane = None;
        }
        let value = self.emit_expr(expression);
        self.selected_lane = previous_lane;
        self.nested_load_site = previous_load;
        self.value_at_lane(value?, lane)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_buffer_load_at_lane(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        execution_lane: &str,
        access_mask: Option<String>,
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
        let selected_mask = self.control_name("load_lane_access_mask");
        let mut mask_expression = format!(
            "WarpMask::from_lanes(std::iter::once({execution_lane})).map_err(|error| EngineError::message(error.to_string()))? & ctx.active_mask()"
        );
        if let Some(access_mask) = access_mask {
            mask_expression = format!("({mask_expression}) & ({access_mask})");
        }
        self.emit_line(&format!("let {selected_mask} = {mask_expression};"));
        let previous_load = self.nested_load_site.replace(NestedLoadSite::Exact);
        let previous_lane = self.selected_lane.take();
        let full_value = self.emit_buffer_load(
            buffer,
            indices,
            Some(selected_mask),
            None,
            zero_fill_invalid,
            result_dtype,
            None,
            source_op_id,
        );
        self.selected_lane = previous_lane;
        self.nested_load_site = previous_load;
        self.value_at_lane(full_value?, execution_lane)
    }

    pub(super) fn physical_index_at_lane(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        lane: &str,
        buffer_load: Option<NestedLoadSite>,
        nonnegative_floor: bool,
    ) -> AResult<RustValue> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        if !info.physical_axes.iter().any(|axis| axis == "m") {
            return unsupported(format!(
                "buffer:{}:does not have a linear physical index",
                buffer_name(buffer)
            ));
        }
        let offset = self.physical_element_offset(buffer, indices)?;
        let previous = mem::replace(&mut self.nonnegative_floor, nonnegative_floor);
        let value = self
            .expression_at_lane(&oref(offset), lane, buffer_load)
            .and_then(|value| self.as_i64(value));
        self.nonnegative_floor = previous;
        value
    }

    pub(super) fn physical_access_predicate_at_lane(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        lane: &str,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<String> {
        let owners = self.physical_owner_coordinates(buffer, indices)?;
        let mut predicates = Vec::new();
        for (axis, expression) in owners {
            let value = self.expression_at_lane(&oref(expression), lane, buffer_load.clone())?;
            let value = self.as_i64(value)?;
            let warps = self.warps_per_warpgroup;
            let current = match axis.as_str() {
                "laneid" => format!("{lane} as i64"),
                "wid_in_wg" => format!("(ctx.warp_id_in_cta() % {warps}) as i64"),
                "tid_in_wg" => {
                    format!("((ctx.warp_id_in_cta() % {warps}) * WARP_SIZE + {lane}) as i64")
                }
                other => return not_covered(format!("owner axis {other} without a coordinate")),
            };
            predicates.push(format!("({}) == ({current})", value.code));
        }
        Ok(if predicates.is_empty() {
            "true".to_owned()
        } else {
            predicates.join(" && ")
        })
    }

    pub(super) fn tmem_coordinates_at_lane(
        &mut self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
        execution_lane: &str,
        source_op_id: Option<i64>,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<(RustValue, RustValue, RustValue)> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        let Some(allocated_addr) = info.allocated_addr.clone() else {
            return unsupported(format!(
                "buffer:{}:TMEM view has no allocated_addr",
                buffer_name(buffer)
            ));
        };
        let (lane, tcol) = self.physical_tmem_coordinates(buffer, indices)?;
        let implicit_load = buffer_load
            .unwrap_or_else(|| NestedLoadSite::AtLaneSite(execution_lane.to_owned(), source_op_id));
        let mut values = Vec::new();
        for expression in [lane, tcol, allocated_addr] {
            let value = self.expression_at_lane(
                &oref(expression),
                execution_lane,
                Some(implicit_load.clone()),
            )?;
            values.push(self.as_i64(value)?);
        }
        let allocated = values.pop().expect("allocated_addr");
        let tcol = values.pop().expect("tcol");
        let lane = values.pop().expect("lane");
        Ok((lane, tcol, allocated))
    }

    pub(super) fn uniform_i64_operand(&mut self, expression: &Any, label: &str) -> AResult<String> {
        match any_value(expression)? {
            AnyValue::Bool(_) => unsupported(format!("{label} must be an integer, got bool")),
            AnyValue::Int(value) => Ok(format!("{value}_i64")),
            AnyValue::Object(node) => Ok(self.uniform_i64(&node, label)?.code),
            other => Err(Failure::Ffi(super::super::analyze::util::ffi_error(
                &format!("{label} cannot be rendered from a {}", other.type_name()),
            ))),
        }
    }

    pub(super) fn tile_variable_scope<R>(
        &mut self,
        body: impl FnOnce(&mut Self) -> AResult<R>,
    ) -> AResult<R> {
        let snapshot = self.scope_snapshot();
        let result = body(self);
        self.restore_scope(snapshot);
        result
    }

    /// One `for` over `extent` elements
    /// whose loop variable is a scoped uniform `i64`.
    pub(super) fn tile_linear_loop<R>(
        &mut self,
        prefix: &str,
        extent: i64,
        body: impl FnOnce(&mut Self, &PrimExpr) -> AResult<R>,
    ) -> AResult<R> {
        let suffix = self.control_name(prefix);
        let rust_name = format!("{prefix}_{suffix}");
        let (variable, linear) = tir_var(&rust_name, "int64")?;
        self.emit_line(&format!("for {rust_name} in 0_i64..{extent}_i64 {{"));
        self.indent += 1;
        let result = self.tile_variable_scope(|emitter| {
            emitter.variables.set(
                variable,
                RustValue::new(rust_name.clone(), "i64", Uniformity::Uniform),
            );
            body(emitter, &linear)
        });
        self.indent -= 1;
        self.emit_line("}");
        result
    }

    /// The manifest topology's CTAs per cluster (the manifest topology is
    /// `extract_topology` of this same PrimFunc).
    pub(super) fn tile_ctas_per_cluster(&self) -> AResult<i64> {
        Ok(self.plan.topology.ctas_per_cluster)
    }

    /// `body` is emitted into a detached
    /// buffer under a split effect scope and kept only when it turned out pure.
    pub(super) fn capture_pure_closure(
        &mut self,
        reject_message: &str,
        body: impl FnOnce(&mut Self, NestedLoadSite) -> AResult<()>,
    ) -> AResult<Option<(Vec<String>, Vec<SplitArgument>)>> {
        let parent_lines = mem::take(&mut self.lines);
        let parent_indent = mem::replace(&mut self.indent, 0);
        self.push_split_effects(false);
        let snapshot = self.scope_snapshot();
        let result = body(self, NestedLoadSite::Reject(reject_message.to_owned()));
        self.restore_scope(snapshot);
        let effects = self.pop_split_effects();
        let lines = mem::replace(&mut self.lines, parent_lines);
        self.indent = parent_indent;
        match result {
            Ok(()) => {
                if effects.unsafe_reason.is_some() || !effects.used_dynamic_references.is_empty() {
                    return Ok(None);
                }
                Ok(Some((lines, effects.arguments)))
            }
            Err(Failure::Unsupported { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

// ----------------------------------------------------------------------
// Async copy emitter.
// ----------------------------------------------------------------------

pub(super) struct AsyncCopyEmitter<'e, 'a> {
    pub kernel: &'e mut Emitter<'a>,
    pub source_op_id: i64,
    gather4: Vec<PrimExpr>,
}

impl<'e, 'a> AsyncCopyEmitter<'e, 'a> {
    pub fn new(kernel: &'e mut Emitter<'a>, source_op_id: i64) -> Self {
        Self {
            kernel,
            source_op_id,
            gather4: Vec::new(),
        }
    }

    fn region_indices(
        &self,
        region: &TileRegion,
        linear: &PrimExpr,
        logical_shape: Option<&[i64]>,
        gather4: &[PrimExpr],
    ) -> AResult<Vec<PrimExpr>> {
        if !gather4.is_empty() {
            let shape = match logical_shape {
                Some(shape) if shape.len() == 2 && region.mins.len() == 2 => shape,
                _ => return unsupported("copy_async gather4 requires rank-2 logical regions"),
            };
            let coordinates = linear_coordinates(linear, shape)?;
            let row = &coordinates[0];
            let mut selected = gather4[gather4.len() - 1].clone();
            for index in (0..gather4.len() - 1).rev() {
                let condition =
                    op_binary("_OpEQ", expr_any(row), expr_any(&int64_imm(index as i64)?))?;
                selected = prim(&oref(Select::new(
                    condition,
                    gather4[index].clone(),
                    selected,
                )?))?;
            }
            let row_dtype = dtype_text(region.mins[0].dtype());
            return Ok(vec![
                Cast::new(PrimType::new(&row_dtype)?, selected)?.into(),
                offset_index(&region.mins[1], &coordinates[1])?,
            ]);
        }
        let coordinates = linear_coordinates(linear, &region.extents)?;
        region
            .mins
            .iter()
            .zip(coordinates.iter())
            .map(|(minimum, coordinate)| offset_index(minimum, coordinate))
            .collect()
    }

    fn axis_statically_in_bounds(
        &mut self,
        minimum: &PrimExpr,
        region_extent: i64,
        buffer_extent: &PrimExpr,
        analyzer: &Analyzer,
    ) -> bool {
        let proved = || -> AResult<bool> {
            let proofs = self.kernel.expression_bindings;
            let minimum = proofs.resolve_expression(minimum)?;
            let buffer_extent = proofs.resolve_expression(buffer_extent)?;
            let dtype = dtype_text(minimum.dtype());
            let zero: PrimExpr = IntImm::new(&dtype, 0)?.into();
            let extent: PrimExpr = IntImm::new(&dtype, region_extent)?.into();
            let lower = op_binary("_OpGE", expr_any(&minimum), expr_any(&zero))?;
            if !analyzer.can_prove(&lower)? {
                return Ok(false);
            }
            let end = op_binary("_OpAdd", expr_any(&minimum), expr_any(&extent))?;
            let bound: PrimExpr = Cast::new(PrimType::new(&dtype)?, buffer_extent)?.into();
            let upper = op_binary("_OpLE", expr_any(&end), expr_any(&bound))?;
            analyzer.can_prove(&upper).map_err(Into::into)
        };
        proved().unwrap_or(false)
    }

    fn bounds_predicate(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        force_all_axes: bool,
    ) -> AResult<PrimExpr> {
        let analyzer = Analyzer::new()?;
        let buffer_shape: Vec<PrimExpr> = region.buffer.buffer_type().shape.iter().collect();
        let mut predicate: Option<PrimExpr> = None;
        for axis in 0..region.mins.len() {
            let minimum = &region.mins[axis];
            let extent = region.extents[axis];
            let index = &indices[axis];
            let buffer_extent = &buffer_shape[axis];
            if !force_all_axes
                && self.axis_statically_in_bounds(minimum, extent, buffer_extent, &analyzer)
            {
                continue;
            }
            let index_dtype = dtype_text(index.dtype());
            let zero: PrimExpr = IntImm::new(&index_dtype, 0)?.into();
            let lower = op_binary("_OpGE", expr_any(index), expr_any(&zero))?;
            let bound: PrimExpr =
                Cast::new(PrimType::new(&index_dtype)?, buffer_extent.clone())?.into();
            let upper = op_binary("_OpLT", expr_any(index), expr_any(&bound))?;
            let condition = prim(&oref(And::new(lower, upper)?))?;
            predicate = Some(match predicate {
                None => condition,
                Some(previous) => prim(&oref(And::new(previous, condition)?))?,
            });
        }
        match predicate {
            Some(predicate) => Ok(predicate),
            None => Ok(IntImm::new("bool", 1)?.into()),
        }
    }

    /// The loader for one issuing lane.
    fn issue_load(&self, lane: &str) -> NestedLoadSite {
        NestedLoadSite::AtLaneSite(
            lane.to_owned(),
            if self.kernel.analysis_capable {
                Some(self.source_op_id)
            } else {
                None
            },
        )
    }

    fn bounds_at_lane(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        lane: &str,
        buffer_load: Option<NestedLoadSite>,
        force_all_axes: bool,
    ) -> AResult<String> {
        let buffer_load = buffer_load.unwrap_or_else(|| self.issue_load(lane));
        let predicate = self.bounds_predicate(region, indices, force_all_axes)?;
        let value = self
            .kernel
            .expression_at_lane(&oref(predicate), lane, Some(buffer_load))?;
        if value.rust_type != "bool" {
            return unsupported("typed copy bounds predicate did not lower to bool");
        }
        Ok(value.code)
    }

    fn snapshot_region_mins(
        &mut self,
        region: &TileRegion,
        lane: &str,
        role: &str,
    ) -> AResult<TileRegion> {
        let mut snapshots = Vec::new();
        for (axis, minimum) in region.mins.iter().enumerate() {
            let dtype = dtype_text(minimum.dtype());
            if !is_integer_dtype(&dtype) {
                return unsupported(format!(
                    "copy_async {role} minimum {axis} must be integer, got {:?}",
                    &dtype
                ));
            }
            if int_imm_expr(minimum).is_some() {
                snapshots.push(minimum.clone());
                continue;
            }
            let load = self.issue_load(lane);
            let value = self
                .kernel
                .expression_at_lane(&oref(minimum.clone()), lane, Some(load))?;
            let name = self
                .kernel
                .control_name(&format!("async_copy_issue_{role}_min_{axis}"));
            self.kernel.emit_line(&format!(
                "let {name}: {} = ({}).clone();",
                value.rust_type, value.code
            ));
            let (variable, expr) = tir_var(&name, &dtype)?;
            self.kernel.variables.set(
                variable,
                RustValue::new(name, value.rust_type.clone(), Uniformity::Uniform),
            );
            snapshots.push(expr);
        }
        Ok(TileRegion {
            mins: snapshots,
            ..region.clone()
        })
    }

    fn snapshot_gather_coordinates(&mut self, lane: &str) -> AResult<()> {
        let mut snapshots = Vec::new();
        for (index, coordinate) in self.gather4.clone().iter().enumerate() {
            let dtype = dtype_text(coordinate.dtype());
            if !is_integer_dtype(&dtype) {
                return unsupported(format!(
                    "copy_async gather4 coordinate {index} must be integer, got {:?}",
                    &dtype
                ));
            }
            let load = self.issue_load(lane);
            let value =
                self.kernel
                    .expression_at_lane(&oref(coordinate.clone()), lane, Some(load))?;
            let name = self
                .kernel
                .control_name(&format!("async_copy_gather4_{index}"));
            self.kernel.emit_line(&format!(
                "let {name}: {} = ({}).clone();",
                value.rust_type, value.code
            ));
            let (variable, expr) = tir_var(&name, &dtype)?;
            self.kernel.variables.set(
                variable,
                RustValue::new(name, value.rust_type.clone(), Uniformity::Uniform),
            );
            snapshots.push(expr);
        }
        self.gather4 = snapshots;
        Ok(())
    }

    pub fn shape_variant(extents: &[i64]) -> AResult<String> {
        let rank = extents.len();
        if !(1..=5).contains(&rank) {
            return unsupported(format!(
                "typed tile ABI supports rank 1..5, got rank {rank}"
            ));
        }
        Ok(format!(
            "v2::tile::variant::Shape{rank}<{}>",
            extents
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    pub fn element_variant(dtype: &str, reduction: Option<&str>) -> AResult<String> {
        if matches!(reduction, Some("and") | Some("or") | Some("xor")) {
            if ["uint32", "int32", "float32"].contains(&dtype) {
                return Ok("v2::reg::variant::B32".to_owned());
            }
            if ["uint64", "int64", "float64"].contains(&dtype) {
                return Ok("v2::reg::variant::B64".to_owned());
            }
        }
        match v2_tile_element(dtype) {
            Some(marker) => Ok(marker.to_owned()),
            None => unsupported(format!(
                "typed copy has no v2 physical element marker for {:?}",
                dtype
            )),
        }
    }

    fn capture_lazy_map(
        &mut self,
        region: &TileRegion,
        logical_shape: &[i64],
        space: &str,
        target_rank: &str,
        gather4: &[PrimExpr],
    ) -> AResult<Option<Capture>> {
        let itemsize = self
            .kernel
            .ctx
            .inspect_layout(&region.buffer, &self.kernel.bindings)?
            .itemsize;
        let kernel: &mut Emitter<'a> = self.kernel;
        // The closure needs the emitter and this emitter's bounds helpers at
        // once; run the body through a detached `AsyncCopyEmitter` view.
        let source_op_id = self.source_op_id;
        kernel.capture_pure_closure(
            "lazy typed-copy element maps cannot read simulated memory",
            |emitter, reject| {
                let mut view = AsyncCopyEmitter {
                    kernel: emitter,
                    source_op_id,
                    gather4: Vec::new(),
                };
                view.lazy_map_body(
                    region,
                    logical_shape,
                    space,
                    target_rank,
                    gather4,
                    itemsize,
                    reject,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn lazy_map_body(
        &mut self,
        region: &TileRegion,
        logical_shape: &[i64],
        space: &str,
        target_rank: &str,
        gather4: &[PrimExpr],
        itemsize: i64,
        reject: NestedLoadSite,
    ) -> AResult<()> {
        self.kernel
            .emit_line("let dimensions = logical.dimensions();");
        self.kernel.emit_line(&format!(
            "if dimensions.len() != {} {{ return Err(EngineError::message(\"typed-copy map rank mismatch\")); }}",
            logical_shape.len()
        ));
        let linear_name = self.kernel.control_name("async_copy_lazy_linear");
        self.kernel
            .emit_line(&format!("let mut {linear_name} = 0_i64;"));
        for (axis, extent) in logical_shape.iter().enumerate() {
            let coordinate = self.kernel.control_name("async_copy_lazy_coord");
            self.kernel.emit_line(&format!(
                "let {coordinate} = i64::try_from(dimensions[{axis}]).map_err(|_| EngineError::message(\"typed-copy logical index overflow\"))?;"
            ));
            self.kernel.emit_line(&format!(
                "if {coordinate} < 0_i64 || {coordinate} >= {extent}_i64 {{ return Err(EngineError::message(\"typed-copy logical index is out of bounds\")); }}"
            ));
            self.kernel.emit_line(&format!(
                "{linear_name} = {linear_name}.checked_mul({extent}_i64).and_then(|value| value.checked_add({coordinate})).ok_or_else(|| EngineError::message(\"typed-copy linear index overflow\"))?;"
            ));
        }
        let (variable, linear) = tir_var(&linear_name, "int64")?;
        self.kernel.variables.set(
            variable,
            RustValue::new(linear_name.clone(), "i64", Uniformity::Uniform),
        );
        let lane = self.kernel.control_name("async_copy_lazy_lane");
        self.kernel
            .emit_line(&format!("let {lane} = map_lane.index();"));
        let indices = self.region_indices(region, &linear, Some(logical_shape), gather4)?;
        let bounds = self.bounds_at_lane(
            region,
            &indices,
            &lane,
            Some(reject.clone()),
            !gather4.is_empty(),
        )?;
        let owner = self.kernel.physical_access_predicate_at_lane(
            &region.buffer,
            &indices,
            &lane,
            Some(reject.clone()),
        )?;
        let index = self.kernel.physical_index_at_lane(
            &region.buffer,
            &indices,
            &lane,
            Some(reject),
            false,
        )?;
        let byte_offset = self.kernel.control_name("async_copy_lazy_byte_offset");
        self.kernel.emit_line(&format!(
            "let {byte_offset} = i128::from({}).checked_mul({itemsize}_i128).ok_or_else(|| EngineError::message(\"typed-copy byte offset overflow\"))?;",
            index.code
        ));
        let unowned = element_ref(space, "unowned", &[target_rank.to_owned()]);
        let in_bounds = element_ref(
            space,
            "in_bounds",
            &[byte_offset.clone(), target_rank.to_owned()],
        );
        let out_of_bounds = element_ref(space, "out_of_bounds", &[target_rank.to_owned()]);
        self.kernel.emit_line(&format!(
            "if !({owner}) {{ Ok({unowned}) }} else if {bounds} {{ Ok({in_bounds}) }} else {{ Ok({out_of_bounds}) }}"
        ));
        Ok(())
    }

    pub fn mapped_view(
        &mut self,
        region: &TileRegion,
        role: &str,
        logical_shape: &[i64],
        target_rank: &str,
        gather4: &[PrimExpr],
    ) -> AResult<String> {
        let Some((space, _mapper)) = v2_space(&region.memory_scope) else {
            return unsupported(format!(
                "typed copy cannot map {:?} through the v2 tile ABI",
                &region.memory_scope
            ));
        };
        let capture = self.capture_lazy_map(region, logical_shape, space, target_rank, gather4)?;
        let Some((body, arguments)) = capture else {
            return self.materialized_mapped_view(
                region,
                role,
                logical_shape,
                target_rank,
                gather4,
            );
        };
        let mapping = self
            .kernel
            .control_name(&format!("async_copy_{role}_mapping"));
        self.kernel.emit_captured_closure(
            &mapping,
            &format!("#[inline(never)] move |logical: v2::LogicalCoord<'_>, map_lane: v2::LaneId| -> Result<v2::ElementRef<{space}>, EngineError> {{"),
            &body,
            &arguments,
        );
        let mapper = self
            .kernel
            .control_name(&format!("async_copy_{role}_mapper"));
        self.kernel.emit_line(&format!(
            "let {mapper} = NumSimFnMap::<{space}>::all_lanes({mapping});"
        ));
        let view = self.kernel.control_name(&format!("async_copy_{role}_view"));
        let buffer_ref = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            space,
            &buffer_ref,
            &buffer_name,
            &mapper,
            FN_MAP_STATE,
        );
        self.kernel.emit_line(&line);
        Ok(view)
    }

    fn materialized_mapped_view(
        &mut self,
        region: &TileRegion,
        role: &str,
        logical_shape: &[i64],
        target_rank: &str,
        gather4: &[PrimExpr],
    ) -> AResult<String> {
        let Some((space, mapper)) = v2_space(&region.memory_scope) else {
            return unsupported(format!(
                "typed copy cannot map {:?} through the v2 tile ABI",
                &region.memory_scope
            ));
        };
        let itemsize = self
            .kernel
            .ctx
            .inspect_layout(&region.buffer, &self.kernel.bindings)?
            .itemsize;
        let table = self.kernel.control_name(&format!("async_copy_{role}_map"));
        let logical_count: i64 = logical_shape.iter().product();
        let empty = element_ref(space, "unowned", &["None".to_owned()]);
        self.kernel.emit_line(&format!(
            "let mut {table} = vec![{empty}; {logical_count}_usize * 32_usize];"
        ));
        let kernel: &mut Emitter<'a> = self.kernel;
        let source_op_id = self.source_op_id;
        let owned_gather4 = gather4.to_vec();
        let looped =
            kernel.tile_linear_loop("async_copy_element", logical_count, |emitter, linear| {
                let mut view = AsyncCopyEmitter {
                    kernel: emitter,
                    source_op_id,
                    gather4: Vec::new(),
                };
                view.materialized_body(
                    region,
                    role,
                    logical_shape,
                    target_rank,
                    &owned_gather4,
                    itemsize,
                    space,
                    &table,
                    linear,
                )
            });
        looped?;
        let view = self.kernel.control_name(&format!("async_copy_{role}_view"));
        if gather4.is_empty() && logical_count != region.element_count() {
            return unsupported(format!(
                "typed copy {role} map shape changes element count: {:?} != {:?}",
                logical_shape, &region.extents
            ));
        }
        let buffer_ref = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            space,
            &buffer_ref,
            &buffer_name,
            mapper,
            &table_state(logical_shape, &table),
        );
        self.kernel.emit_line(&line);
        Ok(view)
    }

    #[allow(clippy::too_many_arguments)]
    fn materialized_body(
        &mut self,
        region: &TileRegion,
        role: &str,
        logical_shape: &[i64],
        target_rank: &str,
        gather4: &[PrimExpr],
        itemsize: i64,
        space: &str,
        table: &str,
        linear: &PrimExpr,
    ) -> AResult<()> {
        let indices = self.region_indices(region, linear, Some(logical_shape), gather4)?;
        let linear_usize = self
            .kernel
            .control_name(&format!("async_copy_{role}_linear"));
        let linear_name = super::super::analyze::util::repr_text(&oref(linear.clone()))?;
        self.kernel.emit_line(&format!(
            "let {linear_usize} = usize::try_from({linear_name}).map_err(|_| EngineError::message(\"negative typed-copy logical index\"))?;"
        ));
        let lane = self
            .kernel
            .control_name(&format!("async_copy_{role}_map_lane"));
        self.kernel
            .emit_line(&format!("for {lane} in ctx.active_mask() {{"));
        self.kernel.indent += 1;
        let bounds = self.bounds_at_lane(region, &indices, &lane, None, !gather4.is_empty())?;
        let load = self.issue_load(&lane);
        let owner = self.kernel.physical_access_predicate_at_lane(
            &region.buffer,
            &indices,
            &lane,
            Some(load.clone()),
        )?;
        let index = self.kernel.physical_index_at_lane(
            &region.buffer,
            &indices,
            &lane,
            Some(load),
            false,
        )?;
        let byte_offset = self
            .kernel
            .control_name(&format!("async_copy_{role}_byte_offset"));
        self.kernel.emit_line(&format!(
            "let {byte_offset} = i128::from({}).checked_mul({itemsize}_i128).ok_or_else(|| EngineError::message(\"typed copy byte offset overflow\"))?;",
            index.code
        ));
        let slot = format!("{linear_usize} * 32_usize + {lane}");
        let unowned = element_ref(space, "unowned", &[target_rank.to_owned()]);
        let in_bounds = element_ref(
            space,
            "in_bounds",
            &[byte_offset.clone(), target_rank.to_owned()],
        );
        let out_of_bounds = element_ref(space, "out_of_bounds", &[target_rank.to_owned()]);
        self.kernel.emit_line(&format!(
            "{table}[{slot}] = if !({owner}) {{ {unowned} }} else if {bounds} {{ {in_bounds} }} else {{ {out_of_bounds} }};"
        ));
        self.kernel.indent -= 1;
        self.kernel.emit_line("}");
        Ok(())
    }

    fn shared_address(&mut self, expression: &ObjectRef, label: &str) -> AResult<String> {
        let pointer = self.kernel.pointer(expression, label)?;
        let name = self.kernel.control_name("async_copy_shared_address");
        self.kernel
            .emit_line(&format!("let {name} = ({}).clone();", pointer.code));
        Ok(abi::address("v2::Shared", &abi::cloned(&name), None))
    }

    /// `emit`.
    fn emit(&mut self, op: &ParsedTileCall) -> AResult<()> {
        if op.kind != TileOpKind::CopyAsync {
            return unsupported("async-copy emitter received another tile operation");
        }
        let Some(source) = op.operands[0].region() else {
            return unsupported("copy_async source must be a buffer region");
        };
        let mut source = source.clone();
        let mut destination = op.destination.clone();
        self.gather4 = match op.attribute("gather4") {
            Some(TileAttr::Values(values)) => values
                .iter()
                .map(|value| PrimExpr::try_from(value.clone()).map_err(Into::into))
                .collect::<AResult<Vec<_>>>()?,
            _ => Vec::new(),
        };
        let dispatch = op.attr_str("dispatch").expect("dispatch").to_owned();
        if dispatch == "tma" && !TYPED_TMA_ELEMENT_DTYPES.contains(&destination.dtype.as_str()) {
            return unsupported(typed_tma_dtype_rejection(&destination.dtype));
        }
        let source_itemsize = self
            .kernel
            .ctx
            .inspect_layout(&source.buffer, &self.kernel.bindings)?
            .itemsize;
        let destination_itemsize = self
            .kernel
            .ctx
            .inspect_layout(&destination.buffer, &self.kernel.bindings)?
            .itemsize;
        if source_itemsize != destination_itemsize {
            return unsupported(format!(
                "copy_async source and destination physical item sizes differ: {source_itemsize} != {destination_itemsize}"
            ));
        }

        if dispatch == "tma" || dispatch == "dsmem" {
            let issuer = self.kernel.control_name("async_copy_issuer_lane");
            self.kernel.emit_line(&format!(
                "let {issuer} = ctx.active_mask().first_active().ok_or_else(|| EngineError::message(\"copy_async has no active issuing lane\"))?;"
            ));
            source = self.snapshot_region_mins(&source, &issuer, "source")?;
            destination = self.snapshot_region_mins(&destination, &issuer, "destination")?;
            if !self.gather4.is_empty() {
                self.snapshot_gather_coordinates(&issuer)?;
            }
        }

        let cta_group = op.attr_int("cta_group").unwrap_or(1);
        let ctas_per_cluster = self.kernel.tile_ctas_per_cluster()?;
        if cta_group > ctas_per_cluster {
            return unsupported(format!(
                "copy_async cta_group={cta_group} exceeds launch cluster size"
            ));
        }
        if cta_group == 2 && ctas_per_cluster % 2 != 0 {
            return unsupported("copy_async cta_group=2 requires complete CTA pairs");
        }

        let mut destination_rank = "None".to_owned();
        if let Some(TileAttr::Value(remote)) = op.attribute("remote_cta_id") {
            let raw = self
                .kernel
                .uniform_i64_operand(remote, "copy_async remote CTA id")?;
            let rank = self.kernel.control_name("async_copy_remote_cta_rank");
            self.kernel.emit_line(&format!(
                "let {rank} = u32::try_from({raw}).map_err(|_| EngineError::message(\"copy_async remote CTA rank is negative or too large\"))?;"
            ));
            destination_rank = format!("Some({rank})");
        }

        let logical_shape = destination.logical_shape();
        let gather4 = self.gather4.clone();
        let source_view = self.mapped_view(&source, "source", &logical_shape, "None", &gather4)?;
        let destination_view = self.mapped_view(
            &destination,
            "destination",
            &logical_shape,
            &destination_rank,
            &[],
        )?;
        let shape = Self::shape_variant(&logical_shape)?;
        let Some(scope) = v2_tile_scope(&op.exec_scope) else {
            return unsupported(format!(
                "copy_async has no v2 scope marker for {:?}",
                &op.exec_scope
            ));
        };
        let reduction = op.attr_str("reduction").map(str::to_owned);
        let element = Self::element_variant(&destination.dtype, reduction.as_deref())?;
        let context = abi::context("ctx");
        let site = self.kernel.v2_site(Some(self.source_op_id));

        let issue = |kernel: &mut Emitter<'a>, instruction: &str, variant: &str, tail: String| {
            let call = abi::warp_call(
                &format!("tile::{instruction}"),
                &site,
                &[
                    format!("&{destination_view}"),
                    format!("&{source_view}"),
                    tail,
                ],
                Some(variant),
                Some(&context),
                false,
                true,
            );
            kernel.emit_line(&format!("{call};"));
        };

        if dispatch == "ldgsts" {
            let bytes_per_instruction = if source_itemsize <= 4 {
                4
            } else {
                source_itemsize
            };
            if ![4, 8, 16].contains(&bytes_per_instruction) {
                return unsupported(format!(
                    "copy_async cp.async has unsupported vector width {bytes_per_instruction}"
                ));
            }
            let variant = format!(
                "v2::tile::variant::CpAsync<{shape}, {element}, {scope}, {bytes_per_instruction}, v2::tile::variant::ZeroFill>"
            );
            issue(self.kernel, "cp_async", &variant, abi::splat("true"));
            return Ok(());
        }

        if dispatch == "dsmem" {
            let barrier = self.shared_address(&attr_node(op, "mbar")?, "copy_async mbarrier")?;
            let variant = format!("v2::tile::variant::BulkS2sCluster<{shape}, {element}, {scope}>");
            issue(self.kernel, "cp_async_bulk", &variant, barrier);
            return Ok(());
        }

        if dispatch != "tma" {
            return unsupported(format!("copy_async dispatch {:?} has no v2 ABI", &dispatch));
        }

        if source.memory_scope == "global" && destination.memory_scope == "shared" {
            let barrier = self.shared_address(&attr_node(op, "mbar")?, "copy_async mbarrier")?;
            let multicast = op.attr_bool("multicast");
            let marker = if multicast {
                "TensorG2sMulticast"
            } else {
                "TensorG2s"
            };
            let fill = if op.attr_str("oob") == Some("nan") {
                "v2::tile::variant::OobNan"
            } else {
                "v2::tile::variant::ZeroFill"
            };
            let conversion = if op.attr_str("tma_dtype") == Some("tf32") {
                "v2::tile::variant::TensorTf32"
            } else {
                "v2::tile::variant::NoTensorConversion"
            };
            let variant = format!(
                "v2::tile::variant::{marker}<{shape}, {element}, {scope}, {cta_group}, {fill}, {conversion}>"
            );
            let mut args = barrier.clone();
            if multicast {
                let Some(TileAttr::Value(cta_mask)) = op.attribute("cta_mask") else {
                    return Err(Failure::Ffi(super::super::analyze::util::ffi_error(
                        "copy_async multicast without cta_mask",
                    )));
                };
                let mask = self
                    .kernel
                    .uniform_i64_operand(cta_mask, "copy_async CTA mask")?;
                args = format!("({barrier}, {})", abi::splat(&mask));
            }
            issue(self.kernel, "cp_async_bulk_tensor", &variant, args);
            return Ok(());
        }

        if source.memory_scope == "shared" && destination.memory_scope == "global" {
            let (variant, instruction) = match reduction.as_deref() {
                None => (
                    format!("v2::tile::variant::TensorS2g<{shape}, {element}, {scope}>"),
                    "cp_async_bulk_tensor",
                ),
                Some(reduction) => {
                    let Some(reduction_marker) = v2_reduction(reduction) else {
                        return unsupported(format!(
                            "copy_async TMA reduction {:?} has no v2 specialization",
                            reduction
                        ));
                    };
                    (
                        format!(
                            "v2::tile::variant::TensorS2gReduce<{shape}, {element}, {scope}, {reduction_marker}>"
                        ),
                        "cp_reduce_async_bulk_tensor",
                    )
                }
            };
            issue(self.kernel, instruction, &variant, "()".to_owned());
            return Ok(());
        }

        unsupported(format!(
            "typed TMA copy direction {}->{} is invalid",
            source.memory_scope, destination.memory_scope
        ))
    }
}

/// One configuration attribute handed through as a TIR node.
fn attr_node(op: &ParsedTileCall, name: &str) -> AResult<ObjectRef> {
    match op.attribute(name) {
        Some(TileAttr::Value(value)) => match any_value(value)? {
            AnyValue::Object(node) => Ok(node),
            other => Err(Failure::Ffi(super::super::analyze::util::ffi_error(
                &format!(
                    "copy_async {name} is a {}, not a TIR node",
                    other.type_name()
                ),
            ))),
        },
        _ => Err(Failure::Ffi(super::super::analyze::util::ffi_error(
            &format!("copy_async has no {name} attribute"),
        ))),
    }
}

impl<'a> Emitter<'a> {
    /// Emit one complete typed async-copy instruction.
    pub fn emit_tile_async_copy(&mut self, op: &ParsedTileCall, source_op_id: i64) -> AResult<()> {
        AsyncCopyEmitter::new(self, source_op_id).emit(op)
    }

    /// `build_mapped_copy_views`: compile two synchronous-copy layouts into
    /// pure v2 mapped views as `(destination_view, source_view, shape,
    /// element, source_space, destination_space)`.
    pub fn build_mapped_copy_views(
        &mut self,
        source: &TileRegion,
        destination: &TileRegion,
        source_op_id: i64,
    ) -> AResult<(String, String, String, String, String, String)> {
        if source.logical_shape() != destination.logical_shape() {
            return unsupported(
                "synchronous tile copy requires equal source/destination logical shapes",
            );
        }
        let logical_shape = destination.logical_shape();
        let shape = AsyncCopyEmitter::shape_variant(&logical_shape)?;
        let element = AsyncCopyEmitter::element_variant(&source.dtype, None)?;
        let space_of = |scope: &str| -> AResult<String> {
            match v2_space(scope) {
                Some((space, _)) => Ok(space.to_owned()),
                None => Err(Failure::Ffi(super::super::analyze::util::ffi_error(
                    &format!("no v2 space for memory scope {scope}"),
                ))),
            }
        };
        let source_space = space_of(&source.memory_scope)?;
        let destination_space = space_of(&destination.memory_scope)?;
        let mut emitter = AsyncCopyEmitter::new(self, source_op_id);
        let source_view = emitter.mapped_view(source, "source", &logical_shape, "None", &[])?;
        let destination_view =
            emitter.mapped_view(destination, "destination", &logical_shape, "None", &[])?;
        Ok((
            destination_view,
            source_view,
            shape,
            element,
            source_space,
            destination_space,
        ))
    }
}
