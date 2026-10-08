//! Typed TCGEN transfers as one whole-tile v2
//! instruction call.

use tvm::ir::{PrimExpr, PrimType};
use tvm::prim::Cast;
use tvm::tirx::{BufferVar, TileLayout};
use tvm::tvm_ffi::{Any, ObjectRefCast};

use super::super::analyze::layout::{axis_name, expr_any, int_any, op_binary, Extent, LayoutInfo};
use super::super::analyze::memory::MemorySpace;
use super::super::analyze::shapes::structural_equal;
use super::super::analyze::tile_forms::{ParsedTileCall, TileOpKind, TileRegion};
use super::super::analyze::util::{dtype_text, oref, static_int, unsupported, AResult, Failure};
use super::abi;
use super::abi::{element_ref, mapped_view, named_buffer, table_state, FN_MAP_STATE};
use super::tile_common::{
    linear_coordinates, offset_index, tir_var, v2_tile_element, var_name, Capture,
};
use super::NestedLoadSite;
use super::{Emitter, RustValue, Uniformity};

fn raw_ldst_shape(name: &str) -> Option<&'static str> {
    Some(match name {
        "16x32bx2" => "Shape16x32bx2",
        "16x64b" => "Shape16x64b",
        "16x128b" => "Shape16x128b",
        "32x32b" => "Shape32x32b",
        "16x256b" => "Shape16x256b",
        _ => return None,
    })
}

const TMEM_TO_LOCAL_M64_SOURCE_LAYOUTS: [&str; 2] = [
    "TileLayout(shard=[4:32@TLane,16:1@TLane,64:1@TCol],replica=[])",
    "TileLayout(shard=[4:32@TLane,16:1@TLane,128:1@TCol],replica=[])",
];

fn tmem_to_local_m64_destination_layout(num: i64) -> Option<&'static str> {
    match num {
        1 => Some("TileLayout(shard=[4:1@wid_in_wg,2:2@m,32:1@laneid,2:1@m],replica=[])"),
        8 => Some(
            "TileLayout(shard=[4:1@wid_in_wg,2:2@m,8:4@laneid,8:4@m,4:1@laneid,2:1@m],replica=[])",
        ),
        _ => None,
    }
}

fn same_expr(lhs: &PrimExpr, rhs: &PrimExpr) -> AResult<bool> {
    structural_equal(&Any::from(lhs.clone()), &Any::from(rhs.clone()))
}

fn same_optional_expr(lhs: &Option<PrimExpr>, rhs: &Option<PrimExpr>) -> AResult<bool> {
    match (lhs, rhs) {
        (None, None) => Ok(true),
        (Some(lhs), Some(rhs)) => same_expr(lhs, rhs),
        _ => Ok(false),
    }
}

fn same_extent(lhs: &Extent, rhs: &Extent) -> AResult<bool> {
    match (lhs, rhs) {
        (Extent::Static(lhs), Extent::Static(rhs)) => Ok(lhs == rhs),
        (Extent::Dynamic(lhs), Extent::Dynamic(rhs)) => same_expr(lhs, rhs),
        _ => Ok(false),
    }
}

/// Whether two TMEM layouts provide the
/// same physical allocation carrier (every piece of physical geometry).
fn same_allocation_only_tmem_view(lhs: &LayoutInfo, rhs: &LayoutInfo) -> AResult<bool> {
    if lhs.shape.len() != rhs.shape.len() {
        return Ok(false);
    }
    for (left, right) in lhs.shape.iter().zip(rhs.shape.iter()) {
        if !same_extent(left, right)? {
            return Ok(false);
        }
    }
    Ok(lhs.itemsize == rhs.itemsize
        && lhs.elem_offset == rhs.elem_offset
        && same_optional_expr(&lhs.dynamic_elem_offset, &rhs.dynamic_elem_offset)?
        && lhs.element_count == rhs.element_count
        && lhs.signature == rhs.signature
        && lhs.physical_axes == rhs.physical_axes
        && lhs.explicit_strides == rhs.explicit_strides
        && same_optional_expr(
            &lhs.dynamic_layout_elem_offset,
            &rhs.dynamic_layout_elem_offset,
        )?
        && lhs.packed_nibble_offset == rhs.packed_nibble_offset
        && same_optional_expr(&lhs.allocated_addr, &rhs.allocated_addr)?
        && lhs.allocated_addr_static == rhs.allocated_addr_static
        && lhs.tmem_lane_span == rhs.tmem_lane_span
        && lhs.tmem_tcol_span_elements == rhs.tmem_tcol_span_elements
        && lhs.tmem_tcol_base_static == rhs.tmem_tcol_base_static)
}

/// The TMEM base coordinates a canonical ld/st carries as one operand tuple.
fn coordinate_operands(values: &[&RustValue]) -> String {
    format!(
        "({})",
        values
            .iter()
            .map(|value| abi::register(&value.code))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `is_tcgen_transfer`.
pub fn is_tcgen_transfer(op: &ParsedTileCall) -> bool {
    if op.kind != TileOpKind::CopyAsync {
        return false;
    }
    let Some(source) = op.operands.first().and_then(|operand| operand.region()) else {
        return false;
    };
    matches!(
        (
            source.memory_scope.as_str(),
            op.destination.memory_scope.as_str()
        ),
        ("shared", "tmem") | ("tmem", "local") | ("local", "tmem")
    )
}

struct TcgenTransferEmitter<'e, 'a> {
    kernel: &'e mut Emitter<'a>,
    source_op_id: i64,
}

impl<'e, 'a> TcgenTransferEmitter<'e, 'a> {
    /// `_static_int`.
    fn static_int(&self, value: &PrimExpr, field: &str) -> AResult<i64> {
        static_int(
            &self.kernel.ctx.analyzer,
            value,
            &format!("TCGEN transfer {field}"),
            "must be static",
        )
    }

    fn indices(
        &self,
        region: &TileRegion,
        linear: &PrimExpr,
        tmem_row_mode: &str,
    ) -> AResult<Vec<PrimExpr>> {
        let mut coordinates = linear_coordinates(linear, &region.extents)?;
        if tmem_row_mode == "m64_d_low_slab" {
            if coordinates.len() != 2 || region.extents[0] != 64 {
                return unsupported(
                    "TCGEN M64 D-layout row remap requires a rank-2, 64-row TMEM region",
                );
            }
            let row = coordinates[0].clone();
            let column = coordinates[1].clone();
            let slab = op_binary("_OpFloorDiv", expr_any(&row), int_any(16))?;
            let scaled = op_binary("_OpMul", expr_any(&slab), int_any(32))?;
            let within = op_binary("_OpFloorMod", expr_any(&row), int_any(16))?;
            let remapped = op_binary("_OpAdd", expr_any(&scaled), expr_any(&within))?;
            coordinates = vec![remapped, column];
        } else if tmem_row_mode != "direct" {
            return unsupported(format!(
                "unknown normalized TCGEN TMEM row mode {:?}",
                tmem_row_mode
            ));
        }
        region
            .mins
            .iter()
            .zip(coordinates.iter())
            .map(|(minimum, coordinate)| offset_index(minimum, coordinate))
            .collect()
    }

    fn replica_offsets(&self, buffer: &BufferVar) -> AResult<Vec<(i64, i64)>> {
        let Some(layout) = buffer.buffer_type().layout.clone() else {
            return super::super::analyze::util::not_covered("buffer without a layout");
        };
        let canonical = layout.canonicalize()?.try_cast::<TileLayout>()?;
        let mut choices: Vec<Vec<(i64, i64)>> = Vec::new();
        for (index, item) in canonical.replica()?.iter().enumerate() {
            let extent = self.static_int(&item.extent, &format!("replica[{index}].extent"))?;
            let stride = self.static_int(&item.stride, &format!("replica[{index}].stride"))?;
            let axis = axis_name(&item.axis)?;
            if axis != "TLane" && axis != "TCol" {
                return unsupported(format!(
                    "TCGEN TMEM replica axis {:?} is not TLane/TCol",
                    &axis
                ));
            }
            choices.push(
                (0..extent)
                    .map(|value| {
                        if axis == "TLane" {
                            (value * stride, 0)
                        } else {
                            (0, value * stride)
                        }
                    })
                    .collect(),
            );
        }
        if choices.is_empty() {
            return Ok(vec![(0, 0)]);
        }
        let mut offsets: Vec<(i64, i64)> = vec![(0, 0)];
        for choice in &choices {
            let mut next = Vec::new();
            for previous in &offsets {
                for value in choice {
                    next.push((previous.0 + value.0, previous.1 + value.1));
                }
            }
            offsets = next;
        }
        Ok(offsets)
    }

    fn access_marker(&self, tmem: &TileRegion) -> AResult<&'static str> {
        let info = self
            .kernel
            .ctx
            .inspect_layout(&tmem.buffer, &self.kernel.bindings)?;
        Ok(if info.allocated_addr_static.is_some() {
            "v2::tcgen05::variant::StaticTmem"
        } else {
            "v2::tcgen05::variant::DynamicTmem"
        })
    }

    fn is_static_zero(&self, value: &PrimExpr) -> bool {
        match self.static_int(value, "canonical region minimum") {
            Ok(value) => value == 0,
            Err(_) => false,
        }
    }

    /// Prove the exact m64 f32 `.16x256b.x1/.x8`
    /// mapping specialization.
    fn is_canonical_f32_m64_ld(
        &self,
        source: &TileRegion,
        destination: &TileRegion,
        row_mode: &str,
        raw_shape: &str,
        num: i64,
    ) -> AResult<bool> {
        let expected_extents: Option<[i64; 2]> = match num {
            1 => Some([64, 8]),
            8 => Some([64, 64]),
            _ => None,
        };
        let Some(expected_extents) = expected_extents else {
            return Ok(false);
        };
        if !((source.memory_scope == "tmem" && destination.memory_scope == "local")
            && source.dtype == "float32"
            && destination.dtype == "float32"
            && source.extents == expected_extents
            && destination.extents == expected_extents
            && row_mode == "direct"
            && raw_shape == "16x256b"
            && source.mins.len() == 2
            && destination.mins.len() == 2
            && self.is_static_zero(&source.mins[0])
            && destination
                .mins
                .iter()
                .all(|value| self.is_static_zero(value)))
        {
            return Ok(false);
        }
        let source_signature = self
            .kernel
            .ctx
            .inspect_layout(&source.buffer, &self.kernel.bindings)?
            .signature;
        if !TMEM_TO_LOCAL_M64_SOURCE_LAYOUTS.contains(&source_signature.as_str()) {
            return Ok(false);
        }
        let destination_signature = self
            .kernel
            .ctx
            .inspect_layout(&destination.buffer, &self.kernel.bindings)?
            .signature;
        if tmem_to_local_m64_destination_layout(num) != Some(destination_signature.as_str()) {
            return Ok(false);
        }
        Ok(self.access_marker(source)?.ends_with("::StaticTmem"))
    }

    /// Prove the standard warpgroup
    /// register/TMEM `.32x32b` mapping.
    fn is_canonical_32x32b_mapping(
        &self,
        register: &TileRegion,
        tmem: &TileRegion,
        row_mode: &str,
        raw_shape: &str,
        num: i64,
    ) -> AResult<bool> {
        if register.memory_scope != "local"
            || tmem.memory_scope != "tmem"
            || register.dtype != tmem.dtype
            || register.extents != tmem.extents
            || register.extents.len() != 2
            || register.extents[0] != 4 * 32
            || row_mode != "direct"
            || raw_shape != "32x32b"
            || num <= 0
            || !register.mins.iter().all(|value| self.is_static_zero(value))
            || !self.is_static_zero(&tmem.mins[0])
        {
            return Ok(false);
        }
        let register_info = self
            .kernel
            .ctx
            .inspect_layout(&register.buffer, &self.kernel.bindings)?;
        if register.extents[1] * register_info.itemsize != num * 4 {
            return Ok(false);
        }
        let proof = || -> AResult<bool> {
            let (_, row) = tir_var("numsim_canonical_st_row", "int64")?;
            let (_, column) = tir_var("numsim_canonical_st_column", "int64")?;
            let indices = |region: &TileRegion| -> AResult<Vec<PrimExpr>> {
                region
                    .mins
                    .iter()
                    .zip([&row, &column])
                    .map(|(minimum, coordinate)| offset_index(minimum, coordinate))
                    .collect()
            };
            let analyzer = tvm::analysis::Analyzer::new()?;
            let register_indices = indices(register)?;
            let owners = self
                .kernel
                .physical_owner_coordinates(&register.buffer, &register_indices)?;
            let axes: Vec<&str> = owners.iter().map(|(axis, _)| axis.as_str()).collect();
            if axes != ["tid_in_wg"] {
                return Ok(false);
            }
            let thread = &owners[0].1;
            let register_offset = self
                .kernel
                .physical_element_offset(&register.buffer, &register_indices)?;
            let cast_to = |target: &PrimExpr, value: &PrimExpr| -> AResult<PrimExpr> {
                Ok(Cast::new(PrimType::new(&dtype_text(target.dtype()))?, value.clone())?.into())
            };
            if !analyzer.can_prove_equal(thread, &cast_to(thread, &row)?)?
                || !analyzer
                    .can_prove_equal(&register_offset, &cast_to(&register_offset, &column)?)?
            {
                return Ok(false);
            }
            let (base_lane, base_tcol) = self
                .kernel
                .physical_tmem_coordinates(&tmem.buffer, &tmem.mins)?;
            let (mapped_lane, mapped_tcol) = self
                .kernel
                .physical_tmem_coordinates(&tmem.buffer, &indices(tmem)?)?;
            let lane_delta = op_binary("_OpSub", expr_any(&mapped_lane), expr_any(&base_lane))?;
            let tcol_delta = op_binary("_OpSub", expr_any(&mapped_tcol), expr_any(&base_tcol))?;
            Ok(
                analyzer.can_prove_equal(&lane_delta, &cast_to(&mapped_lane, &row)?)?
                    && analyzer.can_prove_equal(&tcol_delta, &cast_to(&mapped_tcol, &column)?)?,
            )
        };
        match proof() {
            Ok(value) => Ok(value),
            // An unsupported form or an FFI failure means the proof does not hold.
            Err(Failure::Unsupported { .. }) | Err(Failure::Ffi(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn allocation_only_views(
        &mut self,
        source: &TileRegion,
        destination: &TileRegion,
    ) -> AResult<(String, String, i64)> {
        let source_view = self.allocation_only_view(source, "tcgen_source_view")?;
        let destination_view = self.allocation_only_view(destination, "tcgen_destination_view")?;
        Ok((source_view, destination_view, source.element_count()))
    }

    fn allocation_only_view(&mut self, region: &TileRegion, prefix: &str) -> AResult<String> {
        let space = match region.memory_scope.as_str() {
            "local" => "v2::Register",
            "tmem" => "v2::Tmem",
            other => {
                return Err(Failure::Ffi(super::super::analyze::util::ffi_error(
                    &format!("TCGEN allocation-only operand in {other} space"),
                )))
            }
        };
        let view = self.kernel.control_name(prefix);
        let (buffer, logical_name) = match self.kernel.memory_plan.resolve(&region.buffer) {
            Ok(_) => (
                self.kernel.buffer_ref(&region.buffer)?,
                self.kernel.logical_buffer_name(&region.buffer)?,
            ),
            Err(Failure::Unsupported {
                message,
                unsupported: unsupported_items,
            }) => {
                if region.memory_scope != "tmem" {
                    return Err(Failure::Unsupported {
                        message,
                        unsupported: unsupported_items,
                    });
                }
                let physical = self
                    .kernel
                    .ctx
                    .inspect_layout(&region.buffer, &self.kernel.bindings)?;
                let mut matches: Vec<BufferVar> = Vec::new();
                for (candidate, code) in &self.kernel.buffers {
                    let plan = self.kernel.plan_of(code);
                    if plan.space == MemorySpace::Tmem
                        && same_allocation_only_tmem_view(&physical, &plan.layout)?
                    {
                        matches.push(candidate.clone());
                    }
                }
                if matches.len() != 1 {
                    return unsupported(
                        "allocation-only TCGEN operand has no unique declared physical TMEM carrier",
                    );
                }
                let carrier = matches.remove(0);
                (
                    self.kernel.buffer_ref(&carrier)?,
                    self.kernel.logical_buffer_name(&carrier)?,
                )
            }
            Err(error) => return Err(error),
        };
        let allocation = abi::call(
            &format!("MappedView::<{space}>::allocation_only"),
            &[named_buffer(space, &buffer, &logical_name)],
            &[],
            false,
            false,
        );
        self.kernel
            .emit_line(&format!("let {view} = {allocation};"));
        Ok(view)
    }

    fn regular_ref(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        lane: &str,
        target_rank: &str,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<String> {
        let itemsize = self
            .kernel
            .ctx
            .inspect_layout(&region.buffer, &self.kernel.bindings)?
            .itemsize;
        let index = self.kernel.physical_index_at_lane(
            &region.buffer,
            indices,
            lane,
            buffer_load,
            false,
        )?;
        let byte_offset = self.kernel.control_name("tcgen_regular_byte_offset");
        self.kernel.emit_line(&format!(
            "let {byte_offset} = i128::from({}).checked_mul({itemsize}_i128).ok_or_else(|| EngineError::message(\"TCGEN byte offset overflow\"))?;",
            index.code
        ));
        let space = if region.memory_scope == "shared" {
            "v2::Shared"
        } else {
            "v2::Register"
        };
        Ok(element_ref(
            space,
            "in_bounds",
            &[byte_offset, target_rank.to_owned()],
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn tmem_ref(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        lane: &str,
        target_rank: &str,
        lane_offset: i64,
        column_offset: i64,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<String> {
        let (mapped_lane, column, allocated) = self.kernel.tmem_coordinates_at_lane(
            &region.buffer,
            indices,
            lane,
            None,
            buffer_load,
        )?;
        Ok(element_ref(
            "v2::Tmem",
            "in_bounds_tmem",
            &[
                format!("{} + {lane_offset}_i64", mapped_lane.code),
                format!("{} + {column_offset}_i64", column.code),
                allocated.code,
                target_rank.to_owned(),
            ],
        ))
    }

    /// Capture one pure, stack-local map closure without
    /// materializing a table.
    fn capture_lazy_map(
        &mut self,
        region: &TileRegion,
        owner: &TileRegion,
        row_mode: &str,
        space: &str,
    ) -> AResult<Option<Capture>> {
        let source_op_id = self.source_op_id;
        self.kernel.capture_pure_closure(
            "lazy TCGEN element maps cannot read simulated memory",
            |emitter, reject| {
                let mut view = TcgenTransferEmitter {
                    kernel: emitter,
                    source_op_id,
                };
                let linear_name = view.kernel.control_name("tcgen_lazy_linear");
                let (variable, linear) = tir_var(&linear_name, "int64")?;
                view.kernel
                    .emit_line("let dimensions = logical.dimensions();");
                view.kernel.emit_line(
                    "if dimensions.len() != 1 { return Err(EngineError::message(\"TCGEN map rank mismatch\")); }",
                );
                view.kernel.emit_line(&format!(
                    "let {linear_name} = i64::try_from(dimensions[0]).map_err(|_| EngineError::message(\"TCGEN logical index overflow\"))?;"
                ));
                view.kernel.emit_line(&format!(
                    "if {linear_name} < 0_i64 || {linear_name} >= {}_i64 {{ return Err(EngineError::message(\"TCGEN logical index is out of bounds\")); }}",
                    region.element_count()
                ));
                let lane = view.kernel.control_name("tcgen_lazy_lane");
                view.kernel
                    .emit_line(&format!("let {lane} = map_lane.index();"));
                view.kernel.variables.set(
                    variable,
                    RustValue::new(linear_name.clone(), "i64", Uniformity::Uniform),
                );
                let indices = view.indices(
                    region,
                    &linear,
                    if region.memory_scope == "tmem" {
                        row_mode
                    } else {
                        "direct"
                    },
                )?;
                let owner_indices = view.indices(
                    owner,
                    &linear,
                    if owner.memory_scope == "tmem" {
                        row_mode
                    } else {
                        "direct"
                    },
                )?;
                let owns = if owner.memory_scope != "local" {
                    "true".to_owned()
                } else {
                    view.kernel.physical_access_predicate_at_lane(
                        &owner.buffer,
                        &owner_indices,
                        &lane,
                        Some(reject.clone()),
                    )?
                };
                view.kernel.emit_line(&format!("if !({owns}) {{"));
                view.kernel.indent += 1;
                let unowned = element_ref(space, "unowned", &["None".to_owned()]);
                view.kernel.emit_line(&format!("return Ok({unowned});"));
                view.kernel.indent -= 1;
                view.kernel.emit_line("}");
                let reference = if region.memory_scope == "tmem" {
                    view.tmem_ref(region, &indices, &lane, "None", 0, 0, Some(reject))?
                } else {
                    view.regular_ref(region, &indices, &lane, "None", Some(reject))?
                };
                view.kernel.emit_line(&format!("Ok({reference})"));
                Ok(())
            },
        )
    }

    /// Capture the frontend layout's exact owner
    /// lanes for one logical element.
    fn capture_lazy_owners(
        &mut self,
        owner: &TileRegion,
        row_mode: &str,
    ) -> AResult<Option<Capture>> {
        let source_op_id = self.source_op_id;
        self.kernel.capture_pure_closure(
            "lazy TCGEN ownership maps cannot read simulated memory",
            |emitter, reject| {
                let view = TcgenTransferEmitter {
                    kernel: emitter,
                    source_op_id,
                };
                let linear_name = view.kernel.control_name("tcgen_lazy_owner_linear");
                let (variable, linear) = tir_var(&linear_name, "int64")?;
                view.kernel
                    .emit_line("let dimensions = logical.dimensions();");
                view.kernel.emit_line(
                    "if dimensions.len() != 1 { return Err(EngineError::message(\"TCGEN owner map rank mismatch\")); }",
                );
                view.kernel.emit_line(&format!(
                    "let {linear_name} = i64::try_from(dimensions[0]).map_err(|_| EngineError::message(\"TCGEN logical index overflow\"))?;"
                ));
                view.kernel.emit_line(&format!(
                    "if {linear_name} < 0_i64 || {linear_name} >= {}_i64 {{ return Err(EngineError::message(\"TCGEN logical index is out of bounds\")); }}",
                    owner.element_count()
                ));
                view.kernel.variables.set(
                    variable,
                    RustValue::new(linear_name.clone(), "i64", Uniformity::Uniform),
                );
                let indices = view.indices(
                    owner,
                    &linear,
                    if owner.memory_scope == "tmem" {
                        row_mode
                    } else {
                        "direct"
                    },
                )?;
                let coordinates = view
                    .kernel
                    .physical_owner_coordinates(&owner.buffer, &indices)?;
                let mut axes: Vec<&str> = coordinates.iter().map(|(axis, _)| axis.as_str()).collect();
                axes.sort_unstable();
                let supported: [&[&str]; 4] = [&[], &["laneid"], &["tid_in_wg"], &["laneid", "wid_in_wg"]];
                if !supported.contains(&axes.as_slice()) {
                    return unsupported(format!(
                        "lazy TCGEN ownership requires no owner axis, laneid, tid_in_wg, or wid_in_wg+laneid; got [{}]",
                        axes.iter().map(|axis| (axis).to_string()).collect::<Vec<_>>().join(", ")
                    ));
                }
                let mut values: Vec<(String, RustValue)> = Vec::new();
                let coordinates_clone = coordinates.clone();
                let emitted = view.kernel.with_load_site(Some(reject), |emitter| {
                    for (axis, expression) in &coordinates_clone {
                        let value = emitter.emit_expr(&oref(expression.clone()))?;
                        let value = emitter.as_i64(value)?;
                        if value.uniformity != Uniformity::Uniform {
                            return unsupported(
                                "lazy TCGEN ownership requires lane-uniform logical coordinates",
                            );
                        }
                        values.push((axis.clone(), value));
                    }
                    Ok(())
                });
                emitted?;
                let value_of = |axis: &str| -> String {
                    values
                        .iter()
                        .find(|(name, _)| name == axis)
                        .map(|(_, value)| value.code.clone())
                        .expect("owner coordinate")
                };
                let warps = view.kernel.warps_per_warpgroup;
                if axes.is_empty() {
                    view.kernel.emit_line("Ok(v2::LaneMask::FULL)");
                } else if axes == ["laneid"] {
                    let owner_lane = view.kernel.control_name("tcgen_owner_lane");
                    view.kernel.emit_line(&format!(
                        "let {owner_lane} = usize::try_from({}).map_err(|_| EngineError::message(\"negative TCGEN owner lane\"))?;",
                        value_of("laneid")
                    ));
                    view.kernel
                        .emit_line(&format!("Ok({})", single_lane_mask(&owner_lane)));
                } else if axes == ["tid_in_wg"] {
                    let owner_thread = view.kernel.control_name("tcgen_owner_thread");
                    view.kernel.emit_line(&format!(
                        "let {owner_thread} = usize::try_from({}).map_err(|_| EngineError::message(\"negative TCGEN owner thread\"))?;",
                        value_of("tid_in_wg")
                    ));
                    view.kernel.emit_line(&format!(
                        "if {owner_thread} >= {}_usize {{ return Err(EngineError::message(\"TCGEN owner thread is outside the warpgroup\")); }}",
                        warps * 32
                    ));
                    view.kernel.emit_line(&format!(
                        "if {owner_thread} / WARP_SIZE == ctx.warp_id_in_cta() % {warps}_usize {{"
                    ));
                    view.kernel.indent += 1;
                    let lane = single_lane_mask(&format!("{owner_thread} % WARP_SIZE"));
                    view.kernel.emit_line(&format!("Ok({lane})"));
                    view.kernel.indent -= 1;
                    view.kernel.emit_line("} else {");
                    view.kernel.indent += 1;
                    view.kernel.emit_line("Ok(v2::LaneMask::EMPTY)");
                    view.kernel.indent -= 1;
                    view.kernel.emit_line("}");
                } else {
                    let owner_warp = view.kernel.control_name("tcgen_owner_warp");
                    let owner_lane = view.kernel.control_name("tcgen_owner_lane");
                    view.kernel.emit_line(&format!(
                        "let {owner_warp} = usize::try_from({}).map_err(|_| EngineError::message(\"negative TCGEN owner warp\"))?;",
                        value_of("wid_in_wg")
                    ));
                    view.kernel.emit_line(&format!(
                        "let {owner_lane} = usize::try_from({}).map_err(|_| EngineError::message(\"negative TCGEN owner lane\"))?;",
                        value_of("laneid")
                    ));
                    view.kernel.emit_line(&format!(
                        "if {owner_warp} >= {warps}_usize {{ return Err(EngineError::message(\"TCGEN owner warp is outside the warpgroup\")); }}"
                    ));
                    view.kernel.emit_line(&format!(
                        "if {owner_warp} == ctx.warp_id_in_cta() % {warps}_usize {{"
                    ));
                    view.kernel.indent += 1;
                    view.kernel
                        .emit_line(&format!("Ok({})", single_lane_mask(&owner_lane)));
                    view.kernel.indent -= 1;
                    view.kernel.emit_line("} else {");
                    view.kernel.indent += 1;
                    view.kernel.emit_line("Ok(v2::LaneMask::EMPTY)");
                    view.kernel.indent -= 1;
                    view.kernel.emit_line("}");
                }
                Ok(())
            },
        )
    }

    fn lazy_view(
        &mut self,
        capture: Capture,
        owner_capture: Capture,
        region: &TileRegion,
        space: &str,
        prefix: &str,
    ) -> AResult<String> {
        let (body, arguments) = capture;
        let (owner_body, owner_arguments) = owner_capture;
        let mapping = self.kernel.control_name(&format!("{prefix}_mapping"));
        self.kernel.emit_captured_closure(
            &mapping,
            &format!("#[inline(never)] move |logical: v2::LogicalCoord<'_>, map_lane: v2::LaneId| -> Result<v2::ElementRef<{space}>, EngineError> {{"),
            &body,
            &arguments,
        );

        let owners = self.kernel.control_name(&format!("{prefix}_owners"));
        self.kernel.emit_captured_closure(
            &owners,
            "move |logical: v2::LogicalCoord<'_>| -> Result<v2::LaneMask, EngineError> {",
            &owner_body,
            &owner_arguments,
        );

        let mapper = self.kernel.control_name(&format!("{prefix}_mapper"));
        self.kernel.emit_line(&format!(
            "let {mapper} = NumSimFnMap::<{space}>::new({mapping}, {owners});"
        ));
        let view = self.kernel.control_name(&format!("{prefix}_view"));
        let buffer = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(&view, space, &buffer, &buffer_name, &mapper, FN_MAP_STATE);
        self.kernel.emit_line(&line);
        Ok(view)
    }

    /// Emit pure lazy maps for one unexpanded TCGEN whole tile.
    fn emit_lazy_views(
        &mut self,
        source: &TileRegion,
        destination: &TileRegion,
        row_mode: &str,
    ) -> AResult<Option<(String, String, i64)>> {
        if source.element_count() != destination.element_count() {
            return Ok(None);
        }
        let count = source.element_count();
        let owner = if source.memory_scope == "local" {
            source
        } else {
            destination
        };
        let source_space = tcgen_space(&source.memory_scope);
        let destination_space = tcgen_space(&destination.memory_scope);
        let source_capture = self.capture_lazy_map(source, owner, row_mode, source_space)?;
        let destination_capture =
            self.capture_lazy_map(destination, owner, row_mode, destination_space)?;
        let source_owner_capture = self.capture_lazy_owners(owner, row_mode)?;
        let destination_owner_capture = self.capture_lazy_owners(owner, row_mode)?;
        let (
            Some(source_capture),
            Some(destination_capture),
            Some(source_owner_capture),
            Some(destination_owner_capture),
        ) = (
            source_capture,
            destination_capture,
            source_owner_capture,
            destination_owner_capture,
        )
        else {
            return Ok(None);
        };
        let source_view = self.lazy_view(
            source_capture,
            source_owner_capture,
            source,
            source_space,
            "tcgen_source",
        )?;
        let destination_view = self.lazy_view(
            destination_capture,
            destination_owner_capture,
            destination,
            destination_space,
            "tcgen_destination",
        )?;
        Ok(Some((source_view, destination_view, count)))
    }

    fn emit_views(
        &mut self,
        source: &TileRegion,
        destination: &TileRegion,
        row_mode: &str,
        cta_group: i64,
        replicas: &[(i64, i64)],
    ) -> AResult<(String, String, i64)> {
        if cta_group == 1 && replicas == [(0, 0)] {
            if let Some(lazy) = self.emit_lazy_views(source, destination, row_mode)? {
                return Ok(lazy);
            }
        }
        let base_count = destination.element_count();
        if source.element_count() != base_count {
            return unsupported("typed TCGEN transfer element counts differ");
        }
        let targets: Vec<String> = if cta_group == 2 {
            let pair_base = self.kernel.control_name("tcgen_pair_base");
            self.kernel.emit_line(&format!(
                "let {pair_base}: u32 = u32::try_from(ctx.cta_id_in_cluster() & !1_usize).map_err(|_| EngineError::message(\"TCGEN CTA rank exceeds u32\"))?;"
            ));
            vec![
                format!("Some({pair_base})"),
                format!("Some({pair_base} + 1_u32)"),
            ]
        } else {
            vec!["None".to_owned()]
        };
        let mut expansions: Vec<(String, (i64, i64))> = Vec::new();
        for target in &targets {
            for replica in replicas {
                expansions.push((target.clone(), *replica));
            }
        }
        let count = base_count * expansions.len() as i64;
        let source_space = tcgen_space(&source.memory_scope);
        let destination_space = if destination.memory_scope == "tmem" {
            "v2::Tmem"
        } else {
            "v2::Register"
        };
        let source_table = self.kernel.control_name("tcgen_source_map");
        let destination_table = self.kernel.control_name("tcgen_destination_map");
        let source_empty = element_ref(source_space, "unowned", &["None".to_owned()]);
        let destination_empty = element_ref(destination_space, "unowned", &["None".to_owned()]);
        self.kernel.emit_line(&format!(
            "let mut {source_table} = vec![{source_empty}; {count}_usize * 32_usize];"
        ));
        self.kernel.emit_line(&format!(
            "let mut {destination_table} = vec![{destination_empty}; {count}_usize * 32_usize];"
        ));

        let owner_is_source = source.memory_scope == "local";
        let owner = if owner_is_source { source } else { destination };
        let source_op_id = self.source_op_id;
        let looped = self.kernel.tile_linear_loop("tcgen_transfer_element", base_count, |emitter, linear| {
            let mut view = TcgenTransferEmitter {
                kernel: emitter,
                source_op_id,
            };
            let source_indices = view.indices(
                source,
                linear,
                if source.memory_scope == "tmem" {
                    row_mode
                } else {
                    "direct"
                },
            )?;
            let destination_indices = view.indices(
                destination,
                linear,
                if destination.memory_scope == "tmem" {
                    row_mode
                } else {
                    "direct"
                },
            )?;
            let owner_indices = if owner_is_source {
                &source_indices
            } else {
                &destination_indices
            };
            let linear_usize = view.kernel.control_name("tcgen_logical_linear");
            view.kernel.emit_line(&format!(
                "let {linear_usize} = usize::try_from({}).map_err(|_| EngineError::message(\"negative TCGEN logical index\"))?;",
                var_name(linear)?
            ));
            let lane = view.kernel.control_name("tcgen_map_lane");
            view.kernel
                .emit_line(&format!("for {lane} in ctx.active_mask() {{"));
            view.kernel.indent += 1;
            let owns = if owner.memory_scope != "local" {
                "true".to_owned()
            } else {
                view.kernel.physical_access_predicate_at_lane(
                    &owner.buffer,
                    owner_indices,
                    &lane,
                    None,
                )?
            };
            view.kernel.emit_line(&format!("if {owns} {{"));
            view.kernel.indent += 1;
            for (expansion, (target, (lane_offset, column_offset))) in expansions.iter().enumerate() {
                let slot = format!(
                    "({linear_usize} * {}_usize + {expansion}_usize) * 32_usize + {lane}",
                    expansions.len()
                );
                let source_target = if cta_group == 2 { target.as_str() } else { "None" };
                let source_ref = if source.memory_scope == "tmem" {
                    view.tmem_ref(source, &source_indices, &lane, source_target, 0, 0, None)?
                } else {
                    view.regular_ref(source, &source_indices, &lane, source_target, None)?
                };
                let destination_ref = if destination.memory_scope == "tmem" {
                    view.tmem_ref(
                        destination,
                        &destination_indices,
                        &lane,
                        target,
                        *lane_offset,
                        *column_offset,
                        None,
                    )?
                } else {
                    view.regular_ref(destination, &destination_indices, &lane, "None", None)?
                };
                view.kernel
                    .emit_line(&format!("{source_table}[{slot}] = {source_ref};"));
                view.kernel.emit_line(&format!(
                    "{destination_table}[{slot}] = {destination_ref};"
                ));
            }
            view.kernel.indent -= 1;
            view.kernel.emit_line("}");
            view.kernel.indent -= 1;
            view.kernel.emit_line("}");
            Ok(())
        });
        looped?;

        let source_view = self.kernel.control_name("tcgen_source_view");
        let destination_view = self.kernel.control_name("tcgen_destination_view");
        let source_buffer = self.kernel.buffer_ref(&source.buffer)?;
        let destination_buffer = self.kernel.buffer_ref(&destination.buffer)?;
        let source_mapper = match source.memory_scope.as_str() {
            "shared" => "NumSimSharedMap",
            "local" => "NumSimRegisterMap",
            _ => "NumSimTmemMap",
        };
        let destination_mapper = if destination.memory_scope == "tmem" {
            "NumSimTmemMap"
        } else {
            "NumSimRegisterMap"
        };
        let source_name = self.kernel.logical_buffer_name(&source.buffer)?;
        let line = mapped_view(
            &source_view,
            source_space,
            &source_buffer,
            &source_name,
            source_mapper,
            &table_state(&[count], &source_table),
        );
        self.kernel.emit_line(&line);
        let destination_name = self.kernel.logical_buffer_name(&destination.buffer)?;
        let line = mapped_view(
            &destination_view,
            destination_space,
            &destination_buffer,
            &destination_name,
            destination_mapper,
            &table_state(&[count], &destination_table),
        );
        self.kernel.emit_line(&line);
        Ok((source_view, destination_view, count))
    }

    /// `emit`.
    fn emit(&mut self, op: &ParsedTileCall) -> AResult<()> {
        if !is_tcgen_transfer(op) {
            return unsupported("TCGEN transfer emitter received another tile op");
        }
        let source = op.operands[0]
            .region()
            .expect("tcgen transfer source region")
            .clone();
        let destination = op.destination.clone();
        let pair = (
            source.memory_scope.as_str(),
            destination.memory_scope.as_str(),
        );
        let shared_to_tmem = pair == ("shared", "tmem");
        let cta_group = op.attr_int("cta_group").unwrap_or(1);
        if !shared_to_tmem && cta_group != 1 {
            return unsupported("typed tcgen05.ld/st only supports cta_group=1");
        }
        let row_mode = op.attr_str("tmem_row_mode").unwrap_or("direct").to_owned();
        let replicas = if shared_to_tmem {
            self.replica_offsets(&destination.buffer)?
        } else {
            vec![(0, 0)]
        };
        let raw_shape_name = op.attr_str("tcgen_shape").unwrap_or("").to_owned();
        let mut raw_shape = "";
        let mut num = 0;
        let mut canonical_f32_m64 = false;
        let mut canonical_32x32b_ld = false;
        let mut canonical_32x32b_st = false;
        if !shared_to_tmem {
            raw_shape = match raw_ldst_shape(&raw_shape_name) {
                Some(shape) => shape,
                None => {
                    return unsupported(format!(
                        "typed TCGEN transfer has no exact raw shape {:?}",
                        &raw_shape_name
                    ))
                }
            };
            num = op.attr_int("tcgen_num").unwrap_or(0);
            if num <= 0 {
                return unsupported("typed TCGEN transfer has no positive .xN variant");
            }
            if pair == ("tmem", "local") {
                canonical_f32_m64 = self.is_canonical_f32_m64_ld(
                    &source,
                    &destination,
                    &row_mode,
                    &raw_shape_name,
                    num,
                )?;
                canonical_32x32b_ld = self.is_canonical_32x32b_mapping(
                    &destination,
                    &source,
                    &row_mode,
                    &raw_shape_name,
                    num,
                )?;
            } else {
                canonical_32x32b_st = self.is_canonical_32x32b_mapping(
                    &source,
                    &destination,
                    &row_mode,
                    &raw_shape_name,
                    num,
                )?;
            }
        }
        let tmem = if source.memory_scope == "tmem" {
            &source
        } else {
            &destination
        };
        let (source_view, destination_view, count) =
            if canonical_f32_m64 || canonical_32x32b_ld || canonical_32x32b_st {
                self.allocation_only_views(&source, &destination)?
            } else {
                self.emit_views(&source, &destination, &row_mode, cta_group, &replicas)?
            };
        let (Some(source_element), Some(destination_element)) = (
            v2_tile_element(&source.dtype),
            v2_tile_element(&destination.dtype),
        ) else {
            return unsupported(format!(
                "typed TCGEN transfer has no element marker for {:?} -> {:?}",
                &source.dtype, &destination.dtype
            ));
        };
        let access = self.access_marker(tmem)?;
        let shape = format!("v2::tile::variant::Shape1<{count}>");
        let context = abi::context("ctx");
        let site = self.kernel.v2_site(Some(self.source_op_id));
        let mut instruction_args = "()".to_owned();

        let (variant, instruction) = if shared_to_tmem {
            let raw = format!(
                "v2::tcgen05::variant::Cp<v2::tcgen05::variant::Cp32x128bWarpx4, v2::tcgen05::variant::NoDecompress, {cta_group}, {access}>"
            );
            (
                format!(
                    "v2::tile::variant::Tcgen05Cp<{shape}, {source_element}, {destination_element}, {raw}>"
                ),
                "tcgen05_cp",
            )
        } else {
            let (raw, wrapper, instruction, mapping) = if pair == ("tmem", "local") {
                let raw = format!(
                    "v2::tcgen05::variant::Ld<v2::tcgen05::variant::{raw_shape}, v2::tcgen05::variant::Num<{num}>, false, {access}>"
                );
                let mapping = if canonical_f32_m64 || canonical_32x32b_ld {
                    let (base_lane, base_tcol, allocated_addr) = self
                        .kernel
                        .with_load_site(Some(NestedLoadSite::Site(None)), |emitter| {
                            emitter.tmem_coordinates(&source.buffer, &source.mins, None)
                        })?;
                    instruction_args =
                        coordinate_operands(&[&base_lane, &base_tcol, &allocated_addr]);
                    if canonical_f32_m64 {
                        ", v2::tile::variant::CanonicalF32M64"
                    } else {
                        ", v2::tile::variant::Canonical32x32b"
                    }
                } else {
                    ""
                };
                (raw, "Tcgen05Ld", "tcgen05_ld", mapping)
            } else {
                let raw = format!(
                    "v2::tcgen05::variant::St<v2::tcgen05::variant::{raw_shape}, v2::tcgen05::variant::Num<{num}>, false, {access}>"
                );
                let mapping = if canonical_32x32b_st {
                    let (base_lane, base_tcol, allocated_addr) = self.kernel.with_load_site(
                        Some(NestedLoadSite::Site(None)),
                        |emitter| {
                            emitter.tmem_coordinates(&destination.buffer, &destination.mins, None)
                        },
                    )?;
                    instruction_args =
                        coordinate_operands(&[&base_lane, &base_tcol, &allocated_addr]);
                    ", v2::tile::variant::Canonical32x32b"
                } else {
                    ""
                };
                (raw, "Tcgen05St", "tcgen05_st", mapping)
            };
            (
                format!(
                    "v2::tile::variant::{wrapper}<{shape}, {source_element}, {destination_element}, {raw}{mapping}>"
                ),
                instruction,
            )
        };

        self.kernel.emit_line("if !ctx.active_mask().is_empty() {");
        self.kernel.indent += 1;
        let mut operands = vec![format!("&{destination_view}"), format!("&{source_view}")];
        if instruction == "tcgen05_ld" || instruction == "tcgen05_st" {
            operands.push(instruction_args);
        }
        let call = abi::warp_call(
            &format!("tile::{instruction}"),
            &site,
            &operands,
            Some(&variant),
            Some(&context),
            false,
            true,
        );
        self.kernel.emit_line(&format!("{call};"));
        self.kernel.indent -= 1;
        self.kernel.emit_line("}");
        Ok(())
    }
}

/// The v2 space of one TCGEN transfer operand scope.
fn tcgen_space(scope: &str) -> &'static str {
    match scope {
        "shared" => "v2::Shared",
        "local" => "v2::Register",
        _ => "v2::Tmem",
    }
}

fn single_lane_mask(lane: &str) -> String {
    abi::call("LaneMask::single", &[lane.to_owned()], &[], true, false)
}

impl<'a> Emitter<'a> {
    pub fn emit_tile_tcgen_transfer(
        &mut self,
        op: &ParsedTileCall,
        source_op_id: i64,
    ) -> AResult<()> {
        TcgenTransferEmitter {
            kernel: self,
            source_op_id,
        }
        .emit(op)
    }
}
