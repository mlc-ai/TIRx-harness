//! `tirx.tile.gemm_async` as one typed whole-tile
//! v2 call.  The frontend compiles TIRx layouts into pure on-demand element
//! maps; the engine owns instruction partitioning, descriptor semantics,
//! effects, numeric execution, and completion.

use tvm::ir::PrimExpr;
use tvm::tirx::ComposeLayout;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCast};

use super::super::analyze::layout::{expr_any, int_any, op_binary, tile_layout_is_trivial};
use super::super::analyze::shapes::structural_equal;
use super::super::analyze::tile_forms::{
    any_value, AnyValue, GemmAsyncFacts, ParsedTileCall, TileOpKind, TileRegion, TileScalar,
};
use super::super::analyze::util::{ffi_error, int_imm_expr, same, unsupported, AResult, Failure};
use super::abi;
use super::abi::{element_ref, empty_table_state, mapped_view, table_state, FN_MAP_STATE};
use super::tile_common::{
    int64_imm, tile_logical_region_indices, tir_var, var_name, Capture, LoadFactory,
};
use super::NestedLoadSite;
use super::{Emitter, RustValue, Uniformity};

fn input_marker_of(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "float16" => "v2::reg::variant::F16",
        "bfloat16" => "v2::reg::variant::Bf16",
        "float8_e4m3fn" => "v2::tile::variant::E4m3",
        "float4_e2m1fn" => "v2::tile::variant::E2m1",
        _ => return None,
    })
}

fn scale_marker_of(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "float8_e8m0fnu" => "v2::tile::variant::E8m0",
        "float8_e4m3fn" => "v2::tile::variant::E4m3",
        _ => return None,
    })
}

impl<'a> Emitter<'a> {
    /// Lower a real TileScalar operand with exact source matching.
    pub(super) fn tile_explicit_expression(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        self.with_load_site(Some(NestedLoadSite::Exact), |emitter| {
            emitter.emit_expr(expr)
        })
    }

    /// One bound uniform `i64` coordinate per axis of `extents`.
    pub(super) fn tile_coordinates(
        &mut self,
        linear: &PrimExpr,
        extents: &[i64],
    ) -> AResult<Vec<PrimExpr>> {
        let linear_name = var_name(linear)?;
        let mut coordinates = Vec::new();
        for (axis, extent) in extents.iter().enumerate() {
            let stride: i64 = extents[axis + 1..].iter().product();
            let mut expression = linear_name.clone();
            if stride != 1 {
                expression = format!("({expression} / {stride}_i64)");
            }
            if *extent != 1 {
                expression = format!("({expression} % {extent}_i64)");
            } else {
                expression = "0_i64".to_owned();
            }
            let name = self.control_name(&format!("tile_coord_{axis}"));
            self.emit_line(&format!("let {name}: i64 = {expression};"));
            let (variable, coordinate) = tir_var(&name, "int64")?;
            self.variables
                .set(variable, RustValue::new(name, "i64", Uniformity::Uniform));
            coordinates.push(coordinate);
        }
        Ok(coordinates)
    }
}

/// Which element reference a GEMM operand view renders.
enum GemmReference<'r> {
    Matrix {
        region: &'r TileRegion,
        role: &'r str,
    },
    WsBatchedA {
        region: &'r TileRegion,
        m: i64,
    },
    Destination {
        region: &'r TileRegion,
        m: i64,
        n: i64,
        cta_group: i64,
        weight_stationary: bool,
    },
    Scale {
        region: &'r TileRegion,
        role: &'r str,
        mma_k: i64,
        scale_vector: i64,
        elements_per_instruction: i64,
        physical_elements_per_instruction: i64,
        has_descriptor: bool,
    },
}

struct GemmAsyncEmitter<'e, 'a> {
    kernel: &'e mut Emitter<'a>,
    source_op_id: i64,
}

impl<'e, 'a> GemmAsyncEmitter<'e, 'a> {
    fn logical_shape(region: &TileRegion) -> AResult<(i64, i64)> {
        let shape = region.logical_shape();
        if shape.len() != 2 {
            return unsupported(format!(
                "gemm_async mapped operand must be logical rank 2, got {:?}",
                &shape
            ));
        }
        Ok((shape[0], shape[1]))
    }

    fn table(&mut self, space: &str, shape: (i64, i64), role: &str) -> String {
        let count = shape.0 * shape.1;
        let table = self.kernel.control_name(&format!("gemm_{role}_map"));
        let empty = element_ref(space, "unowned", &["None".to_owned()]);
        self.kernel.emit_line(&format!(
            "let mut {table} = vec![{empty}; {count}_usize * 32_usize];"
        ));
        table
    }

    /// Recognize an exact whole-buffer BF16
    /// swizzled matrix.
    fn bf16_snapshot_parameters(
        &self,
        region: &TileRegion,
        rows: i64,
        columns: i64,
        transpose_storage: bool,
    ) -> AResult<Option<(i64, i64, i64, i64)>> {
        if transpose_storage
            || region.dtype != "bfloat16"
            || region.memory_scope != "shared"
            || region.extents.len() != 2
            || region.extents != [rows, columns]
            || rows * columns == 0
            || (rows * columns).count_ones() != 1
        {
            return Ok(None);
        }
        if region
            .mins
            .iter()
            .any(|value| int_imm_expr(value) != Some(0))
        {
            return Ok(None);
        }
        let shape: Vec<PrimExpr> = region.buffer.buffer_type().shape.iter().collect();
        if shape.len() != 2 || shape.iter().any(|value| int_imm_expr(value).is_none()) {
            return Ok(None);
        }
        let dims: Vec<i64> = shape
            .iter()
            .map(|value| int_imm_expr(value).expect("IntImm"))
            .collect();
        if dims != [rows, columns] {
            return Ok(None);
        }
        let info = self
            .kernel
            .ctx
            .inspect_layout(&region.buffer, &self.kernel.bindings)?;
        if info.element_count != Some(rows * columns) {
            return Ok(None);
        }
        let Some(layout) = region.buffer.buffer_type().layout.clone() else {
            return Ok(None);
        };
        let Ok(compose) = layout.try_cast::<ComposeLayout>() else {
            return Ok(None);
        };
        if !compose.swizzle_inner()? {
            return Ok(None);
        }
        let swizzle_len = i64::from(compose.swizzle_len()?);
        if ![1, 2, 3].contains(&swizzle_len) {
            return Ok(None);
        }
        let atom_columns = (16i64 << swizzle_len) / 2;
        let is_swizzle = tile_layout_is_trivial(&compose.tile_layout()?)?;
        if columns % atom_columns != 0 || (is_swizzle && columns != atom_columns) {
            return Ok(None);
        }
        let per_element_shift = i64::from(compose.per_element()?);
        let outer_mask = i64::from(compose.outer_mask()?);
        let atom_shift = i64::from(compose.atom_len()?);
        if !(0..64).contains(&per_element_shift) || !(0..64).contains(&atom_shift) {
            return Ok(None);
        }
        let quotient_domain = (rows * columns) >> per_element_shift;
        if outer_mask >= quotient_domain {
            return Ok(None);
        }
        Ok(Some((
            atom_columns,
            per_element_shift,
            outer_mask,
            atom_shift,
        )))
    }

    /// Carry an allocation for a specialization
    /// that does not call `map`.
    fn allocation_only_view(
        &mut self,
        region: &TileRegion,
        space: &str,
        mapper: &str,
        shape: (i64, i64),
        role: &str,
    ) -> AResult<String> {
        let view = self.kernel.control_name(&format!("gemm_{role}_view"));
        let buffer = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            space,
            &buffer,
            &buffer_name,
            mapper,
            &empty_table_state(&[shape.0, shape.1], space),
        );
        self.kernel.emit_line(&line);
        Ok(view)
    }

    /// `(marker, destination_tcol, allocation)`
    /// after a full proof.
    #[allow(clippy::too_many_arguments)]
    fn canonical_bf16_ss_mapping(
        &mut self,
        op: &ParsedTileCall,
        facts: &GemmAsyncFacts,
    ) -> AResult<Option<(String, String, String)>> {
        let (left, right) = (&facts.left, &facts.right);
        let (m, n, source_n, k) = (facts.m, facts.n, facts.source_n, facts.k);
        let accumulate_is_false =
            matches!(any_value(&facts.accumulate.expr)?, AnyValue::Bool(false));
        let eligible = !facts.weight_stationary
            && facts.scales.is_none()
            && facts.instruction_descriptor.is_none()
            && facts.predicate.is_none()
            && accumulate_is_false
            && facts.cta_group == 1
            && source_n == n
            && left.dtype == "bfloat16"
            && right.dtype == "bfloat16"
            && op.destination.dtype == "float32"
            && !facts.is_ab_tf32
            && (m == 64 || m == 128)
            && n >= 64
            && k >= 64;
        if !eligible {
            return Ok(None);
        }
        let Some(left_layout) = self.bf16_snapshot_parameters(left, m, k, facts.trans_a)? else {
            return Ok(None);
        };
        let Some(right_layout) =
            self.bf16_snapshot_parameters(right, source_n, k, facts.trans_b)?
        else {
            return Ok(None);
        };
        let mut reuse = m == source_n
            && same(&left.buffer, &right.buffer)
            && left.extents == right.extents
            && facts.trans_a == facts.trans_b;
        if reuse {
            for (left_min, right_min) in left.mins.iter().zip(right.mins.iter()) {
                if !structural_equal(&Any::from(left_min.clone()), &Any::from(right_min.clone()))? {
                    reuse = false;
                    break;
                }
            }
        }
        // The current loader is the
        // exact one without a site, and `tmem_coordinates` receives it.
        let destination = op.destination.clone();
        let (_, destination_tcol, destination_allocated_addr) = self
            .kernel
            .with_load_site(Some(NestedLoadSite::Site(None)), |emitter| {
                emitter.tmem_coordinates(&destination.buffer, &destination.mins, None)
            })?;
        let values = [
            left_layout.0,
            left_layout.1,
            left_layout.2,
            left_layout.3,
            right_layout.0,
            right_layout.1,
            right_layout.2,
            right_layout.3,
        ];
        let marker = format!(
            "v2::tile::variant::CanonicalBf16SsCta1<{}_usize, {}_u32, {}_usize, {}_u32, {}_usize, {}_u32, {}_usize, {}_u32, {}>",
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
            if reuse { "true" } else { "false" }
        );
        Ok(Some((
            marker,
            destination_tcol.code,
            destination_allocated_addr.code,
        )))
    }

    fn slot(&mut self, linear: &PrimExpr, lane: &str, role: &str) -> AResult<String> {
        let linear_usize = self.kernel.control_name(&format!("gemm_{role}_linear"));
        self.kernel.emit_line(&format!(
            "let {linear_usize} = usize::try_from({}).map_err(|_| EngineError::message(\"negative GEMM logical index\"))?;",
            var_name(linear)?
        ));
        Ok(format!("{linear_usize} * 32_usize + {lane}"))
    }

    fn regular_ref(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        lane: &str,
        role: &str,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<String> {
        let buffer_load =
            buffer_load.unwrap_or_else(|| NestedLoadSite::AtLaneSite(lane.to_owned(), None));
        let index = self.kernel.physical_index_at_lane(
            &region.buffer,
            indices,
            lane,
            Some(buffer_load),
            false,
        )?;
        if region.dtype == "float4_e2m1fn" {
            let element = self
                .kernel
                .control_name(&format!("gemm_{role}_packed_element"));
            let byte = self
                .kernel
                .control_name(&format!("gemm_{role}_packed_byte"));
            let bit = self.kernel.control_name(&format!("gemm_{role}_packed_bit"));
            self.kernel.emit_line(&format!(
                "let {element}: i128 = i128::from({});",
                index.code
            ));
            self.kernel
                .emit_line(&format!("let {byte}: i128 = {element}.div_euclid(2_i128);"));
            self.kernel.emit_line(&format!(
                "let {bit}: u8 = u8::try_from({element}.rem_euclid(2_i128) * 4_i128).map_err(|_| EngineError::message(\"GEMM packed bit offset overflow\"))?;"
            ));
            return Ok(element_ref(
                "v2::Shared",
                "in_bounds_bits",
                &[byte, bit, "None".to_owned()],
            ));
        }
        let itemsize = self
            .kernel
            .ctx
            .inspect_layout(&region.buffer, &self.kernel.bindings)?
            .itemsize;
        let byte = self
            .kernel
            .control_name(&format!("gemm_{role}_byte_offset"));
        self.kernel.emit_line(&format!(
            "let {byte}: i128 = i128::from({}).checked_mul({itemsize}_i128).ok_or_else(|| EngineError::message(\"GEMM byte offset overflow\"))?;",
            index.code
        ));
        Ok(element_ref(
            "v2::Shared",
            "in_bounds",
            &[byte, "None".to_owned()],
        ))
    }

    fn tmem_ref(
        &mut self,
        region: &TileRegion,
        indices: &[PrimExpr],
        lane: &str,
        role: &str,
        normalize_scale_cell: bool,
        buffer_load: Option<NestedLoadSite>,
    ) -> AResult<String> {
        let buffer_load =
            buffer_load.unwrap_or_else(|| NestedLoadSite::AtLaneSite(lane.to_owned(), None));
        let (mapped_lane, tcol, allocated) = self.kernel.tmem_coordinates_at_lane(
            &region.buffer,
            indices,
            lane,
            None,
            Some(buffer_load),
        )?;
        let mut tcol_code = tcol.code.clone();
        if normalize_scale_cell {
            let elem_offset = self
                .kernel
                .ctx
                .inspect_layout(&region.buffer, &self.kernel.bindings)?
                .elem_offset;
            let normalized = self.kernel.control_name(&format!("gemm_{role}_scale_cell"));
            self.kernel.emit_line(&format!(
                "let {normalized}: i64 = ({} + {elem_offset}_i64).div_euclid(4_i64) * 4_i64 - {elem_offset}_i64;",
                tcol.code
            ));
            tcol_code = normalized;
        }
        Ok(element_ref(
            "v2::Tmem",
            "in_bounds_tmem",
            &[
                mapped_lane.code,
                tcol_code,
                allocated.code,
                "None".to_owned(),
            ],
        ))
    }

    /// The `reference(coordinates, lane, buffer_load_factory)` closures.
    fn reference(
        &mut self,
        reference: &GemmReference<'_>,
        coordinates: &[PrimExpr],
        lane: &str,
        factory: &LoadFactory,
    ) -> AResult<String> {
        match reference {
            GemmReference::Matrix { region, role } => {
                let indices = tile_logical_region_indices(region, coordinates)?;
                let buffer_load = factory.resolve(self.kernel);
                if region.memory_scope == "shared" {
                    self.regular_ref(region, &indices, lane, role, buffer_load)
                } else {
                    self.tmem_ref(region, &indices, lane, role, false, buffer_load)
                }
            }
            GemmReference::WsBatchedA { region, m } => {
                let flat_row = &coordinates[0];
                let column = &coordinates[1];
                let half = op_binary("_OpFloorDiv", expr_any(flat_row), expr_any(&int64_imm(*m)?))?;
                let row = op_binary("_OpFloorMod", expr_any(flat_row), expr_any(&int64_imm(*m)?))?;
                let indices = tile_logical_region_indices(region, &[half, row, column.clone()])?;
                // The factory itself is the loader of the TMEM reference, so a
                // runtime load rejects the capture.
                let buffer_load = match factory {
                    LoadFactory::None => None,
                    LoadFactory::Current => Some(NestedLoadSite::Reject(
                        "lazy GEMM element maps cannot read simulated memory".to_owned(),
                    )),
                    LoadFactory::Reject(message) => Some(NestedLoadSite::Reject(message.clone())),
                };
                self.tmem_ref(region, &indices, lane, "a", false, buffer_load)
            }
            GemmReference::Destination {
                region,
                m,
                n,
                cta_group,
                weight_stationary,
            } => {
                let row = &coordinates[0];
                let column = &coordinates[1];
                // The factory is read inside the scope, so it sees the exact loader.
                let region_ref: &TileRegion = region;
                let factory = factory.clone();
                let (_, base_tcol, allocated) =
                    self.kernel
                        .with_load_site(Some(NestedLoadSite::Site(None)), |emitter| {
                            let buffer_load = factory.resolve(emitter);
                            emitter.tmem_coordinates_at_lane(
                                &region_ref.buffer,
                                &region_ref.mins,
                                lane,
                                None,
                                buffer_load,
                            )
                        })?;
                let row_code = var_name(row)?;
                let column_code = var_name(column)?;
                let mapped_lane = self.kernel.control_name("gemm_destination_mapped_lane");
                let tcol = self.kernel.control_name("gemm_destination_tcol");
                let (lane_expression, column_expression) = if *m == 128 {
                    (row_code.clone(), column_code.clone())
                } else if *cta_group == 2 || *weight_stationary {
                    let half_n = n / 2;
                    (
                        format!("{row_code} + 64_i64 * ({column_code} / {half_n}_i64)"),
                        format!("{column_code} % {half_n}_i64"),
                    )
                } else {
                    (
                        format!("({row_code} / 16_i64) * 32_i64 + ({row_code} % 16_i64)"),
                        column_code.clone(),
                    )
                };
                self.kernel
                    .emit_line(&format!("let {mapped_lane}: i64 = {lane_expression};"));
                self.kernel.emit_line(&format!(
                    "let {tcol}: i64 = {} + ({column_expression});",
                    base_tcol.code
                ));
                Ok(element_ref(
                    "v2::Tmem",
                    "in_bounds_tmem",
                    &[mapped_lane, tcol, allocated.code, "None".to_owned()],
                ))
            }
            GemmReference::Scale {
                region,
                role,
                mma_k,
                scale_vector,
                elements_per_instruction,
                physical_elements_per_instruction,
                has_descriptor,
            } => {
                let row = &coordinates[0];
                let scale_column = &coordinates[1];
                let scaled = op_binary("_OpMul", expr_any(scale_column), int_any(*scale_vector))?;
                let instruction = op_binary("_OpFloorDiv", expr_any(&scaled), int_any(*mma_k))?;
                let offset = op_binary(
                    "_OpFloorMod",
                    expr_any(scale_column),
                    int_any(*elements_per_instruction),
                )?;
                let scaled_instruction = op_binary(
                    "_OpMul",
                    expr_any(&instruction),
                    int_any(*physical_elements_per_instruction),
                )?;
                let physical_column =
                    op_binary("_OpAdd", expr_any(&scaled_instruction), expr_any(&offset))?;
                let indices = tile_logical_region_indices(region, &[row.clone(), physical_column])?;
                let buffer_load = factory.resolve(self.kernel);
                self.tmem_ref(
                    region,
                    &indices,
                    lane,
                    role,
                    *has_descriptor && *physical_elements_per_instruction == 0,
                    buffer_load,
                )
            }
        }
    }

    /// Capture one pure frontend layout closure for a
    /// GEMM operand.
    fn capture_lazy_view(
        &mut self,
        shape: (i64, i64),
        reference: &GemmReference<'_>,
    ) -> AResult<Option<Capture>> {
        let source_op_id = self.source_op_id;
        self.kernel.capture_pure_closure(
            "lazy GEMM element maps cannot read simulated memory",
            |emitter, reject| {
                let mut view = GemmAsyncEmitter {
                    kernel: emitter,
                    source_op_id,
                };
                view.kernel
                    .emit_line("let dimensions = logical.dimensions();");
                view.kernel.emit_line(
                    "if dimensions.len() != 2 { return Err(EngineError::message(\"GEMM map rank mismatch\")); }",
                );
                let mut coordinates = Vec::new();
                for (axis, extent) in [shape.0, shape.1].iter().enumerate() {
                    let name = view.kernel.control_name("gemm_lazy_coordinate");
                    view.kernel
                        .emit_line(&format!("let {name}: i64 = dimensions[{axis}];"));
                    view.kernel.emit_line(&format!(
                        "if {name} < 0_i64 || {name} >= {extent}_i64 {{ return Err(EngineError::message(\"GEMM logical index is out of bounds\")); }}"
                    ));
                    let (variable, coordinate) = tir_var(&name, "int64")?;
                    view.kernel.variables.set(
                        variable,
                        RustValue::new(name, "i64", Uniformity::Uniform),
                    );
                    coordinates.push(coordinate);
                }
                let lane = view.kernel.control_name("gemm_lazy_lane");
                view.kernel
                    .emit_line(&format!("let {lane} = map_lane.index();"));
                let factory = match reject {
                    NestedLoadSite::Reject(message) => LoadFactory::Reject(message),
                    _ => unreachable!("capture_pure_closure hands over the rejecting loader"),
                };
                let element = view.reference(reference, &coordinates, &lane, &factory)?;
                view.kernel.emit_line(&format!("Ok({element})"));
                Ok(())
            },
        )
    }

    fn lazy_view(
        &mut self,
        region: &TileRegion,
        space: &str,
        role: &str,
        capture: Capture,
    ) -> AResult<String> {
        let (body, arguments) = capture;
        let mapping = self.kernel.control_name(&format!("gemm_{role}_mapping"));
        self.kernel.emit_captured_closure(
            &mapping,
            &format!("#[inline(never)] move |logical: v2::LogicalCoord<'_>, map_lane: v2::LaneId| -> Result<v2::ElementRef<{space}>, EngineError> {{"),
            &body,
            &arguments,
        );
        let mapper = self.kernel.control_name(&format!("gemm_{role}_mapper"));
        self.kernel.emit_line(&format!(
            "let {mapper} = NumSimFnMap::<{space}>::all_lanes({mapping});"
        ));
        let view = self.kernel.control_name(&format!("gemm_{role}_view"));
        let buffer = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(&view, space, &buffer, &buffer_name, &mapper, FN_MAP_STATE);
        self.kernel.emit_line(&line);
        Ok(view)
    }

    /// Materialize `reference` into a per-(element,
    /// lane) table view, evaluating the same reference the lazy path would
    /// have captured.
    #[allow(clippy::too_many_arguments)]
    fn materialized_view(
        &mut self,
        region: &TileRegion,
        role: &str,
        shape: (i64, i64),
        space: &str,
        mapper: &str,
        reference: &GemmReference<'_>,
        buffer_load_factory: &LoadFactory,
        reference_before_slot: bool,
    ) -> AResult<String> {
        let table = self.table(space, shape, role);
        let source_op_id = self.source_op_id;
        let looped = self.kernel.tile_linear_loop(
            &format!("gemm_{role}_element"),
            shape.0 * shape.1,
            |emitter, linear| {
                let mut view = GemmAsyncEmitter {
                    kernel: emitter,
                    source_op_id,
                };
                let coordinates = view.kernel.tile_coordinates(linear, &[shape.0, shape.1])?;
                let lane = view.kernel.control_name(&format!("gemm_{role}_lane"));
                view.kernel
                    .emit_line(&format!("for {lane} in ctx.active_mask() {{"));
                view.kernel.indent += 1;
                let (slot, element) = if reference_before_slot {
                    let element =
                        view.reference(reference, &coordinates, &lane, buffer_load_factory)?;
                    let slot = view.slot(linear, &lane, role)?;
                    (slot, element)
                } else {
                    let slot = view.slot(linear, &lane, role)?;
                    let element =
                        view.reference(reference, &coordinates, &lane, buffer_load_factory)?;
                    (slot, element)
                };
                view.kernel
                    .emit_line(&format!("{table}[{slot}] = {element};"));
                view.kernel.indent -= 1;
                view.kernel.emit_line("}");
                Ok(())
            },
        );
        looped?;
        let view = self.kernel.control_name(&format!("gemm_{role}_view"));
        let buffer = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            space,
            &buffer,
            &buffer_name,
            mapper,
            &table_state(&[shape.0, shape.1], &table),
        );
        self.kernel.emit_line(&line);
        Ok(view)
    }

    /// Emit one GEMM operand view, lazily if its map is provably pure.
    #[allow(clippy::too_many_arguments)]
    fn operand_view(
        &mut self,
        region: &TileRegion,
        role: &str,
        shape: (i64, i64),
        space: &str,
        mapper: &str,
        reference: &GemmReference<'_>,
        buffer_load_factory: LoadFactory,
        reference_before_slot: bool,
    ) -> AResult<String> {
        if let Some(capture) = self.capture_lazy_view(shape, reference)? {
            return self.lazy_view(region, space, role, capture);
        }
        self.materialized_view(
            region,
            role,
            shape,
            space,
            mapper,
            reference,
            &buffer_load_factory,
            reference_before_slot,
        )
    }

    fn matrix_view(&mut self, region: &TileRegion, role: &str) -> AResult<String> {
        let shape = Self::logical_shape(region)?;
        let (space, mapper) = match region.memory_scope.as_str() {
            "shared" => ("v2::Shared", "NumSimSharedMap"),
            "tmem" => ("v2::Tmem", "NumSimTmemMap"),
            other => return unsupported(format!("gemm_async {role} cannot map {:?}", other)),
        };
        let reference = GemmReference::Matrix { region, role };
        self.operand_view(
            region,
            role,
            shape,
            space,
            mapper,
            &reference,
            LoadFactory::None,
            false,
        )
    }

    /// Flatten the exact `[2,M,K]` Layout-E A view for
    /// the v2 WS ABI.
    fn ws_batched_a_view(&mut self, region: &TileRegion, m: i64, k: i64) -> AResult<String> {
        let shape = (2 * m, k);
        let reference = GemmReference::WsBatchedA { region, m };
        if let Some(capture) = self.capture_lazy_view(shape, &reference)? {
            return self.lazy_view(region, "v2::Tmem", "a", capture);
        }
        let table = self.table("v2::Tmem", shape, "a");
        let source_op_id = self.source_op_id;
        let looped =
            self.kernel
                .tile_linear_loop("gemm_a_element", shape.0 * shape.1, |emitter, linear| {
                    let mut view = GemmAsyncEmitter {
                        kernel: emitter,
                        source_op_id,
                    };
                    let coordinates = view.kernel.tile_coordinates(linear, &[shape.0, shape.1])?;
                    let lane = view.kernel.control_name("gemm_a_lane");
                    view.kernel
                        .emit_line(&format!("for {lane} in ctx.active_mask() {{"));
                    view.kernel.indent += 1;
                    let slot = view.slot(linear, &lane, "a")?;
                    let element =
                        view.reference(&reference, &coordinates, &lane, &LoadFactory::None)?;
                    view.kernel
                        .emit_line(&format!("{table}[{slot}] = {element};"));
                    view.kernel.indent -= 1;
                    view.kernel.emit_line("}");
                    Ok(())
                });
        looped?;
        let view = self.kernel.control_name("gemm_a_view");
        let buffer = self.kernel.buffer_ref(&region.buffer)?;
        let buffer_name = self.kernel.logical_buffer_name(&region.buffer)?;
        let line = mapped_view(
            &view,
            "v2::Tmem",
            &buffer,
            &buffer_name,
            "NumSimTmemMap",
            &table_state(&[shape.0, shape.1], &table),
        );
        self.kernel.emit_line(&line);
        Ok(view)
    }

    fn destination_view(
        &mut self,
        region: &TileRegion,
        m: i64,
        n: i64,
        cta_group: i64,
        weight_stationary: bool,
    ) -> AResult<String> {
        let reference = GemmReference::Destination {
            region,
            m,
            n,
            cta_group,
            weight_stationary,
        };
        self.operand_view(
            region,
            "destination",
            (m, n),
            "v2::Tmem",
            "NumSimTmemMap",
            &reference,
            LoadFactory::Current,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scale_view(
        &mut self,
        region: &TileRegion,
        role: &str,
        rows: i64,
        k: i64,
        mma_k: i64,
        scale_vector: i64,
        elements_per_instruction: i64,
        physical_elements_per_instruction: i64,
        has_descriptor: bool,
    ) -> AResult<String> {
        let columns = k / scale_vector;
        let reference = GemmReference::Scale {
            region,
            role,
            mma_k,
            scale_vector,
            elements_per_instruction,
            physical_elements_per_instruction,
            has_descriptor,
        };
        self.operand_view(
            region,
            role,
            (rows, columns),
            "v2::Tmem",
            "NumSimTmemMap",
            &reference,
            LoadFactory::None,
            false,
        )
    }

    fn bool_register(&mut self, scalar: &TileScalar, role: &str) -> AResult<String> {
        let expression = match any_value(&scalar.expr)? {
            AnyValue::Bool(value) => {
                return Ok(abi::splat(if value { "true" } else { "false" }));
            }
            AnyValue::Object(node) => node,
            other => {
                return Err(Failure::Ffi(ffi_error(&format!(
                    "gemm_async {role} is a {}, not a TIR expression",
                    other.type_name()
                ))))
            }
        };
        let value = self.kernel.tile_explicit_expression(&expression)?;
        let value = self.kernel.as_warp_value(value);
        let name = self.kernel.control_name(&format!("gemm_{role}"));
        if value.rust_type == "bool" {
            self.kernel.emit_line(&format!(
                "let {name} = v2_register(({}).clone());",
                value.code
            ));
        } else if ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"]
            .contains(&value.rust_type.as_str())
        {
            self.kernel.emit_line(&format!(
                "let {name} = v2_register(({}).clone()).map(|_, value| value != 0);",
                value.code
            ));
        } else {
            return unsupported(format!(
                "gemm_async {role} lowered to unsupported type {}",
                value.rust_type
            ));
        }
        Ok(name)
    }

    fn descriptor_register(&mut self, scalar: &TileScalar) -> AResult<String> {
        let expression = match any_value(&scalar.expr)? {
            AnyValue::Object(node) => node,
            other => {
                return Err(Failure::Ffi(ffi_error(&format!(
                    "gemm_async descI is a {}, not a TIR expression",
                    other.type_name()
                ))))
            }
        };
        let value = self.kernel.tile_explicit_expression(&expression)?;
        let value = self.kernel.as_warp_value(value);
        if value.rust_type != "u32" {
            return unsupported("gemm_async descI did not lower to uint32");
        }
        let name = self.kernel.control_name("gemm_descriptor");
        self.kernel.emit_line(&format!(
            "let {name} = v2_register(({}).clone());",
            value.code
        ));
        Ok(name)
    }

    fn input_marker(dtype: &str, tf32: bool) -> AResult<String> {
        if tf32 {
            return Ok("v2::tile::variant::Tf32".to_owned());
        }
        match input_marker_of(dtype) {
            Some(marker) => Ok(marker.to_owned()),
            None => unsupported(format!("gemm_async has no v2 input marker for {:?}", dtype)),
        }
    }

    /// `emit`.
    fn emit(&mut self, op: &ParsedTileCall) -> AResult<()> {
        if op.kind != TileOpKind::GemmAsync {
            return unsupported("GEMM emitter received another tile operation");
        }
        let facts = op.gemm_async()?;
        let (left, right) = (&facts.left, &facts.right);
        let accumulate = &facts.accumulate;
        let descriptor = facts.instruction_descriptor.as_ref();
        let predicate = facts.predicate.as_ref();
        let scales = facts.scales.as_ref();
        let (m, n, source_n, k) = (facts.m, facts.n, facts.source_n, facts.k);
        let cta_group = facts.cta_group;
        let weight_stationary = facts.weight_stationary;
        let mapping_form = facts.mapping_form;
        let ws_batched = mapping_form.ws_batched;
        let cta2_banked_a = mapping_form.cta2_banked_a;
        let has_descriptor = descriptor.is_some();
        let has_predicate = predicate.is_some();
        if op.exec_scope == "thread" {
            self.kernel
                .emit_line("if ctx.active_mask().len() != 1_usize {");
            self.kernel.emit_line(
                "    return Err(EngineError::message(\"thread-scope gemm_async requires exactly one active issuing lane\"));",
            );
            self.kernel.emit_line("}");
        }

        let accumulate_register = self.bool_register(accumulate, "accumulate")?;
        let descriptor_register = match descriptor {
            Some(descriptor) => Some(self.descriptor_register(descriptor)?),
            None => None,
        };
        let predicate_register = match predicate {
            Some(predicate) => Some(self.bool_register(predicate, "predicate")?),
            None => None,
        };

        let canonical = self.canonical_bf16_ss_mapping(op, facts)?;
        let (destination_view, left_view, right_view) = match &canonical {
            None => {
                let destination_view =
                    self.destination_view(&op.destination, m, n, cta_group, weight_stationary)?;
                let left_view = if ws_batched || cta2_banked_a {
                    self.ws_batched_a_view(left, m, k)?
                } else {
                    self.matrix_view(left, "a")?
                };
                let right_view = self.matrix_view(right, "b")?;
                (destination_view, left_view, right_view)
            }
            Some(_) => {
                let destination_view = self.allocation_only_view(
                    &op.destination,
                    "v2::Tmem",
                    "NumSimTmemMap",
                    (m, n),
                    "destination",
                )?;
                let left_view =
                    self.allocation_only_view(left, "v2::Shared", "NumSimSharedMap", (m, k), "a")?;
                let right_view = self.allocation_only_view(
                    right,
                    "v2::Shared",
                    "NumSimSharedMap",
                    (source_n, k),
                    "b",
                )?;
                (destination_view, left_view, right_view)
            }
        };

        let mut scale_views: Option<(String, String)> = None;
        if let Some((scale_a, scale_b, numbers)) = scales {
            let a_view = self.scale_view(
                scale_a,
                "scale_a",
                m,
                k,
                facts.instruction_k,
                numbers.vector,
                numbers.values_per_mma,
                numbers.a_elements_per_ki,
                has_descriptor,
            )?;
            let b_view = self.scale_view(
                scale_b,
                "scale_b",
                n,
                k,
                facts.instruction_k,
                numbers.vector,
                numbers.values_per_mma,
                numbers.b_elements_per_ki,
                has_descriptor,
            )?;
            scale_views = Some((a_view, b_view));
        }

        let (mode_base, scale_vector) = match scales {
            None => (
                match (has_descriptor, has_predicate) {
                    (false, false) => "Dense",
                    (false, true) => "DensePredicated",
                    (true, false) => "DenseDescriptor",
                    (true, true) => "DenseDescriptorPredicated",
                }
                .to_owned(),
                1,
            ),
            Some((scale_a, _, numbers)) => {
                let Some(scale_marker) = scale_marker_of(&scale_a.dtype) else {
                    return unsupported(format!(
                        "gemm_async has no v2 scale marker for {:?}",
                        &scale_a.dtype
                    ));
                };
                let mode_name = match (has_descriptor, has_predicate) {
                    (false, false) => "BlockScaled",
                    (false, true) => "BlockScaledPredicated",
                    (true, false) => "BlockScaledDescriptor",
                    (true, true) => "BlockScaledDescriptorPredicated",
                };
                (format!("{mode_name}<{scale_marker}>"), numbers.vector)
            }
        };
        let mode = format!("v2::tile::variant::{mode_base}");

        let is_ab_tf32 = facts.is_ab_tf32;
        let a_input = Self::input_marker(&left.dtype, is_ab_tf32)?;
        let b_input = Self::input_marker(&right.dtype, is_ab_tf32)?;
        let placement = if left.memory_scope == "tmem" {
            "v2::tile::variant::ATmem"
        } else {
            "v2::tile::variant::AShared"
        };
        let access = self.kernel.tmem_access_marker(&op.destination.buffer)?;
        let expected = facts.expected_instruction_descriptor;
        let mask: i64 = if descriptor.is_none() || scales.is_none() {
            0xFFFF_FFFF
        } else {
            0x9FFF_FFCF
        };
        let mapping_marker: Option<String> = match &canonical {
            Some((marker, _, _)) => Some(marker.clone()),
            None => mapping_form.variant_marker.map(str::to_owned),
        };
        let variant = format!(
            "v2::tile::variant::Gemm<{mode}, {a_input}, {b_input}, {placement}, {access}, {m}, {n}, {k}, {}, {}, {}, {cta_group}, {:?}, {:?}, {scale_vector}, 0x{expected:08x}_u32, 0x{mask:08x}_u32{}",
            facts.instruction_m,
            facts.instruction_n,
            facts.instruction_k,
            facts.trans_a,
            facts.trans_b,
            match &mapping_marker {
                None => ">".to_owned(),
                Some(marker) => format!(", {marker}>"),
            }
        );

        let mut args: Vec<String> = Vec::new();
        if let Some((a_view, b_view)) = &scale_views {
            args.push(a_view.clone());
            args.push(b_view.clone());
        }
        args.push(accumulate_register);
        if let Some(register) = descriptor_register {
            args.push(register);
        }
        if let Some(register) = predicate_register {
            args.push(register);
        }
        if let Some((_, tcol, allocation)) = &canonical {
            args.push(format!("v2_register(({tcol}).clone())"));
            args.push(format!("v2_register(({allocation}).clone())"));
        }
        let argument = if args.len() == 1 {
            args[0].clone()
        } else {
            format!("({})", args.join(", "))
        };
        let instruction = if weight_stationary {
            "gemm_async_ws"
        } else {
            "gemm_async"
        };
        let site = self.kernel.v2_site(Some(self.source_op_id));
        let call = abi::warp_call(
            &format!("tile::{instruction}"),
            &site,
            &[
                format!("&{destination_view}"),
                format!("&{left_view}"),
                format!("&{right_view}"),
                argument,
            ],
            Some(&variant),
            None,
            false,
            true,
        );
        self.kernel.emit_line(&format!("{call};"));
        Ok(())
    }
}

impl<'a> Emitter<'a> {
    pub fn emit_tile_gemm_async(&mut self, op: &ParsedTileCall, source_op_id: i64) -> AResult<()> {
        GemmAsyncEmitter {
            kernel: self,
            source_op_id,
        }
        .emit(op)
    }
}
