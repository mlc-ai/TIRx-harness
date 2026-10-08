//! TCGEN layout proofs.

use tvm::analysis::Analyzer;
use tvm::ir::{IntImm, PrimExpr, PrimType, Range, Var};
use tvm::tirx::{ComposeLayout, Layout, TileLayout};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{
    self as tvm_ffi, Any, Array, DLDataType, DLDataTypeExt, ObjectRefCast, ObjectRefCore,
    String as FfiString,
};

use super::super::layout::{expr_any, int_any, op_binary, tile_layout_is_trivial};
use super::super::util::{
    ffi_error, ffi_text, oref, prim, repr_text, unsupported, AResult, Failure,
};
use super::super::Ctx;
use super::coordinates::{
    axes_are, axes_subset, int32_var, int_expr, physical_axes, physical_get,
    physical_layout_coordinates, physical_layout_coordinates_at, prove_equal, simplify_val, vadd,
    vexpr, vfloordiv, vfloormod, vint, vmul, vsub, Physical, Val,
};
use super::parse::{static_int_expr, unmodeled_tile_form};
use super::TileRegion;

pub(super) fn validate_block_scale_layout(
    region: &TileRegion,
    role: &str,
    rows: i64,
    k_iters: i64,
    sf_per_mma: i64,
) -> AResult<(i64, Vec<PrimExpr>)> {
    if k_iters <= 0 {
        return unsupported(
            "TilePrimitiveCall(gemm_async): block-scale K must contain at least one MMA step",
        );
    }
    let scale_cols = region.logical_shape()[1];
    if scale_cols <= 0 {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {role} scale K extent must be positive"
        ));
    }
    let elements_per_ki = scale_cols / k_iters;
    let analyzer = Analyzer::new()?;
    let mut scale_ids: Vec<PrimExpr> = Vec::new();
    let elements_per_tmem_cell = 4;
    for ki in 0..k_iters {
        let base_col = ki * elements_per_ki;
        if base_col + sf_per_mma > scale_cols {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): block-scale region does not contain the {role} bytes addressed by ki={ki}: base={base_col}, width={sf_per_mma}, extent={scale_cols}"
            ));
        }
        let physical_base =
            physical_layout_coordinates(region, &vint(0), &vint(base_col), "gemm_async")?;
        if !axes_are(&physical_base, &["TLane", "TCol"]) {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): block-scale TMEM layout must map exactly to TLane/TCol for {role}, got {:?}",
                &physical_axes(&physical_base)));
        }
        let base_lane = analyzer.simplify(physical_get(&physical_base, "TLane").expect("TLane"))?;
        let base_tcol = analyzer.simplify(physical_get(&physical_base, "TCol").expect("TCol"))?;
        if !prove_equal(&analyzer, &vexpr(&base_lane), &vint(0))? {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): block-scale instruction bases must start at TLane 0 for {role}, got {} at ki={ki}",
                repr_text(&oref(base_lane.clone()))?
            ));
        }
        let scale_id = simplify_val(
            &analyzer,
            &vfloormod(&vexpr(&base_tcol), &vint(elements_per_tmem_cell))?,
        )?;
        let non_negative =
            analyzer.can_prove(&op_binary("_OpGE", expr_any(&scale_id), int_any(0))?)?;
        let in_cell = analyzer.can_prove(&op_binary(
            "_OpLE",
            expr_any(&op_binary(
                "_OpAdd",
                expr_any(&scale_id),
                int_any(sf_per_mma),
            )?),
            int_any(elements_per_tmem_cell),
        )?)?;
        if !non_negative || !in_cell {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): block-scale instruction bytes cross a 32-bit TMEM cell for {role} at ki={ki}: sf_id={}, width={sf_per_mma}",
                repr_text(&oref(scale_id.clone()))?
            ));
        }
        scale_ids.push(scale_id);
        for row in 0..rows {
            for scale_offset in 0..sf_per_mma {
                let physical = physical_layout_coordinates(
                    region,
                    &vint(row),
                    &vint(base_col + scale_offset),
                    "gemm_async",
                )?;
                if !axes_are(&physical, &["TLane", "TCol"]) {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): block-scale TMEM layout must map exactly to TLane/TCol for {role}, got {:?}",
                        &physical_axes(&physical)));
                }
                let expected_lane = row % 32;
                let expected_tcol = simplify_val(
                    &analyzer,
                    &vadd(
                        &vadd(
                            &vexpr(&base_tcol),
                            &vint((row / 32) * elements_per_tmem_cell),
                        )?,
                        &vint(scale_offset),
                    )?,
                )?;
                let lane = physical_get(&physical, "TLane").expect("TLane");
                let tcol = physical_get(&physical, "TCol").expect("TCol");
                if !prove_equal(&analyzer, &vexpr(lane), &vint(expected_lane))?
                    || !analyzer.can_prove_equal(tcol, &expected_tcol)?
                {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): block-scale TMEM layout does not match the instruction ABI for {role} at logical ({row}, {}); expected (TLane={expected_lane}, TCol={}), got (TLane={}, TCol={})",
                        base_col + scale_offset,
                        repr_text(&oref(expected_tcol.clone()))?,
                        repr_text(&oref(lane.clone()))?,
                        repr_text(&oref(tcol.clone()))?
                    ));
                }
            }
        }
    }
    Ok((elements_per_ki, scale_ids))
}

fn layout_kind_name(layout: &Option<Layout>) -> &'static str {
    match layout {
        None => "NoneType",
        Some(layout) => {
            if layout.clone().try_cast::<TileLayout>().is_ok() {
                "TileLayout"
            } else if layout.clone().try_cast::<ComposeLayout>().is_ok() {
                "ComposeLayout"
            } else {
                "Layout"
            }
        }
    }
}

/// `layout.apply(*variables, shape=list(buffer.shape))`.
fn apply_layout(
    layout: &Layout,
    variables: &[PrimExpr],
    shape: &Array<PrimExpr>,
) -> Result<Physical, String> {
    match layout.apply_with_shape(&Array::new(variables.to_vec()), shape) {
        Ok(map) => Ok(map
            .iter()
            .map(|(axis, value)| (ffi_text(&axis), value))
            .collect()),
        Err(error) => Err(error.message().to_owned()),
    }
}

pub(super) fn validate_tcgen_smem_layout(
    ctx: &Ctx,
    region: &TileRegion,
    role: &str,
) -> AResult<()> {
    let buffer = &region.buffer;
    let buffer_type = buffer.buffer_type();
    let shape_exprs: Vec<PrimExpr> = buffer_type.shape.iter().collect();
    let mut shape = Vec::new();
    for (axis, extent) in shape_exprs.iter().enumerate() {
        shape.push(static_int_expr(
            &ctx.analyzer,
            extent,
            &format!("gemm_async.{role}.shape[{axis}]"),
        )?);
    }
    if shape.len() < 2 || shape.iter().any(|extent| *extent <= 0) {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {role} shared buffer must have a positive rank-2-or-higher shape, got {:?}",
            &shape));
    }
    let layout = buffer_type.layout.clone();
    let layout_kind = layout_kind_name(&layout);
    let dtype = DLDataType::try_from_str(&region.dtype)?;
    let bits = i64::from(dtype.bits);
    if dtype.lanes != 1 || ![4, 8, 16, 32].contains(&bits) {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {role} shared dtype {} has no supported TCGEN descriptor element width",
            region.dtype
        ));
    }
    let mut variables: Vec<PrimExpr> = Vec::new();
    for (axis, extent) in shape_exprs.iter().enumerate() {
        if shape[axis] == 1 {
            variables.push(IntImm::new("int32", 0)?.into());
        } else {
            variables.push(prim(&oref(Var::with_type(
                &format!("numsim_{}_{axis}", role.to_lowercase()),
                PrimType::from_dtype(extent.dtype())?,
            )))?);
        }
    }
    let rank = shape.len();
    let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
    let row = &variables[rank - 2];
    let col = &variables[rank - 1];
    let origin_variables = |variables: &[PrimExpr]| -> AResult<Vec<PrimExpr>> {
        let mut origin: Vec<PrimExpr> = variables[..rank - 2].to_vec();
        origin.push(IntImm::from_dtype(row.dtype(), 0)?.into());
        origin.push(IntImm::from_dtype(col.dtype(), 0)?.into());
        Ok(origin)
    };
    let bind_variables = |analyzer: &Analyzer| -> AResult<()> {
        for (variable, extent) in variables.iter().zip(shape.iter()) {
            if variable.as_node::<tvm::ir::VarObj>().is_some() {
                let var = variable.clone().try_cast::<Var>()?;
                analyzer.bind(
                    &var,
                    &Range::from_min_extent(int_expr(0)?, int_expr(*extent)?)?,
                )?;
            }
        }
        Ok(())
    };
    if layout_kind == "TileLayout" {
        let layout = layout.clone().expect("tile layout");
        if bits != 16 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} no-swizzle shared operands require a 16-bit dtype, got {}",
                region.dtype
            ));
        }
        let elements_per_16b = 128 / bits;
        if cols % elements_per_16b != 0 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} no-swizzle shared shape {:?} must have a contiguous extent divisible by {elements_per_16b}",
                &shape));
        }
        let mapped = match apply_layout(&layout, &variables, &buffer_type.shape) {
            Ok(mapped) => mapped,
            Err(error) => {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm_async): cannot apply {role} no-swizzle descriptor layout: {error}"
                ))
            }
        };
        if !axes_are(&mapped, &["m"]) {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} no-swizzle descriptor layout must map to one physical memory axis, got {:?}",
                &physical_axes(&mapped)));
        }
        let origin = match apply_layout(&layout, &origin_variables(&variables)?, &buffer_type.shape) {
            Ok(origin) => origin,
            Err(error) => {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm_async): cannot apply {role} no-swizzle descriptor origin: {error}"
                ))
            }
        };
        let expected_delta = vadd(
            &vadd(
                &vmul(&vexpr(row), &vint(elements_per_16b))?,
                &vmul(
                    &vmul(
                        &vfloordiv(&vexpr(col), &vint(elements_per_16b))?,
                        &vint(rows),
                    )?,
                    &vint(elements_per_16b),
                )?,
            )?,
            &vfloormod(&vexpr(col), &vint(elements_per_16b))?,
        )?;
        let layout_analyzer = Analyzer::new()?;
        bind_variables(&layout_analyzer)?;
        let proven = axes_are(&origin, &["m"])
            && prove_equal(
                &layout_analyzer,
                &vsub(
                    &vexpr(physical_get(&mapped, "m").expect("m")),
                    &vexpr(physical_get(&origin, "m").expect("m")),
                )?,
                &expected_delta,
            )?;
        if !proven {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} TileLayout is not the descriptor-encodable packed-16B no-swizzle layout within each leading-index slice"
            ));
        }
    } else if layout_kind != "ComposeLayout" {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {role} shared operand requires an explicit MMA ComposeLayout or packed-16B no-swizzle TileLayout, got {layout_kind}"
        ));
    } else {
        let swizzle = layout
            .clone()
            .expect("compose layout")
            .try_cast::<ComposeLayout>()?;
        let tile_layout = if tile_layout_is_trivial(&swizzle.tile_layout()?)? {
            None
        } else {
            Some(swizzle.tile_layout()?)
        };
        let swizzle_len = i64::from(swizzle.swizzle_len()?);
        let expected_mask = (1i64 << swizzle_len) - 1;
        let expected_per_element = i64::from(64 - ((128 / bits) as u64).leading_zeros()) - 1;
        if ![1, 2, 3].contains(&swizzle_len)
            || i64::from(swizzle.atom_len()?) != 3
            || i64::from(swizzle.per_element()?) != expected_per_element
            || !swizzle.swizzle_inner()?
            || i64::from(swizzle.inner_mask()?) != expected_mask
            || i64::from(swizzle.outer_mask()?) != expected_mask << 3
        {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} shared swizzle is not a 32B/64B/128B TCGEN matrix-descriptor atom"
            ));
        }
        let atom_bytes = 16i64 << swizzle_len;
        let atom_cols = atom_bytes * 8 / bits;
        if rows % 8 != 0 || cols % atom_cols != 0 {
            return unsupported(format!(
                "TilePrimitiveCall(gemm_async): {role} shared shape {:?} cannot tile {atom_bytes}B descriptor atoms (requires rows multiple of 8 and contiguous extent multiple of {atom_cols})",
                &shape));
        }
        match tile_layout {
            None => {
                if cols != atom_cols {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): {role} bare SwizzleLayout covers one {atom_cols}-element atom, but contiguous extent is {cols}"
                    ));
                }
            }
            Some(tile_layout) => {
                let tile_layout = Layout::from(tile_layout);
                let mapped = match apply_layout(&tile_layout, &variables, &buffer_type.shape) {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        return unsupported(format!(
                            "TilePrimitiveCall(gemm_async): cannot apply {role} descriptor tile: {error}"
                        ))
                    }
                };
                if !axes_are(&mapped, &["m"]) {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): {role} descriptor tile must map to one physical memory axis, got {:?}",
                        &physical_axes(&mapped)));
                }
                let origin =
                    match apply_layout(&tile_layout, &origin_variables(&variables)?, &buffer_type.shape) {
                        Ok(origin) => origin,
                        Err(error) => {
                            return unsupported(format!(
                                "TilePrimitiveCall(gemm_async): cannot apply {role} descriptor origin: {error}"
                            ))
                        }
                    };
                let expected_delta = vadd(
                    &vadd(
                        &vmul(
                            &vmul(&vfloordiv(&vexpr(col), &vint(atom_cols))?, &vint(rows))?,
                            &vint(atom_cols),
                        )?,
                        &vmul(&vexpr(row), &vint(atom_cols))?,
                    )?,
                    &vfloormod(&vexpr(col), &vint(atom_cols))?,
                )?;
                let layout_analyzer = Analyzer::new()?;
                bind_variables(&layout_analyzer)?;
                let proven = axes_are(&origin, &["m"])
                    && prove_equal(
                        &layout_analyzer,
                        &vsub(
                            &vexpr(physical_get(&mapped, "m").expect("m")),
                            &vexpr(physical_get(&origin, "m").expect("m")),
                        )?,
                        &expected_delta,
                    )?;
                if !proven {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): {role} outer layout does not tile {atom_bytes}B descriptor atoms in column-atom/row order within each leading-index slice"
                    ));
                }
            }
        }
    }
    let alignment_elements = 128 / bits;
    let contiguous_min = Analyzer::new()?.simplify(region.mins.last().expect("contiguous axis"))?;
    if !prove_equal(
        &Analyzer::new()?,
        &vfloormod(&vexpr(&contiguous_min), &vint(alignment_elements))?,
        &vint(0),
    )? {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): {role} contiguous-axis slice start must be 16B-aligned ({alignment_elements} {} elements), got {}",
            region.dtype,
            repr_text(&oref(contiguous_min.clone()))?
        ));
    }
    Ok(())
}

pub(super) fn validate_tcgen_tmem_a_layout(
    region: &TileRegion,
    rows: i64,
    cols: i64,
) -> AResult<()> {
    let analyzer = Analyzer::new()?;
    let origin = physical_layout_coordinates(region, &vint(0), &vint(0), "gemm_async")?;
    if !axes_are(&origin, &["TLane", "TCol"]) {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): TMEM A layout must map exactly to TLane/TCol, got {:?}",
            &physical_axes(&origin)
        ));
    }
    let origin_lane = physical_get(&origin, "TLane").expect("TLane").clone();
    let origin_tcol = physical_get(&origin, "TCol").expect("TCol").clone();
    let symbolic_row: PrimExpr = int32_var("numsim_tmem_a_row")?;
    let symbolic_col: PrimExpr = int32_var("numsim_tmem_a_col")?;
    let symbolic = physical_layout_coordinates_at(
        region,
        &[vexpr(&symbolic_row), vexpr(&symbolic_col)],
        "gemm_async",
    )?;
    if axes_are(&symbolic, &["TLane", "TCol"])
        && prove_equal(
            &analyzer,
            &vexpr(physical_get(&symbolic, "TLane").expect("TLane")),
            &vadd(&vexpr(&origin_lane), &vexpr(&symbolic_row))?,
        )?
        && prove_equal(
            &analyzer,
            &vexpr(physical_get(&symbolic, "TCol").expect("TCol")),
            &vadd(&vexpr(&origin_tcol), &vexpr(&symbolic_col))?,
        )?
    {
        return Ok(());
    }
    for row in 0..rows {
        for col in 0..cols {
            let physical =
                physical_layout_coordinates(region, &vint(row), &vint(col), "gemm_async")?;
            let valid_axes = axes_are(&physical, &["TLane", "TCol"]);
            let valid_lane = valid_axes
                && prove_equal(
                    &analyzer,
                    &vexpr(physical_get(&physical, "TLane").expect("TLane")),
                    &vadd(&vexpr(&origin_lane), &vint(row))?,
                )?;
            let valid_column = valid_axes
                && prove_equal(
                    &analyzer,
                    &vexpr(physical_get(&physical, "TCol").expect("TCol")),
                    &vadd(&vexpr(&origin_tcol), &vint(col))?,
                )?;
            if !(valid_lane && valid_column) {
                return unsupported(format!(
                    "TilePrimitiveCall(gemm_async): TMEM A layout does not match the instruction ABI at logical ({row}, {col}); expected TLane/TCol offsets ({row}, {col}) from the slice origin"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_ws_batched_tmem_layout(
    region: &TileRegion,
    rows: i64,
    columns: i64,
    role: &str,
) -> AResult<()> {
    let analyzer = Analyzer::new()?;
    let origin =
        physical_layout_coordinates_at(region, &[vint(0), vint(0), vint(0)], "gemm_async")?;
    if !axes_are(&origin, &["TLane", "TCol"]) {
        return unsupported(format!(
            "TilePrimitiveCall(gemm_async): batched {role} must map exactly to TLane/TCol, got {:?}",
            &physical_axes(&origin)));
    }
    let origin_lane = physical_get(&origin, "TLane").expect("TLane").clone();
    let origin_tcol = physical_get(&origin, "TCol").expect("TCol").clone();
    for half in 0..2 {
        for row in 0..rows {
            for column in 0..columns {
                let physical = physical_layout_coordinates_at(
                    region,
                    &[vint(half), vint(row), vint(column)],
                    "gemm_async",
                )?;
                let valid_axes = axes_are(&physical, &["TLane", "TCol"]);
                let valid_lane = valid_axes
                    && prove_equal(
                        &analyzer,
                        &vexpr(physical_get(&physical, "TLane").expect("TLane")),
                        &vadd(&vexpr(&origin_lane), &vint(half * 64 + row))?,
                    )?;
                let valid_column = valid_axes
                    && prove_equal(
                        &analyzer,
                        &vexpr(physical_get(&physical, "TCol").expect("TCol")),
                        &vadd(&vexpr(&origin_tcol), &vint(column))?,
                    )?;
                if !(valid_lane && valid_column) {
                    return unsupported(format!(
                        "TilePrimitiveCall(gemm_async): batched {role} layout does not match the two-lane-half weight-stationary ABI at logical ({half}, {row}, {column})"
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn matches_ws_packed_tmem_layout(
    region: &TileRegion,
    rows: i64,
    columns: i64,
) -> AResult<bool> {
    if rows != 64 || columns <= 0 || columns % 2 != 0 {
        return Ok(false);
    }
    let row: PrimExpr = int32_var("numsim_ws_packed_row")?;
    let column: PrimExpr = int32_var("numsim_ws_packed_column")?;
    let coordinates = (|| -> AResult<(Physical, Physical)> {
        let origin = physical_layout_coordinates_at(region, &[vint(0), vint(0)], "gemm_async")?;
        let mapped =
            physical_layout_coordinates_at(region, &[vexpr(&row), vexpr(&column)], "gemm_async")?;
        Ok((origin, mapped))
    })();
    let (origin, mapped) = match coordinates {
        Ok(pair) => pair,
        Err(Failure::Unsupported { .. }) => return Ok(false),
        Err(error) => return Err(error),
    };
    if !axes_are(&origin, &["TLane", "TCol"]) || !axes_are(&mapped, &["TLane", "TCol"]) {
        return Ok(false);
    }
    let half_columns = columns / 2;
    let analyzer = Analyzer::new()?;
    let lane_delta = vsub(
        &vexpr(physical_get(&mapped, "TLane").expect("TLane")),
        &vexpr(physical_get(&origin, "TLane").expect("TLane")),
    )?;
    let expected_lane = vadd(
        &vexpr(&row),
        &vmul(&vfloordiv(&vexpr(&column), &vint(half_columns))?, &vint(64))?,
    )?;
    let tcol_delta = vsub(
        &vexpr(physical_get(&mapped, "TCol").expect("TCol")),
        &vexpr(physical_get(&origin, "TCol").expect("TCol")),
    )?;
    let expected_tcol = vfloormod(&vexpr(&column), &vint(half_columns))?;
    Ok(prove_equal(&analyzer, &lane_delta, &expected_lane)?
        && prove_equal(&analyzer, &tcol_delta, &expected_tcol)?)
}

fn matches_coordinate_abi(actual: &[[Val; 4]], expected: &[[Val; 4]]) -> AResult<bool> {
    let analyzer = Analyzer::new()?;
    let mut bases: Option<Vec<PrimExpr>> = None;
    for (actual_coordinates, expected_coordinates) in actual.iter().zip(expected.iter()) {
        let mut offsets = Vec::new();
        for (value, expected_value) in actual_coordinates.iter().zip(expected_coordinates.iter()) {
            offsets.push(simplify_val(&analyzer, &vsub(value, expected_value)?)?);
        }
        match &bases {
            None => bases = Some(offsets),
            Some(bases) => {
                for (offset, base) in offsets.iter().zip(bases.iter()) {
                    if !analyzer.can_prove_equal(offset, base)? {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

fn matches_symbolic_coordinate_abi(
    actual: &[Val; 4],
    expected: &[Val; 4],
    actual_origin: &[Val; 4],
    expected_origin: &[Val; 4],
) -> AResult<bool> {
    let analyzer = Analyzer::new()?;
    for index in 0..4 {
        let lhs = simplify_val(&analyzer, &vsub(&actual[index], &expected[index])?)?;
        let rhs = simplify_val(
            &analyzer,
            &vsub(&actual_origin[index], &expected_origin[index])?,
        )?;
        if !analyzer.can_prove_equal(&lhs, &rhs)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The production `tcgen05_cp._build_plan` check
/// runs in TVM's Python package, so the native route stops here.
pub(super) fn validate_tcgen_cp_layout(
    node: &ObjectRef,
    source: &TileRegion,
    destination: &TileRegion,
) -> AResult<()> {
    let source_type = DLDataType::try_from_str(&source.dtype)?;
    let destination_type = DLDataType::try_from_str(&destination.dtype)?;
    if source_type.bits < 8 || destination_type.bits < 8 {
        return unmodeled_tile_form(
            "copy_async",
            "TilePrimitiveCall(copy_async): tcgen05.cp sub-byte payloads are not implemented",
        );
    }
    if source_type.bits / 8 != destination_type.bits / 8 {
        return unsupported(
            "TilePrimitiveCall(copy_async): tcgen05.cp source/destination physical element widths must match",
        );
    }
    let zeros: Vec<Val> = source.logical_shape().iter().map(|_| vint(0)).collect();
    let source_physical = physical_layout_coordinates_at(source, &zeros, "copy_async")?;
    if !axes_are(&source_physical, &["m"]) {
        return unsupported(format!(
            "TilePrimitiveCall(copy_async): tcgen05.cp source must have one physical memory axis, got {:?}",
            &physical_axes(&source_physical)));
    }
    // TVM's `tcgen05_cp._build_plan` stays the production oracle: Python
    // registers it as a global function returning its rejection text, if any.
    let planner = tvm_ffi::Function::get_global(NATIVE_TCGEN05_CP_PLAN)?;
    let error: Any = planner.call_tuple((Any::from(node.clone()),))?;
    let error = FfiString::try_from(error)?;
    if !error.as_str().is_empty() {
        return unsupported(format!(
            "TilePrimitiveCall(copy_async): tcgen05.cp layout is not accepted by production lowering: {}",
            error.as_str()
        ));
    }
    Ok(())
}

/// `native_frontend.NATIVE_TCGEN05_CP_PLAN`: TVM's copy-plan builder, registered by Python.
const NATIVE_TCGEN05_CP_PLAN: &str = "numsim.frontend.tcgen05_cp_plan_error";

fn tcgen_local_coordinates(physical: &Physical) -> AResult<Option<(Val, Val)>> {
    let slot = match physical_get(physical, "m") {
        Some(value) => vexpr(value),
        None => vint(0),
    };
    if axes_subset(physical, &["m", "tid_in_wg"]) {
        if let Some(thread) = physical_get(physical, "tid_in_wg") {
            return Ok(Some((vexpr(thread), slot)));
        }
    }
    if axes_subset(physical, &["m", "wid_in_wg", "laneid"])
        && (physical_get(physical, "wid_in_wg").is_some()
            || physical_get(physical, "laneid").is_some())
    {
        let warp = match physical_get(physical, "wid_in_wg") {
            Some(value) => vexpr(value),
            None => vint(0),
        };
        let lane = match physical_get(physical, "laneid") {
            Some(value) => vexpr(value),
            None => vint(0),
        };
        return Ok(Some((vadd(&vmul(&warp, &vint(32))?, &lane)?, slot)));
    }
    Ok(None)
}

fn tcgen_16xb_local_coordinates(
    row: &Val,
    col: &Val,
    rows: i64,
    cols: i64,
    fp32_atom_cols: i64,
    elements_per_register: i64,
) -> AResult<(Val, Val)> {
    let (warp, slab, row_in_atom) = if rows == 64 {
        (
            vfloordiv(row, &vint(16))?,
            vint(0),
            vfloormod(row, &vint(16))?,
        )
    } else {
        (
            vfloordiv(row, &vint(32))?,
            vfloordiv(&vfloormod(row, &vint(32))?, &vint(16))?,
            vfloormod(row, &vint(16))?,
        )
    };
    let fp32_cols = cols / elements_per_register;
    let col_fp32 = vfloordiv(col, &vint(elements_per_register))?;
    let packed_element = vfloormod(col, &vint(elements_per_register))?;
    let repetition = vfloordiv(&col_fp32, &vint(fp32_atom_cols))?;
    let col_in_atom = vfloormod(&col_fp32, &vint(fp32_atom_cols))?;
    let repetitions = fp32_cols / fp32_atom_cols;
    let row_mod_8 = vfloormod(&row_in_atom, &vint(8))?;
    let row_div_8 = vfloordiv(&row_in_atom, &vint(8))?;
    let (lane, registers_per_atom, register_slot) = match fp32_atom_cols {
        2 => (
            vadd(
                &vadd(&vmul(&vint(4), &row_mod_8)?, &vmul(&vint(2), &col_in_atom)?)?,
                &row_div_8,
            )?,
            1,
            repetition,
        ),
        4 => (
            vadd(&vmul(&vint(4), &row_mod_8)?, &col_in_atom)?,
            2,
            vadd(&row_div_8, &vmul(&vint(2), &repetition)?)?,
        ),
        8 => (
            vadd(
                &vmul(&vint(4), &row_mod_8)?,
                &vfloordiv(&col_in_atom, &vint(2))?,
            )?,
            4,
            vadd(
                &vadd(
                    &vmul(&vint(2), &row_div_8)?,
                    &vfloormod(&col_in_atom, &vint(2))?,
                )?,
                &vmul(&vint(4), &repetition)?,
            )?,
        ),
        _ => unreachable!("unsupported tcgen05 .16x*b atom width {fp32_atom_cols}"),
    };
    let register_slot = vadd(
        &register_slot,
        &vmul(&vmul(&slab, &vint(registers_per_atom))?, &vint(repetitions))?,
    )?;
    Ok((
        vadd(&vmul(&warp, &vint(32))?, &lane)?,
        vadd(
            &vmul(&register_slot, &vint(elements_per_register))?,
            &packed_element,
        )?,
    ))
}

fn tcgen_16xb_expected_coordinates(
    row: &Val,
    col: &Val,
    rows: i64,
    cols: i64,
    fp32_atom_cols: i64,
    elements_per_register: i64,
    logical_tmem_rows: bool,
) -> AResult<[Val; 4]> {
    let tmem_row = if logical_tmem_rows || rows != 64 {
        row.clone()
    } else {
        vadd(
            &vmul(&vfloordiv(row, &vint(16))?, &vint(32))?,
            &vfloormod(row, &vint(16))?,
        )?
    };
    let (lane, slot) =
        tcgen_16xb_local_coordinates(row, col, rows, cols, fp32_atom_cols, elements_per_register)?;
    Ok([tmem_row, col.clone(), lane, slot])
}

/// One candidate layout row of the tcgen05.ld/st layout validation.
struct LdstCandidate {
    shape: &'static str,
    num: i64,
    form: LdstCandidateForm,
}

enum LdstCandidateForm {
    /// `(row, col, row, col)`.
    Identity,
    /// Datapath B: `row + 64 * (col // half), col % half`.
    HalfFold { half: i64 },
    /// `.16x*b` register atoms.
    Sixteen {
        rows: i64,
        cols: i64,
        fp32_atom_cols: i64,
        elements_per_register: i64,
        logical_tmem_rows: bool,
    },
}

impl LdstCandidate {
    fn expected_at(&self, row: &Val, col: &Val) -> AResult<[Val; 4]> {
        match &self.form {
            LdstCandidateForm::Identity => Ok([row.clone(), col.clone(), row.clone(), col.clone()]),
            LdstCandidateForm::HalfFold { half } => {
                let folded_row = vadd(row, &vmul(&vint(64), &vfloordiv(col, &vint(*half))?)?)?;
                let folded_col = vfloormod(col, &vint(*half))?;
                Ok([
                    folded_row.clone(),
                    folded_col.clone(),
                    folded_row,
                    folded_col,
                ])
            }
            LdstCandidateForm::Sixteen {
                rows,
                cols,
                fp32_atom_cols,
                elements_per_register,
                logical_tmem_rows,
            } => tcgen_16xb_expected_coordinates(
                row,
                col,
                *rows,
                *cols,
                *fp32_atom_cols,
                *elements_per_register,
                *logical_tmem_rows,
            ),
        }
    }
}

fn tcgen_ldst_coordinates_at(
    tmem: &TileRegion,
    local: &TileRegion,
    row: &Val,
    col: &Val,
    tmem_row: Option<&Val>,
) -> AResult<Option<[Val; 4]>> {
    let tmem_physical =
        physical_layout_coordinates(tmem, tmem_row.unwrap_or(row), col, "copy_async")?;
    let local_physical = physical_layout_coordinates(local, row, col, "copy_async")?;
    if !axes_are(&tmem_physical, &["TLane", "TCol"]) {
        return Ok(None);
    }
    let Some((lane, slot)) = tcgen_local_coordinates(&local_physical)? else {
        return Ok(None);
    };
    Ok(Some([
        vexpr(physical_get(&tmem_physical, "TLane").expect("TLane")),
        vexpr(physical_get(&tmem_physical, "TCol").expect("TCol")),
        lane,
        slot,
    ]))
}

#[derive(Clone)]
pub struct TcgenLdstValidation {
    pub row_mode: &'static str,
    pub shape: &'static str,
    pub num: i64,
}

fn validate_tcgen_ldst_layout_uncached(
    ctx: &Ctx,
    tmem: &TileRegion,
    local: &TileRegion,
) -> AResult<TcgenLdstValidation> {
    let tmem_shape = tmem.logical_shape();
    if tmem_shape.len() != 2 {
        return unsupported(
            "TilePrimitiveCall(copy_async): tcgen05.ld/st TMEM operand must be logical rank 2",
        );
    }
    let (rows, cols) = (tmem_shape[0], tmem_shape[1]);
    if local.logical_shape() != [rows, cols] {
        return unsupported(
            "TilePrimitiveCall(copy_async): tcgen05.ld/st operands must have equal rank-2 shape",
        );
    }
    let tmem_buffer_rows = static_int_expr(
        &ctx.analyzer,
        &tmem.buffer.buffer_type().shape.get(0)?,
        "copy_async.tcgen05_ldst.rows",
    )?;
    let mut candidates: Vec<LdstCandidate> = Vec::new();
    let tmem_type = DLDataType::try_from_str(&tmem.dtype)?;
    let tmem_bits = i64::from(tmem_type.bits);
    let elements_per_register = 32 / tmem_bits;
    if rows == 128 && cols % elements_per_register == 0 {
        candidates.push(LdstCandidate {
            shape: "32x32b",
            num: cols / elements_per_register,
            form: LdstCandidateForm::Identity,
        });
    }
    if rows == 64 && tmem_bits == 32 && cols % 2 == 0 {
        let half = cols / 2;
        candidates.push(LdstCandidate {
            shape: "32x32b",
            num: half,
            form: LdstCandidateForm::HalfFold { half },
        });
    }
    if [16, 32].contains(&tmem_bits) && [64, 128].contains(&rows) {
        for fp32_atom_cols in [2, 4, 8] {
            let atom_cols = fp32_atom_cols * elements_per_register;
            if cols % atom_cols != 0 {
                continue;
            }
            let shape = match fp32_atom_cols {
                2 => "16x64b",
                4 => "16x128b",
                _ => "16x256b",
            };
            let num = cols / atom_cols;
            candidates.push(LdstCandidate {
                shape,
                num,
                form: LdstCandidateForm::Sixteen {
                    rows,
                    cols,
                    fp32_atom_cols,
                    elements_per_register,
                    logical_tmem_rows: false,
                },
            });
            if rows == 64 && tmem_buffer_rows == 128 {
                candidates.push(LdstCandidate {
                    shape,
                    num,
                    form: LdstCandidateForm::Sixteen {
                        rows,
                        cols,
                        fp32_atom_cols,
                        elements_per_register,
                        logical_tmem_rows: true,
                    },
                });
            }
        }
    }

    let symbolic_row = vexpr(&int32_var("tcgen_ldst_row")?);
    let symbolic_col = vexpr(&int32_var("tcgen_ldst_col")?);
    let symbolic = (|| -> AResult<(Option<[Val; 4]>, Option<[Val; 4]>)> {
        let symbolic = tcgen_ldst_coordinates_at(tmem, local, &symbolic_row, &symbolic_col, None)?;
        let origin = tcgen_ldst_coordinates_at(tmem, local, &vint(0), &vint(0), None)?;
        Ok((symbolic, origin))
    })();
    let (actual_direct_symbolic, actual_direct_origin) = match symbolic {
        Ok(pair) => pair,
        Err(Failure::Unsupported { .. }) => (None, None),
        Err(error) => return Err(error),
    };
    if let (Some(actual), Some(origin)) = (&actual_direct_symbolic, &actual_direct_origin) {
        for candidate in &candidates {
            if matches_symbolic_coordinate_abi(
                actual,
                &candidate.expected_at(&symbolic_row, &symbolic_col)?,
                origin,
                &candidate.expected_at(&vint(0), &vint(0))?,
            )? {
                return Ok(TcgenLdstValidation {
                    row_mode: "direct",
                    shape: candidate.shape,
                    num: candidate.num,
                });
            }
        }
    }

    let mut actual_m64_symbolic: Option<[Val; 4]> = None;
    let mut actual_m64_origin: Option<[Val; 4]> = None;
    if rows == 64 && tmem_buffer_rows >= 128 {
        let symbolic_remapped_row = vadd(
            &vmul(&vfloordiv(&symbolic_row, &vint(16))?, &vint(32))?,
            &vfloormod(&symbolic_row, &vint(16))?,
        )?;
        let remapped = (|| -> AResult<(Option<[Val; 4]>, Option<[Val; 4]>)> {
            let symbolic = tcgen_ldst_coordinates_at(
                tmem,
                local,
                &symbolic_row,
                &symbolic_col,
                Some(&symbolic_remapped_row),
            )?;
            let origin =
                tcgen_ldst_coordinates_at(tmem, local, &vint(0), &vint(0), Some(&vint(0)))?;
            Ok((symbolic, origin))
        })();
        match remapped {
            Ok((symbolic, origin)) => {
                actual_m64_symbolic = symbolic;
                actual_m64_origin = origin;
            }
            Err(Failure::Unsupported { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    if let (Some(actual), Some(origin)) = (&actual_m64_symbolic, &actual_m64_origin) {
        for candidate in &candidates {
            if matches_symbolic_coordinate_abi(
                actual,
                &candidate.expected_at(&symbolic_row, &symbolic_col)?,
                origin,
                &candidate.expected_at(&vint(0), &vint(0))?,
            )? {
                return Ok(TcgenLdstValidation {
                    row_mode: "m64_d_low_slab",
                    shape: candidate.shape,
                    num: candidate.num,
                });
            }
        }
    }

    let mut actual_direct: Vec<[Val; 4]> = Vec::new();
    let mut actual_m64_d: Vec<[Val; 4]> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let tmem_physical =
                physical_layout_coordinates(tmem, &vint(row), &vint(col), "copy_async")?;
            let local_physical =
                physical_layout_coordinates(local, &vint(row), &vint(col), "copy_async")?;
            if !axes_are(&tmem_physical, &["TLane", "TCol"]) {
                return unsupported(format!(
                    "TilePrimitiveCall(copy_async): tcgen05.ld/st TMEM operand must map exactly to TLane/TCol, got {:?}",
                    &physical_axes(&tmem_physical)));
            }
            let Some((lane, slot)) = tcgen_local_coordinates(&local_physical)? else {
                return unsupported(
                    "TilePrimitiveCall(copy_async): tcgen05.ld/st local operand has no recognized instruction fragment ownership",
                );
            };
            actual_direct.push([
                vexpr(physical_get(&tmem_physical, "TLane").expect("TLane")),
                vexpr(physical_get(&tmem_physical, "TCol").expect("TCol")),
                lane.clone(),
                slot.clone(),
            ]);
            if rows == 64 && tmem_buffer_rows >= 128 {
                let remapped_row = (row / 16) * 32 + row % 16;
                let remapped_tmem = physical_layout_coordinates(
                    tmem,
                    &vint(remapped_row),
                    &vint(col),
                    "copy_async",
                )?;
                actual_m64_d.push([
                    vexpr(physical_get(&remapped_tmem, "TLane").expect("TLane")),
                    vexpr(physical_get(&remapped_tmem, "TCol").expect("TCol")),
                    lane,
                    slot,
                ]);
            }
        }
    }
    let expected_grid = |candidate: &LdstCandidate| -> AResult<Vec<[Val; 4]>> {
        let mut expected = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                expected.push(candidate.expected_at(&vint(row), &vint(col))?);
            }
        }
        Ok(expected)
    };
    for candidate in &candidates {
        if matches_coordinate_abi(&actual_direct, &expected_grid(candidate)?)? {
            return Ok(TcgenLdstValidation {
                row_mode: "direct",
                shape: candidate.shape,
                num: candidate.num,
            });
        }
    }
    if !actual_m64_d.is_empty() {
        for candidate in &candidates {
            if matches_coordinate_abi(&actual_m64_d, &expected_grid(candidate)?)? {
                return Ok(TcgenLdstValidation {
                    row_mode: "m64_d_low_slab",
                    shape: candidate.shape,
                    num: candidate.num,
                });
            }
        }
    }
    unsupported(
        "TilePrimitiveCall(copy_async): TMEM/local layouts do not match any fixed tcgen05.ld/st D, F, or B instruction ABI",
    )
}

// ----------------------------------------------------------------------
// tcgen05.ld/st validation cache (one per analysis context).
// ----------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq)]
struct RegionValidationMetadata {
    dtype: String,
    memory_scope: String,
    extents: Vec<i64>,
    mins: usize,
    rank: usize,
}

fn region_validation_metadata(region: &TileRegion) -> RegionValidationMetadata {
    RegionValidationMetadata {
        dtype: region.dtype.clone(),
        memory_scope: region.memory_scope.clone(),
        extents: region.extents.clone(),
        mins: region.mins.len(),
        rank: region.buffer.buffer_type().shape.len(),
    }
}

/// One cached tcgen05.ld/st structural proof.
pub struct TcgenLdstCacheEntry {
    key: (RegionValidationMetadata, RegionValidationMetadata),
    tmem: TileRegion,
    local: TileRegion,
    result: TcgenLdstValidation,
}

fn any_list(items: Vec<Any>) -> Any {
    Any::from(Array::<Any>::new(items))
}

fn tcgen_ldst_validation_structure(tmem: &TileRegion, local: &TileRegion) -> Any {
    let region_structure = |region: &TileRegion| -> Any {
        let mut items: Vec<Any> = region.mins.iter().map(expr_any).collect();
        items.extend(
            region
                .buffer
                .buffer_type()
                .shape
                .iter()
                .map(|extent| expr_any(&extent)),
        );
        items.push(match region.buffer.buffer_type().layout.clone() {
            Some(layout) => Any::from(layout),
            None => Any::new(),
        });
        any_list(items)
    };
    any_list(vec![region_structure(tmem), region_structure(local)])
}

fn relative_physical_mapping_structure(region: &TileRegion, prefix: &str) -> AResult<Any> {
    let logical_rank = region.extents.iter().filter(|extent| **extent != 1).count();
    let variables: Vec<Val> = (0..logical_rank)
        .map(|axis| Ok(vexpr(&int32_var(&format!("{prefix}_{axis}"))?)))
        .collect::<AResult<_>>()?;
    let zeros: Vec<Val> = (0..logical_rank).map(|_| vint(0)).collect();
    let origin = physical_layout_coordinates_at(region, &zeros, "copy_async")?;
    let mapped = physical_layout_coordinates_at(region, &variables, "copy_async")?;
    if physical_axes(&origin) != physical_axes(&mapped) {
        return Err(Failure::Ffi(ffi_error(
            "physical layout axes change across the logical region",
        )));
    }
    let analyzer = Analyzer::new()?;
    let mut items: Vec<Any> = Vec::new();
    for axis in physical_axes(&mapped) {
        let delta = analyzer.simplify(&op_binary(
            "_OpSub",
            expr_any(physical_get(&mapped, &axis).expect("axis")),
            expr_any(physical_get(&origin, &axis).expect("axis")),
        )?)?;
        items.push(any_list(vec![
            Any::from(FfiString::from(axis.as_str())),
            expr_any(&delta),
        ]));
    }
    Ok(any_list(items))
}

fn relative_tcgen_ldst_validation_structure(tmem: &TileRegion, local: &TileRegion) -> AResult<Any> {
    Ok(any_list(vec![
        relative_physical_mapping_structure(tmem, "tmem")?,
        relative_physical_mapping_structure(local, "local")?,
        expr_any(&tmem.buffer.buffer_type().shape.get(0)?),
    ]))
}

/// `structural_equal(lhs, rhs, map_free_vars=True)`.
fn structural_equal_free_vars(lhs: &Any, rhs: &Any) -> AResult<bool> {
    let result: Any = tvm_ffi::cached_global_func!("ffi.StructuralEqual").call_tuple((
        lhs.clone(),
        rhs.clone(),
        true,
        false,
    ))?;
    Ok(bool::try_from(result)?)
}

/// Whether two validations have the same semantics (metadata already matched).
fn same_tcgen_ldst_validation_semantics(
    left_tmem: &TileRegion,
    left_local: &TileRegion,
    right_tmem: &TileRegion,
    right_local: &TileRegion,
) -> AResult<bool> {
    let relative = (|| -> AResult<bool> {
        structural_equal_free_vars(
            &relative_tcgen_ldst_validation_structure(left_tmem, left_local)?,
            &relative_tcgen_ldst_validation_structure(right_tmem, right_local)?,
        )
    })();
    if let Ok(true) = relative {
        return Ok(true);
    }
    structural_equal_free_vars(
        &tcgen_ldst_validation_structure(left_tmem, left_local),
        &tcgen_ldst_validation_structure(right_tmem, right_local),
    )
}

pub(super) fn validate_tcgen_ldst_layout(
    ctx: &Ctx,
    tmem: &TileRegion,
    local: &TileRegion,
) -> AResult<TcgenLdstValidation> {
    let key = (
        region_validation_metadata(tmem),
        region_validation_metadata(local),
    );
    {
        let entries = ctx.tcgen_ldst_validations.borrow();
        for entry in entries.iter().filter(|entry| entry.key == key) {
            if same_tcgen_ldst_validation_semantics(tmem, local, &entry.tmem, &entry.local)? {
                return Ok(entry.result.clone());
            }
        }
    }
    let result = validate_tcgen_ldst_layout_uncached(ctx, tmem, local)?;
    ctx.tcgen_ldst_validations
        .borrow_mut()
        .push(TcgenLdstCacheEntry {
            key,
            tmem: tmem.clone(),
            local: local.clone(),
            result: result.clone(),
        });
    Ok(result)
}
