//! Static buffer layout inspection.

use tvm::analysis::Analyzer;
use tvm::ir::{IntImm, PrimExpr, Var};
use tvm::tirx::{Axis, BufferVar, ComposeLayout, Iter, Layout, TileLayout};
use tvm::tvm_ffi::{
    self, Any, Array, DLDataType, DLDataTypeExt, Map, ObjectRefCast, String as FfiString,
};

use super::util::{
    buffer_dtype, buffer_name, buffer_scope, ffi_text, int_imm_expr, not_covered, oref, repr_of,
    simplify, static_int as shared_static_int, unsupported, AResult, Failure,
};
use crate::schema::Schema;

const MAX_PARTIAL_ATOM_ENUMERATION: i64 = 1_000_000;

/// Shared primitive operators used by the Python expression constructors.
pub fn op_binary(name: &str, lhs: Any, rhs: Any) -> AResult<PrimExpr> {
    let operator = name.strip_prefix("_Op").expect("primitive constructor prefix");
    let full = format!("prim._Op{operator}");
    let function = tvm_ffi::Function::get_global(&full)?;
    let result: Any = function.call_tuple((lhs, rhs, Any::new()))?;
    Ok(PrimExpr::try_from(result)?)
}

pub fn expr_any(expr: &PrimExpr) -> Any {
    Any::from(expr.clone())
}

pub fn int_any(value: i64) -> Any {
    Any::from(value)
}

/// Physical buffers have compact memory coordinates, with no thread mapping.
/// Explicit strides and element offsets are handled by the ordinary buffer rules.
pub fn buffer_layout(buffer: &BufferVar) -> AResult<Layout> {
    let ty = buffer.buffer_type();
    if let Some(layout) = &ty.layout {
        return Ok(layout.clone());
    }
    let one: PrimExpr = IntImm::new("int64", 1)?.into();
    let mut size = one.clone();
    for extent in ty.shape.iter() {
        size = op_binary("_OpMul", expr_any(&size), expr_any(&extent))?;
    }
    Ok(TileLayout::new(
        vec![Iter::new(size, one, Axis::get("m")?)?],
        Vec::new(),
        Map::new(),
    )?
    .into())
}

/// Query the target TIRx layout contract shared by validation and emission.
pub fn tile_layout_is_trivial(layout: &TileLayout) -> AResult<bool> {
    let result: Any =
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsTrivial").call_tuple((layout.clone(),))?;
    Ok(bool::try_from(result)?)
}

#[derive(Clone)]
pub enum Extent {
    Static(i64),
    Dynamic(PrimExpr),
}

#[derive(Clone)]
pub struct LayoutInfo {
    pub shape: Vec<Extent>,
    pub dtype: String,
    pub itemsize: i64,
    pub elem_offset: i64,
    pub dynamic_elem_offset: Option<PrimExpr>,
    pub element_count: Option<i64>,
    pub signature: String,
    pub physical_axes: Vec<String>,
    pub explicit_strides: Option<Vec<i64>>,
    pub dynamic_layout_elem_offset: Option<PrimExpr>,
    pub packed_nibble_offset: i64,
    pub allocated_addr: Option<PrimExpr>,
    pub allocated_addr_static: Option<i64>,
    pub tmem_lane_span: Option<i64>,
    pub tmem_tcol_span_elements: Option<i64>,
    pub tmem_tcol_base_static: Option<i64>,
}

impl LayoutInfo {
    pub fn byte_end(&self) -> AResult<i64> {
        if self.dynamic_elem_offset.is_some() {
            return unsupported(
                "runtime-offset buffer cannot define a compile-time backing byte end",
            );
        }
        let Some(count) = self.element_count else {
            return unsupported(
                "runtime-sized buffer cannot define a compile-time backing byte end",
            );
        };
        Ok((self.elem_offset + count) * self.itemsize)
    }
}

fn static_int(analyzer: &Analyzer, value: &PrimExpr, field: &str) -> AResult<i64> {
    shared_static_int(
        analyzer,
        value,
        &format!("buffer:{field}:requires"),
        "a static integer",
    )
}

fn shape_extent(
    analyzer: &Analyzer,
    value: &PrimExpr,
    field: &str,
    shape_bindings: &Map<Var, tvm::ir::Expr>,
) -> AResult<Extent> {
    let value = if shape_bindings.is_empty() {
        value.clone()
    } else {
        let substituted = crate::substitute(oref(value.clone()), shape_bindings.clone())?;
        PrimExpr::try_from(Any::from(substituted))?
    };
    let simplified = simplify(analyzer, &value)?;
    if let Some(extent) = int_imm_expr(&simplified) {
        return Ok(Extent::Static(extent));
    }
    let dtype = super::util::dtype_text(simplified.dtype());
    if crate::tables::is_integer_dtype(&dtype) {
        return Ok(Extent::Dynamic(simplified));
    }
    unsupported(format!(
        "buffer:{field}:shape extent must be an integer expression, got {}",
        repr_of(&value)?
    ))
}

fn prove_equal(analyzer: &Analyzer, lhs: &PrimExpr, rhs: &PrimExpr, field: &str) -> AResult<()> {
    let difference = op_binary("_OpSub", expr_any(lhs), expr_any(rhs))?;
    let simplified = simplify(analyzer, &difference)?;
    if int_imm_expr(&simplified) != Some(0) {
        return unsupported(format!(
            "buffer:{field}:cannot prove {} == {}",
            repr_of(lhs)?,
            repr_of(rhs)?
        ));
    }
    Ok(())
}

pub fn axis_name(axis: &tvm::tirx::Axis) -> AResult<String> {
    Ok(ffi_text(&axis.name()?))
}

fn iter_signature(items: &Array<tvm::tirx::Iter>) -> AResult<String> {
    let mut parts = Vec::new();
    for item in items.iter() {
        parts.push(format!(
            "{}:{}@{}",
            repr_of(&item.extent)?,
            repr_of(&item.stride)?,
            axis_name(&item.axis)?
        ));
    }
    Ok(parts.join(","))
}

pub fn layout_signature(layout: &Layout) -> AResult<String> {
    match layout.clone().try_cast::<TileLayout>() {
        Ok(tile) => {
            let canonical = Layout::from(tile.clone())
                .canonicalize()?
                .try_cast::<TileLayout>()?;
            Ok(format!(
                "TileLayout(shard=[{}],replica=[{}])",
                iter_signature(&canonical.shard()?)?,
                iter_signature(&canonical.replica()?)?
            ))
        }
        Err(_) => {
            let kind = if layout.clone().try_cast::<ComposeLayout>().is_ok() {
                "ComposeLayout"
            } else {
                return not_covered("layout kind without a native binding");
            };
            Ok(format!("{kind}({})", repr_of(layout)?))
        }
    }
}

fn validate_static_tile_mapping(
    analyzer: &Analyzer,
    layout: &TileLayout,
    name: &str,
) -> AResult<()> {
    let shard: Vec<tvm::tirx::Iter> = layout.shard()?.iter().collect();
    for index in (1..shard.len()).rev() {
        let extent = static_int(
            analyzer,
            &shard[index].extent,
            &format!("{name}:layout extent[{index}]"),
        )?;
        if extent <= 0 {
            return unsupported(format!(
                "buffer:{name}:layout extent[{index}] must be positive, got {extent}"
            ));
        }
    }
    for (index, item) in shard.iter().enumerate() {
        if axis_name(&item.axis)? != "m" {
            continue;
        }
        let stride = static_int(
            analyzer,
            &item.stride,
            &format!("{name}:layout stride[{index}]"),
        )?;
        if stride < 0 {
            return unsupported(format!(
                "buffer:{name}:negative physical layout stride {stride} is unsupported"
            ));
        }
    }
    for (axis, offset) in layout.offset()?.iter() {
        if axis_name(&axis)? == "m" {
            let offset = PrimExpr::try_from(offset)?;
            static_int(analyzer, &offset, &format!("{name}:layout m offset"))?;
        }
    }
    Ok(())
}

fn tile_repeat_stride(analyzer: &Analyzer, layout: &TileLayout, name: &str) -> AResult<i64> {
    let shard: Vec<tvm::tirx::Iter> = layout.shard()?.iter().collect();
    if shard.is_empty() || axis_name(&shard[0].axis)? != "m" {
        return Ok(0);
    }
    let extent = static_int(
        analyzer,
        &shard[0].extent,
        &format!("{name}:layout extent[0]"),
    )?;
    let stride = static_int(
        analyzer,
        &shard[0].stride,
        &format!("{name}:layout stride[0]"),
    )?;
    if extent <= 0 || stride < 0 {
        return unsupported(format!(
            "buffer:{name}:layout repeat extent/stride must be non-negative and nonzero"
        ));
    }
    Ok(extent * stride)
}

fn full_domain_physical_span(
    analyzer: &Analyzer,
    layout: &Layout,
    logical_elements: i64,
    name: &str,
) -> AResult<i64> {
    if logical_elements == 0 {
        return Ok(0);
    }
    if logical_elements < 0 {
        return unsupported(format!(
            "buffer:{name}:logical element count cannot be negative ({logical_elements})"
        ));
    }
    let (tile, atom_size, atom_span, repeat_stride);
    if let Ok(tile_layout) = layout.clone().try_cast::<TileLayout>() {
        atom_size = static_int(
            analyzer,
            &layout.get_size(None)?,
            &format!("{name}:layout atom size"),
        )?;
        atom_span = static_int(
            analyzer,
            &layout.get_span(Some("m"))?,
            &format!("{name}:layout atom span"),
        )?;
        repeat_stride = tile_repeat_stride(analyzer, &tile_layout, name)?;
        tile = tile_layout;
    } else if let Ok(compose) = layout.clone().try_cast::<ComposeLayout>() {
        let tile_layout = Layout::from(compose.tile_layout()?)
            .canonicalize()?
            .try_cast::<TileLayout>()?;
        atom_size = static_int(
            analyzer,
            &tile_layout.get_size(None)?,
            &format!("{name}:layout atom size"),
        )?;
        atom_span = static_int(
            analyzer,
            &layout.get_span(None)?,
            &format!("{name}:layout atom span"),
        )?;
        repeat_stride = tile_repeat_stride(analyzer, &tile_layout, name)?;
        let swizzle_period =
            1i64 << (compose.per_element()? + compose.swizzle_len()? + compose.atom_len()?);
        if logical_elements > atom_size
            && compose.swizzle_len()? != 0
            && repeat_stride % swizzle_period != 0
        {
            return unsupported(format!(
                "buffer:{name}:repeated ComposeLayout stride {repeat_stride} is not aligned to swizzle period {swizzle_period}"
            ));
        }
        tile = tile_layout;
    } else {
        return unsupported(format!(
            "buffer:{name}:cannot derive a physical span for {}",
            layout_signature(layout)?
        ));
    }
    if atom_size <= 0 || atom_span <= 0 || repeat_stride < 0 {
        return unsupported(format!(
            "buffer:{name}:layout atom size/span/repeat stride must be positive"
        ));
    }
    let full_atoms = logical_elements / atom_size;
    let tail = logical_elements % atom_size;
    let mut maximum: i64 = -1;
    if full_atoms > 0 {
        maximum = (full_atoms - 1) * repeat_stride + atom_span - 1;
    }
    if tail > 0 {
        if tail > MAX_PARTIAL_ATOM_ENUMERATION {
            return unsupported(format!(
                "buffer:{name}:partial layout atom has {tail} elements, exceeding the compile-time limit {MAX_PARTIAL_ATOM_ENUMERATION}"
            ));
        }
        validate_static_tile_mapping(analyzer, &tile, name)?;
        let mut tail_offsets = Vec::new();
        for offset in crate::layout_linear_offsets(layout.clone(), tail)?.iter() {
            tail_offsets.push(static_int(
                analyzer,
                &offset,
                &format!("{name}:layout m offset"),
            )?);
        }
        let minimum = *tail_offsets.iter().min().expect("tail offsets");
        if minimum < 0 {
            return unsupported(format!(
                "buffer:{name}:layout maps a logical element to a negative physical offset"
            ));
        }
        let largest = *tail_offsets.iter().max().expect("tail offsets");
        maximum = maximum.max(full_atoms * repeat_stride + largest);
    }
    Ok(maximum + 1)
}

fn map_get(map: &Map<FfiString, PrimExpr>, key: &str) -> AResult<Option<PrimExpr>> {
    Ok(map.get(&FfiString::from(key))?)
}

pub fn inspect_buffer_layout(
    schema: &Schema,
    analyzer: &Analyzer,
    buffer: &BufferVar,
    shape_bindings: &Map<Var, tvm::ir::Expr>,
) -> AResult<LayoutInfo> {
    let name = buffer_name(buffer);
    let buffer_type = buffer.buffer_type();
    let mut shape = Vec::new();
    for (axis, extent) in buffer_type.shape.iter().enumerate() {
        shape.push(shape_extent(
            analyzer,
            &extent,
            &format!("{name}:shape[{axis}]"),
            shape_bindings,
        )?);
    }
    if shape
        .iter()
        .any(|extent| matches!(extent, Extent::Static(value) if *value < 0))
    {
        let rendered: Vec<String> = shape
            .iter()
            .map(|extent| match extent {
                Extent::Static(value) => Ok(value.to_string()),
                Extent::Dynamic(expr) => repr_of(expr),
            })
            .collect::<AResult<_>>()?;
        return unsupported(format!("buffer:{name}:negative shape {:?}", &rendered));
    }
    if shape.is_empty() {
        return unsupported(format!(
            "buffer:{name}:scalar buffer layout is not implemented"
        ));
    }
    let dtype = buffer_dtype(buffer);
    let data_type = DLDataType::try_from_str(&dtype)?;
    let packed_nibble = dtype == "float4_e2m1fn";
    let itemsize: i64 = if dtype == "bool" && data_type.lanes == 1 {
        1
    } else if let Some((_, _, _, total_bits)) = schema.vector_dtype_abi(&dtype) {
        total_bits / 8
    } else if packed_nibble && data_type.lanes == 1 {
        1
    } else if data_type.lanes != 1 || data_type.bits % 8 != 0 {
        return unsupported(format!(
            "buffer:{name}:packed/vector dtype {dtype} has no byte-address lowering"
        ));
    } else {
        i64::from(data_type.bits) / 8
    };
    let scope = buffer_scope(buffer);
    let simplified_elem_offset = simplify(analyzer, &buffer_type.elem_offset)?;
    let (logical_elem_offset, dynamic_elem_offset) = match int_imm_expr(&simplified_elem_offset) {
        Some(value) => (value, None),
        None => {
            if packed_nibble {
                return unsupported(format!(
                    "buffer:{name}:runtime elem_offset for packed float4 storage is unsupported"
                ));
            }
            (0, Some(simplified_elem_offset.clone()))
        }
    };
    if logical_elem_offset < 0 {
        return unsupported(format!(
            "buffer:{name}:negative elem_offset {logical_elem_offset} is not implemented"
        ));
    }
    let packed_nibble_offset = if packed_nibble {
        logical_elem_offset % 2
    } else {
        0
    };
    let elem_offset = if packed_nibble {
        logical_elem_offset / 2
    } else {
        logical_elem_offset
    };
    let explicit_strides: Option<Vec<i64>> = if buffer_type.strides.is_empty() {
        None
    } else {
        let mut strides = Vec::new();
        for (axis, stride) in buffer_type.strides.iter().enumerate() {
            strides.push(static_int(
                analyzer,
                &stride,
                &format!("{name}:strides[{axis}]"),
            )?);
        }
        Some(strides)
    };
    let static_shape = shape
        .iter()
        .all(|extent| matches!(extent, Extent::Static(_)));
    if let Some(strides) = &explicit_strides {
        if strides.len() != shape.len() {
            return unsupported(format!(
                "buffer:{name}:expected {} explicit strides, got {}",
                shape.len(),
                strides.len()
            ));
        }
        if strides.iter().any(|stride| *stride < 0) {
            let rendered: Vec<String> = strides.iter().map(|s| s.to_string()).collect();
            return unsupported(format!(
                "buffer:{name}:negative explicit strides {:?} are unsupported",
                &rendered
            ));
        }
        if !static_shape {
            return unsupported(format!(
                "buffer:{name}:explicit strides require a static logical shape"
            ));
        }
        if packed_nibble {
            return unsupported(format!(
                "buffer:{name}:explicit strides for float4 storage are unsupported"
            ));
        }
    }
    let layout = buffer_layout(buffer)?;
    let signature = layout_signature(&layout)?;
    let canonical = layout.canonicalize()?;
    let expected_elements: Option<i64> = if static_shape {
        Some(
            shape
                .iter()
                .map(|extent| match extent {
                    Extent::Static(value) => *value,
                    Extent::Dynamic(_) => 1,
                })
                .product(),
        )
    } else {
        None
    };
    // math.prod(buffer.shape): Python multiplies from the integer 1 upward.
    let mut expected_elements_expr: Any = int_any(1);
    for extent in buffer_type.shape.iter() {
        expected_elements_expr = expr_any(&op_binary(
            "_OpMul",
            expected_elements_expr,
            expr_any(&extent),
        )?);
    }
    let expected_elements_expr = PrimExpr::try_from(expected_elements_expr)?;
    let zero_indices: Vec<PrimExpr> = shape
        .iter()
        .map(|_| IntImm::new("int32", 0).map(Into::into))
        .collect::<tvm_ffi::Result<_>>()?;
    let zero_indices = Array::new(zero_indices);
    let zero_extent_shared_placeholder = static_shape
        && expected_elements == Some(0)
        && (scope == "shared" || scope == "shared.dyn");
    let apply_zero = |candidate: &Layout| -> AResult<Map<FfiString, PrimExpr>> {
        if zero_extent_shared_placeholder {
            Ok(candidate.apply(&zero_indices)?)
        } else {
            Ok(candidate.apply_with_shape(&zero_indices, &buffer_type.shape)?)
        }
    };
    let mapped_zero = apply_zero(&canonical)?;
    let mut mapped: Vec<(String, PrimExpr)> = mapped_zero
        .iter()
        .map(|(axis, value)| (ffi_text(&axis), value))
        .collect();
    mapped.sort_by(|left, right| left.0.cmp(&right.0));
    let zero_axes: Vec<String> = mapped.iter().map(|(axis, _)| axis.clone()).collect();
    let mut dynamic_layout_elem_offset: Option<PrimExpr> = None;
    let is_compose = canonical.clone().try_cast::<ComposeLayout>().is_ok();
    if let Some(mapped_m_zero) = map_get(&mapped_zero, "m")? {
        let simplified_m_zero = simplify(analyzer, &mapped_m_zero)?;
        if int_imm_expr(&simplified_m_zero).is_some() {
            set_axis(&mut mapped, "m", simplified_m_zero);
        } else {
            let base_layout: Layout = if is_compose {
                Layout::from(
                    canonical
                        .clone()
                        .try_cast::<ComposeLayout>()?
                        .tile_layout()?,
                )
                .canonicalize()?
            } else {
                canonical.clone()
            };
            let base_map = apply_zero(&base_layout)?;
            let Some(base_m) = map_get(&base_map, "m")? else {
                return not_covered("base layout without an m axis");
            };
            let base_m_zero = simplify(analyzer, &base_m)?;
            if let Some(static_base) = int_imm_expr(&base_m_zero) {
                let static_m_zero = if is_compose {
                    let first = crate::layout_linear_offsets(canonical.clone(), 1)?.get(0)?;
                    static_int(analyzer, &first, &format!("{name}:layout m offset"))?
                } else {
                    static_base
                };
                set_axis(
                    &mut mapped,
                    "m",
                    IntImm::new("int64", static_m_zero)?.into(),
                );
            } else {
                dynamic_layout_elem_offset = Some(base_m_zero);
            }
        }
    }
    if explicit_strides.is_some() && scope != "shared" && scope != "shared.dyn" {
        return unsupported(format!(
            "buffer:{name}:explicit strides are currently supported only for shared storage, got scope {:?}",
            &scope));
    }
    if scope == "tmem" || scope.starts_with("tmem.") {
        return inspect_tmem(
            analyzer,
            buffer,
            &name,
            shape,
            dtype,
            itemsize,
            elem_offset,
            dynamic_elem_offset,
            expected_elements,
            signature,
            &layout,
            &canonical,
            &mapped,
            &zero_axes,
            dynamic_layout_elem_offset,
            packed_nibble,
        );
    }
    if !buffer_type.allocated_addr.is_empty() {
        return unsupported(format!(
            "buffer:{name}:allocated_addr {} is unsupported",
            repr_of(&buffer_type.allocated_addr)?
        ));
    }
    let zero_m = get_axis(&mapped, "m");
    let owner_axes: Vec<String> = zero_axes
        .iter()
        .filter(|axis| schema.register_owner_axes.contains(*axis))
        .cloned()
        .collect();
    let unsupported_axes: Vec<String> = zero_axes
        .iter()
        .filter(|axis| axis.as_str() != "m" && !schema.register_owner_axes.contains(*axis))
        .cloned()
        .collect();
    let implicit_register_slot = zero_m.is_none() && !owner_axes.is_empty();
    if (zero_m.is_none() && !implicit_register_slot) || !unsupported_axes.is_empty() {
        return unsupported(format!(
            "buffer:{name}:layout has unsupported physical axes {:?}: {signature}",
            &zero_axes
        ));
    }
    if !owner_axes.is_empty() && !schema.register_scopes.contains(&scope) {
        return unsupported(format!(
            "buffer:{name}:thread-owner axes {:?} require register/local storage, got scope {:?}: {signature}",
            &owner_axes,
            &scope));
    }
    for axis in &zero_axes {
        if axis == "m" && dynamic_layout_elem_offset.is_some() {
            continue;
        }
        let value = get_axis(&mapped, axis).expect("mapped axis");
        let base = static_int(analyzer, &value, &format!("{name}:{axis} base"))?;
        if axis == "m" {
            if base < 0 {
                return unsupported(format!(
                    "buffer:{name}:negative m layout base is unsupported: {signature}"
                ));
            }
            continue;
        }
        if base != 0 {
            return unsupported(format!(
                "buffer:{name}:nonzero {axis} layout base is unsupported: {signature}"
            ));
        }
    }
    let physical_elements: Option<i64>;
    if let Ok(_tile) = layout.clone().try_cast::<TileLayout>() {
        let canonical_tile = canonical.clone().try_cast::<TileLayout>()?;
        let shard: Vec<tvm::tirx::Iter> = canonical_tile.shard()?.iter().collect();
        let replica: Vec<tvm::tirx::Iter> = canonical_tile.replica()?.iter().collect();
        if !owner_axes.is_empty() {
            if explicit_strides.is_some() {
                return unsupported(format!(
                    "buffer:{name}:explicit strides cannot be combined with register-owner axes: {signature}"
                ));
            }
            if !replica.is_empty() {
                return unsupported(format!(
                    "buffer:{name}:replicated register layouts require explicit fan-out: {signature}"
                ));
            }
            let mut sorted_owner = owner_axes.clone();
            sorted_owner.sort();
            for axis in &sorted_owner {
                let limit = match axis.as_str() {
                    "laneid" => 32,
                    "wid_in_wg" => 4,
                    "tid_in_wg" => 128,
                    _ => return not_covered("register owner axis without a limit"),
                };
                let span = static_int(
                    analyzer,
                    &canonical.get_span(Some(axis))?,
                    &format!("{name}:{axis} span"),
                )?;
                if span <= 0 || span > limit {
                    return unsupported(format!(
                        "buffer:{name}:{axis} span {span} is outside 1..{limit}"
                    ));
                }
            }
            prove_equal(
                analyzer,
                &canonical.get_size(None)?,
                &expected_elements_expr,
                &format!("{name}:layout size"),
            )?;
            let elements = if implicit_register_slot {
                1
            } else {
                static_int(
                    analyzer,
                    &canonical.get_span(Some("m"))?,
                    &format!("{name}:m span"),
                )?
            };
            if elements <= 0 {
                return unsupported(format!(
                    "buffer:{name}:register physical span must be positive, got {elements}"
                ));
            }
            physical_elements = Some(elements);
        } else {
            let is_default = shard.len() == 1
                && replica.is_empty()
                && axis_name(&shard[0].axis)? == "m"
                && static_int(analyzer, &shard[0].stride, &format!("{name}:layout stride"))? == 1;
            if is_default {
                prove_equal(
                    analyzer,
                    &shard[0].extent,
                    &expected_elements_expr,
                    &format!("{name}:layout extent"),
                )?;
                physical_elements = match &explicit_strides {
                    None => expected_elements,
                    Some(strides) => {
                        let extents: Vec<i64> = shape
                            .iter()
                            .map(|extent| match extent {
                                Extent::Static(value) => *value,
                                Extent::Dynamic(_) => 0,
                            })
                            .collect();
                        if extents.is_empty() || extents.iter().any(|extent| *extent == 0) {
                            Some(0)
                        } else {
                            Some(
                                1 + extents
                                    .iter()
                                    .zip(strides.iter())
                                    .map(|(extent, stride)| (extent - 1) * stride)
                                    .sum::<i64>(),
                            )
                        }
                    }
                };
            } else {
                if explicit_strides.is_some() {
                    return unsupported(format!(
                        "buffer:{name}:explicit strides require the default logical layout, got {signature}"
                    ));
                }
                let Some(expected) = expected_elements else {
                    return unsupported(format!(
                        "buffer:{name}:non-default layout requires a static logical shape"
                    ));
                };
                physical_elements = Some(match &dynamic_layout_elem_offset {
                    None => full_domain_physical_span(analyzer, &canonical, expected, &name)?,
                    Some(offset) => {
                        let relative = op_binary(
                            "_OpSub",
                            expr_any(&canonical.get_span(Some("m"))?),
                            expr_any(offset),
                        )?;
                        static_int(
                            analyzer,
                            &relative,
                            &format!("{name}:dynamic-layout relative span"),
                        )?
                    }
                });
            }
        }
    } else if !is_compose {
        return unsupported(format!("buffer:{name}:unsupported layout {signature}"));
    } else {
        if explicit_strides.is_some() {
            return unsupported(format!(
                "buffer:{name}:explicit strides require the default logical layout, got {signature}"
            ));
        }
        if !owner_axes.is_empty() {
            return unsupported(format!(
                "buffer:{name}:thread-owner axes require a TileLayout: {signature}"
            ));
        }
        let Some(expected) = expected_elements else {
            return unsupported(format!(
                "buffer:{name}:ComposeLayout requires a static logical shape"
            ));
        };
        physical_elements = Some(match &dynamic_layout_elem_offset {
            None => full_domain_physical_span(analyzer, &canonical, expected, &name)?,
            Some(offset) => {
                let relative = op_binary(
                    "_OpSub",
                    expr_any(&canonical.get_span(None)?),
                    expr_any(offset),
                )?;
                static_int(
                    analyzer,
                    &relative,
                    &format!("{name}:dynamic-layout relative span"),
                )?
            }
        });
    }
    let physical_elements = if packed_nibble {
        let Some(count) = physical_elements else {
            return unsupported(format!(
                "buffer:{name}:float4 requires a static physical span"
            ));
        };
        Some(if count == 0 {
            0
        } else {
            (packed_nibble_offset + count + 1) / 2
        })
    } else {
        physical_elements
    };
    let mut physical_axes: Vec<String> = zero_axes.clone();
    if implicit_register_slot {
        physical_axes.push("m".to_owned());
    }
    physical_axes.sort();
    physical_axes.dedup();
    Ok(LayoutInfo {
        shape,
        dtype,
        itemsize,
        elem_offset,
        dynamic_elem_offset,
        element_count: physical_elements,
        signature,
        physical_axes,
        explicit_strides,
        dynamic_layout_elem_offset,
        packed_nibble_offset,
        allocated_addr: None,
        allocated_addr_static: None,
        tmem_lane_span: None,
        tmem_tcol_span_elements: None,
        tmem_tcol_base_static: Some(0),
    })
}

fn get_axis(mapped: &[(String, PrimExpr)], axis: &str) -> Option<PrimExpr> {
    mapped
        .iter()
        .find(|(name, _)| name == axis)
        .map(|(_, value)| value.clone())
}

fn set_axis(mapped: &mut Vec<(String, PrimExpr)>, axis: &str, value: PrimExpr) {
    if let Some(entry) = mapped.iter_mut().find(|(name, _)| name == axis) {
        entry.1 = value;
    } else {
        mapped.push((axis.to_owned(), value));
        mapped.sort_by(|left, right| left.0.cmp(&right.0));
    }
}

#[allow(clippy::too_many_arguments)]
fn inspect_tmem(
    analyzer: &Analyzer,
    buffer: &BufferVar,
    name: &str,
    shape: Vec<Extent>,
    dtype: String,
    itemsize: i64,
    elem_offset: i64,
    dynamic_elem_offset: Option<PrimExpr>,
    expected_elements: Option<i64>,
    signature: String,
    layout: &Layout,
    canonical: &Layout,
    mapped: &[(String, PrimExpr)],
    zero_axes: &[String],
    dynamic_layout_elem_offset: Option<PrimExpr>,
    packed_nibble: bool,
) -> AResult<LayoutInfo> {
    if dynamic_elem_offset.is_some() {
        return unsupported(format!(
            "buffer:{name}:dynamic TMEM elem_offset is not implemented"
        ));
    }
    if packed_nibble {
        return unsupported(format!(
            "buffer:{name}:float4 TMEM views are not implemented"
        ));
    }
    let buffer_type = buffer.buffer_type();
    let allocated_addr: Vec<PrimExpr> = buffer_type.allocated_addr.iter().collect();
    if allocated_addr.len() != 1 {
        return unsupported(format!(
            "buffer:{name}:TMEM requires exactly one allocated_addr, got {}",
            repr_of(&buffer_type.allocated_addr)?
        ));
    }
    let Ok(_) = layout.clone().try_cast::<TileLayout>() else {
        return unsupported(format!(
            "buffer:{name}:TMEM requires a TileLayout, got {signature}"
        ));
    };
    if !zero_axes
        .iter()
        .all(|axis| axis == "TLane" || axis == "TCol")
    {
        return unsupported(format!(
            "buffer:{name}:TMEM layout must map only to TLane/TCol, got {:?}: {signature}",
            &zero_axes
        ));
    }
    let canonical_tile = canonical.clone().try_cast::<TileLayout>()?;
    let mut unsupported_axes: Vec<String> = Vec::new();
    for item in canonical_tile
        .shard()?
        .iter()
        .chain(canonical_tile.replica()?.iter())
    {
        let axis = axis_name(&item.axis)?;
        if axis != "TLane" && axis != "TCol" && !unsupported_axes.contains(&axis) {
            unsupported_axes.push(axis);
        }
    }
    unsupported_axes.sort();
    if !unsupported_axes.is_empty() {
        return unsupported(format!(
            "buffer:{name}:TMEM layout uses unsupported axes {:?}: {signature}",
            &unsupported_axes
        ));
    }
    let zero_lane = match get_axis(mapped, "TLane") {
        Some(value) => static_int(analyzer, &value, &format!("{name}:TLane base"))?,
        None => 0,
    };
    let zero_tcol_static: Option<i64> = match get_axis(mapped, "TCol") {
        Some(value) => match static_int(analyzer, &value, &format!("{name}:TCol base")) {
            Ok(base) => Some(base),
            Err(Failure::Unsupported { .. }) => None,
            Err(error) => return Err(error),
        },
        None => Some(0),
    };
    if zero_lane < 0 || zero_tcol_static.is_some_and(|value| value < 0) {
        let tcol = match zero_tcol_static {
            Some(value) => value.to_string(),
            None => "None".to_owned(),
        };
        return unsupported(format!(
            "buffer:{name}:negative TMEM layout base (lane={zero_lane}, tcol={tcol}) is unsupported"
        ));
    }
    let addr_static: Option<i64> = match static_int(
        analyzer,
        &allocated_addr[0],
        &format!("{name}:allocated_addr"),
    ) {
        Ok(value) => Some(value),
        Err(Failure::Unsupported { .. }) => None,
        Err(error) => return Err(error),
    };
    if addr_static.is_some_and(|value| value < 0) {
        return unsupported(format!(
            "buffer:{name}:negative TMEM allocated_addr {}",
            addr_static.unwrap()
        ));
    }
    let lane_span = static_int(
        analyzer,
        &canonical.get_span(Some("TLane"))?,
        &format!("{name}:TLane span"),
    )?;
    if lane_span <= 0 || lane_span > 128 {
        return unsupported(format!(
            "buffer:{name}:TMEM TLane span {lane_span} is outside 1..128"
        ));
    }
    let tcol_capacity = match static_int(
        analyzer,
        &canonical.get_span(Some("TCol"))?,
        &format!("{name}:TCol span"),
    ) {
        Ok(value) => value,
        Err(Failure::Unsupported {
            message,
            unsupported: items,
        }) => {
            if zero_tcol_static.is_some() {
                return Err(Failure::Unsupported {
                    message,
                    unsupported: items,
                });
            }
            let remaining_columns = match addr_static {
                None => 512,
                Some(addr) => 512 - addr,
            };
            if remaining_columns <= 0 {
                return unsupported(format!(
                    "buffer:{name}:TMEM allocated_addr {} leaves no TCol capacity",
                    addr_static.unwrap()
                ));
            }
            remaining_columns * 4 / itemsize
        }
        Err(error) => return Err(error),
    };
    if tcol_capacity <= 0 {
        return unsupported(format!(
            "buffer:{name}:TMEM TCol span must be positive, got {tcol_capacity}"
        ));
    }
    Ok(LayoutInfo {
        shape,
        dtype,
        itemsize,
        elem_offset,
        dynamic_elem_offset,
        element_count: expected_elements,
        signature,
        physical_axes: vec!["TCol".to_owned(), "TLane".to_owned()],
        explicit_strides: None,
        dynamic_layout_elem_offset,
        packed_nibble_offset: 0,
        allocated_addr: Some(allocated_addr[0].clone()),
        allocated_addr_static: addr_static,
        tmem_lane_span: Some(lane_span),
        tmem_tcol_span_elements: Some(tcol_capacity),
        tmem_tcol_base_static: zero_tcol_static,
    })
}
