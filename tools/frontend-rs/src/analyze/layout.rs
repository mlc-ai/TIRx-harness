//! Static buffer layout inspection.

use tvm_tirx::ir::prim::{eq, floordiv, floormod};
use tvm_tirx::ir::{Expr, PrimExpr, Var};
use tvm_tirx::sym::{Analyzer, ProofStrength};
use tvm_tirx::tirx::{Axis, ComposeLayout, Iter, Layout, TileLayout};
use tvm_tirx::tvm_ffi::object::ObjectRefCore;
use tvm_tirx::tvm_ffi::{self, Any, Array, DLDataType, DLDataTypeExt, Map, String as FfiString};

use super::util::{
    buffer_dtype, buffer_name, buffer_scope, dtype_text, enum_name, ffi_text, int_const,
    int_imm_expr, not_covered, oref, prim, prim_expr, repr_of, simplify,
    static_int as shared_static_int, unsupported, AResult, Failure, PrimExprExt, TensorVarExt,
};
use crate::schema::Schema;

const MAX_PARTIAL_ATOM_ENUMERATION: i64 = 1_000_000;

/// Shared primitive operators used by the Python expression constructors.
pub fn op_binary(name: &str, lhs: Any, rhs: Any) -> AResult<Expr> {
    let operator = name
        .strip_prefix("_Op")
        .expect("primitive constructor prefix");
    let full = format!("prim._Op{operator}");
    let function = tvm_ffi::Function::get_global(&full)?;
    let result: Any = function.call_tuple((lhs, rhs, tvm_tirx::ir::UnknownLoc::new()))?;
    Ok(prim_expr(result)?)
}

pub fn expr_any<T: Clone + Into<Expr>>(expr: &T) -> Any {
    Any::from(expr.clone().into())
}

pub fn int_any(value: i64) -> Any {
    Any::from(value)
}

/// Physical buffers have compact memory coordinates, with no thread mapping.
/// Explicit strides and element offsets are handled by the ordinary buffer rules.
pub fn buffer_layout(buffer: &Var) -> AResult<Layout> {
    let ty = buffer.tensor_type();
    if let Some(layout) = &ty.layout {
        return Ok(layout.clone());
    }
    let one: Expr = int_const("int64", 1)?.into();
    let mut size = one.clone();
    for extent in ty.shape.iter() {
        size = op_binary("_OpMul", expr_any(&size), expr_any(&extent))?;
    }
    Ok(TileLayout::new(
        vec![Iter::new(
            PrimExpr::try_from(Any::from(size))?,
            PrimExpr::try_from(Any::from(one))?,
            Axis::get_or_register("m"),
        )],
        Vec::<Iter>::new(),
        Map::<Axis, PrimExpr>::new(),
    )
    .into())
}

/// The element strides of a strided shared tensor, if `buffer` is one.
///
/// A tvm-tirx tensor has no strides field: a strided tensor carries a
/// TileLayout with one memory shard per logical dimension. Like an explicitly
/// strided buffer, it is addressed through its strides and bounds-checked at
/// its executed addresses, so it may span past its allocation. A contiguous
/// layout is not strided and keeps the element-span check.
fn strided_view_strides(
    analyzer: &Analyzer,
    buffer: &Var,
    shape: &[Extent],
    scope: &str,
    packed_nibble: bool,
) -> AResult<Option<Vec<i64>>> {
    if packed_nibble || (scope != "shared" && scope != "shared.dyn") {
        return Ok(None);
    }
    let Some(layout) = buffer.tensor_type().layout.clone() else {
        return Ok(None);
    };
    let Ok(tile) = layout.downcast::<TileLayout>() else {
        return Ok(None);
    };
    let shard: Vec<Iter> = tile.shard.iter().cloned().collect();
    if shard.len() != shape.len() || !tile.replica.is_empty() || !tile.offset.is_empty() {
        return Ok(None);
    }
    let mut strides = Vec::with_capacity(shard.len());
    for (iter, extent) in shard.iter().zip(shape) {
        let Extent::Static(extent) = extent else {
            return Ok(None);
        };
        if axis_name(&iter.axis)? != "m"
            || int_imm_expr(&iter.extent.clone().into()) != Some(*extent)
        {
            return Ok(None);
        }
        match int_imm_expr(&iter.stride.clone().into()) {
            Some(stride) if stride >= 0 => strides.push(stride),
            _ => return Ok(None),
        }
    }
    if tile.is_contiguous(Some(analyzer))? {
        return Ok(None);
    }
    Ok(Some(strides))
}

#[derive(Clone)]
pub enum Extent {
    Static(i64),
    Dynamic(Expr),
}

#[derive(Clone)]
pub struct LayoutInfo {
    pub shape: Vec<Extent>,
    pub dtype: String,
    pub itemsize: i64,
    pub elem_offset: i64,
    pub dynamic_elem_offset: Option<Expr>,
    /// `floormod(byte_offset, itemsize) == 0` for a runtime byte offset the
    /// analyzer could not prove element-aligned, checked at the declaration.
    pub dynamic_elem_offset_alignment: Option<Expr>,
    pub element_count: Option<i64>,
    pub signature: String,
    pub physical_axes: Vec<String>,
    pub explicit_strides: Option<Vec<i64>>,
    pub dynamic_layout_elem_offset: Option<Expr>,
    pub packed_nibble_offset: i64,
    pub allocated_addr: Option<Expr>,
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

fn static_int<T: Clone + Into<Expr>>(analyzer: &Analyzer, value: &T, field: &str) -> AResult<i64> {
    shared_static_int(
        analyzer,
        &value.clone().into(),
        &format!("buffer:{field}:requires"),
        "a static integer",
    )
}

fn shape_extent<T: Clone + Into<Expr>>(
    analyzer: &Analyzer,
    value: &T,
    field: &str,
    shape_bindings: &Map<Var, tvm_tirx::ir::Expr>,
) -> AResult<Extent> {
    let value: Expr = value.clone().into();
    let value = if shape_bindings.is_empty() {
        value.clone()
    } else {
        let substituted = crate::substitute(oref(value.clone()), shape_bindings.clone())?;
        prim_expr(Any::from(substituted))?
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

fn prove_equal<L: Clone + Into<Expr>, R: Clone + Into<Expr>>(
    analyzer: &Analyzer,
    lhs: &L,
    rhs: &R,
    field: &str,
) -> AResult<()> {
    let lhs: &Expr = &lhs.clone().into();
    let rhs: &Expr = &rhs.clone().into();
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

pub fn axis_name(axis: &tvm_tirx::tirx::Axis) -> AResult<String> {
    Ok(enum_name(&**axis))
}

fn iter_signature(items: &Array<tvm_tirx::tirx::Iter>) -> AResult<String> {
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
    match layout.clone().downcast::<TileLayout>() {
        Ok(tile) => {
            let canonical = Layout::from(tile.clone())
                .canonicalize()?
                .downcast::<TileLayout>()
                .map_err(|_| Failure::NotCovered("expected TileLayout".into()))?;
            Ok(format!(
                "TileLayout(shard=[{}],replica=[{}])",
                iter_signature(&canonical.shard)?,
                iter_signature(&canonical.replica)?
            ))
        }
        Err(_) => {
            let kind = if layout.clone().downcast::<ComposeLayout>().is_ok() {
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
    let shard: Vec<tvm_tirx::tirx::Iter> = layout.shard.iter().cloned().collect();
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
    for (axis, offset) in layout.offset.iter() {
        if axis_name(&axis)? == "m" {
            let offset = prim_expr(offset)?;
            static_int(analyzer, &offset, &format!("{name}:layout m offset"))?;
        }
    }
    Ok(())
}

fn tile_repeat_stride(analyzer: &Analyzer, layout: &TileLayout, name: &str) -> AResult<i64> {
    let shard: Vec<tvm_tirx::tirx::Iter> = layout.shard.iter().cloned().collect();
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
    if let Ok(tile_layout) = layout.clone().downcast::<TileLayout>() {
        atom_size = static_int(
            analyzer,
            &layout.size(None)?,
            &format!("{name}:layout atom size"),
        )?;
        atom_span = static_int(
            analyzer,
            &layout.span(Some("m"))?,
            &format!("{name}:layout atom span"),
        )?;
        repeat_stride = tile_repeat_stride(analyzer, &tile_layout, name)?;
        tile = tile_layout;
    } else if let Ok(compose) = layout.clone().downcast::<ComposeLayout>() {
        let tile_layout = Layout::from(compose.tile_layout.clone())
            .canonicalize()?
            .downcast::<TileLayout>()
            .map_err(|_| Failure::NotCovered("expected TileLayout".into()))?;
        atom_size = static_int(
            analyzer,
            &Layout::from(tile_layout.clone()).size(None)?,
            &format!("{name}:layout atom size"),
        )?;
        atom_span = static_int(
            analyzer,
            &layout.span(None)?,
            &format!("{name}:layout atom span"),
        )?;
        repeat_stride = tile_repeat_stride(analyzer, &tile_layout, name)?;
        let swizzle_period = 1i64 << (compose.per_element + compose.swizzle_len + compose.atom_len);
        if logical_elements > atom_size
            && compose.swizzle_len != 0
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

fn map_get(map: &Map<FfiString, Expr>, key: &str) -> AResult<Option<Expr>> {
    Ok(map.get(&FfiString::from(key))?)
}

pub fn inspect_buffer_layout(
    schema: &Schema,
    analyzer: &Analyzer,
    buffer: &Var,
    shape_bindings: &Map<Var, tvm_tirx::ir::Expr>,
    tmem_address: Option<tvm_tirx::ir::Expr>,
    byte_offset: Option<tvm_tirx::ir::Expr>,
) -> AResult<LayoutInfo> {
    let name = buffer_name(buffer);
    let buffer_type = buffer.tensor_type();
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
    // A declared view starts `byte_offset` bytes into its storage root, a whole
    // number of storage elements (bytes for packed float4); a TensorType has no
    // strides, which belong to its layout.
    // A runtime byte offset the analyzer cannot prove element-aligned is
    // checked at runtime, when the view is declared, instead of rejected.
    let mut dynamic_elem_offset_alignment: Option<Expr> = None;
    let (elem_offset, dynamic_elem_offset): (i64, Option<Expr>) = match byte_offset {
        None => (0, None),
        Some(bytes) => {
            let bytes = simplify(analyzer, &bytes)?;
            match int_imm_expr(&bytes) {
                Some(value) if value < 0 => {
                    return unsupported(format!(
                        "buffer:{name}:negative byte offset {value} is not implemented"
                    ))
                }
                Some(value) if value % itemsize == 0 => (value / itemsize, None),
                Some(value) => {
                    return unsupported(format!(
                        "buffer:{name}:byte offset {value} is not a multiple of the {itemsize}-byte element"
                    ))
                }
                None if packed_nibble => {
                    return unsupported(format!(
                        "buffer:{name}:runtime elem_offset for packed float4 storage is unsupported"
                    ))
                }
                None => {
                    let dtype = dtype_text(bytes.dtype());
                    let bytes = PrimExpr::try_from(Any::from(bytes))?;
                    let element = PrimExpr::from(int_const(&dtype, itemsize)?);
                    let aligned = eq(
                        floormod(bytes.clone(), element.clone())?,
                        PrimExpr::from(int_const(&dtype, 0)?),
                    )?;
                    if !analyzer.can_prove(&aligned, ProofStrength::Default)? {
                        dynamic_elem_offset_alignment = Some(aligned.into());
                    }
                    (0, Some(simplify(analyzer, &floordiv(bytes, element)?)?))
                }
            }
        }
    };
    let packed_nibble_offset: i64 = 0;
    let explicit_strides = strided_view_strides(analyzer, buffer, &shape, &scope, packed_nibble)?;
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
    let expected_elements_expr = prim_expr(expected_elements_expr)?;
    let zero_indices: Vec<PrimExpr> = shape
        .iter()
        .map(|_| int_const("int32", 0).map(Into::into))
        .collect::<tvm_ffi::Result<_>>()?;
    let zero_indices = Array::from_vec(zero_indices);
    let zero_extent_shared_placeholder = static_shape
        && expected_elements == Some(0)
        && (scope == "shared" || scope == "shared.dyn");
    let apply_zero = |candidate: &Layout| -> AResult<Map<FfiString, Expr>> {
        if zero_extent_shared_placeholder {
            Ok(candidate
                .apply(&zero_indices, analyzer)?
                .iter()
                .map(|(k, v)| (k.clone(), v.clone().into()))
                .collect())
        } else {
            Ok(candidate
                .apply_with_shape(&zero_indices, &buffer_type.shape, analyzer)?
                .iter()
                .map(|(k, v)| (k.clone(), v.clone().into()))
                .collect())
        }
    };
    let mapped_zero = apply_zero(&canonical)?;
    let mut mapped: Vec<(String, Expr)> = mapped_zero
        .iter()
        .map(|(axis, value)| (ffi_text(&axis), value.clone()))
        .collect();
    mapped.sort_by(|left, right| left.0.cmp(&right.0));
    let zero_axes: Vec<String> = mapped.iter().map(|(axis, _)| axis.clone()).collect();
    let mut dynamic_layout_elem_offset: Option<Expr> = None;
    let is_compose = canonical.clone().downcast::<ComposeLayout>().is_ok();
    if let Some(mapped_m_zero) = map_get(&mapped_zero, "m")? {
        let simplified_m_zero = simplify(analyzer, &mapped_m_zero)?;
        if int_imm_expr(&simplified_m_zero).is_some() {
            set_axis(&mut mapped, "m", simplified_m_zero);
        } else {
            let base_layout: Layout = if is_compose {
                Layout::from(
                    canonical
                        .clone()
                        .downcast::<ComposeLayout>()
                        .map_err(|_| Failure::NotCovered("expected ComposeLayout".into()))?
                        .tile_layout
                        .clone(),
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
                    let first = crate::layout_linear_offsets(canonical.clone(), 1)?
                        .get(0)
                        .ok_or_else(|| Failure::NotCovered("empty layout offsets".into()))?;
                    static_int(analyzer, &first, &format!("{name}:layout m offset"))?
                } else {
                    static_base
                };
                set_axis(&mut mapped, "m", int_const("int64", static_m_zero)?.into());
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
        if let Some(aligned) = &dynamic_elem_offset_alignment {
            return unsupported(format!(
                "buffer:{name}:runtime TMEM offset is not provably element-aligned: {}",
                repr_of(aligned)?
            ));
        }
        return inspect_tmem(
            analyzer,
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
            tmem_address,
        );
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
    if let Ok(_tile) = layout.clone().downcast::<TileLayout>() {
        let canonical_tile = canonical
            .clone()
            .downcast::<TileLayout>()
            .map_err(|_| Failure::NotCovered("expected TileLayout".into()))?;
        let shard: Vec<tvm_tirx::tirx::Iter> = canonical_tile.shard.iter().cloned().collect();
        let replica: Vec<tvm_tirx::tirx::Iter> = canonical_tile.replica.iter().cloned().collect();
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
                    &canonical.span(Some(axis))?,
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
                &canonical.size(None)?,
                &expected_elements_expr,
                &format!("{name}:layout size"),
            )?;
            let elements = if implicit_register_slot {
                1
            } else {
                static_int(
                    analyzer,
                    &canonical.span(Some("m"))?,
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
            if let Some(strides) = &explicit_strides {
                let extents: Vec<i64> = shape
                    .iter()
                    .map(|extent| match extent {
                        Extent::Static(value) => *value,
                        Extent::Dynamic(_) => 0,
                    })
                    .collect();
                physical_elements =
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
                    };
            } else if is_default {
                prove_equal(
                    analyzer,
                    &shard[0].extent,
                    &expected_elements_expr,
                    &format!("{name}:layout extent"),
                )?;
                physical_elements = expected_elements;
            } else {
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
                            expr_any(&canonical.span(Some("m"))?),
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
                let relative =
                    op_binary("_OpSub", expr_any(&canonical.span(None)?), expr_any(offset))?;
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
        dynamic_elem_offset_alignment,
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

fn get_axis(mapped: &[(String, Expr)], axis: &str) -> Option<Expr> {
    mapped
        .iter()
        .find(|(name, _)| name == axis)
        .map(|(_, value)| value.clone())
}

fn set_axis(mapped: &mut Vec<(String, Expr)>, axis: &str, value: Expr) {
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
    name: &str,
    shape: Vec<Extent>,
    dtype: String,
    itemsize: i64,
    elem_offset: i64,
    dynamic_elem_offset: Option<Expr>,
    expected_elements: Option<i64>,
    signature: String,
    layout: &Layout,
    canonical: &Layout,
    mapped: &[(String, Expr)],
    zero_axes: &[String],
    dynamic_layout_elem_offset: Option<Expr>,
    packed_nibble: bool,
    tmem_address: Option<tvm_tirx::ir::Expr>,
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
    // The column address is an operand of the producing tirx.alloc_tensor or
    // tirx.cuda.decl_tmem, not part of the tensor type.
    let Some(tmem_address) = tmem_address else {
        return unsupported(format!(
            "buffer:{name}:TMEM requires the column address of its allocation"
        ));
    };
    let allocated_addr = prim(&oref(tmem_address))?;
    let Ok(_) = layout.clone().downcast::<TileLayout>() else {
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
    let canonical_tile = canonical
        .clone()
        .downcast::<TileLayout>()
        .map_err(|_| Failure::NotCovered("expected TileLayout".into()))?;
    let mut unsupported_axes: Vec<String> = Vec::new();
    for item in canonical_tile
        .shard
        .iter()
        .chain(canonical_tile.replica.iter())
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
    let addr_static: Option<i64> =
        match static_int(analyzer, &allocated_addr, &format!("{name}:allocated_addr")) {
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
        &canonical.span(Some("TLane"))?,
        &format!("{name}:TLane span"),
    )?;
    if lane_span <= 0 || lane_span > 128 {
        return unsupported(format!(
            "buffer:{name}:TMEM TLane span {lane_span} is outside 1..128"
        ));
    }
    let tcol_capacity = match static_int(
        analyzer,
        &canonical.span(Some("TCol"))?,
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
        dynamic_elem_offset_alignment: None,
        element_count: expected_elements,
        signature,
        physical_axes: vec!["TCol".to_owned(), "TLane".to_owned()],
        explicit_strides: None,
        dynamic_layout_elem_offset,
        packed_nibble_offset: 0,
        allocated_addr: Some(allocated_addr),
        allocated_addr_static: addr_static,
        tmem_lane_span: Some(lane_span),
        tmem_tcol_span_elements: Some(tcol_capacity),
        tmem_tcol_base_static: zero_tcol_static,
    })
}
