//! Generation-side layout helpers: physical element offsets and owner
//! coordinates.

use tvm::ir::{IntImm, PrimExpr};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::{Array, Map, String as FfiString};

use super::super::analyze::layout::{buffer_layout, expr_any, int_any, layout_signature, op_binary};
use super::super::analyze::util::{buffer_name, ffi_text, unsupported, AResult};
use super::Emitter;

impl<'a> Emitter<'a> {
    fn mapped_axes(
        &self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<Vec<(String, PrimExpr)>> {
        let layout = buffer_layout(buffer)?;
        let coordinates = Array::new(indices.to_vec());
        let mapped: Map<FfiString, PrimExpr> = layout
            .canonicalize()?
            .apply_with_shape(&coordinates, &buffer.buffer_type().shape)?;
        let mut axes: Vec<(String, PrimExpr)> = mapped
            .iter()
            .map(|(axis, value)| (ffi_text(&axis), value))
            .collect();
        axes.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(axes)
    }

    fn check_index_count(&self, buffer: &BufferVar, indices: &[PrimExpr]) -> AResult<()> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        if indices.len() != info.shape.len() {
            return unsupported(format!(
                "buffer:{}:expected {} indices, got {}",
                buffer_name(buffer),
                info.shape.len(),
                indices.len()
            ));
        }
        Ok(())
    }

    /// `physical_element_offset`.
    pub fn physical_element_offset(
        &self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<PrimExpr> {
        let info = self.ctx.inspect_layout(buffer, &self.bindings)?;
        self.check_index_count(buffer, indices)?;
        let mut offset: PrimExpr;
        if let Some(strides) = &info.explicit_strides {
            offset = IntImm::new("int64", 0)?.into();
            for (index, stride) in indices.iter().zip(strides.iter()) {
                let product = op_binary("_OpMul", expr_any(index), int_any(*stride))?;
                offset = op_binary("_OpAdd", expr_any(&offset), expr_any(&product))?;
            }
        } else {
            let mapped = self.mapped_axes(buffer, indices)?;
            let axes: Vec<String> = mapped.iter().map(|(axis, _)| axis.clone()).collect();
            if axes
                .iter()
                .any(|axis| axis != "m" && !self.ctx.schema.register_owner_axes.contains(axis))
            {
                let layout = buffer.buffer_type().layout.clone().expect("layout");
                return unsupported(format!(
                    "buffer:{}:layout maps to unsupported axes {:?}: {}",
                    buffer_name(buffer),
                    &axes,
                    layout_signature(&layout)?
                ));
            }
            offset = match mapped.iter().find(|(axis, _)| axis == "m") {
                Some((_, value)) => value.clone(),
                None => IntImm::new("int32", 0)?.into(),
            };
            if info.packed_nibble_offset != 0 {
                offset = op_binary(
                    "_OpAdd",
                    expr_any(&offset),
                    int_any(info.packed_nibble_offset),
                )?;
            }
        }
        if let Some(dynamic) = &info.dynamic_elem_offset {
            offset = op_binary("_OpAdd", expr_any(&offset), expr_any(dynamic))?;
        }
        Ok(offset)
    }

    /// `physical_owner_coordinates`: `(axis, expression)` sorted by axis.
    pub fn physical_owner_coordinates(
        &self,
        buffer: &BufferVar,
        indices: &[PrimExpr],
    ) -> AResult<Vec<(String, PrimExpr)>> {
        self.check_index_count(buffer, indices)?;
        let mapped = self.mapped_axes(buffer, indices)?;
        let unsupported_axes: Vec<String> = mapped
            .iter()
            .map(|(axis, _)| axis.clone())
            .filter(|axis| axis != "m" && !self.ctx.schema.register_owner_axes.contains(axis))
            .collect();
        if !unsupported_axes.is_empty() {
            let layout = buffer.buffer_type().layout.clone().expect("layout");
            return unsupported(format!(
                "buffer:{}:layout maps to unsupported owner axes {:?}: {}",
                buffer_name(buffer),
                &unsupported_axes,
                layout_signature(&layout)?
            ));
        }
        Ok(mapped
            .into_iter()
            .filter(|(axis, _)| self.ctx.schema.register_owner_axes.contains(axis))
            .collect())
    }
}
