//! Validation and emission of the ptx_spdecompress instruction family.

use crate::analyze::layout::{expr_any, int_any, op_binary};
use crate::analyze::shapes::structural_equal;
use crate::analyze::util::{
    as_buffer, dtype_of, ffi_error, not_covered, oref, same, simplify, unsupported, AResult,
    Failure,
};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::require_register_call;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::ir::{PrimExpr, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore};

/// `resolve_ptx_spdecompress` result: the register groups and the variant.
pub struct SpDecompressForm {
    pub data: Vec<ObjectRef>,
    pub metadata: Vec<ObjectRef>,
    pub compressed: Vec<ObjectRef>,
    pub variant: String,
}

fn shape_field(decoded: &DecodedPtx, name: &str, prefix: &str) -> AResult<i64> {
    let token = decoded.modifier(name)?;
    token
        .strip_prefix(prefix)
        .unwrap_or(token)
        .parse()
        .map_err(|_| {
            Failure::Ffi(ffi_error(&format!(
                "spdecompress {name} modifier {token:?}"
            )))
        })
}

fn static_shape(decoded: &DecodedPtx) -> AResult<(i64, i64, i64, i64, i64)> {
    let elem_bits = shape_field(decoded, "elemsize", "b")?;
    let index_bits = shape_field(decoded, "idxsize", "b")?;
    let factor = decoded.modifier("spfactor")?;
    let factor = factor.strip_prefix("sp::").unwrap_or(factor);
    let Some((src_text, dst_text)) = factor.split_once(':') else {
        return Err(Failure::Ffi(ffi_error(&format!(
            "spdecompress spfactor modifier {factor:?}"
        ))));
    };
    let parse = |text: &str| -> AResult<i64> {
        text.parse().map_err(|_| {
            Failure::Ffi(ffi_error(&format!(
                "spdecompress spfactor modifier {factor:?}"
            )))
        })
    };
    let num = shape_field(decoded, "num", "x")?;
    Ok((
        elem_bits,
        index_bits,
        parse(src_text)?,
        parse(dst_text)?,
        num,
    ))
}

fn same_register(left: &ObjectRef, right: &ObjectRef) -> AResult<bool> {
    let left_load = left
        .as_node::<TensorLoadObj>()
        .expect("validated TensorLoad");
    let right_load = right
        .as_node::<TensorLoadObj>()
        .expect("validated TensorLoad");
    if !same(&left_load.source, &right_load.source) {
        return Ok(false);
    }
    structural_equal(&Any::from(left.clone()), &Any::from(right.clone()))
}

/// `(name, value)` in `data`, `mdata`, `cdata` order.
pub fn named_registers<'v>(
    data: &'v [ObjectRef],
    metadata: &'v [ObjectRef],
    compressed: &'v [ObjectRef],
) -> Vec<(String, &'v ObjectRef)> {
    let mut named = Vec::new();
    for (group, values) in [("data", data), ("mdata", metadata), ("cdata", compressed)] {
        for (index, value) in values.iter().enumerate() {
            named.push((format!("{group}[{index}]"), value));
        }
    }
    named
}

fn present_lanes(decoded: &DecodedPtx, name: &str) -> AResult<Vec<ObjectRef>> {
    let mut lanes = Vec::new();
    for value in decoded.operand(name)? {
        match value {
            Some(value) => lanes.push(value.clone()),
            None => {
                return Err(Failure::Ffi(ffi_error(&format!(
                    "sunk lane in {}.{name}",
                    decoded.op_name
                ))))
            }
        }
    }
    Ok(lanes)
}

/// `resolve_ptx_spdecompress`.
pub fn resolve_ptx_spdecompress(decoded: &DecodedPtx) -> AResult<SpDecompressForm> {
    require_register_call(decoded, false)?;
    let op_name = decoded.op_name.as_str();
    let data = present_lanes(decoded, "data")?;
    let metadata = present_lanes(decoded, "mdata")?;
    let compressed = present_lanes(decoded, "cdata")?;
    let named = named_registers(&data, &metadata, &compressed);
    for (name, value) in &named {
        if value.as_node::<TensorLoadObj>().is_none() {
            return unsupported(format!("{op_name}.{name} must be a TensorLoad register"));
        }
    }
    for (left_index, (left_name, left)) in named.iter().enumerate() {
        for (right_name, right) in &named[left_index + 1..] {
            if same_register(left, right)? {
                return unsupported(format!(
                    "{op_name} has undefined register overlap between {left_name} and {right_name}"
                ));
            }
        }
    }
    let (elem_bits, index_bits, src, dst, num) = static_shape(decoded)?;
    Ok(SpDecompressForm {
        data,
        metadata,
        compressed,
        variant: if op_name == "tirx.ptx.spdecompress" {
            format!(
                "v2::reg::variant::SpDecompress<{elem_bits}, {index_bits}, {src}, {dst}, {num}>"
            )
        } else {
            format!("v2::reg::variant::SpCompress<{elem_bits}, {index_bits}, {num}>")
        },
    })
}

/// The validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_spdecompress(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_spdecompress(decoded, &parts, source_op_id)?;
    Ok(None)
}

struct RegisterLocation {
    backing: usize,
    start: PrimExpr,
    end: PrimExpr,
    owners: Vec<(String, PrimExpr)>,
}

impl<'a> Emitter<'a> {
    /// `ExpressionBindings.resolve_expression` over this kernel's exact Bind
    /// definitions (`Emitter.bindings` holds the same `Bind` map).
    fn spdecompress_resolve_expression(&self, expression: &PrimExpr) -> AResult<PrimExpr> {
        let substitutions = self.bindings.clone();
        let count = substitutions.iter().count();
        let substituted = if count == 0 {
            expression.clone()
        } else {
            let mut current: ObjectRef = oref(expression.clone());
            let mut resolved: Option<ObjectRef> = None;
            for _ in 0..count + 1 {
                let next = crate::substitute(current.clone(), substitutions.clone())?;
                if structural_equal(&Any::from(current.clone()), &Any::from(next.clone()))? {
                    resolved = Some(next);
                    break;
                }
                current = next;
            }
            let Some(resolved) = resolved else {
                return unsupported(
                    "compile-time expression proof encountered cyclic Bind definitions",
                );
            };
            PrimExpr::try_from(Any::from(resolved))?
        };
        simplify(&self.ctx.analyzer, &substituted)
    }

    fn spdecompress_register_location(
        &self,
        value: &ObjectRef,
    ) -> AResult<Option<RegisterLocation>> {
        let load = value
            .as_node::<TensorLoadObj>()
            .expect("validated TensorLoad");
        let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
            return not_covered("spdecompress register source is not a typed buffer");
        };
        let plan = self.memory_plan.resolve(&buffer)?;
        let Some(backing) = plan.backing_index else {
            return Ok(None);
        };
        let itemsize = plan.layout.itemsize;
        let elem_offset = plan.layout.elem_offset;
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let offset = self.physical_element_offset(&buffer, &indices)?;
        let shifted = op_binary("_OpAdd", expr_any(&offset), int_any(elem_offset))?;
        let scaled = op_binary("_OpMul", expr_any(&shifted), int_any(itemsize))?;
        let start = self.spdecompress_resolve_expression(&scaled)?;
        let end = op_binary("_OpAdd", expr_any(&start), int_any(itemsize))?;
        let mut owners = Vec::new();
        for (axis, coordinate) in self.physical_owner_coordinates(&buffer, &indices)? {
            owners.push((axis, self.spdecompress_resolve_expression(&coordinate)?));
        }
        Ok(Some(RegisterLocation {
            backing,
            start,
            end,
            owners,
        }))
    }

    fn spdecompress_register_relation(
        &self,
        left: &ObjectRef,
        right: &ObjectRef,
    ) -> AResult<&'static str> {
        let (Some(left), Some(right)) = (
            self.spdecompress_register_location(left)?,
            self.spdecompress_register_location(right)?,
        ) else {
            return Ok("unknown");
        };
        if left.backing != right.backing {
            return Ok("disjoint");
        }
        let analyzer = &self.ctx.analyzer;
        let prove = |name: &str, lhs: &PrimExpr, rhs: &PrimExpr| -> AResult<bool> {
            Ok(analyzer.can_prove(&op_binary(name, expr_any(lhs), expr_any(rhs))?)?)
        };
        if prove("_OpLE", &left.end, &right.start)? || prove("_OpLE", &right.end, &left.start)? {
            return Ok("disjoint");
        }
        let same_axes = left.owners.len() == right.owners.len()
            && left
                .owners
                .iter()
                .zip(right.owners.iter())
                .all(|((left_axis, _), (right_axis, _))| left_axis == right_axis);
        if same_axes {
            for ((_, left_owner), (_, right_owner)) in left.owners.iter().zip(right.owners.iter()) {
                if prove("_OpLT", left_owner, right_owner)?
                    || prove("_OpLT", right_owner, left_owner)?
                {
                    return Ok("disjoint");
                }
            }
        }
        let mut same_owner = same_axes;
        if same_owner {
            for ((_, left_owner), (_, right_owner)) in left.owners.iter().zip(right.owners.iter()) {
                if !analyzer.can_prove_equal(left_owner, right_owner)? {
                    same_owner = false;
                    break;
                }
            }
        }
        let overlapping_bytes =
            prove("_OpLT", &left.start, &right.end)? && prove("_OpLT", &right.start, &left.end)?;
        Ok(if same_owner && overlapping_bytes {
            "overlap"
        } else {
            "unknown"
        })
    }

    fn reject_spdecompress_register_overlap(&self, form: &SpDecompressForm) -> AResult<()> {
        let named = named_registers(&form.data, &form.metadata, &form.compressed);
        for (left_index, (left_name, left)) in named.iter().enumerate() {
            for (right_name, right) in &named[left_index + 1..] {
                let relation = self.spdecompress_register_relation(left, right)?;
                if relation == "overlap" {
                    return unsupported(format!(
                        "PTX sparse conversion has undefined register overlap between {left_name} and {right_name} through aliased buffer views"
                    ));
                }
                if relation != "disjoint" {
                    return unsupported(format!(
                        "PTX sparse conversion cannot prove disjoint physical register ranges between {left_name} and {right_name}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn emit_spdecompress_input_vector(
        &mut self,
        values: &[ObjectRef],
        op_name: &str,
        prefix: &str,
    ) -> AResult<String> {
        let mut registers = Vec::new();
        for (index, value) in values.iter().enumerate() {
            let bits =
                self.emit_as_unsigned_bits(value, 32, op_name, &format!("{prefix}_{index}"), None)?;
            registers.push(abi::register(&bits.code));
        }
        Ok(format!("vec![{}]", registers.join(", ")))
    }

    /// `emit_ptx_spdecompress`.
    pub fn emit_ptx_spdecompress(
        &mut self,
        decoded: &DecodedPtx,
        form: &SpDecompressForm,
        source_op_id: i64,
    ) -> AResult<()> {
        self.reject_spdecompress_register_overlap(form)?;
        let destinations: Vec<ObjectRef> = if decoded.op_name == "tirx.ptx.spdecompress" {
            form.data.clone()
        } else {
            form.metadata
                .iter()
                .chain(form.compressed.iter())
                .cloned()
                .collect()
        };
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "sparse_register",
            "sparse conversion predicate must be bool or integer",
        )?;
        let mask = region.mask.clone();
        self.emit_ptx_spdecompress_body(decoded, form, &destinations, source_op_id)?;
        self.close_predicated_region(region);
        for destination in &destinations {
            self.finish_predicated_destinations(
                decoded,
                &[Some(destination.clone())],
                &dtype_of(destination)?,
                &mask,
                source_op_id,
                false,
            )?;
        }
        Ok(())
    }

    fn emit_ptx_spdecompress_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &SpDecompressForm,
        destinations: &[ObjectRef],
        source_op_id: i64,
    ) -> AResult<()> {
        let arguments = if decoded.op_name == "tirx.ptx.spdecompress" {
            let metadata_vector = self.emit_spdecompress_input_vector(
                &form.metadata,
                &decoded.op_name,
                "spdecompress_metadata",
            )?;
            let compressed_vector = self.emit_spdecompress_input_vector(
                &form.compressed,
                &decoded.op_name,
                "spdecompress_compressed",
            )?;
            format!("({metadata_vector}, {compressed_vector})")
        } else {
            let dense_vector = self.emit_spdecompress_input_vector(
                &form.data,
                &decoded.op_name,
                "spcompress_data",
            )?;
            let descriptor = self.emit_expr(&decoded.scalar_operand("spdesc")?)?;
            let descriptor = self.as_warp_value(descriptor);
            format!("({dense_vector}, {})", abi::register(&descriptor.code))
        };
        let raw = self.control_name("spdecompress_raw");
        let results = self.control_name("spdecompress_results");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call(
            if decoded.op_name == "tirx.ptx.spdecompress" {
                "reg::spdecompress"
            } else {
                "reg::spcompress"
            },
            &site,
            &[arguments],
            Some(&form.variant),
        );
        self.emit_line(&format!("let {raw} = {call};"));
        self.emit_line(&format!(
            "let {results}: Vec<_> = {raw}.into_iter().map(v2_register_out).collect();"
        ));
        for (index, destination) in destinations.iter().enumerate() {
            let bits = RustValue::new(format!("{results}[{index}]"), "u32", Uniformity::Varying);
            let value = self.emit_from_unsigned_bits(
                bits,
                &dtype_of(destination)?,
                32,
                &decoded.op_name,
                &format!("spdecompress_result_{index}"),
            )?;
            self.emit_explicit_buffer_store(destination, value, source_op_id, None, None, None)?;
        }
        Ok(())
    }
}
