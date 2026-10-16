//! Expression emission as methods on the kernel emitter.

use crate::tvm_compat::{int_bits, int_value};
use tvm::ir::{FloatImmObj, IntImmObj, PrimExpr, TensorLoadObj};
use tvm::prim::{CastObj, LetObj, NotObj, RampObj, SelectObj, ShuffleObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

use super::super::analyze::util::{
    as_buffer, dtype_of, ffi_text, kind_or_bail, not_covered, oref, unsupported, AResult,
};
use super::super::analyze::vector::{
    classify_contiguous_ramp, classify_vector_buffer_load, classify_vector_extract, VectorForm,
};
use super::super::analyze::{topology_operands, util};
use super::NestedLoadSite;
use super::{
    join_control_provenance, join_uniformity, ControlProvenance, Emitter, RustValue, Uniformity,
};
use crate::decode::projected_buffer;
use crate::tables::{
    dtype_by_rust_type, expr_rust_type, is_integer_rust_type, is_low_precision_float,
    render_float_binary, render_integer_binary,
};

impl<'a> Emitter<'a> {
    pub fn zero_literal(rust_type: &str) -> AResult<String> {
        if rust_type == "bool" {
            return Ok("false".to_owned());
        }
        if rust_type == "f32" || rust_type == "f64" {
            return Ok(format!("0.0_{rust_type}"));
        }
        if is_integer_rust_type(rust_type) {
            return Ok(format!("0_{rust_type}"));
        }
        if rust_type == "F32x4" {
            return Ok("[0.0_f32; 4]".to_owned());
        }
        if rust_type == "U64x2" {
            return Ok("[0_u64; 2]".to_owned());
        }
        unsupported(format!("No numeric zero literal for Rust type {rust_type}"))
    }

    pub fn at_lane(&self, value: &RustValue, lane: &str) -> String {
        let lane = if lane == "lane" {
            self.selected_lane.as_deref().unwrap_or(lane)
        } else {
            lane
        };
        if value.uniformity == Uniformity::Uniform {
            return format!("({})", value.code);
        }
        if value.is_mask {
            return format!("({}).contains({lane})", value.code);
        }
        format!("({})[{lane}]", value.code)
    }

    pub fn materialize(&mut self, value: RustValue, prefix: &str) -> AResult<RustValue> {
        if !value.requires_statement {
            return Ok(value);
        }
        if value.uniformity != Uniformity::Uniform || value.is_mask {
            return unsupported(
                "Only uniform scalar expressions can require statement materialization",
            );
        }
        let name = self.temp(prefix);
        self.emit_line(&format!("let {name} = {};", value.code));
        Ok(RustValue {
            quantized_dtype: value.quantized_dtype.clone(),
            control_provenance: value.control_provenance,
            ..RustValue::new(name, value.rust_type.clone(), Uniformity::Uniform)
        })
    }

    pub fn as_warp_value(&mut self, value: RustValue) -> RustValue {
        if value.uniformity == Uniformity::Varying && !value.is_mask {
            return value;
        }
        if value.is_mask {
            let name = self.temp("mask_values");
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| {}.contains(lane));",
                value.code
            ));
            return RustValue {
                control_provenance: value.control_provenance,
                ..RustValue::new(name, "bool", Uniformity::Varying)
            };
        }
        let name = self.temp("broadcast");
        self.emit_line(&format!("let {name} = WarpValue::splat({});", value.code));
        RustValue {
            control_provenance: value.control_provenance,
            ..RustValue::new(name, value.rust_type.clone(), Uniformity::Varying)
        }
    }

    pub fn value_at_lane(&mut self, value: RustValue, lane: &str) -> AResult<RustValue> {
        let value = self.materialize(value, "selected_lane_input")?;
        if value.uniformity == Uniformity::Uniform {
            return Ok(value);
        }
        let code = if value.is_mask {
            format!("({}).contains({lane})", value.code)
        } else {
            format!("({})[{lane}]", value.code)
        };
        Ok(RustValue {
            quantized_dtype: value.quantized_dtype.clone(),
            control_provenance: value.control_provenance,
            ..RustValue::new(code, value.rust_type.clone(), Uniformity::Uniform)
        })
    }

    pub fn instrument(
        &mut self,
        expr: &ObjectRef,
        kind: &str,
        inputs: &[&RustValue],
        output: RustValue,
    ) -> AResult<RustValue> {
        let result_dtype = dtype_of(expr)?;
        let mut output = output;
        if kind != "TensorLoad"
            && is_low_precision_float(&result_dtype)
            && output.quantized_dtype.as_deref() != Some(result_dtype.as_str())
        {
            let prefix = format!("{}_result", kind_or_bail(expr)?.to_lowercase());
            output = self.coerce_dtype(output, &result_dtype, &prefix)?;
        }
        let provenance = if output.control_provenance == ControlProvenance::ElectSync
            || join_control_provenance(inputs.iter().copied()) == ControlProvenance::ElectSync
        {
            ControlProvenance::ElectSync
        } else {
            ControlProvenance::None
        };
        if provenance != output.control_provenance {
            output.control_provenance = provenance;
        }
        Ok(output)
    }

    pub fn boolean_lane(&self, value: &RustValue) -> AResult<String> {
        if value.rust_type != "bool" {
            return unsupported("A logical expression requires boolean operands");
        }
        Ok(self.at_lane(value, "lane"))
    }

    pub fn emit_varying_mask(
        &mut self,
        predicate: &str,
        control_provenance: ControlProvenance,
    ) -> RustValue {
        let name = self.temp("mask");
        if let Some(lane) = self.selected_lane.clone() {
            self.emit_line(&format!(
                "let {name} = {{ let lane = {lane}; {predicate} }};"
            ));
            return RustValue {
                control_provenance,
                ..RustValue::new(name, "bool", Uniformity::Uniform)
            };
        }
        self.emit_line(&format!(
            "let {name} = WarpMask::from_predicate(|lane| {predicate});"
        ));
        RustValue {
            control_provenance,
            ..RustValue::mask(name)
        }
    }

    pub fn emit_varying_value(
        &mut self,
        prefix: &str,
        code: &str,
        rust_type: &str,
        control_provenance: ControlProvenance,
    ) -> RustValue {
        let name = self.temp(prefix);
        if let Some(lane) = self.selected_lane.clone() {
            self.emit_line(&format!("let {name} = {{ let lane = {lane}; {code} }};"));
            return RustValue {
                control_provenance,
                ..RustValue::new(name, rust_type, Uniformity::Uniform)
            };
        }
        self.emit_line(&format!("let {name} = WarpValue::from_fn(|lane| {code});"));
        RustValue {
            control_provenance,
            ..RustValue::new(name, rust_type, Uniformity::Varying)
        }
    }

    fn can_share_warp_operation(&self) -> bool {
        self.analysis_capable || (self.use_typed_helpers && self.selected_lane.is_none())
    }

    /// One typed `frontend_expr` call.
    fn emit_warp_helper(
        &mut self,
        helper: &str,
        arguments: &[&RustValue],
        result_type: &str,
        prefix: &str,
    ) -> RustValue {
        let codes: Vec<String> = arguments
            .iter()
            .map(|value| {
                if value.uniformity == Uniformity::Varying {
                    format!("&{}", value.code)
                } else {
                    value.code.clone()
                }
            })
            .collect();
        let name = self.temp(prefix);
        self.emit_line(&format!(
            "let {name} = frontend_expr::{helper}({});",
            codes.join(", ")
        ));
        RustValue::new(name, result_type, Uniformity::Varying)
    }

    fn shape_suffix(arguments: &[&RustValue]) -> String {
        arguments
            .iter()
            .map(|value| {
                if value.uniformity == Uniformity::Varying {
                    'v'
                } else {
                    's'
                }
            })
            .collect()
    }

    fn emit_shared_binary(
        &mut self,
        kind: &str,
        lhs: &RustValue,
        rhs: &RustValue,
        rust_type: &str,
    ) -> Option<RustValue> {
        let shared = if is_integer_rust_type(rust_type) {
            matches!(kind, "Add" | "Sub" | "Mul" | "Min" | "Max")
        } else if rust_type == "f32" || rust_type == "f64" {
            matches!(kind, "Add" | "Sub" | "Mul" | "Div")
        } else {
            false
        };
        if !self.can_share_warp_operation()
            || !shared
            || lhs.rust_type != rust_type
            || rhs.rust_type != rust_type
            || lhs.is_mask
            || rhs.is_mask
        {
            return None;
        }
        let shape = Self::shape_suffix(&[lhs, rhs]);
        if shape == "ss" {
            return None;
        }
        let lower = kind.to_lowercase();
        Some(self.emit_warp_helper(
            &format!("{rust_type}::{lower}::{shape}"),
            &[lhs, rhs],
            rust_type,
            &lower,
        ))
    }

    fn emit_shared_integer_cast(
        &mut self,
        value: &RustValue,
        target_type: &str,
        prefix: &str,
    ) -> Option<RustValue> {
        if !self.can_share_warp_operation()
            || value.uniformity != Uniformity::Varying
            || value.is_mask
            || value.requires_statement
            || !is_integer_rust_type(&value.rust_type)
            || !is_integer_rust_type(target_type)
        {
            return None;
        }
        let result = self.emit_warp_helper(
            &format!("{}::cast::{target_type}", value.rust_type),
            &[value],
            target_type,
            prefix,
        );
        Some(RustValue {
            control_provenance: value.control_provenance,
            ..result
        })
    }

    fn emit_shared_call_atom(
        &mut self,
        prefix: &str,
        arguments: &[RustValue],
        result_type: &str,
    ) -> Option<RustValue> {
        if !self.can_share_warp_operation()
            || !arguments
                .iter()
                .any(|value| value.uniformity == Uniformity::Varying)
            || arguments.iter().any(|value| value.requires_statement)
        {
            return None;
        }
        let types: Vec<&str> = arguments.iter().map(|value| value.rust_type.as_str()).collect();
        let references: Vec<&RustValue> = arguments.iter().collect();
        let helper = match prefix {
            "bitwise_and" | "bitwise_or" | "bitwise_xor" | "shift_left" | "shift_right"
            | "make_float2" => {
                let base = if prefix == "make_float2" {
                    if types != ["f32", "f32"] {
                        return None;
                    }
                    prefix.to_owned()
                } else {
                    let rhs = if prefix == "shift_left" || prefix == "shift_right" {
                        "u32"
                    } else {
                        result_type
                    };
                    if !is_integer_rust_type(result_type) || types != [result_type, rhs] {
                        return None;
                    }
                    format!("{result_type}::{prefix}")
                };
                format!("{base}::{}", Self::shape_suffix(&references))
            }
            "reinterpret" => {
                let float = |rust_type: &str| rust_type == "f32" || rust_type == "f64";
                if types.len() != 1
                    || !((is_integer_rust_type(types[0]) && float(result_type))
                        || (float(types[0]) && is_integer_rust_type(result_type)))
                {
                    return None;
                }
                format!("{}::reinterpret::{result_type}", types[0])
            }
            _ => {
                let signature = match prefix {
                    "float2_x" | "float2_y" => ("u64", "f32"),
                    "float_as_uint" => ("f32", "u32"),
                    "uint_as_float" => ("u32", "f32"),
                    "fp16_bits_to_f32" | "bf16_bits_to_f32" => ("u16", "f32"),
                    "f32_to_fp16_bits" | "f32_to_bf16_bits" => ("f32", "u16"),
                    _ => return None,
                };
                if types != [signature.0] || result_type != signature.1 {
                    return None;
                }
                prefix.to_owned()
            }
        };
        Some(self.emit_warp_helper(&helper, &references, result_type, prefix))
    }

    /// `observe_pointer_bits`: an integer observation of the allocation-owned address.
    pub fn observe_pointer_bits(&mut self, value: RustValue) -> RustValue {
        if value.rust_type != "PhysicalPtr" {
            return value;
        }
        let address = self.temp("invocation_address");
        let mask = self
            .register_access_mask
            .clone()
            .filter(|mask| !mask.is_empty())
            .unwrap_or_else(|| "ctx.active_mask()".to_owned());
        self.emit_line(&format!(
            "let {address} = ({}).generic_addresses_u64(&ctx, {mask})?;",
            value.code
        ));
        RustValue {
            control_provenance: value.control_provenance,
            ..RustValue::new(address, "u64", Uniformity::Varying)
        }
    }

    fn cast_atom(
        &self,
        code: &str,
        source_type: &str,
        target_dtype: &str,
    ) -> AResult<(String, String)> {
        let target_type = expr_rust_type(self.ctx.schema, target_dtype)?;
        if self.ctx.schema.high_precision && crate::tables::is_promoted_float(target_dtype) {
            let promoted = if source_type == "bool" {
                format!("if {code} {{ 1.0_f64 }} else {{ 0.0_f64 }}")
            } else {
                format!("({code}) as f64")
            };
            return Ok((promoted, target_type));
        }
        if dtype_by_rust_type(source_type).is_none() {
            return unsupported(format!("Unknown Rust scalar source type {source_type}"));
        }
        match target_dtype {
            "float16" => {
                return Ok((
                    format!("fp16_bits_to_f32(f32_to_fp16_bits(({code}) as f32))"),
                    "f32".into(),
                ))
            }
            "bfloat16" => {
                return Ok((
                    format!("bf16_bits_to_f32(f32_to_bf16_bits(({code}) as f32))"),
                    "f32".into(),
                ))
            }
            "float8_e4m3fn" => {
                return Ok((
                    format!(
                        "float8_e4m3fn_bits_to_f32(f32_to_float8_e4m3fn_bits(({code}) as f32))"
                    ),
                    "f32".into(),
                ))
            }
            "float8_e8m0fnu" => {
                return Ok((
                    format!(
                        "float8_e8m0fnu_bits_to_f32(f32_to_float8_e8m0fnu_bits(({code}) as f32))"
                    ),
                    "f32".into(),
                ))
            }
            _ => {}
        }
        if source_type == target_type {
            return Ok((code.to_owned(), target_type));
        }
        if target_type == "bool" {
            if source_type == "bool" {
                return Ok((code.to_owned(), "bool".into()));
            }
            return Ok((
                format!("({code}) != {}", Self::zero_literal(source_type)?),
                "bool".into(),
            ));
        }
        if target_type == "f32" || target_type == "f64" {
            if source_type == "bool" {
                return Ok((
                    format!("if {code} {{ 1.0_{target_type} }} else {{ 0.0_{target_type} }}"),
                    target_type,
                ));
            }
            return Ok((format!("({code}) as {target_type}"), target_type));
        }
        if is_integer_rust_type(&target_type) {
            if source_type == "bool" {
                return Ok((
                    format!("if {code} {{ 1_{target_type} }} else {{ 0_{target_type} }}"),
                    target_type,
                ));
            }
            return Ok((format!("({code}) as {target_type}"), target_type));
        }
        unsupported(format!(
            "Rust Cast lowering is not implemented for {target_dtype}"
        ))
    }

    pub fn coerce_value(
        &mut self,
        value: RustValue,
        target_type: &str,
        prefix: &str,
    ) -> AResult<RustValue> {
        let value = self.materialize(value, &format!("{prefix}_input"))?;
        if value.rust_type == target_type {
            return Ok(value);
        }
        let Some(target_dtype) = dtype_by_rust_type(target_type) else {
            return not_covered(format!("Rust type {target_type} has no dtype spelling"));
        };
        if value.uniformity == Uniformity::Uniform {
            let (code, rust_type) = self.cast_atom(&value.code, &value.rust_type, target_dtype)?;
            return Ok(RustValue {
                control_provenance: value.control_provenance,
                ..RustValue::new(code, rust_type, Uniformity::Uniform)
            });
        }
        if let Some(result) = self.emit_shared_integer_cast(&value, target_type, prefix) {
            return Ok(result);
        }
        let lane_code = self.at_lane(&value, "lane");
        let (code, rust_type) = self.cast_atom(&lane_code, &value.rust_type, target_dtype)?;
        let result = if rust_type == "bool" {
            self.emit_varying_mask(&code, value.control_provenance)
        } else {
            self.emit_varying_value(prefix, &code, &rust_type, value.control_provenance)
        };
        Ok(RustValue {
            code: result.code,
            rust_type: result.rust_type,
            uniformity: result.uniformity,
            is_mask: result.is_mask,
            requires_statement: result.requires_statement,
            quantized_dtype: None,
            control_provenance: value.control_provenance,
        })
    }

    pub fn coerce_dtype(
        &mut self,
        value: RustValue,
        target_dtype: &str,
        prefix: &str,
    ) -> AResult<RustValue> {
        let value = self.materialize(value, &format!("{prefix}_input"))?;
        let target_type = expr_rust_type(self.ctx.schema, target_dtype)?;
        if value.quantized_dtype.as_deref() == Some(target_dtype) {
            return Ok(value);
        }
        if value.rust_type == target_type && !is_low_precision_float(target_dtype) {
            return Ok(value);
        }
        let quantized = if is_low_precision_float(target_dtype) {
            Some(target_dtype.to_owned())
        } else {
            None
        };
        if value.uniformity == Uniformity::Uniform {
            let (code, rust_type) = self.cast_atom(&value.code, &value.rust_type, target_dtype)?;
            return Ok(RustValue {
                quantized_dtype: quantized,
                control_provenance: value.control_provenance,
                ..RustValue::new(code, rust_type, Uniformity::Uniform)
            });
        }
        if let Some(result) = self.emit_shared_integer_cast(&value, &target_type, prefix) {
            return Ok(result);
        }
        let lane_code = self.at_lane(&value, "lane");
        let (code, rust_type) = self.cast_atom(&lane_code, &value.rust_type, target_dtype)?;
        let result = if rust_type == "bool" {
            self.emit_varying_mask(&code, value.control_provenance)
        } else {
            self.emit_varying_value(prefix, &code, &rust_type, value.control_provenance)
        };
        Ok(RustValue {
            code: result.code,
            rust_type: result.rust_type,
            uniformity: result.uniformity,
            is_mask: result.is_mask,
            requires_statement: result.requires_statement,
            quantized_dtype: quantized,
            control_provenance: value.control_provenance,
        })
    }

    pub fn as_i64(&mut self, value: RustValue) -> AResult<RustValue> {
        if !is_integer_rust_type(&value.rust_type) {
            return unsupported("The signed index ABI requires an integer value");
        }
        self.coerce_value(value, "i64", "index")
    }

    fn numeric_atom(&mut self, value: RustValue, target_type: &str) -> AResult<String> {
        let value = self.materialize(value, "materialized")?;
        if value.rust_type == "bool" || target_type == "bool" {
            return unsupported("A numeric expression cannot use a boolean operand");
        }
        if dtype_by_rust_type(&value.rust_type).is_none() {
            return unsupported(format!(
                "Unknown Rust scalar source type {}",
                value.rust_type
            ));
        }
        let code = self.at_lane(&value, "lane");
        Ok(if value.rust_type == target_type {
            code
        } else {
            format!("(({code}) as {target_type})")
        })
    }

    fn binary_numeric_operands(
        &mut self,
        expr: &ObjectRef,
        a: &ObjectRef,
        b: &ObjectRef,
        lhs: &RustValue,
        rhs: &RustValue,
    ) -> AResult<(String, String, String)> {
        let lhs_dtype = dtype_of(a)?;
        let rhs_dtype = dtype_of(b)?;
        let result_dtype = dtype_of(expr)?;
        if lhs_dtype != result_dtype || rhs_dtype != result_dtype {
            return unsupported("Mixed-dtype binary arithmetic requires an explicit TIRx Cast");
        }
        if self.ctx.schema.vector_dtype_abi(&result_dtype).is_some() {
            return unsupported(format!(
                "{result_dtype} has a packed storage ABI but no ordinary vector arithmetic ABI"
            ));
        }
        let target_type = expr_rust_type(self.ctx.schema, &result_dtype)?;
        if target_type == "bool" {
            return unsupported("Boolean arithmetic is not implemented");
        }
        let lhs_code = self.numeric_atom(lhs.clone(), &target_type)?;
        let rhs_code = self.numeric_atom(rhs.clone(), &target_type)?;
        Ok((lhs_code, rhs_code, target_type))
    }

    fn emit_pointer_arithmetic(
        &mut self,
        kind: &str,
        lhs: &RustValue,
        rhs: &RustValue,
    ) -> AResult<Option<RustValue>> {
        let lhs_is_pointer = lhs.rust_type == "PhysicalPtr";
        let rhs_is_pointer = rhs.rust_type == "PhysicalPtr";
        if !lhs_is_pointer && !rhs_is_pointer {
            return Ok(None);
        }
        // Only one-pointer additive arithmetic can retain an opaque address.
        // Other integer expressions consume its bound numeric address.
        if (kind != "Add" && kind != "Sub")
            || (lhs_is_pointer && rhs_is_pointer)
            || (kind == "Sub" && rhs_is_pointer)
        {
            return Ok(None);
        }
        let (pointer, integer) = if lhs_is_pointer {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        if !is_integer_rust_type(&integer.rust_type) {
            return unsupported("PhysicalPtr byte offsets must be integer scalars or lane vectors");
        }
        let offset = self.as_i64(integer.clone())?;
        let mut offset = self.as_warp_value(offset);
        if kind == "Sub" {
            let negated = self.temp("pointer_sub_offset");
            self.emit_line(&format!(
                "let {negated} = WarpValue::from_fn(|lane| {}[lane].wrapping_neg());",
                offset.code
            ));
            offset = RustValue::new(negated, "i64", Uniformity::Varying);
        }
        let result = self.temp("pointer_byte_offset");
        self.emit_line(&format!(
            "let {result} = {}.with_byte_offset(&{}, 1_usize, ctx.active_mask())?.into_byte_address(ctx.active_mask())?;",
            pointer.code, offset.code
        ));
        Ok(Some(RustValue::new(
            result,
            "PhysicalPtr",
            Uniformity::Varying,
        )))
    }

    fn emit_pointer_zero_comparison(
        &mut self,
        kind: &str,
        a: &ObjectRef,
        b: &ObjectRef,
        lhs: &RustValue,
        rhs: &RustValue,
    ) -> AResult<Option<RustValue>> {
        let lhs_is_pointer = lhs.rust_type == "PhysicalPtr";
        let rhs_is_pointer = rhs.rust_type == "PhysicalPtr";
        if !lhs_is_pointer && !rhs_is_pointer {
            return Ok(None);
        }
        if lhs_is_pointer == rhs_is_pointer {
            return Ok(None);
        }
        let numeric_expr = if lhs_is_pointer { b } else { a };
        if util::int_imm(numeric_expr) != Some(0) || (kind != "EQ" && kind != "NE") {
            return Ok(None);
        }
        let result = self.emit_pointer_is_null(if lhs_is_pointer { lhs } else { rhs })?;
        Ok(Some(if kind == "EQ" {
            result
        } else {
            RustValue {
                code: format!("!({})", result.code),
                ..result
            }
        }))
    }

    pub fn emit_pointer_is_null(&mut self, pointer: &RustValue) -> AResult<RustValue> {
        if pointer.rust_type == "PhysicalPtr" {
            return Ok(RustValue::new("false", "bool", Uniformity::Uniform));
        }
        if pointer.rust_type != "u64" {
            return unsupported("null comparison requires a uint64 address");
        }
        self.emit_call_atom("is_nullptr", &[pointer.clone()], "bool", |codes| {
            format!("({} == 0_u64)", codes[0])
        })
    }

    pub fn emit_call_atom(
        &mut self,
        prefix: &str,
        arguments: &[RustValue],
        result_type: &str,
        atom: impl Fn(&[String]) -> String,
    ) -> AResult<RustValue> {
        self.emit_quantized_call_atom(prefix, arguments, result_type, atom, None)
    }

    pub fn emit_quantized_call_atom(
        &mut self,
        prefix: &str,
        arguments: &[RustValue],
        result_type: &str,
        atom: impl Fn(&[String]) -> String,
        quantized_dtype: Option<&str>,
    ) -> AResult<RustValue> {
        let uniformity = join_uniformity(arguments.iter().map(|argument| argument.uniformity));
        let mut codes = Vec::new();
        for argument in arguments {
            codes.push(if argument.rust_type == "bool" {
                self.boolean_lane(argument)?
            } else {
                let target = argument.rust_type.clone();
                self.numeric_atom(argument.clone(), &target)?
            });
        }
        let code = atom(&codes);
        let result = if uniformity == Uniformity::Uniform {
            RustValue::new(code, result_type, Uniformity::Uniform)
        } else if result_type == "bool" {
            self.emit_varying_mask(&code, ControlProvenance::None)
        } else if let Some(result) = self.emit_shared_call_atom(prefix, arguments, result_type) {
            result
        } else {
            self.emit_varying_value(prefix, &code, result_type, ControlProvenance::None)
        };
        let result = RustValue {
            control_provenance: join_control_provenance(arguments.iter()),
            quantized_dtype: quantized_dtype.map(str::to_owned),
            ..result
        };
        let Some(call_expr) = self.call_expr_stack.last().cloned() else {
            return Ok(result);
        };
        let inputs: Vec<&RustValue> = arguments.iter().collect();
        self.instrument(&call_expr, &format!("Call:{prefix}"), &inputs, result)
    }

    fn merge_conditional_branch(
        &mut self,
        result: &str,
        result_type: &str,
        value: RustValue,
        branch_mask: &str,
    ) {
        if result_type != "bool" {
            let warp_value = self.as_warp_value(value);
            self.emit_line(&format!(
                "{result}.masked_assign({branch_mask}, &{});",
                warp_value.code
            ));
            return;
        }
        if value.uniformity == Uniformity::Uniform {
            self.emit_line(&format!(
                "if {} {{ {result} |= {branch_mask}; }}",
                value.code
            ));
            return;
        }
        if value.is_mask {
            self.emit_line(&format!("{result} |= {branch_mask} & ({});", value.code));
            return;
        }
        let branch_value_mask = self.temp("if_value_mask");
        self.emit_line(&format!(
            "let {branch_value_mask} = {}.to_mask(|_, value| *value) & {branch_mask};",
            value.code
        ));
        self.emit_line(&format!("{result} |= {branch_value_mask};"));
    }

    /// `(result, condition, selected)`.
    pub fn emit_conditional(
        &mut self,
        condition_expr: &ObjectRef,
        true_expr: &ObjectRef,
        false_expr: &ObjectRef,
        result_type: &str,
        prefix: &str,
    ) -> AResult<(RustValue, RustValue, RustValue)> {
        if let Some(lane) = self.selected_lane.clone() {
            let condition = self.emit_expr(condition_expr)?;
            let condition = self.value_at_lane(condition, &lane)?;
            if condition.rust_type != "bool" {
                return unsupported(format!("{prefix} condition did not lower to bool"));
            }
            let result = self.temp(&format!("{prefix}_selected_result"));
            self.emit_line(&format!("let {result}: {result_type};"));
            self.emit_line(&format!("if {} {{", condition.code));
            let true_value = self.emit_expr(true_expr)?;
            let true_value = self.value_at_lane(true_value, &lane)?;
            self.emit_line(&format!("    {result} = {};", true_value.code));
            self.emit_line("} else {");
            let false_value = self.emit_expr(false_expr)?;
            let false_value = self.value_at_lane(false_value, &lane)?;
            self.emit_line(&format!("    {result} = {};", false_value.code));
            self.emit_line("}");
            let quantized_dtype = if true_value.quantized_dtype == false_value.quantized_dtype {
                true_value.quantized_dtype.clone()
            } else {
                None
            };
            let provenance = join_control_provenance([&condition, &true_value, &false_value]);
            let value = RustValue {
                quantized_dtype,
                control_provenance: provenance,
                ..RustValue::new(result, result_type, Uniformity::Uniform)
            };
            return Ok((value.clone(), condition, value));
        }
        let mut condition = self.emit_expr(condition_expr)?;
        if condition.rust_type != "bool" {
            return unsupported(format!("{prefix} condition did not lower to bool"));
        }
        let parent = self.temp(&format!("{prefix}_parent"));
        let parent_context = self.temp(&format!("{prefix}_parent_context"));
        let result = self.temp(&format!("{prefix}_result"));
        self.emit_line(&format!("let {parent} = ctx.active_mask();"));
        let elect_control = condition.control_provenance == ControlProvenance::ElectSync;
        if elect_control && condition.uniformity != Uniformity::Uniform {
            self.emit_line(&format!("let {parent_context} = ctx;"));
        }
        if result_type == "bool" {
            self.emit_line(&format!("let mut {result} = WarpMask::EMPTY;"));
        } else {
            self.emit_line(&format!(
                "let mut {result} = WarpValue::splat({});",
                Self::zero_literal(result_type)?
            ));
        }
        let true_value;
        let false_value;
        if condition.uniformity == Uniformity::Uniform {
            self.emit_line(&format!("if {} {{", condition.code));
            true_value = self.emit_expr(true_expr)?;
            self.merge_conditional_branch(&result, result_type, true_value.clone(), &parent);
            self.emit_line("} else {");
            false_value = self.emit_expr(false_expr)?;
            self.merge_conditional_branch(&result, result_type, false_value.clone(), &parent);
            self.emit_line("}");
        } else {
            if !condition.is_mask {
                let predicate = self.boolean_lane(&condition)?;
                condition = self.emit_varying_mask(&predicate, condition.control_provenance);
            }
            let then_mask = self.temp(&format!("{prefix}_then_mask"));
            let else_mask = self.temp(&format!("{prefix}_else_mask"));
            self.emit_line(&format!(
                "let {then_mask} = {parent} & ({});",
                condition.code
            ));
            self.emit_line(&format!(
                "let {else_mask} = {parent} - ({});",
                condition.code
            ));
            self.emit_line(&format!("if !{then_mask}.is_empty() {{"));
            if elect_control {
                self.emit_line(&format!(
                    "ctx = {parent_context}.with_elect_sync_active_mask({parent}, {then_mask});"
                ));
            } else {
                self.emit_line(&format!("ctx.set_active_mask({then_mask});"));
            }
            true_value = self.emit_expr(true_expr)?;
            self.merge_conditional_branch(&result, result_type, true_value.clone(), &then_mask);
            if elect_control {
                self.emit_line(&format!("ctx = {parent_context};"));
            }
            self.emit_line("}");
            self.emit_line(&format!("if !{else_mask}.is_empty() {{"));
            if elect_control {
                self.emit_line(&format!(
                    "ctx = {parent_context}.with_elect_sync_active_mask({parent}, {else_mask});"
                ));
            } else {
                self.emit_line(&format!("ctx.set_active_mask({else_mask});"));
            }
            false_value = self.emit_expr(false_expr)?;
            self.merge_conditional_branch(&result, result_type, false_value.clone(), &else_mask);
            if elect_control {
                self.emit_line(&format!("ctx = {parent_context};"));
            }
            self.emit_line("}");
        }
        self.emit_line(&format!("ctx.set_active_mask({parent});"));
        let quantized_dtype = if true_value.quantized_dtype == false_value.quantized_dtype {
            true_value.quantized_dtype.clone()
        } else {
            None
        };
        let provenance = join_control_provenance([&condition, &true_value, &false_value]);
        let value = RustValue {
            is_mask: result_type == "bool",
            quantized_dtype,
            control_provenance: provenance,
            ..RustValue::new(result, result_type, Uniformity::Varying)
        };
        Ok((value.clone(), condition, value))
    }

    /// `(result, lhs)`.
    fn emit_lazy_logical(
        &mut self,
        a: &ObjectRef,
        b: &ObjectRef,
        kind: &str,
    ) -> AResult<(RustValue, RustValue)> {
        let lower = kind.to_lowercase();
        let mut lhs = self.emit_expr(a)?;
        if lhs.rust_type != "bool" {
            return unsupported(format!("{kind} requires boolean operands"));
        }
        if let Some(lane) = self.selected_lane.clone() {
            lhs = self.value_at_lane(lhs, &lane)?;
            let result = self.temp(&format!("{lower}_selected_result"));
            self.emit_line(&format!("let {result}: bool;"));
            let branch_on_rhs = if kind == "And" {
                lhs.code.clone()
            } else {
                format!("!({})", lhs.code)
            };
            self.emit_line(&format!("if {branch_on_rhs} {{"));
            let rhs = self.emit_expr(b)?;
            let rhs = self.value_at_lane(rhs, &lane)?;
            if rhs.rust_type != "bool" {
                return unsupported(format!("{kind} requires boolean operands"));
            }
            self.emit_line(&format!("    {result} = {};", rhs.code));
            self.emit_line("} else {");
            self.emit_line(&format!(
                "    {result} = {};",
                if kind == "And" { "false" } else { "true" }
            ));
            self.emit_line("}");
            return Ok((
                RustValue {
                    control_provenance: join_control_provenance([&lhs, &rhs]),
                    ..RustValue::new(result, "bool", Uniformity::Uniform)
                },
                lhs,
            ));
        }
        let parent = self.temp(&format!("{lower}_parent"));
        let parent_context = self.temp(&format!("{lower}_parent_context"));
        let rhs_mask = self.temp(&format!("{lower}_rhs_mask"));
        let result = self.temp(&format!("{lower}_result"));
        self.emit_line(&format!("let {parent} = ctx.active_mask();"));
        let elect_control = lhs.control_provenance == ControlProvenance::ElectSync;
        if elect_control {
            self.emit_line(&format!("let {parent_context} = ctx;"));
        }
        if lhs.uniformity == Uniformity::Uniform {
            lhs = self.materialize(lhs, &format!("{lower}_lhs"))?;
            let rhs_condition = if kind == "And" {
                lhs.code.clone()
            } else {
                format!("!({})", lhs.code)
            };
            self.emit_line(&format!(
                "let {rhs_mask} = if {rhs_condition} {{ {parent} }} else {{ WarpMask::EMPTY }};"
            ));
        } else {
            if !lhs.is_mask {
                let predicate = self.boolean_lane(&lhs)?;
                lhs = self.emit_varying_mask(&predicate, lhs.control_provenance);
            }
            if kind == "And" {
                self.emit_line(&format!("let {rhs_mask} = {parent} & ({});", lhs.code));
            } else {
                self.emit_line(&format!("let {rhs_mask} = {parent} - ({});", lhs.code));
            }
        }
        let initial = if kind == "And" {
            "WarpMask::EMPTY".to_owned()
        } else {
            format!("{parent} - {rhs_mask}")
        };
        self.emit_line(&format!("let mut {result} = {initial};"));
        self.emit_line(&format!("if !{rhs_mask}.is_empty() {{"));
        if elect_control {
            self.emit_line(&format!(
                "ctx = {parent_context}.with_elect_sync_active_mask({parent}, {rhs_mask});"
            ));
        } else {
            self.emit_line(&format!("ctx.set_active_mask({rhs_mask});"));
        }
        let mut rhs = self.emit_expr(b)?;
        if rhs.rust_type != "bool" {
            return unsupported(format!("{kind} requires boolean operands"));
        }
        if rhs.uniformity == Uniformity::Uniform {
            rhs = self.materialize(rhs, &format!("{lower}_rhs"))?;
        }
        self.merge_conditional_branch(&result, "bool", rhs.clone(), &rhs_mask);
        if elect_control {
            self.emit_line(&format!("ctx = {parent_context};"));
        }
        self.emit_line("}");
        self.emit_line(&format!("ctx.set_active_mask({parent});"));
        Ok((
            RustValue {
                control_provenance: join_control_provenance([&lhs, &rhs]),
                ..RustValue::mask(result)
            },
            lhs,
        ))
    }

    fn emit_vector_extract(
        &mut self,
        expr: &ObjectRef,
        shuffle: &ShuffleObj,
    ) -> AResult<RustValue> {
        match classify_vector_extract(self.ctx, expr, shuffle)? {
            VectorForm::Construct {
                vectors,
                vector_dtype,
                element_dtype,
            } => {
                let mut sources = Vec::new();
                for vector in &vectors {
                    sources.push(self.emit_expr(&oref(vector.clone()))?);
                }
                let expected_source_type = expr_rust_type(self.ctx.schema, &element_dtype)?;
                if sources
                    .iter()
                    .any(|source| source.rust_type != expected_source_type || source.is_mask)
                {
                    let lowered: Vec<String> = sources
                        .iter()
                        .map(|source| (&source.rust_type).to_string())
                        .collect();
                    return unsupported(format!(
                        "Shuffle construction sources lowered to {:?}, expected {expected_source_type}",
                        &lowered));
                }
                let atoms: Vec<String> = sources
                    .iter()
                    .map(|source| self.at_lane(source, "lane"))
                    .collect();
                let code = match vector_dtype.as_str() {
                    "bfloat16x2" => format!(
                        "(f32_to_bf16_bits({}) as u32) | ((f32_to_bf16_bits({}) as u32) << 16_u32)",
                        atoms[0], atoms[1]
                    ),
                    "uint32x2" => {
                        format!("({} as u64) | (({} as u64) << 32_u32)", atoms[0], atoms[1])
                    }
                    other => {
                        return not_covered(format!("Shuffle construction lowering missed {other}"))
                    }
                };
                let result_type = expr_rust_type(self.ctx.schema, &vector_dtype)?;
                let uniformity = join_uniformity(sources.iter().map(|source| source.uniformity));
                if uniformity == Uniformity::Uniform {
                    return Ok(RustValue::new(code, result_type, Uniformity::Uniform));
                }
                Ok(self.emit_varying_value(
                    "vector_construct",
                    &code,
                    &result_type,
                    ControlProvenance::None,
                ))
            }
            VectorForm::Extract {
                vector,
                vector_dtype,
                result_dtype,
                index,
            } => {
                let source = self.emit_expr(&oref(vector))?;
                let expected_type = expr_rust_type(self.ctx.schema, &vector_dtype)?;
                if source.rust_type != expected_type || source.is_mask {
                    return unsupported(format!(
                        "Shuffle source {vector_dtype} lowered to {}",
                        source.rust_type
                    ));
                }
                let source_atom = self.at_lane(&source, "lane");
                let code = match vector_dtype.as_str() {
                    "uint32x2" => format!(
                        "((({source_atom}) >> {}_u32) & 0xffff_ffff_u64) as u32",
                        index * 32
                    ),
                    "float16x2" => format!(
                        "fp16_bits_to_f32((({source_atom}) >> {}_u32) as u16)",
                        index * 16
                    ),
                    "bfloat16x2" => format!(
                        "bf16_bits_to_f32((({source_atom}) >> {}_u32) as u16)",
                        index * 16
                    ),
                    "float32x2" => format!(
                        "f32::from_bits(((({source_atom}) >> {}_u32) & 0xffff_ffff_u64) as u32)",
                        index * 32
                    ),
                    "uint64x2" | "float32x4" => format!("({source_atom})[{index}]"),
                    other => {
                        return unsupported(format!("Shuffle lowering missed vector dtype {other}"))
                    }
                };
                let result_type = expr_rust_type(self.ctx.schema, &result_dtype)?;
                if source.uniformity == Uniformity::Uniform {
                    return Ok(RustValue::new(code, result_type, Uniformity::Uniform));
                }
                Ok(self.emit_varying_value(
                    "vector_extract",
                    &code,
                    &result_type,
                    ControlProvenance::None,
                ))
            }
        }
    }

    fn emit_buffer_load_expr(
        &mut self,
        expr: &ObjectRef,
        load: &TensorLoadObj,
    ) -> AResult<RustValue> {
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return not_covered("TensorLoad source is not a typed buffer");
        };
        let result_dtype = classify_vector_buffer_load(self.ctx, expr, load, &source)?;
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let register_access_mask = self.register_access_mask.clone();
        // `self.buffer_load(expr.source, indices, source_node=expr, ...)`: the
        // kernel emitter swaps the loader around tile and TMEM lowerings.
        let value = match self.nested_load_site.clone() {
            None | Some(NestedLoadSite::Exact) => self.emit_buffer_load(
                &source,
                &indices,
                None,
                register_access_mask,
                false,
                result_dtype,
                Some(expr),
                None,
            )?,
            Some(NestedLoadSite::Site(site)) => self.emit_buffer_load(
                &source,
                &indices,
                None,
                register_access_mask,
                false,
                result_dtype,
                None,
                site,
            )?,
            Some(NestedLoadSite::ByBuffer(op_id)) => {
                let site = self.implicit_source_op_id(&source, op_id)?;
                self.emit_buffer_load(
                    &source,
                    &indices,
                    None,
                    register_access_mask,
                    false,
                    result_dtype,
                    None,
                    site,
                )?
            }
            Some(NestedLoadSite::AtLane(lane)) => self.emit_buffer_load_at_lane(
                &source,
                &indices,
                &lane,
                None,
                false,
                result_dtype,
                Some(expr),
                None,
            )?,
            Some(NestedLoadSite::AtLaneSite(lane, site)) => self.emit_buffer_load_at_lane(
                &source,
                &indices,
                &lane,
                None,
                false,
                result_dtype,
                None,
                site,
            )?,
            Some(NestedLoadSite::Reject(message)) => return unsupported(message),
        };
        self.instrument(expr, "TensorLoad", &[], value)
    }

    pub fn emit_expr(&mut self, expr: &ObjectRef) -> AResult<RustValue> {
        if self.ctx.schema.high_precision {
            if let Ok(dtype) = dtype_of(expr) {
                super::high_precision::validate_dtype(&dtype)?;
            }
        }
        if self.diagnostics.invalid_values.contains(expr) {
            return Err(util::Failure::Recorded);
        }
        let kind = kind_or_bail(expr)?;
        if let Some(imm) = expr.as_node::<IntImmObj>() {
            let dtype = dtype_of(expr)?;
            let bits = int_bits(imm)?;
            if dtype == "bool" {
                return Ok(RustValue::new(
                    if bits != 0 { "true" } else { "false" },
                    "bool",
                    Uniformity::Uniform,
                ));
            }
            let rust_type = expr_rust_type(self.ctx.schema, &dtype)?;
            if !is_integer_rust_type(&rust_type) {
                return unsupported(format!("IntImm cannot have dtype {dtype}"));
            }
            let literal = if dtype.starts_with("int") {
                int_value(imm)?.to_string()
            } else {
                bits.to_string()
            };
            return Ok(RustValue::new(
                format!("{literal}_{rust_type}"),
                rust_type,
                Uniformity::Uniform,
            ));
        }
        if let Some(imm) = expr.as_node::<FloatImmObj>() {
            let dtype = dtype_of(expr)?;
            if !matches!(
                dtype.as_str(),
                "float16" | "bfloat16" | "float32" | "float64" | "float8_e4m3fn" | "float8_e8m0fnu"
            ) {
                return unsupported(format!("FloatImm lowering is not implemented for {dtype}"));
            }
            let value = imm.value;
            let rust_float = if dtype == "float64" { "f64" } else { "f32" };
            let mut code = if value.is_nan() {
                format!("{rust_float}::NAN")
            } else if value.is_infinite() {
                if value > 0.0 {
                    format!("{rust_float}::INFINITY")
                } else {
                    format!("{rust_float}::NEG_INFINITY")
                }
            } else {
                format!("{value:?}_{rust_float}")
            };
            code = match dtype.as_str() {
                "float16" => format!("fp16_bits_to_f32(f32_to_fp16_bits({code}))"),
                "bfloat16" => format!("bf16_bits_to_f32(f32_to_bf16_bits({code}))"),
                "float8_e4m3fn" => {
                    format!("float8_e4m3fn_bits_to_f32(f32_to_float8_e4m3fn_bits({code}))")
                }
                "float8_e8m0fnu" => {
                    format!("float8_e8m0fnu_bits_to_f32(f32_to_float8_e8m0fnu_bits({code}))")
                }
                _ => code,
            };
            if self.ctx.schema.high_precision {
                return Ok(RustValue::new(
                    format!("({code}) as f64"),
                    "f64",
                    Uniformity::Uniform,
                ));
            }
            return Ok(RustValue {
                quantized_dtype: if is_low_precision_float(&dtype) {
                    Some(dtype)
                } else {
                    None
                },
                ..RustValue::new(code, rust_float, Uniformity::Uniform)
            });
        }
        if let Some(let_expr) = expr.as_node::<LetObj>() {
            let value = self.emit_expr(&oref(let_expr.value.clone()))?;
            let value = self.materialize(value, "let_value")?;
            let binding_name = self.temp("let_binding");
            self.emit_line(&format!("let {binding_name} = ({}).clone();", value.code));
            let binding = RustValue {
                code: binding_name,
                requires_statement: false,
                ..value
            };
            let key = oref(let_expr.var.clone());
            let previous = self.variables.get(&key).cloned();
            self.variables.set(key.clone(), binding);
            let result = self.emit_expr(&oref(let_expr.body.clone()));
            match previous {
                None => {
                    self.variables.remove(&key);
                }
                Some(previous) => self.variables.set(key, previous),
            }
            return result;
        }
        if let Some(variable) = util::as_var(expr) {
            return match self.lookup_variable(expr) {
                Some(value) => Ok(value),
                None => unsupported(format!(
                    "Unbound TIRx variable in Rust codegen: {}",
                    ffi_text(&variable.name)
                )),
            };
        }
        if let Some(ramp) = expr.as_node::<RampObj>() {
            classify_contiguous_ramp(self.ctx, expr, ramp, None)?;
            return unsupported("Ramp is valid only as a contiguous vector buffer index");
        }
        if let Some(shuffle) = expr.as_node::<ShuffleObj>() {
            return self.emit_vector_extract(expr, shuffle);
        }
        if let Some(load) = expr.as_node::<TensorLoadObj>() {
            return self.emit_buffer_load_expr(expr, load);
        }
        if let Some(cast) = expr.as_node::<CastObj>() {
            let value = self.emit_expr(&oref(cast.value.clone()))?;
            let target_dtype = dtype_of(expr)?;
            let result = self.coerce_dtype(value.clone(), &target_dtype, "cast")?;
            return self.instrument(expr, "Cast", &[&value], result);
        }
        if let Some((name, operands)) = util::bitwise_expr(expr) {
            return self.emit_bitwise_node(expr, name, &operands);
        }
        if self.ctx.schema.integer_binary_node_kinds.contains(kind) {
            let (a, b) = topology_operands(expr).expect("binary node");
            let lower = kind.to_lowercase();
            let lhs = self.emit_expr(&a)?;
            let lhs = self.materialize(lhs, &format!("{lower}_lhs"))?;
            let rhs = self.emit_expr(&b)?;
            let rhs = self.materialize(rhs, &format!("{lower}_rhs"))?;
            if let Some(pointer_result) = self.emit_pointer_arithmetic(kind, &lhs, &rhs)? {
                return Ok(pointer_result);
            }
            let lhs = self.observe_pointer_bits(lhs);
            let rhs = self.observe_pointer_bits(rhs);
            let (lhs_code, rhs_code, rust_type) =
                self.binary_numeric_operands(expr, &a, &b, &lhs, &rhs)?;
            let uniformity = join_uniformity([lhs.uniformity, rhs.uniformity]);
            let (code, requires_statement) = if is_integer_rust_type(&rust_type) {
                render_integer_binary(
                    kind,
                    &lhs_code,
                    &rhs_code,
                    &rust_type,
                    self.nonnegative_floor,
                    self.integer_error_context,
                    self.integer_error_label.as_deref(),
                )?
            } else {
                (
                    render_float_binary(kind, &lhs_code, &rhs_code, &rust_type)?,
                    false,
                )
            };
            let result = if uniformity == Uniformity::Uniform {
                RustValue {
                    requires_statement,
                    ..RustValue::new(code, rust_type, uniformity)
                }
            } else if let Some(result) = self.emit_shared_binary(kind, &lhs, &rhs, &rust_type) {
                result
            } else if requires_statement {
                let name = self.temp(&lower);
                self.emit_line(&format!(
                    "let mut {name} = WarpValue::splat({});",
                    Self::zero_literal(&rust_type)?
                ));
                self.emit_line("for lane in ctx.active_mask() {");
                self.emit_line(&format!("    {name}[lane] = {code};"));
                self.emit_line("}");
                RustValue::new(name, rust_type, Uniformity::Varying)
            } else {
                self.emit_varying_value(&lower, &code, &rust_type, ControlProvenance::None)
            };
            return self.instrument(expr, kind, &[&lhs, &rhs], result);
        }
        if matches!(kind, "LT" | "LE" | "GT" | "GE" | "EQ" | "NE") {
            let operator = match kind {
                "LT" => "<",
                "LE" => "<=",
                "GT" => ">",
                "GE" => ">=",
                "EQ" => "==",
                _ => "!=",
            };
            let (a, b) = topology_operands(expr).expect("comparison node");
            let lhs = self.emit_expr(&a)?;
            let rhs = self.emit_expr(&b)?;
            if let Some(pointer_result) =
                self.emit_pointer_zero_comparison(kind, &a, &b, &lhs, &rhs)?
            {
                return self.instrument(expr, kind, &[&lhs, &rhs], pointer_result);
            }
            let lhs_dtype = dtype_of(&a)?;
            let rhs_dtype = dtype_of(&b)?;
            if lhs_dtype != rhs_dtype {
                return unsupported("Mixed-dtype comparison requires an explicit TIRx Cast");
            }
            if self.ctx.schema.vector_dtype_abi(&lhs_dtype).is_some() {
                return unsupported(format!(
                    "{lhs_dtype} has a packed storage ABI but no ordinary vector comparison ABI"
                ));
            }
            let operand_type = expr_rust_type(self.ctx.schema, &lhs_dtype)?;
            let lhs = self.observe_pointer_bits(lhs);
            let rhs = self.observe_pointer_bits(rhs);
            let (lhs_code, rhs_code) = if operand_type == "bool" {
                if kind != "EQ" && kind != "NE" {
                    return unsupported(format!("{kind} is not defined for boolean operands"));
                }
                (self.boolean_lane(&lhs)?, self.boolean_lane(&rhs)?)
            } else {
                (
                    self.numeric_atom(lhs.clone(), &operand_type)?,
                    self.numeric_atom(rhs.clone(), &operand_type)?,
                )
            };
            let uniformity = join_uniformity([lhs.uniformity, rhs.uniformity]);
            let result = if uniformity == Uniformity::Uniform {
                RustValue::new(
                    format!("({lhs_code}) {operator} ({rhs_code})"),
                    "bool",
                    uniformity,
                )
            } else {
                self.emit_varying_mask(
                    &format!("{lhs_code} {operator} {rhs_code}"),
                    ControlProvenance::None,
                )
            };
            return self.instrument(expr, kind, &[&lhs, &rhs], result);
        }
        if kind == "And" || kind == "Or" {
            let (a, b) = topology_operands(expr).expect("logical node");
            let (result, lhs) = self.emit_lazy_logical(&a, &b, kind)?;
            return self.instrument(expr, kind, &[&lhs], result);
        }
        if let Some(not) = expr.as_node::<NotObj>() {
            let value = self.emit_expr(&oref(not.a.clone()))?;
            if value.rust_type != "bool" {
                return unsupported("Not requires a boolean operand");
            }
            let result = if value.uniformity == Uniformity::Uniform {
                RustValue::new(format!("!({})", value.code), "bool", Uniformity::Uniform)
            } else if value.is_mask {
                RustValue::mask(format!("(!({}))", value.code))
            } else {
                let predicate = format!("!{}", self.boolean_lane(&value)?);
                self.emit_varying_mask(&predicate, ControlProvenance::None)
            };
            return self.instrument(expr, kind, &[&value], result);
        }
        if let Some(select) = expr.as_node::<SelectObj>() {
            let true_expr = oref(select.true_value.clone());
            let false_expr = oref(select.false_value.clone());
            let condition_expr = oref(select.condition.clone());
            let true_dtype = dtype_of(&true_expr)?;
            let false_dtype = dtype_of(&false_expr)?;
            let result_dtype = dtype_of(expr)?;
            if true_dtype != result_dtype || false_dtype != result_dtype {
                return unsupported("Mixed-dtype Select branches require an explicit TIRx Cast");
            }
            let result_type = expr_rust_type(self.ctx.schema, &result_dtype)?;
            let (result, condition, selected) = self.emit_conditional(
                &condition_expr,
                &true_expr,
                &false_expr,
                &result_type,
                "select",
            )?;
            return self.instrument(expr, "Select", &[&condition, &selected], result);
        }
        if kind == "Call" {
            if projected_buffer(expr)?.is_some() {
                return self.emit_buffer_data_pointer(expr);
            }
            return match self.emit_call(expr)? {
                Some(value) => Ok(value),
                None => unsupported(format!(
                    "{} is a statement and cannot be used as an expression",
                    crate::decode::call_name(expr)?.unwrap_or_default()
                )),
            };
        }
        unsupported(format!(
            "Rust expression codegen is not implemented for {kind}"
        ))
    }
}
