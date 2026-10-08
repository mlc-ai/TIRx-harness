//! Validation and emission of the ptx_bitops instruction family.

use crate::analyze::util::{dtype_of, ffi_error, unsupported, AResult, Failure};
use crate::decode::ptx::{unmodeled_form, DecodedPtx};
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, table_marker, PTX_TYPE_MARKERS};
use crate::emit::{Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

/// The Rust carrier type of `dtype`; an unknown dtype is an internal error.
fn carrier_rust_type(dtype: &str) -> AResult<&'static str> {
    Ok(match dtype {
        "int16" => "i16",
        "int32" => "i32",
        "int64" => "i64",
        "uint16" => "u16",
        "uint32" => "u32",
        "uint64" => "u64",
        _ => return Err(Failure::Ffi(ffi_error(&format!("KeyError: {dtype:?}")))),
    })
}

pub struct BitForm {
    pub destination: ObjectRef,
    pub sources: Vec<ObjectRef>,
    pub instruction: &'static str,
    pub variant: String,
    pub source_rust_types: Vec<String>,
    pub source_bit_widths: Vec<Option<i64>>,
    pub output_rust_type: String,
    pub output_bit_width: Option<i64>,
    pub predicate_output: bool,
}

/// `sources[0]` (a missing source is an internal error).
fn first_source(sources: &[ObjectRef]) -> AResult<&ObjectRef> {
    match sources.first() {
        Some(source) => Ok(source),
        None => Err(Failure::Ffi(ffi_error(
            "IndexError: tuple index out of range",
        ))),
    }
}

/// `call.op_name.rsplit(".", 1)[-1]` of one closed instruction.
fn instruction_name(op_name: &str) -> &'static str {
    match op_name {
        "tirx.ptx.and" => "and",
        "tirx.ptx.or" => "or",
        "tirx.ptx.xor" => "xor",
        "tirx.ptx.not" => "not",
        "tirx.ptx.brev" => "brev",
        "tirx.ptx.cnot" => "cnot",
        "tirx.ptx.shl" => "shl",
        "tirx.ptx.shr" => "shr",
        _ => unreachable!("closed bit-manipulation instruction"),
    }
}

fn form(
    destination: ObjectRef,
    sources: Vec<ObjectRef>,
    instruction: &'static str,
    variant: String,
    source_rust_types: Vec<String>,
    source_bit_widths: Vec<Option<i64>>,
    output_rust_type: String,
    output_bit_width: Option<i64>,
) -> BitForm {
    BitForm {
        destination,
        sources,
        instruction,
        variant,
        source_rust_types,
        source_bit_widths,
        output_rust_type,
        output_bit_width,
        predicate_output: false,
    }
}

/// `resolve_ptx_bit_op`.
pub fn resolve_ptx_bit_op(decoded: &DecodedPtx) -> AResult<BitForm> {
    let operands = require_register_call(decoded, false)?;
    let op_name = decoded.op_name.as_str();
    if operands.destinations.len() != 1 {
        return unsupported(format!(
            "{op_name} requires exactly one register destination, got {}",
            operands.destinations.len()
        ));
    }
    let destination = operands.destinations[0].clone();
    let sources = operands.sources;
    let ptx_type = decoded.modifier("type")?;
    let type_marker = table_marker(PTX_TYPE_MARKERS, ptx_type);

    if op_name == "tirx.ptx.shr" && !ptx_type.starts_with('b') {
        if !matches!(
            type_marker,
            Some("I16" | "I32" | "I64" | "U16" | "U32" | "U64")
        ) {
            return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
        }
        let rust_type = carrier_rust_type(&dtype_of(first_source(&sources)?)?)?;
        return Ok(form(
            destination,
            sources,
            "shr",
            format!("v2::reg::variant::{}", type_marker.expect("marker")),
            vec![rust_type.to_owned(), "u32".to_owned()],
            vec![None, None],
            rust_type.to_owned(),
            None,
        ));
    }

    if op_name == "tirx.ptx.selp" {
        // Selection preserves payload bits for every table type. The selector
        // is a predicate operand, independent of instruction predication.
        let digits = ptx_type.get(1..).unwrap_or("");
        let Ok(width) = digits.parse::<i64>() else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "ValueError: invalid literal for int() with base 10: {:?}",
                digits
            ))));
        };
        let reordered = vec![sources[2].clone(), sources[0].clone(), sources[1].clone()];
        return Ok(form(
            destination,
            reordered,
            "selp",
            format!("v2::reg::variant::B{width}"),
            vec!["bool".to_owned(), format!("u{width}"), format!("u{width}")],
            vec![None, Some(width), Some(width)],
            format!("u{width}"),
            Some(width),
        ));
    }

    if op_name == "tirx.ptx.abs" {
        if !matches!(type_marker, Some("I16" | "I32" | "I64")) {
            return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
        }
        let rust_type = carrier_rust_type(&dtype_of(first_source(&sources)?)?)?;
        return Ok(form(
            destination,
            sources,
            "abs",
            format!("v2::reg::variant::{}", type_marker.expect("marker")),
            vec![rust_type.to_owned()],
            vec![None],
            rust_type.to_owned(),
            None,
        ));
    }

    if op_name == "tirx.ptx.bfe" {
        if !matches!(type_marker, Some("U32" | "U64" | "I32" | "I64")) {
            return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
        }
        let value_type = carrier_rust_type(&dtype_of(first_source(&sources)?)?)?;
        return Ok(form(
            destination,
            sources,
            "bfe",
            format!("v2::reg::variant::{}", type_marker.expect("marker")),
            vec![value_type.to_owned(), "u32".to_owned(), "u32".to_owned()],
            vec![None, None, None],
            value_type.to_owned(),
            None,
        ));
    }

    if op_name == "tirx.ptx.bfind" {
        if !matches!(type_marker, Some("U32" | "U64" | "I32" | "I64")) {
            return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
        }
        let mode = if decoded.modifier("shiftamt")?.is_empty() {
            "BitPosition"
        } else {
            "ShiftAmount"
        };
        let variant = format!(
            "v2::reg::variant::Bfind<v2::reg::variant::{}, v2::reg::variant::{mode}>",
            type_marker.expect("marker")
        );
        let value_type = carrier_rust_type(&dtype_of(first_source(&sources)?)?)?;
        return Ok(form(
            destination,
            sources,
            "bfind",
            variant,
            vec![value_type.to_owned()],
            vec![None],
            "u32".to_owned(),
            None,
        ));
    }

    if op_name == "tirx.ptx.szext" {
        if !matches!(type_marker, Some("U32" | "I32")) {
            return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
        }
        let mode = if decoded.modifier("mode")? == "clamp" {
            "Clamp"
        } else {
            "Wrap"
        };
        let variant = format!(
            "v2::reg::variant::Szext<v2::reg::variant::{}, v2::reg::variant::{mode}>",
            type_marker.expect("marker")
        );
        let value_type = carrier_rust_type(&dtype_of(first_source(&sources)?)?)?;
        return Ok(form(
            destination,
            sources,
            "szext",
            variant,
            vec![value_type.to_owned(), "u32".to_owned()],
            vec![None, None],
            value_type.to_owned(),
            None,
        ));
    }

    if matches!(
        op_name,
        "tirx.ptx.and" | "tirx.ptx.or" | "tirx.ptx.xor" | "tirx.ptx.not"
    ) && ptx_type == "pred"
    {
        let count = sources.len();
        let mut form = form(
            destination,
            sources,
            instruction_name(op_name),
            "v2::reg::variant::Pred".to_owned(),
            vec!["bool".to_owned(); count],
            vec![None; count],
            "bool".to_owned(),
            None,
        );
        form.predicate_output = true;
        return Ok(form);
    }

    let Ok(width) = ptx_type
        .strip_prefix('b')
        .unwrap_or(ptx_type)
        .parse::<i64>()
    else {
        return unmodeled_form(decoded, &format!("has no bit width for {:?}", ptx_type));
    };
    let (Some(type_marker), true) = (type_marker, [16, 32, 64].contains(&width)) else {
        return unmodeled_form(decoded, &format!("has no closed {:?} variant", ptx_type));
    };

    let (source_types, bit_widths, instruction, variant, output_type, output_width): (
        Vec<String>,
        Vec<Option<i64>>,
        &'static str,
        String,
        String,
        Option<i64>,
    ) = match op_name {
        "tirx.ptx.bfi" => (
            vec![
                format!("u{width}"),
                format!("u{width}"),
                "u32".to_owned(),
                "u32".to_owned(),
            ],
            vec![Some(width), Some(width), None, None],
            "bfi",
            format!("v2::reg::variant::{type_marker}"),
            format!("u{width}"),
            Some(width),
        ),
        "tirx.ptx.bmsk" => {
            let mode = if decoded.modifier("mode")? == "clamp" {
                "Clamp"
            } else {
                "Wrap"
            };
            (
                vec!["u32".to_owned(), "u32".to_owned()],
                vec![Some(32), Some(32)],
                "bmsk",
                format!("v2::reg::variant::Bmsk<v2::reg::variant::{mode}>"),
                "u32".to_owned(),
                Some(32),
            )
        }
        "tirx.ptx.brev" | "tirx.ptx.cnot" | "tirx.ptx.and" | "tirx.ptx.or" | "tirx.ptx.xor"
        | "tirx.ptx.not" => (
            vec![format!("u{width}"); sources.len()],
            vec![Some(width); sources.len()],
            instruction_name(op_name),
            format!("v2::reg::variant::{type_marker}"),
            format!("u{width}"),
            Some(width),
        ),
        "tirx.ptx.clz" | "tirx.ptx.popc" => (
            vec![format!("u{width}")],
            vec![Some(width)],
            if op_name == "tirx.ptx.clz" {
                "clz"
            } else {
                "popc"
            },
            format!("v2::reg::variant::{type_marker}"),
            "u32".to_owned(),
            None,
        ),
        "tirx.ptx.shl" | "tirx.ptx.shr" => (
            vec![format!("u{width}"), "u32".to_owned()],
            vec![Some(width), None],
            instruction_name(op_name),
            format!("v2::reg::variant::{type_marker}"),
            format!("u{width}"),
            Some(width),
        ),
        "tirx.ptx.prmt" => {
            let mode = decoded.modifier("mode")?;
            let marker = match mode {
                "f4e" => Some("F4e"),
                "b4e" => Some("B4e"),
                "rc8" => Some("Rc8"),
                "ecl" => Some("Ecl"),
                "ecr" => Some("Ecr"),
                "rc16" => Some("Rc16"),
                _ => None,
            };
            if !mode.is_empty() && marker.is_none() {
                return unmodeled_form(
                    decoded,
                    &format!("has no closed {:?} permutation variant", mode),
                );
            }
            let variant = match marker {
                Some(marker) if !mode.is_empty() => {
                    format!("v2::reg::variant::Prmt<v2::reg::variant::{marker}>")
                }
                _ => "v2::reg::variant::B32".to_owned(),
            };
            (
                vec!["u32".to_owned(); 3],
                vec![Some(32); 3],
                "prmt",
                variant,
                "u32".to_owned(),
                Some(32),
            )
        }
        "tirx.ptx.fns" => (
            vec!["u32".to_owned(), "u32".to_owned(), "i32".to_owned()],
            vec![Some(32), Some(32), None],
            "fns",
            "v2::reg::variant::B32".to_owned(),
            "u32".to_owned(),
            Some(32),
        ),
        "tirx.ptx.shf" => {
            let direction = if decoded.modifier("dir")? == "l" {
                "Left"
            } else {
                "Right"
            };
            let mode = if decoded.modifier("mode")? == "clamp" {
                "Clamp"
            } else {
                "Wrap"
            };
            (
                vec!["u32".to_owned(), "u32".to_owned(), "u32".to_owned()],
                vec![Some(32), Some(32), None],
                "shf",
                format!(
                    "v2::reg::variant::Shf<v2::reg::variant::{direction}, v2::reg::variant::{mode}>"
                ),
                "u32".to_owned(),
                Some(32),
            )
        }
        _ => return unmodeled_form(decoded, "has no closed engine instruction"),
    };
    Ok(form(
        destination,
        sources,
        instruction,
        variant,
        source_types,
        bit_widths,
        output_type,
        output_width,
    ))
}

/// `resolve_ptx_bit_op`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_bit_op(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_bit_op(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    fn emit_bitop_source(
        &mut self,
        op_name: &str,
        form: &BitForm,
        index: usize,
    ) -> AResult<RustValue> {
        let expression = &form.sources[index];
        if let Some(bit_width) = form.source_bit_widths[index] {
            return self.emit_as_unsigned_bits(
                expression,
                bit_width,
                op_name,
                &format!("bitop_source_{index}_bits"),
                None,
            );
        }
        let expected = form.source_rust_types[index].as_str();
        let emitted = self.emit_expr(expression)?;
        let emitted = self.coerce_value(emitted, expected, &format!("bitop_source_{index}"))?;
        let value = self.as_warp_value(emitted);
        if value.rust_type != expected {
            return unsupported(format!(
                "{op_name} source {index} lowered to {}, expected {expected}",
                value.rust_type
            ));
        }
        Ok(value)
    }

    /// `emit_ptx_bit_op`.
    pub fn emit_ptx_bit_op(
        &mut self,
        decoded: &DecodedPtx,
        form: &BitForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "bitop",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        self.emit_bit_op_body(decoded, form, source_op_id)?;
        self.close_predicated_region(region);
        self.finish_predicated_destinations(
            decoded,
            &[Some(form.destination.clone())],
            &dtype_of(&form.destination)?,
            &mask,
            source_op_id,
            form.predicate_output,
        )
    }

    fn emit_bit_op_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &BitForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let mut sources = Vec::new();
        for index in 0..form.sources.len() {
            sources.push(self.emit_bitop_source(op_name, form, index)?);
        }
        let result = self.emit_register_result(
            &format!("reg::{}", form.instruction),
            &form.variant,
            &sources,
            source_op_id,
            None,
        );

        let mut output = RustValue::new(
            result.clone(),
            form.output_rust_type.clone(),
            Uniformity::Varying,
        );
        if form.predicate_output {
            let predicate_bits = self.control_name("decoded_bitop_predicate_u32");
            self.emit_line(&format!(
                "let {predicate_bits} = WarpValue::from_fn(|lane| u32::from({result}[lane]));"
            ));
            output = RustValue::new(predicate_bits, "u32", Uniformity::Varying);
        } else if let Some(output_bit_width) = form.output_bit_width {
            let destination_dtype = dtype_of(&form.destination)?;
            if output_bit_width == 16
                && matches!(destination_dtype.as_str(), "float16" | "bfloat16")
            {
                return self.emit_explicit_buffer_store(
                    &form.destination,
                    output,
                    source_op_id,
                    None,
                    None,
                    Some("uint16"),
                );
            }
            output = self.emit_from_unsigned_bits(
                output,
                &destination_dtype,
                output_bit_width,
                op_name,
                "bitop_result_carrier",
            )?;
        }
        self.emit_explicit_buffer_store(&form.destination, output, source_op_id, None, None, None)
    }
}
