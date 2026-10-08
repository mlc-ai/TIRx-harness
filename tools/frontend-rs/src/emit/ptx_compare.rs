//! Validation and emission of the ptx_compare instruction family.

use crate::analyze::util::{dtype_of, ffi_error, unsupported, AResult, Failure};
use crate::decode::ptx::{unmodeled_form, DecodedPtx};
use crate::decode::Decoded;
use crate::emit::register_call::{
    generic_marker, require_register_call, table_marker, PTX_BOOL_MARKERS, PTX_COMPARE_MARKERS,
    PTX_TYPE_MARKERS,
};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

const SET_IDS: [&str; 4] = [
    "tirx.ptx.set",
    "tirx.ptx.set_bool",
    "tirx.ptx.set_half",
    "tirx.ptx.set_half_bool",
];
const BOOL_IDS: [&str; 6] = [
    "tirx.ptx.set_bool",
    "tirx.ptx.set_half_bool",
    "tirx.ptx.setp_bool",
    "tirx.ptx.setp_bool_pq",
    "tirx.ptx.setp_half_bool",
    "tirx.ptx.setp_half_bool_pq",
];
const PAIR_IDS: [&str; 4] = [
    "tirx.ptx.setp_bool_pq",
    "tirx.ptx.setp_half_bool_pq",
    "tirx.ptx.setp_half_pq",
    "tirx.ptx.setp_pq",
];
pub const PREDICATE_IDS: [&str; 9] = [
    "tirx.ptx.setp",
    "tirx.ptx.setp_half",
    "tirx.ptx.setp_bool",
    "tirx.ptx.setp_bool_pq",
    "tirx.ptx.setp_half_bool",
    "tirx.ptx.setp_half_bool_pq",
    "tirx.ptx.setp_half_pq",
    "tirx.ptx.setp_pq",
    "tirx.ptx.testp",
];

/// A missing closed-table entry: an internal error, not a rejection.
fn key_error<T>(token: &str) -> AResult<T> {
    Err(Failure::Ffi(ffi_error(&format!("KeyError: {token:?}"))))
}

fn rust_type(ptx_type: &str) -> AResult<&'static str> {
    Ok(match ptx_type {
        "b16" | "u16" | "f16" | "bf16" => "u16",
        "b32" | "u32" | "f16x2" | "bf16x2" => "u32",
        "b64" | "u64" => "u64",
        "s16" => "i16",
        "s32" => "i32",
        "s64" => "i64",
        "f32" => "f32",
        "f64" => "f64",
        _ => return key_error(ptx_type),
    })
}

fn class_marker(op: &str) -> Option<&'static str> {
    Some(match op {
        "finite" => "Finite",
        "infinite" => "Infinite",
        "number" => "Number",
        "notanumber" => "NotANumber",
        "normal" => "Normal",
        "subnormal" => "Subnormal",
        _ => return None,
    })
}

/// `TABLE[token]` outside a `try` block.
pub fn required_marker(table: &[(&str, &'static str)], token: &str) -> AResult<&'static str> {
    match table_marker(table, token) {
        Some(marker) => Ok(marker),
        None => key_error(token),
    }
}

pub struct CompareForm {
    pub destinations: Vec<ObjectRef>,
    pub sources: Vec<ObjectRef>,
    pub source_types: Vec<String>,
    pub source_bit_widths: Vec<Option<i64>>,
    pub instruction: &'static str,
    pub variant: String,
    pub output_type: String,
    pub output_bit_width: Option<i64>,
}

/// `resolve_ptx_compare`.
pub fn resolve_ptx_compare(decoded: &DecodedPtx) -> AResult<CompareForm> {
    let operands = require_register_call(decoded, false)?;
    let (destinations, sources) = (operands.destinations, operands.sources);
    let op_name = decoded.op_name.as_str();
    let subnormal = if decoded.modifier_or_empty("ftz").is_empty() {
        "PreserveSubnormal"
    } else {
        "Ftz"
    };

    if op_name == "tirx.ptx.testp" {
        let ptx_type = decoded.modifier("type")?;
        let (Some(class), Some(source_marker)) = (
            class_marker(decoded.modifier("op")?),
            table_marker(PTX_TYPE_MARKERS, ptx_type),
        ) else {
            return unmodeled_form(decoded, "has no closed classification variant");
        };
        return Ok(CompareForm {
            destinations,
            sources,
            source_types: vec![rust_type(ptx_type)?.to_owned()],
            source_bit_widths: vec![None],
            instruction: "testp",
            variant: generic_marker("Testp", &[source_marker, class]),
            output_type: "bool".to_owned(),
            output_bit_width: None,
        });
    }

    if op_name == "tirx.ptx.slct" {
        let dtype = decoded.modifier("dtype")?;
        let ctype = decoded.modifier("ctype")?;
        let Ok(width) = dtype[1..].parse::<i64>() else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "ValueError: slct dtype {dtype:?}"
            ))));
        };
        return Ok(CompareForm {
            destinations,
            sources,
            source_types: vec![
                format!("u{width}"),
                format!("u{width}"),
                rust_type(ctype)?.to_owned(),
            ],
            source_bit_widths: vec![Some(width), Some(width), None],
            instruction: "slct",
            variant: generic_marker(
                "Slct",
                &[
                    &format!("B{width}"),
                    required_marker(PTX_TYPE_MARKERS, ctype)?,
                    subnormal,
                ],
            ),
            output_type: format!("u{width}"),
            output_bit_width: Some(width),
        });
    }

    let stype = decoded.modifier_or_empty("stype");
    let source_type = if stype.is_empty() {
        decoded.modifier("type")?
    } else {
        stype
    };
    let (Some(source_marker), Some(compare_marker)) = (
        table_marker(PTX_TYPE_MARKERS, source_type),
        table_marker(PTX_COMPARE_MARKERS, decoded.modifier("cmp")?),
    ) else {
        return unmodeled_form(decoded, "has no closed type/comparison variant");
    };
    let source_width = if matches!(source_type, "b16" | "b32" | "b64") {
        Some(source_type[1..].parse::<i64>().expect("bit width"))
    } else {
        None
    };
    let source_rust = rust_type(source_type)?;

    if SET_IDS.contains(&op_name) {
        let destination_type = decoded.modifier("dtype")?;
        let destination_marker = required_marker(PTX_TYPE_MARKERS, destination_type)?;
        let (variant, source_types, bit_widths) = if BOOL_IDS.contains(&op_name) {
            let bool_marker = required_marker(PTX_BOOL_MARKERS, decoded.modifier("boolop")?)?;
            (
                generic_marker(
                    "SetBool",
                    &[
                        source_marker,
                        compare_marker,
                        destination_marker,
                        bool_marker,
                        subnormal,
                    ],
                ),
                vec![
                    source_rust.to_owned(),
                    source_rust.to_owned(),
                    "bool".to_owned(),
                ],
                vec![source_width, source_width, None],
            )
        } else {
            (
                generic_marker(
                    "Set",
                    &[source_marker, compare_marker, destination_marker, subnormal],
                ),
                vec![source_rust.to_owned(), source_rust.to_owned()],
                vec![source_width, source_width],
            )
        };
        return Ok(CompareForm {
            destinations,
            sources,
            source_types,
            source_bit_widths: bit_widths,
            instruction: "set",
            variant,
            output_type: rust_type(destination_type)?.to_owned(),
            output_bit_width: None,
        });
    }

    let bool_form = BOOL_IDS.contains(&op_name);
    let pair_form = PAIR_IDS.contains(&op_name);
    let mut source_types = vec![source_rust.to_owned(), source_rust.to_owned()];
    let mut bit_widths = vec![source_width, source_width];
    let mut bool_marker = "";
    if bool_form {
        source_types.push("bool".to_owned());
        bit_widths.push(None);
        bool_marker = required_marker(PTX_BOOL_MARKERS, decoded.modifier("boolop")?)?;
    }
    let variant = if pair_form && bool_form {
        generic_marker(
            "SetpPairBool",
            &[source_marker, compare_marker, bool_marker, subnormal],
        )
    } else if pair_form {
        generic_marker("SetpPair", &[source_marker, compare_marker, subnormal])
    } else if bool_form {
        generic_marker(
            "SetpBool",
            &[source_marker, compare_marker, bool_marker, subnormal],
        )
    } else {
        generic_marker("Setp", &[source_marker, compare_marker, subnormal])
    };
    Ok(CompareForm {
        destinations,
        sources,
        source_types,
        source_bit_widths: bit_widths,
        instruction: "setp",
        variant,
        output_type: "bool".to_owned(),
        output_bit_width: None,
    })
}

/// `resolve_ptx_compare`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_compare(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_compare(decoded, &parts, source_op_id)?;
    Ok(None)
}

/// `form.destinations[index]` (a missing destination is an internal error).
fn destination(form: &CompareForm, index: usize) -> AResult<&ObjectRef> {
    match form.destinations.get(index) {
        Some(destination) => Ok(destination),
        None => Err(Failure::Ffi(ffi_error(
            "IndexError: tuple index out of range",
        ))),
    }
}

impl<'a> Emitter<'a> {
    fn emit_compare_source(
        &mut self,
        op_name: &str,
        form: &CompareForm,
        index: usize,
    ) -> AResult<RustValue> {
        let expression = &form.sources[index];
        if let Some(width) = form.source_bit_widths[index] {
            return self.emit_as_unsigned_bits(
                expression,
                width,
                op_name,
                &format!("compare_source_{index}_bits"),
                None,
            );
        }
        let expected = form.source_types[index].as_str();
        let emitted = self.emit_expr(expression)?;
        let emitted = self.observe_pointer_bits(emitted);
        let emitted = self.coerce_value(emitted, expected, &format!("compare_source_{index}"))?;
        let emitted = self.as_warp_value(emitted);
        if emitted.rust_type != expected {
            return unsupported(format!(
                "{op_name} source {index} lowered to {}, expected {expected}",
                emitted.rust_type
            ));
        }
        Ok(emitted)
    }

    fn compare_predicate_u32(&mut self, raw: &str, prefix: &str) -> RustValue {
        let predicate = self.control_name(&format!("{prefix}_predicate"));
        let bits = self.control_name(&format!("{prefix}_u32"));
        self.emit_line(&format!("let {predicate} = v2_register_out({raw});"));
        self.emit_line(&format!(
            "let {bits} = WarpValue::from_fn(|lane| u32::from({predicate}[lane]));"
        ));
        RustValue::new(bits, "u32", Uniformity::Varying)
    }

    fn store_compare_raw_bits(
        &mut self,
        op_name: &str,
        form: &CompareForm,
        raw: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let result = self.control_name(&format!("decoded_{}_bits", form.instruction));
        self.emit_line(&format!("let {result} = v2_register_out({raw});"));
        let value = RustValue::new(result, form.output_type.clone(), Uniformity::Varying);
        let destination = destination(form, 0)?;
        let destination_dtype = dtype_of(destination)?;
        if form.output_bit_width == Some(16)
            && matches!(destination_dtype.as_str(), "float16" | "bfloat16")
        {
            return self.emit_explicit_buffer_store(
                destination,
                value,
                source_op_id,
                None,
                None,
                Some("uint16"),
            );
        }
        let value = self.emit_from_unsigned_bits(
            value,
            &destination_dtype,
            form.output_bit_width.expect("raw bit width"),
            op_name,
            "compare_result_carrier",
        )?;
        self.emit_explicit_buffer_store(destination, value, source_op_id, None, None, None)
    }

    /// `emit_ptx_compare`.
    pub fn emit_ptx_compare(
        &mut self,
        decoded: &DecodedPtx,
        form: &CompareForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "ptx_compare",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        self.emit_ptx_compare_body(decoded, form, source_op_id)?;
        let mask = region.mask.clone();
        self.close_predicated_region(region);
        let predicate_output = PREDICATE_IDS.contains(&decoded.op_name.as_str());
        for destination in &form.destinations {
            let result_dtype = dtype_of(destination)?;
            self.finish_predicated_destinations(
                decoded,
                &[Some(destination.clone())],
                &result_dtype,
                &mask,
                source_op_id,
                predicate_output,
            )?;
        }
        Ok(())
    }

    fn emit_ptx_compare_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &CompareForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let mut sources = Vec::new();
        for index in 0..form.sources.len() {
            sources.push(self.emit_compare_source(op_name, &form, index)?);
        }
        let arguments = if sources.len() == 1 {
            abi::register(&sources[0].code)
        } else {
            format!(
                "({})",
                sources
                    .iter()
                    .map(|source| abi::register(&source.code))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let raw = self.control_name(&format!("decoded_{}_raw", form.instruction));
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call(
            &format!("reg::{}", form.instruction),
            &site,
            &[arguments],
            Some(&form.variant),
        );
        self.emit_line(&format!("let {raw} = {call};"));

        if form.output_bit_width.is_some() {
            return self.store_compare_raw_bits(op_name, &form, &raw, source_op_id);
        }

        if PREDICATE_IDS.contains(&op_name) {
            if form.destinations.len() == 1 {
                let result = self.compare_predicate_u32(&raw, "compare_result");
                return self.emit_explicit_buffer_store(
                    &form.destinations[0],
                    result,
                    source_op_id,
                    None,
                    None,
                    None,
                );
            }
            let first_raw = self.control_name("compare_first_raw");
            let second_raw = self.control_name("compare_second_raw");
            self.emit_line(&format!("let ({first_raw}, {second_raw}) = {raw};"));
            let first = self.compare_predicate_u32(&first_raw, "compare_first");
            let second = self.compare_predicate_u32(&second_raw, "compare_second");
            self.emit_explicit_buffer_store(
                destination(&form, 0)?,
                first,
                source_op_id,
                None,
                None,
                None,
            )?;
            return self.emit_explicit_buffer_store(
                destination(&form, 1)?,
                second,
                source_op_id,
                None,
                None,
                None,
            );
        }

        let result = self.control_name(&format!("decoded_{}", form.instruction));
        self.emit_line(&format!("let {result} = v2_register_out({raw});"));
        self.emit_explicit_buffer_store(
            destination(&form, 0)?,
            RustValue::new(result, form.output_type.clone(), Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }
}
