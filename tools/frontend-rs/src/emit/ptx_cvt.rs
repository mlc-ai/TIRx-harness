//! Validation and emission of the ptx_cvt instruction family.

use crate::analyze::util::{dtype_of, ffi_error, unmodeled, unsupported, AResult, Failure};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::ptx_cvt_variants::{
    lookup_packed_cvt_variant, packed_mode_marker, scalar_cvt_flush_is_inert,
    scalar_cvt_rounding_is_inert, scalar_uses_packed_mode, Modifiers,
};
use crate::emit::register_call::{marker, require_register_call, table_marker, PTX_TYPE_MARKERS};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use crate::tables::{dtype_by_rust_type, dtype_itemsize, expr_rust_type_by_dtype};
use tvm::tvm_ffi::object::ObjectRef;

/// Canonical numeric carriers of the existing Rust conversion ABI. Legal
/// public carriers are validated by the target table, not by this mapping.
fn scalar_carrier(name: &str) -> Option<&'static str> {
    Some(match name {
        "u8" => "uint8",
        "s8" => "int8",
        "u16" => "uint16",
        "s16" => "int16",
        "u32" => "uint32",
        "s32" => "int32",
        "u64" => "uint64",
        "s64" => "int64",
        "f16" => "uint16",
        "bf16" => "uint16",
        "f32" => "float32",
        "f64" => "float64",
        "tf32" => "uint32",
        _ => return None,
    })
}

fn rounding_marker(rounding: &str) -> &'static str {
    match rounding {
        "" => "Unmodified",
        "rn" => "Rn",
        "rna" => "Rna",
        "rz" => "Rz",
        "rm" => "Rm",
        "rp" => "Rp",
        "rni" => "Rni",
        "rzi" => "Rzi",
        "rmi" => "Rmi",
        "rpi" => "Rpi",
        other => unreachable!("cvt rounding marker for {other}"),
    }
}

pub struct CvtForm {
    pub destination: ObjectRef,
    pub sources: Vec<ObjectRef>,
    pub source_rust_types: Vec<&'static str>,
    pub result_type: &'static str,
    pub variant: String,
}

fn unmodeled_form<T>(decoded: &DecodedPtx, message: String) -> AResult<T> {
    unmodeled(format!("call:{}", decoded.op_name), message)
}

fn scalar_variant(m: &Modifiers, packed_mode: bool) -> String {
    let type_marker = |name: &str| {
        marker(table_marker(PTX_TYPE_MARKERS, name).expect("scalar cvt carrier marker"))
    };
    let source_marker = type_marker(m.source);
    let destination_marker = type_marker(m.destination);
    let mode = if packed_mode {
        packed_mode_marker(m.rounding, m.satfinite, m.relu, m.pzo, "")
    } else if m.sat && m.rounding.is_empty() && !m.ftz {
        format!(
            "v2::reg::variant::CvtMode<{}, {}, {}>",
            marker("Unmodified"),
            marker("PreserveSubnormal"),
            marker("Sat")
        )
    } else if m.rounding.is_empty() && !m.ftz {
        if scalar_cvt_rounding_is_inert(m.destination, m.source, m.sat) {
            format!(
                "v2::reg::variant::CvtMode<{}, {}>",
                marker("Exact"),
                marker("PreserveSubnormal")
            )
        } else {
            marker("Unmodified")
        }
    } else {
        let mut round_marker = marker(rounding_marker(m.rounding));
        if scalar_cvt_rounding_is_inert(m.destination, m.source, m.sat) {
            round_marker = marker("Exact");
        }
        let flushes = m.ftz && !scalar_cvt_flush_is_inert(m.destination, m.source);
        let subnormal = marker(if flushes { "Ftz" } else { "PreserveSubnormal" });
        // Float-to-integer SAT is already inherent in that conversion; only
        // floating destinations need the explicit [0, 1] clamp axis here.
        let clamp = if m.sat && matches!(m.destination, "f16" | "f32" | "f64") {
            format!(", {}", marker("Sat"))
        } else {
            String::new()
        };
        format!("v2::reg::variant::CvtMode<{round_marker}, {subnormal}{clamp}>")
    };
    format!("v2::reg::variant::Cvt<{source_marker}, {destination_marker}, {mode}>")
}

/// `resolve_ptx_cvt`.
pub fn resolve_ptx_cvt(decoded: &DecodedPtx) -> AResult<CvtForm> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if operands.destinations.len() != 1 {
        return unsupported(format!("{op_name} requires exactly one destination"));
    }
    let destination_expr = operands.destinations[0].clone();

    let scaled_token = decoded.modifier_or_empty("scaled");
    let m = Modifiers {
        destination: decoded.modifier_or_empty("dtype"),
        source: decoded.modifier_or_empty("atype"),
        rounding: decoded.modifier_or_empty("rnd"),
        ftz: decoded.modifier_or_empty("ftz") == "ftz",
        sat: decoded.modifier_or_empty("sat") == "sat",
        relu: decoded.modifier_or_empty("relu") == "relu",
        satfinite: decoded.modifier_or_empty("satfinite") == "satfinite",
        pzo: decoded.modifier_or_empty("pzo") == "pzo",
        scaled: scaled_token
            .strip_prefix("scaled::")
            .unwrap_or(scaled_token),
    };
    let source_expressions = operands.sources;

    if let Some(packed_variant) = lookup_packed_cvt_variant(&m) {
        let target_types: Vec<&'static str> = packed_variant
            .operand_dtypes
            .iter()
            .map(|dtype| expr_rust_type_by_dtype(dtype).expect("cvt carrier rust type"))
            .collect();
        return Ok(CvtForm {
            destination: destination_expr,
            sources: source_expressions,
            source_rust_types: target_types,
            result_type: expr_rust_type_by_dtype(packed_variant.result_dtype)
                .expect("cvt carrier rust type"),
            variant: packed_variant.specialization,
        });
    }
    if source_expressions.len() != 1 {
        return unmodeled_form(
            decoded,
            "NumSim currently models one-primary scalar PTX cvt spellings and the reviewed \
             packed conversion spellings"
                .to_owned(),
        );
    }
    let (Some(expected_destination), Some(expected_source)) =
        (scalar_carrier(m.destination), scalar_carrier(m.source))
    else {
        return unmodeled_form(
            decoded,
            format!(
                "NumSim has not reviewed modified packed PTX cvt {}.{}",
                m.destination, m.source
            ),
        );
    };
    let packed_mode = scalar_uses_packed_mode(&m);
    let rust_source = expr_rust_type_by_dtype(expected_source).expect("cvt carrier rust type");
    Ok(CvtForm {
        destination: destination_expr,
        sources: source_expressions,
        source_rust_types: vec![rust_source],
        result_type: expr_rust_type_by_dtype(expected_destination).expect("cvt carrier rust type"),
        variant: scalar_variant(&m, packed_mode),
    })
}

/// The validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_cvt(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_cvt(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_cvt`.
    pub fn emit_ptx_cvt(
        &mut self,
        decoded: &DecodedPtx,
        form: &CvtForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "cvt",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        let body = self.emit_cvt_body(decoded, form, source_op_id);
        self.close_predicated_region(region);
        body?;
        let destination_dtype = dtype_of(&form.destination)?;
        self.finish_predicated_destinations(
            decoded,
            &[Some(form.destination.clone())],
            &destination_dtype,
            &mask,
            source_op_id,
            false,
        )
    }

    fn emit_cvt_source(
        &mut self,
        expression: &ObjectRef,
        rust_type: &str,
        op_name: &str,
    ) -> AResult<RustValue> {
        let canonical = dtype_by_rust_type(rust_type).expect("cvt carrier dtype");
        let dtype = dtype_of(expression)?;
        if dtype == canonical {
            let value = self.emit_expr(expression)?;
            let value = self.observe_pointer_bits(value);
            return Ok(self.as_warp_value(value));
        }
        let Some(itemsize) = dtype_itemsize(self.ctx.schema, &dtype) else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "{op_name} source {dtype} has no itemsize"
            ))));
        };
        let mut value =
            self.emit_as_unsigned_bits(expression, itemsize * 8, op_name, "cvt_source_bits", None)?;
        if value.rust_type == "U64x2" {
            let low = self.control_name("cvt_source_low_half");
            self.emit_line(&format!(
                "let {low} = WarpValue::from_fn(|lane| {}[lane][0]);",
                value.code
            ));
            value = RustValue::new(low, "u64", Uniformity::Varying);
        }
        let width: i64 = rust_type[1..].parse().expect("carrier width");
        let value = self.coerce_value(value, &format!("u{width}"), "cvt_source_low_bits")?;
        self.emit_from_unsigned_bits(value, canonical, width, op_name, "cvt_source_value")
    }

    fn emit_cvt_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &CvtForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let mut sources = Vec::new();
        for (expression, rust_type) in form.sources.iter().zip(&form.source_rust_types) {
            sources.push(self.emit_cvt_source(expression, rust_type, &decoded.op_name)?);
        }
        let actual_rust_types: Vec<&str> = sources
            .iter()
            .map(|source| source.rust_type.as_str())
            .collect();
        if actual_rust_types != form.source_rust_types {
            let render = |types: &[&str]| format!("{types:?}");
            return unsupported(format!(
                "{} operands lowered to {}, expected carriers {}",
                decoded.op_name,
                render(&actual_rust_types),
                render(&form.source_rust_types)
            ));
        }
        let registers: Vec<String> = sources
            .iter()
            .map(|source| abi::register(&source.code))
            .collect();
        let arguments = if registers.len() == 1 {
            registers[0].clone()
        } else {
            format!("({})", registers.join(", "))
        };
        let raw_result = self.control_name("decoded_cvt_raw");
        let result = self.control_name("decoded_cvt");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call("reg::cvt", &site, &[arguments], Some(&form.variant));
        self.emit_line(&format!("let {raw_result} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({raw_result});"));
        let output = RustValue::new(result, form.result_type, Uniformity::Varying);
        let width: i64 = form.result_type[1..].parse().expect("result width");
        let (output, storage_dtype) = self.emit_extended_register_value(
            output,
            &dtype_of(&form.destination)?,
            width,
            form.result_type.starts_with('i'),
            &decoded.op_name,
            "cvt_result",
        )?;
        self.emit_explicit_buffer_store(
            &form.destination,
            output,
            source_op_id,
            None,
            None,
            storage_dtype,
        )
    }
}
