//! Validation and emission of the ptx_unary instruction family.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::{unmodeled_form, DecodedPtx};
use crate::decode::Decoded;
use crate::emit::register_call::require_register_call;
use crate::emit::{Emitter, RustValue, Uniformity};
use crate::tables::round_marker;
use tvm::tvm_ffi::object::ObjectRef;

/// The resolved unary form (`resolve_ptx_unary` tuple).
pub struct UnaryForm {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub instruction: &'static str,
    pub variant: String,
    pub rust_type: &'static str,
}

/// `call.op_name.rsplit('.', 1)[-1]`.
fn short_name(op_name: &str) -> &str {
    op_name.rsplit('.').next().unwrap_or(op_name)
}

fn dotted(token: &str) -> String {
    if token.is_empty() {
        String::new()
    } else {
        format!(".{token}")
    }
}

/// `resolve_ptx_unary`.
pub fn resolve_ptx_unary(decoded: &DecodedPtx) -> AResult<UnaryForm> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if operands.destinations.len() != 1 || operands.sources.len() != 1 {
        return unsupported(format!("{op_name} requires one destination and one source"));
    }
    let destination = operands.destinations[0].clone();
    let source = operands.sources[0].clone();

    let instruction: &'static str;
    let variant: String;
    let expected_dtype: &str;
    let rust_type: &'static str;
    if op_name == "tirx.ptx.neg_half" || op_name == "tirx.ptx.abs_half" {
        let ptx_type = decoded.modifier("type")?;
        let ftz = decoded.modifier("ftz")?;
        if !matches!(ptx_type, "f16" | "f16x2" | "bf16" | "bf16x2") {
            return unmodeled_form(
                decoded,
                &format!("{ptx_type} has no half-precision NumSim variant"),
            );
        }
        if !ftz.is_empty() && !matches!(ptx_type, "f16" | "f16x2") {
            return unmodeled_form(
                decoded,
                &format!("{ftz}.{ptx_type} is not a legal closed variant"),
            );
        }
        instruction = if op_name == "tirx.ptx.neg_half" {
            "neg"
        } else {
            "abs"
        };
        let marker = match ptx_type {
            "f16" => "F16",
            "f16x2" => "F16x2",
            "bf16" => "Bf16",
            _ => "Bf16x2",
        };
        variant = format!(
            "v2::reg::variant::{marker}{}",
            if ftz.is_empty() { "" } else { "Ftz" }
        );
        let packed = ptx_type.ends_with("x2");
        expected_dtype = if packed { "uint32" } else { "uint16" };
        rust_type = if packed { "u32" } else { "u16" };
    } else if op_name == "tirx.ptx.neg" || op_name == "tirx.ptx.abs_f" {
        let ptx_type = decoded.modifier("type")?;
        let ftz = decoded.modifier_or_empty("ftz");
        if !(ptx_type == "f32" || ptx_type == "f64") || (!ftz.is_empty() && ptx_type != "f32") {
            return unmodeled_form(
                decoded,
                &format!(
                    "{}{}.{ptx_type} has no exact NumSim engine variant",
                    short_name(op_name),
                    dotted(ftz)
                ),
            );
        }
        instruction = if op_name == "tirx.ptx.neg" {
            "neg"
        } else {
            "abs"
        };
        let marker = if ptx_type == "f64" {
            "F64"
        } else if ftz.is_empty() {
            "F32"
        } else {
            "F32Ftz"
        };
        variant = format!("v2::reg::variant::{marker}");
        expected_dtype = if ptx_type == "f32" {
            "float32"
        } else {
            "float64"
        };
        rust_type = if ptx_type == "f32" { "f32" } else { "f64" };
    } else if op_name == "tirx.ptx.ex2_half" {
        let mode = decoded.modifier("mode")?;
        let ftz = decoded.modifier("ftz")?;
        let ptx_type = decoded.modifier("type")?;
        if mode != "approx"
            || !matches!(ptx_type, "f16" | "f16x2" | "bf16" | "bf16x2")
            || (!ftz.is_empty() != ptx_type.starts_with("bf16"))
        {
            return unmodeled_form(
                decoded,
                &format!(
                    "ex2.{mode}{}.{ptx_type} has no NumSim representative",
                    dotted(ftz)
                ),
            );
        }
        instruction = "exp2";
        let marker = match ptx_type {
            "f16" => "F16",
            "f16x2" => "F16x2",
            "bf16" => "Bf16",
            _ => "Bf16x2",
        };
        variant = format!("v2::reg::variant::{marker}");
        let packed = ptx_type.ends_with("x2");
        expected_dtype = if packed { "uint32" } else { "uint16" };
        rust_type = if packed { "u32" } else { "u16" };
    } else if op_name == "tirx.ptx.cos" || op_name == "tirx.ptx.sin" {
        let mode = decoded.modifier("mode")?;
        let ftz = decoded.modifier("ftz")?;
        let ptx_type = decoded.modifier("type")?;
        if mode != "approx" || ptx_type != "f32" {
            return unmodeled_form(
                decoded,
                &format!(
                    "{}.{mode}.{ptx_type} has no NumSim representative",
                    short_name(op_name)
                ),
            );
        }
        instruction = if op_name == "tirx.ptx.cos" {
            "cos"
        } else {
            "sin"
        };
        variant = format!(
            "v2::reg::variant::{}",
            if ftz.is_empty() { "F32" } else { "F32Ftz" }
        );
        expected_dtype = "float32";
        rust_type = "f32";
    } else if op_name == "tirx.ptx.sqrt" {
        let mode = decoded.modifier("mode")?;
        let ftz = decoded.modifier("ftz")?;
        let ptx_type = decoded.modifier("type")?;
        if mode == "approx" {
            if ptx_type != "f32" {
                return unmodeled_form(
                    decoded,
                    &format!("sqrt.approx.{ptx_type} has no legal NumSim variant"),
                );
            }
            variant = format!(
                "v2::reg::variant::{}",
                if ftz.is_empty() { "F32" } else { "F32Ftz" }
            );
        } else {
            let Some(round_marker) =
                round_marker(mode).filter(|_| ptx_type == "f32" || ptx_type == "f64")
            else {
                return unmodeled_form(
                    decoded,
                    &format!("sqrt.{mode}.{ptx_type} has no closed NumSim variant"),
                );
            };
            if ptx_type == "f64" && !ftz.is_empty() {
                return unmodeled_form(decoded, "sqrt.rnd.ftz.f64 is not a legal PTX form");
            }
            let subnormal = if ftz.is_empty() {
                "PreserveSubnormal"
            } else {
                "Ftz"
            };
            let type_marker = if ptx_type == "f32" { "F32" } else { "F64" };
            variant = format!(
                "v2::reg::variant::Sqrt<v2::reg::variant::{type_marker}, v2::reg::variant::{round_marker}, v2::reg::variant::{subnormal}>"
            );
        }
        instruction = "sqrt";
        expected_dtype = if ptx_type == "f32" {
            "float32"
        } else {
            "float64"
        };
        rust_type = if ptx_type == "f32" { "f32" } else { "f64" };
    } else if op_name == "tirx.ptx.rcp" {
        let mode = decoded.modifier("mode")?;
        let ftz = decoded.modifier("ftz")?;
        let ptx_type = decoded.modifier("type")?;
        if mode == "approx" && ptx_type == "f32" {
            variant = format!(
                "v2::reg::variant::{}",
                if ftz.is_empty() { "F32" } else { "F32RnFtz" }
            );
        } else if (ptx_type == "f32" || ptx_type == "f64")
            && matches!(mode, "rn" | "rz" | "rm" | "rp" | "approx")
        {
            let marker = round_marker(mode).unwrap_or("Approx");
            let mut arithmetic = format!(
                "v2::reg::variant::{}Arithmetic<v2::reg::variant::{marker}",
                if ptx_type == "f32" { "F32" } else { "F64" }
            );
            if ptx_type == "f32" {
                arithmetic.push_str(&format!(
                    ", v2::reg::variant::{}",
                    if ftz.is_empty() {
                        "PreserveSubnormal"
                    } else {
                        "Ftz"
                    }
                ));
            }
            arithmetic.push('>');
            variant = arithmetic;
        } else {
            return unmodeled_form(
                decoded,
                &format!("rcp.{mode}.{ptx_type} has no legal NumSim variant"),
            );
        }
        instruction = "rcp";
        expected_dtype = if ptx_type == "f32" {
            "float32"
        } else {
            "float64"
        };
        rust_type = if ptx_type == "f32" { "f32" } else { "f64" };
    } else if op_name == "tirx.ptx.tanh" || op_name == "tirx.ptx.tanh_half" {
        let mode = decoded.modifier("mode")?;
        let ptx_type = decoded.modifier("type")?;
        if mode != "approx" {
            return unmodeled_form(
                decoded,
                &format!("tanh.{mode}.{ptx_type} has no NumSim representative"),
            );
        }
        instruction = "tanh";
        if op_name == "tirx.ptx.tanh" && ptx_type == "f32" {
            variant = "v2::reg::variant::F32".to_owned();
            expected_dtype = "float32";
            rust_type = "f32";
        } else if op_name == "tirx.ptx.tanh_half" && (ptx_type == "f16" || ptx_type == "bf16") {
            variant = format!(
                "v2::reg::variant::{}",
                if ptx_type == "f16" { "F16" } else { "Bf16" }
            );
            expected_dtype = "uint16";
            rust_type = "u16";
        } else if op_name == "tirx.ptx.tanh_half" && (ptx_type == "f16x2" || ptx_type == "bf16x2") {
            variant = format!(
                "v2::reg::variant::{}",
                if ptx_type == "f16x2" {
                    "F16x2"
                } else {
                    "Bf16x2"
                }
            );
            expected_dtype = "uint32";
            rust_type = "u32";
        } else {
            return unmodeled_form(
                decoded,
                &format!("tanh.{mode}.{ptx_type} has no NumSim representative"),
            );
        }
    } else {
        let mode = decoded.modifier("mode")?;
        let ftz = decoded.modifier("ftz")?;
        let ptx_type = decoded.modifier("type")?;
        let approximate_ftz = mode == "approx" && ftz == "ftz" && ptx_type == "f32";
        let approximate_f32 = mode == "approx" && ftz.is_empty() && ptx_type == "f32";
        let approximate_f64 = op_name == "tirx.ptx.rsqrt" && mode == "approx" && ptx_type == "f64";
        if !(approximate_ftz || approximate_f32 || approximate_f64) {
            return unmodeled_form(
                decoded,
                &format!(
                    "{}.{mode}{}.{ptx_type} has no exact NumSim engine variant",
                    short_name(op_name),
                    dotted(ftz)
                ),
            );
        }
        instruction = match op_name {
            "tirx.ptx.ex2" => "exp2",
            "tirx.ptx.lg2" => "lg2",
            "tirx.ptx.rsqrt" => "rsqrt",
            _ => unreachable!("unary family op name"),
        };
        variant = if approximate_f64 {
            if ftz.is_empty() {
                "v2::reg::variant::F64".to_owned()
            } else {
                "v2::reg::variant::F64Arithmetic<v2::reg::variant::Approx>".to_owned()
            }
        } else if approximate_f32 {
            "v2::reg::variant::F32".to_owned()
        } else {
            "v2::reg::variant::F32RnFtz".to_owned()
        };
        expected_dtype = if approximate_f64 {
            "float64"
        } else {
            "float32"
        };
        rust_type = if ptx_type == "f64" { "f64" } else { "f32" };
    }

    let destination_dtype = dtype_of(&destination)?;
    let source_dtype = dtype_of(&source)?;
    if destination_dtype != expected_dtype || source_dtype != expected_dtype {
        return unsupported(format!(
            "{op_name} expects {expected_dtype} -> {expected_dtype}, got {source_dtype} -> {destination_dtype}"
        ));
    }
    Ok(UnaryForm {
        destination,
        source,
        instruction,
        variant,
        rust_type,
    })
}

/// The validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_unary(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_unary(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_unary`.
    pub fn emit_ptx_unary(
        &mut self,
        decoded: &DecodedPtx,
        form: &UnaryForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "raw_unary",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let source = self.emit_expr(&form.source)?;
        let source = self.as_warp_value(source);
        if source.rust_type != form.rust_type {
            return unsupported(format!(
                "{} operand lowered to {}, expected {}",
                decoded.op_name, source.rust_type, form.rust_type
            ));
        }
        let result = self.emit_register_result(
            &format!("reg::{}", form.instruction),
            &form.variant,
            &[source],
            source_op_id,
            None,
        );
        self.emit_explicit_buffer_store(
            &form.destination,
            RustValue::new(result, form.rust_type, Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )?;
        let mask = region.mask.clone();
        self.close_predicated_region(region);
        let result_dtype = dtype_of(&form.destination)?;
        self.finish_predicated_destinations(
            decoded,
            &[Some(form.destination.clone())],
            &result_dtype,
            &mask,
            source_op_id,
            false,
        )
    }
}
