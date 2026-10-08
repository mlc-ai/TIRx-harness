//! Validation and emission of the ptx_move instruction family.

use crate::analyze::util::{dtype_of, ffi_error, unsupported, AResult, Failure};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{marker, require_register_call};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

/// Existing ABI variants, keyed by (vector element width, vector length).
fn vector_variants(width: i64, length: usize) -> Option<(&'static str, &'static str)> {
    Some(match (width, length) {
        (16, 2) => ("B32", "B32"),
        (32, 2) => ("B64", "B64"),
        (16, 4) => ("B16x4", "B16x4"),
        (64, 2) => ("B128", "B64x2"),
        (32, 4) => ("B32x4", "B32x4"),
        _ => return None,
    })
}

pub struct MoveForm {
    /// `call.operand("d")`, sinks included.
    pub destinations: Vec<Option<ObjectRef>>,
    pub sources: Vec<ObjectRef>,
    pub source_width: i64,
    pub destination_width: i64,
    pub function: &'static str,
    pub specialization: String,
}

/// The bit width a PTX register type names; `.pred` has none.
fn register_width(ptx_type: &str) -> i64 {
    if ptx_type == "pred" {
        0
    } else {
        ptx_type[1..].parse().expect("register type width")
    }
}

/// `resolve_ptx_move`.
pub fn resolve_ptx_move(decoded: &DecodedPtx) -> AResult<MoveForm> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, true)?;
    let destinations: Vec<Option<ObjectRef>> = decoded.operand("d")?.to_vec();
    if operands.destinations.is_empty() {
        return unsupported(format!(
            "{op_name} requires at least one concrete destination"
        ));
    }
    let [destination_slot, source_slot] = decoded.operands.as_slice() else {
        return Err(Failure::Ffi(ffi_error(&format!(
            "{op_name} move entries have a destination and a source operand"
        ))));
    };
    let destination_width = register_width(&destination_slot.operand_type);
    let source_width = register_width(&source_slot.operand_type);
    let (function, specialization) = if destinations.len() == 1 && operands.sources.len() == 1 {
        let specialization = if source_width == 0 {
            "Pred".to_owned()
        } else {
            format!("U{source_width}")
        };
        ("reg::mov", specialization)
    } else {
        let unpack = destinations.len() > 1;
        let (vector_width, vector_length) = if unpack {
            (destination_width, destinations.len())
        } else {
            (source_width, operands.sources.len())
        };
        let Some((pack, unpacked)) = vector_variants(vector_width, vector_length) else {
            return unsupported(format!("{op_name} has no reviewed vector shape"));
        };
        if unpack {
            ("reg::mov_unpack", unpacked.to_owned())
        } else {
            ("reg::mov_pack", pack.to_owned())
        }
    };
    Ok(MoveForm {
        destinations,
        sources: operands.sources,
        source_width,
        destination_width,
        function,
        specialization: marker(&specialization),
    })
}

/// The first non-sink destination lane.
pub fn first_concrete_destination(form: &MoveForm) -> &ObjectRef {
    form.destinations
        .iter()
        .flatten()
        .next()
        .expect("validated concrete destination")
}

/// The validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_move(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_move(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_move`.
    pub fn emit_ptx_move(
        &mut self,
        decoded: &DecodedPtx,
        form: &MoveForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "mov",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        let body = self.emit_move_body(decoded, form, source_op_id);
        self.close_predicated_region(region);
        body?;
        let destination_dtype = dtype_of(first_concrete_destination(form))?;
        self.finish_predicated_destinations(
            decoded,
            &form.destinations,
            &destination_dtype,
            &mask,
            source_op_id,
            form.destination_width == 0,
        )
    }

    fn emit_move_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &MoveForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        // Materialize every input before any destination is written. This also
        // preserves aliasing pack/unpack operands and the original predicate mask.
        let mut arguments = Vec::new();
        for source in &form.sources {
            let value = if form.source_width != 0 {
                self.emit_as_unsigned_bits(source, form.source_width, op_name, "mov_source", None)?
            } else {
                let value = self.emit_expr(source)?;
                let value = self.coerce_value(value, "bool", "mov_predicate")?;
                self.as_warp_value(value)
            };
            arguments.push(abi::register(&value.code));
        }
        let raw = self.control_name("mov_raw");
        let argument = if arguments.len() == 1 {
            arguments.remove(0)
        } else {
            format!("({})", arguments.join(", "))
        };
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call(
            form.function,
            &site,
            &[argument],
            Some(&form.specialization),
        );
        self.emit_line(&format!("let {raw} = {call};"));
        let raw_lanes: Vec<String> = if form.destinations.len() > 1 {
            let lanes: Vec<String> = form
                .destinations
                .iter()
                .map(|_| self.control_name("mov_lane"))
                .collect();
            self.emit_line(&format!("let ({}) = {raw};", lanes.join(", ")));
            lanes
        } else {
            vec![raw]
        };
        let width = form.destination_width;
        for (destination, lane) in form.destinations.iter().zip(&raw_lanes) {
            let Some(destination) = destination else {
                continue;
            };
            let result = self.control_name("mov_value");
            self.emit_line(&format!("let {result} = v2_register_out({lane});"));
            let rust_type = match width {
                128 => "U64x2".to_owned(),
                0 => "bool".to_owned(),
                _ => format!("u{width}"),
            };
            let bits = RustValue::new(result, rust_type, Uniformity::Varying);
            let dtype = dtype_of(destination)?;
            let raw_half = width == 16 && (dtype == "float16" || dtype == "bfloat16");
            let value = if raw_half {
                bits
            } else if width != 0 {
                self.emit_from_unsigned_bits(bits, &dtype, width, op_name, "mov_carrier")?
            } else {
                self.coerce_dtype(bits, &dtype, "mov_predicate_carrier")?
            };
            self.emit_explicit_buffer_store(
                destination,
                value,
                source_op_id,
                None,
                None,
                raw_half.then_some("uint16"),
            )?;
        }
        Ok(())
    }
}
