//! Validation and emission of the ptx_set_packed instruction family.

use crate::analyze::util::{dtype_of, unmodeled, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, table_marker, PTX_COMPARE_MARKERS};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

pub const ABI_FUNCTION: &str = "reg::set";
const UNSIGNED_ALIASES: [&str; 4] = ["lo", "ls", "hi", "hs"];

fn packed_type_marker(ptx_type: &str) -> Option<&'static str> {
    Some(match ptx_type {
        "u8x4" => "U8",
        "s8x4" => "I8",
        "u16x2" => "U16",
        "s16x2" => "I16",
        _ => return None,
    })
}

pub struct SetPackedForm {
    pub destination: ObjectRef,
    pub sources: Vec<ObjectRef>,
    pub variant: String,
}

/// `resolve_ptx_set_packed`.
pub fn resolve_ptx_set_packed(decoded: &DecodedPtx) -> AResult<SetPackedForm> {
    let operands = require_register_call(decoded, false)?;
    let op_name = decoded.op_name.as_str();
    let ptx_type = decoded.modifier("type")?;
    let comparison = decoded.modifier("cmp")?;
    let (Some(type_marker), Some(compare_marker)) = (
        packed_type_marker(ptx_type),
        table_marker(PTX_COMPARE_MARKERS, comparison),
    ) else {
        return unmodeled(
            format!("call:{op_name}"),
            format!("{op_name}.{comparison}.{ptx_type} has no reviewed NumSim variant"),
        );
    };
    if ptx_type.starts_with('s') && UNSIGNED_ALIASES.contains(&comparison) {
        return unmodeled(
            format!("call:{op_name}"),
            format!(
                "{op_name}.{comparison}.{ptx_type} uses an unsigned comparison alias with a signed packed type"
            ),
        );
    }
    if operands.destinations.len() != 1 || operands.sources.len() != 2 {
        return unsupported(format!(
            "{op_name} requires one destination and two sources"
        ));
    }
    Ok(SetPackedForm {
        destination: operands.destinations[0].clone(),
        sources: operands.sources,
        variant: format!(
            "v2::reg::variant::SetPacked<v2::reg::variant::{type_marker}, v2::reg::variant::{compare_marker}>"
        ),
    })
}

/// `resolve_ptx_set_packed`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_set_packed(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_set_packed(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_set_packed`.
    pub fn emit_ptx_set_packed(
        &mut self,
        decoded: &DecodedPtx,
        form: &SetPackedForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "set_packed",
            "packed set predicate must be bool or integer",
        )?;
        let mask = region.mask.clone();
        let body = self.emit_ptx_set_packed_body(decoded, form, source_op_id);
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

    fn emit_ptx_set_packed_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &SetPackedForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let mut sources = Vec::new();
        for (index, expression) in form.sources.iter().enumerate() {
            sources.push(self.emit_as_unsigned_bits(
                expression,
                32,
                op_name,
                &format!("set_packed_source_{index}"),
                None,
            )?);
        }
        let raw = self.control_name("set_packed_raw");
        let bits = self.control_name("set_packed_bits");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call(
            ABI_FUNCTION,
            &site,
            &[format!(
                "({}, {})",
                abi::register(&sources[0].code),
                abi::register(&sources[1].code)
            )],
            Some(&form.variant),
        );
        self.emit_line(&format!("let {raw} = {call};"));
        self.emit_line(&format!("let {bits} = v2_register_out({raw});"));
        let result = self.emit_from_unsigned_bits(
            RustValue::new(bits, "u32", Uniformity::Varying),
            &dtype_of(&form.destination)?,
            32,
            op_name,
            "set_packed_result",
        )?;
        self.emit_explicit_buffer_store(&form.destination, result, source_op_id, None, None, None)
    }
}
