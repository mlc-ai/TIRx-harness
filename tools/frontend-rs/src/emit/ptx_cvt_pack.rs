//! Validation and emission of the ptx_cvt_pack instruction family.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::require_register_call;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

/// The engine function (below `v2::`) the lowering calls.
pub const CVT_PACK: &str = "reg::cvt_pack";
const CONVERT_TYPES: [&str; 8] = ["u2", "s2", "u4", "s4", "u8", "s8", "u16", "s16"];

/// The tuple `resolve_ptx_cvt_pack` returns.
pub struct CvtPackForm {
    pub destination: ObjectRef,
    pub a: ObjectRef,
    pub b: ObjectRef,
    pub c: Option<ObjectRef>,
    pub bits: i64,
    pub signed: bool,
}

/// `resolve_ptx_cvt_pack`.
pub fn resolve_ptx_cvt_pack(decoded: &DecodedPtx) -> AResult<CvtPackForm> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if decoded.modifier("sat")? != "sat" || decoded.modifier("abtype")? != "s32" {
        return unsupported(format!("{op_name} requires cvt.pack.sat.*.s32"));
    }
    let convert = decoded.modifier("convert")?;
    if !CONVERT_TYPES.contains(&convert) {
        return unsupported(format!(
            "{op_name} has unreviewed convert type {:?}",
            convert
        ));
    }
    let signed = convert.starts_with('s');
    let bits: i64 = convert[1..].parse().expect("convert type width");
    let has_c = op_name == "tirx.ptx.cvt_pack_c";
    if has_c != (bits < 16) {
        return unsupported(format!(
            "{op_name}.{convert} operand shape disagrees with the PTX syntax line"
        ));
    }
    if has_c && decoded.modifier("ctype")? != "b32" {
        return unsupported(format!("{op_name} requires the .b32 c-type"));
    }
    if operands.destinations.len() != 1 || operands.sources.len() != 2 + usize::from(has_c) {
        return unsupported(format!("{op_name} operand shape disagrees with its schema"));
    }
    let mut sources = operands.sources.into_iter();
    Ok(CvtPackForm {
        destination: operands.destinations[0].clone(),
        a: sources.next().expect("cvt.pack a"),
        b: sources.next().expect("cvt.pack b"),
        c: sources.next(),
        bits,
        signed,
    })
}

/// `CvtPack<bits, signed>` specialization text.
pub fn cvt_pack_variant(form: &CvtPackForm) -> String {
    format!(
        "v2::reg::variant::CvtPack<{}, {}>",
        form.bits,
        if form.signed { "true" } else { "false" }
    )
}

/// The validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_cvt_pack(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_cvt_pack(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_cvt_pack`.
    pub fn emit_ptx_cvt_pack(
        &mut self,
        decoded: &DecodedPtx,
        form: &CvtPackForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "cvt_pack",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        let body = self.emit_cvt_pack_body(decoded, form, source_op_id);
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

    fn emit_cvt_pack_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &CvtPackForm,
        source_op_id: i64,
    ) -> AResult<()> {
        let a_value = self.emit_expr(&form.a)?;
        let a_value = self.as_warp_value(a_value);
        let b_value = self.emit_expr(&form.b)?;
        let b_value = self.as_warp_value(b_value);
        if a_value.rust_type != "i32" || b_value.rust_type != "i32" {
            return unsupported(format!("{}.a and .b must lower to i32", decoded.op_name));
        }
        let mut arguments = vec![abi::register(&a_value.code), abi::register(&b_value.code)];
        if let Some(c) = &form.c {
            let c_value =
                self.emit_as_unsigned_bits(c, 32, &decoded.op_name, "cvt_pack_c_bits", None)?;
            arguments.push(abi::register(&c_value.code));
        }
        let raw = self.control_name("cvt_pack_raw");
        let result = self.control_name("cvt_pack");
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call(
            CVT_PACK,
            &site,
            &[format!("({})", arguments.join(", "))],
            Some(&cvt_pack_variant(&form)),
        );
        self.emit_line(&format!("let {raw} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({raw});"));
        self.emit_explicit_buffer_store(
            &form.destination,
            RustValue::new(result, "u32", Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }
}
