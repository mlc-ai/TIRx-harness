//! Validation and emission of the ptx_lop3 instruction family.

use crate::analyze::util::{dtype_of, prim, static_int, unsupported, AResult};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, table_marker, PTX_BOOL_MARKERS};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

pub const ABI_FUNCTION: &str = "reg::lop3";

pub struct Lop3Form {
    pub destination: Option<ObjectRef>,
    pub predicate: Option<ObjectRef>,
    pub sources: Vec<ObjectRef>,
    pub predicate_source: Option<ObjectRef>,
    pub variant: String,
}

/// `resolve_ptx_lop3`.
pub fn resolve_ptx_lop3(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<Lop3Form> {
    require_register_call(decoded, false)?;
    let op_name = decoded.op_name.as_str();
    if decoded.modifier("type")? != "b32" {
        return unsupported(format!("{op_name} requires the b32 instruction type"));
    }
    let sources = vec![
        decoded.scalar_operand("a")?,
        decoded.scalar_operand("b")?,
        decoded.scalar_operand("c")?,
    ];
    let lut = static_int(
        &ctx.analyzer,
        &prim(&decoded.scalar_operand("immLut")?)?,
        &format!("{op_name}.immLut"),
        "must be a static integer",
    )?;
    if !(0..=0xFF).contains(&lut) {
        return unsupported(format!("{op_name}.immLut must be in [0, 255], got {lut}"));
    }
    let destination = if op_name == "tirx.ptx.lop3_bool_sink" {
        None
    } else {
        Some(decoded.scalar_operand("d")?)
    };
    if op_name == "tirx.ptx.lop3" {
        return Ok(Lop3Form {
            destination,
            predicate: None,
            sources,
            predicate_source: None,
            variant: format!("v2::reg::variant::Lop3<{lut}>"),
        });
    }
    let predicate = decoded.scalar_operand("p")?;
    let q = decoded.scalar_operand("q")?;
    let bool_op = decoded.modifier("boolop")?;
    if !matches!(bool_op, "and" | "or") {
        return unsupported(format!(
            "{op_name} requires an and/or predicate combiner, got {:?}",
            bool_op
        ));
    }
    Ok(Lop3Form {
        destination,
        predicate: Some(predicate),
        sources,
        predicate_source: Some(q),
        variant: format!(
            "v2::reg::variant::Lop3Bool<{lut}, v2::reg::variant::{}>",
            table_marker(PTX_BOOL_MARKERS, bool_op).expect("validated combiner")
        ),
    })
}

/// `resolve_ptx_lop3`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_lop3(emitter.ctx, decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_lop3(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_lop3`.
    pub fn emit_ptx_lop3(
        &mut self,
        decoded: &DecodedPtx,
        form: &Lop3Form,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "lop3",
            "lop3 instruction predicate must be bool or integer",
        )?;
        self.emit_ptx_lop3_body(decoded, form, source_op_id)?;
        let mask = region.mask.clone();
        self.close_predicated_region(region);
        for (destination, is_predicate) in [(&form.destination, false), (&form.predicate, true)] {
            if let Some(destination) = destination {
                let result_dtype = dtype_of(destination)?;
                self.finish_predicated_destinations(
                    decoded,
                    &[Some(destination.clone())],
                    &result_dtype,
                    &mask,
                    source_op_id,
                    is_predicate,
                )?;
            }
        }
        Ok(())
    }

    fn emit_ptx_lop3_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &Lop3Form,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let mut sources = Vec::new();
        for (index, expression) in form.sources.iter().enumerate() {
            sources.push(self.emit_as_unsigned_bits(
                expression,
                32,
                op_name,
                &format!("lop3_source_{index}"),
                None,
            )?);
        }
        let arguments = sources
            .iter()
            .map(|source| abi::register(&source.code))
            .collect::<Vec<_>>()
            .join(", ");
        let raw = self.control_name("lop3_raw");

        let (raw_bits, raw_predicate) = match &form.predicate_source {
            None => {
                let site = self.v2_site(Some(source_op_id));
                let call = abi::lane_call(
                    ABI_FUNCTION,
                    &site,
                    &[format!("({arguments})")],
                    Some(&form.variant),
                );
                self.emit_line(&format!("let {raw} = {call};"));
                (raw.clone(), None)
            }
            Some(predicate_source) => {
                let q = self.emit_expr(predicate_source)?;
                let q = self.coerce_value(q, "bool", "lop3_predicate_source")?;
                let q = self.as_warp_value(q);
                let site = self.v2_site(Some(source_op_id));
                let call = abi::lane_call(
                    ABI_FUNCTION,
                    &site,
                    &[format!("({arguments}, {})", abi::register(&q.code))],
                    Some(&form.variant),
                );
                self.emit_line(&format!("let {raw} = {call};"));
                let raw_bits = self.control_name("lop3_bits_raw");
                let raw_predicate = self.control_name("lop3_predicate_raw");
                self.emit_line(&format!("let ({raw_bits}, {raw_predicate}) = {raw};"));
                (raw_bits, Some(raw_predicate))
            }
        };

        if let Some(destination) = &form.destination {
            let bits = self.control_name("lop3_bits");
            self.emit_line(&format!("let {bits} = v2_register_out({raw_bits});"));
            let value = self.emit_from_unsigned_bits(
                RustValue::new(bits, "u32", Uniformity::Varying),
                &dtype_of(destination)?,
                32,
                op_name,
                "lop3_result",
            )?;
            self.emit_explicit_buffer_store(destination, value, source_op_id, None, None, None)?;
        }

        if let Some(predicate) = &form.predicate {
            let raw_predicate = raw_predicate.expect("predicate raw");
            let predicate_bits = self.control_name("lop3_predicate");
            let predicate_u32 = self.control_name("lop3_predicate_u32");
            self.emit_line(&format!(
                "let {predicate_bits} = v2_register_out({raw_predicate});"
            ));
            self.emit_line(&format!(
                "let {predicate_u32} = WarpValue::from_fn(|lane| u32::from({predicate_bits}[lane]));"
            ));
            self.emit_explicit_buffer_store(
                predicate,
                RustValue::new(predicate_u32, "u32", Uniformity::Varying),
                source_op_id,
                None,
                None,
                None,
            )?;
        }
        Ok(())
    }
}
