//! Validation and emission support for ptx_clmad.

use crate::analyze::util::{unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, RegisterCall};
use crate::emit::{Emitter, RustValue};

/// `resolve_ptx_clmad`.
pub fn resolve_ptx_clmad(decoded: &DecodedPtx) -> AResult<RegisterCall> {
    let op_name = decoded.op_name.as_str();
    let operands = require_register_call(decoded, false)?;
    if decoded.modifier("type")? != "u64" {
        return unsupported(format!("{op_name} requires the u64 instruction type"));
    }
    let mode = decoded.modifier("mode")?;
    let variant = match mode {
        "hi" => "ClmadHi",
        "lo" => "ClmadLo",
        _ => return unsupported(format!("{op_name} has unsupported mode {:?}", mode)),
    };
    if operands.destinations.len() != 1 || operands.sources.len() != 3 {
        return unsupported(format!(
            "{op_name} requires one destination and three sources"
        ));
    }
    Ok(RegisterCall::new(
        operands.destinations[0].clone(),
        operands.sources,
        "clmad",
        format!("v2::reg::variant::{variant}"),
        vec!["u64"; 3],
        "u64",
    )
    .with_predicate_region(
        "ptx_clmad",
        format!("{op_name} predicate must lower to bool or integer"),
    ))
}

/// `resolve_ptx_clmad`, keeping the validated form.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = resolve_ptx_clmad(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_register_call(decoded, &parts, source_op_id)?;
    Ok(None)
}
