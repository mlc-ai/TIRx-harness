//! Validation and emission of the ptx_cache_policy instruction family.

use crate::analyze::util::{dtype_of, ffi_error, unsupported, AResult, Failure};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::ir::TensorLoadObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

/// The validated parts of one `createpolicy` instruction.
pub struct CachePolicyParts {
    /// The uint64 policy destination lvalue.
    pub destination: ObjectRef,
    pub kind: String,
    /// `(operand, dtype)` register operands in ABI order.
    pub operands: &'static [(&'static str, &'static str)],
}

fn cache_policy_parts(decoded: &DecodedPtx) -> AResult<CachePolicyParts> {
    let op_name = decoded.op_name.as_str();
    let destination = decoded.scalar_operand("cache_policy")?;
    if destination.as_node::<TensorLoadObj>().is_none() || dtype_of(&destination)? != "uint64" {
        return unsupported(format!("{op_name} requires a uint64 destination lvalue"));
    }
    let kind = decoded.modifier("kind")?;
    let operands: &'static [(&'static str, &'static str)] = match kind {
        "range" => &[("primary_size", "uint32"), ("total_size", "uint32")],
        "fractional" if decoded.has_operand("fraction") => &[("fraction", "float32")],
        "fractional" => &[],
        "cvt" => &[("access_property", "uint64")],
        other => return Err(Failure::Ffi(ffi_error(&format!("{:?}", other)))),
    };
    for (name, dtype) in operands {
        if dtype_of(&decoded.scalar_operand(name)?)? != *dtype {
            return unsupported(format!("{op_name}.{name} must be {dtype}"));
        }
    }
    Ok(CachePolicyParts {
        destination,
        kind: kind.to_owned(),
        operands,
    })
}

/// The engine function (below `v2::`) the lowering calls.
pub const CREATEPOLICY: &str = "reg::createpolicy";

/// The resolution of one `createpolicy` instruction.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = cache_policy_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_cache_policy(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_cache_policy`.
    pub fn emit_ptx_cache_policy(
        &mut self,
        decoded: &DecodedPtx,
        parts: &CachePolicyParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "cache_policy",
            "createpolicy predicate must be bool or integer",
        )?;
        let context = region.context.clone();
        let mask = region.mask.clone();
        let result = (|| -> AResult<()> {
            let mut arguments = Vec::new();
            if parts.kind == "range" {
                let pointer = self.emit_address_pointer(
                    &decoded.scalar_operand("addr")?,
                    "global",
                    None,
                    "ctx.active_mask()",
                )?;
                arguments.push(abi::address(
                    "v2::Global",
                    &abi::cloned(&pointer.code),
                    None,
                ));
            }
            for (name, _dtype) in parts.operands {
                let value = self.emit_expr(&decoded.scalar_operand(name)?)?;
                let value = self.as_warp_value(value);
                arguments.push(abi::register(&value.code));
            }
            if parts.kind == "fractional" && parts.operands.is_empty() {
                arguments.push(abi::splat("1.0_f32"));
            }
            let variant = match parts.kind.as_str() {
                "range" => "PolicyRange",
                "fractional" => "PolicyFraction",
                _ => "PolicyConvert",
            };
            let raw = self.control_name("cache_policy_result");
            let argument = if parts.kind == "range" {
                format!("({})", arguments.join(", "))
            } else {
                arguments[0].clone()
            };
            let site = self.v2_site(Some(source_op_id));
            let invocation = abi::lane_call_context(
                CREATEPOLICY,
                &site,
                &[argument],
                Some(&format!("v2::reg::variant::{variant}")),
                context.as_deref(),
            );
            self.emit_line(&format!("let {raw} = {invocation};"));
            self.emit_explicit_buffer_store(
                &parts.destination,
                RustValue::new(
                    format!("v2_register_out({raw})"),
                    "u64",
                    Uniformity::Varying,
                ),
                source_op_id,
                None,
                None,
                None,
            )
        })();
        self.close_predicated_region(region);
        result?;
        self.finish_predicated_destinations(
            decoded,
            &[Some(parts.destination.clone())],
            "uint64",
            &mask,
            source_op_id,
            false,
        )
    }
}
