//! Validation and emission of PTX address wrappers.
use crate::analyze::buffers::call_op_name;
use crate::analyze::util::{ffi_error, unsupported, AResult, Failure, IdSet};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::{Emitter, RustValue, Uniformity};
use tvm::ir::CallObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

// ----------------------------------------------------------------------
// The PTX address wrapper contract.
// ----------------------------------------------------------------------

fn is_addr_call(value: &ObjectRef) -> AResult<bool> {
    use tvm::tvm_ffi::ObjectRefCore;
    let Some(call) = value.as_node::<tvm::ir::CallObj>() else {
        return Ok(false);
    };
    Ok(crate::analyze::buffers::call_op_name(call)?.as_deref() == Some("tirx.ptx.addr"))
}

/// `(base, byte_offset expression, offset)`: the one check of the wrapper
/// contract, shared by call resolution and decoded operand validation.
pub fn ptx_addr_parts(
    ctx: &crate::analyze::Ctx,
    node: &ObjectRef,
) -> AResult<(ObjectRef, ObjectRef, i64)> {
    use tvm::tvm_ffi::ObjectRefCore;
    let Some(call) = node.as_node::<tvm::ir::CallObj>() else {
        return Err(Failure::Ffi(ffi_error(
            "PTX address wrapper expects a tirx Call",
        )));
    };
    let args: Vec<ObjectRef> = call.args.iter().map(crate::analyze::util::oref).collect();
    if args.len() != 2 {
        return unsupported(format!("{} expects base and byte_offset", "tirx.ptx.addr"));
    }
    let (base, byte_offset) = (&args[0], &args[1]);
    if is_addr_call(base)? {
        return unsupported(format!("{} cannot be nested", "tirx.ptx.addr"));
    }
    let offset_dtype = crate::analyze::util::dtype_of(byte_offset)?;
    if !crate::tables::is_integer_dtype(&offset_dtype) || offset_dtype == "bool" {
        return unsupported(format!(
            "{} byte_offset must be an integer scalar, got {offset_dtype}",
            "tirx.ptx.addr"
        ));
    }
    let simplified =
        crate::analyze::util::simplify(&ctx.analyzer, &crate::analyze::util::prim(byte_offset)?)?;
    let Some(offset) = crate::analyze::util::int_imm_expr(&simplified) else {
        return unsupported(format!(
            "{} byte_offset must be a compile-time signed int32 constant",
            "tirx.ptx.addr"
        ));
    };
    if !(-(1i64 << 31)..=(1i64 << 31) - 1).contains(&offset) {
        return unsupported(format!(
            "{} byte_offset {offset} is outside signed int32 range",
            "tirx.ptx.addr"
        ));
    }
    if crate::analyze::util::expr_type(base)
        .is_none_or(|ty| !crate::analyze::util::is_pointer_type(&ty))
    {
        return unsupported(format!(
            "{} base must retain a physical pointer type",
            "tirx.ptx.addr"
        ));
    }
    Ok((base.clone(), byte_offset.clone(), offset))
}

/// A validated `T.ptx.addr`: its pointer base and static byte offset.
pub struct PtxAddr {
    pub base: ObjectRef,
    pub offset: i64,
}

/// The validated operands.

pub fn validate_decoded_ptx_addr_operands(
    ctx: &crate::analyze::Ctx,
    decoded: &DecodedPtx,
) -> AResult<()> {
    for slot in &decoded.operands {
        for value in slot.values.iter().flatten() {
            if !is_addr_call(value)? {
                continue;
            }
            if slot.kind != "addr" || !slot.allow_imm_offset {
                return unsupported(format!(
                    "{}: operand '{}' does not support T.ptx.addr",
                    decoded.op_name, slot.name
                ));
            }
            ptx_addr_parts(ctx, value)?;
        }
    }
    Ok(())
}

impl<'a> Emitter<'a> {
    /// A byte-displaced integer address.
    pub fn emit_ptx_addr(&mut self, addr: &PtxAddr) -> AResult<RustValue> {
        let offset = addr.offset;
        let mut pointer = self.emit_expr(&addr.base)?;
        if pointer.rust_type == "PhysicalPtr" {
            let base = self.temp("ptx_addr_base");
            self.emit_line(&format!(
                "let {base} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
                pointer.code
            ));
            pointer = RustValue::new(base, "u64", Uniformity::Varying);
        }
        let pointer = self.as_warp_value(pointer);
        if pointer.rust_type != "u32" && pointer.rust_type != "u64" {
            return unsupported(format!(
                "{} base lowered to {}, expected u32 or u64",
                "tirx.ptx.addr", pointer.rust_type
            ));
        }
        let result = self.temp("ptx_addr");
        self.emit_line(&format!(
            "let {result} = WarpValue::from_fn(|lane| {}[lane].wrapping_add(({offset}_i64) as {}));",
            pointer.code, pointer.rust_type
        ));
        Ok(RustValue::new(
            result,
            pointer.rust_type,
            Uniformity::Varying,
        ))
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    if !emitter.consumed_address_wrappers.contains(call.node) {
        return unsupported(
            "tirx.ptx.addr must be consumed by a PTX address operand that supports it",
        );
    }
    let (base, _, offset) = ptx_addr_parts(emitter.ctx, call.node)?;
    emitter
        .with_call_expr(call.node, |emitter| {
            emitter.emit_ptx_addr(&PtxAddr { base, offset })
        })
        .map(Some)
}

/// ``T.ptx.addr`` is a trace-time wrapper: the exact wrapper nodes consumed by
/// an eligible target-table address operand.
pub fn consumed_calls(ctx: &Ctx, nodes: &[ObjectRef]) -> AResult<IdSet> {
    let mut consumed = IdSet::default();
    for node in nodes {
        let Some(crate::decode::ptx::PtxDecode::Decoded(decoded)) =
            ctx.decoding.ptx_decode(node)?
        else {
            continue;
        };
        for slot in &decoded.operands {
            if slot.kind != "addr" || !slot.allow_imm_offset {
                continue;
            }
            for value in slot.values.iter().flatten() {
                if let Some(inner) = value.as_node::<CallObj>() {
                    if call_op_name(inner)?.as_deref() == Some("tirx.ptx.addr") {
                        consumed.add(value.clone());
                    }
                }
            }
        }
    }
    Ok(consumed)
}
