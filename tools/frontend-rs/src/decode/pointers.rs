//! Instruction-owned pointer flow facts consumed by generic root analysis.

use tvm::ir::CallObj;
use tvm::tvm_ffi::object::ObjectRef;

use super::ptx::DecodedPtx;
use crate::analyze::buffers::call_op_name;
use crate::analyze::util::{oref, AResult};

pub enum PointerCall {
    Address(ObjectRef),
    Forward(ObjectRef),
    Select(ObjectRef, ObjectRef),
    Opaque,
}

/// Address-taking escapes storage even before call arity is validated.
pub fn address_operand(call: &CallObj) -> AResult<Option<ObjectRef>> {
    Ok(
        if call_op_name(call)?.as_deref() == Some("tirx.address_of") {
            call.args.get(0).ok().map(oref)
        } else {
            None
        },
    )
}

/// Preserve the existing conservative root rules, including arity guards.
/// This classifies provenance only, not instruction validity or execution.
pub fn pointer_call(call: &CallObj) -> AResult<PointerCall> {
    let name = call_op_name(call)?;
    Ok(match name.as_deref() {
        Some("tirx.address_of") if call.args.len() == 1 => {
            PointerCall::Address(oref(call.args.get(0)?))
        }
        Some(
            "tirx.reinterpret"
            | "tirx.ptx.addr"
            | "tirx.ptr_byte_offset"
            | "tirx.handle_add_byte_offset"
            | "tirx.cuda.cvta_generic_to_shared"
            | "tirx.cuda.smem_addr_from_u64"
            | "tirx.cuda.sm100_mbarrier_addr",
        ) if !call.args.is_empty() => PointerCall::Forward(oref(call.args.get(0)?)),
        Some("tirx.tvm_access_ptr") if call.args.len() > 1 => {
            PointerCall::Forward(oref(call.args.get(1)?))
        }
        Some("prim.if_then_else") if call.args.len() == 3 => {
            PointerCall::Select(oref(call.args.get(1)?), oref(call.args.get(2)?))
        }
        _ => PointerCall::Opaque,
    })
}

/// Sources whose pointer roots survive a register write. Other writes keep
/// the pre-existing empty-source result; generic analysis owns the join.
pub fn register_pointer_sources(decoded: &DecodedPtx) -> AResult<Vec<ObjectRef>> {
    Ok(match decoded.op_name.as_str() {
        "tirx.ptx.mov"
            if matches!(decoded.modifier_or_empty("type"), "b64" | "u64")
                && decoded.operand("a")?.len() == 1
                && decoded
                    .operands
                    .iter()
                    .filter(|s| s.kind == "reg" && s.rw.contains('w'))
                    .map(|s| s.values.len())
                    .sum::<usize>()
                    == 1 =>
        {
            decoded.operand("a")?.iter().flatten().cloned().collect()
        }
        "tirx.ptx.selp" if matches!(decoded.modifier_or_empty("type"), "b64" | "u64") => decoded
            .operand("a")?
            .iter()
            .chain(decoded.operand("b")?)
            .flatten()
            .cloned()
            .collect(),
        _ => Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::ptx::PtxOperand;
    use super::*;
    use crate::analyze::util::same;
    use tvm::ir::{Call, Expr, IntImm, Op, PrimType};

    fn initialize() {
        tvm::libinfo::load_compiler().expect("TVM compiler");
    }

    fn call(name: &str, args: Vec<Expr>) -> Call {
        Call::new(
            PrimType::new("uint64").unwrap(),
            Op::get(name).unwrap(),
            args,
        )
    }

    #[test]
    fn pointer_calls_preserve_sources_arity_and_address_escapes() {
        initialize();
        let a: Expr = IntImm::new("int32", 1).unwrap().into();
        let b: Expr = IntImm::new("int32", 2).unwrap().into();
        let address = call("tirx.address_of", vec![a.clone()]);
        assert!(
            matches!(pointer_call(&address).unwrap(), PointerCall::Address(v) if same(&v, &oref(a.clone())))
        );
        let malformed_address = call("tirx.address_of", vec![a.clone(), b.clone()]);
        assert!(matches!(
            pointer_call(&malformed_address).unwrap(),
            PointerCall::Opaque
        ));
        assert!(same(
            &address_operand(&malformed_address).unwrap().unwrap(),
            &oref(a.clone())
        ));
        assert!(address_operand(&call("tirx.address_of", vec![]))
            .unwrap()
            .is_none());
        // Python-registered PTX calls are covered by the integration suite.
        for name in ["tirx.reinterpret", "tirx.ptr_byte_offset"] {
            assert!(
                matches!(pointer_call(&call(name, vec![a.clone()])).unwrap(), PointerCall::Forward(v) if same(&v, &oref(a.clone())))
            );
            assert!(matches!(
                pointer_call(&call(name, vec![])).unwrap(),
                PointerCall::Opaque
            ));
        }
        assert!(
            matches!(pointer_call(&call("tirx.tvm_access_ptr", vec![a.clone(), b.clone()])).unwrap(), PointerCall::Forward(v) if same(&v, &oref(b.clone())))
        );
        let select = call("prim.if_then_else", vec![a.clone(), a.clone(), b.clone()]);
        assert!(
            matches!(pointer_call(&select).unwrap(), PointerCall::Select(x, y) if same(&x, &oref(a.clone())) && same(&y, &oref(b.clone())))
        );
        assert!(matches!(
            pointer_call(&call("prim.if_then_else", vec![a, b])).unwrap(),
            PointerCall::Opaque
        ));
    }

    fn operand(name: &str, rw: &str, value: ObjectRef) -> PtxOperand {
        PtxOperand {
            name: name.into(),
            kind: "reg".into(),
            rw: rw.into(),
            lanes: 1,
            allow_imm_offset: false,
            operand_type: "b64".into(),
            dtypes: vec!["uint64".into()],
            literal: None,
            values: vec![Some(value)],
        }
    }

    #[test]
    fn register_sources_preserve_width_destination_and_sink_guards() {
        initialize();
        let a = oref(IntImm::new("int32", 1).unwrap());
        let b = oref(IntImm::new("int32", 2).unwrap());
        let mut decoded = DecodedPtx {
            op_name: "tirx.ptx.mov".into(),
            operands: vec![operand("d", "w", b.clone()), operand("a", "r", a.clone())],
            modifiers: vec![("type".into(), "b64".into())],
            predicate: None,
            preserve_dst: false,
            result_type: String::new(),
        };
        for width in ["b64", "u64"] {
            decoded.modifiers[0].1 = width.into();
            let sources = register_pointer_sources(&decoded).unwrap();
            assert_eq!(sources.len(), 1);
            assert!(same(&sources[0], &a));
        }
        decoded.modifiers[0].1 = "b32".into();
        assert!(register_pointer_sources(&decoded).unwrap().is_empty());
        decoded.modifiers[0].1 = "b64".into();
        decoded.operands[0].values.push(Some(b.clone()));
        assert!(register_pointer_sources(&decoded).unwrap().is_empty());
        decoded.operands[0].values.pop();
        decoded.operands[1].values[0] = None;
        assert!(register_pointer_sources(&decoded).unwrap().is_empty());
        decoded.operands[1].values[0] = Some(a.clone());
        decoded.op_name = "tirx.ptx.selp".into();
        decoded.operands.push(operand("b", "r", b.clone()));
        let sources = register_pointer_sources(&decoded).unwrap();
        assert_eq!(sources.len(), 2);
        assert!(same(&sources[0], &a) && same(&sources[1], &b));
        decoded.op_name = "tirx.ptx.add".into();
        assert!(register_pointer_sources(&decoded).unwrap().is_empty());
    }
}
