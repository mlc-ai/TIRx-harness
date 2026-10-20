//! Validation and emission of `multimem.ld_reduce`, `multimem.st`, and
//! `multimem.red` (PTX ISA 9.7.10.15).
//!
//! Every form lowers to one `v2::mem::multimem` call whose const codes are
//! decoded by the engine (`MultimemForm::decode`); register groups travel as
//! up to four little-endian 32-bit words.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    LdReduce,
    St,
    Red,
}

struct MultimemParts {
    kind: Kind,
    kind_code: u8,
    type_code: u8,
    op_code: u8,
    sem_code: u8,
    scope_code: u8,
    register_bits: usize,
    address: ObjectRef,
    registers: Vec<Option<ObjectRef>>,
}

fn decoded_parts(decoded: &DecodedPtx) -> AResult<MultimemParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let family = op_name
        .strip_prefix("tirx.ptx.multimem_")
        .unwrap_or_default()
        .trim_end_matches("_vec")
        .trim_end_matches("_f");
    let (kind, kind_code) = match family {
        "ld_reduce" => (Kind::LdReduce, 0),
        "st" => (Kind::St, 1),
        "red" => (Kind::Red, 2),
        _ => return unsupported(format!("{op_name} is not a modeled multimem form")),
    };
    let space = decoded.modifier_or_empty("space");
    if !(space.is_empty() || space == "global") {
        return unsupported(format!("{op_name} has unsupported state space {space:?}"));
    }
    let ptx_type = decoded.modifier("type")?;
    let (type_code, register_bits) = match ptx_type {
        "u32" => (0, 32),
        "s32" => (1, 32),
        "u64" => (2, 64),
        "s64" => (3, 64),
        "b32" => (4, 32),
        "b64" => (5, 64),
        "f16" => (6, 16),
        "f16x2" => (7, 32),
        "bf16" => (8, 16),
        "bf16x2" => (9, 32),
        "f32" => (10, 32),
        "f64" => (11, 64),
        other => return unsupported(format!("{op_name} has unsupported type .{other}")),
    };
    let acc = decoded.modifier_or_empty("acc") == "acc::f32";
    let op_code = match (decoded.modifier_or_empty("op"), acc) {
        ("", false) if kind == Kind::St => 0,
        ("add", false) => 1,
        ("add", true) => 2,
        ("min", false) => 3,
        ("max", false) => 4,
        ("and", false) => 5,
        ("or", false) => 6,
        ("xor", false) => 7,
        (op, _) => {
            return unsupported(format!(
                "{op_name} has unsupported operation {op:?}{}",
                if acc { " with .acc::f32" } else { "" }
            ))
        }
    };
    let semantic = decoded.modifier_or_empty("sem");
    let scope = decoded.modifier_or_empty("scope");
    // `.weak` is its own syntax line with no scope; the ordered forms take a
    // scope, and an omitted pair is weak for ld_reduce/st and `.relaxed.sys`
    // for red.
    let (sem_code, scope_code) = match (kind, semantic, scope) {
        (Kind::LdReduce | Kind::St, "" | "weak", "") => (0, 3),
        (Kind::Red, "", "") => (1, 3),
        (_, "relaxed" | "acquire" | "release", scope) if !scope.is_empty() => {
            let sem_code = match (kind, semantic) {
                (_, "relaxed") => 1,
                (Kind::LdReduce, "acquire") => 2,
                (Kind::St | Kind::Red, "release") => 3,
                _ => {
                    return unsupported(format!(
                        "{op_name} does not take .{semantic}"
                    ))
                }
            };
            let scope_code = match scope {
                "cta" => 0,
                "cluster" => 1,
                "gpu" => 2,
                "sys" => 3,
                other => return unsupported(format!("{op_name} has unsupported scope {other:?}")),
            };
            (sem_code, scope_code)
        }
        _ => {
            return unsupported(format!(
                "{op_name} has an invalid semantic/scope pair {semantic:?}/{scope:?}"
            ))
        }
    };
    let address = decoded.scalar_operand("addr")?;
    let register_slot = if kind == Kind::LdReduce { "d" } else { "b" };
    let registers = decoded.operand(register_slot)?.to_vec();
    let total_bits = registers.len() * register_bits;
    if !matches!(total_bits, 32 | 64 | 128) {
        return unsupported(format!(
            "{op_name} accesses {total_bits} bits; multimem requires 32, 64, or 128"
        ));
    }
    if kind != Kind::LdReduce && registers.iter().any(Option::is_none) {
        return unsupported(format!("{op_name}.b cannot contain a sunk lane"));
    }
    Ok(MultimemParts {
        kind,
        kind_code,
        type_code,
        op_code,
        sem_code,
        scope_code,
        register_bits,
        address,
        registers,
    })
}

impl<'a> Emitter<'a> {
    fn emit_multimem(&mut self, decoded: &DecodedPtx, parts: &MultimemParts, source_op_id: i64) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "multimem",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let pointer =
            self.decoded_atomic_pointer(&parts.address, &op_name, "global", &region.mask, None)?;
        let bits = parts.register_bits;
        let carrier = format!("u{bits}");
        let words = self.control_name("multimem_words");
        if parts.kind == Kind::LdReduce {
            self.emit_line(&format!("let {words} = WarpValue::splat([0_u32; 4]);"));
        } else {
            let mut assignments = Vec::new();
            for (index, register) in parts.registers.iter().enumerate() {
                let register = register.as_ref().expect("validated multimem source lane");
                let value = self.emit_expr(register)?;
                let value = self.atomic_register_bits(value, &carrier);
                let code = &value.code;
                match bits {
                    16 => assignments.push(format!(
                        "words[{}] |= u32::from({code}[lane]) << {};",
                        index / 2,
                        (index % 2) * 16
                    )),
                    32 => assignments.push(format!("words[{index}] = {code}[lane];")),
                    _ => assignments.push(format!(
                        "words[{}] = {code}[lane] as u32; words[{}] = ({code}[lane] >> 32) as u32;",
                        index * 2,
                        index * 2 + 1
                    )),
                }
            }
            self.emit_line(&format!(
                "let {words} = WarpValue::from_fn(|lane| {{ let mut words = [0_u32; 4]; {} words }});",
                assignments.join(" ")
            ));
        }
        let variant = format!(
            "v2::mem::variant::Multimem<{}, {}, {}, {}, {}, {}>",
            parts.kind_code,
            parts.type_code,
            parts.op_code,
            parts.registers.len(),
            parts.sem_code,
            parts.scope_code
        );
        let address = abi::address("v2::Global", &abi::cloned(&pointer.code), None);
        let site = self.v2_site(Some(source_op_id));
        let result = self.control_name("multimem_result");
        let invocation = abi::warp_call(
            "mem::multimem",
            &site,
            &[format!("({address}, {})", abi::register(&words))],
            Some(&variant),
            region.context.as_deref(),
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = {invocation};"));
        if parts.kind == Kind::LdReduce {
            self.emit_line(&format!("let {result} = v2_register_out({result});"));
            for (index, destination) in parts.registers.iter().enumerate() {
                let Some(destination) = destination else {
                    continue;
                };
                let extract = match bits {
                    16 => format!(
                        "({result}[lane][{}] >> {}) as u16",
                        index / 2,
                        (index % 2) * 16
                    ),
                    32 => format!("{result}[lane][{index}]"),
                    _ => format!(
                        "u64::from({result}[lane][{}]) | (u64::from({result}[lane][{}]) << 32)",
                        index * 2,
                        index * 2 + 1
                    ),
                };
                let component = self.control_name("multimem_component");
                self.emit_line(&format!(
                    "let {component} = WarpValue::from_fn(|lane| {extract});"
                ));
                dtype_of(destination)?;
                self.atomic_result_store(
                    destination,
                    RustValue::new(component, carrier.clone(), Uniformity::Varying),
                    source_op_id,
                    &region.mask,
                )?;
            }
        }
        self.close_predicated_region(region);
        Ok(())
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = decoded_parts(decoded)?;
    if parts.kind != Kind::LdReduce {
        // The bytes written are the window's replicas, which no single
        // buffer binding names; leave the written-buffer set unknown.
        emitter.written_global_buffers = None;
    }
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_multimem(decoded, &parts, source_op_id)?;
    Ok(None)
}
