//! Decoded `tirx.ptx.*` calls (`ptx_dialect.DecodedPtxCall`).
//!
//! The target TVM PTX table expresses per-modifier facts (lane counts, operand
//! types, accepted carrier dtypes, sinkability) as Python callables, so the
//! native frontend asks Python to decode each PTX call it meets
//! (the callback installed by `ptx_dialect.register_native_ptx_decoder`) and
//! receives the decoded operands plus the slot facts of that modifier
//! combination. Instruction emitters consume the resulting `DecodedPtx`.

use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{self, Any, Array, Map, String as FfiString};

use crate::analyze::util::{dtype_of, ffi_error, unmodeled, unsupported, AResult, Failure, Json};

// The Python callback constructs named records from these same field lists.
macro_rules! payload_fields {
    ($record:ident { $($field:ident: $name:literal),+ $(,)? }) => {
        struct $record;
        impl $record {
            $(const $field: &'static str = $name;)+
            const NAMES: &'static [&'static str] = &[$($name),+];
        }
    };
}

payload_fields!(CallFields {
    OP_NAME: "op_name",
    MODIFIERS: "modifiers",
    PREDICATE: "predicate",
    PRESERVE_DST: "preserve_dst",
    RESULT_TYPE: "result_type",
    OPERANDS: "operands",
    ERROR: "error",
});

payload_fields!(OperandFields {
    NAME: "name",
    KIND: "kind",
    RW: "rw",
    LANES: "lanes",
    ALLOW_IMM_OFFSET: "allow_imm_offset",
    LITERAL: "literal",
    OPERAND_TYPE: "operand_type",
    DTYPES: "dtypes",
    VALUES: "values",
});

pub fn payload_schema() -> tvm_ffi::Result<Map<FfiString, Array<FfiString>>> {
    Ok([
        ("call", CallFields::NAMES),
        ("operand", OperandFields::NAMES),
    ]
    .into_iter()
    .map(|(name, fields)| {
        (
            FfiString::from(name),
            Array::new(fields.iter().map(|field| FfiString::from(*field)).collect()),
        )
    })
    .collect())
}

/// One table operand slot with the facts of the decoded modifier combination.
#[derive(Clone)]
pub struct PtxOperand {
    pub name: String,
    /// `"reg"`, `"addr"`, `"ptr"` or `"imm"`.
    pub kind: String,
    /// `"r"`, `"w"` or `"rw"`.
    pub rw: String,
    pub lanes: i64,
    pub allow_imm_offset: bool,
    /// `operand_type(slot, modifiers)`; empty for non-register slots.
    pub operand_type: String,
    /// `operand_dtypes(slot, modifiers)`; empty for non-register slots.
    pub dtypes: Vec<String>,
    /// A table-owned literal immediate (no call argument).
    pub literal: Option<String>,
    /// The bound argument nodes in lane order; `None` is a sunk lane (`PTX_SINK`).
    pub values: Vec<Option<ObjectRef>>,
}

#[derive(Clone)]
pub struct DecodedPtx {
    pub op_name: String,
    pub operands: Vec<PtxOperand>,
    /// Every modifier slot name -> token, `""` = omitted, in slot order.
    pub modifiers: Vec<(String, String)>,
    pub predicate: Option<ObjectRef>,
    pub preserve_dst: bool,
    /// `str(call.ty.dtype)`; empty for void calls.
    pub result_type: String,
}

impl DecodedPtx {
    /// An optional slot: zero or one present lanes. A sink is malformed.
    pub fn optional_scalar_operand(&self, name: &str) -> AResult<Option<ObjectRef>> {
        let values = self.operand(name)?;
        if values.is_empty() {
            return Ok(None);
        }
        if values.len() != 1 {
            return unsupported(format!(
                "{}.{name} has {} lanes, expected at most one",
                self.op_name,
                values.len()
            ));
        }
        match &values[0] {
            Some(value) => Ok(Some(value.clone())),
            None => Err(Failure::Ffi(ffi_error("sunk lane in a scalar operand"))),
        }
    }

    pub fn require_cache_policy(&self, policy: Option<&ObjectRef>, enabled: bool) -> AResult<()> {
        if enabled != policy.is_some() {
            return unsupported(format!(
                "{} cache modifier and cache-policy operand disagree",
                self.op_name
            ));
        }
        if let Some(policy) = policy {
            let dtype = dtype_of(policy)?;
            if dtype != "uint64" {
                return unsupported(format!(
                    "{}.cache_policy must be uint64, got {:?}",
                    self.op_name, &dtype
                ));
            }
        }
        Ok(())
    }

    /// The `cache_policy` operand, checked against the `cache` modifier.
    pub fn cache_policy_operand(&self) -> AResult<Option<ObjectRef>> {
        let cache_policy = if self.has_operand("cache_policy") {
            self.optional_scalar_operand("cache_policy")?
        } else {
            None
        };
        let has_cache = self.modifier_or_empty("cache") == "L2::cache_hint";
        self.require_cache_policy(cache_policy.as_ref(), has_cache)?;
        Ok(cache_policy)
    }

    pub fn require_void(&self) -> AResult<()> {
        if !self.result_type.is_empty() {
            return unsupported(format!("{} must return void", self.op_name));
        }
        Ok(())
    }

    /// `require_void` naming the actual result type.
    pub fn require_void_result_type(&self) -> AResult<()> {
        if !self.result_type.is_empty() {
            return unsupported(format!(
                "{} must return void, got result type {}",
                self.op_name, self.result_type
            ));
        }
        Ok(())
    }

    pub fn require_modifiers(&self, expected: &[(&str, &str)]) -> AResult<()> {
        for (name, required) in expected {
            let actual = self.modifier(name)?;
            if actual != *required {
                return unsupported(format!(
                    "{} requires {name}={:?}, got {:?}",
                    self.op_name, required, actual
                ));
            }
        }
        Ok(())
    }

    /// Exactly one lane; a sink is returned as `None`.
    pub fn scalar_lane(&self, name: &str) -> AResult<Option<ObjectRef>> {
        let values = self.operand(name)?;
        if values.len() != 1 {
            return Err(Failure::Ffi(ffi_error(&format!(
                "{} operand {name:?} has {} lanes, expected one",
                self.op_name,
                values.len()
            ))));
        }
        Ok(values[0].clone())
    }

    /// `DecodedPtxCall.operand`.
    pub fn operand(&self, name: &str) -> AResult<&[Option<ObjectRef>]> {
        match self.operands.iter().find(|slot| slot.name == name) {
            Some(slot) => Ok(&slot.values),
            None => Err(Failure::Ffi(ffi_error(&format!(
                "{} has no operand named {name:?}",
                self.op_name
            )))),
        }
    }

    pub fn has_operand(&self, name: &str) -> bool {
        self.operands.iter().any(|slot| slot.name == name)
    }

    /// `DecodedPtxCall.scalar_operand`: exactly one present lane.
    pub fn scalar_operand(&self, name: &str) -> AResult<ObjectRef> {
        match self.scalar_lane(name)? {
            Some(value) => Ok(value),
            None => Err(Failure::Ffi(ffi_error(&format!(
                "{} operand {name:?} is a sunk lane",
                self.op_name
            )))),
        }
    }

    /// `DecodedPtxCall.modifier`: the token of one slot (`""` when omitted).
    pub fn modifier(&self, name: &str) -> AResult<&str> {
        match self.modifiers.iter().find(|(slot, _)| slot == name) {
            Some((_, token)) => Ok(token.as_str()),
            None => Err(Failure::Ffi(ffi_error(&format!(
                "{} has no modifier slot named {name:?}",
                self.op_name
            )))),
        }
    }

    /// `call.modifiers.get(name, "")`.
    pub fn modifier_or_empty(&self, name: &str) -> &str {
        self.modifiers
            .iter()
            .find(|(slot, _)| slot == name)
            .map_or("", |(_, token)| token.as_str())
    }

    /// The `cta_group::N` modifier, 1 or 2; `omitted` is the group of an
    /// instruction without one, `None` when the family requires it.
    pub fn cta_group(&self, omitted: Option<i64>) -> AResult<i64> {
        let token = self.modifier("cta_group")?;
        if let (true, Some(group)) = (token.is_empty(), omitted) {
            return Ok(group);
        }
        let Some(group) = token
            .strip_prefix("cta_group::")
            .filter(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
            .and_then(|digits| digits.parse::<i64>().ok())
        else {
            return unsupported(format!(
                "{} has malformed cta_group modifier {:?}",
                self.op_name, token
            ));
        };
        if group != 1 && group != 2 {
            return unsupported(format!(
                "{} cta_group must be 1 or 2, got {group}",
                self.op_name
            ));
        }
        Ok(group)
    }
}

/// An unmodeled PTX form whose message is prefixed with the op name.
pub fn unmodeled_form<T>(decoded: &DecodedPtx, message: &str) -> AResult<T> {
    unmodeled(
        format!("call:{}", decoded.op_name),
        format!("{} {message}", decoded.op_name),
    )
}

/// The Python decode outcome of one `tirx.ptx.*` call.
#[derive(Clone)]
pub enum PtxDecode {
    Decoded(DecodedPtx),
    /// `PtxCallDecodeError`: the rendered message and the op name it names.
    Failed {
        message: String,
        op_name: String,
    },
}

impl PtxDecode {
    /// The decoded call, or the Python decode failure as an unsupported
    /// rejection carrying `unsupported=(op_name,)`.
    pub fn decoded(&self) -> AResult<&DecodedPtx> {
        match self {
            PtxDecode::Decoded(decoded) => Ok(decoded),
            PtxDecode::Failed { message, op_name } => Err(Failure::Unsupported {
                message: message.clone(),
                unsupported: vec![op_name.clone()],
            }),
        }
    }
}

fn str_field(entry: &Json, key: &str) -> AResult<String> {
    match entry.get(key).and_then(Json::as_str) {
        Some(value) => Ok(value.to_owned()),
        None => Err(Failure::Ffi(ffi_error(&format!(
            "PTX payload entry is missing {key:?}"
        )))),
    }
}

fn bool_field(entry: &Json, key: &str) -> AResult<bool> {
    match entry.get(key).and_then(Json::as_bool) {
        Some(value) => Ok(value),
        None => Err(Failure::Ffi(ffi_error(&format!(
            "PTX payload entry is missing {key:?}"
        )))),
    }
}

fn int_field(entry: &Json, key: &str) -> AResult<i64> {
    match entry.get(key).and_then(Json::as_i64) {
        Some(value) => Ok(value),
        None => Err(Failure::Ffi(ffi_error(&format!(
            "PTX payload entry is missing {key:?}"
        )))),
    }
}

fn argument_at(node: &ObjectRef, index: i64) -> AResult<ObjectRef> {
    use tvm::tvm_ffi::ObjectRefCore;
    let Some(call) = node.as_node::<tvm::ir::CallObj>() else {
        return Err(Failure::Ffi(ffi_error("PTX payload names a non-call node")));
    };
    let Ok(position) = usize::try_from(index) else {
        return Err(Failure::Ffi(ffi_error(
            "PTX payload argument index is negative",
        )));
    };
    if position >= call.args.len() {
        return Err(Failure::Ffi(ffi_error(
            "PTX payload argument index is out of range",
        )));
    }
    Ok(crate::analyze::util::oref(call.args.get(position)?))
}

fn parse_entry(node: &ObjectRef, entry: &Json) -> AResult<PtxDecode> {
    if let Some(message) = entry.get(CallFields::ERROR).and_then(Json::as_str) {
        return Ok(PtxDecode::Failed {
            message: message.to_owned(),
            op_name: str_field(entry, CallFields::OP_NAME)?,
        });
    }
    let mut modifiers = Vec::new();
    for pair in entry
        .get(CallFields::MODIFIERS)
        .and_then(Json::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let Some(items) = pair.as_array() else {
            return Err(Failure::Ffi(ffi_error("PTX payload modifier is malformed")));
        };
        match (
            items.first().and_then(Json::as_str),
            items.get(1).and_then(Json::as_str),
        ) {
            (Some(slot), Some(token)) => modifiers.push((slot.to_owned(), token.to_owned())),
            _ => return Err(Failure::Ffi(ffi_error("PTX payload modifier is malformed"))),
        }
    }
    let mut operands = Vec::new();
    for slot in entry
        .get(CallFields::OPERANDS)
        .and_then(Json::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let mut values = Vec::new();
        for value in slot
            .get(OperandFields::VALUES)
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            match value.as_i64() {
                Some(index) if index >= 0 => values.push(Some(argument_at(node, index)?)),
                Some(_) => values.push(None),
                None => return Err(Failure::Ffi(ffi_error("PTX payload lane is malformed"))),
            }
        }
        let dtypes = slot
            .get(OperandFields::DTYPES)
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Json::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        operands.push(PtxOperand {
            name: str_field(slot, OperandFields::NAME)?,
            kind: str_field(slot, OperandFields::KIND)?,
            rw: str_field(slot, OperandFields::RW)?,
            lanes: int_field(slot, OperandFields::LANES)?,
            allow_imm_offset: bool_field(slot, OperandFields::ALLOW_IMM_OFFSET)?,
            operand_type: slot
                .get(OperandFields::OPERAND_TYPE)
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_owned(),
            dtypes,
            literal: slot
                .get(OperandFields::LITERAL)
                .and_then(Json::as_str)
                .map(str::to_owned),
            values,
        });
    }
    let predicate = match entry.get(CallFields::PREDICATE).and_then(Json::as_i64) {
        Some(index) => Some(argument_at(node, index)?),
        None => None,
    };
    Ok(PtxDecode::Decoded(DecodedPtx {
        op_name: str_field(entry, CallFields::OP_NAME)?,
        operands,
        modifiers,
        predicate,
        preserve_dst: bool_field(entry, CallFields::PRESERVE_DST)?,
        result_type: str_field(entry, CallFields::RESULT_TYPE)?,
    }))
}

/// `ptx_dialect.NATIVE_PTX_DECODER`: Python's per-call decoder.
const NATIVE_PTX_DECODER: &str = "numsim.frontend.decode_ptx_call";

/// Decode one `tirx.ptx.*` call through the Python target table.
pub fn decode_call(node: &ObjectRef) -> AResult<PtxDecode> {
    let decoder = tvm_ffi::Function::get_global(NATIVE_PTX_DECODER)?;
    let payload: Any = decoder.call_tuple((Any::from(node.clone()),))?;
    let payload = FfiString::try_from(payload)?;
    let entry = serde_json::from_str::<Json>(payload.as_str())
        .map_err(|error| Failure::Ffi(ffi_error(&format!("PTX payload is not JSON: {error}"))))?;
    parse_entry(node, &entry)
}
