//! Validation and emission support for register_call.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use std::borrow::Cow;
use tvm::ir::TensorLoadObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

pub const PREDICATE_CARRIER_DTYPES: [&str; 3] = ["bool", "int32", "uint32"];

pub const PTX_TYPE_MARKERS: &[(&str, &str)] = &[
    ("b16", "B16"),
    ("b32", "B32"),
    ("b64", "B64"),
    ("b128", "B128"),
    ("u8", "U8"),
    ("u16", "U16"),
    ("u32", "U32"),
    ("u64", "U64"),
    ("s8", "I8"),
    ("s16", "I16"),
    ("s32", "I32"),
    ("s64", "I64"),
    ("s16x2", "I16x2"),
    ("u16x2", "U16x2"),
    ("f16", "F16"),
    ("bf16", "Bf16"),
    ("f16x2", "F16x2"),
    ("bf16x2", "Bf16x2"),
    ("f32", "F32"),
    ("f64", "F64"),
    ("tf32", "Tf32"),
    ("pred", "Pred"),
];

pub const PTX_COMPARE_MARKERS: &[(&str, &str)] = &[
    ("eq", "Eq"),
    ("ne", "Ne"),
    ("lt", "Lt"),
    ("le", "Le"),
    ("gt", "Gt"),
    ("ge", "Ge"),
    // PTX's unsigned spellings are aliases for the same relation.
    ("lo", "Lt"),
    ("ls", "Le"),
    ("hi", "Gt"),
    ("hs", "Ge"),
    ("equ", "Equ"),
    ("neu", "Neu"),
    ("ltu", "Ltu"),
    ("leu", "Leu"),
    ("gtu", "Gtu"),
    ("geu", "Geu"),
    ("num", "Num"),
    ("nan", "Nan"),
];

pub const PTX_BOOL_MARKERS: &[(&str, &str)] =
    &[("and", "BoolAnd"), ("or", "BoolOr"), ("xor", "BoolXor")];

/// `TABLE[token]` / `TABLE.get(token)` on one of the marker tables above.
pub fn table_marker(table: &[(&str, &'static str)], token: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(key, _)| *key == token)
        .map(|(_, marker)| *marker)
}

/// `RegisterOperands`: schema-validated register lanes, in PTX operand order.
pub struct RegisterOperands {
    pub destinations: Vec<ObjectRef>,
    pub sources: Vec<ObjectRef>,
}

/// A validated, single-result Engine register call. Operands keep their TIRx
/// handles; the signature is shared by validation and emission.
pub struct RegisterCall {
    pub destination: ObjectRef,
    pub sources: Vec<ObjectRef>,
    pub instruction: &'static str,
    pub variant: String,
    pub argument_types: Vec<Cow<'static, str>>,
    pub result_type: Cow<'static, str>,
    /// `(prefix, invalid_message)` of the family's shadowing instruction
    /// predicate region; families without one reject predication when resolving.
    pub predicate_region: Option<(&'static str, String)>,
}

impl RegisterCall {
    pub fn new<T: Into<Cow<'static, str>>>(
        destination: ObjectRef,
        sources: Vec<ObjectRef>,
        instruction: &'static str,
        variant: impl Into<String>,
        argument_types: Vec<T>,
        result_type: T,
    ) -> Self {
        Self {
            destination,
            sources,
            instruction,
            variant: variant.into(),
            argument_types: argument_types.into_iter().map(Into::into).collect(),
            result_type: result_type.into(),
            predicate_region: None,
        }
    }

    /// Gate operand evaluation and writeback with the family's
    /// shadowing predicated region (`open_shadow_predicated_region`).
    pub fn with_predicate_region(self, prefix: &'static str, invalid_message: String) -> Self {
        Self {
            predicate_region: Some((prefix, invalid_message)),
            ..self
        }
    }
}

/// `require_register_call`: validate every register operand of a decoded call.
pub fn require_register_call(decoded: &DecodedPtx, allow_sinks: bool) -> AResult<RegisterOperands> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let mut destinations = Vec::new();
    let mut sources = Vec::new();
    for slot in &decoded.operands {
        if slot.kind != "reg" {
            continue;
        }
        if slot.values.len() as i64 != slot.lanes {
            return unsupported(format!(
                "{op_name}.{} expects {} register lanes, got {}",
                slot.name,
                slot.lanes,
                slot.values.len()
            ));
        }
        let is_predicate = slot.operand_type == "pred";
        let mut lane_dtypes: Vec<String> = Vec::new();
        for value in slot.values.iter().flatten() {
            let dtype = dtype_of(value)?;
            if !lane_dtypes.contains(&dtype) {
                lane_dtypes.push(dtype);
            }
        }
        if lane_dtypes.len() > 1 {
            lane_dtypes.sort();
            return unsupported(format!(
                "{op_name}.{} must use one homogeneous register carrier dtype, got {:?}",
                slot.name, &lane_dtypes
            ));
        }
        for value in &slot.values {
            let Some(value) = value else {
                if !allow_sinks || !(slot.rw == "w" || slot.rw == "rw") {
                    return unsupported(format!(
                        "{op_name}.{} uses an unmodeled sink lane",
                        slot.name
                    ));
                }
                continue;
            };
            let actual = dtype_of(value)?;
            let accepted: Vec<&str> = if is_predicate && slot.rw == "r" {
                PREDICATE_CARRIER_DTYPES.to_vec()
            } else {
                slot.dtypes.iter().map(String::as_str).collect()
            };
            if !accepted.contains(&actual.as_str()) {
                let mut sorted: Vec<&str> = accepted.clone();
                sorted.sort_unstable();
                sorted.dedup();
                return unsupported(format!(
                    "{op_name}.{} expects one of {:?}, got {actual}",
                    slot.name, &sorted
                ));
            }
            if slot.rw == "w" || slot.rw == "rw" {
                if value.as_node::<TensorLoadObj>().is_none() {
                    return unsupported(format!(
                        "{op_name}.{} must be a TensorLoad lvalue",
                        slot.name
                    ));
                }
                destinations.push(value.clone());
            }
            if slot.rw == "r" || slot.rw == "rw" {
                sources.push(value.clone());
            }
            if !matches!(slot.rw.as_str(), "r" | "w" | "rw") {
                return unsupported(format!(
                    "{op_name}.{} has unsupported direction {:?}",
                    slot.name, &slot.rw
                ));
            }
        }
    }
    Ok(RegisterOperands {
        destinations,
        sources,
    })
}

/// `marker`.
/// The engine function (below `v2::`) of one register instruction.
pub fn register_function(instruction: &str) -> String {
    format!("reg::{instruction}")
}

pub fn marker(name: &str) -> String {
    format!("v2::reg::variant::{name}")
}

/// `generic_marker`.
pub fn generic_marker(name: &str, arguments: &[&str]) -> String {
    let rendered: Vec<String> = arguments.iter().map(|argument| marker(argument)).collect();
    format!("v2::reg::variant::{name}<{}>", rendered.join(", "))
}

pub fn bit_carrier_dtypes(width: i64) -> &'static [&'static str] {
    match width {
        8 => &["int8", "uint8"],
        16 => &["bfloat16", "float16", "int16", "uint16"],
        32 => &["float32", "int32", "uint32"],
        64 => &["float64", "int64", "uint64"],
        128 => &["int128", "uint128"],
        _ => &[],
    }
}

impl<'a> Emitter<'a> {
    /// The Engine ABI for a single register result. Families retain operand
    /// conversion, predication and destination-write policy around this call.
    pub(super) fn emit_register_result(
        &mut self,
        function: &str,
        variant: &str,
        operands: &[RustValue],
        source_op_id: i64,
        context: Option<&str>,
    ) -> String {
        let arguments = operands
            .iter()
            .map(|value| abi::register(&value.code))
            .collect::<Vec<_>>()
            .join(", ");
        let instruction = function
            .strip_prefix("reg::")
            .expect("register ABI function");
        let raw = self.control_name(&format!("decoded_{instruction}_raw"));
        let result = self.control_name(&format!("decoded_{instruction}"));
        let site = self.v2_site(Some(source_op_id));
        let call = abi::lane_call_context(
            function,
            &site,
            &[if operands.len() == 1 {
                arguments
            } else {
                format!("({arguments})")
            }],
            Some(variant),
            context,
        );
        self.emit_line(&format!("let {raw} = {call};"));
        self.emit_line(&format!("let {result} = v2_register_out({raw});"));
        result
    }

    /// The family's register call inside its instruction-predicate region,
    /// followed by `finish_predicated_destinations`.
    pub fn emit_register_call(
        &mut self,
        decoded: &DecodedPtx,
        form: &RegisterCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let Some((prefix, invalid_message)) = &form.predicate_region else {
            return self.emit_register_call_body(decoded, form, source_op_id);
        };
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            prefix,
            invalid_message,
        )?;
        self.emit_register_call_body(decoded, form, source_op_id)?;
        let mask = region.mask.clone();
        self.close_predicated_region(region);
        let result_dtype = dtype_of(&form.destination)?;
        self.finish_predicated_destinations(
            decoded,
            &[Some(form.destination.clone())],
            &result_dtype,
            &mask,
            source_op_id,
            false,
        )
    }

    fn emit_register_call_body(
        &mut self,
        decoded: &DecodedPtx,
        form: &RegisterCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let mut operands = Vec::new();
        for expression in &form.sources {
            let value = self.emit_expr(expression)?;
            operands.push(self.as_warp_value(value));
        }
        let actual_types: Vec<&str> = operands
            .iter()
            .map(|value| value.rust_type.as_str())
            .collect();
        let expected_types: Vec<&str> = form.argument_types.iter().map(|ty| ty.as_ref()).collect();
        if actual_types != expected_types {
            return unsupported(format!(
                "{} operands lowered to {:?}, expected {:?}",
                decoded.op_name, actual_types, expected_types,
            ));
        }
        let result = self.emit_register_result(
            &register_function(form.instruction),
            &form.variant,
            &operands,
            source_op_id,
            None,
        );
        self.emit_explicit_buffer_store(
            &form.destination,
            RustValue::new(result, form.result_type.clone(), Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }
}
