//! Promote canonical host-created TensorMaps to implicit device parameters.

use tvm::tirx::TensorMapTypeObj;

use std::collections::HashSet;

use crate::analyze::buffers::{call_op_name, BufferBindings};
use crate::analyze::util::{
    as_buffer, dtype_of, ffi_text, int_imm, kind, oref, same, unsupported, AResult, Json,
};
use crate::decode::projected_buffer;
use tvm::analysis::Analyzer;
use tvm::ir::StringImmObj;
use tvm::ir::{Call, CallObj, DictAttrs, Expr, IntImm, PointerTypeObj, PrimExpr, Var};
use tvm::prim::Add;
use tvm::tirx::{
    AssertStmtObj, AttrStmtObj, BindObj, DeclBufferObj, EvaluateObj, PrimFunc, SeqStmtObj, Stmt,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, Map, ObjectRefCast, ObjectRefCore, String as FfiString};

use super::{implicit_tensor_map_metadata, scalar_parameters, IMPLICIT_TENSOR_MAP_ATTR};

struct Context<'a> {
    func: &'a PrimFunc,
    analyzer: Analyzer,
    scalars: Vec<Var>,
    tvm_version: FfiString,
}

impl Context<'_> {
    fn simplify(&self, value: &ObjectRef) -> AResult<PrimExpr> {
        Ok(self.analyzer.simplify(&value.clone().try_cast()?)?)
    }

    fn serialize_integer(&self, value: &ObjectRef, field: &str) -> AResult<Json> {
        let value = oref(self.simplify(value)?);
        crate::emit::validate_integer_expression(
            &value,
            &|variable| {
                self.scalars
                    .iter()
                    .any(|parameter| same(variable, parameter))
            },
            &format!("host TensorMap {field}"),
        )?;
        Ok(Json::from(ffi_text(
            &crate::serialization::semantic_ir_json(value, self.tvm_version.clone())?,
        )))
    }

    fn static_int(&self, value: &Expr, field: &str) -> AResult<i64> {
        let simplified = oref(self.simplify(&oref(value.clone()))?);
        match int_imm(&simplified) {
            Some(value) => Ok(value),
            None => unsupported(format!(
                "host TensorMap {field} must be a static integer for NumSim, got {}",
                kind(&simplified).unwrap_or("unknown node")
            )),
        }
    }

    fn enum_value(&self, table: &[Option<&str>], value: &Expr, field: &str) -> AResult<Json> {
        let raw = self.static_int(value, field)?;
        match usize::try_from(raw).ok().and_then(|index| table.get(index)) {
            Some(Some(value)) => Ok(Json::from(*value)),
            Some(None) => Ok(Json::Null),
            None => unsupported(format!(
                "host TensorMap {field} enum value {raw} is not supported by NumSim"
            )),
        }
    }

    fn tensor_map_alloca(&self, statement: &Stmt) -> AResult<Option<Var>> {
        let Some(bind) = statement.as_node::<BindObj>() else {
            return Ok(None);
        };
        let Some(call) = bind.value.as_node::<CallObj>() else {
            return Ok(None);
        };
        if call_op_name(call)?.as_deref() != Some("tirx.tvm_stack_alloca")
            || call
                .args
                .get(0)
                .ok()
                .and_then(|value| string_imm(&value))
                .as_deref()
                != Some("tensormap")
        {
            return Ok(None);
        }
        if call.args.len() != 2
            || self.static_int(
                &call.args.get(1).expect("allocation count"),
                "allocation count",
            )? != 1
        {
            return unsupported(
                "NumSim only supports canonical single-descriptor TensorMap stack allocations",
            );
        }
        if !bind
            .var
            .ty
            .as_node::<PointerTypeObj>()
            .is_some_and(|pointer| pointer.element_type.as_node::<TensorMapTypeObj>().is_some())
        {
            return unsupported(
                "host TensorMap allocation must bind a variable with TensorMap type",
            );
        }
        Ok(Some(bind.var.clone()))
    }

    fn base_buffer_and_offset(
        &self,
        bindings: &BufferBindings,
        mut expression: ObjectRef,
    ) -> AResult<(String, Json)> {
        let mut offsets: Vec<Expr> = Vec::new();
        let buffer = loop {
            if let Some(call) = expression.as_node::<CallObj>() {
                if call_op_name(call)?.as_deref() == Some("tirx.handle_add_byte_offset")
                    && call.args.len() == 2
                {
                    offsets.push(call.args.get(1).expect("byte offset"));
                    expression = oref(call.args.get(0).expect("base pointer"));
                    continue;
                }
            }
            let Some(buffer) = projected_buffer(&expression)? else {
                return unsupported(
                    "host TensorMap global address must derive from a typed-buffer data projection",
                );
            };
            let root = bindings.storage_key(&buffer)?;
            if let Some(buffer) = as_buffer(&root) {
                break buffer;
            }
            expression = root;
        };
        let mut offsets = offsets.into_iter();
        let byte_offset = match offsets.next() {
            Some(mut value) => {
                for offset in offsets {
                    value = Add::new(value, offset)?.into();
                }
                oref(value)
            }
            None => oref(IntImm::new("int64", 0)?),
        };
        let serialized = self.serialize_integer(&byte_offset, "global address byte offset")?;
        if self
            .func
            .params
            .iter()
            .any(|parameter| same(&parameter, buffer.as_var()))
        {
            Ok((ffi_text(&buffer.name), serialized))
        } else {
            unsupported(format!(
                "host TensorMap base buffer {:?} is not a PrimFunc parameter",
                ffi_text(&buffer.name)
            ))
        }
    }

    fn parse_encode(
        &self,
        bindings: &BufferBindings,
        call: &Call,
        variable: &Var,
    ) -> AResult<Json> {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 4 || !same(&args[1], variable) {
            return unsupported(format!(
                "host TensorMap encode does not target allocation {:?}",
                ffi_text(&variable.name)
            ));
        }
        let Some(dtype) = string_imm(&args[2]) else {
            return unsupported("host TensorMap dtype must be a static dtype string");
        };
        let rank = self.static_int(&args[3], "rank")?;
        if !(1..=5).contains(&rank) {
            return unsupported(format!("host TensorMap rank must be in 1..5, got {rank}"));
        }
        let rank = rank as usize;
        let expected = 8 + 4 * rank;
        if args.len() != expected && args.len() != expected + 1 {
            return unsupported(format!(
                "host TensorMap rank-{rank} encode has {} arguments; expected {expected} or {}",
                args.len(),
                expected + 1
            ));
        }
        let mut cursor = 5;
        let mut take = |count: usize, field: &str| -> AResult<Json> {
            let result = (0..count)
                .map(|axis| {
                    self.serialize_integer(
                        &oref(args[cursor + axis].clone()),
                        &format!("{field}[{axis}]"),
                    )
                })
                .collect::<AResult<Vec<_>>>()?;
            cursor += count;
            Ok(Json::Array(result))
        };
        let global_shape = take(rank, "global_shape")?;
        let global_strides = take(rank - 1, "global_strides")?;
        let box_shape = take(rank, "box_shape")?;
        let element_strides = take(rank, "element_strides")?;
        let interleave = self.enum_value(
            &[None, Some("16B"), Some("32B")],
            &args[cursor],
            "interleave",
        )?;
        let swizzle = self.enum_value(
            &[None, Some("32B"), Some("64B"), Some("128B")],
            &args[cursor + 1],
            "swizzle",
        )?;
        let l2_promotion = self.enum_value(
            &[None, Some("64B"), Some("128B"), Some("256B")],
            &args[cursor + 2],
            "l2_promotion",
        )?;
        let fill_mode =
            self.enum_value(&[Some("zero"), Some("nan")], &args[cursor + 3], "fill_mode")?;
        cursor += 4;
        let mut tma_dtype = Json::Null;
        let mut fp4_shared_layout = Json::Null;
        if cursor < args.len() {
            match (self.static_int(&args[cursor], "force_cu_dtype")?, dtype.as_str()) {
                (-1, _) => {},
                (11, "float32") => tma_dtype = Json::from("tf32"),
                (13, "float4_e2m1fn") => fp4_shared_layout = Json::from("align8_packed"),
                (14, "float4_e2m1fn") => fp4_shared_layout = Json::from("align16_padded"),
                _ => return unsupported("host TensorMap force_cu_dtype must be -1, TFLOAT32 (11) for float32, or 16U4_ALIGN8B/ALIGN16B (13/14) for float4_e2m1fn"),
            }
        }
        let (base_buffer, base_byte_offset) =
            self.base_buffer_and_offset(bindings, oref(args[4].clone()))?;
        Ok(serde_json::json!({
            "name": ffi_text(&variable.name), "base_buffer": base_buffer,
            "base_byte_offset": base_byte_offset, "dtype": dtype,
            "tma_dtype": tma_dtype, "fp4_shared_layout": fp4_shared_layout,
            "global_shape": global_shape, "global_strides": global_strides,
            "box_shape": box_shape, "element_strides": element_strides,
            "interleave": interleave, "swizzle": swizzle,
            "l2_promotion": l2_promotion, "fill_mode": fill_mode,
        }))
    }
}

fn string_imm(value: &Expr) -> Option<String> {
    value
        .as_node::<StringImmObj>()
        .map(|value| ffi_text(&value.value))
}

fn is_device_entry(statement: &Stmt) -> bool {
    statement.as_node::<AttrStmtObj>().is_some_and(|attribute| {
        matches!(
            attribute.attr_key.as_str(),
            "tirx.device_entry" | "thread_extent"
        )
    })
}

fn encode_call(statement: &Stmt) -> AResult<Option<Call>> {
    let Some(evaluate) = statement.as_node::<EvaluateObj>() else {
        return Ok(None);
    };
    let Ok(call) = evaluate.value.clone().try_cast::<Call>() else {
        return Ok(None);
    };
    if call_op_name(&call)?.as_deref() == Some("tirx.tvm_call_packed")
        && call
            .args
            .get(0)
            .ok()
            .and_then(|value| string_imm(&value))
            .as_deref()
            == Some("runtime.cuTensorMapEncodeTiled")
    {
        Ok(Some(call))
    } else {
        Ok(None)
    }
}

pub fn normalize(func: PrimFunc, tvm_version: FfiString) -> AResult<PrimFunc> {
    if func
        .attrs
        .dict
        .get(&FfiString::from(IMPLICIT_TENSOR_MAP_ATTR))?
        .is_some()
    {
        implicit_tensor_map_metadata(&func)?;
        return Ok(func);
    }
    let body = func.body().clone();
    let statements: Vec<Stmt> = match body.as_node::<SeqStmtObj>() {
        Some(sequence) => sequence.seq.iter().collect(),
        None => vec![body],
    };
    let entries: Vec<_> = statements
        .iter()
        .enumerate()
        .filter(|(_, statement)| is_device_entry(statement))
        .map(|(index, _)| index)
        .collect();
    if entries.is_empty() {
        return Ok(func);
    }
    if entries.len() != 1 {
        return unsupported("NumSim requires exactly one top-level tirx.device_entry");
    }
    let entry_index = entries[0];
    if entry_index == 0 {
        return Ok(func);
    }
    let context = Context {
        func: &func,
        analyzer: Analyzer::new()?,
        scalars: scalar_parameters(&func),
        tvm_version,
    };
    let classified = statements[..entry_index]
        .iter()
        .map(|statement| {
            Ok((
                context.tensor_map_alloca(statement)?,
                encode_call(statement)?,
            ))
        })
        .collect::<AResult<Vec<_>>>()?;
    if classified
        .iter()
        .all(|(variable, call)| variable.is_none() && call.is_none())
    {
        return Ok(func);
    }
    if entry_index + 1 != statements.len() {
        return unsupported(
            "NumSim does not support host statements after the top-level tirx.device_entry",
        );
    }
    let mut allocations: Vec<Var> = Vec::new();
    let mut encodes: Vec<Option<Call>> = Vec::new();
    let mut substitutions: Vec<(Var, Expr)> = Vec::new();
    for (statement, (variable, call)) in statements.iter().zip(classified) {
        if let Some(variable) = variable {
            if allocations
                .iter()
                .any(|candidate| same(&variable, candidate))
            {
                return unsupported(format!(
                    "host TensorMap {:?} is allocated more than once",
                    ffi_text(&variable.name)
                ));
            }
            allocations.push(variable);
            encodes.push(None);
            continue;
        }
        let Some(call) = call else {
            let replacements = substitutions.iter().cloned().collect();
            if let Some(assertion) = statement.as_node::<AssertStmtObj>() {
                let condition = crate::substitute(oref(assertion.condition.clone()), replacements)?;
                if int_imm(&oref(context.simplify(&condition)?)).is_none_or(|value| value == 0) {
                    return unsupported(
                        "host assertion before tirx.device_entry must simplify to true",
                    );
                }
            } else if statement.as_node::<DeclBufferObj>().is_some() {
                // The complete buffer binding table below keeps these aliases.
            } else if let Some(binding) = statement.as_node::<BindObj>() {
                let value = crate::substitute(oref(binding.value.clone()), replacements)?;
                if crate::post_order_nodes(value.clone())?
                    .iter()
                    .any(|node| matches!(kind(&node), Some("Call" | "TensorLoad")))
                {
                    return unsupported(
                        "host scalar binding before tirx.device_entry must be a pure expression",
                    );
                }
                let dtype = dtype_of(&value).unwrap_or_default();
                if !(dtype.starts_with("int") || dtype.starts_with("uint")) {
                    return unsupported(format!("host TensorMap binding {:?} must be an integer expression, got dtype {dtype:?}", ffi_text(&binding.var.name)));
                }
                // Simultaneous substitution uses the last binding for an identity.
                substitutions.retain(|(variable, _)| !same(variable, &binding.var));
                substitutions.push((binding.var.clone(), context.simplify(&value)?.into()));
            } else {
                return unsupported(format!(
                    "unsupported host statement before tirx.device_entry: {}",
                    kind(&oref(statement.clone())).unwrap_or("unknown node")
                ));
            }
            continue;
        };
        let Some(index) = allocations.iter().position(|candidate| {
            call.args
                .get(1)
                .is_ok_and(|target| same(&target, candidate))
        }) else {
            return unsupported("host TensorMap encode targets no preceding allocation");
        };
        if encodes[index].is_some() {
            return unsupported(format!(
                "host TensorMap {:?} is encoded more than once",
                ffi_text(&allocations[index].name)
            ));
        }
        encodes[index] = Some(call);
    }
    if allocations.is_empty() {
        return unsupported(
            "host statements before tirx.device_entry contain no TensorMap allocation",
        );
    }
    let missing: Vec<_> = allocations
        .iter()
        .zip(&encodes)
        .filter(|(_, call)| call.is_none())
        .map(|(variable, _)| ffi_text(&variable.name))
        .collect();
    if !missing.is_empty() {
        return unsupported(format!("host TensorMaps are not encoded: {missing:?}"));
    }
    let bindings = BufferBindings::build(
        &crate::walk_statements(func.body().clone())?
            .iter()
            .collect::<Vec<_>>(),
    )?;
    let substitutions: Map<Var, Expr> = substitutions.into_iter().collect();
    let mut metadata = allocations
        .iter()
        .zip(encodes)
        .map(|(variable, call)| {
            let call = crate::substitute(
                oref(call.expect("encoded allocation")),
                substitutions.clone(),
            )?
            .try_cast()?;
            context.parse_encode(&bindings, &call, variable)
        })
        .collect::<AResult<Vec<_>>>()?;
    let mut used: HashSet<String> = func
        .params
        .iter()
        .map(|parameter| ffi_text(&parameter.name))
        .collect();
    let mut params: Vec<Var> = func.params.iter().collect();
    let mut renames: Vec<(Var, Expr)> = Vec::new();
    for (variable, item) in allocations.iter().zip(&mut metadata) {
        let original = ffi_text(&variable.name);
        let mut name = original.clone();
        if used.contains(&name)
            || allocations
                .iter()
                .filter(|candidate| candidate.name == variable.name)
                .count()
                != 1
        {
            let base = format!("{}_tmap", item["base_buffer"].as_str().expect("base name"));
            name = base.clone();
            let mut suffix = 1;
            while used.contains(&name) {
                name = format!("{base}_{suffix}");
                suffix += 1;
            }
        }
        used.insert(name.clone());
        item["name"] = Json::from(name.clone());
        let normalized = if name == original {
            variable.clone()
        } else {
            let renamed = variable.copy_with(FfiString::from(name), variable.ty.clone());
            renames.push((variable.clone(), renamed.clone().into()));
            renamed
        };
        params.push(normalized);
    }
    let entry = crate::substitute(
        crate::substitute(oref(statements[entry_index].clone()), substitutions)?,
        renames.into_iter().collect(),
    )?;
    let mut attrs: Vec<(FfiString, Any)> = func.attrs.dict.iter().collect();
    attrs.push((
        FfiString::from(IMPLICIT_TENSOR_MAP_ATTR),
        FfiString::from(Json::Array(metadata).to_string()).into(),
    ));
    Ok(PrimFunc::with_metadata(
        params,
        entry.try_cast::<Stmt>()?,
        func.ret_type.clone(),
        DictAttrs::from_dictionary(attrs.into_iter().collect()),
        func.span.as_ref(),
    )?)
}
