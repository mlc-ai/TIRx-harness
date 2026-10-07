//! A declared synchronization-word wait, using the raw load's decoded operands.

use crate::tvm_compat::{int_bits, int_value};
use tvm::ir::{CallObj, IntImm, IntImmObj, PointerTypeObj, PrimExpr, TensorLoadObj};
use tvm::prim::{
    AndObj, CastObj, EQObj, GEObj, GTObj, LEObj, LTObj, NEObj, NotObj, OrObj, SelectObj,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{self, Any, Array, ObjectRefCore, String as FfiString};

use super::memory_support::{v2_memory_space_rust, v2_memory_type_rust_ptx};
use super::raw_memory::{decoded_load_parts, decoded_load_semantics};
use super::{abi, Emitter, RustValue, Uniformity};
use crate::analyze::buffers::call_op_name;
use crate::analyze::util::{
    as_buffer, buffer_name, buffer_scope, dtype_of, dtype_text, expr_type, kind_or_bail, oref,
    prim_dtype, same, static_int, static_string, unsupported, AResult,
};
use crate::decode::ptx::{decode_call, DecodedPtx};
use crate::decode::Decoded;
use crate::tables::{expr_rust_type, json_string, render_bitwise, render_integer_binary};

fn scalar_suffix(dtype: &str) -> Option<&'static str> {
    match dtype {
        "int32" => Some("s32"),
        "uint32" => Some("u32"),
        "int64" => Some("s64"),
        "uint64" => Some("u64"),
        "int128" | "uint128" => Some("b128"),
        _ => None,
    }
}

fn width(suffix: &str) -> Option<u32> {
    match suffix {
        "b32" | "s32" | "u32" => Some(32),
        "b64" | "s64" | "u64" => Some(64),
        "b128" => Some(128),
        _ => None,
    }
}

fn word_suffix(pointer: &ObjectRef, requested: &ObjectRef, name: &str) -> AResult<String> {
    let ty = expr_type(pointer);
    let Some(pointer) = ty.as_ref().and_then(|ty| ty.as_node::<PointerTypeObj>()) else {
        return unsupported(format!("{name} ptr must point at the synchronization word"));
    };
    let dtype = prim_dtype(&pointer.element_type)
        .map(dtype_text)
        .unwrap_or_default();
    let spelling = static_string(requested, &format!("{name} ptx_type"))?;
    if let Some(default) = scalar_suffix(&dtype) {
        if spelling.is_empty() {
            return Ok(default.to_owned());
        }
        if spelling != "b128" && width(&spelling) != width(default) {
            return unsupported(format!(
                "{name} ptx_type {spelling:?} does not spell a {}-bit word",
                width(default).unwrap()
            ));
        }
    } else if spelling.is_empty() {
        return unsupported(format!(
            "{name} ptr is an untyped address; pass ptx_type to say how wide the word is"
        ));
    }
    if width(&spelling).is_none() {
        return unsupported(format!("{name} has an invalid ptx_type {spelling:?}"));
    }
    Ok(spelling)
}

fn raw_load(spelling: &str, operands: &[ObjectRef]) -> AResult<DecodedPtx> {
    let builder = tvm_ffi::Function::get_global("numsim.frontend.build_ptx_call")?;
    let result: Any =
        builder.call_tuple((FfiString::from(spelling), Array::new(operands.to_vec())))?;
    let result = Array::<Any>::try_from(result)?;
    let error = FfiString::try_from(result.get(1)?)?;
    if !error.as_str().is_empty() {
        return unsupported(format!("{spelling}: {}", error.as_str()));
    }
    let node = ObjectRef::try_from(result.get(0)?)?;
    Ok(decode_call(&node)?.decoded()?.clone())
}

struct FrozenBuffer {
    source: tvm::tirx::BufferVar,
    shape: Vec<i64>,
    values: Vec<Option<String>>,
}

/// Compile a wait predicate into a pure test over the candidate word.
///
/// Every value except the synchronization word is captured before the wait.
/// Candidate-indexed local tables are small, immutable snapshots; the engine
/// can therefore replay the same predicate against the word's history without
/// reaching back into mutable kernel state.
struct Predicate<'a, 'b> {
    emitter: &'a mut Emitter<'b>,
    destination: ObjectRef,
    frozen_values: Vec<(ObjectRef, String)>,
    frozen_buffers: Vec<FrozenBuffer>,
}

impl<'a, 'b> Predicate<'a, 'b> {
    fn new(emitter: &'a mut Emitter<'b>, destination: ObjectRef) -> Self {
        Self {
            emitter,
            destination,
            frozen_values: Vec::new(),
            frozen_buffers: Vec::new(),
        }
    }

    fn deep_equal(left: &ObjectRef, right: &ObjectRef) -> AResult<bool> {
        Ok(tvm_ffi::cached_global_func!("prim.expr_deep_equal")
            .call_tuple((Any::from(left.clone()), Any::from(right.clone())))?
            .try_into()?)
    }

    fn lane_value(value: &RustValue, name: &str) -> String {
        if value.is_mask {
            format!("{name}.contains(lane)")
        } else if value.uniformity == Uniformity::Uniform {
            name.to_owned()
        } else {
            format!("{name}[lane]")
        }
    }

    fn freeze(&mut self, node: &ObjectRef) -> AResult<String> {
        for (previous, rendered) in &self.frozen_values {
            if Self::deep_equal(node, previous)? {
                return Ok(rendered.clone());
            }
        }
        let value = self.emitter.emit_expr(node)?;
        let frozen = self.emitter.control_name("declared_wait_capture");
        self.emitter
            .emit_line(&format!("let {frozen} = ({}).clone();", value.code));
        let rendered = Self::lane_value(&value, &frozen);
        self.frozen_values.push((node.clone(), rendered.clone()));
        Ok(rendered)
    }

    fn depends_on_destination(&self, node: &ObjectRef) -> AResult<bool> {
        let destination = self
            .destination
            .as_node::<TensorLoadObj>()
            .expect("wait destination was validated as TensorLoad");
        for candidate in crate::post_order_nodes(node.clone())?.iter() {
            if let Some(load) = candidate.as_node::<TensorLoadObj>() {
                if same(&load.source, &destination.source) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn destination_coordinates(&self, source: &tvm::tirx::BufferVar) -> AResult<Option<Vec<i64>>> {
        let destination = self
            .destination
            .as_node::<TensorLoadObj>()
            .expect("wait destination was validated as TensorLoad");
        if !same(source.as_var(), &destination.source) {
            return Ok(None);
        }
        let mut coordinates = Vec::new();
        for index in destination.indices.iter() {
            let value = static_int(
                &self.emitter.ctx.analyzer,
                &index,
                "wait_until destination index",
                "must be static when its buffer is indexed by the candidate word",
            )?;
            coordinates.push(value);
        }
        Ok(Some(coordinates))
    }

    fn capture_buffer(
        &mut self,
        source: &tvm::tirx::BufferVar,
    ) -> AResult<(Vec<i64>, Vec<Option<String>>)> {
        if let Some(captured) = self
            .frozen_buffers
            .iter()
            .find(|captured| same(captured.source.as_var(), source.as_var()))
        {
            return Ok((captured.shape.clone(), captured.values.clone()));
        }

        let mut shape = Vec::new();
        let mut element_count = 1_i64;
        for dimension in source.buffer_type().shape.iter() {
            let extent = static_int(
                &self.emitter.ctx.analyzer,
                &dimension,
                "candidate-indexed wait predicate local-buffer shape",
                "must be static",
            )?;
            if extent < 0 {
                return unsupported(
                    "a candidate-indexed wait predicate requires a non-negative local-buffer shape",
                );
            }
            element_count = element_count.checked_mul(extent).ok_or_else(|| {
                crate::analyze::util::Failure::Unsupported {
                    message: "a candidate-indexed wait predicate local-buffer shape is too large"
                        .to_owned(),
                    unsupported: Vec::new(),
                }
            })?;
            shape.push(extent);
        }
        let destination = self.destination_coordinates(source)?;
        let capacity = usize::try_from(element_count).map_err(|_| {
            crate::analyze::util::Failure::Unsupported {
                message: "a candidate-indexed wait predicate local-buffer shape exceeds the host address space"
                    .to_owned(),
                unsupported: Vec::new(),
            }
        })?;
        let mut values = Vec::with_capacity(capacity);
        for offset in 0..element_count {
            let mut remainder = offset;
            let mut coordinates = vec![0_i64; shape.len()];
            for (coordinate, extent) in coordinates.iter_mut().zip(shape.iter()).rev() {
                *coordinate = remainder % *extent;
                remainder /= *extent;
            }
            if destination.as_ref() == Some(&coordinates) {
                values.push(None);
                continue;
            }
            let indices: Vec<PrimExpr> = coordinates
                .iter()
                .map(|coordinate| IntImm::new("int32", *coordinate).map(Into::into))
                .collect::<tvm_ffi::Result<_>>()?;
            let value = self.emitter.emit_buffer_load(
                source,
                &indices,
                None,
                self.emitter.register_access_mask.clone(),
                false,
                None,
                None,
                None,
            )?;
            let frozen = self.emitter.control_name("declared_wait_buffer");
            self.emitter
                .emit_line(&format!("let {frozen} = ({}).clone();", value.code));
            values.push(Some(Self::lane_value(&value, &frozen)));
        }
        self.frozen_buffers.push(FrozenBuffer {
            source: source.clone(),
            shape: shape.clone(),
            values: values.clone(),
        });
        Ok((shape, values))
    }

    fn flat_index(&mut self, indices: &[PrimExpr], shape: &[i64]) -> AResult<String> {
        if indices.len() != shape.len() {
            return unsupported(
                "a candidate-indexed wait predicate has an invalid local-buffer rank",
            );
        }
        let mut flattened = "0_i64".to_owned();
        for (index, extent) in indices.iter().zip(shape) {
            flattened = format!(
                "({flattened}).wrapping_mul({extent}_i64).wrapping_add(({}) as i64)",
                self.emit(&oref(index.clone()))?
            );
        }
        Ok(flattened)
    }

    fn dynamic_local_load(&mut self, load: &TensorLoadObj, rust: &str) -> AResult<String> {
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return unsupported("a candidate-indexed wait predicate requires a typed buffer");
        };
        if !matches!(
            buffer_scope(&source).as_str(),
            "local" | "local_scalar" | "register" | "reg"
        ) {
            return unsupported(
                "a candidate-indexed wait predicate may only read thread-local buffers",
            );
        }
        let (shape, values) = self.capture_buffer(&source)?;
        let indices: Vec<PrimExpr> = load.indices.iter().collect();
        let index = self.flat_index(&indices, &shape)?;
        let arms = values
            .iter()
            .enumerate()
            .map(|(offset, value)| {
                let value = value
                    .clone()
                    .unwrap_or_else(|| format!("candidate as {rust}"));
                format!("{offset}_i64 => {value},")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let error = json_string(&format!(
            "wait_until predicate index is outside local buffer {}",
            buffer_name(&source)
        ));
        let index_name = self.emitter.control_name("declared_wait_index");
        Ok(format!(
            "({{ let {index_name} = {index}; match {index_name} {{ {arms} _ => return Err(EngineError::message({error})), }} }})"
        ))
    }

    fn emit(&mut self, node: &ObjectRef) -> AResult<String> {
        let dtype = dtype_of(node)?;
        let rust = expr_rust_type(self.emitter.ctx.schema, &dtype)?;
        if let Some(load) = node.as_node::<TensorLoadObj>() {
            if Self::deep_equal(node, &self.destination)? {
                return Ok(format!("(candidate as {rust})"));
            }
            let destination = self
                .destination
                .as_node::<TensorLoadObj>()
                .expect("wait destination was validated as TensorLoad");
            let candidate_indexed = same(&load.source, &destination.source)
                || load
                    .indices
                    .iter()
                    .map(oref)
                    .try_fold(false, |found, index| {
                        Ok::<_, crate::analyze::util::Failure>(
                            found || self.depends_on_destination(&index)?,
                        )
                    })?;
            if candidate_indexed {
                return self.dynamic_local_load(load, &rust);
            }
        }
        if let Some(value) = node.as_node::<IntImmObj>() {
            let bits = int_bits(value)?;
            return Ok(if dtype == "bool" {
                (bits != 0).to_string()
            } else {
                let literal = if dtype.starts_with("int") {
                    int_value(value)?.to_string()
                } else {
                    bits.to_string()
                };
                format!("({literal}_{rust})")
            });
        }
        let kind = kind_or_bail(node)?;
        if matches!(kind, "Var" | "TensorLoad") {
            return self.freeze(node);
        }
        if let Some(cast) = node.as_node::<CastObj>() {
            let value = oref(cast.value.clone());
            let operand = self.emit(&value)?;
            return Ok(if dtype == "bool" {
                if dtype_of(&value)? == "bool" {
                    operand
                } else {
                    format!("({operand} != 0)")
                }
            } else {
                format!("({operand} as {rust})")
            });
        }
        if self
            .emitter
            .ctx
            .schema
            .integer_binary_node_kinds
            .contains(kind)
        {
            let (a, b) = crate::analyze::topology_operands(node).expect("integer binary node");
            let a = self.emit(&a)?;
            let b = self.emit(&b)?;
            return Ok(render_integer_binary(kind, &a, &b, &rust, false, "engine", None)?.0);
        }
        macro_rules! binary {
            ($ty:ty, $format:literal) => {
                if let Some(binary) = node.as_node::<$ty>() {
                    let a = self.emit(&oref(binary.a.clone()))?;
                    let b = self.emit(&oref(binary.b.clone()))?;
                    return Ok(format!($format, a, b));
                }
            };
        }
        binary!(EQObj, "({} == {})");
        binary!(NEObj, "({} != {})");
        binary!(LTObj, "({} < {})");
        binary!(LEObj, "({} <= {})");
        binary!(GTObj, "({} > {})");
        binary!(GEObj, "({} >= {})");
        binary!(AndObj, "({} && {})");
        binary!(OrObj, "({} || {})");
        if let Some(not) = node.as_node::<NotObj>() {
            return Ok(format!("(!{})", self.emit(&oref(not.a.clone()))?));
        }
        if let Some(select) = node.as_node::<SelectObj>() {
            return Ok(format!(
                "(if {} {{ {} }} else {{ {} }})",
                self.emit(&oref(select.condition.clone()))?,
                self.emit(&oref(select.true_value.clone()))?,
                self.emit(&oref(select.false_value.clone()))?,
            ));
        }
        if let Some((name, operands)) = crate::analyze::util::bitwise_expr(node) {
            let codes = operands
                .iter()
                .map(|operand| self.emit(operand))
                .collect::<AResult<Vec<_>>>()?;
            return Ok(format!("({})", render_bitwise(name, &codes)));
        }
        if let Some(call) = node.as_node::<CallObj>() {
            let name = call_op_name(call)?.unwrap_or_default();
            let name = name.rsplit('.').next().unwrap_or_default();
            let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
            match (name, args.as_slice()) {
                ("if_then_else", [condition, true_value, false_value]) => {
                    return Ok(format!(
                        "(if {} {{ {} }} else {{ {} }})",
                        self.emit(condition)?,
                        self.emit(true_value)?,
                        self.emit(false_value)?,
                    ));
                }
                ("large_uint_imm", [high, low]) => {
                    let part = |value: &ObjectRef| -> AResult<u64> {
                        match value.as_node::<IntImmObj>() {
                            Some(value) => Ok(int_bits(value)? & 0xffffffff),
                            None => Ok(0),
                        }
                    };
                    return Ok(format!("({}_{rust})", (part(high)? << 32) | part(low)?));
                }
                _ => {}
            }
        }
        unsupported(format!("a declared wait's predicate does not model {kind}"))
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let name = &call.op_name;
    let [destination, pointer, condition, scope, space, requested, _backoff] = call.args.as_slice()
    else {
        return unsupported(format!(
            "{name} expects 7 arguments, got {}",
            call.args.len()
        ));
    };
    let suffix = word_suffix(pointer, requested, name)?;
    let local = destination
        .as_node::<TensorLoadObj>()
        .and_then(|load| as_buffer(&oref(load.source.clone())))
        .is_some_and(|buffer| {
            matches!(
                buffer_scope(&buffer).as_str(),
                "local" | "local_scalar" | "register" | "reg"
            )
        });
    if !local {
        return unsupported(format!("{name} dst must be a writable thread-local scalar"));
    }
    let dtype = dtype_of(destination)?;
    let Some(destination_suffix) = scalar_suffix(&dtype) else {
        return unsupported(format!(
            "{name} dst must be a 32/64/128-bit integer scalar, got {dtype}"
        ));
    };
    if width(destination_suffix) != width(&suffix) {
        return unsupported(format!(
            "{name} destination is {} bits but the sync word is {} bits",
            width(destination_suffix).unwrap(),
            width(&suffix).unwrap()
        ));
    }
    let scope = static_string(scope, &format!("{name} modifier"))?;
    let space = static_string(space, &format!("{name} modifier"))?;
    let decoded = raw_load(
        &format!("ld.acquire.{scope}.{space}.{suffix}"),
        &[destination.clone(), pointer.clone()],
    )?;
    let parts = decoded_load_parts(&decoded)?;
    let pointer = emitter.emit_pointer_handle(&parts.address)?;
    if pointer.rust_type != "PhysicalPtr" {
        return unsupported(format!(
            "{} address lost opaque physical-pointer provenance",
            decoded.op_name
        ));
    }
    let closure = Predicate::new(emitter, parts.destination.clone()).emit(condition)?;
    let semantics = decoded_load_semantics(&parts.sem, &parts.scope, parts.mmio);
    let result = emitter.control_name("declared_wait");
    let source_op_id = call.source_op_id(emitter)?;
    let site = emitter.v2_site(Some(source_op_id));
    let space = v2_memory_space_rust(&parts.space)?;
    let carrier = v2_memory_type_rust_ptx(
        emitter.ctx.schema,
        &parts.result_dtype,
        Some(&parts.ptx_type),
    )?;
    let logical_buffer = emitter.address_logical_buffer(&parts.address)?;
    let wait = abi::warp_call(
        "mem::declared_wait",
        &site,
        &[
            abi::address(
                space,
                &abi::cloned(&pointer.code),
                logical_buffer.as_deref(),
            ),
            format!(
                "move |candidate: u64, lane: usize| -> Result<bool, EngineError> {{ let _ = lane; Ok({closure}) }}"
            ),
        ],
        Some(&format!("{carrier}, {space}, {semantics}, _")),
        None,
        true,
        true,
    );
    emitter.emit_suspend_line(&format!("let {result} = v2_register_out({wait});"));
    let rust = expr_rust_type(emitter.ctx.schema, &parts.result_dtype)?;
    emitter.emit_explicit_buffer_store(
        &parts.destination,
        RustValue::new(result, rust, Uniformity::Varying),
        source_op_id,
        None,
        None,
        None,
    )?;
    Ok(None)
}
