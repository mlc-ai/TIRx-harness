//! `analyze_primfunc` producing the kernel manifest and its source nodes.

use crate::analyze::util::{json_object, json_strings};
use tvm::ir::StringImm;
use tvm::ir::{CallObj, PointerTypeObj, TensorLoadObj, Var};
use tvm::prim::{CastObj, RampObj, ShuffleObj};
use tvm::tirx::{
    AttrStmtObj, BufferStoreObj, BufferVar, Evaluate, ForObj, IfThenElseObj, PrimFunc, SeqStmtObj,
    Stmt, TilePrimitiveCallObj, WhileObj,
};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{self, structural_map, Any, ObjectRefCore, String as FfiString, WalkOrder};

use super::buffers::{call_op_name, BufferBindings};
use super::expression_bindings::ExpressionBindings;
use super::memory::{build_memory_plan, MemoryPlan, MemorySpace};
use super::topology::{extract_topology, validate_attr_stmt, validate_for, LaunchTopology};
use super::util::{
    as_buffer, buffer_dtype, buffer_name, buffer_scope, dtype_of, dtype_text, ffi_text,
    kind_or_bail, node_span, not_covered, oref, prim_dtype, repr_of, repr_text, runtime_kind,
    sorted_unique, span_json, AResult, Failure, Json,
};
use super::vector::{
    classify_contiguous_ramp, classify_vector_buffer_load, classify_vector_extract,
};
use super::Ctx;
use crate::emit::raw_tma::TensorMapParameter;

const RAW_TENSOR_MAP_REGISTRY_REQUIREMENT: &str = "raw_tensor_map_registry";
pub use crate::emit::sync::EXTERNAL_GRID_DEPENDENCY_REQUIREMENT;

fn is_supported_buffer_dtype(ctx: &Ctx, buffer: &BufferVar) -> bool {
    let dtype = buffer_dtype(buffer);
    if ctx.schema.is_supported_buffer_dtype_name(&dtype) {
        return true;
    }
    if dtype != "handle" {
        return false;
    }
    let scope = buffer_scope(buffer);
    let shape: Vec<_> = buffer.buffer_type().shape.iter().collect();
    ["local", "local_scalar", "register", "reg"].contains(&scope.as_str())
        && shape.len() == 1
        && super::util::int_imm_expr(&shape[0]) == Some(1)
}

/// A statement header with every child statement rendered as `...`.
fn elided_statement_text(node: &ObjectRef) -> AResult<String> {
    // Child statements have their own source entries.
    let placeholder: Stmt = Evaluate::new(StringImm::new("..."))?.into();
    let root = node.clone();
    let rendered: ObjectRef = structural_map(
        node.clone(),
        move |stmt: Stmt| -> tvm_ffi::Result<Any> {
            Ok(if super::util::same(&stmt, &root) {
                stmt.into()
            } else {
                placeholder.clone().into()
            })
        },
        WalkOrder::PreOrder,
    )?
    .try_into()?;
    let text = repr_text(&rendered)?.trim().to_owned();
    Ok(if super::util::same(&rendered, node) {
        text
    } else {
        text.replace("T.evaluate(\"...\")", "...")
    })
}

fn node_text(node: &ObjectRef) -> AResult<String> {
    if let Some(seq) = node.as_node::<SeqStmtObj>() {
        return Ok(format!("SeqStmt({} statements)", seq.seq.len()));
    }
    let mut text = if node.as_node::<AttrStmtObj>().is_some()
        || node.as_node::<ForObj>().is_some()
        || node.as_node::<WhileObj>().is_some()
        || node.as_node::<IfThenElseObj>().is_some()
    {
        elided_statement_text(node)?
    } else if let Some(text) = crate::emit::pure::source_text(node)? {
        text
    } else {
        repr_text(node)?.trim().to_owned()
    };
    if looks_like_address_printer(&text) {
        return not_covered("node text falls back to the address printer");
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > 1000 {
        text = format!("{}...", chars[..997].iter().collect::<String>());
    }
    Ok(text)
}

/// `^[A-Za-z_][A-Za-z0-9_.]*\(0x[0-9A-Fa-f]+\)$`
fn looks_like_address_printer(text: &str) -> bool {
    let Some(open) = text.find('(') else {
        return false;
    };
    let head = &text[..open];
    let tail = &text[open + 1..];
    if head.is_empty() {
        return false;
    }
    let mut chars = head.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') {
        return false;
    }
    let Some(inner) = tail.strip_suffix(')') else {
        return false;
    };
    let Some(hex) = inner.strip_prefix("0x") else {
        return false;
    };
    !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit())
}

/// One analyzed kernel: its manifest and the facts emission borrows.
pub struct KernelPlan {
    pub func: PrimFunc,
    pub manifest: Json,
    /// Whether the manifest lists no unsupported item.
    pub supported: bool,
    pub statements: Vec<Stmt>,
    pub nodes: Vec<ObjectRef>,
    /// The parsed tile call per op id.
    pub tile_calls: Vec<Option<super::tile_forms::ParsedTileCall>>,
    pub buffer_bindings: BufferBindings,
    pub memory_plan: Option<MemoryPlan>,
    pub expression_bindings: Option<ExpressionBindings>,
    /// The unsupported memory or expression-binding analysis that leaves the kernel
    /// without emission facts.
    pub analysis_failure: Option<(String, Vec<String>)>,
    pub name: String,
    pub topology: LaunchTopology,
    pub buffer_names: Vec<String>,
    pub scalars: Vec<ParameterSpec>,
    pub pointers: Vec<ParameterSpec>,
    pub tensor_maps: Vec<TensorMapParameter>,
    /// The implicit TensorMap metadata attribute of `func`.
    pub implicit_tensor_maps: Vec<Json>,
    pub uses_raw_tensor_map_registry: bool,
    pub requires_implicit_tmem: bool,
    pub uses_dynamic_tmem_lifecycle: bool,
    /// Whether any call is a readonly-proxy load.
    pub uses_readonly_proxy: bool,
}

/// A scalar or pointer PrimFunc parameter.
pub struct ParameterSpec {
    pub name: String,
    pub dtype: String,
    pub parameter_index: i64,
}

pub fn analyze_primfunc(ctx: &Ctx, func: &PrimFunc) -> AResult<KernelPlan> {
    let statements: Vec<Stmt> = crate::walk_statements(func.body().clone())?
        .iter()
        .collect();
    let nodes: Vec<ObjectRef> = crate::post_order_nodes(oref(func.body().clone()))?
        .iter()
        .collect();
    let buffer_bindings = BufferBindings::build(&statements)?;
    let params: Vec<Var> = func.params.iter().collect();

    let mut facts = KernelFacts::default();
    let memory =
        build_memory_plan(ctx, func, &statements, &nodes, &buffer_bindings).and_then(|plan| {
            let bindings = ExpressionBindings::build(&ctx.analyzer, &statements, &buffer_bindings)?;
            Ok((plan, bindings))
        });
    let (memory_plan, expression_bindings) = facts.supported_analysis(memory)?.unzip();
    for node in &nodes {
        facts.record_node(ctx, node)?;
    }
    let tensor_map_facts = crate::emit::raw_tma::scan_tensor_maps(ctx, &nodes, &params)?;
    let uses_raw_tensor_map_registry = tensor_map_facts.uses_registry
        || crate::emit::cuda_helper::requires_tensor_map_registry(&nodes)?;
    let mut semantic_requirements = Vec::new();
    if uses_raw_tensor_map_registry {
        semantic_requirements.push(RAW_TENSOR_MAP_REGISTRY_REQUIREMENT.to_owned());
    }
    if crate::emit::sync::external_grid_dependency(ctx, &nodes)? {
        semantic_requirements.push(EXTERNAL_GRID_DEPENDENCY_REQUIREMENT.to_owned());
    }
    let tmem_requirements = crate::emit::tcgen05::scan_tmem_requirements(&nodes)?;
    let mut unsupported_items = facts.unsupported_node_kinds;
    unsupported_items.extend(facts.semantic_unsupported);
    let (buffers, buffer_names) = buffer_entries(ctx, func, &mut unsupported_items)?;
    let ((scalars, scalar_specs), (pointers, pointer_specs)) =
        scalar_and_pointer_entries(ctx, &params, &mut unsupported_items)?;
    let requires_implicit_tmem = (facts.tile_uses_tmem || tmem_requirements.uses_tmem)
        && memory_plan.as_ref().is_some_and(|plan| {
            !plan
                .backings
                .iter()
                .any(|backing| backing.space == MemorySpace::Tmem)
        });
    let uses_dynamic_tmem_lifecycle = tmem_requirements.dynamic_lifecycle;
    let uses_readonly_proxy = crate::emit::raw_memory::uses_readonly_proxy(&nodes)?;
    let name = kernel_name(func)?;
    let hash = structural_hash(func)?;
    let topology = extract_topology(ctx, func.body(), Default::default())?;
    let implicit_tensor_maps = crate::emit::raw_tma::implicit_tensor_map_metadata(func)?;
    let (tensor_maps, tensor_map_parameters) = crate::emit::raw_tma::tensor_map_specs(
        &implicit_tensor_maps,
        &params,
        &tensor_map_facts.parameter_indices,
    );
    let supported = unsupported_items.is_empty();
    // Source identities and call names live together in the source map.
    let manifest = json_object(vec![
        ("name", Json::String(name.clone())),
        ("structural_hash", Json::String(hash.to_string())),
        ("topology", topology.json()),
        ("buffers", Json::Array(buffers)),
        ("scalars", Json::Array(scalars)),
        ("pointers", Json::Array(pointers)),
        ("tensor_maps", Json::Array(tensor_maps)),
        ("source_map", Json::Array(facts.source_entries)),
        ("requires_implicit_tmem", Json::Bool(requires_implicit_tmem)),
        (
            "uses_dynamic_tmem_lifecycle",
            Json::Bool(uses_dynamic_tmem_lifecycle),
        ),
        (
            "semantic_requirements",
            json_strings(sorted_unique(semantic_requirements)),
        ),
        ("unsupported", json_strings(unsupported_items)),
    ]);
    let mut plan = KernelPlan {
        func: func.clone(),
        manifest,
        supported,
        statements,
        nodes,
        tile_calls: facts.tile_calls,
        buffer_bindings,
        memory_plan,
        expression_bindings,
        analysis_failure: facts.analysis_failure,
        name,
        topology,
        buffer_names,
        scalars: scalar_specs,
        pointers: pointer_specs,
        tensor_maps: tensor_map_parameters,
        implicit_tensor_maps,
        uses_raw_tensor_map_registry,
        requires_implicit_tmem,
        uses_dynamic_tmem_lifecycle,
        uses_readonly_proxy,
    };
    if plan.analysis_failure.is_none() {
        let errors = crate::emit::diagnostics::collect(ctx, &plan)?;
        if let Json::Object(fields) = &mut plan.manifest {
            let unsupported = fields
                .get_mut("unsupported")
                .and_then(Json::as_array_mut)
                .expect("kernel unsupported list");
            for error in errors {
                if !unsupported
                    .iter()
                    .any(|item| item.as_str() == Some(error.as_str()))
                {
                    unsupported.push(Json::String(error));
                }
            }
            plan.supported = unsupported.is_empty();
        }
    }
    Ok(plan)
}

/// What the walk over one kernel's nodes records.
#[derive(Default)]
struct KernelFacts {
    source_entries: Vec<Json>,
    /// `op#<id>:<kind>` for every node kind the frontend does not support.
    unsupported_node_kinds: Vec<String>,
    semantic_unsupported: Vec<String>,
    tile_calls: Vec<Option<super::tile_forms::ParsedTileCall>>,
    /// Whether a tile lowering touches TMEM.
    tile_uses_tmem: bool,
    /// The unsupported memory or expression-binding analysis.
    analysis_failure: Option<(String, Vec<String>)>,
}

impl KernelFacts {
    /// A supported analysis; an unsupported one is recorded and yields `None`.
    fn supported_analysis<T>(&mut self, analysis: AResult<T>) -> AResult<Option<T>> {
        match analysis {
            Ok(analysis) => Ok(Some(analysis)),
            Err(Failure::Unsupported {
                message,
                unsupported,
            }) => {
                self.analysis_failure = Some((message.clone(), unsupported.clone()));
                if unsupported.is_empty() {
                    self.semantic_unsupported.push(message);
                } else {
                    self.semantic_unsupported.extend(unsupported);
                }
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// One node's census, source entry and generic shape/type checks.
    fn record_node(&mut self, ctx: &Ctx, node: &ObjectRef) -> AResult<()> {
        let kind = kind_or_bail(node)?;
        let op_id = self.source_entries.len() as i64;
        if !ctx.schema.runtime_frontend_node_kinds.contains(kind)
            && !ctx.schema.compile_only_frontend_node_kinds.contains(kind)
        {
            self.unsupported_node_kinds
                .push(format!("op#{op_id}:{kind}"));
        }
        self.source_entries
            .push(source_entry(node, &runtime_kind(node)?, op_id)?);
        self.tile_calls.push(None);
        check_node(ctx, node, kind, op_id, &mut self.semantic_unsupported)?;
        self.record_tile_call(ctx, node, op_id)
    }

    /// A TilePrimitiveCall's lowered op and TMEM semantics.
    fn record_tile_call(&mut self, ctx: &Ctx, node: &ObjectRef, op_id: i64) -> AResult<()> {
        let Some(call) = node.as_node::<TilePrimitiveCallObj>() else {
            return Ok(());
        };
        let resolved = super::tile_forms::tile_op_name(call)
            .and_then(|_| super::tile_forms::resolve_tile_call(ctx, node));
        let operation = match resolved {
            Ok(resolved) => resolved,
            Err(Failure::Unsupported { message, .. }) => {
                self.semantic_unsupported
                    .push(format!("op#{op_id}:{message}"));
                return Ok(());
            }
            Err(failure) => return Err(with_node_span(failure, node)),
        };
        self.tile_uses_tmem =
            self.tile_uses_tmem || super::tile_forms::tile_lowering_uses_tmem(&operation);
        self.tile_calls[op_id as usize] = Some(operation);
        Ok(())
    }
}

/// The unsupported dtypes and forms of one non-call node.
fn check_node(
    ctx: &Ctx,
    node: &ObjectRef,
    kind: &str,
    op_id: i64,
    unsupported: &mut Vec<String>,
) -> AResult<()> {
    if let Some((name, operands)) = super::util::bitwise_expr(node) {
        for operand in operands {
            let dtype = dtype_of(&operand)?;
            if !crate::tables::is_integer_dtype(&dtype)
                && !(dtype == "bool" && !name.starts_with("shift_"))
            {
                unsupported.push(format!(
                    "op#{op_id}:{kind} requires scalar integer or boolean operands, got {dtype}"
                ));
            }
        }
    }
    if let Some(attr) = node.as_node::<AttrStmtObj>() {
        if let Some(error) = validate_attr_stmt(ctx, attr)? {
            unsupported.push(format!("op#{op_id}:{error}"));
        }
    }
    if let Some(loop_stmt) = node.as_node::<ForObj>() {
        if let Some(error) = validate_for(ctx, loop_stmt)? {
            unsupported.push(format!("op#{op_id}:{error}"));
        }
    }
    if let Some(store) = node.as_node::<BufferStoreObj>() {
        if !is_supported_buffer_dtype(ctx, &store.buffer) {
            unsupported.push(format!(
                "op#{op_id}:BufferStore(dtype={})",
                buffer_dtype(&store.buffer)
            ));
        }
    }
    if let Some(load) = node.as_node::<TensorLoadObj>() {
        let Some(source) = as_buffer(&oref(load.source.clone())) else {
            return not_covered("TensorLoad source is not a typed buffer");
        };
        if !is_supported_buffer_dtype(ctx, &source) {
            unsupported.push(format!(
                "op#{op_id}:TensorLoad(dtype={})",
                buffer_dtype(&source)
            ));
        }
        if let Err(Failure::Unsupported { message, .. }) =
            classify_vector_buffer_load(ctx, node, load, &source)
        {
            unsupported.push(format!("op#{op_id}:{message}"));
        }
    }
    if let Some(ramp) = node.as_node::<RampObj>() {
        match classify_contiguous_ramp(ctx, node, ramp, None) {
            Ok(_) => {}
            Err(Failure::Unsupported { message, .. }) => {
                unsupported.push(format!("op#{op_id}:{message}"))
            }
            Err(error) => return Err(error),
        }
    }
    if let Some(shuffle) = node.as_node::<ShuffleObj>() {
        match classify_vector_extract(ctx, node, shuffle) {
            Ok(_) => {}
            Err(Failure::Unsupported { message, .. }) => {
                unsupported.push(format!("op#{op_id}:{message}"))
            }
            Err(error) => return Err(error),
        }
    }
    let comparison_kinds = ["LT", "LE", "GT", "GE", "EQ", "NE"];
    if ctx.schema.integer_binary_node_kinds.contains(kind) || comparison_kinds.contains(&kind) {
        let (a, b) = super::topology_operands(node).expect("binary node");
        let mut packed = false;
        for operand in [a, b] {
            let Some(dtype) = prim_dtype(&super::util::expr_type(&operand).expect("typed operand"))
            else {
                return not_covered("binary operand without a primitive type");
            };
            if ctx.schema.vector_dtype_abi(&dtype_text(dtype)).is_some() {
                packed = true;
            }
        }
        if packed {
            unsupported.push(format!(
                "op#{op_id}:{kind} on packed vector storage requires an explicit CUDA/PTX vector operation"
            ));
        }
    }
    if let Some(cast) = node.as_node::<CastObj>() {
        let source_dtype = dtype_of(&oref(cast.value.clone()))?;
        let target_dtype = dtype_of(node)?;
        if !ctx.schema.supported_cast_dtypes.contains(&source_dtype)
            || !ctx.schema.supported_cast_dtypes.contains(&target_dtype)
        {
            unsupported.push(format!("op#{op_id}:Cast({source_dtype}->{target_dtype})"));
        }
    }
    Ok(())
}

/// A resolution failure of `node`: an unmodeled form without a span takes the
/// node's span.
pub(crate) fn with_node_span(failure: Failure, node: &ObjectRef) -> Failure {
    match failure {
        Failure::Unmodeled {
            target,
            message,
            span: None,
        } => Failure::Unmodeled {
            target,
            message,
            span: span_json(node_span(node).as_ref()),
        },
        failure => failure,
    }
}

/// The `source_map` entry of one node.
fn source_entry(node: &ObjectRef, kind: &str, op_id: i64) -> AResult<Json> {
    let mut entry = vec![
        ("op_id", Json::from(op_id)),
        ("kind", Json::from(kind)),
        ("text", Json::String(node_text(node)?)),
        (
            "span",
            span_json(node_span(node).as_ref()).unwrap_or(Json::Null),
        ),
    ];
    if let Some(call) = node.as_node::<CallObj>() {
        if let Some(name) = call_op_name(call)? {
            entry.push((
                "op_name",
                Json::String(crate::decode::canonical_op_name(&name)),
            ));
        }
    }
    Ok(json_object(entry))
}

/// The manifest entries and names of the buffer parameters; unsupported ones
/// are added to `unsupported_items`.
fn buffer_entries(
    ctx: &Ctx,
    func: &PrimFunc,
    unsupported_items: &mut Vec<String>,
) -> AResult<(Vec<Json>, Vec<String>)> {
    let mut buffers: Vec<Json> = Vec::new();
    let mut buffer_names: Vec<String> = Vec::new();
    for parameter in crate::buffer_parameters(func.clone())?.iter() {
        let buffer = BufferVar::try_from(&parameter)?;
        let buffer_type = buffer.buffer_type();
        let name = buffer_name(&buffer);
        buffer_names.push(name.clone());
        let mut shape_texts = Vec::new();
        for value in buffer_type.shape.iter() {
            shape_texts.push(repr_of(&value)?);
        }
        let dtype = buffer_dtype(&buffer);
        let layout = match &buffer_type.layout {
            None => "None".to_owned(),
            Some(layout) => repr_of(layout)?,
        };
        buffers.push(json_object(vec![
            ("parameter", Json::String(name.clone())),
            ("name", Json::String(name.clone())),
            ("shape", json_strings(shape_texts.iter().cloned())),
            ("dtype", Json::String(dtype.clone())),
            ("scope", Json::String(buffer_scope(&buffer))),
            (
                "elem_offset",
                Json::String(repr_of(&buffer_type.elem_offset)?),
            ),
            ("layout", Json::String(layout)),
        ]));
        if !ctx.schema.is_supported_buffer_dtype_name(&dtype) || shape_texts.is_empty() {
            unsupported_items.push(format!(
                "buffer:{name}:requires a supported scalar parameter, got shape={:?}, dtype={dtype}",
                &shape_texts));
        }
    }
    Ok((buffers, buffer_names))
}

/// The manifest entries and specs of the scalar and the pointer parameters;
/// unsupported ones are added to `unsupported_items`.
fn scalar_and_pointer_entries(
    ctx: &Ctx,
    params: &[Var],
    unsupported_items: &mut Vec<String>,
) -> AResult<(
    (Vec<Json>, Vec<ParameterSpec>),
    (Vec<Json>, Vec<ParameterSpec>),
)> {
    let mut scalars: Vec<Json> = Vec::new();
    let mut pointers: Vec<Json> = Vec::new();
    let mut scalar_specs: Vec<ParameterSpec> = Vec::new();
    let mut pointer_specs: Vec<ParameterSpec> = Vec::new();
    for (index, parameter) in params.iter().enumerate() {
        if BufferVar::try_from(parameter).is_ok() {
            continue;
        }
        if let Some(pointer) = parameter.ty.as_node::<PointerTypeObj>() {
            let Some(element) = prim_dtype(&pointer.element_type) else {
                continue;
            };
            let dtype = dtype_text(element);
            if dtype.is_empty() || dtype == "handle" {
                continue;
            }
            let name = ffi_text(&parameter.name);
            let storage_scope = ffi_text(&pointer.storage_scope);
            pointers.push(json_object(vec![
                ("name", Json::String(name.clone())),
                ("dtype", Json::String(dtype.clone())),
                ("storage_scope", Json::String(storage_scope.clone())),
                ("parameter_index", Json::from(index as i64)),
            ]));
            pointer_specs.push(ParameterSpec {
                name: name.clone(),
                dtype: dtype.clone(),
                parameter_index: index as i64,
            });
            if !ctx.schema.is_supported_buffer_dtype_name(&dtype) || storage_scope != "global" {
                unsupported_items.push(format!(
                    "pointer:{name}:requires a supported global scalar pointer, got dtype={dtype}, storage_scope={storage_scope}"
                ));
            }
            continue;
        }
        let Some(dtype) = prim_dtype(&parameter.ty) else {
            return not_covered("scalar parameter without a primitive type");
        };
        let dtype = dtype_text(dtype);
        let name = ffi_text(&parameter.name);
        scalars.push(json_object(vec![
            ("name", Json::String(name.clone())),
            ("dtype", Json::String(dtype.clone())),
            ("parameter_index", Json::from(index as i64)),
        ]));
        scalar_specs.push(ParameterSpec {
            name: name.clone(),
            dtype: dtype.clone(),
            parameter_index: index as i64,
        });
        if !ctx
            .schema
            .supported_parameter_scalar_dtypes
            .contains(&dtype)
        {
            unsupported_items.push(format!(
                "scalar:{name}:PrimFunc parameter dtype {dtype} is not supported"
            ));
        }
    }
    Ok(((scalars, scalar_specs), (pointers, pointer_specs)))
}

/// The `global_symbol` attribute, or `main`.
fn kernel_name(func: &PrimFunc) -> AResult<String> {
    if let Some(symbol) = func.attrs.dict.get(&FfiString::from("global_symbol"))? {
        if let Ok(text) = FfiString::try_from(symbol) {
            return Ok(ffi_text(&text));
        }
    }
    Ok("main".to_owned())
}

fn structural_hash(func: &PrimFunc) -> AResult<u64> {
    let hash: Any = tvm_ffi::cached_global_func!("ffi.StructuralHash").call_tuple((
        Any::from(func.clone()),
        false,
        false,
    ))?;
    // Python renders tvm_ffi.structural_hash as an unsigned 64-bit integer.
    Ok(i64::try_from(hash)? as u64)
}
