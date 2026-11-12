//! `analyze_primfunc` producing the kernel manifest and its source nodes.

use crate::analyze::util::{
    enum_name, function_body, is_tensor_var, json_object, json_strings, tensor_var, TensorVarExt,
};

use tvm_tirx::ir::StringImm;
use tvm_tirx::ir::{CallObj, PtrTypeObj, TensorLoadObj, Var};
use tvm_tirx::ir::prim::{CastObj, RampObj, ShuffleObj};
use tvm_tirx::ir::{
    Evaluate, EvaluateObj, ForObj, IfObj, RegionStmtObj, SeqStmtObj, Stmt, TensorStoreObj, WhileObj,
};
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::{self, structural_map, Any, ObjectRefCore, String as FfiString, WalkOrder};

use super::buffers::{call_op_name, BufferBindings};
use super::expression_bindings::ExpressionBindings;
use super::memory::{build_memory_plan, MemoryPlan, MemorySpace};
use super::topology::{extract_topology, validate_for, validate_region, LaunchTopology};
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

fn is_supported_buffer_dtype(ctx: &Ctx, buffer: &Var) -> bool {
    let dtype = buffer_dtype(buffer);
    if ctx.schema.is_supported_buffer_dtype_name(&dtype) {
        return true;
    }
    if dtype != "handle" {
        return false;
    }
    let scope = buffer_scope(buffer);
    let shape: Vec<_> = buffer.tensor_type().shape.iter().collect();
    ["local", "local_scalar", "register", "reg"].contains(&scope.as_str())
        && shape.len() == 1
        && super::util::int_imm_expr(&shape[0].clone().into()) == Some(1)
}

/// A statement header with every child statement rendered as `...`.
fn elided_statement_text(node: &ObjectRef) -> AResult<String> {
    // Child statements have their own source entries.
    let placeholder: Stmt = Evaluate::new(StringImm::new("...", tvm_tirx::ir::UNKNOWN_LOC), tvm_tirx::ir::UNKNOWN_LOC).into();
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
    if let Some(seq) = node.downcast_ref::<SeqStmtObj>() {
        return Ok(format!("SeqStmt({} statements)", seq.seq.len()));
    }
    let mut text = if node.downcast_ref::<RegionStmtObj>().is_some()
        || node.downcast_ref::<ForObj>().is_some()
        || node.downcast_ref::<WhileObj>().is_some()
        || node.downcast_ref::<IfObj>().is_some()
    {
        elided_statement_text(node)?
    } else if let Some(text) = crate::emit::pure::source_text(node)? {
        text
    } else {
        repr_text(node)?.trim().to_owned()
    };
    if looks_like_address_printer(&text) {
        let Some(fallback) = call_text(node)? else {
            return not_covered("node text falls back to the address printer");
        };
        text = fallback;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > 1000 {
        text = format!("{}...", chars[..997].iter().collect::<String>());
    }
    Ok(text)
}

/// `T.<op>(<operands>)` for a call, or an evaluated call, that the TVMScript
/// printer can only show by address. tvm-tirx's printer recovers a PTX call's
/// surface spelling from its operand count, which two table entries can share
/// (`mbarrier.try_wait.parity` with a report and with a hint); the op is unique.
fn call_text(node: &ObjectRef) -> AResult<Option<String>> {
    let call_node = match node.downcast_ref::<EvaluateObj>() {
        Some(evaluate) => oref(evaluate.value.clone()),
        None => node.clone(),
    };
    let Some(call) = call_node.downcast_ref::<CallObj>() else {
        return Ok(None);
    };
    let Some(op_name) = call_op_name(call)? else {
        return Ok(None);
    };
    let mut operands = Vec::with_capacity(call.args.len());
    for operand in call.args.iter() {
        let text = repr_text(&oref(operand.clone()))?.trim().to_owned();
        if looks_like_address_printer(&text) {
            return Ok(None);
        }
        operands.push(text);
    }
    let name = op_name.strip_prefix("tirx.").unwrap_or(&op_name);
    Ok(Some(format!("T.{name}({})", operands.join(", "))))
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
    pub func: Function,
    pub manifest: Json,
    /// Whether the manifest lists no unsupported item.
    pub supported: bool,
    pub statements: Vec<Stmt>,
    pub nodes: Vec<ObjectRef>,
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

pub fn analyze_primfunc(ctx: &Ctx, func: &Function) -> AResult<KernelPlan> {
    let statements: Vec<Stmt> = crate::walk_statements(function_body(func)?)?
        .iter()
        .cloned()
        .collect();
    let nodes: Vec<ObjectRef> = crate::post_order_nodes(oref(function_body(func)?))?
        .iter()
        .cloned()
        .collect();
    let buffer_bindings = BufferBindings::build(&statements)?;
    ctx.record_tensor_decls(&buffer_bindings)?;
    let params: Vec<Var> = func.params.iter().cloned().collect();

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
    let requires_implicit_tmem = tmem_requirements.uses_tmem
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
    let topology = extract_topology(ctx, &function_body(func)?, Default::default())?;
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
        check_node(ctx, node, kind, op_id, &mut self.semantic_unsupported)
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
    if let Some(region) = node.downcast_ref::<RegionStmtObj>() {
        if let Some(error) = validate_region(region)? {
            unsupported.push(format!("op#{op_id}:{error}"));
        }
    }
    if let Some(loop_stmt) = node.downcast_ref::<ForObj>() {
        if let Some(error) = validate_for(ctx, loop_stmt)? {
            unsupported.push(format!("op#{op_id}:{error}"));
        }
    }
    if let Some(store) = node.downcast_ref::<TensorStoreObj>() {
        let buffer = tensor_var(&oref(store.dest.clone()))?;
        if !is_supported_buffer_dtype(ctx, &buffer) {
            unsupported.push(format!(
                "op#{op_id}:BufferStore(dtype={})",
                buffer_dtype(&buffer)
            ));
        }
    }
    if let Some(load) = node.downcast_ref::<TensorLoadObj>() {
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
    if let Some(ramp) = node.downcast_ref::<RampObj>() {
        match classify_contiguous_ramp(ctx, node, ramp, None) {
            Ok(_) => {}
            Err(Failure::Unsupported { message, .. }) => {
                unsupported.push(format!("op#{op_id}:{message}"))
            }
            Err(error) => return Err(error),
        }
    }
    if let Some(shuffle) = node.downcast_ref::<ShuffleObj>() {
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
    if let Some(cast) = node.downcast_ref::<CastObj>() {
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
    if let Some(call) = node.downcast_ref::<CallObj>() {
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
    func: &Function,
    unsupported_items: &mut Vec<String>,
) -> AResult<(Vec<Json>, Vec<String>)> {
    let mut buffers: Vec<Json> = Vec::new();
    let mut buffer_names: Vec<String> = Vec::new();
    for parameter in crate::buffer_parameters(func.clone())?.iter() {
        let buffer = parameter;
        let buffer_type = buffer.tensor_type();
        let name = buffer_name(&buffer);
        buffer_names.push(name.clone());
        let mut shape_texts = Vec::new();
        for value in buffer_type.shape.iter() {
            shape_texts.push(repr_of(value)?);
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
            // A TensorType has no element offset; its layout carries any offset.
            ("elem_offset", Json::String("0".to_owned())),
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
        if is_tensor_var(parameter) {
            continue;
        }
        if let Some(pointer) = parameter.ty.downcast_ref::<PtrTypeObj>() {
            let Some(element) = prim_dtype(&pointer.element_type) else {
                continue;
            };
            let dtype = dtype_text(element);
            if dtype.is_empty() || dtype == "handle" {
                continue;
            }
            let name = ffi_text(&parameter.name);
            let storage_scope = enum_name(&*pointer.scope);
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
fn kernel_name(func: &Function) -> AResult<String> {
    if let Some(symbol) = func.attrs.dict.get(&FfiString::from("global_symbol"))? {
        if let Ok(text) = FfiString::try_from(symbol) {
            return Ok(ffi_text(&text));
        }
    }
    Ok("main".to_owned())
}

fn structural_hash(func: &Function) -> AResult<u64> {
    let hash: Any = tvm_ffi::cached_global_func!("ffi.StructuralHash").call_tuple((
        Any::from(func.clone()),
        false,
        false,
    ))?;
    // Python renders tvm_ffi.structural_hash as an unsigned 64-bit integer.
    Ok(i64::try_from(hash)? as u64)
}
