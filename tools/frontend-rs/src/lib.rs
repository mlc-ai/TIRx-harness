//! Compile-time TIRx services. Runtime kernels keep using the NumSim engine.

mod analyze;
mod decode;
mod dtypes;
mod emit;
mod host_abi;
#[cfg(test)]
mod ir_tests;
mod registry;
mod schema;
mod serialization;
mod tables;
mod tvm_compat;

use crate::analyze::util::{json_object, json_strings, AResult, Failure, Json};
use std::collections::HashSet;

use tvm::analysis::Analyzer;
use tvm::ir::StringImmObj;
use tvm::ir::{Call, Expr, IntImm, Op, PrimExpr, Var};
use tvm::tirx::{AttrStmtObj, BufferVar, Layout, PrimFunc, Stmt};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{
    structural_map, structural_visit, tvm_ffi_dll_export_typed_func, Any, Array, Error, Map,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String, StructuralView, VisitCallbacks,
    VisitContext, VisitInterrupt, WalkOrder, VALUE_ERROR,
};

fn identity() -> Result<String> {
    Ok(format!(
        "{}:{}",
        env!("CARGO_PKG_VERSION"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/thirdparty/tvm-rust-ext/REVISION"
        ))
        .trim(),
    )
    .into())
}

pub(crate) fn layout_linear_offsets(layout: Layout, count: i64) -> Result<Array<PrimExpr>> {
    let analyzer = Analyzer::new()?;
    let mut offsets = Vec::new();
    for index in 0..count {
        let mapped = layout.apply_linear(&IntImm::new("int64", index)?.into())?;
        let offset = mapped
            .get(&"m".into())?
            .unwrap_or(IntImm::new("int64", 0)?.into());
        offsets.push(analyzer.simplify(&offset)?);
    }
    Ok(Array::new(offsets))
}

pub(crate) fn walk_statements(stmt: Stmt) -> Result<Array<Stmt>> {
    let mut visitor = VisitCallbacks::new(
        Vec::<Stmt>::new(),
        |value: &StructuralView,
         visitor: &mut VisitContext<'_, Vec<Stmt>>|
         -> Result<Option<VisitInterrupt>> {
            if let Some(stmt) = value.cast::<Stmt>() {
                visitor.state_mut().push(stmt);
                // AttrStmt.node is an annotation target, not an executed child.
                if let Some(attr) = value.as_node::<AttrStmtObj>() {
                    return visitor.visit(&attr.body);
                }
                return visitor.visit_children();
            }
            if value.cast::<Array<Stmt>>().is_some() {
                return visitor.visit_children();
            }
            // Do not enter expressions, buffers, spans or annotation maps.
            Ok(None)
        },
    );
    structural_visit(&stmt, &mut visitor)?;
    Ok(Array::new(visitor.into_state()))
}

pub(crate) fn buffer_parameters(func: PrimFunc) -> Result<Array<Var>> {
    Ok(Array::new(
        func.params
            .iter()
            .filter_map(|parameter| BufferVar::try_from(parameter).ok())
            .map(|buffer| buffer.as_var().clone())
            .collect::<Vec<_>>(),
    ))
}

pub(crate) fn post_order_nodes(node: ObjectRef) -> Result<Array<ObjectRef>> {
    // One postorder entry per Expr or
    // Stmt identity, shared subgraphs pruned before descent, and every other
    // value walked by the default reflected traversal. Source op IDs and cache
    // rebinding depend on exactly this ordering and DAG rule.
    type State = (Vec<ObjectRef>, HashSet<ObjectIdentity>);
    let mut visitor = VisitCallbacks::new(
        (Vec::new(), HashSet::new()),
        |value: &StructuralView,
         visitor: &mut VisitContext<'_, State>|
         -> Result<Option<VisitInterrupt>> {
            if value.cast::<Expr>().is_none() && value.cast::<Stmt>().is_none() {
                return visitor.visit_children();
            }
            let node = value
                .cast::<ObjectRef>()
                .ok_or_else(|| Error::new(VALUE_ERROR, "source node is not an object", ""))?;
            if !visitor.state_mut().1.insert(ObjectIdentity::of(&node)) {
                return Ok(None);
            }
            if let Some(interrupt) = visitor.visit_children()? {
                return Ok(Some(interrupt));
            }
            visitor.state_mut().0.push(node);
            Ok(None)
        },
    );
    structural_visit(&node, &mut visitor)?;
    Ok(Array::new(visitor.into_state().0))
}

pub(crate) fn substitute(node: ObjectRef, replacements: Map<Var, Expr>) -> Result<ObjectRef> {
    // Replace each Var occurrence once without revisiting the replacement.
    structural_map(
        node,
        move |variable: Var| -> Result<Any> {
            Ok(match replacements.get(&variable)? {
                Some(replacement) => replacement.into(),
                None => variable.into(),
            })
        },
        WalkOrder::PostOrder,
    )?
    .try_into()
}

fn normalize_host_tensor_maps(func: PrimFunc, tvm_version: String) -> Result<Map<String, Any>> {
    service_result(emit::raw_tma::host_prelude::normalize(func, tvm_version))
}

fn ptx_call_parts(call: Call, modifier_count: i64) -> Result<Array<Any>> {
    let metadata_count = usize::try_from(modifier_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| Error::new(VALUE_ERROR, "invalid PTX modifier count", ""))?;
    // Only unwrap the trailing metadata. Operand expressions retain their handles,
    // including StringImm operands. Python keeps the target TVM table validation.
    let tokens = call
        .args
        .iter()
        .skip(call.args.len().saturating_sub(metadata_count))
        .map(|arg| {
            let token = arg
                .as_node::<StringImmObj>()
                .map(|value| value.value.clone());
            Any::from(token)
        })
        .collect::<Vec<_>>();
    Ok(Array::new(vec![
        call.op.clone().try_cast::<Op>()?.name()?.into(),
        call.args.clone().into(),
        Array::new(tokens).into(),
        call.span.clone().into(),
        call.ty.clone().into(),
    ]))
}

tvm_ffi_dll_export_typed_func!(numsim_layout_linear_offsets, layout_linear_offsets);
tvm_ffi_dll_export_typed_func!(numsim_post_order_nodes, post_order_nodes);
tvm_ffi_dll_export_typed_func!(
    numsim_normalize_host_tensor_maps,
    normalize_host_tensor_maps
);
tvm_ffi_dll_export_typed_func!(numsim_semantic_ir_json, serialization::semantic_ir_json);
tvm_ffi_dll_export_typed_func!(numsim_ptx_call_parts, ptx_call_parts);
tvm_ffi_dll_export_typed_func!(numsim_ptx_payload_schema, decode::ptx::payload_schema);
tvm_ffi_dll_export_typed_func!(
    numsim_split_thresholds,
    emit::scaffold::SplitThresholds::defaults
);

/// Semantic errors are data; unrelated TVM FFI failures retain their native type.
fn failure_payload(failure: Failure) -> Result<Json> {
    Ok(match failure {
        Failure::Recorded => json_object(vec![
            ("kind", Json::from("not_covered")),
            (
                "message",
                Json::from("recorded emission failure escaped collection"),
            ),
        ]),
        Failure::Ffi(error) => return Err(error),
        Failure::Unsupported {
            message,
            unsupported,
        } => json_object(vec![
            ("kind", Json::from("unsupported")),
            ("message", Json::String(message)),
            ("unsupported", json_strings(unsupported)),
        ]),
        Failure::Unmodeled {
            target,
            message,
            span,
        } => json_object(vec![
            ("kind", Json::from("unmodeled")),
            ("target", Json::String(target)),
            ("message", Json::String(message)),
            ("span", span.unwrap_or(Json::Null)),
        ]),
        Failure::NotCovered(reason) => json_object(vec![
            ("kind", Json::from("not_covered")),
            ("message", Json::String(reason)),
        ]),
    })
}

fn payload_value(value: Json) -> Any {
    match value {
        Json::Null => Option::<String>::None.into(),
        Json::Bool(value) => value.into(),
        Json::Number(value) => match value.as_i64() {
            Some(value) => value.into(),
            None => value.as_f64().expect("numeric failure field").into(),
        },
        Json::String(value) => String::from(value).into(),
        Json::Array(values) => {
            Array::new(values.into_iter().map(payload_value).collect::<Vec<_>>()).into()
        }
        Json::Object(fields) => fields
            .into_iter()
            .map(|(key, value)| (String::from(key), payload_value(value)))
            .collect::<Map<String, Any>>()
            .into(),
    }
}

fn service_result<T: Into<Any>>(result: AResult<T>) -> Result<Map<String, Any>> {
    let (value, error) = match result {
        Ok(value) => (value.into(), Option::<String>::None.into()),
        Err(error) => (
            Option::<String>::None.into(),
            payload_value(failure_payload(error)?),
        ),
    };
    Ok([
        (String::from("value"), value),
        (String::from("error"), error),
    ]
    .into_iter()
    .collect())
}

/// The registry rows, contextual op paths and tile op names as JSON.
fn registry(schema: Map<String, Any>) -> AResult<String> {
    let schema = schema::Schema::parse(&schema)?;
    Ok(schema.registry_json().to_string().into())
}

fn analyze_kernels(
    context: &analyze::Ctx,
    funcs: &Array<PrimFunc>,
) -> AResult<Vec<analyze::frontend::KernelPlan>> {
    funcs
        .iter()
        .map(|func| analyze::frontend::analyze_primfunc(context, &func))
        .collect()
}

/// One module's host ABI, or the `HostAbiError` message that rejects it.
type HostAbi = std::result::Result<host_abi::Contract, std::string::String>;

/// `[module_manifest_json, source_nodes per kernel]` and the host ABI the
/// manifest determines.
fn module_manifest(plans: Vec<analyze::frontend::KernelPlan>) -> (Vec<Any>, HostAbi) {
    use analyze::util::Json;
    let mut kernels = Vec::new();
    let mut nodes: Vec<Any> = Vec::new();
    for plan in plans {
        kernels.push(plan.manifest);
        nodes.push(Array::new(plan.nodes).into());
    }
    let manifest = json_object(vec![("kernels", Json::Array(kernels))]);
    let contract = host_abi::contract(&manifest);
    (
        vec![
            String::from(manifest.to_string()).into(),
            Array::new(nodes).into(),
        ],
        contract,
    )
}

/// `[module_manifest_json, source_nodes per kernel, host ABI]` of one kernel
/// sequence.
fn analyze_module(funcs: Array<PrimFunc>, schema: Map<String, Any>) -> AResult<Array<Any>> {
    let schema = schema::Schema::parse(&schema)?;
    let context = analyze::Ctx::new(&schema)?;
    let plans = analyze_kernels(&context, &funcs)?;
    let (mut result, contract) = module_manifest(plans);
    result.push(String::from(host_abi::facts_json(&contract).as_str()).into());
    Ok(Array::new(result))
}

/// `host_abi.build_host_abi(spec)`: the host binding facts `manifest`
/// determines, or the `HostAbiError` that rejects it.
fn module_host_abi(manifest: String) -> AResult<String> {
    let parsed =
        serde_json::from_str::<analyze::util::Json>(manifest.as_str()).map_err(|error| {
            Error::new(
                VALUE_ERROR,
                &format!("invalid module spec manifest: {error}"),
                "",
            )
        })?;
    Ok(String::from(
        host_abi::facts_json(&host_abi::contract(&parsed)).as_str(),
    ))
}

/// `frontend._extract_topology(func, environment)`: one kernel's launch
/// topology under concrete scalar parameter values, as JSON.
fn launch_topology(
    func: PrimFunc,
    schema: Map<String, Any>,
    environment: Map<Var, Any>,
) -> AResult<String> {
    use analyze::topology::StaticValue;
    use tvm::tvm_ffi::TypeIndex;
    let schema = schema::Schema::parse(&schema)?;
    let context = analyze::Ctx::new(&schema)?;
    let mut values = analyze::util::IdMap::default();
    for (variable, value) in environment.iter() {
        let index = value.type_index();
        let value = if index == TypeIndex::kTVMFFIBool as i32 {
            StaticValue::Bool(bool::try_from(value)?)
        } else if index == TypeIndex::kTVMFFIInt as i32 {
            StaticValue::Int(i128::from(i64::try_from(value)?))
        } else if index == TypeIndex::kTVMFFIFloat as i32 {
            StaticValue::Float(f64::try_from(value)?)
        } else {
            return Err(analyze::util::ffi_error(
                "launch scalar values must be bool, integer or float",
            )
            .into());
        };
        values.insert(analyze::util::oref(variable), value);
    }
    let topology = analyze::topology::extract_topology(&context, &func.body(), values)?;
    Ok(topology.json().to_string().into())
}

/// Analyze and emit one kernel sequence: `[module_manifest_json, source_nodes
/// per kernel, host ABI, module template, emission failure]`. A
/// sequence with an unsupported entry is not emitted. An emission failure is
/// returned as structured data, so the caller reports
/// verification and host ABI errors first.
fn compile_module(
    funcs: Array<PrimFunc>,
    schema: Map<String, Any>,
    flags: Map<String, Any>,
) -> AResult<Array<Any>> {
    let flag_bool = |key: &str| -> Result<bool> {
        match flags.get(&String::from(key))? {
            Some(value) => Ok(bool::try_from(value)?),
            None => Ok(false),
        }
    };
    let flag_str = |key: &str| -> Result<Option<String>> {
        match flags.get(&String::from(key))? {
            Some(value) => Ok(Some(String::try_from(value)?)),
            None => Ok(None),
        }
    };
    let abi_version = match flags.get(&String::from("numsim_abi_version"))? {
        Some(value) => i64::try_from(value)?,
        None => return Err(Error::new(VALUE_ERROR, "missing numsim_abi_version flag", "").into()),
    };
    let analysis_checker = flag_str("analysis_checker")?
        .map(|value| value.as_str().to_owned())
        .filter(|value| !value.is_empty());
    let Some(split_thresholds) = flags.get(&String::from("split_thresholds"))? else {
        return Err(Error::new(VALUE_ERROR, "missing split_thresholds flag", "").into());
    };
    let split_thresholds = emit::scaffold::SplitThresholds::parse(split_thresholds.try_into()?)?;
    let request = emit::module::ModuleRequest {
        analysis_capable: flag_bool("analysis_capable")?,
        analysis_checker,
        abi_version,
        split_thresholds,
    };
    let schema = schema::Schema::parse(&schema)?;
    let context = analyze::Ctx::new(&schema)?;
    let plans = analyze_kernels(&context, &funcs)?;
    let mut emitted: Option<std::string::String> = None;
    let mut failure: Option<Any> = None;
    if plans.iter().all(|plan| plan.supported) {
        match emit::module::emit_rust_module(&context, &plans, &request) {
            Ok(source) => emitted = Some(source),
            Err(error) => failure = Some(payload_value(failure_payload(error)?)),
        }
    }
    let (mut result, contract) = module_manifest(plans);
    let template = emitted.map(|source| String::from(source.as_str()));
    result.push(String::from(host_abi::facts_json(&contract).as_str()).into());
    result.push(template.into());
    result.push(failure.unwrap_or_else(|| Option::<String>::None.into()));
    Ok(Array::new(result))
}

macro_rules! export_service {
    ($name:ident, $function:ident, $($argument:ident: $type:ty),*) => {
        fn $name($($argument: $type),*) -> Result<Map<String, Any>> {
            service_result($function($($argument),*))
        }
        tvm_ffi_dll_export_typed_func!($name, $name);
    };
}

export_service!(numsim_registry, registry, schema: Map<String, Any>);
export_service!(numsim_analyze_module, analyze_module, funcs: Array<PrimFunc>, schema: Map<String, Any>);
export_service!(numsim_launch_topology, launch_topology, func: PrimFunc, schema: Map<String, Any>, environment: Map<Var, Any>);
export_service!(numsim_compile_module, compile_module, funcs: Array<PrimFunc>, schema: Map<String, Any>, flags: Map<String, Any>);
export_service!(numsim_host_abi, module_host_abi, manifest: String);
tvm_ffi_dll_export_typed_func!(numsim_frontend_identity, identity);
