//! Native kernel-body emission.
//!
//! The text snapshots pin the emitted text, so temp-name counters, capture
//! attempts and restore points keep their order.

use tvm::ir::{Expr, Var};
use tvm::tirx::{BufferVar, PrimFunc};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Map, ObjectIdentity, ObjectRefCore};

pub mod abi;
pub mod async_copy;
pub mod atomic_bulk;
pub mod calls;
pub mod clc;
pub mod control;
pub mod cuda_helper;
pub mod diagnostics;
pub mod expr;
pub mod layout;
pub mod matrix;
pub mod matrix_variants;
pub mod memory_ops;
pub mod memory_support;
pub mod module;
pub mod module_template;
pub mod ptx_addr;
pub mod ptx_address;
pub mod ptx_arithmetic;
pub mod ptx_bit_carrier;
pub mod ptx_bitops;
pub mod ptx_cache_policy;
pub mod ptx_clmad;
pub mod ptx_compare;
pub mod ptx_cvt;
pub mod ptx_cvt_pack;
pub mod ptx_cvt_variants;
pub mod ptx_integer_arithmetic;
pub mod ptx_lop3;
pub mod ptx_minmax;
pub mod ptx_move;
pub mod ptx_set_packed;
pub mod ptx_spdecompress;
pub mod ptx_unary;
pub mod ptx_warp;
pub mod pure;
pub mod raw_memory;
pub mod raw_tcgen;
pub mod raw_tma;
pub mod register_call;
pub mod scaffold;
pub mod shared_blocks;
pub mod stmt;
pub mod sync;
pub mod tcgen05;
pub mod tcgen_descriptor;
pub mod tile;
pub mod tile_async_copy;
pub mod tile_common;
pub mod tile_gemm_async;
pub mod tile_tcgen05;
pub mod tmem;
pub mod wait_until;

use super::analyze::buffers::BufferBindings;
use super::analyze::expression_bindings::ExpressionBindings;
use super::analyze::frontend::KernelPlan;
use super::analyze::memory::{bind_map, MemoryPlan};
use super::analyze::shapes::Scalar;
use super::analyze::util::{
    buffer_name, dtype_text, ffi_text, not_covered, oref, prim_dtype, repr_of, unsupported,
    AResult, IdMap, IdSet,
};
use super::analyze::Ctx;
use crate::tables::{
    dtype_byte_len, is_integer_rust_type, json_string, phase_binding_name,
    stmt_rust_scalar_by_dtype,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Uniformity {
    Uniform,
    Varying,
}

pub fn join_uniformity<I: IntoIterator<Item = Uniformity>>(values: I) -> Uniformity {
    values.into_iter().max().unwrap_or(Uniformity::Uniform)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlProvenance {
    None,
    ElectSync,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RustValue {
    pub code: String,
    pub rust_type: String,
    pub uniformity: Uniformity,
    pub is_mask: bool,
    pub requires_statement: bool,
    pub quantized_dtype: Option<String>,
    pub control_provenance: ControlProvenance,
}

impl RustValue {
    pub fn new(
        code: impl Into<String>,
        rust_type: impl Into<String>,
        uniformity: Uniformity,
    ) -> Self {
        Self {
            code: code.into(),
            rust_type: rust_type.into(),
            uniformity,
            is_mask: false,
            requires_statement: false,
            quantized_dtype: None,
            control_provenance: ControlProvenance::None,
        }
    }

    pub fn mask(code: impl Into<String>) -> Self {
        Self {
            is_mask: true,
            ..Self::new(code, "bool", Uniformity::Varying)
        }
    }
}

pub fn join_control_provenance<'a, I: IntoIterator<Item = &'a RustValue>>(
    values: I,
) -> ControlProvenance {
    if values
        .into_iter()
        .any(|value| value.control_provenance == ControlProvenance::ElectSync)
    {
        ControlProvenance::ElectSync
    } else {
        ControlProvenance::None
    }
}

/// Variable bindings in deterministic capture order.
pub type Variables = IdMap<RustValue>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SplitArgument {
    pub name: String,
    pub rust_type: String,
    pub call_code: String,
    pub snapshot_before_control: bool,
}

#[derive(Clone)]
pub struct SplitEffects {
    pub outer_variables: Variables,
    pub outer_dynamic_references: Vec<String>,
    pub outer_loop_live_masks: Vec<String>,
    pub borrow_non_copy: bool,
    /// In insertion order.
    pub arguments: Vec<SplitArgument>,
    pub variable_replacements: Variables,
    pub used_dynamic_references: Vec<String>,
    pub loop_live_mask_replacements: Vec<(String, String)>,
    pub allow_outer_loop_live_mutation: bool,
    pub suspend_count: usize,
    pub unsafe_reason: Option<String>,
}

impl SplitEffects {
    pub fn argument(&self, name: &str) -> Option<&SplitArgument> {
        self.arguments.iter().find(|argument| argument.name == name)
    }

    pub fn set_argument(&mut self, argument: SplitArgument) {
        match self
            .arguments
            .iter_mut()
            .find(|existing| existing.name == argument.name)
        {
            Some(existing) => *existing = argument,
            None => self.arguments.push(argument),
        }
    }

    pub fn note_unsafe(&mut self, reason: impl Into<String>) {
        if self.unsafe_reason.is_none() {
            self.unsafe_reason = Some(reason.into());
        }
    }
}

#[derive(Clone)]
pub struct SplitHelper {
    pub name: String,
    pub is_async: bool,
    pub inline_never: bool,
    pub arguments: Vec<SplitArgument>,
    pub body: Vec<String>,
}

/// Emission mode flags.
#[derive(Clone)]
pub struct EmitOptions {
    pub collect_errors: bool,
    pub analysis_capable: bool,
    pub split_thresholds: scaffold::SplitThresholds,
}

/// Per-kernel binding records; module emission reads the names.
pub struct BufferCode {
    pub field: String,
    pub storage_index: Option<usize>,
    pub plan: usize,
    pub input_name: Option<String>,
    pub shape: Vec<Scalar>,
}

pub struct ShapeVarCode {
    pub variable: Var,
    pub name: String,
    pub field: String,
    pub rust_type: String,
    pub sources: Vec<(String, usize)>,
}

pub struct ScalarCode {
    pub variable: Var,
    pub name: String,
    pub binding_name: String,
    pub field: String,
    pub dtype: String,
    pub rust_type: String,
}

pub struct PointerCode {
    pub variable: Var,
    pub binding_name: String,
    pub field: String,
    pub dtype: String,
    pub itemsize: i64,
}

pub struct Emitter<'a> {
    pub diagnostics: diagnostics::State,
    pub ctx: &'a Ctx<'a>,
    pub plan: &'a KernelPlan,
    pub func: &'a PrimFunc,
    pub params: Vec<Var>,
    pub consumed_address_wrappers: IdSet,
    /// Sources whose lowering may await.
    pub suspend_source_nodes: IdSet,
    pub buffer_bindings: &'a BufferBindings,
    pub bindings: Map<Var, Expr>,
    pub memory_plan: &'a MemoryPlan,
    pub expression_bindings: &'a ExpressionBindings,
    /// One serialized entry per node identity.
    pub op_ids: IdMap<i64>,
    pub kernel_index: i64,
    pub analysis_capable: bool,
    pub split_thresholds: scaffold::SplitThresholds,
    pub raw_tma: raw_tma::State,
    pub raw_tcgen: raw_tcgen::State,
    pub memory_candidates: memory_support::CandidateState,
    pub sync: sync::State,
    pub tmem: tmem::State<'a>,
    pub warps_per_warpgroup: i64,
    pub warps_per_cta: i64,
    pub threads_per_warpgroup: i64,
    pub scalars: Vec<ScalarCode>,
    pub pointers: Vec<PointerCode>,
    pub buffers: Vec<(BufferVar, BufferCode)>,
    pub shape_vars: Vec<ShapeVarCode>,
    pub lines: Vec<String>,
    pub indent: usize,
    pub variables: Variables,
    pub next_control: i64,
    pub next_temp: i64,
    pub next_split_helper: i64,
    pub next_sync_helper: i64,
    pub next_root_async_helper: i64,
    pub next_inner_if_helper: i64,
    pub next_inner_varying_if_helper: i64,
    pub next_inner_for_helper: i64,
    pub next_inner_async_helper: i64,
    pub split_helpers: Vec<SplitHelper>,
    /// Static global buffer slots written while lowering; None means unknown.
    /// Monotone across speculative emission: extra slots only retain more history.
    pub written_global_buffers: Option<std::collections::BTreeSet<usize>>,
    pub written_global_parameters: std::collections::BTreeSet<usize>,
    pub written_tensor_maps: std::collections::BTreeSet<usize>,
    pub pointer_targets: Option<crate::analyze::pointer_targets::PointerTargets<'a>>,
    pub recorded_store_count: usize,
    pub control_depth: usize,
    pub split_arm_depth: usize,
    pub loop_live_masks: Vec<String>,
    pub loop_depth: usize,
    pub outer_loop_live_split_permissions: Vec<bool>,
    pub nested_for_statement_split_permissions: Vec<bool>,
    pub capturing_for_statements: IdSet,
    pub dynamic_buffers: Vec<(BufferVar, String)>,
    pub logical_buffer_name_cache: Vec<(BufferVar, String)>,
    pub loop_transfer_cache: IdMap<bool>,
    pub subtree_may_suspend_cache: IdMap<bool>,
    pub failed_inner_uniform_if_splits: IdSet,
    pub failed_inner_async_splits: IdMap<Vec<ObjectIdentity>>,
    pub split_effects: Vec<SplitEffects>,
    pub selected_lane: Option<String>,
    pub register_access_mask: Option<String>,
    /// The buffer loader override (`None` = the exact loader).
    pub nested_load_site: Option<NestedLoadSite>,
    pub nonnegative_floor: bool,
    pub call_expr_stack: Vec<ObjectRef>,
    pub integer_error_context: &'static str,
    pub integer_error_label: Option<String>,
    /// Ordinary numerical artifacts call typed `frontend_expr` and local-memory helpers.
    pub use_typed_helpers: bool,
}

impl<'a> Emitter<'a> {
    /// An emitter for plain (uninstrumented, analysis-free) emission.
    pub fn new(
        ctx: &'a Ctx<'a>,
        plan: &'a KernelPlan,
        kernel_index: i64,
        kernel_count: i64,
        options: &EmitOptions,
    ) -> AResult<Self> {
        let func = &plan.func;
        let nodes = &plan.nodes;
        let buffer_bindings = &plan.buffer_bindings;
        let bindings = bind_map(&plan.statements);
        let params: Vec<Var> = func.params.iter().collect();
        let warps_per_warpgroup = plan.topology.warps_per_warpgroup;
        let warps_per_cta = plan.topology.warps_per_cta;
        // `LaunchTopology.threads_per_warpgroup` is a property, not a manifest field.
        let threads_per_warpgroup = warps_per_warpgroup * 32;
        let mut op_ids = IdMap::default();
        let mut suspend_source_nodes = IdSet::default();
        for (op_id, node) in nodes.iter().enumerate() {
            op_ids.insert(node.clone(), op_id as i64);
            if let Some(name) = crate::decode::call_name(node)? {
                if ctx
                    .schema
                    .registered_ops
                    .get(&name)
                    .is_some_and(|row| row.suspends)
                {
                    suspend_source_nodes.add(node.clone());
                }
            }
        }
        let consumed_address_wrappers = ptx_addr::consumed_calls(ctx, nodes)?;
        // The analysis recorded these failures as unsupported entries.
        if let Some((message, items)) = &plan.analysis_failure {
            return Err(super::analyze::util::Failure::Unsupported {
                message: message.clone(),
                unsupported: items.clone(),
            });
        }
        let (Some(memory_plan), Some(expression_bindings)) =
            (&plan.memory_plan, &plan.expression_bindings)
        else {
            return not_covered("kernel analysis has no memory plan");
        };
        let parameter_at = |index: i64| -> AResult<Var> {
            match usize::try_from(index).ok().and_then(|at| params.get(at)) {
                Some(parameter) => Ok(parameter.clone()),
                None => not_covered("manifest parameter index is out of range"),
            }
        };
        let raw_tma = raw_tma::State::new(plan, &params, kernel_index, kernel_count)?;
        let mut scalars = Vec::new();
        for (index, entry) in plan.scalars.iter().enumerate() {
            let dtype = entry.dtype.clone();
            let rust_type = match stmt_rust_scalar_by_dtype(&dtype) {
                Some(rust_type) => rust_type,
                // Global analysis already rejects these parameter dtypes. A
                // placeholder lets instruction checks report their own errors.
                None if options.collect_errors => "u64",
                None => {
                    return not_covered(format!(
                        "scalar parameter dtype {dtype} has no Rust spelling"
                    ))
                }
            };
            scalars.push(ScalarCode {
                variable: parameter_at(entry.parameter_index)?,
                binding_name: phase_binding_name(kernel_index, &entry.name, kernel_count),
                name: entry.name.clone(),
                field: format!("scalar_{index}"),
                dtype,
                rust_type: rust_type.to_owned(),
            });
        }
        let mut pointers = Vec::new();
        for (index, entry) in plan.pointers.iter().enumerate() {
            pointers.push(PointerCode {
                variable: parameter_at(entry.parameter_index)?,
                binding_name: phase_binding_name(kernel_index, &entry.name, kernel_count),
                field: format!("pointer_{index}"),
                itemsize: dtype_byte_len(ctx.schema, &entry.dtype)?,
                dtype: entry.dtype.clone(),
            });
        }
        let mut buffers: Vec<(BufferVar, BufferCode)> = Vec::new();
        let mut next_buffer_storage_index = 0;
        for (index, plan) in memory_plan.buffers.iter().enumerate() {
            let mut shape = Vec::new();
            for extent in &plan.layout.shape {
                shape.push(match extent {
                    super::analyze::layout::Extent::Static(value) => Scalar::Int(*value),
                    super::analyze::layout::Extent::Dynamic(expr) => expression_bindings
                        .shape_expressions
                        .resolve(&ctx.analyzer, &Scalar::Expr(expr.clone()), &buffer_bindings)?,
                });
            }
            let storage_index = if plan.dynamic_data_var.is_some() {
                None
            } else {
                Some(next_buffer_storage_index)
            };
            buffers.push((
                plan.buffer.clone(),
                BufferCode {
                    field: format!("buffer_{index}"),
                    storage_index,
                    plan: index,
                    input_name: if plan.is_parameter {
                        Some(phase_binding_name(kernel_index, &plan.name, kernel_count))
                    } else {
                        None
                    },
                    shape,
                },
            ));
            if plan.dynamic_data_var.is_none() {
                next_buffer_storage_index += 1;
            }
        }
        let mut shape_sources: IdMap<Vec<(String, usize)>> = IdMap::default();
        let mut shape_types: IdMap<String> = IdMap::default();
        let mut shape_order: Vec<Var> = Vec::new();
        for (buffer, code) in &buffers {
            if !memory_plan.buffers[code.plan].is_parameter {
                continue;
            }
            for (axis, extent) in code.shape.iter().enumerate() {
                let Scalar::Expr(extent) = extent else {
                    continue;
                };
                let extent_ref = oref(extent.clone());
                let Some(variable) = super::analyze::util::as_var(&extent_ref) else {
                    continue;
                };
                let axis_extent = buffer.buffer_type().shape.get(axis)?;
                let dtype = prim_dtype(&axis_extent.ty)
                    .map(dtype_text)
                    .unwrap_or_default();
                let rust_type = stmt_rust_scalar_by_dtype(&dtype);
                if !rust_type.is_some_and(is_integer_rust_type) {
                    return unsupported(format!(
                        "Runtime shape scalar {} has unsupported dtype {dtype}",
                        repr_of(&variable)?
                    ));
                }
                let rust_type = rust_type.unwrap().to_owned();
                if !shape_sources.contains(&extent_ref) {
                    shape_sources.insert(extent_ref.clone(), Vec::new());
                    shape_order.push(variable.clone());
                }
                if !shape_types.contains(&extent_ref) {
                    shape_types.insert(extent_ref.clone(), rust_type.clone());
                }
                if shape_types.get(&extent_ref) != Some(&rust_type) {
                    return unsupported(format!(
                        "Runtime shape scalar {} has inconsistent dtypes",
                        repr_of(&variable)?
                    ));
                }
                let source = (
                    code.input_name
                        .clone()
                        .unwrap_or_else(|| memory_plan.buffers[code.plan].name.clone()),
                    axis,
                );
                let sources = shape_sources.get_mut(&extent_ref).expect("shape sources");
                if !sources.contains(&source) {
                    sources.push(source);
                }
            }
        }
        let mut shape_vars = Vec::new();
        for (index, variable) in shape_order.iter().enumerate() {
            let key = oref(variable.clone());
            shape_vars.push(ShapeVarCode {
                variable: variable.clone(),
                name: ffi_text(&variable.name),
                field: format!("shape_{index}"),
                rust_type: shape_types.get(&key).expect("shape type").clone(),
                sources: shape_sources.get(&key).expect("shape sources").clone(),
            });
        }
        let mut variables = Variables::default();
        for shape in &shape_vars {
            variables.set(
                oref(shape.variable.clone()),
                RustValue::new(
                    format!("buffers.{} as {}", shape.field, shape.rust_type),
                    shape.rust_type.clone(),
                    Uniformity::Uniform,
                ),
            );
        }
        for scalar in &scalars {
            variables.set(
                oref(scalar.variable.clone()),
                RustValue::new(
                    format!("buffers.{}", scalar.field),
                    scalar.rust_type.clone(),
                    Uniformity::Uniform,
                ),
            );
        }
        for pointer in &pointers {
            variables.set(
                oref(pointer.variable.clone()),
                RustValue::new(
                    format!("buffers.{}.clone()", pointer.field),
                    "PhysicalPtr",
                    Uniformity::Uniform,
                ),
            );
        }
        Ok(Self {
            ctx,
            plan,
            func,
            params,
            diagnostics: diagnostics::State {
                enabled: options.collect_errors,
                ..Default::default()
            },
            consumed_address_wrappers,
            suspend_source_nodes,
            buffer_bindings,
            bindings,
            memory_plan,
            expression_bindings,
            op_ids,
            kernel_index,
            analysis_capable: options.analysis_capable,
            split_thresholds: options.split_thresholds.clone(),
            raw_tma,
            raw_tcgen: raw_tcgen::State::default(),
            memory_candidates: memory_support::CandidateState::default(),
            sync: sync::State::default(),
            tmem: tmem::State::new(plan),
            warps_per_warpgroup,
            warps_per_cta,
            threads_per_warpgroup,
            scalars,
            pointers,
            buffers,
            shape_vars,
            lines: Vec::new(),
            indent: 1,
            variables,
            next_control: 0,
            next_temp: 0,
            next_split_helper: 0,
            next_sync_helper: 0,
            next_root_async_helper: 0,
            next_inner_if_helper: 0,
            next_inner_varying_if_helper: 0,
            next_inner_for_helper: 0,
            next_inner_async_helper: 0,
            split_helpers: Vec::new(),
            written_global_buffers: options.analysis_capable.then(Default::default),
            written_global_parameters: Default::default(),
            written_tensor_maps: Default::default(),
            pointer_targets: None,
            recorded_store_count: 0,
            control_depth: 0,
            split_arm_depth: 0,
            loop_live_masks: Vec::new(),
            loop_depth: 0,
            outer_loop_live_split_permissions: Vec::new(),
            nested_for_statement_split_permissions: Vec::new(),
            capturing_for_statements: IdSet::default(),
            dynamic_buffers: Vec::new(),
            logical_buffer_name_cache: Vec::new(),
            loop_transfer_cache: IdMap::default(),
            subtree_may_suspend_cache: IdMap::default(),
            failed_inner_uniform_if_splits: IdSet::default(),
            failed_inner_async_splits: IdMap::default(),
            split_effects: Vec::new(),
            selected_lane: None,
            register_access_mask: None,
            nested_load_site: None,
            nonnegative_floor: false,
            call_expr_stack: Vec::new(),
            integer_error_context: "engine",
            integer_error_label: None,
            use_typed_helpers: !options.analysis_capable,
        })
    }

    pub fn emit(&mut self) -> AResult<String> {
        let body = self.func.body().clone();
        self.emit_stmt(&body, None)?;
        Ok(self.lines.join("\n"))
    }

    pub fn emit_line(&mut self, line: &str) {
        if line.is_empty() {
            self.lines.push(String::new());
        } else {
            self.lines
                .push(format!("{}{line}", "    ".repeat(self.indent)));
        }
    }

    pub fn emit_suspend_line(&mut self, line: &str) {
        self.mark_suspend();
        self.emit_line(line);
    }

    pub fn uniform_i64(&mut self, expr: &ObjectRef, label: &str) -> AResult<RustValue> {
        let value = self.emit_expr(expr)?;
        let value = self.as_i64(value)?;
        if value.uniformity == Uniformity::Uniform {
            return Ok(value);
        }
        if value.is_mask {
            return unsupported(format!("{label} cannot scalarize a lane mask"));
        }
        let scalar = self.control_name("uniform_i64");
        self.emit_line(&format!(
            "let {scalar} = require_uniform_i64(&{}, ctx.active_mask(), {})?;",
            value.code,
            json_string(label)
        ));
        Ok(RustValue::new(scalar, "i64", Uniformity::Uniform))
    }

    pub fn emit_write_call(&mut self, invocation: &str) {
        self.emit_line(&format!("{invocation}?;"));
    }

    pub fn control_name(&mut self, prefix: &str) -> String {
        let value = format!("{prefix}_{}", self.next_control);
        self.next_control += 1;
        value
    }

    pub fn temp(&mut self, prefix: &str) -> String {
        let value = format!("{prefix}_{}", self.next_temp);
        self.next_temp += 1;
        value
    }

    pub fn synthetic_site_id(&mut self) -> u64 {
        let source_op_id = (1u64 << 63) + self.next_control as u64;
        self.next_control += 1;
        source_op_id
    }

    pub fn v2_site(&mut self, source_op_id: Option<i64>) -> String {
        let id = match source_op_id {
            Some(id) => id as u64,
            None => self.synthetic_site_id(),
        };
        abi::site(id)
    }

    /// `static_op_id`: the exact source-map identity of one static IR occurrence.
    pub fn static_op_id(&self, node: &ObjectRef) -> AResult<i64> {
        match self.op_ids.get(node) {
            Some(op_id) => Ok(*op_id),
            None => Err(super::analyze::util::Failure::Ffi(
                super::analyze::util::ffi_error(&format!(
                    "generated Rust provenance could not resolve an exact serialized TIRx source entry for {}",
                    super::analyze::util::kind(node).unwrap_or("node")
                )),
            )),
        }
    }

    /// `lowered_instruction_site`.
    pub fn lowered_instruction_site(&mut self, node: &ObjectRef) -> i64 {
        match self.op_ids.get(node) {
            Some(op_id) => *op_id,
            None => self.synthetic_site_id() as i64,
        }
    }

    pub fn buffer_code(&self, buffer: &BufferVar) -> AResult<&BufferCode> {
        for (candidate, code) in &self.buffers {
            if super::analyze::util::same(candidate.as_var(), buffer.as_var()) {
                return Ok(code);
            }
        }
        let resolved = self.memory_plan.resolve(buffer)?;
        for (_, code) in &self.buffers {
            if std::ptr::eq(&self.memory_plan.buffers[code.plan], resolved) {
                return Ok(code);
            }
        }
        unsupported(format!(
            "Buffer is not a PrimFunc parameter: {}",
            buffer_name(buffer)
        ))
    }

    pub fn plan_of(&self, code: &BufferCode) -> &super::analyze::memory::BufferPlan {
        &self.memory_plan.buffers[code.plan]
    }

    /// `variables[variable]`: the split-aware read.
    pub fn lookup_variable(&mut self, variable: &ObjectRef) -> Option<RustValue> {
        let value = self.variables.get(variable)?.clone();
        Some(self.resolve_split_variable(variable, value))
    }

    /// The variable scope state restored on exit.
    pub fn scope_snapshot(&self) -> (Variables, Vec<(BufferVar, String)>) {
        (self.variables.clone(), self.dynamic_buffers.clone())
    }

    pub fn restore_scope(&mut self, snapshot: (Variables, Vec<(BufferVar, String)>)) {
        self.variables = snapshot.0;
        self.dynamic_buffers = snapshot.1;
    }

    /// Lower a validated pure integer expression with a fresh emitter over the
    /// host bindings.
    pub fn emit_host_integer(
        &mut self,
        expression: &ObjectRef,
        host_variables: &Variables,
        field: &str,
    ) -> AResult<(Vec<String>, RustValue)> {
        validate_integer_expression(
            expression,
            &|variable| host_variables.contains(variable),
            field,
        )?;
        let saved_lines = std::mem::take(&mut self.lines);
        let saved_variables = std::mem::replace(&mut self.variables, host_variables.clone());
        let saved_temp = self.next_temp;
        let saved_indent = self.indent;
        self.next_temp = 0;
        self.indent = 0;
        self.integer_error_context = "python";
        self.integer_error_label = Some(format!("invalid {field}"));
        let result = self.emit_expr(expression);
        self.integer_error_context = "engine";
        self.integer_error_label = None;
        self.indent = saved_indent;
        self.next_temp = saved_temp;
        self.variables = saved_variables;
        let lines = std::mem::replace(&mut self.lines, saved_lines);
        let value = result?;
        if value.uniformity != Uniformity::Uniform || !is_integer_rust_type(&value.rust_type) {
            return unsupported(format!(
                "{field} did not lower to a uniform integer Rust value"
            ));
        }
        Ok((lines, value))
    }
}

/// Validate the pure integer subset used by host expression emission.
fn validate_integer_expression(
    expression: &ObjectRef,
    variable_allowed: &impl Fn(&ObjectRef) -> bool,
    field: &str,
) -> AResult<()> {
    let kind = super::analyze::util::kind(expression).unwrap_or("");
    let dtype = super::analyze::util::expr_type(expression)
        .and_then(|ty| prim_dtype(&ty))
        .map(dtype_text)
        .unwrap_or_default();
    if !(dtype.starts_with("int") || dtype.starts_with("uint")) {
        return unsupported(format!(
            "{field} must be an integer expression, got dtype {:?}",
            &dtype
        ));
    }
    if kind == "IntImm" {
        return Ok(());
    }
    if kind == "Var" {
        if !variable_allowed(expression) {
            return unsupported(format!(
                "{field} references an unavailable scalar variable {}",
                super::analyze::util::repr_text(expression)?
            ));
        }
        return Ok(());
    }
    if let Some(cast) = expression.as_node::<tvm::prim::CastObj>() {
        return validate_integer_expression(&oref(cast.value.clone()), variable_allowed, field);
    }
    if !crate::schema::INTEGER_BINARY_NODE_KINDS.contains(&kind) {
        return unsupported(format!("{field} uses unsupported integer operation {kind}"));
    }
    let (a, b) = super::analyze::topology_operands(expression).expect("binary node");
    validate_integer_expression(&a, variable_allowed, field)?;
    validate_integer_expression(&b, variable_allowed, field)
}

/// The buffer loader swapped in around one emission.
#[derive(Clone)]
pub enum NestedLoadSite {
    /// `emit_buffer_load`: the load's own source node (the default loader,
    /// handed over explicitly).
    Exact,
    Site(Option<i64>),
    ByBuffer(i64),
    /// `emit_buffer_load_at_lane(buffer, indices, lane, source_node=<load>)`.
    AtLane(String),
    /// `emit_(implicit_)buffer_load_at_lane(buffer, indices, lane, source_op_id=...)`.
    AtLaneSite(String, Option<i64>),
    /// Reject a runtime load while capturing a pure closure.
    Reject(String),
}
