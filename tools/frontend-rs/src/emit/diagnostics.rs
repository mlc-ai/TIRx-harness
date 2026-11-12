//! Collect source-located failures during the ordinary emission walk.

use crate::tvm_compat::TryCast;
use tvm_tirx::ir::{CallObj, Expr};
use tvm_tirx::ir::prim::LetObj;
use tvm_tirx::ir::{BindObj, Evaluate, ForObj, RegionStmtObj, Stmt, WhileObj};
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::{
    structural_visit, Any, Array, ObjectRefCore, Result, AnyView, StructuralVisitorView, VisitInterrupt,
};

use super::{EmitOptions, Emitter};
use crate::analyze::frontend::KernelPlan;
use crate::analyze::util::{as_var, oref, same, AResult, Failure, IdSet};
use crate::analyze::Ctx;

#[derive(Default)]
pub struct State {
    pub enabled: bool,
    pub unsupported: Vec<(i64, String)>,
    pub invalid_values: IdSet,
    pub visited_calls: IdSet,
    pub(super) visited_statements: IdSet,
}

// Stop at direct executed children, without entering metadata or region attributes.
struct ChildVisitor {
    root: ObjectRef,
    children: Vec<ObjectRef>,
}

#[tvm_ffi::dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl ChildVisitor {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(node) = value.downcast_ref::<ObjectRef>() {
            if same(node, &self.root) {
                return visitor.default_visit(self, value);
            }
            if value.downcast_ref::<Expr>().is_some() || value.downcast_ref::<Stmt>().is_some() {
                self.children.push(node.clone());
                return Ok(None);
            }
        }
        if value.downcast_ref::<Array<Any>>().is_some() {
            return visitor.default_visit(self, value);
        }
        Ok(None)
    }
}

fn statement_children(stmt: &Stmt) -> Result<Vec<ObjectRef>> {
    if let Some(region) = stmt.downcast_ref::<RegionStmtObj>() {
        let mut children: Vec<ObjectRef> = region.args.iter().map(oref).collect();
        children.push(oref(region.body.clone()));
        return Ok(children);
    }
    let mut visitor = ChildVisitor {
        root: oref(stmt.clone()),
        children: Vec::new(),
    };
    structural_visit(stmt, &mut visitor)?;
    Ok(visitor.children)
}

#[derive(Default)]
struct ExpressionVisitor {
    calls: Vec<Expr>,
    bindings: Vec<ObjectRef>,
    seen: IdSet,
}

#[tvm_ffi::dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl ExpressionVisitor {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(expr) = value.downcast_ref::<Expr>().cloned() {
            let node = oref(expr.clone());
            if !self.seen.add(node.clone()) || as_var(&node).is_some() {
                return Ok(None);
            }
            if let Some(binding) = expr.downcast_ref::<LetObj>() {
                self.bindings.push(oref(binding.var.clone()));
            }
            if let Some(interrupt) = visitor.default_visit(self, value)? {
                return Ok(Some(interrupt));
            }
            if expr.downcast_ref::<CallObj>().is_some() {
                self.calls.push(expr);
            }
        } else if value.downcast_ref::<Array<Any>>().is_some() {
            return visitor.default_visit(self, value);
        }
        Ok(None)
    }
}

fn expression_calls(expression: &ObjectRef) -> Result<(Vec<Expr>, Vec<ObjectRef>)> {
    let mut visitor = ExpressionVisitor::default();
    structural_visit(expression, &mut visitor)?;
    Ok((visitor.calls, visitor.bindings))
}

impl<'a> Emitter<'a> {
    pub(super) fn record_unsupported(&mut self, node: &ObjectRef, message: String) {
        let op_id = self.op_ids.get(node).copied().unwrap_or(-1);
        if !self
            .diagnostics
            .unsupported
            .iter()
            .any(|item| item.0 == op_id && item.1 == message)
        {
            self.diagnostics.unsupported.push((op_id, message));
        }
    }

    pub fn emit_stmt(&mut self, stmt: &Stmt, previous: Option<&Stmt>) -> AResult<()> {
        if !self.diagnostics.enabled {
            return self.emit_stmt_inner(stmt, previous);
        }
        self.diagnostics.visited_statements.add(oref(stmt.clone()));
        let checkpoint = self.partition_checkpoint();
        let lines = self.lines.len();
        let indent = self.indent;
        let control_depth = self.control_depth;
        let split_arm_depth = self.split_arm_depth;
        let loop_depth = self.loop_depth;
        let loop_live_masks = self.loop_live_masks.clone();
        let outer_permissions = self.outer_loop_live_split_permissions.clone();
        let nested_permissions = self.nested_for_statement_split_permissions.clone();
        let call_depth = self.call_expr_stack.len();
        let selected_lane = self.selected_lane.clone();
        let register_access_mask = self.register_access_mask.clone();
        let nested_load_site = self.nested_load_site.clone();
        let node = oref(stmt.clone());
        match self.emit_stmt_inner(stmt, previous) {
            Ok(()) => return Ok(()),
            Err(Failure::Unsupported { message, .. }) => self.record_unsupported(&node, message),
            Err(Failure::Recorded) => {}
            Err(error) => return Err(crate::analyze::frontend::with_node_span(error, &node)),
        }
        let restore = |emitter: &mut Self| {
            emitter.restore_failed_partition(&checkpoint);
            emitter.lines.truncate(lines);
            emitter.indent = indent;
            emitter.control_depth = control_depth;
            emitter.split_arm_depth = split_arm_depth;
            emitter.loop_depth = loop_depth;
            emitter.loop_live_masks = loop_live_masks.clone();
            emitter.outer_loop_live_split_permissions = outer_permissions.clone();
            emitter.nested_for_statement_split_permissions = nested_permissions.clone();
            emitter.call_expr_stack.truncate(call_depth);
            emitter.selected_lane = selected_lane.clone();
            emitter.register_access_mask = register_access_mask.clone();
            emitter.nested_load_site = nested_load_site.clone();
        };
        restore(self);
        if let Some(bind) = stmt.downcast_ref::<BindObj>() {
            self.diagnostics.invalid_values.add(oref(bind.var.clone()));
        }
        let result = self.emit_unvisited_children(stmt);
        restore(self);
        result
    }

    fn emit_unvisited_children(&mut self, stmt: &Stmt) -> AResult<()> {
        let invalid_values = self.diagnostics.invalid_values.clone();
        if let Some(loop_stmt) = stmt.downcast_ref::<ForObj>() {
            // Its bounds failed, so the loop variable has no emitted value.
            self.diagnostics
                .invalid_values
                .add(oref(loop_stmt.var.clone()));
        }
        let is_loop = stmt.downcast_ref::<ForObj>().is_some() || stmt.downcast_ref::<WhileObj>().is_some();
        if is_loop {
            self.loop_depth += 1;
            // Only the enclosing-loop identity is needed for break/continue.
            // All recovery output is discarded along with the failed statement.
            self.loop_live_masks.push("diagnostic_loop_live".to_owned());
        }
        let result = (|| {
            for child in statement_children(stmt)? {
                if let Ok(statement) = child.clone().try_cast::<Stmt>() {
                    if !self.diagnostics.visited_statements.contains(&child) {
                        let scope = self.scope_snapshot();
                        let result = self.emit_stmt(&statement, None);
                        self.restore_scope(scope);
                        result?;
                    }
                } else {
                    let (calls, bindings) = expression_calls(&child)?;
                    let invalid = self.diagnostics.invalid_values.clone();
                    // A skipped Let body must not invent an unbound-variable
                    // error after its lexical binding has already unwound.
                    for binding in bindings {
                        if self.lookup_variable(&binding).is_none() {
                            self.diagnostics.invalid_values.add(binding);
                        }
                    }
                    for call in calls {
                        if !self.diagnostics.visited_calls.contains(&oref(call.clone())) {
                            self.emit_stmt(&Evaluate::new(call, tvm_tirx::ir::UNKNOWN_LOC).into(), None)?;
                        }
                    }
                    self.diagnostics.invalid_values = invalid;
                }
            }
            Ok(())
        })();
        if is_loop {
            self.loop_depth -= 1;
            self.loop_live_masks.pop();
        }
        self.diagnostics.invalid_values = invalid_values;
        result
    }
}

pub fn collect(ctx: &Ctx, plan: &KernelPlan) -> AResult<Vec<String>> {
    let options = EmitOptions {
        collect_errors: true,
        analysis_capable: false,
        split_thresholds: super::scaffold::SplitThresholds::unsplit(),
    };
    let mut emitter = match Emitter::new(ctx, plan, 0, 1, &options) {
        Ok(emitter) => emitter,
        Err(Failure::Unsupported {
            message,
            unsupported,
        }) => {
            return Ok(if unsupported.is_empty() {
                vec![message]
            } else {
                unsupported
            })
        }
        Err(error) => return Err(error),
    };
    emitter.diagnostics.enabled = true;
    emitter.emit()?;
    emitter.diagnostics.unsupported.sort_by_key(|item| item.0);
    Ok(emitter
        .diagnostics
        .unsupported
        .into_iter()
        .map(|(id, message)| {
            if id < 0 {
                message
            } else {
                format!("op#{id}:{message}")
            }
        })
        .collect())
}
