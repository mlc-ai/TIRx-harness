//! Exact bindings and shape substitutions for symbolic expressions.

use tvm_tirx::sym::Analyzer;
use tvm_tirx::ir::{Expr, Var};
use tvm_tirx::ir::Stmt;
use tvm_tirx::tvm_ffi::{Any, Map};

use super::buffers::BufferBindings;
use super::shapes::{structural_equal, ShapeExpressionContext};
use super::util::{oref, prim, repr_of, same, scalar_bind, unsupported, AResult};

pub struct ExpressionBindings {
    pub bindings: Vec<(Var, Expr)>,
    pub shape_expressions: ShapeExpressionContext,
}

impl ExpressionBindings {
    pub(in crate::analyze) fn build(
        analyzer: &Analyzer,
        statements: &[Stmt],
        buffer_bindings: &BufferBindings,
    ) -> AResult<Self> {
        let mut bindings: Vec<(Var, Expr)> = Vec::new();
        for statement in statements {
            let Some(bind) = scalar_bind(statement) else {
                continue;
            };
            if let Some((_, previous)) = bindings.iter().find(|(var, _)| same(var, &bind.var)) {
                if !structural_equal(&Any::from(previous.clone()), &Any::from(bind.value.clone()))?
                {
                    return unsupported(format!(
                        "Bind({}): one variable has conflicting definitions",
                        repr_of(&bind.var)?
                    ));
                }
                continue;
            }
            bindings.push((bind.var.clone(), bind.value.clone()));
        }
        Ok(Self {
            bindings,
            shape_expressions: ShapeExpressionContext::build(
                analyzer,
                statements,
                buffer_bindings,
            )?,
        })
    }

    /// Reuse a substitution map across a sequence of symbolic expressions.
    pub fn resolve_expression_with(
        &self,
        expression: &Expr,
        substitutions: &Map<Var, Expr>,
    ) -> AResult<Expr> {
        let mut current = expression.clone();
        if !self.bindings.is_empty() {
            let mut resolved_once = false;
            for _ in 0..self.bindings.len() + 1 {
                let resolved = prim(&crate::substitute(
                    oref(current.clone()),
                    substitutions.clone(),
                )?)?;
                let equal =
                    structural_equal(&Any::from(current.clone()), &Any::from(resolved.clone()))?;
                current = resolved;
                if equal {
                    resolved_once = true;
                    break;
                }
            }
            if !resolved_once {
                return unsupported(
                    "compile-time expression proof encountered cyclic Bind definitions",
                );
            }
        }
        Ok(Analyzer::new().simplify(&current, 2)?)
    }
}
