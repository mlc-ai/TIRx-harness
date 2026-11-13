//! Exact bindings and shape substitutions for symbolic expressions.

use tvm_tirx::sym::Analyzer;
use tvm_tirx::ir::{Expr, Var};
use tvm_tirx::ir::Stmt;
use tvm_tirx::tvm_ffi::Any;

use super::buffers::BufferBindings;
use super::shapes::{structural_equal, ShapeExpressionContext};
use super::util::{repr_of, same, scalar_bind, unsupported, AResult};

pub struct ExpressionBindings {
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
            shape_expressions: ShapeExpressionContext::build(
                analyzer,
                statements,
                buffer_bindings,
            )?,
        })
    }
}
