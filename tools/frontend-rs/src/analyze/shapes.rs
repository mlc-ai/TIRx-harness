//! Host-evaluable scalar bindings used by runtime buffer shapes.

use tvm::analysis::Analyzer;
use tvm::ir::{Expr, PrimExpr, TensorLoad, TensorLoadObj, Var};
use tvm::tirx::{BindObj, BufferStoreObj, BufferVar, ForObj, IfThenElseObj, Stmt, WhileObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{self, structural_map, Any, Map, ObjectRefCore, WalkOrder};

use super::buffers::{same_buffer, BufferBindings};
use super::layout::{expr_any, int_any, op_binary};
use super::util::{
    as_buffer, buffer_scope, ident, int_imm_expr, oref, simplify, unsupported, AResult, IdSet,
};

pub fn structural_equal(lhs: &Any, rhs: &Any) -> AResult<bool> {
    let result: Any = tvm_ffi::cached_global_func!("ffi.StructuralEqual").call_tuple((
        lhs.clone(),
        rhs.clone(),
        false,
        false,
    ))?;
    Ok(bool::try_from(result)?)
}

fn is_zero_index(analyzer: &Analyzer, value: &PrimExpr) -> AResult<bool> {
    Ok(int_imm_expr(&simplify(analyzer, value)?) == Some(0))
}

/// An `int | PrimExpr` value produced by the shape helpers.
#[derive(Clone)]
pub enum Scalar {
    Int(i64),
    Expr(PrimExpr),
}

impl Scalar {
    pub fn any(&self) -> Any {
        match self {
            Scalar::Int(value) => int_any(*value),
            Scalar::Expr(expr) => expr_any(expr),
        }
    }

    pub fn binary(
        name: &str,
        lhs: &Scalar,
        rhs: &Scalar,
        int_op: fn(i64, i64) -> i64,
    ) -> AResult<Scalar> {
        if let (Scalar::Int(left), Scalar::Int(right)) = (lhs, rhs) {
            return Ok(Scalar::Int(int_op(*left, *right)));
        }
        Ok(Scalar::Expr(op_binary(name, lhs.any(), rhs.any())?))
    }
}

pub struct ShapeExpressionContext {
    pub bindings: Vec<(Var, Expr)>,
    pub scalar_initializers: Vec<(BufferVar, Expr)>,
}

impl ShapeExpressionContext {
    pub fn build(
        analyzer: &Analyzer,
        statements: &[Stmt],
        buffer_bindings: &BufferBindings,
    ) -> AResult<Self> {
        let mut bindings = Vec::new();
        for statement in statements {
            if let Some(bind) = statement.as_node::<BindObj>() {
                bindings.push((bind.var.clone(), bind.value.clone()));
            }
        }
        let mut controlled = IdSet::default();
        for statement in statements {
            if statement.as_node::<IfThenElseObj>().is_some()
                || statement.as_node::<ForObj>().is_some()
                || statement.as_node::<WhileObj>().is_some()
            {
                for child in crate::walk_statements(statement.clone())?.iter() {
                    if child.as_node::<BufferStoreObj>().is_some() {
                        controlled.add(oref(child));
                    }
                }
            }
        }
        let stores: Vec<(ObjectRef, BufferVar, Vec<PrimExpr>, PrimExpr)> = statements
            .iter()
            .filter_map(|statement| {
                statement.as_node::<BufferStoreObj>().map(|store| {
                    (
                        oref(statement.clone()),
                        store.buffer.clone(),
                        store.indices.iter().collect(),
                        store.value.clone(),
                    )
                })
            })
            .collect();
        let mut initializers = Vec::new();
        for (node, buffer, indices, value) in &stores {
            if controlled.contains(node) {
                continue;
            }
            let scope = buffer_scope(buffer);
            if !["local", "local_scalar", "register", "reg"].contains(&scope.as_str()) {
                continue;
            }
            let shape: Vec<PrimExpr> = buffer.buffer_type().shape.iter().collect();
            if shape.len() != 1 {
                continue;
            }
            let extent_minus_one = op_binary("_OpSub", expr_any(&shape[0]), int_any(1))?;
            if !is_zero_index(analyzer, &extent_minus_one)? {
                continue;
            }
            if indices.len() != 1 || !is_zero_index(analyzer, &indices[0])? {
                continue;
            }
            let mut aliases = 0;
            for (_, candidate, _, _) in &stores {
                if same_buffer(buffer_bindings, candidate, buffer)? {
                    aliases += 1;
                }
            }
            if aliases != 1 {
                continue;
            }
            initializers.push((buffer.clone(), value.clone().into()));
        }
        Ok(Self {
            bindings,
            scalar_initializers: initializers,
        })
    }

    pub fn substitutions(&self) -> Map<Var, Expr> {
        // dict(self.bindings): a later binding of the same variable wins.
        map_from_pairs(
            self.bindings
                .iter()
                .map(|(var, value)| (var.clone(), value.clone())),
        )
    }

    /// `resolve`: substitute exact binds and unconditional single-store local scalars.
    pub fn resolve(
        &self,
        analyzer: &Analyzer,
        expression: &Scalar,
        buffer_bindings: &BufferBindings,
    ) -> AResult<Scalar> {
        let Scalar::Expr(expression) = expression else {
            return Ok(expression.clone());
        };
        let substitutions = self.substitutions();
        let limit = self.bindings.len() + self.scalar_initializers.len() + 1;
        let mut current: Any = expr_any(expression);
        for _ in 0..limit {
            let substituted =
                crate::substitute(ObjectRef::try_from(current.clone())?, substitutions.clone())?;
            let mut failure: Option<super::util::Failure> = None;
            let resolved = structural_map(
                Any::from(substituted),
                |load: TensorLoad| -> tvm_ffi::Result<Any> {
                    match self.resolve_load(analyzer, &load, buffer_bindings) {
                        Ok(value) => Ok(value),
                        Err(error) => {
                            failure = Some(error);
                            Ok(Any::from(load))
                        }
                    }
                },
                WalkOrder::PostOrder,
            )?;
            if let Some(error) = failure {
                return Err(error);
            }
            if structural_equal(&current, &resolved)? {
                let simplified = simplify(analyzer, &PrimExpr::try_from(resolved)?)?;
                return Ok(Scalar::Expr(simplified));
            }
            current = resolved;
        }
        unsupported("runtime shape expression contains cyclic scalar definitions")
    }

    fn resolve_load(
        &self,
        analyzer: &Analyzer,
        load: &TensorLoad,
        buffer_bindings: &BufferBindings,
    ) -> AResult<Any> {
        let node = load.as_node::<TensorLoadObj>().expect("TensorLoad node");
        if node.indices.len() == 1 && is_zero_index(analyzer, &node.indices.get(0)?)? {
            let source = oref(node.source.clone());
            if let Some(source_buffer) = as_buffer(&source) {
                let mut matches = Vec::new();
                for (buffer, value) in &self.scalar_initializers {
                    if same_buffer(buffer_bindings, buffer, &source_buffer)? {
                        matches.push(value.clone());
                    }
                }
                if matches.len() == 1 {
                    return Ok(Any::from(matches[0].clone()));
                }
            }
        }
        Ok(Any::from(load.clone()))
    }
}

/// `dict(pairs)`: one FFI map from ordered `(var, value)` pairs, a later
/// pair of the same variable winning.
pub fn map_from_pairs<V: tvm_ffi::AnyCompatible + Clone>(
    pairs: impl IntoIterator<Item = (Var, V)>,
) -> Map<Var, V> {
    let mut positions: std::collections::HashMap<tvm_ffi::ObjectIdentity, usize> =
        std::collections::HashMap::new();
    let mut entries: Vec<(Var, V)> = Vec::new();
    for (key, value) in pairs {
        match positions.get(&ident(&key)) {
            Some(&index) => entries[index] = (key, value),
            None => {
                positions.insert(ident(&key), entries.len());
                entries.push((key, value));
            }
        }
    }
    Map::from_iter(entries)
}
