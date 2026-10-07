//! Typed buffer identities and their explicit physical-data bindings.

use crate::decode::projected_buffer;
use std::cell::RefCell;
use tvm::ir::{CallObj, Expr, Op};
use tvm::tirx::{BufferVar, DeclBufferObj, Stmt};
use tvm::tvm_ffi::object::ObjectRef;

use tvm::tvm_ffi::{ObjectRefCast, ObjectRefCore};

use super::util::{as_buffer, buffer_ref, ffi_text, oref, same, AResult, IdMap};

/// Name of the op called by a `Call` node, if the callee is an `Op`.
pub fn call_op_name(call: &CallObj) -> AResult<Option<String>> {
    match call.op.clone().try_cast::<Op>() {
        Ok(op) => Ok(Some(ffi_text(&op.name()?))),
        Err(_) => Ok(None),
    }
}

pub struct BufferBindings {
    /// `(buffer, data)` in declaration order.
    pub declarations: Vec<(BufferVar, Expr)>,
    storage_keys: RefCell<IdMap<ObjectRef>>,
}

impl BufferBindings {
    pub(crate) fn build(statements: &[Stmt]) -> AResult<Self> {
        let mut declarations = Vec::new();
        for statement in statements {
            if let Some(declaration) = statement.as_node::<DeclBufferObj>() {
                declarations.push((declaration.buffer.clone(), declaration.data.clone()));
            }
        }
        Ok(Self {
            declarations,
            storage_keys: RefCell::new(IdMap::default()),
        })
    }

    /// The last declaration wins.
    pub fn declared_data(&self, buffer: &BufferVar) -> Option<Expr> {
        self.declarations
            .iter()
            .rev()
            .find(|(candidate, _)| same(candidate.as_var(), buffer.as_var()))
            .map(|(_, data)| data.clone())
    }

    /// Typed allocation root (a buffer `Var`) or external pointer expression.
    pub fn storage_key(&self, buffer: &BufferVar) -> AResult<ObjectRef> {
        let mut current = buffer_ref(buffer);
        let mut seen: Vec<ObjectRef> = Vec::new();
        let mut cycle = false;
        while let Some(current_buffer) = as_buffer(&current) {
            if let Some(cached) = self.storage_keys.borrow().get(&current) {
                current = cached.clone();
                break;
            }
            if seen.iter().any(|candidate| same(candidate, &current)) {
                cycle = true;
                break;
            }
            seen.push(current.clone());
            let Some(data) = self.declared_data(&current_buffer) else {
                break;
            };
            let data_ref = oref(data);
            match projected_buffer(&data_ref)? {
                Some(source) => current = buffer_ref(&source),
                None => {
                    current = data_ref;
                    break;
                }
            }
        }
        if !cycle {
            let mut cache = self.storage_keys.borrow_mut();
            for candidate in seen {
                cache.insert(candidate, current.clone());
            }
        }
        Ok(current)
    }

    pub fn same_storage(&self, lhs: &BufferVar, rhs: &BufferVar) -> AResult<bool> {
        let left = self.storage_key(lhs)?;
        let right = self.storage_key(rhs)?;
        Ok(same(&left, &right))
    }
}

pub fn same_buffer(bindings: &BufferBindings, lhs: &BufferVar, rhs: &BufferVar) -> AResult<bool> {
    if same(lhs.as_var(), rhs.as_var()) {
        return Ok(true);
    }
    bindings.same_storage(lhs, rhs)
}
