//! Typed buffer identities and their explicit physical-data bindings.

use crate::decode::{data_source, resolve_tmem_address, tensor_decl, TensorDecl};
use std::cell::RefCell;
use tvm_tirx::ir::prim::add;
use tvm_tirx::ir::{CallObj, Expr, Op, Stmt, Var};
use tvm_tirx::tvm_ffi::object::ObjectRef;

use tvm_tirx::tvm_ffi::ObjectRefCast;

use super::util::{as_buffer, buffer_name, buffer_ref, ffi_text, oref, same, unsupported, AResult, IdMap};

/// Name of the op called by a `Call` node, if the callee is an `Op`.
pub fn call_op_name(call: &CallObj) -> AResult<Option<String>> {
    match call.op.clone().try_cast::<Op>() {
        Ok(op) => Ok(Some(ffi_text(&op.name))),
        Err(_) => Ok(None),
    }
}

pub struct BufferBindings {
    /// `(buffer, data)` in declaration order.
    pub declarations: Vec<(Var, Expr)>,
    /// `(tensor, column address)` of TMEM views, in binding order.
    pub tmem_addresses: Vec<(Var, Expr)>,
    storage_keys: RefCell<IdMap<ObjectRef>>,
}

impl BufferBindings {
    pub(crate) fn build(statements: &[Stmt]) -> AResult<Self> {
        let mut declarations = Vec::new();
        let mut tmem_addresses = Vec::new();
        for statement in statements {
            let Some((tensor, decl)) = tensor_decl(statement)? else {
                continue;
            };
            match decl {
                TensorDecl::View(data) => declarations.push((tensor, data)),
                TensorDecl::Tmem(address) => {
                    if let Some(address) = resolve_tmem_address(&address, &tmem_addresses)? {
                        tmem_addresses.push((tensor, address));
                    }
                }
                TensorDecl::Alloc => {}
            }
        }
        Ok(Self {
            declarations,
            tmem_addresses,
            storage_keys: RefCell::new(IdMap::default()),
        })
    }

    /// The last declaration wins.
    pub fn declared_data(&self, buffer: &Var) -> Option<Expr> {
        self.declarations
            .iter()
            .rev()
            .find(|(candidate, _)| same(candidate, buffer))
            .map(|(_, data)| data.clone())
    }

    /// Typed allocation root (a buffer `Var`) or external pointer expression.
    pub fn storage_key(&self, buffer: &Var) -> AResult<ObjectRef> {
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
            match data_source(&data)? {
                Some((source, _)) => current = buffer_ref(&source),
                None => {
                    current = oref(data);
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

    /// The byte offset of `buffer`'s data from its storage root, or `None` at the root.
    pub fn root_byte_offset(&self, buffer: &Var) -> AResult<Option<Expr>> {
        let mut total: Option<Expr> = None;
        let mut current = buffer.clone();
        for _ in 0..=self.declarations.len() {
            let Some(data) = self.declared_data(&current) else {
                return Ok(total);
            };
            let Some((parent, offsets)) = data_source(&data)? else {
                return Ok(total);
            };
            for offset in offsets {
                total = Some(match total {
                    Some(sum) => add(sum, offset)?,
                    None => offset,
                });
            }
            current = parent;
        }
        unsupported(format!(
            "buffer:{}:tensor declarations form a cycle",
            buffer_name(buffer)
        ))
    }

    pub fn same_storage(&self, lhs: &Var, rhs: &Var) -> AResult<bool> {
        let left = self.storage_key(lhs)?;
        let right = self.storage_key(rhs)?;
        Ok(same(&left, &right))
    }
}

pub fn same_buffer(bindings: &BufferBindings, lhs: &Var, rhs: &Var) -> AResult<bool> {
    if same(lhs, rhs) {
        return Ok(true);
    }
    bindings.same_storage(lhs, rhs)
}
