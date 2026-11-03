//! Typed buffer identities and their explicit physical-data bindings.

use crate::decode::projected_buffer;
use std::cell::RefCell;
use tvm_tirx::ir::prim::{add, AddObj, CastObj};
use tvm_tirx::ir::{BindObj, CallObj, Expr, Op, Stmt, TupleObj, Var};
use tvm_tirx::tirx::DeclBufferObj;
use tvm_tirx::tvm_ffi::object::ObjectRef;

use tvm_tirx::tvm_ffi::{ObjectRefCast, ObjectRefCore};

use super::util::{as_buffer, buffer_ref, buffer_scope, ffi_text, oref, same, AResult, IdMap};

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
    /// `(tensor, column address)` of TMEM allocations and views, in binding order.
    pub tmem_addresses: Vec<(Var, Expr)>,
    storage_keys: RefCell<IdMap<ObjectRef>>,
}

impl BufferBindings {
    pub(crate) fn build(statements: &[Stmt]) -> AResult<Self> {
        let mut declarations = Vec::new();
        let mut tmem_addresses = Vec::new();
        for statement in statements {
            if let Some(declaration) = statement.as_node::<DeclBufferObj>() {
                declarations.push((declaration.buffer.clone(), declaration.data.clone()));
            }
            if let Some(bind) = statement.as_node::<BindObj>() {
                if let Some(address) = tmem_address(bind, &tmem_addresses)? {
                    tmem_addresses.push((bind.var.clone(), address));
                }
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

    pub fn same_storage(&self, lhs: &Var, rhs: &Var) -> AResult<bool> {
        let left = self.storage_key(lhs)?;
        let right = self.storage_key(rhs)?;
        Ok(same(&left, &right))
    }
}

/// The column address a TMEM `Bind` gives its tensor: the placement operand of
/// `tirx.alloc_tensor`, or the address of a `tirx.cuda.decl_tmem` view, whose
/// reading of a parent's data pointer is resolved to the parent's address.
fn tmem_address(bind: &BindObj, known: &[(Var, Expr)]) -> AResult<Option<Expr>> {
    let tensor = as_buffer(&oref(bind.var.clone()));
    if tensor.is_none_or(|tensor| buffer_scope(&tensor) != "tmem") {
        return Ok(None);
    }
    let Some(call) = bind.value.as_node::<CallObj>() else {
        return Ok(None);
    };
    match call_op_name(call)?.as_deref() {
        Some("tirx.alloc_tensor") => Ok(call.args.iter().next().and_then(|placement| {
            let fields: Vec<Expr> = placement.as_node::<TupleObj>()?.fields.iter().collect();
            (fields.len() == 1).then(|| fields[0].clone())
        })),
        Some("tirx.cuda.decl_tmem") => match call.args.iter().next() {
            Some(address) => resolve_tmem_address(&address, known),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

/// `address` with `uint32(reinterpret(tensor_data_ptr(parent), uint64))`, the column
/// address of `parent`, replaced by the address recorded for `parent`.
fn resolve_tmem_address(address: &Expr, known: &[(Var, Expr)]) -> AResult<Option<Expr>> {
    if let Some(sum) = address.as_node::<AddObj>() {
        return match resolve_tmem_address(&sum.a, known)? {
            Some(base) => Ok(Some(add(base, sum.b.clone())?)),
            None => Ok(None),
        };
    }
    match data_pointer_tensor(address)? {
        Some(parent) => Ok(known
            .iter()
            .rev()
            .find(|(tensor, _)| same(tensor, &parent))
            .map(|(_, address)| address.clone())),
        None => Ok(Some(address.clone())),
    }
}

/// The tensor whose data pointer `expr` reinterprets as a TMEM column address.
fn data_pointer_tensor(expr: &Expr) -> AResult<Option<Var>> {
    let Some(cast) = expr.as_node::<CastObj>() else {
        return Ok(None);
    };
    let Some(reinterpret) = cast.value.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(reinterpret)?.as_deref() != Some("tirx.reinterpret") {
        return Ok(None);
    }
    let Some(pointer) = reinterpret.args.iter().next() else {
        return Ok(None);
    };
    let Some(data_pointer) = pointer.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(data_pointer)?.as_deref() != Some("tirx.tensor_data_ptr") {
        return Ok(None);
    }
    Ok(data_pointer
        .args
        .iter()
        .next()
        .and_then(|tensor| as_buffer(&oref(tensor))))
}

pub fn same_buffer(bindings: &BufferBindings, lhs: &Var, rhs: &Var) -> AResult<bool> {
    if same(lhs, rhs) {
        return Ok(true);
    }
    bindings.same_storage(lhs, rhs)
}
