//! Shared registry dispatch and call context.

use tvm::ir::CallObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

use super::super::analyze::util::{oref, unsupported, AResult};
use super::abi;
use super::{Emitter, RustValue};
use crate::decode::Decoded;

pub fn call_args(node: &ObjectRef) -> Vec<ObjectRef> {
    node.as_node::<CallObj>()
        .map(|call| call.args.iter().map(oref).collect())
        .unwrap_or_default()
}

/// A rejected row never reaches emission: Decoded::entry returns its reason.
pub fn rejected(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let entry = call.entry(emitter.ctx)?;
    unsupported(format!("{} is unsupported: {}", call.op_name, entry.reason))
}

impl<'a> Emitter<'a> {
    pub fn with_call_expr<T>(
        &mut self,
        node: &ObjectRef,
        emit: impl FnOnce(&mut Self) -> AResult<T>,
    ) -> AResult<T> {
        self.call_expr_stack.push(node.clone());
        let result = emit(self);
        self.call_expr_stack.pop();
        result
    }

    pub fn emit_call(&mut self, node: &ObjectRef) -> AResult<Option<RustValue>> {
        if self.diagnostics.enabled {
            self.diagnostics.visited_calls.add(node.clone());
        }
        match self.emit_call_inner(node) {
            Err(crate::analyze::util::Failure::Unsupported { message, .. })
                if self.diagnostics.enabled =>
            {
                self.record_unsupported(node, message);
                Err(crate::analyze::util::Failure::Recorded)
            }
            Err(error) => Err(crate::analyze::frontend::with_node_span(error, node)),
            result => result,
        }
    }

    fn emit_call_inner(&mut self, node: &ObjectRef) -> AResult<Option<RustValue>> {
        let params = self.params.clone();
        let decoded = Decoded::new(
            self.ctx,
            node,
            Some(&params),
            self.op_ids.get(node).copied(),
        )?;
        if self.ctx.schema.high_precision {
            if let Some(value) = super::high_precision::emit_float(self, &decoded)? {
                return Ok(Some(value));
            }
            super::high_precision::validate_call(&decoded)?;
        }
        let emit = decoded.entry(self.ctx)?.emit;
        emit(self, &decoded)
    }

    pub(super) fn expr_site(&mut self, node: &ObjectRef) -> String {
        let op_id = self.lowered_instruction_site(node);
        abi::site(op_id as u64)
    }

    pub(super) fn pointer(&mut self, expr: &ObjectRef, label: &str) -> AResult<RustValue> {
        let mut value = self.emit_expr(expr)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_shared_pointer(expr, Some(value), "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} must be a physical SMEM pointer"));
        }
        Ok(value)
    }
}
