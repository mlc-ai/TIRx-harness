//! Control-flow calls return no expression value.

use crate::analyze::util::{dtype_of, oref, unsupported, AResult};
use crate::analyze::Ctx;
use crate::decode::Decoded;
use tvm::ir::CallObj;
use tvm::tvm_ffi::object::ObjectRef;

use super::{ControlProvenance, Emitter, RustValue, Uniformity};

pub fn break_loop(emitter: &mut Emitter, decoded: &Decoded) -> AResult<Option<RustValue>> {
    validate_control_call(emitter.ctx, decoded.node, decoded.call, &decoded.op_name)?;
    emitter.emit_break()?;
    Ok(None)
}

pub fn continue_loop(emitter: &mut Emitter, decoded: &Decoded) -> AResult<Option<RustValue>> {
    validate_control_call(emitter.ctx, decoded.node, decoded.call, &decoded.op_name)?;
    emitter.emit_continue()?;
    Ok(None)
}

pub fn trap_assert(emitter: &mut Emitter, decoded: &Decoded) -> AResult<Option<RustValue>> {
    validate_control_call(emitter.ctx, decoded.node, decoded.call, &decoded.op_name)?;
    emitter.emit_trap_assert(&decoded.args[0])?;
    Ok(None)
}

fn validate_no_args(call: &CallObj, node: &ObjectRef, op_name: &str) -> AResult<()> {
    let result = dtype_of(node)?;
    if !call.args.is_empty() || !result.is_empty() {
        return unsupported(format!(
            "{op_name} expects no arguments and a void result, got {} arguments and result dtype {:?}",
            call.args.len(),
            &result));
    }
    Ok(())
}

fn validate_trap(ctx: &Ctx, call: &CallObj, node: &ObjectRef, op_name: &str) -> AResult<()> {
    let result = dtype_of(node)?;
    let mut actual = Vec::new();
    for argument in call.args.iter() {
        actual.push(dtype_of(&oref(argument))?);
    }
    let predicate_ok = actual.len() == 1 && ctx.schema.call_scalars.contains(&actual[0]);
    if !predicate_ok || !result.is_empty() {
        return unsupported(format!(
            "{op_name} expects one scalar argument and a void result, got arguments {:?} and result dtype {:?}",
            &actual,
            &result));
    }
    Ok(())
}

/// Validate a control-flow or assertion call.
pub fn validate_control_call(
    ctx: &Ctx,
    node: &ObjectRef,
    call: &CallObj,
    op_name: &str,
) -> AResult<()> {
    match op_name {
        "tirx.cuda.trap_when_assert_failed" => validate_trap(ctx, call, node, op_name),
        _ => validate_no_args(call, node, op_name),
    }
}

/// A resolved control-flow or assertion call.

impl<'a> Emitter<'a> {
    pub fn emit_trap_assert(&mut self, condition_expr: &ObjectRef) -> AResult<()> {
        let condition = self.emit_expr(condition_expr)?;
        let condition = self.coerce_value(condition, "bool", "trap_predicate")?;
        if condition.uniformity == Uniformity::Uniform {
            self.emit_line(&format!("if !({}) {{", condition.code));
            self.emit_line(
                "    return Err(EngineError::message(\"GPU assertion condition failed\"));",
            );
            self.emit_line("}");
            return Ok(());
        }
        let condition_mask = if condition.is_mask {
            condition
        } else {
            let predicate = self.boolean_lane(&condition)?;
            self.emit_varying_mask(&predicate, ControlProvenance::None)
        };
        let failed = self.control_name("trap_failed");
        self.emit_line(&format!(
            "let {failed} = ctx.active_mask() - {};",
            condition_mask.code
        ));
        self.emit_line(&format!("if !{failed}.is_empty() {{"));
        self.emit_line(&format!(
            "    return Err(EngineError::message(format!(\"GPU assertion condition failed; lanes={{:?}}\", {failed}.iter().collect::<Vec<_>>() )));"
        ));
        self.emit_line("}");
        Ok(())
    }
}

pub fn is_loop_transfer(node: &ObjectRef) -> AResult<bool> {
    Ok(matches!(
        crate::decode::call_name(node)?.as_deref(),
        Some("tirx.break_loop" | "tirx.continue_loop")
    ))
}
