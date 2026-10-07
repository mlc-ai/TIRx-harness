//! The shared decoding boundary for registry dispatch.

use std::cell::RefCell;

use tvm::ir::{CallObj, Var};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

use crate::analyze::buffers::call_op_name;
pub mod ptx;
pub mod pointers;

use self::ptx::{DecodedPtx, PtxDecode};
use crate::analyze::util::{
    as_buffer, dtype_of, ffi_error, not_covered, oref, unsupported, AResult, Failure, IdMap,
};
use crate::analyze::Ctx;
use crate::emit::ptx_addr::validate_decoded_ptx_addr_operands;
use crate::registry::OpRow;

pub struct Decoded<'a> {
    pub node: &'a ObjectRef,
    pub call: &'a CallObj,
    pub op_name: String,
    pub args: Vec<ObjectRef>,
    pub result_dtype: String,
    pub ptx: Option<DecodedPtx>,
    pub params: Option<&'a [Var]>,
    pub op_id: Option<i64>,
}

impl<'a> Decoded<'a> {
    pub fn source_op_id(&self, emitter: &crate::emit::Emitter) -> AResult<i64> {
        match self.op_id {
            Some(id) => Ok(id),
            None => emitter.static_op_id(self.node),
        }
    }

    pub fn table(&self) -> AResult<&DecodedPtx> {
        match &self.ptx {
            Some(decoded) => Ok(decoded),
            None => not_covered(format!("{} has no PTX table payload", self.op_name)),
        }
    }

    pub fn new(
        ctx: &Ctx,
        node: &'a ObjectRef,
        params: Option<&'a [Var]>,
        op_id: Option<i64>,
    ) -> AResult<Self> {
        let Some(call) = node.as_node::<CallObj>() else {
            return Err(Failure::Ffi(ffi_error("call decoding expects a TIRx Call")));
        };
        let Some(raw_name) = call_op_name(call)? else {
            return not_covered("call without an Op callee");
        };
        let ptx = if raw_name.starts_with("tirx.ptx.") && raw_name != "tirx.ptx.addr" {
            let ptx = ctx.decoding.ptx_decoded(node)?;
            validate_decoded_ptx_addr_operands(ctx, &ptx)?;
            Some(ptx)
        } else {
            None
        };
        let op_name = match &ptx {
            Some(ptx) => ptx.op_name.clone(),
            None => canonical_op_name(&raw_name),
        };
        Ok(Self {
            node,
            call,
            op_name,
            args: call.args.iter().map(oref).collect(),
            result_dtype: dtype_of(node)?,
            ptx,
            params,
            op_id,
        })
    }

    pub fn entry<'c>(&self, ctx: &'c Ctx) -> AResult<&'c OpRow> {
        if crate::emit::raw_tma::is_tensor_map_address_call(ctx, self.node, self.call, self.params)?
        {
            return Ok(&crate::registry::CONTEXTUAL_OP_PATHS[0].1);
        }
        let Some(entry) = ctx.schema.registered_ops.get(&self.op_name) else {
            let raw_name = call_op_name(self.call)?.expect("decoded Op callee");
            let arg_dtypes = self
                .args
                .iter()
                .map(dtype_of)
                .collect::<AResult<Vec<_>>>()?;
            return unsupported(format!(
                "Call({raw_name}({})->{}): unregistered raw call (no NumSim classifier registration)",
                arg_dtypes.join(", "),
                self.result_dtype
            ));
        };
        if entry.support == "rejected" {
            return unsupported(format!("{} is unsupported: {}", self.op_name, entry.reason));
        }
        Ok(entry)
    }
}

pub fn canonical_op_name(name: &str) -> String {
    let Some(basename) = name.strip_prefix("tirx.") else {
        return name.to_owned();
    };
    if basename.contains('.') {
        return name.to_owned();
    }
    if let Some(rest) = basename.strip_prefix("cuda_") {
        return format!("tirx.cuda.{rest}");
    }
    name.to_owned()
}

/// Canonical registry name without decoding the instruction's operand payload.
pub fn call_name(node: &ObjectRef) -> AResult<Option<String>> {
    match node.as_node::<CallObj>() {
        Some(call) => Ok(call_op_name(call)?.map(|name| canonical_op_name(&name))),
        None => Ok(None),
    }
}

#[derive(Default)]
pub struct Cache(RefCell<IdMap<PtxDecode>>);

impl Cache {
    /// The Python decode outcome of one `tirx.ptx.*` call node (`None` for
    /// any other node); decoded on first use and kept for this analysis.
    pub fn ptx_decode(&self, node: &tvm::tvm_ffi::object::ObjectRef) -> AResult<Option<PtxDecode>> {
        use tvm::tvm_ffi::ObjectRefCore;
        if let Some(decode) = self.0.borrow().get(node) {
            return Ok(Some(decode.clone()));
        }
        let Some(call) = node.as_node::<tvm::ir::CallObj>() else {
            return Ok(None);
        };
        let is_table_call = call_op_name(call)?
            .is_some_and(|name| name.starts_with("tirx.ptx.") && name != "tirx.ptx.addr");
        if !is_table_call {
            return Ok(None);
        }
        let decode = ptx::decode_call(node)?;
        self.0.borrow_mut().insert(node.clone(), decode.clone());
        Ok(Some(decode))
    }

    /// The decoded call, or Python's decode rejection.
    pub fn ptx_decoded(&self, node: &tvm::tvm_ffi::object::ObjectRef) -> AResult<DecodedPtx> {
        match self.ptx_decode(node)? {
            Some(decode) => decode.decoded().cloned(),
            None => Err(Failure::Ffi(ffi_error("PTX call has no decode payload"))),
        }
    }
}

/// The typed buffer behind `tirx.buffer_data`.
pub fn projected_buffer(node: &ObjectRef) -> AResult<Option<BufferVar>> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("tirx.buffer_data") || call.args.len() != 1 {
        return Ok(None);
    }
    Ok(as_buffer(&oref(call.args.get(0)?)))
}
