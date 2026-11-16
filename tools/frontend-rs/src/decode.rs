//! The shared decoding boundary for registry dispatch.

use crate::tvm_compat::TryCast;
use std::cell::RefCell;

use tvm_tirx::ir::{BindObj, CallObj, Expr, Op, Stmt, Var};

use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::{ObjectRefCore};

use crate::analyze::buffers::call_op_name;
pub mod ptx;
pub mod pointers;

use self::ptx::{DecodedPtx, PtxDecode};
use crate::analyze::util::{
    as_buffer, buffer_name, buffer_scope, dtype_of, ffi_error, ffi_text, is_tensor_var,
    not_covered, oref, unsupported, AResult, Failure, IdMap,
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
        let Some(call) = node.downcast_ref::<CallObj>() else {
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
            args: call.args.iter().cloned().map(oref).collect(),
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
    match node.downcast_ref::<CallObj>() {
        Some(call) => Ok(call_op_name(call)?.map(|name| canonical_op_name(&name))),
        None => Ok(None),
    }
}

#[derive(Default)]
pub struct Cache(RefCell<IdMap<PtxDecode>>);

impl Cache {
    /// The Python decode outcome of one `tirx.ptx.*` call node (`None` for
    /// any other node); decoded on first use and kept for this analysis.
    pub fn ptx_decode(&self, node: &tvm_tirx::tvm_ffi::object::ObjectRef) -> AResult<Option<PtxDecode>> {
        use tvm_tirx::tvm_ffi::ObjectRefCore;
        if let Some(decode) = self.0.borrow().get(node) {
            return Ok(Some(decode.clone()));
        }
        let Some(call) = node.downcast_ref::<tvm_tirx::ir::CallObj>() else {
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
    pub fn ptx_decoded(&self, node: &tvm_tirx::tvm_ffi::object::ObjectRef) -> AResult<DecodedPtx> {
        match self.ptx_decode(node)? {
            Some(decode) => decode.decoded().cloned(),
            None => Err(Failure::Ffi(ffi_error("PTX call has no decode payload"))),
        }
    }
}

pub const TENSOR_DATA_PTR: &str = "tirx.tensor_data_ptr";
pub const DECL_TENSOR: &str = "tirx.decl_tensor";

/// The tensor whose data pointer `tirx.tensor_data_ptr` reads.
pub fn projected_buffer(node: &ObjectRef) -> AResult<Option<Var>> {
    let Some(call) = node.downcast_ref::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some(TENSOR_DATA_PTR) || call.args.len() != 1 {
        return Ok(None);
    }
    Ok(as_buffer(&oref(call.args.get(0).ok_or_else(|| ffi_error("missing call argument 0"))?)))
}

/// How a `Bind` of a tensor variable introduces it.
pub enum TensorDecl {
    /// `tirx.alloc_tensor` without a placement: new storage.
    Alloc,
    /// `tirx.decl_tensor(data)`: a view of existing data.
    View(Expr),
}

/// Why a declared tensor-memory buffer is rejected: TMEM is only addressed
/// through raw `tcgen05` column operands derived from `tcgen05.alloc`.
pub const TMEM_BUFFER_UNSUPPORTED: &str = "NumSim does not support scope=\"tmem\" buffers \
     (T.tmem, T.alloc_tensor with scope=\"tmem\", T.cuda.decl_tmem); use raw tcgen05 column \
     operands and T.cuda.get_tmem_addr(...)";

/// Whether `scope` names tensor memory.
pub fn is_tmem_scope(scope: &str) -> bool {
    scope == "tmem" || scope.starts_with("tmem.")
}

/// The tensor `stmt` binds and how, or `None` for any other statement.
pub fn tensor_decl(stmt: &Stmt) -> AResult<Option<(Var, TensorDecl)>> {
    let Some(bind) = stmt.downcast_ref::<BindObj>() else {
        return Ok(None);
    };
    if !is_tensor_var(&bind.var) {
        return Ok(None);
    }
    let call = bind.value.downcast_ref::<CallObj>();
    let op = call.and_then(|call| call.op.clone().try_cast::<Op>().ok());
    let op_name = op.map(|op| ffi_text(&op.name));
    if is_tmem_scope(&buffer_scope(&bind.var)) || op_name.as_deref() == Some("tirx.cuda.decl_tmem") {
        return unsupported(format!(
            "buffer:{}:{TMEM_BUFFER_UNSUPPORTED}",
            buffer_name(&bind.var)
        ));
    }
    let decl = match (op_name.as_deref(), call) {
        (Some("tirx.alloc_tensor"), Some(call)) if call.args.is_empty() => TensorDecl::Alloc,
        (Some("tirx.alloc_tensor"), Some(_)) => {
            return unsupported(format!(
                "buffer:{}:alloc_tensor placement is not modeled",
                buffer_name(&bind.var)
            ))
        }
        (Some(DECL_TENSOR), Some(call)) if call.args.len() == 1 => {
            TensorDecl::View(call.args.get(0).ok_or_else(|| ffi_error("missing call argument 0"))?)
        }
        _ => {
            return unsupported(format!(
                "buffer:{}:a tensor must be bound by alloc_tensor or decl_tensor",
                buffer_name(&bind.var)
            ))
        }
    };
    Ok(Some((bind.var.clone(), decl)))
}

/// The pointer under `tirx.ptr_byte_offset` layers, and their byte offsets,
/// outermost first.
fn strip_byte_offsets(data: &Expr) -> AResult<(Expr, Vec<Expr>)> {
    let mut current = data.clone();
    let mut offsets = Vec::new();
    while let Some(call) = current.downcast_ref::<CallObj>() {
        if call_op_name(call)?.as_deref() != Some("tirx.ptr_byte_offset") || call.args.len() != 2 {
            break;
        }
        let base = call.args.get(0).ok_or_else(|| ffi_error("missing call argument 0"))?;
        offsets.push(call.args.get(1).ok_or_else(|| ffi_error("missing call argument 1"))?);
        current = base;
    }
    Ok((current, offsets))
}

/// Whether `data` is a tensor's `tirx.tensor_data_ptr`, under any byte offsets.
pub fn is_tensor_data_pointer(data: &Expr) -> AResult<bool> {
    let (base, _) = strip_byte_offsets(data)?;
    Ok(projected_buffer(&oref(base))?.is_some())
}

/// The tensor whose data `data` points into, with the byte offsets that
/// `tirx.ptr_byte_offset` layers add to it, outermost first, then the offset of
/// an addressed element.
pub fn data_source(data: &Expr) -> AResult<Option<(Var, Vec<Expr>)>> {
    let (base, mut offsets) = strip_byte_offsets(data)?;
    if let Some(tensor) = projected_buffer(&oref(base.clone()))? {
        return Ok(Some((tensor, offsets)));
    }
    if let Some((tensor, byte_offset)) = element_address(&base)? {
        offsets.push(byte_offset);
        return Ok(Some((tensor, offsets)));
    }
    Ok(None)
}

/// `address_of(tensor[indices])` of a default-layout tensor: the tensor and the
/// element's byte offset into its data.
///
/// This is how tvm-tirx spells a view at a runtime element offset
/// (`decl_tensor(data=tensor.ptr_to([offset]))`), where a TIR buffer carried
/// `elem_offset` over its parent's data. A tensor with an explicit layout, or in
/// register storage, keeps its address a plain pointer.
fn element_address(expr: &Expr) -> AResult<Option<(Var, Expr)>> {
    use crate::analyze::util::TensorVarExt;
    use tvm_tirx::ir::TensorLoadObj;
    use tvm_tirx::tvm_ffi::{Any, DLDataType, DLDataTypeExt};

    let Some(call) = expr.downcast_ref::<CallObj>() else {
        return Ok(None);
    };
    if call_op_name(call)?.as_deref() != Some("tirx.address_of") || call.args.len() != 1 {
        return Ok(None);
    }
    let element = call.args.get(0).ok_or_else(|| ffi_error("missing call argument 0"))?;
    let Some(load) = element.downcast_ref::<TensorLoadObj>() else {
        return Ok(None);
    };
    let Some(tensor) = as_buffer(&oref(load.source.clone())) else {
        return Ok(None);
    };
    let ty = tensor.tensor_type();
    if ty.layout.is_some() || load.indices.len() != ty.shape.len() {
        return Ok(None);
    }
    // Views at a runtime offset are bounded by their owner in global and
    // shared memory; register storage keeps such a view a plain pointer.
    if !matches!(buffer_scope(&tensor).as_str(), "global" | "shared" | "shared.dyn") {
        return Ok(None);
    }
    let dtype = crate::analyze::util::buffer_dtype(&tensor);
    let data_type = DLDataType::try_from_str(&dtype)?;
    if data_type.lanes != 1 || data_type.bits % 8 != 0 {
        return Ok(None);
    }
    let itemsize = i64::from(data_type.bits) / 8;
    let op = crate::analyze::layout::op_binary;
    let mut linear: Option<Expr> = None;
    for (index, extent) in load.indices.iter().zip(ty.shape.iter()) {
        linear = Some(match linear {
            None => index.clone().into(),
            Some(outer) => op(
                "_OpAdd",
                Any::from(op("_OpMul", Any::from(outer), Any::from(extent.clone()))?),
                Any::from(index.clone()),
            )?,
        });
    }
    let Some(linear) = linear else {
        return Ok(None);
    };
    let bytes = op("_OpMul", Any::from(linear), Any::from(itemsize))?;
    Ok(Some((tensor, bytes)))
}

