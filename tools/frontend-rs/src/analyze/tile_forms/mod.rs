//! Transient validation facts used by the tile lowerers
//! of every public TIRx tile op.

mod coordinates;
mod copy;
mod gemm;
mod numeric;
mod parse;
mod tcgen_layout;

use tvm::ir::PrimExpr;
use tvm::tirx::{BufferVar, TilePrimitiveCallObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, ObjectRefCore};

use super::util::{ffi_text, oref, unsupported, AResult, Failure};
use super::Ctx;
use copy::{resolve_copy, resolve_copy_async, resolve_permute_layout};
use gemm::{resolve_gemm, resolve_gemm_async};
use numeric::{resolve_cast, resolve_elementwise, resolve_reduction, resolve_unary_elementwise};

pub(crate) use copy::{typed_tma_dtype_rejection, TYPED_TMA_ELEMENT_DTYPES};
pub use gemm::GemmAsyncFacts;
pub use parse::{any_value, AnyValue};
pub use tcgen_layout::TcgenLdstCacheEntry;

// ----------------------------------------------------------------------
// Transient validation facts.
// ----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileOpKind {
    Copy,
    CopyAsync,
    Cast,
    Zero,
    Sqrt,
    Fill,
    Add,
    Sub,
    Mul,
    Maximum,
    Fdiv,
    Fma,
    Reciprocal,
    Silu,
    Exp,
    Exp2,
    Log2,
    Sum,
    Max,
    Min,
    Gemm,
    GemmAsync,
    PermuteLayout,
}

const TILE_OP_PREFIX: &str = "tirx.tile.";

/// Every public TIRx tile op, in IR-name order.
pub const TILE_OPS: &[TileOpKind] = &[
    TileOpKind::Add,
    TileOpKind::Cast,
    TileOpKind::Copy,
    TileOpKind::CopyAsync,
    TileOpKind::Exp,
    TileOpKind::Exp2,
    TileOpKind::Fdiv,
    TileOpKind::Fill,
    TileOpKind::Fma,
    TileOpKind::Gemm,
    TileOpKind::GemmAsync,
    TileOpKind::Log2,
    TileOpKind::Max,
    TileOpKind::Maximum,
    TileOpKind::Min,
    TileOpKind::Mul,
    TileOpKind::PermuteLayout,
    TileOpKind::Reciprocal,
    TileOpKind::Silu,
    TileOpKind::Sqrt,
    TileOpKind::Sub,
    TileOpKind::Sum,
    TileOpKind::Zero,
];

impl TileOpKind {
    /// The IR op name, `tirx.tile.<value>`.
    pub fn op_name(self) -> String {
        format!("{TILE_OP_PREFIX}{}", self.value())
    }

    pub fn from_op_name(op_name: &str) -> Option<TileOpKind> {
        let value = op_name.strip_prefix(TILE_OP_PREFIX)?;
        TILE_OPS.iter().copied().find(|kind| kind.value() == value)
    }

    pub fn value(self) -> &'static str {
        match self {
            TileOpKind::Copy => "copy",
            TileOpKind::CopyAsync => "copy_async",
            TileOpKind::Cast => "cast",
            TileOpKind::Zero => "zero",
            TileOpKind::Sqrt => "sqrt",
            TileOpKind::Fill => "fill",
            TileOpKind::Add => "add",
            TileOpKind::Sub => "sub",
            TileOpKind::Mul => "mul",
            TileOpKind::Maximum => "maximum",
            TileOpKind::Fdiv => "fdiv",
            TileOpKind::Fma => "fma",
            TileOpKind::Reciprocal => "reciprocal",
            TileOpKind::Silu => "silu",
            TileOpKind::Exp => "exp",
            TileOpKind::Exp2 => "exp2",
            TileOpKind::Log2 => "log2",
            TileOpKind::Sum => "sum",
            TileOpKind::Max => "max",
            TileOpKind::Min => "min",
            TileOpKind::Gemm => "gemm",
            TileOpKind::GemmAsync => "gemm_async",
            TileOpKind::PermuteLayout => "permute_layout",
        }
    }
}

/// `TileRegion`.
#[derive(Clone)]
pub struct TileRegion {
    pub buffer: BufferVar,
    pub mins: Vec<PrimExpr>,
    pub extents: Vec<i64>,
    pub dtype: String,
    pub memory_scope: String,
}

impl TileRegion {
    pub fn element_count(&self) -> i64 {
        self.extents.iter().product()
    }

    pub fn logical_shape(&self) -> Vec<i64> {
        let non_unit: Vec<i64> = self
            .extents
            .iter()
            .copied()
            .filter(|extent| *extent != 1)
            .collect();
        if non_unit.is_empty() {
            vec![1]
        } else {
            non_unit
        }
    }
}

/// `TileScalar`: the expression keeps Python's literal-or-node identity.
#[derive(Clone)]
pub struct TileScalar {
    pub expr: Any,
    pub dtype: String,
}

#[derive(Clone)]
pub enum TileOperand {
    Region(TileRegion),
    Scalar(TileScalar),
}

impl TileOperand {
    pub fn dtype(&self) -> &str {
        match self {
            TileOperand::Region(region) => &region.dtype,
            TileOperand::Scalar(scalar) => &scalar.dtype,
        }
    }

    pub fn region(&self) -> Option<&TileRegion> {
        match self {
            TileOperand::Region(region) => Some(region),
            TileOperand::Scalar(_) => None,
        }
    }

    pub fn scalar(&self) -> Option<&TileScalar> {
        match self {
            TileOperand::Region(_) => None,
            TileOperand::Scalar(scalar) => Some(scalar),
        }
    }
}

/// One tile call attribute value.
#[derive(Clone)]
pub enum TileAttr {
    Bool(bool),
    Int(i64),
    Str(String),
    /// A configuration value handed through unchanged (`mbar`, `cta_mask`, ...).
    Value(Any),
    /// `gather4`: the tuple of row coordinates.
    Values(Vec<Any>),
}

#[derive(Clone)]
pub struct ParsedTileCall {
    pub kind: TileOpKind,
    pub exec_scope: String,
    pub destination: TileRegion,
    pub operands: Vec<TileOperand>,
    pub axes: Vec<i64>,
    pub accum: bool,
    pub attributes: Vec<(&'static str, TileAttr)>,
    /// The typed facts of a `gemm_async` call, read by its lowering.
    gemm_async: Option<Box<GemmAsyncFacts>>,
}

impl ParsedTileCall {
    fn new(
        kind: TileOpKind,
        exec_scope: &str,
        destination: TileRegion,
        operands: Vec<TileOperand>,
        attributes: Vec<(&'static str, TileAttr)>,
    ) -> Self {
        Self {
            kind,
            exec_scope: exec_scope.to_owned(),
            destination,
            operands,
            axes: Vec::new(),
            accum: false,
            attributes,
            gemm_async: None,
        }
    }

    fn with_gemm_async(mut self, facts: GemmAsyncFacts) -> Self {
        self.gemm_async = Some(Box::new(facts));
        self
    }

    /// The typed facts of a `gemm_async` call.
    pub fn gemm_async(&self) -> AResult<&GemmAsyncFacts> {
        self.gemm_async.as_deref().ok_or_else(|| {
            Failure::Ffi(super::util::ffi_error(
                "gemm_async call resolved without its facts",
            ))
        })
    }

    pub fn attribute(&self, name: &str) -> Option<&TileAttr> {
        self.attributes
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value)
    }

    pub fn attr_bool(&self, name: &str) -> bool {
        match self.attribute(name) {
            Some(TileAttr::Bool(value)) => *value,
            Some(TileAttr::Int(value)) => *value != 0,
            _ => false,
        }
    }

    pub fn attr_int(&self, name: &str) -> Option<i64> {
        match self.attribute(name) {
            Some(TileAttr::Int(value)) => Some(*value),
            Some(TileAttr::Bool(value)) => Some(i64::from(*value)),
            _ => None,
        }
    }

    pub fn attr_str(&self, name: &str) -> Option<&str> {
        match self.attribute(name) {
            Some(TileAttr::Str(value)) => Some(value.as_str()),
            _ => None,
        }
    }
}

// ----------------------------------------------------------------------
// The tile op registry.
// ----------------------------------------------------------------------

/// `_plain_text(node.op.name)`.
pub fn tile_op_name(call: &TilePrimitiveCallObj) -> AResult<String> {
    Ok(ffi_text(&call.op.name()?))
}

/// The typed kind of this tile op, or fail closed.
fn require_tile_spec(call: &TilePrimitiveCallObj) -> AResult<TileOpKind> {
    let op_name = tile_op_name(call)?;
    let Some(kind) = TileOpKind::from_op_name(&op_name) else {
        return unsupported(format!(
            "TilePrimitiveCall({op_name}, scope={}): unregistered tile operation",
            call.scope.name()?
        ));
    };
    Ok(kind)
}

/// Validate one TIRx tile op and return transient lowering facts.
pub fn resolve_tile_call(ctx: &Ctx, node: &ObjectRef) -> AResult<ParsedTileCall> {
    let call = node.as_node::<TilePrimitiveCallObj>().ok_or_else(|| {
        Failure::Ffi(super::util::ffi_error(
            "resolve_tile_call expects a TilePrimitiveCall",
        ))
    })?;
    match require_tile_spec(call)? {
        TileOpKind::Copy => resolve_copy(ctx, call),
        TileOpKind::CopyAsync => resolve_copy_async(ctx, node, call),
        TileOpKind::Cast => resolve_cast(ctx, call),
        kind @ (TileOpKind::Zero
        | TileOpKind::Sqrt
        | TileOpKind::Fill
        | TileOpKind::Reciprocal
        | TileOpKind::Silu
        | TileOpKind::Exp
        | TileOpKind::Exp2
        | TileOpKind::Log2) => resolve_unary_elementwise(ctx, call, kind),
        kind @ (TileOpKind::Add
        | TileOpKind::Sub
        | TileOpKind::Mul
        | TileOpKind::Maximum
        | TileOpKind::Fdiv
        | TileOpKind::Fma) => resolve_elementwise(ctx, call, kind),
        kind @ (TileOpKind::Sum | TileOpKind::Max | TileOpKind::Min) => {
            resolve_reduction(ctx, call, kind)
        }
        TileOpKind::Gemm => resolve_gemm(ctx, call),
        TileOpKind::GemmAsync => resolve_gemm_async(ctx, call),
        TileOpKind::PermuteLayout => resolve_permute_layout(ctx, call),
    }
}

fn is_tmem_scope(scope: &str) -> bool {
    scope == "tmem" || scope.starts_with("tmem.")
}

pub fn tile_lowering_uses_tmem(operation: &ParsedTileCall) -> bool {
    if is_tmem_scope(&operation.destination.memory_scope) {
        return true;
    }
    operation
        .operands
        .iter()
        .filter_map(TileOperand::region)
        .any(|region| is_tmem_scope(&region.memory_scope))
}

// ----------------------------------------------------------------------
// Unknown calls in the sync backward slice, tile part.
// ----------------------------------------------------------------------

/// The sync backward slice has an unknown call when a sink expression references
/// a `Call` without a resolved source-map entry.  Every expression of the
/// function body is a source-map node, so only the protocol sinks synthesized
/// from normalized tile attributes can carry such a call; this scans exactly
/// those sinks with the same allowlist.
pub fn tile_sinks_have_unknown_call(
    ctx: &Ctx,
    op_ids: &super::util::IdMap<i64>,
    tile_ops: &[&ParsedTileCall],
) -> AResult<bool> {
    let mut sinks: Vec<ObjectRef> = Vec::new();
    let push_any = |sinks: &mut Vec<ObjectRef>, value: &Any| -> AResult<()> {
        if let AnyValue::Object(node) = any_value(value)? {
            sinks.push(node);
        }
        Ok(())
    };
    for operation in tile_ops {
        match operation.kind {
            TileOpKind::CopyAsync => {
                for key in ["mbar", "cta_mask", "remote_cta_id"] {
                    if let Some(TileAttr::Value(value)) = operation.attribute(key) {
                        push_any(&mut sinks, value)?;
                    }
                }
            }
            TileOpKind::GemmAsync => {
                sinks.extend(
                    operation
                        .destination
                        .mins
                        .iter()
                        .map(|min| oref(min.clone())),
                );
                for operand in &operation.operands {
                    match operand {
                        TileOperand::Scalar(scalar) => {
                            // `hasattr(operand.expr, "ty")`: IR expressions only.
                            if let AnyValue::Object(node) = any_value(&scalar.expr)? {
                                if super::util::expr_type(&node).is_some() {
                                    sinks.push(node);
                                }
                            }
                        }
                        TileOperand::Region(region) => {
                            sinks.extend(region.mins.iter().map(|min| oref(min.clone())));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for sink in sinks {
        for candidate in crate::post_order_nodes(sink)?.iter() {
            let Some(call) = candidate.as_node::<tvm::ir::CallObj>() else {
                continue;
            };
            if crate::decode::projected_buffer(&candidate)?.is_some() {
                continue;
            }
            if op_ids.get(&candidate).is_some() {
                // Emission has already checked calls from the source body.
                continue;
            }
            let name = super::buffers::call_op_name(call)?.unwrap_or_default();
            let signature = match name.as_str() {
                "tirx.address_of" => crate::emit::pure::address_of_signature(ctx, &candidate, call),
                "tirx.reinterpret" => {
                    crate::emit::pure::reinterpret_signature(ctx, &candidate, call)
                }
                "tirx.cuda.smem_addr_from_uint64" | "tirx.cuda.sm100_2sm_leader_smem_addr" => {
                    crate::emit::pure::shared_address_signature(ctx, &candidate, call)
                }
                _ => return Ok(true),
            };
            match signature {
                Ok(_) => continue,
                Err(Failure::Unsupported { .. }) => return Ok(true),
                Err(error) => return Err(error),
            }
        }
    }
    Ok(false)
}

/// Whether the tile sinks of the sync slice reference an unknown call, for the
/// kernels the analysis-capable emitter classifies.
pub fn fixed_trace_has_unknown_call(
    ctx: &Ctx,
    op_ids: &super::util::IdMap<i64>,
    tile_calls: &[Option<ParsedTileCall>],
) -> AResult<bool> {
    let tile_ops: Vec<&ParsedTileCall> = tile_calls.iter().flatten().collect();
    tile_sinks_have_unknown_call(ctx, op_ids, &tile_ops)
}
