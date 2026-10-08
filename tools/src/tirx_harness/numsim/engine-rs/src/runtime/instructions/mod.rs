//! Engine-owned implementations behind the public v2 ABI facades.

pub(crate) mod async_copy;
pub(crate) mod collective;
pub(crate) mod control;
pub(crate) mod matrix;
pub(crate) mod mem;
pub(crate) mod mode_axis;
pub(crate) mod reg;
pub(crate) mod sync;
pub(crate) mod tcgen05;
pub(crate) mod tile;
pub(crate) mod tmem;
pub(crate) mod warp;

// Keep implementation files independent of their physical module nesting.
// The public contracts are declared by `abi::v2`; only their sealed execution
// hooks and transport internals are visible here.
pub(crate) mod instruction {
    pub(crate) use crate::abi::v2::instruction::*;
}

pub(crate) mod transport {
    pub(crate) use crate::abi::v2::transport::{engine, ElementLocation};
}

pub(crate) use crate::abi::v2::{
    Address, BufferHandle, DescriptorDomain, DirectAddress, ElementRef, Engine, EngineError,
    ExecCtx, Generic, Global, LaneId, LaneMask, Local, LogicalCoord, MappedView, MemorySpace,
    Register, Shared, SharedCluster, SharedCta, SiteId, TensorMapHandle, Tmem, WarpHandle, R,
};

/// Open the engine operation for one instruction.
///
/// Shared by the typed (`tile::`) and raw (`async_copy::`, `tcgen05::`)
/// families. `async_copy` and `tcgen05` each carried a byte-identical private
/// copy of this; `tile` carried the same body with `kind` and `always` fixed to
/// `AsyncIssue`/`true`, which its call sites now pass explicitly.
///
/// This shares the body only. Each family still owns the surrounding operand
/// validation, issue, and completion sequence.
pub(crate) fn begin(
    warp: &mut impl WarpHandle,
    context: ExecCtx,
    site: SiteId,
    kind: crate::OperationKind,
    always: bool,
) -> Result<Option<crate::OperationContext>, EngineError> {
    transport::engine(warp)
        .begin_optional_operation(context.into_inner(), site.get(), kind, always)
        .map_err(Into::into)
}

/// Close the engine operation opened by [`begin`].
///
/// Was byte-identical in `tile::`, `async_copy::`, and `tcgen05::`.
pub(crate) fn finish(
    warp: &mut impl WarpHandle,
    operation: &Option<crate::OperationContext>,
) -> Result<(), EngineError> {
    transport::engine(warp)
        .finish_optional_operation(operation)
        .map_err(Into::into)
}

/// Resolve the one issuing lane of a single-lane instruction.
///
/// The raw path spelled this `tma_lane` and the typed path
/// `singleton_issue_lane`; the two bodies computed the same thing and already
/// produced a byte-identical error message.
pub(crate) fn singleton_issue_lane(context: ExecCtx, label: &str) -> Result<usize, EngineError> {
    let mask = context.active_mask();
    if mask.len() != 1 {
        return Err(EngineError::message(format!(
            "{label} requires exactly one issuing lane, got mask {:#010x}",
            mask.bits()
        )));
    }
    Ok(mask.first_active().expect("one-lane mask has a lane"))
}
