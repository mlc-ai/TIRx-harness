//! Typed whole-tile ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::tile::{
    copy, cp_async, cp_async_bulk, cp_async_bulk_tensor, cp_reduce_async_bulk_tensor, gemm,
    gemm_async, gemm_async_ws, tcgen05_cp, tcgen05_ld, tcgen05_st, variant, BulkCopyVariant,
    CopyVariant, CpAsyncVariant, GemmAPlacement, GemmAsyncVariant, GemmAsyncWsVariant, GemmMapping,
    GemmMode, GemmVariant, Tcgen05CpVariant, Tcgen05LdMapping, Tcgen05LdVariant, Tcgen05StMapping,
    Tcgen05StVariant, TensorCopyVariant, TensorReduceVariant,
};
