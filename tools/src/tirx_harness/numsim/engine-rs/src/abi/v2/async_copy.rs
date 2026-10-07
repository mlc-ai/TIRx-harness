//! Async-copy instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::async_copy::{
    applypriority, cp_async, cp_async_bulk, cp_async_bulk_commit_group, cp_async_bulk_prefetch,
    cp_async_bulk_prefetch_tensor, cp_async_bulk_tensor, cp_async_bulk_wait_group,
    cp_async_commit_group, cp_async_mbarrier_arrive, cp_async_wait_group, cp_reduce_async_bulk,
    cp_reduce_async_bulk_tensor, override_tensor_map, prefetch_tensormap, prefetch_valid_address,
    red_async, st_async, variant, ApplyPriorityVariant, CpAsyncBulkPrefetchTensorVariant,
    CpAsyncBulkTensorVariant, CpAsyncBulkVariant, CpAsyncBulkWaitGroupVariant,
    CpAsyncMbarrierArriveVariant, CpAsyncVariant, CpAsyncWaitGroupVariant,
    CpReduceAsyncBulkTensorVariant, CpReduceAsyncBulkVariant, RedAsyncVariant, StAsyncVariant,
};
