//! Barrier and fence instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::sync::{
    bar_arrive, bar_sync, barrier_cluster_arrive, barrier_cluster_wait, barrier_sync, fence,
    fence_mbarrier_init, fence_proxy_async, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_check_layout, mbarrier_complete_tx, mbarrier_expect_tx, mbarrier_init, mbarrier_inval,
    mbarrier_pending_count, mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait_until,
    tensormap_cp_fenceproxy, tensormap_replace, variant, BarrierClusterArriveVariant,
    BarrierClusterWaitVariant, FenceProxyAsyncVariant, FenceVariant, MbarrierArriveExpectTxVariant,
    MbarrierArriveVariant, MbarrierCompleteTxVariant, MbarrierExpectTxVariant, MbarrierInitVariant,
    MbarrierTestWaitVariant, MbarrierTryWaitVariant, MbarrierWaitUntilVariant,
    TensorMapCopyVariant, TensorMapReplaceVariant,
};
