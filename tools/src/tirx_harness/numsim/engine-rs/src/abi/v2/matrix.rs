//! Register-fragment matrix instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::matrix::{
    mma_sp_sync, mma_sync, variant, MmaSpSyncVariant, MmaSyncVariant,
};
