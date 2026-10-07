//! Raw TCGEN05 instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::tcgen05::{
    alloc, commit, cp, dealloc, fence, ld, mma, mma_sp, relinquish_alloc_permit, shift, st,
    variant, wait_ld, wait_st, AllocVariant, CommitVariant, CpVariant, DeallocVariant,
    FenceVariant, LdVariant, MmaSpVariant, MmaVariant, RelinquishAllocPermitVariant, ShiftVariant,
    StVariant,
};
