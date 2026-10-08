//! Source-level collective ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::collective::{
    bar_reduce, cta_reduce, cta_vote, grid_sync, participate, scope, variant, BarReduceVariant,
    CtaReduceType, CtaReduceVariant,
};
