//! Warp-scope instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::warp::{
    activemask, bar_warp_sync, context, elect_sync, match_sync, movmatrix, redux_sync,
    shfl_source_mask_bfly, shfl_source_mask_down, shfl_source_mask_idx, shfl_source_mask_up,
    shfl_sync, variant, vote_sync, MatchSyncVariant, MovMatrixVariant, ReduxSyncVariant,
    ShflSyncVariant, VoteSyncVariant,
};
