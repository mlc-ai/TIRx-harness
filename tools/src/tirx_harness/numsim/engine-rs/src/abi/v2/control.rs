//! Control-flow and scheduling ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::control::{
    branch_context, clc_query_cancel, clc_try_cancel, for_enter, for_exit, griddepcontrol,
    nanosleep, setmaxnreg, trap, variant, while_enter, while_exit, BranchVariant,
    ClcQueryCancelVariant, ElectSync, Ordinary, SetmaxnregVariant,
};
