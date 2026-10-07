//! Memory-instruction ABI.
//!
//! This module is intentionally declaration-sized. Concrete specialization
//! bodies live in `runtime/instructions/mem.rs` and are compiled as part of
//! the engine crate.

pub use crate::runtime::instructions::mem::{
    atom, declared_wait, discard, ld, ldmatrix, red, st, st_bulk, stmatrix, variant, AtomVariant,
    LdVariant, LdmatrixVariant, MemoryType, RedVariant, StBulkVariant, StVariant, StmatrixVariant,
};
