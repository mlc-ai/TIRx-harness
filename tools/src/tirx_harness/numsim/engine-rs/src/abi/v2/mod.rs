//! Minimal, instruction-shaped ABI for generated NumSim artifacts.
//!
//! Host launch and generated Rust storage live in the separate, explicitly
//! non-instruction [`crate::artifact_support`] module.

pub mod addr;
pub mod async_copy;
pub mod collective;
pub mod control;
mod error;
pub(crate) mod instruction;
pub mod matrix;
pub mod mem;
pub mod reg;
pub mod sync;
pub mod tcgen05;
pub mod tile;
pub mod tmem;
pub(crate) mod transport;
pub mod warp;

pub use crate::high_precision;
pub use error::EngineError;
pub use transport::{
    Address, AddressViewAccess, BufferHandle, DescriptorDomain, DirectAddress, ElementMap,
    ElementRef, Engine, ExecCtx, Generic, Global, LaneId, LaneIds, LaneMask, Local, LogicalCoord,
    MapError, MappedView, MemorySpace, ReadOnly, ReadWrite, Register, Shared, SharedCluster,
    SharedCta, SiteId, TensorMapHandle, Tmem, ValueOrigin, WarpHandle, WriteOnly, R,
};
