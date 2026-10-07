use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::memory::{
    OwnerPrivateReadSession, OwnerPrivateWriteSession, ReadSource, SemanticProgress,
    SemanticProgressSnapshot, SemanticProgressWatch, UninitializedReadPolicy,
    UninitializedReadReview,
};
use crate::{
    AllocationId, BufferView, EngineError, GlobalMemory, LaunchTopology, MemoryError,
    RuntimeScalar, WarpContext, WarpMask, WarpValue, WARP_SIZE,
};

/// One concrete CTA in a launch topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CtaId {
    cluster_id: usize,
    cta_id_in_cluster: usize,
}

impl CtaId {
    pub fn new(
        topology: LaunchTopology,
        cluster_id: usize,
        cta_id_in_cluster: usize,
    ) -> Result<Self, AddressSpaceError> {
        let cta = Self {
            cluster_id,
            cta_id_in_cluster,
        };
        validate_cta(topology, cta)?;
        Ok(cta)
    }

    pub const fn from_context(context: WarpContext) -> Self {
        Self {
            cluster_id: context.cluster_id(),
            cta_id_in_cluster: context.cta_id_in_cluster(),
        }
    }

    pub const fn cluster_id(self) -> usize {
        self.cluster_id
    }

    pub const fn cta_id_in_cluster(self) -> usize {
        self.cta_id_in_cluster
    }

    pub fn global_cta_id(self, topology: LaunchTopology) -> Result<usize, AddressSpaceError> {
        validate_cta(topology, self)?;
        Ok(self.cluster_id * topology.ctas_per_cluster() + self.cta_id_in_cluster)
    }
}

impl fmt::Display for CtaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cluster {}/CTA {}",
            self.cluster_id, self.cta_id_in_cluster
        )
    }
}

/// One concrete warp in a launch topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WarpId {
    cta: CtaId,
    warp_id_in_cta: usize,
}

impl WarpId {
    pub fn new(
        topology: LaunchTopology,
        cta: CtaId,
        warp_id_in_cta: usize,
    ) -> Result<Self, AddressSpaceError> {
        let warp = Self {
            cta,
            warp_id_in_cta,
        };
        validate_warp(topology, warp)?;
        Ok(warp)
    }

    pub const fn from_context(context: WarpContext) -> Self {
        Self {
            cta: CtaId::from_context(context),
            warp_id_in_cta: context.warp_id_in_cta(),
        }
    }

    pub const fn cta(self) -> CtaId {
        self.cta
    }

    pub const fn warp_id_in_cta(self) -> usize {
        self.warp_id_in_cta
    }

    pub fn global_warp_id(self, topology: LaunchTopology) -> Result<usize, AddressSpaceError> {
        validate_warp(topology, self)?;
        Ok(self.cta.global_cta_id(topology)? * topology.warps_per_cta() + self.warp_id_in_cta)
    }
}

impl fmt::Display for WarpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/warp {}", self.cta, self.warp_id_in_cta)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicalOwner {
    Cta(CtaId),
    Warp(WarpId),
}

impl fmt::Display for PhysicalOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cta(cta) => write!(f, "{cta}"),
            Self::Warp(warp) => write!(f, "{warp}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AddressSpaceError {
    Memory(MemoryError),
    InvalidCta {
        cta: CtaId,
        clusters: usize,
        ctas_per_cluster: usize,
    },
    InvalidWarp {
        warp: WarpId,
        warps_per_cta: usize,
    },
    InvalidLane {
        lane: usize,
    },
    OwnerMismatch {
        allocation: AllocationId,
        expected: PhysicalOwner,
        actual: PhysicalOwner,
    },
    CrossClusterSharedAccess {
        requester: CtaId,
        target: CtaId,
    },
    SizeOverflow,
    TmemRegionOutOfBounds {
        allocation: AllocationId,
        lanes: usize,
        columns: usize,
        lane_offset: usize,
        lane_count: usize,
        column_offset: usize,
        column_count: usize,
    },
    TmemCellByteOutOfBounds {
        byte_offset: usize,
        byte_len: usize,
    },
}

impl fmt::Display for AddressSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(source) => source.fmt(f),
            Self::InvalidCta {
                cta,
                clusters,
                ctas_per_cluster,
            } => write!(
                f,
                "{cta} is outside launch topology ({clusters} clusters, {ctas_per_cluster} CTAs/cluster)"
            ),
            Self::InvalidWarp {
                warp,
                warps_per_cta,
            } => write!(
                f,
                "{warp} is outside launch warp count {warps_per_cta} per CTA"
            ),
            Self::InvalidLane { lane } => {
                write!(f, "lane {lane} is outside warp size {WARP_SIZE}")
            }
            Self::OwnerMismatch {
                allocation,
                expected,
                actual,
            } => write!(
                f,
                "{allocation} is owned by {actual}, not requested owner {expected}"
            ),
            Self::CrossClusterSharedAccess { requester, target } => write!(
                f,
                "{requester} cannot address remote shared memory owned by {target}"
            ),
            Self::SizeOverflow => f.write_str("physical address-space size overflow"),
            Self::TmemRegionOutOfBounds {
                allocation,
                lanes,
                columns,
                lane_offset,
                lane_count,
                column_offset,
                column_count,
            } => write!(
                f,
                "TMEM region lanes [{lane_offset}, {lane_offset}+{lane_count}), columns [{column_offset}, {column_offset}+{column_count}) exceeds {allocation} shape [{lanes}, {columns}]"
            ),
            Self::TmemCellByteOutOfBounds {
                byte_offset,
                byte_len,
            } => write!(
                f,
                "TMEM cell access [{byte_offset}, {byte_offset}+{byte_len}) exceeds {TMEM_CELL_BYTES}-byte cell"
            ),
        }
    }
}

impl Error for AddressSpaceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Memory(source) => Some(source),
            _ => None,
        }
    }
}

impl From<MemoryError> for AddressSpaceError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SharedAllocation {
    allocation: AllocationId,
    raw: BufferView,
    owner: PhysicalOwner,
}

impl SharedAllocation {
    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn owner(&self) -> PhysicalOwner {
        self.owner
    }

    pub const fn byte_len(&self) -> usize {
        self.raw.byte_len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SharedView {
    raw: BufferView,
    owner: PhysicalOwner,
}

impl SharedView {
    pub fn buffer_view(&self) -> BufferView {
        self.raw.clone()
    }

    pub const fn owner(&self) -> PhysicalOwner {
        self.owner
    }

    pub const fn byte_len(&self) -> usize {
        self.raw.byte_len()
    }
}

/// CTA-owned shared memory with explicit same-cluster remote access.
///
/// CTA memory is private through [`SharedMemory::cta_view`].  A remote access
/// must name both requester and target through [`SharedMemory::remote_cta_view`]
/// and is accepted only within one cluster.
#[derive(Clone)]
pub struct SharedMemory {
    topology: LaunchTopology,
    bytes: GlobalMemory,
}

impl SharedMemory {
    pub fn new(topology: LaunchTopology) -> Self {
        Self {
            topology,
            bytes: GlobalMemory::new_queued_owner_private(),
        }
    }

    fn with_semantic_progress(topology: LaunchTopology, progress: SemanticProgress) -> Self {
        Self {
            topology,
            bytes: GlobalMemory::new_queued_owner_private_with_semantic_progress(progress),
        }
    }

    fn with_read_policy(mut self, read_policy: UninitializedReadPolicy) -> Self {
        self.bytes = self.bytes.with_uninitialized_read_policy(read_policy);
        self
    }

    pub(crate) fn with_read_session<R>(
        &self,
        view: &SharedView,
        operation: impl FnOnce(&mut OwnerPrivateReadSession<'_>) -> R,
    ) -> Result<R, AddressSpaceError> {
        Ok(self
            .bytes
            .with_owner_private_read_session(&view.raw, operation)?)
    }

    pub const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    pub fn allocate_cta_uninitialized(
        &self,
        owner: CtaId,
        byte_len: usize,
    ) -> Result<SharedAllocation, AddressSpaceError> {
        validate_cta(self.topology, owner)?;
        self.allocate_uninitialized(PhysicalOwner::Cta(owner), byte_len)
    }

    pub fn allocate_cta_zeroed(
        &self,
        owner: CtaId,
        byte_len: usize,
    ) -> Result<SharedAllocation, AddressSpaceError> {
        validate_cta(self.topology, owner)?;
        self.allocate_zeroed(PhysicalOwner::Cta(owner), byte_len)
    }

    pub fn cta_view(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<SharedView, AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        let expected = PhysicalOwner::Cta(requester);
        validate_owner(allocation.allocation, allocation.owner, expected)?;
        self.make_view(allocation, byte_offset, byte_len)
    }

    pub fn full_cta_view(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
    ) -> Result<SharedView, AddressSpaceError> {
        let byte_len = allocation.raw.byte_len();
        self.cta_view(requester, allocation, 0, byte_len)
    }

    pub fn remote_cta_view(
        &self,
        requester: CtaId,
        target: CtaId,
        allocation: &SharedAllocation,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<SharedView, AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_cta(self.topology, target)?;
        if requester.cluster_id != target.cluster_id {
            return Err(AddressSpaceError::CrossClusterSharedAccess { requester, target });
        }
        let expected = PhysicalOwner::Cta(target);
        validate_owner(allocation.allocation, allocation.owner, expected)?;
        self.make_view(allocation, byte_offset, byte_len)
    }

    pub fn full_remote_cta_view(
        &self,
        requester: CtaId,
        target: CtaId,
        allocation: &SharedAllocation,
    ) -> Result<SharedView, AddressSpaceError> {
        let byte_len = allocation.raw.byte_len();
        self.remote_cta_view(requester, target, allocation, 0, byte_len)
    }

    pub fn read_cta_bytes_into(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(requester),
        )?;
        let absolute = resolve_view_access(
            allocation.allocation,
            allocation.raw.byte_len(),
            view_offset,
            view_len,
            byte_offset,
            target.len(),
        )?;
        self.bytes
            .read_bytes_into(&allocation.raw, absolute, target)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn read_cta_bytes_batch_into(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(requester),
        )?;
        let mut absolutes = WarpValue::splat(0_usize);
        for lane in mask {
            absolutes[lane] = resolve_view_access(
                allocation.allocation,
                allocation.raw.byte_len(),
                view_offset,
                view_len,
                byte_offsets[lane],
                byte_len,
            )?;
        }
        self.bytes.read_owner_private_resolved_bytes_batch_into(
            &allocation.raw,
            &absolutes,
            mask,
            byte_len,
            target_stride,
            target,
        )?;
        Ok(())
    }

    pub fn read_cta_bytes_zero_filled_into(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(requester),
        )?;
        let absolute = resolve_view_access(
            allocation.allocation,
            allocation.raw.byte_len(),
            view_offset,
            view_len,
            byte_offset,
            target.len(),
        )?;
        self.bytes
            .read_bytes_zero_filled_into(&allocation.raw, absolute, target)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_remote_cta_bytes_into(
        &self,
        requester: CtaId,
        target_cta: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_cta(self.topology, target_cta)?;
        if requester.cluster_id != target_cta.cluster_id {
            return Err(AddressSpaceError::CrossClusterSharedAccess {
                requester,
                target: target_cta,
            });
        }
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(target_cta),
        )?;
        let absolute = resolve_view_access(
            allocation.allocation,
            allocation.raw.byte_len(),
            view_offset,
            view_len,
            byte_offset,
            target.len(),
        )?;
        self.bytes
            .read_bytes_into(&allocation.raw, absolute, target)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_cta_bytes(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(requester),
        )?;
        let absolute = resolve_view_access(
            allocation.allocation,
            allocation.raw.byte_len(),
            view_offset,
            view_len,
            byte_offset,
            bytes.len(),
        )?;
        self.bytes.write_bytes(&allocation.raw, absolute, bytes)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn write_cta_bytes_batch(
        &self,
        requester: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        source_stride: usize,
        source: &[u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(requester),
        )?;
        let mut absolutes = WarpValue::splat(0_usize);
        for lane in mask {
            absolutes[lane] = resolve_view_access(
                allocation.allocation,
                allocation.raw.byte_len(),
                view_offset,
                view_len,
                byte_offsets[lane],
                byte_len,
            )?;
        }
        self.bytes.write_owner_private_resolved_bytes_batch(
            &allocation.raw,
            &absolutes,
            mask,
            byte_len,
            source_stride,
            source,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_remote_cta_bytes(
        &self,
        requester: CtaId,
        target_cta: CtaId,
        allocation: &SharedAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_cta(self.topology, target_cta)?;
        if requester.cluster_id != target_cta.cluster_id {
            return Err(AddressSpaceError::CrossClusterSharedAccess {
                requester,
                target: target_cta,
            });
        }
        validate_owner(
            allocation.allocation,
            allocation.owner,
            PhysicalOwner::Cta(target_cta),
        )?;
        let absolute = resolve_view_access(
            allocation.allocation,
            allocation.raw.byte_len(),
            view_offset,
            view_len,
            byte_offset,
            bytes.len(),
        )?;
        self.bytes.write_bytes(&allocation.raw, absolute, bytes)?;
        Ok(())
    }

    pub(crate) fn write_bytes_batch<'a>(
        &self,
        view: &SharedView,
        writes: impl IntoIterator<Item = (usize, &'a [u8])>,
    ) -> Result<(), AddressSpaceError> {
        self.bytes
            .publish_owner_private_bytes_batch(&view.raw, writes)?;
        Ok(())
    }

    pub(crate) fn with_write_session<R>(
        &self,
        view: &SharedView,
        operation: impl FnOnce(&mut OwnerPrivateWriteSession<'_>) -> R,
    ) -> Result<R, AddressSpaceError> {
        Ok(self
            .bytes
            .with_owner_private_write_session(&view.raw, operation)?)
    }

    pub fn subview(
        &self,
        parent: &SharedView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<SharedView, AddressSpaceError> {
        Ok(SharedView {
            raw: self.bytes.subview(&parent.raw, byte_offset, byte_len)?,
            owner: parent.owner,
        })
    }

    fn allocate_uninitialized(
        &self,
        owner: PhysicalOwner,
        byte_len: usize,
    ) -> Result<SharedAllocation, AddressSpaceError> {
        let allocation = self.bytes.allocate_uninitialized(byte_len)?;
        Ok(SharedAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
        })
    }

    fn allocate_zeroed(
        &self,
        owner: PhysicalOwner,
        byte_len: usize,
    ) -> Result<SharedAllocation, AddressSpaceError> {
        let allocation = self.bytes.allocate_zeroed(byte_len)?;
        Ok(SharedAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
        })
    }

    fn make_view(
        &self,
        allocation: &SharedAllocation,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<SharedView, AddressSpaceError> {
        Ok(SharedView {
            raw: self.bytes.subview(&allocation.raw, byte_offset, byte_len)?,
            owner: allocation.owner,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WarpPrivateAllocation {
    allocation: AllocationId,
    raw: BufferView,
    owner: WarpId,
    bytes_per_lane: usize,
}

impl WarpPrivateAllocation {
    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn owner(&self) -> WarpId {
        self.owner
    }

    pub const fn bytes_per_lane(&self) -> usize {
        self.bytes_per_lane
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WarpPrivateView {
    raw: BufferView,
    owner: WarpId,
    lane: usize,
}

impl WarpPrivateView {
    pub fn buffer_view(&self) -> BufferView {
        self.raw.clone()
    }

    pub const fn owner(&self) -> WarpId {
        self.owner
    }

    pub const fn lane(&self) -> usize {
        self.lane
    }

    pub const fn byte_len(&self) -> usize {
        self.raw.byte_len()
    }
}

/// Per-warp backing allocations split into 32 lane-owned byte ranges.
#[derive(Clone)]
pub struct WarpPrivateMemory {
    topology: LaunchTopology,
    bytes: GlobalMemory,
    semantic_progress: Arc<WarpPrivateSemanticProgress>,
}

struct WarpPrivateSemanticProgress {
    enabled: AtomicBool,
    generations: Box<[AtomicU64]>,
}

impl WarpPrivateSemanticProgress {
    fn new(warp_count: usize) -> Self {
        Self {
            enabled: AtomicBool::new(false),
            generations: (0..warp_count)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    fn observes_changes(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    fn record_change(&self, global_warp_id: usize) {
        if !self.observes_changes() {
            return;
        }
        self.generations[global_warp_id]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("NumSim per-warp memory-progress generation exhausted");
    }

    fn snapshot(&self, global_warp_id: usize) -> u64 {
        self.generations[global_warp_id].load(Ordering::Relaxed)
    }

    fn aggregate_snapshot(&self) -> u64 {
        self.generations
            .iter()
            .map(|generation| generation.load(Ordering::Relaxed))
            .try_fold(0_u64, u64::checked_add)
            .expect("NumSim aggregate warp-private memory progress overflowed")
    }
}

/// LOCAL and REG use the same physical ownership rules but normally use
/// distinct manager instances in [`PhysicalMemory`].
pub type LocalMemory = WarpPrivateMemory;
pub type RegisterMemory = WarpPrivateMemory;

impl WarpPrivateMemory {
    pub fn new(topology: LaunchTopology) -> Self {
        Self {
            topology,
            bytes: GlobalMemory::new_owner_private(),
            semantic_progress: Arc::new(WarpPrivateSemanticProgress::new(topology.warp_count())),
        }
    }

    fn with_read_policy(mut self, read_policy: UninitializedReadPolicy) -> Self {
        self.bytes = self.bytes.with_uninitialized_read_policy(read_policy);
        self
    }

    pub const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    fn record_semantic_progress(&self, requester: WarpId, changed: bool) {
        if changed {
            let global_warp_id = requester
                .global_warp_id(self.topology)
                .expect("validated warp-private requester stays inside its topology");
            self.semantic_progress.record_change(global_warp_id);
        }
    }

    pub fn allocate_uninitialized(
        &self,
        owner: WarpId,
        bytes_per_lane: usize,
    ) -> Result<WarpPrivateAllocation, AddressSpaceError> {
        validate_warp(self.topology, owner)?;
        let byte_len = bytes_per_lane
            .checked_mul(WARP_SIZE)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let allocation = self.bytes.allocate_uninitialized(byte_len)?;
        Ok(WarpPrivateAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
            bytes_per_lane,
        })
    }

    pub fn allocate_zeroed(
        &self,
        owner: WarpId,
        bytes_per_lane: usize,
    ) -> Result<WarpPrivateAllocation, AddressSpaceError> {
        validate_warp(self.topology, owner)?;
        let byte_len = bytes_per_lane
            .checked_mul(WARP_SIZE)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let allocation = self.bytes.allocate_zeroed(byte_len)?;
        Ok(WarpPrivateAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
            bytes_per_lane,
        })
    }

    pub fn lane_view(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        lane: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<WarpPrivateView, AddressSpaceError> {
        validate_warp(self.topology, requester)?;
        if lane >= WARP_SIZE {
            return Err(AddressSpaceError::InvalidLane { lane });
        }
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let lane_base = lane
            .checked_mul(allocation.bytes_per_lane)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let lane_full =
            self.bytes
                .subview(&allocation.raw, lane_base, allocation.bytes_per_lane)?;
        Ok(WarpPrivateView {
            raw: self.bytes.subview(&lane_full, byte_offset, byte_len)?,
            owner: requester,
            lane,
        })
    }

    pub fn full_lane_view(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        lane: usize,
    ) -> Result<WarpPrivateView, AddressSpaceError> {
        let bytes_per_lane = allocation.bytes_per_lane;
        self.lane_view(requester, allocation, lane, 0, bytes_per_lane)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_lane_bytes_into(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        lane: usize,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_warp(self.topology, requester)?;
        if lane >= WARP_SIZE {
            return Err(AddressSpaceError::InvalidLane { lane });
        }
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let lane_base = lane
            .checked_mul(allocation.bytes_per_lane)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let lane_relative = resolve_view_access(
            allocation.allocation,
            allocation.bytes_per_lane,
            view_offset,
            view_len,
            byte_offset,
            target.len(),
        )?;
        let absolute = lane_base
            .checked_add(lane_relative)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        self.bytes
            .read_bytes_into(&allocation.raw, absolute, target)?;
        Ok(())
    }

    #[inline]
    fn resolve_lane_batch_offsets(
        allocation: &WarpPrivateAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
    ) -> Result<WarpValue<usize>, AddressSpaceError> {
        let mut absolutes = WarpValue::splat(0_usize);
        let uniform_relative = mask.first_active().and_then(|first_lane| {
            let first = byte_offsets[first_lane];
            mask.into_iter()
                .all(|lane| byte_offsets[lane] == first)
                .then_some(first)
        });
        let resolved_uniform = uniform_relative
            .map(|relative| {
                resolve_view_access(
                    allocation.allocation,
                    allocation.bytes_per_lane,
                    view_offset,
                    view_len,
                    relative,
                    byte_len,
                )
            })
            .transpose()?;
        for lane in mask {
            let lane_base = lane
                .checked_mul(allocation.bytes_per_lane)
                .ok_or(AddressSpaceError::SizeOverflow)?;
            let lane_relative = if let Some(relative) = resolved_uniform {
                relative
            } else {
                resolve_view_access(
                    allocation.allocation,
                    allocation.bytes_per_lane,
                    view_offset,
                    view_len,
                    byte_offsets[lane],
                    byte_len,
                )?
            };
            absolutes[lane] = lane_base
                .checked_add(lane_relative)
                .ok_or(AddressSpaceError::SizeOverflow)?;
        }
        Ok(absolutes)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn read_lane_bytes_batch_into(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        validate_warp(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let absolutes = Self::resolve_lane_batch_offsets(
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            mask,
            byte_len,
        )?;
        self.bytes.read_owner_private_resolved_bytes_batch_into(
            &allocation.raw,
            &absolutes,
            mask,
            byte_len,
            target_stride,
            target,
        )?;
        Ok(())
    }

    #[inline]
    pub(crate) fn read_lane_scalars_batch<T: RuntimeScalar>(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        source: Option<ReadSource>,
    ) -> Result<WarpValue<T>, EngineError> {
        validate_warp(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let absolutes = Self::resolve_lane_batch_offsets(
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            mask,
            T::BYTE_LEN,
        )?;
        self.bytes.read_owner_private_resolved_scalar_batch::<T>(
            &allocation.raw,
            &absolutes,
            mask,
            source,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_lane_bytes_batch(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        source_stride: usize,
        source: &[u8],
    ) -> Result<(), AddressSpaceError> {
        validate_warp(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let absolutes = Self::resolve_lane_batch_offsets(
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            mask,
            byte_len,
        )?;
        let changed = self.bytes.write_owner_private_resolved_bytes_batch(
            &allocation.raw,
            &absolutes,
            mask,
            byte_len,
            source_stride,
            source,
        )?;
        self.record_semantic_progress(requester, changed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn write_lane_scalars_batch<T: RuntimeScalar>(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        view_offset: usize,
        view_len: usize,
        byte_offsets: &WarpValue<usize>,
        values: &WarpValue<T>,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        validate_warp(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let absolutes = Self::resolve_lane_batch_offsets(
            allocation,
            view_offset,
            view_len,
            byte_offsets,
            mask,
            T::BYTE_LEN,
        )?;
        let changed = self.bytes.write_owner_private_resolved_scalar_batch(
            &allocation.raw,
            &absolutes,
            values,
            mask,
        )?;
        self.record_semantic_progress(requester, changed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_lane_bytes(
        &self,
        requester: WarpId,
        allocation: &WarpPrivateAllocation,
        lane: usize,
        view_offset: usize,
        view_len: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), AddressSpaceError> {
        validate_warp(self.topology, requester)?;
        if lane >= WARP_SIZE {
            return Err(AddressSpaceError::InvalidLane { lane });
        }
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Warp(allocation.owner),
            PhysicalOwner::Warp(requester),
        )?;
        let lane_base = lane
            .checked_mul(allocation.bytes_per_lane)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let lane_relative = resolve_view_access(
            allocation.allocation,
            allocation.bytes_per_lane,
            view_offset,
            view_len,
            byte_offset,
            bytes.len(),
        )?;
        let absolute = lane_base
            .checked_add(lane_relative)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let changed = if self.semantic_progress.observes_changes() {
            self.bytes.write_owner_private_bytes_observing_change(
                &allocation.raw,
                absolute,
                bytes,
            )?
        } else {
            self.bytes.write_bytes(&allocation.raw, absolute, bytes)?;
            false
        };
        self.record_semantic_progress(requester, changed);
        Ok(())
    }

    pub fn subview(
        &self,
        parent: &WarpPrivateView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<WarpPrivateView, AddressSpaceError> {
        Ok(WarpPrivateView {
            raw: self.bytes.subview(&parent.raw, byte_offset, byte_len)?,
            owner: parent.owner,
            lane: parent.lane,
        })
    }
}

pub const TMEM_CELL_BYTES: usize = size_of::<u32>();

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TmemAllocation {
    allocation: AllocationId,
    raw: BufferView,
    owner: CtaId,
    lanes: usize,
    columns: usize,
}

impl TmemAllocation {
    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn owner(&self) -> CtaId {
        self.owner
    }

    pub const fn lanes(&self) -> usize {
        self.lanes
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TmemRegion {
    pub lane_offset: usize,
    pub lane_count: usize,
    pub column_offset: usize,
    pub column_count: usize,
}

impl TmemRegion {
    pub const fn new(
        lane_offset: usize,
        lane_count: usize,
        column_offset: usize,
        column_count: usize,
    ) -> Self {
        Self {
            lane_offset,
            lane_count,
            column_offset,
            column_count,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TmemView {
    allocation: TmemAllocation,
    region: TmemRegion,
}

impl TmemView {
    pub fn allocation(&self) -> TmemAllocation {
        self.allocation.clone()
    }

    pub const fn owner(&self) -> CtaId {
        self.allocation.owner
    }

    pub const fn region(&self) -> TmemRegion {
        self.region
    }

    /// Physical TMEM column base. Generated layout code applies logical
    /// coordinate transforms before selecting cells from this view.
    pub const fn allocated_addr(&self) -> usize {
        self.region.column_offset
    }
}

/// CTA-owned physical TMEM cells.
///
/// The engine deliberately does not hard-code an architecture capacity or a
/// logical tensor layout. Generated code supplies the verified lane/column
/// mapping; this type owns byte-validity, aliasing, and physical cell bounds.
#[derive(Clone)]
pub struct TmemMemory {
    topology: LaunchTopology,
    bytes: GlobalMemory,
}

pub(crate) struct TmemWriteSession<'session, 'data> {
    allocation_columns: usize,
    region: TmemRegion,
    session: &'session mut OwnerPrivateWriteSession<'data>,
}

pub(crate) struct TmemReadSession<'session, 'data> {
    allocation_columns: usize,
    region: TmemRegion,
    session: &'session mut OwnerPrivateReadSession<'data>,
}

impl TmemReadSession<'_, '_> {
    #[inline(always)]
    pub(crate) const fn records_uninitialized_read_reviews(&self) -> bool {
        self.session.records_uninitialized_read_reviews()
    }

    #[inline(always)]
    pub(crate) const fn all_bytes_initialized(&self) -> bool {
        self.session.all_bytes_initialized()
    }

    #[inline(always)]
    fn cell_byte_offset_prevalidated(
        &self,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> usize {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column < self.region.column_count);
        debug_assert!(byte_offset
            .checked_add(byte_len)
            .is_some_and(|end| end <= TMEM_CELL_BYTES));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        (physical_lane * self.allocation_columns + physical_column) * TMEM_CELL_BYTES + byte_offset
    }

    #[inline(always)]
    pub(crate) fn validate_initialized_cell_bytes_prevalidated(
        &self,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), MemoryError> {
        let absolute = self.cell_byte_offset_prevalidated(lane, column, byte_offset, byte_len);
        self.session
            .validate_initialized_bytes_prevalidated(absolute, byte_len)
    }

    #[inline(always)]
    pub(crate) fn read_initialized_cell_bytes_into_prevalidated(
        &self,
        lane: usize,
        column: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) {
        let absolute = self.cell_byte_offset_prevalidated(lane, column, byte_offset, target.len());
        self.session
            .read_initialized_bytes_into_prevalidated(absolute, target);
    }

    #[inline(always)]
    pub(crate) fn read_reviewed_cell_bytes_into_prevalidated(
        &mut self,
        lane: usize,
        column: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) {
        let absolute = self.cell_byte_offset_prevalidated(lane, column, byte_offset, target.len());
        self.session
            .read_reviewed_bytes_into_prevalidated(absolute, target);
    }

    #[inline(always)]
    pub(crate) fn read_initialized_f32_cells_into_prevalidated(
        &self,
        lane: usize,
        column: usize,
        target: &mut [f32],
    ) {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column
            .checked_add(target.len())
            .is_some_and(|end| end <= self.region.column_count));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        let absolute =
            (physical_lane * self.allocation_columns + physical_column) * TMEM_CELL_BYTES;
        self.session
            .read_initialized_f32s_into_prevalidated(absolute, target);
    }

    #[inline(always)]
    pub(crate) fn validate_initialized_f32_cells_prevalidated(
        &self,
        lane: usize,
        column: usize,
        columns: usize,
    ) -> Result<(), MemoryError> {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column
            .checked_add(columns)
            .is_some_and(|end| end <= self.region.column_count));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        let absolute =
            (physical_lane * self.allocation_columns + physical_column) * TMEM_CELL_BYTES;
        self.session.validate_initialized_bytes_prevalidated(
            absolute,
            columns
                .checked_mul(size_of::<f32>())
                .expect("validated TMEM f32 validation size overflow"),
        )
    }

    #[inline(always)]
    pub(crate) fn read_reviewed_f32_cells_into_prevalidated(
        &mut self,
        lane: usize,
        column: usize,
        target: &mut [f32],
    ) {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column
            .checked_add(target.len())
            .is_some_and(|end| end <= self.region.column_count));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        let absolute =
            (physical_lane * self.allocation_columns + physical_column) * TMEM_CELL_BYTES;
        self.session
            .read_reviewed_f32s_into_prevalidated(absolute, target);
    }
}

impl TmemWriteSession<'_, '_> {
    /// Write a cell whose lane, column, and byte range were checked by the
    /// instruction preflight. The hot loop performs only address arithmetic
    /// and the backing write.
    #[inline(always)]
    pub(crate) fn write_cell_bytes_prevalidated(
        &mut self,
        lane: usize,
        column: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column < self.region.column_count);
        debug_assert!(byte_offset
            .checked_add(bytes.len())
            .is_some_and(|end| end <= TMEM_CELL_BYTES));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        let absolute = (physical_lane * self.allocation_columns + physical_column)
            * TMEM_CELL_BYTES
            + byte_offset;
        self.session.write_bytes_prevalidated(absolute, bytes);
    }

    #[inline(always)]
    pub(crate) fn write_f32_cells_prevalidated(
        &mut self,
        lane: usize,
        column: usize,
        values: &[f32],
    ) {
        debug_assert!(lane < self.region.lane_count);
        debug_assert!(column
            .checked_add(values.len())
            .is_some_and(|end| end <= self.region.column_count));
        let physical_lane = self.region.lane_offset + lane;
        let physical_column = self.region.column_offset + column;
        let absolute =
            (physical_lane * self.allocation_columns + physical_column) * TMEM_CELL_BYTES;
        self.session.write_f32s_prevalidated(absolute, values);
    }
}

impl TmemMemory {
    pub fn new(topology: LaunchTopology) -> Self {
        Self {
            topology,
            // Instruction gateways own the semantic event for TMEM writes:
            // TCGEN issue covers instruction-selected writes and the direct
            // physical-access gateway covers typed scalar stores. NumSim
            // publishes after the numeric effect; observing modes publish
            // after ordering and checker effects. Keeping the byte arena
            // silent avoids one launch-wide event per destination session.
            bytes: GlobalMemory::new_owner_private_with_semantic_progress(
                SemanticProgress::disabled(),
            ),
        }
    }

    fn with_read_policy(mut self, read_policy: UninitializedReadPolicy) -> Self {
        self.bytes = self.bytes.with_uninitialized_read_policy(read_policy);
        self
    }

    pub(crate) fn with_write_session<R>(
        &self,
        view: &TmemView,
        operation: impl FnOnce(&mut TmemWriteSession<'_, '_>) -> R,
    ) -> Result<R, AddressSpaceError> {
        let allocation_columns = view.allocation.columns;
        let region = view.region;
        Ok(self
            .bytes
            .with_owner_private_write_session(&view.allocation.raw, |session| {
                let mut session = TmemWriteSession {
                    allocation_columns,
                    region,
                    session,
                };
                operation(&mut session)
            })?)
    }

    pub(crate) fn with_read_session<R>(
        &self,
        view: &TmemView,
        operation: impl FnOnce(&mut TmemReadSession<'_, '_>) -> R,
    ) -> Result<R, AddressSpaceError> {
        let allocation_columns = view.allocation.columns;
        let region = view.region;
        Ok(self
            .bytes
            .with_owner_private_read_session(&view.allocation.raw, |session| {
                let mut session = TmemReadSession {
                    allocation_columns,
                    region,
                    session,
                };
                operation(&mut session)
            })?)
    }

    pub(crate) fn validate_cell_rectangle(
        &self,
        view: &TmemView,
        lane_end: usize,
        column_end: usize,
    ) -> Result<(), AddressSpaceError> {
        validate_relative_region(view, TmemRegion::new(0, lane_end, 0, column_end))
    }

    pub const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    pub fn allocate_uninitialized(
        &self,
        owner: CtaId,
        lanes: usize,
        columns: usize,
    ) -> Result<TmemAllocation, AddressSpaceError> {
        validate_cta(self.topology, owner)?;
        let byte_len = tmem_byte_len(lanes, columns)?;
        let allocation = self.bytes.allocate_uninitialized(byte_len)?;
        Ok(TmemAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
            lanes,
            columns,
        })
    }

    pub fn allocate_zeroed(
        &self,
        owner: CtaId,
        lanes: usize,
        columns: usize,
    ) -> Result<TmemAllocation, AddressSpaceError> {
        validate_cta(self.topology, owner)?;
        let byte_len = tmem_byte_len(lanes, columns)?;
        let allocation = self.bytes.allocate_zeroed(byte_len)?;
        Ok(TmemAllocation {
            allocation,
            raw: self.bytes.full_view(allocation)?,
            owner,
            lanes,
            columns,
        })
    }

    pub fn full_view(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
    ) -> Result<TmemView, AddressSpaceError> {
        let lanes = allocation.lanes;
        let columns = allocation.columns;
        self.view(requester, allocation, TmemRegion::new(0, lanes, 0, columns))
    }

    pub fn view(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
        region: TmemRegion,
    ) -> Result<TmemView, AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Cta(allocation.owner),
            PhysicalOwner::Cta(requester),
        )?;
        validate_tmem_region(allocation, region)?;
        Ok(TmemView {
            allocation: allocation.clone(),
            region,
        })
    }

    pub fn subview(
        &self,
        parent: TmemView,
        relative: TmemRegion,
    ) -> Result<TmemView, AddressSpaceError> {
        let lane_offset = parent
            .region
            .lane_offset
            .checked_add(relative.lane_offset)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let column_offset = parent
            .region
            .column_offset
            .checked_add(relative.column_offset)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let region = TmemRegion::new(
            lane_offset,
            relative.lane_count,
            column_offset,
            relative.column_count,
        );
        validate_relative_region(&parent, relative)?;
        validate_tmem_region(&parent.allocation, region)?;
        Ok(TmemView {
            allocation: parent.allocation,
            region,
        })
    }

    pub fn cell_view(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
    ) -> Result<BufferView, AddressSpaceError> {
        self.cell_byte_view(view, lane, column, 0, TMEM_CELL_BYTES)
    }

    pub fn cell_byte_view(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<BufferView, AddressSpaceError> {
        let absolute = self.cell_byte_offset(view, lane, column, byte_offset, byte_len)?;
        Ok(self
            .bytes
            .subview(&view.allocation.raw, absolute, byte_len)?)
    }

    fn cell_byte_offset(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<usize, AddressSpaceError> {
        if byte_offset
            .checked_add(byte_len)
            .filter(|&end| end <= TMEM_CELL_BYTES)
            .is_none()
        {
            return Err(AddressSpaceError::TmemCellByteOutOfBounds {
                byte_offset,
                byte_len,
            });
        }
        if lane >= view.region.lane_count || column >= view.region.column_count {
            return Err(AddressSpaceError::TmemRegionOutOfBounds {
                allocation: view.allocation.allocation,
                lanes: view.region.lane_count,
                columns: view.region.column_count,
                lane_offset: lane,
                lane_count: 1,
                column_offset: column,
                column_count: 1,
            });
        }
        let physical_lane = view
            .region
            .lane_offset
            .checked_add(lane)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let physical_column = view
            .region
            .column_offset
            .checked_add(column)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let cell_index = physical_lane
            .checked_mul(view.allocation.columns)
            .and_then(|base| base.checked_add(physical_column))
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let absolute = cell_index
            .checked_mul(TMEM_CELL_BYTES)
            .and_then(|base| base.checked_add(byte_offset))
            .ok_or(AddressSpaceError::SizeOverflow)?;
        Ok(absolute)
    }

    fn lane_cell_byte_range(
        &self,
        view: &TmemView,
        lane: usize,
        first_column: usize,
        cell_count: usize,
    ) -> Result<(usize, usize), AddressSpaceError> {
        let column_end = first_column
            .checked_add(cell_count)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        if lane >= view.region.lane_count || column_end > view.region.column_count {
            let first_invalid_column = if first_column >= view.region.column_count {
                first_column
            } else if column_end > view.region.column_count {
                view.region.column_count
            } else {
                first_column
            };
            return Err(AddressSpaceError::TmemRegionOutOfBounds {
                allocation: view.allocation.allocation,
                lanes: view.region.lane_count,
                columns: view.region.column_count,
                lane_offset: lane,
                lane_count: 1,
                column_offset: first_invalid_column,
                column_count: 1,
            });
        }
        let physical_lane = view
            .region
            .lane_offset
            .checked_add(lane)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let physical_column = view
            .region
            .column_offset
            .checked_add(first_column)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let cell_index = physical_lane
            .checked_mul(view.allocation.columns)
            .and_then(|base| base.checked_add(physical_column))
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let absolute = cell_index
            .checked_mul(TMEM_CELL_BYTES)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        let byte_len = cell_count
            .checked_mul(TMEM_CELL_BYTES)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        Ok((absolute, byte_len))
    }

    fn allocation_cell_byte_offset(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<usize, AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Cta(allocation.owner),
            PhysicalOwner::Cta(requester),
        )?;
        if byte_offset
            .checked_add(byte_len)
            .filter(|&end| end <= TMEM_CELL_BYTES)
            .is_none()
        {
            return Err(AddressSpaceError::TmemCellByteOutOfBounds {
                byte_offset,
                byte_len,
            });
        }
        if lane >= allocation.lanes || column >= allocation.columns {
            return Err(AddressSpaceError::TmemRegionOutOfBounds {
                allocation: allocation.allocation,
                lanes: allocation.lanes,
                columns: allocation.columns,
                lane_offset: lane,
                lane_count: 1,
                column_offset: column,
                column_count: 1,
            });
        }
        lane.checked_mul(allocation.columns)
            .and_then(|base| base.checked_add(column))
            .and_then(|cell| cell.checked_mul(TMEM_CELL_BYTES))
            .and_then(|base| base.checked_add(byte_offset))
            .ok_or(AddressSpaceError::SizeOverflow)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_allocation_cell_bytes_into(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
        lane: usize,
        column: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        let absolute = self.allocation_cell_byte_offset(
            requester,
            allocation,
            lane,
            column,
            byte_offset,
            target.len(),
        )?;
        self.bytes
            .read_bytes_into(&allocation.raw, absolute, target)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_allocation_cell_bytes(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
        lane: usize,
        column: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), AddressSpaceError> {
        let absolute = self.allocation_cell_byte_offset(
            requester,
            allocation,
            lane,
            column,
            byte_offset,
            bytes.len(),
        )?;
        self.bytes.write_bytes(&allocation.raw, absolute, bytes)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_allocation_f32_rows(
        &self,
        requester: CtaId,
        allocation: &TmemAllocation,
        lanes: &[usize],
        first_column: usize,
        values: &[f32],
        columns: usize,
    ) -> Result<(), AddressSpaceError> {
        validate_cta(self.topology, requester)?;
        validate_owner(
            allocation.allocation,
            PhysicalOwner::Cta(allocation.owner),
            PhysicalOwner::Cta(requester),
        )?;
        let column_end = first_column
            .checked_add(columns)
            .ok_or(AddressSpaceError::SizeOverflow)?;
        if values.len() != lanes.len().saturating_mul(columns) {
            return Err(AddressSpaceError::SizeOverflow);
        }
        let mut row_byte_offsets = Vec::with_capacity(lanes.len());
        for &lane in lanes {
            if lane >= allocation.lanes || column_end > allocation.columns {
                return Err(AddressSpaceError::TmemRegionOutOfBounds {
                    allocation: allocation.allocation,
                    lanes: allocation.lanes,
                    columns: allocation.columns,
                    lane_offset: lane,
                    lane_count: 1,
                    column_offset: first_column,
                    column_count: columns,
                });
            }
            let offset = lane
                .checked_mul(allocation.columns)
                .and_then(|base| base.checked_add(first_column))
                .and_then(|cell| cell.checked_mul(TMEM_CELL_BYTES))
                .ok_or(AddressSpaceError::SizeOverflow)?;
            row_byte_offsets.push(offset);
        }
        self.bytes
            .write_f32_rows(&allocation.raw, &row_byte_offsets, values, columns)?;
        Ok(())
    }

    pub fn read_cell_u32(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
    ) -> Result<u32, AddressSpaceError> {
        let bytes: [u8; TMEM_CELL_BYTES] = self
            .read_cell_bytes(view, lane, column, 0, TMEM_CELL_BYTES)?
            .try_into()
            .expect("TMEM cell read has exactly four bytes");
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn write_cell_u32(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        value: u32,
    ) -> Result<(), AddressSpaceError> {
        self.write_cell_bytes(view, lane, column, 0, &value.to_le_bytes())
    }

    pub fn read_cell_bytes(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<u8>, AddressSpaceError> {
        let absolute = self.cell_byte_offset(view, lane, column, byte_offset, byte_len)?;
        Ok(self
            .bytes
            .read_bytes(&view.allocation.raw, absolute, byte_len)?)
    }

    pub fn read_cell_bytes_into(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        let absolute = self.cell_byte_offset(view, lane, column, byte_offset, target.len())?;
        self.bytes
            .read_bytes_into(&view.allocation.raw, absolute, target)?;
        Ok(())
    }

    pub(crate) fn snapshot_cell_preserving_validity(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
    ) -> Result<([u8; TMEM_CELL_BYTES], [bool; TMEM_CELL_BYTES]), AddressSpaceError> {
        let cell = self.cell_view(view, lane, column)?;
        let mut bytes = [0_u8; TMEM_CELL_BYTES];
        self.bytes
            .read_bytes_zero_filled_into(&cell, 0, &mut bytes)?;
        let validity: [bool; TMEM_CELL_BYTES] = self
            .bytes
            .byte_validity(&cell, 0, TMEM_CELL_BYTES)?
            .try_into()
            .map_err(|_| AddressSpaceError::SizeOverflow)?;
        Ok((bytes, validity))
    }

    /// Read consecutive physical columns from one TMEM lane.
    ///
    /// A failed validity check is reported as the same four-byte cell access
    /// that a scalar column loop would have issued.
    pub fn read_lane_cells_into(
        &self,
        view: &TmemView,
        lane: usize,
        first_column: usize,
        target: &mut [[u8; TMEM_CELL_BYTES]],
    ) -> Result<(), AddressSpaceError> {
        if target.is_empty() {
            return Ok(());
        }
        let (absolute, byte_len) =
            self.lane_cell_byte_range(view, lane, first_column, target.len())?;
        debug_assert_eq!(byte_len, target.as_flattened().len());
        match self
            .bytes
            .read_bytes_into(&view.allocation.raw, absolute, target.as_flattened_mut())
        {
            Ok(()) => Ok(()),
            Err(MemoryError::InvalidRead {
                allocation,
                first_invalid_byte,
                ..
            }) => {
                let relative = first_invalid_byte
                    .checked_sub(absolute)
                    .ok_or(AddressSpaceError::SizeOverflow)?;
                let cell_byte_offset = absolute
                    .checked_add((relative / TMEM_CELL_BYTES) * TMEM_CELL_BYTES)
                    .ok_or(AddressSpaceError::SizeOverflow)?;
                Err(MemoryError::InvalidRead {
                    allocation,
                    byte_offset: cell_byte_offset,
                    byte_len: TMEM_CELL_BYTES,
                    first_invalid_byte,
                }
                .into())
            }
            Err(error) => Err(error.into()),
        }
    }

    pub fn write_cell_bytes(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), AddressSpaceError> {
        let absolute = self.cell_byte_offset(view, lane, column, byte_offset, bytes.len())?;
        self.bytes
            .write_bytes(&view.allocation.raw, absolute, bytes)?;
        Ok(())
    }

    pub(crate) fn write_cell_snapshot(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
        bytes: &[u8; TMEM_CELL_BYTES],
        validity: &[bool; TMEM_CELL_BYTES],
    ) -> Result<(), AddressSpaceError> {
        let cell = self.cell_view(view, lane, column)?;
        self.bytes.write_bytes(&cell, 0, bytes)?;
        for (byte_offset, valid) in validity.iter().copied().enumerate() {
            if !valid {
                self.bytes.invalidate(&cell, byte_offset, 1)?;
            }
        }
        Ok(())
    }

    /// Write consecutive physical columns in one TMEM lane.
    pub fn write_lane_cells(
        &self,
        view: &TmemView,
        lane: usize,
        first_column: usize,
        cells: &[[u8; TMEM_CELL_BYTES]],
    ) -> Result<(), AddressSpaceError> {
        if cells.is_empty() {
            return Ok(());
        }
        let (absolute, byte_len) =
            self.lane_cell_byte_range(view, lane, first_column, cells.len())?;
        debug_assert_eq!(byte_len, cells.as_flattened().len());
        self.bytes
            .write_bytes(&view.allocation.raw, absolute, cells.as_flattened())?;
        Ok(())
    }

    pub fn invalidate_cell(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
    ) -> Result<(), AddressSpaceError> {
        let cell = self.cell_view(view, lane, column)?;
        self.bytes.invalidate(&cell, 0, TMEM_CELL_BYTES)?;
        Ok(())
    }

    pub fn cell_byte_validity(
        &self,
        view: &TmemView,
        lane: usize,
        column: usize,
    ) -> Result<Vec<bool>, AddressSpaceError> {
        let cell = self.cell_view(view, lane, column)?;
        Ok(self.bytes.byte_validity(&cell, 0, TMEM_CELL_BYTES)?)
    }
}

macro_rules! impl_byte_view_io {
    ($memory:ty, $view:ty) => {
        impl $memory {
            pub fn read_bytes(
                &self,
                view: &$view,
                byte_offset: usize,
                byte_len: usize,
            ) -> Result<Vec<u8>, AddressSpaceError> {
                Ok(self.bytes.read_bytes(&view.raw, byte_offset, byte_len)?)
            }

            pub fn read_bytes_into(
                &self,
                view: &$view,
                byte_offset: usize,
                target: &mut [u8],
            ) -> Result<(), AddressSpaceError> {
                self.bytes.read_bytes_into(&view.raw, byte_offset, target)?;
                Ok(())
            }

            pub fn read_bytes_zero_filled(
                &self,
                view: &$view,
                byte_offset: usize,
                byte_len: usize,
            ) -> Result<Vec<u8>, AddressSpaceError> {
                Ok(self
                    .bytes
                    .read_bytes_zero_filled(&view.raw, byte_offset, byte_len)?)
            }

            pub fn read_bytes_zero_filled_into(
                &self,
                view: &$view,
                byte_offset: usize,
                target: &mut [u8],
            ) -> Result<(), AddressSpaceError> {
                self.bytes
                    .read_bytes_zero_filled_into(&view.raw, byte_offset, target)?;
                Ok(())
            }

            pub fn write_bytes(
                &self,
                view: &$view,
                byte_offset: usize,
                bytes: &[u8],
            ) -> Result<(), AddressSpaceError> {
                self.bytes.write_bytes(&view.raw, byte_offset, bytes)?;
                Ok(())
            }

            pub fn invalidate(
                &self,
                view: &$view,
                byte_offset: usize,
                byte_len: usize,
            ) -> Result<(), AddressSpaceError> {
                self.bytes.invalidate(&view.raw, byte_offset, byte_len)?;
                Ok(())
            }

            pub fn read_f32_le(
                &self,
                view: &$view,
                element_index: usize,
            ) -> Result<f32, AddressSpaceError> {
                Ok(self.bytes.read_f32_le(&view.raw, element_index)?)
            }

            pub fn write_f32_le(
                &self,
                view: &$view,
                element_index: usize,
                value: f32,
            ) -> Result<(), AddressSpaceError> {
                self.bytes.write_f32_le(&view.raw, element_index, value)?;
                Ok(())
            }
        }
    };
}

impl_byte_view_io!(SharedMemory, SharedView);
impl_byte_view_io!(WarpPrivateMemory, WarpPrivateView);

/// All physical address spaces for one launch.
#[derive(Clone)]
pub struct PhysicalMemory {
    topology: LaunchTopology,
    global: GlobalMemory,
    shared: SharedMemory,
    local: LocalMemory,
    registers: RegisterMemory,
    tmem: TmemMemory,
    ordering: Arc<crate::OrderingHub>,
    engine_progress: SemanticProgress,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalUninitializedReadReview {
    space: &'static str,
    review: UninitializedReadReview,
}

impl PhysicalUninitializedReadReview {
    pub const fn space(self) -> &'static str {
        self.space
    }

    pub const fn review(self) -> UninitializedReadReview {
        self.review
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhysicalSemanticProgressSnapshot {
    shared: SemanticProgressSnapshot,
    engine: SemanticProgressSnapshot,
    local: u64,
    registers: u64,
}

impl PhysicalSemanticProgressSnapshot {
    pub(crate) fn changed_since(self, observed: Self, memory_only: bool) -> bool {
        self.shared != observed.shared
            || self.local != observed.local
            || self.registers != observed.registers
            || !memory_only && self.engine != observed.engine
    }
}

pub(crate) struct PhysicalSemanticProgressWatch {
    shared: SemanticProgressWatch,
    engine: Option<SemanticProgressWatch>,
}

impl Future for PhysicalSemanticProgressWatch {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if Pin::new(&mut self.shared).poll(context).is_ready() {
            return Poll::Ready(());
        }
        if self
            .engine
            .as_mut()
            .is_some_and(|watch| Pin::new(watch).poll(context).is_ready())
        {
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

impl PhysicalMemory {
    pub fn new(topology: LaunchTopology) -> Self {
        Self::with_global(topology, GlobalMemory::new())
    }

    /// Create launch-local physical spaces over an existing global memory.
    ///
    /// Sequential kernel launches use this constructor so GMEM allocations and
    /// writes survive across phases while SMEM, LOCAL, REG, and TMEM retain
    /// their per-launch lifetime and topology.
    pub fn with_global(topology: LaunchTopology, global: GlobalMemory) -> Self {
        let semantic_progress = global.semantic_progress();
        let read_policy = global.uninitialized_read_policy();
        Self {
            topology,
            global,
            shared: SharedMemory::with_semantic_progress(topology, semantic_progress.clone())
                .with_read_policy(read_policy),
            // Warp-private mutation prevents a finite local induction loop
            // from being classified as idle, but it cannot make another warp's
            // polling predicate ready. Keep it out of the shared wake channel.
            local: LocalMemory::new(topology).with_read_policy(read_policy),
            registers: RegisterMemory::new(topology).with_read_policy(read_policy),
            // Instruction-visible TMEM mutation publishes one engine-progress
            // event at the enclosing TCGEN instruction boundary. Its byte
            // arena must not duplicate that event for every cell.
            tmem: TmemMemory::new(topology).with_read_policy(read_policy),
            ordering: Arc::new(crate::OrderingHub::new(topology)),
            engine_progress: SemanticProgress::disabled(),
        }
    }

    pub(crate) fn convert_ptx_address(
        &self,
        context: &WarpContext,
        addresses: &WarpValue<u64>,
        space: crate::runtime::PtxStateSpace,
        to_generic: bool,
    ) -> Result<WarpValue<u64>, EngineError> {
        use crate::runtime::PtxStateSpace;
        let mask = context.active_mask();
        let mut bits = addresses.clone();
        for lane in mask {
            match space {
                PtxStateSpace::Local => {
                    return Err(EngineError::analysis_incomplete(
                        "unmodeled_ptx_address_conversion",
                    ));
                }
                PtxStateSpace::Global => {
                    if crate::instruction_codec::decode_generic_shared_address(bits[lane]).is_some()
                    {
                        return Err(EngineError::message(
                            "cvta.global requires a global address",
                        ));
                    }
                }
                PtxStateSpace::Generic => {
                    return Err(EngineError::message(
                        "cvta requires a concrete PTX state space",
                    ));
                }
                PtxStateSpace::Shared
                | PtxStateSpace::SharedCta
                | PtxStateSpace::SharedCluster => {
                    let state = if to_generic {
                        u32::try_from(bits[lane]).map_err(|_| {
                            EngineError::message("shared address exceeds uint32")
                        })?
                    } else {
                        crate::instruction_codec::decode_generic_shared_address(bits[lane])
                            .ok_or_else(|| {
                                EngineError::message(
                                    "cvta.to.shared requires a generic shared address",
                                )
                            })?
                    };
                    let rank = crate::instruction_codec::shared_address_cta_rank(state) as usize;
                    if rank >= context.topology().ctas_per_cluster()
                        || (space != PtxStateSpace::SharedCluster
                            && rank != context.cta_id_in_cluster())
                    {
                        return Err(EngineError::message(
                            "cvta shared address does not match its CTA window",
                        ));
                    }
                    bits[lane] = if to_generic {
                        crate::instruction_codec::generic_shared_address(state)
                    } else {
                        u64::from(state)
                    };
                }
            }
        }
        Ok(bits)
    }

    pub const fn topology(&self) -> LaunchTopology {
        self.topology
    }

    pub(crate) fn semantic_progress_snapshot(&self) -> PhysicalSemanticProgressSnapshot {
        PhysicalSemanticProgressSnapshot {
            shared: self.global.semantic_progress().snapshot(),
            engine: self.engine_progress.snapshot(),
            local: self.local.semantic_progress.aggregate_snapshot(),
            registers: self.registers.semantic_progress.aggregate_snapshot(),
        }
    }

    pub(crate) fn semantic_progress_snapshot_for_warp(
        &self,
        global_warp_id: usize,
    ) -> PhysicalSemanticProgressSnapshot {
        debug_assert!(global_warp_id < self.topology.warp_count());
        PhysicalSemanticProgressSnapshot {
            shared: self.global.semantic_progress().snapshot(),
            engine: self.engine_progress.snapshot(),
            local: self.local.semantic_progress.snapshot(global_warp_id),
            registers: self.registers.semantic_progress.snapshot(global_warp_id),
        }
    }

    pub(crate) fn enable_semantic_progress(&self) {
        self.global.enable_semantic_progress();
        self.local.bytes.enable_semantic_progress();
        self.registers.bytes.enable_semantic_progress();
        self.local.semantic_progress.enable();
        self.registers.semantic_progress.enable();
        self.engine_progress.enable();
    }

    pub(crate) fn watch_semantic_progress(
        &self,
        observed: PhysicalSemanticProgressSnapshot,
        memory_only: bool,
    ) -> PhysicalSemanticProgressWatch {
        PhysicalSemanticProgressWatch {
            shared: self.global.semantic_progress().watch(observed.shared),
            engine: (!memory_only).then(|| self.engine_progress.watch(observed.engine)),
        }
    }

    pub(crate) fn record_semantic_progress(&self) {
        self.engine_progress.record_change();
    }

    pub const fn global(&self) -> &GlobalMemory {
        &self.global
    }

    pub const fn shared(&self) -> &SharedMemory {
        &self.shared
    }

    pub const fn local(&self) -> &LocalMemory {
        &self.local
    }

    pub const fn registers(&self) -> &RegisterMemory {
        &self.registers
    }

    pub const fn tmem(&self) -> &TmemMemory {
        &self.tmem
    }

    pub(crate) fn ordering(&self) -> Arc<crate::OrderingHub> {
        self.ordering.clone()
    }

    pub fn take_uninitialized_read_reviews(&self) -> Vec<PhysicalUninitializedReadReview> {
        let mut reviews = Vec::new();
        let mut append = |space: &'static str, memory: &GlobalMemory| {
            reviews.extend(
                memory
                    .take_uninitialized_read_reviews()
                    .into_iter()
                    .map(|review| PhysicalUninitializedReadReview { space, review }),
            );
        };
        append("global", &self.global);
        append("shared", &self.shared.bytes);
        append("local", &self.local.bytes);
        append("register", &self.registers.bytes);
        append("tmem", &self.tmem.bytes);
        reviews.sort_unstable();
        reviews
    }
}

fn validate_cta(topology: LaunchTopology, cta: CtaId) -> Result<(), AddressSpaceError> {
    if cta.cluster_id < topology.clusters() && cta.cta_id_in_cluster < topology.ctas_per_cluster() {
        Ok(())
    } else {
        Err(AddressSpaceError::InvalidCta {
            cta,
            clusters: topology.clusters(),
            ctas_per_cluster: topology.ctas_per_cluster(),
        })
    }
}

#[inline]
fn validate_warp(topology: LaunchTopology, warp: WarpId) -> Result<(), AddressSpaceError> {
    validate_cta(topology, warp.cta)?;
    if warp.warp_id_in_cta < topology.warps_per_cta() {
        Ok(())
    } else {
        Err(AddressSpaceError::InvalidWarp {
            warp,
            warps_per_cta: topology.warps_per_cta(),
        })
    }
}

#[inline]
fn validate_owner(
    allocation: AllocationId,
    actual: PhysicalOwner,
    expected: PhysicalOwner,
) -> Result<(), AddressSpaceError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AddressSpaceError::OwnerMismatch {
            allocation,
            expected,
            actual,
        })
    }
}

#[inline]
fn resolve_view_access(
    allocation: AllocationId,
    allocation_byte_len: usize,
    view_offset: usize,
    view_len: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Result<usize, AddressSpaceError> {
    let view_end = view_offset
        .checked_add(view_len)
        .ok_or(MemoryError::OffsetOverflow)?;
    if view_end > allocation_byte_len {
        return Err(MemoryError::ViewOutOfBounds {
            allocation,
            allocation_byte_len,
            byte_offset: view_offset,
            byte_len: view_len,
        }
        .into());
    }
    let access_end = byte_offset
        .checked_add(byte_len)
        .ok_or(MemoryError::OffsetOverflow)?;
    if access_end > view_len {
        return Err(MemoryError::AccessOutOfBounds {
            allocation,
            view_byte_len: view_len,
            byte_offset,
            byte_len,
        }
        .into());
    }
    view_offset
        .checked_add(byte_offset)
        .ok_or_else(|| MemoryError::OffsetOverflow.into())
}

fn tmem_byte_len(lanes: usize, columns: usize) -> Result<usize, AddressSpaceError> {
    lanes
        .checked_mul(columns)
        .and_then(|cells| cells.checked_mul(TMEM_CELL_BYTES))
        .ok_or(AddressSpaceError::SizeOverflow)
}

fn validate_tmem_region(
    allocation: &TmemAllocation,
    region: TmemRegion,
) -> Result<(), AddressSpaceError> {
    let lane_end = region.lane_offset.checked_add(region.lane_count);
    let column_end = region.column_offset.checked_add(region.column_count);
    if lane_end.is_some_and(|end| end <= allocation.lanes)
        && column_end.is_some_and(|end| end <= allocation.columns)
    {
        Ok(())
    } else {
        Err(AddressSpaceError::TmemRegionOutOfBounds {
            allocation: allocation.allocation,
            lanes: allocation.lanes,
            columns: allocation.columns,
            lane_offset: region.lane_offset,
            lane_count: region.lane_count,
            column_offset: region.column_offset,
            column_count: region.column_count,
        })
    }
}

fn validate_relative_region(
    parent: &TmemView,
    relative: TmemRegion,
) -> Result<(), AddressSpaceError> {
    let lane_end = relative.lane_offset.checked_add(relative.lane_count);
    let column_end = relative.column_offset.checked_add(relative.column_count);
    if lane_end.is_some_and(|end| end <= parent.region.lane_count)
        && column_end.is_some_and(|end| end <= parent.region.column_count)
    {
        Ok(())
    } else {
        Err(AddressSpaceError::TmemRegionOutOfBounds {
            allocation: parent.allocation.allocation,
            lanes: parent.region.lane_count,
            columns: parent.region.column_count,
            lane_offset: relative.lane_offset,
            lane_count: relative.lane_count,
            column_offset: relative.column_offset,
            column_count: relative.column_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> LaunchTopology {
        LaunchTopology::new(2, 2, 2).unwrap()
    }

    fn cta(cluster: usize, rank: usize) -> CtaId {
        CtaId::new(topology(), cluster, rank).unwrap()
    }

    fn warp(cluster: usize, rank: usize, warp: usize) -> WarpId {
        WarpId::new(topology(), cta(cluster, rank), warp).unwrap()
    }

    #[test]
    fn overlapping_smem_views_share_physical_bytes() {
        let smem = SharedMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = smem.allocate_cta_uninitialized(owner, 32).unwrap();
        let wide = smem.cta_view(owner, &allocation, 4, 16).unwrap();
        let alias = smem.cta_view(owner, &allocation, 8, 8).unwrap();
        let nested = smem.subview(&wide, 4, 8).unwrap();

        smem.write_f32_le(&alias, 0, 7.25).unwrap();
        assert_eq!(smem.read_f32_le(&wide, 1).unwrap(), 7.25);
        assert_eq!(smem.read_f32_le(&nested, 0).unwrap(), 7.25);
    }

    #[test]
    fn smem_payload_can_publish_from_a_completion_thread() {
        // PhysicalMemory uses this constructor while native scheduling is
        // active, so this is the path exercised by deferred TMA completion.
        let smem = SharedMemory::with_semantic_progress(topology(), SemanticProgress::default());
        let owner = cta(0, 0);
        let allocation = smem.allocate_cta_zeroed(owner, 8).unwrap();
        let view = smem.full_cta_view(owner, &allocation).unwrap();
        smem.write_bytes(&view, 0, &[1, 2, 3, 4]).unwrap();

        let completion_smem = smem.clone();
        let completion_view = view.clone();
        std::thread::spawn(move || {
            completion_smem
                .write_bytes(&completion_view, 0, &[5, 6, 7, 8])
                .unwrap();
        })
        .join()
        .unwrap();

        assert_eq!(smem.read_bytes(&view, 0, 4).unwrap(), [5, 6, 7, 8]);
    }

    #[test]
    fn smem_owner_isolation_and_explicit_remote_access() {
        let smem = SharedMemory::new(topology());
        let owner = cta(0, 0);
        let peer = cta(0, 1);
        let other_cluster = cta(1, 0);
        let allocation = smem.allocate_cta_uninitialized(owner, 8).unwrap();
        let local = smem.full_cta_view(owner, &allocation).unwrap();
        smem.write_bytes(&local, 0, &[1, 2, 3, 4]).unwrap();

        assert!(matches!(
            smem.full_cta_view(peer, &allocation),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));
        let remote = smem.full_remote_cta_view(peer, owner, &allocation).unwrap();
        assert_eq!(smem.read_bytes(&remote, 0, 4).unwrap(), [1, 2, 3, 4]);
        assert!(matches!(
            smem.full_remote_cta_view(other_cluster, owner, &allocation),
            Err(AddressSpaceError::CrossClusterSharedAccess { .. })
        ));

        let isolated = smem.allocate_cta_zeroed(peer, 8).unwrap();
        let isolated = smem.full_cta_view(peer, &isolated).unwrap();
        assert_eq!(smem.read_bytes(&isolated, 0, 4).unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn direct_smem_access_preserves_owner_cluster_and_view_bounds() {
        let smem = SharedMemory::new(topology());
        let owner = cta(0, 0);
        let peer = cta(0, 1);
        let other_cluster = cta(1, 0);
        let allocation = smem.allocate_cta_uninitialized(owner, 16).unwrap();
        smem.write_cta_bytes(owner, &allocation, 4, 6, 1, &[7, 8])
            .unwrap();
        let mut local = [0_u8; 2];
        smem.read_cta_bytes_into(owner, &allocation, 4, 6, 1, &mut local)
            .unwrap();
        assert_eq!(local, [7, 8]);

        let mut remote = [0_u8; 2];
        smem.read_remote_cta_bytes_into(peer, owner, &allocation, 4, 6, 1, &mut remote)
            .unwrap();
        assert_eq!(remote, [7, 8]);
        assert!(matches!(
            smem.read_cta_bytes_into(peer, &allocation, 4, 6, 1, &mut local),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));
        assert!(matches!(
            smem.read_remote_cta_bytes_into(
                other_cluster,
                owner,
                &allocation,
                4,
                6,
                1,
                &mut remote,
            ),
            Err(AddressSpaceError::CrossClusterSharedAccess { .. })
        ));
        assert!(matches!(
            smem.read_cta_bytes_into(owner, &allocation, 4, 2, 1, &mut local),
            Err(AddressSpaceError::Memory(
                MemoryError::AccessOutOfBounds { .. }
            ))
        ));
    }

    #[test]
    fn warp_private_storage_isolates_warps_and_lanes() {
        let registers = RegisterMemory::new(topology());
        let owner = warp(0, 0, 0);
        let other_warp = warp(0, 0, 1);
        let allocation = registers.allocate_uninitialized(owner, 8).unwrap();
        let lane_three = registers.full_lane_view(owner, &allocation, 3).unwrap();
        let lane_four = registers.full_lane_view(owner, &allocation, 4).unwrap();

        registers
            .write_bytes(&lane_three, 0, &[10, 11, 12, 13])
            .unwrap();
        let alias = registers.subview(&lane_three, 1, 2).unwrap();
        assert_eq!(
            registers.read_bytes(&lane_three, 0, 4).unwrap(),
            [10, 11, 12, 13]
        );
        assert_eq!(registers.read_bytes(&alias, 0, 2).unwrap(), [11, 12]);
        registers.write_bytes(&alias, 0, &[21, 22]).unwrap();
        assert_eq!(
            registers.read_bytes(&lane_three, 0, 4).unwrap(),
            [10, 21, 22, 13]
        );
        assert!(matches!(
            registers.read_bytes(&lane_four, 0, 1),
            Err(AddressSpaceError::Memory(MemoryError::InvalidRead { .. }))
        ));
        assert!(matches!(
            registers.full_lane_view(other_warp, &allocation, 3),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));
    }

    #[test]
    fn direct_warp_private_access_preserves_owner_lane_and_view_bounds() {
        let registers = RegisterMemory::new(topology());
        let owner = warp(0, 0, 0);
        let other_warp = warp(0, 0, 1);
        let allocation = registers.allocate_uninitialized(owner, 12).unwrap();
        registers
            .write_lane_bytes(owner, &allocation, 3, 2, 6, 1, &[9, 10])
            .unwrap();
        let mut bytes = [0_u8; 2];
        registers
            .read_lane_bytes_into(owner, &allocation, 3, 2, 6, 1, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [9, 10]);
        assert!(matches!(
            registers.read_lane_bytes_into(other_warp, &allocation, 3, 2, 6, 1, &mut bytes,),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));
        assert!(matches!(
            registers.read_lane_bytes_into(owner, &allocation, WARP_SIZE, 2, 6, 1, &mut bytes,),
            Err(AddressSpaceError::InvalidLane { lane: WARP_SIZE })
        ));
        assert!(matches!(
            registers.read_lane_bytes_into(owner, &allocation, 3, 2, 2, 1, &mut bytes),
            Err(AddressSpaceError::Memory(
                MemoryError::AccessOutOfBounds { .. }
            ))
        ));
    }

    #[cfg(feature = "analysis-core")]
    #[test]
    fn warp_private_semantic_progress_is_scoped_to_the_writing_warp() {
        let topology = topology();
        let physical = PhysicalMemory::new(topology);
        physical.enable_semantic_progress();
        let first = warp(0, 0, 0);
        let second = warp(0, 0, 1);
        let first_id = first.global_warp_id(topology).unwrap();
        let second_id = second.global_warp_id(topology).unwrap();
        let first_allocation = physical.registers().allocate_zeroed(first, 4).unwrap();
        let second_allocation = physical.registers().allocate_zeroed(second, 4).unwrap();
        let initial_first = physical.semantic_progress_snapshot_for_warp(first_id);
        let initial_second = physical.semantic_progress_snapshot_for_warp(second_id);

        physical
            .registers()
            .write_lane_bytes(first, &first_allocation, 0, 0, 4, 0, &[1])
            .unwrap();
        let changed_first = physical.semantic_progress_snapshot_for_warp(first_id);
        assert_ne!(changed_first, initial_first);
        assert_eq!(
            physical.semantic_progress_snapshot_for_warp(second_id),
            initial_second,
            "another warp's private write is not progress for this warp"
        );

        physical
            .registers()
            .write_lane_bytes(first, &first_allocation, 0, 0, 4, 0, &[1])
            .unwrap();
        assert_eq!(
            physical.semantic_progress_snapshot_for_warp(first_id),
            changed_first,
            "an identical write is not semantic progress"
        );

        physical
            .registers()
            .write_lane_bytes(second, &second_allocation, 0, 0, 4, 0, &[2])
            .unwrap();
        assert_eq!(
            physical.semantic_progress_snapshot_for_warp(first_id),
            changed_first,
            "private progress remains isolated after the peer advances"
        );
        assert_ne!(
            physical.semantic_progress_snapshot_for_warp(second_id),
            initial_second
        );
    }

    #[test]
    fn tmem_views_alias_cells_and_isolate_ctas() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_uninitialized(owner, 8, 16).unwrap();
        let first = tmem
            .view(owner, &allocation, TmemRegion::new(2, 4, 3, 8))
            .unwrap();
        let alias = tmem
            .view(owner, &allocation, TmemRegion::new(3, 2, 5, 4))
            .unwrap();

        tmem.write_cell_u32(&first, 1, 2, 0x1234_5678).unwrap();
        assert_eq!(tmem.read_cell_u32(&alias, 0, 0).unwrap(), 0x1234_5678);
        tmem.write_cell_bytes(&alias, 0, 0, 1, &[0xaa, 0xbb])
            .unwrap();
        assert_eq!(
            tmem.read_cell_bytes(&first, 1, 2, 0, 4).unwrap(),
            [0x78, 0xaa, 0xbb, 0x12]
        );
        assert!(matches!(
            tmem.full_view(cta(0, 1), &allocation),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));

        let other = tmem
            .allocate_zeroed(cta(0, 1), allocation.lanes(), allocation.columns())
            .unwrap();
        let other = tmem.full_view(cta(0, 1), &other).unwrap();
        assert_eq!(tmem.read_cell_u32(&other, 3, 5).unwrap(), 0);
    }

    #[test]
    fn tmem_lane_cell_rows_preserve_physical_layout_aliasing() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_zeroed(owner, 4, 10).unwrap();
        let full = tmem.full_view(owner, &allocation).unwrap();
        let view = tmem
            .view(owner, &allocation, TmemRegion::new(1, 2, 2, 5))
            .unwrap();
        let alias = tmem
            .view(owner, &allocation, TmemRegion::new(2, 1, 3, 3))
            .unwrap();
        let cells = [
            0x1122_3344_u32.to_le_bytes(),
            0x5566_7788_u32.to_le_bytes(),
            0x99aa_bbcc_u32.to_le_bytes(),
        ];

        tmem.write_lane_cells(&view, 1, 1, &cells).unwrap();

        let mut aliased = [[0_u8; TMEM_CELL_BYTES]; 3];
        tmem.read_lane_cells_into(&alias, 0, 0, &mut aliased)
            .unwrap();
        assert_eq!(aliased, cells);
        for (column, expected) in cells.iter().enumerate() {
            assert_eq!(
                tmem.read_cell_u32(&full, 2, 3 + column).unwrap(),
                u32::from_le_bytes(*expected)
            );
        }

        // The row uses the allocation's ten-column lane stride, not the
        // five-column width of `view`.
        assert_eq!(tmem.read_cell_u32(&full, 1, 3).unwrap(), 0);
        assert_eq!(tmem.read_cell_u32(&full, 2, 2).unwrap(), 0);
        assert_eq!(tmem.read_cell_u32(&full, 2, 6).unwrap(), 0);
    }

    #[test]
    fn tmem_lane_cell_rows_preserve_empty_and_out_of_bounds_semantics() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_zeroed(owner, 2, 4).unwrap();
        let full = tmem.full_view(owner, &allocation).unwrap();
        let view = tmem
            .view(owner, &allocation, TmemRegion::new(0, 2, 1, 2))
            .unwrap();
        let mut empty = [[0_u8; TMEM_CELL_BYTES]; 0];

        tmem.read_lane_cells_into(&view, usize::MAX, usize::MAX, &mut empty)
            .unwrap();
        tmem.write_lane_cells(&view, usize::MAX, usize::MAX, &empty)
            .unwrap();

        let two_cells = [[1_u8; TMEM_CELL_BYTES]; 2];
        assert!(matches!(
            tmem.write_lane_cells(&view, 0, 1, &two_cells),
            Err(AddressSpaceError::TmemRegionOutOfBounds {
                lane_offset: 0,
                lane_count: 1,
                column_offset: 2,
                column_count: 1,
                ..
            })
        ));
        assert_eq!(tmem.read_cell_u32(&full, 0, 2).unwrap(), 0);
        assert_eq!(tmem.read_cell_u32(&full, 0, 3).unwrap(), 0);

        let mut one_cell = [[0_u8; TMEM_CELL_BYTES]; 1];
        assert!(matches!(
            tmem.read_lane_cells_into(&view, 2, 0, &mut one_cell),
            Err(AddressSpaceError::TmemRegionOutOfBounds {
                lane_offset: 2,
                column_offset: 0,
                ..
            })
        ));
        assert!(matches!(
            tmem.read_lane_cells_into(&view, 0, 2, &mut one_cell),
            Err(AddressSpaceError::TmemRegionOutOfBounds {
                lane_offset: 0,
                column_offset: 2,
                ..
            })
        ));
    }

    #[test]
    fn tmem_lane_cell_read_preserves_scalar_invalid_read_diagnostics() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_uninitialized(owner, 2, 8).unwrap();
        let view = tmem
            .view(owner, &allocation, TmemRegion::new(1, 1, 2, 4))
            .unwrap();
        let initialized = [
            0x0102_0304_u32.to_le_bytes(),
            0x1112_1314_u32.to_le_bytes(),
            0x2122_2324_u32.to_le_bytes(),
        ];
        tmem.write_lane_cells(&view, 0, 0, &initialized).unwrap();
        tmem.invalidate_cell(&view, 0, 1).unwrap();
        tmem.write_cell_bytes(&view, 0, 1, 0, &[0xaa, 0xbb])
            .unwrap();

        let mut row = [[0_u8; TMEM_CELL_BYTES]; 3];
        let row_error = tmem
            .read_lane_cells_into(&view, 0, 0, &mut row)
            .unwrap_err();
        let mut scalar = [0_u8; TMEM_CELL_BYTES];
        let scalar_error = tmem
            .read_cell_bytes_into(&view, 0, 1, 0, &mut scalar)
            .unwrap_err();
        assert_eq!(row_error, scalar_error);
        assert!(matches!(
            row_error,
            AddressSpaceError::Memory(MemoryError::InvalidRead {
                allocation: found_allocation,
                byte_offset: 44,
                byte_len: TMEM_CELL_BYTES,
                first_invalid_byte: 46,
            }) if found_allocation == allocation.allocation()
        ));

        let fresh = tmem.allocate_uninitialized(owner, 1, 2).unwrap();
        let fresh = tmem.full_view(owner, &fresh).unwrap();
        let mut fresh_row = [[0_u8; TMEM_CELL_BYTES]; 2];
        let row_error = tmem
            .read_lane_cells_into(&fresh, 0, 0, &mut fresh_row)
            .unwrap_err();
        let mut scalar = [0_u8; TMEM_CELL_BYTES];
        let scalar_error = tmem
            .read_cell_bytes_into(&fresh, 0, 0, 0, &mut scalar)
            .unwrap_err();
        assert_eq!(row_error, scalar_error);
    }

    #[test]
    fn direct_tmem_access_preserves_owner_cell_and_byte_bounds() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_uninitialized(owner, 2, 4).unwrap();
        tmem.write_allocation_cell_bytes(owner, &allocation, 1, 2, 1, &[0xaa, 0xbb])
            .unwrap();
        let mut bytes = [0_u8; 2];
        tmem.read_allocation_cell_bytes_into(owner, &allocation, 1, 2, 1, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [0xaa, 0xbb]);
        assert!(matches!(
            tmem.read_allocation_cell_bytes_into(cta(0, 1), &allocation, 1, 2, 1, &mut bytes,),
            Err(AddressSpaceError::OwnerMismatch { .. })
        ));
        assert!(matches!(
            tmem.read_allocation_cell_bytes_into(owner, &allocation, 2, 2, 1, &mut bytes),
            Err(AddressSpaceError::TmemRegionOutOfBounds { .. })
        ));
        assert!(matches!(
            tmem.read_allocation_cell_bytes_into(owner, &allocation, 1, 2, 3, &mut bytes),
            Err(AddressSpaceError::TmemCellByteOutOfBounds {
                byte_offset: 3,
                byte_len: 2,
            })
        ));
    }

    #[test]
    fn uninitialized_tmem_cells_fail_before_partial_initialization_completes() {
        let tmem = TmemMemory::new(topology());
        let owner = cta(0, 0);
        let allocation = tmem.allocate_uninitialized(owner, 2, 2).unwrap();
        let view = tmem.full_view(owner, &allocation).unwrap();

        assert!(matches!(
            tmem.read_cell_u32(&view, 0, 0),
            Err(AddressSpaceError::Memory(MemoryError::InvalidRead { .. }))
        ));
        tmem.write_cell_bytes(&view, 0, 0, 0, &[1, 2]).unwrap();
        assert!(matches!(
            tmem.read_cell_u32(&view, 0, 0),
            Err(AddressSpaceError::Memory(MemoryError::InvalidRead {
                first_invalid_byte: 2,
                ..
            }))
        ));
        tmem.write_cell_bytes(&view, 0, 0, 2, &[3, 4]).unwrap();
        assert_eq!(tmem.read_cell_u32(&view, 0, 0).unwrap(), 0x0403_0201);
    }

    #[test]
    fn sequential_launches_can_share_global_memory_across_topologies() {
        let global = GlobalMemory::new();
        let allocation = global.allocate_from_bytes(vec![0; 4]).unwrap();
        let view = global.view(allocation, 0, 4).unwrap();
        let first = PhysicalMemory::with_global(topology(), global.clone());
        let second_topology = LaunchTopology::new(1, 1, 1).unwrap();
        let second = PhysicalMemory::with_global(second_topology, global);

        first.global().write_f32_le(&view, 0, 7.25).unwrap();

        assert_eq!(second.global().read_f32_le(&view, 0).unwrap(), 7.25);
        assert_eq!(first.topology(), topology());
        assert_eq!(second.topology(), second_topology);
    }

    #[test]
    fn physical_spaces_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PhysicalMemory>();
        assert_send_sync::<SharedMemory>();
        assert_send_sync::<RegisterMemory>();
        assert_send_sync::<TmemMemory>();
    }
}
