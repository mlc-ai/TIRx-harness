//! Engine implementation of typed values transported across the v2 ABI.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, IndexMut, Not, Sub,
};
use std::sync::Arc;

use crate::abi::v2::EngineError;
use crate::engine_mode::EngineMode;
use crate::runtime::operand::PhysicalPtrSlot;
use crate::runtime::{PhysicalPtr, RuntimeBuffer, RuntimeTensorMap};
use crate::{WarpContext, WarpEngine, WarpMask, WarpValue, WARP_SIZE};

/// Stable identity of one source instruction or native-control occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SiteId(u64);

impl SiteId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Active lanes carried across native control flow.
///
/// This intentionally exposes only the set algebra required to express
/// frontend-native control flow.  Barrier, collective, and scheduling
/// semantics remain operations on the engine.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneMask(WarpMask);

impl LaneMask {
    pub const EMPTY: Self = Self(WarpMask::EMPTY);
    pub const FULL: Self = Self(WarpMask::FULL);

    pub const fn from_bits(bits: u32) -> Self {
        Self(WarpMask::from_bits(bits))
    }

    pub fn from_predicate(predicate: impl FnMut(usize) -> bool) -> Self {
        Self(WarpMask::from_predicate(predicate))
    }

    pub fn single(lane: usize) -> Result<Self, EngineError> {
        if lane >= WARP_SIZE {
            return Err(EngineError::message(format!(
                "lane {lane} is outside a {WARP_SIZE}-lane warp"
            )));
        }
        Ok(Self::from_bits(1_u32 << lane))
    }

    /// Build a mask from frontend-selected lane ids.
    pub fn from_lanes(lanes: impl IntoIterator<Item = usize>) -> Result<Self, EngineError> {
        let mut bits = 0_u32;
        for lane in lanes {
            if lane >= WARP_SIZE {
                return Err(EngineError::message(format!(
                    "lane {lane} is outside a {WARP_SIZE}-lane warp"
                )));
            }
            bits |= 1_u32 << lane;
        }
        Ok(Self::from_bits(bits))
    }

    pub const fn bits(self) -> u32 {
        self.0.bits()
    }

    pub const fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub const fn is_full(self) -> bool {
        self.0.is_full()
    }

    pub const fn len(self) -> usize {
        self.0.len()
    }

    pub const fn contains(self, lane: usize) -> bool {
        self.0.contains(lane)
    }

    pub const fn first_active(self) -> Option<usize> {
        self.0.first_active()
    }

    pub const fn iter(self) -> LaneIds {
        LaneIds {
            remaining: self.bits(),
        }
    }

    pub(crate) const fn into_inner(self) -> WarpMask {
        self.0
    }

    pub(crate) const fn from_inner(mask: WarpMask) -> Self {
        Self(mask)
    }
}

impl fmt::Debug for LaneMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LaneMask({:#034b})", self.bits())
    }
}

impl BitAnd for LaneMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for LaneMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for LaneMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for LaneMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for LaneMask {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for LaneMask {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for LaneMask {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Sub for LaneMask {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl IntoIterator for LaneMask {
    type Item = usize;
    type IntoIter = LaneIds;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Ordered lane iterator returned by [`LaneMask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneIds {
    remaining: u32,
}

impl Iterator for LaneIds {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let lane = self.remaining.trailing_zeros() as usize;
        self.remaining &= self.remaining - 1;
        Some(lane)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining.count_ones() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for LaneIds {}

/// One value per simulated lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R<T>(WarpValue<T>);

impl<T> R<T> {
    pub const fn from_lanes(lanes: [T; WARP_SIZE]) -> Self {
        Self(WarpValue::from_lanes(lanes))
    }

    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Self(WarpValue::from_fn(f))
    }

    /// Fast construction used by generated code for copyable lane values.
    pub fn from_fn_copy(f: impl FnMut(usize) -> T) -> Self
    where
        T: Copy,
    {
        Self(WarpValue::from_fn_copy(f))
    }

    pub const fn lanes(&self) -> &[T; WARP_SIZE] {
        self.0.lanes()
    }

    pub fn map<U>(self, f: impl FnMut(usize, T) -> U) -> R<U> {
        R(self.0.map(f))
    }

    pub fn zip_map<U, V>(&self, other: &R<U>, f: impl FnMut(usize, &T, &U) -> V) -> R<V> {
        R(self.0.zip_map(&other.0, f))
    }

    pub fn to_mask(&self, predicate: impl FnMut(usize, &T) -> bool) -> LaneMask {
        LaneMask::from_inner(self.0.to_mask(predicate))
    }

    pub(crate) fn into_inner(self) -> WarpValue<T> {
        self.0
    }

    pub(crate) const fn inner(&self) -> &WarpValue<T> {
        &self.0
    }

    pub(crate) fn from_inner(value: WarpValue<T>) -> Self {
        Self(value)
    }
}

impl<T: Clone> R<T> {
    pub fn splat(value: T) -> Self {
        Self(WarpValue::splat(value))
    }

    pub fn masked_assign(&mut self, mask: LaneMask, source: &Self) {
        self.0.masked_assign(mask.into_inner(), &source.0);
    }

    pub fn masked_fill(&mut self, mask: LaneMask, value: T) {
        self.0.masked_fill(mask.into_inner(), value);
    }
}

impl<T> Index<usize> for R<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for R<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Opaque execution context.  Only its active lanes cross the public boundary;
/// elect-sync provenance remains engine-owned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecCtx(WarpContext);

impl ExecCtx {
    pub const fn active_mask(self) -> LaneMask {
        LaneMask::from_inner(self.0.active_mask())
    }

    /// Derive a context for a structured child region.
    pub fn with_active_mask(self, active: LaneMask) -> Self {
        Self(self.0.with_active_mask(active.into_inner()))
    }

    /// Update active lanes while preserving launch coordinates and control
    /// provenance.
    pub fn set_active_mask(&mut self, active: LaneMask) {
        self.0.set_active_mask(active.into_inner());
    }

    pub(crate) const fn from_inner(context: WarpContext) -> Self {
        Self(context)
    }

    pub(crate) const fn into_inner(self) -> WarpContext {
        self.0
    }
}

mod space_sealed {
    pub trait Sealed {}
}

/// Closed set of PTX-visible storage spaces.
#[allow(private_bounds)]
pub trait MemorySpace: space_sealed::Sealed + Copy + Send + Sync + 'static {}

macro_rules! memory_spaces {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
            pub struct $name;

            impl space_sealed::Sealed for $name {}
            impl MemorySpace for $name {}
        )+
    };
}

memory_spaces!(
    Generic,
    Global,
    Shared,
    SharedCta,
    SharedCluster,
    Local,
    Register,
    Tmem
);

/// Addressable PTX spaces; register and tensor-memory operands are separate.
pub(crate) trait PtxAddressSpace: MemorySpace {
    const PTX_SPACE: crate::runtime::PtxStateSpace;
}

macro_rules! ptx_address_spaces {
    ($($space:ident),+ $(,)?) => {
        $(impl PtxAddressSpace for $space {
            const PTX_SPACE: crate::runtime::PtxStateSpace =
                crate::runtime::PtxStateSpace::$space;
        })+
    };
}
ptx_address_spaces!(
    Generic,
    Global,
    Shared,
    SharedCta,
    SharedCluster,
    Local
);

/// Opaque, lane-wise address in one statically selected PTX state space.
#[derive(Clone)]
pub struct Address<S: MemorySpace> {
    inner: PhysicalPtr,
    logical_buffer: Option<Arc<str>>,
    marker: PhantomData<fn() -> S>,
}

/// Borrowed address of a statically bound TIR buffer.
///
/// This is the direct operand form used by generated scalar `ld`/`st` calls.
/// It avoids turning a frontend buffer binding into an owned raw pointer for
/// every dynamic instruction while the engine still owns bounds, state-space,
/// and access validation.
#[doc(hidden)]
#[derive(Clone)]
pub struct DirectAddress<'a, S: MemorySpace> {
    pub(crate) buffer: &'a RuntimeBuffer,
    pub(crate) indices: &'a WarpValue<i64>,
    pub(crate) itemsize: usize,
    pub(crate) logical_buffer: &'static str,
    marker: PhantomData<fn() -> S>,
}

impl<'a, S: MemorySpace> DirectAddress<'a, S> {
    pub(crate) fn new(
        buffer: &'a RuntimeBuffer,
        indices: &'a WarpValue<i64>,
        itemsize: usize,
        logical_buffer: &'static str,
    ) -> Self {
        Self {
            buffer,
            indices,
            itemsize,
            logical_buffer,
            marker: PhantomData,
        }
    }

    pub(crate) fn physical_pointer(&self) -> PhysicalPtr {
        PhysicalPtr::new(
            (*self.buffer).clone(),
            (*self.indices).clone(),
            self.itemsize,
        )
        .bounded_to_initial_view()
    }
}

impl<S: MemorySpace> Address<S> {
    pub(crate) fn from_inner(inner: PhysicalPtr) -> Self {
        Self {
            inner,
            logical_buffer: None,
            marker: PhantomData,
        }
    }

    pub(crate) fn from_logical_buffer(
        inner: PhysicalPtr,
        logical_buffer: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            inner,
            logical_buffer: Some(logical_buffer.into()),
            marker: PhantomData,
        }
    }

    pub(crate) const fn inner(&self) -> &PhysicalPtr {
        &self.inner
    }

    pub(crate) fn into_inner(self) -> PhysicalPtr {
        self.inner
    }

    pub(crate) fn into_parts(self) -> (PhysicalPtr, Option<Arc<str>>) {
        (self.inner, self.logical_buffer)
    }

    pub(crate) fn cast_space<T: MemorySpace>(self) -> Address<T> {
        Address {
            inner: self.inner,
            logical_buffer: self.logical_buffer,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_inner(&self, inner: PhysicalPtr) -> Self {
        Self {
            inner,
            logical_buffer: self.logical_buffer.clone(),
            marker: PhantomData,
        }
    }

    /// Add one dynamic byte offset per lane while retaining allocation,
    /// bounds, access rights, and alias identity inside the engine.
    pub fn byte_offset(
        &self,
        offsets: &R<i64>,
        pointee_itemsize: usize,
        active: LaneMask,
    ) -> Result<Self, EngineError> {
        self.inner()
            .with_byte_offset(offsets.inner(), pointee_itemsize, active.into_inner())
            .map(|inner| self.with_inner(inner))
            .map_err(Into::into)
    }

    /// Add a lane-wise element offset using the pointee width supplied by the
    /// consuming instruction. Resolved integer addresses intentionally carry
    /// no source-language pointee type, so the ABI boundary owns this scale.
    pub fn element_offset(
        &self,
        offsets: &R<i64>,
        pointee_itemsize: usize,
        active: LaneMask,
    ) -> Result<Self, EngineError> {
        let element_bytes = i64::try_from(pointee_itemsize)
            .map_err(|_| EngineError::message("address itemsize exceeds i64"))?;
        let mut byte_offsets = R::splat(0_i64);
        for lane in active {
            byte_offsets[lane] = offsets[lane].checked_mul(element_bytes).ok_or_else(|| {
                EngineError::message(format!("address element offset overflows on lane {lane}"))
            })?;
        }
        self.byte_offset(&byte_offsets, pointee_itemsize, active)
    }

    /// Restrict one dynamic address to the range and access mode carried by a
    /// frontend access-view expression.
    pub fn view<V: AddressViewAccess>(
        &self,
        element_offsets: &R<i64>,
        element_extents: &R<i64>,
        pointee_itemsize: usize,
        active: LaneMask,
        site: SiteId,
    ) -> Result<Self, EngineError> {
        self.inner()
            .with_element_offset_extent_labeled(
                element_offsets.inner(),
                element_extents.inner(),
                pointee_itemsize,
                active.into_inner(),
                V::ACCESS_MASK,
                &crate::DiagnosticLabel::new(format!("address view at site {}", site.get())),
            )
            .map(|inner| self.with_inner(inner))
            .map_err(Into::into)
    }

    /// Merge lane-selected addresses at a native-control join.
    pub fn masked_assign(
        &mut self,
        selected: LaneMask,
        incoming: &Self,
    ) -> Result<(), EngineError> {
        let mut slot = PhysicalPtrSlot::new();
        slot.store(self.inner(), LaneMask::FULL.into_inner())?;
        slot.store(incoming.inner(), selected.into_inner())?;
        self.inner = slot.load(LaneMask::FULL.into_inner())?;
        self.logical_buffer = (self.logical_buffer == incoming.logical_buffer)
            .then_some(self.logical_buffer.clone())
            .flatten();
        Ok(())
    }
}

mod address_view_sealed {
    pub trait Sealed {}
}

/// Closed compile-time access mode for [`Address::view`].
#[allow(private_bounds)]
pub trait AddressViewAccess: address_view_sealed::Sealed {
    #[doc(hidden)]
    const ACCESS_MASK: u8;
}

macro_rules! address_view_access {
    ($name:ident, $mask:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl address_view_sealed::Sealed for $name {}
        impl AddressViewAccess for $name {
            const ACCESS_MASK: u8 = $mask;
        }
    };
}

address_view_access!(ReadOnly, 1);
address_view_access!(WriteOnly, 2);
address_view_access!(ReadWrite, 3);

impl Address<Shared> {
    /// Map each active lane's shared address to another CTA rank in its
    /// cluster while preserving the physical allocation identity.
    pub fn map_rank(
        &self,
        context: ExecCtx,
        ranks: &R<i64>,
        active: LaneMask,
    ) -> Result<Self, EngineError> {
        self.inner()
            .map_shared_rank(&context.into_inner(), ranks.inner(), active.into_inner())
            .map(|inner| self.with_inner(inner))
            .map_err(Into::into)
    }
}

/// Opaque allocation/view handle in one statically selected memory space.
#[derive(Clone)]
pub struct BufferHandle<S: MemorySpace> {
    inner: RuntimeBuffer,
    logical_buffer: Option<Arc<str>>,
    marker: PhantomData<fn() -> S>,
}

/// Runtime allocation domain used to resolve an encoded hardware descriptor.
///
/// Descriptor bits encode an address and layout, but not the host allocation
/// identity.  The frontend supplies the bound allocation domain; descriptor
/// decoding and all range/access validation remain engine-owned.
#[derive(Clone)]
pub struct DescriptorDomain<S: MemorySpace> {
    buffers: Arc<[RuntimeBuffer]>,
    marker: PhantomData<fn() -> S>,
}

impl<S: MemorySpace> DescriptorDomain<S> {
    pub(crate) fn from_buffers(buffers: impl Into<Arc<[RuntimeBuffer]>>) -> Self {
        Self {
            buffers: buffers.into(),
            marker: PhantomData,
        }
    }

    pub(crate) fn inner(&self) -> &[RuntimeBuffer] {
        &self.buffers
    }
}

impl<S: MemorySpace> BufferHandle<S> {
    pub(crate) fn from_inner(inner: RuntimeBuffer) -> Self {
        Self {
            inner,
            logical_buffer: None,
            marker: PhantomData,
        }
    }

    pub(crate) fn from_logical_buffer(
        inner: RuntimeBuffer,
        logical_buffer: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            inner,
            logical_buffer: Some(logical_buffer.into()),
            marker: PhantomData,
        }
    }

    pub(crate) const fn inner(&self) -> &RuntimeBuffer {
        &self.inner
    }

    pub(crate) fn logical_buffer(&self) -> Option<&str> {
        self.logical_buffer.as_deref()
    }

    /// Build a lane-wise address inside this bound allocation. The frontend
    /// supplies only element indices; allocation identity, bounds, and state
    /// space remain attached by the engine.
    pub fn address(
        &self,
        element_indices: R<i64>,
        itemsize: usize,
    ) -> Result<Address<S>, EngineError> {
        if itemsize == 0 {
            return Err(EngineError::message(
                "buffer address itemsize must be positive",
            ));
        }
        let pointer = PhysicalPtr::new(self.inner.clone(), element_indices.into_inner(), itemsize)
            .bounded_to_initial_view();
        Ok(match &self.logical_buffer {
            Some(logical_buffer) => Address::from_logical_buffer(pointer, logical_buffer.clone()),
            None => Address::from_inner(pointer),
        })
    }
}

/// Opaque identity and validated metadata of one runtime tensor map.
#[derive(Clone)]
pub struct TensorMapHandle(Arc<RuntimeTensorMap>);

impl TensorMapHandle {
    pub(crate) const fn from_inner(inner: Arc<RuntimeTensorMap>) -> Self {
        Self(inner)
    }

    pub(crate) const fn inner(&self) -> &Arc<RuntimeTensorMap> {
        &self.0
    }
}

/// Logical tile coordinate supplied to a frontend-defined pure element map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalCoord<'a>(&'a [i64]);

impl<'a> LogicalCoord<'a> {
    pub const fn new(dimensions: &'a [i64]) -> Self {
        Self(dimensions)
    }

    pub const fn dimensions(self) -> &'a [i64] {
        self.0
    }
}

/// Validated lane identity passed into a frontend element map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(u8);

impl LaneId {
    pub(crate) fn from_index(lane: usize) -> Self {
        debug_assert!(lane < WARP_SIZE);
        Self(lane as u8)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ElementLocation {
    ByteOffset(i128),
    BitOffset {
        byte_offset: i128,
        bit_offset: u8,
    },
    Tmem {
        mapped_lane: i64,
        tcol_element: i64,
        allocated_addr: i64,
        bit_offset: u8,
    },
}

/// Allocation-relative result of a pure frontend layout mapping.
///
/// Ordinary state spaces use a byte offset. TMEM instead uses the PTX-visible
/// `(TLane, TCol, allocated_addr)` coordinate triple: flattening that triple in
/// a frontend would duplicate the engine's allocation/lifecycle rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementRef<S: MemorySpace> {
    location: ElementLocation,
    target_rank: Option<u32>,
    owned: bool,
    in_bounds: bool,
    marker: PhantomData<fn() -> S>,
}

impl<S: MemorySpace> ElementRef<S> {
    pub const fn out_of_bounds(target_rank: Option<u32>) -> Self {
        Self {
            location: ElementLocation::ByteOffset(0),
            target_rank,
            owned: true,
            in_bounds: false,
            marker: PhantomData,
        }
    }

    /// Mark a logical element as owned by another execution lane.
    ///
    /// Ownership and bounds are distinct facts: an out-of-bounds element is
    /// still assigned to this lane and must follow the instruction's fill or
    /// error policy, while an unowned element must be ignored by this lane.
    pub const fn unowned(target_rank: Option<u32>) -> Self {
        Self {
            location: ElementLocation::ByteOffset(0),
            target_rank,
            owned: false,
            in_bounds: false,
            marker: PhantomData,
        }
    }

    /// Return an ordinary allocation-relative byte offset, or `None` for a
    /// TMEM coordinate.
    pub const fn byte_offset(self) -> Option<i128> {
        match self.location {
            ElementLocation::ByteOffset(offset) => Some(offset),
            ElementLocation::BitOffset { byte_offset, .. } => Some(byte_offset),
            ElementLocation::Tmem { .. } => None,
        }
    }

    /// Return a TMEM coordinate, or `None` for a byte-addressed state space.
    pub const fn tmem_coordinates(self) -> Option<(i64, i64, i64)> {
        match self.location {
            ElementLocation::ByteOffset(_) => None,
            ElementLocation::BitOffset { .. } => None,
            ElementLocation::Tmem {
                mapped_lane,
                tcol_element,
                allocated_addr,
                ..
            } => Some((mapped_lane, tcol_element, allocated_addr)),
        }
    }

    /// Bit position within the mapped byte/cell. Byte-sized elements return
    /// zero; packed sub-byte formats currently use 0 or 4.
    pub const fn bit_offset(self) -> u8 {
        match self.location {
            ElementLocation::ByteOffset(_) => 0,
            ElementLocation::BitOffset { bit_offset, .. }
            | ElementLocation::Tmem { bit_offset, .. } => bit_offset,
        }
    }

    pub const fn target_rank(self) -> Option<u32> {
        self.target_rank
    }

    pub const fn is_in_bounds(self) -> bool {
        self.in_bounds
    }

    pub const fn is_owned(self) -> bool {
        self.owned
    }

    pub(crate) const fn location(self) -> ElementLocation {
        self.location
    }
}

macro_rules! byte_element_refs {
    ($($space:ty),+ $(,)?) => {
        $(
            impl ElementRef<$space> {
                pub const fn in_bounds(byte_offset: i128, target_rank: Option<u32>) -> Self {
                    Self {
                        location: ElementLocation::ByteOffset(byte_offset),
                        target_rank,
                        owned: true,
                        in_bounds: true,
                        marker: PhantomData,
                    }
                }

                pub const fn in_bounds_bits(
                    byte_offset: i128,
                    bit_offset: u8,
                    target_rank: Option<u32>,
                ) -> Self {
                    Self {
                        location: ElementLocation::BitOffset {
                            byte_offset,
                            bit_offset,
                        },
                        target_rank,
                        owned: true,
                        in_bounds: true,
                        marker: PhantomData,
                    }
                }
            }
        )+
    };
}

byte_element_refs!(Generic, Global, Shared, Local, Register);

impl ElementRef<Tmem> {
    pub const fn in_bounds_tmem(
        mapped_lane: i64,
        tcol_element: i64,
        allocated_addr: i64,
        target_rank: Option<u32>,
    ) -> Self {
        Self {
            location: ElementLocation::Tmem {
                mapped_lane,
                tcol_element,
                allocated_addr,
                bit_offset: 0,
            },
            target_rank,
            owned: true,
            in_bounds: true,
            marker: PhantomData,
        }
    }

    pub const fn in_bounds_tmem_bits(
        mapped_lane: i64,
        tcol_element: i64,
        allocated_addr: i64,
        bit_offset: u8,
        target_rank: Option<u32>,
    ) -> Self {
        Self {
            location: ElementLocation::Tmem {
                mapped_lane,
                tcol_element,
                allocated_addr,
                bit_offset,
            },
            target_rank,
            owned: true,
            in_bounds: true,
            marker: PhantomData,
        }
    }
}

/// Error produced by a pure frontend layout map before memory is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapError(String);

impl MapError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MapError {}

/// Pure layout program supplied by a frontend.
///
/// Implementations receive no engine or memory handle, so mapping cannot
/// perform an effect or create a second simulation path.
pub trait ElementMap<S: MemorySpace>: Send + Sync {
    type Runtime: Send + Sync;

    /// Lanes in the current warp that own this logical element.
    ///
    /// The default preserves the fully general probing contract. Frontends
    /// with an explicit distributed layout can return the exact owner mask so
    /// the engine does not have to rediscover layout ownership lane by lane.
    fn owners(
        &self,
        _logical: LogicalCoord<'_>,
        _runtime: &Self::Runtime,
    ) -> Result<LaneMask, MapError> {
        Ok(LaneMask::FULL)
    }

    fn map(
        &self,
        logical: LogicalCoord<'_>,
        lane: LaneId,
        runtime: &Self::Runtime,
    ) -> Result<ElementRef<S>, MapError>;
}

trait BoundElementMap<S: MemorySpace>: Send + Sync {
    fn owners(&self, logical: LogicalCoord<'_>) -> Result<LaneMask, MapError>;

    fn map(&self, logical: LogicalCoord<'_>, lane: LaneId) -> Result<ElementRef<S>, MapError>;
}

struct BoundMap<M, R> {
    mapper: M,
    runtime: R,
}

struct Unmapped<S: MemorySpace>(PhantomData<fn() -> S>);

impl<S: MemorySpace> ElementMap<S> for Unmapped<S> {
    type Runtime = ();

    fn map(
        &self,
        _logical: LogicalCoord<'_>,
        _lane: LaneId,
        (): &Self::Runtime,
    ) -> Result<ElementRef<S>, MapError> {
        Err(MapError::new(
            "allocation-only tile operand has no frontend element map",
        ))
    }
}

impl<S, M, R> BoundElementMap<S> for BoundMap<M, R>
where
    S: MemorySpace,
    M: ElementMap<S, Runtime = R>,
    R: Send + Sync,
{
    fn owners(&self, logical: LogicalCoord<'_>) -> Result<LaneMask, MapError> {
        self.mapper.owners(logical, &self.runtime)
    }

    fn map(&self, logical: LogicalCoord<'_>, lane: LaneId) -> Result<ElementRef<S>, MapError> {
        self.mapper.map(logical, lane, &self.runtime)
    }
}

/// One allocation paired with a frontend-owned pure layout program.
///
/// The mapper is type-erased at construction so frontend-specific layout
/// types do not become part of every tile-function signature.
#[derive(Clone)]
pub struct MappedView<S: MemorySpace> {
    allocation: BufferHandle<S>,
    mapping: Arc<dyn BoundElementMap<S>>,
}

impl<S: MemorySpace> MappedView<S> {
    pub fn new<M>(allocation: BufferHandle<S>, mapper: M, runtime: M::Runtime) -> Self
    where
        M: ElementMap<S> + 'static,
        M::Runtime: 'static,
    {
        Self {
            allocation,
            mapping: Arc::new(BoundMap { mapper, runtime }),
        }
    }

    /// Construct the allocation carrier used by a compile-time mapping
    /// specialization. Calling `map`/`owners` on this value is an engine bug;
    /// the selected specialization already fixes the complete PTX mapping.
    pub fn allocation_only(allocation: BufferHandle<S>) -> Self {
        Self::new(allocation, Unmapped(PhantomData), ())
    }

    pub(crate) const fn allocation(&self) -> &BufferHandle<S> {
        &self.allocation
    }

    pub(crate) fn map(
        &self,
        logical: LogicalCoord<'_>,
        lane: LaneId,
    ) -> Result<ElementRef<S>, MapError> {
        self.mapping.map(logical, lane)
    }

    pub(crate) fn owners(&self, logical: LogicalCoord<'_>) -> Result<LaneMask, MapError> {
        self.mapping.owners(logical)
    }
}

/// Opaque token connecting optional source-level observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueOrigin(pub(crate) u64);

mod sealed {
    use super::*;

    pub trait WarpHandle {
        type Mode: EngineMode;

        fn engine(&mut self) -> &mut WarpEngine<Self::Mode>;
    }

    impl<M: EngineMode> WarpHandle for WarpEngine<M> {
        type Mode = M;

        #[inline(always)]
        fn engine(&mut self) -> &mut WarpEngine<Self::Mode> {
            self
        }
    }
}

/// Method-free, sealed handle accepted by v2 operations.
///
/// Generated split helpers can stay generic without naming engine modes or
/// checker types.  The private supertrait gives the engine static dispatch.
#[allow(private_bounds)]
pub trait WarpHandle: sealed::WarpHandle {}

impl<T: sealed::WarpHandle> WarpHandle for T {}

/// Concrete warp engine selected by the artifact build profile.
///
/// NumSim, Synccheck, and Racecheck artifacts are built against distinct
/// engine rlibs.  Keeping the instruction ABI concrete at that rlib boundary
/// lets Rust compile instruction bodies once in the engine crate; generated
/// artifacts only monomorphize their tiny variant adapters.
#[cfg(not(feature = "analysis-core"))]
pub type Engine = WarpEngine<crate::engine_mode::NumSimMode>;
#[cfg(all(feature = "analysis-core", not(feature = "racecheck")))]
pub type Engine = WarpEngine<crate::sync_check::SyncCheckMode>;
#[cfg(feature = "racecheck")]
pub type Engine = WarpEngine<crate::race_check::RaceCheckMode>;

#[inline(always)]
pub(crate) fn engine<W: WarpHandle>(warp: &mut W) -> &mut WarpEngine<W::Mode> {
    sealed::WarpHandle::engine(warp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_mask_exposes_only_native_control_set_algebra() {
        let even = LaneMask::from_predicate(|lane| lane % 2 == 0);
        let low = LaneMask::from_bits(0xff);
        assert_eq!((even & low).iter().collect::<Vec<_>>(), vec![0, 2, 4, 6]);
        assert_eq!((low - even).iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
        assert_eq!(even | !even, LaneMask::FULL);
        assert_eq!(
            LaneMask::single(32).unwrap_err().to_string(),
            "lane 32 is outside a 32-lane warp"
        );
    }

    #[test]
    fn register_values_derive_masks_without_exposing_warp_value() {
        let values = R::from_fn(|lane| lane as i32);
        let selected = values.to_mask(|lane, value| lane % 2 == 0 && *value >= 28);
        assert_eq!(selected.iter().collect::<Vec<_>>(), vec![28, 30]);
    }
}
