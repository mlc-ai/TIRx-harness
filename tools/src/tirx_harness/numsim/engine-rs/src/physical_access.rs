use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::{AllocationId, DynamicOpId, LaunchTopology, OperationContext, WARP_SIZE};

/// Stable physical state space for one analysis-visible memory access.
///
/// Explicit discriminants keep serialized/debug ordering stable as new spaces
/// are added. Variants must be appended rather than inserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PhysicalAccessSpace {
    Global = 0,
    Shared = 1,
    Local = 2,
    Register = 3,
    Tmem = 4,
}

impl PhysicalAccessSpace {
    /// Spaces whose strong accesses can carry a release/read-from relation.
    pub(crate) const fn has_read_from_versions(self) -> bool {
        matches!(self, Self::Global | Self::Shared)
    }
}

impl fmt::Display for PhysicalAccessSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Global => "global",
            Self::Shared => "shared",
            Self::Local => "local",
            Self::Register => "register",
            Self::Tmem => "tmem",
        })
    }
}

/// Read/write semantics of one physical access batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PhysicalAccessKind {
    Read = 0,
    Write = 1,
    AtomicReadModifyWrite = 2,
}

impl PhysicalAccessKind {
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::AtomicReadModifyWrite)
    }

    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::AtomicReadModifyWrite)
    }
}

impl fmt::Display for PhysicalAccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::AtomicReadModifyWrite => "atomic_read_modify_write",
        })
    }
}

/// Ordering semantics attached to one analysis-visible memory operation.
///
/// The discriminants and ordering are part of the diagnostic identity. New
/// variants must be appended rather than inserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryOrder {
    Weak = 0,
    Relaxed = 1,
    Acquire = 2,
    Release = 3,
    AcqRel = 4,
    Sc = 5,
}

impl MemoryOrder {
    pub(crate) const fn is_strong(self) -> bool {
        !matches!(self, Self::Weak)
    }

    pub(crate) const fn has_acquire(self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::Sc)
    }

    pub(crate) const fn has_release(self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::Sc)
    }
}

impl fmt::Display for MemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Weak => "weak",
            Self::Relaxed => "relaxed",
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::Sc => "sc",
        })
    }
}

/// PTX execution scope carried by a strong memory operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryScope {
    Cta = 0,
    Cluster = 1,
    Gpu = 2,
    Sys = 3,
}

impl MemoryScope {
    /// Smallest scope covering two actors of one launch. Warps on different
    /// ranks run on different devices and therefore require `.sys`.
    /// Without topology, only equal warp IDs establish a shared CTA.
    pub(crate) fn required_between_warps(
        topology: Option<LaunchTopology>,
        left: usize,
        right: usize,
    ) -> Self {
        if left == right {
            return Self::Cta;
        }
        let Some(topology) = topology else {
            return Self::Gpu;
        };
        if left / topology.warps_per_cta() == right / topology.warps_per_cta() {
            Self::Cta
        } else if topology.cluster_id_for_warp(left) == topology.cluster_id_for_warp(right) {
            Self::Cluster
        } else if topology.rank_of_warp(left) == topology.rank_of_warp(right) {
            Self::Gpu
        } else {
            Self::Sys
        }
    }
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cta => "cta",
            Self::Cluster => "cluster",
            Self::Gpu => "gpu",
            Self::Sys => "sys",
        })
    }
}

/// Memory proxy through which an operation observes physical bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryProxy {
    Generic = 0,
    Async = 1,
    Mmio = 2,
    /// Generic access through a multicast (multimem) virtual address. PTX
    /// ISA 8.6: distinct virtual aliases behave as different proxies, so
    /// ordering against unicast accesses of the same bytes needs an alias
    /// proxy fence along the synchronization path.
    MulticastAlias = 3,
}

impl MemoryProxy {
    /// Strong atomics at the same physical bytes stay coherent across the
    /// unicast and multicast aliases: a multimem atomic is performed at each
    /// replica's memory. Only data ordering needs `fence.proxy.alias`.
    pub(crate) const fn atomics_cohere(self, other: Self) -> bool {
        self as u8 == other as u8
            || matches!(
                (self, other),
                (Self::Generic, Self::MulticastAlias) | (Self::MulticastAlias, Self::Generic)
            )
    }
}

impl fmt::Display for MemoryProxy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Generic => "generic",
            Self::Async => "async",
            Self::Mmio => "mmio",
            Self::MulticastAlias => "multicast_alias",
        })
    }
}

/// State-space domain used to match a memory access with a proxy fence.
///
/// This is intentionally separate from [`PhysicalAccessSpace`]. The latter
/// identifies the allocation class used by byte shadows, while PTX
/// `fence.proxy.async` distinguishes local `shared::cta` objects from
/// cluster-addressed `shared::cluster` objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub(crate) enum ProxyMemoryDomain {
    Global = 0,
    SharedCta = 1,
    SharedCluster = 2,
    Other = 3,
}

impl ProxyMemoryDomain {
    const fn default_for_space(space: PhysicalAccessSpace) -> Self {
        match space {
            PhysicalAccessSpace::Global => Self::Global,
            PhysicalAccessSpace::Shared => Self::SharedCta,
            PhysicalAccessSpace::Local
            | PhysicalAccessSpace::Register
            | PhysicalAccessSpace::Tmem => Self::Other,
        }
    }
}

/// Atomicity/observation class used by the PTX morally-strong relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryAccessClass {
    Plain = 0,
    Atomic = 1,
    Reduction = 2,
    Async = 3,
}

impl MemoryAccessClass {
    pub(crate) const fn is_atomic_class(self) -> bool {
        matches!(self, Self::Atomic | Self::Reduction)
    }

    pub(crate) const fn can_acquire(self) -> bool {
        matches!(self, Self::Atomic)
    }
}

impl fmt::Display for MemoryAccessClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Plain => "plain",
            Self::Atomic => "atomic",
            Self::Reduction => "reduction",
            Self::Async => "async",
        })
    }
}

/// Complete checker-visible memory-model metadata for one access.
///
/// Plain accesses deliberately carry no implicit scope. Volatile accesses are
/// represented as `relaxed/sys/atomic/generic`, while MMIO remains explicit so
/// unsupported device behavior can fail closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryAccessSemantics {
    order: MemoryOrder,
    scope: Option<MemoryScope>,
    proxy: MemoryProxy,
    class: MemoryAccessClass,
}

impl MemoryAccessSemantics {
    pub const fn plain() -> Self {
        Self {
            order: MemoryOrder::Weak,
            scope: None,
            proxy: MemoryProxy::Generic,
            class: MemoryAccessClass::Plain,
        }
    }

    pub const fn scoped(
        order: MemoryOrder,
        scope: MemoryScope,
        proxy: MemoryProxy,
        class: MemoryAccessClass,
    ) -> Self {
        Self {
            order,
            scope: Some(scope),
            proxy,
            class,
        }
    }

    /// The same access performed through another proxy.
    pub const fn with_proxy(mut self, proxy: MemoryProxy) -> Self {
        self.proxy = proxy;
        self
    }

    pub const fn volatile() -> Self {
        Self::scoped(
            MemoryOrder::Relaxed,
            MemoryScope::Sys,
            MemoryProxy::Generic,
            MemoryAccessClass::Atomic,
        )
    }

    pub const fn mmio() -> Self {
        Self::scoped(
            MemoryOrder::Relaxed,
            MemoryScope::Sys,
            MemoryProxy::Mmio,
            MemoryAccessClass::Atomic,
        )
    }

    /// One completion-time access performed by an asynchronous copy engine.
    ///
    /// Completion/wait and proxy-fence effects, rather than the access itself,
    /// provide ordering and generic-proxy visibility.
    pub(crate) const fn async_proxy() -> Self {
        Self {
            order: MemoryOrder::Weak,
            scope: None,
            proxy: MemoryProxy::Async,
            class: MemoryAccessClass::Async,
        }
    }

    /// One completion-time access performed by classic `cp.async`.
    ///
    /// Classic asynchronous copies are async operations but use the generic
    /// memory proxy, unlike bulk asynchronous copies and TCGEN shared-memory
    /// accesses.
    pub(crate) const fn async_generic() -> Self {
        Self {
            order: MemoryOrder::Weak,
            scope: None,
            proxy: MemoryProxy::Generic,
            class: MemoryAccessClass::Async,
        }
    }

    /// One completion-time atomic reduction performed through the async proxy.
    ///
    /// Tensor-map reductions have the fixed `.relaxed.gpu` contract.  Raw
    /// non-tensor reductions whose PTX spelling defaults to another scope use
    /// [`Self::async_reduction_at`] explicitly at their access planner.
    pub(crate) const fn async_reduction() -> Self {
        Self::async_reduction_at(MemoryScope::Gpu)
    }

    /// One completion-time atomic reduction at an explicitly reviewed scope.
    pub(crate) const fn async_reduction_at(scope: MemoryScope) -> Self {
        Self::scoped(
            MemoryOrder::Relaxed,
            scope,
            MemoryProxy::Async,
            MemoryAccessClass::Reduction,
        )
    }

    /// One independently scheduled access that still uses the generic proxy.
    ///
    /// PTX `st.async` is asynchronous with respect to its issuing thread, but
    /// unlike TMA and tcgen05 its memory access is explicitly performed in the
    /// generic proxy.
    pub(crate) const fn generic_async() -> Self {
        Self {
            order: MemoryOrder::Weak,
            scope: None,
            proxy: MemoryProxy::Generic,
            class: MemoryAccessClass::Async,
        }
    }

    pub(crate) const fn order(self) -> MemoryOrder {
        self.order
    }

    pub(crate) const fn scope(self) -> Option<MemoryScope> {
        self.scope
    }

    pub(crate) const fn proxy(self) -> MemoryProxy {
        self.proxy
    }

    pub(crate) const fn class(self) -> MemoryAccessClass {
        self.class
    }

    /// The same access, reclassified as a reduction.
    ///
    /// `red.*` shares its order, scope and proxy vocabulary with `atom.*` and
    /// is built from the same markers; what separates the two is that a
    /// reduction discards the pre-image instead of handing it back.
    pub(crate) const fn as_reduction(self) -> Self {
        Self {
            class: MemoryAccessClass::Reduction,
            ..self
        }
    }

}

impl Default for MemoryAccessSemantics {
    fn default() -> Self {
        Self::plain()
    }
}

/// Non-zero byte width of one active lane's logical access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalAccessWidth(NonZeroUsize);

impl PhysicalAccessWidth {
    pub const fn new(bytes: usize) -> Option<Self> {
        match NonZeroUsize::new(bytes) {
            Some(bytes) => Some(Self(bytes)),
            None => None,
        }
    }

    pub const fn bytes(self) -> usize {
        self.0.get()
    }
}

impl fmt::Display for PhysicalAccessWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalAccessDescriptorError {
    ZeroWidth,
}

impl fmt::Display for PhysicalAccessDescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => f.write_str("physical access width must be non-zero"),
        }
    }
}

impl Error for PhysicalAccessDescriptorError {}

/// Static semantics shared by every active lane in one dynamic access.
///
/// `width` is the exact per-lane width for ordinary batches and the maximum
/// active-lane width for a lane-varying byte operation. Exact lane footprints
/// always remain available through [`PhysicalAccessBatch::lanes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalAccessDescriptor {
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    width: PhysicalAccessWidth,
    semantics: MemoryAccessSemantics,
    proxy_domain: ProxyMemoryDomain,
}

impl PhysicalAccessDescriptor {
    pub fn new(
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        byte_width: usize,
    ) -> Result<Self, PhysicalAccessDescriptorError> {
        let width =
            PhysicalAccessWidth::new(byte_width).ok_or(PhysicalAccessDescriptorError::ZeroWidth)?;
        Ok(Self {
            kind,
            space,
            width,
            semantics: MemoryAccessSemantics::plain(),
            proxy_domain: ProxyMemoryDomain::default_for_space(space),
        })
    }

    pub fn with_memory_semantics(mut self, semantics: MemoryAccessSemantics) -> Self {
        self.semantics = semantics;
        self
    }

    pub(crate) fn with_proxy_memory_domain(mut self, domain: ProxyMemoryDomain) -> Self {
        self.proxy_domain = domain;
        self
    }

    pub(crate) fn with_kind(mut self, kind: PhysicalAccessKind) -> Self {
        self.kind = kind;
        self
    }

    pub(crate) fn with_byte_width(mut self, byte_width: usize) -> Self {
        self.width = PhysicalAccessWidth::new(byte_width).expect("a transfer unit is nonempty");
        self
    }

    pub const fn kind(self) -> PhysicalAccessKind {
        self.kind
    }

    pub const fn space(self) -> PhysicalAccessSpace {
        self.space
    }

    pub const fn width(self) -> PhysicalAccessWidth {
        self.width
    }

    pub const fn memory_semantics(self) -> MemoryAccessSemantics {
        self.semantics
    }

    pub(crate) const fn proxy_memory_domain(self) -> ProxyMemoryDomain {
        self.proxy_domain
    }
}

/// Stable allocation component of a physical byte identity.
///
/// This intentionally stores only the engine allocation ID. CTA ownership for
/// SMEM and warp/lane ownership for private memory are already represented by
/// distinct physical allocations in the native engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalAllocationId(u64);

impl PhysicalAllocationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<AllocationId> for PhysicalAllocationId {
    fn from(value: AllocationId) -> Self {
        Self(value.as_u64())
    }
}

impl fmt::Display for PhysicalAllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "allocation#{}", self.0)
    }
}

/// One contiguous half-open physical byte interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalByteSpan {
    allocation: PhysicalAllocationId,
    byte_offset: usize,
    byte_len: NonZeroUsize,
}

impl PhysicalByteSpan {
    pub fn new(
        allocation: PhysicalAllocationId,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Self, PhysicalFootprintError> {
        let byte_len = NonZeroUsize::new(byte_len).ok_or(PhysicalFootprintError::EmptySpan)?;
        byte_offset
            .checked_add(byte_len.get())
            .ok_or(PhysicalFootprintError::SpanEndOverflow {
                allocation,
                byte_offset,
                byte_len: byte_len.get(),
            })?;
        Ok(Self {
            allocation,
            byte_offset,
            byte_len,
        })
    }

    pub const fn allocation(self) -> PhysicalAllocationId {
        self.allocation
    }

    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub const fn byte_len(self) -> usize {
        self.byte_len.get()
    }

    pub const fn byte_end(self) -> usize {
        // Construction proves this addition cannot overflow.
        self.byte_offset + self.byte_len.get()
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.allocation.0 == other.allocation.0
            && self.byte_offset < other.byte_end()
            && other.byte_offset < self.byte_end()
    }
}

impl fmt::Display for PhysicalByteSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:[{}, {})",
            self.allocation,
            self.byte_offset,
            self.byte_end()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalFootprintError {
    EmptyFootprint,
    EmptySpan,
    SpanEndOverflow {
        allocation: PhysicalAllocationId,
        byte_offset: usize,
        byte_len: usize,
    },
    OverlappingSpans {
        first: PhysicalByteSpan,
        second: PhysicalByteSpan,
    },
    TotalByteLengthOverflow,
}

impl fmt::Display for PhysicalFootprintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFootprint => {
                f.write_str("physical footprint must contain at least one span")
            }
            Self::EmptySpan => f.write_str("physical byte span must be non-empty"),
            Self::SpanEndOverflow {
                allocation,
                byte_offset,
                byte_len,
            } => write!(
                f,
                "physical byte span {allocation}:[{byte_offset}, {byte_offset}+{byte_len}) overflows usize"
            ),
            Self::OverlappingSpans { first, second } => write!(
                f,
                "physical footprint contains overlapping spans {first} and {second}"
            ),
            Self::TotalByteLengthOverflow => {
                f.write_str("physical footprint total byte length overflows usize")
            }
        }
    }
}

impl Error for PhysicalFootprintError {}

#[derive(Clone)]
enum PhysicalFootprintSpans {
    Inline(PhysicalByteSpan),
    Heap {
        spans: Box<[PhysicalByteSpan]>,
        byte_len: usize,
    },
}

/// Canonical exact set of physical bytes touched by one active lane.
///
/// Input order and adjacent splitting do not affect identity. Overlapping
/// input spans are rejected instead of silently double-counting bytes; an
/// address/layout gateway must resolve aliases before shadow validation.
#[derive(Clone)]
pub struct PhysicalFootprint {
    spans: PhysicalFootprintSpans,
}

impl PhysicalFootprint {
    /// Distinct allocations in canonical span order, without visiting every
    /// byte span when many spans belong to the same allocation.
    pub(crate) fn allocations(&self) -> impl Iterator<Item = PhysicalAllocationId> + '_ {
        let mut remaining = self.spans();
        std::iter::from_fn(move || {
            let allocation = remaining.first()?.allocation();
            let end = if remaining.last().expect("nonempty").allocation() == allocation {
                remaining.len()
            } else {
                remaining.partition_point(|span| span.allocation() == allocation)
            };
            remaining = &remaining[end..];
            Some(allocation)
        })
    }

    pub fn new(mut spans: Vec<PhysicalByteSpan>) -> Result<Self, PhysicalFootprintError> {
        if spans.is_empty() {
            return Err(PhysicalFootprintError::EmptyFootprint);
        }
        if spans.len() == 1 {
            return Ok(Self::single(spans[0]));
        }
        spans.sort_unstable();

        let mut canonical: Vec<PhysicalByteSpan> = Vec::with_capacity(spans.len());
        for span in spans {
            let Some(previous) = canonical.last_mut() else {
                canonical.push(span);
                continue;
            };
            if previous.allocation != span.allocation {
                canonical.push(span);
                continue;
            }
            if span.byte_offset < previous.byte_end() {
                return Err(PhysicalFootprintError::OverlappingSpans {
                    first: *previous,
                    second: span,
                });
            }
            if span.byte_offset == previous.byte_end() {
                let byte_len = previous
                    .byte_len()
                    .checked_add(span.byte_len())
                    .ok_or(PhysicalFootprintError::TotalByteLengthOverflow)?;
                *previous =
                    PhysicalByteSpan::new(previous.allocation, previous.byte_offset, byte_len)?;
            } else {
                canonical.push(span);
            }
        }

        let byte_len = canonical.iter().try_fold(0_usize, |total, span| {
            total
                .checked_add(span.byte_len())
                .ok_or(PhysicalFootprintError::TotalByteLengthOverflow)
        })?;
        if canonical.len() == 1 {
            Ok(Self::single(canonical[0]))
        } else {
            Ok(Self {
                spans: PhysicalFootprintSpans::Heap {
                    spans: canonical.into_boxed_slice(),
                    byte_len,
                },
            })
        }
    }

    /// Build a footprint that keeps adjacent spans separate.
    ///
    /// Spans are sorted and must not overlap, but neighbours are not fused,
    /// so each span stays an individually reportable access unit. Used for
    /// async transfer plans whose witnesses are per transfer unit.
    pub fn new_unmerged(mut spans: Vec<PhysicalByteSpan>) -> Result<Self, PhysicalFootprintError> {
        if spans.is_empty() {
            return Err(PhysicalFootprintError::EmptyFootprint);
        }
        if spans.len() == 1 {
            return Ok(Self::single(spans[0]));
        }
        if !spans.is_sorted() {
            spans.sort_unstable();
        }
        for pair in spans.windows(2) {
            if pair[0].allocation == pair[1].allocation && pair[1].byte_offset < pair[0].byte_end()
            {
                return Err(PhysicalFootprintError::OverlappingSpans {
                    first: pair[0],
                    second: pair[1],
                });
            }
        }
        let byte_len = spans.iter().try_fold(0_usize, |total, span| {
            total
                .checked_add(span.byte_len())
                .ok_or(PhysicalFootprintError::TotalByteLengthOverflow)
        })?;
        Ok(Self {
            spans: PhysicalFootprintSpans::Heap {
                spans: spans.into_boxed_slice(),
                byte_len,
            },
        })
    }

    const fn single(span: PhysicalByteSpan) -> Self {
        Self {
            spans: PhysicalFootprintSpans::Inline(span),
        }
    }

    /// Canonicalize the union of already-valid footprints.
    ///
    /// Ordinary address resolution rejects overlapping spans because overlap
    /// there indicates an invalid lane layout.  Analysis compaction instead
    /// combines several fragments of one semantic operation, where repeated
    /// or overlapping bytes are a valid set union and must not be counted
    /// twice.
    fn from_union(mut spans: Vec<PhysicalByteSpan>) -> Result<Self, PhysicalFootprintError> {
        if spans.is_empty() {
            return Err(PhysicalFootprintError::EmptyFootprint);
        }
        spans.sort_unstable();

        let mut canonical: Vec<PhysicalByteSpan> = Vec::with_capacity(spans.len());
        for span in spans {
            let Some(previous) = canonical.last_mut() else {
                canonical.push(span);
                continue;
            };
            if previous.allocation() != span.allocation()
                || previous.byte_end() < span.byte_offset()
            {
                canonical.push(span);
                continue;
            }
            let byte_end = previous.byte_end().max(span.byte_end());
            *previous = PhysicalByteSpan::new(
                previous.allocation(),
                previous.byte_offset(),
                byte_end - previous.byte_offset(),
            )?;
        }

        let byte_len = canonical.iter().try_fold(0_usize, |total, span| {
            total
                .checked_add(span.byte_len())
                .ok_or(PhysicalFootprintError::TotalByteLengthOverflow)
        })?;
        if canonical.len() == 1 {
            Ok(Self::single(canonical[0]))
        } else {
            Ok(Self {
                spans: PhysicalFootprintSpans::Heap {
                    spans: canonical.into_boxed_slice(),
                    byte_len,
                },
            })
        }
    }

    pub fn spans(&self) -> &[PhysicalByteSpan] {
        match &self.spans {
            PhysicalFootprintSpans::Inline(span) => std::slice::from_ref(span),
            PhysicalFootprintSpans::Heap { spans, .. } => spans,
        }
    }

    pub const fn byte_len(&self) -> usize {
        match &self.spans {
            PhysicalFootprintSpans::Inline(span) => span.byte_len(),
            PhysicalFootprintSpans::Heap { byte_len, .. } => *byte_len,
        }
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        let self_spans = self.spans();
        let other_spans = other.spans();
        let mut left = 0;
        let mut right = 0;
        while left < self_spans.len() && right < other_spans.len() {
            let a = self_spans[left];
            let b = other_spans[right];
            if a.overlaps(b) {
                return true;
            }
            if (a.allocation, a.byte_end()) <= (b.allocation, b.byte_offset) {
                left += 1;
            } else {
                right += 1;
            }
        }
        false
    }
}

impl fmt::Debug for PhysicalFootprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalFootprint")
            .field("spans", &self.spans())
            .field("byte_len", &self.byte_len())
            .finish()
    }
}

impl PartialEq for PhysicalFootprint {
    fn eq(&self, other: &Self) -> bool {
        self.spans() == other.spans() && self.byte_len() == other.byte_len()
    }
}

impl Eq for PhysicalFootprint {}

impl PartialOrd for PhysicalFootprint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhysicalFootprint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.spans()
            .cmp(other.spans())
            .then_with(|| self.byte_len().cmp(&other.byte_len()))
    }
}

impl std::hash::Hash for PhysicalFootprint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.spans(), state);
        std::hash::Hash::hash(&self.byte_len(), state);
    }
}

/// Exact dynamic source and execution lane responsible for one footprint.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneProvenance {
    operation: Arc<DynamicOpId>,
    lane: u8,
}

impl LaneProvenance {
    fn from_shared(
        operation: Arc<DynamicOpId>,
        lane: usize,
    ) -> Result<Self, InvalidLaneProvenance> {
        if lane >= WARP_SIZE {
            return Err(InvalidLaneProvenance { lane });
        }
        Ok(Self {
            operation,
            lane: lane as u8,
        })
    }

    pub fn operation(&self) -> &DynamicOpId {
        self.operation.as_ref()
    }

    pub(crate) fn shared_operation(&self) -> Arc<DynamicOpId> {
        Arc::clone(&self.operation)
    }

    pub const fn lane(&self) -> usize {
        self.lane as usize
    }
}

impl fmt::Display for LaneProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/lane:{}", self.operation.as_ref(), self.lane)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidLaneProvenance {
    lane: usize,
}

impl InvalidLaneProvenance {
    pub const fn lane(self) -> usize {
        self.lane
    }
}

impl fmt::Display for InvalidLaneProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lane {} is outside warp size {WARP_SIZE}", self.lane)
    }
}

impl Error for InvalidLaneProvenance {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LanePhysicalAccess {
    provenance: LaneProvenance,
    footprint: PhysicalFootprint,
}

impl LanePhysicalAccess {
    pub const fn provenance(&self) -> &LaneProvenance {
        &self.provenance
    }

    /// This lane's access narrowed to one of its spans.
    pub(crate) fn with_span(&self, span: PhysicalByteSpan) -> Self {
        Self {
            provenance: self.provenance.clone(),
            footprint: PhysicalFootprint::single(span),
        }
    }

    pub const fn footprint(&self) -> &PhysicalFootprint {
        &self.footprint
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PhysicalAccessLanes {
    Empty,
    Inline(LanePhysicalAccess),
    Heap(Box<[LanePhysicalAccess]>),
}

impl PhysicalAccessLanes {
    fn from_vec(mut lanes: Vec<LanePhysicalAccess>) -> Self {
        match lanes.len() {
            0 => Self::Empty,
            1 => Self::Inline(
                lanes
                    .pop()
                    .expect("a one-lane physical access has one lane"),
            ),
            _ => Self::Heap(lanes.into_boxed_slice()),
        }
    }

    fn as_slice(&self) -> &[LanePhysicalAccess] {
        match self {
            Self::Empty => &[],
            Self::Inline(lane) => std::slice::from_ref(lane),
            Self::Heap(lanes) => lanes,
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl IntoIterator for PhysicalAccessLanes {
    type Item = LanePhysicalAccess;
    type IntoIter = std::vec::IntoIter<LanePhysicalAccess>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Empty => Vec::new(),
            Self::Inline(lane) => vec![lane],
            Self::Heap(lanes) => lanes.into_vec(),
        }
        .into_iter()
    }
}

/// Engine-private, allocation-free view of an ordinary one-span-per-lane access.
///
/// Native analysis summary modes consume this only after every active lane has
/// resolved successfully.  It deliberately stays separate from
/// [`PhysicalAccessBatch`]: retained reports and asynchronous effects still own
/// their complete lane evidence, while the hot synchronous path need not build
/// 32 `LanePhysicalAccess` values merely to discard them after one callback.
pub(crate) struct CompactPhysicalAccessBatch<'a> {
    operation: &'a OperationContext,
    descriptor: PhysicalAccessDescriptor,
    logical_buffer: Option<&'a str>,
    atomic_return_sync_relevant: bool,
    lane_spans: &'a [Option<PhysicalByteSpan>; WARP_SIZE],
}

impl<'a> CompactPhysicalAccessBatch<'a> {
    pub(crate) fn new(
        operation: &'a OperationContext,
        descriptor: PhysicalAccessDescriptor,
        logical_buffer: Option<&'a str>,
        atomic_return_sync_relevant: bool,
        lane_spans: &'a [Option<PhysicalByteSpan>; WARP_SIZE],
    ) -> Self {
        debug_assert!(
            !atomic_return_sync_relevant
                || descriptor.kind() == PhysicalAccessKind::AtomicReadModifyWrite
        );
        debug_assert!(operation
            .active_mask()
            .into_iter()
            .all(|lane| lane_spans[lane].is_some()));
        Self {
            operation,
            descriptor,
            logical_buffer,
            atomic_return_sync_relevant,
            lane_spans,
        }
    }

    pub(crate) const fn operation(&self) -> &OperationContext {
        self.operation
    }

    pub(crate) const fn descriptor(&self) -> PhysicalAccessDescriptor {
        self.descriptor
    }

    pub(crate) const fn logical_buffer(&self) -> Option<&str> {
        self.logical_buffer
    }

    pub(crate) const fn atomic_return_sync_relevant(&self) -> bool {
        self.atomic_return_sync_relevant
    }

    pub(crate) fn lane_spans(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, PhysicalByteSpan)> + '_ {
        self.operation.active_mask().into_iter().map(|lane| {
            (
                lane,
                self.lane_spans[lane].expect("every active compact physical-access lane resolved"),
            )
        })
    }

    pub(crate) fn lane_span(&self, lane: usize) -> Option<PhysicalByteSpan> {
        self.operation
            .active_mask()
            .contains(lane)
            .then(|| self.lane_spans[lane])
            .flatten()
    }
}

/// One dynamic memory operation resolved for every active lane.
///
/// Construction is all-or-nothing: no batch exists until every active lane's
/// address and exact physical footprint have resolved and matched the declared
/// byte width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalAccessBatch {
    operation: OperationContext,
    descriptor: PhysicalAccessDescriptor,
    logical_buffer: Option<Arc<str>>,
    atomic_return_sync_relevant: bool,
    semantic_access_count: usize,
    lanes: PhysicalAccessLanes,
    lane_widths_vary: bool,
    transfer_units: bool,
    // Number of consecutive transfer batches (this one included) that carry
    // the same units to different targets; their reports interleave per unit.
    transfer_siblings: u8,
    /// Non-zero when every span of a transfer batch is a run of consecutive
    /// units of this size rather than one unit: the span count is then the
    /// run count and `semantic_access_count` still counts units.
    transfer_unit_bytes: u32,
    /// What each active lane left in a declared synchronization word, in
    /// `lanes()` order.
    ///
    /// Present only on a write to a declared word, which is the one place a
    /// checker needs the content: a wait states the condition its protocol
    /// completes on, and deciding which write first made that condition hold
    /// is the whole of the wait's happens-before edge. Every other access
    /// stays content-blind, so nothing else pays for this.
    declared_values: Option<Arc<[u64]>>,
}

impl PhysicalAccessBatch {
    /// Resolve the common case where every active lane touches one contiguous span.
    ///
    /// Unlike [`Self::resolve`], this path does not allocate a temporary span
    /// vector or heap storage for each lane's footprint.
    pub fn resolve_single_span<E>(
        operation: OperationContext,
        descriptor: PhysicalAccessDescriptor,
        mut resolve_lane: impl FnMut(&LaneProvenance) -> Result<PhysicalByteSpan, E>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        let shared_operation = operation.shared_id();
        let mut resolve = |lane| {
            let provenance = LaneProvenance::from_shared(Arc::clone(&shared_operation), lane)
                .expect("WarpMask yielded an out-of-range lane");
            let span = resolve_lane(&provenance).map_err(|source| {
                PhysicalAccessBatchError::LaneResolution {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            if span.byte_len() != descriptor.width().bytes() {
                return Err(PhysicalAccessBatchError::WidthMismatch {
                    provenance,
                    expected_byte_len: descriptor.width().bytes(),
                    actual_byte_len: span.byte_len(),
                });
            }
            Ok(LanePhysicalAccess {
                provenance,
                footprint: PhysicalFootprint::single(span),
            })
        };
        let lanes = if operation.active_mask().len() == 1 {
            let lane = operation
                .active_mask()
                .into_iter()
                .next()
                .expect("a one-lane mask has one lane");
            PhysicalAccessLanes::Inline(resolve(lane)?)
        } else {
            let mut lanes = Vec::with_capacity(operation.active_mask().len());
            for lane in operation.active_mask() {
                lanes.push(resolve(lane)?);
            }
            PhysicalAccessLanes::from_vec(lanes)
        };
        Ok(Self {
            operation,
            descriptor,
            logical_buffer: None,
            atomic_return_sync_relevant: false,
            semantic_access_count: 1,
            lanes,
            lane_widths_vary: false,
            transfer_units: false,
            transfer_siblings: 0,
            transfer_unit_bytes: 0,
            declared_values: None,
        })
    }

    pub fn resolve<E>(
        operation: OperationContext,
        descriptor: PhysicalAccessDescriptor,
        resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        Self::resolve_with(operation, descriptor, resolve_lane, PhysicalFootprint::new)
    }

    /// Like [`Self::resolve`], but keeps adjacent spans of a lane separate so
    /// every span remains its own access unit.
    pub fn resolve_unmerged<E>(
        operation: OperationContext,
        descriptor: PhysicalAccessDescriptor,
        resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        let mut batch = Self::resolve_with(
            operation,
            descriptor,
            resolve_lane,
            PhysicalFootprint::new_unmerged,
        )?;
        // Every span is one transfer unit, i.e. one semantic access that
        // reports exactly as it did when each unit was its own batch.
        batch.transfer_units = true;
        batch.semantic_access_count = batch
            .lanes()
            .iter()
            .map(|lane| lane.footprint().spans().len())
            .sum::<usize>()
            .max(1);
        Ok(batch)
    }

    /// Whether every span of every lane is a separately reported transfer
    /// unit (see [`Self::resolve_unmerged`]).
    pub(crate) const fn transfer_units(&self) -> bool {
        self.transfer_units
    }

    /// Resolve a transfer whose lane footprints hold runs of consecutive
    /// `unit_bytes`-sized units instead of one span per unit. The semantic
    /// access count stays the unit count, so statistics and version
    /// identities match the per-unit form.
    pub(crate) fn resolve_transfer_runs<E>(
        operation: OperationContext,
        descriptor: PhysicalAccessDescriptor,
        resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
        unit_bytes: u32,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        debug_assert!(unit_bytes > 0);
        let mut batch = Self::resolve_with(
            operation,
            descriptor,
            resolve_lane,
            PhysicalFootprint::new_unmerged,
        )?;
        debug_assert!(batch.lanes().iter().all(|lane| lane
            .footprint()
            .spans()
            .iter()
            .all(|span| span.byte_len() % unit_bytes as usize == 0)));
        batch.transfer_units = true;
        batch.transfer_unit_bytes = unit_bytes;
        batch.semantic_access_count = batch
            .lanes()
            .iter()
            .flat_map(|lane| lane.footprint().spans())
            .map(|span| span.byte_len() / unit_bytes as usize)
            .sum::<usize>()
            .max(1);
        Ok(batch)
    }

    /// Unit size when the spans are runs of units (see
    /// [`Self::resolve_transfer_runs`]); zero when each span is one unit.
    pub(crate) const fn transfer_unit_bytes(&self) -> u32 {
        self.transfer_unit_bytes
    }

    /// Strong transfers retain their naturally aligned element grid even
    /// when their footprint omits bytes within an element (e.g. ignore_oob).
    /// The footprint remains the actual accessed bytes, not the element hull.
    pub(crate) fn with_aligned_transfer_units(mut self, unit_bytes: u32) -> Self {
        assert!(unit_bytes > 0);
        assert!(self.descriptor.memory_semantics().order().is_strong());
        self.transfer_units = true;
        self.transfer_unit_bytes = unit_bytes;
        let width = unit_bytes as usize;
        self.semantic_access_count = self
            .lanes()
            .iter()
            .map(|lane| {
                let mut previous = None;
                let mut count = 0;
                for span in lane.footprint().spans() {
                    let first = span.byte_offset() / width;
                    let end = span.byte_end().div_ceil(width);
                    let start = match previous {
                        Some((allocation, stop)) if allocation == span.allocation() => {
                            first.max(stop)
                        }
                        _ => first,
                    };
                    count += end.saturating_sub(start);
                    previous = Some((span.allocation(), end));
                }
                count
            })
            .sum::<usize>()
            .max(1);
        self
    }

    /// Mark this transfer batch as one of `count` consecutive batches that
    /// deliver the same units to different targets (a multicast), so reports
    /// list unit by unit across the targets exactly as one batch per unit and
    /// target would.
    pub(crate) fn with_transfer_siblings(mut self, count: usize) -> Self {
        debug_assert!(self.transfer_units);
        self.transfer_siblings = u8::try_from(count).expect("a multicast has few targets");
        self
    }

    pub(crate) const fn transfer_siblings(&self) -> usize {
        self.transfer_siblings as usize
    }

    fn resolve_with<E>(
        operation: OperationContext,
        descriptor: PhysicalAccessDescriptor,
        mut resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
        footprint: fn(Vec<PhysicalByteSpan>) -> Result<PhysicalFootprint, PhysicalFootprintError>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        let mut lanes = Vec::with_capacity(operation.active_mask().len());
        let shared_operation = operation.shared_id();
        for lane in operation.active_mask() {
            let provenance = LaneProvenance::from_shared(Arc::clone(&shared_operation), lane)
                .expect("WarpMask yielded an out-of-range lane");
            let spans = resolve_lane(&provenance).map_err(|source| {
                PhysicalAccessBatchError::LaneResolution {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            let footprint =
                footprint(spans).map_err(|source| PhysicalAccessBatchError::InvalidFootprint {
                    provenance: provenance.clone(),
                    source,
                })?;
            if footprint.byte_len() != descriptor.width().bytes() {
                return Err(PhysicalAccessBatchError::WidthMismatch {
                    provenance,
                    expected_byte_len: descriptor.width().bytes(),
                    actual_byte_len: footprint.byte_len(),
                });
            }
            lanes.push(LanePhysicalAccess {
                provenance,
                footprint,
            });
        }
        Ok(Self {
            operation,
            descriptor,
            logical_buffer: None,
            atomic_return_sync_relevant: false,
            semantic_access_count: 1,
            lanes: PhysicalAccessLanes::from_vec(lanes),
            lane_widths_vary: false,
            transfer_units: false,
            transfer_siblings: 0,
            transfer_unit_bytes: 0,
            declared_values: None,
        })
    }

    /// Resolve an instruction whose byte width is data-dependent per lane.
    ///
    /// The batch retains one operation and therefore one vector-clock event;
    /// splitting these lanes into separate batches would incorrectly impose
    /// program order between lanes of the same SIMT instruction.
    pub fn resolve_lane_widths<E>(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        mut resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        let mut lanes = Vec::with_capacity(operation.active_mask().len());
        let mut minimum_width = usize::MAX;
        let mut maximum_width = 0_usize;
        let shared_operation = operation.shared_id();
        for lane in operation.active_mask() {
            let provenance = LaneProvenance::from_shared(Arc::clone(&shared_operation), lane)
                .expect("WarpMask yielded an out-of-range lane");
            let spans = resolve_lane(&provenance).map_err(|source| {
                PhysicalAccessBatchError::LaneResolution {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            let footprint = PhysicalFootprint::new(spans).map_err(|source| {
                PhysicalAccessBatchError::InvalidFootprint {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            minimum_width = minimum_width.min(footprint.byte_len());
            maximum_width = maximum_width.max(footprint.byte_len());
            lanes.push(LanePhysicalAccess {
                provenance,
                footprint,
            });
        }
        let descriptor = PhysicalAccessDescriptor::new(kind, space, maximum_width.max(1))
            .expect("a positive maximum lane width produces a valid descriptor");
        Ok(Self {
            operation,
            descriptor,
            logical_buffer: None,
            atomic_return_sync_relevant: false,
            semantic_access_count: 1,
            lanes: PhysicalAccessLanes::from_vec(lanes),
            lane_widths_vary: minimum_width != usize::MAX && minimum_width != maximum_width,
            transfer_units: false,
            transfer_siblings: 0,
            transfer_unit_bytes: 0,
            declared_values: None,
        })
    }

    /// Resolve the union of fragments accumulated for each active lane.
    ///
    /// This is engine-private compaction: overlapping fragments are expected
    /// when a layout aliases bytes within one semantic asynchronous copy.
    pub(crate) fn resolve_lane_unions<E>(
        operation: OperationContext,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        mut resolve_lane: impl FnMut(&LaneProvenance) -> Result<Vec<PhysicalByteSpan>, E>,
    ) -> Result<Self, PhysicalAccessBatchError<E>> {
        let mut lanes = Vec::with_capacity(operation.active_mask().len());
        let mut minimum_width = usize::MAX;
        let mut maximum_width = 0_usize;
        let shared_operation = operation.shared_id();
        for lane in operation.active_mask() {
            let provenance = LaneProvenance::from_shared(Arc::clone(&shared_operation), lane)
                .expect("WarpMask yielded an out-of-range lane");
            let spans = resolve_lane(&provenance).map_err(|source| {
                PhysicalAccessBatchError::LaneResolution {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            let footprint = PhysicalFootprint::from_union(spans).map_err(|source| {
                PhysicalAccessBatchError::InvalidFootprint {
                    provenance: provenance.clone(),
                    source,
                }
            })?;
            minimum_width = minimum_width.min(footprint.byte_len());
            maximum_width = maximum_width.max(footprint.byte_len());
            lanes.push(LanePhysicalAccess {
                provenance,
                footprint,
            });
        }
        let descriptor = PhysicalAccessDescriptor::new(kind, space, maximum_width.max(1))
            .expect("a positive maximum lane union width produces a valid descriptor");
        Ok(Self {
            operation,
            descriptor,
            logical_buffer: None,
            atomic_return_sync_relevant: false,
            semantic_access_count: 1,
            lanes: PhysicalAccessLanes::from_vec(lanes),
            lane_widths_vary: minimum_width != usize::MAX && minimum_width != maximum_width,
            transfer_units: false,
            transfer_siblings: 0,
            transfer_unit_bytes: 0,
            declared_values: None,
        })
    }

    /// Attach the source-level buffer name used for this physical access.
    ///
    /// Physical race detection remains allocation-based; this identity is
    /// additive evidence for value-provenance advisories across pool aliases.
    pub fn with_logical_buffer(mut self, logical_buffer: impl Into<Arc<str>>) -> Self {
        self.logical_buffer = Some(logical_buffer.into());
        self
    }

    /// Replace the checker-visible memory semantics without rebuilding the
    /// already-resolved per-lane footprints.
    pub fn with_memory_semantics(mut self, semantics: MemoryAccessSemantics) -> Self {
        self.descriptor = self.descriptor.with_memory_semantics(semantics);
        self
    }

    pub(crate) fn with_proxy_memory_domain(mut self, domain: ProxyMemoryDomain) -> Self {
        self.descriptor = self.descriptor.with_proxy_memory_domain(domain);
        self
    }

    pub(crate) fn with_kind(mut self, kind: PhysicalAccessKind) -> Self {
        debug_assert!(!self.atomic_return_sync_relevant);
        self.descriptor = self.descriptor.with_kind(kind);
        self
    }

    /// Mark a returning atomic whose value is in the backward slice of
    /// synchronization behavior. Non-atomic batches and output-only atomic
    /// returns keep the default `false` value.
    /// Record the post-image each active lane left in a declared word.
    ///
    /// Read back after the write, in `lanes()` order. Nothing runs between the
    /// write and the read -- the two sit in one synchronous span of the same
    /// warp -- so the value is exactly what this access left behind.
    pub(crate) fn with_declared_values(mut self, values: Arc<[u64]>) -> Self {
        debug_assert!(self.descriptor.memory_semantics().class().is_atomic_class());
        debug_assert!(self.descriptor.kind().writes());
        debug_assert_eq!(values.len(), self.lanes().len());
        self.declared_values = Some(values);
        self
    }

    /// What one lane left in a declared word.
    ///
    /// The values parallel `lanes()`, which a consumer walking lanes by number
    /// rather than by position would otherwise have to rediscover; keeping the
    /// convention here means only one place knows it.
    pub(crate) fn declared_value_for_lane(&self, lane: usize) -> Option<u64> {
        let values = self.declared_values.as_deref()?;
        let position = self
            .lanes()
            .iter()
            .position(|entry| entry.provenance().lane() == lane)?;
        values.get(position).copied()
    }

    pub fn with_atomic_return_sync_relevant(mut self, sync_relevant: bool) -> Self {
        debug_assert!(
            !sync_relevant || self.descriptor.kind() == PhysicalAccessKind::AtomicReadModifyWrite
        );
        self.atomic_return_sync_relevant = sync_relevant;
        self
    }

    pub(crate) fn with_semantic_access_count(mut self, count: usize) -> Self {
        debug_assert!(count > 0);
        self.semantic_access_count = count;
        self
    }

    pub const fn operation(&self) -> &OperationContext {
        &self.operation
    }

    pub const fn descriptor(&self) -> PhysicalAccessDescriptor {
        self.descriptor
    }

    pub fn logical_buffer(&self) -> Option<&str> {
        self.logical_buffer.as_deref()
    }

    pub(crate) fn shared_logical_buffer(&self) -> Option<Arc<str>> {
        self.logical_buffer.as_ref().map(Arc::clone)
    }

    pub const fn atomic_return_sync_relevant(&self) -> bool {
        self.atomic_return_sync_relevant
    }

    pub(crate) const fn semantic_access_count(&self) -> usize {
        self.semantic_access_count
    }

    pub fn lanes(&self) -> &[LanePhysicalAccess] {
        self.lanes.as_slice()
    }

    pub const fn lane_widths_vary(&self) -> bool {
        self.lane_widths_vary
    }

    pub fn lane(&self, lane: usize) -> Option<&LanePhysicalAccess> {
        self.lanes()
            .binary_search_by_key(&lane, |access| access.provenance.lane())
            .ok()
            .map(|index| &self.lanes()[index])
    }

    /// Report whether two active lanes touch any common physical byte.
    pub(crate) fn has_inter_lane_overlap(&self) -> bool {
        for (first_index, first) in self.lanes().iter().enumerate() {
            for second in &self.lanes()[first_index + 1..] {
                if first.footprint().overlaps(second.footprint()) {
                    return true;
                }
            }
        }
        false
    }

}

struct PhysicalAccessBatchUnion {
    first: PhysicalAccessBatch,
    spans_by_lane: [Vec<PhysicalByteSpan>; WARP_SIZE],
    batch_count: usize,
    semantic_access_count: usize,
}

impl PhysicalAccessBatchUnion {
    fn has_same_identity(&self, batch: &PhysicalAccessBatch) -> bool {
        // Strong access identity includes element boundaries, not just the
        // union of touched bytes. Keep those batches intact for the shadow.
        !self.first.descriptor.memory_semantics().order().is_strong()
            && self.first.operation == batch.operation
            && self.first.descriptor == batch.descriptor
            && self.first.logical_buffer == batch.logical_buffer
            && self.first.atomic_return_sync_relevant == batch.atomic_return_sync_relevant
    }

    fn push(&mut self, batch: &PhysicalAccessBatch) {
        if self.batch_count == 0 {
            self.batch_count = 1;
            self.semantic_access_count = batch.semantic_access_count;
            // A lone multi-unit transfer batch still unions its units, just as
            // one batch per unit would have been unioned.
            if batch.transfer_units && !batch.descriptor.memory_semantics().order().is_strong() {
                for lane in batch.lanes() {
                    self.spans_by_lane[lane.provenance().lane()]
                        .extend_from_slice(lane.footprint().spans());
                }
                self.batch_count = batch.semantic_access_count.max(2);
            }
            return;
        }
        if self.batch_count == 1 {
            for lane in self.first.lanes() {
                self.spans_by_lane[lane.provenance().lane()]
                    .extend_from_slice(lane.footprint().spans());
            }
        }
        self.batch_count = self.batch_count.saturating_add(1);
        self.semantic_access_count = self
            .semantic_access_count
            .saturating_add(batch.semantic_access_count);
        for lane in batch.lanes() {
            self.spans_by_lane[lane.provenance().lane()]
                .extend_from_slice(lane.footprint().spans());
        }
    }

    fn finish(mut self) -> Result<PhysicalAccessBatch, PhysicalFootprintError> {
        if self.batch_count == 1 {
            return Ok(self.first);
        }

        let mut lanes = Vec::with_capacity(self.first.lanes.len());
        let mut minimum_width = usize::MAX;
        let mut maximum_width = 0_usize;
        let shared_operation = self.first.operation.shared_id();
        for lane in self.first.operation.active_mask() {
            let footprint =
                PhysicalFootprint::from_union(std::mem::take(&mut self.spans_by_lane[lane]))?;
            minimum_width = minimum_width.min(footprint.byte_len());
            maximum_width = maximum_width.max(footprint.byte_len());
            lanes.push(LanePhysicalAccess {
                provenance: LaneProvenance::from_shared(Arc::clone(&shared_operation), lane)
                    .expect("an operation active mask contains only valid lanes"),
                footprint,
            });
        }
        let descriptor = PhysicalAccessDescriptor::new(
            self.first.descriptor.kind(),
            self.first.descriptor.space(),
            maximum_width.max(1),
        )
        .expect("a positive union width produces a valid descriptor")
        .with_memory_semantics(self.first.descriptor.memory_semantics())
        .with_proxy_memory_domain(self.first.descriptor.proxy_memory_domain());
        // A declared word's post-images are attached after the batch is
        // built, so a fragment being merged here never carries any.
        debug_assert!(self.first.declared_values.is_none());
        Ok(PhysicalAccessBatch {
            operation: self.first.operation,
            descriptor,
            logical_buffer: self.first.logical_buffer,
            atomic_return_sync_relevant: self.first.atomic_return_sync_relevant,
            semantic_access_count: self.semantic_access_count,
            lanes: PhysicalAccessLanes::from_vec(lanes),
            lane_widths_vary: minimum_width != usize::MAX && minimum_width != maximum_width,
            transfer_units: false,
            transfer_siblings: 0,
            transfer_unit_bytes: 0,
            declared_values: None,
        })
    }
}

/// Merge fragments of the same semantic memory operation into exact per-lane
/// footprint unions.  This is crate-internal analysis plumbing, not generated
/// kernel ABI: element-level batches remain available for detailed reports.
/// Whether the global-memory transaction of `batches` must be exclusive.
///
/// Exclusivity protects exact read-from: a write publishes a version, and an
/// atomic or ordered access consumes one, so neither may interleave with
/// another transaction on the same allocation between validation, numeric
/// execution, and commit. A weak read (a TMA load of a weight tile, say)
/// consumes no version; it only needs to exclude writers, which the shared
/// side of the lock already does.
pub(crate) fn global_transaction_exclusive<'a>(
    batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
) -> bool {
    batches
        .into_iter()
        .filter(|batch| batch.descriptor().space().has_read_from_versions())
        .any(|batch| {
            let semantics = batch.descriptor().memory_semantics();
            batch.descriptor().kind().writes()
                || semantics.class().is_atomic_class()
                || semantics.order().is_strong()
                || semantics.order().has_acquire()
                || semantics.order().has_release()
        })
}

/// The global/shared byte spans touched by `batches`; the scope of a read-from
/// transaction. Left in lane order: the transaction only maps them to lock
/// stripes, which it sorts and deduplicates itself, so sorting hundreds of
/// tile-row spans per TMA issue here bought nothing.
pub(crate) fn global_batch_spans<'a>(
    batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
) -> Vec<PhysicalByteSpan> {
    batches
        .into_iter()
        .filter(|batch| batch.descriptor().space().has_read_from_versions())
        .flat_map(|batch| batch.lanes())
        .flat_map(|lane| lane.footprint().spans())
        .copied()
        .collect()
}

pub(crate) fn coalesce_physical_access_batches<'a>(
    batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
) -> Result<Box<[PhysicalAccessBatch]>, PhysicalFootprintError> {
    let mut batches = batches.into_iter().peekable();
    let Some(first) = batches.next() else {
        return Ok(Box::new([]));
    };
    if batches.peek().is_none() && !first.transfer_units {
        return Ok(vec![first.clone()].into_boxed_slice());
    }

    let mut groups: Vec<PhysicalAccessBatchUnion> = Vec::new();
    for batch in std::iter::once(first).chain(batches) {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.has_same_identity(batch))
        {
            group.push(batch);
            continue;
        }
        let mut group = PhysicalAccessBatchUnion {
            first: batch.clone(),
            spans_by_lane: std::array::from_fn(|_| Vec::new()),
            batch_count: 0,
            semantic_access_count: 0,
        };
        group.push(batch);
        groups.push(group);
    }
    groups
        .into_iter()
        .map(PhysicalAccessBatchUnion::finish)
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

#[derive(Debug)]
pub enum PhysicalAccessBatchError<E> {
    LaneResolution {
        provenance: LaneProvenance,
        source: E,
    },
    InvalidFootprint {
        provenance: LaneProvenance,
        source: PhysicalFootprintError,
    },
    WidthMismatch {
        provenance: LaneProvenance,
        expected_byte_len: usize,
        actual_byte_len: usize,
    },
}

impl<E> PhysicalAccessBatchError<E> {
    pub const fn provenance(&self) -> &LaneProvenance {
        match self {
            Self::LaneResolution { provenance, .. }
            | Self::InvalidFootprint { provenance, .. }
            | Self::WidthMismatch { provenance, .. } => provenance,
        }
    }
}

impl<E: fmt::Display> fmt::Display for PhysicalAccessBatchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaneResolution { provenance, source } => {
                write!(f, "failed to resolve {provenance}: {source}")
            }
            Self::InvalidFootprint { provenance, source } => {
                write!(f, "invalid footprint for {provenance}: {source}")
            }
            Self::WidthMismatch {
                provenance,
                expected_byte_len,
                actual_byte_len,
            } => write!(
                f,
                "footprint for {provenance} covers {actual_byte_len} bytes, expected {expected_byte_len}"
            ),
        }
    }
}

impl<E: Error + 'static> Error for PhysicalAccessBatchError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LaneResolution { source, .. } => Some(source),
            Self::InvalidFootprint { source, .. } => Some(source),
            Self::WidthMismatch { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{LoopFrame, OperationKind, StaticOpId, WarpMask};

    use super::*;

    fn operation(mask: u32) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(
                2,
                7,
                11,
                StaticOpId::new(19),
                [LoopFrame::new(StaticOpId::new(23), 29)],
            ),
            OperationKind::Load,
            WarpMask::from_bits(mask),
        )
    }

    fn descriptor(width: usize) -> PhysicalAccessDescriptor {
        PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Shared, width)
            .unwrap()
    }

    #[test]
    fn lane_varying_batch_keeps_one_operation_with_exact_lane_footprints() {
        let operation = operation(0b11);
        let batch = PhysicalAccessBatch::resolve_lane_widths(
            operation.clone(),
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            |provenance| {
                let byte_len = (provenance.lane() + 1) * 8;
                Ok::<_, std::convert::Infallible>(vec![span(9, provenance.lane() * 32, byte_len)])
            },
        )
        .unwrap();

        assert_eq!(batch.operation(), &operation);
        assert!(batch.lane_widths_vary());
        assert_eq!(batch.descriptor().width().bytes(), 16);
        assert_eq!(batch.lane(0).unwrap().footprint().byte_len(), 8);
        assert_eq!(batch.lane(1).unwrap().footprint().byte_len(), 16);
    }

    fn span(allocation: u64, offset: usize, len: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(PhysicalAllocationId::new(allocation), offset, len).unwrap()
    }

    #[test]
    fn access_descriptor_has_nonzero_width_and_atomic_is_read_write() {
        assert_eq!(
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Shared, 0,),
            Err(PhysicalAccessDescriptorError::ZeroWidth)
        );
        assert!(PhysicalAccessKind::AtomicReadModifyWrite.reads());
        assert!(PhysicalAccessKind::AtomicReadModifyWrite.writes());
        assert!(!PhysicalAccessKind::Read.writes());
        assert!(!PhysicalAccessKind::Write.reads());
    }

    #[test]
    fn footprint_identity_is_canonical_across_order_and_adjacent_splits() {
        let split = PhysicalFootprint::new(vec![span(3, 12, 4), span(3, 8, 4)]).unwrap();
        let joined = PhysicalFootprint::new(vec![span(3, 8, 8)]).unwrap();
        assert_eq!(split, joined);
        assert_eq!(split.spans(), &[span(3, 8, 8)]);
        assert!(matches!(&split.spans, PhysicalFootprintSpans::Inline(_)));
        assert_eq!(HashSet::from([split, joined]).len(), 1);
    }

    #[test]
    fn single_span_batch_keeps_each_footprint_inline() {
        let operation = operation(0b10101);
        let fast =
            PhysicalAccessBatch::resolve_single_span(operation.clone(), descriptor(4), |lane| {
                Ok::<_, std::convert::Infallible>(span(9, lane.lane() * 4, 4))
            })
            .unwrap();
        let generic = PhysicalAccessBatch::resolve(operation, descriptor(4), |lane| {
            Ok::<_, std::convert::Infallible>(vec![span(9, lane.lane() * 4, 4)])
        })
        .unwrap();

        assert_eq!(fast, generic);
        assert!(fast
            .lanes()
            .iter()
            .all(|access| matches!(&access.footprint.spans, PhysicalFootprintSpans::Inline(_))));
        assert_eq!(fast.clone(), fast);
    }

    #[test]
    fn multispan_footprint_stays_heap_backed_and_slice_ordered() {
        let multiple = PhysicalFootprint::new(vec![span(3, 0, 4), span(3, 8, 4)]).unwrap();
        let single = PhysicalFootprint::new(vec![span(3, 16, 4)]).unwrap();

        assert!(matches!(
            &multiple.spans,
            PhysicalFootprintSpans::Heap { .. }
        ));
        assert_eq!(multiple.spans(), &[span(3, 0, 4), span(3, 8, 4)]);
        assert_eq!(multiple.byte_len(), 8);
        assert!(multiple < single);
        assert!(matches!(
            &multiple.clone().spans,
            PhysicalFootprintSpans::Heap { .. }
        ));
    }

    #[test]
    fn footprint_allocation_runs_match_all_spans_for_every_constructor() {
        use std::collections::BTreeSet;
        for count in [1, 2, 7, 32] {
            let mut spans = Vec::new();
            for allocation in 0..count {
                for index in 0..1 + (allocation * 97 + 511) % 1024 {
                    spans.push(span(allocation as u64 * 3, index * 8, 4));
                }
            }
            spans.reverse();
            let expected = spans
                .iter()
                .map(|span| span.allocation())
                .collect::<BTreeSet<_>>();
            for constructor in [
                PhysicalFootprint::new,
                PhysicalFootprint::new_unmerged,
                PhysicalFootprint::from_union,
            ] {
                let footprint = constructor(spans.clone()).unwrap();
                let actual = footprint.allocations().collect::<Vec<_>>();
                assert_eq!(actual, expected.iter().copied().collect::<Vec<_>>());
                assert_eq!(
                    footprint
                        .spans()
                        .iter()
                        .map(|span| span.allocation())
                        .collect::<BTreeSet<_>>(),
                    expected
                );
            }
        }
        assert_eq!(
            PhysicalFootprint::single(span(9, 5, 1))
                .allocations()
                .collect::<Vec<_>>(),
            vec![PhysicalAllocationId::new(9)]
        );
    }

    #[test]
    fn footprint_rejects_overlapping_aliases_and_span_overflow() {
        assert!(matches!(
            PhysicalFootprint::new(vec![span(1, 4, 8), span(1, 8, 8)]),
            Err(PhysicalFootprintError::OverlappingSpans { .. })
        ));
        assert!(matches!(
            PhysicalByteSpan::new(PhysicalAllocationId::new(1), usize::MAX, 1),
            Err(PhysicalFootprintError::SpanEndOverflow { .. })
        ));
    }

    #[test]
    fn footprint_overlap_uses_allocation_and_exact_byte_intervals() {
        let first = PhysicalFootprint::new(vec![span(2, 0, 4), span(2, 16, 4)]).unwrap();
        let overlap = PhysicalFootprint::new(vec![span(2, 18, 8)]).unwrap();
        let gap = PhysicalFootprint::new(vec![span(2, 4, 12)]).unwrap();
        let other_allocation = PhysicalFootprint::new(vec![span(3, 18, 8)]).unwrap();
        assert!(first.overlaps(&overlap));
        assert!(!first.overlaps(&gap));
        assert!(!first.overlaps(&other_allocation));
    }

    #[test]
    fn analysis_compaction_unions_fragments_without_losing_exact_bytes() {
        let operation = operation(0b101);
        let make_batch = |relative_offset| {
            PhysicalAccessBatch::resolve_single_span(operation.clone(), descriptor(4), |lane| {
                Ok::<_, std::convert::Infallible>(span(9, lane.lane() * 32 + relative_offset, 4))
            })
            .unwrap()
        };
        let fragments = [make_batch(0), make_batch(4), make_batch(2)];

        let compact = coalesce_physical_access_batches(fragments.iter()).unwrap();

        assert_eq!(compact.len(), 1);
        assert_eq!(compact[0].operation(), &operation);
        assert_eq!(compact[0].descriptor().width().bytes(), 8);
        assert_eq!(
            compact[0].lane(0).unwrap().footprint().spans(),
            &[span(9, 0, 8)]
        );
        assert_eq!(
            compact[0].lane(2).unwrap().footprint().spans(),
            &[span(9, 64, 8)]
        );
        assert_eq!(
            fragments.len(),
            3,
            "detailed evidence remains element-granular"
        );
    }

    #[test]
    fn analysis_compaction_keeps_distinct_access_identities_separate() {
        let operation = operation(1);
        let base =
            PhysicalAccessBatch::resolve_single_span(operation.clone(), descriptor(4), |_| {
                Ok::<_, std::convert::Infallible>(span(3, 0, 4))
            })
            .unwrap();
        let first = base.clone().with_logical_buffer("first");
        let second = base.with_logical_buffer("second");
        let write = PhysicalAccessBatch::resolve_single_span(
            operation,
            PhysicalAccessDescriptor::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                4,
            )
            .unwrap(),
            |_| Ok::<_, std::convert::Infallible>(span(3, 0, 4)),
        )
        .unwrap();

        let compact = coalesce_physical_access_batches([&first, &second, &write]).unwrap();

        assert_eq!(compact.len(), 3);
        assert_eq!(compact[0], first);
        assert_eq!(compact[1], second);
        assert_eq!(compact[2], write);
    }

    #[test]
    fn batch_resolves_every_active_lane_in_stable_order_with_full_provenance() {
        let batch = PhysicalAccessBatch::resolve(operation(0b10101), descriptor(4), |lane| {
            Ok::<_, std::convert::Infallible>(vec![span(9, lane.lane() * 4, 4)])
        })
        .unwrap();
        assert_eq!(
            batch
                .lanes()
                .iter()
                .map(|access| access.provenance().lane())
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );
        let provenance = batch.lane(2).unwrap().provenance();
        assert_eq!(provenance.operation(), batch.operation().id());
        assert_eq!(provenance.operation().source_op_id(), StaticOpId::new(19));
        assert_eq!(
            provenance.operation().loop_frames(),
            &[LoopFrame::new(StaticOpId::new(23), 29)]
        );
        assert!(batch.lane(1).is_none());
    }

    #[test]
    fn late_lane_resolution_failure_returns_no_partial_batch() {
        let result = PhysicalAccessBatch::resolve(operation(0b1111), descriptor(4), |lane| {
            if lane.lane() == 3 {
                Err("out of bounds")
            } else {
                Ok(vec![span(4, lane.lane() * 4, 4)])
            }
        });
        assert!(matches!(
            result,
            Err(PhysicalAccessBatchError::LaneResolution {
                ref provenance,
                source: "out of bounds",
            }) if provenance.lane() == 3
        ));
    }

    #[test]
    fn invalid_lane_footprint_returns_no_partial_batch() {
        let result = PhysicalAccessBatch::resolve(operation(0b11), descriptor(4), |lane| {
            if lane.lane() == 1 {
                Ok::<_, std::convert::Infallible>(vec![span(4, 4, 3), span(4, 6, 1)])
            } else {
                Ok(vec![span(4, 0, 4)])
            }
        });
        assert!(matches!(
            result,
            Err(PhysicalAccessBatchError::InvalidFootprint {
                ref provenance,
                source: PhysicalFootprintError::OverlappingSpans { .. },
            }) if provenance.lane() == 1
        ));
    }

    #[test]
    fn lane_width_mismatch_returns_no_partial_batch() {
        let result = PhysicalAccessBatch::resolve(operation(0b11), descriptor(4), |lane| {
            let byte_len = if lane.lane() == 1 { 8 } else { 4 };
            Ok::<_, std::convert::Infallible>(vec![span(4, lane.lane() * 8, byte_len)])
        });
        assert!(matches!(
            result,
            Err(PhysicalAccessBatchError::WidthMismatch {
                ref provenance,
                expected_byte_len: 4,
                actual_byte_len: 8,
            }) if provenance.lane() == 1
        ));
    }
}
