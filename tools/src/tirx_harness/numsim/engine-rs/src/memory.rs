use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};
use std::task::{Context, Poll, Waker};
use thread_local::ThreadLocal;

#[cfg(feature = "python")]
use numsim_host_buffer::{HostByteBuffer, HostByteRegion};

use crate::numpy_backend::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32,
};
use crate::scalar::{cuda_f32_max, cuda_f32_min, F32RoundingMode};
use crate::{
    profile_count_by, EngineError, ProfileKind, ProfileTimer, RuntimeScalar, WarpMask, WarpValue,
    WARP_SIZE,
};

static NEXT_MEMORY_ARENA_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_SEMANTIC_PROGRESS_WAITER_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_MEMORY_THREAD_TOKEN: AtomicU64 = AtomicU64::new(1);

const MEMORY_STRIPE_BYTES: usize = 4096;
static ALL_VALID_MEMORY_STRIPE: [u8; MEMORY_STRIPE_BYTES] = [1; MEMORY_STRIPE_BYTES];

trait ImmutableByteStorage: Send + Sync {
    fn as_bytes(&self) -> &[u8];
}

impl<T> ImmutableByteStorage for T
where
    T: AsRef<[u8]> + Send + Sync,
{
    fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Clone)]
struct SharedInitialBytes(Arc<dyn ImmutableByteStorage>);

impl SharedInitialBytes {
    fn new(bytes: impl AsRef<[u8]> + Send + Sync + 'static) -> Self {
        Self(Arc::new(bytes))
    }

    fn as_slice(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

enum StripeBytes {
    Owned(Box<[u8]>),
    Initial {
        source: SharedInitialBytes,
        start: usize,
        end: usize,
    },
    #[cfg(feature = "python")]
    Host(HostByteRegion),
    AllValid(usize),
}

impl StripeBytes {
    fn initial(source: SharedInitialBytes, start: usize, end: usize) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= source.as_slice().len());
        Self::Initial { source, start, end }
    }

    #[inline(always)]
    fn is_all_valid(&self) -> bool {
        matches!(self, Self::AllValid(_))
    }

    #[inline(always)]
    fn mark_valid(&mut self, start: usize, end: usize) {
        if !self.is_all_valid() {
            self[start..end].fill(1);
        }
    }
}

impl Deref for StripeBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Initial { source, start, end } => &source.as_slice()[*start..*end],
            #[cfg(feature = "python")]
            Self::Host(region) => region,
            Self::AllValid(len) => &ALL_VALID_MEMORY_STRIPE[..*len],
        }
    }
}

impl DerefMut for StripeBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        #[cfg(feature = "python")]
        if let Self::Host(region) = self {
            return region;
        }
        if !matches!(self, Self::Owned(_)) {
            *self = Self::Owned(self.deref().to_vec().into_boxed_slice());
        }
        let Self::Owned(bytes) = self else {
            unreachable!("copy-on-write stripe bytes were materialized")
        };
        bytes
    }
}

impl From<Box<[u8]>> for StripeBytes {
    fn from(bytes: Box<[u8]>) -> Self {
        Self::Owned(bytes)
    }
}

thread_local! {
    static MEMORY_THREAD_TOKEN: u64 = NEXT_MEMORY_THREAD_TOKEN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| next.checked_add(1))
        .expect("NumSim memory thread token space exhausted");
}

/// Source identity for one generated owner-private scalar read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ReadSource {
    pub kernel_index: usize,
    pub source_op_id: u64,
    pub global_warp_id: usize,
}

/// One in-bounds read that observed at least one uninitialized byte.
///
/// Numerical execution may materialize such bytes as zero and report this as
/// a non-fatal review finding.  Direct memory users retain the fail-closed
/// behavior unless they explicitly select the reviewing policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UninitializedReadReview {
    source: Option<ReadSource>,
    allocation: AllocationId,
    byte_offset: usize,
    byte_len: usize,
    first_uninitialized_byte: usize,
}

impl UninitializedReadReview {
    pub(crate) const fn source(self) -> Option<ReadSource> {
        self.source
    }

    pub const fn allocation(self) -> AllocationId {
        self.allocation
    }

    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    pub const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub const fn first_uninitialized_byte(self) -> usize {
        self.first_uninitialized_byte
    }
}

impl fmt::Display for UninitializedReadReview {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "read [{}, {}+{}) from {} includes uninitialized byte {}",
            self.byte_offset,
            self.byte_offset,
            self.byte_len,
            self.allocation,
            self.first_uninitialized_byte,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum UninitializedReadPolicy {
    #[default]
    Error,
    ReviewAndZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(u64);

impl AllocationId {
    pub(crate) const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "allocation#{}", self.0)
    }
}

/// A byte-addressed view into one physical allocation.
///
/// Views contain no data. Multiple views with the same allocation ID therefore
/// preserve aliasing even when their byte ranges overlap with different
/// offsets or element interpretations.
#[derive(Clone)]
pub struct BufferView {
    arena_id: u64,
    allocation: AllocationId,
    allocation_ref: Arc<Allocation>,
    byte_offset: usize,
    byte_len: usize,
}

impl fmt::Debug for BufferView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferView")
            .field("arena_id", &self.arena_id)
            .field("allocation", &self.allocation)
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

impl PartialEq for BufferView {
    fn eq(&self, other: &Self) -> bool {
        self.arena_id == other.arena_id
            && self.allocation == other.allocation
            && self.byte_offset == other.byte_offset
            && self.byte_len == other.byte_len
    }
}

impl Eq for BufferView {}

impl Hash for BufferView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arena_id.hash(state);
        self.allocation.hash(state);
        self.byte_offset.hash(state);
        self.byte_len.hash(state);
    }
}

/// A primitive scalar that can participate in a linearizable global-memory RMW.
///
/// The engine intentionally keeps this trait limited to plain fixed-width
/// values. Generated code with a richer runtime scalar abstraction can use
/// [`GlobalMemory::atomic_update_bytes`] directly.
pub trait AtomicMemoryScalar: Copy {
    const BYTE_LEN: usize;

    fn decode_le(bytes: &[u8]) -> Self;

    fn encode_le(self) -> Vec<u8>;
}

macro_rules! impl_atomic_memory_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl AtomicMemoryScalar for $scalar {
                const BYTE_LEN: usize = size_of::<Self>();

                fn decode_le(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(
                        bytes
                            .try_into()
                            .expect("atomic scalar byte width was validated"),
                    )
                }

                fn encode_le(self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }
            }
        )+
    };
}

impl_atomic_memory_scalar!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl BufferView {
    /// The invocation's address, not the simulator's copied storage address.
    pub(crate) fn observed_allocation_address(&self) -> Option<u64> {
        self.allocation_ref.observed_address.get().copied()
    }

    pub(crate) fn bind_observed_allocation_address(&self, address: u64) -> Result<(), EngineError> {
        self.allocation_ref
            .observed_address
            .set(address)
            .map_err(|_| EngineError::message("allocation address is already bound"))
    }

    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub const fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    pub(crate) fn full_allocation_view(&self) -> Self {
        Self {
            arena_id: self.arena_id,
            allocation: self.allocation,
            allocation_ref: self.allocation_ref.clone(),
            byte_offset: 0,
            byte_len: self.allocation_ref.byte_len,
        }
    }
}

/// Launch-wide generation for concrete simulator state changes.
///
/// Every physical address space in one launch shares this hub.  A write only
/// advances the generation when it changes a byte or initializes/invalidates a
/// validity bit. Native
/// while-loop checkpoints use the generation to distinguish finite local
/// induction from a stuttering poll without inspecting TIR expression shapes.
#[derive(Clone)]
pub(crate) struct SemanticProgress {
    inner: Arc<SemanticProgressInner>,
}

struct SemanticProgressInner {
    enabled: AtomicBool,
    generation: AtomicU64,
    /// Mirror of `waiters.len()`, published only under the `waiters` mutex.
    waiter_count: AtomicUsize,
    waiters: Mutex<BTreeMap<u64, Waker>>,
}

impl Default for SemanticProgress {
    fn default() -> Self {
        Self {
            inner: Arc::new(SemanticProgressInner {
                enabled: AtomicBool::new(true),
                generation: AtomicU64::new(0),
                waiter_count: AtomicUsize::new(0),
                waiters: Mutex::new(BTreeMap::new()),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SemanticProgressSnapshot(u64);

pub(crate) struct SemanticProgressWatch {
    progress: SemanticProgress,
    observed: SemanticProgressSnapshot,
    waiter_id: Option<u64>,
    completed: bool,
}

impl SemanticProgress {
    pub(crate) fn disabled() -> Self {
        Self {
            inner: Arc::new(SemanticProgressInner {
                enabled: AtomicBool::new(false),
                generation: AtomicU64::new(0),
                waiter_count: AtomicUsize::new(0),
                waiters: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub(crate) fn snapshot(&self) -> SemanticProgressSnapshot {
        // SeqCst: half of the handshake documented on `record_change`.
        SemanticProgressSnapshot(self.inner.generation.load(Ordering::SeqCst))
    }

    fn observes_changes(&self) -> bool {
        self.inner.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn enable(&self) {
        self.inner.enabled.store(true, Ordering::Release);
    }

    pub(crate) fn watch(&self, observed: SemanticProgressSnapshot) -> SemanticProgressWatch {
        SemanticProgressWatch {
            progress: self.clone(),
            observed,
            waiter_id: None,
            completed: false,
        }
    }

    pub(crate) fn record_change(&self) {
        if !self.observes_changes() {
            return;
        }
        // Handshake with `SemanticProgressWatch::poll`: publish the generation,
        // then read `waiter_count`; the waiter publishes `waiter_count`, then
        // reads the generation.  SeqCst on all four makes at least one side see
        // the other, so a mid-registration waiter is never left parked stale.
        // Weakening any of them reopens the lost wake.
        self.inner
            .generation
            .fetch_update(Ordering::SeqCst, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("NumSim semantic-progress generation exhausted");
        if self.inner.waiter_count.load(Ordering::SeqCst) == 0 {
            return;
        }
        let waiters = {
            let mut waiters = lock_mutex(&self.inner.waiters);
            let waiters = std::mem::take(&mut *waiters)
                .into_values()
                .collect::<Vec<_>>();
            self.inner.waiter_count.store(0, Ordering::SeqCst);
            waiters
        };
        for waiter in waiters {
            waiter.wake();
        }
    }
}

impl Future for SemanticProgressWatch {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.completed {
            return Poll::Ready(());
        }
        if this.progress.snapshot() != this.observed {
            this.remove_waiter();
            this.completed = true;
            return Poll::Ready(());
        }

        let mut waiters = lock_mutex(&this.progress.inner.waiters);
        let poll_recheck_waker = crate::scheduling::poll_recheck_waker(context.waker());
        // Register BEFORE the staleness check; see `record_change`.
        let waiter_id = match this.waiter_id {
            Some(waiter_id) => {
                waiters.insert(waiter_id, poll_recheck_waker);
                waiter_id
            }
            None => {
                let waiter_id = NEXT_SEMANTIC_PROGRESS_WAITER_ID
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                        next.checked_add(1)
                    })
                    .expect("NumSim semantic-progress waiter ID space exhausted");
                waiters.insert(waiter_id, poll_recheck_waker);
                this.waiter_id = Some(waiter_id);
                waiter_id
            }
        };
        this.progress
            .inner
            .waiter_count
            .store(waiters.len(), Ordering::SeqCst);
        if this.progress.snapshot() != this.observed {
            this.waiter_id = None;
            waiters.remove(&waiter_id);
            this.progress
                .inner
                .waiter_count
                .store(waiters.len(), Ordering::SeqCst);
            this.completed = true;
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

impl SemanticProgressWatch {
    fn remove_waiter(&mut self) {
        let Some(waiter_id) = self.waiter_id.take() else {
            return;
        };
        let mut waiters = lock_mutex(&self.progress.inner.waiters);
        if waiters.remove(&waiter_id).is_some() {
            // Republish the length rather than decrementing: `record_change`'s
            // `mem::take` may have already zeroed the count.
            self.progress
                .inner
                .waiter_count
                .store(waiters.len(), Ordering::SeqCst);
        }
    }
}

impl Drop for SemanticProgressWatch {
    fn drop(&mut self) {
        if !self.completed {
            self.remove_waiter();
        }
    }
}

/// Shared, Rust-owned physical global memory.
///
/// Clones share the same allocations and initialization-validity state.
#[derive(Clone, Default)]
pub struct GlobalMemory {
    inner: Arc<MemoryState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferredGlobalReduction {
    AddU32,
    AddI32,
    AddU64,
    AddF32,
    AddF32Ftz,
    AddF64,
    AddF16,
    AddBf16,
    MinU32,
    MinI32,
    MinU64,
    MinI64,
    MinF16,
    MinBf16,
    MaxU32,
    MaxI32,
    MaxU64,
    MaxI64,
    MaxF16,
    MaxBf16,
    IncU32,
    DecU32,
    AndB32,
    AndB64,
    OrB32,
    OrB64,
    XorB32,
    XorB64,
}

impl DeferredGlobalReduction {
    pub(crate) fn byte_len(self) -> usize {
        match self {
            Self::AddF16
            | Self::AddBf16
            | Self::MinF16
            | Self::MinBf16
            | Self::MaxF16
            | Self::MaxBf16 => 2,
            Self::AddU32
            | Self::AddI32
            | Self::AddF32
            | Self::AddF32Ftz
            | Self::MinU32
            | Self::MinI32
            | Self::MaxU32
            | Self::MaxI32
            | Self::IncU32
            | Self::DecU32
            | Self::AndB32
            | Self::OrB32
            | Self::XorB32 => 4,
            Self::AddU64
            | Self::AddF64
            | Self::MinU64
            | Self::MinI64
            | Self::MaxU64
            | Self::MaxI64
            | Self::AndB64
            | Self::OrB64
            | Self::XorB64 => 8,
        }
    }

    fn apply(self, current: &[u8], source: &[u8]) -> Vec<u8> {
        debug_assert_eq!(current.len(), self.byte_len());
        debug_assert_eq!(source.len(), self.byte_len());
        match self {
            Self::AddU32 => u32::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddI32 => i32::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddU64 => u64::from_le_bytes(current.try_into().unwrap())
                .wrapping_add(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::AddF32 | Self::AddF32Ftz => {
                let add = if matches!(self, Self::AddF32Ftz) {
                    crate::scalar::add_f32_ftz
                } else {
                    crate::scalar::add_f32
                };
                add(
                    f32::from_le_bytes(current.try_into().unwrap()),
                    f32::from_le_bytes(source.try_into().unwrap()),
                    F32RoundingMode::Nearest,
                )
                .to_le_bytes()
                .to_vec()
            }
            Self::AddF64 => crate::scalar::cuda_f64_add(
                f64::from_le_bytes(current.try_into().unwrap()),
                f64::from_le_bytes(source.try_into().unwrap()),
            )
            .to_le_bytes()
            .to_vec(),
            Self::AddF16 => f32_to_fp16_bits(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap()))
                    + fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            )
            .to_le_bytes()
            .to_vec(),
            Self::AddBf16 => f32_to_bf16_bits(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap()))
                    + bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            )
            .to_le_bytes()
            .to_vec(),
            Self::MinU32 => u32::from_le_bytes(current.try_into().unwrap())
                .min(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinI32 => i32::from_le_bytes(current.try_into().unwrap())
                .min(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinU64 => u64::from_le_bytes(current.try_into().unwrap())
                .min(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinI64 => i64::from_le_bytes(current.try_into().unwrap())
                .min(i64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MinF16 => f32_to_fp16_bits(cuda_f32_min(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MinBf16 => f32_to_bf16_bits(cuda_f32_min(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MaxU32 => u32::from_le_bytes(current.try_into().unwrap())
                .max(u32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxI32 => i32::from_le_bytes(current.try_into().unwrap())
                .max(i32::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxU64 => u64::from_le_bytes(current.try_into().unwrap())
                .max(u64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxI64 => i64::from_le_bytes(current.try_into().unwrap())
                .max(i64::from_le_bytes(source.try_into().unwrap()))
                .to_le_bytes()
                .to_vec(),
            Self::MaxF16 => f32_to_fp16_bits(cuda_f32_max(
                fp16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                fp16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::MaxBf16 => f32_to_bf16_bits(cuda_f32_max(
                bf16_bits_to_f32(u16::from_le_bytes(current.try_into().unwrap())),
                bf16_bits_to_f32(u16::from_le_bytes(source.try_into().unwrap())),
            ))
            .to_le_bytes()
            .to_vec(),
            Self::IncU32 => {
                let old = u32::from_le_bytes(current.try_into().unwrap());
                let limit = u32::from_le_bytes(source.try_into().unwrap());
                (if old >= limit { 0 } else { old + 1 })
                    .to_le_bytes()
                    .to_vec()
            }
            Self::DecU32 => {
                let old = u32::from_le_bytes(current.try_into().unwrap());
                let limit = u32::from_le_bytes(source.try_into().unwrap());
                (if old == 0 || old > limit {
                    limit
                } else {
                    old - 1
                })
                .to_le_bytes()
                .to_vec()
            }
            Self::AndB32 => (u32::from_le_bytes(current.try_into().unwrap())
                & u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::AndB64 => (u64::from_le_bytes(current.try_into().unwrap())
                & u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::OrB32 => (u32::from_le_bytes(current.try_into().unwrap())
                | u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::OrB64 => (u64::from_le_bytes(current.try_into().unwrap())
                | u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::XorB32 => (u32::from_le_bytes(current.try_into().unwrap())
                ^ u32::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
            Self::XorB64 => (u64::from_le_bytes(current.try_into().unwrap())
                ^ u64::from_le_bytes(source.try_into().unwrap()))
            .to_le_bytes()
            .to_vec(),
        }
    }
}

enum DeferredGlobalWritePayload {
    Replace(Vec<u8>),
    Masked {
        bytes: Vec<u8>,
        masks: Vec<u8>,
    },
    Reduction {
        bytes: Vec<u8>,
        operation: DeferredGlobalReduction,
    },
}

/// One global-memory write captured at asynchronous issue time and published
/// only when the owning completion protocol reaches full completion.
pub struct DeferredGlobalWrite {
    memory: GlobalMemory,
    view: BufferView,
    byte_offset: usize,
    payload: DeferredGlobalWritePayload,
}

impl fmt::Debug for DeferredGlobalWrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, byte_len) = match &self.payload {
            DeferredGlobalWritePayload::Replace(bytes) => ("replace", bytes.len()),
            DeferredGlobalWritePayload::Masked { bytes, .. } => ("masked", bytes.len()),
            DeferredGlobalWritePayload::Reduction { bytes, .. } => ("reduction", bytes.len()),
        };
        f.debug_struct("DeferredGlobalWrite")
            .field("view", &self.view)
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &byte_len)
            .field("kind", &kind)
            .finish()
    }
}

impl DeferredGlobalWrite {
    pub fn publish(self) -> Result<(), MemoryError> {
        publish_deferred_global_writes(std::iter::once(&self))
    }
}

/// Publish one deferred-write batch as a single global-memory transaction.
///
/// Every range and write epoch is validated before any byte changes, so a
/// failed completion keeps both memory and the owning async-group queue
/// unchanged, including groups that target several global allocations.
pub fn publish_deferred_global_writes<'a>(
    writes: impl IntoIterator<Item = &'a DeferredGlobalWrite>,
) -> Result<(), MemoryError> {
    let (first, resolved, ranges) = {
        let _profile = ProfileTimer::new(ProfileKind::AsyncPublishResolve);
        let writes = writes.into_iter().collect::<Vec<_>>();
        let Some(&first) = writes.first() else {
            return Ok(());
        };
        let arena_id = first.view.arena_id;
        let mut resolved = Vec::with_capacity(writes.len());
        let mut ranges = Vec::with_capacity(writes.len());
        let mut byte_count = 0_u64;
        for &write in &writes {
            if write.view.arena_id != arena_id {
                return Err(MemoryError::UnknownAllocation {
                    allocation: write.view.allocation,
                });
            }
            let allocation = write.memory.allocation_for_view(&write.view)?;
            if allocation.is_owner_private() {
                return Err(MemoryError::OwnerPrivateWriteWaitUnsupported {
                    allocation: write.view.allocation,
                });
            }
            let byte_len = match &write.payload {
                DeferredGlobalWritePayload::Replace(bytes) => bytes.len(),
                DeferredGlobalWritePayload::Masked { bytes, .. } => bytes.len(),
                DeferredGlobalWritePayload::Reduction { bytes, .. } => bytes.len(),
            };
            byte_count = byte_count.saturating_add(byte_len as u64);
            let absolute = resolve_access(
                &write.view,
                allocation.byte_len(),
                write.byte_offset,
                byte_len,
            )?;
            if byte_len != 0 {
                ranges.push(MemoryWriteRange {
                    arena_id,
                    allocation: write.view.allocation,
                    allocation_ref: write.view.allocation_ref.clone(),
                    absolute_byte_offset: absolute,
                    byte_len,
                });
            }
            resolved.push((write, absolute, byte_len));
        }
        profile_count_by(ProfileKind::AsyncPublishWriteCount, writes.len() as u64);
        profile_count_by(ProfileKind::AsyncPublishByteCount, byte_count);
        (first, resolved, ranges)
    };
    if ranges.is_empty() {
        return Ok(());
    }
    let targets = {
        let _profile = ProfileTimer::new(ProfileKind::AsyncPublishPlan);
        let ranges = coalesce_memory_write_ranges(ranges);
        stripe_targets_for_write_ranges(&ranges)
    };

    let (guards, mut data) = {
        let _profile = ProfileTimer::new(ProfileKind::AsyncPublishLock);
        let guards = lock_stripe_targets(&targets);
        let data = begin_stripe_writes(&targets)?;
        for (write, absolute, byte_len) in &resolved {
            if let DeferredGlobalWritePayload::Masked { masks, .. } = &write.payload {
                // Preflight only selected bytes, before publishing any write
                // in the batch. A zero mask neither writes nor initializes.
                let mut relative = 0;
                for run in masks.split_inclusive(|mask| *mask == 0) {
                    let written = run.len() - usize::from(run.last() == Some(&0));
                    validate_readonly_write_locked(
                        &targets,
                        &data,
                        write.view.allocation,
                        *absolute + relative,
                        written,
                    )?;
                    relative += run.len();
                }
            } else {
                validate_readonly_write_locked(
                    &targets,
                    &data,
                    write.view.allocation,
                    *absolute,
                    *byte_len,
                )?;
            }
            if matches!(write.payload, DeferredGlobalWritePayload::Reduction { .. }) {
                validate_initialized_locked(
                    write.view.allocation,
                    &targets,
                    &data,
                    *absolute,
                    *byte_len,
                )?;
            }
        }
        (guards, data)
    };
    let track_semantic_progress = first.memory.inner.semantic_progress.observes_changes();
    let mut semantic_changed = false;
    {
        let _profile = ProfileTimer::new(ProfileKind::AsyncPublishStore);
        for (write, absolute, byte_len) in resolved {
            if byte_len == 0 {
                continue;
            }
            match &write.payload {
                DeferredGlobalWritePayload::Replace(bytes) => {
                    if track_semantic_progress {
                        semantic_changed |=
                            bytes.iter().copied().enumerate().any(|(relative, byte)| {
                                let (old, valid) = locked_byte(
                                    &targets,
                                    &data,
                                    write.view.allocation,
                                    absolute + relative,
                                );
                                old != byte || !valid
                            });
                    }
                    store_initialized_locked(
                        &targets,
                        &mut data,
                        write.view.allocation,
                        absolute,
                        bytes,
                    )?;
                }
                DeferredGlobalWritePayload::Masked { bytes, masks } => {
                    for (relative, (byte, mask)) in
                        bytes.iter().copied().zip(masks.iter().copied()).enumerate()
                    {
                        if mask == 0 {
                            continue;
                        }
                        let index = absolute + relative;
                        let (old, valid) =
                            locked_byte(&targets, &data, write.view.allocation, index);
                        let new = (old & !mask) | (byte & mask);
                        semantic_changed |= track_semantic_progress && (new != old || !valid);
                        store_locked_byte(&targets, &mut data, write.view.allocation, index, new)?;
                    }
                }
                DeferredGlobalWritePayload::Reduction { bytes, operation } => {
                    let mut current = vec![0; byte_len];
                    read_locked_bytes(
                        &targets,
                        &data,
                        write.view.allocation,
                        absolute,
                        &mut current,
                    );
                    let reduced = operation.apply(&current, bytes);
                    debug_assert_eq!(reduced.len(), byte_len);
                    for (relative, byte) in reduced.into_iter().enumerate() {
                        let index = absolute + relative;
                        let (old, valid) =
                            locked_byte(&targets, &data, write.view.allocation, index);
                        semantic_changed |= track_semantic_progress && (old != byte || !valid);
                        store_locked_byte(&targets, &mut data, write.view.allocation, index, byte)?;
                    }
                }
            }
        }
    }
    {
        let _profile = ProfileTimer::new(ProfileKind::AsyncPublishNotify);
        drop(data);
        drop(guards);
    }
    if semantic_changed {
        first.memory.inner.semantic_progress.record_change();
    }
    Ok(())
}

impl GlobalMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create numerical-execution memory that zero-fills in-bounds
    /// uninitialized reads and records them for review.
    pub fn new_reviewing_uninitialized_reads() -> Self {
        Self::new().with_uninitialized_read_policy(UninitializedReadPolicy::ReviewAndZero)
    }

    /// Called at kernel boundaries, before any warp starts. Host input binding
    /// precedes enabling history; a following kernel gets fresh history.
    pub fn set_readonly_proxy_tracking(&self, enabled: bool) -> Result<(), MemoryError> {
        if enabled && self.inner.mode != MemoryMode::Shared {
            return Err(MemoryError::ReadonlyProxyTrackingUnavailable);
        }
        if !self
            .inner
            .readonly_proxy_tracking
            .swap(enabled, Ordering::Relaxed)
            && !enabled
        {
            return Ok(());
        }
        for allocation in read_rwlock(&self.inner.allocations).values() {
            allocation.reset_readonly_proxy_history(enabled);
        }
        Ok(())
    }

    pub(crate) fn observe_readonly_proxy(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        let shared = allocation
            .shared()
            .ok_or(MemoryError::ReadonlyProxyTrackingUnavailable)?;
        if shared.read_only.is_some() || byte_len == 0 {
            return Ok(());
        }
        let end = absolute + byte_len;
        for stripe_index in absolute / MEMORY_STRIPE_BYTES..=(end - 1) / MEMORY_STRIPE_BYTES {
            let stripe = &shared.stripes[stripe_index];
            let mut data = stripe.write_data(view.allocation)?;
            let history = data
                .readonly_proxy
                .as_mut()
                .ok_or(MemoryError::ReadonlyProxyTrackingUnavailable)?;
            let start = absolute.max(stripe.byte_start) - stripe.byte_start;
            let stop = end.min(stripe.byte_start + stripe.byte_len) - stripe.byte_start;
            if history.written.any_valid_in(start, stop) {
                return Err(MemoryError::ReadonlyProxyWriteConflict {
                    allocation: view.allocation,
                    byte_offset: absolute,
                    byte_len,
                });
            }
            history.observed.set_range(start, stop, true);
        }
        Ok(())
    }

    /// Create an arena whose allocations are bound to one CPU thread on first
    /// data access.
    ///
    /// This is used for GPU-private address spaces after the executor pins a
    /// complete cluster to one worker. Runtime owner checks make the unsafe
    /// backing inaccessible from a second live thread instead of relying on an
    /// unchecked synchronization assumption.
    pub fn new_owner_private() -> Self {
        Self {
            inner: Arc::new(MemoryState {
                mode: MemoryMode::OwnerPrivate,
                ..MemoryState::default()
            }),
        }
    }

    pub(crate) fn new_queued_owner_private() -> Self {
        Self {
            inner: Arc::new(MemoryState {
                mode: MemoryMode::QueuedOwnerPrivate,
                ..MemoryState::default()
            }),
        }
    }

    pub(crate) fn new_owner_private_with_semantic_progress(progress: SemanticProgress) -> Self {
        Self {
            inner: Arc::new(MemoryState {
                mode: MemoryMode::OwnerPrivate,
                semantic_progress: progress,
                ..MemoryState::default()
            }),
        }
    }

    pub(crate) fn new_queued_owner_private_with_semantic_progress(
        progress: SemanticProgress,
    ) -> Self {
        Self {
            inner: Arc::new(MemoryState {
                mode: MemoryMode::QueuedOwnerPrivate,
                semantic_progress: progress,
                ..MemoryState::default()
            }),
        }
    }

    pub(crate) fn with_uninitialized_read_policy(
        mut self,
        uninitialized_read_policy: UninitializedReadPolicy,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("read policy can only be selected on fresh memory")
            .uninitialized_read_policy = uninitialized_read_policy;
        self
    }

    pub(crate) fn uninitialized_read_policy(&self) -> UninitializedReadPolicy {
        self.inner.uninitialized_read_policy
    }

    fn records_uninitialized_read_reviews(&self) -> bool {
        matches!(
            self.inner.uninitialized_read_policy,
            UninitializedReadPolicy::ReviewAndZero
        )
    }

    fn record_uninitialized_read_review(&self, review: UninitializedReadReview) {
        debug_assert!(self.records_uninitialized_read_reviews());
        lock_mutex(&self.inner.uninitialized_read_reviews).insert(review);
    }

    pub(crate) fn take_uninitialized_read_reviews(&self) -> Vec<UninitializedReadReview> {
        std::mem::take(&mut *lock_mutex(&self.inner.uninitialized_read_reviews))
            .into_iter()
            .collect()
    }

    pub(crate) fn semantic_progress(&self) -> SemanticProgress {
        self.inner.semantic_progress.clone()
    }

    pub(crate) fn enable_semantic_progress(&self) {
        self.inner.semantic_progress.enable();
    }

    pub fn allocate_from_bytes(
        &self,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<AllocationId, MemoryError> {
        let bytes = bytes.into();
        let valid = vec![true; bytes.len()];
        self.allocate(bytes, valid, false)
    }

    pub fn allocate_from_bytes_with_validity(
        &self,
        bytes: impl Into<Vec<u8>>,
        validity: impl Into<Vec<u8>>,
    ) -> Result<AllocationId, MemoryError> {
        let bytes: Vec<u8> = bytes.into();
        let validity: Vec<u8> = validity.into();
        if bytes.len() != validity.len() {
            return Err(MemoryError::InitializationLengthMismatch {
                byte_len: bytes.len(),
                validity_len: validity.len(),
            });
        }
        let mut valid = Vec::with_capacity(validity.len());
        for (index, value) in validity.into_iter().enumerate() {
            match value {
                0 => valid.push(false),
                1 => valid.push(true),
                _ => {
                    return Err(MemoryError::InvalidValidityByte { index, value });
                }
            }
        }
        self.allocate(bytes, valid, false)
    }

    pub(crate) fn allocate_from_bytes_all_valid(
        &self,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<AllocationId, MemoryError> {
        self.allocate_from_immutable_bytes_all_valid(bytes.into())
    }

    pub(crate) fn allocate_from_immutable_bytes_all_valid(
        &self,
        bytes: impl AsRef<[u8]> + Send + Sync + 'static,
    ) -> Result<AllocationId, MemoryError> {
        let bytes = SharedInitialBytes::new(bytes);
        self.allocate_all_valid(bytes, false)
    }

    #[cfg(feature = "python")]
    pub(crate) fn allocate_from_host_bytes_all_valid(
        &self,
        bytes: HostByteBuffer,
    ) -> Result<AllocationId, MemoryError> {
        debug_assert_eq!(self.inner.mode, MemoryMode::Shared);
        let allocation = self
            .inner
            .next_allocation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(AllocationId)
            .map_err(|_| MemoryError::AllocationIdExhausted)?;
        let allocation_ref = Arc::new(Allocation::new_host_all_valid(self.inner.mode, bytes));
        self.insert_allocation(allocation, allocation_ref);
        Ok(allocation)
    }

    pub(crate) fn allocate_read_only_from_bytes_with_validity(
        &self,
        bytes: impl Into<Vec<u8>>,
        validity: impl Into<Vec<u8>>,
    ) -> Result<AllocationId, MemoryError> {
        let bytes: Vec<u8> = bytes.into();
        let validity: Vec<u8> = validity.into();
        if bytes.len() != validity.len() {
            return Err(MemoryError::InitializationLengthMismatch {
                byte_len: bytes.len(),
                validity_len: validity.len(),
            });
        }
        let mut valid = Vec::with_capacity(validity.len());
        for (index, value) in validity.into_iter().enumerate() {
            match value {
                0 => valid.push(false),
                1 => valid.push(true),
                _ => {
                    return Err(MemoryError::InvalidValidityByte { index, value });
                }
            }
        }
        self.allocate(bytes, valid, true)
    }

    pub fn allocate_zeroed(&self, byte_len: usize) -> Result<AllocationId, MemoryError> {
        self.allocate(vec![0; byte_len], vec![true; byte_len], false)
    }

    pub fn allocate_uninitialized(&self, byte_len: usize) -> Result<AllocationId, MemoryError> {
        self.allocate(vec![0; byte_len], vec![false; byte_len], false)
    }

    pub fn allocation_len(&self, allocation: AllocationId) -> Result<usize, MemoryError> {
        Ok(self.allocation(allocation)?.byte_len())
    }

    pub(crate) fn allocation_is_write_through(
        &self,
        allocation: AllocationId,
    ) -> Result<bool, MemoryError> {
        Ok(self.allocation(allocation)?.is_write_through())
    }

    /// Return the physical byte image for host output transfer.
    ///
    /// This deliberately bypasses simulated-load validity checks. Generated
    /// kernels must use `read_bytes`; only the artifact boundary should use a
    /// raw snapshot to preserve invalid gaps in aliased host allocations.
    pub fn snapshot_allocation_bytes(
        &self,
        allocation: AllocationId,
    ) -> Result<Vec<u8>, MemoryError> {
        let allocation_id = allocation;
        let allocation = self.allocation(allocation)?;
        if allocation.is_owner_private() {
            return allocation.with_private(allocation_id, |data| data.bytes.clone());
        }
        Ok(stable_snapshot(
            allocation
                .shared()
                .expect("shared allocation has shared backing"),
            0,
            allocation.byte_len(),
            SnapshotValidity::Ignore,
        )
        .bytes)
    }

    /// Read a word out of the bytes an allocation was launched with.
    ///
    /// This is deliberately not a view of current memory: it answers what the
    /// address held before the kernel ran, which is the one thing a stripe
    /// stops being able to say once it has been written. `None` means the
    /// launch bytes were not retained -- a host-mapped allocation -- or the
    /// span does not lie inside them, and a caller must then treat the launch
    /// value as unknown rather than assume a zero.
    pub(crate) fn launch_word(
        &self,
        allocation: AllocationId,
        byte_offset: usize,
        byte_len: usize,
    ) -> Option<u64> {
        let allocation = self.allocation(allocation).ok()?;
        let bytes = allocation.launch_bytes(byte_offset, byte_len)?;
        match byte_len {
            4 => Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?))),
            8 => Some(u64::from_le_bytes(bytes.try_into().ok()?)),
            _ => None,
        }
    }

    /// Return the per-byte initialization state for an entire allocation.
    pub fn snapshot_allocation_validity(
        &self,
        allocation: AllocationId,
    ) -> Result<Vec<u8>, MemoryError> {
        let view = self.full_view(allocation)?;
        Ok(self
            .byte_validity(&view, 0, view.byte_len())?
            .into_iter()
            .map(u8::from)
            .collect())
    }

    pub fn full_view(&self, allocation: AllocationId) -> Result<BufferView, MemoryError> {
        let allocation_ref = self.allocation(allocation)?;
        let byte_len = allocation_ref.byte_len();
        Ok(BufferView {
            arena_id: self.inner.arena_id,
            allocation,
            allocation_ref,
            byte_offset: 0,
            byte_len,
        })
    }

    /// Invert the invocation address bindings owned by imported allocations.
    /// Unbound simulator storage is not an address authority. Prefer an interior
    /// match over a neighboring allocation's one-past address; never choose
    /// arbitrarily between overlapping owners.
    pub(crate) fn observed_address_owner(
        &self,
        address: u64,
    ) -> Result<Option<(BufferView, usize)>, EngineError> {
        if address == 0 {
            return Ok(None);
        }
        let mut interior = None;
        let mut one_past = None;
        let mut ambiguous_end = false;
        for (&allocation, allocation_ref) in read_rwlock(&self.inner.allocations).iter() {
            let Some(&base) = allocation_ref.observed_address.get() else {
                continue;
            };
            let end = base
                .checked_add(u64::try_from(allocation_ref.byte_len()).map_err(|_| {
                    EngineError::analysis_incomplete("global_address_range_overflow")
                })?)
                .ok_or_else(|| EngineError::analysis_incomplete("global_address_range_overflow"))?;
            if address < base || address > end {
                continue;
            }
            let candidate = (
                BufferView {
                    arena_id: self.inner.arena_id,
                    allocation,
                    allocation_ref: allocation_ref.clone(),
                    byte_offset: 0,
                    byte_len: allocation_ref.byte_len(),
                },
                usize::try_from(address - base).map_err(|_| {
                    EngineError::analysis_incomplete("global_address_range_overflow")
                })?,
            );
            if address == end {
                ambiguous_end |= one_past.replace(candidate).is_some();
            } else if interior.replace(candidate).is_some() {
                return Err(EngineError::analysis_incomplete(
                    "ambiguous_global_address_owner",
                ));
            }
        }
        if interior.is_some() {
            return Ok(interior);
        }
        if ambiguous_end {
            return Err(EngineError::analysis_incomplete(
                "ambiguous_global_address_owner",
            ));
        }
        Ok(one_past)
    }

    pub fn view(
        &self,
        allocation: AllocationId,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<BufferView, MemoryError> {
        let allocation_ref = self.allocation(allocation)?;
        validate_range(
            allocation,
            allocation_ref.byte_len(),
            byte_offset,
            byte_len,
            RangeKind::View,
        )?;
        Ok(BufferView {
            arena_id: self.inner.arena_id,
            allocation,
            allocation_ref,
            byte_offset,
            byte_len,
        })
    }

    pub fn subview(
        &self,
        parent: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<BufferView, MemoryError> {
        let allocation = self.allocation_for_view(parent)?;
        validate_view(parent, allocation.byte_len())?;
        validate_range(
            parent.allocation,
            parent.byte_len,
            byte_offset,
            byte_len,
            RangeKind::Access,
        )?;
        let absolute_offset = parent
            .byte_offset
            .checked_add(byte_offset)
            .ok_or(MemoryError::OffsetOverflow)?;
        Ok(BufferView {
            arena_id: parent.arena_id,
            allocation: parent.allocation,
            allocation_ref: parent.allocation_ref.clone(),
            byte_offset: absolute_offset,
            byte_len,
        })
    }

    pub fn read_bytes(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<u8>, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateRead
        } else {
            ProfileKind::GmemRead
        });
        if allocation.is_owner_private() {
            let reviewing = self.records_uninitialized_read_reviews();
            let (bytes, review) = allocation
                .with_private(view.allocation, |data| {
                    if reviewing {
                        let mut bytes = vec![0_u8; byte_len];
                        let review = copy_private_bytes_zero_filled(
                            view.allocation,
                            data,
                            absolute,
                            &mut bytes,
                        );
                        Ok((bytes, review))
                    } else {
                        validate_private_initialized(view.allocation, data, absolute, byte_len)?;
                        Ok((data.bytes[absolute..absolute + byte_len].to_vec(), None))
                    }
                })
                .and_then(|result| result)?;
            if let Some(review) = review {
                self.record_uninitialized_read_review(review);
            }
            return Ok(bytes);
        }
        let shared = allocation
            .shared()
            .expect("non-private allocation has shared backing");
        loop {
            let reviewing = self.records_uninitialized_read_reviews();
            let snapshot = stable_snapshot(
                shared,
                absolute,
                byte_len,
                if reviewing {
                    SnapshotValidity::ZeroFill
                } else {
                    SnapshotValidity::RequireInitialized
                },
            );
            let Some(relative) = snapshot.first_invalid else {
                return Ok(snapshot.bytes);
            };
            let invalid_byte = absolute + relative;
            let stripe_index = shared.stripe_index(invalid_byte);
            let stripe = &shared.stripes[stripe_index];
            let _state = lock_mutex(&stripe.state);
            if !shared.is_valid(invalid_byte) {
                let review = UninitializedReadReview {
                    source: None,
                    allocation: view.allocation,
                    byte_offset: absolute,
                    byte_len,
                    first_uninitialized_byte: invalid_byte,
                };
                if reviewing {
                    drop(_state);
                    self.record_uninitialized_read_review(review);
                    return Ok(snapshot.bytes);
                }
                return Err(MemoryError::InvalidRead {
                    allocation: review.allocation,
                    byte_offset: review.byte_offset,
                    byte_len: review.byte_len,
                    first_invalid_byte: review.first_uninitialized_byte,
                });
            }
        }
    }

    /// Read bytes into caller-owned storage.
    ///
    /// Generated scalar paths use this form to keep their fixed-width scratch
    /// storage on the stack instead of allocating a tiny `Vec` per lane.
    pub fn read_bytes_into(
        &self,
        view: &BufferView,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        self.read_bytes_into_impl(view, byte_offset, target, true)
    }

    fn read_bytes_into_impl(
        &self,
        view: &BufferView,
        byte_offset: usize,
        target: &mut [u8],
        profile: bool,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let byte_len = target.len();
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        let _profile_timer = profile.then(|| {
            ProfileTimer::new(if allocation.is_owner_private() {
                ProfileKind::PrivateRead
            } else {
                ProfileKind::GmemRead
            })
        });
        if allocation.is_owner_private() {
            let reviewing = self.records_uninitialized_read_reviews();
            let review = allocation
                .with_private(view.allocation, |data| {
                    if reviewing {
                        Ok(copy_private_bytes_zero_filled(
                            view.allocation,
                            data,
                            absolute,
                            target,
                        ))
                    } else {
                        validate_private_initialized(view.allocation, data, absolute, byte_len)?;
                        target.copy_from_slice(&data.bytes[absolute..absolute + byte_len]);
                        Ok(None)
                    }
                })
                .and_then(|result| result)?;
            if let Some(review) = review {
                self.record_uninitialized_read_review(review);
            }
            return Ok(());
        }
        let shared = allocation
            .shared()
            .expect("non-private allocation has shared backing");
        loop {
            let reviewing = self.records_uninitialized_read_reviews();
            let Some(relative) = stable_snapshot_into(
                shared,
                absolute,
                target,
                if reviewing {
                    SnapshotValidity::ZeroFill
                } else {
                    SnapshotValidity::RequireInitialized
                },
            ) else {
                return Ok(());
            };
            let invalid_byte = absolute + relative;
            let stripe_index = shared.stripe_index(invalid_byte);
            let stripe = &shared.stripes[stripe_index];
            let _state = lock_mutex(&stripe.state);
            if !shared.is_valid(invalid_byte) {
                let review = UninitializedReadReview {
                    source: None,
                    allocation: view.allocation,
                    byte_offset: absolute,
                    byte_len,
                    first_uninitialized_byte: invalid_byte,
                };
                if reviewing {
                    drop(_state);
                    self.record_uninitialized_read_review(review);
                    return Ok(());
                }
                return Err(MemoryError::InvalidRead {
                    allocation: review.allocation,
                    byte_offset: review.byte_offset,
                    byte_len: review.byte_len,
                    first_invalid_byte: review.first_uninitialized_byte,
                });
            }
        }
    }

    /// Read a batch whose offsets were already resolved against this full
    /// allocation view by the owning address-space facade.
    ///
    /// CTA- and warp-private facades must validate their logical subviews
    /// before entering this hot path. Keeping the full-view assertion here
    /// prevents accidentally applying an already-absolute offset twice.
    #[inline]
    pub(crate) fn read_owner_private_resolved_bytes_batch_into(
        &self,
        view: &BufferView,
        absolute_byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        debug_assert!(byte_len <= target_stride);
        debug_assert!(target.len() >= WARP_SIZE * target_stride);
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        debug_assert_eq!(view.byte_offset, 0);
        debug_assert_eq!(view.byte_len, allocation.byte_len());
        for lane in mask {
            debug_assert!(absolute_byte_offsets[lane]
                .checked_add(byte_len)
                .is_some_and(|end| end <= allocation.byte_len()));
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateRead);
        let reviewing = self.records_uninitialized_read_reviews();
        let reviews = allocation
            .with_private(view.allocation, |data| {
                let mut reviews = Vec::new();
                for lane in mask {
                    let absolute = absolute_byte_offsets[lane];
                    let target_start = lane * target_stride;
                    let destination = &mut target[target_start..target_start + byte_len];
                    if reviewing {
                        if let Some(review) = copy_private_bytes_zero_filled(
                            view.allocation,
                            data,
                            absolute,
                            destination,
                        ) {
                            reviews.push(review);
                        }
                    } else {
                        validate_private_initialized(view.allocation, data, absolute, byte_len)?;
                        destination.copy_from_slice(&data.bytes[absolute..absolute + byte_len]);
                    }
                }
                Ok(reviews)
            })
            .and_then(|result| result)?;
        for review in reviews {
            self.record_uninitialized_read_review(review);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn read_owner_private_resolved_scalar_batch<T: RuntimeScalar>(
        &self,
        view: &BufferView,
        absolute_byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        source: Option<ReadSource>,
    ) -> Result<WarpValue<T>, EngineError> {
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        debug_assert_eq!(view.byte_offset, 0);
        debug_assert_eq!(view.byte_len, allocation.byte_len());
        for lane in mask {
            debug_assert!(absolute_byte_offsets[lane]
                .checked_add(T::BYTE_LEN)
                .is_some_and(|end| end <= allocation.byte_len()));
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateRead);
        let reviewing = self.records_uninitialized_read_reviews();
        let (values, reviews) = allocation.with_private(view.allocation, |data| {
            let mut values = WarpValue::splat(T::zero());
            let mut reviews = Vec::new();
            let mut encoded = vec![0_u8; T::BYTE_LEN];
            for lane in mask {
                let absolute = absolute_byte_offsets[lane];
                let bytes = if reviewing {
                    let destination = encoded.as_mut_slice();
                    if let Some(review) =
                        copy_private_bytes_zero_filled(view.allocation, data, absolute, destination)
                    {
                        reviews.push(review);
                    }
                    destination
                } else {
                    validate_private_initialized(view.allocation, data, absolute, T::BYTE_LEN)?;
                    &data.bytes[absolute..absolute + T::BYTE_LEN]
                };
                values[lane] = T::decode_le(bytes)?;
            }
            Ok::<_, EngineError>((values, reviews))
        })??;
        for mut review in reviews {
            review.source = source;
            self.record_uninitialized_read_review(review);
        }
        Ok(values)
    }

    pub(crate) fn read_shared_bytes_batch_into(
        &self,
        view: &BufferView,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        debug_assert!(byte_len <= target_stride);
        debug_assert!(target.len() >= WARP_SIZE * target_stride);
        debug_assert!(byte_len <= MEMORY_STRIPE_BYTES);
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(!allocation.is_owner_private());
        let mut absolutes = WarpValue::splat(0_usize);
        for lane in mask {
            absolutes[lane] =
                resolve_access(view, allocation.byte_len(), byte_offsets[lane], byte_len)?;
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::GmemRead);
        if self.records_uninitialized_read_reviews() {
            for lane in mask {
                let target_start = lane * target_stride;
                self.read_bytes_into_impl(
                    view,
                    byte_offsets[lane],
                    &mut target[target_start..target_start + byte_len],
                    false,
                )?;
            }
            return Ok(());
        }
        allocation
            .shared()
            .expect("shared batch requires shared allocation backing")
            .read_initialized_batch_into(
                view.allocation,
                &absolutes,
                mask,
                byte_len,
                target_stride,
                target,
            )
    }

    /// Write a batch whose offsets were already resolved and validated by the
    /// owning CTA- or warp-private address-space facade.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn write_owner_private_resolved_bytes_batch(
        &self,
        view: &BufferView,
        absolute_byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        source_stride: usize,
        source: &[u8],
    ) -> Result<bool, MemoryError> {
        debug_assert!(byte_len <= source_stride);
        debug_assert!(source.len() >= WARP_SIZE * source_stride);
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        debug_assert_eq!(view.byte_offset, 0);
        debug_assert_eq!(view.byte_len, allocation.byte_len());
        for lane in mask {
            debug_assert!(absolute_byte_offsets[lane]
                .checked_add(byte_len)
                .is_some_and(|end| end <= allocation.byte_len()));
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
        let _profile_detail_timer = ProfileTimer::new(ProfileKind::PrivateWriteResolvedBatch);
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let changed = allocation.with_private(view.allocation, |data| {
            let mut changed = false;
            for lane in mask {
                let absolute = absolute_byte_offsets[lane];
                let source_start = lane * source_stride;
                let lane_source = &source[source_start..source_start + byte_len];
                changed |= track_semantic_progress
                    && (data.bytes[absolute..absolute + byte_len] != *lane_source
                        || data.invalid_byte_count != 0
                            && !data.valid.range_fully_valid(absolute, absolute + byte_len));
                data.bytes[absolute..absolute + byte_len].copy_from_slice(lane_source);
                mark_private_initialized(data, absolute, byte_len);
            }
            changed
        });
        let changed = match changed {
            Ok(changed) => changed,
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                if owner == view.allocation =>
            {
                allocation.enqueue_private_writes(
                    view.allocation,
                    mask.into_iter().map(|lane| PendingPrivateWrite {
                        absolute: absolute_byte_offsets[lane],
                        bytes: source[lane * source_stride..lane * source_stride + byte_len]
                            .to_vec(),
                    }),
                )?;
                track_semantic_progress
            }
            Err(error) => return Err(error),
        };
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(changed)
    }

    #[inline]
    pub(crate) fn write_owner_private_resolved_scalar_batch<T: RuntimeScalar>(
        &self,
        view: &BufferView,
        absolute_byte_offsets: &WarpValue<usize>,
        values: &WarpValue<T>,
        mask: WarpMask,
    ) -> Result<bool, EngineError> {
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        debug_assert_eq!(view.byte_offset, 0);
        debug_assert_eq!(view.byte_len, allocation.byte_len());
        for lane in mask {
            debug_assert!(absolute_byte_offsets[lane]
                .checked_add(T::BYTE_LEN)
                .is_some_and(|end| end <= allocation.byte_len()));
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
        let _profile_detail_timer = ProfileTimer::new(ProfileKind::PrivateWriteResolvedBatch);
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let changed = allocation.with_private(view.allocation, |data| {
            let mut changed = false;
            for lane in mask {
                let absolute = absolute_byte_offsets[lane];
                let destination = &mut data.bytes[absolute..absolute + T::BYTE_LEN];
                if track_semantic_progress {
                    let mut encoded = [0_u8; 16];
                    values[lane].encode_le_into(&mut encoded[..T::BYTE_LEN]);
                    changed |= *destination != encoded[..T::BYTE_LEN]
                        || data.invalid_byte_count != 0
                            && !data
                                .valid
                                .range_fully_valid(absolute, absolute + T::BYTE_LEN);
                    destination.copy_from_slice(&encoded[..T::BYTE_LEN]);
                } else {
                    values[lane].encode_le_into(destination);
                }
                mark_private_initialized(data, absolute, T::BYTE_LEN);
            }
            changed
        });
        let changed = match changed {
            Ok(changed) => changed,
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                if owner == view.allocation =>
            {
                let writes = mask
                    .into_iter()
                    .map(|lane| {
                        let mut bytes = vec![0; T::BYTE_LEN];
                        values[lane].encode_le_into(&mut bytes);
                        PendingPrivateWrite {
                            absolute: absolute_byte_offsets[lane],
                            bytes,
                        }
                    })
                    .collect::<Vec<_>>();
                allocation.enqueue_private_writes(view.allocation, writes)?;
                track_semantic_progress
            }
            Err(error) => return Err(error.into()),
        };
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(changed)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_shared_bytes_batch(
        &self,
        view: &BufferView,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        source_stride: usize,
        source: &[u8],
    ) -> Result<(), MemoryError> {
        debug_assert!(byte_len <= source_stride);
        debug_assert!(source.len() >= WARP_SIZE * source_stride);
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(!allocation.is_owner_private());
        let mut absolutes = WarpValue::splat(0_usize);
        let mut write_ranges = Vec::with_capacity(mask.len());
        for lane in mask {
            let absolute =
                resolve_access(view, allocation.byte_len(), byte_offsets[lane], byte_len)?;
            absolutes[lane] = absolute;
            write_ranges.push((absolute, byte_len));
        }
        if write_ranges.is_empty() {
            return Ok(());
        }

        let _profile_timer = ProfileTimer::new(ProfileKind::GmemWrite);
        let targets =
            stripe_targets_for_ranges(view.allocation, view.allocation_ref.clone(), &write_ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let mut changed = false;
        for lane in mask {
            let absolute = absolutes[lane];
            let source_start = lane * source_stride;
            let lane_source = &source[source_start..source_start + byte_len];
            if track_semantic_progress {
                changed |= lane_source
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(relative, byte)| {
                        let (old, valid) =
                            locked_byte(&targets, &data, view.allocation, absolute + relative);
                        old != byte || !valid
                    });
            }
            store_initialized_locked(&targets, &mut data, view.allocation, absolute, lane_source)?;
        }
        drop(data);
        drop(guards);
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(())
    }

    /// Read a bounded region while materializing invalid bytes as zero.
    ///
    /// This is intentionally separate from `read_bytes`: generated code uses
    /// it only for operations whose hardware data movement traverses padded
    /// storage that is outside the live logical region.
    pub fn read_bytes_zero_filled(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<u8>, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateRead
        } else {
            ProfileKind::GmemRead
        });
        if allocation.is_owner_private() {
            return allocation.with_private(view.allocation, |data| {
                if private_range_fully_valid(data, absolute, byte_len) {
                    return data.bytes[absolute..absolute + byte_len].to_vec();
                }
                data.bytes[absolute..absolute + byte_len]
                    .iter()
                    .enumerate()
                    .map(|(relative, byte)| {
                        if data.valid.get(absolute + relative) {
                            *byte
                        } else {
                            0
                        }
                    })
                    .collect()
            });
        }
        Ok(stable_snapshot(
            allocation
                .shared()
                .expect("non-private allocation has shared backing"),
            absolute,
            byte_len,
            SnapshotValidity::ZeroFill,
        )
        .bytes)
    }

    /// Read a bounded region into caller-owned storage, replacing invalid
    /// bytes with zero without changing their initialization state.
    pub fn read_bytes_zero_filled_into(
        &self,
        view: &BufferView,
        byte_offset: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let byte_len = target.len();
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateRead
        } else {
            ProfileKind::GmemRead
        });
        if allocation.is_owner_private() {
            return allocation.with_private(view.allocation, |data| {
                if private_range_fully_valid(data, absolute, byte_len) {
                    target.copy_from_slice(&data.bytes[absolute..absolute + byte_len]);
                    return;
                }
                for (relative, (target, byte)) in target
                    .iter_mut()
                    .zip(&data.bytes[absolute..absolute + byte_len])
                    .enumerate()
                {
                    *target = if data.valid.get(absolute + relative) {
                        *byte
                    } else {
                        0
                    };
                }
            });
        }
        stable_snapshot_into(
            allocation
                .shared()
                .expect("non-private allocation has shared backing"),
            absolute,
            target,
            SnapshotValidity::ZeroFill,
        );
        Ok(())
    }

    pub fn write_bytes(
        &self,
        view: &BufferView,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateWrite
        } else {
            ProfileKind::GmemWrite
        });
        let _profile_detail_timer = allocation
            .is_owner_private()
            .then(|| ProfileTimer::new(ProfileKind::PrivateWriteScalar));
        #[cfg(feature = "analysis-core")]
        if self.inner.semantic_progress.observes_changes() {
            return self
                .write_bytes_observing_change(allocation, view, absolute, bytes)
                .map(|_| ());
        }
        if allocation.is_owner_private() {
            let result = allocation.with_private(view.allocation, |data| {
                data.bytes[absolute..absolute + bytes.len()].copy_from_slice(bytes);
                mark_private_initialized(data, absolute, bytes.len());
            });
            return match result {
                Ok(()) => Ok(()),
                Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                    if owner == view.allocation =>
                {
                    allocation.enqueue_private_writes(
                        view.allocation,
                        [PendingPrivateWrite {
                            absolute,
                            bytes: bytes.to_vec(),
                        }],
                    )
                }
                Err(error) => Err(error),
            };
        }
        let shared = allocation
            .shared()
            .expect("non-private allocation has shared backing");
        let first_stripe = shared.stripe_index(absolute);
        let last_stripe = shared.stripe_index(absolute + bytes.len() - 1);
        if first_stripe == last_stripe {
            write_single_stripe_bytes(
                shared,
                first_stripe,
                view.allocation,
                absolute,
                bytes,
                false,
            )?;
            return Ok(());
        }
        let write_ranges = [(absolute, bytes.len())];
        let targets =
            stripe_targets_for_ranges(view.allocation, view.allocation_ref.clone(), &write_ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        store_initialized_locked(&targets, &mut data, view.allocation, absolute, bytes)?;
        drop(data);
        drop(guards);
        Ok(())
    }

    /// Write one owner-private range and report whether its concrete value or
    /// validity changed while semantic-progress observation is enabled.
    pub(crate) fn write_owner_private_bytes_observing_change(
        &self,
        view: &BufferView,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<bool, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(false);
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
        let _profile_detail_timer = ProfileTimer::new(ProfileKind::PrivateWriteScalar);
        #[cfg(feature = "analysis-core")]
        if self.inner.semantic_progress.observes_changes() {
            return self.write_bytes_observing_change(allocation, view, absolute, bytes);
        }
        self.write_bytes(view, byte_offset, bytes)?;
        Ok(false)
    }

    /// Run one issue-local write stream against an owner-private allocation.
    ///
    /// The normal path resolves the view and borrows its backing once, then
    /// lets the caller generate destination bytes directly into that borrow.
    /// A cross-thread target falls back to one queued batch without rerunning
    /// the generator.
    #[inline]
    pub(crate) fn with_owner_private_write_session<R>(
        &self,
        view: &BufferView,
        operation: impl FnOnce(&mut OwnerPrivateWriteSession<'_>) -> R,
    ) -> Result<R, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        validate_view(view, allocation.byte_len())?;
        debug_assert!(allocation.is_owner_private());

        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let mut operation = Some(operation);
        let direct = allocation.with_private(view.allocation, |data| {
            let mut session = OwnerPrivateWriteSession::direct(
                view.byte_offset,
                view.byte_len,
                data,
                track_semantic_progress,
            );
            let result =
                operation
                    .take()
                    .expect("private write generator executes once")(&mut session);
            (result, session.changed)
        });
        match direct {
            Ok((result, changed)) => {
                if changed {
                    self.inner.semantic_progress.record_change();
                }
                Ok(result)
            }
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                if owner == view.allocation =>
            {
                let mut session = OwnerPrivateWriteSession::queued(view.byte_offset, view.byte_len);
                let result = operation
                    .take()
                    .expect("failed direct access did not execute the write generator")(
                    &mut session,
                );
                let writes = session.into_queued_writes();
                let changed = track_semantic_progress && !writes.is_empty();
                allocation.enqueue_private_writes(view.allocation, writes)?;
                if changed {
                    self.inner.semantic_progress.record_change();
                }
                Ok(result)
            }
            Err(error) => Err(error),
        }
    }

    /// Run one instruction-local read stream against an owner-private
    /// allocation. The allocation lookup, view validation, pending-write
    /// drain, and backing borrow happen once; the caller owns the address
    /// geometry and byte loop inside the session.
    #[inline]
    pub(crate) fn with_owner_private_read_session<R>(
        &self,
        view: &BufferView,
        operation: impl FnOnce(&mut OwnerPrivateReadSession<'_>) -> R,
    ) -> Result<R, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        validate_view(view, allocation.byte_len())?;
        debug_assert!(allocation.is_owner_private());

        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateRead);
        let reviewing = self.records_uninitialized_read_reviews();
        let (result, reviews) = allocation.with_private(view.allocation, |data| {
            let mut session = OwnerPrivateReadSession {
                allocation: view.allocation,
                view_byte_offset: view.byte_offset,
                view_byte_len: view.byte_len,
                data,
                reviewing,
                reviews: Vec::new(),
            };
            let result = operation(&mut session);
            (result, session.reviews)
        })?;
        for review in reviews {
            self.record_uninitialized_read_review(review);
        }
        Ok(result)
    }

    /// Run one read stream against a resolved shared-backing view.
    ///
    /// The allocation and view are validated once. Individual batches still
    /// validate their element offsets before using the session, but avoid
    /// resolving the same allocation and view for every batch.
    #[inline]
    pub(crate) fn with_shared_read_session<R>(
        &self,
        view: &BufferView,
        operation: impl FnOnce(&SharedReadSession<'_>) -> R,
    ) -> Result<R, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        validate_view(view, allocation.byte_len())?;
        let backing = allocation
            .shared()
            .expect("shared read session requires shared allocation backing");
        let session = SharedReadSession {
            allocation: view.allocation,
            view_byte_offset: view.byte_offset,
            view_byte_len: view.byte_len,
            backing,
        };
        Ok(operation(&session))
    }

    /// Publish multiple replacement writes into one allocation as one memory
    /// transaction.
    ///
    /// All ranges are validated before the first byte changes. Shared backing
    /// locks each touched stripe once, which is important for a deferred TMA
    /// payload containing thousands of small element writes.
    pub(crate) fn write_bytes_batch<'a>(
        &self,
        view: &BufferView,
        writes: impl IntoIterator<Item = (usize, &'a [u8])>,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let mut resolved = Vec::new();
        let mut ranges = Vec::new();
        for (byte_offset, bytes) in writes {
            let absolute = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
            if bytes.is_empty() {
                continue;
            }
            resolved.push((absolute, bytes));
            ranges.push(MemoryWriteRange {
                arena_id: view.arena_id,
                allocation: view.allocation,
                allocation_ref: view.allocation_ref.clone(),
                absolute_byte_offset: absolute,
                byte_len: bytes.len(),
            });
        }
        if resolved.is_empty() {
            return Ok(());
        }

        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateWrite
        } else {
            ProfileKind::GmemWrite
        });
        let _profile_detail_timer = allocation
            .is_owner_private()
            .then(|| ProfileTimer::new(ProfileKind::PrivateWriteBatch));
        if allocation.is_owner_private() {
            let track_semantic_progress = self.inner.semantic_progress.observes_changes();
            let changed = allocation.with_private(view.allocation, |data| {
                let mut changed = false;
                for &(absolute, bytes) in &resolved {
                    changed |= track_semantic_progress
                        && (data.bytes[absolute..absolute + bytes.len()] != *bytes
                            || data.invalid_byte_count != 0
                                && !data
                                    .valid
                                    .range_fully_valid(absolute, absolute + bytes.len()));
                    data.bytes[absolute..absolute + bytes.len()].copy_from_slice(bytes);
                    mark_private_initialized(data, absolute, bytes.len());
                }
                changed
            });
            let changed = match changed {
                Ok(changed) => changed,
                Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                    if owner == view.allocation =>
                {
                    allocation.enqueue_private_writes(
                        view.allocation,
                        resolved
                            .iter()
                            .map(|&(absolute, bytes)| PendingPrivateWrite {
                                absolute,
                                bytes: bytes.to_vec(),
                            }),
                    )?;
                    track_semantic_progress
                }
                Err(error) => return Err(error),
            };
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(());
        }

        let targets = stripe_targets_for_write_ranges(&ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let mut semantic_changed = false;
        for (absolute, bytes) in resolved {
            if track_semantic_progress {
                semantic_changed |= bytes.iter().copied().enumerate().any(|(relative, byte)| {
                    let (old, valid) =
                        locked_byte(&targets, &data, view.allocation, absolute + relative);
                    old != byte || !valid
                });
            }
            store_initialized_locked(&targets, &mut data, view.allocation, absolute, bytes)?;
        }
        drop(data);
        drop(guards);
        if semantic_changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(())
    }

    /// Publish one validated completion batch without claiming an unbound
    /// owner-private allocation.
    ///
    /// Scheduler-owned completion pumping may publish a shared-memory payload
    /// before the destination CTA has touched its backing. It must not claim
    /// that backing on the completion-pump thread. An already-bound physical
    /// owner keeps the direct path; otherwise it drains the queued payload on
    /// its next ordinary access.
    pub(crate) fn publish_owner_private_bytes_batch<'a>(
        &self,
        view: &BufferView,
        writes: impl IntoIterator<Item = (usize, &'a [u8])>,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        debug_assert!(allocation.is_owner_private());
        let mut resolved = Vec::new();
        for (byte_offset, bytes) in writes {
            let absolute = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
            if bytes.is_empty() {
                continue;
            }
            resolved.push((absolute, bytes));
        }
        if resolved.is_empty() {
            return Ok(());
        }

        let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
        let _profile_detail_timer = ProfileTimer::new(ProfileKind::PrivateWritePublishBatch);
        let track_semantic_progress = self.inner.semantic_progress.observes_changes();
        let changed = allocation.with_private_if_bound(view.allocation, |data| {
            let mut changed = false;
            for &(absolute, bytes) in &resolved {
                changed |= track_semantic_progress
                    && (data.bytes[absolute..absolute + bytes.len()] != *bytes
                        || data.invalid_byte_count != 0
                            && !data
                                .valid
                                .range_fully_valid(absolute, absolute + bytes.len()));
                data.bytes[absolute..absolute + bytes.len()].copy_from_slice(bytes);
                mark_private_initialized(data, absolute, bytes.len());
            }
            changed
        });
        let changed = match changed {
            Ok(changed) => changed,
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                if owner == view.allocation =>
            {
                allocation.enqueue_private_writes(
                    view.allocation,
                    resolved
                        .iter()
                        .map(|&(absolute, bytes)| PendingPrivateWrite {
                            absolute,
                            bytes: bytes.to_vec(),
                        }),
                )?;
                // The payload is intentionally not read on a non-owner
                // completion-pump thread. Conservatively report progress.
                track_semantic_progress
            }
            Err(error) => return Err(error),
        };
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(())
    }

    pub(crate) fn write_f32_rows(
        &self,
        view: &BufferView,
        row_byte_offsets: &[usize],
        values: &[f32],
        columns: usize,
    ) -> Result<(), MemoryError> {
        debug_assert_eq!(values.len(), row_byte_offsets.len() * columns);
        if row_byte_offsets.is_empty() || columns == 0 {
            return Ok(());
        }
        let allocation = self.allocation_for_view(view)?;
        let row_byte_len = columns
            .checked_mul(size_of::<f32>())
            .ok_or(MemoryError::OffsetOverflow)?;
        let absolutes = row_byte_offsets
            .iter()
            .map(|&offset| resolve_access(view, allocation.byte_len(), offset, row_byte_len))
            .collect::<Result<Vec<_>, _>>()?;
        if allocation.is_owner_private() {
            let _profile_timer = ProfileTimer::new(ProfileKind::PrivateWrite);
            let _profile_detail_timer = ProfileTimer::new(ProfileKind::PrivateWriteRows);
            let track_semantic_progress = self.inner.semantic_progress.observes_changes();
            let changed = allocation.with_private(view.allocation, |data| {
                let mut changed = false;
                for (&absolute, row) in absolutes.iter().zip(values.chunks_exact(columns)) {
                    let destination = &mut data.bytes[absolute..absolute + row_byte_len];
                    for (encoded, value) in destination.chunks_exact_mut(4).zip(row) {
                        let replacement = value.to_le_bytes();
                        changed |= track_semantic_progress && *encoded != replacement;
                        encoded.copy_from_slice(&replacement);
                    }
                    changed |= track_semantic_progress
                        && !data
                            .valid
                            .range_fully_valid(absolute, absolute + row_byte_len);
                    mark_private_initialized(data, absolute, row_byte_len);
                }
                changed
            })?;
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(());
        }
        for (&offset, row) in row_byte_offsets.iter().zip(values.chunks_exact(columns)) {
            let mut encoded = Vec::with_capacity(row_byte_len);
            for value in row {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            self.write_bytes(view, offset, &encoded)?;
        }
        Ok(())
    }

    #[cfg(feature = "analysis-core")]
    #[inline(never)]
    fn write_bytes_observing_change(
        &self,
        allocation: &Allocation,
        view: &BufferView,
        absolute: usize,
        bytes: &[u8],
    ) -> Result<bool, MemoryError> {
        if allocation.is_owner_private() {
            let changed = allocation.with_private(view.allocation, |data| {
                let changed = data.bytes[absolute..absolute + bytes.len()] != *bytes
                    || data.invalid_byte_count != 0
                        && !data
                            .valid
                            .range_fully_valid(absolute, absolute + bytes.len());
                data.bytes[absolute..absolute + bytes.len()].copy_from_slice(bytes);
                mark_private_initialized(data, absolute, bytes.len());
                changed
            });
            let changed = match changed {
                Ok(changed) => changed,
                Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation: owner })
                    if owner == view.allocation =>
                {
                    allocation.enqueue_private_writes(
                        view.allocation,
                        [PendingPrivateWrite {
                            absolute,
                            bytes: bytes.to_vec(),
                        }],
                    )?;
                    true
                }
                Err(error) => return Err(error),
            };
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(changed);
        }
        let shared = allocation
            .shared()
            .expect("non-private allocation has shared backing");
        let first_stripe = shared.stripe_index(absolute);
        let last_stripe = shared.stripe_index(absolute + bytes.len() - 1);
        if first_stripe == last_stripe {
            let changed = write_single_stripe_bytes(
                shared,
                first_stripe,
                view.allocation,
                absolute,
                bytes,
                true,
            )?;
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(changed);
        }
        let write_ranges = [(absolute, bytes.len())];
        let targets =
            stripe_targets_for_ranges(view.allocation, view.allocation_ref.clone(), &write_ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        let changed = bytes.iter().copied().enumerate().any(|(relative, byte)| {
            let (old, valid) = locked_byte(&targets, &data, view.allocation, absolute + relative);
            old != byte || !valid
        });
        store_initialized_locked(&targets, &mut data, view.allocation, absolute, bytes)?;
        drop(data);
        drop(guards);
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(changed)
    }

    pub fn defer_write_bytes(
        &self,
        view: &BufferView,
        byte_offset: usize,
        bytes: Vec<u8>,
    ) -> Result<DeferredGlobalWrite, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let _ = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        Ok(DeferredGlobalWrite {
            memory: self.clone(),
            view: view.clone(),
            byte_offset,
            payload: DeferredGlobalWritePayload::Replace(bytes),
        })
    }

    /// Append one exact replacement write to the preceding write when both
    /// ranges are contiguous parts of the same deferred transaction.
    ///
    /// Tensor-map stores discover their global coordinates element by
    /// element, but a dense row is still one byte range at publication time.
    /// Coalescing it here preserves the transaction's write union while
    /// avoiding one allocation and two `Arc` references per scalar element.
    pub(crate) fn defer_or_extend_write_bytes(
        &self,
        writes: &mut Vec<DeferredGlobalWrite>,
        view: &BufferView,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let _ = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        if let Some(previous) = writes.last_mut() {
            let contiguous = previous.byte_offset.checked_add(match &previous.payload {
                DeferredGlobalWritePayload::Replace(previous_bytes) => previous_bytes.len(),
                _ => 0,
            }) == Some(byte_offset);
            if contiguous
                && Arc::ptr_eq(&previous.memory.inner, &self.inner)
                && previous.view == *view
            {
                if let DeferredGlobalWritePayload::Replace(previous_bytes) = &mut previous.payload {
                    previous_bytes.extend_from_slice(bytes);
                    return Ok(());
                }
            }
        }
        writes.push(DeferredGlobalWrite {
            memory: self.clone(),
            view: view.clone(),
            byte_offset,
            payload: DeferredGlobalWritePayload::Replace(bytes.to_vec()),
        });
        Ok(())
    }

    pub fn defer_masked_write_bytes(
        &self,
        view: &BufferView,
        byte_offset: usize,
        bytes: Vec<u8>,
        masks: Vec<u8>,
    ) -> Result<DeferredGlobalWrite, MemoryError> {
        if bytes.len() != masks.len() {
            return Err(MemoryError::DeferredWriteMaskLengthMismatch {
                byte_len: bytes.len(),
                mask_len: masks.len(),
            });
        }
        let allocation = self.allocation_for_view(view)?;
        let _ = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        Ok(DeferredGlobalWrite {
            memory: self.clone(),
            view: view.clone(),
            byte_offset,
            payload: DeferredGlobalWritePayload::Masked { bytes, masks },
        })
    }

    pub fn defer_reduction_write_bytes(
        &self,
        view: &BufferView,
        byte_offset: usize,
        bytes: Vec<u8>,
        operation: DeferredGlobalReduction,
    ) -> Result<DeferredGlobalWrite, MemoryError> {
        let expected_byte_len = operation.byte_len();
        if bytes.len() != expected_byte_len {
            return Err(MemoryError::DeferredReductionLengthMismatch {
                expected_byte_len,
                actual_byte_len: bytes.len(),
            });
        }
        let allocation = self.allocation_for_view(view)?;
        let _ = resolve_access(view, allocation.byte_len(), byte_offset, bytes.len())?;
        Ok(DeferredGlobalWrite {
            memory: self.clone(),
            view: view.clone(),
            byte_offset,
            payload: DeferredGlobalWritePayload::Reduction { bytes, operation },
        })
    }

    /// Atomically read, transform, and write one physical byte range.
    ///
    /// Bounds and initialization checks, the callback, byte replacement, and
    /// metadata updates execute while holding every range stripe touched by
    /// the operation. Overlapping RMWs therefore have one linear order, while
    /// independent allocations and disjoint stripes can proceed concurrently.
    pub fn atomic_update_bytes<R, E>(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
        update: impl FnOnce(&[u8]) -> Result<(R, Vec<u8>), E>,
    ) -> Result<R, E>
    where
        E: From<MemoryError>,
    {
        if byte_len == 0 {
            return Err(E::from(MemoryError::EmptyAtomicUpdateRange));
        }

        let allocation = self.allocation_for_view(view).map_err(E::from)?;
        let absolute =
            resolve_access(view, allocation.byte_len(), byte_offset, byte_len).map_err(E::from)?;
        let _profile_timer = ProfileTimer::new(if allocation.is_owner_private() {
            ProfileKind::PrivateAtomic
        } else {
            ProfileKind::GmemAtomic
        });
        if allocation.is_owner_private() {
            let old_bytes = allocation
                .with_private(view.allocation, |data| {
                    validate_private_initialized(view.allocation, data, absolute, byte_len)?;
                    Ok(data.bytes[absolute..absolute + byte_len].to_vec())
                })
                .map_err(E::from)?
                .map_err(E::from)?;
            let (result, new_bytes) = update(&old_bytes)?;
            if new_bytes.len() != byte_len {
                return Err(E::from(MemoryError::AtomicUpdateLengthMismatch {
                    expected_byte_len: byte_len,
                    actual_byte_len: new_bytes.len(),
                }));
            }
            let changed = new_bytes != old_bytes;
            allocation
                .with_private(view.allocation, |data| {
                    data.bytes[absolute..absolute + byte_len].copy_from_slice(&new_bytes);
                    mark_private_initialized(data, absolute, byte_len);
                })
                .map_err(E::from)?;
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(result);
        }
        let write_ranges = [(absolute, byte_len)];
        let targets =
            stripe_targets_for_ranges(view.allocation, view.allocation_ref.clone(), &write_ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets).map_err(E::from)?;
        validate_initialized_locked(view.allocation, &targets, &data, absolute, byte_len)
            .map_err(E::from)?;
        let mut old_bytes = vec![0; byte_len];
        read_locked_bytes(&targets, &data, view.allocation, absolute, &mut old_bytes);
        let (result, new_bytes) = update(&old_bytes)?;
        if new_bytes.len() != byte_len {
            return Err(E::from(MemoryError::AtomicUpdateLengthMismatch {
                expected_byte_len: byte_len,
                actual_byte_len: new_bytes.len(),
            }));
        }
        let changed = new_bytes != old_bytes;

        store_initialized_locked(&targets, &mut data, view.allocation, absolute, &new_bytes)?;
        drop(data);
        drop(guards);
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(result)
    }

    pub fn atomic_update_scalar_le<T: AtomicMemoryScalar>(
        &self,
        view: &BufferView,
        element_index: usize,
        update: impl FnOnce(T) -> T,
    ) -> Result<T, MemoryError> {
        let byte_offset = element_offset(element_index, T::BYTE_LEN)?;
        self.atomic_update_bytes(view, byte_offset, T::BYTE_LEN, |old_bytes| {
            let old_value = T::decode_le(old_bytes);
            Ok((old_value, update(old_value).encode_le()))
        })
    }

    /// Return byte initialization state.
    pub fn byte_validity(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<bool>, MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        if allocation.is_owner_private() {
            return allocation.with_private(view.allocation, |data| {
                data.valid.to_bools(absolute, absolute + byte_len)
            });
        }
        Ok(allocation
            .shared()
            .expect("non-private allocation has shared backing")
            .validity(absolute, byte_len))
    }

    pub fn invalidate(
        &self,
        view: &BufferView,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        let absolute = resolve_access(view, allocation.byte_len(), byte_offset, byte_len)?;
        if byte_len == 0 {
            return Ok(());
        }
        if allocation.is_owner_private() {
            let track_semantic_progress = self.inner.semantic_progress.observes_changes();
            let changed = allocation.with_private(view.allocation, |data| {
                let changed = track_semantic_progress
                    && (data.invalid_byte_count == 0
                        || data.valid.any_valid_in(absolute, absolute + byte_len));
                mark_private_invalid(data, absolute, byte_len);
                changed
            })?;
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(());
        }
        let targets = stripe_targets_for_ranges(
            view.allocation,
            view.allocation_ref.clone(),
            &[(absolute, byte_len)],
        );
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        let changed = invalidate_locked(&targets, &mut data, view.allocation, absolute, byte_len)?;
        drop(data);
        drop(guards);
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(())
    }

    pub fn read_f32_le(&self, view: &BufferView, element_index: usize) -> Result<f32, MemoryError> {
        let byte_offset = element_offset(element_index, size_of::<f32>())?;
        let bytes = self.read_bytes(view, byte_offset, size_of::<f32>())?;
        Ok(f32::from_le_bytes(
            bytes.try_into().expect("read length is exactly one f32"),
        ))
    }

    pub fn write_f32_le(
        &self,
        view: &BufferView,
        element_index: usize,
        value: f32,
    ) -> Result<(), MemoryError> {
        let byte_offset = element_offset(element_index, size_of::<f32>())?;
        self.write_bytes(view, byte_offset, &value.to_le_bytes())
    }

    /// Load active lane indices and preserve inactive lanes in `destination`.
    ///
    /// All active accesses are validated before any destination lane changes.
    pub fn load_f32_indexed(
        &self,
        view: &BufferView,
        element_indices: &WarpValue<usize>,
        mask: WarpMask,
        destination: &mut WarpValue<f32>,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        validate_view(view, allocation.byte_len())?;
        let mut loaded = [None; WARP_SIZE];

        for lane in mask {
            let relative = element_offset(element_indices[lane], size_of::<f32>())?;
            let bytes: [u8; 4] = self
                .read_bytes(view, relative, size_of::<f32>())?
                .as_slice()
                .try_into()
                .expect("validated f32 range has four bytes");
            loaded[lane] = Some(f32::from_le_bytes(bytes));
        }

        for lane in mask {
            destination[lane] = loaded[lane].expect("active lane was loaded");
        }
        Ok(())
    }

    /// Store active lane values at their indexed f32 locations.
    ///
    /// The operation is fail-closed: every active access is bounds-checked
    /// before any byte or validity bit is modified. Duplicate active indices
    /// are applied in ascending lane order.
    pub fn store_f32_indexed(
        &self,
        view: &BufferView,
        element_indices: &WarpValue<usize>,
        values: &WarpValue<f32>,
        mask: WarpMask,
    ) -> Result<(), MemoryError> {
        let allocation = self.allocation_for_view(view)?;
        validate_view(view, allocation.byte_len())?;
        let mut writes = Vec::with_capacity(mask.len());

        for lane in mask {
            let relative = element_offset(element_indices[lane], size_of::<f32>())?;
            let absolute = resolve_access(view, allocation.byte_len(), relative, size_of::<f32>())?;
            writes.push((absolute, values[lane].to_le_bytes()));
        }

        let write_ranges = writes
            .iter()
            .map(|(absolute, _)| (*absolute, size_of::<f32>()))
            .collect::<Vec<_>>();
        if write_ranges.is_empty() {
            return Ok(());
        }
        if allocation.is_owner_private() {
            if !self.inner.semantic_progress.observes_changes() {
                return allocation.with_private(view.allocation, |data| {
                    for (absolute, bytes) in writes {
                        data.bytes[absolute..absolute + size_of::<f32>()].copy_from_slice(&bytes);
                        mark_private_initialized(data, absolute, size_of::<f32>());
                    }
                });
            }
            let changed = allocation.with_private(view.allocation, |data| {
                let mut changed = false;
                for (absolute, bytes) in writes {
                    changed |= data.bytes[absolute..absolute + size_of::<f32>()] != bytes
                        || data.invalid_byte_count != 0
                            && !data
                                .valid
                                .range_fully_valid(absolute, absolute + size_of::<f32>());
                    data.bytes[absolute..absolute + size_of::<f32>()].copy_from_slice(&bytes);
                    mark_private_initialized(data, absolute, size_of::<f32>());
                }
                changed
            })?;
            if changed {
                self.inner.semantic_progress.record_change();
            }
            return Ok(());
        }
        let targets =
            stripe_targets_for_ranges(view.allocation, view.allocation_ref.clone(), &write_ranges);
        let guards = lock_stripe_targets(&targets);
        let mut data = begin_stripe_writes(&targets)?;
        let mut changed = false;
        for (absolute, bytes) in writes {
            changed |= bytes.into_iter().enumerate().any(|(relative, byte)| {
                let (old, valid) =
                    locked_byte(&targets, &data, view.allocation, absolute + relative);
                old != byte || !valid
            });
            store_initialized_locked(&targets, &mut data, view.allocation, absolute, &bytes)?;
        }
        drop(data);
        drop(guards);
        if changed {
            self.inner.semantic_progress.record_change();
        }
        Ok(())
    }

    fn allocate(
        &self,
        bytes: Vec<u8>,
        valid: Vec<bool>,
        read_only: bool,
    ) -> Result<AllocationId, MemoryError> {
        debug_assert_eq!(bytes.len(), valid.len());
        let allocation = self
            .inner
            .next_allocation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(AllocationId)
            .map_err(|_| MemoryError::AllocationIdExhausted)?;
        let allocation_ref = Arc::new(Allocation::new(self.inner.mode, bytes, valid, read_only));
        self.insert_allocation(allocation, allocation_ref);
        Ok(allocation)
    }

    fn allocate_all_valid(
        &self,
        bytes: SharedInitialBytes,
        read_only: bool,
    ) -> Result<AllocationId, MemoryError> {
        let allocation = self
            .inner
            .next_allocation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(AllocationId)
            .map_err(|_| MemoryError::AllocationIdExhausted)?;
        let allocation_ref = Arc::new(Allocation::new_all_valid(self.inner.mode, bytes, read_only));
        self.insert_allocation(allocation, allocation_ref);
        Ok(allocation)
    }

    fn allocation(&self, allocation: AllocationId) -> Result<Arc<Allocation>, MemoryError> {
        read_rwlock(&self.inner.allocations)
            .get(&allocation)
            .cloned()
            .ok_or(MemoryError::UnknownAllocation { allocation })
    }

    fn insert_allocation(&self, id: AllocationId, allocation: Arc<Allocation>) {
        if self.inner.readonly_proxy_tracking.load(Ordering::Relaxed) {
            allocation.reset_readonly_proxy_history(true);
        }
        write_rwlock(&self.inner.allocations).insert(id, allocation);
    }

    fn allocation_for_view<'a>(&self, view: &'a BufferView) -> Result<&'a Allocation, MemoryError> {
        if view.arena_id != self.inner.arena_id {
            return Err(MemoryError::UnknownAllocation {
                allocation: view.allocation,
            });
        }
        Ok(&view.allocation_ref)
    }
}

struct MemoryState {
    arena_id: u64,
    next_allocation_id: AtomicU64,
    allocations: RwLock<BTreeMap<AllocationId, Arc<Allocation>>>,
    mode: MemoryMode,
    semantic_progress: SemanticProgress,
    uninitialized_read_policy: UninitializedReadPolicy,
    uninitialized_read_reviews: Mutex<BTreeSet<UninitializedReadReview>>,
    readonly_proxy_tracking: AtomicBool,
}

impl Default for MemoryState {
    fn default() -> Self {
        Self {
            arena_id: NEXT_MEMORY_ARENA_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                    next.checked_add(1)
                })
                .expect("NumSim memory arena ID space exhausted"),
            // Allocation IDs are physical identities within one address-space
            // arena. Keeping the counter arena-local makes pristine replay
            // diagnostics deterministic instead of depending on earlier runs
            // in the hosting Python process.
            next_allocation_id: AtomicU64::new(0),
            allocations: RwLock::new(BTreeMap::new()),
            mode: MemoryMode::Shared,
            semantic_progress: SemanticProgress::default(),
            uninitialized_read_policy: UninitializedReadPolicy::Error,
            uninitialized_read_reviews: Mutex::new(BTreeSet::new()),
            readonly_proxy_tracking: AtomicBool::new(false),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryMode {
    Shared,
    OwnerPrivate,
    QueuedOwnerPrivate,
}

struct Allocation {
    byte_len: usize,
    observed_address: OnceLock<u64>,
    backing: AllocationBacking,
    write_through: bool,
    /// The bytes this allocation was launched with, kept for the whole run.
    ///
    /// A stripe drops its view of them on its first write (`StripeBytes::Owned`),
    /// so the only way to answer "what did this word hold before the kernel
    /// touched anything" is to hold the launch buffer here. It is the same
    /// `Arc` the stripes start from, so keeping it costs one pointer.
    ///
    /// `None` where the launch bytes are a host region rather than an owned
    /// buffer; a caller that needs them must treat that as "unknown".
    launch_bytes: Option<SharedInitialBytes>,
}

enum AllocationBacking {
    Shared(SharedAllocationBacking),
    OwnerPrivate(Box<OwnerPrivateAllocationBacking>),
}

struct SharedAllocationBacking {
    read_only: Option<SharedStripeData>,
    stripes: Box<[MemoryStripe]>,
}

struct OwnerPrivateAllocationBacking {
    data: ThreadLocal<RefCell<OwnerPrivateData>>,
    initial_data: Mutex<Option<OwnerPrivateData>>,
    pending_writes: Mutex<Vec<PendingPrivateWrite>>,
    pending_writes_present: AtomicBool,
    queue_cross_thread_writes: bool,
}

/// Per-byte validity of an owner-private allocation, stored one bit per byte
/// so range checks cost a few word operations instead of a byte scan.
struct PrivateValidity {
    words: Vec<u64>,
    byte_len: usize,
}

impl PrivateValidity {
    const WORD_BITS: usize = u64::BITS as usize;

    fn new_filled(byte_len: usize, valid: bool) -> Self {
        let fill = if valid { u64::MAX } else { 0 };
        Self {
            words: vec![fill; byte_len.div_ceil(Self::WORD_BITS)],
            byte_len,
        }
    }

    fn from_bools(valid: &[bool]) -> Self {
        let mut this = Self::new_filled(valid.len(), false);
        for (index, valid) in valid.iter().enumerate() {
            if *valid {
                this.words[index / Self::WORD_BITS] |= 1 << (index % Self::WORD_BITS);
            }
        }
        this
    }

    #[inline(always)]
    fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.byte_len);
        self.words[index / Self::WORD_BITS] & (1 << (index % Self::WORD_BITS)) != 0
    }

    /// Visit every word overlapping `start..end` with the mask of in-range bits.
    #[inline(always)]
    fn masked_words(start: usize, end: usize, mut apply: impl FnMut(usize, u64) -> bool) {
        debug_assert!(start <= end);
        let mut index = start;
        while index < end {
            let word_index = index / Self::WORD_BITS;
            let word_start = word_index * Self::WORD_BITS;
            let mut mask = u64::MAX << (index - word_start);
            let word_end = word_start + Self::WORD_BITS;
            if word_end > end {
                mask &= u64::MAX >> (word_end - end);
            }
            if !apply(word_index, mask) {
                return;
            }
            index = word_end;
        }
    }

    #[inline]
    fn first_invalid_in(&self, start: usize, end: usize) -> Option<usize> {
        debug_assert!(end <= self.byte_len);
        let mut first = None;
        Self::masked_words(start, end, |word_index, mask| {
            let invalid = !self.words[word_index] & mask;
            if invalid == 0 {
                return true;
            }
            first = Some(word_index * Self::WORD_BITS + invalid.trailing_zeros() as usize);
            false
        });
        first
    }

    #[inline]
    fn range_fully_valid(&self, start: usize, end: usize) -> bool {
        self.first_invalid_in(start, end).is_none()
    }

    #[inline]
    fn any_valid_in(&self, start: usize, end: usize) -> bool {
        debug_assert!(end <= self.byte_len);
        let mut any = false;
        Self::masked_words(start, end, |word_index, mask| {
            any = self.words[word_index] & mask != 0;
            !any
        });
        any
    }

    #[inline]
    fn count_invalid_in(&self, start: usize, end: usize) -> usize {
        debug_assert!(end <= self.byte_len);
        let mut count = 0;
        Self::masked_words(start, end, |word_index, mask| {
            count += (!self.words[word_index] & mask).count_ones() as usize;
            true
        });
        count
    }

    #[inline]
    fn set_range(&mut self, start: usize, end: usize, valid: bool) {
        debug_assert!(end <= self.byte_len);
        Self::masked_words(start, end, |word_index, mask| {
            if valid {
                self.words[word_index] |= mask;
            } else {
                self.words[word_index] &= !mask;
            }
            true
        });
    }

    fn to_bools(&self, start: usize, end: usize) -> Vec<bool> {
        (start..end).map(|index| self.get(index)).collect()
    }
}

struct OwnerPrivateData {
    owner_thread: u64,
    bytes: Vec<u8>,
    valid: PrivateValidity,
    invalid_byte_count: usize,
}

struct PendingPrivateWrite {
    absolute: usize,
    bytes: Vec<u8>,
}

pub(crate) struct OwnerPrivateReadSession<'a> {
    allocation: AllocationId,
    view_byte_offset: usize,
    view_byte_len: usize,
    data: &'a OwnerPrivateData,
    reviewing: bool,
    reviews: Vec<UninitializedReadReview>,
}

impl OwnerPrivateReadSession<'_> {
    #[inline(always)]
    pub(crate) const fn records_uninitialized_read_reviews(&self) -> bool {
        self.reviewing
    }

    #[inline(always)]
    pub(crate) const fn all_bytes_initialized(&self) -> bool {
        self.data.invalid_byte_count == 0
    }

    #[inline(always)]
    fn absolute_prevalidated(&self, byte_offset: usize, byte_len: usize) -> usize {
        debug_assert!(byte_offset
            .checked_add(byte_len)
            .is_some_and(|end| end <= self.view_byte_len));
        self.view_byte_offset
            .checked_add(byte_offset)
            .expect("validated owner-private read offset overflow")
    }

    /// Validate one range during the instruction's preflight pass.
    #[inline(always)]
    pub(crate) fn validate_initialized_bytes_prevalidated(
        &self,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), MemoryError> {
        let absolute = self.absolute_prevalidated(byte_offset, byte_len);
        validate_private_initialized(self.allocation, self.data, absolute, byte_len)
    }

    /// Copy a range that the same instruction already validated.
    #[inline(always)]
    pub(crate) fn read_initialized_bytes_into_prevalidated(
        &self,
        byte_offset: usize,
        target: &mut [u8],
    ) {
        let absolute = self.absolute_prevalidated(byte_offset, target.len());
        debug_assert!(validate_private_initialized(
            self.allocation,
            self.data,
            absolute,
            target.len()
        )
        .is_ok());
        target.copy_from_slice(&self.data.bytes[absolute..absolute + target.len()]);
    }

    /// Copy one range under the review policy while retaining the scalar
    /// zero-fill diagnostic contract.
    #[inline(always)]
    pub(crate) fn read_reviewed_bytes_into_prevalidated(
        &mut self,
        byte_offset: usize,
        target: &mut [u8],
    ) {
        debug_assert!(self.reviewing);
        let absolute = self.absolute_prevalidated(byte_offset, target.len());
        if let Some(review) =
            copy_private_bytes_zero_filled(self.allocation, self.data, absolute, target)
        {
            self.reviews.push(review);
        }
    }

    /// Decode one contiguous binary32 row whose byte range was validated by
    /// the instruction preflight.
    #[inline(always)]
    pub(crate) fn read_initialized_f32s_into_prevalidated(
        &self,
        byte_offset: usize,
        target: &mut [f32],
    ) {
        let byte_len = target
            .len()
            .checked_mul(size_of::<f32>())
            .expect("validated owner-private f32 read size overflow");
        let absolute = self.absolute_prevalidated(byte_offset, byte_len);
        debug_assert!(
            validate_private_initialized(self.allocation, self.data, absolute, byte_len).is_ok()
        );
        for (value, encoded) in target
            .iter_mut()
            .zip(self.data.bytes[absolute..absolute + byte_len].chunks_exact(size_of::<f32>()))
        {
            *value = f32::from_le_bytes(
                encoded
                    .try_into()
                    .expect("validated f32 cell has four bytes"),
            );
        }
    }

    /// Decode one contiguous binary32 row under the review policy while
    /// retaining byte-precise zero filling and diagnostics.
    #[inline(always)]
    pub(crate) fn read_reviewed_f32s_into_prevalidated(
        &mut self,
        byte_offset: usize,
        target: &mut [f32],
    ) {
        debug_assert!(self.reviewing);
        let byte_len = target
            .len()
            .checked_mul(size_of::<f32>())
            .expect("validated owner-private f32 read size overflow");
        let absolute = self.absolute_prevalidated(byte_offset, byte_len);
        // A fully initialized row decodes straight from the backing bytes;
        // the per-cell review loop below produces no reviews and the same
        // values for such rows.
        if private_range_fully_valid(self.data, absolute, byte_len) {
            let (chunks, _) =
                self.data.bytes[absolute..absolute + byte_len].as_chunks::<{ size_of::<f32>() }>();
            for (value, encoded) in target.iter_mut().zip(chunks) {
                *value = f32::from_le_bytes(*encoded);
            }
            return;
        }
        for (column, value) in target.iter_mut().enumerate() {
            let mut encoded = [0_u8; size_of::<f32>()];
            self.read_reviewed_bytes_into_prevalidated(
                byte_offset + column * size_of::<f32>(),
                &mut encoded,
            );
            *value = f32::from_le_bytes(encoded);
        }
    }
}

enum OwnerPrivateWriteTarget<'a> {
    Direct(&'a mut OwnerPrivateData),
    Queued(Vec<PendingPrivateWrite>),
}

pub(crate) struct SharedReadSession<'a> {
    allocation: AllocationId,
    view_byte_offset: usize,
    view_byte_len: usize,
    backing: &'a SharedAllocationBacking,
}

impl SharedReadSession<'_> {
    #[inline(always)]
    pub(crate) const fn byte_len(&self) -> usize {
        self.view_byte_len
    }

    /// Read ranges already checked against the view that created this session.
    #[inline(always)]
    pub(crate) fn read_bytes_batch_into_prevalidated(
        &self,
        byte_offsets: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        debug_assert!(byte_len <= target_stride);
        debug_assert!(target.len() >= WARP_SIZE * target_stride);
        let mut absolutes = WarpValue::splat(0_usize);
        for lane in mask {
            debug_assert!(byte_offsets[lane]
                .checked_add(byte_len)
                .is_some_and(|end| end <= self.view_byte_len));
            absolutes[lane] = self
                .view_byte_offset
                .checked_add(byte_offsets[lane])
                .expect("validated shared read offset overflow");
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::GmemRead);
        self.backing.read_initialized_batch_into(
            self.allocation,
            &absolutes,
            mask,
            byte_len,
            target_stride,
            target,
        )
    }
}

pub(crate) struct OwnerPrivateWriteSession<'a> {
    view_byte_offset: usize,
    view_byte_len: usize,
    target: OwnerPrivateWriteTarget<'a>,
    track_semantic_progress: bool,
    changed: bool,
}

impl<'a> OwnerPrivateWriteSession<'a> {
    fn direct(
        view_byte_offset: usize,
        view_byte_len: usize,
        data: &'a mut OwnerPrivateData,
        track_semantic_progress: bool,
    ) -> Self {
        Self {
            view_byte_offset,
            view_byte_len,
            target: OwnerPrivateWriteTarget::Direct(data),
            track_semantic_progress,
            changed: false,
        }
    }

    fn queued(view_byte_offset: usize, view_byte_len: usize) -> Self {
        Self {
            view_byte_offset,
            view_byte_len,
            target: OwnerPrivateWriteTarget::Queued(Vec::new()),
            track_semantic_progress: false,
            changed: false,
        }
    }

    #[inline(always)]
    pub(crate) const fn byte_len(&self) -> usize {
        self.view_byte_len
    }

    /// Write a range already checked against the runtime buffer that created
    /// this session.
    #[inline(always)]
    pub(crate) fn write_bytes_prevalidated(&mut self, byte_offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        debug_assert!(byte_offset
            .checked_add(bytes.len())
            .is_some_and(|end| end <= self.view_byte_len));
        let absolute = self
            .view_byte_offset
            .checked_add(byte_offset)
            .expect("validated owner-private write offset overflow");
        match &mut self.target {
            OwnerPrivateWriteTarget::Direct(data) => {
                self.changed |= self.track_semantic_progress
                    && (data.bytes[absolute..absolute + bytes.len()] != *bytes
                        || !data
                            .valid
                            .range_fully_valid(absolute, absolute + bytes.len()));
                data.bytes[absolute..absolute + bytes.len()].copy_from_slice(bytes);
                mark_private_initialized(data, absolute, bytes.len());
            }
            OwnerPrivateWriteTarget::Queued(writes) => {
                if let Some(previous) = writes.last_mut() {
                    if previous.absolute.checked_add(previous.bytes.len()) == Some(absolute) {
                        previous.bytes.extend_from_slice(bytes);
                        return;
                    }
                }
                writes.push(PendingPrivateWrite {
                    absolute,
                    bytes: bytes.to_vec(),
                });
            }
        }
    }

    /// Encode one contiguous binary32 row whose destination range was
    /// validated by the instruction preflight.
    #[inline(always)]
    pub(crate) fn write_f32s_prevalidated(&mut self, byte_offset: usize, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let byte_len = values
            .len()
            .checked_mul(size_of::<f32>())
            .expect("validated owner-private f32 write size overflow");
        debug_assert!(byte_offset
            .checked_add(byte_len)
            .is_some_and(|end| end <= self.view_byte_len));
        let absolute = self
            .view_byte_offset
            .checked_add(byte_offset)
            .expect("validated owner-private f32 write offset overflow");
        match &mut self.target {
            OwnerPrivateWriteTarget::Direct(data) => {
                // Fixed-width chunks keep the encode loop free of per-cell
                // outlined memcpy calls; the change probe runs only until the
                // first difference is known.
                let destination = &mut data.bytes[absolute..absolute + byte_len];
                let (chunks, _) = destination.as_chunks_mut::<{ size_of::<f32>() }>();
                if self.track_semantic_progress && !self.changed {
                    self.changed = chunks
                        .iter()
                        .zip(values)
                        .any(|(encoded, value)| *encoded != value.to_le_bytes());
                }
                for (encoded, value) in chunks.iter_mut().zip(values) {
                    *encoded = value.to_le_bytes();
                }
                self.changed |= self.track_semantic_progress
                    && !data.valid.range_fully_valid(absolute, absolute + byte_len);
                mark_private_initialized(data, absolute, byte_len);
            }
            OwnerPrivateWriteTarget::Queued(writes) => {
                let mut bytes = Vec::with_capacity(byte_len);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                if let Some(previous) = writes.last_mut() {
                    if previous.absolute.checked_add(previous.bytes.len()) == Some(absolute) {
                        previous.bytes.extend_from_slice(&bytes);
                        return;
                    }
                }
                writes.push(PendingPrivateWrite { absolute, bytes });
            }
        }
    }

    /// Write fixed-width elements whose destination offsets were validated by
    /// the session caller. The target dispatch is shared by the whole batch.
    #[inline(always)]
    pub(crate) fn write_strided_batch_prevalidated<const N: usize>(
        &mut self,
        byte_offsets: &[usize; N],
        source: &[u8],
        source_stride: usize,
        byte_len: usize,
        count: usize,
    ) {
        debug_assert!(count <= N);
        debug_assert!(byte_len <= source_stride);
        debug_assert!(source.len() >= count.saturating_mul(source_stride));
        match &mut self.target {
            OwnerPrivateWriteTarget::Direct(data) => {
                if self.track_semantic_progress {
                    for element in 0..count {
                        let byte_offset = byte_offsets[element];
                        debug_assert!(byte_offset
                            .checked_add(byte_len)
                            .is_some_and(|end| end <= self.view_byte_len));
                        let absolute = self
                            .view_byte_offset
                            .checked_add(byte_offset)
                            .expect("validated owner-private write offset overflow");
                        let source_start = element * source_stride;
                        let bytes = &source[source_start..source_start + byte_len];
                        self.changed |= data.bytes[absolute..absolute + byte_len] != *bytes
                            || !data.valid.range_fully_valid(absolute, absolute + byte_len);
                        data.bytes[absolute..absolute + byte_len].copy_from_slice(bytes);
                        mark_private_initialized(data, absolute, byte_len);
                    }
                } else {
                    for element in 0..count {
                        let byte_offset = byte_offsets[element];
                        debug_assert!(byte_offset
                            .checked_add(byte_len)
                            .is_some_and(|end| end <= self.view_byte_len));
                        let absolute = self
                            .view_byte_offset
                            .checked_add(byte_offset)
                            .expect("validated owner-private write offset overflow");
                        let source_start = element * source_stride;
                        data.bytes[absolute..absolute + byte_len]
                            .copy_from_slice(&source[source_start..source_start + byte_len]);
                        mark_private_initialized(data, absolute, byte_len);
                    }
                }
            }
            OwnerPrivateWriteTarget::Queued(writes) => {
                for element in 0..count {
                    let byte_offset = byte_offsets[element];
                    debug_assert!(byte_offset
                        .checked_add(byte_len)
                        .is_some_and(|end| end <= self.view_byte_len));
                    let absolute = self
                        .view_byte_offset
                        .checked_add(byte_offset)
                        .expect("validated owner-private write offset overflow");
                    let source_start = element * source_stride;
                    let bytes = &source[source_start..source_start + byte_len];
                    if let Some(previous) = writes.last_mut() {
                        if previous.absolute.checked_add(previous.bytes.len()) == Some(absolute) {
                            previous.bytes.extend_from_slice(bytes);
                            continue;
                        }
                    }
                    writes.push(PendingPrivateWrite {
                        absolute,
                        bytes: bytes.to_vec(),
                    });
                }
            }
        }
    }

    fn into_queued_writes(self) -> Vec<PendingPrivateWrite> {
        match self.target {
            OwnerPrivateWriteTarget::Queued(writes) => writes,
            OwnerPrivateWriteTarget::Direct(_) => {
                unreachable!("direct write session does not retain queued writes")
            }
        }
    }
}

struct MemoryStripe {
    byte_start: usize,
    byte_len: usize,
    data: Option<RwLock<SharedStripeData>>,
    state: Mutex<()>,
}

struct SharedStripeData {
    bytes: StripeBytes,
    valid: StripeBytes,
    readonly_proxy: Option<Box<ReadonlyProxyHistory>>,
}

/// Kernel-lifetime history, owned by the same locked bytes as the data.
/// A write conflicts with a readonly-proxy read in either execution order,
/// including a write of the value already stored at that address.
struct ReadonlyProxyHistory {
    written: PrivateValidity,
    observed: PrivateValidity,
}

impl SharedStripeData {
    fn record_write(
        &mut self,
        allocation: AllocationId,
        absolute: usize,
        start: usize,
        end: usize,
    ) -> Result<(), MemoryError> {
        if let Some(history) = &mut self.readonly_proxy {
            if history.observed.any_valid_in(start, end) {
                return Err(MemoryError::ReadonlyProxyWriteConflict {
                    allocation,
                    byte_offset: absolute,
                    byte_len: end - start,
                });
            }
            history.written.set_range(start, end, true);
        }
        Ok(())
    }
}

impl MemoryStripe {
    fn read_data(&self) -> std::sync::RwLockReadGuard<'_, SharedStripeData> {
        read_rwlock(
            self.data
                .as_ref()
                .expect("read-write stripe has mutable data"),
        )
    }

    fn write_data(
        &self,
        allocation: AllocationId,
    ) -> Result<std::sync::RwLockWriteGuard<'_, SharedStripeData>, MemoryError> {
        self.data
            .as_ref()
            .map(write_rwlock)
            .ok_or(MemoryError::WriteToReadOnlyAllocation { allocation })
    }
}

impl Allocation {
    fn reset_readonly_proxy_history(&self, enabled: bool) {
        let Some(shared) = self.shared() else { return };
        for stripe in &shared.stripes {
            if let Some(data) = &stripe.data {
                write_rwlock(data).readonly_proxy = enabled.then(|| {
                    Box::new(ReadonlyProxyHistory {
                        written: PrivateValidity::new_filled(stripe.byte_len, false),
                        observed: PrivateValidity::new_filled(stripe.byte_len, false),
                    })
                });
            }
        }
    }

    fn new(mode: MemoryMode, bytes: Vec<u8>, valid: Vec<bool>, read_only: bool) -> Self {
        let byte_len = bytes.len();
        let launch = SharedInitialBytes::new(bytes);
        let backing = match mode {
            MemoryMode::Shared => {
                AllocationBacking::Shared(SharedAllocationBacking::new_with_initial_bytes(
                    launch.clone(),
                    Some(valid),
                    read_only,
                ))
            }
            MemoryMode::OwnerPrivate | MemoryMode::QueuedOwnerPrivate => {
                debug_assert!(!read_only);
                let invalid_byte_count = valid.iter().filter(|valid| !**valid).count();
                AllocationBacking::OwnerPrivate(Box::new(OwnerPrivateAllocationBacking {
                    data: ThreadLocal::new(),
                    initial_data: Mutex::new(Some(OwnerPrivateData {
                        owner_thread: 0,
                        bytes: launch.as_slice().to_vec(),
                        valid: PrivateValidity::from_bools(&valid),
                        invalid_byte_count,
                    })),
                    pending_writes: Mutex::new(Vec::new()),
                    pending_writes_present: AtomicBool::new(false),
                    queue_cross_thread_writes: mode == MemoryMode::QueuedOwnerPrivate,
                }))
            }
        };
        Self {
            byte_len,
            backing,
            write_through: false,
            observed_address: OnceLock::new(),
            launch_bytes: Some(launch),
        }
    }

    fn new_all_valid(mode: MemoryMode, bytes: SharedInitialBytes, read_only: bool) -> Self {
        let byte_len = bytes.as_slice().len();
        let launch = bytes.clone();
        let backing = match mode {
            MemoryMode::Shared => {
                AllocationBacking::Shared(SharedAllocationBacking::new_all_valid(bytes, read_only))
            }
            MemoryMode::OwnerPrivate | MemoryMode::QueuedOwnerPrivate => {
                debug_assert!(!read_only);
                AllocationBacking::OwnerPrivate(Box::new(OwnerPrivateAllocationBacking {
                    data: ThreadLocal::new(),
                    initial_data: Mutex::new(Some(OwnerPrivateData {
                        owner_thread: 0,
                        bytes: bytes.as_slice().to_vec(),
                        valid: PrivateValidity::new_filled(byte_len, true),
                        invalid_byte_count: 0,
                    })),
                    pending_writes: Mutex::new(Vec::new()),
                    pending_writes_present: AtomicBool::new(false),
                    queue_cross_thread_writes: mode == MemoryMode::QueuedOwnerPrivate,
                }))
            }
        };
        Self {
            byte_len,
            backing,
            write_through: false,
            observed_address: OnceLock::new(),
            launch_bytes: Some(launch),
        }
    }

    #[cfg(feature = "python")]
    fn new_host_all_valid(mode: MemoryMode, bytes: HostByteBuffer) -> Self {
        debug_assert_eq!(mode, MemoryMode::Shared);
        let byte_len = bytes.byte_len();
        Self {
            byte_len,
            backing: AllocationBacking::Shared(SharedAllocationBacking::new_host_all_valid(bytes)),
            write_through: true,
            observed_address: OnceLock::new(),
            // A host region is mapped, not copied, so there is no launch
            // snapshot to keep; callers see `None` and treat it as unknown.
            launch_bytes: None,
        }
    }

    /// The bytes this allocation was launched with, if they were retained.
    fn launch_bytes(&self, byte_offset: usize, byte_len: usize) -> Option<&[u8]> {
        let bytes = self.launch_bytes.as_ref()?.as_slice();
        bytes.get(byte_offset..byte_offset.checked_add(byte_len)?)
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn shared(&self) -> Option<&SharedAllocationBacking> {
        match &self.backing {
            AllocationBacking::Shared(backing) => Some(backing),
            AllocationBacking::OwnerPrivate(_) => None,
        }
    }

    fn is_owner_private(&self) -> bool {
        matches!(self.backing, AllocationBacking::OwnerPrivate(_))
    }

    fn is_write_through(&self) -> bool {
        self.write_through
    }

    #[inline]
    fn with_private<R>(
        &self,
        allocation: AllocationId,
        operation: impl FnOnce(&mut OwnerPrivateData) -> R,
    ) -> Result<R, MemoryError> {
        let AllocationBacking::OwnerPrivate(backing) = &self.backing else {
            unreachable!("private backing access requires an owner-private allocation");
        };
        let token = MEMORY_THREAD_TOKEN.with(|token| *token);
        let data = backing.data.get_or_try(|| {
            let mut initial_data = lock_mutex(&backing.initial_data)
                .take()
                .ok_or(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation })?;
            initial_data.owner_thread = token;
            Ok(RefCell::new(initial_data))
        })?;
        Self::with_private_data(backing, allocation, token, data, operation)
    }

    fn with_private_if_bound<R>(
        &self,
        allocation: AllocationId,
        operation: impl FnOnce(&mut OwnerPrivateData) -> R,
    ) -> Result<R, MemoryError> {
        let AllocationBacking::OwnerPrivate(backing) = &self.backing else {
            unreachable!("private backing access requires an owner-private allocation");
        };
        let token = MEMORY_THREAD_TOKEN.with(|token| *token);
        let data = backing
            .data
            .get()
            .ok_or(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation })?;
        Self::with_private_data(backing, allocation, token, data, operation)
    }

    #[inline]
    fn with_private_data<R>(
        backing: &OwnerPrivateAllocationBacking,
        allocation: AllocationId,
        token: u64,
        data: &RefCell<OwnerPrivateData>,
        operation: impl FnOnce(&mut OwnerPrivateData) -> R,
    ) -> Result<R, MemoryError> {
        let mut data = data.borrow_mut();
        if data.owner_thread != token {
            return Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation });
        }
        if backing.queue_cross_thread_writes
            && backing.pending_writes_present.load(Ordering::Acquire)
        {
            let pending = {
                let mut pending = lock_mutex(&backing.pending_writes);
                let pending = std::mem::take(&mut *pending);
                backing
                    .pending_writes_present
                    .store(false, Ordering::Release);
                pending
            };
            for write in pending {
                let bytes = &write.bytes;
                data.bytes[write.absolute..write.absolute + bytes.len()].copy_from_slice(bytes);
                mark_private_initialized(&mut data, write.absolute, bytes.len());
            }
        }
        Ok(operation(&mut data))
    }

    fn enqueue_private_writes(
        &self,
        allocation: AllocationId,
        writes: impl IntoIterator<Item = PendingPrivateWrite>,
    ) -> Result<(), MemoryError> {
        let AllocationBacking::OwnerPrivate(backing) = &self.backing else {
            unreachable!("queued private writes require owner-private backing");
        };
        if !backing.queue_cross_thread_writes {
            return Err(MemoryError::OwnerPrivateAccessFromDifferentThread { allocation });
        }
        let mut pending = lock_mutex(&backing.pending_writes);
        pending.extend(writes);
        backing
            .pending_writes_present
            .store(true, Ordering::Release);
        Ok(())
    }
}

impl SharedAllocationBacking {
    fn new(bytes: Vec<u8>, valid: Vec<bool>, read_only: bool) -> Self {
        Self::new_with_optional_validity(bytes, Some(valid), read_only)
    }

    fn new_all_valid(bytes: SharedInitialBytes, read_only: bool) -> Self {
        Self::new_with_initial_bytes(bytes, None, read_only)
    }

    #[cfg(feature = "python")]
    fn new_host_all_valid(bytes: HostByteBuffer) -> Self {
        let byte_len = bytes.byte_len();
        let stripes = bytes
            .into_regions(MEMORY_STRIPE_BYTES)
            .into_iter()
            .enumerate()
            .map(|(stripe_index, region)| {
                let byte_start = stripe_index * MEMORY_STRIPE_BYTES;
                let stripe_byte_len = MEMORY_STRIPE_BYTES.min(byte_len - byte_start);
                MemoryStripe {
                    byte_start,
                    byte_len: stripe_byte_len,
                    data: Some(RwLock::new(SharedStripeData {
                        bytes: StripeBytes::Host(region),
                        valid: StripeBytes::AllValid(stripe_byte_len),
                        readonly_proxy: None,
                    })),
                    state: Mutex::new(()),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            read_only: None,
            stripes,
        }
    }

    fn new_with_optional_validity(
        bytes: Vec<u8>,
        valid: Option<Vec<bool>>,
        read_only: bool,
    ) -> Self {
        Self::new_with_initial_bytes(SharedInitialBytes::new(bytes), valid, read_only)
    }

    fn new_with_initial_bytes(
        bytes: SharedInitialBytes,
        valid: Option<Vec<bool>>,
        read_only: bool,
    ) -> Self {
        let byte_len = bytes.as_slice().len();
        let read_only_data = if read_only {
            Some(SharedStripeData {
                readonly_proxy: None,
                bytes: StripeBytes::initial(bytes.clone(), 0, byte_len),
                valid: valid.as_ref().map_or_else(
                    || StripeBytes::Owned(Box::default()),
                    |valid| {
                        StripeBytes::Owned(
                            valid
                                .iter()
                                .copied()
                                .map(u8::from)
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                        )
                    },
                ),
            })
        } else {
            None
        };
        let stripes = (0..byte_len.div_ceil(MEMORY_STRIPE_BYTES))
            .map(|stripe_index| {
                let byte_start = stripe_index * MEMORY_STRIPE_BYTES;
                let stripe_byte_len = MEMORY_STRIPE_BYTES.min(byte_len - byte_start);
                MemoryStripe {
                    byte_start,
                    byte_len: stripe_byte_len,
                    data: if read_only_data.is_some() {
                        None
                    } else {
                        Some(RwLock::new(SharedStripeData {
                            readonly_proxy: None,
                            bytes: StripeBytes::initial(
                                bytes.clone(),
                                byte_start,
                                byte_start + stripe_byte_len,
                            ),
                            valid: valid.as_ref().map_or_else(
                                || StripeBytes::AllValid(stripe_byte_len),
                                |valid| {
                                    StripeBytes::Owned(
                                        valid[byte_start..byte_start + stripe_byte_len]
                                            .iter()
                                            .copied()
                                            .map(u8::from)
                                            .collect::<Vec<_>>()
                                            .into_boxed_slice(),
                                    )
                                },
                            ),
                        }))
                    },
                    state: Mutex::new(()),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            read_only: read_only_data,
            stripes,
        }
    }

    fn validity(&self, absolute: usize, byte_len: usize) -> Vec<bool> {
        if byte_len == 0 {
            return Vec::new();
        }
        if let Some(data) = &self.read_only {
            if data.valid.is_empty() {
                return vec![true; byte_len];
            }
            return data.valid[absolute..absolute + byte_len]
                .iter()
                .map(|valid| *valid != 0)
                .collect();
        }
        let first_stripe = self.stripe_index(absolute);
        let last_stripe = self.stripe_index(absolute + byte_len - 1);
        let guards = self.stripes[first_stripe..=last_stripe]
            .iter()
            .map(MemoryStripe::read_data)
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(byte_len);
        for (stripe, data) in self.stripes[first_stripe..=last_stripe].iter().zip(&guards) {
            let overlap_start = absolute.max(stripe.byte_start);
            let overlap_end = (absolute + byte_len).min(stripe.byte_start + stripe.byte_len);
            if data.valid.is_all_valid() {
                result.extend(std::iter::repeat_n(true, overlap_end - overlap_start));
            } else {
                result.extend(
                    data.valid[overlap_start - stripe.byte_start..overlap_end - stripe.byte_start]
                        .iter()
                        .map(|valid| *valid != 0),
                );
            }
        }
        result
    }

    fn snapshot_into(
        &self,
        absolute: usize,
        target: &mut [u8],
        validity: SnapshotValidity,
    ) -> Option<usize> {
        if target.is_empty() {
            return None;
        }
        if let Some(data) = &self.read_only {
            let source = &data.bytes[absolute..absolute + target.len()];
            if data.valid.is_empty() || matches!(validity, SnapshotValidity::Ignore) {
                target.copy_from_slice(source);
                return None;
            }
            let valid = &data.valid[absolute..absolute + target.len()];
            if valid.iter().all(|valid| *valid != 0) {
                target.copy_from_slice(source);
                return None;
            }
            let mut first_invalid = None;
            for (relative, ((target, source), valid)) in
                target.iter_mut().zip(source).zip(valid).enumerate()
            {
                if *valid == 0 {
                    first_invalid.get_or_insert(relative);
                }
                *target = if *valid != 0 || matches!(validity, SnapshotValidity::RequireInitialized)
                {
                    *source
                } else {
                    0
                };
            }
            return first_invalid;
        }
        let first_stripe = self.stripe_index(absolute);
        let last_stripe = self.stripe_index(absolute + target.len() - 1);
        let guards = self.stripes[first_stripe..=last_stripe]
            .iter()
            .map(MemoryStripe::read_data)
            .collect::<Vec<_>>();
        let mut first_invalid = None;
        for (stripe, data) in self.stripes[first_stripe..=last_stripe].iter().zip(&guards) {
            let overlap_start = absolute.max(stripe.byte_start);
            let overlap_end = (absolute + target.len()).min(stripe.byte_start + stripe.byte_len);
            let source_start = overlap_start - stripe.byte_start;
            let source_end = overlap_end - stripe.byte_start;
            let target_start = overlap_start - absolute;
            let target_end = overlap_end - absolute;
            let source = &data.bytes[source_start..source_end];
            let destination = &mut target[target_start..target_end];
            match validity {
                SnapshotValidity::Ignore => destination.copy_from_slice(source),
                SnapshotValidity::RequireInitialized | SnapshotValidity::ZeroFill => {
                    if data.valid.is_all_valid() {
                        destination.copy_from_slice(source);
                        continue;
                    }
                    let valid = &data.valid[source_start..source_end];
                    for (relative, ((destination, source), valid)) in
                        destination.iter_mut().zip(source).zip(valid).enumerate()
                    {
                        if *valid == 0 {
                            first_invalid.get_or_insert(target_start + relative);
                        }
                        *destination = if *valid != 0
                            || matches!(validity, SnapshotValidity::RequireInitialized)
                        {
                            *source
                        } else {
                            0
                        };
                    }
                }
            }
        }
        first_invalid
    }

    #[allow(clippy::too_many_arguments)]
    fn read_initialized_batch_into(
        &self,
        allocation: AllocationId,
        absolutes: &WarpValue<usize>,
        mask: WarpMask,
        byte_len: usize,
        target_stride: usize,
        target: &mut [u8],
    ) -> Result<(), MemoryError> {
        const MAX_BATCH_STRIPES: usize = WARP_SIZE * 2;

        if let Some(data) = &self.read_only {
            crate::profile_count(ProfileKind::GmemReadReadonly);
            if mask.is_full() && target_stride == byte_len {
                let first_absolute = absolutes[0];
                let contiguous = (1..WARP_SIZE).all(|lane| {
                    first_absolute.checked_add(lane * byte_len) == Some(absolutes[lane])
                });
                if contiguous {
                    crate::profile_count(ProfileKind::GmemReadContiguous);
                    let batch_byte_len = WARP_SIZE * byte_len;
                    target[..batch_byte_len].copy_from_slice(
                        &data.bytes[first_absolute..first_absolute + batch_byte_len],
                    );
                    if data.valid.is_empty() {
                        return Ok(());
                    }
                    if let Some(relative) = data.valid
                        [first_absolute..first_absolute + batch_byte_len]
                        .iter()
                        .position(|valid| *valid == 0)
                    {
                        let lane = relative / byte_len;
                        let invalid_byte = first_absolute + relative;
                        return Err(MemoryError::InvalidRead {
                            allocation,
                            byte_offset: absolutes[lane],
                            byte_len,
                            first_invalid_byte: invalid_byte,
                        });
                    }
                    return Ok(());
                }
            }
            if let Some(first_lane) = mask.first_active() {
                let absolute = absolutes[first_lane];
                if mask.into_iter().all(|lane| absolutes[lane] == absolute) {
                    crate::profile_count(ProfileKind::GmemReadUniform);
                    let destination_start = first_lane * target_stride;
                    let destination = &mut target[destination_start..destination_start + byte_len];
                    destination.copy_from_slice(&data.bytes[absolute..absolute + byte_len]);
                    if !data.valid.is_empty() {
                        if let Some(relative) = data.valid[absolute..absolute + byte_len]
                            .iter()
                            .position(|valid| *valid == 0)
                        {
                            let invalid_byte = absolute + relative;
                            return Err(MemoryError::InvalidRead {
                                allocation,
                                byte_offset: absolute,
                                byte_len,
                                first_invalid_byte: invalid_byte,
                            });
                        }
                    }
                    for lane in mask {
                        if lane != first_lane {
                            target.copy_within(
                                destination_start..destination_start + byte_len,
                                lane * target_stride,
                            );
                        }
                    }
                    return Ok(());
                }
            }
            for lane in mask {
                let absolute = absolutes[lane];
                let destination_start = lane * target_stride;
                let destination = &mut target[destination_start..destination_start + byte_len];
                destination.copy_from_slice(&data.bytes[absolute..absolute + byte_len]);
                if data.valid.is_empty() {
                    continue;
                }
                if let Some(relative) = data.valid[absolute..absolute + byte_len]
                    .iter()
                    .position(|valid| *valid == 0)
                {
                    let invalid_byte = absolute + relative;
                    return Err(MemoryError::InvalidRead {
                        allocation,
                        byte_offset: absolute,
                        byte_len,
                        first_invalid_byte: invalid_byte,
                    });
                }
            }
            return Ok(());
        }

        // Scalar warp loads overwhelmingly stay inside one 4 KiB stripe. The
        // general path below sorts and deduplicates up to 64 stripe indices,
        // initializes a 64-guard array, and performs a binary search for every
        // lane chunk. Preserve the same one-lock snapshot while avoiding that
        // machinery when every active lane touches one common stripe.
        if let Some(first_lane) = mask.first_active() {
            let first_absolute = absolutes[first_lane];
            let common_stripe = self.stripe_index(first_absolute);
            let one_stripe = self.stripe_index(first_absolute + byte_len - 1) == common_stripe
                && mask.into_iter().all(|lane| {
                    self.stripe_index(absolutes[lane]) == common_stripe
                        && self.stripe_index(absolutes[lane] + byte_len - 1) == common_stripe
                });
            if one_stripe {
                crate::profile_count(ProfileKind::GmemReadOneStripe);
                let stripe = &self.stripes[common_stripe];
                loop {
                    let stripe_data = stripe.read_data();
                    let mut first_invalid = None;
                    let contiguous_full_warp = mask.is_full()
                        && target_stride == byte_len
                        && (1..WARP_SIZE).all(|lane| {
                            absolutes[0].checked_add(lane * byte_len) == Some(absolutes[lane])
                        });
                    if contiguous_full_warp {
                        crate::profile_count(ProfileKind::GmemReadContiguous);
                        let absolute = absolutes[0];
                        let source_start = absolute - stripe.byte_start;
                        let batch_byte_len = WARP_SIZE * byte_len;
                        target[..batch_byte_len].copy_from_slice(
                            &stripe_data.bytes[source_start..source_start + batch_byte_len],
                        );
                        if !stripe_data.valid.is_all_valid() {
                            first_invalid = stripe_data.valid
                                [source_start..source_start + batch_byte_len]
                                .iter()
                                .position(|valid| *valid == 0)
                                .map(|relative| {
                                    let lane = relative / byte_len;
                                    (absolutes[lane] + relative % byte_len, absolutes[lane])
                                });
                        }
                    } else {
                        let uniform = mask
                            .into_iter()
                            .all(|lane| absolutes[lane] == first_absolute);
                        if uniform {
                            crate::profile_count(ProfileKind::GmemReadUniform);
                            let source_start = first_absolute - stripe.byte_start;
                            let destination_start = first_lane * target_stride;
                            target[destination_start..destination_start + byte_len]
                                .copy_from_slice(
                                    &stripe_data.bytes[source_start..source_start + byte_len],
                                );
                            if !stripe_data.valid.is_all_valid() {
                                first_invalid = stripe_data.valid
                                    [source_start..source_start + byte_len]
                                    .iter()
                                    .position(|valid| *valid == 0)
                                    .map(|relative| (first_absolute + relative, first_absolute));
                            }
                            if first_invalid.is_none() {
                                for lane in mask {
                                    if lane != first_lane {
                                        target.copy_within(
                                            destination_start..destination_start + byte_len,
                                            lane * target_stride,
                                        );
                                    }
                                }
                            }
                        } else {
                            for lane in mask {
                                let absolute = absolutes[lane];
                                let source_start = absolute - stripe.byte_start;
                                let destination_start = lane * target_stride;
                                target[destination_start..destination_start + byte_len]
                                    .copy_from_slice(
                                        &stripe_data.bytes[source_start..source_start + byte_len],
                                    );
                                if first_invalid.is_none() && !stripe_data.valid.is_all_valid() {
                                    first_invalid = stripe_data.valid
                                        [source_start..source_start + byte_len]
                                        .iter()
                                        .position(|valid| *valid == 0)
                                        .map(|relative| (absolute + relative, absolute));
                                }
                            }
                        }
                    }
                    drop(stripe_data);

                    let Some((invalid_byte, access_absolute)) = first_invalid else {
                        return Ok(());
                    };
                    let _state = lock_mutex(&stripe.state);
                    if !self.is_valid(invalid_byte) {
                        return Err(MemoryError::InvalidRead {
                            allocation,
                            byte_offset: access_absolute,
                            byte_len,
                            first_invalid_byte: invalid_byte,
                        });
                    }
                }
            }
        }

        crate::profile_count(ProfileKind::GmemReadGeneral);
        let mut stripe_indices = [0_usize; MAX_BATCH_STRIPES];
        let mut stripe_count = 0;
        for lane in mask {
            let absolute = absolutes[lane];
            stripe_indices[stripe_count] = self.stripe_index(absolute);
            stripe_count += 1;
            let last_stripe = self.stripe_index(absolute + byte_len - 1);
            if last_stripe != stripe_indices[stripe_count - 1] {
                stripe_indices[stripe_count] = last_stripe;
                stripe_count += 1;
            }
        }
        stripe_indices[..stripe_count].sort_unstable();
        let mut unique_count = 0;
        for source in 0..stripe_count {
            if source == 0 || stripe_indices[source] != stripe_indices[source - 1] {
                stripe_indices[unique_count] = stripe_indices[source];
                unique_count += 1;
            }
        }

        loop {
            let mut data: [Option<std::sync::RwLockReadGuard<'_, SharedStripeData>>;
                MAX_BATCH_STRIPES] = std::array::from_fn(|_| None);
            for position in 0..unique_count {
                data[position] = Some(self.stripes[stripe_indices[position]].read_data());
            }

            let mut first_invalid = None;
            for lane in mask {
                let absolute = absolutes[lane];
                let end = absolute + byte_len;
                let mut current = absolute;
                while current < end {
                    let stripe_index = self.stripe_index(current);
                    let position = stripe_indices[..unique_count]
                        .binary_search(&stripe_index)
                        .expect("batch read locked every touched stripe");
                    let stripe = &self.stripes[stripe_index];
                    let stripe_data = data[position]
                        .as_ref()
                        .expect("batch read guard is present");
                    let chunk_end = end.min(stripe.byte_start + stripe.byte_len);
                    let source_start = current - stripe.byte_start;
                    let source_end = chunk_end - stripe.byte_start;
                    let destination_start = lane * target_stride + current - absolute;
                    let destination_end = destination_start + chunk_end - current;
                    target[destination_start..destination_end]
                        .copy_from_slice(&stripe_data.bytes[source_start..source_end]);
                    if first_invalid.is_none() && !stripe_data.valid.is_all_valid() {
                        if let Some(relative) = stripe_data.valid[source_start..source_end]
                            .iter()
                            .position(|valid| *valid == 0)
                        {
                            first_invalid = Some((lane, current + relative, absolute, byte_len));
                        }
                    }
                    current = chunk_end;
                }
            }
            drop(data);

            let Some((_lane, invalid_byte, access_absolute, access_byte_len)) = first_invalid
            else {
                return Ok(());
            };
            let stripe = &self.stripes[self.stripe_index(invalid_byte)];
            let _state = lock_mutex(&stripe.state);
            if !self.is_valid(invalid_byte) {
                return Err(MemoryError::InvalidRead {
                    allocation,
                    byte_offset: access_absolute,
                    byte_len: access_byte_len,
                    first_invalid_byte: invalid_byte,
                });
            }
        }
    }

    fn stripe_index(&self, absolute_byte_offset: usize) -> usize {
        absolute_byte_offset / MEMORY_STRIPE_BYTES
    }

    fn is_valid(&self, absolute: usize) -> bool {
        if let Some(data) = &self.read_only {
            return data.valid.is_empty() || data.valid[absolute] != 0;
        }
        let stripe = &self.stripes[self.stripe_index(absolute)];
        let data = stripe.read_data();
        data.valid.is_all_valid() || data.valid[absolute - stripe.byte_start] != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StripeKey {
    allocation: AllocationId,
    stripe_index: usize,
}

#[derive(Clone)]
struct StripeTarget {
    key: StripeKey,
    allocation: Arc<Allocation>,
}

impl StripeTarget {
    fn key(&self) -> StripeKey {
        self.key
    }

    fn stripe(&self) -> &MemoryStripe {
        &self
            .allocation
            .shared()
            .expect("stripe targets only contain shared allocations")
            .stripes[self.key.stripe_index]
    }
}

struct StripeWriteData<'a> {
    guards: Vec<std::sync::RwLockWriteGuard<'a, SharedStripeData>>,
}

#[derive(Clone)]
struct MemoryWriteRange {
    arena_id: u64,
    allocation: AllocationId,
    allocation_ref: Arc<Allocation>,
    absolute_byte_offset: usize,
    byte_len: usize,
}

impl PartialEq for MemoryWriteRange {
    fn eq(&self, other: &Self) -> bool {
        self.arena_id == other.arena_id
            && self.allocation == other.allocation
            && self.absolute_byte_offset == other.absolute_byte_offset
            && self.byte_len == other.byte_len
    }
}

impl Eq for MemoryWriteRange {}

impl PartialOrd for MemoryWriteRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MemoryWriteRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.arena_id,
            self.allocation,
            self.absolute_byte_offset,
            self.byte_len,
        )
            .cmp(&(
                other.arena_id,
                other.allocation,
                other.absolute_byte_offset,
                other.byte_len,
            ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    ReadonlyProxyTrackingUnavailable,
    ReadonlyProxyWriteConflict {
        allocation: AllocationId,
        byte_offset: usize,
        byte_len: usize,
    },
    AllocationIdExhausted,
    OwnerPrivateAccessFromDifferentThread {
        allocation: AllocationId,
    },
    OwnerPrivateWriteWaitUnsupported {
        allocation: AllocationId,
    },
    WriteToReadOnlyAllocation {
        allocation: AllocationId,
    },
    EmptyAtomicUpdateRange,
    AtomicUpdateLengthMismatch {
        expected_byte_len: usize,
        actual_byte_len: usize,
    },
    DeferredWriteMaskLengthMismatch {
        byte_len: usize,
        mask_len: usize,
    },
    DeferredReductionLengthMismatch {
        expected_byte_len: usize,
        actual_byte_len: usize,
    },
    InitializationLengthMismatch {
        byte_len: usize,
        validity_len: usize,
    },
    InvalidValidityByte {
        index: usize,
        value: u8,
    },
    UnknownAllocation {
        allocation: AllocationId,
    },
    ViewOutOfBounds {
        allocation: AllocationId,
        allocation_byte_len: usize,
        byte_offset: usize,
        byte_len: usize,
    },
    AccessOutOfBounds {
        allocation: AllocationId,
        view_byte_len: usize,
        byte_offset: usize,
        byte_len: usize,
    },
    InvalidRead {
        allocation: AllocationId,
        byte_offset: usize,
        byte_len: usize,
        first_invalid_byte: usize,
    },
    OffsetOverflow,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadonlyProxyTrackingUnavailable => f.write_str(
                "readonly-proxy load requires kernel-lifetime global write tracking",
            ),
            Self::ReadonlyProxyWriteConflict { allocation, byte_offset, byte_len } => write!(
                f,
                "write overlaps readonly bytes (readonly-proxy load or invocation binding): {allocation}, byte offset {byte_offset}, width {byte_len}",
            ),
            Self::AllocationIdExhausted => f.write_str("global-memory allocation IDs exhausted"),
            Self::OwnerPrivateAccessFromDifferentThread { allocation } => write!(
                f,
                "owner-private {allocation} was accessed from a different CPU thread"
            ),
            Self::OwnerPrivateWriteWaitUnsupported { allocation } => write!(
                f,
                "owner-private {allocation} does not support cross-thread write watches"
            ),
            Self::WriteToReadOnlyAllocation { allocation } => {
                write!(f, "write targets read-only {allocation}")
            }
            Self::EmptyAtomicUpdateRange => {
                f.write_str("global-memory atomic update requires a non-empty byte range")
            }
            Self::AtomicUpdateLengthMismatch {
                expected_byte_len,
                actual_byte_len,
            } => write!(
                f,
                "global-memory atomic update produced {actual_byte_len} bytes for a {expected_byte_len}-byte range"
            ),
            Self::DeferredWriteMaskLengthMismatch { byte_len, mask_len } => write!(
                f,
                "deferred global write has {byte_len} data bytes but {mask_len} mask bytes"
            ),
            Self::DeferredReductionLengthMismatch {
                expected_byte_len,
                actual_byte_len,
            } => write!(
                f,
                "deferred global reduction has {actual_byte_len} source bytes, expected {expected_byte_len}"
            ),
            Self::InitializationLengthMismatch {
                byte_len,
                validity_len,
            } => write!(
                f,
                "allocation initializer has {byte_len} data bytes but {validity_len} validity bytes"
            ),
            Self::InvalidValidityByte { index, value } => write!(
                f,
                "allocation initializer validity byte {index} is {value}, expected 0 or 1"
            ),
            Self::UnknownAllocation { allocation } => {
                write!(f, "unknown global-memory {allocation}")
            }
            Self::ViewOutOfBounds {
                allocation,
                allocation_byte_len,
                byte_offset,
                byte_len,
            } => write!(
                f,
                "view [{byte_offset}, {byte_offset}+{byte_len}) exceeds {allocation} length {allocation_byte_len}"
            ),
            Self::AccessOutOfBounds {
                allocation,
                view_byte_len,
                byte_offset,
                byte_len,
            } => write!(
                f,
                "access [{byte_offset}, {byte_offset}+{byte_len}) exceeds {allocation} view length {view_byte_len}"
            ),
            Self::InvalidRead {
                allocation,
                byte_offset,
                byte_len,
                first_invalid_byte,
            } => write!(
                f,
                "read [{byte_offset}, {byte_offset}+{byte_len}) from {allocation} includes invalid byte {first_invalid_byte}"
            ),
            Self::OffsetOverflow => f.write_str("global-memory byte offset overflow"),
        }
    }
}

impl Error for MemoryError {}

#[derive(Clone, Copy)]
enum RangeKind {
    View,
    Access,
}

fn validate_view(view: &BufferView, allocation_byte_len: usize) -> Result<(), MemoryError> {
    validate_range(
        view.allocation,
        allocation_byte_len,
        view.byte_offset,
        view.byte_len,
        RangeKind::View,
    )
}

fn resolve_access(
    view: &BufferView,
    allocation_byte_len: usize,
    byte_offset: usize,
    byte_len: usize,
) -> Result<usize, MemoryError> {
    validate_view(view, allocation_byte_len)?;
    validate_range(
        view.allocation,
        view.byte_len,
        byte_offset,
        byte_len,
        RangeKind::Access,
    )?;
    view.byte_offset
        .checked_add(byte_offset)
        .ok_or(MemoryError::OffsetOverflow)
}

fn validate_range(
    allocation: AllocationId,
    container_byte_len: usize,
    byte_offset: usize,
    byte_len: usize,
    kind: RangeKind,
) -> Result<(), MemoryError> {
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or(MemoryError::OffsetOverflow)?;
    if end <= container_byte_len {
        return Ok(());
    }
    match kind {
        RangeKind::View => Err(MemoryError::ViewOutOfBounds {
            allocation,
            allocation_byte_len: container_byte_len,
            byte_offset,
            byte_len,
        }),
        RangeKind::Access => Err(MemoryError::AccessOutOfBounds {
            allocation,
            view_byte_len: container_byte_len,
            byte_offset,
            byte_len,
        }),
    }
}

fn validate_initialized_locked(
    allocation_id: AllocationId,
    targets: &[StripeTarget],
    data: &StripeWriteData<'_>,
    absolute: usize,
    byte_len: usize,
) -> Result<(), MemoryError> {
    if let Some(relative) = first_invalid_locked(targets, data, allocation_id, absolute, byte_len) {
        let first_invalid_byte = absolute + relative;
        return Err(MemoryError::InvalidRead {
            allocation: allocation_id,
            byte_offset: absolute,
            byte_len,
            first_invalid_byte,
        });
    }
    Ok(())
}

// The allocation-global invalid count says nothing about one range; a fully
// initialized range must not pay the per-byte validity path just because
// other bytes of the allocation stay uninitialized (for example padding).
#[inline]
fn private_range_fully_valid(
    allocation: &OwnerPrivateData,
    absolute: usize,
    byte_len: usize,
) -> bool {
    allocation.invalid_byte_count == 0
        || allocation
            .valid
            .range_fully_valid(absolute, absolute + byte_len)
}

#[inline]
fn validate_private_initialized(
    allocation_id: AllocationId,
    allocation: &OwnerPrivateData,
    absolute: usize,
    byte_len: usize,
) -> Result<(), MemoryError> {
    if allocation.invalid_byte_count == 0 {
        return Ok(());
    }
    let Some(first_invalid_byte) = allocation
        .valid
        .first_invalid_in(absolute, absolute + byte_len)
    else {
        return Ok(());
    };
    Err(MemoryError::InvalidRead {
        allocation: allocation_id,
        byte_offset: absolute,
        byte_len,
        first_invalid_byte,
    })
}

fn copy_private_bytes_zero_filled(
    allocation_id: AllocationId,
    allocation: &OwnerPrivateData,
    absolute: usize,
    target: &mut [u8],
) -> Option<UninitializedReadReview> {
    let source = &allocation.bytes[absolute..absolute + target.len()];
    if private_range_fully_valid(allocation, absolute, target.len()) {
        target.copy_from_slice(source);
        return None;
    }
    let mut first_invalid = None;
    for (relative, (destination, source)) in target.iter_mut().zip(source).enumerate() {
        let valid = allocation.valid.get(absolute + relative);
        if !valid {
            first_invalid.get_or_insert(relative);
        }
        *destination = if valid { *source } else { 0 };
    }
    first_invalid.map(|relative| UninitializedReadReview {
        source: None,
        allocation: allocation_id,
        byte_offset: absolute,
        byte_len: target.len(),
        first_uninitialized_byte: absolute + relative,
    })
}

#[inline]
fn mark_private_initialized(allocation: &mut OwnerPrivateData, absolute: usize, byte_len: usize) {
    if allocation.invalid_byte_count == 0 || byte_len == 0 {
        return;
    }
    let newly_initialized = allocation
        .valid
        .count_invalid_in(absolute, absolute + byte_len);
    allocation
        .valid
        .set_range(absolute, absolute + byte_len, true);
    allocation.invalid_byte_count = allocation
        .invalid_byte_count
        .checked_sub(newly_initialized)
        .expect("private allocation invalid-byte accounting underflow");
}

#[inline]
fn mark_private_invalid(allocation: &mut OwnerPrivateData, absolute: usize, byte_len: usize) {
    if byte_len == 0 {
        return;
    }
    let newly_invalid = byte_len
        - allocation
            .valid
            .count_invalid_in(absolute, absolute + byte_len);
    allocation
        .valid
        .set_range(absolute, absolute + byte_len, false);
    allocation.invalid_byte_count = allocation
        .invalid_byte_count
        .checked_add(newly_invalid)
        .expect("private allocation invalid-byte accounting overflow");
}

fn stripe_targets_for_ranges(
    allocation: AllocationId,
    allocation_ref: Arc<Allocation>,
    write_ranges: &[(usize, usize)],
) -> Vec<StripeTarget> {
    let mut stripe_indices = BTreeSet::new();
    let shared = allocation_ref
        .shared()
        .expect("stripe targets require shared backing");
    for &(absolute, byte_len) in write_ranges {
        if byte_len == 0 {
            continue;
        }
        let last = absolute + byte_len - 1;
        for stripe_index in shared.stripe_index(absolute)..=shared.stripe_index(last) {
            stripe_indices.insert(stripe_index);
        }
    }
    stripe_indices
        .into_iter()
        .map(|stripe_index| StripeTarget {
            key: StripeKey {
                allocation,
                stripe_index,
            },
            allocation: allocation_ref.clone(),
        })
        .collect()
}

fn stripe_targets_for_write_ranges(ranges: &[MemoryWriteRange]) -> Vec<StripeTarget> {
    let mut targets = Vec::new();
    for range in ranges {
        let shared = range
            .allocation_ref
            .shared()
            .expect("write watches require shared backing");
        let last = range.absolute_byte_offset + range.byte_len - 1;
        for stripe_index in
            shared.stripe_index(range.absolute_byte_offset)..=shared.stripe_index(last)
        {
            targets.push(StripeTarget {
                key: StripeKey {
                    allocation: range.allocation,
                    stripe_index,
                },
                allocation: range.allocation_ref.clone(),
            });
        }
    }
    targets.sort_unstable_by_key(StripeTarget::key);
    targets.dedup_by_key(|target| target.key);
    targets
}

/// Canonicalize the exact union of one write transaction's byte ranges.
///
/// Write watches observe whether their range overlaps the transaction, not
/// how many adjacent element writes formed it. Merging overlapping or
/// contiguous ranges therefore preserves wakeups and touched stripes while
/// avoiding one retained `Arc` and repeated overlap work per scalar element.
fn coalesce_memory_write_ranges(mut ranges: Vec<MemoryWriteRange>) -> Vec<MemoryWriteRange> {
    ranges.sort_unstable();
    let mut coalesced: Vec<MemoryWriteRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        let Some(previous) = coalesced.last_mut() else {
            coalesced.push(range);
            continue;
        };
        let previous_end = previous
            .absolute_byte_offset
            .checked_add(previous.byte_len)
            .expect("validated write range end fits usize");
        let range_end = range
            .absolute_byte_offset
            .checked_add(range.byte_len)
            .expect("validated write range end fits usize");
        if previous.arena_id == range.arena_id
            && previous.allocation == range.allocation
            && range.absolute_byte_offset <= previous_end
        {
            previous.byte_len = previous_end
                .max(range_end)
                .checked_sub(previous.absolute_byte_offset)
                .expect("coalesced range end follows its start");
        } else {
            coalesced.push(range);
        }
    }
    coalesced
}

fn lock_stripe_targets(targets: &[StripeTarget]) -> Vec<MutexGuard<'_, ()>> {
    targets
        .iter()
        .map(|target| lock_mutex(&target.stripe().state))
        .collect()
}

fn begin_stripe_writes(targets: &[StripeTarget]) -> Result<StripeWriteData<'_>, MemoryError> {
    let guards = targets
        .iter()
        .map(|target| target.stripe().write_data(target.key.allocation))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StripeWriteData { guards })
}

fn stripe_data_guard_index(
    targets: &[StripeTarget],
    allocation: AllocationId,
    stripe_index: usize,
) -> usize {
    let key = StripeKey {
        allocation,
        stripe_index,
    };
    targets
        .binary_search_by_key(&key, StripeTarget::key)
        .expect("every locked byte belongs to a target stripe")
}

fn locked_byte(
    targets: &[StripeTarget],
    data: &StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
) -> (u8, bool) {
    let stripe_index = absolute / MEMORY_STRIPE_BYTES;
    let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
    let stripe = targets[guard_index].stripe();
    let offset = absolute - stripe.byte_start;
    (
        data.guards[guard_index].bytes[offset],
        data.guards[guard_index].valid.is_all_valid()
            || data.guards[guard_index].valid[offset] != 0,
    )
}

fn store_locked_byte(
    targets: &[StripeTarget],
    data: &mut StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    byte: u8,
) -> Result<(), MemoryError> {
    let stripe_index = absolute / MEMORY_STRIPE_BYTES;
    let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
    let stripe = targets[guard_index].stripe();
    let offset = absolute - stripe.byte_start;
    data.guards[guard_index].record_write(allocation, absolute, offset, offset + 1)?;
    data.guards[guard_index].bytes[offset] = byte;
    data.guards[guard_index]
        .valid
        .mark_valid(offset, offset + 1);
    Ok(())
}

fn read_locked_bytes(
    targets: &[StripeTarget],
    data: &StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    target: &mut [u8],
) {
    let end = absolute + target.len();
    let first_stripe = absolute / MEMORY_STRIPE_BYTES;
    let last_stripe = (end - 1) / MEMORY_STRIPE_BYTES;
    for stripe_index in first_stripe..=last_stripe {
        let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
        let stripe = targets[guard_index].stripe();
        let overlap_start = absolute.max(stripe.byte_start);
        let overlap_end = end.min(stripe.byte_start + stripe.byte_len);
        target[overlap_start - absolute..overlap_end - absolute].copy_from_slice(
            &data.guards[guard_index].bytes
                [overlap_start - stripe.byte_start..overlap_end - stripe.byte_start],
        );
    }
}

fn first_invalid_locked(
    targets: &[StripeTarget],
    data: &StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    byte_len: usize,
) -> Option<usize> {
    let end = absolute + byte_len;
    let first_stripe = absolute / MEMORY_STRIPE_BYTES;
    let last_stripe = (end - 1) / MEMORY_STRIPE_BYTES;
    for stripe_index in first_stripe..=last_stripe {
        let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
        let stripe = targets[guard_index].stripe();
        let overlap_start = absolute.max(stripe.byte_start);
        let overlap_end = end.min(stripe.byte_start + stripe.byte_len);
        if !data.guards[guard_index].valid.is_all_valid() {
            if let Some(relative) = data.guards[guard_index].valid
                [overlap_start - stripe.byte_start..overlap_end - stripe.byte_start]
                .iter()
                .position(|valid| *valid == 0)
            {
                return Some(overlap_start - absolute + relative);
            }
        }
    }
    None
}

fn validate_readonly_write_locked(
    targets: &[StripeTarget],
    data: &StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    byte_len: usize,
) -> Result<(), MemoryError> {
    if byte_len == 0 {
        return Ok(());
    }
    let end = absolute + byte_len;
    for stripe_index in absolute / MEMORY_STRIPE_BYTES..=(end - 1) / MEMORY_STRIPE_BYTES {
        let index = stripe_data_guard_index(targets, allocation, stripe_index);
        let stripe = targets[index].stripe();
        if let Some(history) = &data.guards[index].readonly_proxy {
            let start = absolute.max(stripe.byte_start) - stripe.byte_start;
            let stop = end.min(stripe.byte_start + stripe.byte_len) - stripe.byte_start;
            if history.observed.any_valid_in(start, stop) {
                return Err(MemoryError::ReadonlyProxyWriteConflict {
                    allocation,
                    byte_offset: absolute,
                    byte_len,
                });
            }
        }
    }
    Ok(())
}

fn store_initialized_locked(
    targets: &[StripeTarget],
    data: &mut StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    bytes: &[u8],
) -> Result<(), MemoryError> {
    validate_readonly_write_locked(targets, data, allocation, absolute, bytes.len())?;
    let end = absolute + bytes.len();
    let first_stripe = absolute / MEMORY_STRIPE_BYTES;
    let last_stripe = (end - 1) / MEMORY_STRIPE_BYTES;
    for stripe_index in first_stripe..=last_stripe {
        let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
        let stripe = targets[guard_index].stripe();
        let overlap_start = absolute.max(stripe.byte_start);
        let overlap_end = end.min(stripe.byte_start + stripe.byte_len);
        let source_start = overlap_start - absolute;
        let source_end = overlap_end - absolute;
        let destination_start = overlap_start - stripe.byte_start;
        let destination_end = overlap_end - stripe.byte_start;
        data.guards[guard_index].record_write(
            allocation,
            overlap_start,
            destination_start,
            destination_end,
        )?;
        data.guards[guard_index].bytes[destination_start..destination_end]
            .copy_from_slice(&bytes[source_start..source_end]);
        data.guards[guard_index]
            .valid
            .mark_valid(destination_start, destination_end);
    }
    Ok(())
}

fn invalidate_locked(
    targets: &[StripeTarget],
    data: &mut StripeWriteData<'_>,
    allocation: AllocationId,
    absolute: usize,
    byte_len: usize,
) -> Result<bool, MemoryError> {
    validate_readonly_write_locked(targets, data, allocation, absolute, byte_len)?;
    let end = absolute + byte_len;
    let first_stripe = absolute / MEMORY_STRIPE_BYTES;
    let last_stripe = (end - 1) / MEMORY_STRIPE_BYTES;
    let mut changed = false;
    for stripe_index in first_stripe..=last_stripe {
        let guard_index = stripe_data_guard_index(targets, allocation, stripe_index);
        let stripe = targets[guard_index].stripe();
        let overlap_start = absolute.max(stripe.byte_start);
        let overlap_end = end.min(stripe.byte_start + stripe.byte_len);
        data.guards[guard_index].record_write(
            allocation,
            overlap_start,
            overlap_start - stripe.byte_start,
            overlap_end - stripe.byte_start,
        )?;
        let valid = &mut data.guards[guard_index].valid
            [overlap_start - stripe.byte_start..overlap_end - stripe.byte_start];
        changed |= valid.iter().any(|valid| *valid != 0);
        valid.fill(0);
    }
    Ok(changed)
}

fn write_single_stripe_bytes(
    shared: &SharedAllocationBacking,
    stripe_index: usize,
    allocation: AllocationId,
    absolute: usize,
    bytes: &[u8],
    track_semantic_progress: bool,
) -> Result<bool, MemoryError> {
    let stripe = &shared.stripes[stripe_index];
    let state = lock_mutex(&stripe.state);
    let mut data = stripe.write_data(allocation)?;
    let stripe_offset = absolute - stripe.byte_start;
    data.record_write(
        allocation,
        absolute,
        stripe_offset,
        stripe_offset + bytes.len(),
    )?;
    let changed = track_semantic_progress
        && (data.bytes[stripe_offset..stripe_offset + bytes.len()] != *bytes
            || !data.valid.is_all_valid()
                && data.valid[stripe_offset..stripe_offset + bytes.len()]
                    .iter()
                    .any(|valid| *valid == 0));
    data.bytes[stripe_offset..stripe_offset + bytes.len()].copy_from_slice(bytes);
    data.valid
        .mark_valid(stripe_offset, stripe_offset + bytes.len());
    drop(data);
    drop(state);
    Ok(changed)
}

#[derive(Clone, Copy)]
enum SnapshotValidity {
    Ignore,
    RequireInitialized,
    ZeroFill,
}

struct StableSnapshot {
    bytes: Vec<u8>,
    first_invalid: Option<usize>,
}

fn stable_snapshot(
    allocation: &SharedAllocationBacking,
    absolute: usize,
    byte_len: usize,
    validity: SnapshotValidity,
) -> StableSnapshot {
    if byte_len == 0 {
        return StableSnapshot {
            bytes: Vec::new(),
            first_invalid: None,
        };
    }
    let mut bytes = vec![0; byte_len];
    let first_invalid = allocation.snapshot_into(absolute, &mut bytes, validity);
    StableSnapshot {
        bytes,
        first_invalid,
    }
}

fn stable_snapshot_into(
    allocation: &SharedAllocationBacking,
    absolute: usize,
    target: &mut [u8],
    validity: SnapshotValidity,
) -> Option<usize> {
    allocation.snapshot_into(absolute, target, validity)
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_rwlock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_rwlock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn element_offset(element_index: usize, element_size: usize) -> Result<usize, MemoryError> {
    element_index
        .checked_mul(element_size)
        .ok_or(MemoryError::OffsetOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{mpsc, Barrier};
    use std::task::Wake;
    use std::thread;
    use std::time::Duration;

    // Gated alongside its sole user, the `analysis-core` semantic-progress
    // test; a default-features build would otherwise see it as dead code.
    #[cfg(feature = "analysis-core")]
    struct RecordingWake {
        id: usize,
        events: Arc<Mutex<Vec<usize>>>,
    }

    #[cfg(feature = "analysis-core")]
    impl Wake for RecordingWake {
        fn wake(self: Arc<Self>) {
            self.events.lock().unwrap().push(self.id);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.events.lock().unwrap().push(self.id);
        }
    }

    #[test]
    fn observed_address_resolution_has_one_allocation_owner() {
        let memory = GlobalMemory::new();
        let first = memory
            .full_view(memory.allocate_zeroed(16).unwrap())
            .unwrap();
        let second = memory
            .full_view(memory.allocate_zeroed(16).unwrap())
            .unwrap();
        first.bind_observed_allocation_address(0x1000).unwrap();
        assert_eq!(
            memory.observed_address_owner(0x1010).unwrap(),
            Some((first.clone(), 16))
        );
        assert!(memory.observed_address_owner(0).unwrap().is_none());
        assert!(memory.observed_address_owner(0x1020).unwrap().is_none());
        second.bind_observed_allocation_address(0x1010).unwrap();
        assert_eq!(
            memory.observed_address_owner(0x1010).unwrap(),
            Some((second.clone(), 0))
        );
        assert_eq!(
            memory.observed_address_owner(0x1018).unwrap(),
            Some((second, 8))
        );
        let overlapping = memory
            .full_view(memory.allocate_zeroed(16).unwrap())
            .unwrap();
        overlapping
            .bind_observed_allocation_address(0x1008)
            .unwrap();
        assert!(memory
            .observed_address_owner(0x100c)
            .unwrap_err()
            .to_string()
            .contains("ambiguous_global_address_owner"));
        let memory = GlobalMemory::new();
        memory
            .full_view(memory.allocate_zeroed(16).unwrap())
            .unwrap()
            .bind_observed_allocation_address(u64::MAX - 8)
            .unwrap();
        assert!(memory
            .observed_address_owner(u64::MAX)
            .unwrap_err()
            .to_string()
            .contains("global_address_range_overflow"));
    }

    #[test]
    fn readonly_proxy_covers_atomic_and_deferred_same_value_writes() {
        let memory = GlobalMemory::new();
        memory.set_readonly_proxy_tracking(true).unwrap();
        let allocation = memory.allocate_zeroed(16).unwrap();
        let view = memory.full_view(allocation).unwrap();
        memory.observe_readonly_proxy(&view, 4, 4).unwrap();
        assert!(matches!(
            memory.invalidate(&view, 4, 4),
            Err(MemoryError::ReadonlyProxyWriteConflict { .. })
        ));
        for write_first in [false, true] {
            for deferred in [false, true] {
                memory.set_readonly_proxy_tracking(true).unwrap();
                let write = || {
                    if deferred {
                        let pending = memory.defer_write_bytes(&view, 4, vec![0; 4])?;
                        publish_deferred_global_writes(&[pending])
                    } else {
                        memory.atomic_update_bytes::<_, MemoryError>(&view, 4, 4, |old| {
                            Ok(((), old.to_vec()))
                        })
                    }
                };
                if write_first {
                    write().unwrap();
                    assert!(matches!(
                        memory.observe_readonly_proxy(&view, 4, 4),
                        Err(MemoryError::ReadonlyProxyWriteConflict { .. })
                    ));
                } else {
                    memory.observe_readonly_proxy(&view, 4, 4).unwrap();
                    assert!(matches!(
                        write(),
                        Err(MemoryError::ReadonlyProxyWriteConflict { .. })
                    ));
                }
            }
        }
        memory.set_readonly_proxy_tracking(true).unwrap();
        memory.observe_readonly_proxy(&view, 4, 4).unwrap();
        let writes = [
            memory.defer_write_bytes(&view, 0, vec![1; 4]).unwrap(),
            memory.defer_write_bytes(&view, 4, vec![2; 4]).unwrap(),
        ];
        assert!(matches!(
            publish_deferred_global_writes(&writes),
            Err(MemoryError::ReadonlyProxyWriteConflict { .. })
        ));
        assert_eq!(memory.read_bytes(&view, 0, 8).unwrap(), vec![0; 8]);
    }

    #[test]
    fn allocation_ids_are_deterministic_per_memory_arena() {
        let first = GlobalMemory::new();
        assert_eq!(first.allocate_zeroed(1).unwrap().as_u64(), 0);
        assert_eq!(first.allocate_zeroed(1).unwrap().as_u64(), 1);

        let shared_clone = first.clone();
        assert_eq!(shared_clone.allocate_zeroed(1).unwrap().as_u64(), 2);

        let pristine_replay = GlobalMemory::new();
        assert_eq!(pristine_replay.allocate_zeroed(1).unwrap().as_u64(), 0);
        assert_eq!(pristine_replay.allocate_zeroed(1).unwrap().as_u64(), 1);
    }

    #[test]
    fn initialized_writes_preserve_the_all_valid_stripe_sentinel() {
        let memory = GlobalMemory::new();
        let allocation = memory
            .allocate_from_bytes_all_valid(vec![0; MEMORY_STRIPE_BYTES + 8])
            .unwrap();
        let view = memory.full_view(allocation).unwrap();
        let allocation_ref = memory.allocation(allocation).unwrap();
        let shared = allocation_ref.shared().unwrap();

        assert!(shared
            .stripes
            .iter()
            .all(|stripe| stripe.read_data().valid.is_all_valid()));
        memory.write_bytes(&view, 2, &[1, 2, 3, 4]).unwrap();
        assert!(shared
            .stripes
            .iter()
            .all(|stripe| stripe.read_data().valid.is_all_valid()));
        assert_eq!(memory.read_bytes(&view, 2, 4).unwrap(), [1, 2, 3, 4]);

        memory.invalidate(&view, 3, 1).unwrap();
        assert_eq!(
            memory.byte_validity(&view, 2, 3).unwrap(),
            [true, false, true]
        );
        memory.write_bytes(&view, 3, &[9]).unwrap();
        assert_eq!(memory.read_bytes(&view, 2, 3).unwrap(), [1, 9, 3]);
    }

    #[test]
    fn overlapping_views_share_one_physical_backing() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(16).unwrap();
        let full = memory.full_view(allocation).unwrap();
        let alias = memory.view(allocation, 4, 8).unwrap();
        let nested = memory.subview(&full, 4, 8).unwrap();

        memory.write_f32_le(&alias, 0, 3.5).unwrap();
        assert_eq!(memory.read_f32_le(&full, 1).unwrap(), 3.5);
        assert_eq!(memory.read_f32_le(&nested, 0).unwrap(), 3.5);

        memory.write_f32_le(&full, 2, -7.25).unwrap();
        assert_eq!(memory.read_f32_le(&alias, 1).unwrap(), -7.25);
        assert_eq!(
            memory.read_bytes(&full, 8, 4).unwrap(),
            (-7.25_f32).to_le_bytes()
        );

        let shared_clone = memory.clone();
        shared_clone.write_f32_le(&nested, 1, 11.0).unwrap();
        assert_eq!(memory.read_f32_le(&full, 2).unwrap(), 11.0);
    }

    #[test]
    fn batch_write_validates_before_publishing_and_preserves_order() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(8).unwrap();
        let view = memory.full_view(allocation).unwrap();
        memory
            .write_bytes_batch(&view, [(4, &[4_u8, 5][..]), (1, &[1_u8, 2, 3][..])])
            .unwrap();
        assert_eq!(
            memory.read_bytes(&view, 0, 8).unwrap(),
            [0, 1, 2, 3, 4, 5, 0, 0]
        );
        let error = memory
            .write_bytes_batch(&view, [(0, &[9_u8][..]), (8, &[10_u8][..])])
            .unwrap_err();
        assert!(matches!(error, MemoryError::AccessOutOfBounds { .. }));
        assert_eq!(
            memory.read_bytes(&view, 0, 8).unwrap(),
            [0, 1, 2, 3, 4, 5, 0, 0]
        );
    }

    #[test]
    fn contiguous_deferred_replacements_coalesce_exactly() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(12).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let mut writes = Vec::new();

        memory
            .defer_or_extend_write_bytes(&mut writes, &view, 0, &[1, 2])
            .unwrap();
        memory
            .defer_or_extend_write_bytes(&mut writes, &view, 2, &[3, 4])
            .unwrap();
        memory
            .defer_or_extend_write_bytes(&mut writes, &view, 4, &[5, 6])
            .unwrap();
        memory
            .defer_or_extend_write_bytes(&mut writes, &view, 6, &[7, 8])
            .unwrap();
        memory
            .defer_or_extend_write_bytes(&mut writes, &view, 10, &[11, 12])
            .unwrap();

        assert_eq!(writes.len(), 2);
        publish_deferred_global_writes(&writes).unwrap();
        assert_eq!(
            memory.read_bytes(&view, 0, 12).unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 11, 12]
        );
    }

    #[test]
    fn deferred_masked_writes_read_the_destination_at_publish_time() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_from_bytes([0xa0]).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let low = memory
            .defer_masked_write_bytes(&view, 0, vec![0x05], vec![0x0f])
            .unwrap();
        let high = memory
            .defer_masked_write_bytes(&view, 0, vec![0x30], vec![0xf0])
            .unwrap();

        assert_eq!(memory.read_bytes(&view, 0, 1).unwrap(), [0xa0]);
        low.publish().unwrap();
        assert_eq!(memory.read_bytes(&view, 0, 1).unwrap(), [0xa5]);
        high.publish().unwrap();
        assert_eq!(memory.read_bytes(&view, 0, 1).unwrap(), [0x35]);
    }

    #[test]
    fn deferred_reduction_variants_match_ptx_scalar_semantics() {
        let f16 = |value: f32| f32_to_fp16_bits(value).to_le_bytes().to_vec();
        let bf16 = |value: f32| f32_to_bf16_bits(value).to_le_bytes().to_vec();
        let cases = vec![
            (
                DeferredGlobalReduction::AddU32,
                7_u32.to_le_bytes().to_vec(),
                9_u32.to_le_bytes().to_vec(),
                16_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AddI32,
                (-7_i32).to_le_bytes().to_vec(),
                3_i32.to_le_bytes().to_vec(),
                (-4_i32).to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AddU64,
                (u64::MAX - 2).to_le_bytes().to_vec(),
                5_u64.to_le_bytes().to_vec(),
                2_u64.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AddF32Ftz,
                1.5_f32.to_le_bytes().to_vec(),
                2.25_f32.to_le_bytes().to_vec(),
                3.75_f32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AddF16,
                f16(1.5),
                f16(2.0),
                f16(3.5),
            ),
            (
                DeferredGlobalReduction::AddBf16,
                bf16(1.5),
                bf16(2.0),
                bf16(3.5),
            ),
            (
                DeferredGlobalReduction::MinU32,
                7_u32.to_le_bytes().to_vec(),
                9_u32.to_le_bytes().to_vec(),
                7_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MinI32,
                (-7_i32).to_le_bytes().to_vec(),
                3_i32.to_le_bytes().to_vec(),
                (-7_i32).to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MinU64,
                11_u64.to_le_bytes().to_vec(),
                5_u64.to_le_bytes().to_vec(),
                5_u64.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MinI64,
                (-11_i64).to_le_bytes().to_vec(),
                (-5_i64).to_le_bytes().to_vec(),
                (-11_i64).to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MinF16,
                f16(1.5),
                f16(-2.0),
                f16(-2.0),
            ),
            (
                DeferredGlobalReduction::MinBf16,
                bf16(1.5),
                bf16(-2.0),
                bf16(-2.0),
            ),
            (
                DeferredGlobalReduction::MaxU32,
                7_u32.to_le_bytes().to_vec(),
                9_u32.to_le_bytes().to_vec(),
                9_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MaxI32,
                (-7_i32).to_le_bytes().to_vec(),
                3_i32.to_le_bytes().to_vec(),
                3_i32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MaxU64,
                11_u64.to_le_bytes().to_vec(),
                5_u64.to_le_bytes().to_vec(),
                11_u64.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MaxI64,
                (-11_i64).to_le_bytes().to_vec(),
                (-5_i64).to_le_bytes().to_vec(),
                (-5_i64).to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::MaxF16,
                f16(1.5),
                f16(-2.0),
                f16(1.5),
            ),
            (
                DeferredGlobalReduction::MaxBf16,
                bf16(1.5),
                bf16(-2.0),
                bf16(1.5),
            ),
            (
                DeferredGlobalReduction::IncU32,
                2_u32.to_le_bytes().to_vec(),
                4_u32.to_le_bytes().to_vec(),
                3_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::DecU32,
                0_u32.to_le_bytes().to_vec(),
                4_u32.to_le_bytes().to_vec(),
                4_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AndB32,
                0xf0f0_00ff_u32.to_le_bytes().to_vec(),
                0x0ff0_f00f_u32.to_le_bytes().to_vec(),
                0x00f0_000f_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::AndB64,
                0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
                0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
                0x00f0_000f_00f0_000f_u64.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::OrB32,
                0xf0f0_00ff_u32.to_le_bytes().to_vec(),
                0x0ff0_f00f_u32.to_le_bytes().to_vec(),
                0xfff0_f0ff_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::OrB64,
                0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
                0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
                0xfff0_f0ff_fff0_f0ff_u64.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::XorB32,
                0xf0f0_00ff_u32.to_le_bytes().to_vec(),
                0x0ff0_f00f_u32.to_le_bytes().to_vec(),
                0xff00_f0f0_u32.to_le_bytes().to_vec(),
            ),
            (
                DeferredGlobalReduction::XorB64,
                0xf0f0_00ff_f0f0_00ff_u64.to_le_bytes().to_vec(),
                0x0ff0_f00f_0ff0_f00f_u64.to_le_bytes().to_vec(),
                0xff00_f0f0_ff00_f0f0_u64.to_le_bytes().to_vec(),
            ),
        ];

        for (operation, current, source, expected) in cases {
            assert_eq!(
                operation.apply(&current, &source),
                expected,
                "reduction {operation:?}"
            );
        }
    }

    #[test]
    fn deferred_reduction_reads_destination_at_publish_time() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_from_bytes(1_u32.to_le_bytes()).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let deferred = memory
            .defer_reduction_write_bytes(
                &view,
                0,
                2_u32.to_le_bytes().to_vec(),
                DeferredGlobalReduction::AddU32,
            )
            .unwrap();

        memory.write_bytes(&view, 0, &10_u32.to_le_bytes()).unwrap();
        deferred.publish().unwrap();

        assert_eq!(
            u32::from_le_bytes(memory.read_bytes(&view, 0, 4).unwrap().try_into().unwrap()),
            12
        );
    }

    #[test]
    fn deferred_add_f32_reduction_flushes_subnormal_inputs_and_outputs() {
        let smallest_normal = f32::MIN_POSITIVE;
        let smallest_subnormal = f32::from_bits(1);

        assert_eq!(
            DeferredGlobalReduction::AddF32Ftz
                .apply(
                    &smallest_normal.to_le_bytes(),
                    &(-smallest_subnormal).to_le_bytes(),
                )
                .as_slice(),
            smallest_normal.to_le_bytes()
        );
        assert_eq!(
            DeferredGlobalReduction::AddF32Ftz
                .apply(
                    &smallest_subnormal.to_le_bytes(),
                    &smallest_subnormal.to_le_bytes(),
                )
                .as_slice(),
            0.0_f32.to_le_bytes()
        );
    }

    #[test]
    fn byte_initialization_is_little_endian_and_ids_do_not_cross_arenas() {
        let first = GlobalMemory::new();
        let bytes = [1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat();
        let allocation = first.allocate_from_bytes(bytes).unwrap();
        let view = first.full_view(allocation).unwrap();
        assert_eq!(first.read_f32_le(&view, 0).unwrap(), 1.25);
        assert_eq!(first.read_f32_le(&view, 1).unwrap(), -2.5);

        let second = GlobalMemory::new();
        second.allocate_zeroed(8).unwrap();
        assert_eq!(
            second.read_f32_le(&view, 0),
            Err(MemoryError::UnknownAllocation { allocation })
        );
    }

    #[test]
    fn uninitialized_and_invalidated_reads_fail_closed() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_uninitialized(8).unwrap();
        let view = memory.full_view(allocation).unwrap();

        assert_eq!(
            memory.read_f32_le(&view, 0),
            Err(MemoryError::InvalidRead {
                allocation,
                byte_offset: 0,
                byte_len: 4,
                first_invalid_byte: 0,
            })
        );
        memory.write_f32_le(&view, 0, 2.0).unwrap();
        assert_eq!(memory.read_f32_le(&view, 0).unwrap(), 2.0);
        memory.invalidate(&view, 2, 1).unwrap();
        assert_eq!(
            memory.read_f32_le(&view, 0),
            Err(MemoryError::InvalidRead {
                allocation,
                byte_offset: 0,
                byte_len: 4,
                first_invalid_byte: 2,
            })
        );
    }

    #[test]
    fn owner_private_write_session_keeps_invalid_byte_count_exact() {
        let memory = GlobalMemory::new_owner_private();
        let allocation = memory.allocate_uninitialized(4).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let invalid_byte_count = || {
            memory
                .allocation_for_view(&view)
                .unwrap()
                .with_private(allocation, |data| data.invalid_byte_count)
                .unwrap()
        };

        assert_eq!(invalid_byte_count(), 4);
        memory
            .with_owner_private_write_session(&view, |session| {
                session.write_bytes_prevalidated(0, &[1, 2]);
            })
            .unwrap();
        assert_eq!(invalid_byte_count(), 2);

        memory
            .with_owner_private_write_session(&view, |session| {
                session.write_strided_batch_prevalidated(&[2, 3], &[3, 4], 1, 1, 2);
            })
            .unwrap();
        assert_eq!(invalid_byte_count(), 0);
        assert_eq!(memory.read_bytes(&view, 0, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn private_validity_handles_word_boundaries_exactly() {
        for len in [1, 63, 64, 65, 130] {
            let mut validity = PrivateValidity::new_filled(len, false);
            assert_eq!(validity.count_invalid_in(0, len), len);
            assert_eq!(validity.first_invalid_in(0, len), Some(0));
            assert!(!validity.any_valid_in(0, len));

            validity.set_range(0, len, true);
            assert_eq!(validity.count_invalid_in(0, len), 0);
            assert!(validity.range_fully_valid(0, len));

            if len > 2 {
                validity.set_range(1, 2, false);
                assert!(!validity.get(1));
                assert!(validity.get(0));
                assert!(validity.get(2));
                assert_eq!(validity.first_invalid_in(0, len), Some(1));
                assert_eq!(validity.first_invalid_in(2, len), None);
                assert!(validity.range_fully_valid(2, len));
                assert_eq!(validity.count_invalid_in(0, len), 1);
                assert!(validity.any_valid_in(0, len));
                assert_eq!(
                    validity.to_bools(0, 3.min(len)),
                    [true, false, true][..3.min(len)]
                );
            }
        }

        let mut validity = PrivateValidity::new_filled(130, true);
        validity.set_range(63, 65, false);
        assert_eq!(validity.first_invalid_in(0, 130), Some(63));
        assert_eq!(validity.first_invalid_in(64, 130), Some(64));
        assert_eq!(validity.first_invalid_in(65, 130), None);
        assert_eq!(validity.count_invalid_in(0, 130), 2);
        assert_eq!(validity.count_invalid_in(64, 130), 1);
        assert!(validity.range_fully_valid(0, 63));
        assert!(!validity.range_fully_valid(0, 64));
        assert!(validity.any_valid_in(63, 66));
        assert!(!validity.any_valid_in(63, 65));
        let from_bools = PrivateValidity::from_bools(&validity.to_bools(0, 130));
        assert_eq!(from_bools.count_invalid_in(0, 130), 2);
        assert_eq!(from_bools.first_invalid_in(0, 130), Some(63));
    }

    #[test]
    fn owner_private_range_reads_ignore_invalid_bytes_outside_the_range() {
        let memory = GlobalMemory::new_owner_private();
        let allocation = memory.allocate_uninitialized(8).unwrap();
        let view = memory.full_view(allocation).unwrap();

        memory.write_bytes(&view, 2, &[7, 8, 9, 10]).unwrap();

        assert_eq!(memory.read_bytes(&view, 2, 4).unwrap(), [7, 8, 9, 10]);
        assert_eq!(
            memory.read_bytes_zero_filled(&view, 2, 4).unwrap(),
            [7, 8, 9, 10]
        );
        let mut exact = [99_u8; 4];
        memory
            .read_bytes_zero_filled_into(&view, 2, &mut exact)
            .unwrap();
        assert_eq!(exact, [7, 8, 9, 10]);

        assert_eq!(
            memory.read_bytes_zero_filled(&view, 0, 8).unwrap(),
            [0, 0, 7, 8, 9, 10, 0, 0]
        );
        let mut zero_filled = [99_u8; 8];
        memory
            .read_bytes_zero_filled_into(&view, 0, &mut zero_filled)
            .unwrap();
        assert_eq!(zero_filled, [0, 0, 7, 8, 9, 10, 0, 0]);
        assert!(matches!(
            memory.read_bytes(&view, 1, 3),
            Err(MemoryError::InvalidRead {
                first_invalid_byte: 1,
                ..
            })
        ));
    }

    #[test]
    fn explicit_zero_filled_read_preserves_valid_bytes_and_does_not_initialize_padding() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_uninitialized(4).unwrap();
        let view = memory.full_view(allocation).unwrap();

        memory.write_bytes(&view, 1, &[7, 8]).unwrap();
        assert_eq!(
            memory.read_bytes_zero_filled(&view, 0, 4).unwrap(),
            [0, 7, 8, 0]
        );
        let mut zero_filled = [99_u8; 4];
        memory
            .read_bytes_zero_filled_into(&view, 0, &mut zero_filled)
            .unwrap();
        assert_eq!(zero_filled, [0, 7, 8, 0]);
        let mut initialized = [0_u8; 2];
        memory.read_bytes_into(&view, 1, &mut initialized).unwrap();
        assert_eq!(initialized, [7, 8]);
        assert!(matches!(
            memory.read_bytes(&view, 0, 4),
            Err(MemoryError::InvalidRead {
                first_invalid_byte: 0,
                ..
            })
        ));
    }

    #[test]
    fn host_initializer_preserves_invalid_gaps_and_supports_raw_copyback() {
        let memory = GlobalMemory::new();
        let bytes = vec![9, 8, 7, 6, 5, 4];
        let allocation = memory
            .allocate_from_bytes_with_validity(bytes.clone(), vec![0, 0, 1, 1, 1, 1])
            .unwrap();
        let view = memory.full_view(allocation).unwrap();

        assert!(matches!(
            memory.read_bytes(&view, 0, 1),
            Err(MemoryError::InvalidRead {
                first_invalid_byte: 0,
                ..
            })
        ));
        assert_eq!(memory.read_bytes(&view, 2, 4).unwrap(), [7, 6, 5, 4]);
        assert_eq!(memory.snapshot_allocation_bytes(allocation).unwrap(), bytes);
        assert_eq!(
            memory.allocate_from_bytes_with_validity([0, 1], [1]),
            Err(MemoryError::InitializationLengthMismatch {
                byte_len: 2,
                validity_len: 1,
            })
        );
        assert_eq!(
            memory.allocate_from_bytes_with_validity([0], [2]),
            Err(MemoryError::InvalidValidityByte { index: 0, value: 2 })
        );
    }

    #[test]
    fn read_only_backing_preserves_aliases_and_rejects_batch_writes_atomically() {
        let memory = GlobalMemory::new();
        let bytes = [11_u32.to_le_bytes(), 29_u32.to_le_bytes()].concat();
        let allocation = memory
            .allocate_read_only_from_bytes_with_validity(bytes.clone(), vec![1; bytes.len()])
            .unwrap();
        let full = memory.full_view(allocation).unwrap();
        let alias = memory.view(allocation, 4, 4).unwrap();
        assert_eq!(
            memory.read_bytes(&alias, 0, 4).unwrap(),
            29_u32.to_le_bytes()
        );

        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        let mut offsets = WarpValue::splat(usize::MAX);
        offsets[0] = 0;
        offsets[1] = 4;
        let mut target = vec![0xff; WARP_SIZE * 4];
        memory
            .read_shared_bytes_batch_into(&full, &offsets, mask, 4, 4, &mut target)
            .unwrap();
        assert_eq!(&target[0..4], &11_u32.to_le_bytes());
        assert_eq!(&target[4..8], &29_u32.to_le_bytes());
        assert_eq!(&target[8..12], &[0xff; 4]);

        let mut source = vec![0; WARP_SIZE * 4];
        source[0..4].copy_from_slice(&37_u32.to_le_bytes());
        source[4..8].copy_from_slice(&41_u32.to_le_bytes());
        let error = memory
            .write_shared_bytes_batch(&full, &offsets, mask, 4, 4, &source)
            .unwrap_err();
        assert_eq!(error, MemoryError::WriteToReadOnlyAllocation { allocation });
        assert_eq!(memory.snapshot_allocation_bytes(allocation).unwrap(), bytes);
        assert_eq!(
            memory.write_bytes(&alias, 0, &37_u32.to_le_bytes()),
            Err(MemoryError::WriteToReadOnlyAllocation { allocation })
        );
    }

    #[test]
    fn queued_private_batch_writes_commit_in_issue_order_on_the_owner() {
        let memory = GlobalMemory::new_queued_owner_private();
        let allocation = memory.allocate_zeroed(8).unwrap();
        let view = memory.full_view(allocation).unwrap();
        assert_eq!(memory.read_bytes(&view, 0, 8).unwrap(), [0; 8]);

        let writer_memory = memory.clone();
        let writer_view = view.clone();
        thread::spawn(move || {
            let mask = WarpMask::from_lanes([0, 1]).unwrap();
            let mut offsets = WarpValue::splat(usize::MAX);
            offsets[0] = 0;
            offsets[1] = 4;
            let mut first = vec![0; WARP_SIZE * 4];
            first[0..4].copy_from_slice(&11_u32.to_le_bytes());
            first[4..8].copy_from_slice(&29_u32.to_le_bytes());
            writer_memory
                .write_owner_private_resolved_bytes_batch(
                    &writer_view,
                    &offsets,
                    mask,
                    4,
                    4,
                    &first,
                )
                .unwrap();

            let lane_zero = WarpMask::from_lanes([0]).unwrap();
            let mut second = vec![0; WARP_SIZE * 4];
            second[0..4].copy_from_slice(&37_u32.to_le_bytes());
            writer_memory
                .write_owner_private_resolved_bytes_batch(
                    &writer_view,
                    &offsets,
                    lane_zero,
                    4,
                    4,
                    &second,
                )
                .unwrap();
        })
        .join()
        .unwrap();

        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        let mut offsets = WarpValue::splat(usize::MAX);
        offsets[0] = 0;
        offsets[1] = 4;
        let mut target = vec![0xff; WARP_SIZE * 4];
        memory
            .read_owner_private_resolved_bytes_batch_into(&view, &offsets, mask, 4, 4, &mut target)
            .unwrap();
        assert_eq!(&target[0..4], &37_u32.to_le_bytes());
        assert_eq!(&target[4..8], &29_u32.to_le_bytes());
        assert_eq!(&target[8..12], &[0xff; 4]);
    }

    #[test]
    fn queued_private_payload_does_not_claim_an_unbound_owner() {
        let memory = GlobalMemory::new_queued_owner_private();
        let allocation = memory.allocate_uninitialized(4).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let remote_memory = memory.clone();
        let remote_view = view.clone();

        thread::spawn(move || {
            remote_memory
                .publish_owner_private_bytes_batch(
                    &remote_view,
                    [(0, &[7_u8, 11][..]), (2, &[13_u8, 17][..])],
                )
                .unwrap();
        })
        .join()
        .unwrap();

        assert_eq!(memory.read_bytes(&view, 0, 4).unwrap(), [7, 11, 13, 17]);
    }

    #[test]
    fn view_and_access_bounds_are_checked_without_partial_writes() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(16).unwrap();
        assert!(matches!(
            memory.view(allocation, 12, 8),
            Err(MemoryError::ViewOutOfBounds { .. })
        ));

        let view = memory.view(allocation, 4, 4).unwrap();
        assert!(matches!(
            memory.read_f32_le(&view, 1),
            Err(MemoryError::AccessOutOfBounds { .. })
        ));
        assert_eq!(
            memory.read_f32_le(&view, usize::MAX),
            Err(MemoryError::OffsetOverflow)
        );

        let indices = WarpValue::from_fn(|lane| if lane == 0 { 0 } else { 99 });
        let values = WarpValue::from_fn(|lane| lane as f32 + 1.0);
        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        assert!(matches!(
            memory.store_f32_indexed(&view, &indices, &values, mask),
            Err(MemoryError::AccessOutOfBounds { .. })
        ));
        assert_eq!(memory.read_f32_le(&view, 0).unwrap(), 0.0);
    }

    #[test]
    fn masked_vector_load_store_ignore_inactive_indices() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(WARP_SIZE * 4).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let mut indices = WarpValue::splat(usize::MAX);
        indices[0] = 2;
        indices[3] = 7;
        indices[31] = 29;
        let values = WarpValue::from_fn(|lane| 100.0 + lane as f32);
        let mask = WarpMask::from_lanes([0, 3, 31]).unwrap();

        memory
            .store_f32_indexed(&view, &indices, &values, mask)
            .unwrap();
        assert_eq!(memory.read_f32_le(&view, 2).unwrap(), 100.0);
        assert_eq!(memory.read_f32_le(&view, 7).unwrap(), 103.0);
        assert_eq!(memory.read_f32_le(&view, 29).unwrap(), 131.0);
        assert_eq!(memory.read_f32_le(&view, 0).unwrap(), 0.0);

        let mut loaded = WarpValue::splat(-1.0_f32);
        memory
            .load_f32_indexed(&view, &indices, mask, &mut loaded)
            .unwrap();
        assert_eq!(loaded[0], 100.0);
        assert_eq!(loaded[3], 103.0);
        assert_eq!(loaded[31], 131.0);
        assert_eq!(loaded[1], -1.0);
        assert_eq!(loaded[30], -1.0);
    }

    #[test]
    fn scalar_atomic_updates_are_linearizable_across_threads() {
        const THREAD_COUNT: usize = 8;
        const UPDATES_PER_THREAD: usize = 256;

        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(size_of::<u32>()).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let start = Arc::new(Barrier::new(THREAD_COUNT));
        let mut handles = Vec::new();

        for _worker in 0..THREAD_COUNT {
            let memory = memory.clone();
            let start = Arc::clone(&start);
            let worker_view = view.clone();
            handles.push(thread::spawn(move || {
                start.wait();
                let mut previous_values = Vec::with_capacity(UPDATES_PER_THREAD);
                for _sequence in 0..UPDATES_PER_THREAD {
                    let previous = memory
                        .atomic_update_scalar_le::<u32>(&worker_view, 0, |value| {
                            value.checked_add(1).unwrap()
                        })
                        .unwrap();
                    previous_values.push(previous);
                }
                previous_values
            }));
        }

        let mut previous_values = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        previous_values.sort_unstable();
        let update_count = THREAD_COUNT * UPDATES_PER_THREAD;
        assert_eq!(
            previous_values,
            (0..u32::try_from(update_count).unwrap()).collect::<Vec<_>>()
        );
        let final_value = u32::from_le_bytes(
            memory
                .read_bytes(&view, 0, size_of::<u32>())
                .unwrap()
                .try_into()
                .unwrap(),
        );
        assert_eq!(final_value, u32::try_from(update_count).unwrap());
    }

    #[test]
    fn disjoint_stripe_read_progresses_while_atomic_callback_is_blocked() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(MEMORY_STRIPE_BYTES * 2).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let writer_memory = memory.clone();
        let writer_view = view.clone();
        let writer = thread::spawn(move || {
            writer_memory
                .atomic_update_bytes::<_, MemoryError>(&writer_view, 0, 4, |old| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(((), old.to_vec()))
                })
                .unwrap();
        });
        entered_rx.recv().unwrap();

        let (read_tx, read_rx) = mpsc::channel();
        let reader_memory = memory.clone();
        let reader_view = view.clone();
        let reader = thread::spawn(move || {
            let bytes = reader_memory
                .read_bytes(&reader_view, MEMORY_STRIPE_BYTES, 4)
                .unwrap();
            read_tx.send(bytes).unwrap();
        });
        let read_result = read_rx.recv_timeout(Duration::from_secs(1));
        release_tx.send(()).unwrap();
        writer.join().unwrap();
        reader.join().unwrap();
        assert_eq!(read_result.unwrap(), [0; 4]);
    }

    #[test]
    fn stable_snapshot_never_observes_a_torn_stripe_write() {
        const BYTE_LEN: usize = 256;
        const WRITE_COUNT: usize = 2_000;

        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(BYTE_LEN).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let writer_memory = memory.clone();
        let writer_view = view.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let writer_finished = Arc::clone(&finished);
        let writer = thread::spawn(move || {
            for sequence in 1..=WRITE_COUNT {
                let byte = u8::try_from(sequence % 251 + 1).unwrap();
                writer_memory
                    .write_bytes(&writer_view, 0, &vec![byte; BYTE_LEN])
                    .unwrap();
            }
            writer_finished.store(true, AtomicOrdering::Release);
        });

        while !finished.load(AtomicOrdering::Acquire) {
            let snapshot = memory.read_bytes(&view, 0, BYTE_LEN).unwrap();
            assert!(snapshot.iter().all(|byte| *byte == snapshot[0]));
        }
        writer.join().unwrap();
        let snapshot = memory.read_bytes(&view, 0, BYTE_LEN).unwrap();
        assert!(snapshot.iter().all(|byte| *byte == snapshot[0]));
    }

    #[test]
    fn disjoint_stripe_atomic_callbacks_can_overlap() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(MEMORY_STRIPE_BYTES * 2).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (first_release_tx, first_release_rx) = mpsc::channel();
        let (second_release_tx, second_release_rx) = mpsc::channel();

        let first_memory = memory.clone();
        let first_entered = entered_tx.clone();
        let first_view = view.clone();
        let first = thread::spawn(move || {
            first_memory
                .atomic_update_bytes::<_, MemoryError>(&first_view, 0, 4, |old| {
                    first_entered.send(0).unwrap();
                    first_release_rx.recv().unwrap();
                    Ok(((), old.to_vec()))
                })
                .unwrap();
        });
        assert_eq!(entered_rx.recv().unwrap(), 0);

        let second_memory = memory.clone();
        let second_view = view.clone();
        let second = thread::spawn(move || {
            second_memory
                .atomic_update_bytes::<_, MemoryError>(
                    &second_view,
                    MEMORY_STRIPE_BYTES,
                    4,
                    |old| {
                        entered_tx.send(1).unwrap();
                        second_release_rx.recv().unwrap();
                        Ok(((), old.to_vec()))
                    },
                )
                .unwrap();
        });
        let overlapped = entered_rx.recv_timeout(Duration::from_secs(1));
        first_release_tx.send(()).unwrap();
        if overlapped.is_err() {
            assert_eq!(entered_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        }
        second_release_tx.send(()).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(overlapped.unwrap(), 1);
    }

    #[test]
    fn owner_private_memory_preserves_validity() {
        let memory = GlobalMemory::new_owner_private();
        let allocation = memory.allocate_uninitialized(4).unwrap();
        let view = memory.full_view(allocation).unwrap();

        assert_eq!(memory.byte_validity(&view, 0, 4).unwrap(), [false; 4]);
        memory.write_bytes(&view, 1, &[7, 8]).unwrap();

        assert_eq!(
            memory.byte_validity(&view, 0, 4).unwrap(),
            [false, true, true, false]
        );
    }

    #[test]
    fn owner_private_allocations_bind_independent_threads() {
        let memory = GlobalMemory::new_owner_private();
        let first_id = memory.allocate_zeroed(4).unwrap();
        let second_id = memory.allocate_zeroed(4).unwrap();
        let first_allocation = memory.allocation(first_id).unwrap();
        let second_allocation = memory.allocation(second_id).unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (first_release_tx, first_release_rx) = mpsc::channel();
        let (second_release_tx, second_release_rx) = mpsc::channel();

        let first_entered = entered_tx.clone();
        let first = thread::spawn(move || {
            first_allocation
                .with_private(first_id, |_| {
                    first_entered.send(0).unwrap();
                    first_release_rx.recv().unwrap();
                })
                .unwrap();
        });
        assert_eq!(entered_rx.recv().unwrap(), 0);

        let second = thread::spawn(move || {
            second_allocation
                .with_private(second_id, |_| {
                    entered_tx.send(1).unwrap();
                    second_release_rx.recv().unwrap();
                })
                .unwrap();
        });
        let overlapped = entered_rx.recv_timeout(Duration::from_secs(1));
        first_release_tx.send(()).unwrap();
        if overlapped.is_err() {
            assert_eq!(entered_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        }
        second_release_tx.send(()).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(overlapped.unwrap(), 1);
    }

    #[test]
    fn owner_private_allocation_rejects_a_second_thread() {
        let memory = GlobalMemory::new_owner_private();
        let allocation_id = memory.allocate_zeroed(4).unwrap();
        let allocation = memory.allocation(allocation_id).unwrap();
        let view = memory.full_view(allocation_id).unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let owner = thread::spawn(move || {
            allocation
                .with_private(allocation_id, |_| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                })
                .unwrap();
        });
        entered_rx.recv().unwrap();
        let contender_memory = memory.clone();
        let contender = thread::spawn(move || contender_memory.read_bytes(&view, 0, 4));
        let result = contender.join().unwrap();
        release_tx.send(()).unwrap();
        owner.join().unwrap();
        assert_eq!(
            result,
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread {
                allocation: allocation_id,
            })
        );
    }

    #[test]
    fn owner_private_allocation_rejects_recycled_thread_local_slot() {
        let memory = GlobalMemory::new_owner_private();
        let allocation_id = memory.allocate_zeroed(4).unwrap();
        let view = memory.full_view(allocation_id).unwrap();

        let owner_memory = memory.clone();
        let owner_view = view.clone();
        thread::spawn(move || owner_memory.write_bytes(&owner_view, 0, &[1, 2, 3, 4]))
            .join()
            .unwrap()
            .unwrap();

        let contender = thread::spawn(move || memory.read_bytes(&view, 0, 4));
        assert_eq!(
            contender.join().unwrap(),
            Err(MemoryError::OwnerPrivateAccessFromDifferentThread {
                allocation: allocation_id,
            })
        );
    }

    #[cfg(feature = "analysis-core")]
    #[test]
    fn semantic_progress_ignores_same_value_writes_and_wakes_after_real_change() {
        let memory = GlobalMemory::new();
        let allocation = memory.allocate_zeroed(4).unwrap();
        let view = memory.full_view(allocation).unwrap();
        let progress = memory.semantic_progress();
        let snapshot = progress.snapshot();
        let events = Arc::new(Mutex::new(Vec::new()));
        let waker = Waker::from(Arc::new(RecordingWake {
            id: 7,
            events: Arc::clone(&events),
        }));
        let mut context = Context::from_waker(&waker);
        let mut watch = Box::pin(progress.watch(snapshot));

        assert_eq!(watch.as_mut().poll(&mut context), Poll::Pending);
        memory.write_bytes(&view, 0, &[0, 0, 0, 0]).unwrap();
        assert_eq!(progress.snapshot(), snapshot);
        assert_eq!(watch.as_mut().poll(&mut context), Poll::Pending);
        assert!(events.lock().unwrap().is_empty());

        memory.write_bytes(&view, 0, &[1, 0, 0, 0]).unwrap();
        assert_ne!(progress.snapshot(), snapshot);
        assert_eq!(*events.lock().unwrap(), vec![7]);
        assert_eq!(watch.as_mut().poll(&mut context), Poll::Ready(()));
    }

    struct FlagWake(Arc<AtomicBool>);

    impl Wake for FlagWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, AtomicOrdering::Release);
        }
    }

    /// A watcher that parks against generation `g` must be woken by the
    /// `record_change` that advances the generation past `g`.
    ///
    /// Regression guard for the lost semantic-progress wake that made
    /// `record_change` read `waiter_count` outside the `waiters` mutex while
    /// `poll` published its registration after its staleness check: the
    /// notifier saw no waiter, woke nobody, and the waiter parked forever on a
    /// generation it already knew was stale.  In production that strands a
    /// native while-loop checkpoint and the executor reports a false
    /// `deadlock`.
    ///
    /// Round discipline makes the witness exact: the notifier performs exactly
    /// ONE `record_change` per round and signals completion, and the watcher is
    /// judged only after that signal.  A watcher parked on a stale generation
    /// with no wake delivered at that point is therefore permanently stranded --
    /// no further `record_change` exists to rescue it.
    #[test]
    fn semantic_progress_never_strands_a_registering_watcher() {
        let progress = SemanticProgress::default();
        let round = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(usize::MAX));
        let stop = Arc::new(AtomicBool::new(false));

        let notifier_progress = progress.clone();
        let notifier_round = Arc::clone(&round);
        let notifier_done = Arc::clone(&done);
        let notifier_stop = Arc::clone(&stop);
        let notifier = thread::spawn(move || {
            let mut served = usize::MAX;
            let mut skew = 0_usize;
            while !notifier_stop.load(AtomicOrdering::Relaxed) {
                let current = notifier_round.load(AtomicOrdering::Acquire);
                if current == served {
                    std::hint::spin_loop();
                    continue;
                }
                // Sweep the notifier's arrival across the watcher's
                // registration path so every offset is exercised.
                for _ in 0..skew {
                    std::hint::spin_loop();
                }
                skew = (skew + 1) % 96;
                notifier_progress.record_change();
                served = current;
                notifier_done.store(current, AtomicOrdering::Release);
            }
        });

        let rounds = 400_000_usize;
        let mut parked = 0_usize;
        let mut witness: Option<String> = None;
        for index in 0..rounds {
            let observed = progress.snapshot();
            let woken = Arc::new(AtomicBool::new(false));
            let waker = Waker::from(Arc::new(FlagWake(Arc::clone(&woken))));
            let mut context = Context::from_waker(&waker);
            let mut watch = Box::pin(progress.watch(observed));
            round.store(index, AtomicOrdering::Release);
            let outcome = watch.as_mut().poll(&mut context);
            while done.load(AtomicOrdering::Acquire) != index {
                std::hint::spin_loop();
            }
            if outcome == Poll::Pending {
                parked += 1;
                let current = progress.snapshot();
                if current != observed && !woken.load(AtomicOrdering::Acquire) {
                    // `waiter_count` must also still mirror the registry: a
                    // persistent disagreement is what would make a strand
                    // permanent, since every later `record_change` would keep
                    // taking the `waiter_count == 0` fast path.
                    let waiters_len = lock_mutex(&progress.inner.waiters).len();
                    let count = progress.inner.waiter_count.load(Ordering::SeqCst);
                    let strand = format!(
                        "STRAND observed={} current={} round={index} waiter_count={count} \
                         waiters_len={waiters_len}",
                        observed.0, current.0
                    );
                    eprintln!("{strand}");
                    witness = Some(strand);
                    break;
                }
                assert_eq!(
                    progress.inner.waiter_count.load(Ordering::SeqCst),
                    lock_mutex(&progress.inner.waiters).len(),
                    "waiter_count stopped mirroring the waiters registry at round {index}"
                );
            }
            drop(watch);
        }

        stop.store(true, AtomicOrdering::Relaxed);
        notifier.join().unwrap();

        assert!(parked > 0, "test never exercised the registration path");
        assert!(
            witness.is_none(),
            "record_change stranded a registered watcher: {}\n\
             The notifier published a generation strictly newer than the one this watcher \
             parked on, took `record_change`'s unlocked `waiter_count == 0` fast path, and \
             woke nobody -- so the watcher is parked forever on a generation it already \
             knew was stale. {parked} of {rounds} rounds reached the registration path. \
             Check that `poll` still registers BEFORE its staleness check and that both \
             halves of the handshake (`generation` and `waiter_count`) are still SeqCst.",
            witness.unwrap_or_default()
        );
    }

    #[test]
    fn global_memory_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_send<T: Send>() {}
        assert_send_sync::<GlobalMemory>();
        assert_send::<SemanticProgressWatch>();
    }
}
