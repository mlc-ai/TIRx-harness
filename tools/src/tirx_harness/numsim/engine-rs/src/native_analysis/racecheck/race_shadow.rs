//! Racecheck-owned online happens-before detection over resolved physical byte batches.
//!
//! This module intentionally owns only the byte-shadow and vector-clock layer.
//! `RaceCheckMode` supplies ordinary warp events, independently clocked async
//! token completions, and barrier release/acquire payloads. This per-cluster
//! shadow owns shared memory and TMEM. Launch-wide global-memory conflicts,
//! exact read-from, and scoped publication are owned by the separate
//! `global_race` state.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use crate::operation::LoopFrame;
use crate::physical_access::{CompactPhysicalAccessBatch, ProxyMemoryDomain};
use crate::transactional_interval_map::TransactionalIntervalMap;
use crate::{
    profile_count, AsyncTokenId, DynamicOpId, LanePhysicalAccess, MemoryProxy, PhysicalAccessBatch,
    PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
    PhysicalAllocationId, PhysicalByteSpan, ProfileKind, ProfileTimer, ProxyAsyncFenceScope,
    WarpMask, WARP_SIZE,
};

const AUTOMATIC_GC_SAFE_POINT_INTERVAL: usize = 1_024;
const PROXY_MEMORY_DOMAIN_COUNT: usize = 3;
const PROXY_BRIDGE_DIRECTION_COUNT: usize = 2;
const PROXY_BRIDGE_SLOT_COUNT: usize =
    PROXY_BRIDGE_DIRECTION_COUNT * PROXY_MEMORY_DOMAIN_COUNT * PROXY_MEMORY_DOMAIN_COUNT;
const MAX_EXACT_RETIRED_PROXY_ALLOCATIONS: usize = 4_096;
const MAX_RETIRED_RANGES_PER_SLOT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
struct ProxyAccessClass(u8);

impl ProxyAccessClass {
    const PROXY_MASK: u8 = 0x3;
    const DOMAIN_SHIFT: u32 = 2;
    const DOMAIN_MASK: u8 = 0x3 << Self::DOMAIN_SHIFT;
    const SENSITIVE_BIT: u8 = 1 << 4;

    const fn new(proxy: MemoryProxy, domain: ProxyMemoryDomain, proxy_sensitive: bool) -> Self {
        Self(
            proxy as u8
                | ((domain as u8) << Self::DOMAIN_SHIFT)
                | if proxy_sensitive {
                    Self::SENSITIVE_BIT
                } else {
                    0
                },
        )
    }

    #[inline(always)]
    fn proxy(self) -> MemoryProxy {
        match self.0 & Self::PROXY_MASK {
            0 => MemoryProxy::Generic,
            1 => MemoryProxy::Async,
            2 => MemoryProxy::Mmio,
            _ => unreachable!("proxy access class retains a valid memory proxy"),
        }
    }

    #[inline(always)]
    fn domain(self) -> ProxyMemoryDomain {
        match (self.0 & Self::DOMAIN_MASK) >> Self::DOMAIN_SHIFT {
            0 => ProxyMemoryDomain::Global,
            1 => ProxyMemoryDomain::SharedCta,
            2 => ProxyMemoryDomain::SharedCluster,
            3 => ProxyMemoryDomain::Other,
            _ => unreachable!("two domain bits are exhaustive"),
        }
    }

    const fn is_sensitive(self) -> bool {
        self.0 & Self::SENSITIVE_BIT != 0
    }
}

/// Whether Racecheck currently claims conflict coverage for this memory space.
///
/// Global accesses are deliberately excluded here because the launch-wide
/// global shadow owns their conflict and ordering state.
pub(crate) const fn tracks_race_conflicts(space: PhysicalAccessSpace) -> bool {
    matches!(
        space,
        PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem
    )
}

#[derive(Debug, Default)]
struct AsyncClockRegistry {
    inner: Mutex<AsyncClockRegistryInner>,
    chunk_joins: AsyncChunkJoinMemo,
    warp_chunk_joins: WarpChunkJoinMemo,
}

/// Async actor epochs carry the slot's generation above the epoch, so a slot
/// can be handed to a new token once every access of its previous token is
/// gone from the shadow: a clock that only ever observed an earlier
/// generation holds a value below every epoch of the current one, and
/// therefore never appears to have observed the new token. The packed value
/// stays below `RaceEventTimestamp::ASYNC_ACTOR_BIT` so async timestamps keep
/// their compact encoding.
const ASYNC_EPOCH_BITS: u32 = 16;
const ASYNC_EPOCH_MASK: u64 = (1 << ASYNC_EPOCH_BITS) - 1;
const ASYNC_GENERATION_LIMIT: u32 = 1 << 14;
/// Minimum number of fresh slot allocations between two reclamation passes.
/// A pass is also never due before `ASYNC_RECLAIM_OCCUPANCY_FACTOR` times as
/// many fresh slots as stayed occupied after the previous pass, so the slot
/// count stays within a small multiple of the live token set while passes
/// remain rare. The dominated-frontier GC alone runs only every
/// `AUTOMATIC_GC_SAFE_POINT_INTERVAL` safe points — rarer than a shard's
/// live token set turns over — so slot growth is bounded here.
const ASYNC_RECLAIM_ALLOCATION_INTERVAL: usize = 64;
const ASYNC_RECLAIM_OCCUPANCY_FACTOR: usize = 4;

const fn pack_async_epoch(generation: u32, epoch: u64) -> u64 {
    ((generation as u64) << ASYNC_EPOCH_BITS) | epoch
}

const fn unpack_async_epoch(value: u64) -> (u32, u64) {
    ((value >> ASYNC_EPOCH_BITS) as u32, value & ASYNC_EPOCH_MASK)
}

impl AsyncClockRegistryInner {
    fn reclaim_due(&self) -> bool {
        self.fresh_since_reclaim
            >= ASYNC_RECLAIM_ALLOCATION_INTERVAL
                .max(self.occupied_after_reclaim * ASYNC_RECLAIM_OCCUPANCY_FACTOR)
    }
}

#[derive(Debug, Default)]
struct AsyncClockRegistryInner {
    handles: HashMap<AsyncTokenId, usize>,
    // Per slot: the token currently holding it (`None` while free or
    // exhausted) and its generation.
    tokens: Vec<Option<AsyncTokenId>>,
    generations: Vec<u32>,
    free: Vec<usize>,
    fresh_since_reclaim: usize,
    occupied_after_reclaim: usize,
    issue_events: Vec<Option<AsyncIssueEvent>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AsyncIssueEvent {
    // The async clock does not advance its issuer component. Reserve the next
    // epoch as an issue marker so a release observed before issue cannot be
    // mistaken for a post-issue ordinary handoff.
    issuer_warp: usize,
    epoch: u64,
}

impl AsyncClockRegistry {
    /// Resolve `token` to its slot and the slot's current generation,
    /// allocating (preferably by reusing a freed slot) when unseen.
    fn register_with_generation(&self, token: &AsyncTokenId) -> (usize, u32) {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned");
        if let Some(&index) = inner.handles.get(token) {
            return (index, inner.generations[index]);
        }
        let index = if let Some(index) = inner.free.pop() {
            index
        } else {
            inner.tokens.push(None);
            inner.generations.push(0);
            inner.issue_events.push(None);
            inner.fresh_since_reclaim += 1;
            inner.tokens.len() - 1
        };
        inner.tokens[index] = Some(token.clone());
        // A slot's issue event belongs to the token holding it.
        inner.issue_events[index] = None;
        inner.handles.insert(token.clone(), index);
        (index, inner.generations[index])
    }

    /// Resolve `token` to its slot and record its issue event (one per
    /// token): the issuing warp and the epoch reserved as its issue marker.
    fn register_issue(&self, token: &AsyncTokenId, issuer_warp: usize, epoch: u64) -> (usize, u32) {
        let (index, generation) = self.register_with_generation(token);
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned");
        let issue = AsyncIssueEvent { issuer_warp, epoch };
        let registered = &mut inner.issue_events[index];
        if let Some(registered) = registered {
            assert_eq!(*registered, issue, "one async token has one issue event");
        } else {
            *registered = Some(issue);
        }
        (index, generation)
    }

    fn index(&self, token: &AsyncTokenId) -> Option<usize> {
        self.inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned")
            .handles
            .get(token)
            .copied()
    }

    fn issue_event(&self, index: usize) -> Option<AsyncIssueEvent> {
        self.inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned")
            .issue_events
            .get(index)
            .copied()
            .flatten()
    }

    #[cfg(test)]
    fn index_with_generation(&self, token: &AsyncTokenId) -> Option<(usize, u32)> {
        let inner = self
            .inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned");
        let index = *inner.handles.get(token)?;
        Some((index, inner.generations[index]))
    }

    fn slot_count(&self) -> usize {
        self.inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned")
            .tokens
            .len()
    }

    fn reclaim_due(&self) -> bool {
        self.inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned")
            .reclaim_due()
    }

    /// Free every occupied slot `dead` accepts and return their indices. A
    /// freed slot is reused under the next generation; a slot whose
    /// generation counter is exhausted is simply left unused.
    fn reclaim(&self, mut dead: impl FnMut(usize) -> bool) -> Vec<usize> {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned");
        let mut freed = Vec::new();
        for index in 0..inner.tokens.len() {
            if inner.tokens[index].is_none() || !dead(index) {
                continue;
            }
            let token = inner.tokens[index]
                .take()
                .expect("an occupied async slot names its token");
            inner.handles.remove(&token);
            let next = inner.generations[index] + 1;
            if next < ASYNC_GENERATION_LIMIT {
                inner.generations[index] = next;
                inner.free.push(index);
            }
            freed.push(index);
        }
        inner.fresh_since_reclaim = 0;
        inner.occupied_after_reclaim = inner.tokens.iter().filter(|token| token.is_some()).count();
        freed
    }
}

/// Epochs per actor index in immutable, shared chunks of `CHUNK` slots.
///
/// Async actor indices are launch-local and never recycled, so a flat vector
/// grows with every token ever issued and every merge or copy pays for all of
/// them; the per-warp vector is copied whole on every changing merge. Chunks
/// are immutable and shared between clocks: a join compares chunk pointers
/// first and only touches the chunks that actually differ, a dominated side
/// adopts the dominating side's chunk outright so clocks that synchronized
/// through the same payload keep sharing storage, and joins of two differing
/// chunks are memoized per pair of chunk identities in the registry. An
/// absent chunk holds zeros; `len` is the number of addressable slots.
#[derive(Clone, Debug)]
struct EpochChunks<const CHUNK: usize> {
    chunks: Arc<[Option<Arc<EpochChunk<CHUNK>>>]>,
    len: usize,
}

impl<const CHUNK: usize> Default for EpochChunks<CHUNK> {
    fn default() -> Self {
        Self {
            chunks: Arc::from([]),
            len: 0,
        }
    }
}

const ASYNC_EPOCH_CHUNK: usize = 64;
const WARP_EPOCH_CHUNK: usize = 32;

type AsyncEpochs = EpochChunks<ASYNC_EPOCH_CHUNK>;
type WarpEpochs = EpochChunks<WARP_EPOCH_CHUNK>;

/// One immutable chunk; every content change makes a new chunk with a new
/// identity.
#[derive(Debug)]
struct EpochChunk<const CHUNK: usize> {
    id: u32,
    epochs: [u64; CHUNK],
}

static NEXT_EPOCH_CHUNK_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
impl<const CHUNK: usize> EpochChunk<CHUNK> {
    fn new(epochs: [u64; CHUNK]) -> Arc<Self> {
        let id = NEXT_EPOCH_CHUNK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert!(id != u32::MAX, "racecheck epoch chunk identity overflow");
        Arc::new(Self { id, epochs })
    }

    // Express the fixed-size comparison as a reduction without early exits.
    fn dominates(&self, other: &Self) -> bool {
        self.epochs
            .iter()
            .zip(other.epochs.iter())
            .fold(true, |all, (l, r)| all & (l >= r))
    }

    fn is_zero(&self) -> bool {
        self.epochs.iter().all(|epoch| *epoch == 0)
    }
}

/// Result of joining two chunks.
#[derive(Clone)]
enum EpochChunkJoin<const CHUNK: usize> {
    Current,
    Incoming,
    New(Arc<EpochChunk<CHUNK>>),
}

/// Multiplicative hash for the packed pair of node identities.
pub(crate) type GroupJoinHasherBuilder = std::hash::BuildHasherDefault<GroupJoinHasher>;

#[derive(Default)]
pub(crate) struct GroupJoinHasher(u64);

impl std::hash::Hasher for GroupJoinHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = (value ^ (value >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 ^= self.0 >> 32;
    }

    fn write_u128(&mut self, value: u128) {
        self.write_u64(value as u64);
        let low = self.0;
        self.write_u64((value >> 64) as u64);
        self.0 ^= low.rotate_left(32);
    }
}

/// Memo of incomparable chunk joins keyed by `(current id << 32) | incoming id`,
/// sharded by key so concurrent shards rarely touch the same lock. Chunks are
/// held weakly: repeated joins share live results without extending their life.
struct EpochChunkJoinMemo<const CHUNK: usize> {
    joins: Box<[Mutex<EpochChunkJoinShard<CHUNK>>]>,
}

struct EpochChunkJoinShard<const CHUNK: usize> {
    joins: HashMap<u64, std::sync::Weak<EpochChunk<CHUNK>>, GroupJoinHasherBuilder>,
    inserts_since_prune: u32,
}

const EPOCH_MEMO_SHARDS: usize = 64;
/// Entries per memo shard before dead entries are pruned (and, if that frees
/// little, the shard is cleared): chunk identities are never reused, so
/// entries for dropped chunks only waste memory.
const EPOCH_MEMO_SHARD_CAP: usize = 1 << 15;
/// Inserts per shard between prunes of the dead entries. A memoized join
/// holds its chunk weakly, and a `Weak` keeps the chunk's allocation (528 B
/// for an async chunk) until the entry goes: with 74 racecheck shards each
/// memoizing up to 2 M joins, the dead entries pinned 14 M chunk
/// allocations on MegaMoE t128 (≈ 9 GB, 60 × the live chunks), so a shard
/// prunes on a cadence instead of only at its cap.
const EPOCH_MEMO_PRUNE_EVERY: u32 = 1 << 8;

impl<const CHUNK: usize> Default for EpochChunkJoinMemo<CHUNK> {
    fn default() -> Self {
        Self {
            joins: (0..EPOCH_MEMO_SHARDS)
                .map(|_| {
                    Mutex::new(EpochChunkJoinShard {
                        joins: HashMap::default(),
                        inserts_since_prune: 0,
                    })
                })
                .collect(),
        }
    }
}

impl<const CHUNK: usize> fmt::Debug for EpochChunkJoinMemo<CHUNK> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpochChunkJoinMemo").finish_non_exhaustive()
    }
}

impl<const CHUNK: usize> EpochChunkJoinMemo<CHUNK> {
    fn shard(key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize
    }

    fn get(&self, key: u64) -> Option<EpochChunkJoin<CHUNK>> {
        let mut shard = self.joins[Self::shard(key)]
            .lock()
            .expect("racecheck epoch chunk join memo lock was poisoned");
        let joins = &mut shard.joins;
        match joins.get(&key)?.upgrade() {
            Some(chunk) => Some(EpochChunkJoin::New(chunk)),
            None => {
                // The joined chunk died with its last clock; forget it so
                // the map does not fill with lapsed entries.
                joins.remove(&key);
                None
            }
        }
    }

    fn insert(&self, key: u64, chunk: &Arc<EpochChunk<CHUNK>>) {
        let mut shard = self.joins[Self::shard(key)]
            .lock()
            .expect("racecheck epoch chunk join memo lock was poisoned");
        shard.joins.insert(key, Arc::downgrade(chunk));
        shard.inserts_since_prune += 1;
        if shard.inserts_since_prune >= EPOCH_MEMO_PRUNE_EVERY
            || shard.joins.len() >= EPOCH_MEMO_SHARD_CAP
        {
            shard.inserts_since_prune = 0;
            shard.joins.retain(|_, chunk| chunk.strong_count() != 0);
            if shard.joins.len() >= EPOCH_MEMO_SHARD_CAP / 2 {
                shard.joins.clear();
            }
        }
    }
}

type AsyncChunkJoinMemo = EpochChunkJoinMemo<ASYNC_EPOCH_CHUNK>;
type WarpChunkJoinMemo = EpochChunkJoinMemo<WARP_EPOCH_CHUNK>;

fn join_epoch_chunks<const CHUNK: usize>(
    current: &Arc<EpochChunk<CHUNK>>,
    incoming: &Arc<EpochChunk<CHUNK>>,
    memo: &EpochChunkJoinMemo<CHUNK>,
) -> EpochChunkJoin<CHUNK> {
    if Arc::ptr_eq(current, incoming) {
        return EpochChunkJoin::Current;
    }
    if current.dominates(incoming) {
        return EpochChunkJoin::Current;
    }
    if incoming.dominates(current) {
        return EpochChunkJoin::Incoming;
    }
    let key = ((current.id as u64) << 32) | incoming.id as u64;
    if let Some(join) = memo.get(key) {
        return join;
    }
    let mut merged = current.epochs;
    for (slot, epoch) in merged.iter_mut().zip(incoming.epochs.iter()) {
        *slot = (*slot).max(*epoch);
    }
    let chunk = EpochChunk::new(merged);
    memo.insert(key, &chunk);
    EpochChunkJoin::New(chunk)
}

impl<const CHUNK: usize> EpochChunks<CHUNK> {
    /// `len` addressable zero slots.
    fn zeros(len: usize) -> Self {
        Self {
            chunks: vec![None; len.div_ceil(CHUNK)].into(),
            len,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.chunks, &other.chunks)
    }

    #[inline(always)]
    fn get(&self, index: usize) -> u64 {
        self.chunks
            .get(index / CHUNK)
            .and_then(|chunk| chunk.as_ref())
            .map_or(0, |chunk| chunk.epochs[index % CHUNK])
    }

    /// The slot's epoch, or `None` past the addressable length.
    fn slot(&self, index: usize) -> Option<u64> {
        (index < self.len).then(|| self.get(index))
    }

    /// Replace one slot, giving the touched chunk a fresh identity and
    /// growing the addressable length if needed.
    fn write(&mut self, index: usize, epoch: u64) {
        let chunk_index = index / CHUNK;
        if chunk_index >= self.chunks.len() {
            let mut chunks = self.chunks.to_vec();
            chunks.resize(chunk_index + 1, None);
            self.chunks = chunks.into();
        }
        self.len = self.len.max(index + 1);
        let chunks = Arc::make_mut(&mut self.chunks);
        let mut epochs = chunks[chunk_index]
            .as_ref()
            .map_or([0; CHUNK], |chunk| chunk.epochs);
        epochs[index % CHUNK] = epoch;
        chunks[chunk_index] = Some(EpochChunk::new(epochs));
    }

    fn set(&mut self, index: usize, epoch: u64) {
        if self.get(index) == epoch {
            return;
        }
        self.write(index, epoch);
    }

    fn raise(&mut self, index: usize, epoch: u64) {
        if self.get(index) >= epoch {
            return;
        }
        self.write(index, epoch);
    }

    /// Every `(index, epoch)` with a non-zero epoch, skipping absent chunks.
    fn nonzero(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.chunks
            .iter()
            .enumerate()
            .filter_map(|(chunk_index, chunk)| chunk.as_ref().map(|chunk| (chunk_index, chunk)))
            .flat_map(move |(chunk_index, chunk)| {
                chunk
                    .epochs
                    .iter()
                    .enumerate()
                    .filter(|(_, epoch)| **epoch != 0)
                    .map(move |(slot, epoch)| (chunk_index * CHUNK + slot, *epoch))
            })
    }

    /// Whether every epoch of `self` is covered by `other`.
    fn happens_before(&self, other: &Self, memo: &EpochChunkJoinMemo<CHUNK>) -> bool {
        if self.shares_storage_with(other) {
            return true;
        }
        self.chunks.iter().enumerate().all(|(chunk_index, own)| {
            let Some(own) = own else { return true };
            match other
                .chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.as_ref())
            {
                Some(theirs) => {
                    // `self ⊑ other` exactly when joining `self` into `other`
                    // changes nothing on `other`'s side.
                    matches!(
                        join_epoch_chunks(theirs, own, memo),
                        EpochChunkJoin::Current
                    )
                }
                None => own.is_zero(),
            }
        })
    }

    fn equals(&self, other: &Self, memo: &EpochChunkJoinMemo<CHUNK>) -> bool {
        self.happens_before(other, memo) && other.happens_before(self, memo)
    }

    /// Raise every epoch to at least `other`'s.
    fn merge(&mut self, other: &Self, memo: &EpochChunkJoinMemo<CHUNK>) {
        if self.shares_storage_with(other) {
            return;
        }
        let len = self.chunks.len().max(other.chunks.len());
        let mut replacements = Vec::new();
        let mut current_changed = false;
        let mut incoming_changed = self.len > other.len;
        for chunk_index in 0..len {
            let own = self
                .chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.as_ref());
            let theirs = other
                .chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.as_ref());
            match (own, theirs) {
                (None, None) => {}
                (Some(_), None) => incoming_changed = true,
                (None, Some(theirs)) => {
                    current_changed = true;
                    replacements.push((chunk_index, Arc::clone(theirs)));
                }
                (Some(own), Some(theirs)) => match join_epoch_chunks(own, theirs, memo) {
                    EpochChunkJoin::Current => {
                        if !Arc::ptr_eq(own, theirs) {
                            incoming_changed = true;
                        }
                    }
                    EpochChunkJoin::Incoming => {
                        current_changed = true;
                        replacements.push((chunk_index, Arc::clone(theirs)));
                    }
                    EpochChunkJoin::New(chunk) => {
                        current_changed = true;
                        incoming_changed = true;
                        replacements.push((chunk_index, chunk));
                    }
                },
            }
        }
        self.len = self.len.max(other.len);
        if !current_changed {
            return;
        }
        if !incoming_changed {
            self.chunks = Arc::clone(&other.chunks);
            return;
        }
        if self.chunks.len() < len {
            let mut chunks = self.chunks.to_vec();
            chunks.resize(len, None);
            self.chunks = chunks.into();
        }
        let chunks = Arc::make_mut(&mut self.chunks);
        for (chunk_index, chunk) in replacements {
            chunks[chunk_index] = Some(chunk);
        }
    }

    /// Zero the listed slots for which `should_clear(epoch, tag)` holds.
    /// Chunks without a listed slot to clear keep sharing their storage.
    fn clear_where(
        &mut self,
        slots_by_chunk: &[Vec<(u16, u64)>],
        should_clear: impl Fn(u64, u64) -> bool,
    ) {
        let chunk_count = self.chunks.len().min(slots_by_chunk.len());
        for chunk_index in 0..chunk_count {
            let Some(chunk) = self.chunks[chunk_index].as_ref() else {
                continue;
            };
            let slots = &slots_by_chunk[chunk_index];
            if !slots
                .iter()
                .any(|&(slot, tag)| should_clear(chunk.epochs[usize::from(slot)], tag))
            {
                continue;
            }
            let mut epochs = chunk.epochs;
            for &(slot, tag) in slots {
                if should_clear(epochs[usize::from(slot)], tag) {
                    epochs[usize::from(slot)] = 0;
                }
            }
            Arc::make_mut(&mut self.chunks)[chunk_index] = Some(EpochChunk::new(epochs));
        }
    }

    /// Lower every epoch to at most `other`'s.
    fn meet(&mut self, other: &Self) {
        let mut replacement: Option<Vec<Option<Arc<EpochChunk<CHUNK>>>>> = None;
        for chunk_index in 0..self.chunks.len() {
            let Some(own) = self.chunks[chunk_index].as_ref() else {
                continue;
            };
            let theirs = other
                .chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.as_ref());
            let lowered = match theirs {
                Some(theirs) if Arc::ptr_eq(own, theirs) || theirs.dominates(own) => {
                    continue;
                }
                Some(theirs) => {
                    let mut lowered = own.epochs;
                    for (slot, epoch) in lowered.iter_mut().zip(theirs.epochs.iter()) {
                        *slot = (*slot).min(*epoch);
                    }
                    Some(EpochChunk::new(lowered))
                }
                None => None,
            };
            replacement.get_or_insert_with(|| self.chunks.to_vec())[chunk_index] = lowered;
        }
        if let Some(chunks) = replacement {
            self.chunks = chunks.into();
        }
    }
}

/// Happens-before timestamp used by the native physical race shadow.
///
/// Async actors receive launch-local dense component indices. The previous
/// `BTreeMap<AsyncTokenId, u64>` representation cloned and compared hundreds
/// of heap nodes at every barrier merge after a long pipeline. Dense slices
/// preserve the exact vector-clock relation while turning that work into a
/// contiguous max/min pass.
#[derive(Clone, Debug)]
pub struct RaceVectorClock {
    components: WarpEpochs,
    async_components: AsyncEpochs,
    async_registry: Arc<AsyncClockRegistry>,
    proxy_metadata: Option<Arc<ProxyClockMetadata>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProxyClockMetadata {
    bridges: Option<Arc<LaneProxyBridgeFrontiers>>,
    lane_mask: WarpMask,
}

impl PartialEq for RaceVectorClock {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.async_registry, &other.async_registry)
            && self
                .components
                .equals(&other.components, &self.async_registry.warp_chunk_joins)
            && self
                .async_components
                .equals(&other.async_components, &self.async_registry.chunk_joins)
            && self.proxy_metadata == other.proxy_metadata
    }
}

impl Eq for RaceVectorClock {}

impl RaceVectorClock {
    fn zero(warp_count: usize, async_registry: Arc<AsyncClockRegistry>) -> Self {
        Self {
            components: WarpEpochs::zeros(warp_count),
            async_components: AsyncEpochs::default(),
            async_registry,
            proxy_metadata: None,
        }
    }

    #[inline(always)]
    fn proxy_lane_mask(&self) -> WarpMask {
        self.proxy_metadata
            .as_deref()
            .map_or(WarpMask::FULL, |metadata| metadata.lane_mask)
    }

    #[inline(always)]
    fn proxy_bridges_arc(&self) -> Option<&Arc<LaneProxyBridgeFrontiers>> {
        self.proxy_metadata
            .as_deref()
            .and_then(|metadata| metadata.bridges.as_ref())
    }

    #[inline(always)]
    fn proxy_bridges(&self) -> Option<&LaneProxyBridgeFrontiers> {
        self.proxy_bridges_arc().map(Arc::as_ref)
    }

    fn proxy_metadata_mut(&mut self) -> &mut ProxyClockMetadata {
        Arc::make_mut(self.proxy_metadata.get_or_insert_with(|| {
            Arc::new(ProxyClockMetadata {
                bridges: None,
                lane_mask: WarpMask::FULL,
            })
        }))
    }

    fn proxy_bridges_mut_or_default(&mut self) -> &mut LaneProxyBridgeFrontiers {
        let metadata = self.proxy_metadata_mut();
        Arc::make_mut(
            metadata
                .bridges
                .get_or_insert_with(|| Arc::new(LaneProxyBridgeFrontiers::default())),
        )
    }

    fn proxy_bridges_mut(&mut self) -> Option<&mut LaneProxyBridgeFrontiers> {
        self.proxy_metadata
            .as_mut()
            .and_then(|metadata| Arc::make_mut(metadata).bridges.as_mut())
            .map(Arc::make_mut)
    }

    fn clear_proxy_bridges(&mut self) {
        let lane_mask = self.proxy_lane_mask();
        if let Some(metadata) = self.proxy_metadata.as_mut() {
            Arc::make_mut(metadata).bridges = None;
        }
        if lane_mask.is_full() {
            self.proxy_metadata = None;
        }
    }

    fn set_proxy_lane_mask(&mut self, mask: WarpMask) {
        if mask.is_full()
            && self
                .proxy_metadata
                .as_deref()
                .is_none_or(|metadata| metadata.bridges.is_none())
        {
            self.proxy_metadata = None;
            return;
        }
        self.proxy_metadata_mut().lane_mask = mask;
    }

    pub fn warp_count(&self) -> usize {
        self.components.len()
    }

    #[inline(always)]
    pub fn component(&self, warp_id: usize) -> Option<u64> {
        self.components.slot(warp_id)
    }

    #[inline(always)]
    fn direct_component(&self, warp_id: u32) -> u64 {
        debug_assert!((warp_id as usize) < self.components.len());
        self.components.get(warp_id as usize)
    }

    /// The epoch of `token` this clock has observed; zero when the slot has
    /// since been handed to a newer token.
    #[cfg(test)]
    pub fn async_component(&self, token: &AsyncTokenId) -> u64 {
        self.async_registry
            .index_with_generation(token)
            .map(|(index, generation)| {
                let (seen_generation, epoch) = unpack_async_epoch(self.async_component_at(index));
                if seen_generation == generation {
                    epoch
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }

    /// Import an unpacked `(token, epoch)` frontier, packing each epoch under
    /// the slot generation this registry currently holds for the token.
    fn merge_async_frontier(&mut self, frontier: impl IntoIterator<Item = (AsyncTokenId, u64)>) {
        for (token, incoming) in frontier {
            if incoming == 0 {
                continue;
            }
            let (index, generation) = self.async_registry.register_with_generation(&token);
            self.async_components.raise(
                index,
                pack_async_epoch(generation, incoming.min(ASYNC_EPOCH_MASK)),
            );
        }
    }

    fn async_component_at(&self, index: usize) -> u64 {
        self.async_components.get(index)
    }

    fn async_actor_index(&self, token: &AsyncTokenId) -> Option<usize> {
        self.async_registry.index(token)
    }

    /// Zero every TCGEN token component that has not observed the token's
    /// completion. `tcgen_slots_by_chunk` lists, per async-epoch chunk, the
    /// TCGEN slots and their completed epoch (0 while uncompleted).
    fn clear_uncompleted_async_components(&mut self, tcgen_slots_by_chunk: &[Vec<(u16, u64)>]) {
        self.async_components
            .clear_where(tcgen_slots_by_chunk, |component, completed| {
                component != 0 && !(completed != 0 && component >= completed)
            });
    }

    pub fn happens_before(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.async_registry, &other.async_registry)
            && self.components.len() == other.components.len()
            && self
                .components
                .happens_before(&other.components, &self.async_registry.warp_chunk_joins)
            && self
                .async_components
                .happens_before(&other.async_components, &self.async_registry.chunk_joins)
    }

    fn tick(&mut self, warp_id: usize) -> Result<(), RaceShadowError> {
        let warp_count = self.warp_count();
        let component = self
            .components
            .slot(warp_id)
            .ok_or(RaceShadowError::InvalidWarp {
                warp_id,
                warp_count,
            })?;
        let epoch = component
            .checked_add(1)
            .ok_or(RaceShadowError::ClockOverflow { warp_id })?;
        self.components.write(warp_id, epoch);
        Ok(())
    }

    fn acquire_payload(
        &mut self,
        payload: &BarrierClockPayload,
        mask: WarpMask,
    ) -> Result<(), RaceShadowError> {
        self.merge_causal(&payload.clock)?;
        if let Some(bridges) = payload.proxy_bridges.as_deref() {
            self.proxy_bridges_mut_or_default()
                .merge_collapsed(mask, bridges)?;
        }
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        if self.warp_count() != other.warp_count() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: other.warp_count(),
            });
        }
        if !Arc::ptr_eq(&self.async_registry, &other.async_registry) {
            return Err(RaceShadowError::AsyncClockRegistryMismatch);
        }
        self.components
            .merge(&other.components, &self.async_registry.warp_chunk_joins);
        self.async_components
            .merge(&other.async_components, &self.async_registry.chunk_joins);
        if let Some(other_bridges) = other.proxy_bridges_arc() {
            let metadata = self.proxy_metadata_mut();
            if let Some(current_bridges) = metadata.bridges.as_mut() {
                if !Arc::ptr_eq(current_bridges, other_bridges) {
                    Arc::make_mut(current_bridges).merge(other_bridges)?;
                }
            } else {
                metadata.bridges = Some(Arc::clone(other_bridges));
            }
        }
        let merged_mask = self.proxy_lane_mask() | other.proxy_lane_mask();
        self.set_proxy_lane_mask(merged_mask);
        Ok(())
    }

    fn tick_async(&mut self, token: &AsyncTokenId) -> Result<(), RaceShadowError> {
        let (index, generation) = self.async_registry.register_with_generation(token);
        self.tick_async_at(index, generation, token)
    }

    fn tick_async_at(
        &mut self,
        index: usize,
        generation: u32,
        token: &AsyncTokenId,
    ) -> Result<(), RaceShadowError> {
        let (seen_generation, epoch) = unpack_async_epoch(self.async_components.get(index));
        // A value left behind by the slot's previous token restarts the count.
        let epoch = if seen_generation == generation {
            epoch
        } else {
            0
        };
        let epoch = epoch
            .checked_add(1)
            .filter(|epoch| *epoch <= ASYNC_EPOCH_MASK)
            .ok_or_else(|| RaceShadowError::AsyncClockOverflow {
                token: token.clone(),
            })?;
        self.async_components
            .set(index, pack_async_epoch(generation, epoch));
        Ok(())
    }

    fn async_issue_index(
        &self,
        token: &AsyncTokenId,
        issuer_warp: usize,
    ) -> Result<(usize, u32), RaceShadowError> {
        let issue_epoch = self
            .component(issuer_warp)
            .expect("a validated async issuer has a vector-clock component")
            .checked_add(1)
            .ok_or(RaceShadowError::ClockOverflow {
                warp_id: issuer_warp,
            })?;
        Ok(self
            .async_registry
            .register_issue(token, issuer_warp, issue_epoch))
    }

    fn register_async_issue(
        &self,
        token: &AsyncTokenId,
        issuer_warp: usize,
    ) -> Result<(), RaceShadowError> {
        self.async_issue_index(token, issuer_warp).map(|_| ())
    }

    fn tick_async_issue(
        &mut self,
        token: &AsyncTokenId,
        issuer_warp: usize,
    ) -> Result<(), RaceShadowError> {
        let (index, generation) = self.async_issue_index(token, issuer_warp)?;
        self.tick_async_at(index, generation, token)
    }

    pub(crate) fn apply_proxy_async_fence(
        &mut self,
        scope: ProxyAsyncFenceScope,
        active_mask: WarpMask,
    ) -> Result<(), RaceShadowError> {
        let frontier = ProxyClockFrontier::from_clock(self);
        self.apply_proxy_fence(scope, active_mask, &frontier)
    }

    fn apply_proxy_fence(
        &mut self,
        scope: ProxyAsyncFenceScope,
        active_mask: WarpMask,
        frontier: &ProxyClockFrontier,
    ) -> Result<(), RaceShadowError> {
        let lane_bridges = self.proxy_bridges_mut_or_default();
        let prior_domains: &[ProxyMemoryDomain] = match scope {
            ProxyAsyncFenceScope::All => &[
                ProxyMemoryDomain::Global,
                ProxyMemoryDomain::SharedCta,
                ProxyMemoryDomain::SharedCluster,
            ],
            ProxyAsyncFenceScope::Global => &[ProxyMemoryDomain::Global],
            ProxyAsyncFenceScope::SharedCta => &[ProxyMemoryDomain::SharedCta],
            ProxyAsyncFenceScope::SharedCluster => &[ProxyMemoryDomain::SharedCluster],
            // Shared memory has no modeled virtual aliases.
            ProxyAsyncFenceScope::Alias => &[],
        };
        lane_bridges.merge_masked_with(active_mask, |bridges| {
            for prior_domain in prior_domains.iter().copied() {
                for current_domain in proxy_alias_domains(prior_domain).iter().copied() {
                    bridges.merge_direction(
                        ProxyBridgeDirection::GenericToAsync,
                        prior_domain,
                        current_domain,
                        frontier,
                    )?;
                    bridges.merge_direction(
                        ProxyBridgeDirection::AsyncToGeneric,
                        prior_domain,
                        current_domain,
                        frontier,
                    )?;
                }
            }
            Ok(())
        })
    }

    fn proxy_bridge_observes_frontier(
        &self,
        prior_proxy: MemoryProxy,
        current_proxy: MemoryProxy,
        prior_domain: ProxyMemoryDomain,
        current_domain: ProxyMemoryDomain,
        retired: &ProxyClockFrontier,
        current_lane: usize,
    ) -> bool {
        let Some(direction) = ProxyBridgeDirection::between(prior_proxy, current_proxy) else {
            return false;
        };
        self.proxy_bridges()
            .and_then(|bridges| bridges.lane(current_lane))
            .and_then(|bridges| bridges.frontier(direction, prior_domain, current_domain))
            .is_some_and(|frontier| frontier.observes_frontier(retired))
    }

    pub(crate) fn apply_implicit_async_completion(
        &mut self,
        domain: ProxyMemoryDomain,
    ) -> Result<(), RaceShadowError> {
        let frontier = ProxyClockFrontier::from_clock(self);
        let proxy_lane_mask = self.proxy_lane_mask();
        let lane_bridges = self.proxy_bridges_mut_or_default();
        lane_bridges.merge_masked_with(proxy_lane_mask, |bridges| {
            for current_domain in proxy_alias_domains(domain).iter().copied() {
                bridges.merge_direction(
                    ProxyBridgeDirection::AsyncToGeneric,
                    domain,
                    current_domain,
                    &frontier,
                )?;
            }
            Ok(())
        })
    }

    /// A complete-tx observation orders this copy, not arbitrary instructions
    /// that preceded its issue. Keep only its actor and its completion bridges.
    pub(crate) fn copy_completion_projection(
        &self,
        token: &AsyncTokenId,
    ) -> Result<Self, RaceShadowError> {
        let index =
            self.async_actor_index(token)
                .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                    token: token.clone(),
                })?;
        let mut completion = Self::zero(self.warp_count(), Arc::clone(&self.async_registry));
        completion
            .async_components
            .set(index, self.async_components.get(index));
        let mask = self.proxy_lane_mask();
        completion.set_proxy_lane_mask(mask);
        let frontier = ProxyClockFrontier::from_clock(&completion);
        for domain in [
            ProxyMemoryDomain::Global,
            ProxyMemoryDomain::SharedCta,
            ProxyMemoryDomain::SharedCluster,
        ] {
            if mask.into_iter().all(|lane| {
                self.proxy_bridge_observes_frontier(
                    MemoryProxy::Async,
                    MemoryProxy::Generic,
                    domain,
                    domain,
                    &frontier,
                    lane,
                )
            }) {
                completion.apply_implicit_async_completion(domain)?;
            }
        }
        Ok(completion)
    }

    fn proxy_bridge_observes(
        &self,
        prior_proxy: MemoryProxy,
        current_proxy: MemoryProxy,
        prior_domain: ProxyMemoryDomain,
        current_domain: ProxyMemoryDomain,
        timestamp: RaceEventTimestamp,
        registry: &OperationRegistry,
        current_lane: usize,
        source_lane: (usize, usize),
    ) -> bool {
        let Some(direction) = ProxyBridgeDirection::between(prior_proxy, current_proxy) else {
            return false;
        };
        self.proxy_bridges()
            .and_then(|bridges| bridges.lane(current_lane))
            .and_then(|bridges| bridges.frontier(direction, prior_domain, current_domain))
            .is_some_and(|frontier| timestamp.observed_by_frontier(frontier, registry, source_lane))
    }

    fn restrict_proxy_lanes(&mut self, mask: WarpMask) {
        if let Some(bridges) = self
            .proxy_metadata
            .as_mut()
            .and_then(|metadata| Arc::make_mut(metadata).bridges.as_mut())
        {
            Arc::make_mut(bridges).retain(mask);
        }
        self.set_proxy_lane_mask(mask);
    }

    fn merge_causal(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        let current_metadata = self.proxy_metadata.take();
        self.merge(other)?;
        self.proxy_metadata = current_metadata;
        Ok(())
    }

    fn synchronize_proxy_lanes(&mut self, mask: WarpMask) -> Result<(), RaceShadowError> {
        let collapsed = self
            .proxy_bridges()
            .and_then(|bridges| bridges.collapsed(mask));
        let Some(collapsed) = collapsed.as_ref() else {
            return Ok(());
        };
        self.proxy_bridges_mut_or_default()
            .merge_collapsed(mask, collapsed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProxyBridgeDirection {
    GenericToAsync,
    AsyncToGeneric,
}

impl ProxyBridgeDirection {
    const fn between(prior: MemoryProxy, current: MemoryProxy) -> Option<Self> {
        match (prior, current) {
            (MemoryProxy::Generic, MemoryProxy::Async) => Some(Self::GenericToAsync),
            (MemoryProxy::Async, MemoryProxy::Generic) => Some(Self::AsyncToGeneric),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct ProxyClockFrontier {
    components: WarpEpochs,
    async_components: AsyncEpochs,
    async_registry: Arc<AsyncClockRegistry>,
    shared_lanes: SharedLaneEpochs,
}

impl PartialEq for ProxyClockFrontier {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.async_registry, &other.async_registry)
            && self.shared_lanes == other.shared_lanes
            && self
                .components
                .equals(&other.components, &self.async_registry.warp_chunk_joins)
            && self
                .async_components
                .equals(&other.async_components, &self.async_registry.chunk_joins)
    }
}

impl Eq for ProxyClockFrontier {}

impl ProxyClockFrontier {
    fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.async_registry, &other.async_registry)
            && self.components.len() == other.components.len()
            && self.async_components.len() == other.async_components.len()
            && self.components.shares_storage_with(&other.components)
            && self
                .async_components
                .shares_storage_with(&other.async_components)
            && self.shared_lanes.as_ref().map(Arc::as_ptr)
                == other.shared_lanes.as_ref().map(Arc::as_ptr)
    }

    fn from_clock(clock: &RaceVectorClock) -> Self {
        Self {
            components: clock.components.clone(),
            async_components: clock.async_components.clone(),
            async_registry: Arc::clone(&clock.async_registry),
            shared_lanes: None,
        }
    }

    /// A frontier that has observed nothing yet.
    fn empty(warp_count: usize, async_registry: Arc<AsyncClockRegistry>) -> Self {
        Self {
            components: WarpEpochs::zeros(warp_count),
            async_components: AsyncEpochs::zeros(0),
            async_registry,
            shared_lanes: None,
        }
    }

    /// Extend the frontier so that it observes `timestamp`.
    fn raise_timestamp(
        &mut self,
        timestamp: RaceEventTimestamp,
        registry: &OperationRegistry,
        source_lane: Option<(usize, usize)>,
    ) -> Result<(), RaceShadowError> {
        if let (Some((warp, lane)), Some(stamp)) = (source_lane, timestamp.lane_stamp(registry)) {
            let lanes = self
                .shared_lanes
                .get_or_insert_with(|| Arc::new(SharedLaneEpochMap::default()));
            let epoch = &mut Arc::make_mut(lanes).get_or_insert(warp)[lane];
            *epoch = (*epoch).max(stamp.epoch.get());
        }
        match timestamp.resolve(registry) {
            (TimestampActor::Warp(warp), epoch) => self.components.raise(warp, epoch),
            (TimestampActor::Async(index), epoch) => self.async_components.raise(index, epoch),
            (TimestampActor::RegisteredVector(index), _) => {
                let clock = registry.registered_vector_timestamp_clock(index);
                self.merge(&Self::from_clock(&clock))?;
            }
        }
        Ok(())
    }

    fn component(&self, warp_id: usize) -> Option<u64> {
        self.components.slot(warp_id)
    }

    fn async_component_at(&self, index: usize) -> u64 {
        self.async_components.get(index)
    }

    fn observes_clock(&self, clock: &RaceVectorClock) -> bool {
        Arc::ptr_eq(&self.async_registry, &clock.async_registry)
            && self.components.len() == clock.components.len()
            && clock
                .components
                .happens_before(&self.components, &self.async_registry.warp_chunk_joins)
            && clock
                .async_components
                .happens_before(&self.async_components, &self.async_registry.chunk_joins)
    }

    fn observes_frontier(&self, frontier: &Self) -> bool {
        Arc::ptr_eq(&self.async_registry, &frontier.async_registry)
            && shared_lane_epochs_dominate(
                self.shared_lanes.as_deref(), frontier.shared_lanes.as_deref(),
            )
            && self.components.len() == frontier.components.len()
            && frontier
                .components
                .happens_before(&self.components, &self.async_registry.warp_chunk_joins)
            && frontier
                .async_components
                .happens_before(&self.async_components, &self.async_registry.chunk_joins)
    }

    fn merge(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        if self.components.len() != other.components.len() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.components.len(),
                actual_warps: other.components.len(),
            });
        }
        if !Arc::ptr_eq(&self.async_registry, &other.async_registry) {
            return Err(RaceShadowError::AsyncClockRegistryMismatch);
        }
        self.components
            .merge(&other.components, &self.async_registry.warp_chunk_joins);
        self.async_components
            .merge(&other.async_components, &self.async_registry.chunk_joins);
        merge_shared_lane_epochs(&mut self.shared_lanes, &other.shared_lanes);
        Ok(())
    }

    fn meet(&mut self, other: &Self) {
        debug_assert_eq!(self.components.len(), other.components.len());
        debug_assert!(Arc::ptr_eq(&self.async_registry, &other.async_registry));
        self.components.meet(&other.components);
        self.async_components.meet(&other.async_components);
        match (&mut self.shared_lanes, &other.shared_lanes) {
            (Some(current), Some(other)) => Arc::make_mut(current).retain(|warp, lanes| {
                let Some(incoming) = other.get(warp) else {
                    return false;
                };
                for (epoch, incoming) in lanes.iter_mut().zip(incoming) {
                    *epoch = (*epoch).min(*incoming);
                }
                lanes.iter().any(|epoch| *epoch != 0)
            }),
            _ => self.shared_lanes = None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ProxyBridgeFrontiers {
    // A slot is shared by every clock and lane that acquired the same
    // frontier, so merging it into itself is a pointer comparison.
    frontiers: [Option<Arc<ProxyClockFrontier>>; PROXY_BRIDGE_SLOT_COUNT],
}

impl ProxyBridgeFrontiers {
    fn frontier(
        &self,
        direction: ProxyBridgeDirection,
        prior_domain: ProxyMemoryDomain,
        current_domain: ProxyMemoryDomain,
    ) -> Option<&ProxyClockFrontier> {
        let index = proxy_bridge_slot(direction, prior_domain, current_domain)?;
        self.frontiers[index].as_deref()
    }

    fn merge_direction(
        &mut self,
        direction: ProxyBridgeDirection,
        prior_domain: ProxyMemoryDomain,
        current_domain: ProxyMemoryDomain,
        frontier: &ProxyClockFrontier,
    ) -> Result<(), RaceShadowError> {
        let Some(index) = proxy_bridge_slot(direction, prior_domain, current_domain) else {
            return Ok(());
        };
        let slot = &mut self.frontiers[index];
        if let Some(current) = slot {
            Arc::make_mut(current).merge(frontier)
        } else {
            *slot = Some(Arc::new(frontier.clone()));
            Ok(())
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        // Equal immutable input pairs across slots have the same exact join.
        let mut last: Option<(
            Arc<ProxyClockFrontier>,
            &Arc<ProxyClockFrontier>,
            Arc<ProxyClockFrontier>,
        )> = None;
        for (slot, incoming) in self.frontiers.iter_mut().zip(other.frontiers.iter()) {
            let Some(incoming) = incoming else {
                continue;
            };
            match slot {
                None => *slot = Some(Arc::clone(incoming)),
                Some(current) if Arc::ptr_eq(current, incoming) => {}
                Some(current) => {
                    if let Some((prior, prior_incoming, merged)) = &last {
                        if prior.shares_storage_with(current)
                            && prior_incoming.shares_storage_with(incoming)
                        {
                            *current = Arc::clone(merged);
                            continue;
                        }
                    }
                    let prior = Arc::clone(current);
                    let mut merged = current.as_ref().clone();
                    merged.merge(incoming)?;
                    if merged.shares_storage_with(incoming) {
                        *current = Arc::clone(incoming);
                    } else if !merged.shares_storage_with(current) {
                        *current = Arc::new(merged);
                    }
                    last = Some((prior, incoming, Arc::clone(current)));
                }
            }
        }
        Ok(())
    }

    fn meet(&mut self, other: &Self) {
        debug_assert_eq!(self.frontiers.len(), PROXY_BRIDGE_SLOT_COUNT);
        debug_assert_eq!(other.frontiers.len(), PROXY_BRIDGE_SLOT_COUNT);
        for (slot, candidate) in self.frontiers.iter_mut().zip(other.frontiers.iter()) {
            match (slot.as_mut(), candidate.as_deref()) {
                (Some(current), Some(candidate)) => Arc::make_mut(current).meet(candidate),
                (Some(_), None) => *slot = None,
                (None, _) => {}
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaneProxyBridgeFrontiers {
    lanes: [Option<Arc<ProxyBridgeFrontiers>>; WARP_SIZE],
}

impl Default for LaneProxyBridgeFrontiers {
    fn default() -> Self {
        Self {
            lanes: std::array::from_fn(|_| None),
        }
    }
}

impl LaneProxyBridgeFrontiers {
    fn lane(&self, lane: usize) -> Option<&ProxyBridgeFrontiers> {
        self.lanes.get(lane).and_then(Option::as_deref)
    }

    fn lane_mut(&mut self, lane: usize) -> &mut ProxyBridgeFrontiers {
        Arc::make_mut(
            self.lanes[lane].get_or_insert_with(|| Arc::new(ProxyBridgeFrontiers::default())),
        )
    }

    fn merge_masked_with(
        &mut self,
        mask: WarpMask,
        mut merge: impl FnMut(&mut ProxyBridgeFrontiers) -> Result<(), RaceShadowError>,
    ) -> Result<(), RaceShadowError> {
        // Lanes that share one bridge set get one join and keep sharing it.
        let mut groups: Vec<(Option<*const ProxyBridgeFrontiers>, u32)> = Vec::new();
        for lane in mask {
            let identity = self.lanes[lane].as_ref().map(Arc::as_ptr);
            match groups.iter_mut().find(|(current, _)| *current == identity) {
                Some((_, lanes)) => *lanes |= 1 << lane,
                None => groups.push((identity, 1 << lane)),
            }
        }
        for (_, lanes) in groups {
            let lanes = WarpMask::from_bits(lanes);
            let first = lanes.into_iter().next().expect("a lane group is not empty");
            let mut merged = self.lanes[first].as_deref().cloned().unwrap_or_default();
            merge(&mut merged)?;
            let merged = Arc::new(merged);
            for lane in lanes {
                self.lanes[lane] = Some(Arc::clone(&merged));
            }
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        for (lane, incoming) in other.lanes.iter().enumerate() {
            if let Some(incoming) = incoming {
                self.lane_mut(lane).merge(incoming)?;
            }
        }
        Ok(())
    }

    fn retain(&mut self, mask: WarpMask) {
        for (lane, bridges) in self.lanes.iter_mut().enumerate() {
            if !mask.contains(lane) {
                *bridges = None;
            }
        }
    }

    fn collapsed(&self, mask: WarpMask) -> Option<ProxyBridgeFrontiers> {
        let mut collapsed: Option<ProxyBridgeFrontiers> = None;
        let mut last_incoming = None;
        for lane in mask {
            let Some(incoming) = self.lanes[lane].as_ref() else {
                continue;
            };
            let incoming_ptr = Arc::as_ptr(incoming);
            if last_incoming == Some(incoming_ptr) {
                continue;
            }
            last_incoming = Some(incoming_ptr);
            if let Some(current) = &mut collapsed {
                current
                    .merge(incoming)
                    .expect("one race clock uses one proxy-frontier registry");
            } else {
                collapsed = Some(incoming.as_ref().clone());
            }
        }
        collapsed
    }

    fn merge_collapsed(
        &mut self,
        mask: WarpMask,
        incoming: &ProxyBridgeFrontiers,
    ) -> Result<(), RaceShadowError> {
        self.merge_masked_with(mask, |bridges| bridges.merge(incoming))
    }

    fn meet_active(&mut self, other: &Self, other_mask: WarpMask) {
        for lane in other_mask {
            match (self.lanes[lane].as_mut(), other.lanes[lane].as_ref()) {
                (Some(current), Some(candidate)) => Arc::make_mut(current).meet(candidate),
                (Some(_), None) => self.lanes[lane] = None,
                (None, _) => {}
            }
        }
    }
}

const fn proxy_domain_index(domain: ProxyMemoryDomain) -> Option<usize> {
    match domain {
        ProxyMemoryDomain::Global => Some(0),
        ProxyMemoryDomain::SharedCta => Some(1),
        ProxyMemoryDomain::SharedCluster => Some(2),
        ProxyMemoryDomain::Other => None,
    }
}

const fn proxy_bridge_slot(
    direction: ProxyBridgeDirection,
    prior_domain: ProxyMemoryDomain,
    current_domain: ProxyMemoryDomain,
) -> Option<usize> {
    let Some(prior_index) = proxy_domain_index(prior_domain) else {
        return None;
    };
    let Some(current_index) = proxy_domain_index(current_domain) else {
        return None;
    };
    let direction_index = match direction {
        ProxyBridgeDirection::GenericToAsync => 0,
        ProxyBridgeDirection::AsyncToGeneric => 1,
    };
    Some(
        direction_index * PROXY_MEMORY_DOMAIN_COUNT * PROXY_MEMORY_DOMAIN_COUNT
            + prior_index * PROXY_MEMORY_DOMAIN_COUNT
            + current_index,
    )
}

const fn proxy_memory_domains() -> [ProxyMemoryDomain; PROXY_MEMORY_DOMAIN_COUNT] {
    [
        ProxyMemoryDomain::Global,
        ProxyMemoryDomain::SharedCta,
        ProxyMemoryDomain::SharedCluster,
    ]
}

const fn proxy_alias_domains(domain: ProxyMemoryDomain) -> &'static [ProxyMemoryDomain] {
    match domain {
        ProxyMemoryDomain::Global => &[ProxyMemoryDomain::Global],
        ProxyMemoryDomain::SharedCta | ProxyMemoryDomain::SharedCluster => &[
            ProxyMemoryDomain::SharedCta,
            ProxyMemoryDomain::SharedCluster,
        ],
        ProxyMemoryDomain::Other => &[],
    }
}

/// Release-clock payload carried by a barrier implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierClockPayload {
    clock: RaceVectorClock,
    proxy_bridges: Option<Arc<ProxyBridgeFrontiers>>,
}

impl BarrierClockPayload {
    pub fn from_clock(clock: RaceVectorClock) -> Self {
        let mask = clock.proxy_lane_mask();
        Self::from_clock_for_mask(clock, mask)
    }

    fn from_clock_for_mask(mut clock: RaceVectorClock, mask: WarpMask) -> Self {
        let proxy_bridges = clock
            .proxy_bridges()
            .and_then(|bridges| bridges.collapsed(mask))
            .map(Arc::new);
        clock.proxy_metadata = Some(Arc::new(ProxyClockMetadata {
            bridges: None,
            lane_mask: WarpMask::EMPTY,
        }));
        Self {
            clock,
            proxy_bridges,
        }
    }

    pub const fn clock(&self) -> &RaceVectorClock {
        &self.clock
    }

    /// Join another contributor into the payload for a multi-producer barrier.
    pub fn merge(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        self.clock.merge_causal(&other.clock)?;
        if let Some(incoming) = other.proxy_bridges.as_ref() {
            if let Some(current) = self.proxy_bridges.as_mut() {
                if !Arc::ptr_eq(current, incoming) {
                    Arc::make_mut(current).merge(incoming)?;
                }
            } else {
                self.proxy_bridges = Some(Arc::clone(incoming));
            }
        }
        Ok(())
    }
}

pub(crate) type LaneFrontier = [u64; WARP_SIZE];
/// Lane epochs of the warps a frontier has observed, as rows sorted by warp.
/// A frontier holds few rows (a shard's own warps), so joins and dominance
/// checks are one linear pass over both row lists with no hashing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SharedLaneEpochMap {
    rows: Vec<(usize, LaneFrontier)>,
}

impl SharedLaneEpochMap {
    fn position(&self, warp: usize) -> Result<usize, usize> {
        self.rows.binary_search_by_key(&warp, |(row_warp, _)| *row_warp)
    }

    fn get(&self, warp: &usize) -> Option<&LaneFrontier> {
        self.position(*warp).ok().map(|index| &self.rows[index].1)
    }

    fn get_or_insert(&mut self, warp: usize) -> &mut LaneFrontier {
        let index = match self.position(warp) {
            Ok(index) => index,
            Err(index) => {
                self.rows.insert(index, (warp, [0; WARP_SIZE]));
                index
            }
        };
        &mut self.rows[index].1
    }

    fn iter(&self) -> impl Iterator<Item = (&usize, &LaneFrontier)> {
        self.rows.iter().map(|(warp, lanes)| (warp, lanes))
    }

    fn keys(&self) -> impl Iterator<Item = &usize> {
        self.rows.iter().map(|(warp, _)| warp)
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn retain(&mut self, mut keep: impl FnMut(&usize, &mut LaneFrontier) -> bool) {
        self.rows.retain_mut(|(warp, lanes)| keep(warp, lanes));
    }

    /// Every row of `self` is at or below `current`'s row; a row `current`
    /// lacks counts as all zeros.
    fn dominated_by(&self, current: &Self) -> bool {
        let mut rows = current.rows.iter().peekable();
        self.rows.iter().all(|(warp, lanes)| {
            while rows.peek().is_some_and(|(current_warp, _)| current_warp < warp) {
                rows.next();
            }
            match rows.peek() {
                Some((current_warp, current_lanes)) if current_warp == warp => current_lanes
                    .iter()
                    .zip(lanes)
                    .fold(true, |all, (a, b)| all & (a >= b)),
                _ => lanes.iter().all(|epoch| *epoch == 0),
            }
        })
    }

    /// Every row of `self` is present in `current` at or below its row, so a
    /// join would leave `current` unchanged.
    fn joins_nothing_into(&self, current: &Self) -> bool {
        let mut rows = current.rows.iter().peekable();
        self.rows.iter().all(|(warp, lanes)| {
            while rows.peek().is_some_and(|(current_warp, _)| current_warp < warp) {
                rows.next();
            }
            rows.peek().is_some_and(|(current_warp, current_lanes)| {
                current_warp == warp
                    && current_lanes
                        .iter()
                        .zip(lanes)
                        .fold(true, |all, (a, b)| all & (a >= b))
            })
        })
    }

    /// Lane-wise maximum with `incoming`; rows only `incoming` has are added.
    fn join(&mut self, incoming: &Self) {
        let mut merged = Vec::with_capacity(self.rows.len() + incoming.rows.len());
        let mut own = self.rows.iter().peekable();
        let mut theirs = incoming.rows.iter().peekable();
        loop {
            match (own.peek(), theirs.peek()) {
                (Some((warp, lanes)), Some((incoming_warp, incoming_lanes))) => {
                    if warp < incoming_warp {
                        merged.push((*warp, *lanes));
                        own.next();
                    } else if warp > incoming_warp {
                        merged.push((*incoming_warp, *incoming_lanes));
                        theirs.next();
                    } else {
                        let mut joined = *lanes;
                        for (epoch, incoming) in joined.iter_mut().zip(incoming_lanes) {
                            *epoch = (*epoch).max(*incoming);
                        }
                        merged.push((*warp, joined));
                        own.next();
                        theirs.next();
                    }
                }
                (Some(_), None) => {
                    merged.extend(own.copied());
                    break;
                }
                (None, Some(_)) => {
                    merged.extend(theirs.copied());
                    break;
                }
                (None, None) => break,
            }
        }
        self.rows = merged;
    }
}

type SharedLaneEpochs = Option<Arc<SharedLaneEpochMap>>;
pub(crate) type SharedLaneFrontiers = BTreeMap<usize, SharedClockFrontier>;

fn merge_shared_lane_epochs(current: &mut SharedLaneEpochs, incoming: &SharedLaneEpochs) {
    let Some(incoming) = incoming else {
        return;
    };
    let Some(current) = current else {
        *current = Some(Arc::clone(incoming));
        return;
    };
    if Arc::ptr_eq(current, incoming) {
        return;
    }
    // A dominating publication is already the exact join. Keep its identity
    // so other lanes acquiring the same payload can share it as well.
    if current.len() <= incoming.len()
        && shared_lane_epochs_dominate(Some(incoming), Some(current))
    {
        *current = Arc::clone(incoming);
        return;
    }
    if incoming.joins_nothing_into(current) {
        return;
    }
    Arc::make_mut(current).join(incoming);
}

/// Join `incoming` into `current`, keeping only rows for warps in
/// `[base, base + count)`.
///
/// Lane epochs answer point queries for a prior access's `(warp, lane)`, and a
/// shard only validates accesses recorded by its own warps
/// (`RaceShadow::local_warp_id` rejects the rest). Rows for other shards are
/// therefore never read here; the shards that own them acquire them directly
/// from the global model.
fn merge_shared_lane_epochs_within(
    current: &mut SharedLaneEpochs,
    incoming: &SharedLaneEpochs,
    base: usize,
    count: usize,
) {
    let Some(incoming_map) = incoming else {
        return;
    };
    if incoming_map
        .keys()
        .all(|warp| warp.wrapping_sub(base) < count)
    {
        merge_shared_lane_epochs(current, incoming);
        return;
    }
    for (&warp, frontier) in incoming_map
        .iter()
        .filter(|(warp, _)| warp.wrapping_sub(base) < count)
    {
        let current_map = current.get_or_insert_with(|| Arc::new(SharedLaneEpochMap::default()));
        if current_map.get(&warp).is_some_and(|prior| {
            prior
                .iter()
                .zip(frontier)
                .fold(true, |all, (a, b)| all & (a >= b))
        }) {
            continue;
        }
        let prior = Arc::make_mut(current_map).get_or_insert(warp);
        for (prior, incoming) in prior.iter_mut().zip(frontier) {
            *prior = (*prior).max(*incoming);
        }
    }
}

fn shared_lane_epochs_dominate(
    current: Option<&SharedLaneEpochMap>,
    prior: Option<&SharedLaneEpochMap>,
) -> bool {
    prior.is_none_or(|prior| match current {
        Some(current) => prior.dominated_by(current),
        None => prior.iter().all(|(_, lanes)| lanes.iter().all(|epoch| *epoch == 0)),
    })
}

/// History carried by a thread's memory synchronization. Direct accesses use
/// lane epochs; asynchronous actors and proxy bridges retain their existing
/// shard-local clock payload. Relays keep foreign shards' payloads intact.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SharedClockFrontier {
    releases: SharedLaneEpochs,
    clocks: Option<Arc<BTreeMap<usize, BarrierClockPayload>>>,
}

impl SharedClockFrontier {
    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        self.releases.as_ref().map(Arc::as_ptr) == other.releases.as_ref().map(Arc::as_ptr)
            && self.clocks.as_ref().map(Arc::as_ptr) == other.clocks.as_ref().map(Arc::as_ptr)
    }

    pub(crate) fn single(warp_id: usize, frontier: LaneFrontier) -> Self {
        Self {
            releases: Some(Arc::new(SharedLaneEpochMap {
                rows: vec![(warp_id, frontier)],
            })),
            clocks: None,
        }
    }

    #[inline]
    pub(crate) fn release_for(&self, warp_id: usize) -> Option<&LaneFrontier> {
        self.releases.as_ref()?.get(&warp_id)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.releases.is_none() && self.clocks.is_none()
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        merge_shared_lane_epochs(&mut self.releases, &other.releases);
        self.merge_clocks_from(other);
    }

    /// Merge `other` but keep only lane rows for warps in `[base, base + count)`.
    pub(crate) fn merge_within(&mut self, other: &Self, base: usize, count: usize) {
        merge_shared_lane_epochs_within(&mut self.releases, &other.releases, base, count);
        self.merge_clocks_from(other);
    }

    fn merge_clocks_from(&mut self, other: &Self) {
        if let Some(incoming) = &other.clocks {
            match &mut self.clocks {
                None => self.clocks = Some(Arc::clone(incoming)),
                Some(current) if !Arc::ptr_eq(current, incoming) => {
                    for (&base, clock) in incoming.iter() {
                        self.merge_clock(base, clock);
                    }
                }
                _ => {}
            }
        }
    }

    pub(crate) fn merge_clock(&mut self, global_warp_base: usize, clock: &BarrierClockPayload) {
        let clocks = self.clocks.get_or_insert_with(|| Arc::new(BTreeMap::new()));
        match Arc::make_mut(clocks).entry(global_warp_base) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(clock.clone());
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry
                    .get_mut()
                    .merge(clock)
                    .expect("one shared-clock shard per launch range");
            }
        }
    }

    fn clock_for(&self, global_warp_base: usize) -> Option<&BarrierClockPayload> {
        self.clocks.as_ref()?.get(&global_warp_base)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PhysicalRaceKind {
    WriteRead = 0,
    ReadWrite = 1,
    WriteWrite = 2,
}

impl fmt::Display for PhysicalRaceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WriteRead => "unordered write/read",
            Self::ReadWrite => "unordered read/write",
            Self::WriteWrite => "unordered write/write",
        })
    }
}

/// Layer at which an ordering proof failed, derived from the exact failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PhysicalRaceOrderingDomain {
    Execution = 0,
    Memory = 1,
    Completion = 2,
}

impl fmt::Display for PhysicalRaceOrderingDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Execution => "execution",
            Self::Memory => "memory",
            Self::Completion => "completion",
        })
    }
}

/// Checker-visible proxy-fence domain retained only on the cold finding path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PhysicalRaceProxyDomain {
    Global = 0,
    SharedCta = 1,
    SharedCluster = 2,
    Other = 3,
}

impl From<ProxyMemoryDomain> for PhysicalRaceProxyDomain {
    fn from(value: ProxyMemoryDomain) -> Self {
        match value {
            ProxyMemoryDomain::Global => Self::Global,
            ProxyMemoryDomain::SharedCta => Self::SharedCta,
            ProxyMemoryDomain::SharedCluster => Self::SharedCluster,
            ProxyMemoryDomain::Other => Self::Other,
        }
    }
}

impl fmt::Display for PhysicalRaceProxyDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Global => "global",
            Self::SharedCta => "shared::cta",
            Self::SharedCluster => "shared::cluster",
            Self::Other => "other",
        })
    }
}

/// Exact ordering proof that failed for a physical conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhysicalRaceOrderingFailure {
    MissingInterActorSynchronization,
    MissingSameWarpLaneOrder,
    MissingReleaseAcquire,
    AsyncLifetimeNotDrained,
    MissingProxyBridge {
        prior_proxy: MemoryProxy,
        current_proxy: MemoryProxy,
        prior_domain: PhysicalRaceProxyDomain,
        current_domain: PhysicalRaceProxyDomain,
    },
}

impl PhysicalRaceOrderingFailure {
    pub const fn domain(self) -> PhysicalRaceOrderingDomain {
        match self {
            Self::MissingInterActorSynchronization | Self::MissingSameWarpLaneOrder => {
                PhysicalRaceOrderingDomain::Execution
            }
            Self::MissingReleaseAcquire | Self::MissingProxyBridge { .. } => {
                PhysicalRaceOrderingDomain::Memory
            }
            Self::AsyncLifetimeNotDrained => PhysicalRaceOrderingDomain::Completion,
        }
    }
}

impl fmt::Display for PhysicalRaceOrderingFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingInterActorSynchronization => {
                f.write_str("no synchronization edge orders the actors")
            }
            Self::MissingSameWarpLaneOrder => {
                f.write_str("different lanes in the same warp have no lane-order edge")
            }
            Self::MissingReleaseAcquire => {
                f.write_str("no release/acquire edge orders the accesses")
            }
            Self::AsyncLifetimeNotDrained => {
                f.write_str("an asynchronous access is not completed before the conflicting reuse")
            }
            Self::MissingProxyBridge {
                prior_proxy,
                current_proxy,
                prior_domain,
                current_domain,
            } => write!(
                f,
                "no {prior_proxy}->{current_proxy} proxy bridge from {prior_domain} to \
                 {current_domain} orders the accesses"
            ),
        }
    }
}

/// Exact source, lane, and physical byte span for one side of a race.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalRaceWitness {
    operation: Arc<DynamicOpId>,
    lane: u8,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    span: PhysicalByteSpan,
}

impl PhysicalRaceWitness {
    pub(crate) fn new(
        operation: DynamicOpId,
        lane: usize,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
    ) -> Self {
        debug_assert!(lane < crate::WARP_SIZE);
        Self {
            operation: Arc::new(operation),
            lane: lane as u8,
            kind,
            space,
            span,
        }
    }

    pub(crate) fn from_lane(
        lane: &LanePhysicalAccess,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
    ) -> Self {
        Self {
            operation: lane.provenance().shared_operation(),
            lane: lane.provenance().lane() as u8,
            kind,
            space,
            span,
        }
    }

    pub(crate) fn from_parts_shared(
        operation: Arc<DynamicOpId>,
        lane: usize,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
    ) -> Self {
        debug_assert!(lane < crate::WARP_SIZE);
        Self {
            operation,
            lane: lane as u8,
            kind,
            space,
            span,
        }
    }

    pub fn operation(&self) -> &DynamicOpId {
        self.operation.as_ref()
    }

    pub(crate) fn shared_operation(&self) -> Arc<DynamicOpId> {
        Arc::clone(&self.operation)
    }

    /// This witness over `span` (same operation, lane, kind and space).
    pub(crate) fn with_span(&self, span: PhysicalByteSpan) -> Self {
        Self {
            operation: Arc::clone(&self.operation),
            lane: self.lane,
            kind: self.kind,
            space: self.space,
            span,
        }
    }

    pub const fn lane(&self) -> usize {
        self.lane as usize
    }

    pub const fn kind(&self) -> PhysicalAccessKind {
        self.kind
    }

    pub const fn space(&self) -> PhysicalAccessSpace {
        self.space
    }

    pub const fn span(&self) -> PhysicalByteSpan {
        self.span
    }
}

impl fmt::Display for PhysicalRaceWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/lane:{} {} {} {}",
            self.operation.as_ref(),
            self.lane,
            self.kind,
            self.space,
            self.span
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RegisteredOperation(NonZeroU64);

impl RegisteredOperation {
    fn new(index: usize, global_warp_id: usize) -> Self {
        // Store index + 1 in the low word. Besides preserving the existing
        // "fewer than u32::MAX operations" contract, this gives Rust a niche
        // for Option<RegisteredOperation> and, transitively, for the retained
        // witnesses that dominate Racecheck's byte shadow.
        let encoded_index = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .expect("one launch registers fewer than u32::MAX operations");
        let global_warp_id =
            u32::try_from(global_warp_id).expect("one launch has fewer than u32::MAX warps");
        Self(
            NonZeroU64::new((u64::from(global_warp_id) << 32) | u64::from(encoded_index))
                .expect("an encoded operation index is nonzero"),
        )
    }

    fn index(self) -> usize {
        (self.0.get() as u32 - 1) as usize
    }

    fn global_warp_id(self) -> usize {
        (self.0.get() >> 32) as u32 as usize
    }
}

#[derive(Debug)]
pub(crate) struct OperationRegistry {
    global_warp_base: usize,
    topology: Option<crate::LaunchTopology>,
    inner: Mutex<OperationRegistryInner>,
    // Read on every retained-witness comparison, so these stay outside the
    // registry mutex: appends serialize on their own small lock, reads are
    // lock-free.
    wide_timestamps: AppendOnlyTable<WideRetainedTimestamp>,
    wide_witnesses: AppendOnlyTable<WideRetainedWitness>,
}

impl Default for OperationRegistry {
    fn default() -> Self {
        Self::for_warp_range(0)
    }
}

impl OperationRegistry {
    fn for_warp_range(global_warp_base: usize) -> Self {
        Self {
            global_warp_base,
            topology: None,
            inner: Mutex::new(OperationRegistryInner::default()),
            wide_timestamps: AppendOnlyTable::default(),
            wide_witnesses: AppendOnlyTable::default(),
        }
    }
}

const APPEND_ONLY_CHUNK: usize = 4096;
const APPEND_ONLY_MAX_CHUNKS: usize = 1 << 16;

/// Append-only table of `Copy` records with lock-free reads.
///
/// Records are immutable once pushed. Appends serialize on a small mutex;
/// every published index stays readable without locking, which keeps the
/// retained-witness hot path off the registry mutex.
struct AppendOnlyTable<T> {
    chunks: std::sync::OnceLock<Box<[std::sync::OnceLock<Box<[std::sync::OnceLock<T>]>>]>>,
    len: std::sync::atomic::AtomicUsize,
    write: Mutex<()>,
}

impl<T> Default for AppendOnlyTable<T> {
    fn default() -> Self {
        Self {
            chunks: std::sync::OnceLock::new(),
            len: std::sync::atomic::AtomicUsize::new(0),
            write: Mutex::new(()),
        }
    }
}

impl<T: Copy> fmt::Debug for AppendOnlyTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppendOnlyTable")
            .field("len", &self.len.load(std::sync::atomic::Ordering::Acquire))
            .finish()
    }
}

impl<T: Copy> AppendOnlyTable<T> {
    fn push(&self, value: T) -> usize {
        let _guard = self
            .write
            .lock()
            .expect("racecheck append-only table lock was poisoned");
        let index = self.len.load(std::sync::atomic::Ordering::Relaxed);
        let chunk_index = index / APPEND_ONLY_CHUNK;
        assert!(
            chunk_index < APPEND_ONLY_MAX_CHUNKS,
            "racecheck append-only table exhausted"
        );
        let chunks = self.chunks.get_or_init(|| {
            (0..APPEND_ONLY_MAX_CHUNKS)
                .map(|_| std::sync::OnceLock::new())
                .collect()
        });
        let chunk = chunks[chunk_index].get_or_init(|| {
            (0..APPEND_ONLY_CHUNK)
                .map(|_| std::sync::OnceLock::new())
                .collect()
        });
        if chunk[index % APPEND_ONLY_CHUNK].set(value).is_err() {
            panic!("racecheck append-only table slot was written twice");
        }
        self.len
            .store(index + 1, std::sync::atomic::Ordering::Release);
        index
    }

    fn get(&self, index: usize) -> Option<T> {
        if index >= self.len.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        self.chunks
            .get()
            .and_then(|chunks| chunks[index / APPEND_ONLY_CHUNK].get())
            .and_then(|chunk| chunk[index % APPEND_ONLY_CHUNK].get())
            .copied()
    }
}

#[derive(Debug, Default)]
struct OperationRegistryInner {
    kernel_index: Option<usize>,
    operation_indices_by_warp: Vec<Vec<u32>>,
    // Retain operation metadata inline on the race-clean hot path. A finding
    // materializes the cold-path `Arc` only when it needs to escape the shadow.
    operations: OperationStore,
    loop_frames: Vec<Arc<[LoopFrame]>>,
    loop_frame_cache_by_warp: Vec<Option<LoopFrameCacheEntry>>,
    wide_source_op_ids: Vec<crate::StaticOpId>,
    wide_per_warp_sequences: Vec<u64>,
    tmem_load_register_sources: HashSet<RegisteredOperation>,
    vector_timestamp_actors: Vec<Arc<RaceVectorClock>>,
}

const OPERATION_CHUNK_SIZE: usize = 1 << 16;

#[derive(Clone, Copy, Debug)]
struct RegisteredOperationMetadata {
    per_warp_sequence: RegisteredPerWarpSequence,
    source_op_id: RegisteredStaticOpId,
    loop_frames_index: u32,
}

#[derive(Clone, Copy, Debug)]
struct RegisteredPerWarpSequence(u32);

impl RegisteredPerWarpSequence {
    const WIDE_BIT: u32 = 1_u32 << 31;

    fn register(inner: &mut OperationRegistryInner, sequence: u64) -> Self {
        if sequence < u64::from(Self::WIDE_BIT) {
            return Self(sequence as u32);
        }
        let index = u32::try_from(inner.wide_per_warp_sequences.len())
            .expect("one launch retains fewer than 2^31 wide per-warp sequences");
        assert!(
            index < Self::WIDE_BIT,
            "one launch retains fewer than 2^31 wide per-warp sequences"
        );
        inner.wide_per_warp_sequences.push(sequence);
        Self(Self::WIDE_BIT | index)
    }

    fn resolve(self, inner: &OperationRegistryInner) -> u64 {
        if self.0 & Self::WIDE_BIT == 0 {
            return u64::from(self.0);
        }
        inner.wide_per_warp_sequences[(self.0 & !Self::WIDE_BIT) as usize]
    }
}

#[derive(Clone, Copy, Debug)]
struct RegisteredStaticOpId(u32);

impl RegisteredStaticOpId {
    const WIDE_BIT: u32 = 1_u32 << 31;

    fn register(inner: &mut OperationRegistryInner, source_op_id: crate::StaticOpId) -> Self {
        if let Ok(compact) = u32::try_from(source_op_id.get()) {
            if compact < Self::WIDE_BIT {
                return Self(compact);
            }
        }
        let index = u32::try_from(inner.wide_source_op_ids.len())
            .expect("one launch retains fewer than 2^31 wide source operation IDs");
        assert!(
            index < Self::WIDE_BIT,
            "one launch retains fewer than 2^31 wide source operation IDs"
        );
        inner.wide_source_op_ids.push(source_op_id);
        Self(Self::WIDE_BIT | index)
    }

    fn resolve(self, inner: &OperationRegistryInner) -> crate::StaticOpId {
        if self.0 & Self::WIDE_BIT == 0 {
            return crate::StaticOpId::new(u64::from(self.0));
        }
        inner.wide_source_op_ids[(self.0 & !Self::WIDE_BIT) as usize]
    }
}

#[derive(Clone, Copy, Debug)]
struct LoopFrameCacheEntry {
    pointer: usize,
    len: usize,
    index: u32,
}

#[derive(Debug, Default)]
struct OperationStore {
    chunks: Vec<Vec<RegisteredOperationMetadata>>,
    len: usize,
}

impl OperationStore {
    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, operation: RegisteredOperationMetadata) {
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.len() == OPERATION_CHUNK_SIZE)
        {
            self.chunks.push(Vec::with_capacity(OPERATION_CHUNK_SIZE));
        }
        self.chunks
            .last_mut()
            .expect("an operation chunk was just provisioned")
            .push(operation);
        self.len += 1;
    }

    fn get(&self, index: usize) -> Option<&RegisteredOperationMetadata> {
        if index >= self.len {
            return None;
        }
        self.chunks
            .get(index / OPERATION_CHUNK_SIZE)?
            .get(index % OPERATION_CHUNK_SIZE)
    }

    fn iter(&self) -> impl Iterator<Item = &RegisteredOperationMetadata> {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }
}

impl OperationRegistry {
    fn register_operation_inner(
        inner: &mut OperationRegistryInner,
        operation: &DynamicOpId,
    ) -> RegisteredOperation {
        match inner.kernel_index {
            Some(kernel_index) => assert_eq!(
                kernel_index,
                operation.kernel_index(),
                "one race shadow only registers operations from one kernel phase"
            ),
            None => inner.kernel_index = Some(operation.kernel_index()),
        }
        let global_warp_id = operation.global_warp_id();
        if inner.loop_frame_cache_by_warp.len() <= global_warp_id {
            inner
                .loop_frame_cache_by_warp
                .resize(global_warp_id + 1, None);
        }
        let loop_frames = operation.shared_loop_frames();
        let pointer = loop_frames.as_ptr() as usize;
        let len = loop_frames.len();
        let loop_frames_index = match inner.loop_frame_cache_by_warp[global_warp_id] {
            Some(cached) if cached.pointer == pointer && cached.len == len => cached.index,
            _ => {
                let index = inner.loop_frames.len();
                let compact_index = u32::try_from(index)
                    .expect("one launch retains fewer than u32::MAX loop-frame snapshots");
                inner.loop_frames.push(Arc::clone(loop_frames));
                inner.loop_frame_cache_by_warp[global_warp_id] = Some(LoopFrameCacheEntry {
                    pointer,
                    len,
                    index: compact_index,
                });
                compact_index
            }
        };
        let handle = RegisteredOperation::new(inner.operations.len(), global_warp_id);
        let source_op_id = RegisteredStaticOpId::register(inner, operation.source_op_id());
        let per_warp_sequence =
            RegisteredPerWarpSequence::register(inner, operation.per_warp_sequence());
        inner.operations.push(RegisteredOperationMetadata {
            per_warp_sequence,
            source_op_id,
            loop_frames_index,
        });
        handle
    }

    fn register(&self, operation: &DynamicOpId) -> RegisteredOperation {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        Self::register_inner(&mut inner, operation)
    }

    fn register_exclusive(&mut self, operation: &DynamicOpId) -> RegisteredOperation {
        let inner = self
            .inner
            .get_mut()
            .expect("racecheck operation registry lock was poisoned");
        Self::register_inner(inner, operation)
    }

    fn register_unique(&self, operation: &DynamicOpId) -> RegisteredOperation {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        Self::register_unique_inner(&mut inner, operation)
    }

    fn register_unique_exclusive(&mut self, operation: &DynamicOpId) -> RegisteredOperation {
        let inner = self
            .inner
            .get_mut()
            .expect("racecheck operation registry lock was poisoned");
        Self::register_unique_inner(inner, operation)
    }

    fn register_unique_inner(
        inner: &mut OperationRegistryInner,
        operation: &DynamicOpId,
    ) -> RegisteredOperation {
        Self::register_operation_inner(inner, operation)
    }

    fn register_inner(
        inner: &mut OperationRegistryInner,
        operation: &DynamicOpId,
    ) -> RegisteredOperation {
        if let Some(kernel_index) = inner.kernel_index {
            assert_eq!(
                kernel_index,
                operation.kernel_index(),
                "one race shadow only registers operations from one kernel phase"
            );
        }
        let global_warp_id = operation.global_warp_id();
        if inner.operation_indices_by_warp.len() <= global_warp_id {
            inner
                .operation_indices_by_warp
                .resize_with(global_warp_id + 1, Vec::new);
        }
        let sequence = operation.per_warp_sequence();
        let position = {
            let indices = &inner.operation_indices_by_warp[global_warp_id];
            let lower = indices.partition_point(|index| {
                inner
                    .operations
                    .get(*index as usize)
                    .expect("a per-warp operation index remains registered")
                    .per_warp_sequence
                    .resolve(inner)
                    < sequence
            });
            let upper = indices[lower..].partition_point(|index| {
                inner
                    .operations
                    .get(*index as usize)
                    .expect("a per-warp operation index remains registered")
                    .per_warp_sequence
                    .resolve(inner)
                    == sequence
            }) + lower;
            indices[lower..upper]
                .iter()
                .position(|index| {
                    let registered = inner
                        .operations
                        .get(*index as usize)
                        .expect("a per-warp operation index remains registered");
                    registered.source_op_id.resolve(inner) == operation.source_op_id()
                        && inner.loop_frames[registered.loop_frames_index as usize].as_ref()
                            == operation.loop_frames()
                })
                .map(|offset| lower + offset)
                .ok_or(upper)
        };
        if let Ok(position) = position {
            let index = inner.operation_indices_by_warp[global_warp_id][position];
            return RegisteredOperation::new(index as usize, global_warp_id);
        }
        let handle = Self::register_operation_inner(inner, operation);
        let compact_index = u32::try_from(handle.index())
            .expect("one launch registers fewer than u32::MAX operations");
        let indices = &mut inner.operation_indices_by_warp[global_warp_id];
        let position = position.expect_err("a missing operation has an insertion position");
        if position == indices.len() {
            indices.push(compact_index);
        } else {
            indices.insert(position, compact_index);
        }
        handle
    }

    fn register_tmem_load_register_source(&self, operation: &DynamicOpId) -> RegisteredOperation {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        let handle = Self::register_inner(&mut inner, operation);
        inner.tmem_load_register_sources.insert(handle);
        handle
    }

    fn register_tmem_load_register_source_exclusive(
        &mut self,
        operation: &DynamicOpId,
    ) -> RegisteredOperation {
        let inner = self
            .inner
            .get_mut()
            .expect("racecheck operation registry lock was poisoned");
        let handle = Self::register_inner(inner, operation);
        inner.tmem_load_register_sources.insert(handle);
        handle
    }

    fn is_tmem_load_register_source(&self, operation: RegisteredOperation) -> bool {
        self.inner
            .lock()
            .expect("racecheck operation registry lock was poisoned")
            .tmem_load_register_sources
            .contains(&operation)
    }

    /// Repeated spans of the same dynamic operation are already validated.
    /// A shared warp or static site does not prove order between async actors.
    fn owns_witnesses(
        &self,
        current: RegisteredOperation,
        mut witnesses: impl Iterator<Item = RegisteredOperation>,
    ) -> bool {
        witnesses.all(|operation| operation == current)
    }

    fn owned_spans_for_batch<'a>(
        &self,
        current: RegisteredOperation,
        batch: &PhysicalAccessBatch,
        segments_of: impl Fn(PhysicalByteSpan) -> Option<&'a ShadowSegments>,
    ) -> Vec<bool> {
        batch
            .lanes()
            .iter()
            .flat_map(|lane| lane.footprint().spans().iter().copied())
            .map(|span| {
                segments_of(span)
                    .and_then(|segments| segments.get(span.byte_offset()))
                    .filter(|segment| segment.end == span.byte_end())
                    .is_some_and(|segment| {
                        self.owns_witnesses(
                            current,
                            segment
                                .state
                                .writes
                                .iter()
                                .chain(segment.state.reads.iter())
                                .map(|retained| retained.witness.operation(self)),
                        )
                    })
            })
            .collect()
    }


    fn operation(&self, handle: RegisteredOperation) -> Arc<DynamicOpId> {
        let inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        let operation = *inner
            .operations
            .get(handle.index())
            .expect("a retained race witness has a registered operation");
        let kernel_index = inner
            .kernel_index
            .expect("a registered operation has a kernel phase");
        Arc::new(DynamicOpId::new_shared(
            kernel_index,
            handle.global_warp_id(),
            operation.per_warp_sequence.resolve(&inner),
            operation.source_op_id.resolve(&inner),
            Arc::clone(
                inner
                    .loop_frames
                    .get(operation.loop_frames_index as usize)
                    .expect("a registered operation has retained loop frames"),
            ),
        ))
    }

    fn register_vector_timestamp_actor(&self, clock: Arc<RaceVectorClock>) -> usize {
        let mut inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        let index = inner.vector_timestamp_actors.len();
        inner.vector_timestamp_actors.push(clock);
        index
    }

    fn registered_vector_timestamp_clock(&self, index: usize) -> Arc<RaceVectorClock> {
        Arc::clone(
            self.inner
                .lock()
                .expect("racecheck operation registry lock was poisoned")
                .vector_timestamp_actors
                .get(index)
                .expect("a retained race timestamp has a registered actor"),
        )
    }

    fn registered_vector_timestamp_observed_by(
        &self,
        index: usize,
        clock: &RaceVectorClock,
    ) -> bool {
        self.inner
            .lock()
            .expect("racecheck operation registry lock was poisoned")
            .vector_timestamp_actors
            .get(index)
            .expect("a retained race timestamp has a registered actor")
            .happens_before(clock)
    }

    fn registered_vector_timestamp_ordinary_observed_by(
        &self,
        index: usize,
        clock: &RaceVectorClock,
        current_issue: Option<AsyncIssueEvent>,
    ) -> bool {
        let inner = self
            .inner
            .lock()
            .expect("racecheck operation registry lock was poisoned");
        let prior = inner
            .vector_timestamp_actors
            .get(index)
            .expect("a retained race timestamp has a registered actor");
        let observed = prior.components.nonzero().all(|(warp_id, prior)| {
            let observed = current_issue
                .filter(|issue| issue.issuer_warp == warp_id)
                .map_or(clock.components.get(warp_id), |issue| {
                    clock.components.get(warp_id).max(issue.epoch)
                });
            prior <= observed
        });
        observed
    }

    fn registered_vector_timestamp_observed_by_frontier(
        &self,
        index: usize,
        frontier: &ProxyClockFrontier,
    ) -> bool {
        frontier.observes_clock(
            self.inner
                .lock()
                .expect("racecheck operation registry lock was poisoned")
                .vector_timestamp_actors
                .get(index)
                .expect("a retained race timestamp has a registered actor"),
        )
    }

    fn register_wide_timestamp(&self, timestamp: WideRetainedTimestamp) -> usize {
        let index = self.wide_timestamps.push(timestamp);
        assert!(
            u64::try_from(index).is_ok_and(|index| index < RaceEventTimestamp::WIDE_BIT),
            "one launch retains fewer than 2^63 wide timestamps"
        );
        index
    }

    fn wide_timestamp(&self, index: usize) -> WideRetainedTimestamp {
        self.wide_timestamps
            .get(index)
            .expect("a retained wide timestamp has a registered value")
    }

    fn register_wide_witness(&self, witness: WideRetainedWitness) -> usize {
        let index = self.wide_witnesses.push(witness);
        assert!(
            u64::try_from(index).is_ok_and(|index| index < RetainedRaceWitness::WIDE_BIT),
            "one launch retains fewer than 2^63 wide race witnesses"
        );
        index
    }

    fn wide_witness(&self, index: usize) -> WideRetainedWitness {
        self.wide_witnesses
            .get(index)
            .expect("a retained wide race witness has a registered value")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalRaceWitnessRef {
    operation: RegisteredOperation,
    lane: u8,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy: MemoryProxy,
    proxy_domain: ProxyMemoryDomain,
    strong_scope: Option<crate::MemoryScope>,
    span: PhysicalByteSpan,
}

impl PhysicalRaceWitnessRef {
    fn from_lane(
        operation: RegisteredOperation,
        lane: &LanePhysicalAccess,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
        span: PhysicalByteSpan,
    ) -> Self {
        Self {
            operation,
            lane: lane.provenance().lane() as u8,
            kind,
            space,
            proxy,
            proxy_domain,
            strong_scope: None,
            span,
        }
    }

    fn from_parts(
        operation: RegisteredOperation,
        lane: usize,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
        span: PhysicalByteSpan,
    ) -> Self {
        debug_assert!(lane < WARP_SIZE);
        Self {
            operation,
            lane: lane as u8,
            kind,
            space,
            proxy,
            proxy_domain,
            strong_scope: None,
            span,
        }
    }

    fn same_operation(&self, other: &Self) -> bool {
        self.operation == other.operation
    }

    fn with_memory_semantics(mut self, semantics: crate::MemoryAccessSemantics) -> Self {
        debug_assert_eq!(self.proxy, semantics.proxy());
        self.strong_scope = semantics
            .scope()
            .filter(|_| semantics.order().is_strong() && semantics.class().is_atomic_class());
        self
    }

    fn global_warp_id(&self) -> usize {
        self.operation.global_warp_id()
    }

    fn lane(&self) -> usize {
        self.lane as usize
    }

    fn kind(&self) -> PhysicalAccessKind {
        self.kind
    }

    fn space(&self) -> PhysicalAccessSpace {
        self.space
    }

    fn proxy(&self) -> MemoryProxy {
        self.proxy
    }

    fn proxy_domain(&self) -> ProxyMemoryDomain {
        self.proxy_domain
    }

    fn span(&self) -> PhysicalByteSpan {
        self.span
    }

    fn materialize(&self, registry: &OperationRegistry) -> PhysicalRaceWitness {
        PhysicalRaceWitness {
            operation: registry.operation(self.operation),
            lane: self.lane,
            kind: self.kind,
            space: self.space,
            span: self.span,
        }
    }

    fn retained(self, registry: &OperationRegistry) -> RetainedRaceWitness {
        RetainedRaceWitness::new(
            self.operation,
            self.span,
            self.lane,
            self.kind,
            self.space,
            self.proxy,
            self.proxy_domain,
            self.strong_scope,
            registry,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideRetainedWitness {
    operation: RegisteredOperation,
    span: PhysicalByteSpan,
    lane: u8,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy: MemoryProxy,
    proxy_domain: ProxyMemoryDomain,
    strong_scope: Option<crate::MemoryScope>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetainedRaceWitness(NonZeroU64);

impl RetainedRaceWitness {
    fn subsumes_contract(self, registry: &OperationRegistry, prior: Self) -> bool {
        if self.resolved_kind(registry) != prior.resolved_kind(registry) {
            return false;
        }
        if self.0.get() & Self::WIDE_BIT == 0 {
            return true; // Compact witnesses cannot carry a strong scope.
        }
        let current = registry.wide_witness((self.0.get() & Self::VALUE_MASK) as usize);
        if current.strong_scope.is_none() {
            return true;
        }
        let prior = prior.resolve(registry, current.span.allocation());
        current.strong_scope == prior.strong_scope
            && current.proxy == prior.proxy
            && current.span == prior.span
            && crate::race_check::scope_covers_warps(
                registry.topology,
                current.strong_scope.expect("strong access"),
                current.operation.global_warp_id(),
                prior.operation.global_warp_id(),
            )
    }

    // The compact representation covers the launch-local ranges used by the
    // byte-shadow hot path:
    //
    //   operation index + 1  18 bits
    //   shard-local warp id   5 bits  (a two-CTA cluster shard holds 32 warps)
    //   byte offset          23 bits  (8 MiB; shared and TMEM windows are far smaller)
    //   byte length class     7 bits
    //   lane                  5 bits
    //   access kind           2 bits
    //   memory space          1 bit
    //   shared proxy/domain   2 bits
    //   wide marker           1 bit
    //
    // Values outside any compact field retain the exact operation and access
    // in the registry's wide table. Keeping the common witness in one word
    // makes ClockedWitness 16 bytes instead of 24 without weakening evidence.
    const WIDE_BIT: u64 = 1_u64 << 63;
    const VALUE_MASK: u64 = Self::WIDE_BIT - 1;
    const OPERATION_INDEX_BITS: u32 = 18;
    const OPERATION_INDEX_MASK: u64 = (1_u64 << Self::OPERATION_INDEX_BITS) - 1;
    const LOCAL_WARP_SHIFT: u32 = Self::OPERATION_INDEX_BITS;
    const LOCAL_WARP_BITS: u32 = 5;
    const LOCAL_WARP_MASK: u64 = (1_u64 << Self::LOCAL_WARP_BITS) - 1;
    const OPERATION_MASK: u64 = (1_u64 << (Self::OPERATION_INDEX_BITS + Self::LOCAL_WARP_BITS)) - 1;
    const BYTE_OFFSET_SHIFT: u32 = Self::LOCAL_WARP_SHIFT + Self::LOCAL_WARP_BITS;
    const BYTE_OFFSET_BITS: u32 = 23;
    const BYTE_OFFSET_MASK: u64 = (1_u64 << Self::BYTE_OFFSET_BITS) - 1;
    const BYTE_LENGTH_SHIFT: u32 = Self::BYTE_OFFSET_SHIFT + Self::BYTE_OFFSET_BITS;
    const BYTE_LENGTH_BITS: u32 = 7;
    const BYTE_LENGTH_MASK: u64 = (1_u64 << Self::BYTE_LENGTH_BITS) - 1;
    const BYTE_LENGTH_POWER_OF_TWO_BIT: u64 = 1_u64 << (Self::BYTE_LENGTH_BITS - 1);
    const LANE_SHIFT: u32 = Self::BYTE_LENGTH_SHIFT + Self::BYTE_LENGTH_BITS;
    const KIND_SHIFT: u32 = Self::LANE_SHIFT + 5;
    const SPACE_SHIFT: u32 = Self::KIND_SHIFT + 2;
    const SHARED_PROXY_DOMAIN_SHIFT: u32 = Self::SPACE_SHIFT + 1;

    #[inline(always)]
    fn encode_byte_len(byte_len: usize) -> Option<u64> {
        if byte_len.is_power_of_two() {
            let exponent = u64::from(byte_len.trailing_zeros());
            return (exponent < Self::BYTE_LENGTH_POWER_OF_TWO_BIT)
                .then_some(Self::BYTE_LENGTH_POWER_OF_TWO_BIT | exponent);
        }
        byte_len.checked_sub(1).and_then(|encoded| {
            ((encoded as u64) < Self::BYTE_LENGTH_POWER_OF_TWO_BIT).then_some(encoded as u64)
        })
    }

    #[inline(always)]
    fn decode_byte_len(encoded: u64) -> usize {
        if encoded & Self::BYTE_LENGTH_POWER_OF_TWO_BIT != 0 {
            return 1_usize
                .checked_shl((encoded & (Self::BYTE_LENGTH_POWER_OF_TWO_BIT - 1)) as u32)
                .expect("a compact retained access has a representable power-of-two length");
        }
        encoded as usize + 1
    }

    #[inline(always)]
    fn new(
        operation: RegisteredOperation,
        span: PhysicalByteSpan,
        lane: u8,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
        strong_scope: Option<crate::MemoryScope>,
        registry: &OperationRegistry,
    ) -> Self {
        let registered_operation = operation.0.get();
        let encoded_operation_index = registered_operation as u32 as u64;
        let global_warp_id = registered_operation >> 32;
        let local_warp_id = (global_warp_id as usize).checked_sub(registry.global_warp_base);
        let byte_offset = span.byte_offset();
        let byte_len = span.byte_len();
        let compact_kind = match kind {
            PhysicalAccessKind::Read => 0_u64,
            PhysicalAccessKind::Write => 1_u64,
            PhysicalAccessKind::AtomicReadModifyWrite => 2_u64,
        };
        let compact_space = match space {
            PhysicalAccessSpace::Shared => Some(0_u64),
            PhysicalAccessSpace::Tmem => Some(1_u64),
            _ => None,
        };
        let compact_proxy_domain = match (space, proxy, proxy_domain) {
            (PhysicalAccessSpace::Shared, MemoryProxy::Generic, ProxyMemoryDomain::SharedCta) => {
                Some(0_u64)
            }
            (
                PhysicalAccessSpace::Shared,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCluster,
            ) => Some(1_u64),
            (PhysicalAccessSpace::Shared, MemoryProxy::Async, ProxyMemoryDomain::SharedCta) => {
                Some(2_u64)
            }
            (PhysicalAccessSpace::Shared, MemoryProxy::Async, ProxyMemoryDomain::SharedCluster) => {
                Some(3_u64)
            }
            (PhysicalAccessSpace::Tmem, MemoryProxy::Generic, ProxyMemoryDomain::Other) => {
                Some(0_u64)
            }
            _ => None,
        };
        if let (Some(local_warp_id), Some(space), Some(encoded_byte_len), Some(proxy_domain)) = (
            local_warp_id,
            compact_space,
            Self::encode_byte_len(byte_len),
            compact_proxy_domain,
        ) {
            if encoded_operation_index <= Self::OPERATION_INDEX_MASK
                && (local_warp_id as u64) <= Self::LOCAL_WARP_MASK
                && (byte_offset as u64) <= Self::BYTE_OFFSET_MASK
                && (encoded_byte_len as u64) <= Self::BYTE_LENGTH_MASK
                && usize::from(lane) < WARP_SIZE
                && strong_scope.is_none()
            {
                let value = encoded_operation_index
                    | ((local_warp_id as u64) << Self::LOCAL_WARP_SHIFT)
                    | ((byte_offset as u64) << Self::BYTE_OFFSET_SHIFT)
                    | ((encoded_byte_len as u64) << Self::BYTE_LENGTH_SHIFT)
                    | (u64::from(lane) << Self::LANE_SHIFT)
                    | (compact_kind << Self::KIND_SHIFT)
                    | (space << Self::SPACE_SHIFT)
                    | (proxy_domain << Self::SHARED_PROXY_DOMAIN_SHIFT);
                return Self(
                    NonZeroU64::new(value)
                        .expect("a compact witness has a nonzero encoded operation index"),
                );
            }
        }
        let index = registry.register_wide_witness(WideRetainedWitness {
            operation,
            span,
            lane,
            kind,
            space,
            proxy,
            proxy_domain,
            strong_scope,
        });
        Self(
            NonZeroU64::new(
                Self::WIDE_BIT
                    | u64::try_from(index)
                        .expect("a wide witness index fits in the retained handle"),
            )
            .expect("a wide race witness has its marker bit set"),
        )
    }

    fn resolve(
        self,
        registry: &OperationRegistry,
        allocation: PhysicalAllocationId,
    ) -> WideRetainedWitness {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            return registry.wide_witness((value & Self::VALUE_MASK) as usize);
        }
        let encoded_operation_index = value & Self::OPERATION_INDEX_MASK;
        let local_warp_id = ((value >> Self::LOCAL_WARP_SHIFT) & Self::LOCAL_WARP_MASK) as usize;
        let operation = RegisteredOperation::new(
            (encoded_operation_index - 1) as usize,
            registry.global_warp_base + local_warp_id,
        );
        let byte_offset = ((value >> Self::BYTE_OFFSET_SHIFT) & Self::BYTE_OFFSET_MASK) as usize;
        let byte_len =
            Self::decode_byte_len((value >> Self::BYTE_LENGTH_SHIFT) & Self::BYTE_LENGTH_MASK);
        let lane = ((value >> Self::LANE_SHIFT) & 0x1f) as u8;
        let kind = match (value >> Self::KIND_SHIFT) & 0x3 {
            0 => PhysicalAccessKind::Read,
            1 => PhysicalAccessKind::Write,
            2 => PhysicalAccessKind::AtomicReadModifyWrite,
            _ => unreachable!("compact retained access has a valid kind"),
        };
        let space = if (value >> Self::SPACE_SHIFT) & 0x1 == 0 {
            PhysicalAccessSpace::Shared
        } else {
            PhysicalAccessSpace::Tmem
        };
        let (proxy, proxy_domain) = Self::compact_proxy_and_domain(value);
        WideRetainedWitness {
            operation,
            span: PhysicalByteSpan::new(allocation, byte_offset, byte_len)
                .expect("a compact retained access has a valid non-empty span"),
            lane,
            kind,
            space,
            proxy,
            proxy_domain,
            strong_scope: None,
        }
    }

    #[inline(always)]
    fn compact_proxy_and_domain(value: u64) -> (MemoryProxy, ProxyMemoryDomain) {
        if (value >> Self::SPACE_SHIFT) & 0x1 != 0 {
            return (MemoryProxy::Generic, ProxyMemoryDomain::Other);
        }
        match (value >> Self::SHARED_PROXY_DOMAIN_SHIFT) & 0x3 {
            0 => (MemoryProxy::Generic, ProxyMemoryDomain::SharedCta),
            1 => (MemoryProxy::Generic, ProxyMemoryDomain::SharedCluster),
            2 => (MemoryProxy::Async, ProxyMemoryDomain::SharedCta),
            3 => (MemoryProxy::Async, ProxyMemoryDomain::SharedCluster),
            _ => unreachable!("two-bit shared proxy/domain class is exhaustive"),
        }
    }

    #[inline(always)]
    const fn compact_proxy_and_domain_key(
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
    ) -> Option<u64> {
        let shared_class = match (proxy, proxy_domain) {
            (MemoryProxy::Generic, ProxyMemoryDomain::SharedCta) => 0_u64,
            (MemoryProxy::Generic, ProxyMemoryDomain::SharedCluster) => 1_u64,
            (MemoryProxy::Async, ProxyMemoryDomain::SharedCta) => 2_u64,
            (MemoryProxy::Async, ProxyMemoryDomain::SharedCluster) => 3_u64,
            (MemoryProxy::Generic, ProxyMemoryDomain::Other) => {
                return Some(1_u64 << Self::SPACE_SHIFT);
            }
            _ => return None,
        };
        Some(shared_class << Self::SHARED_PROXY_DOMAIN_SHIFT)
    }

    #[inline(always)]
    fn operation(self, registry: &OperationRegistry) -> RegisteredOperation {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            return registry
                .wide_witness((value & Self::VALUE_MASK) as usize)
                .operation;
        }
        RegisteredOperation::new(
            ((value & Self::OPERATION_INDEX_MASK) - 1) as usize,
            registry.global_warp_base
                + ((value >> Self::LOCAL_WARP_SHIFT) & Self::LOCAL_WARP_MASK) as usize,
        )
    }

    #[inline(always)]
    fn compact_operation_key(self) -> Option<u64> {
        let value = self.0.get();
        (value & Self::WIDE_BIT == 0).then_some(value & Self::OPERATION_MASK)
    }

    #[inline(always)]
    fn resolved_lane(self, registry: &OperationRegistry) -> usize {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            return registry
                .wide_witness((value & Self::VALUE_MASK) as usize)
                .lane as usize;
        }
        ((value >> Self::LANE_SHIFT) & 0x1f) as usize
    }

    #[inline(always)]
    fn resolved_kind(self, registry: &OperationRegistry) -> PhysicalAccessKind {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            return registry
                .wide_witness((value & Self::VALUE_MASK) as usize)
                .kind;
        }
        match (value >> Self::KIND_SHIFT) & 0x3 {
            0 => PhysicalAccessKind::Read,
            1 => PhysicalAccessKind::Write,
            2 => PhysicalAccessKind::AtomicReadModifyWrite,
            _ => unreachable!("compact retained access has a valid kind"),
        }
    }

    #[inline(always)]
    fn proxy_and_domain(self, registry: &OperationRegistry) -> (MemoryProxy, ProxyMemoryDomain) {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            let witness = registry.wide_witness((value & Self::VALUE_MASK) as usize);
            return (witness.proxy, witness.proxy_domain);
        }
        Self::compact_proxy_and_domain(value)
    }

    #[inline(always)]
    fn same_proxy_and_domain(
        self,
        registry: &OperationRegistry,
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
    ) -> bool {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            let witness = registry.wide_witness((value & Self::VALUE_MASK) as usize);
            return witness.proxy == proxy && witness.proxy_domain == proxy_domain;
        }
        let class_mask =
            (1_u64 << Self::SPACE_SHIFT) | (0x3_u64 << Self::SHARED_PROXY_DOMAIN_SHIFT);
        Self::compact_proxy_and_domain_key(proxy, proxy_domain)
            .is_some_and(|key| value & class_mask == key)
    }

    #[inline(always)]
    fn same_proxy(self, registry: &OperationRegistry, proxy: MemoryProxy) -> bool {
        let value = self.0.get();
        if value & Self::WIDE_BIT != 0 {
            return registry
                .wide_witness((value & Self::VALUE_MASK) as usize)
                .proxy
                == proxy;
        }
        if (value >> Self::SPACE_SHIFT) & 0x1 != 0 {
            return proxy == MemoryProxy::Generic;
        }
        let retained_proxy = (value >> (Self::SHARED_PROXY_DOMAIN_SHIFT + 1)) & 0x1;
        retained_proxy == proxy as u64
    }

    #[inline(always)]
    fn same_operation(
        self,
        registry: &OperationRegistry,
        current: &PhysicalRaceWitnessRef,
    ) -> bool {
        if let Some(key) = self.compact_operation_key() {
            let operation = current.operation.0.get();
            let encoded_index = operation as u32 as u64;
            let global_warp_id = operation >> 32;
            let local_warp_id = (global_warp_id as usize).checked_sub(registry.global_warp_base);
            return local_warp_id.is_some_and(|local_warp_id| {
                encoded_index <= Self::OPERATION_INDEX_MASK
                    && (local_warp_id as u64) <= Self::LOCAL_WARP_MASK
                    && key == encoded_index | ((local_warp_id as u64) << Self::LOCAL_WARP_SHIFT)
            });
        }
        self.operation(registry) == current.operation
    }

    #[inline(always)]
    fn same_retained_operation(self, registry: &OperationRegistry, other: Self) -> bool {
        if let (Some(left), Some(right)) =
            (self.compact_operation_key(), other.compact_operation_key())
        {
            return left == right;
        }
        self.operation(registry) == other.operation(registry)
    }

    #[inline(always)]
    fn same_retained_operation_and_proxy(self, registry: &OperationRegistry, other: Self) -> bool {
        if !self.same_retained_operation(registry, other)
            || self.resolved_lane(registry) != other.resolved_lane(registry)
        {
            return false;
        }
        let left = self.0.get();
        let right = other.0.get();
        if (left | right) & Self::WIDE_BIT == 0 {
            let class_mask =
                (1_u64 << Self::SPACE_SHIFT) | (0x3_u64 << Self::SHARED_PROXY_DOMAIN_SHIFT);
            return left & class_mask == right & class_mask;
        }
        self.proxy_and_domain(registry) == other.proxy_and_domain(registry)
    }

    #[inline(always)]
    fn global_warp_id(self, registry: &OperationRegistry) -> usize {
        let value = self.0.get();
        if value & Self::WIDE_BIT == 0 {
            return registry.global_warp_base
                + ((value >> Self::LOCAL_WARP_SHIFT) & Self::LOCAL_WARP_MASK) as usize;
        }
        self.operation(registry).global_warp_id()
    }

    fn span(
        self,
        registry: &OperationRegistry,
        allocation: PhysicalAllocationId,
    ) -> PhysicalByteSpan {
        self.resolve(registry, allocation).span
    }

    fn materialize(
        self,
        registry: &OperationRegistry,
        allocation: PhysicalAllocationId,
    ) -> PhysicalRaceWitness {
        let witness = self.resolve(registry, allocation);
        PhysicalRaceWitness {
            operation: registry.operation(witness.operation),
            lane: witness.lane,
            kind: witness.kind,
            space: witness.space,
            span: witness.span,
        }
    }
}

fn witness_same_site(left: &PhysicalRaceWitness, right: &PhysicalRaceWitness) -> bool {
    left.lane == right.lane
        && left.kind == right.kind
        && left.space == right.space
        && left.span.allocation() == right.span.allocation()
        && (Arc::ptr_eq(&left.operation, &right.operation) || left.operation == right.operation)
}

/// The smallest span of one allocation covering both.
pub(crate) fn span_hull(left: PhysicalByteSpan, right: PhysicalByteSpan) -> PhysicalByteSpan {
    debug_assert_eq!(left.allocation(), right.allocation());
    let byte_offset = left.byte_offset().min(right.byte_offset());
    let byte_end = left.byte_end().max(right.byte_end());
    PhysicalByteSpan::new(left.allocation(), byte_offset, byte_end - byte_offset)
        .expect("the hull of two spans of one allocation is nonempty and representable")
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalRaceFinding {
    kind: PhysicalRaceKind,
    ordering_failure: PhysicalRaceOrderingFailure,
    prior: PhysicalRaceWitness,
    current: PhysicalRaceWitness,
    overlap: PhysicalByteSpan,
    reviewed_tmem_load_handle: Option<RegisteredOperation>,
}

impl PhysicalRaceFinding {
    pub(crate) fn new(
        kind: PhysicalRaceKind,
        ordering_failure: PhysicalRaceOrderingFailure,
        prior: PhysicalRaceWitness,
        current: PhysicalRaceWitness,
        overlap: PhysicalByteSpan,
    ) -> Self {
        Self {
            kind,
            ordering_failure,
            prior,
            current,
            overlap,
            reviewed_tmem_load_handle: None,
        }
    }

    pub const fn kind(&self) -> PhysicalRaceKind {
        self.kind
    }

    /// Whether `other` reports the same pair of access sites: same kind,
    /// same operations, lanes, access kinds and spaces on both sides, in
    /// one allocation — only the byte ranges may differ.
    pub(crate) fn same_site(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.ordering_failure == other.ordering_failure
            && self.reviewed_tmem_load_handle.is_some() == other.reviewed_tmem_load_handle.is_some()
            && witness_same_site(&self.prior, &other.prior)
            && witness_same_site(&self.current, &other.current)
    }

    /// This finding widened to cover `other`, a finding of the same site:
    /// each span becomes the hull of the two.
    pub(crate) fn hull(&self, other: &Self) -> Self {
        debug_assert!(self.same_site(other));
        Self {
            kind: self.kind,
            ordering_failure: self.ordering_failure,
            prior: self
                .prior
                .with_span(span_hull(self.prior.span(), other.prior.span())),
            current: self
                .current
                .with_span(span_hull(self.current.span(), other.current.span())),
            overlap: span_hull(self.overlap, other.overlap),
            reviewed_tmem_load_handle: self.reviewed_tmem_load_handle,
        }
    }

    pub const fn ordering_failure(&self) -> PhysicalRaceOrderingFailure {
        self.ordering_failure
    }

    pub const fn prior(&self) -> &PhysicalRaceWitness {
        &self.prior
    }

    pub const fn current(&self) -> &PhysicalRaceWitness {
        &self.current
    }

    pub const fn overlap(&self) -> PhysicalByteSpan {
        self.overlap
    }

    /// Whether this conflict may be ordered by a true register dependency from
    /// an earlier `tcgen05.ld` that native Racecheck deliberately does not
    /// model.
    ///
    /// TMEM conflicts without that exact load provenance remain hard errors;
    /// issue order alone does not order asynchronous TCGEN operations.
    pub(crate) fn requires_unwaited_tmem_load_review(&self) -> bool {
        self.reviewed_tmem_load_handle.is_some()
    }

    pub(crate) fn reviewed_tmem_load_operation(&self) -> Option<&DynamicOpId> {
        self.reviewed_tmem_load_handle
            .map(|_| self.prior.operation())
    }

    fn reviewed_tmem_load_handle(&self) -> Option<RegisteredOperation> {
        self.reviewed_tmem_load_handle
    }
}

impl fmt::Display for PhysicalRaceFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} on {} between {} and {}; {} ordering failure: {}",
            self.kind,
            self.overlap,
            self.prior,
            self.current,
            self.ordering_failure.domain(),
            self.ordering_failure
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RaceShadowError {
    InvalidWarp {
        warp_id: usize,
        warp_count: usize,
    },
    ClockOverflow {
        warp_id: usize,
    },
    AsyncClockOverflow {
        token: AsyncTokenId,
    },
    DuplicateAsyncActor {
        token: AsyncTokenId,
    },
    MissingAsyncActor {
        token: AsyncTokenId,
    },
    ClockDimensionMismatch {
        expected_warps: usize,
        actual_warps: usize,
    },
    AsyncClockRegistryMismatch,
    RetiredCrossProxyHistory {
        space: PhysicalAccessSpace,
        allocation: PhysicalAllocationId,
    },
    Race(PhysicalRaceFinding),
}

impl RaceShadowError {
    pub const fn finding(&self) -> Option<&PhysicalRaceFinding> {
        match self {
            Self::Race(finding) => Some(finding),
            _ => None,
        }
    }
}

impl fmt::Display for RaceShadowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWarp {
                warp_id,
                warp_count,
            } => write!(
                f,
                "warp {warp_id} is outside launch warp count {warp_count}"
            ),
            Self::ClockOverflow { warp_id } => {
                write!(f, "vector-clock component for warp {warp_id} overflowed")
            }
            Self::AsyncClockOverflow { token } => {
                write!(
                    f,
                    "vector-clock component for async token {token:?} overflowed"
                )
            }
            Self::DuplicateAsyncActor { token } => {
                write!(f, "async actor for token {token:?} is already active")
            }
            Self::MissingAsyncActor { token } => {
                write!(f, "async actor for token {token:?} is not active")
            }
            Self::ClockDimensionMismatch {
                expected_warps,
                actual_warps,
            } => write!(
                f,
                "barrier clock has {actual_warps} warps, expected {expected_warps}"
            ),
            Self::AsyncClockRegistryMismatch => {
                f.write_str("barrier clocks belong to different racecheck shards")
            }
            Self::RetiredCrossProxyHistory { space, allocation } => write!(
                f,
                "cross-proxy history for {space} allocation {allocation} was retired before its \
                 first async-proxy access; no matching proxy bridge proves that history ordered"
            ),
            Self::Race(finding) => finding.fmt(f),
        }
    }
}

impl Error for RaceShadowError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AllocationKey {
    space: PhysicalAccessSpace,
    allocation: PhysicalAllocationId,
}

#[derive(Clone, Debug, Default)]
struct RetiredProxyFrontiers {
    reads: [Option<ProxyClockFrontier>; PROXY_MEMORY_DOMAIN_COUNT],
    writes: [Option<ProxyClockFrontier>; PROXY_MEMORY_DOMAIN_COUNT],
}

impl RetiredProxyFrontiers {
    fn is_empty(&self) -> bool {
        self.reads.iter().all(Option::is_none) && self.writes.iter().all(Option::is_none)
    }

    /// Extend the frontiers of the slots `kind` touches in `domain` so that
    /// they observe `timestamp`.
    fn raise_timestamp(
        &mut self,
        kind: PhysicalAccessKind,
        domain: ProxyMemoryDomain,
        timestamp: RaceEventTimestamp,
        registry: &OperationRegistry,
        warp_count: usize,
        async_registry: &Arc<AsyncClockRegistry>,
        source_lane: (usize, usize),
    ) -> Result<(), RaceShadowError> {
        let Some(index) = proxy_domain_index(domain) else {
            return Ok(());
        };
        if kind.reads() {
            self.reads[index]
                .get_or_insert_with(|| {
                    ProxyClockFrontier::empty(warp_count, Arc::clone(async_registry))
                })
                .raise_timestamp(timestamp, registry, Some(source_lane))?;
        }
        if kind.writes() {
            self.writes[index]
                .get_or_insert_with(|| {
                    ProxyClockFrontier::empty(warp_count, Arc::clone(async_registry))
                })
                .raise_timestamp(timestamp, registry, Some(source_lane))?;
        }
        Ok(())
    }

    fn merge_from(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        for (slot, incoming) in self
            .reads
            .iter_mut()
            .zip(&other.reads)
            .chain(self.writes.iter_mut().zip(&other.writes))
        {
            if let Some(incoming) = incoming {
                merge_optional_proxy_frontier(slot, incoming)?;
            }
        }
        Ok(())
    }

    fn has_unordered_conflict(
        &self,
        current_kind: PhysicalAccessKind,
        current_domain: ProxyMemoryDomain,
        current_clock: &RaceVectorClock,
        current_lane: usize,
    ) -> bool {
        for prior_domain in proxy_memory_domains() {
            let index = proxy_domain_index(prior_domain)
                .expect("the supported proxy-domain list contains indexed domains");
            let conflicting = self.writes[index].iter().chain(
                current_kind
                    .writes()
                    .then_some(&self.reads[index])
                    .into_iter()
                    .flatten(),
            );
            for frontier in conflicting {
                if !current_clock.proxy_bridge_observes_frontier(
                    MemoryProxy::Generic,
                    MemoryProxy::Async,
                    prior_domain,
                    current_domain,
                    frontier,
                    current_lane,
                ) {
                    return true;
                }
            }
        }
        false
    }
}

fn merge_optional_proxy_frontier(
    slot: &mut Option<ProxyClockFrontier>,
    frontier: &ProxyClockFrontier,
) -> Result<(), RaceShadowError> {
    if let Some(current) = slot {
        current.merge(frontier)
    } else {
        *slot = Some(frontier.clone());
        Ok(())
    }
}

/// One retired byte range of an allocation and the join of the timestamps
/// of the generic witnesses retired from it.
#[derive(Clone, Debug)]
struct RetiredRange {
    start: usize,
    end: usize,
    frontier: ProxyClockFrontier,
}

/// Retired ranges of one (kind, domain) slot: sorted by start and pairwise
/// disjoint. Overlapping insertions fold into one range with the joined
/// frontier; past `MAX_RETIRED_RANGES_PER_SLOT`, the closest neighbours
/// coalesce, which is only ever conservative.
#[derive(Clone, Debug, Default)]
struct RetiredRangeFrontiers {
    ranges: Vec<RetiredRange>,
}

impl RetiredRangeFrontiers {
    fn insert(
        &mut self,
        start: usize,
        end: usize,
        frontier: &ProxyClockFrontier,
    ) -> Result<(), RaceShadowError> {
        let first = self.ranges.partition_point(|range| range.end <= start);
        let mut last = first;
        while last < self.ranges.len() && self.ranges[last].start < end {
            last += 1;
        }
        if first == last {
            self.ranges.insert(
                first,
                RetiredRange {
                    start,
                    end,
                    frontier: frontier.clone(),
                },
            );
        } else {
            let mut merged = RetiredRange {
                start: start.min(self.ranges[first].start),
                end: end.max(self.ranges[last - 1].end),
                frontier: frontier.clone(),
            };
            for range in self.ranges.drain(first..last) {
                merged.frontier.merge(&range.frontier)?;
            }
            self.ranges.insert(first, merged);
        }
        while self.ranges.len() > MAX_RETIRED_RANGES_PER_SLOT {
            let index = (0..self.ranges.len() - 1)
                .min_by_key(|&index| self.ranges[index + 1].start - self.ranges[index].end)
                .expect("a list past the cap has at least two ranges");
            let next = self.ranges.remove(index + 1);
            let current = &mut self.ranges[index];
            current.end = next.end;
            current.frontier.merge(&next.frontier)?;
        }
        Ok(())
    }

    fn merge_from(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        for range in &other.ranges {
            self.insert(range.start, range.end, &range.frontier)?;
        }
        Ok(())
    }

    fn collapse_into(&self, slot: &mut Option<ProxyClockFrontier>) -> Result<(), RaceShadowError> {
        for range in &self.ranges {
            merge_optional_proxy_frontier(slot, &range.frontier)?;
        }
        Ok(())
    }

    fn overlapping(&self, start: usize, end: usize) -> impl Iterator<Item = &ProxyClockFrontier> {
        let first = self.ranges.partition_point(|range| range.end <= start);
        self.ranges[first..]
            .iter()
            .take_while(move |range| range.start < end)
            .map(|range| &range.frontier)
    }
}

/// What one allocation's retired generic witnesses assert, per access kind,
/// proxy domain, and byte range. The live cross-proxy check compares
/// overlapping bytes only, so the retired history must keep ranges too:
/// otherwise an initializing store far from a TMA's target bytes would
/// demand a proxy fence that the hardware never required.
#[derive(Clone, Debug, Default)]
struct RetiredAllocationHistory {
    reads: [RetiredRangeFrontiers; PROXY_MEMORY_DOMAIN_COUNT],
    writes: [RetiredRangeFrontiers; PROXY_MEMORY_DOMAIN_COUNT],
}

impl RetiredAllocationHistory {
    fn is_empty(&self) -> bool {
        self.reads
            .iter()
            .chain(&self.writes)
            .all(|slot| slot.ranges.is_empty())
    }

    /// Record the frontiers retired from one segment `[start, end)`.
    fn insert_segment(
        &mut self,
        start: usize,
        end: usize,
        frontiers: &RetiredProxyFrontiers,
    ) -> Result<(), RaceShadowError> {
        for index in 0..PROXY_MEMORY_DOMAIN_COUNT {
            if let Some(frontier) = &frontiers.reads[index] {
                self.reads[index].insert(start, end, frontier)?;
            }
            if let Some(frontier) = &frontiers.writes[index] {
                self.writes[index].insert(start, end, frontier)?;
            }
        }
        Ok(())
    }

    fn merge_from(&mut self, other: &Self) -> Result<(), RaceShadowError> {
        for (slot, incoming) in self
            .reads
            .iter_mut()
            .zip(&other.reads)
            .chain(self.writes.iter_mut().zip(&other.writes))
        {
            slot.merge_from(incoming)?;
        }
        Ok(())
    }

    /// Fold every range into range-less frontiers (the overflow form).
    fn collapse_into(&self, frontiers: &mut RetiredProxyFrontiers) -> Result<(), RaceShadowError> {
        for index in 0..PROXY_MEMORY_DOMAIN_COUNT {
            self.reads[index].collapse_into(&mut frontiers.reads[index])?;
            self.writes[index].collapse_into(&mut frontiers.writes[index])?;
        }
        Ok(())
    }

    fn has_unordered_conflict(
        &self,
        current_kind: PhysicalAccessKind,
        current_domain: ProxyMemoryDomain,
        start: usize,
        end: usize,
        current_clock: &RaceVectorClock,
        current_lane: usize,
    ) -> bool {
        for prior_domain in proxy_memory_domains() {
            let index = proxy_domain_index(prior_domain)
                .expect("the supported proxy-domain list contains indexed domains");
            let conflicting = self.writes[index].overlapping(start, end).chain(
                current_kind
                    .writes()
                    .then(|| self.reads[index].overlapping(start, end))
                    .into_iter()
                    .flatten(),
            );
            for frontier in conflicting {
                if !current_clock.proxy_bridge_observes_frontier(
                    MemoryProxy::Generic,
                    MemoryProxy::Async,
                    prior_domain,
                    current_domain,
                    frontier,
                    current_lane,
                ) {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Clone, Debug, Default)]
struct RetiredGenericHistory {
    exact: HashMap<AllocationKey, RetiredAllocationHistory>,
    overflow: RetiredProxyFrontiers,
    overflowed: bool,
}

impl RetiredGenericHistory {
    /// Record what was just retired from `key`; once the exact table is
    /// full, everything further folds range-less into `overflow` (committed
    /// by [`Self::commit_overflow`]).
    fn record(
        &mut self,
        key: AllocationKey,
        history: &RetiredAllocationHistory,
        overflow: &mut RetiredProxyFrontiers,
    ) -> Result<(), RaceShadowError> {
        if history.is_empty() {
            return Ok(());
        }
        if !self.overflowed
            && (self.exact.contains_key(&key)
                || self.exact.len() < MAX_EXACT_RETIRED_PROXY_ALLOCATIONS)
        {
            self.exact.entry(key).or_default().merge_from(history)
        } else {
            self.overflowed = true;
            history.collapse_into(overflow)
        }
    }

    fn commit_overflow(
        &mut self,
        frontiers: &RetiredProxyFrontiers,
    ) -> Result<(), RaceShadowError> {
        self.overflow.merge_from(frontiers)
    }

    /// Whether any retired history could constrain accesses to `key`.
    fn applies_to(&self, key: AllocationKey) -> bool {
        self.overflowed || self.exact.contains_key(&key)
    }

    fn has_unordered_conflict(
        &self,
        key: AllocationKey,
        current_kind: PhysicalAccessKind,
        current_domain: ProxyMemoryDomain,
        start: usize,
        end: usize,
        current_clock: &RaceVectorClock,
        current_lane: usize,
    ) -> bool {
        self.exact.get(&key).is_some_and(|history| {
            history.has_unordered_conflict(
                current_kind,
                current_domain,
                start,
                end,
                current_clock,
                current_lane,
            )
        }) || self.overflow.has_unordered_conflict(
            current_kind,
            current_domain,
            current_clock,
            current_lane,
        )
    }

    fn remove_exact(&mut self, key: &AllocationKey) {
        self.exact.remove(key);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimestampActor {
    Warp(usize),
    Async(usize),
    RegisteredVector(usize),
}

/// Compact exact event timestamp.
///
/// Ordinary kernels fit a 31-bit actor index and epoch directly in one word.
/// Exceptional launch sizes or long-running epochs retain the full values in
/// the operation registry and use the high bit as a lossless wide handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RaceEventTimestamp(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideRetainedTimestamp {
    actor: TimestampActor,
    epoch: u64,
    lane_stamp: Option<RetainedRaceLaneStamp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetainedRaceLaneStamp {
    epoch: NonZeroU64,
}

#[derive(Clone, Copy)]
struct DirectAccessActor {
    global_warp_id: usize,
    local_warp_id: usize,
}

#[derive(Clone, Copy)]
enum DirectTimestampEncoder {
    Compact {
        base: u64,
        timestamp: RaceEventTimestamp,
        kind: PhysicalAccessKind,
    },
    Wide {
        timestamp: RaceEventTimestamp,
        kind: PhysicalAccessKind,
    },
}

impl RetainedRaceLaneStamp {
    fn new(epoch: u64) -> Self {
        Self {
            epoch: NonZeroU64::new(epoch)
                .expect("a retained lane event always has a positive epoch"),
        }
    }
}

/// A lane's clock after acquiring the order's incoming payloads: the clock
/// itself when there is nothing to acquire, otherwise a shared joined copy.
enum AcquiredClock<'b> {
    Borrowed(&'b RaceVectorClock),
    Shared(Arc<RaceVectorClock>),
}

impl std::ops::Deref for AcquiredClock<'_> {
    type Target = RaceVectorClock;

    fn deref(&self) -> &RaceVectorClock {
        match self {
            Self::Borrowed(clock) => clock,
            Self::Shared(clock) => clock,
        }
    }
}

struct AcquiredClockEntry {
    current: usize,
    lane: usize,
    clock: Arc<RaceVectorClock>,
}

/// Current per-lane order for direct warp accesses; async actors use their
/// independent vector-clock timestamp.
pub(crate) struct RaceLaneOrder<'a> {
    global_warp_id: usize,
    common: &'a [u64; WARP_SIZE],
    observed: &'a [[u64; WARP_SIZE]; WARP_SIZE],
    global_warp_base: usize,
    incoming_common: Option<&'a SharedClockFrontier>,
    incoming_lanes: Option<&'a [SharedClockFrontier; WARP_SIZE]>,
    // Repeated byte/lane checks share this immutable publication. Cache only
    // a borrowed lookup, never another copy of its epochs or synchronization.
    common_release: Cell<Option<(usize, Option<&'a LaneFrontier>)>>,
    // One batch checks a lane's access against many priors with the same
    // clock; keep the last joined clock so each prior does not rebuild it.
    acquired_clock: Cell<Option<AcquiredClockEntry>>,
}

impl<'a> RaceLaneOrder<'a> {
    pub(crate) const fn new(
        global_warp_id: usize,
        common: &'a [u64; WARP_SIZE],
        observed: &'a [[u64; WARP_SIZE]; WARP_SIZE],
    ) -> Self {
        Self {
            global_warp_id,
            common,
            observed,
            global_warp_base: 0,
            incoming_common: None,
            incoming_lanes: None,
            common_release: Cell::new(None),
            acquired_clock: Cell::new(None),
        }
    }

    pub(crate) fn with_incoming(
        mut self,
        global_warp_base: usize,
        common: &'a SharedClockFrontier,
        lanes: Option<&'a [SharedClockFrontier; WARP_SIZE]>,
    ) -> Self {
        self.global_warp_base = global_warp_base;
        self.incoming_common = Some(common);
        self.incoming_lanes = lanes;
        self.common_release.set(None);
        self.acquired_clock.set(None);
        self
    }

    fn incoming_payloads(
        &self,
        lane: usize,
    ) -> impl Iterator<Item = &'a BarrierClockPayload> + '_ {
        self.incoming_common
            .into_iter()
            .chain(self.incoming_lanes.map(|lanes| &lanes[lane]))
            .filter_map(|frontier| frontier.clock_for(self.global_warp_base))
    }

    fn acquired_clock<'b>(&self, current: &'b RaceVectorClock, lane: usize) -> AcquiredClock<'b> {
        if self.incoming_payloads(lane).next().is_none() {
            return AcquiredClock::Borrowed(current);
        }
        let key = (current as *const RaceVectorClock as usize, lane);
        if let Some(entry) = self.acquired_clock.take() {
            if (entry.current, entry.lane) == key {
                let clock = Arc::clone(&entry.clock);
                self.acquired_clock.set(Some(entry));
                return AcquiredClock::Shared(clock);
            }
        }
        let mut clock = current.clone();
        for payload in self.incoming_payloads(lane) {
            clock
                .acquire_payload(payload, WarpMask::from_bits(1 << lane))
                .expect("memory acquisition keeps each shard's clock identity");
        }
        let clock = Arc::new(clock);
        self.acquired_clock.set(Some(AcquiredClockEntry {
            current: key.0,
            lane,
            clock: Arc::clone(&clock),
        }));
        AcquiredClock::Shared(clock)
    }

    /// `timestamp.observed_by(&self.acquired_clock(current, lane))` without
    /// building the joined clock when the timestamp names one actor: the
    /// join's component is the maximum over `current` and the payloads.
    fn timestamp_observed_by_acquired(
        &self,
        timestamp: &RaceEventTimestamp,
        current: &RaceVectorClock,
        lane: usize,
        registry: &OperationRegistry,
    ) -> bool {
        if timestamp.observed_by(current, registry) {
            return true;
        }
        let mut payloads = self.incoming_payloads(lane).peekable();
        if payloads.peek().is_none() {
            return false;
        }
        if timestamp.has_single_actor(registry) {
            payloads.any(|payload| timestamp.observed_by(payload.clock(), registry))
        } else {
            timestamp.observed_by(&self.acquired_clock(current, lane), registry)
        }
    }

    fn publication_lanes(&self, lane: usize) -> SharedLaneEpochs {
        let own = std::array::from_fn(|source| self.component(lane, source));
        let mut epochs = Some(Arc::new(SharedLaneEpochMap {
            rows: vec![(self.global_warp_id, own)],
        }));
        for frontier in self
            .incoming_common
            .into_iter()
            .chain(self.incoming_lanes.map(|lanes| &lanes[lane]))
        {
            merge_shared_lane_epochs(&mut epochs, &frontier.releases);
        }
        epochs
    }

    /// An async actor may inherit its issuing lanes, never their sibling lanes.
    /// The joined clock belongs to the actor; it is not a warp-clock update.
    fn acquired_mask_clock(&self, current: &RaceVectorClock, mask: WarpMask) -> RaceVectorClock {
        // Joining one acquired clock per lane repeats the common payload's
        // causal merge and its proxy-bridge merge once per lane; acquiring it
        // under the whole mask gives the same join in one pass.
        let mut inherited = current.clone();
        if let Some(payload) = self
            .incoming_common
            .and_then(|frontier| frontier.clock_for(self.global_warp_base))
        {
            inherited
                .acquire_payload(payload, mask)
                .expect("memory acquisition keeps each shard's clock identity");
        }
        if let Some(lanes) = self.incoming_lanes {
            for lane in mask {
                if let Some(payload) = lanes[lane].clock_for(self.global_warp_base) {
                    inherited
                        .acquire_payload(payload, WarpMask::from_bits(1 << lane))
                        .expect("memory acquisition keeps each shard's clock identity");
                }
            }
        }
        inherited
    }

    fn component(&self, lane: usize, source_lane: usize) -> u64 {
        self.common[source_lane].max(self.observed[lane][source_lane])
    }

    fn retained_stamp(&self, lane: usize) -> RetainedRaceLaneStamp {
        RetainedRaceLaneStamp::new(
            self.component(lane, lane)
                .checked_add(1)
                .expect("lane-order validation rejects epoch overflow first"),
        )
    }

    fn observes(
        &self,
        current_lane: usize,
        prior_global_warp_id: usize,
        prior_lane: usize,
        prior: RetainedRaceLaneStamp,
    ) -> bool {
        if self.global_warp_id != prior_global_warp_id {
            let common = match self.common_release.get() {
                Some((warp, frontier)) if warp == prior_global_warp_id => frontier,
                _ => {
                    let frontier = self.incoming_common
                        .and_then(|frontier| frontier.release_for(prior_global_warp_id));
                    self.common_release.set(Some((prior_global_warp_id, frontier)));
                    frontier
                }
            };
            return common.is_some_and(|frontier| frontier[prior_lane] >= prior.epoch.get())
                || self.incoming_lanes
                    .and_then(|lanes| lanes[current_lane].release_for(prior_global_warp_id))
                    .is_some_and(|frontier| frontier[prior_lane] >= prior.epoch.get());
        }
        let observed_epoch = if current_lane == prior_lane {
            self.component(current_lane, current_lane)
                .checked_add(1)
                .expect("lane-order validation rejects epoch overflow first")
        } else {
            self.component(current_lane, prior_lane)
        };
        observed_epoch >= prior.epoch.get()
    }
}

impl RaceEventTimestamp {
    const WIDE_BIT: u64 = 1_u64 << 63;
    const LANE_STAMP_BIT: u64 = 1_u64 << 62;
    const REGISTERED_ACTOR_BIT: u32 = 1_u32 << 31;
    const ASYNC_ACTOR_BIT: u32 = 1_u32 << 30;
    const REGISTERED_INDEX_LIMIT: u32 = Self::ASYNC_ACTOR_BIT;
    const COMPACT_VALUE_MASK: u64 = Self::WIDE_BIT - 1;
    const LANE_METADATA_BITS: u32 = 14;
    const LANE_WARP_BITS: u32 = 4;
    const LANE_WARP_MASK: u64 = (1_u64 << Self::LANE_WARP_BITS) - 1;
    const LANE_LANE_SHIFT: u32 = Self::LANE_WARP_BITS;
    const LANE_KIND_SHIFT: u32 = Self::LANE_LANE_SHIFT + 5;
    const LANE_EPOCH_BITS: u32 = 24;
    const LANE_METADATA_MASK: u64 = (1_u64 << Self::LANE_METADATA_BITS) - 1;
    const LANE_EPOCH_MASK: u64 = (1_u64 << Self::LANE_EPOCH_BITS) - 1;
    const LANE_EVENT_EPOCH_SHIFT: u32 = Self::LANE_METADATA_BITS;
    const LANE_STAMP_SHIFT: u32 = Self::LANE_METADATA_BITS + Self::LANE_EPOCH_BITS;

    fn new(actor: TimestampActor, epoch: u64, registry: &OperationRegistry) -> Self {
        let compact_actor = match actor {
            TimestampActor::Warp(index) => u32::try_from(index)
                .ok()
                .filter(|index| *index < Self::REGISTERED_ACTOR_BIT),
            TimestampActor::Async(index) => u32::try_from(index)
                .ok()
                .filter(|index| *index < Self::REGISTERED_INDEX_LIMIT)
                .map(|index| Self::REGISTERED_ACTOR_BIT | Self::ASYNC_ACTOR_BIT | index),
            TimestampActor::RegisteredVector(index) => u32::try_from(index)
                .ok()
                .filter(|index| *index < Self::REGISTERED_INDEX_LIMIT)
                .map(|index| Self::REGISTERED_ACTOR_BIT | index),
        };
        if let (Some(actor), Ok(epoch)) = (compact_actor, u32::try_from(epoch)) {
            // Bit 62 distinguishes the lane-stamped direct representation.
            // Keeping it clear here leaves the two encodings unambiguous.
            if epoch < Self::ASYNC_ACTOR_BIT {
                return Self((u64::from(epoch) << 32) | u64::from(actor));
            }
        }
        Self::new_wide(
            WideRetainedTimestamp {
                actor,
                epoch,
                lane_stamp: None,
            },
            registry,
        )
    }

    fn new_wide(timestamp: WideRetainedTimestamp, registry: &OperationRegistry) -> Self {
        let index = registry.register_wide_timestamp(timestamp);
        Self(
            Self::WIDE_BIT
                | u64::try_from(index).expect("a wide timestamp index fits in the retained handle"),
        )
    }

    #[inline(always)]
    fn with_lane_stamp(
        self,
        lane_stamp: Option<RetainedRaceLaneStamp>,
        lane: usize,
        kind: PhysicalAccessKind,
        registry: &OperationRegistry,
    ) -> Self {
        let Some(lane_stamp) = lane_stamp else {
            return self;
        };
        if self.0 & (Self::WIDE_BIT | Self::LANE_STAMP_BIT) == 0 {
            let actor = self.0 as u32;
            let epoch = self.0 >> 32;
            let lane_epoch = lane_stamp.epoch.get();
            let compact_kind = match kind {
                PhysicalAccessKind::Read => 0_u64,
                PhysicalAccessKind::Write => 1_u64,
                PhysicalAccessKind::AtomicReadModifyWrite => 2_u64,
            };
            if actor & Self::REGISTERED_ACTOR_BIT == 0
                && u64::from(actor) <= Self::LANE_WARP_MASK
                && lane < WARP_SIZE
                && epoch <= Self::LANE_EPOCH_MASK
                && lane_epoch <= Self::LANE_EPOCH_MASK
            {
                let metadata = u64::from(actor)
                    | ((lane as u64) << Self::LANE_LANE_SHIFT)
                    | (compact_kind << Self::LANE_KIND_SHIFT);
                debug_assert!(metadata <= Self::LANE_METADATA_MASK);
                return Self(
                    Self::LANE_STAMP_BIT
                        | (lane_epoch << Self::LANE_STAMP_SHIFT)
                        | (epoch << Self::LANE_EVENT_EPOCH_SHIFT)
                        | metadata,
                );
            }
        }
        let resolved = self.resolve_full(registry);
        Self::new_wide(
            WideRetainedTimestamp {
                actor: resolved.actor,
                epoch: resolved.epoch,
                lane_stamp: Some(lane_stamp),
            },
            registry,
        )
    }

    #[inline(always)]
    fn resolve_full(self, registry: &OperationRegistry) -> WideRetainedTimestamp {
        if self.0 & Self::WIDE_BIT != 0 {
            return registry.wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize);
        }
        if self.0 & Self::LANE_STAMP_BIT != 0 {
            return WideRetainedTimestamp {
                actor: TimestampActor::Warp((self.0 & Self::LANE_WARP_MASK) as usize),
                epoch: (self.0 >> Self::LANE_EVENT_EPOCH_SHIFT) & Self::LANE_EPOCH_MASK,
                lane_stamp: Some(RetainedRaceLaneStamp::new(
                    (self.0 >> Self::LANE_STAMP_SHIFT) & Self::LANE_EPOCH_MASK,
                )),
            };
        }
        let actor = self.0 as u32;
        let actor = if actor & Self::REGISTERED_ACTOR_BIT == 0 {
            TimestampActor::Warp(actor as usize)
        } else if actor & Self::ASYNC_ACTOR_BIT != 0 {
            TimestampActor::Async(
                (actor & !(Self::REGISTERED_ACTOR_BIT | Self::ASYNC_ACTOR_BIT)) as usize,
            )
        } else {
            TimestampActor::RegisteredVector((actor & !Self::REGISTERED_ACTOR_BIT) as usize)
        };
        WideRetainedTimestamp {
            actor,
            epoch: self.0 >> 32,
            lane_stamp: None,
        }
    }

    #[inline(always)]
    fn resolve(self, registry: &OperationRegistry) -> (TimestampActor, u64) {
        let resolved = self.resolve_full(registry);
        (resolved.actor, resolved.epoch)
    }

    fn is_async_actor(self, registry: &OperationRegistry) -> bool {
        matches!(self.resolve_full(registry).actor, TimestampActor::Async(_))
    }

    fn async_issue_event(
        self,
        clock: &RaceVectorClock,
        registry: &OperationRegistry,
    ) -> Option<AsyncIssueEvent> {
        let TimestampActor::Async(index) = self.resolve_full(registry).actor else {
            return None;
        };
        clock.async_registry.issue_event(index)
    }

    fn ordinary_observed_by(
        self,
        current_timestamp: Self,
        current_clock: &RaceVectorClock,
        registry: &OperationRegistry,
    ) -> bool {
        let current_issue = current_timestamp.async_issue_event(current_clock, registry);
        let observed_component = |warp_id: usize| {
            current_clock.component(warp_id).map(|epoch| {
                current_issue
                    .filter(|issue| issue.issuer_warp == warp_id)
                    .map_or(epoch, |issue| epoch.max(issue.epoch))
            })
        };
        match self.resolve_full(registry).actor {
            TimestampActor::Warp(warp_id) => observed_component(warp_id)
                .is_some_and(|epoch| epoch >= self.resolve_full(registry).epoch),
            TimestampActor::Async(index) => current_clock
                .async_registry
                .issue_event(index)
                .is_some_and(|issue| {
                    observed_component(issue.issuer_warp).is_some_and(|epoch| epoch >= issue.epoch)
                }),
            TimestampActor::RegisteredVector(index) => registry
                .registered_vector_timestamp_ordinary_observed_by(
                    index,
                    current_clock,
                    current_issue,
                ),
        }
    }

    #[inline(always)]
    fn lane_stamp(self, registry: &OperationRegistry) -> Option<RetainedRaceLaneStamp> {
        if self.0 & Self::WIDE_BIT != 0 {
            return registry
                .wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize)
                .lane_stamp;
        }
        if self.0 & Self::LANE_STAMP_BIT == 0 {
            return None;
        }
        Some(RetainedRaceLaneStamp::new(
            (self.0 >> Self::LANE_STAMP_SHIFT) & Self::LANE_EPOCH_MASK,
        ))
    }

    #[inline(always)]
    fn compact_direct_warp_components(
        self,
    ) -> Option<(u32, u32, RetainedRaceLaneStamp, usize, PhysicalAccessKind)> {
        if self.0 & (Self::WIDE_BIT | Self::LANE_STAMP_BIT) != Self::LANE_STAMP_BIT {
            return None;
        }
        let kind = match (self.0 >> Self::LANE_KIND_SHIFT) & 0x3 {
            0 => PhysicalAccessKind::Read,
            1 => PhysicalAccessKind::Write,
            2 => PhysicalAccessKind::AtomicReadModifyWrite,
            _ => unreachable!("compact direct timestamp has a valid access kind"),
        };
        Some((
            (self.0 & Self::LANE_WARP_MASK) as u32,
            ((self.0 >> Self::LANE_EVENT_EPOCH_SHIFT) & Self::LANE_EPOCH_MASK) as u32,
            RetainedRaceLaneStamp::new((self.0 >> Self::LANE_STAMP_SHIFT) & Self::LANE_EPOCH_MASK),
            ((self.0 >> Self::LANE_LANE_SHIFT) & 0x1f) as usize,
            kind,
        ))
    }

    #[cold]
    #[inline(never)]
    fn wide_direct_warp_components(
        self,
        registry: &OperationRegistry,
    ) -> Option<(usize, u64, RetainedRaceLaneStamp)> {
        if self.0 & Self::WIDE_BIT == 0 {
            return None;
        }
        let timestamp = registry.wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize);
        match (timestamp.actor, timestamp.lane_stamp) {
            (TimestampActor::Warp(warp_id), Some(lane_stamp)) => {
                Some((warp_id, timestamp.epoch, lane_stamp))
            }
            _ => None,
        }
    }

    #[inline(always)]
    fn same_event(self, other: Self, registry: &OperationRegistry) -> bool {
        if self.0 & (Self::WIDE_BIT | Self::LANE_STAMP_BIT) == 0
            && other.0 & (Self::WIDE_BIT | Self::LANE_STAMP_BIT) == 0
        {
            return self == other;
        }
        self.resolve(registry) == other.resolve(registry)
    }

    fn for_warp(
        clock: &RaceVectorClock,
        local_warp_id: usize,
        registry: &OperationRegistry,
    ) -> Self {
        Self::new(
            TimestampActor::Warp(local_warp_id),
            clock
                .component(local_warp_id)
                .expect("a validated local warp has a vector-clock component"),
            registry,
        )
    }

    fn for_async(
        clock: &RaceVectorClock,
        token: &AsyncTokenId,
        registry: &OperationRegistry,
    ) -> Self {
        let actor_index = clock
            .async_actor_index(token)
            .expect("an async event clock has registered its token");
        Self::new(
            TimestampActor::Async(actor_index),
            clock.async_component_at(actor_index),
            registry,
        )
    }

    fn for_vector(clock: Arc<RaceVectorClock>, registry: &OperationRegistry) -> Self {
        let index = registry.register_vector_timestamp_actor(clock);
        Self::new(TimestampActor::RegisteredVector(index), 0, registry)
    }

    #[inline(always)]
    /// Whether this timestamp is one actor's epoch, so a join of clocks
    /// observes it exactly when one of the joined clocks does.
    fn has_single_actor(&self, registry: &OperationRegistry) -> bool {
        if self.0 & Self::WIDE_BIT == 0 {
            if self.0 & Self::LANE_STAMP_BIT != 0 {
                return true;
            }
            let actor = self.0 as u32;
            return actor & Self::REGISTERED_ACTOR_BIT == 0 || actor & Self::ASYNC_ACTOR_BIT != 0;
        }
        let timestamp = registry.wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize);
        !matches!(timestamp.actor, TimestampActor::RegisteredVector(_))
    }

    fn observed_by(&self, clock: &RaceVectorClock, registry: &OperationRegistry) -> bool {
        if self.0 & Self::WIDE_BIT == 0 {
            if self.0 & Self::LANE_STAMP_BIT != 0 {
                let actor = (self.0 & Self::LANE_WARP_MASK) as usize;
                let epoch = (self.0 >> Self::LANE_EVENT_EPOCH_SHIFT) & Self::LANE_EPOCH_MASK;
                return clock.component(actor).is_some_and(|seen| seen >= epoch);
            }
            let actor = self.0 as u32;
            let epoch = self.0 >> 32;
            if actor & Self::REGISTERED_ACTOR_BIT == 0 {
                return clock
                    .component(actor as usize)
                    .is_some_and(|seen| seen >= epoch);
            }
            if actor & Self::ASYNC_ACTOR_BIT != 0 {
                let index =
                    (actor & !(Self::REGISTERED_ACTOR_BIT | Self::ASYNC_ACTOR_BIT)) as usize;
                return clock.async_component_at(index) >= epoch;
            }
            return registry.registered_vector_timestamp_observed_by(
                (actor & !Self::REGISTERED_ACTOR_BIT) as usize,
                clock,
            );
        }
        let timestamp = registry.wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize);
        match timestamp.actor {
            TimestampActor::Warp(index) => clock
                .component(index)
                .is_some_and(|seen| seen >= timestamp.epoch),
            TimestampActor::Async(index) => clock.async_component_at(index) >= timestamp.epoch,
            TimestampActor::RegisteredVector(index) => {
                registry.registered_vector_timestamp_observed_by(index, clock)
            }
        }
    }

    fn observed_by_frontier(
        &self,
        frontier: &ProxyClockFrontier,
        registry: &OperationRegistry,
        source_lane: (usize, usize),
    ) -> bool {
        if let (Some(lanes), Some(stamp)) = (&frontier.shared_lanes, self.lane_stamp(registry)) {
            return lanes
                .get(&source_lane.0)
                .is_some_and(|epochs| epochs[source_lane.1] >= stamp.epoch.get());
        }
        if self.0 & Self::WIDE_BIT == 0 {
            if self.0 & Self::LANE_STAMP_BIT != 0 {
                let actor = (self.0 & Self::LANE_WARP_MASK) as usize;
                let epoch = (self.0 >> Self::LANE_EVENT_EPOCH_SHIFT) & Self::LANE_EPOCH_MASK;
                return frontier.component(actor).is_some_and(|seen| seen >= epoch);
            }
            let actor = self.0 as u32;
            let epoch = self.0 >> 32;
            if actor & Self::REGISTERED_ACTOR_BIT == 0 {
                return frontier
                    .component(actor as usize)
                    .is_some_and(|seen| seen >= epoch);
            }
            if actor & Self::ASYNC_ACTOR_BIT != 0 {
                let index =
                    (actor & !(Self::REGISTERED_ACTOR_BIT | Self::ASYNC_ACTOR_BIT)) as usize;
                return frontier.async_component_at(index) >= epoch;
            }
            return registry.registered_vector_timestamp_observed_by_frontier(
                (actor & !Self::REGISTERED_ACTOR_BIT) as usize,
                frontier,
            );
        }
        let timestamp = registry.wide_timestamp((self.0 & Self::COMPACT_VALUE_MASK) as usize);
        match timestamp.actor {
            TimestampActor::Warp(index) => frontier
                .component(index)
                .is_some_and(|seen| seen >= timestamp.epoch),
            TimestampActor::Async(index) => frontier.async_component_at(index) >= timestamp.epoch,
            TimestampActor::RegisteredVector(index) => {
                registry.registered_vector_timestamp_observed_by_frontier(index, frontier)
            }
        }
    }
}

impl DirectTimestampEncoder {
    #[inline(always)]
    fn new(timestamp: RaceEventTimestamp, kind: PhysicalAccessKind) -> Self {
        if timestamp.0 & (RaceEventTimestamp::WIDE_BIT | RaceEventTimestamp::LANE_STAMP_BIT) == 0 {
            let actor = timestamp.0 as u32;
            let epoch = timestamp.0 >> 32;
            let compact_kind = match kind {
                PhysicalAccessKind::Read => 0_u64,
                PhysicalAccessKind::Write => 1_u64,
                PhysicalAccessKind::AtomicReadModifyWrite => 2_u64,
            };
            if actor & RaceEventTimestamp::REGISTERED_ACTOR_BIT == 0
                && u64::from(actor) <= RaceEventTimestamp::LANE_WARP_MASK
                && epoch <= RaceEventTimestamp::LANE_EPOCH_MASK
            {
                return Self::Compact {
                    base: RaceEventTimestamp::LANE_STAMP_BIT
                        | (epoch << RaceEventTimestamp::LANE_EVENT_EPOCH_SHIFT)
                        | u64::from(actor)
                        | (compact_kind << RaceEventTimestamp::LANE_KIND_SHIFT),
                    timestamp,
                    kind,
                };
            }
        }
        Self::Wide { timestamp, kind }
    }

    #[inline(always)]
    fn encode(
        self,
        lane_stamp: RetainedRaceLaneStamp,
        lane: usize,
        registry: &OperationRegistry,
    ) -> RaceEventTimestamp {
        debug_assert!(lane < WARP_SIZE);
        match self {
            Self::Compact { base, .. }
                if lane_stamp.epoch.get() <= RaceEventTimestamp::LANE_EPOCH_MASK =>
            {
                RaceEventTimestamp(
                    base | (lane_stamp.epoch.get() << RaceEventTimestamp::LANE_STAMP_SHIFT)
                        | ((lane as u64) << RaceEventTimestamp::LANE_LANE_SHIFT),
                )
            }
            Self::Compact {
                timestamp, kind, ..
            }
            | Self::Wide { timestamp, kind } => {
                timestamp.with_lane_stamp(Some(lane_stamp), lane, kind, registry)
            }
        }
    }

    #[inline(always)]
    fn event_timestamp(self) -> RaceEventTimestamp {
        match self {
            Self::Compact { timestamp, .. } | Self::Wide { timestamp, .. } => timestamp,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClockedWitness {
    timestamp: RaceEventTimestamp,
    witness: RetainedRaceWitness,
}

impl ClockedWitness {
    fn represents_same_direct_event(
        &self,
        registry: &OperationRegistry,
        timestamp: &RaceEventTimestamp,
        current: &PhysicalRaceWitnessRef,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> bool {
        let Some(lane_order) = lane_order else {
            return false;
        };
        self.timestamp.same_event(*timestamp, registry)
            && self.witness.same_operation(registry, current)
            && self.witness.resolved_lane(registry) == current.lane()
            && self.witness.resolved_kind(registry) == current.kind()
            && self
                .witness
                .same_proxy_and_domain(registry, current.proxy(), current.proxy_domain())
            && self.timestamp.lane_stamp(registry)
                == Some(lane_order.retained_stamp(current.lane()))
    }

    fn observed_by_lane(
        &self,
        registry: &OperationRegistry,
        lane_order: Option<&RaceLaneOrder<'_>>,
        current_lane: usize,
    ) -> Option<bool> {
        let order = lane_order?;
        let prior = self.timestamp.lane_stamp(registry)?;
        let prior_warp = self.witness.global_warp_id(registry);
        (prior_warp == order.global_warp_id || order.incoming_common.is_some()).then(|| {
            order.observes(
                current_lane,
                prior_warp,
                self.witness.resolved_lane(registry),
                prior,
            )
        })
    }

    fn ordered_before(
        &self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current: &PhysicalRaceWitnessRef,
        current_lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
    ) -> bool {
        let (prior_proxy, prior_proxy_domain) = self.witness.proxy_and_domain(registry);
        if prior_proxy == current.proxy() {
            // Same-proxy lane history already includes the acquired releases.
            // Only cross-proxy and vector-clock queries need the merged clock.
            if allow_same_operation && self.witness.same_operation(registry, current) {
                return true;
            }
            if let Some(ordered) = self.observed_by_lane(registry, current_lane_order, current.lane()) {
                return ordered;
            }
            // Sibling lanes of one batch remain simultaneous.
            if self.witness.same_operation(registry, current) {
                return false;
            }
        }
        let acquired_clock =
            current_lane_order.map(|order| order.acquired_clock(current_clock, current.lane()));
        let current_clock = acquired_clock.as_deref().unwrap_or(current_clock);
        if prior_proxy != current.proxy() {
            return current_clock.proxy_bridge_observes(
                prior_proxy,
                current.proxy(),
                prior_proxy_domain,
                current.proxy_domain(),
                self.timestamp,
                registry,
                current.lane(),
                (
                    self.witness.global_warp_id(registry),
                    self.witness.resolved_lane(registry),
                ),
            );
        }
        self.timestamp.observed_by(current_clock, registry)
    }

    fn ordering_failure(
        &self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current: &PhysicalRaceWitnessRef,
        current_timestamp: RaceEventTimestamp,
        current_lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> PhysicalRaceOrderingFailure {
        let (prior_proxy, prior_domain) = self.witness.proxy_and_domain(registry);
        if prior_proxy != current.proxy() {
            return PhysicalRaceOrderingFailure::MissingProxyBridge {
                prior_proxy,
                current_proxy: current.proxy(),
                prior_domain: prior_domain.into(),
                current_domain: current.proxy_domain().into(),
            };
        }
        if let (Some(prior), Some(current_lane_order)) =
            (self.timestamp.lane_stamp(registry), current_lane_order)
        {
            if self.witness.global_warp_id(registry) == current.global_warp_id()
                && !current_lane_order.observes(
                    current.lane(),
                    self.witness.global_warp_id(registry),
                    self.witness.resolved_lane(registry),
                    prior,
                )
            {
                return PhysicalRaceOrderingFailure::MissingSameWarpLaneOrder;
            }
        }
        let missing_ordinary_order = || {
            if self.witness.global_warp_id(registry) == current.global_warp_id()
                && self.witness.resolved_lane(registry) != current.lane()
            {
                return PhysicalRaceOrderingFailure::MissingSameWarpLaneOrder;
            }
            let prior_kind = self.witness.resolved_kind(registry);
            if prior_kind == PhysicalAccessKind::AtomicReadModifyWrite
                || current.kind() == PhysicalAccessKind::AtomicReadModifyWrite
            {
                PhysicalRaceOrderingFailure::MissingReleaseAcquire
            } else {
                PhysicalRaceOrderingFailure::MissingInterActorSynchronization
            }
        };
        if !self
            .timestamp
            .ordinary_observed_by(current_timestamp, current_clock, registry)
        {
            return missing_ordinary_order();
        }
        if self.timestamp.is_async_actor(registry) || current_timestamp.is_async_actor(registry) {
            PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained
        } else {
            missing_ordinary_order()
        }
    }

    fn cross_proxy_ordered_before(
        &self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current_proxy: MemoryProxy,
        current_proxy_domain: ProxyMemoryDomain,
        current_lane: usize,
    ) -> Option<bool> {
        if self.witness.same_proxy(registry, current_proxy) {
            return None;
        }
        let (prior_proxy, prior_proxy_domain) = self.witness.proxy_and_domain(registry);
        Some(current_clock.proxy_bridge_observes(
            prior_proxy,
            current_proxy,
            prior_proxy_domain,
            current_proxy_domain,
            self.timestamp,
            registry,
            current_lane,
            (
                self.witness.global_warp_id(registry),
                self.witness.resolved_lane(registry),
            ),
        ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum AccessFrontier {
    #[default]
    Empty,
    One(ClockedWitness),
    Two(Box<[ClockedWitness; 2]>),
    Many(Box<Vec<ClockedWitness>>),
}

impl AccessFrontier {
    fn single(&self) -> Option<&ClockedWitness> {
        match self {
            Self::One(access) => Some(access),
            _ => None,
        }
    }

    fn from_witnesses(mut reads: Vec<ClockedWitness>) -> Self {
        match reads.len() {
            0 => Self::Empty,
            1 => Self::One(reads.pop().expect("one read")),
            2 => {
                let second = reads.pop().expect("two reads");
                let first = reads.pop().expect("two reads");
                Self::Two(Box::new([first, second]))
            }
            _ => Self::Many(Box::new(reads)),
        }
    }

    fn as_slice(&self) -> &[ClockedWitness] {
        match self {
            Self::Empty => &[],
            Self::One(read) => std::slice::from_ref(read),
            Self::Two(reads) => reads.as_slice(),
            Self::Many(reads) => reads,
        }
    }

    fn iter(&self) -> std::slice::Iter<'_, ClockedWitness> {
        self.as_slice().iter()
    }

    fn record(
        &mut self,
        read: ClockedWitness,
        current_clock: &RaceVectorClock,
        registry: &OperationRegistry,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) {
        let (current_proxy, current_proxy_domain) = read.witness.proxy_and_domain(registry);
        let current_lane = read.witness.resolved_lane(registry);
        self.record_with(read, registry, |prior| {
            if let Some(ordered) = prior.cross_proxy_ordered_before(
                registry,
                current_clock,
                current_proxy,
                current_proxy_domain,
                current_lane,
            ) {
                return ordered;
            }
            if let Some(ordered) = prior.observed_by_lane(registry, lane_order, current_lane) {
                return ordered;
            }
            prior.timestamp.observed_by(current_clock, registry)
        });
    }

    /// Record the reads of one operation's aliased lanes, with the same result
    /// as recording them lane by lane with `record`. Whether a lane observes a
    /// retained read does not depend on the frontier's contents, so a prior
    /// read is dropped exactly when some lane observes it. A read of another
    /// warp judged by the common release, or one judged by the clock, has the
    /// same verdict for every lane and is checked once instead of per lane.
    /// Returns `false` and changes nothing when the frontier is not a list of
    /// reads or the group has a shape only the lane-by-lane path handles.
    fn record_lane_group(
        &mut self,
        reads: &[ClockedWitness],
        current_clock: &RaceVectorClock,
        registry: &OperationRegistry,
        lane_order: &RaceLaneOrder<'_>,
    ) -> bool {
        let Self::Many(priors) = self else {
            return false;
        };
        let Some(first) = reads.first() else {
            return false;
        };
        if reads.len() < 2
            || reads
                .iter()
                .any(|read| read.witness.0.get() & RetainedRaceWitness::WIDE_BIT != 0)
        {
            return false;
        }
        let lanes = reads
            .iter()
            .map(|read| read.witness.resolved_lane(registry))
            .collect::<Vec<_>>();
        if lanes
            .iter()
            .try_fold(0_u32, |seen, &lane| (seen & (1 << lane) == 0).then_some(seen | 1 << lane))
            .is_none()
        {
            return false;
        }
        let (current_proxy, current_proxy_domain) = first.witness.proxy_and_domain(registry);
        let observed_by = |prior: &ClockedWitness, lane: usize| {
            if let Some(ordered) = prior.cross_proxy_ordered_before(
                registry,
                current_clock,
                current_proxy,
                current_proxy_domain,
                lane,
            ) {
                return ordered;
            }
            if let Some(ordered) = prior.observed_by_lane(registry, Some(lane_order), lane) {
                return ordered;
            }
            prior.timestamp.observed_by(current_clock, registry)
        };
        let lane_independent = |prior: &ClockedWitness| {
            prior.witness.same_proxy(registry, current_proxy)
                && lane_order.incoming_lanes.is_none()
                && (prior.timestamp.lane_stamp(registry).is_none()
                    || prior.witness.global_warp_id(registry) != lane_order.global_warp_id)
        };
        let mut keep = Vec::with_capacity(priors.len());
        for prior in priors.iter() {
            if prior.witness.same_retained_operation(registry, first.witness) {
                return false;
            }
            let observed = first.witness.subsumes_contract(registry, prior.witness)
                && if lane_independent(prior) {
                    observed_by(prior, lanes[0])
                } else {
                    lanes.iter().any(|&lane| observed_by(prior, lane))
                };
            keep.push(!observed);
        }
        if !keep.iter().any(|keep| *keep) {
            return false;
        }
        for _ in reads {
            profile_count(ProfileKind::RaceAccessFrontierMany);
        }
        let mut index = 0;
        priors.retain(|_| {
            index += 1;
            keep[index - 1]
        });
        // Lane by lane, a later lane of the group also drops an earlier
        // lane's read that it observes.
        let start = priors.len();
        for (read, &lane) in reads.iter().zip(&lanes) {
            let mut position = start;
            while position < priors.len() {
                if read.witness.subsumes_contract(registry, priors[position].witness)
                    && observed_by(&priors[position], lane)
                {
                    priors.remove(position);
                } else {
                    position += 1;
                }
            }
            priors.push(read.clone());
        }
        true
    }

    #[inline(always)]
    fn record_with(
        &mut self,
        read: ClockedWitness,
        registry: &OperationRegistry,
        mut observed: impl FnMut(&ClockedWitness) -> bool,
    ) {
        // Happens-before alone cannot discard an older access if the newer
        // one can be morally strong with a future access that still races
        // with the older one. Preserve differing conflict contracts.
        let current = read.witness;
        let subsumes = |prior: &ClockedWitness| current.subsumes_contract(registry, prior.witness);
        let mut observed = |prior: &ClockedWitness| subsumes(prior) && observed(prior);
        match self {
            Self::Empty => {
                profile_count(ProfileKind::RaceAccessFrontierEmpty);
                *self = Self::One(read);
            }
            Self::One(prior) => {
                if prior
                    .witness
                    .same_retained_operation_and_proxy(registry, read.witness)
                    && subsumes(prior)
                {
                    profile_count(ProfileKind::RaceAccessFrontierOneSame);
                    return;
                }
                if observed(prior) {
                    profile_count(ProfileKind::RaceAccessFrontierOneReplace);
                    *prior = read;
                    return;
                }
                profile_count(ProfileKind::RaceAccessFrontierOneMany);
                let Self::One(prior) = std::mem::replace(self, Self::Empty) else {
                    unreachable!("the read frontier was matched as one element");
                };
                *self = Self::Two(Box::new([prior, read]));
            }
            Self::Two(reads) => {
                profile_count(ProfileKind::RaceAccessFrontierMany);
                let same_first = reads[0]
                    .witness
                    .same_retained_operation_and_proxy(registry, read.witness)
                    && subsumes(&reads[0]);
                let keep_first = same_first || !observed(&reads[0]);
                let same_second = reads[1]
                    .witness
                    .same_retained_operation_and_proxy(registry, read.witness)
                    && subsumes(&reads[1]);
                let keep_second = same_second || !observed(&reads[1]);
                let already_represented = same_first || same_second;

                if already_represented {
                    match (keep_first, keep_second) {
                        (true, true) => {}
                        (true, false) | (false, true) => {
                            let Self::Two(mut reads) = std::mem::replace(self, Self::Empty) else {
                                unreachable!("the read frontier was matched as two elements");
                            };
                            if !keep_first {
                                reads.swap(0, 1);
                            }
                            let [read, _] = *reads;
                            *self = Self::One(read);
                        }
                        (false, false) => {
                            unreachable!("a represented read remains in the frontier")
                        }
                    }
                    return;
                }

                match (keep_first, keep_second) {
                    (false, false) => *self = Self::One(read),
                    (true, false) => reads[1] = read,
                    (false, true) => {
                        reads.swap(0, 1);
                        reads[1] = read;
                    }
                    (true, true) => {
                        let Self::Two(reads) = std::mem::replace(self, Self::Empty) else {
                            unreachable!("the read frontier was matched as two elements");
                        };
                        let reads: Box<[ClockedWitness]> = reads;
                        let mut reads = Vec::from(reads);
                        reads.push(read);
                        *self = Self::Many(Box::new(reads));
                    }
                }
            }
            Self::Many(reads) => {
                profile_count(ProfileKind::RaceAccessFrontierMany);
                let mut already_represented = false;
                reads.retain(|prior| {
                    let same_operation = prior
                        .witness
                        .same_retained_operation_and_proxy(registry, read.witness)
                        && subsumes(prior);
                    already_represented |= same_operation;
                    same_operation || !observed(prior)
                });
                if !already_represented {
                    reads.push(read);
                }
                if reads.len() == 1 {
                    let read = reads
                        .pop()
                        .expect("a one-element read frontier has one read");
                    *self = Self::One(read);
                }
            }
        }
    }

    fn retain(&mut self, mut keep: impl FnMut(&ClockedWitness) -> bool) {
        match self {
            Self::Empty => {}
            Self::One(read) => {
                if !keep(read) {
                    *self = Self::Empty;
                }
            }
            Self::Two(reads) => {
                let keep_first = keep(&reads[0]);
                let keep_second = keep(&reads[1]);
                match (keep_first, keep_second) {
                    (true, true) => {}
                    (false, false) => *self = Self::Empty,
                    (true, false) | (false, true) => {
                        let Self::Two(mut reads) = std::mem::replace(self, Self::Empty) else {
                            unreachable!("the read frontier was matched as two elements");
                        };
                        if !keep_first {
                            reads.swap(0, 1);
                        }
                        let [read, _] = *reads;
                        *self = Self::One(read);
                    }
                }
            }
            Self::Many(reads) => {
                reads.retain(|read| keep(read));
                match reads.len() {
                    0 => *self = Self::Empty,
                    1 => {
                        let read = reads
                            .pop()
                            .expect("a one-element read frontier has one read");
                        *self = Self::One(read);
                    }
                    _ => {}
                }
            }
        }
    }

    fn clear(&mut self) {
        *self = Self::Empty;
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ShadowState {
    writes: AccessFrontier,
    reads: AccessFrontier,
}

impl ShadowState {
    fn direct_shared_prior_is_ordered(
        prior: &ClockedWitness,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current_operation: RegisteredOperation,
        current_global_warp_id: usize,
        current_local_warp_id: usize,
        current_kind: PhysicalAccessKind,
        proxy_access: ProxyAccessClass,
        current_lane: usize,
        lane_order: &RaceLaneOrder<'_>,
    ) -> bool {
        if current_kind == PhysicalAccessKind::AtomicReadModifyWrite
            && prior.witness.resolved_kind(registry) == PhysicalAccessKind::AtomicReadModifyWrite
        {
            profile_count(ProfileKind::RacePriorAtomic);
            return true;
        }
        if proxy_access.is_sensitive() && !prior.witness.same_proxy(registry, proxy_access.proxy()) {
            let acquired_clock = lane_order.acquired_clock(current_clock, current_lane);
            if let Some(ordered) = prior.cross_proxy_ordered_before(
                registry,
                &acquired_clock,
                proxy_access.proxy(),
                proxy_access.domain(),
                current_lane,
            ) {
                return ordered;
            }
        }
        // One GPU lane is a single program-order actor. Direct compact
        // accesses reach this path only after numeric execution, and a lane
        // cannot execute its next batch before its previous batch. Avoid
        // reconstructing the lane-clock component for this dominant exact
        // case; cross-lane accesses still require the retained lane frontier.
        if let Some((prior_local_warp_id, prior_epoch, prior_lane_stamp, prior_lane, prior_kind)) =
            prior.timestamp.compact_direct_warp_components()
        {
            if current_local_warp_id == prior_local_warp_id as usize {
                profile_count(ProfileKind::RacePriorSameWarp);
                if prior_lane == current_lane {
                    profile_count(ProfileKind::RacePriorSameLane);
                    return true;
                }
                profile_count(ProfileKind::RacePriorOtherLane);
                return lane_order.observes(
                    current_lane,
                    current_global_warp_id,
                    prior_lane,
                    prior_lane_stamp,
                );
            }
            if current_kind == PhysicalAccessKind::AtomicReadModifyWrite
                && prior_kind == PhysicalAccessKind::AtomicReadModifyWrite
            {
                profile_count(ProfileKind::RacePriorAtomic);
                return true;
            }
            profile_count(ProfileKind::RacePriorCrossWarp);
            if lane_order.incoming_common.is_some() {
                return lane_order.observes(
                    current_lane,
                    lane_order.global_warp_base + prior_local_warp_id as usize,
                    prior_lane,
                    prior_lane_stamp,
                );
            }
            return current_clock.direct_component(prior_local_warp_id) >= u64::from(prior_epoch);
        }
        if let Some((prior_local_warp_id, prior_epoch, prior_lane_stamp)) =
            prior.timestamp.wide_direct_warp_components(registry)
        {
            if current_local_warp_id == prior_local_warp_id {
                profile_count(ProfileKind::RacePriorSameWarp);
                let prior_lane = prior.witness.resolved_lane(registry);
                if prior_lane == current_lane {
                    profile_count(ProfileKind::RacePriorSameLane);
                    return true;
                }
                profile_count(ProfileKind::RacePriorOtherLane);
                return lane_order.observes(
                    current_lane,
                    current_global_warp_id,
                    prior_lane,
                    prior_lane_stamp,
                );
            }
            if current_kind == PhysicalAccessKind::AtomicReadModifyWrite
                && prior.witness.resolved_kind(registry)
                    == PhysicalAccessKind::AtomicReadModifyWrite
            {
                profile_count(ProfileKind::RacePriorAtomic);
                return true;
            }
            profile_count(ProfileKind::RacePriorCrossWarp);
            if lane_order.incoming_common.is_some() {
                return lane_order.observes(
                    current_lane,
                    lane_order.global_warp_base + prior_local_warp_id,
                    prior.witness.resolved_lane(registry),
                    prior_lane_stamp,
                );
            }
            return current_clock
                .component(prior_local_warp_id)
                .is_some_and(|seen| seen >= prior_epoch);
        }
        if current_kind == PhysicalAccessKind::AtomicReadModifyWrite
            && prior.witness.resolved_kind(registry) == PhysicalAccessKind::AtomicReadModifyWrite
        {
            profile_count(ProfileKind::RacePriorAtomic);
            return true;
        }
        if prior.witness.operation(registry) == current_operation {
            profile_count(ProfileKind::RacePriorSameOperation);
            return false;
        }
        profile_count(ProfileKind::RacePriorCrossWarp);
        lane_order.timestamp_observed_by_acquired(
            &prior.timestamp,
            current_clock,
            current_lane,
            registry,
        )
    }

    /// Fast clean-path proof for a direct shared-memory access.
    ///
    /// Compact direct accesses already have an exact allocation and span, so
    /// the general validator's retained-witness reconstruction is only needed
    /// when a conflict is possible. Same-warp predecessors still use the exact
    /// per-lane clock; cross-warp and async predecessors use their vector-clock
    /// timestamp. This is the same ordering predicate as `validate_access`,
    /// specialized to shared memory where no TMEM review is required.
    #[allow(clippy::too_many_arguments)]
    fn direct_shared_access_is_ordered(
        &self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current_operation: RegisteredOperation,
        actor: DirectAccessActor,
        current_kind: PhysicalAccessKind,
        proxy_access: ProxyAccessClass,
        current_lane: usize,
        lane_order: &RaceLaneOrder<'_>,
    ) -> bool {
        if self.writes.iter().any(|prior| {
            !Self::direct_shared_prior_is_ordered(
                prior,
                registry,
                current_clock,
                current_operation,
                actor.global_warp_id,
                actor.local_warp_id,
                current_kind,
                proxy_access,
                current_lane,
                lane_order,
            )
        }) {
            return false;
        }
        !current_kind.writes()
            || self.reads.iter().all(|prior| {
                Self::direct_shared_prior_is_ordered(
                    prior,
                    registry,
                    current_clock,
                    current_operation,
                    actor.global_warp_id,
                    actor.local_warp_id,
                    current_kind,
                    proxy_access,
                    current_lane,
                    lane_order,
                )
            })
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn validate_and_record_direct_lane(
        &mut self,
        registry: &OperationRegistry,
        event_clock: &RaceVectorClock,
        timestamp_encoder: DirectTimestampEncoder,
        operation: RegisteredOperation,
        actor: DirectAccessActor,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy_access: ProxyAccessClass,
        span: PhysicalByteSpan,
        state_span: PhysicalByteSpan,
        lane: usize,
        lane_order: &RaceLaneOrder<'_>,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<(), RaceShadowError> {
        if space != PhysicalAccessSpace::Shared
            || !self.direct_shared_access_is_ordered(
                registry,
                event_clock,
                operation,
                actor,
                kind,
                proxy_access,
                lane,
                lane_order,
            )
        {
            let current = PhysicalRaceWitnessRef::from_parts(
                operation,
                lane,
                kind,
                space,
                proxy_access.proxy(),
                proxy_access.domain(),
                span,
            );
            self.validate_access(
                registry,
                event_clock,
                timestamp_encoder.event_timestamp(),
                &current,
                state_span,
                Some(lane_order),
                false,
                review_findings,
            )?;
        }
        self.record_unique_direct_lane(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            kind,
            space,
            proxy_access,
            span,
            lane,
            lane_order,
        );
        Ok(())
    }

    /// Record one lane from geometry that excludes same-batch span aliases.
    ///
    /// `record_validated` first searches for a duplicate witness from the same
    /// semantic event. A unique direct span cannot already contain this
    /// registered operation, so that search is redundant; reads still pass
    /// through `AccessFrontier::record` for cross-operation subsumption.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn record_unique_direct_lane(
        &mut self,
        registry: &OperationRegistry,
        event_clock: &RaceVectorClock,
        timestamp_encoder: DirectTimestampEncoder,
        operation: RegisteredOperation,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy_access: ProxyAccessClass,
        span: PhysicalByteSpan,
        lane: usize,
        lane_order: &RaceLaneOrder<'_>,
    ) {
        let clocked = Self::direct_lane_witness(
            registry,
            timestamp_encoder,
            operation,
            kind,
            space,
            proxy_access,
            span,
            lane,
            lane_order,
        );
        if kind.writes() {
            self.writes
                .record(clocked, event_clock, registry, Some(lane_order));
            if kind == PhysicalAccessKind::Write {
                self.reads.clear();
            }
        } else {
            self.reads
                .record(clocked, event_clock, registry, Some(lane_order));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn direct_lane_witness(
        registry: &OperationRegistry,
        timestamp_encoder: DirectTimestampEncoder,
        operation: RegisteredOperation,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy_access: ProxyAccessClass,
        span: PhysicalByteSpan,
        lane: usize,
        lane_order: &RaceLaneOrder<'_>,
    ) -> ClockedWitness {
        ClockedWitness {
            timestamp: timestamp_encoder.encode(lane_order.retained_stamp(lane), lane, registry),
            witness: RetainedRaceWitness::new(
                operation,
                span,
                lane as u8,
                kind,
                space,
                proxy_access.proxy(),
                proxy_access.domain(),
                None,
                registry,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn validate_and_record_direct_lane_group(
        &mut self,
        registry: &OperationRegistry,
        event_clock: &RaceVectorClock,
        timestamp_encoder: DirectTimestampEncoder,
        operation: RegisteredOperation,
        actor: DirectAccessActor,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy_access: ProxyAccessClass,
        span: PhysicalByteSpan,
        state_span: PhysicalByteSpan,
        lanes: &[u8],
        lane_order: &RaceLaneOrder<'_>,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<(), RaceShadowError> {
        // Every aliased lane observes the same pre-batch state. Do not let an
        // earlier lane's publication hide a lane-dependent conflict for a
        // later lane in the same semantic operation.
        for &lane in lanes {
            let lane = usize::from(lane);
            if space == PhysicalAccessSpace::Shared
                && self.direct_shared_access_is_ordered(
                    registry,
                    event_clock,
                    operation,
                    actor,
                    kind,
                    proxy_access,
                    lane,
                    lane_order,
                )
            {
                continue;
            }
            let current = PhysicalRaceWitnessRef::from_parts(
                operation,
                lane,
                kind,
                space,
                proxy_access.proxy(),
                proxy_access.domain(),
                span,
            );
            self.validate_access(
                registry,
                event_clock,
                timestamp_encoder.event_timestamp(),
                &current,
                state_span,
                Some(lane_order),
                false,
                review_findings,
            )?;
        }
        self.record_direct_lane_group(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            kind,
            space,
            proxy_access,
            span,
            lanes,
            lane_order,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_direct_lane_group(
        &mut self,
        registry: &OperationRegistry,
        event_clock: &RaceVectorClock,
        timestamp_encoder: DirectTimestampEncoder,
        operation: RegisteredOperation,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy_access: ProxyAccessClass,
        span: PhysicalByteSpan,
        lanes: &[u8],
        lane_order: &RaceLaneOrder<'_>,
    ) {
        // Each aliased lane has its own synchronization history. Validating
        // all lanes and then retaining just one loses later partial-warp races.
        if !kind.writes() && lanes.len() > 1 && matches!(self.reads, AccessFrontier::Many(_)) {
            let reads = lanes
                .iter()
                .map(|&lane| {
                    Self::direct_lane_witness(
                        registry,
                        timestamp_encoder,
                        operation,
                        kind,
                        space,
                        proxy_access,
                        span,
                        usize::from(lane),
                        lane_order,
                    )
                })
                .collect::<Vec<_>>();
            if !self
                .reads
                .record_lane_group(&reads, event_clock, registry, lane_order)
            {
                for read in reads {
                    self.reads
                        .record(read, event_clock, registry, Some(lane_order));
                }
            }
            return;
        }
        for &lane in lanes {
            self.record_unique_direct_lane(
                registry,
                event_clock,
                timestamp_encoder,
                operation,
                kind,
                space,
                proxy_access,
                span,
                usize::from(lane),
                lane_order,
            );
        }
    }

    fn records_same_operation_kind(
        &self,
        registry: &OperationRegistry,
        current: &PhysicalRaceWitnessRef,
    ) -> bool {
        if current.kind().writes() {
            return self.writes.single().is_some_and(|prior| {
                prior.witness.same_operation(registry, current)
                    && prior.witness.resolved_kind(registry) == current.kind()
                    && prior.witness.same_proxy_and_domain(
                        registry,
                        current.proxy(),
                        current.proxy_domain(),
                    )
            });
        }
        self.reads.iter().any(|prior| {
            prior.witness.same_operation(registry, current)
                && prior.witness.resolved_kind(registry) == current.kind()
                && prior.witness.same_proxy_and_domain(
                    registry,
                    current.proxy(),
                    current.proxy_domain(),
                )
        })
    }

    fn check_and_record(
        &mut self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        timestamp: &RaceEventTimestamp,
        current: PhysicalRaceWitnessRef,
        overlap: PhysicalByteSpan,
        current_lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<(), RaceShadowError> {
        self.validate_access(
            registry,
            current_clock,
            *timestamp,
            &current,
            overlap,
            current_lane_order,
            allow_same_operation,
            review_findings,
        )?;
        if !reviewed_tmem_load(review_findings, registry, current.operation) {
            self.record_validated(
                registry,
                current_clock,
                timestamp,
                current,
                current_lane_order,
            );
        }
        Ok(())
    }

    fn validate_access(
        &self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        current_timestamp: RaceEventTimestamp,
        current: &PhysicalRaceWitnessRef,
        _overlap: PhysicalByteSpan,
        current_lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<(), RaceShadowError> {
        if reviewed_tmem_load(review_findings, registry, current.operation) {
            return Ok(());
        }
        if current.kind().reads() {
            for prior in self.writes.iter() {
                if !atomic_modification_order_pair(registry, &prior.witness, &current)
                    && !prior.ordered_before(
                        registry,
                        current_clock,
                        &current,
                        current_lane_order,
                        allow_same_operation,
                    )
                {
                    record_review_or_reject(
                        review_findings,
                        physical_race_finding(
                            registry,
                            PhysicalRaceKind::WriteRead,
                            prior,
                            current_clock,
                            current,
                            current_timestamp,
                            current_lane_order,
                        ),
                    )?;
                }
            }
        }

        if current.kind().writes() {
            for prior in self.reads.iter() {
                if reviewed_tmem_load(review_findings, registry, prior.witness.operation(registry))
                    || atomic_modification_order_pair(registry, &prior.witness, current)
                    || prior.ordered_before(
                        registry,
                        current_clock,
                        &current,
                        current_lane_order,
                        allow_same_operation,
                    )
                {
                    continue;
                }
                record_review_or_reject(
                    review_findings,
                    physical_race_finding(
                        registry,
                        PhysicalRaceKind::ReadWrite,
                        prior,
                        current_clock,
                        current,
                        current_timestamp,
                        current_lane_order,
                    ),
                )?;
            }
            for prior in self.writes.iter() {
                if !atomic_modification_order_pair(registry, &prior.witness, &current)
                    && !prior.ordered_before(
                        registry,
                        current_clock,
                        &current,
                        current_lane_order,
                        allow_same_operation,
                    )
                {
                    record_review_or_reject(
                        review_findings,
                        physical_race_finding(
                            registry,
                            PhysicalRaceKind::WriteWrite,
                            prior,
                            current_clock,
                            current,
                            current_timestamp,
                            current_lane_order,
                        ),
                    )?;
                }
            }
        }

        Ok(())
    }

    fn record_validated(
        &mut self,
        registry: &OperationRegistry,
        current_clock: &RaceVectorClock,
        timestamp: &RaceEventTimestamp,
        current: PhysicalRaceWitnessRef,
        current_lane_order: Option<&RaceLaneOrder<'_>>,
    ) {
        if current.kind().writes() {
            if self.writes.single().is_some_and(|write| {
                write.represents_same_direct_event(
                    registry,
                    timestamp,
                    &current,
                    current_lane_order,
                )
            }) {
                if current.kind() == PhysicalAccessKind::Write && current.strong_scope.is_none() {
                    self.reads.clear();
                }
                return;
            }
        } else if self.reads.iter().any(|read| {
            read.represents_same_direct_event(registry, timestamp, &current, current_lane_order)
        }) {
            return;
        }

        let writes = current.kind().writes();
        let clocked = ClockedWitness {
            timestamp: timestamp.with_lane_stamp(
                current_lane_order.map(|order| order.retained_stamp(current.lane())),
                current.lane(),
                current.kind(),
                registry,
            ),
            witness: current.retained(registry),
        };
        if writes {
            self.writes
                .record(clocked, current_clock, registry, current_lane_order);
            if current.kind() == PhysicalAccessKind::Write && current.strong_scope.is_none() {
                self.reads.clear();
            }
        } else {
            // Aliased lanes remain distinct ordering actors; only a later
            // access that observes a witness may replace that witness.
            self.reads
                .record(clocked, current_clock, registry, current_lane_order);
        }
    }

    /// Whether any retained witness is dominated by `observed_frontier`.
    fn has_globally_observed_witness(
        &self,
        registry: &OperationRegistry,
        observed_frontier: &RaceVectorClock,
        proxy_sensitive: bool,
    ) -> bool {
        self.writes.iter().chain(self.reads.iter()).any(|witness| {
            witness_is_globally_observed(witness, registry, observed_frontier, proxy_sensitive)
        })
    }

    /// Mark the async clock slots the retained witnesses refer to.
    fn mark_referenced_async(&self, registry: &OperationRegistry, referenced: &mut AsyncIndexSet) {
        for witness in self.writes.iter().chain(self.reads.iter()) {
            referenced.mark_witness(witness, registry);
        }
    }

    fn retire_globally_observed(
        &mut self,
        registry: &OperationRegistry,
        observed_frontier: &RaceVectorClock,
        proxy_sensitive: bool,
        retired: &mut RetiredProxyFrontiers,
        warp_count: usize,
        async_registry: &Arc<AsyncClockRegistry>,
        referenced_async: &mut AsyncIndexSet,
    ) -> Result<usize, RaceShadowError> {
        let mut retired_count = 0;
        let mut write_retirement = Ok(());
        self.writes.retain(|write| {
            if witness_is_globally_observed(write, registry, observed_frontier, proxy_sensitive) {
                if !proxy_sensitive && write_retirement.is_ok() {
                    write_retirement =
                        note_retired_generic(write, registry, retired, warp_count, async_registry);
                }
                retired_count += 1;
                false
            } else {
                referenced_async.mark_witness(write, registry);
                true
            }
        });
        write_retirement?;
        let before = self.reads.len();
        let mut noted = Ok(());
        self.reads.retain(|read| {
            let retire =
                witness_is_globally_observed(read, registry, observed_frontier, proxy_sensitive);
            if retire && !proxy_sensitive && noted.is_ok() {
                noted = note_retired_generic(read, registry, retired, warp_count, async_registry);
            }
            if !retire {
                referenced_async.mark_witness(read, registry);
            }
            !retire
        });
        noted?;
        Ok(retired_count + before - self.reads.len())
    }

    fn retire_reviewed_tmem_loads(
        &mut self,
        registry: &OperationRegistry,
        operations: &[RegisteredOperation],
    ) -> usize {
        let before = self.reads.len();
        self.reads
            .retain(|read| !operations.contains(&read.witness.operation(registry)));
        before - self.reads.len()
    }

    fn is_empty(&self) -> bool {
        self.writes.is_empty() && self.reads.is_empty()
    }
}

fn reviewed_tmem_load(
    review_findings: &[PhysicalRaceFinding],
    registry: &OperationRegistry,
    operation: RegisteredOperation,
) -> bool {
    // The clean path has no review findings. Check the local slice before
    // consulting the shared operation registry; doing this in the opposite
    // order acquired its mutex for every ordinary physical-byte access.
    review_findings
        .iter()
        .any(|finding| finding.reviewed_tmem_load_handle() == Some(operation))
        && registry.is_tmem_load_register_source(operation)
}

fn physical_race_finding(
    registry: &OperationRegistry,
    kind: PhysicalRaceKind,
    prior: &ClockedWitness,
    current_clock: &RaceVectorClock,
    current: &PhysicalRaceWitnessRef,
    current_timestamp: RaceEventTimestamp,
    current_lane_order: Option<&RaceLaneOrder<'_>>,
) -> PhysicalRaceFinding {
    let allocation = current.span().allocation();
    let mut finding = PhysicalRaceFinding::new(
        kind,
        prior.ordering_failure(
            registry,
            current_clock,
            current,
            current_timestamp,
            current_lane_order,
        ),
        prior.witness.materialize(registry, allocation),
        current.materialize(registry),
        witness_overlap(registry, &prior.witness, current),
    );
    if finding.ordering_failure() == PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained {
        let prior_is_tmem_load = prior.witness.resolved_kind(registry).reads()
            && registry.is_tmem_load_register_source(prior.witness.operation(registry));
        if prior_is_tmem_load {
            finding.reviewed_tmem_load_handle = Some(prior.witness.operation(registry));
        }
    }
    finding
}

fn record_review_or_reject(
    review_findings: &mut Vec<PhysicalRaceFinding>,
    finding: PhysicalRaceFinding,
) -> Result<(), RaceShadowError> {
    if finding.requires_unwaited_tmem_load_review() {
        review_findings.push(finding);
        Ok(())
    } else {
        Err(RaceShadowError::Race(finding))
    }
}

fn atomic_modification_order_pair(
    registry: &OperationRegistry,
    prior: &RetainedRaceWitness,
    current: &PhysicalRaceWitnessRef,
) -> bool {
    if prior.resolved_kind(registry) == PhysicalAccessKind::AtomicReadModifyWrite
        && current.kind() == PhysicalAccessKind::AtomicReadModifyWrite
    {
        return true;
    }
    let Some(current_scope) = current.strong_scope else {
        return false;
    };
    let prior = prior.resolve(registry, current.span().allocation());
    let Some(prior_scope) = prior.strong_scope else {
        return false;
    };
    let covers = |scope, source, target| {
        crate::race_check::scope_covers_warps(registry.topology, scope, source, target)
    };
    prior.span == current.span()
        && prior.proxy == current.proxy()
        && covers(
            prior_scope,
            prior.operation.global_warp_id(),
            current.global_warp_id(),
        )
        && covers(
            current_scope,
            current.global_warp_id(),
            prior.operation.global_warp_id(),
        )
}

fn witness_overlap(
    registry: &OperationRegistry,
    prior: &RetainedRaceWitness,
    current: &PhysicalRaceWitnessRef,
) -> PhysicalByteSpan {
    let current_span = current.span();
    let prior_span = prior.span(registry, current_span.allocation());
    debug_assert!(prior_span.overlaps(current_span));
    let start = prior_span.byte_offset().max(current_span.byte_offset());
    let end = prior_span.byte_end().min(current_span.byte_end());
    PhysicalByteSpan::new(prior_span.allocation(), start, end - start)
        .expect("overlapping valid witness spans have a nonempty intersection")
}

/// Bitset over async clock slots, used by the dominated-frontier GC to find
/// the slots that retained witnesses still refer to.
struct AsyncIndexSet {
    words: Vec<u64>,
}

impl AsyncIndexSet {
    fn new(slots: usize) -> Self {
        Self {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    fn mark(&mut self, index: usize) {
        if let Some(word) = self.words.get_mut(index / 64) {
            *word |= 1 << (index % 64);
        }
    }

    fn contains(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }

    /// Mark every slot `witness` refers to: its own actor for an async-proxy
    /// event, and every async component of the registered clock for a
    /// vector-timestamped one, whose defining component may be an async
    /// actor's rather than a warp's.
    fn mark_witness(&mut self, witness: &ClockedWitness, registry: &OperationRegistry) {
        match witness.timestamp.resolve(registry) {
            (TimestampActor::Async(index), _) => self.mark(index),
            (TimestampActor::RegisteredVector(index), _) => {
                let clock = registry.registered_vector_timestamp_clock(index);
                for (index, _) in clock.async_components.nonzero() {
                    self.mark(index);
                }
            }
            (TimestampActor::Warp(_), _) => {}
        }
    }
}

fn witness_is_globally_observed(
    witness: &ClockedWitness,
    registry: &OperationRegistry,
    observed_frontier: &RaceVectorClock,
    proxy_sensitive: bool,
) -> bool {
    // This floor contains warp/async clocks, not a meet of all lane clocks.
    // Lane-stamped evidence can only be subsumed by a proven lane observation.
    if witness.timestamp.lane_stamp(registry).is_some() {
        return false;
    }
    if !witness.timestamp.observed_by(observed_frontier, registry) {
        return false;
    }
    if !proxy_sensitive {
        return true;
    }
    let (prior_proxy, prior_domain) = witness.witness.proxy_and_domain(registry);
    let current_proxy = match prior_proxy {
        MemoryProxy::Generic => MemoryProxy::Async,
        MemoryProxy::Async => MemoryProxy::Generic,
        MemoryProxy::Mmio | MemoryProxy::MulticastAlias => return false,
    };
    let current_domains: &[ProxyMemoryDomain] = match prior_domain {
        ProxyMemoryDomain::Global => &[ProxyMemoryDomain::Global],
        ProxyMemoryDomain::SharedCta | ProxyMemoryDomain::SharedCluster => &[
            ProxyMemoryDomain::SharedCta,
            ProxyMemoryDomain::SharedCluster,
        ],
        // No modeled async-proxy access uses TMEM/local/register space.
        ProxyMemoryDomain::Other => return true,
    };
    observed_frontier.proxy_lane_mask().into_iter().all(|lane| {
        current_domains.iter().copied().all(|current_domain| {
            observed_frontier.proxy_bridge_observes(
                prior_proxy,
                current_proxy,
                prior_domain,
                current_domain,
                witness.timestamp,
                registry,
                lane,
                (
                    witness.witness.global_warp_id(registry),
                    witness.witness.resolved_lane(registry),
                ),
            )
        })
    })
}

/// Keep what a retired generic-proxy witness asserts about cross-proxy
/// ordering: its exact timestamp, in the slots of its kind and domain.
fn note_retired_generic(
    witness: &ClockedWitness,
    registry: &OperationRegistry,
    retired: &mut RetiredProxyFrontiers,
    warp_count: usize,
    async_registry: &Arc<AsyncClockRegistry>,
) -> Result<(), RaceShadowError> {
    let (proxy, domain) = witness.witness.proxy_and_domain(registry);
    if proxy != MemoryProxy::Generic {
        return Ok(());
    }
    retired.raise_timestamp(
        witness.witness.resolved_kind(registry),
        domain,
        witness.timestamp,
        registry,
        warp_count,
        async_registry,
        (
            witness.witness.global_warp_id(registry),
            witness.witness.resolved_lane(registry),
        ),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ShadowSegment {
    start: usize,
    end: usize,
    state: ShadowState,
}

type ShadowSegments = TransactionalIntervalMap<ShadowSegment>;

impl ShadowSegment {
    fn new(start: usize, end: usize, state: ShadowState) -> Self {
        debug_assert!(start < end);
        Self { start, end, state }
    }
}

/// Prevalidated all-lane shadow update, committed after numerical success.
pub struct RaceBatchValidation {
    warp_id: usize,
    event_clock: Arc<RaceVectorClock>,
    allocation_updates: BTreeMap<AllocationKey, AllocationPatch>,
    proxy_sensitive_allocations: Option<Box<BTreeSet<AllocationKey>>>,
    review_findings: Vec<PhysicalRaceFinding>,
}

/// Transactional synchronous accesses plus the async actor they launch.
///
/// Async payload issue can contain several exact physical batches.  Validate
/// their sequential warp events into one sparse overlay, then publish both the
/// final warp clock and the forked async actor only after numeric issue
/// succeeds.  This preserves every exact batch timestamp without cloning and
/// copy-on-write mutating the complete committed shadow.
pub(crate) struct RaceAsyncIssueValidation {
    warp_validation: Option<RaceBatchValidation>,
    token: AsyncTokenId,
    token_clock: RaceVectorClock,
}

/// Transactional token for the allocation-free synchronous access path.
///
/// Exact, non-overlapping lane geometry can be approved read-only and then
/// committed in place after numeric success. Irregular geometry retains the
/// general sparse overlay so the optimization never changes Racecheck
/// semantics.
pub(crate) enum RaceCompactBatchValidation {
    Direct(RaceDirectBatchValidation),
    Sparse(RaceBatchValidation),
}

pub(crate) struct RaceDirectBatchValidation {
    warp_id: usize,
    event_clock: Arc<RaceVectorClock>,
    operation: RegisteredOperation,
    allocation: AllocationKey,
    proxy_sensitive: bool,
    defer_clock_commit: bool,
    review_findings: Vec<PhysicalRaceFinding>,
}

#[derive(Clone, Copy)]
pub(crate) struct CompactDirectGeometry {
    allocation: AllocationKey,
    byte_offset: usize,
    byte_end: usize,
    contiguous_coverage: bool,
    duplicate_lane_order: [u8; WARP_SIZE],
    duplicate_lane_count: u8,
}

impl CompactDirectGeometry {
    pub(crate) const fn allocation(self) -> PhysicalAllocationId {
        self.allocation.allocation
    }

    pub(crate) const fn byte_range(self) -> (usize, usize) {
        (self.byte_offset, self.byte_end)
    }

    pub(crate) const fn has_contiguous_coverage(self) -> bool {
        self.contiguous_coverage
    }

    pub(crate) fn duplicate_lane_order(&self) -> &[u8] {
        &self.duplicate_lane_order[..usize::from(self.duplicate_lane_count)]
    }
}

/// In-place direct-access state for one uninterrupted warp execution segment.
///
/// Direct compact accesses reach Racecheck only after their numeric mutation
/// succeeds. The executor cannot expose another warp in this scheduling domain
/// without first entering a Racecheck boundary, so these accesses can update
/// the committed interval geometry immediately. The clock itself remains
/// pending until the segment boundary.
pub(crate) struct RaceDirectSegment {
    warp_id: usize,
    event_clock: Arc<RaceVectorClock>,
    proxy_sensitive_allocations: BTreeSet<AllocationKey>,
    review_findings: Vec<PhysicalRaceFinding>,
}

/// Transactional shadow update for one or more batches sharing an explicit
/// async-actor timestamp.  Keeping this as a sparse validation token avoids
/// cloning the complete byte shadow merely to bracket a deferred completion.
pub(crate) struct RaceClockedBatchValidation {
    allocation_updates: BTreeMap<AllocationKey, AllocationPatch>,
    proxy_sensitive_allocations: BTreeSet<AllocationKey>,
    review_findings: Vec<PhysicalRaceFinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AllocationPatch {
    base_len: usize,
    base_geometry_exact: bool,
    // Compact physical batches normally visit disjoint lane spans in address
    // order. Keep those staged states in a flat vector: the byte occupancy
    // index used by the general overlay costs more to build than it saves for
    // an ephemeral, at-most-one-warp transaction.
    ordered_segments: Option<Vec<ShadowSegment>>,
    // Sparse transactional overlay. These segments contain the complete final
    // state only for byte ranges touched by the current lane batch; gaps fall
    // through to the committed allocation.
    segments: TransactionalIntervalMap<ShadowSegment>,
}

impl AllocationPatch {
    fn new(existing: &ShadowSegments) -> Self {
        Self {
            base_len: existing.len(),
            base_geometry_exact: true,
            ordered_segments: Some(Vec::new()),
            segments: TransactionalIntervalMap::default(),
        }
    }

    #[cfg(test)]
    fn staged_segment_count(&self) -> usize {
        self.ordered_segments
            .as_ref()
            .map_or_else(|| self.segments.len(), Vec::len)
    }

    fn apply(
        &mut self,
        registry: &OperationRegistry,
        existing: &ShadowSegments,
        current: PhysicalRaceWitnessRef,
        event_clock: &Arc<RaceVectorClock>,
        timestamp: &RaceEventTimestamp,
        lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<(), RaceShadowError> {
        let _profile = ProfileTimer::new(ProfileKind::RaceSparsePatchApply);
        debug_assert_eq!(existing.len(), self.base_len);
        let span = current.span();
        let start = span.byte_offset();
        let end = span.byte_end();
        if let Some(ordered) = &mut self.ordered_segments {
            if let Some(segment) = ordered
                .last_mut()
                .filter(|segment| segment.start == start && segment.end == end)
            {
                if allow_same_operation
                    && segment
                        .state
                        .records_same_operation_kind(registry, &current)
                {
                    return Ok(());
                }
                segment.state.check_and_record(
                    registry,
                    event_clock,
                    timestamp,
                    current,
                    span,
                    lane_order,
                    allow_same_operation,
                    review_findings,
                )?;
                return Ok(());
            }
            if ordered.last().is_none_or(|segment| segment.end <= start) {
                let (base_state, base_end) = segment_map_state_until(existing, start, end);
                if base_end == end {
                    let _profile = ProfileTimer::new(ProfileKind::RaceSparsePatchBaseExact);
                    self.base_geometry_exact &= existing
                        .get(start)
                        .is_some_and(|segment| segment.end == end);
                    let mut state = base_state.cloned().unwrap_or_default();
                    state.check_and_record(
                        registry,
                        event_clock,
                        timestamp,
                        current,
                        span,
                        lane_order,
                        allow_same_operation,
                        review_findings,
                    )?;
                    ordered.push(ShadowSegment::new(start, end, state));
                    return Ok(());
                }
            }

            for segment in self
                .ordered_segments
                .take()
                .expect("ordered patch exists until promotion")
            {
                self.segments
                    .insert_prepared(segment.start, segment.end, segment);
            }
        }
        // Repeated tile loops overwhelmingly revisit the exact interval shape
        // established by their first iteration.  Updating that staged segment
        // in place avoids both occupancy-table preparation and rebuilding the
        // sparse overlay for every scalar access in the loop.
        if let Some(segment) = self
            .segments
            .get_mut(start)
            .filter(|segment| segment.end == end)
        {
            if allow_same_operation
                && segment
                    .state
                    .records_same_operation_kind(registry, &current)
            {
                return Ok(());
            }
            segment.state.check_and_record(
                registry,
                event_clock,
                timestamp,
                current,
                span,
                lane_order,
                allow_same_operation,
                review_findings,
            )?;
            return Ok(());
        }
        self.segments.prepare_span(start, end);
        let (patch_state, patch_end) = patch_segment_state_until(&self.segments, start, end);
        if patch_state.is_none() && patch_end == end {
            let (base_state, base_end) = segment_map_state_until(existing, start, end);
            if base_end == end {
                let _profile = ProfileTimer::new(ProfileKind::RaceSparsePatchBaseExact);
                self.base_geometry_exact &= existing
                    .get(start)
                    .is_some_and(|segment| segment.end == end);
                let mut state = base_state.cloned().unwrap_or_default();
                state.check_and_record(
                    registry,
                    event_clock,
                    timestamp,
                    current,
                    span,
                    lane_order,
                    allow_same_operation,
                    review_findings,
                )?;
                self.segments.insert_after_prepare(
                    start,
                    end,
                    ShadowSegment::new(start, end, state),
                );
                return Ok(());
            }
        }
        let _profile = ProfileTimer::new(ProfileKind::RaceSparsePatchGeneral);
        self.base_geometry_exact = false;
        let mut cursor = start;
        let mut replacement = Vec::new();
        while cursor < end {
            let (effective_state, next) = self.effective_state_until(existing, cursor, end);
            debug_assert!(next > cursor, "race-shadow overlay scan must advance");
            let overlap = PhysicalByteSpan::new(span.allocation(), cursor, next - cursor)
                .expect("subspan of a valid physical span is valid");
            let mut state = effective_state.cloned().unwrap_or_default();
            state.check_and_record(
                registry,
                event_clock,
                timestamp,
                current.clone(),
                overlap,
                lane_order,
                allow_same_operation,
                review_findings,
            )?;
            replacement.push(ShadowSegment::new(cursor, next, state));
            cursor = next;
        }
        merge_adjacent_segments(&mut replacement);
        if let Err(replacement) = self.segments.try_replace_range(
            start,
            end,
            replacement,
            |segment| segment.start,
            |segment| segment.end,
            |segment, split_start, split_end| {
                ShadowSegment::new(split_start, split_end, segment.state.clone())
            },
        ) {
            replace_patch_segment_range(self.segments.general_mut(), start, end, replacement);
        }
        Ok(())
    }

    fn effective_state_until<'a>(
        &'a self,
        existing: &'a ShadowSegments,
        byte_offset: usize,
        byte_end: usize,
    ) -> (Option<&'a ShadowState>, usize) {
        let (patch_state, patch_end) =
            patch_segment_state_until(&self.segments, byte_offset, byte_end);
        if patch_state.is_some() {
            return (patch_state, patch_end);
        }
        let (base_state, base_end) = segment_map_state_until(existing, byte_offset, byte_end);
        (base_state, base_end.min(patch_end))
    }

    fn commit_into(self, existing: &mut ShadowSegments) {
        debug_assert_eq!(existing.len(), self.base_len);
        if let Some(mut segments) = self.ordered_segments {
            debug_assert!(self.segments.is_empty());
            if self.base_geometry_exact {
                let _profile = ProfileTimer::new(ProfileKind::RaceCommitOrderedExact);
                for replacement in segments {
                    existing
                        .get_mut(replacement.start)
                        .expect("exact base geometry remains present until commit")
                        .state = replacement.state;
                }
                return;
            }
            if segments.len() == 1 {
                let _profile = ProfileTimer::new(ProfileKind::RaceCommitOrderedSingle);
                let replacement = segments
                    .pop()
                    .expect("a single-segment patch has one replacement");
                replace_segment_map_range_after_exact_miss(
                    existing,
                    replacement.start,
                    replacement.end,
                    vec![replacement],
                );
                return;
            }
            if segments.windows(2).all(|pair| pair[0].end == pair[1].start) {
                let _profile = ProfileTimer::new(ProfileKind::RaceCommitOrderedContiguous);
                let start = segments
                    .first()
                    .expect("a nonempty ordered patch has a first segment")
                    .start;
                let end = segments
                    .last()
                    .expect("a nonempty ordered patch has a last segment")
                    .end;
                replace_segment_map_range_after_exact_miss(existing, start, end, segments);
                return;
            }
            let _profile = ProfileTimer::new(ProfileKind::RaceCommitOrderedGroups);
            let mut group_end = segments.len();
            while group_end > 0 {
                let mut group_start = group_end - 1;
                while group_start > 0
                    && segments[group_start - 1].end == segments[group_start].start
                {
                    group_start -= 1;
                }
                let replacement = segments.split_off(group_start);
                let start = replacement
                    .first()
                    .expect("a nonempty patch group has a first segment")
                    .start;
                let end = replacement
                    .last()
                    .expect("a nonempty patch group has a last segment")
                    .end;
                replace_segment_map_range_after_exact_miss(existing, start, end, replacement);
                group_end = group_start;
            }
            return;
        }
        if self.base_geometry_exact {
            for replacement in self.segments.into_values() {
                existing
                    .get_mut(replacement.start)
                    .expect("exact base geometry remains present until commit")
                    .state = replacement.state;
            }
            return;
        }
        if self.segments.len() == 1 {
            let replacement = self
                .segments
                .into_sorted_values()
                .pop()
                .expect("a single-segment patch has one replacement");
            replace_segment_map_range(
                existing,
                replacement.start,
                replacement.end,
                vec![replacement],
            );
            return;
        }
        let mut segments = self.segments.into_sorted_values();
        let mut group_end = segments.len();
        while group_end > 0 {
            let mut group_start = group_end - 1;
            while group_start > 0 && segments[group_start - 1].end == segments[group_start].start {
                group_start -= 1;
            }
            let replacement = segments.split_off(group_start);
            let start = replacement
                .first()
                .expect("a nonempty patch group has a first segment")
                .start;
            let end = replacement
                .last()
                .expect("a nonempty patch group has a last segment")
                .end;
            replace_segment_map_range(existing, start, end, replacement);
            group_end = group_start;
        }
    }
}

fn patch_segment_state_until(
    segments: &TransactionalIntervalMap<ShadowSegment>,
    byte_offset: usize,
    byte_end: usize,
) -> (Option<&ShadowState>, usize) {
    let (segment, next) = segments.state_until(byte_offset, byte_end, |segment| segment.end);
    (segment.map(|segment| &segment.state), next)
}

fn replace_patch_segment_range(
    segments: &mut BTreeMap<usize, ShadowSegment>,
    start: usize,
    end: usize,
    replacement: Vec<ShadowSegment>,
) {
    debug_assert!(start < end);
    debug_assert_eq!(
        replacement.first().map(|segment| segment.start),
        Some(start)
    );
    debug_assert_eq!(replacement.last().map(|segment| segment.end), Some(end));

    let mut overlapping = Vec::new();
    if let Some((&key, segment)) = segments.range(..=start).next_back() {
        if segment.end > start {
            overlapping.push(key);
        }
    }
    overlapping.extend(segments.range(start..end).map(|(&key, _)| key));
    overlapping.sort_unstable();
    overlapping.dedup();

    let left = overlapping
        .first()
        .and_then(|key| segments.get(key))
        .filter(|segment| segment.start < start)
        .map(|segment| ShadowSegment::new(segment.start, start, segment.state.clone()));
    let right = overlapping
        .last()
        .and_then(|key| segments.get(key))
        .filter(|segment| segment.end > end)
        .map(|segment| ShadowSegment::new(end, segment.end, segment.state.clone()));

    for key in overlapping {
        segments.remove(&key);
    }
    if let Some(segment) = left {
        segments.insert(segment.start, segment);
    }
    for segment in replacement {
        segments.insert(segment.start, segment);
    }
    if let Some(segment) = right {
        segments.insert(segment.start, segment);
    }
}

fn segment_map_state_until(
    segments: &ShadowSegments,
    byte_offset: usize,
    byte_end: usize,
) -> (Option<&ShadowState>, usize) {
    let (segment, next) = segments.state_until(byte_offset, byte_end, |segment| segment.end);
    (segment.map(|segment| &segment.state), next)
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn validate_and_record_direct_lane_nonexact(
    segments: &mut ShadowSegments,
    registry: &OperationRegistry,
    event_clock: &RaceVectorClock,
    timestamp_encoder: DirectTimestampEncoder,
    operation: RegisteredOperation,
    actor: DirectAccessActor,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy_access: ProxyAccessClass,
    span: PhysicalByteSpan,
    lane: usize,
    lane_order: &RaceLaneOrder<'_>,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<(), RaceShadowError> {
    let start = span.byte_offset();
    let end = span.byte_end();
    if kind.writes() {
        let current = PhysicalRaceWitnessRef::from_parts(
            operation,
            lane,
            kind,
            space,
            proxy_access.proxy(),
            proxy_access.domain(),
            span,
        );
        let mut cursor = start;
        while cursor < end {
            let (existing, next) = segment_map_state_until(segments, cursor, end);
            debug_assert!(next > cursor, "race-shadow validation scan must advance");
            if let Some(state) = existing {
                let state_span = PhysicalByteSpan::new(span.allocation(), cursor, next - cursor)
                    .expect("subspan of a valid physical span is valid");
                if space != PhysicalAccessSpace::Shared
                    || !state.direct_shared_access_is_ordered(
                        registry,
                        event_clock,
                        operation,
                        actor,
                        kind,
                        proxy_access,
                        lane,
                        lane_order,
                    )
                {
                    state.validate_access(
                        registry,
                        event_clock,
                        timestamp_encoder.event_timestamp(),
                        &current,
                        state_span,
                        Some(lane_order),
                        false,
                        review_findings,
                    )?;
                }
            }
            cursor = next;
        }

        // A validated write replaces every prior writer and clears every read
        // over its complete span. Its final state is therefore uniform even
        // when historical access widths split the committed shadow into many
        // fragments. Record the witness once and collapse those boundaries.
        let mut state = ShadowState::default();
        state.record_unique_direct_lane(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            kind,
            space,
            proxy_access,
            span,
            lane,
            lane_order,
        );
        replace_segment_map_range(
            segments,
            start,
            end,
            vec![ShadowSegment::new(start, end, state)],
        );
        return Ok(());
    }

    let mut cursor = start;
    let mut replacement = Vec::new();
    while cursor < end {
        let (existing, next) = segment_map_state_until(segments, cursor, end);
        debug_assert!(next > cursor, "race-shadow committed scan must advance");
        let state_span = PhysicalByteSpan::new(span.allocation(), cursor, next - cursor)
            .expect("subspan of a valid physical span is valid");
        let mut state = existing.cloned().unwrap_or_default();
        state.validate_and_record_direct_lane(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            actor,
            kind,
            space,
            proxy_access,
            span,
            state_span,
            lane,
            lane_order,
            review_findings,
        )?;
        replacement.push(ShadowSegment::new(cursor, next, state));
        cursor = next;
    }
    replace_segment_map_range(segments, start, end, replacement);
    Ok(())
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn validate_and_record_direct_lane_group_nonexact(
    segments: &mut ShadowSegments,
    registry: &OperationRegistry,
    event_clock: &RaceVectorClock,
    timestamp_encoder: DirectTimestampEncoder,
    operation: RegisteredOperation,
    actor: DirectAccessActor,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    proxy_access: ProxyAccessClass,
    span: PhysicalByteSpan,
    lanes: &[u8],
    lane_order: &RaceLaneOrder<'_>,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<(), RaceShadowError> {
    let start = span.byte_offset();
    let end = span.byte_end();
    if kind.writes() {
        let mut cursor = start;
        while cursor < end {
            let (existing, next) = segment_map_state_until(segments, cursor, end);
            debug_assert!(next > cursor, "race-shadow validation scan must advance");
            if let Some(state) = existing {
                let state_span = PhysicalByteSpan::new(span.allocation(), cursor, next - cursor)
                    .expect("subspan of a valid physical span is valid");
                // Every aliased lane observes the same pre-write state. Finish
                // all validation before the one uniform replacement so an
                // earlier lane cannot hide a later lane's conflict.
                for &lane in lanes {
                    let lane = usize::from(lane);
                    if space == PhysicalAccessSpace::Shared
                        && state.direct_shared_access_is_ordered(
                            registry,
                            event_clock,
                            operation,
                            actor,
                            kind,
                            proxy_access,
                            lane,
                            lane_order,
                        )
                    {
                        continue;
                    }
                    let current = PhysicalRaceWitnessRef::from_parts(
                        operation,
                        lane,
                        kind,
                        space,
                        proxy_access.proxy(),
                        proxy_access.domain(),
                        span,
                    );
                    state.validate_access(
                        registry,
                        event_clock,
                        timestamp_encoder.event_timestamp(),
                        &current,
                        state_span,
                        Some(lane_order),
                        false,
                        review_findings,
                    )?;
                }
            }
            cursor = next;
        }

        let mut state = ShadowState::default();
        state.record_direct_lane_group(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            kind,
            space,
            proxy_access,
            span,
            lanes,
            lane_order,
        );
        replace_segment_map_range(
            segments,
            start,
            end,
            vec![ShadowSegment::new(start, end, state)],
        );
        return Ok(());
    }

    let mut cursor = start;
    let mut replacement = Vec::new();
    while cursor < end {
        let (existing, next) = segment_map_state_until(segments, cursor, end);
        debug_assert!(next > cursor, "race-shadow committed scan must advance");
        let state_span = PhysicalByteSpan::new(span.allocation(), cursor, next - cursor)
            .expect("subspan of a valid physical span is valid");
        let mut state = existing.cloned().unwrap_or_default();
        state.validate_and_record_direct_lane_group(
            registry,
            event_clock,
            timestamp_encoder,
            operation,
            actor,
            kind,
            space,
            proxy_access,
            span,
            state_span,
            lanes,
            lane_order,
            review_findings,
        )?;
        replacement.push(ShadowSegment::new(cursor, next, state));
        cursor = next;
    }
    replace_segment_map_range(segments, start, end, replacement);
    Ok(())
}

fn stage_allocation_patches(
    registry: &OperationRegistry,
    allocations: &BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    batch: &PhysicalAccessBatch,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
    lane_order: Option<&RaceLaneOrder<'_>>,
    allow_same_operation: bool,
    tmem_load_register_source: bool,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<BTreeMap<AllocationKey, AllocationPatch>, RaceShadowError> {
    let mut patches = BTreeMap::new();
    stage_batch_into_allocation_patches(
        registry,
        allocations,
        &mut patches,
        batch,
        event_clock,
        timestamp,
        lane_order,
        allow_same_operation,
        tmem_load_register_source,
        review_findings,
    )?;
    Ok(patches)
}

fn stage_batch_into_allocation_patches(
    registry: &OperationRegistry,
    allocations: &BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    patches: &mut BTreeMap<AllocationKey, AllocationPatch>,
    batch: &PhysicalAccessBatch,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
    lane_order: Option<&RaceLaneOrder<'_>>,
    allow_same_operation: bool,
    tmem_load_register_source: bool,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<(), RaceShadowError> {
    let descriptor = batch.descriptor();
    if !tracks_race_conflicts(descriptor.space()) {
        return Ok(());
    }
    let operation = if tmem_load_register_source {
        registry.register_tmem_load_register_source(batch.operation().id())
    } else {
        registry.register(batch.operation().id())
    };
    let kind = descriptor.kind();
    let space = descriptor.space();
    let proxy = descriptor.memory_semantics().proxy();
    let proxy_domain = descriptor.proxy_memory_domain();
    let empty_segments = ShadowSegments::default();
    if let [lane] = batch.lanes() {
        if let [span] = lane.footprint().spans() {
            let current = PhysicalRaceWitnessRef::from_lane(
                operation,
                lane,
                kind,
                space,
                proxy,
                proxy_domain,
                *span,
            )
            .with_memory_semantics(descriptor.memory_semantics());
            let key = AllocationKey {
                space,
                allocation: span.allocation(),
            };
            let existing = allocations
                .get(&key)
                .map(Arc::as_ref)
                .unwrap_or(&empty_segments);
            patches
                .entry(key)
                .or_insert_with(|| AllocationPatch::new(existing))
                .apply(
                    registry,
                    existing,
                    current,
                    event_clock,
                    timestamp,
                    lane_order,
                    allow_same_operation,
                    review_findings,
                )?;
            return Ok(());
        }
    }
    let mut spans = batch
        .lanes()
        .iter()
        .flat_map(|lane| lane.footprint().spans());
    let common_allocation = spans
        .next()
        .map(|span| span.allocation())
        .filter(|allocation| spans.all(|span| span.allocation() == *allocation));
    if let Some(allocation) = common_allocation {
        let key = AllocationKey { space, allocation };
        let existing = allocations
            .get(&key)
            .map(Arc::as_ref)
            .unwrap_or(&empty_segments);
        let patch = patches
            .entry(key)
            .or_insert_with(|| AllocationPatch::new(existing));
        for lane in batch.lanes() {
            for span in lane.footprint().spans().iter().copied() {
                let witness = PhysicalRaceWitnessRef::from_lane(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    span,
                )
                .with_memory_semantics(descriptor.memory_semantics());
                patch.apply(
                    registry,
                    existing,
                    witness,
                    event_clock,
                    timestamp,
                    lane_order,
                    allow_same_operation,
                    review_findings,
                )?;
            }
        }
        return Ok(());
    }
    for lane in batch.lanes() {
        for span in lane.footprint().spans().iter().copied() {
            let witness = PhysicalRaceWitnessRef::from_lane(
                operation,
                lane,
                kind,
                space,
                proxy,
                proxy_domain,
                span,
            )
            .with_memory_semantics(descriptor.memory_semantics());
            let key = AllocationKey {
                space,
                allocation: witness.span().allocation(),
            };
            let existing = allocations
                .get(&key)
                .map(Arc::as_ref)
                .unwrap_or(&empty_segments);
            patches
                .entry(key)
                .or_insert_with(|| AllocationPatch::new(existing))
                .apply(
                    registry,
                    existing,
                    witness,
                    event_clock,
                    timestamp,
                    lane_order,
                    allow_same_operation,
                    review_findings,
                )?;
        }
    }
    Ok(())
}

fn stage_compact_batch_into_allocation_patches(
    registry: &OperationRegistry,
    allocations: &BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    patches: &mut BTreeMap<AllocationKey, AllocationPatch>,
    batch: &CompactPhysicalAccessBatch<'_>,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
    lane_order: &RaceLaneOrder<'_>,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<(), RaceShadowError> {
    let descriptor = batch.descriptor();
    if !tracks_race_conflicts(descriptor.space()) {
        return Ok(());
    }
    let operation = registry.register(batch.operation().id());
    let kind = descriptor.kind();
    let space = descriptor.space();
    let proxy = descriptor.memory_semantics().proxy();
    let proxy_domain = descriptor.proxy_memory_domain();
    let empty_segments = ShadowSegments::default();
    let mut spans = batch.lane_spans();
    let common_allocation = spans
        .next()
        .map(|(_, span)| span.allocation())
        .filter(|allocation| spans.all(|(_, span)| span.allocation() == *allocation));
    if let Some(allocation) = common_allocation {
        let key = AllocationKey { space, allocation };
        let existing = allocations
            .get(&key)
            .map(Arc::as_ref)
            .unwrap_or(&empty_segments);
        let patch = patches
            .entry(key)
            .or_insert_with(|| AllocationPatch::new(existing));
        for (lane, span) in batch.lane_spans() {
            patch.apply(
                registry,
                existing,
                PhysicalRaceWitnessRef::from_parts(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    span,
                ),
                event_clock,
                timestamp,
                Some(lane_order),
                false,
                review_findings,
            )?;
        }
        return Ok(());
    }

    for (lane, span) in batch.lane_spans() {
        let key = AllocationKey {
            space,
            allocation: span.allocation(),
        };
        let existing = allocations
            .get(&key)
            .map(Arc::as_ref)
            .unwrap_or(&empty_segments);
        patches
            .entry(key)
            .or_insert_with(|| AllocationPatch::new(existing))
            .apply(
                registry,
                existing,
                PhysicalRaceWitnessRef::from_parts(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    span,
                ),
                event_clock,
                timestamp,
                Some(lane_order),
                false,
                review_findings,
            )?;
    }
    Ok(())
}

fn record_validated_clocked_witness(
    registry: &OperationRegistry,
    segments: &mut ShadowSegments,
    current: PhysicalRaceWitnessRef,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
) {
    let span = current.span();
    let start = span.byte_offset();
    let end = span.byte_end();
    // Validation has already proved the complete span conflict-free. A write
    // replaces every prior writer and clears every reader in that span, so its
    // final state is uniform regardless of the committed interval geometry.
    // Publish one exact interval instead of rescanning and preserving thousands
    // of stale scalar fragments beneath a large TCGEN write.
    if current.kind().writes() {
        let mut state = ShadowState::default();
        state.record_validated(registry, event_clock, timestamp, current, None);
        replace_segment_map_range(
            segments,
            start,
            end,
            vec![ShadowSegment::new(start, end, state)],
        );
        return;
    }
    if let Some(segment) = segments.get_mut(start).filter(|segment| segment.end == end) {
        if segment
            .state
            .records_same_operation_kind(registry, &current)
        {
            return;
        }
        segment
            .state
            .record_validated(registry, event_clock, timestamp, current, None);
        return;
    }
    if current.space() == PhysicalAccessSpace::Tmem && end - start == 1 {
        // One byte cannot cross an interval boundary. Keep the exact byte state
        // and reuse the indexed split without building a one-element Vec.
        let mut state = segment_map_state_until(segments, start, end)
            .0
            .cloned()
            .unwrap_or_default();
        if !state.records_same_operation_kind(registry, &current) {
            state.record_validated(registry, event_clock, timestamp, current, None);
        }
        replace_segment_map_one(segments, ShadowSegment::new(start, end, state));
        return;
    }
    if segments.try_for_each_exact_range_mut(
        start,
        end,
        |segment| segment.end,
        |segment| {
            !segment
                .state
                .records_same_operation_kind(registry, &current)
        },
        |segment| {
            segment
                .state
                .record_validated(registry, event_clock, timestamp, current, None);
        },
    ) {
        return;
    }

    let mut cursor = start;
    let mut replacement = Vec::new();
    while cursor < end {
        let (existing, next) = segment_map_state_until(segments, cursor, end);
        debug_assert!(next > cursor, "race-shadow committed scan must advance");
        let mut state = existing.cloned().unwrap_or_default();
        if !state.records_same_operation_kind(registry, &current) {
            state.record_validated(registry, event_clock, timestamp, current, None);
        }
        replacement.push(ShadowSegment::new(cursor, next, state));
        cursor = next;
    }
    replace_segment_map_range(segments, start, end, replacement);
}

fn validate_clocked_witness_in_segments(
    registry: &OperationRegistry,
    segments: &ShadowSegments,
    current: PhysicalRaceWitnessRef,
    event_clock: &RaceVectorClock,
    timestamp: RaceEventTimestamp,
    review_findings: &mut Vec<PhysicalRaceFinding>,
) -> Result<(), RaceShadowError> {
    let span = current.span();
    let mut cursor = span.byte_offset();
    while cursor < span.byte_end() {
        let (state, next) = segment_map_state_until(segments, cursor, span.byte_end());
        debug_assert!(next > cursor, "race-shadow validation scan must advance");
        if let Some(state) = state {
            state.validate_access(
                registry,
                event_clock,
                timestamp,
                &current,
                span,
                None,
                true,
                review_findings,
            )?;
        }
        cursor = next;
    }
    Ok(())
}

/// Online physical-byte shadow memory for one fixed launch topology.
///
/// One memory batch advances its warp clock exactly once. Every active lane is
/// checked against the same pre-commit timestamp and a sparse interval overlay;
/// the real shadow is updated only after the complete batch validates.
#[derive(Clone)]
pub struct RaceShadow {
    global_warp_base: usize,
    warp_clocks: Vec<RaceVectorClock>,
    active_async_actor_clocks: BTreeMap<AsyncTokenId, RaceVectorClock>,
    tcgen_async_indices: BTreeSet<usize>,
    tcgen_completed_epochs: BTreeMap<usize, u64>,
    // The same facts laid out per async-epoch chunk: `(slot in chunk, completed
    // epoch or 0)` so that clearing a forked clock walks only the chunks the
    // clock actually holds, with no tree lookups.
    tcgen_slots_by_chunk: Vec<Vec<(u16, u64)>>,
    async_clock_registry: Arc<AsyncClockRegistry>,
    operation_registry: Arc<OperationRegistry>,
    allocations: BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    /// Tile regions of shared memory whose pieces are tracked as classes of
    /// equal state instead of one segment per piece (see `TileWindow`).
    tile_windows: BTreeMap<AllocationKey, Vec<TileWindow>>,
    tile_windows_enabled: bool,
    proxy_sensitive_allocations: BTreeSet<AllocationKey>,
    retired_generic_history: RetiredGenericHistory,
    /// Allocations that received witnesses checked at an explicit clock —
    /// the only witnesses whose timestamps (async actor or registered
    /// vector) refer to async clock slots. A reclamation pass visits these
    /// alone.
    slot_referencing_allocations: HashSet<AllocationKey>,
    observed_frontier: RaceVectorClock,
    safe_points_since_gc: usize,
    revision: u64,
}

impl RaceShadow {
    pub fn new(warp_count: usize) -> Self {
        Self::for_warp_range(0, warp_count)
    }

    /// Construct a shadow for one contiguous scheduling domain while keeping
    /// externally visible operation IDs in launch-global warp coordinates.
    pub fn for_warp_range(global_warp_base: usize, warp_count: usize) -> Self {
        Self::for_warp_range_with_topology(global_warp_base, warp_count, None)
    }

    pub(crate) fn for_warp_range_with_topology(
        global_warp_base: usize,
        warp_count: usize,
        topology: Option<crate::LaunchTopology>,
    ) -> Self {
        let async_clock_registry = Arc::new(AsyncClockRegistry::default());
        let zero = RaceVectorClock::zero(warp_count, Arc::clone(&async_clock_registry));
        let mut operation_registry = OperationRegistry::for_warp_range(global_warp_base);
        operation_registry.topology = topology;
        Self {
            global_warp_base,
            warp_clocks: vec![zero; warp_count],
            active_async_actor_clocks: BTreeMap::new(),
            tcgen_async_indices: BTreeSet::new(),
            tcgen_completed_epochs: BTreeMap::new(),
            tcgen_slots_by_chunk: Vec::new(),
            async_clock_registry: Arc::clone(&async_clock_registry),
            operation_registry: Arc::new(operation_registry),
            allocations: BTreeMap::new(),
            tile_windows: BTreeMap::new(),
            tile_windows_enabled: true,
            proxy_sensitive_allocations: BTreeSet::new(),
            retired_generic_history: RetiredGenericHistory::default(),
            slot_referencing_allocations: HashSet::new(),
            observed_frontier: RaceVectorClock::zero(warp_count, async_clock_registry),
            safe_points_since_gc: 0,
            revision: 0,
        }
    }

    /// Monotonic identity of the state used to validate physical accesses.
    /// A caller may reuse a validation token only while this value is unchanged.
    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn empty_clock(&self) -> RaceVectorClock {
        RaceVectorClock::zero(self.warp_count(), Arc::clone(&self.async_clock_registry))
    }

    pub(crate) fn tcgen_frontier_from_async_components(
        &mut self,
        frontier: impl IntoIterator<Item = (AsyncTokenId, u64)>,
    ) -> RaceVectorClock {
        let frontier = frontier.into_iter().collect::<Vec<_>>();
        let mut clock = self.empty_clock();
        clock.merge_async_frontier(frontier.iter().cloned());
        for (token, _) in frontier {
            self.register_tcgen_async_index(
                clock
                    .async_actor_index(&token)
                    .expect("an imported TCGEN frontier registers every token"),
            );
        }
        clock
    }

    fn register_tcgen_async_index(&mut self, index: usize) {
        if !self.tcgen_async_indices.insert(index) {
            return;
        }
        let chunk = index / ASYNC_EPOCH_CHUNK;
        if chunk >= self.tcgen_slots_by_chunk.len() {
            self.tcgen_slots_by_chunk.resize_with(chunk + 1, Vec::new);
        }
        let completed = self
            .tcgen_completed_epochs
            .get(&index)
            .copied()
            .unwrap_or(0);
        self.tcgen_slots_by_chunk[chunk].push((
            u16::try_from(index % ASYNC_EPOCH_CHUNK).expect("an async chunk slot fits u16"),
            completed,
        ));
    }

    pub(crate) fn tcgen_async_frontier(&self, clock: &RaceVectorClock) -> Vec<(AsyncTokenId, u64)> {
        debug_assert!(Arc::ptr_eq(
            &self.async_clock_registry,
            &clock.async_registry
        ));
        let registry = self
            .async_clock_registry
            .inner
            .lock()
            .expect("racecheck async-clock registry lock was poisoned");
        let mut frontier = registry
            .handles
            .iter()
            .filter(|(_, index)| self.tcgen_async_indices.contains(index))
            .filter_map(|(token, index)| {
                let (generation, epoch) = unpack_async_epoch(clock.async_component_at(*index));
                (generation == registry.generations[*index] && epoch != 0)
                    .then(|| (token.clone(), epoch))
            })
            .collect::<Vec<_>>();
        frontier.sort_unstable();
        frontier
    }

    pub(crate) fn commit_direct_segment(
        &mut self,
        segment: RaceDirectSegment,
    ) -> Vec<PhysicalRaceFinding> {
        self.warp_clocks[segment.warp_id] = segment.event_clock.as_ref().clone();
        self.commit_proxy_sensitive_allocations(segment.proxy_sensitive_allocations);
        self.retire_reviewed_tmem_load_findings(&segment.review_findings);
        self.note_memory_safe_point();
        self.bump_revision();
        segment.review_findings
    }

    pub(crate) fn extend_batch_validation_with_lane_order(
        &mut self,
        validation: &mut RaceBatchValidation,
        batch: &PhysicalAccessBatch,
        lane_order: &RaceLaneOrder<'_>,
    ) -> Result<(), RaceShadowError> {
        self.extend_batch_validation_with_optional_lane_order(
            validation,
            batch,
            Some(lane_order),
            false,
        )
    }

    fn extend_batch_validation_with_optional_lane_order(
        &mut self,
        validation: &mut RaceBatchValidation,
        batch: &PhysicalAccessBatch,
        lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
    ) -> Result<(), RaceShadowError> {
        self.demote_tile_windows_for_batch(batch);
        self.extend_batch_validation_with_optional_lane_order_unchecked(
            validation,
            batch,
            lane_order,
            allow_same_operation,
        )
    }

    fn extend_batch_validation_with_optional_lane_order_unchecked(
        &self,
        validation: &mut RaceBatchValidation,
        batch: &PhysicalAccessBatch,
        lane_order: Option<&RaceLaneOrder<'_>>,
        allow_same_operation: bool,
    ) -> Result<(), RaceShadowError> {
        let local_warp_id = self.local_warp_id(batch.operation().id().global_warp_id())?;
        if local_warp_id != validation.warp_id {
            return Err(RaceShadowError::InvalidWarp {
                warp_id: batch.operation().id().global_warp_id(),
                warp_count: self.global_warp_base.saturating_add(self.warp_count()),
            });
        }
        let timestamp = RaceEventTimestamp::for_warp(
            &validation.event_clock,
            validation.warp_id,
            &self.operation_registry,
        );
        let proxy_sensitive_allocations =
            self.proxy_allocations_for_batch(batch, &validation.event_clock)?;
        if !proxy_sensitive_allocations.is_empty() {
            validation
                .proxy_sensitive_allocations
                .get_or_insert_with(|| Box::new(BTreeSet::new()))
                .extend(proxy_sensitive_allocations);
        }
        stage_batch_into_allocation_patches(
            &self.operation_registry,
            &self.allocations,
            &mut validation.allocation_updates,
            batch,
            &validation.event_clock,
            &timestamp,
            lane_order,
            allow_same_operation,
            false,
            &mut validation.review_findings,
        )
    }

    pub(crate) fn validate_batch_with_lane_order(
        &mut self,
        batch: &PhysicalAccessBatch,
        lane_order: &RaceLaneOrder<'_>,
    ) -> Result<RaceBatchValidation, RaceShadowError> {
        self.demote_tile_windows_for_batch(batch);
        let warp_id = batch.operation().id().global_warp_id();
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut event_clock = self.warp_clocks[local_warp_id].clone();
        event_clock.tick(local_warp_id)?;
        let event_clock = Arc::new(event_clock);
        let mut validation = RaceBatchValidation {
            warp_id: local_warp_id,
            event_clock,
            allocation_updates: BTreeMap::new(),
            proxy_sensitive_allocations: None,
            review_findings: Vec::new(),
        };
        self.extend_batch_validation_with_lane_order(&mut validation, batch, lane_order)?;
        Ok(validation)
    }

    pub(crate) fn validate_compact_batch_with_lane_order(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
    ) -> Result<RaceBatchValidation, RaceShadowError> {
        self.demote_tile_windows_for_compact_batch(batch);
        let warp_id = batch.operation().id().global_warp_id();
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut event_clock = self.warp_clocks[local_warp_id].clone();
        event_clock.tick(local_warp_id)?;
        let event_clock = Arc::new(event_clock);
        let timestamp =
            RaceEventTimestamp::for_warp(&event_clock, local_warp_id, &self.operation_registry);
        let proxy_sensitive_allocations =
            self.proxy_allocations_for_compact_batch(batch, &event_clock)?;
        let mut allocation_updates = BTreeMap::new();
        let mut review_findings = Vec::new();
        stage_compact_batch_into_allocation_patches(
            &self.operation_registry,
            &self.allocations,
            &mut allocation_updates,
            batch,
            &event_clock,
            &timestamp,
            lane_order,
            &mut review_findings,
        )?;
        Ok(RaceBatchValidation {
            warp_id: local_warp_id,
            event_clock,
            allocation_updates,
            proxy_sensitive_allocations: (!proxy_sensitive_allocations.is_empty())
                .then(|| Box::new(proxy_sensitive_allocations)),
            review_findings,
        })
    }

    pub(crate) fn compact_direct_geometry(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Option<CompactDirectGeometry> {
        let descriptor = batch.descriptor();
        let mut lane_spans = batch.lane_spans();
        let (_, first_span) = lane_spans.next()?;
        let allocation = first_span.allocation();
        let key = AllocationKey {
            space: descriptor.space(),
            allocation,
        };
        let mut range_count = 0_usize;
        let mut monotonic_nonoverlap = true;
        let mut monotonic_contiguous = true;
        let mut previous_end = None;
        let mut byte_offset = first_span.byte_offset();
        let mut byte_end = first_span.byte_end();
        let mut all_spans_identical = true;
        for (_lane, span) in batch.lane_spans() {
            if span.allocation() != allocation {
                profile_count(ProfileKind::RaceSparseAllocationMismatch);
                return None;
            }
            byte_offset = byte_offset.min(span.byte_offset());
            byte_end = byte_end.max(span.byte_end());
            all_spans_identical &= span == first_span;
            monotonic_nonoverlap &= previous_end.is_none_or(|end| end <= span.byte_offset());
            monotonic_contiguous &= previous_end.is_none_or(|end| end == span.byte_offset());
            previous_end = Some(span.byte_end());
            range_count += 1;
        }
        let mut duplicate_lane_order = [0_u8; WARP_SIZE];
        let mut has_duplicate_spans = false;
        let mut contiguous_coverage = monotonic_contiguous;
        if !monotonic_nonoverlap {
            if all_spans_identical {
                has_duplicate_spans = range_count > 1;
                for (destination, (lane, _)) in
                    duplicate_lane_order.iter_mut().zip(batch.lane_spans())
                {
                    *destination = u8::try_from(lane).expect("a compact lane fits in u8");
                }
            } else {
                // The dominant monotonic case needs no lane permutation. Build
                // this comparatively large scratch array only after an
                // overlap or out-of-order span actually requires sorting.
                let mut ranges = [(0_usize, 0_usize, 0_u8); WARP_SIZE];
                for (destination, (lane, span)) in ranges.iter_mut().zip(batch.lane_spans()) {
                    *destination = (
                        span.byte_offset(),
                        span.byte_end(),
                        u8::try_from(lane).expect("a compact lane fits in u8"),
                    );
                }
                ranges[..range_count].sort_unstable();
                let mut covered_end = ranges[0].1;
                for pair in ranges[..range_count].windows(2) {
                    if pair[0].1 <= pair[1].0 {
                        contiguous_coverage &= pair[0].1 == pair[1].0;
                        covered_end = pair[1].1;
                        continue;
                    }
                    if pair[0].0 != pair[1].0 || pair[0].1 != pair[1].1 {
                        profile_count(ProfileKind::RaceSparsePartialOverlap);
                        return None;
                    }
                    has_duplicate_spans = true;
                    covered_end = covered_end.max(pair[1].1);
                }
                contiguous_coverage &= covered_end == byte_end;
                if has_duplicate_spans {
                    for (destination, range) in
                        duplicate_lane_order.iter_mut().zip(&ranges[..range_count])
                    {
                        *destination = range.2;
                    }
                }
            }
            if has_duplicate_spans {
                profile_count(ProfileKind::RaceGeometryDuplicate);
            }
        } else {
            profile_count(ProfileKind::RaceGeometryMonotonic);
        }
        Some(CompactDirectGeometry {
            allocation: key,
            byte_offset,
            byte_end,
            contiguous_coverage,
            duplicate_lane_order,
            duplicate_lane_count: if has_duplicate_spans {
                u8::try_from(range_count).expect("a compact batch has at most one warp of lanes")
            } else {
                0
            },
        })
    }

    /// Approve a compact synchronous batch without publishing its memory
    /// effect. Exact disjoint lane intervals need no heap-backed sparse
    /// overlay: validation reads the committed states, and the token permits
    /// an in-place commit after numeric success. All other geometry uses the
    /// general transactional path.
    pub(crate) fn validate_compact_batch_for_direct_commit(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
        geometry: Option<CompactDirectGeometry>,
    ) -> Result<RaceCompactBatchValidation, RaceShadowError> {
        self.validate_compact_batch_for_optional_segment(batch, lane_order, geometry, None, false)
    }

    pub(crate) fn validate_compact_batch_for_new_direct_segment(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
        geometry: CompactDirectGeometry,
    ) -> Result<RaceCompactBatchValidation, RaceShadowError> {
        self.validate_compact_batch_for_optional_segment(
            batch,
            lane_order,
            Some(geometry),
            None,
            true,
        )
    }

    pub(crate) fn validate_compact_batch_for_direct_segment(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
        geometry: CompactDirectGeometry,
        segment: &RaceDirectSegment,
    ) -> Result<RaceCompactBatchValidation, RaceShadowError> {
        self.validate_compact_batch_for_optional_segment(
            batch,
            lane_order,
            Some(geometry),
            Some(segment),
            true,
        )
    }

    fn validate_compact_batch_for_optional_segment(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
        geometry: Option<CompactDirectGeometry>,
        segment: Option<&RaceDirectSegment>,
        defer_clock_commit: bool,
    ) -> Result<RaceCompactBatchValidation, RaceShadowError> {
        self.demote_tile_windows_for_compact_batch(batch);
        let Some(geometry) = geometry else {
            debug_assert!(segment.is_none());
            return self
                .validate_compact_batch_with_lane_order(batch, lane_order)
                .map(RaceCompactBatchValidation::Sparse);
        };
        let allocation = geometry.allocation;
        let global_warp_id = batch.operation().id().global_warp_id();
        let warp_id = self.local_warp_id(global_warp_id)?;
        let actor = DirectAccessActor {
            global_warp_id,
            local_warp_id: warp_id,
        };
        let event_clock = match segment {
            Some(segment) => {
                if segment.warp_id != warp_id {
                    return Err(RaceShadowError::InvalidWarp {
                        warp_id: global_warp_id,
                        warp_count: self.global_warp_base.saturating_add(self.warp_count()),
                    });
                }
                Arc::clone(&segment.event_clock)
            }
            None => {
                let mut event_clock = self.warp_clocks[warp_id].clone();
                event_clock.tick(warp_id)?;
                Arc::new(event_clock)
            }
        };
        let timestamp =
            RaceEventTimestamp::for_warp(&event_clock, warp_id, &self.operation_registry);
        let descriptor = batch.descriptor();
        self.proxy_allocations_for_compact_batch(batch, &event_clock)?;
        let operation = {
            match Arc::get_mut(&mut self.operation_registry) {
                Some(registry) => registry.register_unique_exclusive(batch.operation().id()),
                None => self
                    .operation_registry
                    .register_unique(batch.operation().id()),
            }
        };
        let kind = descriptor.kind();
        let proxy = descriptor.memory_semantics().proxy();
        let proxy_domain = descriptor.proxy_memory_domain();
        let proxy_sensitive = allocation.space == PhysicalAccessSpace::Shared
            && (proxy == MemoryProxy::Async
                || self.proxy_sensitive_allocations.contains(&allocation)
                || segment.is_some_and(|segment| {
                    segment.proxy_sensitive_allocations.contains(&allocation)
                }));
        let proxy_access = ProxyAccessClass::new(proxy, proxy_domain, proxy_sensitive);
        let mut review_findings = Vec::new();
        {
            if let Some(segments) = self.allocations.get(&allocation).map(Arc::as_ref) {
                for (lane, span) in batch.lane_spans() {
                    let Some(segment) = segments.get(span.byte_offset()) else {
                        continue;
                    };
                    debug_assert_eq!(segment.end, span.byte_end());
                    if allocation.space == PhysicalAccessSpace::Shared
                        && segment.state.direct_shared_access_is_ordered(
                            &self.operation_registry,
                            &event_clock,
                            operation,
                            actor,
                            kind,
                            proxy_access,
                            lane,
                            lane_order,
                        )
                    {
                        continue;
                    }
                    let current = PhysicalRaceWitnessRef::from_parts(
                        operation,
                        lane,
                        kind,
                        allocation.space,
                        proxy,
                        proxy_domain,
                        span,
                    );
                    segment.state.validate_access(
                        &self.operation_registry,
                        &event_clock,
                        timestamp,
                        &current,
                        span,
                        Some(lane_order),
                        false,
                        &mut review_findings,
                    )?;
                }
            }
        }
        Ok(RaceCompactBatchValidation::Direct(
            RaceDirectBatchValidation {
                warp_id,
                event_clock,
                operation,
                allocation,
                proxy_sensitive: descriptor.memory_semantics().proxy() == MemoryProxy::Async
                    && allocation.space == PhysicalAccessSpace::Shared,
                defer_clock_commit,
                review_findings,
            },
        ))
    }

    /// Validate and publish one compact batch after its numeric effect succeeds.
    ///
    /// Direct geometry guarantees that distinct lane-span groups do not
    /// overlap. Exact duplicates are validated as one simultaneous group, so
    /// recording one group cannot affect the validation of a later group in the
    /// same batch. Existing shadow boundaries are handled during this single
    /// traversal instead of being preflighted before numeric execution.
    /// Irregular lane geometry retains the fully transactional two-pass
    /// implementation.
    pub(crate) fn apply_compact_batch_after_numeric(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
        geometry: Option<CompactDirectGeometry>,
        direct_segment: Option<&RaceDirectSegment>,
    ) -> Result<(Option<RaceDirectSegment>, Vec<PhysicalRaceFinding>), RaceShadowError> {
        self.demote_tile_windows_for_compact_batch(batch);
        let Some(geometry) = geometry else {
            profile_count(ProfileKind::RaceSparseGeometry);
            profile_count(if batch.descriptor().kind().writes() {
                ProfileKind::RaceSparseWrite
            } else {
                ProfileKind::RaceSparseRead
            });
            profile_count(match batch.descriptor().space() {
                PhysicalAccessSpace::Global => ProfileKind::RaceSparseGlobal,
                PhysicalAccessSpace::Shared => ProfileKind::RaceSparseShared,
                PhysicalAccessSpace::Tmem => ProfileKind::RaceSparseTmem,
                _ => unreachable!("compact race accesses only track global, shared, or TMEM"),
            });
            let _profile = ProfileTimer::new(ProfileKind::RaceSparseShadow);
            debug_assert!(direct_segment.is_none());
            let validation = {
                let _profile = ProfileTimer::new(ProfileKind::RaceSparseValidate);
                self.validate_compact_batch_for_direct_commit(batch, lane_order, None)?
            };
            let result = {
                let _profile = ProfileTimer::new(ProfileKind::RaceSparseCommit);
                self.commit_compact_validation(validation, batch, lane_order)
            };
            return Ok(result);
        };
        profile_count(ProfileKind::RaceDirectGeometry);
        if geometry.duplicate_lane_count != 0 {
            profile_count(ProfileKind::RaceDuplicateGeometry);
        }

        let allocation = geometry.allocation;
        let global_warp_id = batch.operation().id().global_warp_id();
        let warp_id = self.local_warp_id(global_warp_id)?;
        let actor = DirectAccessActor {
            global_warp_id,
            local_warp_id: warp_id,
        };
        let event_clock = match direct_segment {
            Some(segment) => {
                if segment.warp_id != warp_id {
                    return Err(RaceShadowError::InvalidWarp {
                        warp_id: global_warp_id,
                        warp_count: self.global_warp_base.saturating_add(self.warp_count()),
                    });
                }
                Arc::clone(&segment.event_clock)
            }
            None => {
                let mut event_clock = self.warp_clocks[warp_id].clone();
                event_clock.tick(warp_id)?;
                Arc::new(event_clock)
            }
        };
        let event_timestamp =
            RaceEventTimestamp::for_warp(&event_clock, warp_id, &self.operation_registry);
        let descriptor = batch.descriptor();
        self.proxy_allocations_for_compact_batch(batch, &event_clock)?;
        let kind = descriptor.kind();
        let proxy = descriptor.memory_semantics().proxy();
        let proxy_domain = descriptor.proxy_memory_domain();
        let proxy_sensitive = allocation.space == PhysicalAccessSpace::Shared
            && (proxy == MemoryProxy::Async
                || self.proxy_sensitive_allocations.contains(&allocation)
                || direct_segment.is_some_and(|segment| {
                    segment.proxy_sensitive_allocations.contains(&allocation)
                }));
        let proxy_access = ProxyAccessClass::new(proxy, proxy_domain, proxy_sensitive);
        let operation = {
            let _profile = ProfileTimer::new(ProfileKind::RaceOperationRegister);
            match Arc::get_mut(&mut self.operation_registry) {
                Some(registry) => registry.register_unique_exclusive(batch.operation().id()),
                None => self
                    .operation_registry
                    .register_unique(batch.operation().id()),
            }
        };
        let registry = &self.operation_registry;
        let timestamp_encoder = DirectTimestampEncoder::new(event_timestamp, kind);
        profile_count(if kind.writes() {
            ProfileKind::RaceShadowWrite
        } else {
            ProfileKind::RaceShadowRead
        });
        if allocation.space == PhysicalAccessSpace::Tmem {
            profile_count(ProfileKind::RaceShadowTmem);
        }
        let mut review_findings = Vec::new();
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceShadowSegments);
            let segments = Arc::make_mut(self.allocations.entry(allocation).or_default());
            if geometry.duplicate_lane_count == 0 {
                for (lane, span) in batch.lane_spans() {
                    if let Some(segment) = segments
                        .get_mut(span.byte_offset())
                        .filter(|segment| segment.end == span.byte_end())
                    {
                        profile_count(ProfileKind::RaceDirectSegmentExact);
                        segment.state.validate_and_record_direct_lane(
                            registry,
                            &event_clock,
                            timestamp_encoder,
                            operation,
                            actor,
                            kind,
                            allocation.space,
                            proxy_access,
                            span,
                            span,
                            lane,
                            lane_order,
                            &mut review_findings,
                        )?;
                        continue;
                    }
                    let (existing, next) =
                        segment_map_state_until(segments, span.byte_offset(), span.byte_end());
                    if existing.is_none() && next == span.byte_end() {
                        profile_count(ProfileKind::RaceDirectSegmentEmpty);
                        let mut state = ShadowState::default();
                        state.validate_and_record_direct_lane(
                            registry,
                            &event_clock,
                            timestamp_encoder,
                            operation,
                            actor,
                            kind,
                            allocation.space,
                            proxy_access,
                            span,
                            span,
                            lane,
                            lane_order,
                            &mut review_findings,
                        )?;
                        segments.prepare_span(span.byte_offset(), span.byte_end());
                        segments.insert_after_prepare(
                            span.byte_offset(),
                            span.byte_end(),
                            ShadowSegment::new(span.byte_offset(), span.byte_end(), state),
                        );
                        continue;
                    }
                    profile_count(ProfileKind::RaceDirectSegmentNonexact);
                    validate_and_record_direct_lane_nonexact(
                        segments,
                        registry,
                        &event_clock,
                        timestamp_encoder,
                        operation,
                        actor,
                        kind,
                        allocation.space,
                        proxy_access,
                        span,
                        lane,
                        lane_order,
                        &mut review_findings,
                    )?;
                }
            } else {
                let lane_ordering =
                    &geometry.duplicate_lane_order[..usize::from(geometry.duplicate_lane_count)];
                let mut group_start = 0_usize;
                while group_start < lane_ordering.len() {
                    let first_lane = usize::from(lane_ordering[group_start]);
                    let span = batch
                        .lane_span(first_lane)
                        .expect("direct geometry retains only active lanes");
                    let mut group_end = group_start + 1;
                    while group_end < lane_ordering.len()
                        && batch
                            .lane_span(usize::from(lane_ordering[group_end]))
                            .is_some_and(|candidate| candidate == span)
                    {
                        group_end += 1;
                    }
                    let lanes = &lane_ordering[group_start..group_end];
                    if let Some(segment) = segments
                        .get_mut(span.byte_offset())
                        .filter(|segment| segment.end == span.byte_end())
                    {
                        profile_count(ProfileKind::RaceDirectSegmentExact);
                        segment.state.validate_and_record_direct_lane_group(
                            registry,
                            &event_clock,
                            timestamp_encoder,
                            operation,
                            actor,
                            kind,
                            allocation.space,
                            proxy_access,
                            span,
                            span,
                            lanes,
                            lane_order,
                            &mut review_findings,
                        )?;
                    } else {
                        let (existing, next) =
                            segment_map_state_until(segments, span.byte_offset(), span.byte_end());
                        if existing.is_none() && next == span.byte_end() {
                            profile_count(ProfileKind::RaceDirectSegmentEmpty);
                            let mut state = ShadowState::default();
                            state.validate_and_record_direct_lane_group(
                                registry,
                                &event_clock,
                                timestamp_encoder,
                                operation,
                                actor,
                                kind,
                                allocation.space,
                                proxy_access,
                                span,
                                span,
                                lanes,
                                lane_order,
                                &mut review_findings,
                            )?;
                            segments.prepare_span(span.byte_offset(), span.byte_end());
                            segments.insert_after_prepare(
                                span.byte_offset(),
                                span.byte_end(),
                                ShadowSegment::new(span.byte_offset(), span.byte_end(), state),
                            );
                        } else {
                            profile_count(ProfileKind::RaceDirectSegmentNonexact);
                            validate_and_record_direct_lane_group_nonexact(
                                segments,
                                registry,
                                &event_clock,
                                timestamp_encoder,
                                operation,
                                actor,
                                kind,
                                allocation.space,
                                proxy_access,
                                span,
                                lanes,
                                lane_order,
                                &mut review_findings,
                            )?;
                        }
                    }
                    group_start = group_end;
                }
            }
        }
        if proxy == MemoryProxy::Async && allocation.space == PhysicalAccessSpace::Shared {
            self.commit_proxy_sensitive_allocations([allocation]);
        }
        self.bump_revision();
        self.retire_reviewed_tmem_load_findings(&review_findings);
        let new_segment = direct_segment.is_none().then(|| RaceDirectSegment {
            warp_id,
            event_clock,
            proxy_sensitive_allocations: BTreeSet::new(),
            review_findings: Vec::new(),
        });
        Ok((new_segment, review_findings))
    }

    pub(crate) fn commit_compact_validation(
        &mut self,
        validation: RaceCompactBatchValidation,
        batch: &CompactPhysicalAccessBatch<'_>,
        lane_order: &RaceLaneOrder<'_>,
    ) -> (Option<RaceDirectSegment>, Vec<PhysicalRaceFinding>) {
        self.demote_tile_windows_for_compact_batch(batch);
        let RaceCompactBatchValidation::Direct(validation) = validation else {
            let RaceCompactBatchValidation::Sparse(validation) = validation else {
                unreachable!()
            };
            let review_findings = self.commit_validation(validation);
            return (None, review_findings);
        };
        debug_assert_eq!(
            self.local_warp_id(batch.operation().id().global_warp_id())
                .expect("validated compact warp remains in range"),
            validation.warp_id
        );
        debug_assert_eq!(batch.descriptor().space(), validation.allocation.space);
        debug_assert!(batch
            .lane_spans()
            .all(|(_, span)| span.allocation() == validation.allocation.allocation));
        let timestamp = RaceEventTimestamp::for_warp(
            &validation.event_clock,
            validation.warp_id,
            &self.operation_registry,
        );
        let kind = batch.descriptor().kind();
        let semantics = batch.descriptor().memory_semantics();
        let proxy_domain = batch.descriptor().proxy_memory_domain();
        {
            let segments =
                Arc::make_mut(self.allocations.entry(validation.allocation).or_default());
            for (lane, span) in batch.lane_spans() {
                let current = PhysicalRaceWitnessRef::from_parts(
                    validation.operation,
                    lane,
                    kind,
                    validation.allocation.space,
                    semantics.proxy(),
                    proxy_domain,
                    span,
                );
                if let Some(segment) = segments
                    .get_mut(span.byte_offset())
                    .filter(|segment| segment.end == span.byte_end())
                {
                    segment.state.record_validated(
                        &self.operation_registry,
                        &validation.event_clock,
                        &timestamp,
                        current,
                        Some(lane_order),
                    );
                    continue;
                }
                segments.prepare_span(span.byte_offset(), span.byte_end());
                let mut state = ShadowState::default();
                state.record_validated(
                    &self.operation_registry,
                    &validation.event_clock,
                    &timestamp,
                    current,
                    Some(lane_order),
                );
                segments.insert_after_prepare(
                    span.byte_offset(),
                    span.byte_end(),
                    ShadowSegment::new(span.byte_offset(), span.byte_end(), state),
                );
            }
        }
        if validation.proxy_sensitive {
            self.commit_proxy_sensitive_allocations([validation.allocation]);
        }
        let segment = validation.defer_clock_commit.then(|| RaceDirectSegment {
            warp_id: validation.warp_id,
            event_clock: Arc::clone(&validation.event_clock),
            proxy_sensitive_allocations: BTreeSet::new(),
            review_findings: Vec::new(),
        });
        if validation.defer_clock_commit {
            self.bump_revision();
        } else {
            self.warp_clocks[validation.warp_id] = validation.event_clock.as_ref().clone();
            self.note_memory_safe_point();
            self.bump_revision();
        }
        self.retire_reviewed_tmem_load_findings(&validation.review_findings);
        (segment, validation.review_findings)
    }

    fn retire_reviewed_tmem_load_findings(&mut self, findings: &[PhysicalRaceFinding]) -> usize {
        if !findings.is_empty() {
            self.demote_all_tile_windows();
        }
        let mut operations = findings
            .iter()
            .filter_map(PhysicalRaceFinding::reviewed_tmem_load_handle)
            .collect::<Vec<_>>();
        operations.sort_unstable();
        operations.dedup();
        if operations.is_empty() {
            return 0;
        }

        let mut retired = 0;
        self.allocations.retain(|_, segments| {
            let segments = Arc::make_mut(segments);
            let mut retained = std::mem::take(segments).into_sorted_values();
            for segment in &mut retained {
                retired += segment
                    .state
                    .retire_reviewed_tmem_loads(&self.operation_registry, &operations);
            }
            retained.retain(|segment| !segment.state.is_empty());
            merge_adjacent_segments(&mut retained);
            for segment in retained {
                segments.insert_prepared(segment.start, segment.end, segment);
            }
            !segments.is_empty()
        });
        retired
    }

    fn bump_revision(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("race-shadow revision overflowed");
    }

    pub const fn warp_count(&self) -> usize {
        self.warp_clocks.len()
    }

    pub fn warp_clock(&self, warp_id: usize) -> Option<&RaceVectorClock> {
        self.local_warp_id(warp_id)
            .ok()
            .and_then(|local_warp_id| self.warp_clocks.get(local_warp_id))
    }

    pub(crate) fn memory_publication(
        &self,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<BarrierClockPayload, RaceShadowError> {
        let local_warp = self.local_warp_id(global_warp_id)?;
        Ok(BarrierClockPayload::from_clock_for_mask(
            self.warp_clocks[local_warp].clone(),
            mask,
        ))
    }

    fn local_warp_id(&self, warp_id: usize) -> Result<usize, RaceShadowError> {
        warp_id
            .checked_sub(self.global_warp_base)
            .filter(|local_warp_id| *local_warp_id < self.warp_count())
            .ok_or(RaceShadowError::InvalidWarp {
                warp_id,
                warp_count: self.global_warp_base.saturating_add(self.warp_count()),
            })
    }

    /// Whether an async-proxy access to `allocation` must be checked against
    /// retired generic history: only before the allocation's first
    /// async-proxy access, and only if some history was retired at all.
    /// Memoized per allocation because a batch's spans nearly always share
    /// one allocation while a TMA footprint holds many spans.
    fn retired_history_applies_memo(
        &self,
        allocation: AllocationKey,
        memo: &mut Option<(AllocationKey, bool)>,
    ) -> bool {
        if let Some((known, applies)) = *memo {
            if known == allocation {
                return applies;
            }
        }
        let applies = !self.proxy_sensitive_allocations.contains(&allocation)
            && self.retired_generic_history.applies_to(allocation);
        *memo = Some((allocation, applies));
        applies
    }

    fn validate_retired_proxy_allocation(
        &self,
        descriptor: PhysicalAccessDescriptor,
        allocation: AllocationKey,
        range: (usize, usize),
        current_clock: &RaceVectorClock,
        current_lane: usize,
    ) -> Result<(), RaceShadowError> {
        if descriptor.memory_semantics().proxy() != MemoryProxy::Async
            || descriptor.space() != PhysicalAccessSpace::Shared
            || self.proxy_sensitive_allocations.contains(&allocation)
        {
            return Ok(());
        }
        let current_domain = descriptor.proxy_memory_domain();
        if self.retired_generic_history.has_unordered_conflict(
            allocation,
            descriptor.kind(),
            current_domain,
            range.0,
            range.1,
            current_clock,
            current_lane,
        ) {
            return Err(RaceShadowError::RetiredCrossProxyHistory {
                space: allocation.space,
                allocation: allocation.allocation,
            });
        }
        Ok(())
    }

    fn proxy_allocations_for_batch(
        &self,
        batch: &PhysicalAccessBatch,
        current_clock: &RaceVectorClock,
    ) -> Result<BTreeSet<AllocationKey>, RaceShadowError> {
        let descriptor = batch.descriptor();
        if descriptor.memory_semantics().proxy() != MemoryProxy::Async
            || descriptor.space() != PhysicalAccessSpace::Shared
        {
            return Ok(BTreeSet::new());
        }
        let mut allocations = BTreeSet::new();
        let mut applies = None;
        for lane in batch.lanes() {
            let lane_index = lane.provenance().lane();
            for span in lane.footprint().spans() {
                let allocation = AllocationKey {
                    space: descriptor.space(),
                    allocation: span.allocation(),
                };
                if self.retired_history_applies_memo(allocation, &mut applies) {
                    self.validate_retired_proxy_allocation(
                        descriptor,
                        allocation,
                        (span.byte_offset(), span.byte_end()),
                        current_clock,
                        lane_index,
                    )?;
                }
                if allocations.last() != Some(&allocation) {
                    allocations.insert(allocation);
                }
            }
        }
        Ok(allocations)
    }

    fn proxy_allocations_for_compact_batch(
        &self,
        batch: &CompactPhysicalAccessBatch<'_>,
        current_clock: &RaceVectorClock,
    ) -> Result<BTreeSet<AllocationKey>, RaceShadowError> {
        let descriptor = batch.descriptor();
        if descriptor.memory_semantics().proxy() != MemoryProxy::Async
            || descriptor.space() != PhysicalAccessSpace::Shared
        {
            return Ok(BTreeSet::new());
        }
        let mut allocations = BTreeSet::new();
        let mut applies = None;
        for (lane, span) in batch.lane_spans() {
            let allocation = AllocationKey {
                space: descriptor.space(),
                allocation: span.allocation(),
            };
            if self.retired_history_applies_memo(allocation, &mut applies) {
                self.validate_retired_proxy_allocation(
                    descriptor,
                    allocation,
                    (span.byte_offset(), span.byte_end()),
                    current_clock,
                    lane,
                )?;
            }
            if allocations.last() != Some(&allocation) {
                allocations.insert(allocation);
            }
        }
        Ok(allocations)
    }

    fn commit_proxy_sensitive_allocations(
        &mut self,
        allocations: impl IntoIterator<Item = AllocationKey>,
    ) {
        for allocation in allocations {
            self.proxy_sensitive_allocations.insert(allocation);
            self.retired_generic_history.remove_exact(&allocation);
        }
    }

    pub fn check_batch(&mut self, batch: PhysicalAccessBatch) -> Result<(), RaceShadowError> {
        self.demote_tile_windows_for_batch(&batch);
        let validation = self.validate_batch(&batch)?;
        self.commit_validation(validation);
        Ok(())
    }

    /// Fork an independently advancing async actor from the issuer's current
    /// clock without advancing the issuer warp itself.
    pub fn fork_async_token(
        &mut self,
        warp_id: usize,
        token: &AsyncTokenId,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        self.fork_async_token_for_mask(warp_id, WarpMask::FULL, token)
    }

    pub(crate) fn fork_async_token_for_mask(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        token: &AsyncTokenId,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        self.fork_async_token_after_clock(warp_id, active_mask, token, None, None)
    }

    /// Fork an async actor after an engine-level pipeline predecessor.
    ///
    /// The issuer warp itself is deliberately not advanced: only operations
    /// whose hardware pipeline guarantees execution order may supply an
    /// inherited clock here.
    pub(crate) fn fork_async_token_after_clock(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        token: &AsyncTokenId,
        predecessor: Option<&RaceVectorClock>,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        let clock = self.preview_async_token_after_clock(
            warp_id,
            active_mask,
            token,
            predecessor,
            lane_order,
        )?;
        let local_warp_id = self.local_warp_id(warp_id)?;
        clock.register_async_issue(token, local_warp_id)?;
        self.active_async_actor_clocks
            .insert(token.clone(), clock.clone());
        self.bump_revision();
        Ok(clock)
    }

    /// Fork one TCGEN actor from the issuer's ordinary thread clock without
    /// implicitly inheriting unordered TCGEN work.
    ///
    /// Standard memory-model and barrier ordering still precede the issue.
    /// TCGEN async components that reached the thread through synchronization
    /// are removed unless an architected TCGEN pipeline or an observed
    /// completion frontier supplies them as a predecessor.
    pub(crate) fn fork_tcgen_token_after_clock(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        token: &AsyncTokenId,
        predecessor: Option<&RaceVectorClock>,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        if self.active_async_actor_clocks.contains_key(token) {
            return Err(RaceShadowError::DuplicateAsyncActor {
                token: token.clone(),
            });
        }
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut clock = self.warp_clocks[local_warp_id].clone();
        if let Some(order) = lane_order {
            clock = order.acquired_mask_clock(&clock, active_mask);
        }
        clock.clear_uncompleted_async_components(&self.tcgen_slots_by_chunk);
        if let Some(predecessor) = predecessor {
            clock.merge(predecessor)?;
        }
        clock.restrict_proxy_lanes(active_mask);
        clock.tick_async_issue(token, local_warp_id)?;
        self.register_tcgen_async_index(
            clock
                .async_actor_index(token)
                .expect("a forked TCGEN clock registers its token"),
        );
        self.active_async_actor_clocks
            .insert(token.clone(), clock.clone());
        self.bump_revision();
        Ok(clock)
    }

    /// Compute the exact clock a new async actor would receive without
    /// publishing it. Transactional issue paths use this to validate all
    /// memory effects once before the numeric operation, then publish the same
    /// clock only if the numeric operation succeeds.
    pub(crate) fn preview_async_token_after_clock(
        &self,
        warp_id: usize,
        active_mask: WarpMask,
        token: &AsyncTokenId,
        predecessor: Option<&RaceVectorClock>,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        if self.active_async_actor_clocks.contains_key(token) {
            return Err(RaceShadowError::DuplicateAsyncActor {
                token: token.clone(),
            });
        }
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut clock = self.warp_clocks[local_warp_id].clone();
        if let Some(order) = lane_order {
            clock = order.acquired_mask_clock(&clock, active_mask);
        }
        if let Some(predecessor) = predecessor {
            clock.merge(predecessor)?;
        }
        clock.restrict_proxy_lanes(active_mask);
        clock.tick_async(token)?;
        Ok(clock)
    }

    /// Retire an async actor only after its completion accesses and release
    /// payloads have committed. Completion witnesses retain the token clock,
    /// so their records still require observation by every warp.
    pub fn retire_async_actor(&mut self, token: &AsyncTokenId) -> Result<(), RaceShadowError> {
        self.active_async_actor_clocks
            .remove(token)
            .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                token: token.clone(),
            })?;
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(())
    }

    /// Retire async actors whose remaining same-thread ordering is delegated
    /// to a review finding. Unlike a real wait/completion, this publishes no
    /// clock to the issuing warp or to any other warp.
    pub(crate) fn retire_async_actors_for_review(
        &mut self,
        tokens: &[AsyncTokenId],
    ) -> Result<(), RaceShadowError> {
        for token in tokens {
            if !self.active_async_actor_clocks.contains_key(token) {
                return Err(RaceShadowError::MissingAsyncActor {
                    token: token.clone(),
                });
            }
        }
        for token in tokens {
            self.active_async_actor_clocks
                .remove(token)
                .expect("every reviewed async actor was prevalidated");
        }
        if !tokens.is_empty() {
            self.note_memory_safe_point();
            self.bump_revision();
        }
        Ok(())
    }

    /// Advance one independently scheduled async actor to its next milestone.
    pub fn advance_async_actor(
        &mut self,
        token: &AsyncTokenId,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        let clock = self
            .active_async_actor_clocks
            .get_mut(token)
            .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                token: token.clone(),
            })?;
        clock.tick_async(token)?;
        let clock = clock.clone();
        self.bump_revision();
        Ok(clock)
    }

    pub(crate) fn apply_implicit_async_completion(
        &mut self,
        token: &AsyncTokenId,
        domains: impl IntoIterator<Item = ProxyMemoryDomain>,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        let clock = self
            .active_async_actor_clocks
            .get_mut(token)
            .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                token: token.clone(),
            })?;
        for domain in domains {
            clock.apply_implicit_async_completion(domain)?;
        }
        let clock = clock.clone();
        self.bump_revision();
        Ok(clock)
    }

    /// Compute an async actor's next milestone without publishing it.
    ///
    /// Completion hooks use this to validate all memory effects before the
    /// numeric completion and publish the same clock only after it succeeds.
    pub(crate) fn preview_advance_async_actor(
        &mut self,
        token: &AsyncTokenId,
    ) -> Result<RaceVectorClock, RaceShadowError> {
        let mut clock = self
            .active_async_actor_clocks
            .get(token)
            .cloned()
            .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                token: token.clone(),
            })?;
        clock.tick_async(token)?;
        Ok(clock)
    }

    /// Atomically advance and retire one completed set of async actors.
    ///
    /// TCGEN wait/commit completions have no numeric memory mutation to roll
    /// back. Preview every clock first, then remove all actors and collect the
    /// dominated frontier once. This preserves the all-or-nothing contract
    /// without cloning the complete byte shadow merely to make retirement
    /// transactional.
    pub(crate) fn complete_async_actors(
        &mut self,
        tokens: &[AsyncTokenId],
    ) -> Result<Vec<RaceVectorClock>, RaceShadowError> {
        let mut completed = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            if tokens[..index].contains(token) {
                return Err(RaceShadowError::MissingAsyncActor {
                    token: token.clone(),
                });
            }
            let mut clock = self
                .active_async_actor_clocks
                .get(token)
                .cloned()
                .ok_or_else(|| RaceShadowError::MissingAsyncActor {
                    token: token.clone(),
                })?;
            clock.tick_async(token)?;
            completed.push(clock);
        }
        for (token, clock) in tokens.iter().zip(&completed) {
            let Some(index) = clock.async_actor_index(token) else {
                continue;
            };
            if !self.tcgen_async_indices.contains(&index) {
                continue;
            }
            let completed_epoch = clock.async_component_at(index);
            let known = self
                .tcgen_completed_epochs
                .entry(index)
                .and_modify(|known| *known = (*known).max(completed_epoch))
                .or_insert(completed_epoch);
            let slot = self.tcgen_slots_by_chunk[index / ASYNC_EPOCH_CHUNK]
                .iter_mut()
                .find(|(slot, _)| usize::from(*slot) == index % ASYNC_EPOCH_CHUNK)
                .expect("a completed TCGEN token was registered in its chunk table");
            slot.1 = *known;
        }
        for token in tokens {
            self.active_async_actor_clocks
                .remove(token)
                .expect("every completed async actor was prevalidated");
        }
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(completed)
    }

    /// Commit one memory batch at an explicit async-actor timestamp. Warp
    /// clocks are deliberately unchanged.
    pub fn check_batch_at_clock(
        &mut self,
        batch: &PhysicalAccessBatch,
        event_clock: &RaceVectorClock,
    ) -> Result<Vec<PhysicalRaceFinding>, RaceShadowError> {
        self.check_batch_at_clock_with_policy(batch, event_clock, false, None, false)
    }

    pub(crate) fn check_batch_at_clock_within_operation_for_async_token(
        &mut self,
        batch: &PhysicalAccessBatch,
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
    ) -> Result<Vec<PhysicalRaceFinding>, RaceShadowError> {
        self.check_batch_at_clock_with_policy(batch, event_clock, true, Some(token), false)
    }

    pub(crate) fn check_batch_at_clock_for_async_token(
        &mut self,
        batch: &PhysicalAccessBatch,
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
    ) -> Result<Vec<PhysicalRaceFinding>, RaceShadowError> {
        self.check_batch_at_clock_with_policy(batch, event_clock, false, Some(token), false)
    }

    /// Validate several explicit-clock batches as one sparse transaction.
    ///
    /// Deferred completions are bracketed by `before_completion` and
    /// `after_completion`.  A full `RaceShadow` clone here used to force a
    /// copy-on-write clone of every tracked interval before each completion.
    /// This token retains only touched intervals and is committed after the
    /// numeric completion succeeds.
    pub(crate) fn validate_batches_at_clock<'a>(
        &mut self,
        batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
    ) -> Result<RaceClockedBatchValidation, RaceShadowError> {
        self.validate_batches_at_clock_with_policy(batches, event_clock, token, false)
    }

    /// Validate and commit exact accesses at their already-resolved actor
    /// clock.
    ///
    /// Deferred completion and TCGEN work issue are terminal checker events:
    /// the runtime transition has already succeeded, and a checker rejection
    /// ends the numerical run. Validate and publish each witness immediately,
    /// avoiding a retained sparse rollback patch and a second traversal of all
    /// batches. Earlier witnesses from the same semantic operation are allowed
    /// by `validate_clocked_witness`; publishing one cannot hide an external
    /// conflict from an overlapping later witness because the earlier witness
    /// has already proved that same prior byte state ordered.
    pub(crate) fn validate_and_commit_batches_at_clock(
        &mut self,
        batches: &[PhysicalAccessBatch],
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
        tmem_load_register_source: bool,
    ) -> Result<Vec<PhysicalRaceFinding>, RaceShadowError> {
        if event_clock.warp_count() != self.warp_count() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: event_clock.warp_count(),
            });
        }
        let event_clock = Arc::new(event_clock.clone());
        let mut proxy_sensitive_allocations = BTreeSet::new();
        let mut review_findings = Vec::new();
        let timestamp =
            RaceEventTimestamp::for_async(&event_clock, token, &self.operation_registry);
        for batch in batches {
            self.note_slot_referencing_batch(batch);
            let descriptor = batch.descriptor();
            proxy_sensitive_allocations
                .extend(self.proxy_allocations_for_batch(batch, &event_clock)?);
            let operation = match Arc::get_mut(&mut self.operation_registry) {
                Some(registry) if tmem_load_register_source => {
                    registry.register_tmem_load_register_source_exclusive(batch.operation().id())
                }
                Some(registry) => registry.register_exclusive(batch.operation().id()),
                None if tmem_load_register_source => self
                    .operation_registry
                    .register_tmem_load_register_source(batch.operation().id()),
                None => self.operation_registry.register(batch.operation().id()),
            };
            let kind = descriptor.kind();
            let space = descriptor.space();
            let proxy = descriptor.memory_semantics().proxy();
            let proxy_domain = descriptor.proxy_memory_domain();
            if self.tile_window_batch(
                batch,
                operation,
                kind,
                space,
                proxy,
                proxy_domain,
                &event_clock,
                &timestamp,
                &mut review_findings,
            )? {
                continue;
            }
            let mut allocations = batch
                .lanes()
                .iter()
                .flat_map(|lane| lane.footprint().allocations());
            let common_allocation = allocations
                .next()
                .filter(|allocation| allocations.all(|other| other == *allocation));
            if let Some(allocation) = common_allocation {
                profile_count(ProfileKind::RaceClockedCommonAllocation);
                let registry = &self.operation_registry;
                let segments = Arc::make_mut(
                    self.allocations
                        .entry(AllocationKey { space, allocation })
                        .or_default(),
                );
                let owned = {
                    let segments: &ShadowSegments = segments;
                    registry.owned_spans_for_batch(operation, batch, |_| Some(segments))
                };
                let mut owned = owned.into_iter();
                for lane in batch.lanes() {
                    for span in lane.footprint().spans().iter().copied() {
                        let current = PhysicalRaceWitnessRef::from_lane(
                            operation,
                            lane,
                            kind,
                            space,
                            proxy,
                            proxy_domain,
                            span,
                        )
                        .with_memory_semantics(descriptor.memory_semantics());
                        if owned.next().unwrap_or(false) {
                        } else {
                            validate_clocked_witness_in_segments(
                                registry,
                                segments,
                                current,
                                &event_clock,
                                timestamp,
                                &mut review_findings,
                            )?;
                        }
                        record_validated_clocked_witness(
                            registry,
                            segments,
                            current,
                            &event_clock,
                            &timestamp,
                        );
                    }
                }
                continue;
            }
            profile_count(ProfileKind::RaceClockedMixedAllocation);
            let owned = {
                let allocations = &self.allocations;
                self.operation_registry
                    .owned_spans_for_batch(operation, batch, |span| {
                        allocations
                            .get(&AllocationKey {
                                space,
                                allocation: span.allocation(),
                            })
                            .map(Arc::as_ref)
                    })
            };
            let mut owned = owned.into_iter();
            for lane in batch.lanes() {
                // A footprint's spans are sorted by allocation: look the shadow
                // up and unshare it once per allocation, not once per span. An
                // allocation without a shadow validates nothing either way.
                for run in lane
                    .footprint()
                    .spans()
                    .chunk_by(|left, right| left.allocation() == right.allocation())
                {
                    let key = AllocationKey {
                        space,
                        allocation: run[0].allocation(),
                    };
                    let registry = &self.operation_registry;
                    let segments = Arc::make_mut(self.allocations.entry(key).or_default());
                    for span in run.iter().copied() {
                        let current = PhysicalRaceWitnessRef::from_lane(
                            operation,
                            lane,
                            kind,
                            space,
                            proxy,
                            proxy_domain,
                            span,
                        )
                        .with_memory_semantics(descriptor.memory_semantics());
                        if owned.next().unwrap_or(false) {
                        } else {
                            validate_clocked_witness_in_segments(
                                registry,
                                segments,
                                current,
                                &event_clock,
                                timestamp,
                                &mut review_findings,
                            )?;
                        }
                        record_validated_clocked_witness(
                            registry,
                            segments,
                            current,
                            &event_clock,
                            &timestamp,
                        );
                    }
                }
            }
        }
        self.commit_proxy_sensitive_allocations(proxy_sensitive_allocations);
        self.retire_reviewed_tmem_load_findings(&review_findings);
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(review_findings)
    }

    fn validate_batches_at_clock_with_policy<'a>(
        &mut self,
        batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
        tmem_load_register_source: bool,
    ) -> Result<RaceClockedBatchValidation, RaceShadowError> {
        let batches = batches.into_iter().collect::<Vec<_>>();
        for batch in batches.iter().copied() {
            self.demote_tile_windows_for_batch(batch);
        }
        if event_clock.warp_count() != self.warp_count() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: event_clock.warp_count(),
            });
        }
        let event_clock = Arc::new(event_clock.clone());
        let timestamp =
            RaceEventTimestamp::for_async(&event_clock, token, &self.operation_registry);
        let mut allocation_updates = BTreeMap::new();
        let mut proxy_sensitive_allocations = BTreeSet::new();
        let mut review_findings = Vec::new();
        for batch in batches {
            proxy_sensitive_allocations
                .extend(self.proxy_allocations_for_batch(batch, &event_clock)?);
            stage_batch_into_allocation_patches(
                &self.operation_registry,
                &self.allocations,
                &mut allocation_updates,
                batch,
                &event_clock,
                &timestamp,
                None,
                true,
                tmem_load_register_source,
                &mut review_findings,
            )?;
        }
        Ok(RaceClockedBatchValidation {
            allocation_updates,
            proxy_sensitive_allocations,
            review_findings,
        })
    }

    /// Start an exact multi-actor completion transaction.
    pub(crate) fn empty_clocked_batch_validation(&self) -> RaceClockedBatchValidation {
        RaceClockedBatchValidation {
            allocation_updates: BTreeMap::new(),
            proxy_sensitive_allocations: BTreeSet::new(),
            review_findings: Vec::new(),
        }
    }

    /// Add exact accesses at one async actor milestone to an existing
    /// transaction. Later actors observe earlier staged updates through the
    /// shared sparse overlay, matching sequential actor advancement without
    /// cloning the committed shadow.
    pub(crate) fn extend_batches_at_clock<'a>(
        &mut self,
        validation: &mut RaceClockedBatchValidation,
        batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
        event_clock: &RaceVectorClock,
        token: &AsyncTokenId,
    ) -> Result<(), RaceShadowError> {
        let batches = batches.into_iter().collect::<Vec<_>>();
        for batch in batches.iter().copied() {
            self.demote_tile_windows_for_batch(batch);
        }
        if event_clock.warp_count() != self.warp_count() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: event_clock.warp_count(),
            });
        }
        let event_clock = Arc::new(event_clock.clone());
        let timestamp =
            RaceEventTimestamp::for_async(&event_clock, token, &self.operation_registry);
        for batch in batches {
            validation
                .proxy_sensitive_allocations
                .extend(self.proxy_allocations_for_batch(batch, &event_clock)?);
            stage_batch_into_allocation_patches(
                &self.operation_registry,
                &self.allocations,
                &mut validation.allocation_updates,
                batch,
                &event_clock,
                &timestamp,
                None,
                false,
                false,
                &mut validation.review_findings,
            )?;
        }
        Ok(())
    }

    pub(crate) fn validate_async_issue_batches<'a>(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        token: &AsyncTokenId,
        batches: impl IntoIterator<Item = &'a PhysicalAccessBatch>,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> Result<RaceAsyncIssueValidation, RaceShadowError> {
        let batches = batches.into_iter().collect::<Vec<_>>();
        for batch in batches.iter().copied() {
            self.demote_tile_windows_for_batch(batch);
        }
        if self.active_async_actor_clocks.contains_key(token) {
            return Err(RaceShadowError::DuplicateAsyncActor {
                token: token.clone(),
            });
        }
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut current_clock = self.warp_clocks[local_warp_id].clone();
        let mut final_event_clock = None;
        let mut allocation_updates = BTreeMap::new();
        let mut proxy_sensitive_allocations = BTreeSet::new();
        let mut review_findings = Vec::new();
        for batch in batches {
            if batch.operation().id().global_warp_id() != warp_id {
                return Err(RaceShadowError::InvalidWarp {
                    warp_id: batch.operation().id().global_warp_id(),
                    warp_count: self.global_warp_base.saturating_add(self.warp_count()),
                });
            }
            current_clock.tick(local_warp_id)?;
            let event_clock = Arc::new(current_clock.clone());
            proxy_sensitive_allocations
                .extend(self.proxy_allocations_for_batch(batch, &event_clock)?);
            let timestamp =
                RaceEventTimestamp::for_warp(&event_clock, local_warp_id, &self.operation_registry);
            stage_batch_into_allocation_patches(
                &self.operation_registry,
                &self.allocations,
                &mut allocation_updates,
                batch,
                &event_clock,
                &timestamp,
                lane_order,
                false,
                false,
                &mut review_findings,
            )?;
            final_event_clock = Some(event_clock);
        }
        let warp_validation = final_event_clock.map(|event_clock| RaceBatchValidation {
            warp_id: local_warp_id,
            event_clock,
            allocation_updates,
            proxy_sensitive_allocations: (!proxy_sensitive_allocations.is_empty())
                .then(|| Box::new(proxy_sensitive_allocations)),
            review_findings,
        });
        let mut token_clock = lane_order.map_or_else(
            || current_clock.clone(),
            |order| order.acquired_mask_clock(&current_clock, active_mask),
        );
        token_clock.restrict_proxy_lanes(active_mask);
        token_clock.tick_async_issue(token, local_warp_id)?;
        Ok(RaceAsyncIssueValidation {
            warp_validation,
            token: token.clone(),
            token_clock,
        })
    }

    pub(crate) fn commit_async_issue_validation(
        &mut self,
        validation: RaceAsyncIssueValidation,
    ) -> Result<(RaceVectorClock, Vec<PhysicalRaceFinding>), RaceShadowError> {
        if self
            .active_async_actor_clocks
            .contains_key(&validation.token)
        {
            return Err(RaceShadowError::DuplicateAsyncActor {
                token: validation.token,
            });
        }
        // Validation has registered the token's slot, but issue witnesses carry
        // warp timestamps. Pin the actor before committing those witnesses can
        // reach a reclamation safe point and mistake the slot for an unused one.
        self.active_async_actor_clocks
            .insert(validation.token, validation.token_clock.clone());
        let mut review_findings = Vec::new();
        if let Some(warp_validation) = validation.warp_validation {
            self.warp_clocks[warp_validation.warp_id] =
                warp_validation.event_clock.as_ref().clone();
            for (allocation, update) in warp_validation.allocation_updates {
                self.slot_referencing_allocations.insert(allocation);
                commit_allocation_update(&mut self.allocations, allocation, update);
            }
            if let Some(allocations) = warp_validation.proxy_sensitive_allocations {
                self.commit_proxy_sensitive_allocations(*allocations);
            }
            review_findings = warp_validation.review_findings;
            self.retire_reviewed_tmem_load_findings(&review_findings);
            self.note_memory_safe_point();
        }
        self.bump_revision();
        Ok((validation.token_clock, review_findings))
    }

    pub(crate) fn commit_clocked_batches(
        &mut self,
        validation: RaceClockedBatchValidation,
    ) -> Vec<PhysicalRaceFinding> {
        for (allocation, update) in validation.allocation_updates {
            self.slot_referencing_allocations.insert(allocation);
            commit_allocation_update(&mut self.allocations, allocation, update);
        }
        self.commit_proxy_sensitive_allocations(validation.proxy_sensitive_allocations);
        self.retire_reviewed_tmem_load_findings(&validation.review_findings);
        self.note_memory_safe_point();
        self.bump_revision();
        validation.review_findings
    }

    /// Commit a previously validated synchronous batch. The validation token
    /// already owns every shadow update; retaining or cloning the source batch
    /// until commit would add no safety information.
    pub(crate) fn commit_validation(
        &mut self,
        validation: RaceBatchValidation,
    ) -> Vec<PhysicalRaceFinding> {
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceCommitClock);
            self.warp_clocks[validation.warp_id] = validation.event_clock.as_ref().clone();
        }
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceCommitAllocation);
            for (allocation, update) in validation.allocation_updates {
                self.slot_referencing_allocations.insert(allocation);
                commit_allocation_update(&mut self.allocations, allocation, update);
            }
        }
        if let Some(allocations) = validation.proxy_sensitive_allocations {
            self.commit_proxy_sensitive_allocations(*allocations);
        }
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceCommitRetire);
            self.retire_reviewed_tmem_load_findings(&validation.review_findings);
        }
        {
            let _profile = ProfileTimer::new(ProfileKind::RaceCommitSafePoint);
            self.note_memory_safe_point();
        }
        self.bump_revision();
        validation.review_findings
    }

    fn check_batch_at_clock_with_policy(
        &mut self,
        batch: &PhysicalAccessBatch,
        event_clock: &RaceVectorClock,
        allow_same_operation: bool,
        witness_async_actor: Option<&AsyncTokenId>,
        tmem_load_register_source: bool,
    ) -> Result<Vec<PhysicalRaceFinding>, RaceShadowError> {
        self.demote_tile_windows_for_batch(batch);
        if event_clock.warp_count() != self.warp_count() {
            return Err(RaceShadowError::ClockDimensionMismatch {
                expected_warps: self.warp_count(),
                actual_warps: event_clock.warp_count(),
            });
        }
        let event_clock = Arc::new(event_clock.clone());
        self.note_slot_referencing_batch(batch);
        let proxy_sensitive_allocations = self.proxy_allocations_for_batch(batch, &event_clock)?;
        let timestamp = witness_async_actor.map_or_else(
            || RaceEventTimestamp::for_vector(Arc::clone(&event_clock), &self.operation_registry),
            |token| RaceEventTimestamp::for_async(&event_clock, token, &self.operation_registry),
        );
        let mut review_findings = Vec::new();
        let allocation_updates = stage_allocation_patches(
            &self.operation_registry,
            &self.allocations,
            batch,
            &event_clock,
            &timestamp,
            None,
            allow_same_operation,
            tmem_load_register_source,
            &mut review_findings,
        )?;
        for (allocation, update) in allocation_updates {
            self.slot_referencing_allocations.insert(allocation);
            commit_allocation_update(&mut self.allocations, allocation, update);
        }
        self.commit_proxy_sensitive_allocations(proxy_sensitive_allocations);
        self.retire_reviewed_tmem_load_findings(&review_findings);
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(review_findings)
    }

    /// Create a release payload after one barrier-arrive event on `warp_id`.
    pub fn barrier_release(
        &mut self,
        warp_id: usize,
    ) -> Result<BarrierClockPayload, RaceShadowError> {
        self.barrier_release_masked(warp_id, WarpMask::FULL)
    }

    pub(crate) fn barrier_release_masked(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
    ) -> Result<BarrierClockPayload, RaceShadowError> {
        let local_warp_id = self.local_warp_id(warp_id)?;
        let clock = &mut self.warp_clocks[local_warp_id];
        clock.tick(local_warp_id)?;
        let payload = BarrierClockPayload::from_clock_for_mask(clock.clone(), active_mask);
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(payload)
    }

    /// Merge a completed barrier's contributor clocks, then record the wait.
    pub fn barrier_acquire(
        &mut self,
        warp_id: usize,
        payload: &BarrierClockPayload,
    ) -> Result<(), RaceShadowError> {
        self.barrier_acquire_masked(warp_id, WarpMask::FULL, payload)
    }

    pub(crate) fn barrier_acquire_masked(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        payload: &BarrierClockPayload,
    ) -> Result<(), RaceShadowError> {
        let local_warp_id = self.local_warp_id(warp_id)?;
        let current = &self.warp_clocks[local_warp_id];
        let mut acquired = current.clone();
        acquired.acquire_payload(payload, active_mask)?;
        acquired.tick(local_warp_id)?;
        self.warp_clocks[local_warp_id] = acquired;
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(())
    }

    pub(crate) fn proxy_async_fence(
        &mut self,
        warp_id: usize,
        scope: ProxyAsyncFenceScope,
    ) -> Result<(), RaceShadowError> {
        self.proxy_async_fence_masked(warp_id, WarpMask::FULL, scope, None)
    }

    pub(crate) fn proxy_async_fence_masked(
        &mut self,
        warp_id: usize,
        active_mask: WarpMask,
        scope: ProxyAsyncFenceScope,
        lane_order: Option<&RaceLaneOrder<'_>>,
    ) -> Result<(), RaceShadowError> {
        let local_warp_id = self.local_warp_id(warp_id)?;
        let clock = &mut self.warp_clocks[local_warp_id];
        if let Some(order) = lane_order {
            for lane in active_mask {
                let mut frontier =
                    ProxyClockFrontier::from_clock(&order.acquired_clock(clock, lane));
                frontier.shared_lanes = order.publication_lanes(lane);
                clock.apply_proxy_fence(scope, WarpMask::from_bits(1 << lane), &frontier)?;
            }
        } else {
            clock.apply_proxy_async_fence(scope, active_mask)?;
        }
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(())
    }

    pub(crate) fn warp_sync(
        &mut self,
        warp_id: usize,
        participating_mask: WarpMask,
    ) -> Result<(), RaceShadowError> {
        let local_warp_id = self.local_warp_id(warp_id)?;
        self.warp_clocks[local_warp_id].synchronize_proxy_lanes(participating_mask)?;
        self.note_memory_safe_point();
        self.bump_revision();
        Ok(())
    }

    /// Retire records that every warp's current clock has observed.
    ///
    /// This is deliberately conservative: records remain live until all warps
    /// and every still-active async actor have clocks that dominate them.
    pub fn gc_dominated_frontier(&mut self) -> usize {
        self.safe_points_since_gc = 0;
        let observed_frontier = self.dominated_frontier();
        if observed_frontier == self.observed_frontier {
            return 0;
        }
        let retired = self.retire_dominated(&observed_frontier, None);
        self.observed_frontier = observed_frontier;
        self.bump_revision();
        retired
    }

    /// Hand back dead async clock slots without a full pass: retire only
    /// inside the allocations whose witnesses can refer to slots, then free
    /// every slot that no surviving witness and no active actor names.
    fn reclaim_async_slots(&mut self) {
        let observed_frontier = self.dominated_frontier();
        let only = std::mem::take(&mut self.slot_referencing_allocations);
        self.retire_dominated(&observed_frontier, Some(&only));
        self.slot_referencing_allocations = only;
        self.bump_revision();
    }

    /// Remember the allocations a batch checked at an explicit clock touches:
    /// its witnesses carry async-actor or registered-vector timestamps, the
    /// only ones that refer to async clock slots.
    fn note_slot_referencing_batch(&mut self, batch: &PhysicalAccessBatch) {
        let space = batch.descriptor().space();
        // A batch's spans nearly always share one allocation and this runs
        // per issue: hash only when the allocation changes.
        let mut noted = None;
        for lane in batch.lanes() {
            for allocation in lane.footprint().allocations() {
                if noted == Some(allocation) {
                    continue;
                }
                noted = Some(allocation);
                self.slot_referencing_allocations
                    .insert(AllocationKey { space, allocation });
            }
        }
    }

    /// The meet of every warp's and every active async actor's clock.
    fn dominated_frontier(&self) -> RaceVectorClock {
        minimum_actor_clock(
            self.warp_clocks
                .iter()
                .chain(self.active_async_actor_clocks.values()),
            self.warp_count(),
            Arc::clone(&self.async_clock_registry),
        )
    }

    /// Retire every record `observed_frontier` dominates — in every
    /// allocation, or in `only` — record the retired generic history, and
    /// reclaim the async clock slots nothing refers to any more.
    fn retire_dominated(
        &mut self,
        observed_frontier: &RaceVectorClock,
        only: Option<&HashSet<AllocationKey>>,
    ) -> usize {
        let mut retired = 0;
        // The retired generic history records exactly what was retired: per
        // access kind and proxy domain, the join of the retired witnesses' own
        // timestamps — the same facts the live cross-proxy check consults.
        // Snapshotting the whole observed frontier instead would also carry
        // async actors every warp happened to observe by GC time, which a
        // proxy fence issued before them could never dominate, and would
        // report a spurious unordered cross-proxy history.
        let warp_count = self.warp_count();
        let async_registry = Arc::clone(&self.async_clock_registry);
        let mut overflow_frontiers = RetiredProxyFrontiers::default();
        let mut referenced_async = AsyncIndexSet::new(self.async_clock_registry.slot_count());
        retired += self.retire_tile_windows(
            observed_frontier,
            only,
            warp_count,
            &async_registry,
            &mut overflow_frontiers,
            &mut referenced_async,
        );
        self.allocations.retain(|key, segments| {
            if only.is_some_and(|only| !only.contains(key)) {
                return true;
            }
            let proxy_sensitive = self.proxy_sensitive_allocations.contains(key);
            // Rebuilding the segment map is the expensive part of a pass;
            // an allocation with nothing dominated only needs its slot
            // references marked.
            if !segments.iter().any(|(_, segment)| {
                segment.state.has_globally_observed_witness(
                    &self.operation_registry,
                    observed_frontier,
                    proxy_sensitive,
                )
            }) {
                for (_, segment) in segments.iter() {
                    segment
                        .state
                        .mark_referenced_async(&self.operation_registry, &mut referenced_async);
                }
                return true;
            }
            let segments = Arc::make_mut(segments);
            let mut retained = std::mem::take(segments).into_sorted_values();
            let mut retired_history = RetiredAllocationHistory::default();
            for segment in &mut retained {
                let mut segment_frontiers = RetiredProxyFrontiers::default();
                retired += segment
                    .state
                    .retire_globally_observed(
                        &self.operation_registry,
                        observed_frontier,
                        proxy_sensitive,
                        &mut segment_frontiers,
                        warp_count,
                        &async_registry,
                        &mut referenced_async,
                    )
                    .expect("one race shadow's retirement frontiers are compatible");
                if !segment_frontiers.is_empty() {
                    retired_history
                        .insert_segment(segment.start, segment.end, &segment_frontiers)
                        .expect("one race shadow's retirement frontiers are compatible");
                }
            }
            self.retired_generic_history
                .record(*key, &retired_history, &mut overflow_frontiers)
                .expect("one race shadow's retirement frontiers are compatible");
            retained.retain(|segment| !segment.state.is_empty());
            merge_adjacent_segments(&mut retained);
            for segment in retained {
                segments.insert_prepared(segment.start, segment.end, segment);
            }
            !segments.is_empty()
        });
        self.retired_generic_history
            .commit_overflow(&overflow_frontiers)
            .expect("one race shadow's retirement frontiers are compatible");
        self.reclaim_async_indices(&mut referenced_async);
        retired
    }

    /// Hand back every async clock slot that no live actor and no retained
    /// witness refers to any more. `referenced` already lists the witnesses
    /// that survived the retirement pass.
    fn reclaim_async_indices(&mut self, referenced: &mut AsyncIndexSet) {
        for token in self.active_async_actor_clocks.keys() {
            if let Some(index) = self.async_clock_registry.index(token) {
                referenced.mark(index);
            }
        }
        let freed = self
            .async_clock_registry
            .reclaim(|index| !referenced.contains(index));
        for index in freed {
            if self.tcgen_async_indices.remove(&index) {
                self.tcgen_completed_epochs.remove(&index);
                if let Some(slots) = self.tcgen_slots_by_chunk.get_mut(index / ASYNC_EPOCH_CHUNK) {
                    slots.retain(|(slot, _)| usize::from(*slot) != index % ASYNC_EPOCH_CHUNK);
                }
            }
        }
    }

    pub fn tracked_interval_count(&self) -> usize {
        self.allocations
            .values()
            .map(|segments| segments.len())
            .sum::<usize>()
            + self
                .tile_windows
                .values()
                .flat_map(|windows| windows.iter())
                .map(|window| window.classes.len())
                .sum::<usize>()
    }

    pub fn active_async_actor_count(&self) -> usize {
        self.active_async_actor_clocks.len()
    }

    fn note_memory_safe_point(&mut self) {
        self.safe_points_since_gc = self.safe_points_since_gc.saturating_add(1);
        if self.safe_points_since_gc >= AUTOMATIC_GC_SAFE_POINT_INTERVAL {
            self.gc_dominated_frontier();
        } else if self.async_clock_registry.reclaim_due() {
            self.reclaim_async_slots();
        }
    }
}

fn minimum_actor_clock<'a>(
    mut actor_clocks: impl Iterator<Item = &'a RaceVectorClock>,
    warp_count: usize,
    async_registry: Arc<AsyncClockRegistry>,
) -> RaceVectorClock {
    let Some(first) = actor_clocks.next() else {
        return RaceVectorClock::zero(warp_count, async_registry);
    };
    debug_assert_eq!(first.warp_count(), warp_count);
    let mut minimum = first.clone();
    for clock in actor_clocks {
        debug_assert_eq!(clock.warp_count(), warp_count);
        minimum.components.meet(&clock.components);
        minimum.async_components.meet(&clock.async_components);
        let proxy_lane_mask = clock.proxy_lane_mask();
        if let Some(candidate) = clock.proxy_bridges() {
            if let Some(current) = minimum.proxy_bridges_mut() {
                current.meet_active(candidate, proxy_lane_mask);
            }
        } else if proxy_lane_mask.is_full() {
            minimum.clear_proxy_bridges();
        } else if let Some(current) = minimum.proxy_bridges_mut() {
            current.retain(!proxy_lane_mask);
        }
    }
    minimum
}

impl RaceShadow {
    fn validate_batch(
        &self,
        batch: &PhysicalAccessBatch,
    ) -> Result<RaceBatchValidation, RaceShadowError> {
        let warp_id = batch.operation().id().global_warp_id();
        let local_warp_id = self.local_warp_id(warp_id)?;
        let mut event_clock = self.warp_clocks[local_warp_id].clone();
        event_clock.tick(local_warp_id)?;
        let event_clock = Arc::new(event_clock);
        let mut validation = RaceBatchValidation {
            warp_id: local_warp_id,
            event_clock,
            allocation_updates: BTreeMap::new(),
            proxy_sensitive_allocations: None,
            review_findings: Vec::new(),
        };
        debug_assert!(
            !self.tile_windows_overlap_batch(batch),
            "the generic batch target is validated without tile windows over its spans"
        );
        self.extend_batch_validation_with_optional_lane_order_unchecked(
            &mut validation,
            batch,
            None,
            false,
        )?;
        Ok(validation)
    }
}

fn commit_allocation_update(
    allocations: &mut BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    key: AllocationKey,
    update: AllocationPatch,
) {
    let existing = allocations.entry(key).or_default();
    update.commit_into(Arc::make_mut(existing));
}

fn replace_segment_map_range(
    segments: &mut ShadowSegments,
    start: usize,
    end: usize,
    replacement: Vec<ShadowSegment>,
) {
    debug_assert!(start < end);
    debug_assert_eq!(
        replacement.first().map(|segment| segment.start),
        Some(start)
    );
    debug_assert_eq!(replacement.last().map(|segment| segment.end), Some(end));
    debug_assert!(replacement
        .windows(2)
        .all(|pair| pair[0].end == pair[1].start));

    // Repeated tiled accesses normally preserve interval geometry. Replace all
    // matching states in place instead of removing and reinserting thousands
    // of scalar intervals at every warp scheduling boundary.
    if replacement.iter().all(|replacement| {
        segments
            .get(replacement.start)
            .is_some_and(|segment| segment.end == replacement.end)
    }) {
        for replacement in replacement {
            segments
                .get_mut(replacement.start)
                .expect("the exact segment was just observed")
                .state = replacement.state;
        }
        return;
    }

    replace_segment_map_range_after_exact_miss(segments, start, end, replacement);
}

fn replace_segment_map_one(segments: &mut ShadowSegments, replacement: ShadowSegment) {
    let start = replacement.start;
    let end = replacement.end;
    let start_of = |segment: &ShadowSegment| segment.start;
    let end_of = |segment: &ShadowSegment| segment.end;
    let split = |segment: &ShadowSegment, start, end| {
        ShadowSegment::new(start, end, segment.state.clone())
    };
    let indexed = {
        // Count the same logical replacement once, including a fast-path miss.
        let _profile = ProfileTimer::new(ProfileKind::RaceCommitIndexedReplace);
        segments
            .try_replace_one(start, end, replacement, start_of, end_of, split)
            .or_else(|replacement| {
                segments.try_replace_range(start, end, vec![replacement], start_of, end_of, split)
            })
    };
    if let Err(replacement) = indexed {
        let _profile = ProfileTimer::new(ProfileKind::RaceCommitGeneralReplace);
        replace_patch_segment_range(segments.general_mut(), start, end, replacement);
    }
}

fn replace_segment_map_range_after_exact_miss(
    segments: &mut ShadowSegments,
    start: usize,
    end: usize,
    replacement: Vec<ShadowSegment>,
) {
    let indexed = {
        let _profile = ProfileTimer::new(ProfileKind::RaceCommitIndexedReplace);
        segments.try_replace_range(
            start,
            end,
            replacement,
            |segment| segment.start,
            |segment| segment.end,
            |segment, split_start, split_end| {
                ShadowSegment::new(split_start, split_end, segment.state.clone())
            },
        )
    };
    if let Err(replacement) = indexed {
        let _profile = ProfileTimer::new(ProfileKind::RaceCommitGeneralReplace);
        replace_patch_segment_range(segments.general_mut(), start, end, replacement);
    }
}

#[cfg(test)]
fn apply_access_to_span_in_place(
    registry: &OperationRegistry,
    segments: &mut Vec<ShadowSegment>,
    lane: &LanePhysicalAccess,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    span: PhysicalByteSpan,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
) -> Result<(), RaceShadowError> {
    let (start_index, end_index, replacement) = plan_access_to_span(
        registry,
        segments,
        lane,
        kind,
        space,
        span,
        event_clock,
        timestamp,
    )?;
    segments.splice(start_index..end_index, replacement);
    Ok(())
}

#[cfg(test)]
fn plan_access_to_span(
    registry: &OperationRegistry,
    segments: &[ShadowSegment],
    lane: &LanePhysicalAccess,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    span: PhysicalByteSpan,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
) -> Result<(usize, usize, Vec<ShadowSegment>), RaceShadowError> {
    let operation = registry.register(lane.provenance().operation());
    let proxy_domain = match space {
        PhysicalAccessSpace::Global => ProxyMemoryDomain::Global,
        PhysicalAccessSpace::Shared => ProxyMemoryDomain::SharedCta,
        PhysicalAccessSpace::Local | PhysicalAccessSpace::Register | PhysicalAccessSpace::Tmem => {
            ProxyMemoryDomain::Other
        }
    };
    let current = PhysicalRaceWitnessRef::from_lane(
        operation,
        lane,
        kind,
        space,
        MemoryProxy::Generic,
        proxy_domain,
        span,
    );
    let start = span.byte_offset();
    let end = span.byte_end();
    let first = segments.partition_point(|segment| segment.end <= start);
    let last = first + segments[first..].partition_point(|segment| segment.start < end);

    if first == segments.len() {
        return Ok((
            first,
            first,
            vec![new_access_segment(
                registry,
                start,
                end,
                event_clock,
                timestamp,
                &current,
            )],
        ));
    }

    if last == first + 1 && segments[first].start == start && segments[first].end == end {
        let overlap = PhysicalByteSpan::new(span.allocation(), start, end - start)
            .expect("an exact segment match is a valid physical span");
        let mut state = segments[first].state.clone();
        let mut review_findings = Vec::new();
        state.check_and_record(
            registry,
            event_clock,
            timestamp,
            current,
            overlap,
            None,
            false,
            &mut review_findings,
        )?;
        return Ok((first, last, vec![ShadowSegment::new(start, end, state)]));
    }

    let mut cursor = start;
    let mut replacement = Vec::with_capacity(last - first + 2);

    for segment in &segments[first..last] {
        if segment.start < start {
            replacement.push(ShadowSegment::new(
                segment.start,
                start,
                segment.state.clone(),
            ));
        }

        let overlap_start = segment.start.max(start);
        let overlap_end = segment.end.min(end);
        if cursor < overlap_start {
            replacement.push(new_access_segment(
                registry,
                cursor,
                overlap_start,
                event_clock,
                timestamp,
                &current,
            ));
        }

        let mut overlap_state = segment.state.clone();
        let mut review_findings = Vec::new();
        let overlap = PhysicalByteSpan::new(
            span.allocation(),
            overlap_start,
            overlap_end - overlap_start,
        )
        .expect("intersection of valid physical spans is valid");
        overlap_state.check_and_record(
            registry,
            event_clock,
            timestamp,
            current.clone(),
            overlap,
            None,
            false,
            &mut review_findings,
        )?;
        replacement.push(ShadowSegment::new(
            overlap_start,
            overlap_end,
            overlap_state,
        ));
        cursor = overlap_end;

        if segment.end > end {
            replacement.push(ShadowSegment::new(end, segment.end, segment.state.clone()));
        }
    }

    if cursor < end {
        replacement.push(new_access_segment(
            registry,
            cursor,
            end,
            event_clock,
            timestamp,
            &current,
        ));
    }
    merge_adjacent_segments(&mut replacement);

    let mut splice_start = first;
    let mut splice_end = last;
    if splice_start > 0
        && same_shadow_state(
            &segments[splice_start - 1].state,
            &replacement
                .first()
                .expect("a nonempty access produces a replacement segment")
                .state,
        )
    {
        replacement
            .first_mut()
            .expect("a nonempty access produces a replacement segment")
            .start = segments[splice_start - 1].start;
        splice_start -= 1;
    }
    if splice_end < segments.len()
        && same_shadow_state(
            &replacement
                .last()
                .expect("a nonempty access produces a replacement segment")
                .state,
            &segments[splice_end].state,
        )
    {
        replacement
            .last_mut()
            .expect("a nonempty access produces a replacement segment")
            .end = segments[splice_end].end;
        splice_end += 1;
    }
    Ok((splice_start, splice_end, replacement))
}

#[cfg(test)]
fn new_access_segment(
    registry: &OperationRegistry,
    start: usize,
    end: usize,
    event_clock: &Arc<RaceVectorClock>,
    timestamp: &RaceEventTimestamp,
    current: &PhysicalRaceWitnessRef,
) -> ShadowSegment {
    let mut state = ShadowState::default();
    let mut review_findings = Vec::new();
    state
        .check_and_record(
            registry,
            event_clock,
            timestamp,
            current.clone(),
            PhysicalByteSpan::new(current.span().allocation(), start, end - start)
                .expect("subspan of a valid physical span is valid"),
            None,
            false,
            &mut review_findings,
        )
        .expect("an empty shadow state cannot race");
    ShadowSegment::new(start, end, state)
}

fn merge_adjacent_segments(segments: &mut Vec<ShadowSegment>) {
    if segments.len() < 2 {
        return;
    }
    let mut merged: Vec<ShadowSegment> = Vec::with_capacity(segments.len());
    for segment in segments.drain(..) {
        if let Some(previous) = merged.last_mut() {
            if previous.end == segment.start && same_shadow_state(&previous.state, &segment.state) {
                previous.end = segment.end;
                continue;
            }
        }
        merged.push(segment);
    }
    *segments = merged;
}

fn same_shadow_state(left: &ShadowState, right: &ShadowState) -> bool {
    left == right
}

// ---- piece masks ---------------------------------------------------------
//
// One bit per fixed-size piece of a byte region: how a tile access names the
// hundreds of equal, aligned pieces it touches.

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PieceMask {
    words: Box<[u64]>,
    pieces: usize,
}

impl PieceMask {
    pub(crate) fn empty(pieces: usize) -> Self {
        Self {
            words: vec![0; pieces.div_ceil(64)].into_boxed_slice(),
            pieces,
        }
    }

    pub(crate) fn pieces(&self) -> usize {
        self.pieces
    }

    /// Mask of the spans inside `[region_start, region_start + pieces * piece)`.
    /// `None` when a span is not piece-aligned or falls outside the region.
    pub(crate) fn from_spans(
        region_start: usize,
        piece: usize,
        pieces: usize,
        spans: impl IntoIterator<Item = PhysicalByteSpan>,
    ) -> Option<Self> {
        let mut mask = Self::empty(pieces);
        for span in spans {
            let offset = span.byte_offset().checked_sub(region_start)?;
            if offset % piece != 0 || span.byte_len() % piece != 0 {
                return None;
            }
            let first = offset / piece;
            let count = span.byte_len() / piece;
            if first + count > pieces {
                return None;
            }
            for index in first..first + count {
                mask.words[index / 64] |= 1 << (index % 64);
            }
        }
        Some(mask)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub(crate) fn set_range(&mut self, first: usize, count: usize) {
        debug_assert!(first + count <= self.pieces);
        for index in first..first + count {
            self.words[index / 64] |= 1 << (index % 64);
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        debug_assert_eq!(self.pieces, other.pieces);
        self.words.iter().zip(other.words.iter()).any(|(a, b)| a & b != 0)
    }

    /// Index of the lowest piece in both masks.
    pub(crate) fn first_common_piece(&self, other: &Self) -> Option<usize> {
        for (index, (a, b)) in self.words.iter().zip(other.words.iter()).enumerate() {
            let both = a & b;
            if both != 0 {
                return Some(index * 64 + both.trailing_zeros() as usize);
            }
        }
        None
    }

    pub(crate) fn first_piece(&self) -> Option<usize> {
        for (index, word) in self.words.iter().enumerate() {
            if *word != 0 {
                return Some(index * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    pub(crate) fn is_subset_of(&self, other: &Self) -> bool {
        self.words.iter().zip(other.words.iter()).all(|(a, b)| a & !b == 0)
    }

    pub(crate) fn intersection(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(other.words.iter()).map(|(a, b)| a & b).collect(),
            pieces: self.pieces,
        }
    }

    pub(crate) fn difference(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(other.words.iter()).map(|(a, b)| a & !b).collect(),
            pieces: self.pieces,
        }
    }

    pub(crate) fn union(&self, other: &Self) -> Self {
        Self {
            words: self.words.iter().zip(other.words.iter()).map(|(a, b)| a | b).collect(),
            pieces: self.pieces,
        }
    }

    pub(crate) fn contains(&self, index: usize) -> bool {
        index < self.pieces && self.words[index / 64] & (1 << (index % 64)) != 0
    }

    /// Runs of consecutive set pieces as `(first, count)`.
    pub(crate) fn runs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut index = 0usize;
        std::iter::from_fn(move || {
            while index < self.pieces && !self.contains(index) {
                // Skip whole clear words.
                if index % 64 == 0 && self.words[index / 64] == 0 {
                    index += 64;
                } else {
                    index += 1;
                }
            }
            if index >= self.pieces {
                return None;
            }
            let first = index;
            while index < self.pieces && self.contains(index) {
                index += 1;
            }
            Some((first, index - first))
        })
    }
}

#[cfg(test)]
mod piece_mask_tests {
    use super::PieceMask;
    use crate::{PhysicalAllocationId, PhysicalByteSpan};

    fn span(offset: usize, len: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(PhysicalAllocationId::new(1), offset, len).unwrap()
    }

    #[test]
    fn masks_follow_spans_and_set_operations() {
        let a = PieceMask::from_spans(1000, 8, 16, [span(1000, 8), span(1016, 8), span(1120, 8)]).unwrap();
        assert_eq!(a.count(), 3);
        assert!(a.contains(0) && a.contains(2) && a.contains(15) && !a.contains(1));
        let b = PieceMask::from_spans(1000, 8, 16, [span(1016, 16)]).unwrap();
        assert!(a.overlaps(&b));
        assert_eq!(a.first_common_piece(&b), Some(2));
        assert_eq!(a.difference(&b).count(), 2);
        assert_eq!(a.intersection(&b).count(), 1);
        assert_eq!(a.union(&b).count(), 4);
        assert!(!b.is_subset_of(&a));
        assert!(a.intersection(&b).is_subset_of(&a));
        assert_eq!(a.runs().collect::<Vec<_>>(), vec![(0, 1), (2, 1), (15, 1)]);
        assert_eq!(b.runs().collect::<Vec<_>>(), vec![(2, 2)]);
        assert!(PieceMask::from_spans(1000, 8, 16, [span(1004, 8)]).is_none());
        assert!(PieceMask::from_spans(1000, 8, 16, [span(1128, 8)]).is_none());
        assert!(PieceMask::from_spans(1000, 8, 16, [span(996, 8)]).is_none());
    }

    #[test]
    fn runs_cross_word_boundaries() {
        let m = PieceMask::from_spans(0, 4, 200, [span(248, 32), span(400, 4)]).unwrap();
        assert_eq!(m.runs().collect::<Vec<_>>(), vec![(62, 8), (100, 1)]);
        assert_eq!(m.first_piece(), Some(62));
    }
}

// ---- tile windows -------------------------------------------------------
//
// A transfer or MMA operand touches a tile as hundreds of equal, piece-aligned
// spans (padded FP4 slots, swizzled K-slices). Their shadow states are the
// same for every piece one event touched, so a tile region is tracked as
// classes of pieces with one `ShadowState` each, and a batch validates and
// records against a class once instead of once per piece. Witness spans
// inside a class are canonical: a witness whose span is one piece long stands
// for every piece of the class and is re-addressed to the piece it is asked
// about; larger spans belong to the whole access and are kept. Any access
// that does not fit the window form demotes the window back into segments,
// so the segment map stays the source of truth for everything else.

const TILE_WINDOW_MIN_SPANS: usize = 32;
const TILE_WINDOW_MAX_PIECES: usize = 1 << 16;
/// Windows start and end on this boundary, so every access to one tile —
/// a K-slice read, the whole-tile transfer write, a scale-row read — lands
/// in the same window instead of demoting and re-promoting it per stage.
const TILE_WINDOW_ALIGN: usize = 1024;
/// TMEM accumulators mix 64 B rows, 16 B column writes and 1 B scale reads
/// in one region; a bit-per-piece mask over that needs a row/column form,
/// so TMEM stays on segments for now.
const TILE_WINDOW_SPACES: [PhysicalAccessSpace; 1] = [PhysicalAccessSpace::Shared];

#[derive(Clone, Debug)]
struct TileClass {
    mask: PieceMask,
    state: ShadowState,
}

#[derive(Clone, Debug)]
struct TileWindow {
    start: usize,
    end: usize,
    piece: usize,
    classes: Vec<TileClass>,
}

impl TileWindow {
    fn pieces(&self) -> usize {
        (self.end - self.start) / self.piece
    }

    fn piece_span(&self, allocation: PhysicalAllocationId, index: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(allocation, self.start + index * self.piece, self.piece)
            .expect("a tile piece is a valid span")
    }

    fn contains(&self, start: usize, end: usize) -> bool {
        self.start <= start && end <= self.end
    }

    fn overlaps(&self, start: usize, end: usize) -> bool {
        self.start < end && start < self.end
    }

    /// Re-express the window in pieces of `new_piece`, a divisor of the
    /// current piece; every class mask expands bit by bit.
    fn refine(&mut self, new_piece: usize) {
        debug_assert!(new_piece > 0 && self.piece % new_piece == 0);
        if new_piece == self.piece {
            return;
        }
        let factor = self.piece / new_piece;
        let pieces = (self.end - self.start) / new_piece;
        for class in &mut self.classes {
            let mut mask = PieceMask::empty(pieces);
            for (first, count) in class.mask.runs() {
                mask.set_range(first * factor, count * factor);
            }
            class.mask = mask;
        }
        self.piece = new_piece;
    }
}

/// Whether a witness span is one block of the window's regular blocks: a
/// multiple of the piece, aligned to its own length from the window start.
/// Such spans are canonical inside a class and re-addressed per piece.
fn regular_block(span: PhysicalByteSpan, window_start: usize, piece: usize) -> bool {
    let len = span.byte_len();
    len > 0
        && len % piece == 0
        && span.byte_offset() >= window_start
        && (span.byte_offset() - window_start) % len == 0
}

/// The witness re-addressed to the block containing `piece_index`.
fn witness_for_piece(
    witness: &ClockedWitness,
    registry: &OperationRegistry,
    allocation: PhysicalAllocationId,
    window_start: usize,
    piece: usize,
    piece_index: usize,
) -> ClockedWitness {
    let wide = witness.witness.resolve(registry, allocation);
    if !regular_block(wide.span, window_start, piece) {
        return witness.clone();
    }
    let len = wide.span.byte_len();
    let block_start = window_start + ((piece_index * piece) / len) * len;
    if wide.span.byte_offset() == block_start {
        return witness.clone();
    }
    let span = PhysicalByteSpan::new(allocation, block_start, len).expect("a block span is valid");
    ClockedWitness {
        timestamp: witness.timestamp,
        witness: RetainedRaceWitness::new(
            wide.operation,
            span,
            wide.lane,
            wide.kind,
            wide.space,
            wide.proxy,
            wide.proxy_domain,
            wide.strong_scope,
            registry,
        ),
    }
}

fn state_for_piece(
    state: &ShadowState,
    registry: &OperationRegistry,
    allocation: PhysicalAllocationId,
    window_start: usize,
    piece: usize,
    piece_index: usize,
) -> ShadowState {
    ShadowState {
        writes: AccessFrontier::from_witnesses(
            state
                .writes
                .iter()
                .map(|witness| {
                    witness_for_piece(
                        witness,
                        registry,
                        allocation,
                        window_start,
                        piece,
                        piece_index,
                    )
                })
                .collect(),
        ),
        reads: AccessFrontier::from_witnesses(
            state
                .reads
                .iter()
                .map(|witness| {
                    witness_for_piece(
                        witness,
                        registry,
                        allocation,
                        window_start,
                        piece,
                        piece_index,
                    )
                })
                .collect(),
        ),
    }
}

/// A witness with its regular-block span reduced to its length, for
/// comparing class states.
fn erased_witness(
    witness: &ClockedWitness,
    registry: &OperationRegistry,
    allocation: PhysicalAllocationId,
    window_start: usize,
    piece: usize,
) -> (RaceEventTimestamp, WideRetainedWitness) {
    let mut wide = witness.witness.resolve(registry, allocation);
    if regular_block(wide.span, window_start, piece) {
        wide.span = PhysicalByteSpan::new(allocation, 0, wide.span.byte_len())
            .expect("a block span is valid");
    }
    (witness.timestamp, wide)
}

fn same_class_state(
    left: &ShadowState,
    right: &ShadowState,
    registry: &OperationRegistry,
    allocation: PhysicalAllocationId,
    window_start: usize,
    piece: usize,
) -> bool {
    if left.writes.as_slice().len() != right.writes.as_slice().len()
        || !left.writes.iter().zip(right.writes.iter()).all(|(a, b)| {
            erased_witness(a, registry, allocation, window_start, piece)
                == erased_witness(b, registry, allocation, window_start, piece)
        })
    {
        return false;
    }
    left.reads.as_slice().len() == right.reads.as_slice().len()
        && left.reads.iter().zip(right.reads.iter()).all(|(a, b)| {
            erased_witness(a, registry, allocation, window_start, piece)
                == erased_witness(b, registry, allocation, window_start, piece)
        })
}

fn merge_tile_classes(
    classes: &mut Vec<TileClass>,
    registry: &OperationRegistry,
    allocation: PhysicalAllocationId,
    window_start: usize,
    piece: usize,
) {
    let mut merged: Vec<TileClass> = Vec::with_capacity(classes.len());
    for class in classes.drain(..) {
        if class.mask.is_empty() {
            continue;
        }
        if let Some(existing) = merged.iter_mut().find(|existing| {
            same_class_state(
                &existing.state,
                &class.state,
                registry,
                allocation,
                window_start,
                piece,
            )
        }) {
            existing.mask = existing.mask.union(&class.mask);
        } else {
            merged.push(class);
        }
    }
    *classes = merged;
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl RaceShadow {
    fn tile_window_index(&self, key: &AllocationKey, start: usize, end: usize) -> Option<usize> {
        self.tile_windows
            .get(key)?
            .iter()
            .position(|window| window.contains(start, end))
    }

    /// Move every window overlapping `[start, end)` back into the segment map.
    fn demote_tile_windows_overlapping(&mut self, key: AllocationKey, start: usize, end: usize) {
        let Some(windows) = self.tile_windows.get_mut(&key) else {
            return;
        };
        let mut demoted = Vec::new();
        windows.retain(|window| {
            if window.overlaps(start, end) {
                demoted.push(window.clone());
                false
            } else {
                true
            }
        });
        if windows.is_empty() {
            self.tile_windows.remove(&key);
        }
        for window in demoted {
            self.demote_tile_window(key, window);
        }
    }

    /// Move every tile window back into the segment map.
    pub(crate) fn demote_all_tile_windows(&mut self) {
        let all = std::mem::take(&mut self.tile_windows);
        for (key, windows) in all {
            for window in windows {
                self.demote_tile_window(key, window);
            }
        }
    }

    fn tile_windows_overlap_batch(&self, batch: &PhysicalAccessBatch) -> bool {
        let space = batch.descriptor().space();
        batch.lanes().iter().any(|lane| {
            lane.footprint().spans().iter().any(|span| {
                self.tile_windows
                    .get(&AllocationKey {
                        space,
                        allocation: span.allocation(),
                    })
                    .is_some_and(|windows| {
                        windows
                            .iter()
                            .any(|window| window.overlaps(span.byte_offset(), span.byte_end()))
                    })
            })
        })
    }

    fn demote_tile_windows_for_batch(&mut self, batch: &PhysicalAccessBatch) {
        if self.tile_windows.is_empty() {
            return;
        }
        let space = batch.descriptor().space();
        if !matches!(space, PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem) {
            return;
        }
        for lane in batch.lanes() {
            for span in lane.footprint().spans() {
                let key = AllocationKey {
                    space,
                    allocation: span.allocation(),
                };
                if self.tile_windows.contains_key(&key) {
                    self.demote_tile_windows_overlapping(key, span.byte_offset(), span.byte_end());
                }
            }
        }
    }

    fn demote_tile_windows_for_compact_batch(&mut self, batch: &CompactPhysicalAccessBatch<'_>) {
        if self.tile_windows.is_empty() {
            return;
        }
        let space = batch.descriptor().space();
        if !matches!(space, PhysicalAccessSpace::Shared | PhysicalAccessSpace::Tmem) {
            return;
        }
        for (_, span) in batch.lane_spans() {
            let key = AllocationKey {
                space,
                allocation: span.allocation(),
            };
            if self.tile_windows.contains_key(&key) {
                self.demote_tile_windows_overlapping(key, span.byte_offset(), span.byte_end());
            }
        }
    }

    fn demote_tile_window(&mut self, key: AllocationKey, window: TileWindow) {
        let registry = &self.operation_registry;
        let allocation = key.allocation;
        let mut demoted: Vec<ShadowSegment> = Vec::new();
        for class in &window.classes {
            for (first, count) in class.mask.runs() {
                for index in first..first + count {
                    let span = window.piece_span(allocation, index);
                    demoted.push(ShadowSegment::new(
                        span.byte_offset(),
                        span.byte_end(),
                        state_for_piece(
                            &class.state,
                            registry,
                            allocation,
                            window.start,
                            window.piece,
                            index,
                        ),
                    ));
                }
            }
        }
        if demoted.is_empty() {
            return;
        }
        demoted.sort_by_key(|segment| segment.start);
        merge_adjacent_segments(&mut demoted);
        let segments = Arc::make_mut(self.allocations.entry(key).or_default());
        for segment in demoted {
            segments.insert_prepared(segment.start, segment.end, segment);
        }
    }

    /// Turn the segments inside `[start, end)` into a window of `piece`-sized
    /// pieces. Fails (leaving everything as it was) when a segment straddles
    /// the region or a piece boundary.
    fn promote_tile_window(
        &mut self,
        key: AllocationKey,
        start: usize,
        end: usize,
        piece: usize,
    ) -> Option<usize> {
        let pieces = (end - start) / piece;
        if pieces == 0 || pieces > TILE_WINDOW_MAX_PIECES || (end - start) % piece != 0 {
            return None;
        }
        let mut classes = Vec::new();
        if let Some(segments) = self.allocations.get(&key) {
            for (_, segment) in segments.iter() {
                if segment.end <= start || segment.start >= end {
                    continue;
                }
                if segment.start < start
                    || segment.end > end
                    || (segment.start - start) % piece != 0
                    || (segment.end - start) % piece != 0
                {
                    return None;
                }
                let span = PhysicalByteSpan::new(
                    key.allocation,
                    segment.start,
                    segment.end - segment.start,
                )
                .expect("a segment is a valid span");
                let mask = PieceMask::from_spans(start, piece, pieces, std::iter::once(span))
                    .expect("an aligned segment inside the region is a mask");
                classes.push(TileClass {
                    mask,
                    state: segment.state.clone(),
                });
            }
            if !classes.is_empty() {
                let segments = Arc::make_mut(
                    self.allocations
                        .get_mut(&key)
                        .expect("the allocation was just read"),
                );
                let all = std::mem::take(segments).into_sorted_values();
                for segment in all {
                    if segment.end <= start || segment.start >= end {
                        segments.insert_prepared(segment.start, segment.end, segment);
                    }
                }
            }
        }
        merge_tile_classes(
            &mut classes,
            &self.operation_registry,
            key.allocation,
            start,
            piece,
        );
        let windows = self.tile_windows.entry(key).or_default();
        let index = windows
            .iter()
            .position(|window| window.start > start)
            .unwrap_or(windows.len());
        windows.insert(
            index,
            TileWindow {
                start,
                end,
                piece,
                classes,
            },
        );
        Some(index)
    }

    /// Validate and record a clocked batch against tile windows when the batch
    /// is one lane's piece-aligned spans of shared memory (one group per
    /// allocation it touches). Returns `Ok(false)` when the batch takes the
    /// segment path instead; the windows are then demoted first so the
    /// segment map holds every byte the batch touches.
    #[allow(clippy::too_many_arguments)]
    fn tile_window_batch(
        &mut self,
        batch: &PhysicalAccessBatch,
        operation: RegisteredOperation,
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        proxy: MemoryProxy,
        proxy_domain: ProxyMemoryDomain,
        event_clock: &Arc<RaceVectorClock>,
        timestamp: &RaceEventTimestamp,
        review_findings: &mut Vec<PhysicalRaceFinding>,
    ) -> Result<bool, RaceShadowError> {
        if !self.tile_windows_enabled || !TILE_WINDOW_SPACES.contains(&space) {
            return Ok(false);
        }
        let [lane] = batch.lanes() else {
            self.demote_tile_windows_for_batch(batch);
            return Ok(false);
        };
        let spans = lane.footprint().spans();
        if spans.is_empty() {
            return Ok(false);
        }
        // Spans are sorted by allocation, so each allocation is one run.
        let mut groups: Vec<(AllocationKey, usize, usize)> = Vec::new();
        for (index, span) in spans.iter().enumerate() {
            match groups.last_mut() {
                Some((key, _, end)) if key.allocation == span.allocation() => *end = index + 1,
                _ => groups.push((
                    AllocationKey {
                        space,
                        allocation: span.allocation(),
                    },
                    index,
                    index + 1,
                )),
            }
        }
        struct Placed {
            key: AllocationKey,
            index: usize,
            window_start: usize,
            piece: usize,
            block: usize,
            mask: PieceMask,
        }
        let mut placed: Vec<Placed> = Vec::with_capacity(groups.len());
        let mut fits = true;
        for &(key, first, end) in &groups {
            let group = &spans[first..end];
            let block = group[0].byte_len();
            if block == 0 || group.iter().any(|span| span.byte_len() != block) {
                fits = false;
                break;
            }
            let start = group.iter().map(|span| span.byte_offset()).min().expect("nonempty");
            let stop = group.iter().map(|span| span.byte_end()).max().expect("nonempty");
            let index = match self.tile_window_index(&key, start, stop) {
                Some(index) => index,
                None => {
                    if group.len() < TILE_WINDOW_MIN_SPANS {
                        fits = false;
                        break;
                    }
                    let aligned_start = start - start % TILE_WINDOW_ALIGN;
                    let aligned_stop = stop.next_multiple_of(TILE_WINDOW_ALIGN);
                    if self.tile_windows.contains_key(&key) {
                        self.demote_tile_windows_overlapping(key, aligned_start, aligned_stop);
                    }
                    match self.promote_tile_window(key, aligned_start, aligned_stop, block) {
                        Some(index) => index,
                        None => {
                            fits = false;
                            break;
                        }
                    }
                }
            };
            let window = &mut self.tile_windows.get_mut(&key).expect("window")[index];
            // A finer access refines the window's pieces; a coarser one must
            // consist of whole blocks aligned to the window.
            if block % window.piece != 0 {
                let piece = gcd(window.piece, block);
                if (window.end - window.start) % piece != 0
                    || (window.end - window.start) / piece > TILE_WINDOW_MAX_PIECES
                {
                    fits = false;
                    break;
                }
                window.refine(piece);
            }
            let (window_start, piece, pieces) = (window.start, window.piece, window.pieces());
            if group
                .iter()
                .any(|span| !regular_block(*span, window_start, piece))
            {
                fits = false;
                break;
            }
            let Some(mask) = PieceMask::from_spans(window_start, piece, pieces, group.iter().copied())
            else {
                fits = false;
                break;
            };
            placed.push(Placed {
                key,
                index,
                window_start,
                piece,
                block,
                mask,
            });
        }
        if !fits {
            // Hand every touched region back to the segment map before the
            // batch takes the per-span path.
            for &(key, first, end) in &groups {
                if self.tile_windows.contains_key(&key) {
                    let group = &spans[first..end];
                    let start = group.iter().map(|span| span.byte_offset()).min().expect("nonempty");
                    let stop = group.iter().map(|span| span.byte_end()).max().expect("nonempty");
                    self.demote_tile_windows_overlapping(key, start, stop);
                }
            }
            return Ok(false);
        }
        let registry = &self.operation_registry;
        // Validation: once per class a group touches, against the class state
        // addressed to the lowest piece the group shares with it, so a finding
        // names exactly the piece the per-piece path would name (groups and
        // pieces are visited in span order, and the first race wins). Classes
        // holding only this dynamic operation's own state need none.
        for group in &placed {
            let allocation = group.key.allocation;
            let window = &self.tile_windows[&group.key][group.index];
            let block_span = |piece_index: usize| -> PhysicalByteSpan {
                let block_start =
                    group.window_start + ((piece_index * group.piece) / group.block) * group.block;
                PhysicalByteSpan::new(allocation, block_start, group.block)
                    .expect("a tile block is a valid span")
            };
            let mut race: Option<(usize, RaceShadowError)> = None;
            for class in &window.classes {
                if !class.mask.overlaps(&group.mask) {
                    continue;
                }
                let owned = registry.owns_witnesses(
                    operation,
                    class
                        .state
                        .writes
                        .iter()
                        .chain(class.state.reads.iter())
                        .map(|witness| witness.witness.operation(registry)),
                );
                if owned {
                    continue;
                }
                let piece_index = class
                    .mask
                    .first_common_piece(&group.mask)
                    .expect("the class overlaps the group");
                if race.as_ref().is_some_and(|(prior, _)| *prior <= piece_index) {
                    continue;
                }
                let state = state_for_piece(
                    &class.state,
                    registry,
                    allocation,
                    group.window_start,
                    group.piece,
                    piece_index,
                );
                let current = PhysicalRaceWitnessRef::from_lane(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    block_span(piece_index),
                )
                .with_memory_semantics(batch.descriptor().memory_semantics());
                if let Err(error) = state.validate_access(
                    registry,
                    event_clock,
                    *timestamp,
                    &current,
                    current.span(),
                    None,
                    true,
                    review_findings,
                ) {
                    race = Some((piece_index, error));
                }
            }
            if let Some((_, error)) = race {
                return Err(error);
            }
        }
        // Recording: split the classes a group touches and record once per
        // part, exactly as the per-piece path records each piece.
        for group in &placed {
            let allocation = group.key.allocation;
            let mut classes = std::mem::take(
                &mut self
                    .tile_windows
                    .get_mut(&group.key)
                    .expect("window")[group.index]
                    .classes,
            );
            let current_span = |piece_index: usize| -> PhysicalByteSpan {
                let block_start =
                    group.window_start + ((piece_index * group.piece) / group.block) * group.block;
                PhysicalByteSpan::new(allocation, block_start, group.block)
                    .expect("a tile block is a valid span")
            };
            let mut next: Vec<TileClass> = Vec::with_capacity(classes.len() + 2);
            let mut covered = PieceMask::empty(group.mask.pieces());
            for class in classes.drain(..) {
                if !class.mask.overlaps(&group.mask) {
                    next.push(class);
                    continue;
                }
                let inside = class.mask.intersection(&group.mask);
                let outside = class.mask.difference(&group.mask);
                if !outside.is_empty() {
                    next.push(TileClass {
                        mask: outside,
                        state: class.state.clone(),
                    });
                }
                let piece_index = inside.first_piece().expect("the class overlaps the group");
                let current = PhysicalRaceWitnessRef::from_lane(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    current_span(piece_index),
                )
                .with_memory_semantics(batch.descriptor().memory_semantics());
                let mut state = class.state;
                if kind.writes() {
                    state.record_validated(registry, event_clock, timestamp, current, None);
                } else if !state.records_same_operation_kind(registry, &current) {
                    state.record_validated(registry, event_clock, timestamp, current, None);
                }
                covered = covered.union(&inside);
                next.push(TileClass {
                    mask: inside,
                    state,
                });
            }
            let fresh = group.mask.difference(&covered);
            if !fresh.is_empty() {
                let piece_index = fresh.first_piece().expect("nonempty");
                let current = PhysicalRaceWitnessRef::from_lane(
                    operation,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    current_span(piece_index),
                )
                .with_memory_semantics(batch.descriptor().memory_semantics());
                let mut state = ShadowState::default();
                state.record_validated(registry, event_clock, timestamp, current, None);
                next.push(TileClass { mask: fresh, state });
            }
            merge_tile_classes(
                &mut next,
                registry,
                allocation,
                group.window_start,
                group.piece,
            );
            self.tile_windows.get_mut(&group.key).expect("window")[group.index].classes = next;
        }
        Ok(true)
    }

    fn retire_tile_windows(
        &mut self,
        observed_frontier: &RaceVectorClock,
        only: Option<&HashSet<AllocationKey>>,
        warp_count: usize,
        async_registry: &Arc<AsyncClockRegistry>,
        overflow_frontiers: &mut RetiredProxyFrontiers,
        referenced_async: &mut AsyncIndexSet,
    ) -> usize {
        let mut retired = 0;
        let Self {
            tile_windows,
            retired_generic_history,
            operation_registry,
            proxy_sensitive_allocations,
            ..
        } = self;
        for (key, windows) in tile_windows.iter_mut() {
            if only.is_some_and(|only| !only.contains(key)) {
                continue;
            }
            let proxy_sensitive = proxy_sensitive_allocations.contains(key);
            let observed = windows.iter().any(|window| {
                window.classes.iter().any(|class| {
                    class.state.has_globally_observed_witness(
                        operation_registry,
                        observed_frontier,
                        proxy_sensitive,
                    )
                })
            });
            if !observed {
                for class in windows.iter().flat_map(|window| window.classes.iter()) {
                    class
                        .state
                        .mark_referenced_async(operation_registry, referenced_async);
                }
                continue;
            }
            let mut retired_history = RetiredAllocationHistory::default();
            for window in windows.iter_mut() {
                for class in window.classes.iter_mut() {
                    let mut frontiers = RetiredProxyFrontiers::default();
                    retired += class
                        .state
                        .retire_globally_observed(
                            operation_registry,
                            observed_frontier,
                            proxy_sensitive,
                            &mut frontiers,
                            warp_count,
                            async_registry,
                            referenced_async,
                        )
                        .expect("one race shadow's retirement frontiers are compatible");
                    if !frontiers.is_empty() {
                        for (first, count) in class.mask.runs() {
                            let start = window.start + first * window.piece;
                            retired_history
                                .insert_segment(start, start + count * window.piece, &frontiers)
                                .expect("one race shadow's retirement frontiers are compatible");
                        }
                    }
                }
                window.classes.retain(|class| !class.state.is_empty());
                merge_tile_classes(
                    &mut window.classes,
                    operation_registry,
                    key.allocation,
                    window.start,
                    window.piece,
                );
            }
            retired_generic_history
                .record(*key, &retired_history, overflow_frontiers)
                .expect("one race shadow's retirement frontiers are compatible");
        }
        retired
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn clocked_multi_lane_fallback_retains_promoted_tile_writes() {
        for windows in [false, true] {
            let mut shadow = RaceShadow::new(2);
            shadow.tile_windows_enabled = windows;
            let write_op = operation(0, 1, 100, 1);
            let write_token = AsyncTokenId::new(write_op.id().clone(), 0);
            let write_clock = shadow.fork_async_token(0, &write_token).unwrap();
            let write = tile_batch(
                0,
                1,
                100,
                PhysicalAccessKind::Write,
                (0..64).map(tile_piece).collect(),
            );
            shadow
                .validate_and_commit_batches_at_clock(&[write], &write_clock, &write_token, false)
                .unwrap();
            assert_eq!(!shadow.tile_windows.is_empty(), windows);
            let read_op = operation(1, 1, 200, 3);
            let read_token = AsyncTokenId::new(read_op.id().clone(), 0);
            let read_clock = shadow.fork_async_token(1, &read_token).unwrap();
            let read = PhysicalAccessBatch::resolve_unmerged(
                read_op,
                PhysicalAccessDescriptor::new(
                    PhysicalAccessKind::Read,
                    PhysicalAccessSpace::Shared,
                    8,
                )
                .unwrap(),
                |lane| Ok::<_, Infallible>(vec![tile_piece(lane.lane())]),
            )
            .unwrap()
            .with_memory_semantics(MemoryAccessSemantics::async_proxy());
            let result = shadow.validate_and_commit_batches_at_clock(
                &[read],
                &read_clock,
                &read_token,
                false,
            );
            let error = result.expect_err(&format!(
                "unordered tile write was missed, windows={windows}"
            ));
            let conflict = finding(error);
            assert_eq!(conflict.kind(), PhysicalRaceKind::WriteRead);
            assert_eq!(conflict.overlap(), tile_piece(0));
        }
    }

    #[test]
    fn proxy_frontier_join_retains_dominating_storage_without_aliasing_mutation() {
        let registry = Arc::new(AsyncClockRegistry::default());
        let mut left = ProxyClockFrontier::empty(2, Arc::clone(&registry));
        left.components.set(0, 3);
        let left = Arc::new(left);
        let mut right = left.as_ref().clone();
        right.async_components.set(0, 5);
        right.shared_lanes = Some(Arc::new(SharedLaneEpochMap {
            rows: vec![(1, [7; WARP_SIZE])],
        }));
        let right = Arc::new(right);
        let mut current = ProxyBridgeFrontiers::default();
        current.frontiers[0] = Some(Arc::clone(&left));
        let mut incoming = ProxyBridgeFrontiers::default();
        incoming.frontiers[0] = Some(Arc::clone(&right));
        current.merge(&incoming).unwrap();
        assert!(Arc::ptr_eq(current.frontiers[0].as_ref().unwrap(), &right));
        incoming.frontiers[0] = Some(Arc::clone(&left));
        current.merge(&incoming).unwrap();
        assert!(Arc::ptr_eq(current.frontiers[0].as_ref().unwrap(), &right));

        let mut independent = left.as_ref().clone();
        independent.components.set(1, 11);
        incoming.frontiers[0] = Some(Arc::new(independent));
        current.merge(&incoming).unwrap();
        let joined = current.frontiers[0].as_ref().unwrap();
        assert_eq!(joined.components.get(0), 3);
        assert_eq!(joined.components.get(1), 11);
        assert_eq!(joined.async_components.get(0), 5);
        assert_eq!(
            joined.shared_lanes.as_ref().unwrap().get(&1),
            Some(&[7; WARP_SIZE])
        );
        assert_eq!(right.components.get(1), 0);
        assert_eq!(left.async_components.get(0), 0);
        assert!(left.shared_lanes.is_none());
    }

    #[test]
    fn proxy_slot_joins_match_independent_frontiers() {
        let registry = Arc::new(AsyncClockRegistry::default());
        let mut actual = ProxyBridgeFrontiers::default();
        let mut reference = actual.clone();
        for step in 0..16 {
            let mut first = ProxyClockFrontier::empty(2, Arc::clone(&registry));
            first.components.set(step % 2, step as u64 + 1);
            first.async_components.set(step % 3, step as u64 + 3);
            first.shared_lanes = Some(Arc::new(SharedLaneEpochMap {
                rows: vec![(1, [step as u64 + 2; WARP_SIZE])],
            }));
            let mut second = first.clone();
            second.async_components.set(1, step as u64 + 5);
            let mut incoming = ProxyBridgeFrontiers::default();
            for index in [0, 1, 4, 5, 8, 9] {
                // Distinct outer allocations can still share all their inputs.
                let frontier = if index == 5 { &second } else { &first };
                incoming.frontiers[index] = Some(Arc::new(frontier.clone()));
            }
            actual.merge(&incoming).unwrap();
            for (slot, value) in reference.frontiers.iter_mut().zip(&incoming.frontiers) {
                if let Some(value) = value {
                    match slot {
                        None => *slot = Some(Arc::clone(value)),
                        Some(current) => Arc::make_mut(current).merge(value).unwrap(),
                    }
                }
            }
            assert_eq!(actual, reference, "step {step}");
            let snapshot = actual.clone();
            Arc::make_mut(actual.frontiers[0].as_mut().unwrap())
                .components
                .set(0, 100);
            Arc::make_mut(reference.frontiers[0].as_mut().unwrap())
                .components
                .set(0, 100);
            assert_eq!(actual.frontiers[1], snapshot.frontiers[1]);
            assert_eq!(actual, reference);
        }
        let mut invalid = ProxyBridgeFrontiers::default();
        invalid.frontiers[0] = Some(Arc::new(ProxyClockFrontier::empty(
            2,
            Arc::new(AsyncClockRegistry::default()),
        )));
        assert!(matches!(
            actual.merge(&invalid),
            Err(RaceShadowError::AsyncClockRegistryMismatch)
        ));
    }

    #[test]
    fn common_release_lookup_keeps_source_warp_lane_and_publication_identity() {
        use super::{RaceLaneOrder, RetainedRaceLaneStamp, SharedClockFrontier, WARP_SIZE};
        let common = [0; WARP_SIZE];
        let observed = [[0; WARP_SIZE]; WARP_SIZE];
        let mut released = [0; WARP_SIZE];
        released[3] = 7;
        let publication = SharedClockFrontier::single(1, released);
        let empty = SharedClockFrontier::default();
        let stamp = RetainedRaceLaneStamp::new(7);
        let order = RaceLaneOrder::new(0, &common, &observed);
        assert!(!order.observes(0, 1, 3, stamp));
        let order = order.with_incoming(0, &publication, None);
        for warp in [1, 1, 2, 2, 1] {
            assert_eq!(order.observes(0, warp, 3, stamp), warp == 1);
            assert!(!order.observes(0, warp, 4, stamp));
        }
        let order = order.with_incoming(0, &empty, None);
        assert!(!order.observes(0, 1, 3, stamp));
    }

    #[test]
    fn chunked_async_epochs_match_a_dense_reference() {
        use super::{AsyncChunkJoinMemo, AsyncEpochs, ASYNC_EPOCH_CHUNK};
        let memo = AsyncChunkJoinMemo::default();
        let width = ASYNC_EPOCH_CHUNK * 3 + 7;
        let mut seed = 0x9E37_79B9_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let dense_of = |epochs: &AsyncEpochs| (0..width).map(|i| epochs.get(i)).collect::<Vec<_>>();
        let mut left = AsyncEpochs::default();
        let mut right = AsyncEpochs::default();
        let mut left_dense = vec![0_u64; width];
        let mut right_dense = vec![0_u64; width];
        for step in 0..4000 {
            let index = (next() % width as u64) as usize;
            let epoch = next() % 5;
            match next() % 6 {
                0 => {
                    left.set(index, epoch);
                    left_dense[index] = epoch;
                }
                1 => {
                    right.raise(index, epoch);
                    right_dense[index] = right_dense[index].max(epoch);
                }
                2 => {
                    left.merge(&right, &memo);
                    for (l, r) in left_dense.iter_mut().zip(&right_dense) {
                        *l = (*l).max(*r);
                    }
                }
                3 => {
                    right.meet(&left);
                    for (r, l) in right_dense.iter_mut().zip(&left_dense) {
                        *r = (*r).min(*l);
                    }
                }
                4 => {
                    right = left.clone();
                    right_dense = left_dense.clone();
                }
                _ => {
                    let shared = right.chunks.clone();
                    let len = right.len;
                    left.merge(
                        &AsyncEpochs {
                            chunks: shared,
                            len,
                        },
                        &memo,
                    );
                    for (l, r) in left_dense.iter_mut().zip(&right_dense) {
                        *l = (*l).max(*r);
                    }
                }
            }
            assert_eq!(dense_of(&left), left_dense, "step {step}");
            assert_eq!(dense_of(&right), right_dense, "step {step}");
            let dense_before = left_dense.iter().zip(&right_dense).all(|(l, r)| l <= r);
            assert_eq!(
                left.happens_before(&right, &memo),
                dense_before,
                "step {step}"
            );
            let dense_after = right_dense.iter().zip(&left_dense).all(|(r, l)| r <= l);
            assert_eq!(
                right.happens_before(&left, &memo),
                dense_after,
                "step {step}"
            );
            assert_eq!(
                left.equals(&right, &memo),
                left_dense == right_dense,
                "step {step}"
            );
        }
    }

    use std::convert::Infallible;
    use std::time::Instant;

    use crate::{
        DynamicOpId, MemoryAccessSemantics, OperationContext, OperationKind,
        PhysicalAccessDescriptor, StaticOpId, WarpMask,
    };

    use super::*;

    fn operation(warp_id: usize, sequence: u64, source: u64, mask: u32) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(0, warp_id, sequence, StaticOpId::new(source), Vec::new()),
            OperationKind::Load,
            WarpMask::from_bits(mask),
        )
    }

    fn span(allocation: u64, offset: usize, len: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(PhysicalAllocationId::new(allocation), offset, len).unwrap()
    }

    fn batch(
        warp_id: usize,
        sequence: u64,
        source: u64,
        mask: u32,
        kind: PhysicalAccessKind,
        width: usize,
        mut lane_span: impl FnMut(usize) -> PhysicalByteSpan,
    ) -> PhysicalAccessBatch {
        PhysicalAccessBatch::resolve(
            operation(warp_id, sequence, source, mask),
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, width).unwrap(),
            |lane| Ok::<_, Infallible>(vec![lane_span(lane.lane())]),
        )
        .unwrap()
    }

    fn compact_batch<'a>(
        operation: &'a OperationContext,
        kind: PhysicalAccessKind,
        width: usize,
        lane_spans: &'a [Option<PhysicalByteSpan>; WARP_SIZE],
    ) -> CompactPhysicalAccessBatch<'a> {
        CompactPhysicalAccessBatch::new(
            operation,
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, width).unwrap(),
            None,
            false,
            lane_spans,
        )
    }

    fn single(
        warp_id: usize,
        sequence: u64,
        source: u64,
        kind: PhysicalAccessKind,
        allocation: u64,
        offset: usize,
        len: usize,
    ) -> PhysicalAccessBatch {
        batch(warp_id, sequence, source, 1, kind, len, |_| {
            span(allocation, offset, len)
        })
    }

    fn single_proxy(
        warp_id: usize,
        sequence: u64,
        source: u64,
        kind: PhysicalAccessKind,
        allocation: u64,
        proxy: MemoryProxy,
        domain: ProxyMemoryDomain,
    ) -> PhysicalAccessBatch {
        let semantics = match proxy {
            MemoryProxy::Generic => MemoryAccessSemantics::plain(),
            MemoryProxy::Async => MemoryAccessSemantics::async_proxy(),
            MemoryProxy::Mmio | MemoryProxy::MulticastAlias => {
                unreachable!("proxy-async tests do not use MMIO or multicast aliases")
            }
        };
        single(warp_id, sequence, source, kind, allocation, 0, 4)
            .with_memory_semantics(semantics)
            .with_proxy_memory_domain(domain)
    }

    fn single_proxy_lane(
        warp_id: usize,
        lane: usize,
        sequence: u64,
        source: u64,
        kind: PhysicalAccessKind,
        allocation: u64,
        proxy: MemoryProxy,
        domain: ProxyMemoryDomain,
    ) -> PhysicalAccessBatch {
        let semantics = match proxy {
            MemoryProxy::Generic => MemoryAccessSemantics::plain(),
            MemoryProxy::Async => MemoryAccessSemantics::async_proxy(),
            MemoryProxy::Mmio | MemoryProxy::MulticastAlias => {
                unreachable!("proxy-async tests do not use MMIO or multicast aliases")
            }
        };
        batch(warp_id, sequence, source, 1_u32 << lane, kind, 4, |_| {
            span(allocation, 0, 4)
        })
        .with_memory_semantics(semantics)
        .with_proxy_memory_domain(domain)
    }

    fn finding(error: RaceShadowError) -> PhysicalRaceFinding {
        match error {
            RaceShadowError::Race(finding) => finding,
            other => panic!("expected race finding, got {other}"),
        }
    }

    /// One lane's async-proxy access made of many shared-memory spans.
    fn tile_batch(
        warp_id: usize,
        sequence: u64,
        source: u64,
        kind: PhysicalAccessKind,
        spans: Vec<PhysicalByteSpan>,
    ) -> PhysicalAccessBatch {
        let width = spans.iter().map(|span| span.byte_len()).sum();
        PhysicalAccessBatch::resolve_unmerged(
            operation(warp_id, sequence, source, 1),
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, width).unwrap(),
            |_| Ok::<_, Infallible>(spans.clone()),
        )
        .unwrap()
        .with_memory_semantics(MemoryAccessSemantics::async_proxy())
    }

    fn tile_piece(index: usize) -> PhysicalByteSpan {
        span(6, 1024 + 16 * index, 8)
    }

    /// Drive the same clocked tile traffic through a windowed and a plain
    /// shadow and return what every per-piece probe reports afterwards.
    fn run_tile_traffic(windows: bool) -> (RaceShadow, Vec<String>, String) {
        let mut shadow = RaceShadow::new(2);
        shadow.tile_windows_enabled = windows;
        let mut sequence = 1;
        let mut clocked =
            |shadow: &mut RaceShadow, warp: usize, source: u64, kind, pieces: Vec<usize>| {
                let issue = operation(warp, sequence, source, 1);
                let token = AsyncTokenId::new(issue.id().clone(), 0);
                let clock = shadow.fork_async_token(warp, &token).unwrap();
                let batch = tile_batch(
                    warp,
                    sequence,
                    source,
                    kind,
                    pieces.into_iter().map(tile_piece).collect(),
                );
                sequence += 1;
                let reviews = shadow
                    .validate_and_commit_batches_at_clock(&[batch], &clock, &token, false)
                    .map(|reviews| reviews.len())
                    .map_err(|error| format!("{:?}", finding(error)))?;
                // The transfer or MMA completes and its warp waits for it, as the
                // pipeline's barriers do, so the warp's next access is ordered.
                let completed = shadow
                    .complete_async_actors(std::slice::from_ref(&token))
                    .unwrap()
                    .pop()
                    .unwrap();
                shadow
                    .barrier_acquire(warp, &BarrierClockPayload::from_clock(completed))
                    .unwrap();
                Ok::<usize, String>(reviews)
            };
        // Stage 1: the transfer writes every padded piece, then four K-slice reads.
        clocked(
            &mut shadow,
            0,
            101,
            PhysicalAccessKind::Write,
            (0..1024).collect(),
        )
        .unwrap();
        for k in 0..4 {
            clocked(
                &mut shadow,
                0,
                110 + k as u64,
                PhysicalAccessKind::Read,
                (0..1024).filter(|index| index % 4 == k).collect(),
            )
            .unwrap();
        }
        // A partial re-read and a re-write of the lower half by the same warp.
        clocked(
            &mut shadow,
            0,
            120,
            PhysicalAccessKind::Read,
            (0..512).collect(),
        )
        .unwrap();
        clocked(
            &mut shadow,
            0,
            121,
            PhysicalAccessKind::Write,
            (0..512).collect(),
        )
        .unwrap();
        // A second tile in another allocation, then one MMA operand read that
        // spans both allocations, then a whole-region single-span rewrite of
        // the first tile (the contiguous-transfer form).
        {
            let issue = operation(0, 900, 140, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let clock = shadow.fork_async_token(0, &token).unwrap();
            let batch = tile_batch(
                0,
                900,
                140,
                PhysicalAccessKind::Write,
                (0..1024).map(|index| span(7, 4096 + 16 * index, 8)).collect(),
            );
            shadow
                .validate_and_commit_batches_at_clock(&[batch], &clock, &token, false)
                .unwrap();
            let completed = shadow
                .complete_async_actors(std::slice::from_ref(&token))
                .unwrap()
                .pop()
                .unwrap();
            shadow
                .barrier_acquire(0, &BarrierClockPayload::from_clock(completed))
                .unwrap();
            let issue = operation(0, 901, 141, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let clock = shadow.fork_async_token(0, &token).unwrap();
            let mut spans = (0..1024)
                .filter(|index| index % 4 == 1)
                .map(tile_piece)
                .collect::<Vec<_>>();
            spans.extend(
                (0..1024)
                    .filter(|index| index % 4 == 2)
                    .map(|index| span(7, 4096 + 16 * index, 8)),
            );
            let batch = tile_batch(0, 901, 141, PhysicalAccessKind::Read, spans);
            shadow
                .validate_and_commit_batches_at_clock(&[batch], &clock, &token, false)
                .unwrap();
            let completed = shadow
                .complete_async_actors(std::slice::from_ref(&token))
                .unwrap()
                .pop()
                .unwrap();
            shadow
                .barrier_acquire(0, &BarrierClockPayload::from_clock(completed))
                .unwrap();
            let issue = operation(0, 902, 142, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let clock = shadow.fork_async_token(0, &token).unwrap();
            let batch = tile_batch(
                0,
                902,
                142,
                PhysicalAccessKind::Write,
                vec![span(6, 1024, 1023 * 16 + 8)],
            );
            shadow
                .validate_and_commit_batches_at_clock(&[batch], &clock, &token, false)
                .unwrap();
            let completed = shadow
                .complete_async_actors(std::slice::from_ref(&token))
                .unwrap()
                .pop()
                .unwrap();
            shadow
                .barrier_acquire(0, &BarrierClockPayload::from_clock(completed))
                .unwrap();
        }
        clocked(
            &mut shadow,
            0,
            143,
            PhysicalAccessKind::Read,
            (0..1024).filter(|index| index % 4 == 3).collect(),
        )
        .unwrap();
        // Another warp reads a slice without synchronization: a race the two
        // paths must report identically.
        let racing = clocked(
            &mut shadow,
            1,
            130,
            PhysicalAccessKind::Read,
            (256..768).filter(|index| index % 2 == 1).collect(),
        )
        .unwrap_err();
        let mut probes = Vec::with_capacity(1024);
        for index in 0..1024 {
            let mut probe = shadow.clone();
            let outcome = probe
                .check_batch(single(
                    1,
                    500 + index as u64,
                    300,
                    PhysicalAccessKind::Write,
                    6,
                    1024 + 16 * index,
                    8,
                ))
                .err()
                .map(|error| format!("{:?}", finding(error)));
            probes.push(format!("{outcome:?}"));
        }
        (shadow, probes, racing)
    }

    /// One lane's generic-proxy TMEM access made of many spans.
    fn tmem_batch(
        warp_id: usize,
        sequence: u64,
        source: u64,
        kind: PhysicalAccessKind,
        spans: Vec<PhysicalByteSpan>,
    ) -> PhysicalAccessBatch {
        let width = spans.iter().map(|span| span.byte_len()).sum();
        PhysicalAccessBatch::resolve_unmerged(
            operation(warp_id, sequence, source, 1),
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Tmem, width).unwrap(),
            |_| Ok::<_, Infallible>(spans.clone()),
        )
        .unwrap()
    }

    /// Accumulator rows (64 B at a 2 KB lane pitch), narrower row writes that
    /// refine the window, and 1 B scale reads, through both shadow forms.
    fn run_tmem_traffic(windows: bool) -> (RaceShadow, Vec<String>, String) {
        let mut shadow = RaceShadow::new(2);
        shadow.tile_windows_enabled = windows;
        let mut sequence = 1;
        let mut clocked = |shadow: &mut RaceShadow, warp: usize, source: u64, kind, spans: Vec<PhysicalByteSpan>| {
            let issue = operation(warp, sequence, source, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let clock = shadow.fork_async_token(warp, &token).unwrap();
            let batch = tmem_batch(warp, sequence, source, kind, spans);
            sequence += 1;
            let reviews = shadow
                .validate_and_commit_batches_at_clock(&[batch], &clock, &token, false)
                .map(|reviews| reviews.len())
                .map_err(|error| format!("{:?}", finding(error)))?;
            let completed = shadow
                .complete_async_actors(std::slice::from_ref(&token))
                .unwrap()
                .pop()
                .unwrap();
            shadow
                .barrier_acquire(warp, &BarrierClockPayload::from_clock(completed))
                .unwrap();
            Ok::<usize, String>(reviews)
        };
        let rows = |len: usize, column: usize| -> Vec<PhysicalByteSpan> {
            (0..128).map(|row| span(9, row * 2048 + column, len)).collect()
        };
        for step in 0..3 {
            // Read-modify-write of the accumulator, then a 16 B column write
            // (refines the 64 B window to 16 B pieces), then scale reads.
            clocked(
                &mut shadow,
                0,
                200 + step,
                PhysicalAccessKind::Read,
                rows(64, 0),
            )
            .unwrap();
            clocked(
                &mut shadow,
                0,
                210 + step,
                PhysicalAccessKind::Write,
                rows(64, 0),
            )
            .unwrap();
            clocked(
                &mut shadow,
                0,
                220 + step,
                PhysicalAccessKind::Write,
                rows(16, 32),
            )
            .unwrap();
            clocked(
                &mut shadow,
                0,
                230 + step,
                PhysicalAccessKind::Read,
                (0..256).map(|index| span(10, 4096 + 4 * index, 1)).collect(),
            )
            .unwrap();
        }
        // A second warp reads the accumulator rows without synchronization.
        let racing = clocked(&mut shadow, 1, 240, PhysicalAccessKind::Read, rows(64, 0)).unwrap_err();
        let mut probes = Vec::with_capacity(128 * 4);
        for row in 0..128 {
            for column in [0usize, 16, 32, 48] {
                let mut probe = shadow.clone();
                let outcome = probe
                    .check_batch(
                        single(
                            1,
                            600 + (row * 4 + column / 16) as u64,
                            301,
                            PhysicalAccessKind::Write,
                            9,
                            row * 2048 + column,
                            16,
                        )
                        .with_memory_semantics(MemoryAccessSemantics::plain()),
                    )
                    .err()
                    .map(|error| format!("{:?}", finding(error)));
                probes.push(format!("{outcome:?}"));
            }
        }
        (shadow, probes, racing)
    }

    #[test]
    fn tmem_byte_reads_preserve_neighbor_witnesses_and_cloned_shadow() {
        for general in [false, true] {
            for byte in 0..4 {
                let mut shadow = RaceShadow::new(2);
                shadow
                    .check_batch(tmem_batch(
                        0,
                        1,
                        100,
                        PhysicalAccessKind::Write,
                        vec![span(9, 0, 4)],
                    ))
                    .unwrap();
                let published = shadow.barrier_release(0).unwrap();
                shadow.barrier_acquire(1, &published).unwrap();
                if general {
                    for segments in shadow.allocations.values_mut() {
                        Arc::make_mut(segments).general_mut();
                    }
                }
                let before = shadow.clone();
                let issue = operation(0, 2, 101, 1);
                let token = AsyncTokenId::new(issue.id().clone(), 0);
                let clock = shadow.fork_async_token(0, &token).unwrap();
                let read = tmem_batch(0, 2, 101, PhysicalAccessKind::Read, vec![span(9, byte, 1)]);
                assert!(shadow
                    .validate_and_commit_batches_at_clock(&[read], &clock, &token, false)
                    .unwrap()
                    .is_empty());
                for probe_byte in 0..4 {
                    let write = tmem_batch(
                        1,
                        3,
                        102,
                        PhysicalAccessKind::Write,
                        vec![span(9, probe_byte, 1)],
                    );
                    // The old clone contains only the already acquired wide write.
                    before.clone().check_batch(write.clone()).unwrap();
                    let outcome = shadow.clone().check_batch(write);
                    if probe_byte == byte {
                        let conflict = finding(outcome.unwrap_err());
                        assert_eq!(conflict.overlap(), span(9, byte, 1));
                    } else {
                        outcome.unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn tmem_tile_windows_match_per_piece_segments() {
        let (mut windowed, windowed_probes, windowed_race) = run_tmem_traffic(true);
        let (plain, plain_probes, plain_race) = run_tmem_traffic(false);
        assert_eq!(
            !windowed.tile_windows.is_empty(),
            TILE_WINDOW_SPACES.contains(&PhysicalAccessSpace::Tmem),
            "TMEM traffic promotes a window exactly when TMEM windows are enabled"
        );
        assert_eq!(windowed_race, plain_race);
        for (index, (left, right)) in windowed_probes.iter().zip(plain_probes.iter()).enumerate() {
            assert_eq!(left, right, "probe {index} differs");
        }
        windowed.demote_all_tile_windows();
        let mut left = windowed
            .allocations
            .values()
            .flat_map(|s| {
                s.iter()
                    .map(|(_, seg)| (seg.start, seg.end, format!("{:?}", seg.state)))
            })
            .collect::<Vec<_>>();
        let mut right = plain
            .allocations
            .values()
            .flat_map(|s| {
                s.iter()
                    .map(|(_, seg)| (seg.start, seg.end, format!("{:?}", seg.state)))
            })
            .collect::<Vec<_>>();
        left.sort(); right.sort();
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            assert_eq!(l, r);
        }
    }

    #[test]
    fn tile_windows_match_per_piece_segments() {
        let (mut windowed, windowed_probes, windowed_race) = run_tile_traffic(true);
        let (plain, plain_probes, plain_race) = run_tile_traffic(false);
        assert!(!windowed.tile_windows.is_empty(), "the tile traffic promotes a window");
        assert_eq!(windowed_race, plain_race);
        for (index, (left, right)) in windowed_probes.iter().zip(plain_probes.iter()).enumerate() {
            assert_eq!(left, right, "probe on piece {index} differs");
        }
        windowed.demote_all_tile_windows();
        assert!(windowed.tile_windows.is_empty());
        let mut left = windowed
            .allocations
            .values()
            .flat_map(|s| {
                s.iter()
                    .map(|(_, seg)| (seg.start, seg.end, format!("{:?}", seg.state)))
            })
            .collect::<Vec<_>>();
        let mut right = plain
            .allocations
            .values()
            .flat_map(|s| {
                s.iter()
                    .map(|(_, seg)| (seg.start, seg.end, format!("{:?}", seg.state)))
            })
            .collect::<Vec<_>>();
        left.sort(); right.sort();
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            assert_eq!(l, r);
        }
    }

    #[test]
    fn retained_witness_round_trips_compact_and_wide_values() {
        let registry = OperationRegistry::for_warp_range(100);
        for (operation, access_span, lane, kind, space, proxy, proxy_domain, wide) in [
            (
                RegisteredOperation::new(7, 103),
                span(7, 32, 16),
                3,
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                MemoryProxy::Async,
                ProxyMemoryDomain::SharedCluster,
                false,
            ),
            (
                RegisteredOperation::new(9, 104),
                span(8, 65_536, 32_768),
                17,
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCta,
                false,
            ),
            (
                RegisteredOperation::new(11, 105),
                span(9, RetainedRaceWitness::BYTE_OFFSET_MASK as usize + 7, 8),
                31,
                PhysicalAccessKind::AtomicReadModifyWrite,
                PhysicalAccessSpace::Tmem,
                MemoryProxy::Generic,
                ProxyMemoryDomain::Other,
                true,
            ),
            (
                RegisteredOperation::new(13, 106),
                span(10, 96, 129),
                5,
                PhysicalAccessKind::Read,
                PhysicalAccessSpace::Shared,
                MemoryProxy::Async,
                ProxyMemoryDomain::SharedCta,
                true,
            ),
        ] {
            let retained = RetainedRaceWitness::new(
                operation,
                access_span,
                lane,
                kind,
                space,
                proxy,
                proxy_domain,
                None,
                &registry,
            );
            assert_eq!(retained.0.get() & RetainedRaceWitness::WIDE_BIT != 0, wide);
            assert_eq!(
                retained.resolve(&registry, access_span.allocation()),
                WideRetainedWitness {
                    operation,
                    span: access_span,
                    lane,
                    kind,
                    space,
                    proxy,
                    proxy_domain,
                    strong_scope: None,
                }
            );
        }
    }

    #[test]
    fn event_timestamp_round_trips_compact_and_wide_epochs() {
        let registry = OperationRegistry::default();
        for (actor, epoch) in [
            (TimestampActor::Warp(3), 17),
            (TimestampActor::Warp(7), 1_u64 << 40),
            (TimestampActor::Async(5), 19),
            (TimestampActor::Async(7), 1_u64 << 40),
            (TimestampActor::RegisteredVector(11), 23),
        ] {
            let timestamp = RaceEventTimestamp::new(actor, epoch, &registry);
            assert_eq!(timestamp.resolve(&registry), (actor, epoch));
        }
        for (event_epoch, lane_epoch) in [(17, 29), (1_u64 << 40, 1_u64 << 41)] {
            let event = RaceEventTimestamp::new(TimestampActor::Warp(3), event_epoch, &registry);
            let stamped = event.with_lane_stamp(
                Some(RetainedRaceLaneStamp::new(lane_epoch)),
                5,
                PhysicalAccessKind::Read,
                &registry,
            );
            assert_eq!(
                stamped.resolve(&registry),
                (TimestampActor::Warp(3), event_epoch)
            );
            assert_eq!(
                stamped.lane_stamp(&registry),
                Some(RetainedRaceLaneStamp::new(lane_epoch))
            );
            assert!(stamped.same_event(event, &registry));
        }
        let compact_event = RaceEventTimestamp::new(TimestampActor::Warp(3), 17, &registry);
        for (lane, kind) in [
            (5, PhysicalAccessKind::Read),
            (31, PhysicalAccessKind::Write),
            (7, PhysicalAccessKind::AtomicReadModifyWrite),
        ] {
            let lane_stamp = RetainedRaceLaneStamp::new(29);
            let stamped = compact_event.with_lane_stamp(Some(lane_stamp), lane, kind, &registry);
            assert_eq!(
                stamped.compact_direct_warp_components(),
                Some((3, 17, lane_stamp, lane, kind))
            );
            assert!(stamped.wide_direct_warp_components(&registry).is_none());
        }
        assert_eq!(std::mem::size_of::<RetainedRaceWitness>(), 8);
        assert_eq!(std::mem::size_of::<ClockedWitness>(), 16);
        assert_eq!(std::mem::size_of::<Option<ClockedWitness>>(), 16);
        assert_eq!(std::mem::size_of::<AccessFrontier>(), 24);
        // Both read and write frontiers keep their common single witness inline.
        // Unordered strong writes cannot share the old one-writer slot.
        assert_eq!(std::mem::size_of::<ShadowState>(), 48);
        assert_eq!(std::mem::size_of::<ShadowSegment>(), 64);
        assert_eq!(std::mem::size_of::<RegisteredOperationMetadata>(), 12);
    }

    #[test]
    fn lane_ordered_shadow_detects_unordered_same_warp_lanes() {
        let mut shadow = RaceShadow::new(1);
        let mut observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let first = batch(0, 0, 10, 1, PhysicalAccessKind::Write, 4, |_| span(1, 0, 4));
        let common = [0; WARP_SIZE];
        let first_order = RaceLaneOrder::new(0, &common, &observed);
        let validation = shadow
            .validate_batch_with_lane_order(&first, &first_order)
            .unwrap();
        shadow.commit_validation(validation);
        observed[0][0] = 1;

        let second = batch(0, 1, 11, 2, PhysicalAccessKind::Read, 4, |_| span(1, 0, 4));
        let second_order = RaceLaneOrder::new(0, &common, &observed);
        let race = match shadow.validate_batch_with_lane_order(&second, &second_order) {
            Ok(_) => panic!("unordered same-warp lanes must race"),
            Err(error) => error,
        };
        assert_eq!(finding(race).kind(), PhysicalRaceKind::WriteRead);

        observed[1][0] = 1;
        let acquired_order = RaceLaneOrder::new(0, &common, &observed);
        shadow
            .validate_batch_with_lane_order(&second, &acquired_order)
            .unwrap();
    }

    #[test]
    fn direct_segment_retains_exact_source_evidence_without_rebuilding_same_source() {
        let mut shadow = RaceShadow::new(2);
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let common = [0; WARP_SIZE];
        let order = RaceLaneOrder::new(0, &common, &observed);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        let mut segment = None;
        for (sequence, source) in [(0, 10), (1, 10), (2, 11)] {
            let operation = operation(0, sequence, source, 1);
            let batch = compact_batch(&operation, PhysicalAccessKind::Write, 4, &spans);
            let geometry = shadow.compact_direct_geometry(&batch).unwrap();
            let (new_segment, reviews) = shadow
                .apply_compact_batch_after_numeric(&batch, &order, Some(geometry), segment.as_ref())
                .unwrap();
            assert!(reviews.is_empty());
            if segment.is_none() {
                segment = new_segment;
            } else {
                assert!(new_segment.is_none());
            }
        }
        shadow.commit_direct_segment(segment.unwrap());

        let reader = single(1, 0, 12, PhysicalAccessKind::Read, 1, 0, 4);
        let race = match shadow.validate_batch(&reader) {
            Ok(_) => panic!("unordered warp write and read must race"),
            Err(error) => finding(error),
        };
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.prior().operation().source_op_id().get(), 11);
    }

    #[test]
    fn direct_segment_still_detects_unordered_lanes_in_one_warp() {
        let mut shadow = RaceShadow::new(1);
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let common = [0; WARP_SIZE];
        let order = RaceLaneOrder::new(0, &common, &observed);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        let writer_op = operation(0, 0, 10, 1);
        let writer = compact_batch(&writer_op, PhysicalAccessKind::Write, 4, &spans);
        let geometry = shadow.compact_direct_geometry(&writer).unwrap();
        let (segment, reviews) = shadow
            .apply_compact_batch_after_numeric(&writer, &order, Some(geometry), None)
            .unwrap();
        assert!(reviews.is_empty());
        assert!(segment.is_some());

        spans.swap(0, 1);
        let reader_op = operation(0, 1, 11, 2);
        let reader = compact_batch(&reader_op, PhysicalAccessKind::Read, 4, &spans);
        let geometry = shadow.compact_direct_geometry(&reader).unwrap();
        let error = shadow.apply_compact_batch_after_numeric(
            &reader,
            &order,
            Some(geometry),
            segment.as_ref(),
        );
        let race = match error {
            Ok(_) => panic!("unordered same-warp lanes must race"),
            Err(error) => finding(error),
        };
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
    }

    #[test]
    fn postnumeric_duplicate_lane_groups_match_transactional_commit() {
        let base = RaceShadow::new(2);
        let mut postnumeric = base.clone();
        let mut transactional = base;
        let operation = operation(0, 0, 12, 0b1111);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        spans[1] = Some(span(1, 0, 4));
        spans[2] = Some(span(1, 8, 4));
        spans[3] = Some(span(1, 8, 4));
        let batch = compact_batch(&operation, PhysicalAccessKind::Read, 4, &spans);
        let common = [0; WARP_SIZE];
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let lane_order = RaceLaneOrder::new(0, &common, &observed);

        let geometry = postnumeric
            .compact_direct_geometry(&batch)
            .expect("exact duplicate groups have direct geometry");
        let (segment, reviews) = postnumeric
            .apply_compact_batch_after_numeric(&batch, &lane_order, Some(geometry), None)
            .unwrap();
        assert!(reviews.is_empty());
        postnumeric.commit_direct_segment(segment.expect("a first direct batch starts a segment"));

        let validation = transactional
            .validate_compact_batch_with_lane_order(&batch, &lane_order)
            .unwrap();
        assert!(transactional.commit_validation(validation).is_empty());

        assert_eq!(postnumeric.warp_clocks, transactional.warp_clocks);
        assert_eq!(
            postnumeric.tracked_interval_count(),
            transactional.tracked_interval_count()
        );
        let conflicting = single(1, 0, 16, PhysicalAccessKind::Write, 1, 0, 4);
        let postnumeric_race = finding(postnumeric.check_batch(conflicting.clone()).unwrap_err());
        let transactional_race = finding(transactional.check_batch(conflicting).unwrap_err());
        assert_eq!(postnumeric_race, transactional_race);

        // The first aliased reader cannot stand in for its unsynchronized peer.
        let overwrite = single(0, 1, 17, PhysicalAccessKind::Write, 1, 0, 4);
        for shadow in [&mut postnumeric, &mut transactional] {
            let error = shadow.validate_batch_with_lane_order(&overwrite, &lane_order);
            let race = match error {
                Ok(_) => panic!("lane 1's read must remain visible to lane 0's write"),
                Err(error) => finding(error),
            };
            assert_eq!(race.kind(), PhysicalRaceKind::ReadWrite);
            assert_eq!(race.prior().lane(), 1);
            assert_eq!(race.current().lane(), 0);
        }
    }

    #[test]
    fn postnumeric_disjoint_lanes_crossing_prior_boundaries_match_transactional_commit() {
        let mut base = RaceShadow::new(3);
        base.check_batch(single(0, 0, 10, PhysicalAccessKind::Write, 1, 0, 2))
            .unwrap();
        base.check_batch(single(0, 1, 11, PhysicalAccessKind::Write, 1, 2, 2))
            .unwrap();
        let release = base.barrier_release(0).unwrap();
        base.barrier_acquire(1, &release).unwrap();

        let mut postnumeric = base.clone();
        let mut transactional = base;
        let operation = operation(1, 0, 12, 0b11);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        spans[1] = Some(span(1, 8, 4));
        let batch = compact_batch(&operation, PhysicalAccessKind::Read, 4, &spans);
        let common = [0; WARP_SIZE];
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let lane_order = RaceLaneOrder::new(1, &common, &observed);

        let geometry = postnumeric
            .compact_direct_geometry(&batch)
            .expect("disjoint current lanes are safe across historical shadow boundaries");
        let (segment, reviews) = postnumeric
            .apply_compact_batch_after_numeric(&batch, &lane_order, Some(geometry), None)
            .unwrap();
        assert!(reviews.is_empty());
        postnumeric.commit_direct_segment(segment.expect("a first direct batch starts a segment"));

        let validation = transactional
            .validate_compact_batch_with_lane_order(&batch, &lane_order)
            .unwrap();
        assert!(transactional.commit_validation(validation).is_empty());
        assert_eq!(postnumeric.warp_clocks, transactional.warp_clocks);
        assert_eq!(
            postnumeric.tracked_interval_count(),
            transactional.tracked_interval_count()
        );

        let conflicting = single(2, 0, 13, PhysicalAccessKind::Write, 1, 0, 12);
        let postnumeric_race = finding(postnumeric.check_batch(conflicting.clone()).unwrap_err());
        let transactional_race = finding(transactional.check_batch(conflicting).unwrap_err());
        assert_eq!(postnumeric_race.kind(), transactional_race.kind());
        assert_eq!(postnumeric_race.overlap(), transactional_race.overlap());
        assert_eq!(
            postnumeric_race.prior().operation().source_op_id(),
            transactional_race.prior().operation().source_op_id()
        );
    }

    #[test]
    fn postnumeric_nonexact_write_collapses_historical_boundaries() {
        let mut base = RaceShadow::new(3);
        base.check_batch(single(0, 0, 10, PhysicalAccessKind::Write, 1, 0, 2))
            .unwrap();
        base.check_batch(single(0, 1, 11, PhysicalAccessKind::Write, 1, 2, 2))
            .unwrap();
        let release = base.barrier_release(0).unwrap();
        base.barrier_acquire(1, &release).unwrap();

        let mut postnumeric = base.clone();
        let mut transactional = base;
        let operation = operation(1, 0, 12, 1);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        let batch = compact_batch(&operation, PhysicalAccessKind::Write, 4, &spans);
        let common = [0; WARP_SIZE];
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let lane_order = RaceLaneOrder::new(1, &common, &observed);

        let geometry = postnumeric
            .compact_direct_geometry(&batch)
            .expect("one exact lane has direct geometry");
        let (segment, reviews) = postnumeric
            .apply_compact_batch_after_numeric(&batch, &lane_order, Some(geometry), None)
            .unwrap();
        assert!(reviews.is_empty());
        postnumeric.commit_direct_segment(segment.expect("a first direct batch starts a segment"));
        let committed = postnumeric
            .allocations
            .get(&AllocationKey {
                space: PhysicalAccessSpace::Shared,
                allocation: PhysicalAllocationId::new(1),
            })
            .expect("the write retained its allocation");
        assert_eq!(
            committed.get(0).map(|segment| (segment.start, segment.end)),
            Some((0, 4))
        );

        let validation = transactional
            .validate_compact_batch_with_lane_order(&batch, &lane_order)
            .unwrap();
        assert!(transactional.commit_validation(validation).is_empty());

        let conflicting = single(2, 0, 13, PhysicalAccessKind::Read, 1, 0, 4);
        let postnumeric_race = finding(postnumeric.check_batch(conflicting.clone()).unwrap_err());
        let transactional_race = finding(transactional.check_batch(conflicting).unwrap_err());
        assert_eq!(postnumeric_race, transactional_race);
        assert_eq!(
            postnumeric_race.prior().operation().source_op_id().get(),
            12
        );
    }

    #[test]
    fn postnumeric_duplicate_nonexact_write_retains_both_lane_witnesses() {
        let mut base = RaceShadow::new(3);
        base.check_batch(single(0, 0, 10, PhysicalAccessKind::Read, 1, 0, 2))
            .unwrap();
        base.check_batch(single(0, 1, 11, PhysicalAccessKind::Read, 1, 2, 2))
            .unwrap();
        let release = base.barrier_release(0).unwrap();
        base.barrier_acquire(1, &release).unwrap();

        let mut postnumeric = base.clone();
        let operation = operation(1, 0, 12, 0b11);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        spans[1] = Some(span(1, 0, 4));
        let batch = compact_batch(&operation, PhysicalAccessKind::Write, 4, &spans);
        let common = [0; WARP_SIZE];
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let lane_order = RaceLaneOrder::new(1, &common, &observed);

        let geometry = postnumeric
            .compact_direct_geometry(&batch)
            .expect("exact duplicate lanes have direct geometry");
        let (segment, reviews) = postnumeric
            .apply_compact_batch_after_numeric(&batch, &lane_order, Some(geometry), None)
            .unwrap();
        assert!(reviews.is_empty());
        postnumeric.commit_direct_segment(segment.expect("a first direct batch starts a segment"));

        let conflicting = single(2, 0, 13, PhysicalAccessKind::Read, 1, 0, 4);
        let race = finding(postnumeric.check_batch(conflicting).unwrap_err());
        assert_eq!(race.prior().operation().source_op_id().get(), 12);
        assert_eq!(race.prior().lane(), 0);

        // Each duplicate lane retains its own history. A later read by either
        // writer must still conflict with the other writer, not a group proxy.
        for lane in 0..2 {
            let read = self::batch(
                1, 1, 14, 1 << lane, PhysicalAccessKind::Read, 4, |_| span(1, 0, 4),
            );
            let race = match postnumeric.validate_batch_with_lane_order(&read, &lane_order) {
                Ok(_) => panic!("the peer lane's write must remain visible"),
                Err(error) => finding(error),
            };
            assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
            assert_eq!(race.prior().operation().source_op_id().get(), 12);
            assert_eq!(race.prior().lane(), 1 - lane);
            assert_eq!(race.current().lane(), lane);
            assert_eq!(race.overlap(), span(1, 0, 4));
        }
    }

    #[test]
    fn postnumeric_duplicate_groups_detect_a_late_partial_overlap_race() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 13, PhysicalAccessKind::Write, 1, 0, 4))
            .unwrap();
        let release = shadow.barrier_release(0).unwrap();
        shadow.barrier_acquire(1, &release).unwrap();
        shadow
            .check_batch(single(0, 1, 14, PhysicalAccessKind::Write, 1, 8, 8))
            .unwrap();

        let operation = operation(1, 0, 15, 0b1111);
        let mut spans = [None; WARP_SIZE];
        spans[0] = Some(span(1, 0, 4));
        spans[1] = Some(span(1, 0, 4));
        spans[2] = Some(span(1, 10, 4));
        spans[3] = Some(span(1, 10, 4));
        let batch = compact_batch(&operation, PhysicalAccessKind::Read, 4, &spans);
        let common = [0; WARP_SIZE];
        let observed = [[0_u64; WARP_SIZE]; WARP_SIZE];
        let lane_order = RaceLaneOrder::new(1, &common, &observed);
        let geometry = shadow
            .compact_direct_geometry(&batch)
            .expect("the first exact group permits the guarded direct path");

        let race = match shadow.apply_compact_batch_after_numeric(
            &batch,
            &lane_order,
            Some(geometry),
            None,
        ) {
            Ok(_) => panic!("the late partially overlapping group must race"),
            Err(error) => finding(error),
        };
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.overlap(), span(1, 10, 4));
    }

    struct ReferenceBatchValidation {
        warp_id: usize,
        event_clock: Arc<RaceVectorClock>,
        allocation_updates: BTreeMap<AllocationKey, Vec<ShadowSegment>>,
    }

    fn reference_validate_batch(
        shadow: &RaceShadow,
        batch: &PhysicalAccessBatch,
    ) -> Result<ReferenceBatchValidation, RaceShadowError> {
        let warp_id = batch.operation().id().global_warp_id();
        let local_warp_id = shadow.local_warp_id(warp_id)?;
        let mut event_clock = shadow.warp_clocks[local_warp_id].clone();
        event_clock.tick(local_warp_id)?;
        let event_clock = Arc::new(event_clock);
        let timestamp =
            RaceEventTimestamp::for_warp(&event_clock, local_warp_id, &shadow.operation_registry);
        let descriptor = batch.descriptor();
        let mut allocation_updates = BTreeMap::new();
        if tracks_race_conflicts(descriptor.space()) {
            for lane in batch.lanes() {
                for span in lane.footprint().spans().iter().copied() {
                    let key = AllocationKey {
                        space: descriptor.space(),
                        allocation: span.allocation(),
                    };
                    let existing = shadow
                        .allocations
                        .get(&key)
                        .map(|segments| segments.as_ref().clone().into_sorted_values())
                        .unwrap_or_default();
                    let segments = allocation_updates.entry(key).or_insert(existing);
                    apply_access_to_span_in_place(
                        &shadow.operation_registry,
                        segments,
                        lane,
                        descriptor.kind(),
                        descriptor.space(),
                        span,
                        &event_clock,
                        &timestamp,
                    )?;
                }
            }
        }
        Ok(ReferenceBatchValidation {
            warp_id: local_warp_id,
            event_clock,
            allocation_updates,
        })
    }

    fn materialize_sparse_validation(
        shadow: &RaceShadow,
        validation: &RaceBatchValidation,
    ) -> BTreeMap<AllocationKey, Arc<ShadowSegments>> {
        let mut allocations = shadow.allocations.clone();
        for (key, patch) in &validation.allocation_updates {
            commit_allocation_update(&mut allocations, *key, patch.clone());
        }
        allocations
    }

    fn materialize_reference_validation(
        shadow: &RaceShadow,
        validation: &ReferenceBatchValidation,
    ) -> BTreeMap<AllocationKey, Arc<ShadowSegments>> {
        let mut allocations = shadow.allocations.clone();
        for (key, segments) in &validation.allocation_updates {
            let mut materialized = ShadowSegments::default();
            for segment in segments.iter().cloned() {
                materialized.insert_prepared(segment.start, segment.end, segment);
            }
            allocations.insert(*key, Arc::new(materialized));
        }
        allocations
    }

    fn canonicalize_shadow_segments(
        mut allocations: BTreeMap<AllocationKey, Arc<ShadowSegments>>,
    ) -> BTreeMap<AllocationKey, Arc<ShadowSegments>> {
        for segments in allocations.values_mut() {
            let segments = Arc::make_mut(segments);
            let mut canonical = std::mem::take(segments).into_sorted_values();
            merge_adjacent_segments(&mut canonical);
            for segment in canonical {
                segments.insert_prepared(segment.start, segment.end, segment);
            }
        }
        allocations
    }

    #[test]
    fn monotonic_allocation_growth_uses_a_sparse_patch() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .check_batch(single(0, 0, 9, PhysicalAccessKind::Write, 1, 0, 4))
            .unwrap();
        let next = single(0, 1, 10, PhysicalAccessKind::Write, 1, 8, 4);
        let validation = shadow.validate_batch(&next).unwrap();
        assert!(matches!(
            validation.allocation_updates.values().next(),
            Some(patch) if patch.base_len == 1 && patch.staged_segment_count() == 1
        ));
        shadow.commit_validation(validation);
        assert_eq!(shadow.tracked_interval_count(), 2);
    }

    #[test]
    fn overlapping_single_span_stages_only_the_touched_range() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .check_batch(single(0, 0, 9, PhysicalAccessKind::Write, 1, 0, 16))
            .unwrap();
        let next = single(0, 1, 10, PhysicalAccessKind::Write, 1, 4, 4);
        let validation = shadow.validate_batch(&next).unwrap();
        assert!(matches!(
            validation.allocation_updates.values().next(),
            Some(patch) if patch.base_len == 1 && patch.staged_segment_count() == 1
        ));
        shadow.commit_validation(validation);
        assert_eq!(shadow.tracked_interval_count(), 3);
    }

    #[test]
    fn fragmented_allocation_tiny_lane_batch_stages_a_bounded_patch() {
        const SEGMENT_COUNT: usize = 512;

        let mut shadow = RaceShadow::new(1);
        for byte_offset in 0..SEGMENT_COUNT {
            shadow
                .check_batch(single(
                    0,
                    byte_offset as u64,
                    9,
                    PhysicalAccessKind::Write,
                    1,
                    byte_offset,
                    1,
                ))
                .unwrap();
        }
        assert_eq!(shadow.tracked_interval_count(), SEGMENT_COUNT);

        let overwrite = single(0, SEGMENT_COUNT as u64, 10, PhysicalAccessKind::Write, 1, 127, 1);
        let validation = shadow.validate_batch(&overwrite).unwrap();
        assert_eq!(validation.allocation_updates.len(), 1);
        let patch = validation.allocation_updates.values().next().unwrap();
        assert_eq!(patch.base_len, SEGMENT_COUNT);
        assert_eq!(patch.staged_segment_count(), 1);
        drop(validation);
        assert_eq!(shadow.tracked_interval_count(), SEGMENT_COUNT);

        let next = batch(
            0,
            SEGMENT_COUNT as u64,
            10,
            0b11,
            PhysicalAccessKind::Read,
            1,
            |_| span(1, SEGMENT_COUNT / 2, 1),
        );
        let validation = shadow.validate_batch(&next).unwrap();
        let patch = validation
            .allocation_updates
            .values()
            .next()
            .expect("the touched allocation has one sparse patch");
        assert_eq!(patch.base_len, SEGMENT_COUNT);
        assert_eq!(patch.staged_segment_count(), 1);
        assert_eq!(shadow.tracked_interval_count(), SEGMENT_COUNT);

        shadow.commit_validation(validation);
        assert_eq!(shadow.tracked_interval_count(), SEGMENT_COUNT);
        assert_eq!(
            shadow.warp_clock(0).and_then(|clock| clock.component(0)),
            Some((SEGMENT_COUNT + 1) as u64),
        );
    }

    #[test]
    fn discarded_sparse_validation_does_not_commit_clock_or_shadow() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .check_batch(single(0, 0, 11, PhysicalAccessKind::Write, 1, 0, 4))
            .unwrap();
        let clock_before = shadow.warp_clock(0).unwrap().clone();
        let allocations_before = shadow.allocations.clone();

        let next = single(0, 1, 12, PhysicalAccessKind::Read, 1, 0, 4);
        let validation = shadow.validate_batch(&next).unwrap();
        drop(validation);

        assert_eq!(shadow.warp_clock(0), Some(&clock_before));
        assert_eq!(shadow.allocations, allocations_before);
    }

    #[test]
    fn same_warp_intervals_split_and_coalesce_independently_of_span_width() {
        for byte_len in [16, 16 * 1024 * 1024] {
            let mut shadow = RaceShadow::new(2);
            for (sequence, offset, length, interval_count) in
                [(0, 0, byte_len, 1), (1, 4, 8, 3), (2, 0, byte_len, 1)]
            {
                shadow
                    .check_batch(single(0, sequence, 9, PhysicalAccessKind::Write, 7, offset, length))
                    .unwrap();
                assert_eq!(shadow.tracked_interval_count(), interval_count);
            }
            let segments = shadow.allocations.values().next().unwrap();
            let (_, segment) = segments.iter().next().unwrap();
            assert_eq!((segment.start, segment.end), (0, byte_len));
        }
    }

    #[test]
    fn same_lane_loop_keeps_byte_shadow_bounded_by_footprint() {
        // Keep the second warp unobserving so GC cannot hide retained state.
        let mut shadow = RaceShadow::new(2);
        let started = Instant::now();
        for sequence in 0..50_000_u64 {
            shadow
                .check_batch(single(0, sequence, 9, PhysicalAccessKind::Write, 7, 0, 4))
                .unwrap();
        }
        assert_eq!(shadow.allocations.len(), 1);
        assert_eq!(shadow.tracked_interval_count(), 1);
        let segments = shadow.allocations.values().next().unwrap();
        let (_, segment) = segments.iter().next().unwrap();
        assert_eq!((segment.start, segment.end), (0, 4));
        assert_eq!(segment.state.writes.iter().count(), 1);
        assert_eq!(segment.state.reads.iter().count(), 0);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(30),
            "50k fixed-footprint accesses took {:?}", started.elapsed(),
        );
    }

    #[test]
    fn cloned_shadow_copies_only_the_touched_allocation_vector() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .check_batch(single(0, 0, 9, PhysicalAccessKind::Write, 1, 0, 4))
            .unwrap();
        shadow
            .check_batch(single(0, 1, 10, PhysicalAccessKind::Write, 2, 0, 4))
            .unwrap();

        let mut candidate = shadow.clone();
        candidate
            .check_batch(single(0, 2, 11, PhysicalAccessKind::Write, 2, 8, 4))
            .unwrap();

        let key_one = AllocationKey {
            space: PhysicalAccessSpace::Shared,
            allocation: PhysicalAllocationId::new(1),
        };
        let key_two = AllocationKey {
            space: PhysicalAccessSpace::Shared,
            allocation: PhysicalAllocationId::new(2),
        };
        assert!(Arc::ptr_eq(
            shadow.allocations.get(&key_one).unwrap(),
            candidate.allocations.get(&key_one).unwrap(),
        ));
        assert!(!Arc::ptr_eq(
            shadow.allocations.get(&key_two).unwrap(),
            candidate.allocations.get(&key_two).unwrap(),
        ));
    }

    #[test]
    fn read_read_is_clean_and_adjacent_intervals_do_not_alias() {
        let mut shadow = RaceShadow::new(3);
        shadow
            .check_batch(single(0, 0, 10, PhysicalAccessKind::Read, 1, 0, 2))
            .unwrap();
        shadow
            .check_batch(single(1, 0, 11, PhysicalAccessKind::Read, 1, 0, 2))
            .unwrap();
        shadow
            .check_batch(single(2, 0, 12, PhysicalAccessKind::Write, 1, 2, 2))
            .unwrap();
    }

    #[test]
    fn fp16_subword_aliases_fp32_physical_bytes() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 20, PhysicalAccessKind::Write, 7, 0, 4))
            .unwrap();
        let race = finding(
            shadow
                .check_batch(single(1, 0, 21, PhysicalAccessKind::Read, 7, 2, 2))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.prior().span(), span(7, 0, 4));
        assert_eq!(race.current().span(), span(7, 2, 2));
        assert_eq!(race.overlap(), span(7, 2, 2));
    }

    #[test]
    fn distinct_logical_views_alias_when_physical_span_matches() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 30, PhysicalAccessKind::Read, 9, 64, 4))
            .unwrap();
        let race = finding(
            shadow
                .check_batch(single(1, 0, 31, PhysicalAccessKind::Write, 9, 64, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::ReadWrite);
        assert_eq!(race.prior().operation().source_op_id(), StaticOpId::new(30));
        assert_eq!(
            race.current().operation().source_op_id(),
            StaticOpId::new(31)
        );
    }

    #[test]
    fn unordered_cross_warp_write_read_is_an_error() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 40, PhysicalAccessKind::Write, 3, 8, 4))
            .unwrap();
        let race = finding(
            shadow
                .check_batch(single(1, 0, 41, PhysicalAccessKind::Read, 3, 8, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.prior().lane(), 0);
        assert_eq!(race.current().lane(), 0);
    }

    #[test]
    fn barrier_release_acquire_orders_the_consumer() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 50, PhysicalAccessKind::Write, 4, 0, 4))
            .unwrap();
        let payload = shadow.barrier_release(0).unwrap();
        shadow.barrier_acquire(1, &payload).unwrap();
        shadow
            .check_batch(single(1, 0, 51, PhysicalAccessKind::Read, 4, 0, 4))
            .unwrap();
    }

    #[test]
    fn imported_global_release_orders_async_shared_access() {
        let run = |import_frontier: bool, proxy_fence: bool| {
            let mut shadow = RaceShadow::new(2);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    52,
                    PhysicalAccessKind::Write,
                    41,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            if proxy_fence {
                shadow
                    .proxy_async_fence(0, ProxyAsyncFenceScope::SharedCta)
                    .unwrap();
            }
            let mut incoming = SharedClockFrontier::default();
            if import_frontier {
                incoming.merge_clock(
                    0,
                    &shadow
                        .memory_publication(0, WarpMask::from_bits(1))
                        .unwrap(),
                );
            }
            let common = [0; WARP_SIZE];
            let observed = [[0; WARP_SIZE]; WARP_SIZE];
            let lane_order =
                RaceLaneOrder::new(1, &common, &observed).with_incoming(0, &incoming, None);
            let issue = operation(1, 0, 53, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow
                .fork_async_token_after_clock(
                    1,
                    WarpMask::from_bits(1),
                    &token,
                    None,
                    Some(&lane_order),
                )
                .unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    1,
                    0,
                    53,
                    PhysicalAccessKind::Read,
                    41,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        assert!(matches!(run(false, true), Err(RaceShadowError::Race(_))));
        assert!(matches!(run(true, false), Err(RaceShadowError::Race(_))));
        run(true, true).unwrap();
    }

    #[test]
    fn barrier_payload_joins_multiple_releases() {
        let mut shadow = RaceShadow::new(3);
        let mut payload = shadow.barrier_release(0).unwrap();
        let second = shadow.barrier_release(1).unwrap();
        payload.merge(&second).unwrap();
        shadow.barrier_acquire(2, &payload).unwrap();
        assert_eq!(shadow.warp_clock(2).unwrap().component(0), Some(1));
        assert_eq!(shadow.warp_clock(2).unwrap().component(1), Some(1));
        assert_eq!(shadow.warp_clock(2).unwrap().component(2), Some(1));
    }

    #[test]
    fn generic_to_async_requires_a_matching_proxy_fence() {
        let run = |scope: Option<ProxyAsyncFenceScope>| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    500,
                    PhysicalAccessKind::Write,
                    50,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            if let Some(scope) = scope {
                shadow.proxy_async_fence(0, scope).unwrap();
            }
            let issue = operation(0, 1, 501, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    501,
                    PhysicalAccessKind::Read,
                    50,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        assert!(matches!(run(None), Err(RaceShadowError::Race(_))));
        assert!(matches!(
            run(Some(ProxyAsyncFenceScope::SharedCluster)),
            Err(RaceShadowError::Race(_))
        ));
        run(Some(ProxyAsyncFenceScope::SharedCta)).unwrap();
        run(Some(ProxyAsyncFenceScope::All)).unwrap();
    }

    #[test]
    fn proxy_fence_applies_only_to_active_lanes() {
        let run = |fence_lane: usize, issue_lane: usize| {
            let mut shadow = RaceShadow::new(2);
            shadow
                .check_batch(single_proxy_lane(
                    1,
                    0,
                    0,
                    530,
                    PhysicalAccessKind::Write,
                    54,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            let release = shadow.barrier_release(1).unwrap();
            shadow.barrier_acquire(0, &release).unwrap();
            shadow
                .proxy_async_fence_masked(
                    0,
                    WarpMask::from_bits(1_u32 << fence_lane),
                    ProxyAsyncFenceScope::SharedCta,
                    None,
                )
                .unwrap();
            let issue = operation(0, 1, 531, 1_u32 << issue_lane);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow
                .fork_async_token_for_mask(0, WarpMask::from_bits(1_u32 << issue_lane), &token)
                .unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy_lane(
                    0,
                    issue_lane,
                    1,
                    531,
                    PhysicalAccessKind::Read,
                    54,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        run(0, 0).unwrap();
        let race = finding(run(0, 1).unwrap_err());
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(
            race.prior().operation().source_op_id(),
            StaticOpId::new(530)
        );
        assert_eq!(
            race.current().operation().source_op_id(),
            StaticOpId::new(531)
        );
        assert_eq!(race.current().lane(), 1);
        assert_eq!(race.overlap(), span(54, 0, 4));
    }

    #[test]
    fn warp_sync_transfers_proxy_bridge_only_to_participating_lanes() {
        let run = |issue_lane: usize| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy_lane(
                    0,
                    0,
                    0,
                    535,
                    PhysicalAccessKind::Write,
                    56,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            shadow
                .proxy_async_fence_masked(
                    0,
                    WarpMask::from_bits(1),
                    ProxyAsyncFenceScope::SharedCta,
                    None,
                )
                .unwrap();
            shadow
                .warp_sync(0, WarpMask::from_lanes([0, 1]).unwrap())
                .unwrap();
            let issue = operation(0, 1, 536, 1_u32 << issue_lane);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow
                .fork_async_token_for_mask(0, WarpMask::from_bits(1_u32 << issue_lane), &token)
                .unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy_lane(
                    0,
                    issue_lane,
                    1,
                    536,
                    PhysicalAccessKind::Read,
                    56,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        run(1).unwrap();
        let race = finding(run(2).unwrap_err());
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(
            race.prior().operation().source_op_id(),
            StaticOpId::new(535)
        );
        assert_eq!(
            race.current().operation().source_op_id(),
            StaticOpId::new(536)
        );
        assert_eq!(race.current().lane(), 2);
        assert_eq!(race.overlap(), span(56, 0, 4));
    }

    #[test]
    fn async_completion_acquire_applies_only_to_acquiring_lanes() {
        let run = |acquiring_lane: usize| {
            let mut shadow = RaceShadow::new(2);
            let issue = operation(1, 0, 540, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow
                .fork_async_token_for_mask(1, WarpMask::from_bits(1), &token)
                .unwrap();
            shadow
                .check_batch_at_clock_for_async_token(
                    &single_proxy_lane(
                        1,
                        0,
                        0,
                        540,
                        PhysicalAccessKind::Write,
                        55,
                        MemoryProxy::Async,
                        ProxyMemoryDomain::SharedCta,
                    ),
                    &token_clock,
                    &token,
                )
                .unwrap();
            let completion = shadow
                .apply_implicit_async_completion(&token, [ProxyMemoryDomain::SharedCta])
                .unwrap();
            shadow
                .barrier_acquire_masked(
                    0,
                    WarpMask::from_bits(1_u32 << acquiring_lane),
                    &BarrierClockPayload::from_clock(completion),
                )
                .unwrap();
            shadow.check_batch(single_proxy_lane(
                0,
                1,
                1,
                541,
                PhysicalAccessKind::Read,
                55,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCta,
            ))
        };

        run(1).unwrap();
        let race = finding(run(0).unwrap_err());
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(
            race.prior().operation().source_op_id(),
            StaticOpId::new(540)
        );
        assert_eq!(
            race.current().operation().source_op_id(),
            StaticOpId::new(541)
        );
        assert_eq!(race.current().lane(), 1);
        assert_eq!(race.overlap(), span(55, 0, 4));
    }

    #[test]
    fn proxy_fence_does_not_retroactively_order_a_later_generic_access() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .proxy_async_fence(0, ProxyAsyncFenceScope::SharedCta)
            .unwrap();
        shadow
            .check_batch(single_proxy(
                0,
                0,
                502,
                PhysicalAccessKind::Write,
                50,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCta,
            ))
            .unwrap();
        let issue = operation(0, 1, 503, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        assert!(matches!(
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    503,
                    PhysicalAccessKind::Read,
                    50,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            ),
            Err(RaceShadowError::Race(_)),
        ));
    }

    #[test]
    fn qualified_proxy_fence_orders_aliases_from_its_selected_prior_domain() {
        let run = |scope: Option<ProxyAsyncFenceScope>| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    505,
                    PhysicalAccessKind::Write,
                    50,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            if let Some(scope) = scope {
                shadow.proxy_async_fence(0, scope).unwrap();
            }
            let issue = operation(0, 1, 506, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    506,
                    PhysicalAccessKind::Read,
                    50,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCluster,
                ),
                &token_clock,
                &token,
            )
        };

        assert!(matches!(run(None), Err(RaceShadowError::Race(_))));
        run(Some(ProxyAsyncFenceScope::SharedCta)).unwrap();
        assert!(matches!(
            run(Some(ProxyAsyncFenceScope::SharedCluster)),
            Err(RaceShadowError::Race(_))
        ));
        run(Some(ProxyAsyncFenceScope::All)).unwrap();
    }

    #[test]
    fn qualified_cluster_proxy_fence_orders_cluster_to_cta_alias() {
        let run = |scope: Option<ProxyAsyncFenceScope>| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    505,
                    PhysicalAccessKind::Write,
                    50,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCluster,
                ))
                .unwrap();
            if let Some(scope) = scope {
                shadow.proxy_async_fence(0, scope).unwrap();
            }
            let issue = operation(0, 1, 506, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    506,
                    PhysicalAccessKind::Read,
                    50,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        assert!(matches!(run(None), Err(RaceShadowError::Race(_))));
        assert!(matches!(
            run(Some(ProxyAsyncFenceScope::SharedCta)),
            Err(RaceShadowError::Race(_))
        ));
        run(Some(ProxyAsyncFenceScope::SharedCluster)).unwrap();
        run(Some(ProxyAsyncFenceScope::All)).unwrap();
    }

    #[test]
    fn explicit_proxy_fence_orders_completed_async_to_generic_access() {
        let run = |scope: Option<ProxyAsyncFenceScope>| {
            let mut shadow = RaceShadow::new(1);
            let issue = operation(0, 0, 507, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow
                .check_batch_at_clock_for_async_token(
                    &single_proxy(
                        0,
                        0,
                        507,
                        PhysicalAccessKind::Write,
                        51,
                        MemoryProxy::Async,
                        ProxyMemoryDomain::SharedCta,
                    ),
                    &token_clock,
                    &token,
                )
                .unwrap();
            let completed = shadow
                .complete_async_actors(std::slice::from_ref(&token))
                .unwrap()
                .pop()
                .unwrap();
            shadow
                .barrier_acquire(0, &BarrierClockPayload::from_clock(completed))
                .unwrap();
            if let Some(scope) = scope {
                shadow.proxy_async_fence(0, scope).unwrap();
            }
            shadow.check_batch(single_proxy(
                0,
                1,
                508,
                PhysicalAccessKind::Read,
                51,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCta,
            ))
        };

        assert!(matches!(run(None), Err(RaceShadowError::Race(_))));
        assert!(matches!(
            run(Some(ProxyAsyncFenceScope::SharedCluster)),
            Err(RaceShadowError::Race(_))
        ));
        run(Some(ProxyAsyncFenceScope::SharedCta)).unwrap();
        run(Some(ProxyAsyncFenceScope::All)).unwrap();
    }

    #[test]
    fn ordinary_barrier_does_not_replace_async_to_generic_proxy_ordering() {
        let mut shadow = RaceShadow::new(2);
        let issue = operation(0, 0, 510, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        shadow
            .check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    0,
                    510,
                    PhysicalAccessKind::Write,
                    51,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCluster,
                ),
                &token_clock,
                &token,
            )
            .unwrap();
        shadow
            .barrier_acquire(1, &BarrierClockPayload::from_clock(token_clock))
            .unwrap();

        assert!(matches!(
            shadow.check_batch(single_proxy(
                1,
                0,
                511,
                PhysicalAccessKind::Read,
                51,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCluster,
            )),
            Err(RaceShadowError::Race(_))
        ));
    }

    #[test]
    fn implicit_async_completion_bridge_propagates_through_a_barrier() {
        let mut shadow = RaceShadow::new(2);
        let issue = operation(0, 0, 520, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        shadow
            .check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    0,
                    520,
                    PhysicalAccessKind::Write,
                    52,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCluster,
                ),
                &token_clock,
                &token,
            )
            .unwrap();
        let completion = shadow
            .apply_implicit_async_completion(&token, [ProxyMemoryDomain::SharedCluster])
            .unwrap();
        shadow
            .barrier_acquire(1, &BarrierClockPayload::from_clock(completion))
            .unwrap();
        shadow
            .check_batch(single_proxy(
                1,
                0,
                521,
                PhysicalAccessKind::Read,
                52,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCta,
            ))
            .unwrap();
    }

    #[test]
    fn implicit_cta_completion_orders_a_later_cluster_alias() {
        let mut shadow = RaceShadow::new(2);
        let issue = operation(0, 0, 522, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        shadow
            .check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    0,
                    522,
                    PhysicalAccessKind::Write,
                    53,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
            .unwrap();
        let completion = shadow
            .apply_implicit_async_completion(&token, [ProxyMemoryDomain::SharedCta])
            .unwrap();
        shadow
            .barrier_acquire(1, &BarrierClockPayload::from_clock(completion))
            .unwrap();
        shadow
            .check_batch(single_proxy(
                1,
                0,
                523,
                PhysicalAccessKind::Read,
                53,
                MemoryProxy::Generic,
                ProxyMemoryDomain::SharedCluster,
            ))
            .unwrap();
    }

    #[test]
    fn lane_stamped_history_survives_gc_until_a_proxy_fence_covers_it() {
        let run = |lane_epoch: u64, fence_lane: Option<usize>| {
            let mut shadow = RaceShadow::new(1);
            let common = [0; WARP_SIZE];
            let mut observed = [[0; WARP_SIZE]; WARP_SIZE];
            observed[0][0] = lane_epoch - 1;
            let validation = shadow
                .validate_batch_with_lane_order(
                    &single_proxy(
                        0,
                        0,
                        525,
                        PhysicalAccessKind::Write,
                        53,
                        MemoryProxy::Generic,
                        ProxyMemoryDomain::SharedCta,
                    ),
                    &RaceLaneOrder::new(0, &common, &observed),
                )
                .unwrap();
            shadow.commit_validation(validation);
            observed[0][0] = lane_epoch;
            assert_eq!(shadow.gc_dominated_frontier(), 0);
            assert_eq!(shadow.tracked_interval_count(), 1);
            let order = RaceLaneOrder::new(0, &common, &observed);
            if let Some(lane) = fence_lane {
                shadow
                    .proxy_async_fence_masked(
                        0,
                        WarpMask::from_bits(1 << lane),
                        ProxyAsyncFenceScope::SharedCta,
                        Some(&order),
                    )
                    .unwrap();
            }
            // Transfer any bridge to the issuing lane without retroactively
            // publishing a write the fence's own lane had not observed.
            shadow.warp_sync(0, WarpMask::from_bits(3)).unwrap();
            let issue = operation(0, 1, 526, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow
                .fork_async_token_after_clock(0, WarpMask::from_bits(1), &token, None, Some(&order))
                .unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    526,
                    PhysicalAccessKind::Read,
                    53,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        for epoch in [1, RaceEventTimestamp::LANE_EPOCH_MASK + 1] {
            for fence_lane in [None, Some(1)] {
                let race = finding(run(epoch, fence_lane).unwrap_err());
                assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
                assert_eq!(race.prior().operation().source_op_id().get(), 525);
                assert_eq!(race.current().operation().source_op_id().get(), 526);
            }
            run(epoch, Some(0)).unwrap();
        }
    }

    #[test]
    fn retired_generic_history_fails_closed_until_a_proxy_fence_covers_it() {
        let run = |with_fence: bool| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    525,
                    PhysicalAccessKind::Write,
                    53,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            assert_eq!(shadow.gc_dominated_frontier(), 1);
            assert_eq!(shadow.tracked_interval_count(), 0);
            if with_fence {
                shadow
                    .proxy_async_fence(0, ProxyAsyncFenceScope::SharedCta)
                    .unwrap();
            }
            let issue = operation(0, 1, 526, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    1,
                    526,
                    PhysicalAccessKind::Read,
                    53,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        assert!(matches!(
            run(false),
            Err(RaceShadowError::RetiredCrossProxyHistory { .. })
        ));
        run(true).unwrap();
    }

    #[test]
    fn retired_generic_history_keeps_reads_and_writes_in_their_own_slots() {
        // A fenced generic write followed by unfenced generic reads, all
        // retired before the first async-proxy access. An async read only
        // conflicts with the write, which the fence covers; folding the
        // later reads into the same frontier would report it unordered.
        let run = |async_kind: PhysicalAccessKind| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    525,
                    PhysicalAccessKind::Write,
                    53,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            shadow
                .proxy_async_fence(0, ProxyAsyncFenceScope::SharedCta)
                .unwrap();
            for (sequence, op) in [(1, 527), (2, 528)] {
                shadow
                    .check_batch(single_proxy(
                        0,
                        sequence,
                        op,
                        PhysicalAccessKind::Read,
                        53,
                        MemoryProxy::Generic,
                        ProxyMemoryDomain::SharedCta,
                    ))
                    .unwrap();
            }
            assert!(shadow.gc_dominated_frontier() >= 2);
            assert_eq!(shadow.tracked_interval_count(), 0);
            let issue = operation(0, 3, 526, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single_proxy(
                    0,
                    3,
                    526,
                    async_kind,
                    53,
                    MemoryProxy::Async,
                    ProxyMemoryDomain::SharedCta,
                ),
                &token_clock,
                &token,
            )
        };

        run(PhysicalAccessKind::Read).unwrap();
        // An async write also conflicts with the unfenced reads.
        assert!(matches!(
            run(PhysicalAccessKind::Write),
            Err(RaceShadowError::RetiredCrossProxyHistory { .. })
        ));
    }

    #[test]
    fn retired_generic_history_only_covers_the_retired_bytes() {
        // An unfenced generic store to bytes [0, 4) retired before the first
        // async-proxy access: an async write to [64, 68) of the same
        // allocation never conflicted with it, while one to [0, 4) did.
        let run = |offset: usize| {
            let mut shadow = RaceShadow::new(1);
            shadow
                .check_batch(single_proxy(
                    0,
                    0,
                    525,
                    PhysicalAccessKind::Write,
                    53,
                    MemoryProxy::Generic,
                    ProxyMemoryDomain::SharedCta,
                ))
                .unwrap();
            assert_eq!(shadow.gc_dominated_frontier(), 1);
            let issue = operation(0, 1, 526, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow.check_batch_at_clock_for_async_token(
                &single(0, 1, 526, PhysicalAccessKind::Write, 53, offset, 4)
                    .with_memory_semantics(MemoryAccessSemantics::async_proxy())
                    .with_proxy_memory_domain(ProxyMemoryDomain::SharedCta),
                &token_clock,
                &token,
            )
        };

        run(64).unwrap();
        assert!(matches!(
            run(0),
            Err(RaceShadowError::RetiredCrossProxyHistory { .. })
        ));
    }

    #[test]
    fn async_token_clock_is_independent_until_wait_acquire() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 52, PhysicalAccessKind::Read, 40, 0, 4))
            .unwrap();
        let issue = operation(0, 0, 53, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        assert_eq!(token_clock.component(0), Some(1));
        assert_eq!(token_clock.async_component(&token), 1);

        shadow
            .check_batch_at_clock(
                &single(0, 1, 54, PhysicalAccessKind::Write, 41, 0, 4),
                &token_clock,
            )
            .unwrap();
        shadow
            .check_batch(single(0, 2, 55, PhysicalAccessKind::Read, 42, 0, 4))
            .unwrap();

        let later_issuer = shadow.warp_clock(0).unwrap().clone();
        assert_eq!(later_issuer.component(0), Some(2));
        assert_eq!(later_issuer.async_component(&token), 0);
        assert!(!token_clock.happens_before(&later_issuer));
        assert!(!later_issuer.happens_before(&token_clock));

        let race = finding(
            shadow
                .check_batch(single(1, 0, 56, PhysicalAccessKind::Read, 41, 0, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);

        let payload = BarrierClockPayload::from_clock(token_clock.clone());
        shadow.barrier_acquire(1, &payload).unwrap();
        let waiter = shadow.warp_clock(1).unwrap();
        assert_eq!(waiter.component(0), Some(1));
        assert_eq!(waiter.async_component(&token), 1);
        shadow
            .check_batch(single(1, 1, 57, PhysicalAccessKind::Read, 41, 0, 4))
            .unwrap();
    }

    #[test]
    fn async_completion_failure_requires_ordinary_post_issue_handoff() {
        let run = |prior_kind: PhysicalAccessKind, handoff_after_issue: Option<bool>| {
            let mut shadow = RaceShadow::new(2);
            if handoff_after_issue == Some(false) {
                let payload = shadow.barrier_release(0).unwrap();
                shadow.barrier_acquire(1, &payload).unwrap();
            }

            let issue = operation(0, 0, 560, 1);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let token_clock = shadow.fork_async_token(0, &token).unwrap();
            shadow
                .check_batch_at_clock_for_async_token(
                    &single(0, 0, 560, prior_kind, 41, 0, 4),
                    &token_clock,
                    &token,
                )
                .unwrap();

            if handoff_after_issue == Some(true) {
                let payload = shadow.barrier_release(0).unwrap();
                shadow.barrier_acquire(1, &payload).unwrap();
            }
            let current_kind = if prior_kind.reads() {
                PhysicalAccessKind::Write
            } else {
                PhysicalAccessKind::Read
            };
            finding(
                shadow
                    .check_batch(single(1, 0, 561, current_kind, 41, 0, 4))
                    .unwrap_err(),
            )
            .ordering_failure()
        };

        for prior_kind in [PhysicalAccessKind::Read, PhysicalAccessKind::Write] {
            assert_eq!(
                run(prior_kind, None),
                PhysicalRaceOrderingFailure::MissingInterActorSynchronization
            );
            assert_eq!(
                run(prior_kind, Some(false)),
                PhysicalRaceOrderingFailure::MissingInterActorSynchronization
            );
            assert_eq!(
                run(prior_kind, Some(true)),
                PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained
            );
        }
    }

    #[test]
    fn current_async_access_reports_missing_ordinary_ordering() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(1, 0, 562, PhysicalAccessKind::Write, 41, 0, 4))
            .unwrap();

        let issue = operation(0, 0, 563, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        let race = finding(
            shadow
                .check_batch_at_clock_for_async_token(
                    &single(0, 0, 563, PhysicalAccessKind::Read, 41, 0, 4),
                    &token_clock,
                    &token,
                )
                .unwrap_err(),
        );

        assert_eq!(
            race.ordering_failure(),
            PhysicalRaceOrderingFailure::MissingInterActorSynchronization
        );
    }

    #[test]
    fn dense_async_clocks_keep_independent_actors_unordered_and_join_exactly() {
        let mut shadow = RaceShadow::new(2);
        let first_issue = operation(0, 0, 580, 1);
        let second_issue = operation(1, 0, 581, 1);
        let first = AsyncTokenId::new(first_issue.id().clone(), 7);
        let second = AsyncTokenId::new(second_issue.id().clone(), 3);

        // Register in the opposite order from the assertions so correctness
        // cannot depend on a token's dense component index.
        let second_clock = shadow.fork_async_token(1, &second).unwrap();
        let first_clock = shadow.fork_async_token(0, &first).unwrap();
        assert_eq!(first_clock.async_component(&first), 1);
        assert_eq!(first_clock.async_component(&second), 0);
        assert_eq!(second_clock.async_component(&first), 0);
        assert_eq!(second_clock.async_component(&second), 1);
        assert!(!first_clock.happens_before(&second_clock));
        assert!(!second_clock.happens_before(&first_clock));

        let mut joined = BarrierClockPayload::from_clock(first_clock);
        joined
            .merge(&BarrierClockPayload::from_clock(second_clock))
            .unwrap();
        assert_eq!(joined.clock().async_component(&first), 1);
        assert_eq!(joined.clock().async_component(&second), 1);
    }

    #[test]
    fn vector_clocks_from_different_launch_registries_cannot_merge() {
        let mut first = RaceShadow::new(1);
        let mut second = RaceShadow::new(1);
        let mut first_payload = first.barrier_release(0).unwrap();
        let second_payload = second.barrier_release(0).unwrap();

        assert!(matches!(
            first_payload.merge(&second_payload),
            Err(RaceShadowError::AsyncClockRegistryMismatch)
        ));
    }

    #[test]
    fn ordinary_events_share_unchanged_async_clock_components() {
        let mut shadow = RaceShadow::new(1);
        let issue = operation(0, 0, 58, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        let payload = BarrierClockPayload::from_clock(token_clock);
        shadow.barrier_acquire(0, &payload).unwrap();

        let before = shadow.warp_clock(0).unwrap().async_components.clone();
        shadow
            .check_batch(single(0, 0, 59, PhysicalAccessKind::Read, 43, 0, 4))
            .unwrap();
        let after = &shadow.warp_clock(0).unwrap().async_components;

        assert!(before.shares_storage_with(after));
        assert_eq!(shadow.warp_clock(0).unwrap().async_component(&token), 1);
    }

    #[test]
    fn async_epoch_packing_round_trips_and_orders_by_generation() {
        assert_eq!(unpack_async_epoch(pack_async_epoch(0, 7)), (0, 7));
        assert_eq!(
            unpack_async_epoch(pack_async_epoch(3, ASYNC_EPOCH_MASK)),
            (3, ASYNC_EPOCH_MASK)
        );
        assert!(pack_async_epoch(1, 1) > pack_async_epoch(0, ASYNC_EPOCH_MASK));
        assert!(
            pack_async_epoch(ASYNC_GENERATION_LIMIT - 1, ASYNC_EPOCH_MASK)
                < u64::from(RaceEventTimestamp::ASYNC_ACTOR_BIT)
        );
    }

    #[test]
    fn async_issue_pins_its_actor_before_commit_can_reclaim_slots() {
        let mut shadow = RaceShadow::new(1);
        let access = single(0, 0, 70, PhysicalAccessKind::Write, 3_000, 0, 16);
        let token = AsyncTokenId::new(access.operation().id().clone(), 0);
        let validation = shadow
            .validate_async_issue_batches(0, WarpMask::FULL, &token, [&access], None)
            .unwrap();
        shadow.safe_points_since_gc = AUTOMATIC_GC_SAFE_POINT_INTERVAL - 1;
        let (clock, _) = shadow.commit_async_issue_validation(validation).unwrap();
        assert_eq!(clock.async_component(&token), 1);
        assert_eq!(shadow.active_async_actor_count(), 1);
        shadow
            .validate_and_commit_batches_at_clock(&[], &clock, &token, false)
            .unwrap();
        shadow.retire_async_actor(&token).unwrap();
        shadow.reclaim_async_slots();
        assert_eq!(shadow.async_clock_registry.index(&token), None);
    }

    #[test]
    fn retired_async_slot_is_reused_under_a_new_generation() {
        let mut shadow = RaceShadow::new(1);
        let issue_a = operation(0, 0, 70, 1);
        let token_a = AsyncTokenId::new(issue_a.id().clone(), 0);
        let clock_a = shadow.fork_async_token(0, &token_a).unwrap();
        let index_a = clock_a.async_actor_index(&token_a).unwrap();
        assert_eq!(clock_a.async_component(&token_a), 1);
        shadow
            .check_batch_at_clock(
                &single(0, 0, 71, PhysicalAccessKind::Write, 3_000, 0, 4),
                &clock_a,
            )
            .unwrap();

        // The issuing warp observes A's completion; A's witness is then
        // globally observed and retired, and its slot comes free.
        let payload = BarrierClockPayload::from_clock(clock_a.clone());
        shadow.retire_async_actor(&token_a).unwrap();
        shadow.barrier_acquire(0, &payload).unwrap();
        assert_eq!(shadow.gc_dominated_frontier(), 1);
        assert_eq!(shadow.async_clock_registry.index(&token_a), None);

        let issue_b = operation(0, 1, 72, 1);
        let token_b = AsyncTokenId::new(issue_b.id().clone(), 0);
        let clock_b = shadow.fork_async_token(0, &token_b).unwrap();
        assert_eq!(clock_b.async_actor_index(&token_b), Some(index_a));
        assert_eq!(clock_b.async_component(&token_b), 1);
        assert_eq!(
            unpack_async_epoch(clock_b.async_component_at(index_a)),
            (1, 1)
        );

        // A clock that only ever saw A's generation has not observed B.
        let stamp_b = RaceEventTimestamp::for_async(&clock_b, &token_b, &shadow.operation_registry);
        assert!(stamp_b.observed_by(&clock_b, &shadow.operation_registry));
        assert!(!stamp_b.observed_by(&clock_a, &shadow.operation_registry));

        // B's write races with a warp read that never observed B, and stops
        // racing once the warp acquires B's completion.
        shadow
            .check_batch_at_clock(
                &single(0, 1, 73, PhysicalAccessKind::Write, 3_000, 0, 4),
                &clock_b,
            )
            .unwrap();
        let race = finding(
            shadow
                .check_batch(single(0, 2, 74, PhysicalAccessKind::Read, 3_000, 0, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        let payload_b = BarrierClockPayload::from_clock(clock_b.clone());
        shadow.retire_async_actor(&token_b).unwrap();
        shadow.barrier_acquire(0, &payload_b).unwrap();
        shadow
            .check_batch(single(0, 3, 75, PhysicalAccessKind::Read, 3_000, 0, 4))
            .unwrap();
    }

    #[test]
    fn reclaimed_tcgen_slot_exports_unpacked_epochs_and_forgets_completion() {
        let mut shadow = RaceShadow::new(1);
        let issue_a = operation(0, 0, 80, 1);
        let token_a = AsyncTokenId::new(issue_a.id().clone(), 0);
        let clock_a = shadow
            .fork_tcgen_token_after_clock(0, WarpMask::FULL, &token_a, None, None)
            .unwrap();
        let index_a = clock_a.async_actor_index(&token_a).unwrap();
        assert_eq!(
            shadow.tcgen_async_frontier(&clock_a),
            vec![(token_a.clone(), 1)]
        );
        shadow
            .check_batch_at_clock(
                &single(0, 0, 81, PhysicalAccessKind::Write, 4_000, 0, 4),
                &clock_a,
            )
            .unwrap();
        let completed = shadow.complete_async_actors(&[token_a.clone()]).unwrap();
        assert!(shadow.tcgen_completed_epochs.contains_key(&index_a));
        shadow
            .barrier_acquire(0, &BarrierClockPayload::from_clock(completed[0].clone()))
            .unwrap();
        assert_eq!(shadow.gc_dominated_frontier(), 1);
        assert!(!shadow.tcgen_async_indices.contains(&index_a));
        assert!(!shadow.tcgen_completed_epochs.contains_key(&index_a));

        let issue_b = operation(0, 1, 82, 1);
        let token_b = AsyncTokenId::new(issue_b.id().clone(), 0);
        let clock_b = shadow
            .fork_tcgen_token_after_clock(0, WarpMask::FULL, &token_b, None, None)
            .unwrap();
        assert_eq!(clock_b.async_actor_index(&token_b), Some(index_a));
        // Exported frontiers carry the plain epoch, never the generation.
        assert_eq!(
            shadow.tcgen_async_frontier(&clock_b),
            vec![(token_b.clone(), 1)]
        );
        // A stale value from A's generation is not a completion of B.
        assert_eq!(clock_a.async_component(&token_b), 0);
    }

    #[test]
    fn active_async_actor_prevents_retiring_unordered_warp_evidence() {
        let mut shadow = RaceShadow::new(1);
        let issue = operation(0, 0, 58, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();

        for sequence in 0..AUTOMATIC_GC_SAFE_POINT_INTERVAL {
            shadow
                .check_batch(single(
                    0,
                    sequence as u64,
                    59,
                    PhysicalAccessKind::Write,
                    1_000 + sequence as u64,
                    0,
                    4,
                ))
                .unwrap();
        }
        assert_eq!(shadow.active_async_actor_count(), 1);
        assert_eq!(
            shadow.tracked_interval_count(),
            AUTOMATIC_GC_SAFE_POINT_INTERVAL
        );

        let completion_race = finding(
            shadow
                .check_batch_at_clock(
                    &single(0, 0, 60, PhysicalAccessKind::Read, 1_000, 0, 4),
                    &token_clock,
                )
                .unwrap_err(),
        );
        assert_eq!(completion_race.kind(), PhysicalRaceKind::WriteRead);

        shadow.retire_async_actor(&token).unwrap();
        assert_eq!(shadow.active_async_actor_count(), 0);
        assert_eq!(
            shadow.tracked_interval_count(),
            AUTOMATIC_GC_SAFE_POINT_INTERVAL
        );
        assert_eq!(
            shadow.gc_dominated_frontier(),
            AUTOMATIC_GC_SAFE_POINT_INTERVAL
        );
        assert_eq!(shadow.tracked_interval_count(), 0);
    }

    #[test]
    fn completed_token_actor_retires_before_witness_becomes_globally_observed() {
        let mut shadow = RaceShadow::new(2);
        let issue = operation(0, 0, 61, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();
        shadow
            .check_batch_at_clock(
                &single(0, 0, 62, PhysicalAccessKind::Write, 2_000, 0, 4),
                &token_clock,
            )
            .unwrap();
        assert_eq!(shadow.active_async_actor_count(), 1);
        assert_eq!(shadow.tracked_interval_count(), 1);

        let payload = BarrierClockPayload::from_clock(token_clock);
        shadow.retire_async_actor(&token).unwrap();
        assert_eq!(shadow.active_async_actor_count(), 0);
        assert_eq!(shadow.tracked_interval_count(), 1);

        shadow.barrier_acquire(0, &payload).unwrap();
        assert_eq!(shadow.tracked_interval_count(), 1);
        shadow.barrier_acquire(1, &payload).unwrap();
        assert_eq!(shadow.tracked_interval_count(), 1);
        assert_eq!(shadow.gc_dominated_frontier(), 1);
        assert_eq!(shadow.tracked_interval_count(), 0);
    }

    #[test]
    fn late_conflict_cannot_partially_commit_earlier_lanes() {
        let mut shadow = RaceShadow::new(3);
        shadow
            .check_batch(single(0, 0, 60, PhysicalAccessKind::Write, 5, 128, 4))
            .unwrap();

        let failed = batch(
            1,
            0,
            61,
            (1 << 0) | (1 << 31),
            PhysicalAccessKind::Write,
            4,
            |lane| {
                if lane == 31 {
                    span(5, 128, 4)
                } else {
                    span(5, 0, 4)
                }
            },
        );
        assert_eq!(
            finding(shadow.check_batch(failed).unwrap_err()).kind(),
            PhysicalRaceKind::WriteWrite
        );

        // If lane 0 had leaked from the failed transaction, this read would race.
        shadow
            .check_batch(single(2, 0, 62, PhysicalAccessKind::Read, 5, 0, 4))
            .unwrap();
    }

    #[test]
    fn same_warp_lanes_in_one_batch_are_not_ordered_by_iteration() {
        let mut shadow = RaceShadow::new(2);
        let clock_before = shadow.warp_clock(0).unwrap().clone();
        let aliased_lanes = batch(0, 0, 70, 0b11, PhysicalAccessKind::Write, 4, |_| {
            span(6, 0, 4)
        });
        let race = finding(shadow.check_batch(aliased_lanes).unwrap_err());
        assert_eq!(race.kind(), PhysicalRaceKind::WriteWrite);
        assert_eq!(race.prior().lane(), 0);
        assert_eq!(race.current().lane(), 1);
        assert_eq!(race.prior().operation(), race.current().operation());
        assert_eq!(shadow.warp_clock(0), Some(&clock_before));
        assert_eq!(shadow.tracked_interval_count(), 0);

        shadow
            .check_batch(batch(0, 1, 71, 0b111, PhysicalAccessKind::Write, 4, |lane| {
                span(6, lane * 4, 4)
            }))
            .unwrap();
        let race = finding(
            shadow
                .check_batch(single(1, 0, 72, PhysicalAccessKind::Read, 6, 8, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.prior().lane(), 2);
    }

    #[test]
    fn multi_footprint_operation_does_not_race_with_itself_but_remains_visible() {
        let mut shadow = RaceShadow::new(2);
        let issue = operation(0, 0, 71, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = shadow.fork_async_token(0, &token).unwrap();

        let write = single(0, 0, 71, PhysicalAccessKind::Write, 6, 0, 4);
        let read = single(0, 0, 71, PhysicalAccessKind::Read, 6, 0, 4);
        shadow
            .check_batch_at_clock_within_operation_for_async_token(&write, &token_clock, &token)
            .unwrap();
        shadow
            .check_batch_at_clock_within_operation_for_async_token(&read, &token_clock, &token)
            .unwrap();

        let race = finding(
            shadow
                .check_batch(single(1, 0, 72, PhysicalAccessKind::Read, 6, 0, 4))
                .unwrap_err(),
        );
        assert_eq!(race.kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(race.prior().operation(), issue.id());
    }

    #[test]
    fn fused_clocked_batch_commit_matches_staged_commit() {
        let mut staged = RaceShadow::new(2);
        let issue = operation(0, 0, 73, 1);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let token_clock = staged.fork_async_token(0, &token).unwrap();
        let mut fused = staged.clone();
        let writes = [
            single(0, 0, 73, PhysicalAccessKind::Write, 7, 0, 4),
            single(0, 0, 73, PhysicalAccessKind::Write, 7, 8, 4),
        ];

        let validation = staged
            .validate_batches_at_clock(writes.iter(), &token_clock, &token)
            .unwrap();
        let staged_reviews = staged.commit_clocked_batches(validation);
        let fused_reviews = fused
            .validate_and_commit_batches_at_clock(&writes, &token_clock, &token, false)
            .unwrap();

        assert_eq!(fused_reviews, staged_reviews);
        assert_eq!(
            fused.tracked_interval_count(),
            staged.tracked_interval_count()
        );

        for allocation_offset in [0, 8] {
            let probe = single(
                1,
                allocation_offset as u64,
                74,
                PhysicalAccessKind::Read,
                7,
                allocation_offset,
                4,
            );
            let staged_race = finding(staged.check_batch(probe.clone()).unwrap_err());
            let fused_race = finding(fused.check_batch(probe).unwrap_err());
            assert_eq!(fused_race, staged_race);
        }
    }

    #[test]
    fn sequential_operations_in_one_warp_remain_program_ordered() {
        let mut shadow = RaceShadow::new(1);
        shadow
            .check_batch(single(0, 0, 80, PhysicalAccessKind::Write, 8, 0, 4))
            .unwrap();
        shadow
            .check_batch(single(0, 1, 81, PhysicalAccessKind::Read, 8, 0, 4))
            .unwrap();
    }

    #[test]
    fn atomic_rmw_operations_share_modification_order_but_plain_accesses_still_race() {
        let mut atomics = RaceShadow::new(2);
        atomics
            .check_batch(single(
                0,
                0,
                82,
                PhysicalAccessKind::AtomicReadModifyWrite,
                8,
                0,
                4,
            ))
            .unwrap();
        atomics
            .check_batch(single(
                1,
                0,
                83,
                PhysicalAccessKind::AtomicReadModifyWrite,
                8,
                0,
                4,
            ))
            .unwrap();

        let mut atomic_then_plain = RaceShadow::new(2);
        atomic_then_plain
            .check_batch(single(
                0,
                0,
                84,
                PhysicalAccessKind::AtomicReadModifyWrite,
                8,
                0,
                4,
            ))
            .unwrap();
        assert_eq!(
            finding(
                atomic_then_plain
                    .check_batch(single(1, 0, 85, PhysicalAccessKind::Read, 8, 0, 4))
                    .unwrap_err(),
            )
            .kind(),
            PhysicalRaceKind::WriteRead
        );

        let mut plain_then_atomic = RaceShadow::new(2);
        plain_then_atomic
            .check_batch(single(0, 0, 86, PhysicalAccessKind::Write, 8, 0, 4))
            .unwrap();
        assert_eq!(
            finding(
                plain_then_atomic
                    .check_batch(single(
                        1,
                        0,
                        87,
                        PhysicalAccessKind::AtomicReadModifyWrite,
                        8,
                        0,
                        4,
                    ))
                    .unwrap_err(),
            )
            .kind(),
            PhysicalRaceKind::WriteRead
        );
    }

    #[test]
    fn sparse_patch_matches_full_materialization_reference() {
        fn next_random(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        let mut shadow = RaceShadow::new(2);
        let mut sequences = [0_u64; 2];
        let mut seed = 0x6a09_e667_f3bc_c909_u64;
        for step in 0..2_000_u64 {
            if step > 0 && next_random(&mut seed).is_multiple_of(13) {
                let producer = (next_random(&mut seed) % 2) as usize;
                let consumer = 1 - producer;
                let payload = shadow.barrier_release(producer).unwrap();
                shadow.barrier_acquire(consumer, &payload).unwrap();
                continue;
            }

            let warp_id = (next_random(&mut seed) % 2) as usize;
            let sequence = sequences[warp_id];
            sequences[warp_id] += 1;
            let mut mask = (next_random(&mut seed) as u32) & 0b1111;
            if mask == 0 {
                mask = 1 << (next_random(&mut seed) % 4);
            }
            let kind = match next_random(&mut seed) % 3 {
                0 => PhysicalAccessKind::Read,
                1 => PhysicalAccessKind::Write,
                _ => PhysicalAccessKind::AtomicReadModifyWrite,
            };
            let space = match next_random(&mut seed) % 3 {
                0 => PhysicalAccessSpace::Shared,
                1 => PhysicalAccessSpace::Global,
                _ => PhysicalAccessSpace::Tmem,
            };
            let mut lane_spans = BTreeMap::new();
            for lane in WarpMask::from_bits(mask) {
                let first_len = (next_random(&mut seed) % 8 + 1) as usize;
                let first_offset = (next_random(&mut seed) % 128) as usize;
                let mut spans = vec![span(
                    1 + next_random(&mut seed) % 2,
                    first_offset,
                    first_len,
                )];
                if next_random(&mut seed).is_multiple_of(3) {
                    let second_len = (next_random(&mut seed) % 8 + 1) as usize;
                    let second_offset = (next_random(&mut seed) % 128) as usize;
                    spans.push(span(
                        3 + next_random(&mut seed) % 2,
                        second_offset,
                        second_len,
                    ));
                }
                lane_spans.insert(lane, spans);
            }
            let operation = operation(warp_id, sequence, 100 + step, mask);
            let batch =
                PhysicalAccessBatch::resolve_lane_widths(operation, kind, space, |provenance| {
                    Ok::<_, Infallible>(
                        lane_spans
                            .get(&provenance.lane())
                            .expect("every active lane has generated spans")
                            .clone(),
                    )
                })
                .unwrap();

            match (
                reference_validate_batch(&shadow, &batch),
                shadow.validate_batch(&batch),
            ) {
                (Ok(reference), Ok(actual)) => {
                    assert_eq!(actual.warp_id, reference.warp_id, "warp at step {step}");
                    assert_eq!(
                        actual.event_clock, reference.event_clock,
                        "clock at step {step}",
                    );
                    assert_eq!(
                        canonicalize_shadow_segments(materialize_sparse_validation(
                            &shadow, &actual,
                        )),
                        canonicalize_shadow_segments(materialize_reference_validation(
                            &shadow, &reference,
                        )),
                        "shadow at step {step}",
                    );
                    shadow.commit_validation(actual);
                }
                (Err(reference), Err(actual)) => {
                    assert_eq!(actual, reference, "finding at step {step}");
                }
                (Ok(_), Err(actual)) => {
                    panic!("sparse patch found an extra race at step {step}: {actual:?}");
                }
                (Err(reference), Ok(_)) => {
                    panic!("sparse patch missed a race at step {step}: {reference:?}");
                }
            }
        }
    }

    #[test]
    fn automatic_gc_bounds_a_million_unique_accesses_deterministically() {
        const OPERATION_COUNT: usize = 1_000_000;

        fn run() -> (usize, usize, usize) {
            let mut shadow = RaceShadow::new(1);
            let mut sampled_peak = 0;
            for sequence in 0..OPERATION_COUNT {
                shadow
                    .check_batch(single(
                        0,
                        sequence as u64,
                        90,
                        PhysicalAccessKind::Write,
                        10_000 + sequence as u64,
                        0,
                        4,
                    ))
                    .unwrap();
                if sequence % 127 == 0 {
                    sampled_peak = sampled_peak.max(shadow.tracked_interval_count());
                }
            }
            let tail = shadow.tracked_interval_count();
            let retired_tail = shadow.gc_dominated_frontier();
            assert_eq!(shadow.tracked_interval_count(), 0);
            (sampled_peak.max(tail), tail, retired_tail)
        }

        let started = Instant::now();
        let first = run();
        let second = run();
        eprintln!(
            "race shadow automatic-GC stress: {} operations x2 in {:?}, summary={first:?}",
            OPERATION_COUNT,
            started.elapsed(),
        );

        assert_eq!(first, second);
        assert!(first.0 <= AUTOMATIC_GC_SAFE_POINT_INTERVAL);
        assert_eq!(first.1, OPERATION_COUNT % AUTOMATIC_GC_SAFE_POINT_INTERVAL);
        assert_eq!(first.2, first.1);
    }

    #[test]
    fn barrier_safe_points_periodically_collect_globally_observed_frontier() {
        let mut shadow = RaceShadow::new(2);
        shadow
            .check_batch(single(0, 0, 91, PhysicalAccessKind::Write, 10, 0, 4))
            .unwrap();
        assert_eq!(shadow.gc_dominated_frontier(), 0);
        let payload = shadow.barrier_release(0).unwrap();
        shadow.barrier_acquire(1, &payload).unwrap();
        assert_eq!(shadow.tracked_interval_count(), 1);
        for _ in 2..AUTOMATIC_GC_SAFE_POINT_INTERVAL {
            shadow.barrier_acquire(1, &payload).unwrap();
        }
        assert_eq!(shadow.tracked_interval_count(), 0);
        assert_eq!(shadow.gc_dominated_frontier(), 0);
    }
}
