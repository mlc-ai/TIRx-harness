//! Launch-wide, lane-precise global-memory happens-before analysis.
//!
//! This state is deliberately separate from the cluster-sharded SMEM/TMEM
//! shadow. Ordinary mutex/transaction ordering is not a happens-before edge.
//! SC fences additionally select a runtime order, linking only pairs whose
//! scopes mutually cover their issuing threads.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering as AtomicOrdering,
};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::effect::TensorMapObservation;
use crate::physical_access::CompactPhysicalAccessBatch;
use crate::race_shadow::{
    GroupJoinHasherBuilder, PhysicalRaceOrderingFailure, PhysicalRaceProxyDomain, SharedClockFrontier,
    SharedLaneFrontiers,
};
use crate::transactional_interval_map::TransactionalIntervalMap;
use crate::{
    AsyncGroupMilestone, AsyncTokenId, DynamicOpId, LaunchTopology, MemoryAccessSemantics,
    MemoryFenceEffect, MemoryOrder, MemoryProxy, MemoryScope, OperationContext,
    PhysicalAccessBatch, PhysicalAccessDescriptor, PhysicalAccessKind, PhysicalAccessSpace,
    PhysicalAllocationId, PhysicalBarrierId, PhysicalByteSpan, PhysicalRaceFinding,
    PhysicalRaceKind, PhysicalRaceWitness, ProfileKind, ProfileTimer, ProxyAsyncFenceEffect,
    ProxyAsyncFenceScope, WarpMask,
};

use crate::effect::DeclaredWordWaitPlan;

use super::tcgen_fence::{TcgenFenceFrontier, TcgenLaneFrontiers};
use super::RaceCheckIncompleteReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GlobalActor {
    global_warp_id: usize,
    lane: u8,
}

impl GlobalActor {
    fn new(global_warp_id: usize, lane: usize) -> Self {
        debug_assert!(lane < crate::WARP_SIZE);
        Self {
            global_warp_id,
            lane: lane as u8,
        }
    }

    fn dense_index(self) -> usize {
        self.global_warp_id
            .checked_mul(crate::WARP_SIZE)
            .and_then(|base| base.checked_add(self.lane as usize))
            .expect("global lane clock index must fit usize")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GlobalActorRelation {
    SameCta,
    SameCluster,
    CrossCluster,
}

impl fmt::Display for GlobalActorRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SameCta => "same_cta",
            Self::SameCluster => "same_cluster",
            Self::CrossCluster => "cross_cluster",
        })
    }
}

/// One address claimed by a declared synchronization word that some access
/// reached without going through the primitive.
///
/// Design §2.5: a protocol owns its word, so every access to it -- the
/// initialization and the reset included -- goes through the primitive. An
/// access that bypasses it is a defect whether it reads or writes, whichever
/// side ran first, and whether or not the two are ordered. That is what makes
/// the declaration checkable rather than advisory: the exemption in §2.4
/// covers the protocol's own accesses, and this covers everyone else's.
/// A word two actors use as a cross-thread protocol that nothing declares.
///
/// Not a race: both sides may be perfectly ordered. It says the checker was
/// given no rules for this word, so whatever it concludes about the data
/// behind it rests on an agreement it cannot see. Declaring the word with
/// `wait_until` retires this and puts the word under the checks that can
/// actually judge it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UndeclaredProtocolWordDiagnostic {
    /// The word itself, in the shape every other finding reports a location.
    overlap: PhysicalByteSpan,
    write_operation: DynamicOpId,
    peer_operation: DynamicOpId,
    writer_warp_id: usize,
    writer_lane: u8,
    peer_warp_id: usize,
    peer_lane: u8,
}

impl UndeclaredProtocolWordDiagnostic {
    pub const fn overlap(&self) -> PhysicalByteSpan {
        self.overlap
    }

    pub const fn write_operation(&self) -> &DynamicOpId {
        &self.write_operation
    }

    pub const fn peer_operation(&self) -> &DynamicOpId {
        &self.peer_operation
    }

    pub const fn writer_warp_id(&self) -> usize {
        self.writer_warp_id
    }

    pub const fn writer_lane(&self) -> u8 {
        self.writer_lane
    }

    pub const fn peer_warp_id(&self) -> usize {
        self.peer_warp_id
    }

    pub const fn peer_lane(&self) -> u8 {
        self.peer_lane
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeclaredWordBypassDiagnostic {
    declared_operation: DynamicOpId,
    bypassing_operation: DynamicOpId,
    declared_warp_id: usize,
    declared_lane: u8,
    bypassing_warp_id: usize,
    bypassing_lane: u8,
    overlap: PhysicalByteSpan,
}

impl DeclaredWordBypassDiagnostic {
    pub const fn declared_operation(&self) -> &DynamicOpId {
        &self.declared_operation
    }

    pub const fn bypassing_operation(&self) -> &DynamicOpId {
        &self.bypassing_operation
    }

    pub const fn declared_warp_id(&self) -> usize {
        self.declared_warp_id
    }

    pub const fn declared_lane(&self) -> u8 {
        self.declared_lane
    }

    pub const fn bypassing_warp_id(&self) -> usize {
        self.bypassing_warp_id
    }

    pub const fn bypassing_lane(&self) -> u8 {
        self.bypassing_lane
    }

    pub const fn overlap(&self) -> PhysicalByteSpan {
        self.overlap
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalScopeMismatchDiagnostic {
    release_operation: DynamicOpId,
    acquire_operation: DynamicOpId,
    release_scope: MemoryScope,
    acquire_scope: MemoryScope,
    release_warp_id: usize,
    release_lane: u8,
    acquire_warp_id: usize,
    acquire_lane: u8,
    relation: GlobalActorRelation,
}

impl GlobalScopeMismatchDiagnostic {
    pub const fn release_operation(&self) -> &DynamicOpId {
        &self.release_operation
    }

    pub const fn acquire_operation(&self) -> &DynamicOpId {
        &self.acquire_operation
    }

    pub const fn release_scope(&self) -> MemoryScope {
        self.release_scope
    }

    pub const fn acquire_scope(&self) -> MemoryScope {
        self.acquire_scope
    }

    pub const fn release_warp_id(&self) -> usize {
        self.release_warp_id
    }

    pub const fn release_lane(&self) -> usize {
        self.release_lane as usize
    }

    pub const fn acquire_warp_id(&self) -> usize {
        self.acquire_warp_id
    }

    pub const fn acquire_lane(&self) -> usize {
        self.acquire_lane as usize
    }

    pub const fn relation(&self) -> GlobalActorRelation {
        self.relation
    }
}

/// One async slot's component of a clock: the slot's generation (bumped
/// every time the registry recycles the slot, so a stale component of an
/// earlier token never reads as progress of the current one) and the
/// epoch the token reached. Both are `u32`: a launch recycles a slot tens of
/// thousands of times and a token ticks a handful of times, and 32 slots
/// per chunk at 8 B each halve the chunk that every clock join copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct AsyncClockComponent {
    generation: u32,
    epoch: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AsyncClockHandle {
    index: usize,
    generation: u32,
}

#[derive(Debug, Default)]
struct AsyncClockRegistryInner {
    generations: Vec<u32>,
    free: Vec<usize>,
}

#[derive(Debug, Default)]
struct AsyncClockRegistry {
    inner: Mutex<AsyncClockRegistryInner>,
    blocks: BlockArena,
    /// Join of every floor the collector retired frontier entries under, and
    /// whether any retirement happened. A lane whose clock does not dominate
    /// the watermark may still race with a retired record, so its global
    /// accesses are reported as an analysis gap instead of silently checked
    /// against a thinned frontier.
    retirement: RwLock<RetirementWatermark>,
    retired_any: std::sync::atomic::AtomicBool,
}

/// Sparse join of retirement floors: `(dense lane index, epoch)` and
/// `(async slot, component)` entries above zero, sorted by index.
#[derive(Debug, Default)]
struct RetirementWatermark {
    lanes: Vec<(usize, u64)>,
    asyncs: Vec<(usize, AsyncClockComponent)>,
}

impl RetirementWatermark {
    fn join(&mut self, floor: &GlobalFloor) {
        /// Merge a dense floor into the sparse watermark, keeping the entries
        /// of each that the other does not name. A lane the floor holds at
        /// top (no state yet) contributes nothing, so `join` returns the
        /// prior value for it and the leftover tops are dropped below.
        fn merge_sparse<T: Copy>(
            previous: &[(usize, T)],
            dense: &[T],
            is_zero: impl Fn(&T) -> bool,
            join: impl Fn(T, T) -> T,
        ) -> Vec<(usize, T)> {
            let mut merged = Vec::with_capacity(previous.len());
            let mut prior = 0;
            for (index, value) in dense.iter().enumerate() {
                let mut value = *value;
                while prior < previous.len() && previous[prior].0 < index {
                    merged.push(previous[prior]);
                    prior += 1;
                }
                if prior < previous.len() && previous[prior].0 == index {
                    value = join(previous[prior].1, value);
                    prior += 1;
                }
                if !is_zero(&value) {
                    merged.push((index, value));
                }
            }
            merged.extend_from_slice(&previous[prior..]);
            merged
        }
        let lanes = merge_sparse(
            &self.lanes,
            &floor.lanes,
            |epoch| *epoch == 0,
            |prior, current| {
                if current == u64::MAX {
                    prior
                } else {
                    prior.max(current)
                }
            },
        );
        let asyncs = merge_sparse(
            &self.asyncs,
            &floor.asyncs,
            |component| component.epoch == 0,
            |prior, current| {
                if current.generation == u32::MAX {
                    prior
                } else if prior.generation > current.generation
                    || (prior.generation == current.generation && prior.epoch > current.epoch)
                {
                    prior
                } else {
                    current
                }
            },
        );
        // Top entries without a prior value must not survive as MAX: drop them.
        self.lanes = lanes.into_iter().filter(|(_, epoch)| *epoch != u64::MAX).collect();
        self.asyncs = asyncs
            .into_iter()
            .filter(|(_, component)| component.generation != u32::MAX)
            .collect();
    }

    fn is_empty(&self) -> bool {
        self.lanes.is_empty() && self.asyncs.is_empty()
    }

    /// Whether `clock` has observed every retired epoch.
    fn dominated_by(&self, clock: &SparseLaneClock) -> bool {
        if self
            .lanes
            .iter()
            .any(|(index, epoch)| clock.component_epoch(*index) < *epoch)
        {
            return false;
        }
        let arena = clock.arena();
        !self.asyncs.iter().any(|(index, floor)| {
            clock
                .async_components
                .chunk(index / ASYNC_CHUNK, arena)
                .map(|chunk| unpack_component(arena.chunks.view(chunk).word(index % ASYNC_CHUNK)))
                .is_none_or(|component| {
                    component.generation != floor.generation || component.epoch < floor.epoch
                })
        })
    }
}

impl AsyncClockRegistry {
    /// Records that the collector is about to retire entries every present
    /// actor dominates under `floor`.
    fn note_retirement_floor(&self, floor: &GlobalFloor) {
        self.retirement
            .write()
            .expect("retirement watermark lock was poisoned")
            .join(floor);
        self.retired_any
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn retired_any(&self) -> bool {
        self.retired_any.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether a lane with `clock` may be checked against the thinned
    /// frontiers without missing a retired record.
    fn watermark_dominated_by(&self, clock: &SparseLaneClock) -> bool {
        let watermark = self
            .retirement
            .read()
            .expect("retirement watermark lock was poisoned");
        watermark.is_empty() || watermark.dominated_by(clock)
    }
}

impl AsyncClockRegistry {
    fn lease(
        self: &Arc<Self>,
        token: AsyncTokenId,
        issue_epoch: u64,
    ) -> Result<Arc<AsyncClockLease>, String> {
        let mut inner = self
            .inner
            .lock()
            .expect("global racecheck async-clock registry lock was poisoned");
        let handle = if let Some(index) = inner.free.pop() {
            let generation = inner.generations[index].checked_add(1).ok_or_else(|| {
                format!("global async clock generation overflowed at slot {index}")
            })?;
            inner.generations[index] = generation;
            AsyncClockHandle { index, generation }
        } else {
            let index = inner.generations.len();
            inner.generations.push(1);
            AsyncClockHandle {
                index,
                generation: 1,
            }
        };
        drop(inner);
        Ok(Arc::new(AsyncClockLease {
            token,
            handle,
            registry: Arc::downgrade(self),
            issue_epoch,
        }))
    }

    fn release(&self, handle: AsyncClockHandle) {
        let mut inner = self
            .inner
            .lock()
            .expect("global racecheck async-clock registry lock was poisoned");
        debug_assert_eq!(
            inner.generations.get(handle.index),
            Some(&handle.generation)
        );
        debug_assert!(!inner.free.contains(&handle.index));
        inner.free.push(handle.index);
    }

    #[cfg(test)]
    fn slot_counts(&self) -> (usize, usize) {
        let inner = self
            .inner
            .lock()
            .expect("global racecheck async-clock registry lock was poisoned");
        (inner.generations.len(), inner.free.len())
    }
}

#[derive(Debug)]
struct AsyncClockLease {
    token: AsyncTokenId,
    handle: AsyncClockHandle,
    registry: std::sync::Weak<AsyncClockRegistry>,
    /// The epoch the issuing lane had at issue (its own component after the
    /// issue tick): the async clock does not advance the issuer's component,
    /// and a finding against an access of this token is classified by
    /// whether the other side observed the issue.
    issue_epoch: u64,
}

impl Drop for AsyncClockLease {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.release(self.handle);
        }
    }
}

impl PartialEq for AsyncClockLease {
    fn eq(&self, other: &Self) -> bool {
        self.token == other.token && self.handle == other.handle
    }
}

impl Eq for AsyncClockLease {}

impl PartialOrd for AsyncClockLease {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AsyncClockLease {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.token, self.handle).cmp(&(&other.token, other.handle))
    }
}

/// Async slots per shared chunk of an async clock vector.
const ASYNC_CHUNK: usize = 32;

/// Chunks per second-level node of an async clock vector.
const ASYNC_CHUNKS_PER_GROUP: usize = 32;

/// A component packed into one node word: generation high, epoch low, so
/// the `u64` order is the component order and the default packs to zero.
#[inline]
const fn pack_component(component: AsyncClockComponent) -> u64 {
    ((component.generation as u64) << 32) | component.epoch as u64
}

#[inline]
const fn unpack_component(word: u64) -> AsyncClockComponent {
    AsyncClockComponent {
        generation: (word >> 32) as u32,
        epoch: word as u32,
    }
}

type AsyncGroupEntry = (u32, NodeId);

/// The chunk node `slot` of async group `group`, if any.
#[inline]
fn group_slot(group: NodeView<'_>, slot: usize) -> Option<NodeId> {
    let word = group.word(slot);
    (word != NodeId::NONE.0).then_some(NodeId(word))
}

/// Every component of chunk `left` is `<=` the component of `right`
/// (packed words order like components).
fn async_chunk_happens_before(arena: &BlockArena, left: NodeId, right: NodeId) -> bool {
    let (left, right) = (arena.chunks.view(left), arena.chunks.view(right));
    (0..ASYNC_CHUNK).all(|slot| left.word(slot) <= right.word(slot))
}

fn async_chunk_is_default(arena: &BlockArena, chunk: NodeId) -> bool {
    let chunk = arena.chunks.view(chunk);
    (0..ASYNC_CHUNK).all(|slot| chunk.word(slot) == 0)
}

/// Every slot of async group `left` is `<=` the slot of `right`.
fn async_group_happens_before(arena: &BlockArena, left: NodeId, right: NodeId) -> bool {
    let (left, right) = (
        arena.async_groups.view(left),
        arena.async_groups.view(right),
    );
    (0..ASYNC_CHUNKS_PER_GROUP).all(|slot| {
        match (group_slot(left, slot), group_slot(right, slot)) {
            (None, _) => true,
            (Some(own), None) => async_chunk_is_default(arena, own),
            (Some(own), Some(theirs)) => {
                own == theirs || async_chunk_happens_before(arena, own, theirs)
            }
        }
    })
}

fn async_group_is_default(arena: &BlockArena, group: NodeId) -> bool {
    let group = arena.async_groups.view(group);
    (0..ASYNC_CHUNKS_PER_GROUP).all(|slot| {
        group_slot(group, slot).is_none_or(|chunk| async_chunk_is_default(arena, chunk))
    })
}

/// Join two async groups chunk by chunk, memoized per ordered pair of group
/// identities like [`merge_groups`]. Nothing is allocated unless the join is
/// genuinely new.
fn merge_async_groups(current: NodeId, incoming: NodeId, arena: &BlockArena) -> MemoizedNode {
    if current == incoming {
        return MemoizedNode::Current;
    }
    let key = ((current.0 as u128) << 64) | incoming.0 as u128;
    if let Some(join) = arena.async_group_join(key) {
        return join;
    }
    let join = merge_node_groups::<ASYNC_CHUNK>(
        current,
        incoming,
        &arena.async_groups,
        &arena.chunks,
    );
    arena.remember_async_group_join(key, join);
    join
}

/// Async-token components of a clock, stored as two levels of shared nodes:
/// groups of `ASYNC_CHUNKS_PER_GROUP` chunks, each chunk holding
/// `ASYNC_CHUNK` slots.
///
/// The registry recycles thousands of slots in a large launch, and clocks that
/// derive from one another share almost all of them. Sharing keeps a dominance
/// check or join at one pointer comparison per shared group instead of one
/// comparison per slot.
#[derive(Clone, Debug, Default)]
struct AsyncClockVector {
    groups: Option<Arc<AsyncGroupList>>,
}

/// The group list of an async clock vector, tagged with a value identity so
/// whole-list joins can be memoized: a fresh or freshly mutated list gets a
/// new identity, so equal identities always mean equal contents (identity 0
/// is reserved for the shared empty default).
#[derive(Clone, Debug, Default)]
struct AsyncGroupList {
    id: u64,
    entries: Vec<AsyncGroupEntry>,
}

/// Combined length below which an async list join is walked directly: the
/// walk of a short list costs less than the memo lookup and insertion.
const ASYNC_LIST_MEMO_MIN: usize = 8;

/// Result of joining two async group lists.
#[derive(Clone)]
enum AsyncListJoin {
    Current,
    Incoming,
    New(Arc<AsyncGroupList>),
}

impl AsyncClockVector {
    fn groups(&self) -> &[AsyncGroupEntry] {
        self.groups
            .as_deref()
            .map_or(&[], |list| list.entries.as_slice())
    }

    fn is_empty(&self) -> bool {
        self.groups.is_none()
    }

    fn shares_storage_with(&self, other: &Self) -> bool {
        match (&self.groups, &other.groups) {
            (None, None) => true,
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn storage_identity(&self) -> usize {
        self.groups
            .as_ref()
            .map_or(0, |groups| Arc::as_ptr(groups) as usize)
    }

    fn group(&self, group: usize) -> Option<NodeId> {
        let groups = self.groups();
        groups
            .binary_search_by_key(&(group as u32), |(index, _)| *index)
            .ok()
            .map(|position| groups[position].1)
    }

    fn chunk(&self, chunk: usize, arena: &BlockArena) -> Option<NodeId> {
        self.group(chunk / ASYNC_CHUNKS_PER_GROUP)
            .and_then(|group| {
                group_slot(
                    arena.async_groups.view(group),
                    chunk % ASYNC_CHUNKS_PER_GROUP,
                )
            })
    }

    fn get(&self, index: usize, arena: &BlockArena) -> AsyncClockComponent {
        self.chunk(index / ASYNC_CHUNK, arena)
            .map_or_else(AsyncClockComponent::default, |chunk| {
                unpack_component(arena.chunks.view(chunk).word(index % ASYNC_CHUNK))
            })
    }

    fn set(&mut self, index: usize, component: AsyncClockComponent, arena: &BlockArena) {
        let chunk_index = index / ASYNC_CHUNK;
        let group_index = chunk_index / ASYNC_CHUNKS_PER_GROUP;
        // Lists are never patched in place: the registry holds them weakly,
        // and `Arc::make_mut` would move a list out from under its root. A
        // changed list is a fresh value (which the memos want anyway), and
        // so is a changed group (joins are memoized by group identity).
        let mut groups = self.groups().to_vec();
        let position = match groups.binary_search_by_key(&(group_index as u32), |(index, _)| *index)
        {
            Ok(position) => position,
            Err(position) => {
                groups.insert(position, (group_index as u32, arena.empty_async_group()));
                position
            }
        };
        let mut group = arena.async_groups.view(groups[position].1).words();
        let slot = chunk_index % ASYNC_CHUNKS_PER_GROUP;
        let mut chunk = if group[slot] == NodeId::NONE.0 {
            [0_u64; NODE_WORDS]
        } else {
            arena.chunks.view(NodeId(group[slot])).words()
        };
        chunk[index % ASYNC_CHUNK] = pack_component(component);
        group[slot] = arena.chunks.alloc(&chunk).0;
        groups[position].1 = arena.async_groups.alloc(&group);
        groups.shrink_to_fit();
        let list = Arc::new(AsyncGroupList {
            id: arena.next_async_list_id(),
            entries: groups,
        });
        arena.async_lists.register(&list);
        self.groups = Some(list);
    }

    fn iter<'a>(&'a self, arena: &'a BlockArena) -> impl Iterator<Item = AsyncClockComponent> + 'a {
        self.groups().iter().flat_map(move |(_, group)| {
            let group = arena.async_groups.view(*group);
            (0..ASYNC_CHUNKS_PER_GROUP)
                .filter_map(move |slot| group_slot(group, slot))
                .flat_map(move |chunk| {
                    let chunk = arena.chunks.view(chunk);
                    (0..ASYNC_CHUNK).map(move |slot| unpack_component(chunk.word(slot)))
                })
        })
    }

    /// `self <= other` on every slot (absent slots are the default).
    fn happens_before(&self, other: &Self, arena: &BlockArena) -> bool {
        if self.shares_storage_with(other) {
            return true;
        }
        let own = self.groups();
        let theirs = other.groups();
        let mut their_index = 0;
        own.iter().all(|(group, own_group)| {
            while their_index < theirs.len() && theirs[their_index].0 < *group {
                their_index += 1;
            }
            match theirs.get(their_index) {
                Some((their_group, their_node)) if their_group == group => {
                    own_group == their_node
                        || async_group_happens_before(arena, *own_group, *their_node)
                }
                _ => async_group_is_default(arena, *own_group),
            }
        })
    }

    /// Join `incoming` into `self`, sharing storage wherever one side already
    /// dominates. Groups both sides share are skipped by identity, joins of
    /// differing groups are memoized in `arena`, and the group list is patched
    /// in place when this clock owns it alone.
    fn merge(&mut self, incoming: &Self, arena: &BlockArena) {
        if self.shares_storage_with(incoming) {
            return;
        }
        let current = self.groups();
        let theirs = incoming.groups();
        if current.is_empty() {
            *self = incoming.clone();
            return;
        }
        if theirs.is_empty() {
            return;
        }
        // Whole-list joins repeat massively (many lanes and warps join the
        // same release frontier), so the outcome is memoized per ordered
        // pair of list identities — but only for lists long enough that the
        // walk costs more than the memo bookkeeping (short lists are the
        // norm while slot recycling keeps up).
        let memoize = current.len() + theirs.len() >= ASYNC_LIST_MEMO_MIN;
        let current_list = self
            .groups
            .as_ref()
            .expect("a non-empty async clock vector holds groups");
        let incoming_list = incoming
            .groups
            .as_ref()
            .expect("a non-empty async clock vector holds groups");
        let key = ((current_list.id as u128) << 64) | incoming_list.id as u128;
        if memoize {
            if let Some(join) = arena.async_list_join(key) {
                match join {
                    AsyncListJoin::Current => {}
                    AsyncListJoin::Incoming => *self = incoming.clone(),
                    AsyncListJoin::New(list) => self.groups = Some(list),
                }
                return;
            }
        }
        let mut replacements: Vec<(usize, NodeId)> = Vec::new();
        let mut insertions: Vec<AsyncGroupEntry> = Vec::new();
        let mut incoming_changed = false;
        let mut current_index = 0;
        let mut incoming_index = 0;
        while current_index < current.len() || incoming_index < theirs.len() {
            match (current.get(current_index), theirs.get(incoming_index)) {
                (Some((own_index, own)), Some((their_index, their)))
                    if own_index == their_index =>
                {
                    match merge_async_groups(*own, *their, arena) {
                        MemoizedNode::Current => {
                            if own != their {
                                incoming_changed = true;
                            }
                        }
                        MemoizedNode::Incoming => {
                            replacements.push((current_index, *their));
                        }
                        MemoizedNode::New(group) => {
                            replacements.push((current_index, group));
                            incoming_changed = true;
                        }
                    }
                    current_index += 1;
                    incoming_index += 1;
                }
                (Some((own_index, _)), Some((their_index, _))) if own_index < their_index => {
                    incoming_changed = true;
                    current_index += 1;
                }
                (Some(_), Some((their_index, their))) => {
                    insertions.push((*their_index, *their));
                    incoming_index += 1;
                }
                (Some(_), None) => {
                    incoming_changed = true;
                    current_index += 1;
                }
                (None, Some((their_index, their))) => {
                    insertions.push((*their_index, *their));
                    incoming_index += 1;
                }
                (None, None) => break,
            }
        }
        if replacements.is_empty() && insertions.is_empty() {
            if memoize {
                arena.remember_async_list_join(key, &AsyncListJoin::Current);
            }
            return;
        }
        if !incoming_changed {
            if memoize {
                arena.remember_async_list_join(key, &AsyncListJoin::Incoming);
            }
            *self = incoming.clone();
            return;
        }
        // Build the joined list as a fresh shared value so the memo can hand
        // it to every later identical join.
        let mut entries = current.to_vec();
        for (position, group) in replacements {
            entries[position].1 = group;
        }
        for (group_index, group) in insertions {
            let position = entries
                .binary_search_by_key(&group_index, |(index, _)| *index)
                .unwrap_or_else(|position| position);
            entries.insert(position, (group_index, group));
        }
        entries.shrink_to_fit();
        let list = Arc::new(AsyncGroupList {
            id: arena.next_async_list_id(),
            entries,
        });
        arena.async_lists.register(&list);
        if memoize {
            arena.remember_async_list_join(key, &AsyncListJoin::New(Arc::clone(&list)));
        }
        self.groups = Some(list);
    }
}

fn async_clock_happens_before(
    left: &AsyncClockVector,
    right: &AsyncClockVector,
    arena: &BlockArena,
) -> bool {
    left.happens_before(right, arena)
}

fn async_clock_equal(
    left: &AsyncClockVector,
    right: &AsyncClockVector,
    arena: &BlockArena,
) -> bool {
    async_clock_happens_before(left, right, arena) && async_clock_happens_before(right, left, arena)
}

/// Lane clock for launch-wide global-memory happens-before state.
///
/// Lane components are stored per warp: a sorted list of `(warp, block)`
/// pairs where each block holds the 32 lane epochs of that warp behind an
/// `Arc`. Clocks derived from one another share both the list and the blocks,
/// so joining two clocks costs one pointer comparison per warp they have in
/// common and only touches the blocks that actually differ; the cost no longer
/// scales with the number of lanes a clock has heard of.
///
/// Async actors use launch-local dense component indices. The previous
/// `BTreeMap<AsyncTokenId, u64>` frontier cloned and compared long token keys
/// at every barrier merge. A shared registry preserves the exact identity
/// relation while dense slices make comparable frontiers cheap to adopt.
#[derive(Clone, Debug)]
struct SparseLaneClock {
    // Synchronization makes the common clock wide, while one actor normally
    // advances only its own component afterward. Share the block list and keep
    // those actor-local advances in a small update layer so a warp sync does
    // not rebuild the list once per participating lane.
    component_base: SparseComponentBase,
    component_updates: SparseComponentUpdates,
    async_components: AsyncClockVector,
    async_registry: Arc<AsyncClockRegistry>,
    proxy_bridges: Option<Arc<GlobalProxyBridgeFrontiers>>,
}

impl SparseLaneClock {
    fn new(async_registry: Arc<AsyncClockRegistry>) -> Self {
        Self {
            component_base: SparseComponentBase::Empty,
            component_updates: SparseComponentUpdates::Empty,
            async_components: AsyncClockVector::default(),
            async_registry,
            proxy_bridges: None,
        }
    }
}

impl Default for SparseLaneClock {
    fn default() -> Self {
        Self::new(Arc::new(AsyncClockRegistry::default()))
    }
}

/// Words per clock node: every node kind — lane block, warp group, async
/// group, async chunk — is exactly 32 words.
const NODE_WORDS: usize = 32;

/// Identity of a clock node: the slot's generation in the high 32 bits and
/// its index in the low 32, so a reclaimed and reused slot never answers to
/// the identity of its previous occupant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct NodeId(u64);

impl NodeId {
    /// The absent node (an empty warp or chunk slot of a group).
    const NONE: Self = Self(u64::MAX);

    const fn new(index: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | index as u64)
    }

    const fn index(self) -> usize {
        (self.0 & 0xffff_ffff) as usize
    }

    const fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// One slot of a [`NodeSlab`]: a generation (odd while the slot holds a
/// node, even while it is free) and the node's 32 words.
///
/// The words are atomics so that the slab needs no `unsafe`: a node is
/// written before its identity is published (release on the generation),
/// read through relaxed loads that are plain loads on every target this
/// runs on, and only reclaimed under the collector's quiescent point, when
/// no reader can hold its identity.
struct NodeSlot {
    generation: AtomicU32,
    words: [AtomicU64; NODE_WORDS],
}

impl NodeSlot {
    fn new() -> Self {
        Self {
            generation: AtomicU32::new(0),
            words: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

/// A validated view of one live node.
#[derive(Clone, Copy)]
struct NodeView<'a> {
    slot: &'a NodeSlot,
}

impl NodeView<'_> {
    #[inline]
    fn word(&self, index: usize) -> u64 {
        self.slot.words[index].load(AtomicOrdering::Relaxed)
    }

    fn words(&self) -> [u64; NODE_WORDS] {
        std::array::from_fn(|index| self.word(index))
    }
}

const ARENA_CHUNK_BLOCKS: usize = 4096;
const ARENA_MAX_CHUNKS: usize = 1 << 16;

/// Slab of clock nodes addressed by [`NodeId`], with slot reuse.
///
/// Storage is chunked so a slot's address never moves; reads are lock-free
/// (`OnceLock::get` plus atomic loads); allocation serializes on the free
/// list's mutex. Slots are reclaimed by [`Self::sweep`] under the global
/// floor collector, which holds every shard lock — the only time no clock
/// operation can be reading a node — and frees each slot that no live group
/// list reaches (see [`BlockArena::sweep_nodes`]). Earlier versions either
/// never freed nodes (24 GB of dead clocks on MegaMoE t128_m128) or shared
/// them by `Arc`, whose adoption traffic cost 13 % of worker time on the
/// e384 configs: an adoption is again one word copy.
struct NodeSlab {
    chunks: OnceLock<Box<[OnceLock<Box<[NodeSlot]>>]>>,
    high: AtomicU32,
    free: Mutex<Vec<u32>>,
    /// Nodes ever allocated — the only counter an allocation touches; the
    /// live count and the allocations since the last sweep are derived from
    /// it and the two counters the sweep maintains.
    allocated: AtomicUsize,
    freed: AtomicUsize,
    allocated_at_sweep: AtomicUsize,
    /// Mark bits, double-buffered: `marks[current]` is the map of the mark
    /// in progress (or the last one), the other map is clear and becomes
    /// current at the next `begin_mark`. While `marking` is set, every
    /// allocation marks its slot in the current map — the write barrier
    /// that lets the mark walk run without the shard locks.
    marks: [MarkBitmap; 2],
    current: AtomicUsize,
    marking: AtomicBool,
}

/// One bit per slot, chunked like the slots so it grows lazily and never
/// moves.
struct MarkBitmap {
    chunks: OnceLock<Box<[OnceLock<Box<[AtomicU64]>>]>>,
}

const MARK_WORDS_PER_CHUNK: usize = ARENA_CHUNK_BLOCKS / 64;

impl MarkBitmap {
    fn new() -> Self {
        Self {
            chunks: OnceLock::new(),
        }
    }

    fn word(&self, index: usize) -> &AtomicU64 {
        let chunks = self
            .chunks
            .get_or_init(|| (0..ARENA_MAX_CHUNKS).map(|_| OnceLock::new()).collect());
        let chunk = chunks[index / ARENA_CHUNK_BLOCKS].get_or_init(|| {
            (0..MARK_WORDS_PER_CHUNK)
                .map(|_| AtomicU64::new(0))
                .collect()
        });
        &chunk[(index % ARENA_CHUNK_BLOCKS) / 64]
    }

    /// Sets the bit; returns whether it was clear.
    fn set(&self, index: usize) -> bool {
        let bit = 1 << (index % 64);
        self.word(index).fetch_or(bit, AtomicOrdering::Relaxed) & bit == 0
    }

    fn get(&self, index: usize) -> bool {
        self.chunks
            .get()
            .and_then(|chunks| chunks[index / ARENA_CHUNK_BLOCKS].get())
            .is_some_and(|chunk| {
                chunk[(index % ARENA_CHUNK_BLOCKS) / 64].load(AtomicOrdering::Relaxed)
                    & (1 << (index % 64))
                    != 0
            })
    }

    fn clear(&self) {
        if let Some(chunks) = self.chunks.get() {
            for chunk in chunks.iter() {
                if let Some(chunk) = chunk.get() {
                    for word in chunk.iter() {
                        word.store(0, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
    }
}

impl NodeSlab {
    fn new() -> Self {
        Self {
            chunks: OnceLock::new(),
            high: AtomicU32::new(0),
            free: Mutex::new(Vec::new()),
            allocated: AtomicUsize::new(0),
            freed: AtomicUsize::new(0),
            allocated_at_sweep: AtomicUsize::new(0),
            marks: [MarkBitmap::new(), MarkBitmap::new()],
            current: AtomicUsize::new(0),
            marking: AtomicBool::new(false),
        }
    }

    fn current_marks(&self) -> &MarkBitmap {
        &self.marks[self.current.load(AtomicOrdering::Acquire) & 1]
    }

    /// Mark `id` live in the mark in progress; returns whether this was the
    /// first visit.
    fn mark(&self, id: NodeId) -> bool {
        self.current_marks().set(id.index())
    }

    /// Switch to the clear bitmap and start marking allocations. Only under
    /// the collector's quiescent point, so no allocation straddles the switch.
    fn begin_mark(&self) {
        debug_assert!(!self.marking.load(AtomicOrdering::Relaxed));
        self.current.fetch_xor(1, AtomicOrdering::AcqRel);
        self.marking.store(true, AtomicOrdering::Release);
    }

    fn slot(&self, index: usize) -> &NodeSlot {
        let chunks = self
            .chunks
            .get_or_init(|| (0..ARENA_MAX_CHUNKS).map(|_| OnceLock::new()).collect());
        let chunk = chunks[index / ARENA_CHUNK_BLOCKS]
            .get_or_init(|| (0..ARENA_CHUNK_BLOCKS).map(|_| NodeSlot::new()).collect());
        &chunk[index % ARENA_CHUNK_BLOCKS]
    }

    fn published_slot(&self, index: usize) -> Option<&NodeSlot> {
        self.chunks
            .get()
            .and_then(|chunks| chunks.get(index / ARENA_CHUNK_BLOCKS)?.get())
            .map(|chunk| &chunk[index % ARENA_CHUNK_BLOCKS])
    }

    fn alloc(&self, words: &[u64; NODE_WORDS]) -> NodeId {
        let index = {
            let mut free = self
                .free
                .lock()
                .expect("global racecheck clock node free list was poisoned");
            free.pop()
        }
        .unwrap_or_else(|| {
            let index = self.high.fetch_add(1, AtomicOrdering::Relaxed);
            assert!(
                (index as usize) / ARENA_CHUNK_BLOCKS < ARENA_MAX_CHUNKS,
                "global racecheck clock node slab exhausted"
            );
            index
        });
        let slot = self.slot(index as usize);
        for (word, value) in slot.words.iter().zip(words) {
            word.store(*value, AtomicOrdering::Relaxed);
        }
        let generation = slot
            .generation
            .load(AtomicOrdering::Relaxed)
            .wrapping_add(1);
        debug_assert!(generation % 2 == 1, "a free slot has an even generation");
        if self.marking.load(AtomicOrdering::Acquire) {
            // Allocated during a mark: live by definition. Marked before the
            // generation publishes the node, so a sweep that reads the odd
            // generation reads the mark as well.
            self.current_marks().set(index as usize);
        }
        slot.generation.store(generation, AtomicOrdering::Release);
        self.allocated.fetch_add(1, AtomicOrdering::Relaxed);
        NodeId::new(index, generation)
    }

    /// The node `id` names; panics if the slot was reclaimed, which the
    /// collector's protocol rules out for any identity a clock still holds.
    #[inline]
    fn view(&self, id: NodeId) -> NodeView<'_> {
        let slot = self
            .published_slot(id.index())
            .expect("global racecheck clock node identity is not published");
        assert_eq!(
            slot.generation.load(AtomicOrdering::Acquire),
            id.generation(),
            "global racecheck clock node was reclaimed while still referenced"
        );
        NodeView { slot }
    }

    fn is_live(&self, id: NodeId) -> bool {
        self.published_slot(id.index())
            .is_some_and(|slot| slot.generation.load(AtomicOrdering::Acquire) == id.generation())
    }

    fn high(&self) -> usize {
        self.high.load(AtomicOrdering::Relaxed) as usize
    }

    /// Free the live slots in `start..end` that the finished mark did not
    /// reach; returns how many. Runs with the workers going: the caller holds
    /// the memo shard locks that a hit adopting an unmarked node would need,
    /// the free list lock orders the range against allocations, and a node
    /// allocated since the mark began is marked before it is published.
    fn sweep_range(&self, start: usize, end: usize) -> usize {
        debug_assert!(self.marking.load(AtomicOrdering::Relaxed));
        let marks = self.current_marks();
        let mut free = self
            .free
            .lock()
            .expect("global racecheck clock node free list was poisoned");
        let mut freed = 0usize;
        for index in start..end {
            let Some(slot) = self.published_slot(index) else {
                continue;
            };
            let generation = slot.generation.load(AtomicOrdering::Acquire);
            if generation % 2 == 0 || marks.get(index) {
                continue;
            }
            slot.generation
                .store(generation.wrapping_add(1), AtomicOrdering::Release);
            free.push(index as u32);
            freed += 1;
        }
        freed
    }

    /// End the mark once every range was swept.
    fn finish_sweep(&self, freed: usize) {
        self.marking.store(false, AtomicOrdering::Release);
        self.freed.fetch_add(freed, AtomicOrdering::Relaxed);
        self.allocated_at_sweep.store(
            self.allocated.load(AtomicOrdering::Relaxed),
            AtomicOrdering::Relaxed,
        );
    }

    fn live(&self) -> usize {
        self.allocated
            .load(AtomicOrdering::Relaxed)
            .saturating_sub(self.freed.load(AtomicOrdering::Relaxed))
    }

    fn since_sweep(&self) -> usize {
        self.allocated
            .load(AtomicOrdering::Relaxed)
            .saturating_sub(self.allocated_at_sweep.load(AtomicOrdering::Relaxed))
    }

    /// Clear the bitmap that the next mark will use (outside the quiescent
    /// point: nobody marks into the non-current map).
    fn clear_next_marks(&self) {
        self.marks[(self.current.load(AtomicOrdering::Acquire) & 1) ^ 1].clear();
    }
}

/// Registry of the live group lists of a launch — the roots of every clock
/// node. Lists are held weakly: a dropped list simply fails to upgrade and
/// is pruned at the next walk, so no `Drop` bookkeeping runs on the hot path.
struct ListRegistry<T> {
    shards: Box<[Mutex<Vec<std::sync::Weak<T>>>]>,
}

impl<T> ListRegistry<T> {
    fn new() -> Self {
        Self {
            shards: (0..MEMO_SHARDS).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }

    fn register(&self, list: &Arc<T>) {
        let shard = (Arc::as_ptr(list) as usize >> 6) % MEMO_SHARDS;
        self.shards[shard]
            .lock()
            .expect("global racecheck list registry lock was poisoned")
            .push(Arc::downgrade(list));
    }

    /// Every live list, held strongly, pruning the dead entries.
    fn snapshot(&self) -> Vec<Arc<T>> {
        let mut lists = Vec::new();
        for shard in self.shards.iter() {
            let mut entries = shard
                .lock()
                .expect("global racecheck list registry lock was poisoned");
            entries.retain(|weak| match weak.upgrade() {
                Some(list) => {
                    lists.push(list);
                    true
                }
                None => false,
            });
        }
        lists
    }
}

/// The root lists of a mark, held alive for its duration.
pub(crate) struct MarkSnapshot {
    base_lists: Vec<Arc<WarpGroupList>>,
    async_lists: Vec<Arc<AsyncGroupList>>,
}

impl MarkSnapshot {
}

/// Nodes allocated since the last sweep before the collector is asked for a
/// pass on the slab's account.
const NODE_SWEEP_EVERY: usize = 1 << 20;
/// A pass is also due once the allocations since the last sweep reach this
/// fraction of the live nodes: with the collection on its own thread the
/// workers only pay the two short lock windows, so the slabs are allowed a
/// quarter of garbage rather than half.
const NODE_SWEEP_DIVISOR: usize = 4;

/// Slots a sweep frees under one hold of the memo shard locks.
const NODE_SWEEP_RANGE: usize = 1 << 14;

/// The clock nodes of a launch, their identities, the roots that keep them
/// alive and the join memos.
///
/// Every node kind lives in a [`NodeSlab`]; clocks reference nodes by
/// [`NodeId`], so cloning a clock or adopting a group touches no reference
/// count. The group lists that hold the identities register themselves, and
/// [`Self::sweep_nodes`] frees every node no live list reaches. Memoized
/// fresh values are identities too, validated on every hit.
///
/// A collection is a mark walk between two quiescent points: `begin_mark`
/// (switch bitmaps, snapshot the roots) and `sweep_nodes` (free the
/// unmarked); the walk itself runs while the workers go on, allocations made
/// meanwhile marking themselves.
struct BlockArena {
    blocks: NodeSlab,
    groups: NodeSlab,
    async_groups: NodeSlab,
    chunks: NodeSlab,
    empty_group: OnceLock<NodeId>,
    empty_async_group: OnceLock<NodeId>,
    base_lists: ListRegistry<WarpGroupList>,
    async_lists: ListRegistry<AsyncGroupList>,
    // Value identities for async group lists and clock-base group lists
    // (0 = the empty default) and the memoized whole-list joins keyed by
    // ordered identity pair. A memoized fresh value is held weakly: the memo
    // hands repeats of one join the same shared value, but must not keep
    // values alive that every clock has already moved past.
    next_async_list_id: AtomicU64,
    async_list_joins:
        OnceLock<Box<[Mutex<HashMap<u128, MemoizedJoin<AsyncGroupList>, GroupJoinHasherBuilder>>]>>,
    next_base_list_id: AtomicU64,
    base_list_joins:
        OnceLock<Box<[Mutex<HashMap<u128, MemoizedJoin<WarpGroupList>, GroupJoinHasherBuilder>>]>>,
    // Memoized group joins keyed by `(current id << 64) | incoming id`,
    // sharded by key so concurrent shards rarely touch the same lock.
    group_joins: OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
    async_group_joins: OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
}

const MEMO_SHARDS: usize = 64;
/// Entries per memo shard before the shard is pruned (and, if pruning frees
/// little, cleared). Join memos are caches over identities that are never
/// reused, so entries for dead values only ever waste memory.
const MEMO_SHARD_CAP: usize = 1 << 15;

fn bound_memo_shard<K, V>(
    joins: &mut HashMap<K, V, GroupJoinHasherBuilder>,
    is_dead: impl Fn(&V) -> bool,
) {
    if joins.len() < MEMO_SHARD_CAP {
        return;
    }
    joins.retain(|_, join| !is_dead(join));
    if joins.len() >= MEMO_SHARD_CAP / 2 {
        joins.clear();
    }
}

fn memoized_join_is_dead<T>(join: &MemoizedJoin<T>) -> bool {
    matches!(join, MemoizedJoin::New(weak) if weak.strong_count() == 0)
}

impl Default for BlockArena {
    fn default() -> Self {
        Self {
            blocks: NodeSlab::new(),
            groups: NodeSlab::new(),
            async_groups: NodeSlab::new(),
            chunks: NodeSlab::new(),
            empty_group: OnceLock::new(),
            empty_async_group: OnceLock::new(),
            base_lists: ListRegistry::new(),
            async_lists: ListRegistry::new(),
            next_async_list_id: AtomicU64::new(1),
            async_list_joins: OnceLock::new(),
            next_base_list_id: AtomicU64::new(1),
            base_list_joins: OnceLock::new(),
            group_joins: OnceLock::new(),
            async_group_joins: OnceLock::new(),
        }
    }
}

/// A memoized join outcome whose fresh value is only weakly retained.
enum MemoizedJoin<T> {
    Current,
    Incoming,
    New(std::sync::Weak<T>),
}

/// A memoized group join: the fresh node is an identity, validated on use.
#[derive(Clone, Copy)]
enum MemoizedNode {
    Current,
    Incoming,
    New(NodeId),
}

fn memo_shards<K, V>() -> Box<[Mutex<HashMap<K, V, GroupJoinHasherBuilder>>]> {
    (0..MEMO_SHARDS)
        .map(|_| Mutex::new(HashMap::default()))
        .collect()
}

impl fmt::Debug for BlockArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockArena")
            .field("blocks_live", &self.blocks.live())
            .field("chunks_live", &self.chunks.live())
            .finish()
    }
}

impl BlockArena {
    fn empty_group(&self) -> NodeId {
        *self
            .empty_group
            .get_or_init(|| self.groups.alloc(&[NodeId::NONE.0; NODE_WORDS]))
    }

    fn empty_async_group(&self) -> NodeId {
        *self
            .empty_async_group
            .get_or_init(|| self.async_groups.alloc(&[NodeId::NONE.0; NODE_WORDS]))
    }

    /// Whether enough nodes were allocated since the last sweep to warrant
    /// asking the collector for a quiescent point: at least
    /// `NODE_SWEEP_EVERY`, and at least a fraction (`NODE_SWEEP_DIVISOR`) of
    /// the live node count, so a sweep (whose mark walks every live list)
    /// amortizes against the garbage it can free on the configs that keep
    /// millions of nodes live.
    fn sweep_due(&self) -> bool {
        let slabs = [&self.blocks, &self.groups, &self.async_groups, &self.chunks];
        let since: usize = slabs.iter().map(|slab| slab.since_sweep()).sum();
        let live: usize = slabs.iter().map(|slab| slab.live()).sum();
        since >= NODE_SWEEP_EVERY.max(live / NODE_SWEEP_DIVISOR)
    }

    /// Start a collection: switch every slab to its clear bitmap and take
    /// the root lists. Only under the collector's quiescent point (every
    /// shard's global state locked), so that no clock operation straddles
    /// the switch: every node a worker holds at the switch is in a list of
    /// the snapshot, from here on every allocation marks itself and every
    /// memo hit marks the node it adopts, and any node a later operation
    /// adopts comes from a snapshot list, a memo hit, or such an allocation.
    /// (The snapshot must be taken here: a list alive at the switch may be
    /// gone before a walk outside the lock reaches its registry shard, after
    /// an operation adopted its blocks into a group allocated meanwhile.)
    fn begin_mark(&self) -> MarkSnapshot {
        for slab in [&self.blocks, &self.groups, &self.async_groups, &self.chunks] {
            slab.begin_mark();
        }
        MarkSnapshot {
            base_lists: self.base_lists.snapshot(),
            async_lists: self.async_lists.snapshot(),
        }
    }

    /// Mark every node the snapshot reaches. Needs no lock: lists are
    /// immutable, the snapshot holds them alive, and nothing is freed until
    /// `sweep_nodes`.
    fn mark(&self, snapshot: &MarkSnapshot) {
        if let Some(id) = self.empty_group.get() {
            self.groups.mark(*id);
        }
        if let Some(id) = self.empty_async_group.get() {
            self.async_groups.mark(*id);
        }
        // Lists share most of their groups: a group's members are walked on
        // its first visit only.
        for list in &snapshot.base_lists {
            for (_, group) in &list.entries {
                self.mark_group_deep(*group);
            }
        }
        for list in &snapshot.async_lists {
            for (_, group) in &list.entries {
                self.mark_async_group_deep(*group);
            }
        }
    }

    /// Free every node the mark did not reach, with the workers going. Every
    /// identity a list holds was marked (by the walk, by the allocation, or
    /// by the registration of the list); the one way a worker adopts a node
    /// no list holds is a memo hit, which marks the node and its members
    /// under the memo's shard lock, so each range is swept under those locks:
    /// a hit either marked the node before its range or finds it reclaimed.
    /// Groups go before their members, so a member is freed only once no
    /// group that could still be adopted names it.
    fn sweep_nodes(&self) {
        Self::sweep_slabs(&self.group_joins, &self.groups, &self.blocks);
        Self::sweep_slabs(&self.async_group_joins, &self.async_groups, &self.chunks);
        for slab in [&self.blocks, &self.groups, &self.async_groups, &self.chunks] {
            slab.clear_next_marks();
        }
    }

    fn sweep_slabs(
        memo: &OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
        groups: &NodeSlab,
        members: &NodeSlab,
    ) {
        let mut freed = [0usize; 2];
        for (slab, freed) in [groups, members].into_iter().zip(freed.iter_mut()) {
            let end = slab.high();
            let mut start = 0;
            while start < end {
                let range_end = end.min(start + NODE_SWEEP_RANGE);
                let _memo = Self::lock_memo_shards(memo);
                *freed += slab.sweep_range(start, range_end);
                start = range_end;
            }
        }
        groups.finish_sweep(freed[0]);
        members.finish_sweep(freed[1]);
    }

    fn lock_memo_shards<'a>(
        memo: &'a OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
    ) -> Vec<MutexGuard<'a, HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>> {
        memo.get()
            .map(|shards| {
                shards
                    .iter()
                    .map(|shard| {
                        shard
                            .lock()
                            .expect("global racecheck group join memo lock was poisoned")
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn memo_shard(key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize
    }

    fn pair_shard(key: u128) -> usize {
        Self::memo_shard((key as u64) ^ ((key >> 64) as u64))
    }

    /// A memo hit; `mark_deep` marks the node and its members while the
    /// memo shard is still locked, so that a sweep (which holds every memo
    /// shard while it frees a range) either sees the mark or reclaimed the
    /// node before the hit could validate it.
    fn node_join(
        table: &OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
        slab: &NodeSlab,
        key: u128,
        mark_deep: impl FnOnce(NodeId),
    ) -> Option<MemoizedNode> {
        let mut joins = table.get()?[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck group join memo lock was poisoned");
        let join = *joins.get(&key)?;
        if let MemoizedNode::New(id) = join {
            if !slab.is_live(id) {
                joins.remove(&key);
                return None;
            }
            if slab.marking.load(AtomicOrdering::Acquire) {
                mark_deep(id);
            }
        }
        Some(join)
    }

    fn remember_node_join(
        table: &OnceLock<Box<[Mutex<HashMap<u128, MemoizedNode, GroupJoinHasherBuilder>>]>>,
        slab: &NodeSlab,
        key: u128,
        join: MemoizedNode,
    ) {
        let mut joins = table.get_or_init(memo_shards)[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck group join memo lock was poisoned");
        joins.insert(key, join);
        bound_memo_shard(
            &mut joins,
            |join| matches!(join, MemoizedNode::New(id) if !slab.is_live(*id)),
        );
    }

    /// Mark a warp group and its blocks (a memo hit during a mark adopts a
    /// node the snapshot may not reach: it is live because the memo kept its
    /// identity valid, not because any root list holds it).
    fn mark_group_deep(&self, group: NodeId) {
        if self.groups.mark(group) {
            let view = self.groups.view(group);
            for slot in 0..WARPS_PER_GROUP {
                if let Some(block) = group_slot(view, slot) {
                    self.blocks.mark(block);
                }
            }
        }
    }

    fn mark_async_group_deep(&self, group: NodeId) {
        if self.async_groups.mark(group) {
            let view = self.async_groups.view(group);
            for slot in 0..ASYNC_CHUNKS_PER_GROUP {
                if let Some(chunk) = group_slot(view, slot) {
                    self.chunks.mark(chunk);
                }
            }
        }
    }

    fn group_join(&self, key: u128) -> Option<MemoizedNode> {
        Self::node_join(&self.group_joins, &self.groups, key, |id| {
            self.mark_group_deep(id)
        })
    }

    fn remember_group_join(&self, key: u128, join: MemoizedNode) {
        Self::remember_node_join(&self.group_joins, &self.groups, key, join);
    }

    fn async_group_join(&self, key: u128) -> Option<MemoizedNode> {
        Self::node_join(&self.async_group_joins, &self.async_groups, key, |id| {
            self.mark_async_group_deep(id)
        })
    }

    fn remember_async_group_join(&self, key: u128, join: MemoizedNode) {
        Self::remember_node_join(&self.async_group_joins, &self.async_groups, key, join);
    }

    fn next_async_list_id(&self) -> u64 {
        self.next_async_list_id
            .fetch_add(1, AtomicOrdering::Relaxed)
    }

    fn next_base_list_id(&self) -> u64 {
        self.next_base_list_id.fetch_add(1, AtomicOrdering::Relaxed)
    }

    fn base_list_join(&self, key: u128) -> Option<BaseListJoin> {
        let mut joins = self.base_list_joins.get()?[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck base list join memo lock was poisoned");
        let join = match joins.get(&key)? {
            MemoizedJoin::Current => Some(BaseListJoin::Current),
            MemoizedJoin::Incoming => Some(BaseListJoin::Incoming),
            MemoizedJoin::New(list) => list.upgrade().map(BaseListJoin::New),
        };
        if join.is_none() {
            joins.remove(&key);
        }
        join
    }

    fn remember_base_list_join(&self, key: u128, join: &BaseListJoin) {
        let memoized = match join {
            BaseListJoin::Current => MemoizedJoin::Current,
            BaseListJoin::Incoming => MemoizedJoin::Incoming,
            BaseListJoin::New(list) => MemoizedJoin::New(Arc::downgrade(list)),
        };
        let mut joins = self.base_list_joins.get_or_init(memo_shards)[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck base list join memo lock was poisoned");
        joins.insert(key, memoized);
        bound_memo_shard(&mut joins, memoized_join_is_dead);
    }

    fn async_list_join(&self, key: u128) -> Option<AsyncListJoin> {
        let mut joins = self.async_list_joins.get()?[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck async list join memo lock was poisoned");
        let join = match joins.get(&key)? {
            MemoizedJoin::Current => Some(AsyncListJoin::Current),
            MemoizedJoin::Incoming => Some(AsyncListJoin::Incoming),
            MemoizedJoin::New(list) => list.upgrade().map(AsyncListJoin::New),
        };
        if join.is_none() {
            joins.remove(&key);
        }
        join
    }

    fn remember_async_list_join(&self, key: u128, join: &AsyncListJoin) {
        let memoized = match join {
            AsyncListJoin::Current => MemoizedJoin::Current,
            AsyncListJoin::Incoming => MemoizedJoin::Incoming,
            AsyncListJoin::New(list) => MemoizedJoin::New(Arc::downgrade(list)),
        };
        let mut joins = self.async_list_joins.get_or_init(memo_shards)[Self::pair_shard(key)]
            .lock()
            .expect("global racecheck async list join memo lock was poisoned");
        joins.insert(key, memoized);
        bound_memo_shard(&mut joins, memoized_join_is_dead);
    }
}

/// Warps per second-level node of a clock base.
const WARPS_PER_GROUP: usize = 32;

/// One group in a clock base: `(group index, warp-group node)`. A warp
/// group's 32 words are the lane-block identities of 32 consecutive warps
/// (`NodeId::NONE` where a warp has none); groups are shared between
/// clocks, so a join skips every group both clocks already share with one
/// identity comparison and only walks the groups that differ.
type GroupEntry = (usize, NodeId);

/// Lane epochs and packed async generations/epochs share component-wise max.
/// Each caller retains its own slabs, memo table, and reclamation domain.
fn merge_node_groups<const COMPONENTS: usize>(
    current: NodeId,
    incoming: NodeId,
    groups: &NodeSlab,
    components: &NodeSlab,
) -> MemoizedNode {
    let dominates = |left: NodeView<'_>, right: NodeView<'_>| {
        (0..COMPONENTS).all(|slot| left.word(slot) >= right.word(slot))
    };
    let (current_group, incoming_group) = (groups.view(current), groups.view(incoming));
    let mut merged = [NodeId::NONE.0; NODE_WORDS];
    let mut current_changed = false;
    let mut incoming_changed = false;
    for (slot, word) in merged.iter_mut().enumerate() {
        *word = match (
            group_slot(current_group, slot),
            group_slot(incoming_group, slot),
        ) {
            (None, None) => NodeId::NONE.0,
            (Some(own), None) => {
                incoming_changed = true;
                own.0
            }
            (None, Some(theirs)) => {
                current_changed = true;
                theirs.0
            }
            (Some(own), Some(theirs)) => {
                if own == theirs {
                    own.0
                } else {
                    let (own_node, their_node) =
                        (components.view(own), components.view(theirs));
                    if dominates(own_node, their_node) {
                        incoming_changed = true;
                        own.0
                    } else if dominates(their_node, own_node) {
                        current_changed = true;
                        theirs.0
                    } else {
                        let mut block = [0_u64; NODE_WORDS];
                        for (lane, epoch) in block.iter_mut().enumerate() {
                            *epoch = own_node.word(lane).max(their_node.word(lane));
                        }
                        current_changed = true;
                        incoming_changed = true;
                        components.alloc(&block).0
                    }
                }
            }
        };
    }
    if !current_changed {
        MemoizedNode::Current
    } else if !incoming_changed {
        MemoizedNode::Incoming
    } else {
        MemoizedNode::New(groups.alloc(&merged))
    }
}

/// Join two groups block by block.
///
/// Joins are memoized per ordered pair of group identities: the same two
/// groups are joined again and again (every lane of a warp, every warp of a
/// CTA, every acquire of the same publication), and the join of two
/// immutable groups never changes. A hit returns the very same shared group,
/// so later joins keep skipping it by identity.
fn merge_groups(current: NodeId, incoming: NodeId, arena: &BlockArena) -> MemoizedNode {
    if current == incoming {
        return MemoizedNode::Current;
    }
    let key = ((current.0 as u128) << 64) | incoming.0 as u128;
    if let Some(join) = arena.group_join(key) {
        return join;
    }
    let join =
        merge_node_groups::<{ crate::WARP_SIZE }>(current, incoming, &arena.groups, &arena.blocks);
    arena.remember_group_join(key, join);
    join
}

#[derive(Clone, Debug, Default)]
enum SparseComponentBase {
    #[default]
    Empty,
    Many(Arc<WarpGroupList>),
}

/// The group list of a clock base, tagged with a value identity so
/// whole-list joins can be memoized; a fresh or freshly mutated list gets a
/// new identity, so equal identities always mean equal contents.
#[derive(Clone, Debug, Default)]
struct WarpGroupList {
    id: u64,
    entries: Vec<GroupEntry>,
}

/// Result of joining two clock-base group lists.
#[derive(Clone)]
enum BaseListJoin {
    Current,
    Incoming,
    New(Arc<WarpGroupList>),
}

impl SparseComponentBase {
    /// Address of the shared group list, `0` when empty.
    fn storage_identity(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Many(groups) => Arc::as_ptr(groups) as usize,
        }
    }

    fn groups(&self) -> &[GroupEntry] {
        match self {
            Self::Empty => &[],
            Self::Many(groups) => groups.entries.as_slice(),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn shares_storage_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, Self::Empty) => true,
            (Self::Many(left), Self::Many(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn group(&self, group: usize) -> Option<NodeId> {
        let groups = self.groups();
        groups
            .binary_search_by_key(&group, |(group_index, _)| *group_index)
            .ok()
            .map(|position| groups[position].1)
    }

    fn block(&self, warp: usize, arena: &BlockArena) -> Option<NodeId> {
        self.group(warp / WARPS_PER_GROUP)
            .and_then(|group| group_slot(arena.groups.view(group), warp % WARPS_PER_GROUP))
    }

    fn get(&self, index: usize, arena: &BlockArena) -> Option<u64> {
        self.block(index / crate::WARP_SIZE, arena)
            .map(|block| arena.blocks.view(block).word(index % crate::WARP_SIZE))
    }

    /// Every stored `(warp, block)` pair in warp order.
    fn blocks<'a>(&'a self, arena: &'a BlockArena) -> impl Iterator<Item = (usize, NodeId)> + 'a {
        self.groups().iter().flat_map(move |(group, warps)| {
            let warps = arena.groups.view(*warps);
            (0..WARPS_PER_GROUP).filter_map(move |slot| {
                group_slot(warps, slot).map(|block| (group * WARPS_PER_GROUP + slot, block))
            })
        })
    }

    /// Largest component index + 1 covered by the stored blocks.
    fn component_len(&self, arena: &BlockArena) -> usize {
        self.blocks(arena)
            .last()
            .map_or(0, |(warp, _)| (warp + 1) * crate::WARP_SIZE)
    }
}

#[derive(Clone, Debug, Default)]
enum SparseComponentUpdates {
    #[default]
    Empty,
    One((usize, u64)),
    Many(Arc<Vec<(usize, u64)>>),
}

impl SparseComponentUpdates {
    fn as_slice(&self) -> &[(usize, u64)] {
        match self {
            Self::Empty => &[],
            Self::One(update) => std::slice::from_ref(update),
            Self::Many(updates) => updates,
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn shares_storage_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, Self::Empty) => true,
            (Self::One(left), Self::One(right)) => left == right,
            (Self::Many(left), Self::Many(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        self.as_slice()
            .binary_search_by_key(&index, |(component, _)| *component)
            .ok()
            .map(|position| self.as_slice()[position].1)
    }

    fn set(&mut self, index: usize, epoch: u64) {
        match self {
            Self::Empty => *self = Self::One((index, epoch)),
            Self::One((component, current)) if *component == index => *current = epoch,
            Self::One((component, current)) => {
                let mut updates = vec![(*component, *current), (index, epoch)];
                updates.sort_unstable_by_key(|(component, _)| *component);
                *self = Self::Many(Arc::new(updates));
            }
            Self::Many(updates) => {
                let updates = Arc::make_mut(updates);
                match updates.binary_search_by_key(&index, |(component, _)| *component) {
                    Ok(position) => updates[position].1 = epoch,
                    Err(position) => updates.insert(position, (index, epoch)),
                }
            }
        }
    }

    fn from_sorted(updates: Vec<(usize, u64)>) -> Self {
        match updates.len() {
            0 => Self::Empty,
            1 => Self::One(updates[0]),
            _ => Self::Many(Arc::new(updates)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedRepresentationKey {
    base: usize,
    asyncs: usize,
    bridges: usize,
}

impl PartialEq for SparseLaneClock {
    fn eq(&self, other: &Self) -> bool {
        if !Arc::ptr_eq(&self.async_registry, &other.async_registry)
            || !async_clock_equal(
                &self.async_components,
                &other.async_components,
                &self.async_registry.blocks,
            )
            || self.proxy_bridges != other.proxy_bridges
        {
            return false;
        }
        self.lane_components_dominate(other) && other.lane_components_dominate(self)
    }
}

impl Eq for SparseLaneClock {}

impl SparseLaneClock {
    #[inline]
    fn arena(&self) -> &BlockArena {
        &self.async_registry.blocks
    }

    fn shares_causal_representation_with(&self, other: &Self) -> bool {
        self.component_base
            .shares_storage_with(&other.component_base)
            && self
                .component_updates
                .shares_storage_with(&other.component_updates)
            && Arc::ptr_eq(&self.async_registry, &other.async_registry)
            && self
                .async_components
                .shares_storage_with(&other.async_components)
    }

    /// Identity of the storage this clock shares with its siblings, ignoring
    /// the lane-local update layer: equal keys mean equal bases, async
    /// components and proxy bridges. Valid while the clocks behind the key
    /// stay alive.
    fn shared_representation_key(&self) -> SharedRepresentationKey {
        let bridges = match &self.proxy_bridges {
            None => 0,
            Some(bridges) if bridges.is_empty() => 0,
            Some(bridges) => Arc::as_ptr(bridges) as usize,
        };
        SharedRepresentationKey {
            base: self.component_base.storage_identity(),
            asyncs: self.async_components.storage_identity(),
            bridges,
        }
    }

    fn shares_representation_with(&self, other: &Self) -> bool {
        self.shares_causal_representation_with(other)
            && match (&self.proxy_bridges, &other.proxy_bridges) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    Arc::ptr_eq(left, right) || (left.is_empty() && right.is_empty())
                }
                _ => false,
            }
    }

    fn tick(&mut self, actor: GlobalActor) -> Result<(), String> {
        let index = actor.dense_index();
        let epoch = self.component_epoch(index).checked_add(1).ok_or_else(|| {
            format!(
                "global lane clock overflowed for warp {} lane {}",
                actor.global_warp_id, actor.lane
            )
        })?;
        self.set_component_epoch(index, epoch);
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalClockMerge);
        self.merge_unfrozen(other);
        self.freeze_component_updates();
    }

    /// Join every clock in `others`, folding the update layer into the base
    /// once at the end instead of after each join. Equivalent to sequential
    /// `merge` calls: freezing only changes the representation, never the
    /// epochs a lookup observes.
    fn merge_all<'a>(&mut self, others: impl IntoIterator<Item = &'a Self>) {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalClockMerge);
        for other in others {
            self.merge_unfrozen(other);
        }
        self.freeze_component_updates();
    }

    fn merge_unfrozen(&mut self, other: &Self) {
        if self.shares_representation_with(other) {
            return;
        }
        if !self.shares_causal_representation_with(other) {
            self.merge_causal_frontier_unfrozen(other);
        }
        if let Some(other_bridges) = other.proxy_bridges.as_ref() {
            if let Some(current_bridges) = self.proxy_bridges.as_mut() {
                if !Arc::ptr_eq(current_bridges, other_bridges) {
                    Arc::make_mut(current_bridges).merge(other_bridges);
                }
            } else {
                self.proxy_bridges = Some(Arc::clone(other_bridges));
            }
        }
    }

    fn merge_causal_frontier(&mut self, other: &Self) {
        self.merge_causal_frontier_unfrozen(other);
        self.freeze_component_updates();
    }

    fn merge_causal_frontier_unfrozen(&mut self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.async_registry, &other.async_registry),
            "global lane clocks must share one async-clock registry"
        );
        if self.component_base.is_empty() && self.component_updates.is_empty() {
            self.component_base = other.component_base.clone();
            self.component_updates = other.component_updates.clone();
        } else {
            if !self
                .component_base
                .shares_storage_with(&other.component_base)
            {
                self.merge_block_base(&other.component_base);
            }
            self.merge_sparse_improvements(other.component_updates.as_slice());
        }
        self.merge_async_token_epochs(&other.async_components);
    }

    /// Join `incoming`'s warp blocks into this clock's base.
    ///
    /// Walks both sorted group lists once. Groups and blocks that share
    /// storage are skipped, a block dominated by its counterpart adopts the
    /// counterpart's storage, and only genuinely interleaved blocks allocate a
    /// new one. The base keeps sharing `incoming`'s list outright when every
    /// block of this clock was dominated, and keeps its own list when nothing
    /// improved. The update layer is re-validated against the joined base so
    /// it only ever shadows a base slot with a larger epoch.
    fn merge_block_base(&mut self, incoming: &SparseComponentBase) {
        let incoming_groups = incoming.groups();
        if incoming_groups.is_empty() {
            return;
        }
        let current_groups = self.component_base.groups();
        if current_groups.is_empty() {
            self.component_base = incoming.clone();
            self.revalidate_updates();
            return;
        }
        let arena = self.arena();
        // Whole-list joins repeat massively (many lanes and warps join the
        // same release frontier), so the outcome is memoized per ordered
        // pair of list identities.
        let (SparseComponentBase::Many(current_list), SparseComponentBase::Many(incoming_list)) =
            (&self.component_base, incoming)
        else {
            unreachable!("non-empty clock bases hold groups");
        };
        let key = ((current_list.id as u128) << 64) | incoming_list.id as u128;
        if let Some(join) = arena.base_list_join(key) {
            match join {
                BaseListJoin::Current => return,
                BaseListJoin::Incoming => self.component_base = incoming.clone(),
                BaseListJoin::New(list) => self.component_base = SparseComponentBase::Many(list),
            }
            self.revalidate_updates();
            return;
        }
        // First pass: decide per group without touching any reference count.
        // `replacements` holds `(position in current, group)` for groups that
        // change, `insertions` the incoming groups current lacks.
        let mut replacements: Vec<(usize, NodeId)> = Vec::new();
        let mut insertions: Vec<GroupEntry> = Vec::new();
        let mut incoming_changed = false;
        let mut current_index = 0;
        let mut incoming_index = 0;
        while current_index < current_groups.len() || incoming_index < incoming_groups.len() {
            match (
                current_groups.get(current_index),
                incoming_groups.get(incoming_index),
            ) {
                (Some((current_group, current_warps)), Some((incoming_group, incoming_warps)))
                    if current_group == incoming_group =>
                {
                    match merge_groups(*current_warps, *incoming_warps, arena) {
                        MemoizedNode::Current => {
                            if current_warps != incoming_warps {
                                incoming_changed = true;
                            }
                        }
                        MemoizedNode::Incoming => {
                            replacements.push((current_index, *incoming_warps));
                        }
                        MemoizedNode::New(group) => {
                            replacements.push((current_index, group));
                            incoming_changed = true;
                        }
                    }
                    current_index += 1;
                    incoming_index += 1;
                }
                (Some((current_group, _)), Some((incoming_group, _)))
                    if current_group < incoming_group =>
                {
                    incoming_changed = true;
                    current_index += 1;
                }
                (Some(_), Some((incoming_group, incoming_warps))) => {
                    insertions.push((*incoming_group, *incoming_warps));
                    incoming_index += 1;
                }
                (Some(_), None) => {
                    incoming_changed = true;
                    current_index += 1;
                }
                (None, Some((incoming_group, incoming_warps))) => {
                    insertions.push((*incoming_group, *incoming_warps));
                    incoming_index += 1;
                }
                (None, None) => break,
            }
        }
        if replacements.is_empty() && insertions.is_empty() {
            arena.remember_base_list_join(key, &BaseListJoin::Current);
            return;
        }
        if !incoming_changed {
            arena.remember_base_list_join(key, &BaseListJoin::Incoming);
            self.component_base = incoming.clone();
            self.revalidate_updates();
            return;
        }
        // Second pass: build the joined list as a fresh shared value so the
        // memo can hand it to every later identical join.
        let mut entries = current_groups.to_vec();
        for (position, group) in replacements {
            entries[position].1 = group;
        }
        for (group_index, group) in insertions {
            let position = entries
                .binary_search_by_key(&group_index, |(index, _)| *index)
                .unwrap_or_else(|position| position);
            entries.insert(position, (group_index, group));
        }
        entries.shrink_to_fit();
        let list = Arc::new(WarpGroupList {
            id: arena.next_base_list_id(),
            entries,
        });
        arena.base_lists.register(&list);
        arena.remember_base_list_join(key, &BaseListJoin::New(Arc::clone(&list)));
        self.component_base = SparseComponentBase::Many(list);
        self.revalidate_updates();
    }

    /// Drop update-layer entries the base has caught up with, preserving the
    /// invariant that an update always exceeds the base slot it shadows.
    fn revalidate_updates(&mut self) {
        if self.component_updates.is_empty() {
            return;
        }
        let kept = self
            .component_updates
            .as_slice()
            .iter()
            .copied()
            .filter(|&(index, epoch)| {
                epoch > self.component_base.get(index, self.arena()).unwrap_or(0)
            })
            .collect::<Vec<_>>();
        if kept.len() != self.component_updates.as_slice().len() {
            self.component_updates = SparseComponentUpdates::from_sorted(kept);
        }
    }

    /// Raise lane components to `incoming` wherever it is ahead.
    ///
    /// `incoming` must be sorted by component index. Improvements are applied
    /// as one sorted merge into the update layer rather than as repeated
    /// sorted-vector insertions.
    fn merge_sparse_improvements(&mut self, incoming: &[(usize, u64)]) {
        let mut improvements = Vec::new();
        for &(index, epoch) in incoming {
            if epoch > self.component_epoch(index) {
                improvements.push((index, epoch));
            }
        }
        self.apply_sorted_improvements(improvements);
    }

    /// Apply strictly improving, index-sorted updates in one pass.
    fn apply_sorted_improvements(&mut self, improvements: Vec<(usize, u64)>) {
        debug_assert!(improvements.windows(2).all(|pair| pair[0].0 < pair[1].0));
        match improvements.len() {
            0 => {}
            1 => self.set_component_epoch(improvements[0].0, improvements[0].1),
            _ => {
                let current = self.component_updates.as_slice();
                let mut merged = Vec::with_capacity(current.len() + improvements.len());
                let mut current_index = 0;
                let mut improvement_index = 0;
                while current_index < current.len() || improvement_index < improvements.len() {
                    match (
                        current.get(current_index),
                        improvements.get(improvement_index),
                    ) {
                        (Some(&(current_component, _)), Some(&(component, epoch)))
                            if current_component == component =>
                        {
                            merged.push((component, epoch));
                            current_index += 1;
                            improvement_index += 1;
                        }
                        (Some(&(current_component, current_epoch)), Some(&(component, _)))
                            if current_component < component =>
                        {
                            merged.push((current_component, current_epoch));
                            current_index += 1;
                        }
                        (Some(_), Some(&(component, epoch))) => {
                            merged.push((component, epoch));
                            improvement_index += 1;
                        }
                        (Some(&(current_component, current_epoch)), None) => {
                            merged.push((current_component, current_epoch));
                            current_index += 1;
                        }
                        (None, Some(&(component, epoch))) => {
                            merged.push((component, epoch));
                            improvement_index += 1;
                        }
                        (None, None) => break,
                    }
                }
                self.component_updates = SparseComponentUpdates::Many(Arc::new(merged));
            }
        }
    }

    /// Clone this clock without its lane-local update layer.
    fn without_component_updates(&self) -> Self {
        Self {
            component_base: self.component_base.clone(),
            component_updates: SparseComponentUpdates::Empty,
            async_components: self.async_components.clone(),
            async_registry: Arc::clone(&self.async_registry),
            proxy_bridges: self.proxy_bridges.clone(),
        }
    }

    // A clock's proxy-bridge frontiers are always dominated by its own causal
    // frontier: they are only ever recorded from the owning clock itself or
    // merged in together with the causal frontier of the clock they came from,
    // and causal frontiers never move backwards. Recording the current frontier
    // therefore replaces the stored one outright instead of joining it.
    fn apply_proxy_async_fence(&mut self) {
        let frontier = SparseClockFrontier::from_clock(self);
        let bridges = Arc::make_mut(
            self.proxy_bridges
                .get_or_insert_with(|| Arc::new(GlobalProxyBridgeFrontiers::default())),
        );
        bridges.replace_direction(GlobalProxyBridgeDirection::GenericToAsync, frontier.clone());
        bridges.replace_direction(GlobalProxyBridgeDirection::AsyncToGeneric, frontier);
    }

    fn apply_implicit_async_completion(&mut self) {
        let frontier = SparseClockFrontier::from_clock(self);
        Arc::make_mut(
            self.proxy_bridges
                .get_or_insert_with(|| Arc::new(GlobalProxyBridgeFrontiers::default())),
        )
        .replace_direction(GlobalProxyBridgeDirection::AsyncToGeneric, frontier);
    }

    fn proxy_bridge_observes(
        &self,
        prior_proxy: MemoryProxy,
        current_proxy: MemoryProxy,
        frontier_actor: &GlobalFrontierActor,
        frontier_epoch: u64,
    ) -> bool {
        let Some(direction) = GlobalProxyBridgeDirection::between(prior_proxy, current_proxy)
        else {
            return false;
        };
        self.proxy_bridges
            .as_deref()
            .and_then(|bridges| bridges.frontier(direction))
            .is_some_and(|frontier| frontier.frontier_epoch(frontier_actor) >= frontier_epoch)
    }

    fn tick_async(&mut self, lease: &AsyncClockLease) -> Result<(), String> {
        let handle = lease.handle;
        let arena = &self.async_registry.blocks;
        let mut component = self.async_components.get(handle.index, arena);
        if component.generation != handle.generation {
            component = AsyncClockComponent {
                generation: handle.generation,
                epoch: 0,
            };
        }
        component.epoch = component
            .epoch
            .checked_add(1)
            .ok_or_else(|| format!("global async clock overflowed for token {:?}", lease.token))?;
        self.async_components.set(handle.index, component, arena);
        Ok(())
    }

    fn actor_epoch(&self, actor: GlobalActor) -> u64 {
        self.component_epoch(actor.dense_index())
    }

    fn component_epoch(&self, index: usize) -> u64 {
        self.component_updates
            .get(index)
            .or_else(|| self.component_base.get(index, self.arena()))
            .unwrap_or_default()
    }

    fn component_len(&self) -> usize {
        self.component_base.component_len(self.arena()).max(
            self.component_updates
                .as_slice()
                .last()
                .map_or(0, |(index, _)| index + 1),
        )
    }

    /// Whether every lane component of `other` is covered by this clock.
    fn lane_components_dominate(&self, other: &Self) -> bool {
        let arena = self.arena();
        let covers_block = |warp: usize, block: NodeId| {
            let own = self
                .component_base
                .block(warp, arena)
                .map(|own| arena.blocks.view(own));
            let block = arena.blocks.view(block);
            (0..crate::WARP_SIZE).all(|lane| {
                let epoch = block.word(lane);
                epoch == 0 || {
                    let index = warp * crate::WARP_SIZE + lane;
                    self.component_updates
                        .get(index)
                        .or_else(|| own.map(|own| own.word(lane)))
                        .unwrap_or(0)
                        >= epoch
                }
            })
        };
        other.component_base.groups().iter().all(|(group, warps)| {
            self.component_base.group(*group) == Some(*warps)
                || {
                    let warps = arena.groups.view(*warps);
                    (0..WARPS_PER_GROUP).all(|slot| {
                        group_slot(warps, slot).is_none_or(|block| {
                            let warp = group * WARPS_PER_GROUP + slot;
                            self.component_base.block(warp, arena) == Some(block)
                                || covers_block(warp, block)
                        })
                    })
                }
        }) && other
            .component_updates
            .as_slice()
            .iter()
            .all(|&(index, epoch)| self.component_epoch(index) >= epoch)
    }

    fn dominates(&self, other: &Self) -> bool {
        if !Arc::ptr_eq(&self.async_registry, &other.async_registry) {
            return false;
        }
        self.lane_components_dominate(other)
            && async_clock_happens_before(
                &other.async_components,
                &self.async_components,
                &self.async_registry.blocks,
            )
    }

    fn is_bottom(&self) -> bool {
        // Blocks only come from ticks (epoch >= 1) or joins of such blocks, so
        // a non-empty base is never all-zero; checking emptiness keeps this
        // O(1) instead of scanning every lane the clock has heard of.
        self.component_base.is_empty()
            && self
                .component_updates
                .as_slice()
                .iter()
                .all(|(_, epoch)| *epoch == 0)
            && self
                .async_components
                .iter(&self.async_registry.blocks)
                .all(|component| component.epoch == 0)
    }

    fn set_component_epoch(&mut self, index: usize, epoch: u64) {
        // The update layer must stay monotone over the base slots it shadows;
        // `merge_block_base` and `freeze_component_updates` rely on it.
        debug_assert!(epoch >= self.component_epoch(index));
        self.component_updates.set(index, epoch);
    }

    /// Fold a multi-entry update layer into the block base.
    ///
    /// Only the groups and blocks touched by the updates get fresh storage;
    /// everything else keeps sharing with the clocks this one derives from,
    /// and the group list is patched in place when this clock owns it alone.
    fn freeze_component_updates(&mut self) {
        let updates = std::mem::take(&mut self.component_updates);
        let SparseComponentUpdates::Many(updates) = updates else {
            self.component_updates = updates;
            return;
        };
        let arena = &self.async_registry.blocks;
        // A fresh list (see `AsyncClockVector::set` for why lists are never
        // patched in place).
        let mut entries = self.component_base.groups().to_vec();
        let list = &mut entries;
        let mut update_index = 0;
        while update_index < updates.len() {
            let group_index = updates[update_index].0 / (crate::WARP_SIZE * WARPS_PER_GROUP);
            let position = match list.binary_search_by_key(&group_index, |(index, _)| *index) {
                Ok(position) => position,
                Err(position) => {
                    list.insert(position, (group_index, arena.empty_group()));
                    position
                }
            };
            let mut blocks = arena.groups.view(list[position].1).words();
            while update_index < updates.len()
                && updates[update_index].0 / (crate::WARP_SIZE * WARPS_PER_GROUP) == group_index
            {
                let warp = updates[update_index].0 / crate::WARP_SIZE;
                let slot = warp % WARPS_PER_GROUP;
                let mut block = if blocks[slot] == NodeId::NONE.0 {
                    [0; crate::WARP_SIZE]
                } else {
                    arena.blocks.view(NodeId(blocks[slot])).words()
                };
                while update_index < updates.len()
                    && updates[update_index].0 / crate::WARP_SIZE == warp
                {
                    let (index, epoch) = updates[update_index];
                    let lane = &mut block[index % crate::WARP_SIZE];
                    *lane = (*lane).max(epoch);
                    update_index += 1;
                }
                blocks[slot] = arena.blocks.alloc(&block).0;
            }
            list[position].1 = arena.groups.alloc(&blocks);
        }
        entries.shrink_to_fit();
        let groups = Arc::new(WarpGroupList {
            id: arena.next_base_list_id(),
            entries,
        });
        arena.base_lists.register(&groups);
        self.component_base = SparseComponentBase::Many(groups);
    }

    fn frontier_epoch(&self, actor: &GlobalFrontierActor) -> u64 {
        match actor {
            GlobalFrontierActor::Lane(actor) => self.actor_epoch(*actor),
            GlobalFrontierActor::Async(lease) => self.async_component(lease.handle),
        }
    }

    fn async_component(&self, handle: AsyncClockHandle) -> u64 {
        let component = self
            .async_components
            .get(handle.index, &self.async_registry.blocks);
        if component.generation == handle.generation {
            u64::from(component.epoch)
        } else {
            0
        }
    }

    fn merge_async_token_epochs(&mut self, incoming: &AsyncClockVector) {
        self.async_components
            .merge(incoming, &self.async_registry.blocks);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlobalProxyBridgeDirection {
    GenericToAsync,
    AsyncToGeneric,
}

impl GlobalProxyBridgeDirection {
    const fn between(prior: MemoryProxy, current: MemoryProxy) -> Option<Self> {
        match (prior, current) {
            (MemoryProxy::Generic, MemoryProxy::Async) => Some(Self::GenericToAsync),
            (MemoryProxy::Async, MemoryProxy::Generic) => Some(Self::AsyncToGeneric),
            _ => None,
        }
    }
}

/// One immutable causal frontier captured by a proxy fence.
///
/// This deliberately excludes proxy-bridge state so clocks do not become
/// recursively nested. The block storage remains shared until two
/// independently captured frontiers must be joined at a barrier.
#[derive(Clone, Debug)]
struct SparseClockFrontier {
    clock: SparseLaneClock,
}

impl SparseClockFrontier {
    fn from_clock(clock: &SparseLaneClock) -> Self {
        Self {
            clock: SparseLaneClock {
                component_base: clock.component_base.clone(),
                component_updates: clock.component_updates.clone(),
                async_components: clock.async_components.clone(),
                async_registry: Arc::clone(&clock.async_registry),
                proxy_bridges: None,
            },
        }
    }

    fn as_clock(&self) -> SparseLaneClock {
        self.clock.clone()
    }

    fn merge(&mut self, other: &Self) {
        self.clock.merge_causal_frontier(&other.clock);
    }

    fn frontier_epoch(&self, actor: &GlobalFrontierActor) -> u64 {
        self.clock.frontier_epoch(actor)
    }
}

impl PartialEq for SparseClockFrontier {
    fn eq(&self, other: &Self) -> bool {
        self.clock == other.clock
    }
}

impl Eq for SparseClockFrontier {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GlobalProxyBridgeFrontiers {
    generic_to_async: Option<SparseClockFrontier>,
    async_to_generic: Option<SparseClockFrontier>,
    // Descriptor generations are assigned by the runtime registry. An acquire
    // is propagated by the existing causal clock, not by warp membership. PTX
    // proxy-preserved causality for a non-generic proxy is CTA-local.
    tensor_map_acquired: BTreeMap<(PhysicalByteSpan, usize), TensorMapAcquired>,
    // Generic releases cover earlier causally visible writes. Copy-release
    // covers only its destination. These immutable frontiers travel through
    // the same synchronization snapshots as every other proxy bridge.
    tensor_map_released:
        BTreeMap<(Option<PhysicalByteSpan>, GlobalActor, MemoryScope), SparseClockFrontier>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorMapAcquired {
    generation: u64,
    writes: SparseClockFrontier,
}

impl TensorMapAcquired {
    fn merge(&mut self, other: &Self) {
        self.generation = self.generation.max(other.generation);
        self.writes.merge(&other.writes);
    }
}

impl GlobalProxyBridgeFrontiers {
    const fn frontier(
        &self,
        direction: GlobalProxyBridgeDirection,
    ) -> Option<&SparseClockFrontier> {
        match direction {
            GlobalProxyBridgeDirection::GenericToAsync => self.generic_to_async.as_ref(),
            GlobalProxyBridgeDirection::AsyncToGeneric => self.async_to_generic.as_ref(),
        }
    }

    fn merge_direction(
        &mut self,
        direction: GlobalProxyBridgeDirection,
        frontier: &SparseClockFrontier,
    ) {
        let slot = match direction {
            GlobalProxyBridgeDirection::GenericToAsync => &mut self.generic_to_async,
            GlobalProxyBridgeDirection::AsyncToGeneric => &mut self.async_to_generic,
        };
        if let Some(current) = slot {
            current.merge(frontier);
        } else {
            *slot = Some(frontier.clone());
        }
    }

    /// Record `frontier` as the new frontier for `direction`.
    ///
    /// Callers guarantee that `frontier` dominates whatever is stored, so no
    /// join is needed; the check is kept as a debug assertion.
    fn replace_direction(
        &mut self,
        direction: GlobalProxyBridgeDirection,
        frontier: SparseClockFrontier,
    ) {
        let slot = match direction {
            GlobalProxyBridgeDirection::GenericToAsync => &mut self.generic_to_async,
            GlobalProxyBridgeDirection::AsyncToGeneric => &mut self.async_to_generic,
        };
        debug_assert!(
            slot.as_ref()
                .is_none_or(|current| frontier.as_clock().dominates(&current.as_clock())),
            "a recorded proxy-bridge frontier must dominate the stored frontier"
        );
        *slot = Some(frontier);
    }

    fn merge(&mut self, other: &Self) {
        for direction in [
            GlobalProxyBridgeDirection::GenericToAsync,
            GlobalProxyBridgeDirection::AsyncToGeneric,
        ] {
            if let Some(frontier) = other.frontier(direction) {
                self.merge_direction(direction, frontier);
            }
        }
        for (&key, acquired) in &other.tensor_map_acquired {
            self.tensor_map_acquired
                .entry(key)
                .and_modify(|current| current.merge(acquired))
                .or_insert_with(|| acquired.clone());
        }
        for (&key, released) in &other.tensor_map_released {
            self.tensor_map_released
                .entry(key)
                .and_modify(|current| current.merge(released))
                .or_insert_with(|| released.clone());
        }
    }

    fn is_empty(&self) -> bool {
        self.generic_to_async.is_none()
            && self.async_to_generic.is_none()
            && self.tensor_map_acquired.is_empty()
            && self.tensor_map_released.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReleaseHeadKey {
    actor: GlobalActor,
    scope: MemoryScope,
    proxy: MemoryProxy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReleaseHead {
    key: ReleaseHeadKey,
    operation: DynamicOpId,
    clock: SparseLaneClock,
    tcgen: TcgenFenceFrontier,
    shared_frontier: SharedClockFrontier,
}

#[derive(Clone, Debug, Default)]
struct ReleasePayload {
    // Heads are shared between a version and the versions that inherit its
    // release history (RMW chains), so adopting a predecessor's heads is a
    // map of reference counts rather than a deep copy of every head clock.
    heads: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    /// Join of the shared/TMEM clock frontiers of every release the payload
    /// carries. Acquirers merge a frontier component-wise, so the join of
    /// the frontiers orders an acquirer exactly as merging each one would,
    /// and an RMW chain inherits one merged frontier instead of the growing
    /// list of every predecessor's frontier.
    shared_frontier: SharedClockFrontier,
    tcgen: TcgenFenceFrontier,
    /// Join of every head clock in `heads`, maintained incrementally.
    ///
    /// RMW ancestry chains make a version carry one head per earlier releasing
    /// actor, so an acquire that consumes every head would otherwise join each
    /// head clock separately per acquiring lane. The join is kept exact: it is
    /// dropped (`None` while `heads` is non-empty) whenever a head is replaced
    /// by one that does not dominate it, and acquirers then fall back to the
    /// per-head joins.
    joined: Option<SparseLaneClock>,
    /// Join of every head's TCGEN frontier, kept exact the same way: dropped
    /// when a head is replaced by one whose frontier does not cover it.
    joined_tcgen: Option<TcgenFenceFrontier>,
    /// The narrowest release scope among the heads. Together with a uniform
    /// proxy it lets an acquire settle every head's scope and proxy checks
    /// at once instead of visiting each of the (often hundreds of) heads.
    scope_floor: Option<MemoryScope>,
    /// The proxy shared by every head, `None` once the heads mix proxies.
    uniform_proxy: Option<MemoryProxy>,
    mixed_proxy: bool,
}

impl PartialEq for ReleasePayload {
    fn eq(&self, other: &Self) -> bool {
        self.heads == other.heads
            && self.shared_frontier == other.shared_frontier
            && self.tcgen == other.tcgen
    }
}

impl Eq for ReleasePayload {}

impl ReleasePayload {
    /// The exact join of all head clocks, when it is still tracked.
    fn joined_heads(&self) -> Option<&SparseLaneClock> {
        self.joined.as_ref()
    }

    /// The exact join of all head TCGEN frontiers, when it is still tracked.
    fn joined_tcgen(&self) -> Option<&TcgenFenceFrontier> {
        self.joined_tcgen.as_ref()
    }

    /// Whether an acquire with `scope` and `proxy` is known to consume every
    /// head without any per-head scope or proxy check: all heads publish
    /// through `proxy`, and both the releases and the acquire are at least
    /// GPU-scoped, which [`scope_covers`] accepts for any actor pair.
    fn every_head_acquirable_by(&self, scope: MemoryScope, proxy: MemoryProxy) -> bool {
        !self.heads.is_empty()
            && self.uniform_proxy == Some(proxy)
            && self.scope_floor.is_some_and(|floor| floor >= MemoryScope::Gpu)
            && scope >= MemoryScope::Gpu
    }

    fn join_head_clock(&mut self, clock: &SparseLaneClock) {
        match &mut self.joined {
            Some(joined) => joined.merge(clock),
            None if self.heads.is_empty() => self.joined = Some(clock.clone()),
            // The join was invalidated earlier; it stays untracked.
            None => {}
        }
    }

    fn join_head_tcgen(&mut self, tcgen: &TcgenFenceFrontier) {
        match &mut self.joined_tcgen {
            Some(joined) => {
                joined.merge(tcgen);
            }
            None if self.heads.is_empty() => self.joined_tcgen = Some(tcgen.clone()),
            None => {}
        }
    }

    fn insert(&mut self, head: Arc<ReleaseHead>) {
        let (replaced_dominated, tcgen_covered, new_key) = match self.heads.get(&head.key) {
            None => (Some(true), true, true),
            Some(current) => {
                let current_epoch = current.clock.actor_epoch(head.key.actor);
                let new_epoch = head.clock.actor_epoch(head.key.actor);
                (
                    (current_epoch <= new_epoch).then(|| head.clock.dominates(&current.clock)),
                    head.tcgen.covers(&current.tcgen),
                    false,
                )
            }
        };
        let Some(replaced_dominated) = replaced_dominated else {
            return;
        };
        if replaced_dominated {
            self.join_head_clock(&head.clock);
        } else {
            self.joined = None;
        }
        if tcgen_covered {
            self.join_head_tcgen(&head.tcgen);
        } else {
            self.joined_tcgen = None;
        }
        if new_key {
            // A replaced head keeps its key, so scope and proxy summaries only
            // move when a new releasing actor joins the payload.
            self.scope_floor = Some(
                self.scope_floor
                    .map_or(head.key.scope, |floor| floor.min(head.key.scope)),
            );
            if self.heads.is_empty() {
                self.uniform_proxy = Some(head.key.proxy);
                self.mixed_proxy = false;
            } else if !self.mixed_proxy && self.uniform_proxy != Some(head.key.proxy) {
                self.uniform_proxy = None;
                self.mixed_proxy = true;
            }
        }
        self.shared_frontier.merge(&head.shared_frontier);
        self.heads.insert(head.key, head);
    }

    fn extend_release_heads(&mut self, other: &Self) {
        if self.heads.is_empty() {
            // Adopt the other payload's heads and their tracked joins wholesale
            // instead of re-joining every head clock.
            self.heads = other.heads.clone();
            self.joined = other.joined.clone();
            self.joined_tcgen = other.joined_tcgen.clone();
            self.scope_floor = other.scope_floor;
            self.uniform_proxy = other.uniform_proxy;
            self.mixed_proxy = other.mixed_proxy;
            return;
        }
        for head in other.heads.values().cloned() {
            self.insert(head);
        }
    }

    fn extend(&mut self, other: &Self) {
        self.extend_release_heads(other);
        self.shared_frontier.merge(&other.shared_frontier);
        self.tcgen.merge(&other.tcgen);
    }
}

/// Per-batch memo of `base ⊔ head clocks` joins for acquiring lanes.
///
/// Lanes of one warp normally share a synchronized causal base and differ only
/// in their lane-local update layer, while a compact batch acquires the same
/// release heads for every lane. Joining the shared base with the heads once
/// and overlaying each lane's private updates is exact because the join is
/// associative and commutative, and it leaves all lanes sharing one storage.
#[derive(Default)]
struct AcquireJoinCache {
    entries: Vec<AcquireJoinEntry>,
}

struct AcquireJoinEntry {
    base: SparseLaneClock,
    /// Addresses of the joined clocks, kept valid by `version`.
    identities: Vec<usize>,
    joined: SparseLaneClock,
    /// Pins the version so the addresses in `identities` cannot be reused.
    _version: Arc<GlobalVersion>,
}

impl AcquireJoinCache {
    fn join_heads(
        &mut self,
        actor_state: &mut GlobalActorState,
        version: &Arc<GlobalVersion>,
        heads: &[&ReleaseHead],
    ) {
        let identities = heads
            .iter()
            .map(|head| std::ptr::from_ref::<ReleaseHead>(head) as usize)
            .collect::<Vec<_>>();
        self.join_clocks(
            actor_state,
            version,
            identities,
            heads.iter().map(|head| &head.clock),
        );
    }

    /// Join `clocks` into the lane clock; `identities` names them for the memo
    /// and must stay valid for as long as `version` is alive.
    fn join_clocks<'a>(
        &mut self,
        actor_state: &mut GlobalActorState,
        version: &Arc<GlobalVersion>,
        identities: Vec<usize>,
        clocks: impl Iterator<Item = &'a SparseLaneClock>,
    ) {
        let base = actor_state.clock.without_component_updates();
        let mut joined = if let Some(entry) = self.entries.iter().find(|entry| {
            entry.identities == identities && entry.base.shares_representation_with(&base)
        }) {
            entry.joined.clone()
        } else {
            let mut joined = base.clone();
            joined.merge_all(clocks);
            self.entries.push(AcquireJoinEntry {
                base,
                identities,
                joined: joined.clone(),
                _version: Arc::clone(version),
            });
            joined
        };
        joined.merge_sparse_improvements(actor_state.clock.component_updates.as_slice());
        actor_state.clock = joined;
    }
}

#[derive(Clone, Debug, Default)]
struct GlobalActorState {
    clock: SparseLaneClock,
    release_fence: Option<Arc<ReleaseHead>>,
    pending_acquire: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    last_global_acquire: Option<GlobalAcquireCache>,
    /// The lane appeared after the collector retired entries and has not yet
    /// been shown to dominate the retirement watermark: its next global
    /// access is reported as `RetiredRecordsUnobserved` (see
    /// `AsyncClockRegistry::retirement`).
    laggard: bool,
}

/// Clears a lane's laggard mark once its clock dominates the watermark, or
/// reports the analysis gap for its access.
fn note_laggard_access(
    state: &mut GlobalActorState,
    registry: &AsyncClockRegistry,
    operation: &DynamicOpId,
) -> Option<RaceCheckIncompleteReason> {
    if !state.laggard {
        return None;
    }
    state.laggard = false;
    if registry.watermark_dominated_by(&state.clock) {
        return None;
    }
    Some(RaceCheckIncompleteReason::RetiredRecordsUnobserved {
        operation: operation.clone(),
    })
}

#[derive(Clone, Debug)]
struct GlobalAcquireCache {
    version_id: u64,
    scope: MemoryScope,
    proxy: MemoryProxy,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GlobalActorStates {
    slots: Vec<Option<GlobalActorState>>,
    // Dense lane index of `slots[0]`; a shard only holds its own warps.
    base_slot: usize,
    len: usize,
    async_registry: Arc<AsyncClockRegistry>,
}

impl GlobalActorStates {
    fn with_topology(topology: LaunchTopology, async_registry: Arc<AsyncClockRegistry>) -> Self {
        let slot_count = topology
            .warp_count()
            .checked_mul(crate::WARP_SIZE)
            .expect("global actor-state slot count must fit usize");
        Self {
            slots: std::iter::repeat_with(|| None).take(slot_count).collect(),
            base_slot: 0,
            len: 0,
            async_registry,
        }
    }

    fn with_registry(async_registry: Arc<AsyncClockRegistry>) -> Self {
        Self {
            slots: Vec::new(),
            base_slot: 0,
            len: 0,
            async_registry,
        }
    }

    /// A table for the warps `[base_warp, base_warp + warp_count)` only.
    fn with_warp_range(
        base_warp: usize,
        warp_count: usize,
        async_registry: Arc<AsyncClockRegistry>,
    ) -> Self {
        Self {
            slots: std::iter::repeat_with(|| None)
                .take(warp_count * crate::WARP_SIZE)
                .collect(),
            base_slot: base_warp * crate::WARP_SIZE,
            len: 0,
            async_registry,
        }
    }

    fn slot_index(&self, actor: &GlobalActor) -> usize {
        let index = actor.dense_index();
        debug_assert!(
            index >= self.base_slot,
            "global actor {actor:?} is below this shard's warp range"
        );
        index - self.base_slot
    }

    fn empty_clock(&self) -> SparseLaneClock {
        SparseLaneClock::new(Arc::clone(&self.async_registry))
    }

    fn empty_state(&self) -> GlobalActorState {
        GlobalActorState {
            clock: self.empty_clock(),
            laggard: self.async_registry.retired_any(),
            ..GlobalActorState::default()
        }
    }

    fn get(&self, actor: &GlobalActor) -> Option<&GlobalActorState> {
        self.slots
            .get(self.slot_index(actor))
            .and_then(Option::as_ref)
    }

    fn get_or_insert_default(&mut self, actor: GlobalActor) -> &mut GlobalActorState {
        let index = self.slot_index(&actor);
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        let slot = &mut self.slots[index];
        if slot.is_none() {
            *slot = Some(GlobalActorState {
                clock: SparseLaneClock::new(Arc::clone(&self.async_registry)),
                laggard: self.async_registry.retired_any(),
                ..GlobalActorState::default()
            });
            self.len += 1;
        }
        slot.as_mut()
            .expect("global actor state was inserted into its dense slot")
    }

    fn insert(&mut self, actor: GlobalActor, state: GlobalActorState) {
        let slot = self.get_or_insert_default(actor);
        *slot = state;
    }

    fn extend(&mut self, updates: impl IntoIterator<Item = (GlobalActor, GlobalActorState)>) {
        for (actor, state) in updates {
            self.insert(actor, state);
        }
    }

    fn values(&self) -> impl Iterator<Item = &GlobalActorState> {
        self.slots.iter().filter_map(Option::as_ref)
    }

    const fn len(&self) -> usize {
        self.len
    }

    const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// What the byte state keeps of a committed access.
///
/// Racecheck commits accesses in an order consistent with happens-before:
/// synchronization is processed before its causal descendants, a completion
/// is applied before its waiters resume, and an atomic access waits for the
/// in-flight writes it overlaps. A recorded access is therefore always the
/// earlier side of a later conflict check, and only its frontier stamp
/// (`frontier_actor`, `frontier_epoch`) is consulted against the later
/// access's clock; its own clock is never needed again and is not kept.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordedGlobalAccess {
    /// The witness operation, shared with every other record of it.
    operation: Arc<DynamicOpId>,
    byte_offset: usize,
    allocation: PhysicalAllocationId,
    /// The async actor whose clock slot stamps the frontier; `None` for a
    /// lane access, whose frontier actor is the accessing lane itself.
    frontier_lease: Option<Arc<AsyncClockLease>>,
    frontier_epoch: u64,
    /// The accessing lane as a dense clock index (`warp * 32 + lane`).
    actor_index: u32,
    byte_len: u32,
    /// Nonzero when the witness span stands for a run of equal transfer units
    /// of this many bytes. Strong runs may clip their naturally aligned edge
    /// units; the witness always keeps the actual accessed bytes. The byte state keeps one entry for
    /// the whole run; findings report the units that conflict as one range
    /// (see `validate_pair_in`), so a run is indistinguishable from one
    /// access per unit in every report.
    unit_bytes: u32,
    semantics: MemoryAccessSemantics,
    lane: u8,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
}

impl RecordedGlobalAccess {
    fn new(
        actor: GlobalActor,
        frontier_actor: GlobalFrontierActor,
        frontier_epoch: u64,
        witness: PhysicalRaceWitness,
        semantics: MemoryAccessSemantics,
        unit_bytes: u32,
    ) -> Self {
        let span = witness.span();
        let frontier_lease = match frontier_actor {
            GlobalFrontierActor::Lane(lane_actor) => {
                debug_assert_eq!(lane_actor, actor, "a lane access is its own frontier actor");
                None
            }
            GlobalFrontierActor::Async(lease) => Some(lease),
        };
        Self {
            operation: witness.shared_operation(),
            byte_offset: span.byte_offset(),
            allocation: span.allocation(),
            frontier_lease,
            frontier_epoch,
            actor_index: u32::try_from(actor.dense_index())
                .expect("a global lane index fits u32"),
            byte_len: u32::try_from(span.byte_len())
                .expect("a recorded global access span fits u32"),
            unit_bytes,
            semantics,
            lane: witness.lane() as u8,
            kind: witness.kind(),
            space: witness.space(),
        }
    }

    fn actor(&self) -> GlobalActor {
        GlobalActor::new(
            self.actor_index as usize / crate::WARP_SIZE,
            self.actor_index as usize % crate::WARP_SIZE,
        )
    }

    fn actor_index(&self) -> usize {
        self.actor_index as usize
    }

    /// The epoch the issuing lane had at the access: the record's own
    /// frontier epoch for a lane access, the issue epoch of the token for an
    /// async one.
    fn issue_epoch(&self) -> u64 {
        match &self.frontier_lease {
            None => self.frontier_epoch,
            Some(lease) => lease.issue_epoch,
        }
    }

    fn frontier_actor(&self) -> GlobalFrontierActor {
        match &self.frontier_lease {
            Some(lease) => GlobalFrontierActor::Async(Arc::clone(lease)),
            None => GlobalFrontierActor::Lane(self.actor()),
        }
    }

    /// Whether both records are stamped by the same frontier actor.
    fn same_frontier(&self, other: &Self) -> bool {
        match (&self.frontier_lease, &other.frontier_lease) {
            (None, None) => self.actor_index == other.actor_index,
            (Some(left), Some(right)) => left == right,
            _ => false,
        }
    }

    /// `clock`'s epoch for this record's frontier actor.
    /// Which clock component positions this access, without retaining the
    /// lease that names it.
    fn frontier_key(&self) -> WordFrontier {
        match &self.frontier_lease {
            Some(lease) => WordFrontier::Async(lease.handle),
            None => WordFrontier::Lane(self.actor()),
        }
    }

    fn frontier_epoch_in(&self, clock: &SparseLaneClock) -> u64 {
        match &self.frontier_lease {
            Some(lease) => clock.async_component(lease.handle),
            None => clock.actor_epoch(self.actor()),
        }
    }

    fn span(&self) -> PhysicalByteSpan {
        PhysicalByteSpan::new(self.allocation, self.byte_offset, self.byte_len as usize)
            .expect("a recorded global access span is nonempty and representable")
    }

    fn operation(&self) -> &DynamicOpId {
        &self.operation
    }

    fn shared_operation(&self) -> Arc<DynamicOpId> {
        Arc::clone(&self.operation)
    }

    const fn lane(&self) -> usize {
        self.lane as usize
    }

    const fn kind(&self) -> PhysicalAccessKind {
        self.kind
    }

    const fn space(&self) -> PhysicalAccessSpace {
        self.space
    }

    /// The witness a finding reports for this record.
    fn witness(&self) -> PhysicalRaceWitness {
        PhysicalRaceWitness::from_parts_shared(
            self.shared_operation(),
            self.lane(),
            self.kind,
            self.space,
            self.span(),
        )
    }

    /// Strong transfers use naturally aligned elements even at clipped edges.
    fn unit_origin(&self) -> usize {
        if self.unit_bytes != 0 && self.semantics.order().is_strong() {
            self.byte_offset / self.unit_bytes as usize * self.unit_bytes as usize
        } else {
            self.byte_offset
        }
    }

    /// Restrict the covered range without changing its semantic element width.
    fn with_span(&self, span: PhysicalByteSpan) -> Self {
        debug_assert_eq!(span.allocation(), self.allocation);
        Self {
            byte_offset: span.byte_offset(),
            byte_len: u32::try_from(span.byte_len())
                .expect("a recorded global access span fits u32"),
            ..self.clone()
        }
    }

    /// This access restricted to one of its units.
    fn with_unit_span(&self, span: PhysicalByteSpan) -> Self {
        Self {
            unit_bytes: 0,
            ..self.with_span(span)
        }
    }
}

/// A recorded access shared by every frontier entry that stands for it.
///
/// One access commits into every byte-state segment its span covers, and a
/// wide span (a TMA run, a coalesced warp access) is split by other actors'
/// accesses into many segments: on MegaMoE t128_m128 the byte shadow held
/// 48.5 M entries over far fewer distinct accesses. Sharing the record keeps
/// one copy per access and an `Arc` per entry.
type RecordRef = Arc<RecordedGlobalAccess>;

/// An access in flight: the recorded part plus the event clock that the
/// checks and supersession scans against earlier recorded accesses read.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GlobalAccess {
    recorded: RecordRef,
    clock: SparseLaneClock,
}

impl std::ops::Deref for GlobalAccess {
    type Target = RecordedGlobalAccess;

    fn deref(&self) -> &RecordedGlobalAccess {
        &self.recorded
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GlobalFrontierActor {
    Lane(GlobalActor),
    Async(Arc<AsyncClockLease>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GlobalVersion {
    /// Identity of the version; a run carrier (`carrier.unit_bytes != 0`)
    /// owns one consecutive identity per transfer unit starting here, so the
    /// numbering is the one separate per-unit versions would have produced.
    id: u64,
    carrier: RecordRef,
    payload: ReleasePayload,
}

impl GlobalVersion {
    /// The per-unit version identity of the unit holding `byte`.
    fn unit_id_at(&self, byte: usize) -> u64 {
        if self.carrier.unit_bytes == 0 {
            return self.id;
        }
        let run = self.carrier.span();
        debug_assert!(byte >= run.byte_offset() && byte < run.byte_end());
        self.id + ((byte - run.byte_offset()) / self.carrier.unit_bytes as usize) as u64
    }

    /// The version exactly as a separate per-unit version for the unit
    /// holding `byte` would look: own identity, unit-wide carrier witness.
    fn unit_version_at(self: &Arc<Self>, byte: usize) -> Arc<Self> {
        if self.carrier.unit_bytes == 0 {
            return Arc::clone(self);
        }
        let run = self.carrier.span();
        let unit_bytes = self.carrier.unit_bytes as usize;
        let index = (byte - run.byte_offset()) / unit_bytes;
        let unit_span = PhysicalByteSpan::new(
            run.allocation(),
            run.byte_offset() + index * unit_bytes,
            unit_bytes,
        )
        .expect("a transfer unit span is nonempty and representable");
        Arc::new(Self {
            id: self.id + index as u64,
            carrier: Arc::new(self.carrier.with_unit_span(unit_span)),
            payload: self.payload.clone(),
        })
    }
}

/// Lane readers or writers of one segment, grouped by warp.
///
/// Grouping lets a supersession pass probe the superseding clock once per warp
/// and skip every reader of a warp that clock has never observed: an event by
/// an unobserved actor cannot happen-before the superseding access, and the
/// proxy-bridge frontiers a cross-proxy check consults are dominated by that
/// same clock.
/// A supersession scan of one warp's entries is a pure function of the
/// entries, the scanning access's kind and semantics, and the superseding
/// clock's view of that warp's lanes (its base block). A repeat of the same
/// scan over unchanged entries removes nothing again, so its key is kept on
/// the entries and invalidated by every insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SupersedeScanKey {
    block: NodeId,
    kind: PhysicalAccessKind,
    semantics: MemoryAccessSemantics,
}

#[derive(Clone, Debug, Default)]
struct WarpFrontierEntries {
    // Sorted by lane; at most `WARP_SIZE` entries.
    lanes: Vec<(u8, RecordRef)>,
    // The last scan whose outcome still holds for `lanes`.
    scanned: Option<SupersedeScanKey>,
}

impl PartialEq for WarpFrontierEntries {
    fn eq(&self, other: &Self) -> bool {
        self.lanes == other.lanes
    }
}

impl Eq for WarpFrontierEntries {}

/// One async-actor reader or writer of a segment: its clock slot handle, the
/// lease that keeps the slot alive, and the recorded access.
type AsyncFrontierEntry = (Arc<AsyncClockLease>, RecordRef);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ManyGlobalFrontiers {
    lanes: BTreeMap<usize, WarpFrontierEntries>,
    // Sorted by slot handle, so the entries of one `ASYNC_CHUNK`-slot clock
    // chunk are contiguous and a supersession pass can still skip every chunk
    // the superseding clock has never observed. A flat vector grown exactly:
    // most segments hold one entry per chunk, and per-chunk vectors with the
    // allocator's minimum capacity cost 13 GB of slack on MegaMoE t128_m128
    // (36 M single-entry buckets).
    asyncs: Vec<AsyncFrontierEntry>,
    len: usize,
    // Entries that are mutually morally strong with any GPU-scoped atomic
    // unit access (see `strong_gpu_unit`). When every entry is one, such an
    // access can skip the frontier outright: validation returns for each of
    // those pairs before any happens-before or witness work.
    strong_gpu_len: usize,
}

/// Whether an access is a non-run atomic-class access whose scope covers any
/// actor pair, so `mutually_morally_strong` holds against every such access.
fn strong_gpu_unit(access: &RecordedGlobalAccess) -> bool {
    access.unit_bytes == 0
        && access.semantics.class().is_atomic_class()
        && access.semantics.scope().is_some_and(|scope| scope >= MemoryScope::Gpu)
}

impl ManyGlobalFrontiers {
    fn insert(&mut self, actor: GlobalFrontierActor, access: RecordRef) {
        let strong = usize::from(strong_gpu_unit(&access));
        match actor {
            GlobalFrontierActor::Lane(actor) => {
                let entries = self.lanes.entry(actor.global_warp_id).or_default();
                entries.scanned = None;
                match entries
                    .lanes
                    .binary_search_by_key(&actor.lane, |(lane, _)| *lane)
                {
                    Ok(position) => {
                        self.strong_gpu_len -= usize::from(strong_gpu_unit(&entries.lanes[position].1));
                        entries.lanes[position].1 = access;
                    }
                    Err(position) => {
                        entries.lanes.reserve_exact(1);
                        entries.lanes.insert(position, (actor.lane, access));
                        self.len += 1;
                    }
                }
            }
            GlobalFrontierActor::Async(lease) => {
                let handle = lease.handle;
                match self
                    .asyncs
                    .binary_search_by_key(&handle, |(lease, _)| lease.handle)
                {
                    Ok(position) => {
                        self.strong_gpu_len -=
                            usize::from(strong_gpu_unit(&self.asyncs[position].1));
                        self.asyncs[position].1 = access;
                    }
                    Err(position) => {
                        self.asyncs.reserve_exact(1);
                        self.asyncs.insert(position, (lease, access));
                        self.len += 1;
                    }
                }
            }
        }
        self.strong_gpu_len += strong;
    }

    fn remove(&mut self, actor: &GlobalFrontierActor) {
        match actor {
            GlobalFrontierActor::Lane(actor) => {
                if let Some(entries) = self.lanes.get_mut(&actor.global_warp_id) {
                    if let Ok(position) = entries
                        .lanes
                        .binary_search_by_key(&actor.lane, |(lane, _)| *lane)
                    {
                        let (_, removed) = entries.lanes.remove(position);
                        self.strong_gpu_len -= usize::from(strong_gpu_unit(&removed));
                        self.len -= 1;
                        if entries.lanes.is_empty() {
                            self.lanes.remove(&actor.global_warp_id);
                        }
                    }
                }
            }
            GlobalFrontierActor::Async(lease) => {
                let handle = lease.handle;
                if let Ok(position) = self
                    .asyncs
                    .binary_search_by_key(&handle, |(lease, _)| lease.handle)
                {
                    let (_, removed) = self.asyncs.remove(position);
                    self.strong_gpu_len -= usize::from(strong_gpu_unit(&removed));
                    self.len -= 1;
                    self.asyncs.shrink_to_fit();
                }
            }
        }
    }

    fn remove_superseded_by(&mut self, current: &GlobalAccess) {
        let clock = &current.clock;
        let mut removed = 0;
        let own_component = current.actor_index();
        let updates = clock.component_updates.as_slice();
        let kind = current.kind();
        let semantics = current.semantics;
        self.lanes.retain(|warp, entries| {
            let block = clock.component_base.block(*warp, clock.arena());
            // Update-layer components of this warp: the superseding actor's
            // own component only ever decides its own entry, which the
            // following insertion replaces anyway, so it does not affect the
            // scan key; any other lane's update does.
            let mut warp_updated = false;
            let mut foreign_update = false;
            for (index, _) in updates {
                if index / crate::WARP_SIZE == *warp {
                    warp_updated = true;
                    if *index != own_component {
                        foreign_update = true;
                    }
                }
            }
            // No component of this warp is known to the superseding clock, so
            // none of its events can happen-before the superseding access.
            if block.is_none() && !warp_updated {
                return true;
            }
            let key = block
                .filter(|_| !foreign_update)
                .map(|block| SupersedeScanKey {
                    block,
                    kind,
                    semantics,
                });
            if key.is_some() && entries.scanned == key {
                return true;
            }
            entries.scanned = key;
            let before = entries.lanes.len();
            entries
                .lanes
                .retain(|(_, access)| !access_is_superseded_by(access, current));
            removed += before - entries.lanes.len();
            if entries.lanes.len() < before {
                entries.lanes.shrink_to_fit();
            }
            !entries.lanes.is_empty()
        });
        if !clock.async_components.is_empty() {
            let arena = clock.arena();
            let before = self.asyncs.len();
            let mut current_chunk = usize::MAX;
            let mut chunk_observed = false;
            self.asyncs.retain(|(lease, access)| {
                let chunk = lease.handle.index / ASYNC_CHUNK;
                if chunk != current_chunk {
                    current_chunk = chunk;
                    chunk_observed = clock.async_components.chunk(chunk, arena).is_some();
                }
                !chunk_observed || !access_is_superseded_by(access, current)
            });
            removed += before - self.asyncs.len();
            if self.asyncs.len() < before {
                self.asyncs.shrink_to_fit();
            }
        }
        self.len -= removed;
    }

    /// Visit the entries `current` may conflict with: a same-proxy lane entry
    /// whose epoch the current clock already covers is ordered before it
    /// (`event_happens_before`), and validation would only return early for
    /// it, so it is skipped without the call. The polled-counter frontier
    /// holds one reader entry per polling lane, and this keeps a write's
    /// scan to the few entries it is actually unordered with.
    fn for_each_unordered_before(
        &self,
        current: &GlobalAccess,
        visit: &mut impl FnMut(&RecordedGlobalAccess),
    ) {
        debug_assert!(self.strong_gpu_len <= self.len);
        if self.strong_gpu_len == self.len && strong_gpu_unit(&current.recorded) {
            // Every pair is mutually morally strong: `validate_pair_in`
            // returns before its happens-before check for each of them.
            return;
        }
        let clock = &current.clock;
        let arena = clock.arena();
        let proxy = current.semantics.proxy();
        for (&warp, entries) in &self.lanes {
            let block = clock
                .component_base
                .block(warp, arena)
                .map(|block| arena.blocks.view(block));
            for (lane, access) in &entries.lanes {
                if access.semantics.proxy() == proxy {
                    let lane = *lane as usize;
                    let epoch = clock
                        .component_updates
                        .get(warp * crate::WARP_SIZE + lane)
                        .or_else(|| block.map(|block| block.word(lane)))
                        .unwrap_or(0);
                    if epoch >= access.frontier_epoch {
                        continue;
                    }
                }
                visit(&**access);
            }
        }
        for (_, access) in &self.asyncs {
            visit(&**access);
        }
    }

    fn into_single(self) -> (GlobalFrontierActor, RecordRef) {
        debug_assert_eq!(self.len, 1);
        if let Some((warp, entries)) = self.lanes.into_iter().next() {
            let (lane, access) = entries
                .lanes
                .into_iter()
                .next()
                .expect("one-entry global frontier has a lane entry");
            return (
                GlobalFrontierActor::Lane(GlobalActor::new(warp, lane as usize)),
                access,
            );
        }
        let (lease, access) = self
            .asyncs
            .into_iter()
            .next()
            .expect("one-entry global frontier has an async entry");
        (GlobalFrontierActor::Async(lease), access)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum GlobalFrontiers {
    #[default]
    Empty,
    One(GlobalFrontierActor, RecordRef),
    Many(Box<ManyGlobalFrontiers>),
}

impl GlobalFrontiers {
    fn for_each(&self, mut visit: impl FnMut(&RecordedGlobalAccess)) {
        match self {
            Self::Empty => {}
            Self::One(_, access) => visit(access),
            Self::Many(accesses) => {
                for entries in accesses.lanes.values() {
                    for (_, access) in &entries.lanes {
                        visit(access);
                    }
                }
                for (_, access) in &accesses.asyncs {
                    visit(access);
                }
            }
        }
    }

    fn insert(&mut self, actor: GlobalFrontierActor, access: RecordRef) {
        match self {
            Self::Empty => *self = Self::One(actor, access),
            Self::One(current_actor, current_access) if *current_actor == actor => {
                *current_access = access;
            }
            Self::One(current_actor, current_access) => {
                let mut accesses = ManyGlobalFrontiers::default();
                accesses.insert(current_actor.clone(), current_access.clone());
                accesses.insert(actor, access);
                *self = Self::Many(Box::new(accesses));
            }
            Self::Many(accesses) => {
                accesses.insert(actor, access);
            }
        }
    }

    fn remove(&mut self, actor: &GlobalFrontierActor) {
        match self {
            Self::Empty => {}
            Self::One(current_actor, _) if current_actor == actor => *self = Self::Empty,
            Self::One(_, _) => {}
            Self::Many(accesses) => {
                accesses.remove(actor);
                if accesses.len == 1 {
                    let (actor, access) = std::mem::take(accesses.as_mut()).into_single();
                    *self = Self::One(actor, access);
                }
            }
        }
    }

    fn remove_superseded_by(&mut self, current: &GlobalAccess) {
        let previous = std::mem::take(self);
        *self = match previous {
            Self::Empty => Self::Empty,
            Self::One(actor, access) => {
                if access_is_superseded_by(&access, current) {
                    Self::Empty
                } else {
                    Self::One(actor, access)
                }
            }
            Self::Many(mut accesses) => {
                accesses.remove_superseded_by(current);
                match accesses.len {
                    0 => Self::Empty,
                    1 => {
                        let (actor, access) = accesses.into_single();
                        Self::One(actor, access)
                    }
                    _ => Self::Many(accesses),
                }
            }
        };
    }

    fn for_each_unordered_before(
        &self,
        current: &GlobalAccess,
        mut visit: impl FnMut(&RecordedGlobalAccess),
    ) {
        match self {
            Self::Empty => {}
            Self::One(_, access) => visit(&**access),
            Self::Many(accesses) => accesses.for_each_unordered_before(current, &mut visit),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GlobalByteState {
    writers: GlobalFrontiers,
    readers: GlobalFrontiers,
    current_version: Option<Arc<GlobalVersion>>,
}

impl GlobalByteState {
    fn apply(&mut self, prepared: &PreparedGlobalAccess) {
        // Shared conflicts belong to RaceShadow. This shadow owns only the
        // exact version of the bytes, including invalidation by weak writes.
        if prepared.access.space() == PhysicalAccessSpace::Shared {
            if prepared.access.kind().writes() {
                self.current_version = prepared.new_version.clone();
            }
            return;
        }
        // Supersession needs an identical non-atomic signature, and no
        // non-atomic prior shares an atomic access's semantics, so an atomic
        // never supersedes anything: its frontier scans are skipped.
        let supersedes = !prepared.access.semantics.class().is_atomic_class();
        if prepared.access.kind().reads() {
            if supersedes {
                self.readers.remove_superseded_by(&prepared.access);
            }
            self.readers.insert(
                prepared.access.frontier_actor(),
                prepared.access.recorded.clone(),
            );
        }
        if prepared.access.kind().writes() {
            if supersedes {
                self.writers.remove_superseded_by(&prepared.access);
            }
            self.writers.insert(
                prepared.access.frontier_actor(),
                prepared.access.recorded.clone(),
            );
            self.readers.remove(&prepared.access.frontier_actor());
            self.current_version = prepared.new_version.clone();
        }
    }
}

fn access_is_superseded_by(prior: &RecordedGlobalAccess, current: &GlobalAccess) -> bool {
    // Racecheck processes synchronization before its causal descendants. For
    // identical non-atomic access signatures, a newer event therefore covers
    // every future conflict that the older event could expose. Atomic and
    // reduction events are deliberately excluded: actor scope can make two
    // otherwise identical atomic accesses morally strong with different
    // future observers.
    !prior.semantics.class().is_atomic_class()
        && prior.kind() == current.kind()
        && prior.semantics == current.semantics
        && event_happens_before(prior, current)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GlobalSegment {
    byte_offset: usize,
    byte_end: usize,
    state: GlobalByteState,
}

#[derive(Clone, Debug, Default)]
struct GlobalAllocationShadow {
    // Piecewise-constant byte state. Repeated vector accesses normally revisit
    // one exact interval instead of cloning the same frontier into every byte.
    segments: TransactionalIntervalMap<GlobalSegment>,
    // A dense, previously untouched weak-write batch needs one interval-map
    // node, not one node per active lane. Keep the exact lane witnesses inside
    // that batch node so a later overlapping access still receives the same
    // lane-precise conflict and happens-before validation.
    first_touches: TransactionalIntervalMap<GlobalFirstTouchSegment>,
}

impl GlobalAllocationShadow {
    fn segment_until(
        &self,
        byte_offset: usize,
        byte_end: usize,
    ) -> (Option<&GlobalSegment>, usize) {
        self.segments
            .state_until(byte_offset, byte_end, |segment| segment.byte_end)
    }

    fn first_touch_segment_until(
        &self,
        byte_offset: usize,
        byte_end: usize,
    ) -> (Option<&GlobalFirstTouchSegment>, usize) {
        self.first_touches
            .state_until(byte_offset, byte_end, |segment| segment.byte_end)
    }

    fn range_is_untouched(&self, byte_offset: usize, byte_end: usize) -> bool {
        let mut cursor = byte_offset;
        while cursor < byte_end {
            let (segment, next) = self.segment_until(cursor, byte_end);
            if segment.is_some() {
                return false;
            }
            debug_assert!(next > cursor, "global shadow gap scan must advance");
            cursor = next;
        }
        let mut cursor = byte_offset;
        while cursor < byte_end {
            let (segment, next) = self.first_touch_segment_until(cursor, byte_end);
            if segment.is_some() {
                return false;
            }
            debug_assert!(
                next > cursor,
                "global first-touch shadow gap scan must advance"
            );
            cursor = next;
        }
        true
    }

    /// Whether `prepared` (sorted by span) is one contiguous run of weak,
    /// unversioned writes; returns its byte range when it is.
    fn contiguous_weak_write_range(prepared: &[PreparedGlobalAccess]) -> Option<(usize, usize)> {
        let first_span = prepared.first()?.access.span();
        let byte_offset = first_span.byte_offset();
        let mut byte_end = byte_offset;
        let contiguous = prepared.iter().all(|prepared| {
            let span = prepared.access.span();
            let contiguous = span.allocation() == first_span.allocation()
                && prepared.access.space() == PhysicalAccessSpace::Global
                && span.byte_offset() == byte_end
                && prepared.access.kind().writes()
                && prepared.new_version.is_none();
            byte_end = span.byte_end();
            contiguous
        });
        contiguous.then_some((byte_offset, byte_end))
    }

    fn sort_prepared(prepared: &mut [PreparedGlobalAccess]) {
        let key = |prepared: &PreparedGlobalAccess| {
            let span = prepared.access.span();
            (span.byte_offset(), span.byte_end())
        };
        if !prepared.is_sorted_by_key(key) {
            prepared.sort_unstable_by_key(key);
        }
    }

    /// Apply one access to the part of the byte state in `[lo, hi)`.
    fn apply(&mut self, prepared: &PreparedGlobalAccess, lo: usize, hi: usize) {
        let span = prepared.access.span();
        let byte_offset = span.byte_offset().max(lo);
        let byte_end = span.byte_end().min(hi);
        self.update(byte_offset, byte_end, |state| state.apply(prepared));
    }

    fn update(
        &mut self,
        byte_offset: usize,
        byte_end: usize,
        mut update: impl FnMut(&mut GlobalByteState),
    ) {
        if byte_offset >= byte_end {
            return;
        }

        if let Some(segment) = self.segments.get_mut(byte_offset) {
            if segment.byte_end == byte_end {
                update(&mut segment.state);
                return;
            }
        }

        let mut replacement = Vec::new();
        let mut cursor = byte_offset;
        while cursor < byte_end {
            let (existing, next) = self.segment_until(cursor, byte_end);
            debug_assert!(next > cursor, "global shadow scan must advance");
            // A segment the access covers whole is dropped by the range
            // replacement below, so its state is moved out rather than
            // cloned; only the two boundary segments, which survive as split
            // remnants, still need a copy.
            let covered_whole = existing.is_some_and(|segment| {
                segment.byte_offset == cursor && segment.byte_end <= byte_end
            });
            let mut state = if covered_whole {
                std::mem::take(
                    &mut self
                        .segments
                        .get_mut(cursor)
                        .expect("a segment covered whole starts at the cursor")
                        .state,
                )
            } else {
                existing
                    .map(|segment| segment.state.clone())
                    .unwrap_or_default()
            };
            update(&mut state);
            replacement.push(GlobalSegment {
                byte_offset: cursor,
                byte_end: next,
                state,
            });
            cursor = next;
        }
        if let Err(replacement) = self.segments.try_replace_range(
            byte_offset,
            byte_end,
            replacement,
            |segment| segment.byte_offset,
            |segment| segment.byte_end,
            |segment, split_start, split_end| GlobalSegment {
                byte_offset: split_start,
                byte_end: split_end,
                state: segment.state.clone(),
            },
        ) {
            replace_global_segment_range_general(
                self.segments.general_mut(),
                byte_offset,
                byte_end,
                replacement,
            );
        }
    }

    /// Visit the prior accesses recorded in `[lo, hi)` that may conflict
    /// with an access of `current_kind` over `span`, in place: nothing is
    /// cloned out of the byte state.
    fn for_each_candidate(
        &self,
        span: PhysicalByteSpan,
        lo: usize,
        hi: usize,
        current: &GlobalAccess,
        visit: &mut impl FnMut(&RecordedGlobalAccess),
    ) {
        let mut cursor = lo;
        while cursor < hi {
            let (segment, next) = self.segment_until(cursor, hi);
            debug_assert!(next > cursor, "global candidate scan must advance");
            if let Some(segment) = segment {
                segment
                    .state
                    .writers
                    .for_each_unordered_before(current, &mut *visit);
                if current.kind().writes() {
                    segment
                        .state
                        .readers
                        .for_each_unordered_before(current, &mut *visit);
                }
            }
            cursor = next;
        }
        let mut cursor = lo;
        while cursor < hi {
            let (segment, next) = self.first_touch_segment_until(cursor, hi);
            debug_assert!(
                next > cursor,
                "global first-touch candidate scan must advance"
            );
            if let Some(segment) = segment {
                for prior in segment.overlapping_accesses(span) {
                    visit(&prior);
                }
            }
            cursor = next;
        }
    }

    /// Inspect actual writer records without inventing a memory access for a
    /// proxy fence. First-touch runs retain the same lane-specific witnesses.
    fn for_each_writer(
        &self,
        span: PhysicalByteSpan,
        lo: usize,
        hi: usize,
        visit: &mut impl FnMut(&RecordedGlobalAccess),
    ) {
        let mut cursor = lo;
        while cursor < hi {
            let (segment, next) = self.segment_until(cursor, hi);
            if let Some(segment) = segment {
                segment.state.writers.for_each(&mut *visit);
            }
            cursor = next;
        }
        let mut cursor = lo;
        while cursor < hi {
            let (segment, next) = self.first_touch_segment_until(cursor, hi);
            if let Some(segment) = segment {
                for access in segment.overlapping_accesses(span) {
                    visit(&access);
                }
            }
            cursor = next;
        }
    }

    #[cfg(test)]
    fn tracked_byte_count(&self) -> usize {
        let mut ranges = self
            .segments
            .iter()
            .map(|(_, segment)| (segment.byte_offset, segment.byte_end))
            .chain(
                self.first_touches
                    .iter()
                    .map(|(_, segment)| (segment.byte_offset, segment.byte_end)),
            )
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        let mut total = 0;
        let mut current: Option<(usize, usize)> = None;
        for (start, end) in ranges {
            match current {
                Some((current_start, current_end)) if start <= current_end => {
                    current = Some((current_start, current_end.max(end)));
                }
                Some((current_start, current_end)) => {
                    total += current_end - current_start;
                    current = Some((start, end));
                }
                None => current = Some((start, end)),
            }
        }
        if let Some((start, end)) = current {
            total += end - start;
        }
        total
    }
}

fn replace_global_segment_range_general(
    segments: &mut BTreeMap<usize, GlobalSegment>,
    byte_offset: usize,
    byte_end: usize,
    replacement: Vec<GlobalSegment>,
) {
    let mut overlapping = Vec::new();
    if let Some((&key, segment)) = segments.range(..=byte_offset).next_back() {
        if segment.byte_end > byte_offset {
            overlapping.push(key);
        }
    }
    overlapping.extend(segments.range(byte_offset..byte_end).map(|(&key, _)| key));
    overlapping.sort_unstable();
    overlapping.dedup();
    let left = overlapping
        .first()
        .and_then(|key| segments.get(key))
        .filter(|segment| segment.byte_offset < byte_offset)
        .map(|segment| GlobalSegment {
            byte_offset: segment.byte_offset,
            byte_end: byte_offset,
            state: segment.state.clone(),
        });
    let right = overlapping
        .last()
        .and_then(|key| segments.get(key))
        .filter(|segment| segment.byte_end > byte_end)
        .map(|segment| GlobalSegment {
            byte_offset: byte_end,
            byte_end: segment.byte_end,
            state: segment.state.clone(),
        });
    for key in overlapping {
        segments.remove(&key);
    }
    for segment in left.into_iter().chain(replacement).chain(right) {
        segments.insert(segment.byte_offset, segment);
    }
}

/// One write a protocol made to a word it declared.
///
/// Design API §3: an acquiring wait takes its edge from the first write whose
/// value satisfies the wait's predicate, so the checker needs that word's
/// writes in modification order with what each left behind and what each
/// released. Only declared addresses are kept this way, which is what makes
/// keeping them complete affordable -- a protocol owns a handful of words,
/// not every atomic address in the program.
///
/// The order is the order racecheck commits the writes. For an arrival
/// (`red`/`atom`) that is the modification order, because the engine holds
/// the atomic ordering reservation across the commit. A publication (`store`)
/// relies on the protocol having one publisher per word, which is what
/// publishing means.
/// How one word has been reached, for the undeclared-protocol claim.
///
/// Deliberately not a happens-before question: it records who touched the
/// word, never in what order, so the answer cannot move with the schedule the
/// way a race verdict does.
/// One actor's contact with a word: who reached it, and with what.
type WordContact = (GlobalActor, DynamicOpId);

/// A wait that declared a word: the actor, the operation, and the epoch that
/// actor stood at. A wait adjudicates no access, so this is all it has.
type WaitEvent = (GlobalActor, DynamicOpId, u64);

/// Which component of a clock positions an access.
///
/// An access carried by an async lease is positioned by that lease's component
/// and not by its lane's; comparing one against the other would be comparing
/// different counters. Only the handle is kept, never the lease itself -- a
/// retained `Arc<AsyncClockLease>` would hold a retired slot open and stop the
/// registry from reclaiming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WordFrontier {
    Lane(GlobalActor),
    Async(AsyncClockHandle),
}

impl WordFrontier {
    fn epoch_in(self, clock: &SparseLaneClock) -> u64 {
        match self {
            Self::Lane(actor) => clock.actor_epoch(actor),
            Self::Async(handle) => clock.async_component(handle),
        }
    }
}

/// A plain access that reached a word: whose it was, what positions it, the
/// epoch it stood at, and the operation to name in the report.
type BypassEvent = (GlobalActor, WordFrontier, u64, DynamicOpId);

/// Note one actor in a fixed-size slot, first two distinct actors win.
fn note_wait_event(slots: &mut [Option<WaitEvent>; 2], event: WaitEvent) {
    if slots
        .iter()
        .any(|slot| slot.as_ref().is_some_and(|(seen, _, _)| *seen == event.0))
    {
        return;
    }
    if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(event);
    }
}

fn note_bypass_event(slots: &mut [Option<BypassEvent>; 2], event: BypassEvent) {
    if slots
        .iter()
        .any(|slot| slot.as_ref().is_some_and(|(seen, _, _, _)| *seen == event.0))
    {
        return;
    }
    if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(event);
    }
}

/// A recorded wait by another actor that `clock` does not cover.
///
/// Epoch zero is "no position yet", not "a position everything covers".
/// `apply_declared_word_wait` gives each wait a nonzero event epoch even though
/// it adjudicates no memory access. Keep the zero guard so an unpositioned
/// event cannot accidentally prove ordering.
fn unordered_wait_under<'a>(
    slots: &'a [Option<WaitEvent>; 2],
    actor: GlobalActor,
    clock: &SparseLaneClock,
) -> Option<&'a WaitEvent> {
    slots.iter().flatten().find(|(other, _, epoch)| {
        *other != actor && (*epoch == 0 || clock.actor_epoch(*other) < *epoch)
    })
}

/// A recorded plain access by another actor that `clock` does not cover.
///
/// The same test `event_happens_before` makes: covering an access's own
/// frontier epoch covers its complete causal past, and the frontier is the
/// lease's when one carries the access and the lane's otherwise.
fn unordered_plain_under<'a>(
    slots: &'a [Option<BypassEvent>; 2],
    actor: GlobalActor,
    clock: &SparseLaneClock,
) -> Option<&'a BypassEvent> {
    slots.iter().flatten().find(|(owner, frontier, epoch, _)| {
        *owner != actor && frontier.epoch_in(clock) < *epoch
    })
}

#[derive(Clone, Debug, Default)]
struct ProtocolWordUse {
    /// Up to two distinct actors seen writing the word, with an operation
    /// apiece. Two is enough to answer "is some writer not this reader?"
    /// whatever order they arrive in, and keeps the record a fixed size on a
    /// word every CTA contributes to.
    writers: [Option<WordContact>; 2],
    /// Up to two distinct actors seen reading the word and nothing else.
    readers: [Option<WordContact>; 2],
    /// A read of this word handed its reader a write it did not already hold.
    /// That is what makes a word a protocol rather than a value: the reader
    /// learned something here, and nothing else told it. A word read only
    /// behind an order the program already had -- a barrier, an acquire on
    /// some other word -- delivers nothing, and is not claimed.
    delivered_order: bool,
    /// Some access to this word went through the primitive, so the protocol
    /// is declared and the claim does not apply.
    declared: bool,
    /// Waits that declared the word, and plain accesses that reached it.
    ///
    /// A wait performs no adjudicated access of its own, so a word whose only
    /// primitive use is a wait has no pair for `validate_pair_in` to catch --
    /// the claim has to be made on the address.
    ///
    /// Reaching a declared word plainly is only a defect if it is *concurrent*
    /// with the protocol: an initializing store a barrier separates from every
    /// wait is ordered, and ordinary. A wait adjudicates no access, so
    /// `validate_pair_in` never sees this pair -- the order has to be settled
    /// here, from the epochs.
    ///
    /// Two distinct actors a side, the same fixed size `writers` and `readers`
    /// keep, so the record does not grow with how long a kernel runs. Both
    /// hooks run on every occurrence -- each plain access is judged against the
    /// waits recorded so far, and each wait against the plain accesses -- so an
    /// unordered pair is proven whenever either of its two members is among the
    /// first two actors on its own side. A word more than two actors reach
    /// plainly *and* more than two wait on can therefore lose a pair whose both
    /// members arrived late; that direction is a missed report, never a
    /// fabricated one.
    declared_ops: [Option<WaitEvent>; 2],
    bypassing_ops: [Option<BypassEvent>; 2],
    /// Some wait and some plain access, by different actors, that neither
    /// side's clock ordered. Only then is the bypass proven, and reported.
    bypass: Option<DeclaredWordBypassDiagnostic>,
    /// The word's span, kept because the map key drops its width.
    span: Option<PhysicalByteSpan>,
}

impl ProtocolWordUse {
    /// Remember one actor in a fixed-size slot, first two distinct ones win.
    fn note(slots: &mut [Option<WordContact>; 2], actor: GlobalActor, op: &DynamicOpId) {
        if slots
            .iter()
            .any(|slot| slot.as_ref().is_some_and(|(seen, _)| *seen == actor))
        {
            return;
        }
        if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some((actor, op.clone()));
        }
    }

    /// One actor writes it, a different one waits on it: a protocol.
    ///
    /// Asked as "is there a pair that differs", never as "who came first", so
    /// the answer cannot move with the schedule.
    fn protocol_pair(&self) -> Option<(&WordContact, &WordContact)> {
        for writer in self.writers.iter().flatten() {
            for reader in self.readers.iter().flatten() {
                if writer.0 != reader.0 {
                    return Some((writer, reader));
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
struct DeclaredWordWrite {
    value: u64,
    /// The version this write created, kept whole so the wait that accepts it
    /// takes its edge through `apply_load_ordering`, the same path an ordinary
    /// acquire load uses.
    version: Arc<GlobalVersion>,
    operation: DynamicOpId,
}

#[derive(Clone, Debug)]
struct PreparedGlobalAccess {
    access: GlobalAccess,
    new_version: Option<Arc<GlobalVersion>>,
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Debug)]
enum ReplayFrontierActor {
    Lane(GlobalActor),
    // Boxed so the replay record stays smaller than a live `GlobalAccess`.
    Async(Box<(AsyncTokenId, AsyncClockHandle, u64)>),
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Debug)]
struct GlobalReplayAccess {
    actor: GlobalActor,
    frontier_actor: ReplayFrontierActor,
    frontier_epoch: u64,
    clock_snapshot: u32,
    /// Zero means no write version; otherwise the original first unit's ID.
    version_id: u64,
    witness: PhysicalRaceWitness,
    semantics: MemoryAccessSemantics,
    unit_bytes: u32,
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ReplayClockUpdatesKey {
    Empty,
    One(usize, u64),
    Many(usize),
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReplayClockKey {
    component_base: usize,
    component_updates: ReplayClockUpdatesKey,
    async_components: usize,
    async_registry: usize,
    proxy_bridges: Option<usize>,
}

#[cfg(any(test, feature = "profile"))]
impl SparseLaneClock {
    fn replay_key(&self) -> ReplayClockKey {
        let component_base = match &self.component_base {
            SparseComponentBase::Empty => 0,
            SparseComponentBase::Many(updates) => Arc::as_ptr(updates) as usize,
        };
        let component_updates = match &self.component_updates {
            SparseComponentUpdates::Empty => ReplayClockUpdatesKey::Empty,
            SparseComponentUpdates::One((component, epoch)) => {
                ReplayClockUpdatesKey::One(*component, *epoch)
            }
            SparseComponentUpdates::Many(updates) => {
                ReplayClockUpdatesKey::Many(Arc::as_ptr(updates) as usize)
            }
        };
        let proxy_bridges = self.proxy_bridges.as_ref().map(|bridges| {
            if bridges.is_empty() {
                0
            } else {
                Arc::as_ptr(bridges) as usize
            }
        });
        let async_components = self.async_components.storage_identity();
        ReplayClockKey {
            component_base,
            component_updates,
            async_components,
            async_registry: Arc::as_ptr(&self.async_registry) as usize,
            proxy_bridges,
        }
    }
}

#[cfg(any(test, feature = "profile"))]
impl GlobalReplayAccess {
    fn from_access(
        access: &GlobalAccess,
        clock_snapshot: u32,
        version: Option<&Arc<GlobalVersion>>,
    ) -> Self {
        let frontier_actor = match &access.frontier_actor() {
            GlobalFrontierActor::Lane(actor) => ReplayFrontierActor::Lane(*actor),
            GlobalFrontierActor::Async(lease) => ReplayFrontierActor::Async(Box::new((
                lease.token.clone(),
                lease.handle,
                lease.issue_epoch,
            ))),
        };
        Self {
            actor: access.actor(),
            frontier_actor,
            frontier_epoch: access.frontier_epoch,
            clock_snapshot,
            version_id: version
                .map_or(0, |version| version.unit_id_at(access.span().byte_offset())),
            witness: access.witness(),
            semantics: access.semantics,
            unit_bytes: access.unit_bytes,
        }
    }

    fn into_prepared_access(
        self,
        clock_snapshots: &[Arc<SparseLaneClock>],
        async_leases: &mut BTreeMap<(AsyncTokenId, AsyncClockHandle), Arc<AsyncClockLease>>,
    ) -> PreparedGlobalAccess {
        let version_id = self.version_id;
        let frontier_actor = match self.frontier_actor {
            ReplayFrontierActor::Lane(actor) => GlobalFrontierActor::Lane(actor),
            ReplayFrontierActor::Async(actor) => {
                let (token, handle, issue_epoch) = *actor;
                let lease = async_leases
                    .entry((token.clone(), handle))
                    .or_insert_with(|| {
                        Arc::new(AsyncClockLease {
                            token,
                            handle,
                            registry: std::sync::Weak::new(),
                            issue_epoch,
                        })
                    });
                GlobalFrontierActor::Async(Arc::clone(lease))
            }
        };
        let clock = clock_snapshots
            .get(self.clock_snapshot as usize)
            .expect("replay access must reference a recorded clock snapshot")
            .as_ref()
            .clone();
        let access = GlobalAccess {
            recorded: Arc::new(RecordedGlobalAccess::new(
                self.actor,
                frontier_actor,
                self.frontier_epoch,
                self.witness,
                self.semantics,
                self.unit_bytes,
            )),
            clock: clock,
        };
        let new_version = (version_id != 0).then(|| {
            let carrier = if access.unit_bytes == 0 {
                access.recorded.clone()
            } else {
                let width = access.unit_bytes as usize;
                let origin = access.unit_origin();
                let span = PhysicalByteSpan::new(
                    access.span().allocation(),
                    origin,
                    (access.span().byte_end() - origin).div_ceil(width) * width,
                )
                .expect("recorded transfer elements have representable bounds");
                Arc::new(access.recorded.with_span(span))
            };
            Arc::new(GlobalVersion {
                id: version_id,
                carrier,
                payload: ReleasePayload::default(),
            })
        });
        PreparedGlobalAccess {
            access,
            new_version,
        }
    }
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Debug)]
struct GlobalReplayBatch {
    accesses: Box<[GlobalReplayAccess]>,
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayAsyncPhase {
    Issue,
    Complete(AsyncGroupMilestone),
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Copy, Debug)]
struct ReplayAsyncState {
    handle: AsyncClockHandle,
    issue_epoch: u64,
    current_snapshot: u32,
    source_read_snapshot: Option<u32>,
    full_snapshot: Option<u32>,
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Debug)]
enum GlobalReplayEvent {
    AccessBatch(GlobalReplayBatch),
    AsyncLifecycle {
        phase: ReplayAsyncPhase,
        token: AsyncTokenId,
        state: Option<ReplayAsyncState>,
    },
    AsyncRetire {
        token: AsyncTokenId,
    },
    AsyncAcquire {
        global_warp_id: usize,
        mask: WarpMask,
        clock_snapshot: Option<u32>,
        read_observations: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    },
    AsyncPublishPhysical {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        clock_snapshot: Option<u32>,
        read_observations: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    },
    ProxyFence {
        operation: DynamicOpId,
        kind: crate::OperationKind,
        mask: WarpMask,
        effect: ProxyAsyncFenceEffect,
    },
    TensorMap {
        operation: DynamicOpId,
        kind: crate::OperationKind,
        mask: WarpMask,
        observation: TensorMapObservation,
    },
    Fence {
        operation: DynamicOpId,
        kind: crate::OperationKind,
        mask: WarpMask,
        effect: MemoryFenceEffect,
    },
    WarpSync {
        global_warp_id: usize,
        mask: WarpMask,
    },
    ResetPhysicalBarriers {
        barrier_ids: Box<[PhysicalBarrierId]>,
    },
    RetainPhysicalBarrierGenerations {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        conditional: Option<u64>,
    },
    PhysicalBarrierRelease {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    },
    PhysicalBarrierAcquire {
        barrier_id: PhysicalBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
        acquire: bool,
    },
    NamedBarrierRelease {
        barrier_id: crate::NamedBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    },
    NamedBarrierAcquire {
        barrier_id: crate::NamedBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    },
    ClusterBarrierRelease {
        barrier_id: crate::ClusterBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
        publishes_memory: bool,
    },
    ClusterBarrierAcquire {
        barrier_id: crate::ClusterBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    },
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Debug)]
struct GlobalReplaySummary {
    findings: BTreeSet<PhysicalRaceFinding>,
    scope_diagnostics: BTreeSet<GlobalScopeMismatchDiagnostic>,
    declared_word_bypasses: BTreeSet<DeclaredWordBypassDiagnostic>,
    incomplete_reasons: Vec<RaceCheckIncompleteReason>,
    access_count: usize,
    batch_count: usize,
}

#[cfg(any(test, feature = "profile"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GlobalReplayStorageStats {
    access_count: usize,
    batch_count: usize,
    snapshot_count: usize,
    access_record_bytes: usize,
    snapshot_record_bytes: usize,
    event_count: usize,
    event_record_bytes: usize,
}

#[derive(Clone, Debug)]
struct GlobalFirstTouchSegment {
    byte_offset: usize,
    byte_end: usize,
    operation: Arc<DynamicOpId>,
    byte_width: usize,
    lanes: Box<[GlobalFirstTouchLane]>,
}

#[derive(Clone, Debug)]
struct GlobalFirstTouchLane {
    lane: u8,
    /// The lane's frontier epoch at the write (see [`RecordedGlobalAccess`]).
    epoch: u64,
}

impl GlobalFirstTouchSegment {
    fn from_prepared(
        byte_offset: usize,
        byte_end: usize,
        prepared: Vec<PreparedGlobalAccess>,
    ) -> Self {
        let first = prepared
            .first()
            .expect("a first-touch segment has at least one lane");
        let operation = Arc::new(first.access.operation().clone());
        let byte_width = first.access.span().byte_len();
        let lanes = prepared
            .into_iter()
            .map(|prepared| {
                debug_assert_eq!(
                    prepared.access.operation(),
                    operation.as_ref(),
                    "one compact first-touch batch has one operation",
                );
                debug_assert_eq!(
                    prepared.access.span().byte_len(),
                    byte_width,
                    "one compact first-touch batch has one lane width",
                );
                debug_assert_eq!(
                    prepared.access.frontier_actor(),
                    GlobalFrontierActor::Lane(prepared.access.actor()),
                    "a weak direct first-touch write has a lane frontier",
                );
                GlobalFirstTouchLane {
                    lane: prepared.access.actor().lane,
                    epoch: prepared.access.frontier_epoch,
                }
            })
            .collect();
        Self {
            byte_offset,
            byte_end,
            operation,
            byte_width,
            lanes,
        }
    }

    fn overlapping_accesses(
        &self,
        span: PhysicalByteSpan,
    ) -> impl Iterator<Item = RecordedGlobalAccess> + '_ {
        self.lanes
            .iter()
            .enumerate()
            .filter_map(move |(index, lane)| {
                let byte_offset = self
                    .byte_offset
                    .checked_add(index.checked_mul(self.byte_width)?)
                    .expect("first-touch lane offset must fit usize");
                let lane_span =
                    PhysicalByteSpan::new(span.allocation(), byte_offset, self.byte_width)
                        .expect("first-touch lane span is nonempty and representable");
                lane_span.overlaps(span).then(|| {
                    let actor =
                        GlobalActor::new(self.operation.global_warp_id(), usize::from(lane.lane));
                    RecordedGlobalAccess::new(
                        actor,
                        GlobalFrontierActor::Lane(actor),
                        lane.epoch,
                        PhysicalRaceWitness::from_parts_shared(
                            Arc::clone(&self.operation),
                            usize::from(lane.lane),
                            PhysicalAccessKind::Write,
                            PhysicalAccessSpace::Global,
                            lane_span,
                        ),
                        MemoryAccessSemantics::plain(),
                        0,
                    )
                })
            })
    }
}

#[derive(Clone, Debug)]
struct StagedGlobalBatch {
    actor_updates: BTreeMap<GlobalActor, GlobalActorState>,
    accesses: Box<[PreparedGlobalAccess]>,
}

#[derive(Clone, Debug)]
struct GlobalAsyncTokenState {
    lease: Arc<AsyncClockLease>,
    current: SparseLaneClock,
    source_read: Option<SparseLaneClock>,
    full: Option<SparseLaneClock>,
    // Captured at issue, never at completion: later issuer work must not leak
    // into the publication of an asynchronous release.
    release_fence: Option<Arc<ReleaseHead>>,
    // Read-from is captured with the source access, but is not available to
    // a thread's acquire fence until that thread observes completion.
    read_observations: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    tcgen_publication: TcgenFenceFrontier,
    shared_frontier: Option<SharedClockFrontier>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GlobalExecutionPayload {
    clock: SparseLaneClock,
    read_observations: BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
}

impl GlobalExecutionPayload {
    fn merge(&mut self, other: &Self) {
        self.clock.merge(&other.clock);
        merge_pending_acquire(
            &mut self.read_observations,
            other.read_observations.values().cloned(),
        );
    }

    fn is_bottom(&self) -> bool {
        self.clock.is_bottom() && self.read_observations.is_empty()
    }
}

pub(crate) type GlobalAsyncPublication = GlobalExecutionPayload;

fn merge_pending_acquire(
    pending: &mut BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>,
    heads: impl IntoIterator<Item = Arc<ReleaseHead>>,
) {
    for head in heads {
        // Heads with the same key were released by the same lane. A later
        // read can observe an older release via another location; retain the
        // dominating head rather than replacing it with that older snapshot.
        if pending
            .get(&head.key)
            .is_some_and(|prior| Arc::ptr_eq(prior, &head) || prior.clock.dominates(&head.clock))
        {
            continue;
        }
        pending.insert(head.key, head);
    }
}

/// Launch-wide global-memory race state shared by every shard.
///
/// Everything here is keyed by memory, not by actor: the byte state of each
/// allocation (under its own lock, so shards working on different buffers
/// never wait for each other), the tracked-allocation filter, the version
/// identity counter, and the report sets. Actor clocks, staged batches, async
/// tokens and barrier tables stay in the per-shard [`GlobalRaceState`].
/// Byte-state cells split a large allocation into independently locked
/// stripes of this many bytes, so shards working on different tiles of one
/// weight buffer do not queue on one lock.
const SHADOW_STRIPE_BYTES: usize = 1 << 16;

/// Allocation numbers are arena-local: global and shared may use the same ID.
/// The space is therefore part of every version/shadow cell's identity.
type ShadowCellKey = ((PhysicalAccessSpace, PhysicalAllocationId), usize);

fn shadow_cell_keys(
    (space, span): (PhysicalAccessSpace, PhysicalByteSpan),
) -> impl Iterator<Item = ShadowCellKey> {
    let first = span.byte_offset() / SHADOW_STRIPE_BYTES;
    let last = (span.byte_end() - 1) / SHADOW_STRIPE_BYTES;
    (first..=last).map(move |stripe| ((space, span.allocation()), stripe))
}

/// The part of `span` inside `stripe`, as a byte range.
fn clip_to_stripe(span: PhysicalByteSpan, stripe: usize) -> (usize, usize) {
    let lo = stripe * SHADOW_STRIPE_BYTES;
    let hi = lo + SHADOW_STRIPE_BYTES;
    (span.byte_offset().max(lo), span.byte_end().min(hi))
}

/// How many weak reads a cell's pending log may hold before the next locker
/// drains it. The bound keeps drain latency and the async-clock leases pinned
/// by undrained entries proportional to the cell count, not the read count.
const PENDING_READS_CAP: usize = 64;

/// One byte-state stripe: the interval-map shadow plus a FIFO log of weak
/// reads published while only the read side of `state` was held. Every path
/// that mutates the shadow or consults its reader frontiers drains the log
/// first, so the shadow always holds the exact per-cell apply sequence the
/// immediate scheme would have produced.
#[derive(Debug, Default)]
pub(crate) struct ShadowCell {
    state: RwLock<GlobalAllocationShadow>,
    pending_reads: Mutex<Vec<PreparedGlobalAccess>>,
}

impl ShadowCell {
    fn pending_len(&self) -> usize {
        self.pending_reads
            .lock()
            .expect("global racecheck pending-read lock was poisoned")
            .len()
    }

    /// Apply every pending weak read of this `stripe` to the shadow, oldest
    /// first.
    fn drain_pending(&self, stripe: usize, shadow: &mut GlobalAllocationShadow) {
        let pending = std::mem::take(
            &mut *self
                .pending_reads
                .lock()
                .expect("global racecheck pending-read lock was poisoned"),
        );
        for prepared in &pending {
            let (lo, hi) = clip_to_stripe(prepared.access.span(), stripe);
            shadow.apply(prepared, lo, hi);
        }
    }
}

/// How many merged findings the global byte shadow retains before it only
/// counts further ones.
const RACECHECK_MAX_RETAINED_FINDINGS: usize = 1 << 20;

/// How many writes one word's history holds before the record stops being the
/// whole history. A wait names a position in it, so a truncated record cannot
/// be indexed into; reaching this is reported, never silently absorbed.
const RACECHECK_MAX_DECLARED_WORD_WRITES: usize = 1 << 16;

/// How many distinct unpollable-width strong global write spans are kept.
///
/// These exist only to make a declared-word claim fail closed, and one span per
/// static store site is enough for that; the cap bounds a kernel that writes a
/// different wide span every iteration.
const RACECHECK_MAX_UNRECORDED_STRONG_WRITES: usize = 1 << 14;

/// One side of a finding without its byte range.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct WitnessSite {
    /// The *static* source position, not the dynamic instance. A spin polls
    /// the same read a million times and every poll is a distinct
    /// `DynamicOpId` -- different sequence, different loop ordinal -- so
    /// keying on the instance would keep one finding per poll and the retained
    /// set would grow with how long the kernel ran. The report still names a
    /// concrete instance: the first one merged into the site carries it.
    kernel_index: usize,
    global_warp_id: usize,
    source_op: u64,
    loop_sites: Vec<u64>,
    lane: usize,
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    allocation: PhysicalAllocationId,
}

impl WitnessSite {
    fn of(witness: &PhysicalRaceWitness) -> Self {
        let operation = witness.shared_operation();
        Self {
            kernel_index: operation.kernel_index(),
            global_warp_id: operation.global_warp_id(),
            source_op: operation.source_op_id().get(),
            loop_sites: operation
                .loop_frames()
                .iter()
                .map(|frame| frame.loop_site_id().get())
                .collect(),
            lane: witness.lane(),
            kind: witness.kind(),
            space: witness.space(),
            allocation: witness.span().allocation(),
        }
    }
}

/// The pair of access sites a finding reports; findings of one site differ
/// only in their byte ranges.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FindingSite {
    kind: PhysicalRaceKind,
    prior: WitnessSite,
    current: WitnessSite,
    ordering_failure: PhysicalRaceOrderingFailure,
    reviewed_tmem_load: bool,
}

impl FindingSite {
    fn of(finding: &PhysicalRaceFinding) -> Self {
        Self {
            kind: finding.kind(),
            prior: WitnessSite::of(finding.prior()),
            current: WitnessSite::of(finding.current()),
            ordering_failure: finding.ordering_failure(),
            reviewed_tmem_load: finding.requires_unwaited_tmem_load_review(),
        }
    }
}

/// The retained race findings of a launch: merged per site into byte ranges
/// and bounded.
///
/// One access pair used to produce one finding per overlapping byte or
/// transfer unit — a 3.5 KB write against byte-wise reads made 3,584 findings,
/// each carrying its own normalized copies of both operations — and the set
/// was unbounded: MegaMoE t1536_m1536 retained 231 M findings (132 GB, 77 % of
/// its RSS) over about a thousand distinct operation pairs. Findings of one
/// site (same kind, operations, lanes, access kinds and spaces) whose
/// overlaps touch are now kept as one finding over the hull of their spans,
/// so a site costs one record per contiguous conflicting range, and the set
/// keeps at most `cap` findings: a finding that would add a range beyond the
/// cap is counted in `dropped` and reported as
/// `RaceCheckIncompleteReason::FindingsTruncated`. A finding that merges into
/// a retained range is always kept, whatever the cap.
pub(crate) struct FindingAggregate {
    // Per site, findings sorted by overlap offset with no two touching.
    sites: BTreeMap<FindingSite, Vec<PhysicalRaceFinding>>,
    len: usize,
    cap: usize,
    dropped: u64,
}

impl Default for FindingAggregate {
    fn default() -> Self {
        Self::with_cap(RACECHECK_MAX_RETAINED_FINDINGS)
    }
}

impl FindingAggregate {
    fn with_cap(cap: usize) -> Self {
        Self {
            sites: BTreeMap::new(),
            len: 0,
            cap,
            dropped: 0,
        }
    }

    /// Merge `finding` into its site's ranges; returns whether it was
    /// retained (merged or added) rather than counted as dropped.
    fn insert(&mut self, finding: PhysicalRaceFinding) -> bool {
        let site = FindingSite::of(&finding);
        let at_cap = self.len >= self.cap;
        let entries = if at_cap {
            match self.sites.get_mut(&site) {
                Some(entries) => entries,
                None => {
                    self.dropped += 1;
                    return false;
                }
            }
        } else {
            self.sites.entry(site).or_default()
        };
        let overlap = finding.overlap();
        // Entries touching the new range form one contiguous run: those
        // ending before it are skipped, those starting after its end are
        // untouched.
        let lo = entries.partition_point(|entry| entry.overlap().byte_end() < overlap.byte_offset());
        let hi = entries.partition_point(|entry| entry.overlap().byte_offset() <= overlap.byte_end());
        if lo == hi {
            if at_cap {
                self.dropped += 1;
                return false;
            }
            entries.insert(lo, finding);
            self.len += 1;
            return true;
        }
        let merged = entries[lo..hi]
            .iter()
            .fold(finding, |merged, entry| merged.hull(entry));
        entries.drain(lo..hi);
        entries.insert(lo, merged);
        self.len -= hi - lo - 1;
        true
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dropped(&self) -> u64 {
        self.dropped
    }

    fn iter(&self) -> impl Iterator<Item = &PhysicalRaceFinding> {
        self.sites.values().flatten()
    }

    /// Every retained finding in the report's order.
    fn findings(&self) -> Vec<PhysicalRaceFinding> {
        let mut findings = self.iter().cloned().collect::<Vec<_>>();
        findings.sort_unstable();
        findings
    }
}

#[derive(Default)]
pub(crate) struct GlobalRaceShared {
    bytes: RwLock<BTreeMap<ShadowCellKey, Arc<ShadowCell>>>,
    tracked_allocations: Option<RwLock<BTreeSet<PhysicalAllocationId>>>,
    // Bumped on every insertion so per-thread copies of the tracked set know
    // when to refresh; the set only grows, so a stale "tracked" answer is
    // still right and only a stale "untracked" answer needs a refresh.
    tracked_generation: AtomicU64,
    findings: Mutex<FindingAggregate>,
    scope_diagnostics: Mutex<BTreeSet<GlobalScopeMismatchDiagnostic>>,
    /// Per declared word (allocation, byte offset), what the protocol wrote.
    declared_word_writes: Mutex<BTreeMap<(PhysicalAllocationId, usize), Vec<DeclaredWordWrite>>>,
    /// Per word, which actors reached it with a scoped access and whether any
    /// primitive ever claimed it. A word two actors use this way, with at
    /// least one writing and no declaration anywhere, is a protocol the
    /// checker was never told the rules of.
    protocol_word_use: Mutex<BTreeMap<(PhysicalAllocationId, usize), ProtocolWordUse>>,
    /// Strong global writes whose width no wait can poll, by span.
    ///
    /// The post-image read-back only covers 4- and 8-byte writes, because those
    /// are the widths `wait_until` polls. A `b128` release store can still
    /// land on a word a wait declares; its value never reaches the word's
    /// history, so the wait would read an incomplete record. Keeping the spans
    /// lets a claim that overlaps one report incomplete rather than call a
    /// published protocol unpublished. Deduplicated by span and capped, so this
    /// does not grow with how long a kernel runs.
    unrecorded_strong_writes:
        Mutex<BTreeMap<(PhysicalAllocationId, usize, usize), DynamicOpId>>,
    incomplete_reasons: Mutex<Vec<RaceCheckIncompleteReason>>,
    next_version: AtomicU64,
    async_registry: Arc<AsyncClockRegistry>,
    /// Latest SC head per source CTA, scope and proxy. Heads with the same
    /// key mutually synchronize, so the latest dominates its predecessors.
    /// Keep narrow and wide scopes separate: a later CTA fence must not hide
    /// a GPU fence from an observer outside that CTA.
    sc_fences: Mutex<BTreeMap<(usize, MemoryScope, MemoryProxy), Arc<ReleaseHead>>>,
}

impl fmt::Debug for GlobalRaceShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlobalRaceShared")
            .field(
                "next_version",
                &self.next_version.load(AtomicOrdering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl GlobalRaceShared {
    pub(crate) fn new(tracked_allocations: Option<BTreeSet<PhysicalAllocationId>>) -> Self {
        Self {
            tracked_allocations: tracked_allocations.map(RwLock::new),
            next_version: AtomicU64::new(1),
            ..Self::default()
        }
    }

    /// Records the floor the collector retires under (see
    /// `AsyncClockRegistry::retirement`).
    pub(crate) fn note_retirement_floor(&self, floor: &GlobalFloor) {
        self.async_registry.note_retirement_floor(floor);
    }

    pub(crate) fn findings(&self) -> Vec<PhysicalRaceFinding> {
        self.findings
            .lock()
            .expect("global racecheck findings lock was poisoned")
            .findings()
    }

    /// Retained findings after merging, and how many were dropped at the cap.
    pub(crate) fn findings_retained(&self) -> (usize, u64) {
        let findings = self
            .findings
            .lock()
            .expect("global racecheck findings lock was poisoned");
        (findings.len(), findings.dropped())
    }

    pub(crate) fn scope_diagnostics(&self) -> Vec<GlobalScopeMismatchDiagnostic> {
        self.scope_diagnostics
            .lock()
            .expect("global racecheck diagnostics lock was poisoned")
            .iter()
            .cloned()
            .collect()
    }

    pub(crate) fn declared_word_bypasses(&self) -> Vec<DeclaredWordBypassDiagnostic> {
        self.claimed_word_bypasses()
    }

    pub(crate) fn incomplete_reasons(&self) -> Vec<RaceCheckIncompleteReason> {
        let mut reasons = self.incomplete_reasons
            .lock()
            .expect("global racecheck incomplete lock was poisoned")
            .clone();
        // Adding an allocation after its first write cannot recover reads
        // omitted before that write. Never certify such a speculative seed.
        if self.tracked_generation.load(AtomicOrdering::Acquire) != 0 {
            reasons.push(RaceCheckIncompleteReason::GlobalWriteSeedIncomplete);
        }
        reasons
    }

    fn insert_finding(&self, finding: PhysicalRaceFinding) {
        let _ = self
            .findings
            .lock()
            .expect("global racecheck findings lock was poisoned")
            .insert(finding);
    }

    fn insert_scope_diagnostic(&self, diagnostic: GlobalScopeMismatchDiagnostic) {
        self.scope_diagnostics
            .lock()
            .expect("global racecheck diagnostics lock was poisoned")
            .insert(diagnostic);
    }

    fn record_declared_word_write(
        &self,
        span: PhysicalByteSpan,
        value: u64,
        version: Arc<GlobalVersion>,
        operation: DynamicOpId,
    ) -> bool {
        let mut writes = self
            .declared_word_writes
            .lock()
            .expect("global racecheck declared-word lock was poisoned");
        let history = writes
            .entry((span.allocation(), span.byte_offset()))
            .or_default();
        // Retained state must not grow with how long a kernel runs. A word
        // that accumulates without ever being waited on -- an `atom.max` amax,
        // a float reduction into a workspace -- would otherwise keep one entry
        // per contribution. Past the cap the history stops being the whole
        // history, so it stops being usable: the caller turns that into an
        // incomplete reason rather than letting a wait index into a truncated
        // record.
        if history.len() >= RACECHECK_MAX_DECLARED_WORD_WRITES {
            return false;
        }
        history.push(DeclaredWordWrite {
            value,
            version,
            operation,
        });
        true
    }

    /// The values a protocol wrote to one declared word, in order.
    ///
    /// This is what leaves the checker: the wait evaluates its predicate where
    /// it was written, against these, and names back a position. The versions
    /// stay here, so nothing but the checker ever handles causality.
    pub(crate) fn declared_word_values(&self, span: PhysicalByteSpan) -> Vec<u64> {
        self.declared_word_writes
            .lock()
            .expect("global racecheck declared-word lock was poisoned")
            .get(&(span.allocation(), span.byte_offset()))
            .map(|writes| writes.iter().map(|write| write.value).collect())
            .unwrap_or_default()
    }

    /// The writes a protocol made to one declared word, in order.
    fn declared_word_writes_at(&self, span: PhysicalByteSpan) -> Vec<DeclaredWordWrite> {
        self.declared_word_writes
            .lock()
            .expect("global racecheck declared-word lock was poisoned")
            .get(&(span.allocation(), span.byte_offset()))
            .cloned()
            .unwrap_or_default()
    }

    /// Note one access to a word, for the undeclared-protocol claim.
    ///
    /// Only `claim_protocol_word` declares a word now, off the address the
    /// wait names; an access carries no declaration of its own.
    ///
    /// `reads_only` separates a waiter from a contributor. A word that only
    /// ever accumulates -- `atom.max` for an amax, a float reduction into a
    /// workspace -- has no waiter and is not a protocol, which the inventory
    /// settled on its own terms: "a word with no waiter is not a protocol
    /// word". Requiring one plain load from another actor keeps those out.
    fn record_protocol_word_use(
        &self,
        span: PhysicalByteSpan,
        actor: GlobalActor,
        operation: &DynamicOpId,
        writes: bool,
        reads_only: bool,
        plain: Option<(&RecordRef, &SparseLaneClock)>,
    ) {
        let mut uses = self
            .protocol_word_use
            .lock()
            .expect("global racecheck protocol-word lock was poisoned");
        let entry = uses
            .entry((span.allocation(), span.byte_offset()))
            .or_default();
        entry.span.get_or_insert(span);
        // A publisher is spelled in raw PTX now -- `st.release`, `red`, `atom`
        // -- so reaching the word without the primitive is what a protocol
        // ordinarily looks like, not a defect. What the word is still entitled
        // to refuse is a *plain* access: one that carries no ordering and no
        // atomicity, and so takes part in no agreement at all.
        //
        // A plain write that the summary-only path compacted never reaches
        // here, so the address does not learn about it. The protocol error is
        // still reported -- such a write is not in the word's history, so a
        // wait it releases exits on a value nothing explains and comes out as
        // `declared_word_wait_unexplained` -- but the bypass is not attributed
        // to the access. Attributing it would mean routing every plain global
        // write through this recorder, which is the dense-output fast path.
        if let Some((record, clock)) = plain {
            note_bypass_event(
                &mut entry.bypassing_ops,
                (
                    record.actor(),
                    record.frontier_key(),
                    record.frontier_epoch,
                    operation.clone(),
                ),
            );
            // Waits that already ran are judged here, because none of them will
            // come back for this access. The other order -- this access first,
            // the wait after -- is judged in `claim_protocol_word`.
            if entry.bypass.is_none() {
                if let Some((waiter, wait_operation, _)) =
                    unordered_wait_under(&entry.declared_ops, actor, clock)
                {
                    entry.bypass = Some(DeclaredWordBypassDiagnostic {
                        declared_operation: wait_operation.clone(),
                        bypassing_operation: operation.clone(),
                        declared_warp_id: waiter.global_warp_id,
                        declared_lane: waiter.lane,
                        bypassing_warp_id: actor.global_warp_id,
                        bypassing_lane: actor.lane,
                        overlap: span,
                    });
                }
            }
        }
        // Asked once per actor, at its first contact only: from the second
        // access on, the clock carries whatever edge this word itself handed
        // out, and a poller would talk itself into being ordered by the very
        // word in question. At first contact nothing of the sort is in it, so
        // what it already covers is an order the word did not supply.
        if writes {
            ProtocolWordUse::note(&mut entry.writers, actor, operation);
        }
        if reads_only {
            ProtocolWordUse::note(&mut entry.readers, actor, operation);
        }
    }

    /// Note that a read took delivery of a write its reader did not hold.
    ///
    /// Asked before the merge, which is the only moment the answer is about
    /// anything but the word itself: afterwards the clock carries the edge
    /// this very read handed out, and every later poll would report that it
    /// was already ordered -- by the word it is polling.
    fn note_word_delivery(&self, span: PhysicalByteSpan, delivered: bool) {
        if !delivered {
            return;
        }
        let mut uses = self
            .protocol_word_use
            .lock()
            .expect("global racecheck protocol-word lock was poisoned");
        let entry = uses
            .entry((span.allocation(), span.byte_offset()))
            .or_default();
        entry.span.get_or_insert(span);
        entry.delivered_order = true;
    }

    /// Remember a strong global write no wait could have polled.
    ///
    /// Recorded whether or not any word is declared yet: a wait can declare its
    /// word long after the write ran, and the claim checks back here.
    fn note_unrecorded_strong_write(&self, span: PhysicalByteSpan, operation: &DynamicOpId) {
        let mut dropped = false;
        {
            let mut spans = self
                .unrecorded_strong_writes
                .lock()
                .expect("global racecheck unrecorded-write lock was poisoned");
            let key = (span.allocation(), span.byte_offset(), span.byte_len());
            if spans.len() >= RACECHECK_MAX_UNRECORDED_STRONG_WRITES && !spans.contains_key(&key) {
                // Past the cap the set stops being the whole set, so a later
                // claim could look clear when it is not. Report rather than
                // drop it silently.
                dropped = true;
            } else {
                spans.entry(key).or_insert_with(|| operation.clone());
            }
        }
        // The write may land on a word that was declared earlier in the run;
        // the claim below only covers the other order.
        let already_declared = {
            let uses = self
                .protocol_word_use
                .lock()
                .expect("global racecheck protocol-word lock was poisoned");
            uses.values()
                .any(|use_| use_.declared && use_.span.is_some_and(|word| word.overlaps(span)))
        };
        if already_declared || dropped {
            self.push_incomplete(RaceCheckIncompleteReason::DeclaredWordWriteUnrecorded {
                operation: operation.clone(),
            });
        }
    }

    /// Whether any unpollable-width strong write landed on this word.
    fn unrecorded_write_on(&self, span: PhysicalByteSpan) -> Option<DynamicOpId> {
        self.unrecorded_strong_writes
            .lock()
            .expect("global racecheck unrecorded-write lock was poisoned")
            .iter()
            .find(|((allocation, byte_offset, byte_len), _)| {
                PhysicalByteSpan::new(*allocation, *byte_offset, *byte_len)
                    .is_ok_and(|write| write.overlaps(span))
            })
            .map(|(_, operation)| operation.clone())
    }

    /// Note that a primitive claimed this word without accessing it.
    ///
    /// A wait performs no adjudicated access: the polling is the engine's, not
    /// the memory model's. The claim on the address still has to be made, or a
    /// word whose only primitive use is a wait looks unclaimed and a raw write
    /// to it goes unreported.
    fn claim_protocol_word(
        &self,
        span: PhysicalByteSpan,
        actor: GlobalActor,
        operation: &DynamicOpId,
        clock: &SparseLaneClock,
    ) {
        let mut uses = self
            .protocol_word_use
            .lock()
            .expect("global racecheck protocol-word lock was poisoned");
        let entry = uses
            .entry((span.allocation(), span.byte_offset()))
            .or_default();
        entry.span.get_or_insert(span);
        // Asked the first time this word is claimed, not on every wait: the
        // question is about the word, and a spin claims it once per iteration.
        let first_claim = !entry.declared;
        entry.declared = true;
        note_wait_event(
            &mut entry.declared_ops,
            (actor, operation.clone(), clock.actor_epoch(actor)),
        );
        // The clock is the waiter's as of entry, before the word's own acquire
        // edge is merged, so what it covers is order the program already had
        // rather than order this wait is about to take from the word.
        if entry.bypass.is_none() {
            if let Some((plain_actor, _, _, plain_operation)) =
                unordered_plain_under(&entry.bypassing_ops, actor, clock)
            {
                entry.bypass = Some(DeclaredWordBypassDiagnostic {
                    declared_operation: operation.clone(),
                    bypassing_operation: plain_operation.clone(),
                    declared_warp_id: actor.global_warp_id,
                    declared_lane: actor.lane,
                    bypassing_warp_id: plain_actor.global_warp_id,
                    bypassing_lane: plain_actor.lane,
                    overlap: span,
                });
            }
        }
        drop(uses);
        if first_claim {
            if let Some(write) = self.unrecorded_write_on(span) {
                self.push_incomplete(RaceCheckIncompleteReason::DeclaredWordWriteUnrecorded {
                    operation: write,
                });
            }
        }
    }

    /// Raw accesses that reached a word a primitive claims (design 2.5).
    ///
    /// Read off the address rather than off a conflicting pair, so it holds
    /// when the primitive's only use of the word is a wait.
    fn claimed_word_bypasses(&self) -> Vec<DeclaredWordBypassDiagnostic> {
        self.protocol_word_use
            .lock()
            .expect("global racecheck protocol-word lock was poisoned")
            .iter()
            .filter_map(|(_key, use_)| use_.bypass.clone())
            .collect()
    }

    /// The words two actors use as a protocol that no primitive declares.
    pub(crate) fn undeclared_protocol_words(&self) -> Vec<UndeclaredProtocolWordDiagnostic> {
        self.protocol_word_use
            .lock()
            .expect("global racecheck protocol-word lock was poisoned")
            .iter()
            .filter_map(|(_key, use_)| {
                if use_.declared || !use_.delivered_order {
                    return None;
                }
                let (writer, reader) = use_.protocol_pair()?;
                Some(UndeclaredProtocolWordDiagnostic {
                    overlap: use_.span?,
                    write_operation: writer.1.clone(),
                    peer_operation: reader.1.clone(),
                    writer_warp_id: writer.0.global_warp_id,
                    writer_lane: writer.0.lane,
                    peer_warp_id: reader.0.global_warp_id,
                    peer_lane: reader.0.lane,
                })
            })
            .collect()
    }

    fn push_incomplete(&self, reason: RaceCheckIncompleteReason) {
        let mut reasons = self
            .incomplete_reasons
            .lock()
            .expect("global racecheck incomplete lock was poisoned");
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    }

    /// Reserve `count` consecutive version identities; returns the first.
    fn allocate_versions(&self, count: u64) -> Result<u64, String> {
        let id = self.next_version.fetch_add(count, AtomicOrdering::Relaxed);
        id.checked_add(count)
            .map(|_| id)
            .ok_or_else(|| "global racecheck version identity overflow".to_string())
    }

    fn track_allocations(&self, allocations: impl IntoIterator<Item = PhysicalAllocationId>) {
        let Some(tracked) = &self.tracked_allocations else {
            return;
        };
        let mut tracked = tracked
            .write()
            .expect("global racecheck tracked-allocation lock was poisoned");
        let before = tracked.len();
        tracked.extend(allocations);
        if tracked.len() != before {
            self.tracked_generation
                .fetch_add(1, AtomicOrdering::Release);
        }
    }

    fn tracks(&self, allocation: PhysicalAllocationId) -> bool {
        let Some(tracked) = self.tracked_allocations.as_ref() else {
            return true;
        };
        thread_local! {
            // (shared identity, generation, tracked set) as last seen by this
            // thread.
            static CACHE: std::cell::RefCell<(usize, u64, BTreeSet<PhysicalAllocationId>)> =
                const { std::cell::RefCell::new((0, u64::MAX, BTreeSet::new())) };
        }
        let identity = self as *const Self as usize;
        let generation = self.tracked_generation.load(AtomicOrdering::Acquire);
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.0 == identity && cache.1 == generation && cache.2.contains(&allocation) {
                return true;
            }
            if cache.0 != identity || cache.1 != generation {
                let current = tracked
                    .read()
                    .expect("global racecheck tracked-allocation lock was poisoned");
                *cache = (identity, generation, current.clone());
            }
            cache.2.contains(&allocation)
        })
    }

    /// The byte-state cells covering `spans`, creating missing ones when
    /// `create` is set. Lock them through [`LockedShadows::lock`]; the sorted
    /// key order keeps lock acquisition consistent across shards.
    fn shadows(
        &self,
        spans: impl IntoIterator<Item = (PhysicalAccessSpace, PhysicalByteSpan)>,
        create: bool,
    ) -> Vec<(ShadowCellKey, Arc<ShadowCell>)> {
        let mut keys = spans
            .into_iter()
            .flat_map(shadow_cell_keys)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        self.shadows_for_keys(keys, create)
    }

    /// The cells of `keys`, which are sorted and distinct.
    fn shadows_for_keys(
        &self,
        keys: Vec<ShadowCellKey>,
        create: bool,
    ) -> Vec<(ShadowCellKey, Arc<ShadowCell>)> {
        let mut cells = Vec::with_capacity(keys.len());
        let mut missing = Vec::new();
        {
            let bytes = self
                .bytes
                .read()
                .expect("global racecheck byte-state lock was poisoned");
            for &key in keys.iter() {
                match bytes.get(&key) {
                    Some(cell) => cells.push((key, Arc::clone(cell))),
                    None => missing.push(key),
                }
            }
        }
        if create && !missing.is_empty() {
            let mut bytes = self
                .bytes
                .write()
                .expect("global racecheck byte-state lock was poisoned");
            for key in missing {
                let cell = Arc::clone(bytes.entry(key).or_default());
                cells.push((key, cell));
            }
            cells.sort_unstable_by_key(|(key, _)| *key);
        }
        cells
    }

    #[cfg(test)]
    fn shadow(&self, allocation: PhysicalAllocationId) -> Option<GlobalAllocationShadow> {
        // Tests look at small allocations that live in their first stripe.
        self.bytes
            .read()
            .expect("global racecheck byte-state lock was poisoned")
            .get(&((PhysicalAccessSpace::Global, allocation), 0))
            .map(|cell| {
                let mut state = {
                    cell.state
                        .write()
                        .expect("global racecheck allocation lock was poisoned")
                };
                cell.drain_pending(0, &mut state);
                state.clone()
            })
    }

    #[cfg(test)]
    fn tracked_byte_count(&self) -> usize {
        self.bytes
            .read()
            .expect("global racecheck byte-state lock was poisoned")
            .iter()
            .map(|(key, cell)| {
                let mut state = {
                    cell.state
                        .write()
                        .expect("global racecheck allocation lock was poisoned")
                };
                cell.drain_pending(key.1, &mut state);
                state.tracked_byte_count()
            })
            .sum()
    }
}

enum ShadowCellGuard<'a> {
    Read(RwLockReadGuard<'a, GlobalAllocationShadow>),
    Write(RwLockWriteGuard<'a, GlobalAllocationShadow>),
}

impl std::ops::Deref for ShadowCellGuard<'_> {
    type Target = GlobalAllocationShadow;

    fn deref(&self) -> &GlobalAllocationShadow {
        match self {
            Self::Read(guard) => guard,
            Self::Write(guard) => guard,
        }
    }
}

impl ShadowCellGuard<'_> {
    fn get_mut(&mut self) -> &mut GlobalAllocationShadow {
        match self {
            Self::Read(_) => unreachable!("a deferred-read shadow guard is never mutated"),
            Self::Write(guard) => guard,
        }
    }
}

/// The locked byte-state cells of one batch's spans.
struct LockedShadows<'a> {
    cells: Vec<(ShadowCellKey, &'a ShadowCell, ShadowCellGuard<'a>)>,
    // A deferred batch (weak reads only) holds the read side of every cell
    // and publishes its frontier updates through the pending logs.
    deferred: bool,
}

impl<'a> LockedShadows<'a> {
    /// Exclusive access. Each cell's pending weak reads are drained on
    /// acquisition, so the shadow reflects every prior access.
    fn lock(cells: &'a [(ShadowCellKey, Arc<ShadowCell>)]) -> Self {
        Self {
            cells: cells
                .iter()
                .map(|(key, cell)| {
                    let cell = Arc::as_ref(cell);
                    let mut guard = {
                        cell.state
                            .write()
                            .expect("global racecheck allocation lock was poisoned")
                    };
                    cell.drain_pending(key.1, &mut guard);
                    (*key, cell, ShadowCellGuard::Write(guard))
                })
                .collect(),
            deferred: false,
        }
    }

    /// Shared access for a batch of weak reads: validation reads the shadow
    /// concurrently and the frontier updates go to the pending logs. A log
    /// at capacity is drained first so it stays bounded.
    fn lock_deferred(cells: &'a [(ShadowCellKey, Arc<ShadowCell>)]) -> Self {
        Self {
            cells: cells
                .iter()
                .map(|(key, cell)| {
                    let cell = Arc::as_ref(cell);
                    if cell.pending_len() >= PENDING_READS_CAP {
                        let mut guard = {
                            cell.state
                                .write()
                                .expect("global racecheck allocation lock was poisoned")
                        };
                        cell.drain_pending(key.1, &mut guard);
                    }
                    let guard = cell
                        .state
                        .read()
                        .expect("global racecheck allocation lock was poisoned");
                    (*key, cell, ShadowCellGuard::Read(guard))
                })
                .collect(),
            deferred: true,
        }
    }

    fn position(&self, key: ShadowCellKey) -> Option<usize> {
        self.cells.binary_search_by_key(&key, |(id, _, _)| *id).ok()
    }

    /// Visit the locked cells `span` overlaps, in byte order, with the part
    /// of the span inside each cell.
    fn for_each_overlapping(
        &self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        mut visit: impl FnMut(&GlobalAllocationShadow, usize, usize),
    ) {
        for key in shadow_cell_keys((space, span)) {
            if let Some(position) = self.position(key) {
                let (lo, hi) = clip_to_stripe(span, key.1);
                visit(&self.cells[position].2, lo, hi);
            }
        }
    }

    fn for_each_overlapping_mut(
        &mut self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        mut visit: impl FnMut(&mut GlobalAllocationShadow, usize, usize),
    ) {
        for key in shadow_cell_keys((space, span)) {
            if let Some(position) = self.position(key) {
                let (lo, hi) = clip_to_stripe(span, key.1);
                visit(self.cells[position].2.get_mut(), lo, hi);
            }
        }
    }

    /// Record one prepared access: append a deferred batch's weak read to
    /// the pending log of every cell its span overlaps, or apply it to the
    /// exclusively held shadow.
    fn publish(&mut self, prepared: &PreparedGlobalAccess) {
        let span = prepared.access.span();
        if !self.deferred {
            self.for_each_overlapping_mut(prepared.access.space(), span, |shadow, lo, hi| {
                shadow.apply(prepared, lo, hi);
            });
            return;
        }
        debug_assert!(prepared_defers(prepared));
        // The pending log outlives the batch: enqueue the clock in its
        // frozen, structurally shared form instead of a private update
        // vector (tens of KB per undrained read otherwise).
        let mut frozen = prepared.clone();
        frozen.access.clock.freeze_component_updates();
        for key in shadow_cell_keys((prepared.access.space(), span)) {
            if let Some(position) = self.position(key) {
                self.cells[position]
                    .1
                    .pending_reads
                    .lock()
                    .expect("global racecheck pending-read lock was poisoned")
                    .push(frozen.clone());
            }
        }
    }
}

/// Whether an access may publish its reader-frontier update through a cell's
/// pending log: only weak, non-atomic pure reads qualify. Everything else
/// observes or mutates the exact shadow state, so it takes the write side and
/// drains the log first.
impl LockedShadows<'_> {
    /// Append a batch of deferred reads to the pending logs, locking each
    /// cell's log once: the lanes of one batch almost always share a cell.
    fn publish_pending(&self, accesses: &[PreparedGlobalAccess]) {
        debug_assert!(self.deferred);
        // See `publish`: the logs hold the accesses until a writer drains
        // them, so the enqueued clocks are frozen once per access here and
        // the per-cell clones below share their storage.
        let frozen = accesses
            .iter()
            .map(|prepared| {
                debug_assert!(prepared_defers(prepared));
                let mut prepared = prepared.clone();
                prepared.access.clock.freeze_component_updates();
                prepared
            })
            .collect::<Vec<_>>();
        let mut targets = Vec::with_capacity(frozen.len());
        for prepared in &frozen {
            for key in shadow_cell_keys((prepared.access.space(), prepared.access.span())) {
                if let Some(position) = self.position(key) {
                    targets.push((position, prepared));
                }
            }
        }
        targets.sort_by_key(|(position, _)| *position);
        let mut targets = targets.into_iter().peekable();
        while let Some((position, prepared)) = targets.next() {
            let mut pending = self.cells[position]
                .1
                .pending_reads
                .lock()
                .expect("global racecheck pending-read lock was poisoned");
            pending.push(prepared.clone());
            while let Some((_, prepared)) = targets.next_if(|(next, _)| *next == position) {
                pending.push(prepared.clone());
            }
        }
    }
}

fn weak_read_defers(kind: PhysicalAccessKind, semantics: MemoryAccessSemantics) -> bool {
    kind.reads() && !kind.writes() && !semantics.order().has_release()
}

fn prepared_defers(prepared: &PreparedGlobalAccess) -> bool {
    prepared.new_version.is_none()
        && weak_read_defers(prepared.access.kind(), prepared.access.semantics)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GlobalRaceState {
    topology: Option<LaunchTopology>,
    shared: Arc<GlobalRaceShared>,
    actors: GlobalActorStates,
    staged: BTreeMap<DynamicOpId, StagedGlobalBatch>,
    async_tokens: BTreeMap<AsyncTokenId, GlobalAsyncTokenState>,
    cp_async_completed: BTreeMap<(usize, usize), GlobalAsyncPublication>,
    physical_barriers: BTreeMap<(PhysicalBarrierId, u64), GlobalAsyncPublication>,
    physical_copy_barriers: BTreeMap<(PhysicalBarrierId, u64), GlobalAsyncPublication>,
    named_barriers: BTreeMap<(crate::NamedBarrierId, u64), GlobalExecutionPayload>,
    cluster_barriers: BTreeMap<(crate::ClusterBarrierId, u64), GlobalExecutionPayload>,
    /// First generation still retained per barrier (see
    /// `RETAINED_BARRIER_GENERATIONS`); an acquire below it is an analysis gap.
    physical_barrier_floors: BTreeMap<PhysicalBarrierId, (u64, Option<u64>)>,
    named_barrier_floors: BTreeMap<crate::NamedBarrierId, u64>,
    cluster_barrier_floors: BTreeMap<crate::ClusterBarrierId, u64>,
    #[cfg(any(test, feature = "profile"))]
    replay_events: Vec<GlobalReplayEvent>,
    #[cfg(any(test, feature = "profile"))]
    replay_batch_count: usize,
    #[cfg(any(test, feature = "profile"))]
    replay_clock_snapshots: Vec<Arc<SparseLaneClock>>,
    #[cfg(any(test, feature = "profile"))]
    replay_clock_snapshot_index: BTreeMap<ReplayClockKey, u32>,
    #[cfg(any(test, feature = "profile"))]
    replay_access_count: usize,
    #[cfg(any(test, feature = "profile"))]
    replay_overflowed: bool,
    #[cfg(any(test, feature = "profile"))]
    replay_recording: bool,
}

pub(crate) type GlobalClockFrontier = SharedLaneFrontiers;

#[cfg(test)]
const MAX_GLOBAL_REPLAY_BATCHES: usize = 4096;
#[cfg(test)]
const MAX_GLOBAL_REPLAY_ACCESSES: usize = 16_384;
#[cfg(test)]
const MAX_GLOBAL_REPLAY_EVENTS: usize = 32_768;
#[cfg(all(feature = "profile", not(test)))]
const MAX_GLOBAL_REPLAY_BATCHES: usize = 1_000_000;
#[cfg(all(feature = "profile", not(test)))]
const MAX_GLOBAL_REPLAY_ACCESSES: usize = 2_000_000;
#[cfg(all(feature = "profile", not(test)))]
const MAX_GLOBAL_REPLAY_EVENTS: usize = 2_000_000;

impl GlobalRaceState {
    pub(crate) fn new(topology: Option<LaunchTopology>) -> Self {
        Self::with_tracked_allocations(topology, None)
    }

    pub(crate) fn with_tracked_allocations(
        topology: Option<LaunchTopology>,
        tracked_allocations: Option<BTreeSet<PhysicalAllocationId>>,
    ) -> Self {
        Self::with_shared(
            topology,
            Arc::new(GlobalRaceShared::new(tracked_allocations)),
        )
    }

    /// One shard's state over `shared`; every shard of a launch shares one
    /// `GlobalRaceShared` and one async-clock registry.
    pub(crate) fn with_shared(
        topology: Option<LaunchTopology>,
        shared: Arc<GlobalRaceShared>,
    ) -> Self {
        let mut state = Self {
            topology,
            actors: topology
                .map(|topology| {
                    GlobalActorStates::with_topology(topology, Arc::clone(&shared.async_registry))
                })
                .unwrap_or_else(|| {
                    GlobalActorStates::with_registry(Arc::clone(&shared.async_registry))
                }),
            shared,
            ..Self::default()
        };
        #[cfg(any(test, feature = "profile"))]
        {
            state.replay_recording =
                cfg!(test) || std::env::var_os("NUMSIM_REPLAY_METRICS").is_some();
        }
        state
    }

    /// One shard's state for the warps `[base_warp, base_warp + warp_count)`.
    pub(crate) fn for_warp_range(
        topology: Option<LaunchTopology>,
        base_warp: usize,
        warp_count: usize,
        shared: Arc<GlobalRaceShared>,
    ) -> Self {
        let mut state = Self::with_shared(topology, Arc::clone(&shared));
        state.actors = GlobalActorStates::with_warp_range(
            base_warp,
            warp_count,
            Arc::clone(&shared.async_registry),
        );
        state
    }

    pub(crate) fn findings(&self) -> impl Iterator<Item = PhysicalRaceFinding> {
        self.shared.findings().into_iter()
    }

    pub(crate) fn declared_word_bypasses(
        &self,
    ) -> impl Iterator<Item = DeclaredWordBypassDiagnostic> {
        self.shared.declared_word_bypasses().into_iter()
    }

    pub(crate) fn scope_diagnostics(&self) -> impl Iterator<Item = GlobalScopeMismatchDiagnostic> {
        self.shared.scope_diagnostics().into_iter()
    }

    pub(crate) fn incomplete_reasons(&self) -> impl Iterator<Item = RaceCheckIncompleteReason> {
        self.shared.incomplete_reasons().into_iter()
    }

    #[cfg(any(test, feature = "profile"))]
    fn replay_analysis(&self) -> Option<GlobalReplaySummary> {
        if self.replay_overflowed {
            return None;
        }
        let tracked_allocations = self.shared.tracked_allocations.as_ref().map(|tracked| {
            tracked
                .read()
                .expect("global racecheck tracked-allocation lock was poisoned")
                .clone()
        });
        let replay_shared = Arc::new(GlobalRaceShared {
            async_registry: Arc::clone(&self.shared.async_registry),
            ..GlobalRaceShared::new(tracked_allocations)
        });
        let mut replay = Self::with_shared(self.topology, replay_shared);
        replay.replay_recording = false;
        let mut async_leases = BTreeMap::new();
        for event in &self.replay_events {
            match event {
                GlobalReplayEvent::AccessBatch(batch) => {
                    let prepared = batch
                        .accesses
                        .iter()
                        .cloned()
                        .map(|access| {
                            access.into_prepared_access(
                                &self.replay_clock_snapshots,
                                &mut async_leases,
                            )
                        })
                        .collect::<Vec<_>>();
                    replay.validate_and_commit_prepared(prepared);
                }
                _ => replay.replay_event(event, &self.replay_clock_snapshots),
            }
        }
        Some(GlobalReplaySummary {
            findings: replay.shared.findings().into_iter().collect(),
            scope_diagnostics: replay.shared.scope_diagnostics().into_iter().collect(),
            declared_word_bypasses: replay
                .shared
                .declared_word_bypasses()
                .into_iter()
                .collect(),
            incomplete_reasons: replay.shared.incomplete_reasons(),
            access_count: self.replay_access_count,
            batch_count: self.replay_batch_count,
        })
    }

    #[cfg(any(test, feature = "profile"))]
    fn replay_event(&mut self, event: &GlobalReplayEvent, snapshots: &[Arc<SparseLaneClock>]) {
        match event {
            GlobalReplayEvent::AccessBatch(_) => unreachable!("access batches are replayed inline"),
            GlobalReplayEvent::AsyncLifecycle {
                token, state: None, ..
            } => {
                let _ = token;
            }
            GlobalReplayEvent::AsyncLifecycle {
                token,
                phase,
                state: Some(state),
                ..
            } => {
                match phase {
                    ReplayAsyncPhase::Issue => {}
                    ReplayAsyncPhase::Complete(AsyncGroupMilestone::SourceReadComplete) => {
                        debug_assert!(state.source_read_snapshot.is_some());
                    }
                    ReplayAsyncPhase::Complete(AsyncGroupMilestone::FullComplete) => {
                        debug_assert!(state.full_snapshot.is_some());
                    }
                }
                let lease = Arc::new(AsyncClockLease {
                    token: token.clone(),
                    handle: state.handle,
                    registry: std::sync::Weak::new(),
                    issue_epoch: state.issue_epoch,
                });
                let current = snapshots
                    .get(state.current_snapshot as usize)
                    .expect("async replay state must reference a clock snapshot")
                    .as_ref()
                    .clone();
                let source_read = state.source_read_snapshot.map(|snapshot| {
                    snapshots
                        .get(snapshot as usize)
                        .expect("async source-read state must reference a clock snapshot")
                        .as_ref()
                        .clone()
                });
                let full = state.full_snapshot.map(|snapshot| {
                    snapshots
                        .get(snapshot as usize)
                        .expect("async full state must reference a clock snapshot")
                        .as_ref()
                        .clone()
                });
                self.async_tokens.insert(
                    token.clone(),
                    GlobalAsyncTokenState {
                        lease,
                        current,
                        source_read,
                        full,
                        // Replay consumes prepared memory batches (including
                        // their release payload), not live completion planning.
                        release_fence: None,
                        read_observations: BTreeMap::new(),
                        tcgen_publication: TcgenFenceFrontier::default(),
                        shared_frontier: None,
                    },
                );
            }
            GlobalReplayEvent::AsyncRetire { token } => {
                self.async_tokens.remove(token);
            }
            GlobalReplayEvent::AsyncAcquire {
                global_warp_id,
                mask,
                clock_snapshot,
                read_observations,
            } => {
                let Some(snapshot) = clock_snapshot else {
                    return;
                };
                let clock = snapshots
                    .get(*snapshot as usize)
                    .expect("async acquire must reference a clock snapshot");
                self.acquire_mask(
                    *global_warp_id,
                    *mask,
                    &GlobalExecutionPayload {
                        clock: clock.as_ref().clone(),
                        read_observations: read_observations.clone(),
                    },
                )
                .expect("replaying async acquire must succeed");
            }
            GlobalReplayEvent::AsyncPublishPhysical {
                barrier_id,
                generation,
                clock_snapshot,
                read_observations,
            } => {
                let Some(snapshot) = clock_snapshot else {
                    return;
                };
                let clock = snapshots
                    .get(*snapshot as usize)
                    .expect("async publish must reference a clock snapshot")
                    .as_ref()
                    .clone();
                Self::merge_physical_barrier_payload(
                    &mut self.physical_copy_barriers,
                    (*barrier_id, *generation),
                    GlobalAsyncPublication {
                        clock,
                        read_observations: read_observations.clone(),
                    },
                );
            }
            GlobalReplayEvent::ProxyFence {
                operation,
                kind,
                mask,
                effect,
            } => {
                let operation = OperationContext::new(operation.clone(), *kind, *mask);
                self.proxy_async_fence(&operation, *effect)
                    .expect("replaying proxy fence must succeed");
            }
            GlobalReplayEvent::Fence {
                operation,
                kind,
                mask,
                effect,
            } => {
                let operation = OperationContext::new(operation.clone(), *kind, *mask);
                self.fence(&operation, *effect, &TcgenLaneFrontiers::new(), None)
                    .expect("replaying fence must succeed");
            }
            GlobalReplayEvent::TensorMap {
                operation,
                kind,
                mask,
                observation,
            } => {
                let operation = OperationContext::new(operation.clone(), *kind, *mask);
                self.tensor_map_observation(&operation, *observation)
                    .expect("replaying a successful TensorMap observation must succeed");
            }
            GlobalReplayEvent::WarpSync {
                global_warp_id,
                mask,
            } => self
                .warp_sync(*global_warp_id, *mask)
                .expect("replaying warp sync must succeed"),
            GlobalReplayEvent::ResetPhysicalBarriers { barrier_ids } => {
                self.reset_physical_barriers(barrier_ids);
            }
            GlobalReplayEvent::RetainPhysicalBarrierGenerations {
                barrier_id,
                generation,
                conditional,
            } => {
                self.retain_physical_barrier_generations(*barrier_id, *generation, *conditional);
            }
            GlobalReplayEvent::PhysicalBarrierRelease {
                barrier_id,
                generation,
                global_warp_id,
                mask,
            } => self
                .physical_barrier_release(*barrier_id, *generation, *global_warp_id, *mask)
                .expect("replaying physical barrier release must succeed"),
            GlobalReplayEvent::PhysicalBarrierAcquire {
                barrier_id,
                generation,
                global_warp_id,
                mask,
                acquire,
            } => self
                .physical_barrier_acquire(
                    None,
                    *barrier_id,
                    *generation,
                    *global_warp_id,
                    *mask,
                    *acquire,
                )
                .expect("replaying physical barrier acquire must succeed"),
            GlobalReplayEvent::NamedBarrierRelease {
                barrier_id,
                generation,
                global_warp_id,
                mask,
            } => self
                .named_barrier_release(*barrier_id, *generation, *global_warp_id, *mask)
                .expect("replaying named barrier release must succeed"),
            GlobalReplayEvent::NamedBarrierAcquire {
                barrier_id,
                generation,
                global_warp_id,
                mask,
            } => self
                .named_barrier_acquire(None, *barrier_id, *generation, *global_warp_id, *mask)
                .expect("replaying named barrier acquire must succeed"),
            GlobalReplayEvent::ClusterBarrierRelease {
                barrier_id,
                generation,
                global_warp_id,
                mask,
                publishes_memory,
            } => self
                .cluster_barrier_release(
                    *barrier_id,
                    *generation,
                    *global_warp_id,
                    *mask,
                    *publishes_memory,
                )
                .expect("replaying cluster barrier release must succeed"),
            GlobalReplayEvent::ClusterBarrierAcquire {
                barrier_id,
                generation,
                global_warp_id,
                mask,
            } => self
                .cluster_barrier_acquire(None, *barrier_id, *generation, *global_warp_id, *mask)
                .expect("replaying cluster barrier acquire must succeed"),
        }
    }

    #[cfg(any(test, feature = "profile"))]
    fn replay_storage_stats(&self) -> GlobalReplayStorageStats {
        GlobalReplayStorageStats {
            access_count: self.replay_access_count,
            batch_count: self.replay_batch_count,
            snapshot_count: self.replay_clock_snapshots.len(),
            access_record_bytes: std::mem::size_of::<GlobalReplayAccess>(),
            snapshot_record_bytes: std::mem::size_of::<SparseLaneClock>(),
            event_count: self.replay_events.len(),
            event_record_bytes: std::mem::size_of::<GlobalReplayEvent>(),
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn replay_diagnostic_line(&self) -> String {
        let started = Instant::now();
        let replay = self.replay_analysis();
        let replay_wall_us = started.elapsed().as_micros();
        let storage = self.replay_storage_stats();
        let findings_equal = replay.as_ref().is_some_and(|summary| {
            summary.findings == self.shared.findings().into_iter().collect::<BTreeSet<_>>()
        });
        let scope_equal = replay.as_ref().is_some_and(|summary| {
            summary.scope_diagnostics
                == self
                    .shared
                    .scope_diagnostics()
                    .into_iter()
                    .collect::<BTreeSet<_>>()
        });
        let declared_equal = replay.as_ref().is_some_and(|summary| {
            summary.declared_word_bypasses
                == self
                    .shared
                    .declared_word_bypasses()
                    .into_iter()
                    .collect::<BTreeSet<_>>()
        });
        let incomplete_equal = replay
            .as_ref()
            .is_some_and(|summary| summary.incomplete_reasons == self.shared.incomplete_reasons());
        let replay_exact = replay.as_ref().is_some_and(|summary| {
            findings_equal
                && scope_equal
                && declared_equal
                && incomplete_equal
                && summary.access_count == self.replay_access_count
                && summary.batch_count == self.replay_batch_count
        });
        format!(
            "{{\"access_count\":{},\"batch_count\":{},\"snapshot_count\":{},\"event_count\":{},\"access_record_bytes\":{},\"snapshot_record_bytes\":{},\"event_record_bytes\":{},\"overflowed\":{},\"replay_available\":{},\"replay_exact\":{},\"findings_equal\":{},\"scope_equal\":{},\"incomplete_equal\":{},\"eager_findings\":{},\"replay_findings\":{},\"eager_scope\":{},\"replay_scope\":{},\"eager_incomplete\":{},\"replay_incomplete\":{},\"replay_wall_us\":{}}}",
            storage.access_count,
            storage.batch_count,
            storage.snapshot_count,
            storage.event_count,
            storage.access_record_bytes,
            storage.snapshot_record_bytes,
            storage.event_record_bytes,
            self.replay_overflowed,
            replay.is_some(),
            replay_exact,
            findings_equal,
            scope_equal,
            incomplete_equal,
            self.shared.findings().len(),
            replay.as_ref().map_or(0, |summary| summary.findings.len()),
            self.shared.scope_diagnostics().len(),
            replay
                .as_ref()
                .map_or(0, |summary| summary.scope_diagnostics.len()),
            self.shared.incomplete_reasons().len(),
            replay
                .as_ref()
                .map_or(0, |summary| summary.incomplete_reasons.len()),
            replay_wall_us,
        )
    }

    #[cfg(any(test, feature = "profile"))]
    fn intern_replay_clock_snapshot(&mut self, clock: &SparseLaneClock) -> u32 {
        let key = clock.replay_key();
        if let Some(&index) = self.replay_clock_snapshot_index.get(&key) {
            debug_assert!(
                self.replay_clock_snapshots[index as usize].shares_representation_with(clock)
            );
            return u32::try_from(index).expect("replay snapshot index must fit u32");
        }
        let index = self.replay_clock_snapshots.len();
        self.replay_clock_snapshots.push(Arc::new(clock.clone()));
        let index = u32::try_from(index).expect("replay snapshot index must fit u32");
        self.replay_clock_snapshot_index.insert(key, index);
        index
    }

    #[cfg(any(test, feature = "profile"))]
    fn record_replay_event(&mut self, event: GlobalReplayEvent) {
        if !self.replay_recording {
            return;
        }
        if self.replay_events.len() >= MAX_GLOBAL_REPLAY_EVENTS {
            self.replay_overflowed = true;
            return;
        }
        self.replay_events.push(event);
    }

    #[cfg(any(test, feature = "profile"))]
    fn record_replay_async_lifecycle(&mut self, phase: ReplayAsyncPhase, token: &AsyncTokenId) {
        let captured = self.async_tokens.get(token).map(|state| {
            (
                state.lease.handle,
                state.lease.issue_epoch,
                state.current.clone(),
                state.source_read.clone(),
                state.full.clone(),
            )
        });
        let state =
            captured.map(
                |(handle, issue_epoch, current, source_read, full)| ReplayAsyncState {
                    handle,
                    issue_epoch,
                    current_snapshot: self.intern_replay_clock_snapshot(&current),
                    source_read_snapshot: source_read
                        .as_ref()
                        .map(|clock| self.intern_replay_clock_snapshot(clock)),
                    full_snapshot: full
                        .as_ref()
                        .map(|clock| self.intern_replay_clock_snapshot(clock)),
                },
            );
        self.record_replay_event(GlobalReplayEvent::AsyncLifecycle {
            phase,
            token: token.clone(),
            state,
        });
    }

    #[cfg(any(test, feature = "profile"))]
    fn record_replay_async_acquire(
        &mut self,
        global_warp_id: usize,
        mask: WarpMask,
        payload: Option<&GlobalExecutionPayload>,
    ) {
        let clock_snapshot =
            payload.map(|payload| self.intern_replay_clock_snapshot(&payload.clock));
        self.record_replay_event(GlobalReplayEvent::AsyncAcquire {
            global_warp_id,
            mask,
            clock_snapshot,
            read_observations: payload
                .map(|payload| payload.read_observations.clone())
                .unwrap_or_default(),
        });
    }

    #[cfg(any(test, feature = "profile"))]
    fn record_replay_async_publish(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        payload: Option<&GlobalExecutionPayload>,
    ) {
        let clock_snapshot =
            payload.map(|payload| self.intern_replay_clock_snapshot(&payload.clock));
        self.record_replay_event(GlobalReplayEvent::AsyncPublishPhysical {
            barrier_id,
            generation,
            clock_snapshot,
            read_observations: payload
                .map(|payload| payload.read_observations.clone())
                .unwrap_or_default(),
        });
    }

    #[cfg(any(test, feature = "profile"))]
    fn record_replay_batch(&mut self, accesses: &[PreparedGlobalAccess]) {
        if !self.replay_recording || accesses.is_empty() {
            return;
        }
        let exceeds_batches = self.replay_batch_count >= MAX_GLOBAL_REPLAY_BATCHES;
        let exceeds_accesses =
            self.replay_access_count.saturating_add(accesses.len()) > MAX_GLOBAL_REPLAY_ACCESSES;
        if exceeds_batches || exceeds_accesses {
            self.replay_overflowed = true;
            return;
        }
        self.replay_access_count += accesses.len();
        self.replay_batch_count += 1;
        let replay_accesses = accesses
            .iter()
            .map(|prepared| {
                let clock_snapshot = self.intern_replay_clock_snapshot(&prepared.access.clock);
                GlobalReplayAccess::from_access(
                    &prepared.access,
                    clock_snapshot,
                    prepared.new_version.as_ref(),
                )
            })
            .collect::<Vec<_>>();
        self.record_replay_event(GlobalReplayEvent::AccessBatch(GlobalReplayBatch {
            accesses: replay_accesses.into_boxed_slice(),
        }));
    }

    pub(crate) fn staged_operations(&self) -> impl Iterator<Item = &DynamicOpId> {
        self.staged.keys()
    }

    #[cfg(test)]
    pub(crate) fn retained_state_counts(&self) -> (usize, usize, usize) {
        let pending = self
            .actors
            .values()
            .map(|state| state.pending_acquire.len())
            .sum();
        let bytes = self.shared.tracked_byte_count();
        (self.actors.len(), bytes, pending)
    }

    pub(crate) fn before_batch(
        &mut self,
        batch: &PhysicalAccessBatch,
        tcgen_publications: &TcgenLaneFrontiers,
        shared_frontier: Option<&SharedLaneFrontiers>,
    ) -> Result<(TcgenLaneFrontiers, GlobalClockFrontier), String> {
        let descriptor = batch.descriptor();
        if !descriptor.space().has_read_from_versions() {
            return Ok((TcgenLaneFrontiers::new(), GlobalClockFrontier::default()));
        }
        if descriptor.space() == PhysicalAccessSpace::Shared
            && !direct_write_requires_version(descriptor.memory_semantics())
        {
            return Ok((TcgenLaneFrontiers::new(), GlobalClockFrontier::default()));
        }
        self.register_written_allocations(std::slice::from_ref(batch));
        if self.staged.contains_key(batch.operation().id()) {
            return Err(format!(
                "global racecheck operation {} already has a staged batch",
                batch.operation().id()
            ));
        }
        let semantics = descriptor.memory_semantics();
        if semantics.proxy() == MemoryProxy::Mmio {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: batch.operation().id().clone(),
                kind: "mmio_device_behavior_unmodeled",
                reason: "MMIO device behavior has no exact read-from model".to_string(),
            });
        }
        // The batch's allocations stay locked from the read-from lookup
        // through validation and (for reads) the frontier update, so no other
        // shard's access to the same bytes can slip between them. A batch of
        // weak reads holds only the read side and defers its frontier update
        // to the pending logs.
        let mut actor_updates = BTreeMap::new();
        let mut prepared = Vec::new();
        let mut tcgen_acquisitions = TcgenLaneFrontiers::new();
        let mut acquired_frontier = SharedLaneFrontiers::new();
        // Overlapping active-lane RMW footprints leave the lane serialization
        // order unconstrained, so no lane's read-from version is a fact. That
        // suppresses the exact predecessor below and nothing more: withholding a
        // read-from edge only removes happens-before, which can make the verdict
        // more conservative but never less. Atomicity already excludes a race
        // between these lanes, so the race verdict itself stays decidable and
        // this is not a coverage gap.
        let ambiguous_rmw_read_from = descriptor.kind()
            == PhysicalAccessKind::AtomicReadModifyWrite
            && self.has_overlapping_tracked_lane_spans(batch);
        let mut join_cache = AcquireJoinCache::default();
        // Per-lane setup that needs no byte-state lock runs before the cells
        // are locked; see `before_compact_batch`.
        let lanes = batch
            .lanes()
            .iter()
            .filter_map(|lane_access| {
                let spans = lane_access
                    .footprint()
                    .spans()
                    .iter()
                    .copied()
                    .filter(|span| self.tracks_span(descriptor.space(), *span))
                    .collect::<Vec<_>>();
                if spans.is_empty() {
                    return None;
                }
                let lane = lane_access.provenance().lane();
                let actor = GlobalActor::new(batch.operation().id().global_warp_id(), lane);
                let actor_state = self
                    .actors
                    .get(&actor)
                    .cloned()
                    .unwrap_or_else(|| self.actors.empty_state());
                let tcgen_publication = tcgen_publications.get(&lane).cloned().unwrap_or_default();
                Some((
                    lane_access,
                    lane,
                    actor,
                    actor_state,
                    tcgen_publication,
                    spans,
                ))
            })
            .collect::<Vec<_>>();
        let cells = self.shared.shadows(
            batch
                .lanes()
                .iter()
                .flat_map(|lane| lane.footprint().spans())
                .copied()
                .filter(|span| self.tracks_span(descriptor.space(), *span))
                .map(|span| (descriptor.space(), span)),
            true,
        );
        let mut shadows = if weak_read_defers(descriptor.kind(), semantics) {
            LockedShadows::lock_deferred(&cells)
        } else {
            LockedShadows::lock(&cells)
        };
        let registry = Arc::clone(&self.actors.async_registry);
        let mut laggard_gaps = Vec::new();
        for (lane_access, lane, actor, mut actor_state, tcgen_publication, spans) in lanes {
            if let Some(gap) =
                note_laggard_access(&mut actor_state, &registry, batch.operation().id())
            {
                laggard_gaps.push(gap);
            }
            let mut tcgen_acquisition = TcgenFenceFrontier::default();
            let mut lane_shared_acquisition = SharedClockFrontier::default();

            let predecessors = if descriptor.kind().reads()
                && semantics.class().is_atomic_class()
                && !ambiguous_rmw_read_from
            {
                self.read_versions(
                    &shadows,
                    batch.operation().id(),
                    &spans,
                    descriptor.space(),
                    semantics,
                    &actor_state.clock,
                )
            } else {
                Vec::new()
            };
            // `predecessors` is empty unless this access reads, so the
            // gate is upstream's. A declared wait is not among these: it
            // adjudicates no access and takes its edge in
            // `apply_declared_word_wait`, from the write its predicate
            // accepted -- never from what a late schedule showed it.
            for predecessor in &predecessors {
                self.apply_load_ordering(
                    batch.operation().id(),
                    actor,
                    semantics,
                    descriptor.kind() == PhysicalAccessKind::Read,
                    predecessor,
                    &mut actor_state,
                    &mut tcgen_acquisition,
                    &mut lane_shared_acquisition,
                    &mut join_cache,
                );
            }
            let mut lane_shared_publication = shared_frontier
                .and_then(|all| all.get(&lane))
                .cloned()
                .unwrap_or_default();
            lane_shared_publication.merge(&lane_shared_acquisition);
            actor_state.clock.tick(actor)?;
            // TCGEN before-fence state is carried only by execution-ordering
            // operations.  Attaching it to an ordinary memory event would
            // contaminate the generic happens-before relation when an acquire
            // deliberately splits the specialized TCGEN frontier back out.
            let event_clock = actor_state.clock.clone();

            for span in spans {
                let witness = PhysicalRaceWitness::from_lane(
                    lane_access,
                    descriptor.kind(),
                    descriptor.space(),
                    span,
                );
                let access = GlobalAccess {
                    recorded: Arc::new(RecordedGlobalAccess::new(
    actor,
    GlobalFrontierActor::Lane(actor),
    event_clock.actor_epoch(actor),
    witness,
    semantics,
    0,
)),
                    clock: event_clock.clone(),
                };
                let new_version = (descriptor.kind().writes()
                    && direct_write_requires_version(semantics))
                .then(|| {
                    let payload = self.publication_for_write(
                        batch.operation().id(),
                        &access,
                        &predecessors,
                        actor_state.release_fence.as_ref(),
                        &tcgen_publication,
                        Some(&lane_shared_publication),
                    );
                    let id = self
                        .shared
                        .allocate_versions(1)
                        .expect("global racecheck version identity overflow");
                    Arc::new(GlobalVersion {
                        id,
                        carrier: access.recorded.clone(),
                        payload,
                    })
                });
                prepared.push(PreparedGlobalAccess {
                    access,
                    new_version,
                });
            }
            if !tcgen_acquisition.is_empty() {
                tcgen_acquisitions.insert(lane, tcgen_acquisition);
            }
            if !lane_shared_acquisition.is_empty() {
                acquired_frontier.insert(lane, lane_shared_acquisition);
            }
            actor_updates.insert(actor, actor_state);
        }
        for gap in laggard_gaps {
            self.push_incomplete(gap);
        }
        self.validate_prepared_accesses(&shadows, &prepared);

        let clock_frontier = acquired_frontier;
        self.stage_or_commit_batch(
            &mut shadows,
            batch.operation().id(),
            descriptor.kind().writes(),
            actor_updates,
            prepared,
        );
        Ok((tcgen_acquisitions, clock_frontier))
    }

    pub(crate) fn after_batch(&mut self, batch: &PhysicalAccessBatch) -> Result<(), String> {
        if !batch.descriptor().space().has_read_from_versions() {
            return Ok(());
        }
        if !batch.descriptor().kind().writes() {
            return Ok(());
        }
        if batch.descriptor().space() == PhysicalAccessSpace::Shared
            && !direct_write_requires_version(batch.descriptor().memory_semantics())
        {
            self.invalidate_shared_versions(|| {
                batch
                    .lanes()
                    .iter()
                    .flat_map(|lane| lane.footprint().spans())
                    .copied()
            });
            return Ok(());
        }
        // A word a protocol uses keeps its whole write history: the value the
        // access left and the version it created. This is the first moment
        // both exist -- the version is built when the batch is staged, before
        // the write executes, and the value is read back after it does -- so
        // the history is appended here rather than at either end (API §3).
        //
        // Every strong write is kept, not only a primitive's. The publisher of
        // a declared word is spelled in raw PTX now -- `st.release`, `red`,
        // `atom` -- so gating on the operation's identity would leave the word
        // with no history at all and the wait unable to name the write it
        // accepted. A plain write cannot be a publication, so it is still
        // skipped, and the history is capped per word below.
        // Global only: this function now also sees shared writes that
        // carry a version, and those take no post-image read-back.
        if batch.descriptor().space() == PhysicalAccessSpace::Global
            && batch
                .descriptor()
                .memory_semantics()
                .class()
                .is_atomic_class()
        {
            self.record_declared_word_writes(batch);
        }
        self.commit_staged_batch(batch.operation().id())
    }

    fn invalidate_shared_versions<I: Iterator<Item = PhysicalByteSpan>>(&self, spans: impl Fn() -> I) {
        // A batch's spans (one per lane and row of a copy) fall into a few
        // stripes, and those stripes usually have no shadow yet: collect the
        // distinct stripe keys without sorting every span's key, and stop
        // here when none of them has a cell.
        let mut keys: Vec<ShadowCellKey> = Vec::new();
        for key in spans()
            .map(|span| (PhysicalAccessSpace::Shared, span))
            .flat_map(shadow_cell_keys)
        {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        if keys.is_empty() {
            return;
        }
        keys.sort_unstable();
        let cells = self.shared.shadows_for_keys(keys, false);
        if cells.is_empty() {
            return;
        }
        let mut shadows = LockedShadows::lock(&cells);
        for span in spans() {
            shadows.for_each_overlapping_mut(
                PhysicalAccessSpace::Shared,
                span,
                |shadow, lo, hi| {
                    if !shadow.range_is_untouched(lo, hi) {
                        shadow.update(lo, hi, |state| state.current_version = None);
                    }
                },
            );
        }
    }

    fn invalidate_weak_shared_batches(&self, batches: &[PhysicalAccessBatch]) {
        self.invalidate_shared_versions(|| {
            batches
                .iter()
                .filter(|batch| {
                    batch.descriptor().space() == PhysicalAccessSpace::Shared
                        && batch.descriptor().kind().writes()
                        && !batch.descriptor().memory_semantics().order().is_strong()
                })
                .flat_map(|batch| batch.lanes())
                .flat_map(|lane| lane.footprint().spans())
                .copied()
        });
    }

    /// Append one declared batch's writes to their word's history.
    fn record_declared_word_writes(&self, batch: &PhysicalAccessBatch) {
        let Some(staged) = self.staged.get(batch.operation().id()) else {
            return;
        };
        for prepared in staged.accesses.iter() {
            let Some(version) = prepared.new_version.as_ref() else {
                continue;
            };
            let span = prepared.access.recorded.span();
            let lane = prepared.access.recorded.actor().lane as usize;
            let Some(value) = batch.declared_value_for_lane(lane) else {
                // No post-image was taken, so this write is not one a wait can
                // poll -- the width is not 4 or 8 bytes. It may still overlap a
                // declared word, which the claim has to know about.
                self.shared
                    .note_unrecorded_strong_write(span, batch.operation().id());
                continue;
            };
            if !self.shared.record_declared_word_write(
                span,
                value,
                Arc::clone(version),
                batch.operation().id().clone(),
            ) {
                self.shared.push_incomplete(RaceCheckIncompleteReason::DeclaredWordHistoryTruncated {
                    operation: batch.operation().id().clone(),
                });
            }
        }
    }

    pub(crate) fn before_compact_batch(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
        tcgen_publications: &TcgenLaneFrontiers,
        shared_frontier: Option<&SharedLaneFrontiers>,
    ) -> Result<(TcgenLaneFrontiers, GlobalClockFrontier), String> {
        let descriptor = batch.descriptor();
        debug_assert!(descriptor.space().has_read_from_versions());
        if descriptor.space() == PhysicalAccessSpace::Shared
            && !direct_write_requires_version(descriptor.memory_semantics())
        {
            return Ok((TcgenLaneFrontiers::new(), GlobalClockFrontier::default()));
        }
        if self.staged.contains_key(batch.operation().id()) {
            return Err(format!(
                "global racecheck operation {} already has a staged batch",
                batch.operation().id()
            ));
        }
        if descriptor.space() == PhysicalAccessSpace::Global
            && descriptor.kind().writes()
            && self.shared.tracked_allocations.is_some()
        {
            self.shared
                .track_allocations(batch.lane_spans().map(|(_, span)| span.allocation()));
        }
        let semantics = descriptor.memory_semantics();
        if semantics.proxy() == MemoryProxy::Mmio {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: batch.operation().id().clone(),
                kind: "mmio_device_behavior_unmodeled",
                reason: "MMIO device behavior has no exact read-from model".to_string(),
            });
        }
        // Suppresses the exact predecessor only; see `before_batch`.
        let ambiguous_rmw_read_from =
            if descriptor.kind() == PhysicalAccessKind::AtomicReadModifyWrite {
                let spans = batch
                    .lane_spans()
                    .map(|(_, span)| span)
                    .filter(|span| self.tracks_span(descriptor.space(), *span))
                    .collect::<Vec<_>>();
                spans.iter().enumerate().any(|(index, span)| {
                    spans[index + 1..].iter().any(|other| span.overlaps(*other))
                })
            } else {
                false
            };
        let shared_operation = batch.operation().shared_id();
        let mut actor_updates = BTreeMap::new();
        let mut prepared = Vec::with_capacity(batch.operation().active_mask().len());
        let mut tcgen_acquisitions = TcgenLaneFrontiers::new();
        let mut acquired_frontier = SharedLaneFrontiers::new();
        let mut join_cache = AcquireJoinCache::default();
        // Per-lane setup that needs no byte-state lock runs before the cells
        // are locked, so the lock is held only for the read-from lookup,
        // validation and publication: a polled counter's cell is contended by
        // every poller and by the writer waiting for them to drain.
        let defers = weak_read_defers(descriptor.kind(), semantics);
        let lanes = batch
            .lane_spans()
            .filter(|(_, span)| self.tracks_span(descriptor.space(), *span))
            .map(|(lane, span)| {
                let actor = GlobalActor::new(batch.operation().id().global_warp_id(), lane);
                let actor_state = self
                    .actors
                    .get(&actor)
                    .cloned()
                    .unwrap_or_else(|| self.actors.empty_state());
                let tcgen_publication =
                    tcgen_publications.get(&lane).cloned().unwrap_or_default();
                (lane, span, actor, actor_state, tcgen_publication)
            })
            .collect::<Vec<_>>();
        let cells = self.shared.shadows(
            batch
                .lane_spans()
                .map(|(_, span)| span)
                .filter(|span| self.tracks_span(descriptor.space(), *span))
                .map(|span| (descriptor.space(), span)),
            true,
        );
        // A batch that publishes through the pending logs needs the byte
        // state only to find each lane's read-from version and, once the
        // records exist, to validate and publish them. The acquire joins,
        // event clocks and records in between run with the cells unlocked, so
        // pollers of one counter do not hold its cell across that work while
        // the writer they wait for is queued behind them. A writer that lands
        // between the two locks either drains the pending log before this
        // batch reaches it, in which case this batch's validation sees the
        // writer in the frontier, or after, in which case the drain sees the
        // records: every pair is still checked exactly once.
        let mut shadows = Some(if defers {
            LockedShadows::lock_deferred(&cells)
        } else {
            LockedShadows::lock(&cells)
        });
        let predecessors = lanes
            .iter()
            .map(|(_, span, _, actor_state, _)| {
                if descriptor.kind().reads()
                    && semantics.class().is_atomic_class()
                    && !ambiguous_rmw_read_from
                {
                    self.read_versions(
                        shadows.as_ref().expect("cells are locked for the read-from lookup"),
                        batch.operation().id(),
                        std::slice::from_ref(span),
                        descriptor.space(),
                        semantics,
                        &actor_state.clock,
                    )
                } else {
                    Vec::new()
                }
            })
            .collect::<Vec<_>>();
        if defers {
            shadows = None;
        }
        let registry = Arc::clone(&self.actors.async_registry);
        let mut laggard_gaps = Vec::new();
        for ((lane, span, actor, mut actor_state, tcgen_publication), predecessors) in
            lanes.into_iter().zip(predecessors)
        {
            if let Some(gap) =
                note_laggard_access(&mut actor_state, &registry, batch.operation().id())
            {
                laggard_gaps.push(gap);
            }
            let mut tcgen_acquisition = TcgenFenceFrontier::default();
            let mut lane_shared_acquisition = SharedClockFrontier::default();
            for predecessor in &predecessors {
                self.apply_load_ordering(
                    batch.operation().id(),
                    actor,
                    semantics,
                    descriptor.kind() == PhysicalAccessKind::Read,
                    predecessor,
                    &mut actor_state,
                    &mut tcgen_acquisition,
                    &mut lane_shared_acquisition,
                    &mut join_cache,
                );
            }
            let mut lane_shared_publication = shared_frontier
                .and_then(|all| all.get(&lane))
                .cloned()
                .unwrap_or_default();
            lane_shared_publication.merge(&lane_shared_acquisition);
            actor_state.clock.tick(actor)?;
            let event_clock = actor_state.clock.clone();
            let access = GlobalAccess {
                recorded: Arc::new(RecordedGlobalAccess::new(
    actor,
    GlobalFrontierActor::Lane(actor),
    event_clock.actor_epoch(actor),
    PhysicalRaceWitness::from_parts_shared(
                        Arc::clone(&shared_operation),
                        lane,
                        descriptor.kind(),
                        descriptor.space(),
                        span,
                    ),
    semantics,
    0,
)),
                clock: event_clock,
            };
            let new_version = (descriptor.kind().writes()
                && direct_write_requires_version(semantics))
            .then(|| {
                let payload = self.publication_for_write(
                    batch.operation().id(),
                    &access,
                    &predecessors,
                    actor_state.release_fence.as_ref(),
                    &tcgen_publication,
                    Some(&lane_shared_publication),
                );
                let id = self
                    .shared
                    .allocate_versions(1)
                    .expect("global racecheck version identity overflow");
                Arc::new(GlobalVersion {
                    id,
                    carrier: access.recorded.clone(),
                    payload,
                })
            });
            prepared.push(PreparedGlobalAccess {
                access,
                new_version,
            });
            if !tcgen_acquisition.is_empty() {
                tcgen_acquisitions.insert(lane, tcgen_acquisition);
            }
            if !lane_shared_acquisition.is_empty() {
                acquired_frontier.insert(lane, lane_shared_acquisition);
            }
            actor_updates.insert(actor, actor_state);
        }
        for gap in laggard_gaps {
            self.push_incomplete(gap);
        }
        let mut shadows =
            shadows.unwrap_or_else(|| LockedShadows::lock_deferred(&cells));
        self.validate_prepared_accesses(&shadows, &prepared);
        let clock_frontier = acquired_frontier;
        self.stage_or_commit_batch(
            &mut shadows,
            batch.operation().id(),
            descriptor.kind().writes(),
            actor_updates,
            prepared,
        );
        Ok((tcgen_acquisitions, clock_frontier))
    }

    pub(crate) fn after_compact_batch(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<(), String> {
        debug_assert!(batch.descriptor().space().has_read_from_versions());
        if !batch.descriptor().kind().writes() {
            return Ok(());
        }
        if batch.descriptor().space() == PhysicalAccessSpace::Shared
            && !direct_write_requires_version(batch.descriptor().memory_semantics())
        {
            self.invalidate_shared_versions(|| batch.lane_spans().map(|(_, span)| span));
            return Ok(());
        }
        self.commit_staged_batch(batch.operation().id())
    }

    pub(crate) fn apply_plain_compact_write_after_numeric(
        &mut self,
        batch: &CompactPhysicalAccessBatch<'_>,
    ) -> Result<GlobalClockFrontier, String> {
        if batch.descriptor().space() != PhysicalAccessSpace::Global
            || batch.descriptor().kind() != PhysicalAccessKind::Write
            || batch.descriptor().memory_semantics() != MemoryAccessSemantics::plain()
        {
            return Err(format!(
                "global racecheck after-numeric fast path requires one plain global write, got {:?}",
                batch.descriptor(),
            ));
        }
        if self.staged.contains_key(batch.operation().id()) {
            return Err(format!(
                "global racecheck operation {} already has a staged batch",
                batch.operation().id()
            ));
        }
        let lane_spans = batch.lane_spans().collect::<Vec<_>>();
        if self.shared.tracked_allocations.is_some() {
            self.shared
                .track_allocations(lane_spans.iter().map(|(_, span)| span.allocation()));
        }
        let dense_candidate = lane_spans.first().and_then(|(_, first_span)| {
            let allocation = first_span.allocation();
            let byte_offset = first_span.byte_offset();
            let mut byte_end = byte_offset;
            let contiguous = lane_spans.iter().all(|(_, span)| {
                let contiguous = span.allocation() == allocation && span.byte_offset() == byte_end;
                byte_end = span.byte_end();
                contiguous
            });
            contiguous.then_some((allocation, byte_offset, byte_end, first_span.byte_len()))
        });
        // The allocation stays locked from the untouched check to the
        // first-touch insert.
        let dense_range = dense_candidate.map(|(allocation, byte_offset, byte_end, _)| {
            PhysicalByteSpan::new(allocation, byte_offset, byte_end - byte_offset)
                .expect("a contiguous lane-span batch range is representable")
        });
        let cells = dense_range
            .map(|range| {
                self.shared
                    .shadows(std::iter::once((PhysicalAccessSpace::Global, range)), true)
            })
            .unwrap_or_default();
        let mut shadows = LockedShadows::lock(&cells);
        let dense_first_touch = dense_candidate.filter(|_| {
            let mut untouched = true;
            shadows.for_each_overlapping(
                PhysicalAccessSpace::Global,
                dense_range.expect("a dense candidate has a range"),
                |shadow, lo, hi| untouched &= shadow.range_is_untouched(lo, hi),
            );
            untouched
        });
        if let Some((allocation, byte_offset, byte_end, byte_width)) = dense_first_touch {
            let _actor_profile = ProfileTimer::new(ProfileKind::RaceGlobalDenseActors);
            let shared_operation = batch.operation().shared_id();
            let mut first_touch_lanes = Vec::with_capacity(lane_spans.len());
            #[cfg(any(test, feature = "profile"))]
            let mut replay_prepared = Vec::with_capacity(lane_spans.len());
            let registry = Arc::clone(&self.actors.async_registry);
            let mut laggard_gaps = Vec::new();
            for (lane, span) in lane_spans.iter().copied() {
                let actor = GlobalActor::new(batch.operation().id().global_warp_id(), lane);
                let actor_state = self.actors.get_or_insert_default(actor);
                if let Some(gap) =
                    note_laggard_access(actor_state, &registry, batch.operation().id())
                {
                    laggard_gaps.push(gap);
                }
                actor_state.clock.tick(actor)?;
                #[cfg(any(test, feature = "profile"))]
                let event_clock = actor_state.clock.clone();
                first_touch_lanes.push(GlobalFirstTouchLane {
                    lane: lane as u8,
                    epoch: actor_state.clock.actor_epoch(actor),
                });
                #[cfg(any(test, feature = "profile"))]
                if self.replay_recording {
                    replay_prepared.push(PreparedGlobalAccess {
                        access: GlobalAccess {
                            recorded: Arc::new(RecordedGlobalAccess::new(
    actor,
    GlobalFrontierActor::Lane(actor),
    event_clock.actor_epoch(actor),
    PhysicalRaceWitness::from_parts_shared(
                                    Arc::clone(&shared_operation),
                                    lane,
                                    PhysicalAccessKind::Write,
                                    PhysicalAccessSpace::Global,
                                    span,
                                ),
    MemoryAccessSemantics::plain(),
    0,
)),
                            clock: event_clock,
                        },
                        new_version: None,
                    });
                }
            }
            for gap in laggard_gaps {
                self.push_incomplete(gap);
            }
            #[cfg(any(test, feature = "profile"))]
            self.record_replay_batch(&replay_prepared);
            drop(_actor_profile);
            let node = GlobalFirstTouchSegment {
                byte_offset,
                byte_end,
                operation: shared_operation,
                byte_width,
                lanes: first_touch_lanes.into_boxed_slice(),
            };
            let _ = allocation;
            shadows.for_each_overlapping_mut(
                PhysicalAccessSpace::Global,
                dense_range.expect("a dense candidate has a range"),
                |shadow, lo, hi| {
                    shadow.first_touches.insert_prepared(lo, hi, node.clone());
                },
            );
            return Ok(GlobalClockFrontier::default());
        }
        drop(shadows);
        let (_, clock_frontier) =
            self.before_compact_batch(batch, &TcgenLaneFrontiers::new(), None)?;
        if !self.staged.contains_key(batch.operation().id()) {
            return Err(format!(
                "global racecheck operation {} has no staged plain-write batch",
                batch.operation().id()
            ));
        }
        self.commit_staged_batch(batch.operation().id())?;
        Ok(clock_frontier)
    }

    fn stage_or_commit_batch(
        &mut self,
        shadows: &mut LockedShadows<'_>,
        operation: &DynamicOpId,
        writes: bool,
        actor_updates: BTreeMap<GlobalActor, GlobalActorState>,
        accesses: Vec<PreparedGlobalAccess>,
    ) {
        if writes {
            self.staged.insert(
                operation.clone(),
                StagedGlobalBatch {
                    actor_updates,
                    accesses: accesses.into_boxed_slice(),
                },
            );
            return;
        }
        self.actors.extend(actor_updates);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_batch(&accesses);
        if shadows.deferred {
            shadows.publish_pending(&accesses);
            return;
        }
        for prepared in accesses {
            shadows.publish(&prepared);
        }
    }

    fn commit_staged_batch(&mut self, operation: &DynamicOpId) -> Result<(), String> {
        let staged = self.staged.remove(operation).ok_or_else(|| {
            format!(
                "global racecheck operation {} has no staged batch",
                operation
            )
        })?;
        self.actors.extend(staged.actor_updates);
        let prepared = staged.accesses.into_vec();
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_batch(&prepared);
        // Plain global accesses do not hold the exact-read-from transaction
        // across numerical execution. Revalidate at commit, under the
        // allocation locks, so two concurrently staged accesses cannot both
        // miss each other before either one reaches the launch-wide shadow.
        self.validate_and_commit_prepared(prepared);
        Ok(())
    }

    /// Validate `prepared` against the current byte state and commit it, with
    /// the touched allocations locked across both steps. A batch of weak
    /// reads holds only the read side and appends to the pending logs in
    /// exactly the order [`Self::commit_prepared_accesses`] would have
    /// applied it.
    fn validate_and_commit_prepared(&mut self, prepared: Vec<PreparedGlobalAccess>) {
        let defers = !prepared.is_empty() && prepared.iter().all(prepared_defers);
        let cells = self.shared.shadows(
            prepared
                .iter()
                .map(|prepared| (prepared.access.space(), prepared.access.span())),
            true,
        );
        let mut shadows = if defers {
            LockedShadows::lock_deferred(&cells)
        } else {
            LockedShadows::lock(&cells)
        };
        self.validate_prepared_accesses(&shadows, &prepared);
        if defers {
            let mut by_allocation = BTreeMap::<
                (PhysicalAccessSpace, PhysicalAllocationId),
                Vec<PreparedGlobalAccess>,
            >::new();
            for prepared in prepared {
                let span = prepared.access.span();
                by_allocation
                    .entry((prepared.access.space(), span.allocation()))
                    .or_default()
                    .push(prepared);
            }
            for (_, mut prepared) in by_allocation {
                GlobalAllocationShadow::sort_prepared(&mut prepared);
                for prepared in &prepared {
                    shadows.publish(prepared);
                }
            }
        } else {
            self.commit_prepared_accesses(&mut shadows, prepared);
        }
    }

    pub(crate) fn discard_batch(&mut self, operation: &DynamicOpId) {
        self.staged.remove(operation);
    }

    pub(crate) fn begin_async_token(
        &mut self,
        token: &AsyncTokenId,
        operation: &OperationContext,
        issue_accesses: &[PhysicalAccessBatch],
        completion_accesses: &[PhysicalAccessBatch],
        commit_issue_accesses: bool,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalAsyncIssue);
        if commit_issue_accesses {
            self.invalidate_weak_shared_batches(issue_accesses);
        }
        self.register_written_allocations(issue_accesses);
        self.register_written_allocations(completion_accesses);
        let global_batches = issue_accesses
            .iter()
            .chain(completion_accesses)
            .filter(|batch| {
                (batch.descriptor().space() == PhysicalAccessSpace::Global
                    || (batch.descriptor().space() == PhysicalAccessSpace::Shared
                        && batch.descriptor().memory_semantics().order().is_strong()))
                    && self.batch_has_tracked_span(batch)
            })
            .collect::<Vec<_>>();
        if global_batches.is_empty() {
            #[cfg(any(test, feature = "profile"))]
            self.record_replay_async_lifecycle(ReplayAsyncPhase::Issue, token);
            return Ok(());
        }
        if self.async_tokens.contains_key(token) {
            return Err(format!(
                "global racecheck async token {token:?} is already active"
            ));
        }

        let issuers = global_batches
            .iter()
            .flat_map(|batch| batch.lanes())
            .map(|lane| GlobalActor::new(operation.id().global_warp_id(), lane.provenance().lane()))
            .collect::<BTreeSet<_>>();
        if issuers.len() != 1 {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: operation.id().clone(),
                kind: "multi_lane_async_publication",
                reason: format!(
                    "async global token {token:?} has {} issuing lanes; exact per-lane token ownership is unavailable",
                    issuers.len()
                ),
            });
        }

        let mut clock = self.actors.empty_clock();
        let registry = Arc::clone(&self.actors.async_registry);
        let mut laggard_gaps = Vec::new();
        let mut issue_epoch = 0;
        for actor in &issuers {
            let state = self.actors.get_or_insert_default(*actor);
            if let Some(gap) = note_laggard_access(state, &registry, token.issue_operation()) {
                laggard_gaps.push(gap);
            }
            state.clock.tick(*actor)?;
            issue_epoch = issue_epoch.max(state.clock.actor_epoch(*actor));
            clock.merge(&state.clock);
        }
        for gap in laggard_gaps {
            self.push_incomplete(gap);
        }
        let lease = self.actors.async_registry.lease(token.clone(), issue_epoch)?;
        clock.tick_async(&lease)?;
        let release_fence = issuers
            .first()
            .and_then(|actor| self.actors.get(actor))
            .and_then(|state| state.release_fence.clone());
        // Numerical async sources are snapshotted at issue, even when their
        // lifetime/read completion is reported later. Retain these exact
        // versions on the token; only an observed completion can expose them.
        let read_observations = self.observe_async_read_versions(issue_accesses, &clock)?;
        if commit_issue_accesses {
            self.commit_async_batches(
                &lease,
                issue_accesses,
                &clock,
                release_fence.as_ref(),
                &TcgenFenceFrontier::default(),
                None,
            )?;
            if issue_accesses.iter().any(|batch| {
                batch.descriptor().space() == PhysicalAccessSpace::Global
                    && batch.descriptor().memory_semantics().proxy() == MemoryProxy::Async
            }) {
                clock.apply_implicit_async_completion();
            }
        }
        self.async_tokens.insert(
            token.clone(),
            GlobalAsyncTokenState {
                lease,
                current: clock,
                source_read: None,
                full: None,
                release_fence,
                read_observations,
                tcgen_publication: TcgenFenceFrontier::default(),
                shared_frontier: None,
            },
        );
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_async_lifecycle(ReplayAsyncPhase::Issue, token);
        Ok(())
    }

    pub(crate) fn complete_async_token(
        &mut self,
        token: &AsyncTokenId,
        milestone: AsyncGroupMilestone,
        accesses: &[PhysicalAccessBatch],
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalAsyncComplete);
        // Weak shared writes carry no publication. They still overwrite the
        // current bytes, even when this copy needed no read-from token.
        self.invalidate_weak_shared_batches(accesses);
        let Some((current, lease, release_fence, tcgen_publication, shared_frontier)) =
            self.async_tokens.get(token).map(|state| {
                (
                    state.current.clone(),
                    Arc::clone(&state.lease),
                    state.release_fence.clone(),
                    state.tcgen_publication.clone(),
                    state.shared_frontier.clone(),
                )
            })
        else {
            #[cfg(any(test, feature = "profile"))]
            self.record_replay_async_lifecycle(ReplayAsyncPhase::Complete(milestone), token);
            return Ok(());
        };
        let mut clock = current;
        clock.tick_async(&lease)?;
        self.commit_async_batches(
            &lease,
            accesses,
            &clock,
            release_fence.as_ref(),
            &tcgen_publication,
            shared_frontier.as_ref(),
        )?;
        if accesses.iter().any(|batch| {
            batch.descriptor().space() == PhysicalAccessSpace::Global
                && batch.descriptor().memory_semantics().proxy() == MemoryProxy::Async
        }) {
            clock.apply_implicit_async_completion();
        }
        let state = self
            .async_tokens
            .get_mut(token)
            .expect("global async token remains active through completion");
        state.current = clock.clone();
        match milestone {
            AsyncGroupMilestone::SourceReadComplete => state.source_read = Some(clock.clone()),
            AsyncGroupMilestone::FullComplete => {
                state.full = Some(clock.clone());
            }
        }
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_async_lifecycle(ReplayAsyncPhase::Complete(milestone), token);
        Ok(())
    }

    fn copy_completion_payload(
        &self,
        token: &AsyncTokenId,
    ) -> Result<Option<GlobalAsyncPublication>, String> {
        let Some(state) = self.async_tokens.get(token) else {
            return Ok(None);
        };
        let full = state.full.as_ref().ok_or_else(|| {
            format!("global racecheck async token {token:?} has not fully completed")
        })?;
        // Complete-tx does not publish arbitrary earlier issuer instructions.
        let mut clock = SparseLaneClock::new(Arc::clone(&full.async_registry));
        clock.async_components.set(
            state.lease.handle.index,
            full.async_components
                .get(state.lease.handle.index, &full.async_registry.blocks),
            &full.async_registry.blocks,
        );
        clock.apply_implicit_async_completion();
        Ok(Some(GlobalAsyncPublication {
            clock,
            read_observations: state.read_observations.clone(),
        }))
    }

    fn publish_copy_completion(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        payload: Option<GlobalAsyncPublication>,
    ) {
        if let Some(payload) = &payload {
            Self::merge_physical_barrier_payload(
                &mut self.physical_copy_barriers,
                (barrier_id, generation),
                payload.clone(),
            );
        }
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_async_publish(barrier_id, generation, payload.as_ref());
    }

    pub(crate) fn publish_async_token_to_physical_barrier(
        &mut self,
        token: &AsyncTokenId,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalAsyncPublish);
        let payload = self.copy_completion_payload(token)?;
        self.publish_copy_completion(barrier_id, generation, payload);
        Ok(())
    }

    pub(crate) fn retain_cp_async_completion(
        &mut self,
        warp: usize,
        lane: usize,
        token: &AsyncTokenId,
    ) -> Result<(), String> {
        if let Some(payload) = self.copy_completion_payload(token)? {
            match self.cp_async_completed.entry((warp, lane)) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(payload);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().merge(&payload);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn publish_cp_async_completion(
        &mut self,
        warp: usize,
        lane: usize,
        barrier_id: PhysicalBarrierId,
        generation: u64,
    ) {
        let payload = self.cp_async_completed.get(&(warp, lane)).cloned();
        self.publish_copy_completion(barrier_id, generation, payload);
    }

    pub(crate) fn acquire_async_token(
        &mut self,
        operation: &OperationContext,
        token: &AsyncTokenId,
        milestone: AsyncGroupMilestone,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalAsyncAcquire);
        let Some(state) = self.async_tokens.get(token) else {
            #[cfg(any(test, feature = "profile"))]
            self.record_replay_async_acquire(
                operation.id().global_warp_id(),
                operation.active_mask(),
                None,
            );
            return Ok(());
        };
        let clock = match milestone {
            AsyncGroupMilestone::SourceReadComplete => state.source_read.as_ref(),
            AsyncGroupMilestone::FullComplete => state.full.as_ref(),
        }
        .cloned()
        .ok_or_else(|| {
            format!("global racecheck async token {token:?} has not reached {milestone:?}")
        })?;
        let payload = GlobalExecutionPayload {
            clock,
            read_observations: state.read_observations.clone(),
        };
        let global_warp_id = operation.id().global_warp_id();
        let mask = operation.active_mask();
        self.acquire_mask(global_warp_id, mask, &payload)?;
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_async_acquire(global_warp_id, mask, Some(&payload));
        Ok(())
    }

    pub(crate) fn retire_async_token(&mut self, token: &AsyncTokenId) {
        self.async_tokens.remove(token);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::AsyncRetire {
            token: token.clone(),
        });
    }

    pub(crate) fn tensor_map_observation(
        &mut self,
        operation: &OperationContext,
        observation: TensorMapObservation,
    ) -> Result<(), String> {
        let warp = operation.id().global_warp_id();
        match observation {
            TensorMapObservation::Release { scope } => {
                self.tensor_map_release(operation, None, scope)?;
            }
            TensorMapObservation::CopyRelease { descriptor, scope } => {
                // The fused instruction's .sync.aligned copy joins all lane
                // words before it releases this one descriptor.
                let payload = self.release_mask(warp, operation.active_mask())?;
                self.acquire_mask_from_own_release(warp, operation.active_mask(), &payload);
                self.tensor_map_release(operation, Some(descriptor), scope)?;
            }
            TensorMapObservation::Acquire {
                descriptor,
                generation,
                lane,
                cta,
                scope,
            } => {
                if !operation.active_mask().contains(lane) {
                    return Err("TensorMap acquire lane is outside its instruction mask".into());
                }
                let actor = GlobalActor::new(warp, lane);
                let mut writes = SparseClockFrontier::from_clock(&self.actors.empty_clock());
                if let Some(bridges) = self
                    .actors
                    .get(&actor)
                    .and_then(|state| state.clock.proxy_bridges.as_ref())
                {
                    for (&(range, publisher, release_scope), frontier) in
                        &bridges.tensor_map_released
                    {
                        let required = MemoryScope::required_between_warps(
                            self.topology,
                            publisher.global_warp_id,
                            warp,
                        );
                        if range.is_none_or(|range| range == descriptor)
                            && release_scope >= required
                            && scope >= required
                        {
                            writes.merge(frontier);
                        }
                    }
                }
                let published = self.tensor_map_writes_covered(descriptor, &writes);
                let state = self.actors.get_or_insert_default(actor);
                if published {
                    let acquired = TensorMapAcquired { generation, writes };
                    Arc::make_mut(
                        state
                            .clock
                            .proxy_bridges
                            .get_or_insert_with(|| Arc::new(GlobalProxyBridgeFrontiers::default())),
                    )
                    .tensor_map_acquired
                    .entry((descriptor, cta))
                    .and_modify(|current| current.merge(&acquired))
                    .or_insert(acquired);
                }
                state.clock.tick(actor)?;
            }
            TensorMapObservation::Consume {
                descriptor,
                generation,
                cta,
            } => {
                for lane in operation.active_mask() {
                    let actor = GlobalActor::new(warp, lane);
                    let acquired = self
                        .actors
                        .get(&actor)
                        .and_then(|state| state.clock.proxy_bridges.as_ref())
                        .and_then(|bridges| bridges.tensor_map_acquired.get(&(descriptor, cta)));
                    if acquired.is_none_or(|acquired| {
                        acquired.generation < generation
                            || !self.tensor_map_writes_covered(descriptor, &acquired.writes)
                    }) {
                        return Err(format!(
                            "TensorMap descriptor {descriptor} generation {generation} is not acquired \
                             by warp {warp} lane {lane} through program order or synchronization"
                        ));
                    }
                }
            }
        }
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::TensorMap {
            operation: operation.id().clone(),
            kind: operation.kind(),
            mask: operation.active_mask(),
            observation,
        });
        Ok(())
    }

    fn tensor_map_release(
        &mut self,
        operation: &OperationContext,
        descriptor: Option<PhysicalByteSpan>,
        scope: MemoryScope,
    ) -> Result<(), String> {
        let warp = operation.id().global_warp_id();
        for lane in operation.active_mask() {
            let actor = GlobalActor::new(warp, lane);
            let state = self.actors.get_or_insert_default(actor);
            let frontier = SparseClockFrontier::from_clock(&state.clock);
            Arc::make_mut(
                state
                    .clock
                    .proxy_bridges
                    .get_or_insert_with(|| Arc::new(GlobalProxyBridgeFrontiers::default())),
            )
            .tensor_map_released
            .entry((descriptor, actor, scope))
            .and_modify(|current| current.merge(&frontier))
            .or_insert(frontier);
            state.clock.tick(actor)?;
        }
        Ok(())
    }

    fn tensor_map_writes_covered(
        &self,
        descriptor: PhysicalByteSpan,
        frontier: &SparseClockFrontier,
    ) -> bool {
        let cells = self
            .shared
            .shadows([(PhysicalAccessSpace::Global, descriptor)], false);
        let shadows = LockedShadows::lock_deferred(&cells);
        let mut covered = true;
        shadows.for_each_overlapping(PhysicalAccessSpace::Global, descriptor, |shadow, lo, hi| {
            shadow.for_each_writer(descriptor, lo, hi, &mut |writer| {
                covered &= writer.frontier_epoch_in(&frontier.clock) >= writer.frontier_epoch;
            });
        });
        covered
    }

    pub(crate) fn proxy_async_fence(
        &mut self,
        operation: &OperationContext,
        effect: ProxyAsyncFenceEffect,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalProxyFence);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::ProxyFence {
            operation: operation.id().clone(),
            kind: operation.kind(),
            mask: operation.active_mask(),
            effect,
        });
        if !matches!(
            effect.scope(),
            ProxyAsyncFenceScope::All | ProxyAsyncFenceScope::Global
        ) {
            return Ok(());
        }
        for lane in operation.active_mask() {
            let actor = GlobalActor::new(operation.id().global_warp_id(), lane);
            let state = self.actors.get_or_insert_default(actor);
            state.clock.apply_proxy_async_fence();
            state.clock.tick(actor)?;
        }
        Ok(())
    }

    pub(crate) fn bind_async_release_frontiers(
        &mut self,
        token: &AsyncTokenId,
        tcgen: TcgenFenceFrontier,
        shared: SharedClockFrontier,
    ) {
        if let Some(state) = self.async_tokens.get_mut(token) {
            state.tcgen_publication = tcgen;
            state.shared_frontier = Some(shared);
        }
    }

    fn observe_async_read_versions(
        &mut self,
        batches: &[PhysicalAccessBatch],
        clock: &SparseLaneClock,
    ) -> Result<BTreeMap<ReleaseHeadKey, Arc<ReleaseHead>>, String> {
        let mut observations = BTreeMap::new();
        for batch in batches.iter().filter(|batch| {
            batch.descriptor().space().has_read_from_versions()
                && batch.descriptor().kind() == PhysicalAccessKind::Read
                && batch.descriptor().memory_semantics().order().is_strong()
        }) {
            let semantics = batch.descriptor().memory_semantics();
            for lane in batch.lanes() {
                let actor = GlobalActor::new(
                    batch.operation().id().global_warp_id(),
                    lane.provenance().lane(),
                );
                let spans = lane
                    .footprint()
                    .spans()
                    .iter()
                    .copied()
                    .filter(|span| self.tracks_span(batch.descriptor().space(), *span))
                    .collect::<Vec<_>>();
                let runs = match batch.transfer_unit_bytes() {
                    0 => coalesce_transfer_runs(spans),
                    width => spans.into_iter().map(|span| (span, width)).collect(),
                };
                for (span, unit_bytes) in runs {
                    let cells = self
                        .shared
                        .shadows([(batch.descriptor().space(), span)], false);
                    let shadows = LockedShadows::lock(&cells);
                    let width = if unit_bytes == 0 {
                        span.byte_len()
                    } else {
                        unit_bytes as usize
                    };
                    let topology = self.topology;
                    let mut cursor = span.byte_offset();
                    while cursor < span.byte_end() {
                        let element =
                            PhysicalByteSpan::new(span.allocation(), cursor / width * width, width)
                                .map_err(|error| error.to_string())?;
                        let end = span.byte_end().min(element.byte_end());
                        let observed =
                            PhysicalByteSpan::new(span.allocation(), cursor, end - cursor)
                                .map_err(|error| error.to_string())?;
                        self.visit_read_versions_for_element(
                            &shadows,
                            batch.operation().id(),
                            observed,
                            element,
                            batch.descriptor().space(),
                            semantics,
                            clock,
                            |version| {
                                if let Some(version) = version {
                                    if semantics_are_mutually_morally_strong(
                                        topology,
                                        version.carrier.semantics,
                                        version.carrier.actor(),
                                        semantics,
                                        actor,
                                    ) {
                                        merge_pending_acquire(
                                            &mut observations,
                                            version.payload.heads.values().cloned(),
                                        );
                                    }
                                }
                            },
                        );
                        cursor = end;
                    }
                }
            }
        }
        Ok(observations)
    }

    fn commit_async_batches(
        &mut self,
        lease: &Arc<AsyncClockLease>,
        batches: &[PhysicalAccessBatch],
        clock: &SparseLaneClock,
        release_fence: Option<&Arc<ReleaseHead>>,
        tcgen_publication: &TcgenFenceFrontier,
        shared_frontier: Option<&SharedClockFrontier>,
    ) -> Result<(), String> {
        let mut prepared = Vec::new();
        for batch in batches.iter().filter(|batch| {
            batch.descriptor().space() == PhysicalAccessSpace::Global
                || (batch.descriptor().space() == PhysicalAccessSpace::Shared
                    && batch.descriptor().kind().writes()
                    && batch.descriptor().memory_semantics().order().is_strong())
        }) {
            let semantics = batch.descriptor().memory_semantics();
            let has_async_metadata = semantics.class() == crate::MemoryAccessClass::Async
                || (semantics.class().is_atomic_class()
                    && semantics.order().is_strong()
                    && semantics.proxy() == MemoryProxy::Async)
                || (semantics.class().is_atomic_class()
                    && semantics.order().is_strong()
                    && matches!(semantics.proxy(), MemoryProxy::Generic | MemoryProxy::Mmio));
            if !has_async_metadata {
                self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                    operation: batch.operation().id().clone(),
                    kind: "async_access_missing_proxy_metadata",
                    reason: format!(
                        "async token {:?} carried a {} {}-proxy access",
                        lease.token,
                        semantics.class(),
                        semantics.proxy(),
                    ),
                });
            }
            for lane_access in batch.lanes() {
                let actor = GlobalActor::new(
                    batch.operation().id().global_warp_id(),
                    lane_access.provenance().lane(),
                );
                let spans = lane_access
                    .footprint()
                    .spans()
                    .iter()
                    .copied()
                    .filter(|span| self.tracks_span(batch.descriptor().space(), *span))
                    .collect::<Vec<_>>();
                // Contiguous equal-width units become one run entry (see
                // `GlobalAccess::unit_bytes`); a written run owns one version
                // identity per unit (see `GlobalVersion::unit_id_at`). A batch
                // planned as runs already carries its unit size.
                let runs = if batch.descriptor().kind() == PhysicalAccessKind::AtomicReadModifyWrite
                {
                    // RMW read-from and release ancestry belong to each element.
                    spans.into_iter().map(|span| (span, 0)).collect()
                } else {
                    match batch.transfer_unit_bytes() {
                        0 => coalesce_transfer_runs(spans),
                        unit_bytes => spans.into_iter().map(|span| (span, unit_bytes)).collect(),
                    }
                };
                for group in runs.chunk_by(|left, right| {
                    left.0.allocation() == right.0.allocation() && left.1 == right.1
                }) {
                    // A strong write owns one identity per logical element,
                    // shared by every selected fragment. Only the actual spans
                    // below enter the byte shadow; the hull is version metadata.
                    let (first, width) = group[0];
                    let publication_hull = if width != 0
                        && semantics.order().is_strong()
                        && batch.descriptor().kind() == PhysicalAccessKind::Write
                    {
                        let width = width as usize;
                        let origin = first.byte_offset() / width * width;
                        let end = group
                            .last()
                            .unwrap()
                            .0
                            .byte_end()
                            .div_ceil(width)
                            .checked_mul(width)
                            .ok_or("strong transfer element range overflow")?;
                        Some(
                            PhysicalByteSpan::new(first.allocation(), origin, end - origin)
                                .map_err(|error| error.to_string())?,
                        )
                    } else {
                        None
                    };
                    let mut shared_version: Option<Arc<GlobalVersion>> = None;
                    for &(span, unit_bytes) in group {
                        let witness = PhysicalRaceWitness::from_lane(
                            lane_access,
                            batch.descriptor().kind(),
                            batch.descriptor().space(),
                            span,
                        );
                        let access = GlobalAccess {
                            recorded: Arc::new(RecordedGlobalAccess::new(
                                actor,
                                GlobalFrontierActor::Async(Arc::clone(lease)),
                                clock.async_component(lease.handle),
                                witness,
                                semantics,
                                unit_bytes,
                            )),
                            clock: clock.clone(),
                        };
                        let new_version = if let Some(version) = &shared_version {
                            Some(Arc::clone(version))
                        } else if batch.descriptor().kind().writes() {
                            let predecessors = if batch.descriptor().kind()
                                == PhysicalAccessKind::AtomicReadModifyWrite
                            {
                                let cells = self
                                    .shared
                                    .shadows([(batch.descriptor().space(), span)], false);
                                let shadows = LockedShadows::lock(&cells);
                                self.read_versions(
                                    &shadows,
                                    batch.operation().id(),
                                    &[span],
                                    batch.descriptor().space(),
                                    semantics,
                                    clock,
                                )
                            } else {
                                Vec::new()
                            };
                            let payload = self.publication_for_write(
                                batch.operation().id(),
                                &access,
                                &predecessors,
                                release_fence,
                                tcgen_publication,
                                shared_frontier,
                            );
                            let carrier = publication_hull.map_or_else(
                                || Arc::clone(&access.recorded),
                                |hull| Arc::new(access.recorded.with_span(hull)),
                            );
                            let unit_count = if unit_bytes == 0 {
                                1
                            } else {
                                (carrier.span().byte_len() / unit_bytes as usize) as u64
                            };
                            let id = self.shared.allocate_versions(unit_count)?;
                            let version = Arc::new(GlobalVersion {
                                id,
                                carrier,
                                payload,
                            });
                            if publication_hull.is_some() {
                                shared_version = Some(Arc::clone(&version));
                            }
                            Some(version)
                        } else {
                            None
                        };
                        prepared.push(PreparedGlobalAccess {
                            access,
                            new_version,
                        });
                    }
                }
            }
        }
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_batch(&prepared);
        self.validate_and_commit_prepared(prepared);
        Ok(())
    }

    fn commit_prepared_accesses(
        &mut self,
        shadows: &mut LockedShadows<'_>,
        prepared: impl IntoIterator<Item = PreparedGlobalAccess>,
    ) {
        let mut by_allocation = BTreeMap::<
            (PhysicalAccessSpace, PhysicalAllocationId),
            Vec<PreparedGlobalAccess>,
        >::new();
        for prepared in prepared {
            let span = prepared.access.span();
            by_allocation
                .entry((prepared.access.space(), span.allocation()))
                .or_default()
                .push(prepared);
        }
        for ((space, allocation), mut prepared) in by_allocation {
            if prepared.is_empty() {
                continue;
            }
            GlobalAllocationShadow::sort_prepared(&mut prepared);
            // A dense, previously untouched weak-write batch becomes one
            // first-touch node (per stripe it covers) instead of one segment
            // per lane; "untouched" is judged over the whole batch range.
            if let Some((byte_offset, byte_end)) =
                GlobalAllocationShadow::contiguous_weak_write_range(&prepared)
            {
                let range = PhysicalByteSpan::new(allocation, byte_offset, byte_end - byte_offset)
                    .expect("a contiguous weak-write batch range is representable");
                let mut untouched = true;
                shadows.for_each_overlapping(space, range, |shadow, lo, hi| {
                    untouched &= shadow.range_is_untouched(lo, hi);
                });
                if untouched {
                    let node =
                        GlobalFirstTouchSegment::from_prepared(byte_offset, byte_end, prepared);
                    shadows.for_each_overlapping_mut(space, range, |shadow, lo, hi| {
                        shadow.first_touches.insert_prepared(lo, hi, node.clone());
                    });
                    continue;
                }
            }
            for prepared in &prepared {
                let span = prepared.access.span();
                shadows.for_each_overlapping_mut(space, span, |shadow, lo, hi| {
                    shadow.apply(prepared, lo, hi);
                });
            }
        }
    }

    pub(crate) fn fence(
        &mut self,
        operation: &OperationContext,
        effect: MemoryFenceEffect,
        tcgen_publications: &TcgenLaneFrontiers,
        shared_frontier: Option<&SharedLaneFrontiers>,
    ) -> Result<(TcgenLaneFrontiers, GlobalClockFrontier), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalFence);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::Fence {
            operation: operation.id().clone(),
            kind: operation.kind(),
            mask: operation.active_mask(),
            effect,
        });
        let mut tcgen_acquisitions = TcgenLaneFrontiers::new();
        let mut acquired_shared = SharedLaneFrontiers::new();
        let shared = Arc::clone(&self.shared);
        let mut sc_fences = (effect.order() == MemoryOrder::Sc).then(|| {
            shared
                .sc_fences
                .lock()
                .expect("SC fence order lock was poisoned")
        });
        for lane in operation.active_mask() {
            let actor = GlobalActor::new(operation.id().global_warp_id(), lane);
            let mut state = self
                .actors
                .get(&actor)
                .cloned()
                .unwrap_or_else(|| self.actors.empty_state());
            let mut tcgen_publication = tcgen_publications.get(&lane).cloned().unwrap_or_default();
            let mut tcgen_acquisition = TcgenFenceFrontier::default();
            let mut lane_shared_acquisition = SharedClockFrontier::default();
            if effect.order().has_acquire() {
                let pending = std::mem::take(&mut state.pending_acquire);
                for head in pending.into_values() {
                    if self.try_acquire_head(
                        operation.id(),
                        actor,
                        effect.scope(),
                        effect.proxy(),
                        &head,
                        &mut state,
                        &mut tcgen_acquisition,
                        true,
                    ) {
                        lane_shared_acquisition.merge(&head.shared_frontier);
                    }
                }
            }
            if let Some(heads) = &sc_fences {
                for head in heads.values() {
                    if head.key.proxy == effect.proxy()
                        && self.scope_covers(head.key.scope, head.key.actor, actor)
                        && self.scope_covers(effect.scope(), actor, head.key.actor)
                    {
                        state.clock.merge(&head.clock);
                        tcgen_acquisition.merge(&head.tcgen);
                        lane_shared_acquisition.merge(&head.shared_frontier);
                    }
                }
            }
            state.clock.tick(actor)?;
            if effect.order().has_release() {
                // AcqRel/SC relay what this fence just acquired; capturing only
                // the pre-fence publication loses a transitive TCGEN frontier.
                tcgen_publication.merge(&tcgen_acquisition);
                let mut lane_shared_publication = shared_frontier
                    .and_then(|all| all.get(&lane))
                    .cloned()
                    .unwrap_or_default();
                lane_shared_publication.merge(&lane_shared_acquisition);
                let head = Arc::new(ReleaseHead {
                    key: ReleaseHeadKey {
                        actor,
                        scope: effect.scope(),
                        proxy: effect.proxy(),
                    },
                    operation: operation.id().clone(),
                    clock: state.clock.clone(),
                    tcgen: tcgen_publication,
                    shared_frontier: lane_shared_publication,
                });
                if let Some(heads) = &mut sc_fences {
                    let cta = self.topology.map_or(actor.global_warp_id, |topology| {
                        actor.global_warp_id / topology.warps_per_cta()
                    });
                    heads.insert((cta, effect.scope(), effect.proxy()), Arc::clone(&head));
                }
                state.release_fence = Some(head);
            }
            if !tcgen_acquisition.is_empty() {
                tcgen_acquisitions.insert(lane, tcgen_acquisition);
            }
            if !lane_shared_acquisition.is_empty() {
                acquired_shared.insert(lane, lane_shared_acquisition);
            }
            self.actors.insert(actor, state);
        }
        Ok((tcgen_acquisitions, acquired_shared))
    }

    pub(crate) fn warp_sync(
        &mut self,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalWarpSync);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::WarpSync {
            global_warp_id,
            mask,
        });
        let payload = self.release_mask(global_warp_id, mask)?;
        if payload.is_bottom() {
            return Ok(());
        }
        // Racecheck clocks tick at global memory events. A synchronization
        // event only joins those event frontiers, so no lane component is
        // manufactured for the barrier itself. Worker-local TCGEN ordering is
        // carried separately by RaceCheckState and never enters this payload.
        // The payload joins every masked lane, so each lane's acquire result
        // is the payload itself. Handing every lane the same clock also
        // leaves the warp with one representation, which the next barrier
        // acquire joins once instead of once per lane.
        self.acquire_mask_from_own_release(global_warp_id, mask, &payload);
        Ok(())
    }

    pub(crate) fn reset_physical_barriers(&mut self, barrier_ids: &[PhysicalBarrierId]) {
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::ResetPhysicalBarriers {
            barrier_ids: barrier_ids.to_vec().into_boxed_slice(),
        });
        self.physical_barriers
            .retain(|(barrier_id, _), _| !barrier_ids.contains(barrier_id));
        self.physical_copy_barriers
            .retain(|(barrier_id, _), _| !barrier_ids.contains(barrier_id));
        for barrier_id in barrier_ids {
            self.physical_barrier_floors.remove(barrier_id);
        }
    }

    pub(crate) fn physical_barrier_release(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalPhysicalRelease);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::PhysicalBarrierRelease {
            barrier_id,
            generation,
            global_warp_id,
            mask,
        });
        let payload = self.release_mask(global_warp_id, mask)?;
        if payload.is_bottom() {
            return Ok(());
        }
        Self::merge_physical_barrier_payload(
            &mut self.physical_barriers,
            (barrier_id, generation),
            payload,
        );
        Ok(())
    }

    pub(crate) fn physical_barrier_acquire(
        &mut self,
        operation: Option<&DynamicOpId>,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
        acquire: bool,
    ) -> Result<(), String> {
        if self
            .physical_barrier_floors
            .get(&barrier_id)
            .is_some_and(|&(floor, pin)| generation < floor && Some(generation) != pin)
        {
            if let Some(operation) = operation {
                self.push_incomplete(RaceCheckIncompleteReason::BarrierPayloadUnavailable {
                    operation: operation.clone(),
                    barrier_id,
                    generation,
                });
            }
            return Ok(());
        }
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalPhysicalAcquire);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::PhysicalBarrierAcquire {
            barrier_id,
            generation,
            global_warp_id,
            mask,
            acquire,
        });
        let key = (barrier_id, generation);
        // Relaxed queries observe copy completion (and its source reads), but
        // do not acquire the ordinary arriving threads' release history.
        if let Some(payload) = self.physical_copy_barriers.get(&key).cloned() {
            self.acquire_mask(global_warp_id, mask, &payload)?;
        }
        if acquire {
            if let Some(payload) = self.physical_barriers.get(&key).cloned() {
                self.acquire_mask(global_warp_id, mask, &payload)?;
            }
        }
        Ok(())
    }

    pub(crate) fn named_barrier_release(
        &mut self,
        barrier_id: crate::NamedBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalNamedCluster);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::NamedBarrierRelease {
            barrier_id,
            generation,
            global_warp_id,
            mask,
        });
        let payload = self.release_mask(global_warp_id, mask)?;
        if payload.is_bottom() {
            return Ok(());
        }
        Self::merge_barrier_payload(
            &mut self.named_barriers,
            &mut self.named_barrier_floors,
            (barrier_id, generation),
            payload,
        );
        Ok(())
    }

    pub(crate) fn named_barrier_acquire(
        &mut self,
        operation: Option<&DynamicOpId>,
        barrier_id: crate::NamedBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<(), String> {
        if Self::generation_retired(&self.named_barrier_floors, &barrier_id, generation) {
            if let Some(operation) = operation {
                self.push_incomplete(RaceCheckIncompleteReason::ShadowRejected {
                    operation: operation.clone(),
                    reason: format!(
                        "named barrier {barrier_id:?} generation {generation} release payload was retired"
                    ),
                });
            }
            return Ok(());
        }
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalNamedCluster);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::NamedBarrierAcquire {
            barrier_id,
            generation,
            global_warp_id,
            mask,
        });
        let Some(payload) = self.named_barriers.get(&(barrier_id, generation)).cloned() else {
            return Ok(());
        };
        // A sync resume is paired with this same warp/mask's earlier
        // registration release for the generation. The completed payload
        // therefore already dominates every resuming lane. Reuse it directly
        // so all lanes retain one immutable clock base.
        self.acquire_mask_from_own_release(global_warp_id, mask, &payload);
        Ok(())
    }

    pub(crate) fn cluster_barrier_release(
        &mut self,
        barrier_id: crate::ClusterBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
        publishes_memory: bool,
    ) -> Result<(), String> {
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalNamedCluster);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::ClusterBarrierRelease {
            barrier_id,
            generation,
            global_warp_id,
            mask,
            publishes_memory,
        });
        if !publishes_memory {
            return Ok(());
        }
        let payload = self.release_mask(global_warp_id, mask)?;
        if payload.is_bottom() {
            return Ok(());
        }
        Self::merge_barrier_payload(
            &mut self.cluster_barriers,
            &mut self.cluster_barrier_floors,
            (barrier_id, generation),
            payload,
        );
        Ok(())
    }

    pub(crate) fn cluster_barrier_acquire(
        &mut self,
        operation: Option<&DynamicOpId>,
        barrier_id: crate::ClusterBarrierId,
        generation: u64,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<(), String> {
        if Self::generation_retired(&self.cluster_barrier_floors, &barrier_id, generation) {
            if let Some(operation) = operation {
                self.push_incomplete(RaceCheckIncompleteReason::ShadowRejected {
                    operation: operation.clone(),
                    reason: format!(
                        "cluster barrier {barrier_id:?} generation {generation} release payload was retired"
                    ),
                });
            }
            return Ok(());
        }
        let _profile = ProfileTimer::new(ProfileKind::RaceGlobalNamedCluster);
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::ClusterBarrierAcquire {
            barrier_id,
            generation,
            global_warp_id,
            mask,
        });
        let Some(payload) = self
            .cluster_barriers
            .get(&(barrier_id, generation))
            .cloned()
        else {
            return Ok(());
        };
        self.acquire_mask(global_warp_id, mask, &payload)
    }

    /// A numeric barrier outcome, not an individual clock merge, authorizes
    /// retirement. The pin also records a valid bottom/GC-retired release.
    pub(crate) fn retain_physical_barrier_generations(
        &mut self,
        barrier_id: PhysicalBarrierId,
        generation: u64,
        conditional: Option<u64>,
    ) {
        #[cfg(any(test, feature = "profile"))]
        self.record_replay_event(GlobalReplayEvent::RetainPhysicalBarrierGenerations {
            barrier_id,
            generation,
            conditional,
        });
        let floor = crate::sync_causality::retire_barrier_generations_except(
            &mut self.physical_barriers,
            barrier_id,
            generation,
            super::RETAINED_BARRIER_GENERATIONS,
            conditional,
        );
        crate::sync_causality::retire_barrier_generations_except(
            &mut self.physical_copy_barriers,
            barrier_id,
            generation,
            super::RETAINED_BARRIER_GENERATIONS,
            conditional,
        );
        let retained = self
            .physical_barrier_floors
            .entry(barrier_id)
            .or_insert((0, None));
        *retained = (retained.0.max(floor), conditional);
    }

    fn merge_physical_barrier_payload<B: Ord + Copy>(
        barriers: &mut BTreeMap<(B, u64), GlobalAsyncPublication>,
        key: (B, u64),
        payload: GlobalAsyncPublication,
    ) {
        match barriers.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(payload);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry.get_mut().merge(&payload);
            }
        }
    }

    fn merge_barrier_payload<B: Ord + Copy>(
        barriers: &mut BTreeMap<(B, u64), GlobalExecutionPayload>,
        floors: &mut BTreeMap<B, u64>,
        key: (B, u64),
        payload: GlobalExecutionPayload,
    ) {
        match barriers.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(payload);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry.get_mut().merge(&payload);
            }
        }
        Self::retire_generations(
            barriers,
            floors,
            key,
            super::retire_named_barrier_generations,
        );
    }

    /// Applies the generation window after a publication and remembers the
    /// barrier's retained floor so a late acquire below it is reported.
    fn retire_generations<B: Ord + Copy, V>(
        barriers: &mut BTreeMap<(B, u64), V>,
        floors: &mut BTreeMap<B, u64>,
        key: (B, u64),
        retire: fn(&mut BTreeMap<(B, u64), V>, B, u64) -> u64,
    ) {
        let floor = retire(barriers, key.0, key.1);
        if floor > 0 {
            let retained = floors.entry(key.0).or_insert(0);
            *retained = (*retained).max(floor);
        }
    }

    /// Whether `generation` of `barrier` was retired by the generation window.
    fn generation_retired<B: Ord>(floors: &BTreeMap<B, u64>, barrier: &B, generation: u64) -> bool {
        floors.get(barrier).is_some_and(|floor| generation < *floor)
    }

    fn release_mask(
        &mut self,
        global_warp_id: usize,
        mask: WarpMask,
    ) -> Result<GlobalExecutionPayload, String> {
        let mut clocks = mask
            .into_iter()
            .filter_map(|lane| self.actors.get(&GlobalActor::new(global_warp_id, lane)))
            .map(|state| &state.clock);
        let Some(first) = clocks.next() else {
            return Ok(GlobalExecutionPayload {
                clock: self.actors.empty_clock(),
                read_observations: BTreeMap::new(),
            });
        };
        let mut clock = first.clone();
        clock.merge_all(clocks);
        Ok(GlobalExecutionPayload {
            clock,
            read_observations: BTreeMap::new(),
        })
    }

    fn acquire_mask(
        &mut self,
        global_warp_id: usize,
        mask: WarpMask,
        payload: &GlobalExecutionPayload,
    ) -> Result<(), String> {
        if payload.is_bottom() {
            return Ok(());
        }
        // Sibling lanes normally share one synchronized base and differ only
        // in their own lane-local update layer. Join that shared base with the
        // payload once, then overlay each lane's private updates on the shared
        // result: `(base ⊔ updates) ⊔ payload == (base ⊔ payload) ⊔ updates`.
        // This replaces one full join per lane with one per distinct base and
        // leaves the whole mask sharing the joined storage afterwards.
        // Keyed by the storage the lanes share; the actor states keep every
        // Arc behind a key alive for the duration of the call.
        let mut joins = Vec::<(SharedRepresentationKey, SparseLaneClock)>::new();
        let registry = Arc::clone(&self.actors.async_registry);
        for lane in mask {
            let actor = GlobalActor::new(global_warp_id, lane);
            let state = self.actors.get_or_insert_default(actor);
            merge_pending_acquire(
                &mut state.pending_acquire,
                payload.read_observations.values().cloned(),
            );
            let key = state.clock.shared_representation_key();
            let mut joined = if let Some((_, joined)) = joins.iter().find(|(prior, _)| *prior == key)
            {
                joined.clone()
            } else {
                let mut joined = state.clock.without_component_updates();
                joined.merge(&payload.clock);
                joins.push((key, joined.clone()));
                joined
            };
            joined.merge_sparse_improvements(state.clock.component_updates.as_slice());
            state.clock = joined;
            if state.laggard {
                state.laggard = !registry.watermark_dominated_by(&state.clock);
            }
        }
        Ok(())
    }

    fn acquire_mask_from_own_release(
        &mut self,
        global_warp_id: usize,
        mask: WarpMask,
        payload: &GlobalExecutionPayload,
    ) {
        if payload.is_bottom() {
            return;
        }
        let registry = Arc::clone(&self.actors.async_registry);
        for lane in mask {
            let actor = GlobalActor::new(global_warp_id, lane);
            let state = self.actors.get_or_insert_default(actor);
            debug_assert!(
                payload.clock.dominates(&state.clock),
                "a barrier sync payload must include every resuming lane's release"
            );
            state.clock = payload.clock.clone();
            if state.laggard {
                state.laggard = !registry.watermark_dominated_by(&state.clock);
            }
        }
    }

    fn read_versions(
        &mut self,
        shadows: &LockedShadows<'_>,
        operation: &DynamicOpId,
        spans: &[PhysicalByteSpan],
        space: PhysicalAccessSpace,
        semantics: MemoryAccessSemantics,
        clock: &SparseLaneClock,
    ) -> Vec<Arc<GlobalVersion>> {
        if spans.len() != 1 {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: operation.clone(),
                kind: "mixed_or_partial_read_from",
                reason: "scoped global observation has a multi-span footprint".to_string(),
            });
            return Vec::new();
        }
        let span = spans[0];
        let mut versions: Vec<Arc<GlobalVersion>> = Vec::new();
        let complete = self.visit_read_versions_for_element(
            shadows,
            operation,
            span,
            span,
            space,
            semantics,
            clock,
            |current| {
                if let Some(current) = current {
                    if !versions.iter().any(|existing| existing.id == current.id) {
                        versions.push(current);
                    }
                }
            },
        );
        if !complete {
            versions.clear();
        }
        versions
    }

    /// Visit same-element writes contributing actual bytes to this read.
    /// Masked writes may leave several such versions. Different-width writes
    /// supply values but cannot create observation order: their locations do
    /// not completely overlap. The ordinary byte-shadow race checks still
    /// validate every actual access, including those different-width pairs.
    fn visit_read_versions_for_element(
        &mut self,
        shadows: &LockedShadows<'_>,
        operation: &DynamicOpId,
        span: PhysicalByteSpan,
        element: PhysicalByteSpan,
        space: PhysicalAccessSpace,
        semantics: MemoryAccessSemantics,
        clock: &SparseLaneClock,
        mut observe: impl FnMut(Option<Arc<GlobalVersion>>),
    ) -> bool {
        debug_assert_eq!(span.allocation(), element.allocation());
        debug_assert!(
            span.byte_offset() >= element.byte_offset() && span.byte_end() <= element.byte_end()
        );
        if element.byte_offset() % element.byte_len() != 0 {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: operation.clone(),
                kind: "unaligned_read_from",
                reason: format!("scoped global observation is not aligned: {span}"),
            });
            return false;
        }
        let mut cursor = span.byte_offset();
        while cursor < span.byte_end() {
            let stripe = cursor / SHADOW_STRIPE_BYTES;
            let stripe_end = ((stripe + 1) * SHADOW_STRIPE_BYTES).min(span.byte_end());
            let (segment, next) = shadows
                .position(((space, span.allocation()), stripe))
                .map(|position| shadows.cells[position].2.segment_until(cursor, stripe_end))
                .unwrap_or((None, stripe_end));
            debug_assert!(next > cursor, "global read-from scan must advance");
            let current = segment.and_then(|segment| segment.state.current_version.clone());
            let version = current.map(|value| value.unit_version_at(cursor));
            if let Some(version) = &version {
                if version.carrier.span() != element {
                    cursor = next;
                    continue;
                }
                if version.carrier.space() == PhysicalAccessSpace::Shared
                    && semantics.proxy() != version.carrier.semantics.proxy()
                {
                    // Proxy fences order shared conflicts in RaceShadow, but
                    // different proxies cannot form observation order (PTX 8.7).
                    cursor = next;
                    continue;
                }
                if semantics.proxy() != version.carrier.semantics.proxy()
                    && !clock.proxy_bridge_observes(
                        version.carrier.semantics.proxy(),
                        semantics.proxy(),
                        &version.carrier.frontier_actor(),
                        version.carrier.frontier_epoch,
                    )
                {
                    self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                        operation: operation.clone(),
                        kind: "cross_proxy_publication_unmodeled",
                        reason: format!(
                            "{} proxy read observes {} proxy version {} without a modeled proxy edge",
                            semantics.proxy(),
                            version.carrier.semantics.proxy(),
                            version.id,
                        ),
                    });
                    return false;
                }
            }
            observe(version);
            cursor = next;
        }
        true
    }

    /// The writes a wait could still be released by, and how many precede them.
    ///
    /// A word's history outlives one wait: a barrier counter carries every
    /// generation's arrivals. A value from a generation this waiter has
    /// already passed can satisfy the predicate again -- a sense bit flips
    /// back, a counter wraps a threshold -- and taking the earliest such write
    /// would hand the waiter an edge from far behind where it actually is,
    /// which is weaker than what the protocol gave it and makes everything it
    /// reads afterwards look unordered.
    ///
    /// A write this actor has already observed cannot be the one that released
    /// it: the actor is causally after it. The prefix of such writes is
    /// returned as a count rather than dropped, because a pre-tested wait may
    /// legitimately leave on a value it already held -- and then the right
    /// answer is no new edge, not an unexplained exit.
    pub(crate) fn declared_word_candidates(
        &self,
        span: PhysicalByteSpan,
        warp_id: usize,
        lane: usize,
    ) -> (usize, Vec<u64>) {
        let writes = self.shared.declared_word_writes_at(span);
        let actor = GlobalActor::new(warp_id, lane);
        let Some(state) = self.actors.get(&actor) else {
            return (0, writes.iter().map(|write| write.value).collect());
        };
        let skip = writes
            .iter()
            .take_while(|write| {
                let carrier = &write.version.carrier;
                carrier.frontier_epoch_in(&state.clock) >= carrier.frontier_epoch
            })
            .count();
        (skip, writes.iter().map(|write| write.value).collect())
    }

    /// Take an acquiring wait's edge from the write its predicate accepted.
    ///
    /// API §3: the accepted position is the *earliest* write in the word's
    /// history whose value satisfies the predicate, which is the exit that
    /// carries the least. Judging by that one makes the conclusion hold for
    /// every schedule the loop could have left on, not just the one that ran.
    ///
    /// The edge itself is ordinary: the accepted write's version goes through
    /// the same `apply_load_ordering` an acquire load uses, so the payload is
    /// the one accumulated along the read-modify-write chain and the scope
    /// rules are the ones already in force.
    /// Apply a declared wait's ordering, returning the TCGEN execution-ordering
    /// frontier it acquired.
    ///
    /// The wait adjudicates no access, so this is the only path by which the
    /// specialized `tcgen05.fence` frontier can reach the waiting actor; the
    /// batch paths collect the same frontier per lane and hand it to
    /// `merge_tcgen_acquisitions`, and so does this one's caller.
    pub(crate) fn apply_declared_word_wait(
        &mut self,
        operation: &DynamicOpId,
        plan: DeclaredWordWaitPlan,
    ) -> Result<(TcgenLaneFrontiers, SharedLaneFrontiers), String> {
        // The wait adjudicates no access of its own, so this is the only place
        // it can say the word belongs to a protocol. The actor's state is read
        // first because the claim needs the clock the waiter arrived with, and
        // `apply_load_ordering` below is what changes it.
        let actor = GlobalActor::new(plan.warp_id(), plan.lane());
        let mut actor_state = self
            .actors
            .get(&actor)
            .cloned()
            .unwrap_or_else(|| self.actors.empty_state());
        // A wait is a causal event even though its polling loads do not enter
        // the ordinary memory-access checker. Give it its own position before
        // claiming the word, and retain that position on every successful exit
        // so a later barrier or release can publish it. Reusing the preceding
        // access's epoch both rejects ordered first-event waits (epoch zero)
        // and lets a barrier *before* a wait appear to order the wait itself.
        // This local tick adds no cross-actor HB edge; the claim still sees no
        // ordering acquired from the word by this wait.
        actor_state.clock.tick(actor)?;
        self.shared
            .claim_protocol_word(plan.span(), actor, operation, &actor_state.clock);
        let Some(index) = plan.accepted() else {
            if !plan.satisfied_on_entry() {
                // The history could not name the write this predicate
                // accepted, but the wait did perform a load, and that load did
                // read something. An asynchronous publication is the case that
                // reaches here: `st_async.release` and `red_async.release`
                // stage their version when the group is issued and land their
                // bytes when it completes, so value and version never meet in
                // `record_declared_word_writes` and the word's history stays
                // silent about a publication that really happened.
                //
                // Fall back to what the load itself is worth. Taking no edge
                // is the strictly worse failure: it reports the data a correct
                // publication released as a race, inventing a finding, where
                // the ordinary read-from edge is exactly what the
                // hand-written `ld.acquire` loop this wait replaced already
                // got. What is lost is the history's schedule independence --
                // the edge comes from the version this run observed rather
                // than from the earliest write any run could have exited on --
                // so this is parity with the raw spelling, not the stronger
                // guarantee a recorded history gives.
                //
                // The wait is only *unexplained* when that fallback finds
                // nothing either. Reporting it whenever the history is silent
                // would make every asynchronous publication permanently
                // incomplete, which says the checker gave up on a protocol it
                // in fact adjudicated.
                let cells = self.shared.shadows(
                    std::iter::once((PhysicalAccessSpace::Global, plan.span())),
                    false,
                );
                let versions = {
                    let shadows = LockedShadows::lock(&cells);
                    self.read_versions(
                        &shadows,
                        operation,
                        &[plan.span()],
                        PhysicalAccessSpace::Global,
                        plan.semantics(),
                        &actor_state.clock,
                    )
                };
                // A wait the launch value already satisfies needed nobody to
                // publish, and that is a different answer from "a publication
                // happened that the recorder never saw". The predicate tells
                // the two apart: it accepts the launch value in the first case
                // and rejects it in the second -- rejecting it is exactly why
                // such a waiter had to wait at all.
                //
                // Judging by the launch value is the same rule the accepted
                // position follows. That position is the *earliest* write the
                // predicate accepts, so that the conclusion holds for every
                // schedule; the launch value is earlier than every write, and
                // the host's stores precede every actor, so an exit on it
                // carries no edge anyone still owes.
                //
                // `None` means the launch bytes were not retained, and then
                // the conservative answer stands.
                if versions.is_empty() && plan.satisfied_by_launch_value() != Some(true) {
                    self.push_incomplete(RaceCheckIncompleteReason::DeclaredWordWaitUnexplained {
                        operation: operation.clone(),
                    });
                }
                let mut tcgen_acquisition = TcgenFenceFrontier::default();
                let mut lane_shared_acquisition = SharedClockFrontier::default();
                let mut join_cache = AcquireJoinCache::default();
                for version in &versions {
                    self.apply_load_ordering(
                        operation,
                        actor,
                        plan.semantics(),
                        false,
                        version,
                        &mut actor_state,
                        &mut tcgen_acquisition,
                        &mut lane_shared_acquisition,
                        &mut join_cache,
                    );
                }
                self.actors.insert(actor, actor_state);
                let mut acquisitions = TcgenLaneFrontiers::new();
                if !tcgen_acquisition.is_empty() {
                    acquisitions.insert(plan.lane(), tcgen_acquisition);
                }
                let mut acquired_shared = SharedLaneFrontiers::new();
                if !lane_shared_acquisition.is_empty() {
                    acquired_shared.insert(plan.lane(), lane_shared_acquisition);
                }
                return Ok((acquisitions, acquired_shared));
            }
            self.actors.insert(actor, actor_state);
            return Ok((TcgenLaneFrontiers::new(), SharedLaneFrontiers::new()));
        };
        // A relaxed wait builds no acquire edge of its own -- it states that
        // what it consumes is the exit value itself -- but PTX ISA §8.7 gives
        // the second form of the acquire pattern to a strong read followed in
        // program order by an `acq_rel`/`sc` fence, and the load this wait
        // emits is strong (§8.4.2 puts `.volatile` and `.relaxed` in the same
        // class). So it still runs the ordering below, which for a relaxed
        // order only leaves the accepted write's release heads in
        // `pending_acquire` for such a fence to consume. The writer's half of
        // that idiom already composes: a relaxed strong write picks up
        // `release_fence`.
        let writes = self.shared.declared_word_writes_at(plan.span());
        let Some(write) = writes.get(index) else {
            return Err(format!(
                "declared word wait at {operation} accepted write #{index} of {}, which the word's history does not hold",
                writes.len(),
            ));
        };
        let mut tcgen_acquisition = TcgenFenceFrontier::default();
        let mut lane_shared_acquisition = SharedClockFrontier::default();
        let mut join_cache = AcquireJoinCache::default();
        self.apply_load_ordering(
            operation,
            actor,
            plan.semantics(),
            false,
            &write.version,
            &mut actor_state,
            &mut tcgen_acquisition,
            &mut lane_shared_acquisition,
            &mut join_cache,
        );
        self.actors.insert(actor, actor_state);
        let mut acquisitions = TcgenLaneFrontiers::new();
        if !tcgen_acquisition.is_empty() {
            acquisitions.insert(plan.lane(), tcgen_acquisition);
        }
        // The shared frontier the accepted publication carried leaves with the
        // wait, the way a fence's and a batch's do. Dropping it would lose the
        // edge to whatever the publisher put in shared memory before releasing
        // the word -- an omission that shows up as a race on that data, not as
        // a missing report.
        let mut acquired_shared = SharedLaneFrontiers::new();
        if !lane_shared_acquisition.is_empty() {
            acquired_shared.insert(plan.lane(), lane_shared_acquisition);
        }
        Ok((acquisitions, acquired_shared))
    }

    fn apply_load_ordering(
        &mut self,
        operation: &DynamicOpId,
        actor: GlobalActor,
        semantics: MemoryAccessSemantics,
        pure_read: bool,
        version: &Arc<GlobalVersion>,
        actor_state: &mut GlobalActorState,
        tcgen_acquisition: &mut TcgenFenceFrontier,
        shared_acquisition: &mut SharedClockFrontier,
        join_cache: &mut AcquireJoinCache,
    ) {
        // A pure read is the only access that can be waiting on a word: a
        // read-modify-write contributes, and what it reads is its own
        // contribution's predecessor, not a message. Taking delivery of
        // another actor's write that this one did not already hold is what
        // makes the word a protocol.
        //
        // Only a read that does not acquire claims the word. An `ld.acquire`
        // loop is a protocol that explains itself: whichever value it got, it
        // took the edge with it, and the checker adjudicated the handoff
        // rather than being unable to. What it cannot tell apart is a
        // `ld.relaxed`/`ld.volatile` poll -- a wait that retries versus a read
        // that took whatever it got -- and that is the shape worth naming.
        if pure_read
            && semantics.class().is_atomic_class()
            && !semantics.order().has_acquire()
        {
            let carrier = &version.carrier;
            let delivered = carrier.actor() != actor
                && carrier.frontier_epoch_in(&actor_state.clock) < carrier.frontier_epoch;
            self.shared.note_word_delivery(carrier.span(), delivered);
        }
        let acquire_scope = semantics
            .order()
            .has_acquire()
            .then(|| semantics.scope())
            .flatten();
        let cache_hit = acquire_scope.is_some_and(|scope| {
            actor_state
                .last_global_acquire
                .as_ref()
                .is_some_and(|cache| {
                    cache.version_id == version.id
                        && cache.scope == scope
                        && cache.proxy == semantics.proxy()
                })
        });
        if cache_hit
            && version.payload.joined_heads().is_some()
            && version.payload.joined_tcgen().is_some()
            && acquire_scope.is_some_and(|scope| {
                version
                    .payload
                    .every_head_acquirable_by(scope, semantics.proxy())
            })
        {
            // A poller re-reading the version it last acquired at this scope
            // and proxy already holds every head clock and TCGEN frontier the
            // immutable payload carries; repeating the merges changes nothing.
            shared_acquisition.merge(&version.payload.shared_frontier);
            return;
        }
        if semantics_are_mutually_morally_strong(
            self.topology,
            version.carrier.semantics,
            version.carrier.actor(),
            semantics,
            actor,
        ) {
            // A morally-strong read-from pair is an execution-ordering link
            // for the specialized TCGEN fence composition, even when the
            // accesses are `.relaxed` and therefore do not create an ordinary
            // acquire edge. The version payload keeps this specialized
            // frontier separate from the ordinary happens-before clock.
            tcgen_acquisition.merge(&version.payload.tcgen);
        }
        if !semantics.class().can_acquire() {
            return;
        }
        if semantics.order().has_acquire() {
            let Some(scope) = semantics.scope() else {
                return;
            };
            // Every head passes its scope and proxy checks, and both tracked
            // joins are exact: consume the payload in one clock join and one
            // TCGEN frontier merge instead of visiting each head. This is the
            // per-head loop below with every `try_acquire_head` known to
            // succeed, so the results are identical.
            if let (Some(joined), Some(joined_tcgen)) = (
                version.payload.joined_heads(),
                version.payload.joined_tcgen(),
            ) {
                if version
                    .payload
                    .every_head_acquirable_by(scope, semantics.proxy())
                {
                    tcgen_acquisition.merge(joined_tcgen);
                    shared_acquisition.merge(&version.payload.shared_frontier);
                    if !cache_hit {
                        join_cache.join_clocks(
                            actor_state,
                            version,
                            vec![Arc::as_ptr(version) as usize],
                            std::iter::once(joined),
                        );
                    }
                    actor_state.last_global_acquire = Some(GlobalAcquireCache {
                        version_id: version.id,
                        scope,
                        proxy: semantics.proxy(),
                    });
                    return;
                }
            }
            let mut all_heads_acquired = true;
            // Acquired head clocks are joined through the per-batch cache so
            // sibling lanes that share a synchronized base pay for one join.
            // A cross-proxy head consults the acquirer's proxy bridges, which
            // earlier heads may have extended, so pending joins are flushed
            // before such a head is checked to keep the sequential semantics.
            // When every head is same-proxy the checks are order-independent,
            // and acquiring all of them joins the payload's tracked join once
            // instead of one head clock at a time.
            let same_proxy_heads = version
                .payload
                .heads
                .values()
                .all(|head| head.key.proxy == semantics.proxy());
            let mut pending_heads = Vec::<&ReleaseHead>::new();
            for head in version.payload.heads.values() {
                if !cache_hit
                    && !same_proxy_heads
                    && head.key.proxy != semantics.proxy()
                    && !pending_heads.is_empty()
                {
                    join_cache.join_heads(actor_state, version, &pending_heads);
                    pending_heads.clear();
                }
                let acquired = self.try_acquire_head(
                    operation,
                    actor,
                    scope,
                    semantics.proxy(),
                    head,
                    actor_state,
                    tcgen_acquisition,
                    false,
                );
                all_heads_acquired &= acquired;
                if acquired {
                    shared_acquisition.merge(&head.shared_frontier);
                }
                if acquired && !cache_hit {
                    pending_heads.push(&**head);
                }
            }
            if !pending_heads.is_empty() {
                match version.payload.joined_heads() {
                    Some(joined) if same_proxy_heads && all_heads_acquired => {
                        join_cache.join_clocks(
                            actor_state,
                            version,
                            vec![Arc::as_ptr(version) as usize],
                            std::iter::once(joined),
                        );
                    }
                    _ => join_cache.join_heads(actor_state, version, &pending_heads),
                }
            }
            if all_heads_acquired {
                actor_state.last_global_acquire = Some(GlobalAcquireCache {
                    version_id: version.id,
                    scope,
                    proxy: semantics.proxy(),
                });
            }
        } else if semantics.order() == MemoryOrder::Relaxed {
            merge_pending_acquire(
                &mut actor_state.pending_acquire,
                version.payload.heads.values().cloned(),
            );
        }
    }

    fn try_acquire_head(
        &mut self,
        acquire_operation: &DynamicOpId,
        acquire_actor: GlobalActor,
        acquire_scope: MemoryScope,
        acquire_proxy: MemoryProxy,
        head: &ReleaseHead,
        actor_state: &mut GlobalActorState,
        tcgen_acquisition: &mut TcgenFenceFrontier,
        merge_clock: bool,
    ) -> bool {
        if acquire_proxy != head.key.proxy
            && !actor_state.clock.proxy_bridge_observes(
                head.key.proxy,
                acquire_proxy,
                &GlobalFrontierActor::Lane(head.key.actor),
                head.clock.actor_epoch(head.key.actor),
            )
        {
            self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                operation: acquire_operation.clone(),
                kind: "cross_proxy_publication_unmodeled",
                reason: format!(
                    "{} proxy acquire cannot consume {} proxy publication from {}",
                    acquire_proxy, head.key.proxy, head.operation
                ),
            });
            return false;
        }
        let relation = self.actor_relation(head.key.actor, acquire_actor);
        if !self.scope_covers(head.key.scope, head.key.actor, acquire_actor)
            || !self.scope_covers(acquire_scope, acquire_actor, head.key.actor)
        {
            self.shared
                .insert_scope_diagnostic(GlobalScopeMismatchDiagnostic {
                    release_operation: normalized_operation(&head.operation),
                    acquire_operation: normalized_operation(acquire_operation),
                    release_scope: head.key.scope,
                    acquire_scope,
                    release_warp_id: head.key.actor.global_warp_id,
                    release_lane: head.key.actor.lane,
                    acquire_warp_id: acquire_actor.global_warp_id,
                    acquire_lane: acquire_actor.lane,
                    relation,
                });
            return false;
        }
        if merge_clock {
            actor_state.clock.merge(&head.clock);
        }
        tcgen_acquisition.merge(&head.tcgen);
        true
    }

    fn publication_for_write(
        &mut self,
        operation: &DynamicOpId,
        access: &GlobalAccess,
        predecessors: &[Arc<GlobalVersion>],
        release_fence: Option<&Arc<ReleaseHead>>,
        tcgen_publication: &TcgenFenceFrontier,
        shared_frontier: Option<&SharedClockFrontier>,
    ) -> ReleasePayload {
        let mut payload = ReleasePayload::default();
        payload.tcgen.merge(tcgen_publication);
        if access.kind() == PhysicalAccessKind::AtomicReadModifyWrite {
            for predecessor in predecessors {
                let cross_proxy_ordered = predecessor.carrier.semantics.proxy()
                    != access.semantics.proxy()
                    && access.clock.proxy_bridge_observes(
                        predecessor.carrier.semantics.proxy(),
                        access.semantics.proxy(),
                        &predecessor.carrier.frontier_actor(),
                        predecessor.carrier.frontier_epoch,
                    );
                if mutually_morally_strong(self.topology, &predecessor.carrier, access) {
                    payload.extend(&predecessor.payload);
                } else if cross_proxy_ordered {
                    payload.extend_release_heads(&predecessor.payload);
                } else if predecessor.carrier.semantics.proxy() != access.semantics.proxy() {
                    self.push_incomplete(RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
                        operation: operation.clone(),
                        kind: "cross_proxy_rmw_ancestry_unmodeled",
                        reason: format!(
                            "RMW {} cannot inherit proxy-mismatched predecessor version {}",
                            operation, predecessor.id
                        ),
                    });
                }
            }
        }
        if access.semantics.order().has_release() {
            if let Some(scope) = access.semantics.scope() {
                payload.insert(Arc::new(ReleaseHead {
                    key: ReleaseHeadKey {
                        actor: access.actor(),
                        scope,
                        proxy: access.semantics.proxy(),
                    },
                    operation: operation.clone(),
                    clock: access.clock.clone(),
                    tcgen: tcgen_publication.clone(),
                    shared_frontier: shared_frontier.cloned().unwrap_or_default(),
                }));
            }
        } else if access.semantics.order() == MemoryOrder::Relaxed
            && access.semantics.class().is_atomic_class()
        {
            if let Some(head) = release_fence.cloned() {
                payload.insert(head);
            }
        }
        payload
    }

    /// Note a scoped access for the undeclared-protocol claim.
    ///
    /// Per access, not per conflicting pair: the claim is about who reached
    /// the word at all, so it must not depend on which pairs the frontier
    /// happened to compare or on what order they ran in.
    fn record_word_use(&self, access: &GlobalAccess) {
        let strong = access.semantics.class().is_atomic_class();
        // The protocol inventory -- who writes the word, who only reads it --
        // is about strong accesses; a plain one joins no protocol. It is still
        // recorded, because a plain access to a word some wait claims is the
        // one contact the declaration is entitled to refuse.
        self.shared.record_protocol_word_use(
            access.span(),
            access.actor(),
            access.operation(),
            strong && access.kind().writes(),
            strong && access.kind() == PhysicalAccessKind::Read,
            (!strong).then(|| (&access.recorded, &access.clock)),
        );
    }

    fn validate_prepared_accesses(
        &mut self,
        shadows: &LockedShadows<'_>,
        prepared: &[PreparedGlobalAccess],
    ) {
        for prepared in prepared {
            if prepared.access.space() == PhysicalAccessSpace::Global {
                // A declared word is global, so the protocol-word record
                // follows the same gate the validation does.
                self.record_word_use(&prepared.access);
                self.validate_access(shadows, &prepared.access);
            }
        }
        let mut accesses = prepared
            .iter()
            .filter(|prepared| prepared.access.space() == PhysicalAccessSpace::Global)
            .map(|prepared| &prepared.access)
            .collect::<Vec<_>>();
        let key = |access: &&GlobalAccess| {
            let span = access.span();
            (span.allocation(), span.byte_offset(), span.byte_end())
        };
        if !accesses.is_sorted_by_key(key) {
            accesses.sort_unstable_by_key(key);
        }
        for (index, prior) in accesses.iter().enumerate() {
            let prior_span = prior.span();
            for current in &accesses[index + 1..] {
                let current_span = current.span();
                if current_span.allocation() != prior_span.allocation()
                    || current_span.byte_offset() >= prior_span.byte_end()
                {
                    break;
                }
                self.validate_access_pair(prior, current);
            }
        }
    }

    fn validate_access(&mut self, shadows: &LockedShadows<'_>, current: &GlobalAccess) {
        let span = current.span();
        // Candidates are validated where they sit in the byte state. Findings
        // and diagnostics land in order-independent sets, so visiting order
        // does not matter, and no `GlobalAccess` (with its clock and async
        // lease) is ever cloned out of a frontier just to be checked.
        let topology = self.topology;
        let shared = &self.shared;
        shadows.for_each_overlapping(current.space(), span, |shadow, lo, hi| {
            shadow.for_each_candidate(span, lo, hi, current, &mut |prior| {
                validate_recorded_pair_in(topology, shared, prior, current);
            });
        });
    }

    fn validate_access_pair(&mut self, prior: &GlobalAccess, current: &GlobalAccess) {
        validate_batch_pair_in(self.topology, &self.shared, prior, current);
    }
}

/// A recorded access against one in flight. The byte state is committed in
/// happens-before order (see [`RecordedGlobalAccess`]), so only
/// `prior hb current` can order the pair.
fn validate_recorded_pair_in(
    topology: Option<LaunchTopology>,
    shared: &GlobalRaceShared,
    prior: &RecordedGlobalAccess,
    current: &GlobalAccess,
) {
    validate_pair_in(
        topology,
        shared,
        prior,
        current,
        || event_happens_before(prior, current),
        || false,
    );
}

/// Two accesses of one batch, both in flight: neither is committed yet, so
/// happens-before is checked both ways.
fn validate_batch_pair_in(
    topology: Option<LaunchTopology>,
    shared: &GlobalRaceShared,
    prior: &GlobalAccess,
    current: &GlobalAccess,
) {
    validate_pair_in(
        topology,
        shared,
        prior,
        current,
        || event_happens_before(prior, current) || event_happens_before(current, prior),
        || ordinary_event_happens_before(current, prior),
    );
}

/// A load: a read the generic proxy issued.
///
/// This is the side the morally-strong exemption is not entitled to speak for.
/// It takes a value away, and which value it took is decided by the order the
/// two ran in, so whether that order is the right one rests on an agreement
/// the pair does not have -- the pair falls through to happens-before instead.
///
/// The generic proxy is what makes it a load. An async-proxy transfer also
/// reads, but it is `cp.async`/TMA, not `ld`: its ordering is the async
/// lifecycle's -- issue, completion, acquire -- and the exemption over a
/// completed transfer is the one PTX ISA 8.7 states. Treating those as loads
/// reports every relaxed async pair on one address.
fn is_generic_proxy_load(kind: PhysicalAccessKind, semantics: MemoryAccessSemantics) -> bool {
    kind == PhysicalAccessKind::Read && semantics.proxy() == MemoryProxy::Generic
}

fn validate_pair_in(
    topology: Option<LaunchTopology>,
    shared: &GlobalRaceShared,
    prior: &RecordedGlobalAccess,
    current: &GlobalAccess,
    ordered: impl FnOnce() -> bool,
    current_before_prior: impl FnOnce() -> bool,
) {
    let same_fragmented_event = prior.same_frontier(current)
        && prior.operation() == current.operation()
        && prior.lane() == current.lane();
    if same_fragmented_event || !accesses_conflict(prior, current) {
        return;
    }
    // A declared word owns its address (design §2.5). An access that reaches
    // it without going through the primitive is a defect on its own, so it is
    // settled before both early returns below: the morally-strong exemption
    // would otherwise swallow the one shape the declaration exists to catch --
    // a raw `red`/`atom` contribution alongside the primitive's, which is the
    // same instruction and so is mutually morally strong -- and the
    // happens-before return would swallow an ordered raw read.
    // Two mutually morally-strong accesses, split by what the exemption was
    // ever entitled to say. Runs are excluded: their spans are only comparable
    // unit by unit, below.
    if prior.unit_bytes == 0
        && current.unit_bytes == 0
        && mutually_morally_strong(topology, prior, current)
    {
        // Neither side is a plain read: each is one indivisible access to one
        // naturally aligned word -- a single copy that only writes, or a
        // read-modify-write the hardware serializes -- so coherence totally
        // orders them and neither can tear the other. That is the shape PTX
        // ISA 8.7's exemption is reasoning about, and it covers a release
        // publication against the contributions that join its release
        // sequence, not only a read-modify-write against another.
        if !is_generic_proxy_load(prior.kind(), prior.semantics)
            && !is_generic_proxy_load(current.kind(), current.semantics)
        {
            return;
        }
        // A plain read is the one side coherence cannot answer for: it takes a
        // value away, and which value it took is decided by the order the two
        // ran in, so whether that order is the right one depends on an
        // agreement between them. The exemption was never entitled to say that
        // agreement holds, so the pair falls through to happens-before, where
        // an edge either orders it or does not.
        //
        // The spin that used to make this unsafe is no longer here to catch: a
        // declared wait adjudicates no access of its own, so its failed polls
        // -- which race the publisher by construction -- never reach this
        // function. What does reach it is every *other* read of the word: a
        // hand-written seed load beside the primitive, or a reader that skipped
        // the protocol entirely. Those are the reads the declaration exists to
        // adjudicate, and an unordered one is a finding, not an exemption.
    }
    // Happens-before does not depend on the spans, so it is settled once
    // for a run; only the unordered pairs are expanded unit by unit.
    if ordered() {
        return;
    }
    // The failure classification does not depend on the spans either.
    let failure = global_ordering_failure(prior, current, current_before_prior());
    if prior.unit_bytes == 0 && current.unit_bytes == 0 {
        validate_unordered_pair_in(topology, shared, prior, current, failure);
        return;
    }
    // A run's units tile its span, so the units that overlap the other side
    // form one contiguous sub-span: the run clipped to the other access and
    // rounded outward to unit boundaries. One finding over the two clipped
    // spans reports exactly the hull of what one finding per overlapping
    // unit pair reported (the checks below do not depend on the spans),
    // without one record per unit: see `FindingAggregate`.
    let (Some(prior_span), Some(current_span)) = (
        clip_run_to(
            prior.span(),
            prior.unit_bytes,
            prior.unit_origin(),
            current.span(),
        ),
        clip_run_to(
            current.span(),
            current.unit_bytes,
            current.unit_origin(),
            prior.span(),
        ),
    ) else {
        return;
    };
    let prior_unit = prior.with_span(prior_span);
    let current_unit = current.with_span(current_span);
    validate_unordered_pair_in(topology, shared, &prior_unit, &current_unit, failure);
}

/// The units of `run` (of `unit_bytes` each; `0` = one unit) that overlap
/// `other`, as one span; `None` when none does.
fn clip_run_to(
    run: PhysicalByteSpan,
    unit_bytes: u32,
    origin: usize,
    other: PhysicalByteSpan,
) -> Option<PhysicalByteSpan> {
    let lo = run.byte_offset().max(other.byte_offset());
    let hi = run.byte_end().min(other.byte_end());
    if lo >= hi {
        return None;
    }
    if unit_bytes == 0 {
        return Some(run);
    }
    let unit = unit_bytes as usize;
    let start = (origin + (lo - origin) / unit * unit).max(run.byte_offset());
    let end = (origin + (hi - origin).div_ceil(unit) * unit).min(run.byte_end());
    PhysicalByteSpan::new(run.allocation(), start, end - start).ok()
}

/// The exact per-access report for two conflicting, unordered accesses.
fn validate_unordered_pair_in(
    topology: Option<LaunchTopology>,
    shared: &GlobalRaceShared,
    prior: &RecordedGlobalAccess,
    current: &RecordedGlobalAccess,
    ordering_failure: PhysicalRaceOrderingFailure,
) {
    // The same split `validate_pair_in` makes, and for the same reason: the
    // exemption speaks for accesses coherence totally orders, and a plain read
    // is not one of those -- which value it took is decided by the order the
    // two ran in. A pair that got here has already been through
    // happens-before, so its edge -- or the absence of one -- is the answer.
    if mutually_morally_strong(topology, prior, current)
        && !is_generic_proxy_load(prior.kind(), prior.semantics)
        && !is_generic_proxy_load(current.kind(), current.semantics)
    {
        return;
    }
    if atomics_have_scope_mismatch(topology, prior, current) {
        shared.insert_scope_diagnostic(scope_mismatch_diagnostic(topology, prior, current));
        return;
    }
    shared.insert_finding(normalized_finding(prior, current, ordering_failure));
}

impl GlobalRaceState {
    fn register_written_allocations(&mut self, batches: &[PhysicalAccessBatch]) {
        if self.shared.tracked_allocations.is_none() {
            return;
        }
        self.shared.track_allocations(
            batches
                .iter()
                .filter(|batch| {
                    batch.descriptor().space() == PhysicalAccessSpace::Global
                        && batch.descriptor().kind().writes()
                })
                .flat_map(|batch| batch.lanes())
                .flat_map(|lane| lane.footprint().spans())
                .map(|span| span.allocation()),
        );
    }

    fn tracks_span(&self, space: PhysicalAccessSpace, span: PhysicalByteSpan) -> bool {
        space == PhysicalAccessSpace::Shared || self.shared.tracks(span.allocation())
    }

    fn batch_has_tracked_span(&self, batch: &PhysicalAccessBatch) -> bool {
        batch
            .lanes()
            .iter()
            .flat_map(|lane| lane.footprint().spans())
            .copied()
            .any(|span| self.tracks_span(batch.descriptor().space(), span))
    }

    fn has_overlapping_tracked_lane_spans(&self, batch: &PhysicalAccessBatch) -> bool {
        let spans = batch
            .lanes()
            .iter()
            .flat_map(|lane| lane.footprint().spans())
            .copied()
            .filter(|span| self.tracks_span(batch.descriptor().space(), *span))
            .collect::<Vec<_>>();
        spans
            .iter()
            .enumerate()
            .any(|(index, span)| spans[index + 1..].iter().any(|other| span.overlaps(*other)))
    }

    fn actor_relation(&self, left: GlobalActor, right: GlobalActor) -> GlobalActorRelation {
        actor_relation(self.topology, left, right)
    }

    fn scope_covers(&self, scope: MemoryScope, source: GlobalActor, target: GlobalActor) -> bool {
        scope_covers(self.topology, scope, source, target)
    }

    fn push_incomplete(&mut self, reason: RaceCheckIncompleteReason) {
        self.shared.push_incomplete(reason);
    }
}

fn event_happens_before(left: &RecordedGlobalAccess, right: &GlobalAccess) -> bool {
    if left.semantics.proxy() != right.semantics.proxy() {
        return right.clock.proxy_bridge_observes(
            left.semantics.proxy(),
            right.semantics.proxy(),
            &left.frontier_actor(),
            left.frontier_epoch,
        );
    }
    // For an event clock, observing the event's own frontier epoch also
    // observes its complete causal past. This is equivalent to comparing the
    // two complete vector clocks, without scanning every lane component for
    // every conflicting frontier candidate.
    left.frontier_epoch_in(&right.clock) >= left.frontier_epoch
}

/// Lane-level order between two accesses in flight: whether `right`'s clock
/// covers the epoch `left`'s issuing lane had at the access.
fn ordinary_event_happens_before(left: &GlobalAccess, right: &GlobalAccess) -> bool {
    right.clock.actor_epoch(left.actor()) >= left.issue_epoch()
}

fn atomics_have_scope_mismatch(
    topology: Option<LaunchTopology>,
    left: &RecordedGlobalAccess,
    right: &RecordedGlobalAccess,
) -> bool {
    left.semantics.class().is_atomic_class()
        && right.semantics.class().is_atomic_class()
        && left.semantics.order().is_strong()
        && right.semantics.order().is_strong()
        && same_access_elements(left, right)
        && left.semantics.proxy() == right.semantics.proxy()
        && left.semantics.scope().is_some()
        && right.semantics.scope().is_some()
        && (!scope_covers(
            topology,
            left.semantics.scope().expect("checked above"),
            left.actor(),
            right.actor(),
        ) || !scope_covers(
            topology,
            right.semantics.scope().expect("checked above"),
            right.actor(),
            left.actor(),
        ))
}

fn scope_mismatch_diagnostic(
    topology: Option<LaunchTopology>,
    left: &RecordedGlobalAccess,
    right: &RecordedGlobalAccess,
) -> GlobalScopeMismatchDiagnostic {
    let left_is_release = left.semantics.order().has_release();
    let left_is_acquire = left.semantics.order().has_acquire();
    let right_is_release = right.semantics.order().has_release();
    let right_is_acquire = right.semantics.order().has_acquire();
    let (release, acquire) = if left_is_release && right_is_acquire {
        (left, right)
    } else if right_is_release && left_is_acquire {
        (right, left)
    } else if normalized_witness(&left.witness()) <= normalized_witness(&right.witness()) {
        (left, right)
    } else {
        (right, left)
    };
    GlobalScopeMismatchDiagnostic {
        release_operation: normalized_operation(release.operation()),
        acquire_operation: normalized_operation(acquire.operation()),
        release_scope: release
            .semantics
            .scope()
            .expect("scope-mismatched atomic has a scope"),
        acquire_scope: acquire
            .semantics
            .scope()
            .expect("scope-mismatched atomic has a scope"),
        release_warp_id: release.actor().global_warp_id,
        release_lane: release.actor().lane,
        acquire_warp_id: acquire.actor().global_warp_id,
        acquire_lane: acquire.actor().lane,
        relation: actor_relation(topology, release.actor(), acquire.actor()),
    }
}

fn accesses_conflict(left: &RecordedGlobalAccess, right: &RecordedGlobalAccess) -> bool {
    left.span().overlaps(right.span())
        && (left.kind().writes() || right.kind().writes())
}

fn direct_write_requires_version(semantics: MemoryAccessSemantics) -> bool {
    // A weak generic write carries no release payload and contributes no
    // morally-strong RMW ancestry. Its writer frontier is still retained for
    // conflict detection, including against a later atomic observer; keeping
    // a duplicate in `current_version` would add no ordering and makes dense
    // output-only kernels pay for an unused Arc/version object.
    semantics.class().is_atomic_class() || semantics.proxy() != MemoryProxy::Generic
}

pub(crate) fn tcgen_publication_required(descriptor: PhysicalAccessDescriptor) -> bool {
    descriptor.kind().writes() && direct_write_requires_version(descriptor.memory_semantics())
}

fn mutually_morally_strong(
    topology: Option<LaunchTopology>,
    left: &RecordedGlobalAccess,
    right: &RecordedGlobalAccess,
) -> bool {
    same_access_elements(left, right)
        && semantics_are_mutually_morally_strong(
            topology,
            left.semantics,
            left.actor(),
            right.semantics,
            right.actor(),
        )
}

/// Runs may cover different windows, but every conflicting pair must refer to
/// the same complete element. Equal hulls alone do not prove equal elements.
fn same_access_elements(left: &RecordedGlobalAccess, right: &RecordedGlobalAccess) -> bool {
    let width = |access: &RecordedGlobalAccess| {
        if access.unit_bytes == 0 {
            access.byte_len
        } else {
            access.unit_bytes
        }
    };
    let left_width = width(left) as usize;
    left.span().overlaps(right.span())
        && left_width == width(right) as usize
        && left.unit_origin() % left_width == right.unit_origin() % left_width
}

/// PTX ISA 8.7: each side is strong and names a scope that reaches the other.
///
/// `is_atomic_class`, not `is_read_modify_write`. Narrowing this to the
/// read-modify-writes was tried,
/// and it does not hold up: a correct undeclared flag protocol
/// (`st.release.gpu` published, `ld.acquire.gpu` polled in a loop) then reports
/// on the schedules where a poll reads the pre-publication value and stays
/// clean on the schedules where it does not -- measured at 11 errors in 20
/// runs with two workers, 0 in 20 with one, on
/// `global_scoped_message_passing` mode 0. A verdict that moves with the
/// worker count is worse than the hole it closes, and PTX ISA 8.7 counts a
/// scoped single copy as strong, so the exemption is not wrong here, only
/// blunt. Catching an undeclared protocol needs a finding that does not depend
/// on which poll won the race -- an address-level claim like
/// `declared_word_bypassed`, not a happens-before verdict. Until that exists
/// the class stays out of it.
fn semantics_are_mutually_morally_strong(
    topology: Option<LaunchTopology>,
    left: MemoryAccessSemantics,
    left_actor: GlobalActor,
    right: MemoryAccessSemantics,
    right_actor: GlobalActor,
) -> bool {
    left.class().is_atomic_class()
        && right.class().is_atomic_class()
        && left.order().is_strong()
        && right.order().is_strong()
        && left.proxy() == right.proxy()
        && scopes_mutually_cover(topology, left, left_actor, right, right_actor)
}

/// Each side names a scope that reaches the other side's actor.
fn scopes_mutually_cover(
    topology: Option<LaunchTopology>,
    left: MemoryAccessSemantics,
    left_actor: GlobalActor,
    right: MemoryAccessSemantics,
    right_actor: GlobalActor,
) -> bool {
    left.scope()
        .is_some_and(|scope| scope_covers(topology, scope, left_actor, right_actor))
        && right
            .scope()
            .is_some_and(|scope| scope_covers(topology, scope, right_actor, left_actor))
}

fn actor_relation(
    topology: Option<LaunchTopology>,
    left: GlobalActor,
    right: GlobalActor,
) -> GlobalActorRelation {
    match MemoryScope::required_between_warps(topology, left.global_warp_id, right.global_warp_id) {
        MemoryScope::Cta => GlobalActorRelation::SameCta,
        MemoryScope::Cluster => GlobalActorRelation::SameCluster,
        MemoryScope::Gpu | MemoryScope::Sys => GlobalActorRelation::CrossCluster,
    }
}

fn scope_covers(
    topology: Option<LaunchTopology>,
    scope: MemoryScope,
    source: GlobalActor,
    target: GlobalActor,
) -> bool {
    scope
        >= MemoryScope::required_between_warps(
            topology,
            source.global_warp_id,
            target.global_warp_id,
        )
}

pub(crate) fn scope_covers_warps(
    topology: Option<LaunchTopology>,
    scope: MemoryScope,
    source: usize,
    target: usize,
) -> bool {
    // PTX scopes do not distinguish lanes within one warp.
    scope_covers(
        topology,
        scope,
        GlobalActor::new(source, 0),
        GlobalActor::new(target, 0),
    )
}

/// Group sorted, non-overlapping transfer-unit spans into maximal contiguous
/// runs of equal width: `(run span, unit bytes)`. A run of one unit keeps
/// `unit bytes == 0`, i.e. it is the unit itself.
fn coalesce_transfer_runs(mut spans: Vec<PhysicalByteSpan>) -> Vec<(PhysicalByteSpan, u32)> {
    if !spans.is_sorted() {
        spans.sort_unstable();
    }
    let mut runs: Vec<(PhysicalByteSpan, u32)> = Vec::new();
    let mut unit_count = 0_usize;
    for span in spans {
        if let Some((run, unit_bytes)) = runs.last_mut() {
            let width = if *unit_bytes == 0 {
                run.byte_len()
            } else {
                *unit_bytes as usize
            };
            if run.allocation() == span.allocation()
                && run.byte_end() == span.byte_offset()
                && span.byte_len() == width
                && u32::try_from(width).is_ok()
            {
                *run = PhysicalByteSpan::new(
                    run.allocation(),
                    run.byte_offset(),
                    run.byte_len() + span.byte_len(),
                )
                .expect("a transfer run is nonempty and representable");
                *unit_bytes = width as u32;
                unit_count += 1;
                continue;
            }
        }
        runs.push((span, 0));
        unit_count += 1;
    }
    debug_assert!(runs.iter().all(|(run, unit_bytes)| {
        *unit_bytes == 0 || run.byte_len() % *unit_bytes as usize == 0
    }));
    let _ = unit_count;
    runs
}

fn normalized_finding(
    left: &RecordedGlobalAccess,
    right: &RecordedGlobalAccess,
    ordering_failure: PhysicalRaceOrderingFailure,
) -> PhysicalRaceFinding {
    let left = normalized_witness(&left.witness());
    let right = normalized_witness(&right.witness());
    let (prior, current) = if left <= right {
        (&left, &right)
    } else {
        (&right, &left)
    };
    // The failure was classified in processing order; the proxies of a
    // missing bridge follow the reported order.
    let ordering_failure = match ordering_failure {
        PhysicalRaceOrderingFailure::MissingProxyBridge {
            prior_proxy,
            current_proxy,
            prior_domain,
            current_domain,
        } if !std::ptr::eq(prior, &left) => PhysicalRaceOrderingFailure::MissingProxyBridge {
            prior_proxy: current_proxy,
            current_proxy: prior_proxy,
            prior_domain: current_domain,
            current_domain: prior_domain,
        },
        failure => failure,
    };
    let kind = match (prior.kind().writes(), current.kind().writes()) {
        (true, false) => PhysicalRaceKind::WriteRead,
        (false, true) => PhysicalRaceKind::ReadWrite,
        (true, true) => PhysicalRaceKind::WriteWrite,
        (false, false) => unreachable!("a race finding always contains a writer"),
    };
    let start = prior.span().byte_offset().max(current.span().byte_offset());
    let end = prior.span().byte_end().min(current.span().byte_end());
    let overlap = PhysicalByteSpan::new(prior.span().allocation(), start, end - start)
        .expect("overlapping access witnesses have a nonempty representable intersection");
    PhysicalRaceFinding::new(
        kind,
        ordering_failure,
        prior.clone(),
        current.clone(),
        overlap,
    )
}

/// Why `prior` and `current`, which conflict and are unordered, are
/// unordered. `prior` is a committed record without its clock, so the
/// ordinary (lane-level) order is read from `current`'s clock against the
/// epoch `prior`'s issuing lane had at the access; `current_before_prior`
/// is the other direction, which only a caller holding both clocks can
/// settle (`validate_batch_pair_in`) and is `false` otherwise.
fn global_ordering_failure(
    prior: &RecordedGlobalAccess,
    current: &GlobalAccess,
    current_before_prior: bool,
) -> PhysicalRaceOrderingFailure {
    if prior.semantics.proxy() != current.semantics.proxy() {
        return PhysicalRaceOrderingFailure::MissingProxyBridge {
            prior_proxy: prior.semantics.proxy(),
            current_proxy: current.semantics.proxy(),
            prior_domain: PhysicalRaceProxyDomain::Global,
            current_domain: PhysicalRaceProxyDomain::Global,
        };
    }
    let ordinarily_ordered = current_before_prior
        || current.clock.actor_epoch(prior.actor()) >= prior.issue_epoch();
    if !ordinarily_ordered {
        let (prior_actor, current_actor) = (prior.actor(), current.actor());
        if prior_actor.global_warp_id == current_actor.global_warp_id
            && prior_actor.lane != current_actor.lane
        {
            return PhysicalRaceOrderingFailure::MissingSameWarpLaneOrder;
        }
        let participates_in_atomic_order = |access: &RecordedGlobalAccess| {
            access.kind() == PhysicalAccessKind::AtomicReadModifyWrite
                || access.semantics.class().is_atomic_class()
        };
        return if participates_in_atomic_order(prior) || participates_in_atomic_order(current) {
            PhysicalRaceOrderingFailure::MissingReleaseAcquire
        } else {
            PhysicalRaceOrderingFailure::MissingInterActorSynchronization
        };
    }
    if prior.frontier_lease.is_some() || current.frontier_lease.is_some() {
        PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained
    } else {
        PhysicalRaceOrderingFailure::MissingInterActorSynchronization
    }
}

fn normalized_witness(witness: &PhysicalRaceWitness) -> PhysicalRaceWitness {
    PhysicalRaceWitness::new(
        normalized_operation(witness.operation()),
        witness.lane(),
        witness.kind(),
        witness.space(),
        witness.span(),
    )
}

fn normalized_operation(operation: &DynamicOpId) -> DynamicOpId {
    let loop_frames = operation
        .loop_frames()
        .iter()
        .map(|frame| crate::LoopFrame::new(frame.loop_site_id(), 0))
        .collect::<Vec<_>>();
    DynamicOpId::new(
        operation.kernel_index(),
        operation.global_warp_id(),
        0,
        operation.source_op_id(),
        loop_frames,
    )
}

/// The meet of every actor clock in the launch, projected onto lane and async
/// clock components.
///
/// A frontier entry at or below the floor happened-before the current point of
/// every actor, so no access any actor can still perform conflicts with it:
/// the entry (or a barrier payload the floor dominates) can be dropped without
/// changing a single verdict.
///
/// Clocks are accumulated per shared base list, so a base contributes once no
/// matter how many actors share it; each actor's update layer is folded in
/// exactly (a component every actor of the group overrides takes the smallest
/// override, any other component keeps the base value, or nothing).
pub(crate) struct GlobalFloor {
    warp_count: usize,
    lanes: Vec<u64>,
    asyncs: Vec<AsyncClockComponent>,
    bottom: bool,
    finished: bool,
    /// Clocks met that carried no async-component knowledge (they zero the
    /// async floor); reported by the collector trace.
    clocks_without_async: usize,
    groups: HashMap<usize, FloorGroup>,
    seen_async_lists: HashSet<usize>,
    covered_slots: Vec<bool>,
}

/// Every clock that shares one lane-component base list.
struct FloorGroup {
    base: Option<Arc<WarpGroupList>>,
    clocks: usize,
    // Per component index: how many clocks of the group override it, and the
    // smallest override.
    updates: HashMap<usize, (usize, u64)>,
}

impl GlobalFloor {
    pub(crate) fn new(warp_count: usize, async_slot_count: usize) -> Self {
        let top = AsyncClockComponent {
            generation: u32::MAX,
            epoch: u32::MAX,
        };
        Self {
            warp_count,
            lanes: vec![u64::MAX; warp_count * crate::WARP_SIZE],
            asyncs: vec![top; async_slot_count],
            bottom: false,
            finished: false,
            clocks_without_async: 0,
            groups: HashMap::new(),
            seen_async_lists: HashSet::new(),
            covered_slots: vec![false; async_slot_count],
        }
    }

    /// A cheap summary of a finished floor: the count and sum of its lane
    /// epochs and of its async epochs. Two passes with the same summary
    /// dominate the same records (the floor only ever grows between passes,
    /// so an equal summary means an equal floor), and a summary of zero
    /// dominates nothing.
    pub(crate) fn signature(&self) -> (usize, u64, usize, u64) {
        let mut lanes = 0usize;
        let mut lane_sum = 0u64;
        for epoch in &self.lanes {
            if *epoch > 0 && *epoch != u64::MAX {
                lanes += 1;
                lane_sum = lane_sum.wrapping_add(*epoch);
            }
        }
        let mut asyncs = 0usize;
        let mut async_sum = 0u64;
        for component in &self.asyncs {
            if component.epoch > 0 && component.generation != u32::MAX {
                asyncs += 1;
                async_sum = async_sum
                    .wrapping_add(u64::from(component.epoch))
                    .wrapping_add(
                        u64::from(component.generation).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
            }
        }
        (lanes, lane_sum, asyncs, async_sum)
    }

    pub(crate) fn is_bottom(&self) -> bool {
        self.bottom
    }

    /// Account for `clock`; call [`Self::finish`] once every clock is in.
    pub(crate) fn meet(&mut self, clock: &SparseLaneClock) {
        debug_assert!(!self.finished, "a finished floor accepts no more clocks");
        if self.bottom {
            return;
        }
        let (key, base) = match &clock.component_base {
            SparseComponentBase::Empty => (0, None),
            SparseComponentBase::Many(list) => (Arc::as_ptr(list) as usize, Some(Arc::clone(list))),
        };
        let group = self.groups.entry(key).or_insert_with(|| FloorGroup {
            base,
            clocks: 0,
            updates: HashMap::new(),
        });
        group.clocks += 1;
        for (index, epoch) in clock.component_updates.as_slice() {
            let entry = group.updates.entry(*index).or_insert((0, u64::MAX));
            entry.0 += 1;
            entry.1 = entry.1.min(*epoch);
        }
        let arena = clock.arena();
        match &clock.async_components.groups {
            None => {
                self.clocks_without_async += 1;
                self.asyncs.fill(AsyncClockComponent::default())
            }
            Some(list) => {
                if self.seen_async_lists.insert(Arc::as_ptr(list) as usize) {
                    self.covered_slots.fill(false);
                    for (group_index, group) in &list.entries {
                        let group = arena.async_groups.view(*group);
                        for chunk_index in 0..ASYNC_CHUNKS_PER_GROUP {
                            let Some(chunk) = group_slot(group, chunk_index) else {
                                continue;
                            };
                            let chunk = arena.chunks.view(chunk);
                            let base = (*group_index as usize * ASYNC_CHUNKS_PER_GROUP
                                + chunk_index)
                                * ASYNC_CHUNK;
                            for slot in 0..ASYNC_CHUNK {
                                let index = base + slot;
                                if index >= self.asyncs.len() {
                                    break;
                                }
                                let component = unpack_component(chunk.word(slot));
                                self.covered_slots[index] = true;
                                let floor = &mut self.asyncs[index];
                                if component < *floor {
                                    *floor = component;
                                }
                            }
                        }
                    }
                    for (index, covered) in self.covered_slots.iter().enumerate() {
                        if !covered {
                            self.asyncs[index] = AsyncClockComponent::default();
                        }
                    }
                }
            }
        }
    }

    /// Fold the accumulated groups into the lane floors.
    pub(crate) fn finish(&mut self, shared: &GlobalRaceShared) {
        self.finish_in(&shared.async_registry.blocks);
    }

    fn finish_in(&mut self, arena: &BlockArena) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.bottom {
            return;
        }
        let lane_count = self.lanes.len();
        let mut covered = vec![false; lane_count];
        let mut values = vec![0_u64; lane_count];
        for group in self.groups.values() {
            covered.fill(false);
            if let Some(list) = &group.base {
                for (group_index, warp_group) in &list.entries {
                    for slot in 0..WARPS_PER_GROUP {
                        let warp = group_index * WARPS_PER_GROUP + slot;
                        if warp >= self.warp_count {
                            break;
                        }
                        let Some(block) = group_slot(arena.groups.view(*warp_group), slot) else {
                            continue;
                        };
                        let block = arena.blocks.view(block);
                        let base = warp * crate::WARP_SIZE;
                        for (lane, epoch) in (0..crate::WARP_SIZE).map(|lane| (lane, block.word(lane))) {
                            covered[base + lane] = true;
                            values[base + lane] = epoch;
                        }
                    }
                }
            }
            for (&index, &(count, smallest)) in &group.updates {
                if index >= lane_count {
                    continue;
                }
                if count == group.clocks {
                    // Every clock of the group overrides this component.
                    covered[index] = true;
                    values[index] = smallest;
                }
                // Otherwise some clock keeps the base value, which an
                // override never undercuts.
            }
            for index in 0..lane_count {
                let value = if covered[index] { values[index] } else { 0 };
                let floor = &mut self.lanes[index];
                *floor = (*floor).min(value);
            }
        }
        self.groups.clear();
    }

    fn lane_floor(&self, actor: GlobalActor) -> u64 {
        debug_assert!(self.finished, "the floor is only meaningful once finished");
        if self.bottom {
            return 0;
        }
        self.lanes.get(actor.dense_index()).copied().unwrap_or(0)
    }

    fn dominates_access(&self, access: &RecordedGlobalAccess) -> bool {
        if self.bottom {
            return false;
        }
        match &access.frontier_lease {
            None => self.lane_floor(access.actor()) >= access.frontier_epoch,
            Some(lease) => self.asyncs.get(lease.handle.index).is_some_and(|floor| {
                floor.generation == lease.handle.generation
                    && u64::from(floor.epoch) >= access.frontier_epoch
            }),
        }
    }

    /// Whether every component `clock` carries is at or below the floor, so
    /// merging it into any actor's clock could no longer change anything.
    fn dominates_clock(&self, clock: &SparseLaneClock) -> bool {
        debug_assert!(self.finished, "the floor is only meaningful once finished");
        if self.bottom || clock.proxy_bridges.is_some() {
            return false;
        }
        let arena = clock.arena();
        let warp_count = self.warp_count;
        for (group_index, group) in clock.component_base.groups() {
            let group = arena.groups.view(*group);
            for slot in 0..WARPS_PER_GROUP {
                let warp = group_index * WARPS_PER_GROUP + slot;
                let Some(block) = group_slot(group, slot) else {
                    continue;
                };
                if warp >= warp_count {
                    return false;
                }
                let block = arena.blocks.view(block);
                let base = warp * crate::WARP_SIZE;
                if (0..crate::WARP_SIZE).any(|lane| block.word(lane) > self.lanes[base + lane]) {
                    return false;
                }
            }
        }
        for (index, epoch) in clock.component_updates.as_slice() {
            if self.lanes.get(*index).is_none_or(|floor| *epoch > *floor) {
                return false;
            }
        }
        for (group_index, group) in clock.async_components.groups() {
            let group = arena.async_groups.view(*group);
            for chunk_index in 0..ASYNC_CHUNKS_PER_GROUP {
                let Some(chunk) = group_slot(group, chunk_index) else {
                    continue;
                };
                let chunk = arena.chunks.view(chunk);
                let base =
                    (*group_index as usize * ASYNC_CHUNKS_PER_GROUP + chunk_index) * ASYNC_CHUNK;
                for slot in 0..ASYNC_CHUNK {
                    let component = unpack_component(chunk.word(slot));
                    if self
                        .asyncs
                        .get(base + slot)
                        .is_none_or(|floor| component > *floor)
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl ManyGlobalFrontiers {
    fn retire_dominated(&mut self, floor: &GlobalFloor) -> usize {
        let mut removed = 0;
        self.lanes.retain(|_, entries| {
            let before = entries.lanes.len();
            entries
                .lanes
                .retain(|(_, access)| !floor.dominates_access(access));
            let retired = before - entries.lanes.len();
            if retired != 0 {
                removed += retired;
                entries.scanned = None;
            }
            !entries.lanes.is_empty()
        });
        let before = self.asyncs.len();
        self.asyncs
            .retain(|(_, access)| !floor.dominates_access(access));
        removed += before - self.asyncs.len();
        if self.asyncs.len() < before {
            self.asyncs.shrink_to_fit();
        }
        self.len -= removed;
        removed
    }
}

impl GlobalFrontiers {
    fn retire_dominated(&mut self, floor: &GlobalFloor) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_, access) => {
                if floor.dominates_access(access) {
                    *self = Self::Empty;
                    1
                } else {
                    0
                }
            }
            Self::Many(accesses) => {
                let removed = accesses.retire_dominated(floor);
                match accesses.len {
                    0 => *self = Self::Empty,
                    1 => {
                        let (actor, access) = std::mem::take(accesses.as_mut()).into_single();
                        *self = Self::One(actor, access);
                    }
                    _ => {}
                }
                removed
            }
        }
    }
}

impl GlobalByteState {
    fn retire_dominated(&mut self, floor: &GlobalFloor) -> usize {
        self.writers.retire_dominated(floor) + self.readers.retire_dominated(floor)
    }
}

impl GlobalAllocationShadow {
    fn retire_dominated(&mut self, floor: &GlobalFloor) -> usize {
        let mut retired = 0;
        self.segments
            .for_each_value_mut(|segment| retired += segment.state.retire_dominated(floor));
        // A first-touch node maps lanes to bytes by position, so it retires
        // whole or not at all; an emptied node yields no accesses.
        self.first_touches.for_each_value_mut(|node| {
            if node.lanes.is_empty() {
                return;
            }
            let warp = node.operation.global_warp_id();
            let dominated = node.lanes.iter().all(|lane| {
                let actor = GlobalActor::new(warp, usize::from(lane.lane));
                floor.lane_floor(actor) >= lane.epoch
            });
            if dominated {
                retired += node.lanes.len();
                node.lanes = Box::new([]);
            }
        });
        retired
    }
}

impl GlobalRaceShared {
    pub(crate) fn async_slot_count(&self) -> usize {
        self.async_registry
            .inner
            .lock()
            .expect("global racecheck async-clock registry lock was poisoned")
            .generations
            .len()
    }

    /// Drop every frontier entry the floor dominates. Cells another worker is
    /// writing right now are skipped; they get their turn on a later pass.
    pub(crate) fn retire_dominated(&self, floor: &GlobalFloor) -> (usize, usize) {
        let cells = self
            .bytes
            .read()
            .expect("global racecheck byte state lock was poisoned")
            .iter()
            .map(|(key, cell)| (*key, Arc::clone(cell)))
            .collect::<Vec<_>>();
        let (mut retired, mut skipped) = (0, 0);
        for (key, cell) in cells {
            match cell.state.try_write() {
                Ok(mut state) => {
                    // A deferred read waits in the cell's pending log with a
                    // full clock snapshot until a writer drains the log; a
                    // read-only allocation never gets one, so its log — and
                    // every clock node those snapshots pin — would live for
                    // the launch. The collector holds the write side anyway.
                    cell.drain_pending(key.1, &mut state);
                    retired += state.retire_dominated(floor);
                }
                Err(_) => skipped += 1,
            }
        }
        (retired, skipped)
    }
}

impl GlobalRaceState {
    /// Lower `floor` to what every actor of this shard has observed. A lane
    /// that has no state yet has observed nothing, and may still access
    /// memory later, so it makes the floor bottom.
    /// Meets the present actors' clocks into `floor` and returns how many
    /// there were. Lanes without global state are skipped: they never
    /// touched global memory nor acquired a global release, so their clocks
    /// are bottom by construction and would pin every record forever (two
    /// warps per cluster on MegaMoE). A lane that appears after a retirement
    /// is marked a laggard instead and reports an analysis gap on its first
    /// global access unless it has caught up with the retirement watermark.
    pub(crate) fn meet_actor_clocks(&self, floor: &mut GlobalFloor) -> usize {
        let mut present = 0;
        for state in self.actors.slots.iter().flatten() {
            floor.meet(&state.clock);
            present += 1;
        }
        present
    }

    /// Drop barrier payloads every actor already dominates: acquiring a
    /// missing global payload is a no-op, exactly like merging one every
    /// component of which the acquirer already observes.
    pub(crate) fn retire_dominated_barriers(&mut self, floor: &GlobalFloor) -> usize {
        let before = self.physical_barriers.len()
            + self.physical_copy_barriers.len()
            + self.named_barriers.len()
            + self.cluster_barriers.len();
        self.physical_barriers
            .retain(|_, payload| !floor.dominates_clock(&payload.clock));
        // Dominating the copy clock does not prove every thread has observed
        // its read-from heads; keep those until the numeric generation retires.
        self.physical_copy_barriers.retain(|_, payload| {
            !payload.read_observations.is_empty() || !floor.dominates_clock(&payload.clock)
        });
        self.named_barriers
            .retain(|_, payload| !floor.dominates_clock(&payload.clock));
        self.cluster_barriers
            .retain(|_, payload| !floor.dominates_clock(&payload.clock));
        before
            - (self.physical_barriers.len()
                + self.physical_copy_barriers.len()
                + self.named_barriers.len()
                + self.cluster_barriers.len())
    }
}

impl GlobalRaceShared {
    /// Whether the clock-node slabs want a collector pass.
    pub(crate) fn clock_nodes_sweep_due(&self) -> bool {
        self.async_registry.blocks.sweep_due()
    }

    /// Begin a clock-node collection: switch the mark bitmaps and take the
    /// root lists. Only under the collector's quiescent point (every shard's
    /// global state locked).
    pub(crate) fn begin_clock_node_mark(&self) -> MarkSnapshot {
        self.async_registry.blocks.begin_mark()
    }

    /// Mark the nodes `snapshot` reaches; no lock needed.
    pub(crate) fn mark_clock_nodes(&self, snapshot: &MarkSnapshot) {
        self.async_registry.blocks.mark(snapshot);
    }

    /// Free every unmarked node, after the mark; no shard lock needed.
    pub(crate) fn sweep_clock_nodes(&self) {
        self.async_registry.blocks.sweep_nodes();
    }
}



#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::{MemoryAccessClass, OperationKind, PhysicalAccessDescriptor, StaticOpId};

    fn operation(warp: usize, lane: usize, sequence: u64, kind: OperationKind) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(0, warp, sequence, StaticOpId::new(sequence), Vec::new()),
            kind,
            WarpMask::from_bits(1_u32 << lane),
        )
    }

    fn batch(
        warp: usize,
        lane: usize,
        sequence: u64,
        kind: PhysicalAccessKind,
        byte_offset: usize,
        semantics: MemoryAccessSemantics,
    ) -> PhysicalAccessBatch {
        batch_in_allocation(
            PhysicalAllocationId::new(1),
            warp,
            lane,
            sequence,
            kind,
            byte_offset,
            semantics,
        )
    }

    fn batch_in_allocation(
        allocation: PhysicalAllocationId,
        warp: usize,
        lane: usize,
        sequence: u64,
        kind: PhysicalAccessKind,
        byte_offset: usize,
        semantics: MemoryAccessSemantics,
    ) -> PhysicalAccessBatch {
        batch_range_in_allocation(
            allocation,
            warp,
            lane,
            sequence,
            kind,
            byte_offset,
            4,
            semantics,
        )
    }

    fn multi_lane_batch(
        warp: usize,
        active_mask: WarpMask,
        sequence: u64,
        kind: PhysicalAccessKind,
        semantics: MemoryAccessSemantics,
        resolve_byte_offset: impl Fn(usize) -> usize,
    ) -> PhysicalAccessBatch {
        let operation_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        let operation = OperationContext::new(
            DynamicOpId::new(0, warp, sequence, StaticOpId::new(sequence), Vec::new()),
            operation_kind,
            active_mask,
        );
        let descriptor = PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, 4)
            .unwrap()
            .with_memory_semantics(semantics);
        PhysicalAccessBatch::resolve_single_span(operation, descriptor, |provenance| {
            Ok::<_, Infallible>(
                PhysicalByteSpan::new(
                    PhysicalAllocationId::new(1),
                    resolve_byte_offset(provenance.lane()),
                    4,
                )
                .unwrap(),
            )
        })
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_range_in_allocation(
        allocation: PhysicalAllocationId,
        warp: usize,
        lane: usize,
        sequence: u64,
        kind: PhysicalAccessKind,
        byte_offset: usize,
        byte_len: usize,
        semantics: MemoryAccessSemantics,
    ) -> PhysicalAccessBatch {
        let operation_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        let operation = operation(warp, lane, sequence, operation_kind);
        let descriptor = PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, byte_len)
            .unwrap()
            .with_memory_semantics(semantics);
        PhysicalAccessBatch::resolve_single_span(operation, descriptor, |_| {
            Ok::<_, Infallible>(PhysicalByteSpan::new(allocation, byte_offset, byte_len).unwrap())
        })
        .unwrap()
    }

    fn commit(state: &mut GlobalRaceState, batch: &PhysicalAccessBatch) {
        let _ = state
            .before_batch(batch, &TcgenLaneFrontiers::new(), None)
            .unwrap();
        state.after_batch(batch).unwrap();
    }

    /// One access re-executed at the same source position, as a loop does.
    ///
    /// A spin's polls share a `StaticOpId` and differ only in their dynamic
    /// instance; a helper that invents a new source position per poll models a
    /// straight-line program instead, and hides whether findings merge.
    fn same_site_batch_with_semantics(
        warp: usize,
        sequence: u64,
        kind: PhysicalAccessKind,
        byte_offset: usize,
        semantics: MemoryAccessSemantics,
    ) -> PhysicalAccessBatch {
        let operation_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            PhysicalAccessKind::Write => OperationKind::Store,
            PhysicalAccessKind::AtomicReadModifyWrite => OperationKind::Atomic,
        };
        let operation = OperationContext::new(
            DynamicOpId::new(0, warp, sequence, StaticOpId::new(11), Vec::new()),
            operation_kind,
            WarpMask::from_bits(1),
        );
        let descriptor = PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, 4)
            .unwrap()
            .with_memory_semantics(semantics);
        PhysicalAccessBatch::resolve_single_span(operation, descriptor, |_| {
            Ok::<_, Infallible>(
                PhysicalByteSpan::new(PhysicalAllocationId::new(1), byte_offset, 4).unwrap(),
            )
        })
        .unwrap()
    }

    fn same_site_batch(
        warp: usize,
        sequence: u64,
        kind: PhysicalAccessKind,
        byte_offset: usize,
        byte_len: usize,
    ) -> PhysicalAccessBatch {
        let operation_kind = match kind {
            PhysicalAccessKind::Read => OperationKind::Load,
            _ => OperationKind::Store,
        };
        let operation = OperationContext::new(
            DynamicOpId::new(0, warp, sequence, StaticOpId::new(7), Vec::new()),
            operation_kind,
            WarpMask::from_bits(1),
        );
        let descriptor =
            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, byte_len)
                .unwrap()
                .with_memory_semantics(MemoryAccessSemantics::plain());
        PhysicalAccessBatch::resolve_single_span(operation, descriptor, |_| {
            Ok::<_, Infallible>(
                PhysicalByteSpan::new(PhysicalAllocationId::new(1), byte_offset, byte_len).unwrap(),
            )
        })
        .unwrap()
    }

    fn finding_at(
        prior_lane: usize,
        current_lane: usize,
        byte_offset: usize,
        byte_len: usize,
    ) -> PhysicalRaceFinding {
        let span = PhysicalByteSpan::new(PhysicalAllocationId::new(1), byte_offset, byte_len).unwrap();
        let prior = PhysicalRaceWitness::from_parts_shared(
            Arc::new(DynamicOpId::new(0, 0, 0, StaticOpId::new(1), Vec::new())),
            prior_lane,
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Global,
            span,
        );
        let current = PhysicalRaceWitness::from_parts_shared(
            Arc::new(DynamicOpId::new(0, 1, 0, StaticOpId::new(2), Vec::new())),
            current_lane,
            PhysicalAccessKind::Read,
            PhysicalAccessSpace::Global,
            span,
        );
        PhysicalRaceFinding::new(
            PhysicalRaceKind::WriteRead,
            PhysicalRaceOrderingFailure::MissingInterActorSynchronization,
            prior,
            current,
            span,
        )
    }

    #[test]
    fn frontier_records_and_entries_stay_compact() {
        assert!(std::mem::size_of::<RecordedGlobalAccess>() <= 64);
        assert_eq!(std::mem::size_of::<AsyncFrontierEntry>(), 16);
        assert_eq!(std::mem::size_of::<(u8, RecordRef)>(), 16);
    }

    #[test]
    fn signal_diagnostic_retains_the_unordered_access_not_an_earlier_ordered_one() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let plain_actor = GlobalActor::new(0, 0);
        let waiter = GlobalActor::new(1, 0);
        let span = PhysicalByteSpan::new(PhysicalAllocationId::new(1), 0, 4).unwrap();
        commit(&mut state, &batch(0, 0, 1, PhysicalAccessKind::Read, 0,
            MemoryAccessSemantics::plain()));
        // Model synchronization after the first plain read and before the wait.
        let mut clock = state.actors.get(&plain_actor).unwrap().clock.clone();
        clock.tick(waiter).unwrap();
        let wait = operation(1, 0, 2, OperationKind::Load);
        state.shared.claim_protocol_word(span, waiter, wait.id(), &clock);
        assert!(state.shared.claimed_word_bypasses().is_empty());
        // This later access has not acquired the wait event. It is the actual
        // witness, even though the retained per-actor plain slot names seq 1.
        let later = batch(0, 0, 3, PhysicalAccessKind::Read, 0,
            MemoryAccessSemantics::plain());
        commit(&mut state, &later);
        let findings = state.shared.claimed_word_bypasses();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].declared_operation(), wait.id());
        assert_eq!(findings[0].bypassing_operation(), later.operation().id());
    }

    #[test]
    fn swept_clock_nodes_are_reclaimed_and_live_clocks_survive() {
        let registry = Arc::new(AsyncClockRegistry::default());
        let arena = &registry.blocks;
        let live = |slab: &NodeSlab| slab.live();
        let mut kept = SparseLaneClock::new(Arc::clone(&registry));
        kept.tick(GlobalActor::new(0, 3)).unwrap();
        kept.tick(GlobalActor::new(2, 7)).unwrap();
        kept.merge(&kept.clone());
        let lease = registry
            .lease(
                AsyncTokenId::new(
                    operation(0, 0, 1, OperationKind::AsyncIssue).id().clone(),
                    0,
                ),
                0,
            )
            .unwrap();
        kept.tick_async(&lease).unwrap();
        {
            let mut dropped = SparseLaneClock::new(Arc::clone(&registry));
            dropped.tick(GlobalActor::new(1, 5)).unwrap();
            dropped.tick(GlobalActor::new(1, 6)).unwrap();
            dropped.merge(&kept);
            dropped.tick_async(&lease).unwrap();
            dropped.tick_async(&lease).unwrap();
            assert_eq!(dropped.async_component(lease.handle), 3);
        }
        let (blocks, groups, async_groups, chunks) = (
            live(&arena.blocks),
            live(&arena.groups),
            live(&arena.async_groups),
            live(&arena.chunks),
        );
        let snapshot = arena.begin_mark();
        arena.mark(&snapshot);
        drop(snapshot);
        arena.sweep_nodes();
        // Every kind of node the dropped clock held alone is freed.
        assert!(live(&arena.blocks) < blocks);
        assert!(live(&arena.groups) < groups);
        assert!(live(&arena.async_groups) < async_groups);
        assert!(live(&arena.chunks) < chunks);
        // The surviving clock still reads every component it holds.
        assert_eq!(kept.actor_epoch(GlobalActor::new(0, 3)), 1);
        assert_eq!(kept.actor_epoch(GlobalActor::new(2, 7)), 1);
        assert_eq!(kept.async_component(lease.handle), 1);
        // A second sweep with nothing new frees nothing more.
        let live_now = [
            &arena.blocks,
            &arena.groups,
            &arena.async_groups,
            &arena.chunks,
        ]
        .map(live);
        let snapshot = arena.begin_mark();
        arena.mark(&snapshot);
        drop(snapshot);
        arena.sweep_nodes();
        assert_eq!(
            [&arena.blocks, &arena.groups, &arena.async_groups, &arena.chunks].map(live),
            live_now
        );
        // Freed slots are reused by the next allocation.
        let slots = arena.blocks.high();
        let mut later = SparseLaneClock::new(Arc::clone(&registry));
        later.tick(GlobalActor::new(4, 1)).unwrap();
        later.tick(GlobalActor::new(4, 2)).unwrap();
        later.merge(&later.clone());
        assert_eq!(arena.blocks.high(), slots);
    }

    #[test]
    fn a_run_against_a_conflicting_access_yields_one_finding_over_the_overlapping_units() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        // Warp 0 writes eight 4-byte units at [0, 32) as one transfer run.
        let issue = operation(0, 0, 1, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Global,
            32,
        )
        .unwrap()
        .with_memory_semantics(MemoryAccessSemantics::async_proxy());
        let run = PhysicalAccessBatch::resolve_transfer_runs(
            operation(0, 0, 1, OperationKind::AsyncIssue),
            descriptor,
            |_| {
                Ok::<_, Infallible>(vec![PhysicalByteSpan::new(
                    PhysicalAllocationId::new(1),
                    0,
                    32,
                )
                .unwrap()])
            },
            4,
        )
        .unwrap();
        state
            .begin_async_token(&token, &issue, &[], std::slice::from_ref(&run), false)
            .unwrap();
        state
            .complete_async_token(
                &token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&run),
            )
            .unwrap();
        // Warp 1 reads [6, 22) without acquiring the token: units 1..=5 of
        // the run conflict, and they are reported as one finding over [4, 24).
        commit(
            &mut state,
            &batch_range_in_allocation(
                PhysicalAllocationId::new(1),
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                6,
                16,
                MemoryAccessSemantics::plain(),
            ),
        );
        let findings = state.findings().collect::<Vec<_>>();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(findings[0].prior().span().byte_offset(), 4);
        assert_eq!(findings[0].prior().span().byte_len(), 20);
        assert_eq!(findings[0].current().span().byte_offset(), 6);
        assert_eq!(findings[0].current().span().byte_len(), 16);
        assert_eq!(findings[0].overlap().byte_offset(), 6);
        assert_eq!(findings[0].overlap().byte_len(), 16);
    }

    #[test]
    fn findings_of_one_site_are_kept_per_contiguous_range() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &same_site_batch(0, 1, PhysicalAccessKind::Write, 0, 16),
        );
        // Warp 1 reads bytes 0..4 and 8..12 one at a time from one source
        // operation (the per-warp sequence is normalized away in a report).
        for byte in (0..4).chain(8..12) {
            commit(
                &mut state,
                &same_site_batch(1, 10 + byte as u64, PhysicalAccessKind::Read, byte, 1),
            );
        }
        let findings = state.findings().collect::<Vec<_>>();
        assert_eq!(findings.len(), 2);
        let ranges = findings
            .iter()
            .map(|finding| {
                (
                    finding.overlap().byte_offset(),
                    finding.overlap().byte_len(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(ranges, vec![(0, 4), (8, 4)]);
        assert_eq!(findings[0].prior().span().byte_len(), 16);
        assert_eq!(findings[0].current().span().byte_offset(), 0);
        assert_eq!(findings[0].current().span().byte_len(), 4);
        assert_eq!(state.shared.findings_retained(), (2, 0));
    }

    #[test]
    fn retained_findings_are_bounded_and_the_overflow_is_counted() {
        let mut findings = FindingAggregate::with_cap(2);
        assert!(findings.insert(finding_at(0, 0, 0, 4)));
        assert!(findings.insert(finding_at(0, 1, 0, 4)));
        // A third site does not fit; the same finding again, and a range
        // touching a retained one, still merge.
        assert!(!findings.insert(finding_at(0, 2, 0, 4)));
        assert!(findings.insert(finding_at(0, 0, 0, 4)));
        assert!(findings.insert(finding_at(0, 1, 4, 4)));
        assert!(!findings.insert(finding_at(0, 1, 12, 4)));
        assert_eq!((findings.len(), findings.sites.len(), findings.dropped()), (2, 2, 2));
        let retained = findings.findings();
        assert_eq!(retained.len(), 2);
        assert_eq!(retained[1].current().lane(), 1);
        assert_eq!(retained[1].overlap().byte_offset(), 0);
        assert_eq!(retained[1].overlap().byte_len(), 8);
        assert_eq!(retained[1].prior().span().byte_len(), 8);
    }

    #[test]
    fn a_range_bridging_two_retained_ranges_merges_them() {
        let mut findings = FindingAggregate::with_cap(usize::MAX);
        assert!(findings.insert(finding_at(0, 0, 0, 4)));
        assert!(findings.insert(finding_at(0, 0, 8, 4)));
        assert_eq!(findings.len(), 2);
        assert!(findings.insert(finding_at(0, 0, 4, 4)));
        assert_eq!(findings.len(), 1);
        let retained = findings.findings();
        assert_eq!(retained[0].overlap().byte_offset(), 0);
        assert_eq!(retained[0].overlap().byte_len(), 12);
    }

    #[test]
    fn a_declared_word_keeps_every_strong_write_made_to_it() {
        // API §3: an acquiring wait finds the write its predicate first
        // accepts, so the word's writes have to be kept whole -- value and
        // released payload, in order. A publisher is spelled in raw PTX now,
        // so the history is kept for every strong write rather than only for
        // a primitive's; a plain write publishes nothing and is not kept.
        let mut state = GlobalRaceState::new(None);
        let span = PhysicalByteSpan::new(PhysicalAllocationId::new(1), 0, 4).unwrap();
        let publisher = atomic(MemoryOrder::Release, MemoryScope::Gpu);

        for (sequence, value) in [(1_u64, 5_u64), (2, 9)] {
            let batch = multi_lane_batch(
                0,
                WarpMask::from_bits(1),
                sequence,
                PhysicalAccessKind::AtomicReadModifyWrite,
                publisher,
                |_| 0,
            )
            .with_declared_values(vec![value].into());
            commit(&mut state, &batch);
        }

        let history = state.shared.declared_word_writes_at(span);
        assert_eq!(
            history.iter().map(|write| write.value).collect::<Vec<_>>(),
            vec![5, 9],
            "the word's writes are kept in the order the protocol made them"
        );
        assert_eq!(history[0].operation.per_warp_sequence(), 1);
        assert_eq!(history[1].operation.per_warp_sequence(), 2);
        assert!(history.iter().all(|write| write.version.carrier.span() == span));

        // A plain write to the same address adds nothing: it carries neither
        // ordering nor atomicity, so it publishes nothing a wait could name.
        let plain = multi_lane_batch(
            1,
            WarpMask::from_bits(1),
            3,
            PhysicalAccessKind::Write,
            MemoryAccessSemantics::plain(),
            |_| 0,
        );
        commit(&mut state, &plain);
        assert_eq!(state.shared.declared_word_writes_at(span).len(), 2);
    }

    /// One `atom.*` on a flag word: the shape a test uses when it needs a
    /// synchronizing access that is not itself the subject of the check.
    fn atomic(order: MemoryOrder, scope: MemoryScope) -> MemoryAccessSemantics {
        MemoryAccessSemantics::scoped(order, scope, MemoryProxy::Generic, MemoryAccessClass::Atomic)
    }

    /// One scoped `ld`/`st`: strong, but not a read-modify-write.
    fn strong(order: MemoryOrder, scope: MemoryScope) -> MemoryAccessSemantics {
        MemoryAccessSemantics::scoped(order, scope, MemoryProxy::Generic, MemoryAccessClass::Atomic)
    }

    #[test]
    fn sparse_clock_equality_is_independent_of_base_and_delta_representation() {
        let first = GlobalActor::new(0, 0);
        let second = GlobalActor::new(3, 7);
        let async_registry = Arc::new(AsyncClockRegistry::default());
        let mut left = SparseLaneClock::new(Arc::clone(&async_registry));
        left.tick(first).unwrap();
        left.tick(second).unwrap();

        let mut first_clock = SparseLaneClock::new(Arc::clone(&async_registry));
        first_clock.tick(first).unwrap();
        let mut second_clock = SparseLaneClock::new(async_registry);
        second_clock.tick(second).unwrap();
        let mut right = first_clock;
        right.merge(&second_clock);

        assert_eq!(left, right);
        assert_eq!(left.actor_epoch(first), 1);
        assert_eq!(left.actor_epoch(second), 1);
    }

    #[test]
    fn compact_replay_snapshot_index_reuses_exact_clock_representation() {
        let mut state = GlobalRaceState::new(None);
        let clock = SparseLaneClock::default();
        let first = state.intern_replay_clock_snapshot(&clock);
        let second = state.intern_replay_clock_snapshot(&clock.clone());
        assert_eq!(first, second);

        let mut advanced = clock;
        advanced.tick(GlobalActor::new(0, 0)).unwrap();
        let third = state.intern_replay_clock_snapshot(&advanced);
        assert_ne!(first, third);
        assert_eq!(state.replay_clock_snapshots.len(), 2);
    }

    #[test]
    fn bounded_global_replay_oracle_matches_eager_conflict_analysis() {
        let mut eager = GlobalRaceState::new(None);
        let first = batch_range_in_allocation(
            PhysicalAllocationId::new(1),
            0,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        let second = batch_range_in_allocation(
            PhysicalAllocationId::new(1),
            1,
            0,
            1,
            PhysicalAccessKind::AtomicReadModifyWrite,
            0,
            4,
            atomic(MemoryOrder::Relaxed, MemoryScope::Gpu),
        );
        commit(&mut eager, &first);
        commit(&mut eager, &second);

        let replay = eager
            .replay_analysis()
            .expect("bounded replay log should cover this two-event case");
        assert_eq!(eager.findings().collect::<BTreeSet<_>>(), replay.findings);
        assert_eq!(
            eager.scope_diagnostics().collect::<BTreeSet<_>>(),
            replay.scope_diagnostics
        );
        assert_eq!(
            eager.incomplete_reasons().collect::<Vec<_>>(),
            replay.incomplete_reasons
        );
        assert_eq!(replay.batch_count, 2);
        assert_eq!(replay.access_count, 2);
        let storage = eager.replay_storage_stats();
        assert_eq!(storage.batch_count, replay.batch_count);
        assert_eq!(storage.access_count, replay.access_count);
        assert_eq!(storage.snapshot_count, 2);
        // A replay record is more compact than what a live access holds: its
        // shared record plus its clock.
        assert!(
            storage.access_record_bytes
                < std::mem::size_of::<RecordedGlobalAccess>() + std::mem::size_of::<SparseLaneClock>()
        );
        assert!(storage.snapshot_record_bytes > 0);
        assert!(eager.replay_storage_stats().event_count >= 2);
        assert!(!eager.replay_overflowed);
    }

    #[test]
    fn conditional_release_survives_retirement_without_acquiring_newer_writes() {
        let mut state = GlobalRaceState::new(Some(LaunchTopology::new(2, 1, 1).unwrap()));
        let barrier = PhysicalBarrierId::new(7, 0, 0);
        let allocation = PhysicalAllocationId::new(9);
        for generation in 0..21 {
            let write = batch_range_in_allocation(
                allocation,
                0,
                0,
                generation + 1,
                PhysicalAccessKind::Write,
                generation as usize * 4,
                4,
                MemoryAccessSemantics::plain(),
            );
            commit(&mut state, &write);
            state
                .physical_barrier_release(barrier, generation, 0, WarpMask::from_bits(1))
                .unwrap();
            state.retain_physical_barrier_generations(
                barrier,
                generation,
                (generation >= 1).then_some(1),
            );
        }
        assert!(state.physical_barriers.contains_key(&(barrier, 1)));
        assert!(!state.physical_barriers.contains_key(&(barrier, 2)));
        assert!(
            state.physical_barriers.len()
                <= super::super::RETAINED_BARRIER_GENERATIONS as usize + 2
        );
        state
            .physical_barrier_acquire(None, barrier, 1, 1, WarpMask::from_bits(1), true)
            .unwrap();
        let old_read = batch_range_in_allocation(
            allocation,
            1,
            0,
            1,
            PhysicalAccessKind::Read,
            4,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut state, &old_read);
        assert_eq!(state.findings().count(), 0);
        let newer_read = batch_range_in_allocation(
            allocation,
            1,
            0,
            2,
            PhysicalAccessKind::Read,
            80,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut state, &newer_read);
        assert_eq!(state.findings().count(), 1);
        assert_eq!(state.incomplete_reasons().count(), 0);
        let replay = state.replay_analysis().unwrap();
        assert_eq!(replay.findings, state.findings().collect::<BTreeSet<_>>());
        assert!(replay.incomplete_reasons.is_empty());
        state.reset_physical_barriers(&[barrier]);
        state.retain_physical_barrier_generations(barrier, 30, Some(0));
        let wait = operation(0, 1, 3, OperationKind::Barrier);
        state
            .physical_barrier_acquire(Some(wait.id()), barrier, 0, 1, WarpMask::from_bits(1), true)
            .unwrap();
        assert_eq!(state.incomplete_reasons().count(), 0); // A valid bottom release needs no stored clock.
    }

    #[test]
    fn bounded_replay_preserves_a_physical_barrier_without_global_access() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut eager = GlobalRaceState::new(Some(topology));
        let barrier = PhysicalBarrierId::new(7, 0, 0);
        let write = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            0,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        let read = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            1,
            0,
            1,
            PhysicalAccessKind::Read,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut eager, &write);
        eager
            .physical_barrier_release(barrier, 0, 0, WarpMask::from_bits(1))
            .unwrap();
        eager
            .physical_barrier_acquire(None, barrier, 0, 1, WarpMask::from_bits(1), true)
            .unwrap();
        commit(&mut eager, &read);

        assert!(eager.findings().next().is_none());
        let replay = eager
            .replay_analysis()
            .expect("barrier event stream should remain within the bounded oracle");
        assert_eq!(eager.findings().collect::<BTreeSet<_>>(), replay.findings);
        assert_eq!(
            eager.scope_diagnostics().collect::<BTreeSet<_>>(),
            replay.scope_diagnostics
        );
        assert_eq!(
            eager.incomplete_reasons().collect::<Vec<_>>(),
            replay.incomplete_reasons
        );
        assert_eq!(replay.access_count, 2);
        assert!(eager.replay_storage_stats().event_count >= 4);
    }

    #[test]
    fn global_floor_skips_stateless_lanes_and_flags_lanes_that_appear_after_a_retirement() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let barrier = PhysicalBarrierId::new(7, 0, 0);
        let write = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            0,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        let read = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            0,
            1,
            2,
            PhysicalAccessKind::Read,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut state, &write);
        state
            .physical_barrier_release(barrier, 0, 0, WarpMask::from_bits(1))
            .unwrap();
        state
            .physical_barrier_acquire(None, barrier, 0, 0, WarpMask::from_bits(1 << 1), true)
            .unwrap();
        state
            .physical_barrier_acquire(None, barrier, 0, 0, WarpMask::from_bits(1 << 3), true)
            .unwrap();
        commit(&mut state, &read);
        assert!(state.findings().next().is_none());

        // Lanes without global state never touched global memory nor acquired
        // a global release: they are skipped, so the floor is the meet of the
        // three present lanes. All of them observed the write's release, so
        // the write retires; the read is only known to lane 1 and stays.
        let mut floor = GlobalFloor::new(1, state.shared.async_slot_count());
        assert_eq!(state.meet_actor_clocks(&mut floor), 3);
        floor.finish(&state.shared);
        assert!(!floor.is_bottom());
        state.shared.note_retirement_floor(&floor);
        let retired = state.shared.retire_dominated(&floor);
        let again = state.shared.retire_dominated(&floor);
        let barriers = state.retire_dominated_barriers(&floor);
        assert_eq!(retired, (1, 0));
        assert_eq!(again, (0, 0));
        assert_eq!(barriers, 1);

        // Lane 3 was present at the retirement: its read is checked normally.
        let late_read = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            0,
            3,
            3,
            PhysicalAccessKind::Read,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut state, &late_read);
        assert!(state.shared.incomplete_reasons().is_empty());

        // Lane 2 appears only now, with a clock below the watermark: a race
        // against the retired write can no longer be checked, so its access
        // is reported as an analysis gap rather than passed silently.
        let laggard_read = batch_range_in_allocation(
            PhysicalAllocationId::new(9),
            0,
            2,
            4,
            PhysicalAccessKind::Read,
            0,
            4,
            MemoryAccessSemantics::plain(),
        );
        commit(&mut state, &laggard_read);
        assert!(state.findings().next().is_none());
        let reasons = state.shared.incomplete_reasons();
        assert_eq!(reasons.len(), 1);
        assert!(matches!(
            reasons[0],
            RaceCheckIncompleteReason::RetiredRecordsUnobserved { .. }
        ));
    }

    #[test]
    fn merged_sparse_clock_keeps_history_in_an_immutable_shared_base() {
        let first = GlobalActor::new(0, 0);
        let second = GlobalActor::new(3, 7);
        let async_registry = Arc::new(AsyncClockRegistry::default());
        let mut clock = SparseLaneClock::new(Arc::clone(&async_registry));
        clock.tick(first).unwrap();
        let mut second_clock = SparseLaneClock::new(async_registry);
        second_clock.tick(second).unwrap();
        clock.merge(&second_clock);

        let history = clock.clone();
        clock.tick(first).unwrap();

        let (SparseComponentBase::Many(current_base), SparseComponentBase::Many(history_base)) =
            (&clock.component_base, &history.component_base)
        else {
            panic!("a two-actor merge must freeze into a sparse base");
        };
        assert!(Arc::ptr_eq(current_base, history_base));
        assert_eq!(history.actor_epoch(first), 1);
        assert_eq!(clock.actor_epoch(first), 2);
        assert_eq!(clock.actor_epoch(second), 1);
        assert!(matches!(
            clock.component_updates,
            SparseComponentUpdates::One((_, 2))
        ));
    }

    #[test]
    fn sparse_clock_linear_base_merge_preserves_interleaved_epochs_and_deltas() {
        let first = GlobalActor::new(0, 0);
        let middle = GlobalActor::new(1, 7);
        let last = GlobalActor::new(3, 31);
        let async_registry = Arc::new(AsyncClockRegistry::default());

        let mut left = SparseLaneClock::new(Arc::clone(&async_registry));
        for actor in [first, last] {
            let mut update = SparseLaneClock::new(Arc::clone(&async_registry));
            update.tick(actor).unwrap();
            left.merge(&update);
        }
        left.tick(first).unwrap();

        let mut right = SparseLaneClock::new(Arc::clone(&async_registry));
        for actor in [middle, last] {
            let mut update = SparseLaneClock::new(Arc::clone(&async_registry));
            update.tick(actor).unwrap();
            right.merge(&update);
        }
        right.tick(last).unwrap();

        left.merge(&right);

        assert_eq!(left.actor_epoch(first), 2);
        assert_eq!(left.actor_epoch(middle), 1);
        assert_eq!(left.actor_epoch(last), 2);
    }

    #[test]
    fn recycled_async_clock_slot_does_not_alias_a_stale_generation() {
        let registry = Arc::new(AsyncClockRegistry::default());
        let first_token = AsyncTokenId::new(
            operation(0, 0, 1, OperationKind::AsyncIssue).id().clone(),
            0,
        );
        let first_lease = registry.lease(first_token, 0).unwrap();
        let first_handle = first_lease.handle;
        let mut stale = SparseLaneClock::new(Arc::clone(&registry));
        stale.tick_async(&first_lease).unwrap();
        assert_eq!(stale.async_component(first_handle), 1);
        drop(first_lease);
        assert_eq!(registry.slot_counts(), (1, 1));

        let second_token = AsyncTokenId::new(
            operation(0, 0, 2, OperationKind::AsyncIssue).id().clone(),
            0,
        );
        let second_lease = registry.lease(second_token, 0).unwrap();
        let second_handle = second_lease.handle;
        assert_eq!(second_handle.index, first_handle.index);
        assert!(second_handle.generation > first_handle.generation);
        assert_eq!(stale.async_component(second_handle), 0);

        let mut current = stale.clone();
        current.tick_async(&second_lease).unwrap();
        assert_eq!(current.async_component(first_handle), 0);
        assert_eq!(current.async_component(second_handle), 1);
    }

    #[test]
    fn retired_async_clock_slot_is_reclaimed_after_its_frontier_is_superseded() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let destination = |sequence| {
            batch(
                0,
                0,
                sequence,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::async_proxy(),
            )
        };

        let first_issue = operation(0, 0, 1, OperationKind::AsyncIssue);
        let first_token = AsyncTokenId::new(first_issue.id().clone(), 0);
        let first_destination = destination(1);
        state
            .begin_async_token(
                &first_token,
                &first_issue,
                &[],
                std::slice::from_ref(&first_destination),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &first_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&first_destination),
            )
            .unwrap();
        state
            .acquire_async_token(
                &first_issue,
                &first_token,
                AsyncGroupMilestone::FullComplete,
            )
            .unwrap();
        state.retire_async_token(&first_token);
        assert_eq!(state.actors.async_registry.slot_counts(), (1, 0));

        let second_issue = operation(0, 0, 2, OperationKind::AsyncIssue);
        let second_token = AsyncTokenId::new(second_issue.id().clone(), 0);
        let second_destination = destination(2);
        state
            .begin_async_token(
                &second_token,
                &second_issue,
                &[],
                std::slice::from_ref(&second_destination),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &second_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&second_destination),
            )
            .unwrap();

        // The second ordered write replaces both the writer frontier and the
        // version carrier for the first token, releasing its one shared lease.
        assert_eq!(state.actors.async_registry.slot_counts(), (2, 1));
        assert!(state.findings().next().is_none());

        let first_handle = AsyncClockHandle {
            index: 0,
            generation: 1,
        };
        state
            .acquire_async_token(
                &second_issue,
                &second_token,
                AsyncGroupMilestone::FullComplete,
            )
            .unwrap();
        state.retire_async_token(&second_token);
        let third_issue = operation(0, 0, 3, OperationKind::AsyncIssue);
        let third_token = AsyncTokenId::new(third_issue.id().clone(), 0);
        let third_destination = destination(3);
        state
            .begin_async_token(
                &third_token,
                &third_issue,
                &[],
                std::slice::from_ref(&third_destination),
                false,
            )
            .unwrap();
        let third_handle = state
            .async_tokens
            .get(&third_token)
            .expect("third token is active")
            .lease
            .handle;
        assert_eq!(third_handle.index, first_handle.index);
        assert!(third_handle.generation > first_handle.generation);
        assert_eq!(state.actors.async_registry.slot_counts(), (2, 0));
    }

    #[test]
    fn barriers_without_global_events_do_not_manufacture_actor_clocks() {
        let topology = LaunchTopology::new(4, 1, 8).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let barrier_id = PhysicalBarrierId::new(1, 0, 0);

        for warp in 0..topology.warp_count() {
            state
                .physical_barrier_release(barrier_id, warp as u64, warp, WarpMask::FULL)
                .unwrap();
            state
                .physical_barrier_acquire(None, barrier_id, warp as u64, warp, WarpMask::FULL, true)
                .unwrap();
            state.warp_sync(warp, WarpMask::FULL).unwrap();
        }

        assert!(state.actors.is_empty());
        assert!(state.physical_barriers.is_empty());
    }

    #[test]
    fn named_barrier_sync_resume_reuses_the_dominating_release_payload() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let barrier_id = crate::NamedBarrierId::new(0, 3);
        let mask = WarpMask::from_bits(1);
        let first = GlobalActor::new(0, 0);
        let second = GlobalActor::new(1, 0);
        state
            .actors
            .get_or_insert_default(first)
            .clock
            .tick(first)
            .unwrap();
        state
            .actors
            .get_or_insert_default(second)
            .clock
            .tick(second)
            .unwrap();

        state.named_barrier_release(barrier_id, 0, 0, mask).unwrap();
        state.named_barrier_release(barrier_id, 0, 1, mask).unwrap();
        state.named_barrier_acquire(None, barrier_id, 0, 0, mask).unwrap();
        state.named_barrier_acquire(None, barrier_id, 0, 1, mask).unwrap();

        let first_clock = &state.actors.get(&first).unwrap().clock;
        let second_clock = &state.actors.get(&second).unwrap().clock;
        assert_eq!(first_clock.actor_epoch(first), 1);
        assert_eq!(first_clock.actor_epoch(second), 1);
        assert!(first_clock.shares_representation_with(second_clock));
    }

    #[test]
    fn warp_sync_reuses_the_dominating_release_payload() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let first = GlobalActor::new(0, 0);
        let second = GlobalActor::new(0, 1);
        state
            .actors
            .get_or_insert_default(first)
            .clock
            .tick(first)
            .unwrap();
        state
            .actors
            .get_or_insert_default(second)
            .clock
            .tick(second)
            .unwrap();

        state.warp_sync(0, WarpMask::from_bits(0b11)).unwrap();

        let first_clock = &state.actors.get(&first).unwrap().clock;
        let second_clock = &state.actors.get(&second).unwrap().clock;
        assert_eq!(first_clock.actor_epoch(first), 1);
        assert_eq!(first_clock.actor_epoch(second), 1);
        assert!(first_clock.shares_representation_with(second_clock));
    }

    #[test]
    fn physical_barrier_orders_existing_memory_event_frontiers() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let barrier_id = PhysicalBarrierId::new(1, 0, 0);
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        state
            .physical_barrier_release(barrier_id, 0, 0, WarpMask::from_bits(1))
            .unwrap();
        state
            .physical_barrier_acquire(None, barrier_id, 0, 1, WarpMask::from_bits(1), true)
            .unwrap();
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state.findings().next().is_none());
        assert_eq!(
            state
                .actors
                .get(&GlobalActor::new(1, 0))
                .unwrap()
                .clock
                .actor_epoch(GlobalActor::new(0, 0)),
            1,
        );
    }

    #[test]
    fn large_global_footprint_uses_one_interval_not_one_entry_per_byte() {
        const BYTE_LEN: usize = 1 << 20;
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let allocation = PhysicalAllocationId::new(1);
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch_range_in_allocation(
                allocation,
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                BYTE_LEN,
                MemoryAccessSemantics::plain(),
            ),
        );

        let allocation = state.shared.shadow(allocation).unwrap();
        assert_eq!(allocation.tracked_byte_count(), BYTE_LEN);
        assert_eq!(allocation.segments.len(), 0);
        assert_eq!(allocation.first_touches.len(), 1);
    }

    #[test]
    fn dense_first_touch_write_batch_uses_one_interval_with_exact_lane_witnesses() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let allocation = PhysicalAllocationId::new(1);
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &multi_lane_batch(
                0,
                WarpMask::FULL,
                1,
                PhysicalAccessKind::Write,
                MemoryAccessSemantics::plain(),
                |lane| lane * 4,
            ),
        );

        let allocation = state.shared.shadow(allocation).unwrap();
        assert_eq!(allocation.tracked_byte_count(), 32 * 4);
        assert_eq!(allocation.segments.len(), 0);
        assert_eq!(allocation.first_touches.len(), 1);
        let (_, first_touch) = allocation.first_touches.iter().next().unwrap();
        assert_eq!(first_touch.byte_offset, 0);
        assert_eq!(first_touch.byte_end, 32 * 4);
        assert_eq!(first_touch.lanes.len(), 32);
        assert_eq!(first_touch.lanes[17].lane, 17);
        assert_eq!(
            first_touch.byte_offset + 17 * first_touch.byte_width,
            17 * 4,
        );
    }

    #[test]
    fn later_access_reports_exact_lane_from_dense_first_touch_batch() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &multi_lane_batch(
                0,
                WarpMask::FULL,
                1,
                PhysicalAccessKind::Write,
                MemoryAccessSemantics::plain(),
                |lane| lane * 4,
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                17 * 4,
                MemoryAccessSemantics::plain(),
            ),
        );

        let findings = state.findings().collect::<Vec<_>>();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(findings[0].prior().lane(), 17);
        assert_eq!(findings[0].current().lane(), 0);
        assert_eq!(findings[0].overlap().byte_offset(), 17 * 4);
        assert_eq!(findings[0].overlap().byte_len(), 4);
    }

    #[test]
    fn barrier_orders_later_access_after_dense_first_touch_batch() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &multi_lane_batch(
                0,
                WarpMask::FULL,
                1,
                PhysicalAccessKind::Write,
                MemoryAccessSemantics::plain(),
                |lane| lane * 4,
            ),
        );
        let barrier_id = PhysicalBarrierId::new(1, 0, 0);
        state
            .physical_barrier_release(barrier_id, 0, 0, WarpMask::FULL)
            .unwrap();
        state
            .physical_barrier_acquire(None, barrier_id, 0, 1, WarpMask::from_bits(1), true)
            .unwrap();
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                17 * 4,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state.findings().next().is_none());
    }

    #[test]
    fn overlapping_lanes_in_one_batch_are_still_checked() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &multi_lane_batch(
                0,
                WarpMask::from_bits(0b11),
                1,
                PhysicalAccessKind::Write,
                MemoryAccessSemantics::plain(),
                |_| 0,
            ),
        );

        let findings = state.findings().collect::<Vec<_>>();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind(), PhysicalRaceKind::WriteWrite);
        assert_eq!(findings[0].prior().lane(), 0);
        assert_eq!(findings[0].current().lane(), 1);
        assert_eq!(findings[0].overlap().byte_offset(), 0);
        assert_eq!(findings[0].overlap().byte_len(), 4);
    }

    #[test]
    fn exact_release_acquire_publication_orders_data_across_clusters() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        // The second acquire observes the same version with the same
        // scope/proxy and exercises the repeated-version clock-merge cache.
        commit(
            &mut state,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                3,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state.findings().next().is_none());
        assert!(state.scope_diagnostics().next().is_none());
        assert!(state.incomplete_reasons().next().is_none());
    }

    #[test]
    fn sc_fences_order_only_participating_lanes_at_mutually_covering_scopes() {
        let topology = LaunchTopology::new(2, 2, 2).unwrap();
        let check = |order, release_scope, acquire_scope, reader_warp, publisher, acquirer| {
            let shared = Arc::new(GlobalRaceShared::new(None));
            let mut producer =
                GlobalRaceState::for_warp_range(Some(topology), 0, 4, Arc::clone(&shared));
            let mut remote =
                GlobalRaceState::for_warp_range(Some(topology), 4, 4, Arc::clone(&shared));
            commit(
                &mut producer,
                &batch(
                    0,
                    0,
                    1,
                    PhysicalAccessKind::Write,
                    0,
                    MemoryAccessSemantics::plain(),
                ),
            );
            producer
                .fence(
                    &operation(0, publisher, 2, OperationKind::Fence),
                    MemoryFenceEffect::new(order, release_scope, MemoryProxy::Generic),
                    &TcgenLaneFrontiers::new(),
                    None,
                )
                .unwrap();
            let consumer = if reader_warp < 4 {
                &mut producer
            } else {
                &mut remote
            };
            consumer
                .fence(
                    &operation(reader_warp, acquirer, 3, OperationKind::Fence),
                    MemoryFenceEffect::new(order, acquire_scope, MemoryProxy::Generic),
                    &TcgenLaneFrontiers::new(),
                    None,
                )
                .unwrap();
            commit(
                consumer,
                &batch(
                    reader_warp,
                    2,
                    4,
                    PhysicalAccessKind::Read,
                    0,
                    MemoryAccessSemantics::plain(),
                ),
            );
            let required = match reader_warp {
                0 | 1 => MemoryScope::Cta,
                2 => MemoryScope::Cluster,
                4 => MemoryScope::Gpu,
                _ => unreachable!(),
            };
            let ordered = order == MemoryOrder::Sc
                && release_scope >= required
                && acquire_scope >= required
                && publisher == 0
                && acquirer == 2;
            assert_eq!(
                shared.findings().is_empty(), ordered,
                "{order:?} {release_scope:?}->{acquire_scope:?}, warp {reader_warp}, lanes {publisher}->{acquirer}",
            );
            // Unrelated narrow fences are not an invalid release/read-from pair.
            assert!(shared.scope_diagnostics().is_empty());
            assert!(shared.incomplete_reasons().is_empty());
        };
        use MemoryScope::{Cluster, Cta, Gpu, Sys};
        for reader_warp in [0, 1, 2, 4] {
            // Each scope boundary in both directions; no Cartesian product
            // of wider scopes that all take the same containment branch.
            for (release, acquire) in [(Cta, Cta), (Cta, Sys), (Sys, Cta),
                (Cluster, Cluster), (Cluster, Sys), (Sys, Cluster), (Gpu, Gpu), (Sys, Sys)]
            {
                check(MemoryOrder::Sc, release, acquire, reader_warp, 0, 2);
            }
            check(MemoryOrder::AcqRel, Sys, Sys, reader_warp, 0, 2);
            for (publisher, acquirer) in [(1, 2), (0, 3)] {
                check(
                    MemoryOrder::Sc,
                    MemoryScope::Gpu,
                    MemoryScope::Gpu,
                    reader_warp,
                    publisher,
                    acquirer,
                );
            }
        }
    }

    #[test]
    fn relaxed_publication_does_not_invent_a_release_edge() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Relaxed, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        let mut findings = state.findings().collect::<Vec<_>>();
        findings.sort_by_key(|finding| finding.overlap().byte_offset());
        // Byte 0 is the payload: the relaxed publication carries no release,
        // so the acquire on the flag has nothing to synchronize with and the
        // payload is unordered. Byte 4 is the flag itself: one side of that
        // pair is a load, and which value it took is decided by the order the
        // two ran in, so the exemption does not speak for it either.
        assert_eq!(findings.len(), 2);
        assert!(findings
            .iter()
            .all(|finding| finding.kind() == PhysicalRaceKind::WriteRead));
        assert_eq!(findings[0].overlap().byte_offset(), 0);
        assert_eq!(findings[1].overlap().byte_offset(), 4);
    }

    #[test]
    fn insufficient_scope_keeps_the_race_and_records_actor_relation() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Cluster),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Cluster),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state
            .findings()
            .any(|finding| finding.overlap().byte_offset() == 0));
        let diagnostics = state.scope_diagnostics().collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].release_scope(), MemoryScope::Cluster);
        assert_eq!(diagnostics[0].acquire_scope(), MemoryScope::Cluster);
        assert_eq!(diagnostics[0].relation(), GlobalActorRelation::CrossCluster);
    }

    #[test]
    fn publication_is_lane_precise_and_warp_sync_transfers_it() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let mut unsynchronized = GlobalRaceState::new(Some(topology));
        commit(
            &mut unsynchronized,
            &batch(
                0,
                1,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        commit(
            &mut unsynchronized,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut unsynchronized,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut unsynchronized,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        assert!(unsynchronized.findings().next().is_some());

        let mut synchronized = GlobalRaceState::new(Some(topology));
        commit(
            &mut synchronized,
            &batch(
                0,
                1,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        synchronized
            .warp_sync(0, WarpMask::from_bits(0b11))
            .unwrap();
        commit(
            &mut synchronized,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut synchronized,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut synchronized,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        assert!(synchronized.findings().next().is_none());
    }

    #[test]
    fn fence_halves_bridge_volatile_accesses_without_sc_execution_edges() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        let producer_fence = operation(0, 0, 2, OperationKind::Fence);
        state
            .fence(
                &producer_fence,
                MemoryFenceEffect::new(MemoryOrder::Sc, MemoryScope::Gpu, MemoryProxy::Generic),
                &TcgenLaneFrontiers::new(),
                None,
            )
            .unwrap();
        commit(
            &mut state,
            &batch(
                0,
                0,
                3,
                PhysicalAccessKind::Write,
                4,
                MemoryAccessSemantics::volatile(),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                MemoryAccessSemantics::volatile(),
            ),
        );
        let consumer_fence = operation(1, 0, 2, OperationKind::Fence);
        state
            .fence(
                &consumer_fence,
                MemoryFenceEffect::new(MemoryOrder::Sc, MemoryScope::Gpu, MemoryProxy::Generic),
                &TcgenLaneFrontiers::new(),
                None,
            )
            .unwrap();
        commit(
            &mut state,
            &batch(
                1,
                0,
                3,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        // The data word is bridged: the fence halves carry the plain write to
        // the plain read with no execution edge between the volatile pair.
        // The flag word itself is not, and is not meant to be. A volatile
        // write against a volatile read is two single copies, not a
        // read-modify-write, so it takes no exemption from PTX ISA 8.7 and is
        // judged on happens-before -- which relaxed accesses do not create.
        // Declaring the flag is what states the edge; raw, it is a race.
        let findings = state.findings().collect::<Vec<_>>();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind(), PhysicalRaceKind::WriteRead);
        assert_eq!(findings[0].overlap().byte_offset(), 4);
        assert!(state.incomplete_reasons().next().is_none());
    }

    #[test]
    fn barrier_without_global_projection_acquires_the_bottom_clock() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));

        state
            .physical_barrier_acquire(None,
                PhysicalBarrierId::new(1, 0, 1),
                0,
                1,
                WarpMask::from_bits(1),
                true,
            )
            .unwrap();

        assert!(state.findings().next().is_none());
        assert!(state.incomplete_reasons().next().is_none());
    }

    #[test]
    fn release_payload_propagates_only_through_exact_rmw_predecessors() {
        let topology = LaunchTopology::new(3, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::AtomicReadModifyWrite,
                4,
                atomic(MemoryOrder::Relaxed, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                2,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                2,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state.findings().next().is_none());
    }

    #[test]
    fn repeated_relaxed_polls_keep_state_bounded_by_actors_and_bytes() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        for sequence in 1..=1_000_000 {
            commit(
                &mut state,
                &same_site_batch_with_semantics(
                    1,
                    sequence,
                    PhysicalAccessKind::Read,
                    4,
                    atomic(MemoryOrder::Relaxed, MemoryScope::Gpu),
                ),
            );
        }
        let (actors, bytes, pending) = state.retained_state_counts();
        assert_eq!(actors, 2);
        assert_eq!(bytes, 4);
        assert_eq!(pending, 1);
        // A relaxed poll is a plain read, so the pair goes to happens-before
        // and an unordered one is reported -- which is the answer that asks
        // the author for `wait_until`. What must not grow with the poll
        // count is the record of it: one merged finding, not one per read.
        assert_eq!(state.findings().count(), 1);
    }

    #[test]
    fn write_allocation_filter_skips_read_only_input_shadow() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let output = PhysicalAllocationId::new(2);
        let mut state = GlobalRaceState::with_tracked_allocations(
            Some(topology),
            Some(BTreeSet::from([output])),
        );
        commit(
            &mut state,
            &batch_in_allocation(
                PhysicalAllocationId::new(1),
                0,
                0,
                1,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        assert_eq!(state.retained_state_counts(), (0, 0, 0));

        commit(
            &mut state,
            &batch_in_allocation(
                output,
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        assert_eq!(state.retained_state_counts(), (1, 4, 0));
    }

    #[test]
    fn overlapping_multi_lane_rmw_is_not_a_coverage_gap() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 1, StaticOpId::new(1), Vec::new()),
            OperationKind::Atomic,
            WarpMask::from_bits(0b11),
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::AtomicReadModifyWrite,
            PhysicalAccessSpace::Global,
            4,
        )
        .unwrap()
        .with_memory_semantics(atomic(MemoryOrder::Relaxed, MemoryScope::Gpu));
        let rmw = PhysicalAccessBatch::resolve_single_span(operation, descriptor, |_| {
            Ok::<_, Infallible>(PhysicalByteSpan::new(PhysicalAllocationId::new(1), 0, 4).unwrap())
        })
        .unwrap();

        commit(&mut state, &rmw);

        // Two lanes RMW the same span, so their serialization order -- and with
        // it each lane's read-from version -- is unconstrained. That withholds
        // the exact predecessor, which only removes happens-before and cannot
        // make the verdict unsound. Atomicity already excludes a race between
        // the lanes, so this is not a coverage gap and must not be reported as
        // one.
        assert_eq!(state.incomplete_reasons().count(), 0);
    }

    // Tests issue real async stores; this helper only shares the identical
    // begin/full-completion sequence, never an observation or acquire.
    fn complete_store(state: &mut GlobalRaceState, store: &PhysicalAccessBatch) {
        let token = AsyncTokenId::new(store.operation().id().clone(), 0);
        let writes = std::slice::from_ref(store);
        state
            .begin_async_token(&token, store.operation(), &[], writes, false)
            .unwrap();
        state
            .complete_async_token(&token, AsyncGroupMilestone::FullComplete, writes)
            .unwrap();
    }

    fn gpu_fence(state: &mut GlobalRaceState, warp: usize, sequence: u64, order: MemoryOrder) {
        state
            .fence(
                &operation(warp, 0, sequence, OperationKind::Fence),
                MemoryFenceEffect::new(order, MemoryScope::Gpu, MemoryProxy::Generic),
                &TcgenLaneFrontiers::new(),
                None,
            )
            .unwrap();
    }

    #[test]
    fn masked_element_read_observes_each_contributing_release() {
        for (second_release, same_producer) in [(false, false), (true, false), (true, true)] {
            let mut state = GlobalRaceState::new(Some(LaunchTopology::new(3, 1, 1).unwrap()));
            let semantics = MemoryAccessSemantics::scoped(
                MemoryOrder::Relaxed,
                MemoryScope::Gpu,
                MemoryProxy::Async,
                MemoryAccessClass::Atomic,
            );
            let transfer = |warp, sequence, kind, start, width| {
                PhysicalAccessBatch::resolve_single_span(
                    operation(warp, 0, sequence, OperationKind::AsyncIssue),
                    PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, width)
                        .unwrap()
                        .with_memory_semantics(semantics),
                    |_| {
                        Ok::<_, Infallible>(
                            PhysicalByteSpan::new(PhysicalAllocationId::new(1), start, width)
                                .unwrap(),
                        )
                    },
                )
                .unwrap()
                .with_aligned_transfer_units(16)
            };
            for index in 0..2 {
                let warp = if same_producer { 0 } else { index };
                let sequence = index as u64 * 3;
                commit(
                    &mut state,
                    &batch(
                        warp,
                        0,
                        sequence + 1,
                        PhysicalAccessKind::Write,
                        index * 4,
                        MemoryAccessSemantics::plain(),
                    ),
                );
                if index == 0 || second_release {
                    gpu_fence(&mut state, warp, sequence + 2, MemoryOrder::Release);
                }
                // Address-order scanning sees the newer release first. An older
                // head from the same lane must not overwrite that observation.
                let store = transfer(
                    warp,
                    sequence + 3,
                    PhysicalAccessKind::Write,
                    24 - index * 8,
                    8,
                );
                complete_store(&mut state, &store);
            }
            let load = transfer(2, 3, PhysicalAccessKind::Read, 16, 16);
            // A direct whole-element observation must preserve both byte
            // contributors too, rather than rejecting or fabricating one
            // carrier. RMW publication inherits exactly these release heads.
            let span = PhysicalByteSpan::new(PhysicalAllocationId::new(1), 16, 16).unwrap();
            let cells = state
                .shared
                .shadows([(PhysicalAccessSpace::Global, span)], false);
            let versions = state.read_versions(
                &LockedShadows::lock(&cells),
                load.operation().id(),
                &[span],
                PhysicalAccessSpace::Global,
                semantics,
                &state.actors.empty_clock(),
            );
            assert_eq!(versions.len(), 2);
            assert_ne!(versions[0].id, versions[1].id);
            assert_eq!(versions[0].carrier.span(), span);
            assert_eq!(versions[1].carrier.span(), span);
            let token = AsyncTokenId::new(load.operation().id().clone(), 0);
            state
                .begin_async_token(
                    &token,
                    load.operation(),
                    std::slice::from_ref(&load),
                    &[],
                    true,
                )
                .unwrap();
            state
                .complete_async_token(&token, AsyncGroupMilestone::FullComplete, &[])
                .unwrap();
            state
                .acquire_async_token(
                    &operation(2, 0, 4, OperationKind::Barrier),
                    &token,
                    AsyncGroupMilestone::FullComplete,
                )
                .unwrap();
            assert_eq!(
                state
                    .actors
                    .get(&GlobalActor::new(2, 0))
                    .unwrap()
                    .pending_acquire
                    .len(),
                1 + usize::from(second_release && !same_producer)
            );
            gpu_fence(&mut state, 2, 5, MemoryOrder::Acquire);
            for index in 0..2 {
                commit(
                    &mut state,
                    &batch(
                        2,
                        0,
                        6 + index as u64,
                        PhysicalAccessKind::Read,
                        index * 4,
                        MemoryAccessSemantics::plain(),
                    ),
                );
            }
            assert_eq!(state.findings().count(), usize::from(!second_release));
            assert_eq!(state.incomplete_reasons().count(), 0);
            let replay = state.replay_analysis().unwrap();
            assert_eq!(replay.findings.len(), state.findings().count());
            assert!(replay.incomplete_reasons.is_empty());
        }
    }

    #[test]
    fn strong_async_read_requires_completion_observation_then_acquire_fence() {
        // None = group wait, Some = mbarrier query with acquire/relaxed semantics.
        for (barrier_wait, clipped, masked) in
            [None, Some(true), Some(false)]
                .into_iter()
                .flat_map(|wait| {
                    [(false, false), (true, false), (false, true), (true, true)]
                        .map(|(clipped, masked)| (wait, clipped, masked))
                })
        {
            for observe in [false, true] {
                // The wait selector is unused when the thread observes nothing.
                if !observe && barrier_wait.is_some() {
                    continue;
                }
                for acquire in [false, true] {
                    let mut state =
                        GlobalRaceState::new(Some(LaunchTopology::new(2, 1, 1).unwrap()));
                    commit(
                        &mut state,
                        &batch(
                            0,
                            0,
                            1,
                            PhysicalAccessKind::Write,
                            0,
                            MemoryAccessSemantics::plain(),
                        ),
                    );
                    gpu_fence(&mut state, 0, 2, MemoryOrder::Release);
                    let semantics = MemoryAccessSemantics::scoped(
                        MemoryOrder::Relaxed,
                        MemoryScope::Gpu,
                        MemoryProxy::Async,
                        MemoryAccessClass::Atomic,
                    );
                    let transfer = |warp, sequence, kind: PhysicalAccessKind| {
                        let (start, width) = if clipped && kind.reads() {
                            (19, 24)
                        } else {
                            (16, 32)
                        };
                        let spans = if masked && kind.writes() {
                            [(16, 1), (31, 2), (47, 1)]
                                .into_iter()
                                .map(|(start, width)| {
                                    PhysicalByteSpan::new(
                                        PhysicalAllocationId::new(1),
                                        start,
                                        width,
                                    )
                                    .unwrap()
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![
                                PhysicalByteSpan::new(PhysicalAllocationId::new(1), start, width)
                                    .unwrap(),
                            ]
                        };
                        let width = spans.iter().map(|span| span.byte_len()).sum();
                        PhysicalAccessBatch::resolve_unmerged(
                            operation(warp, 0, sequence, OperationKind::AsyncIssue),
                            PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Global, width)
                                .unwrap()
                                .with_memory_semantics(semantics),
                            |_| Ok::<_, Infallible>(spans.clone()),
                        )
                        .unwrap()
                        .with_aligned_transfer_units(16)
                    };
                    let store = transfer(0, 3, PhysicalAccessKind::Write);
                    complete_store(&mut state, &store);
                    let load = transfer(1, 1, PhysicalAccessKind::Read);
                    let consumer = AsyncTokenId::new(load.operation().id().clone(), 0);
                    state
                        .begin_async_token(
                            &consumer,
                            load.operation(),
                            std::slice::from_ref(&load),
                            &[],
                            true,
                        )
                        .unwrap();
                    state
                        .complete_async_token(&consumer, AsyncGroupMilestone::FullComplete, &[])
                        .unwrap();
                    let actor = GlobalActor::new(1, 0);
                    assert!(
                        state.actors.get(&actor).unwrap().pending_acquire.is_empty(),
                        "numeric completion alone is not an observation by the thread"
                    );
                    if observe {
                        let wait = operation(1, 0, 2, OperationKind::Barrier);
                        if let Some(acquire_wait) = barrier_wait {
                            let barrier = PhysicalBarrierId::new(7, 1, 0);
                            state
                                .publish_async_token_to_physical_barrier(&consumer, barrier, 0)
                                .unwrap();
                            // Clock dominance alone must not collect the read
                            // observations before the querying thread sees them.
                            let mut floor = GlobalFloor::new(2, state.shared.async_slot_count());
                            floor.meet(&state.physical_copy_barriers[&(barrier, 0)].clock);
                            floor.finish(&state.shared);
                            state.retire_dominated_barriers(&floor);
                            assert!(state.physical_copy_barriers.contains_key(&(barrier, 0)));
                            state
                                .physical_barrier_acquire(
                                    Some(wait.id()),
                                    barrier,
                                    0,
                                    1,
                                    WarpMask::from_bits(1),
                                    acquire_wait,
                                )
                                .unwrap();
                        } else {
                            state
                                .acquire_async_token(
                                    &wait,
                                    &consumer,
                                    AsyncGroupMilestone::FullComplete,
                                )
                                .unwrap();
                        }
                        assert_eq!(state.actors.get(&actor).unwrap().pending_acquire.len(), 1);
                    }
                    if acquire {
                        gpu_fence(&mut state, 1, 3, MemoryOrder::Acquire);
                    }
                    commit(
                        &mut state,
                        &batch(
                            1,
                            0,
                            4,
                            PhysicalAccessKind::Read,
                            0,
                            MemoryAccessSemantics::plain(),
                        ),
                    );
                    assert_eq!(state.findings().count(), usize::from(!(observe && acquire)), "clipped={clipped}, wait={barrier_wait:?}, observe={observe}, acquire={acquire}: {:?}", state.findings().collect::<Vec<_>>());
                    assert_eq!(state.incomplete_reasons().count(), 0);
                    let replay = state.replay_analysis().unwrap();
                    assert_eq!(replay.findings.len(), state.findings().count());
                    assert!(replay.incomplete_reasons.is_empty());
                }
            }
        }
    }

    #[test]
    fn shared_async_read_keeps_issue_version_and_weak_overwrites_invalidate_it() {
        let shared_batch = |warp, sequence, kind, semantics| {
            let op = operation(warp, 0, sequence, OperationKind::AsyncIssue);
            let descriptor = PhysicalAccessDescriptor::new(kind, PhysicalAccessSpace::Shared, 16)
                .unwrap()
                .with_memory_semantics(semantics);
            PhysicalAccessBatch::resolve_single_span(op, descriptor, |_| {
                Ok::<_, Infallible>(
                    PhysicalByteSpan::new(PhysicalAllocationId::new(7), 0, 16).unwrap(),
                )
            })
            .unwrap()
        };
        let semantics = MemoryAccessSemantics::scoped(
            MemoryOrder::Relaxed,
            MemoryScope::Cta,
            MemoryProxy::Async,
            MemoryAccessClass::Atomic,
        );
        let publish = |state: &mut GlobalRaceState, warp, epoch| {
            state
                .fence(
                    &operation(warp, 0, 1, OperationKind::Fence),
                    MemoryFenceEffect::new(
                        MemoryOrder::Release,
                        MemoryScope::Cta,
                        MemoryProxy::Generic,
                    ),
                    &TcgenLaneFrontiers::new(),
                    Some(&BTreeMap::from([(
                        0,
                        SharedClockFrontier::single(warp, [epoch; crate::WARP_SIZE]),
                    )])),
                )
                .unwrap();
            let write = shared_batch(warp, 2, PhysicalAccessKind::Write, semantics);
            complete_store(state, &write);
        };
        for overwrite in 0..3 {
            let invalidate_before_issue = overwrite != 0;
            let mut state = GlobalRaceState::new(Some(LaunchTopology::new(1, 1, 4).unwrap()));
            publish(&mut state, 0, 7);
            if overwrite == 1 {
                commit(
                    &mut state,
                    &shared_batch(
                        0,
                        3,
                        PhysicalAccessKind::Write,
                        MemoryAccessSemantics::plain(),
                    ),
                );
            } else if overwrite == 2 {
                let write = shared_batch(
                    0,
                    3,
                    PhysicalAccessKind::Write,
                    MemoryAccessSemantics::async_proxy(),
                );
                complete_store(&mut state, &write);
            }
            let read = shared_batch(1, 1, PhysicalAccessKind::Read, semantics);
            let token = AsyncTokenId::new(read.operation().id().clone(), 0);
            state
                .begin_async_token(
                    &token,
                    read.operation(),
                    std::slice::from_ref(&read),
                    &[],
                    false,
                )
                .unwrap();
            assert!(state
                .actors
                .get(&GlobalActor::new(1, 0))
                .unwrap()
                .pending_acquire
                .is_empty());
            // A later overwrite must not change the version of the saved source bytes.
            publish(&mut state, 2, 9);
            state
                .complete_async_token(&token, AsyncGroupMilestone::SourceReadComplete, &[read])
                .unwrap();
            state
                .acquire_async_token(
                    &operation(1, 0, 2, OperationKind::Barrier),
                    &token,
                    AsyncGroupMilestone::SourceReadComplete,
                )
                .unwrap();
            let pending = &state
                .actors
                .get(&GlobalActor::new(1, 0))
                .unwrap()
                .pending_acquire;
            assert_eq!(pending.len(), usize::from(!invalidate_before_issue));
            if !invalidate_before_issue {
                assert_eq!(
                    pending.values().next().unwrap().key.actor,
                    GlobalActor::new(0, 0)
                );
            }
            let (_, acquired) = state
                .fence(
                    &operation(1, 0, 3, OperationKind::Fence),
                    MemoryFenceEffect::new(
                        MemoryOrder::Acquire,
                        MemoryScope::Cta,
                        MemoryProxy::Generic,
                    ),
                    &TcgenLaneFrontiers::new(),
                    None,
                )
                .unwrap();
            assert_eq!(
                acquired,
                if invalidate_before_issue {
                    BTreeMap::new()
                } else {
                    BTreeMap::from([(0, SharedClockFrontier::single(0, [7; crate::WARP_SIZE]))])
                }
            );
            let fresh_read = shared_batch(3, 1, PhysicalAccessKind::Read, semantics);
            let fresh_token = AsyncTokenId::new(fresh_read.operation().id().clone(), 0);
            state
                .begin_async_token(
                    &fresh_token,
                    fresh_read.operation(),
                    &[fresh_read.clone()],
                    &[],
                    false,
                )
                .unwrap();
            assert_eq!(
                state.async_tokens[&fresh_token]
                    .read_observations
                    .values()
                    .next()
                    .unwrap()
                    .key
                    .actor,
                GlobalActor::new(2, 0)
            );
            assert_eq!(state.incomplete_reasons().count(), 0);
            assert_eq!(
                state.findings().count(),
                0,
                "shared conflicts remain owned by RaceShadow"
            );
        }
    }

    #[test]
    fn async_global_completion_wait_bridges_to_generic_publication() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let issue = operation(0, 0, 1, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let destination = batch(
            0,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &token,
                &issue,
                &[],
                std::slice::from_ref(&destination),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&destination),
            )
            .unwrap();
        let wait = operation(0, 0, 2, OperationKind::Barrier);
        state
            .acquire_async_token(&wait, &token, AsyncGroupMilestone::FullComplete)
            .unwrap();
        state.retire_async_token(&token);
        commit(
            &mut state,
            &batch(
                0,
                0,
                3,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state.findings().next().is_none());
        assert!(state.incomplete_reasons().next().is_none());
    }

    #[test]
    fn async_global_completion_failure_requires_ordinary_post_issue_handoff() {
        let run = |handoff_after_issue: Option<bool>| {
            let topology = LaunchTopology::new(1, 2, 1).unwrap();
            let mut state = GlobalRaceState::new(Some(topology));
            let barrier = PhysicalBarrierId::new(19, 0, 0);
            let lane_zero = WarpMask::from_bits(1);
            if handoff_after_issue == Some(false) {
                state
                    .physical_barrier_release(barrier, 0, 0, lane_zero)
                    .unwrap();
                state
                    .physical_barrier_acquire(None, barrier, 0, 1, lane_zero, true)
                    .unwrap();
            }

            let issue = operation(0, 0, 1, OperationKind::AsyncIssue);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let destination = batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::async_proxy(),
            );
            state
                .begin_async_token(
                    &token,
                    &issue,
                    &[],
                    std::slice::from_ref(&destination),
                    false,
                )
                .unwrap();

            if handoff_after_issue == Some(true) {
                state
                    .physical_barrier_release(barrier, 0, 0, lane_zero)
                    .unwrap();
                state
                    .physical_barrier_acquire(None, barrier, 0, 1, lane_zero, true)
                    .unwrap();
            }
            state
                .complete_async_token(
                    &token,
                    AsyncGroupMilestone::FullComplete,
                    std::slice::from_ref(&destination),
                )
                .unwrap();
            commit(
                &mut state,
                &batch(
                    1,
                    0,
                    2,
                    PhysicalAccessKind::Read,
                    0,
                    MemoryAccessSemantics::async_proxy(),
                ),
            );

            let failure = state
                .findings()
                .next()
                .expect("the undrained async write conflicts with the read")
                .ordering_failure();
            failure
        };

        assert_eq!(
            run(None),
            PhysicalRaceOrderingFailure::MissingInterActorSynchronization
        );
        assert_eq!(
            run(Some(false)),
            PhysicalRaceOrderingFailure::MissingInterActorSynchronization
        );
        assert_eq!(
            run(Some(true)),
            PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained
        );
    }

    #[test]
    fn full_async_global_tma_wait_orders_same_cta_read_write_and_write_reuse() {
        let topology = LaunchTopology::new(1, 8, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let lane_zero = WarpMask::from_bits(1);
        let load_complete = PhysicalBarrierId::new(1, 0, 0);
        let store_ready = PhysicalBarrierId::new(2, 0, 0);

        let load_issue = operation(5, 0, 1, OperationKind::AsyncIssue);
        let load_token = AsyncTokenId::new(load_issue.id().clone(), 0);
        let state_read = batch(
            5,
            0,
            1,
            PhysicalAccessKind::Read,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &load_token,
                &load_issue,
                std::slice::from_ref(&state_read),
                &[],
                true,
            )
            .unwrap();
        state
            .complete_async_token(&load_token, AsyncGroupMilestone::FullComplete, &[])
            .unwrap();
        state
            .publish_async_token_to_physical_barrier(&load_token, load_complete, 0)
            .unwrap();
        state.retire_async_token(&load_token);

        state
            .physical_barrier_acquire(None, load_complete, 0, 0, lane_zero, true)
            .unwrap();
        state
            .physical_barrier_release(store_ready, 0, 0, lane_zero)
            .unwrap();
        state
            .physical_barrier_acquire(None, store_ready, 0, 6, lane_zero, true)
            .unwrap();

        let first_store_issue = operation(6, 0, 2, OperationKind::AsyncIssue);
        let first_store_token = AsyncTokenId::new(first_store_issue.id().clone(), 0);
        let first_state_write = batch(
            6,
            0,
            2,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &first_store_token,
                &first_store_issue,
                &[],
                std::slice::from_ref(&first_state_write),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &first_store_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&first_state_write),
            )
            .unwrap();
        state
            .acquire_async_token(
                &first_store_issue,
                &first_store_token,
                AsyncGroupMilestone::FullComplete,
            )
            .unwrap();
        state.retire_async_token(&first_store_token);

        let second_store_issue = operation(6, 0, 3, OperationKind::AsyncIssue);
        let second_store_token = AsyncTokenId::new(second_store_issue.id().clone(), 0);
        let second_state_write = batch(
            6,
            0,
            3,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &second_store_token,
                &second_store_issue,
                &[],
                std::slice::from_ref(&second_state_write),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &second_store_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&second_state_write),
            )
            .unwrap();

        assert!(state.findings().next().is_none());
        assert!(state.incomplete_reasons().next().is_none());

        let replay = state
            .replay_analysis()
            .expect("bounded replay log should cover the async/barrier event stream");
        assert_eq!(state.findings().collect::<BTreeSet<_>>(), replay.findings);
        assert_eq!(
            state.scope_diagnostics().collect::<BTreeSet<_>>(),
            replay.scope_diagnostics
        );
        assert_eq!(
            state.incomplete_reasons().collect::<Vec<_>>(),
            replay.incomplete_reasons
        );
        assert_eq!(replay.batch_count, state.replay_batch_count);
        assert_eq!(replay.access_count, state.replay_access_count);
        assert!(state.replay_storage_stats().event_count > replay.batch_count);
    }

    #[test]
    fn source_read_async_wait_does_not_order_global_destination_completion() {
        let topology = LaunchTopology::new(1, 8, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));

        let first_store_issue = operation(6, 0, 1, OperationKind::AsyncIssue);
        let first_store_token = AsyncTokenId::new(first_store_issue.id().clone(), 0);
        let first_state_write = batch(
            6,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &first_store_token,
                &first_store_issue,
                &[],
                std::slice::from_ref(&first_state_write),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &first_store_token,
                AsyncGroupMilestone::SourceReadComplete,
                &[],
            )
            .unwrap();
        state
            .acquire_async_token(
                &first_store_issue,
                &first_store_token,
                AsyncGroupMilestone::SourceReadComplete,
            )
            .unwrap();
        state
            .complete_async_token(
                &first_store_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&first_state_write),
            )
            .unwrap();

        let next_load_issue = operation(5, 0, 2, OperationKind::AsyncIssue);
        let next_load_token = AsyncTokenId::new(next_load_issue.id().clone(), 0);
        let next_state_read = batch(
            5,
            0,
            2,
            PhysicalAccessKind::Read,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &next_load_token,
                &next_load_issue,
                std::slice::from_ref(&next_state_read),
                &[],
                true,
            )
            .unwrap();

        let second_store_issue = operation(6, 0, 3, OperationKind::AsyncIssue);
        let second_store_token = AsyncTokenId::new(second_store_issue.id().clone(), 0);
        let second_state_write = batch(
            6,
            0,
            3,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &second_store_token,
                &second_store_issue,
                &[],
                std::slice::from_ref(&second_state_write),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &second_store_token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&second_state_write),
            )
            .unwrap();

        let kinds = state
            .findings()
            .map(|finding| finding.kind())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            kinds,
            BTreeSet::from([PhysicalRaceKind::ReadWrite, PhysicalRaceKind::WriteWrite])
        );
        assert!(state.incomplete_reasons().next().is_none());
    }

    #[test]
    fn async_global_publication_without_wait_or_proxy_fence_is_not_clean() {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        let issue = operation(0, 0, 1, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let destination = batch(
            0,
            0,
            1,
            PhysicalAccessKind::Write,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(
                &token,
                &issue,
                &[],
                std::slice::from_ref(&destination),
                false,
            )
            .unwrap();
        state
            .complete_async_token(
                &token,
                AsyncGroupMilestone::FullComplete,
                std::slice::from_ref(&destination),
            )
            .unwrap();
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                4,
                atomic(MemoryOrder::Release, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                1,
                PhysicalAccessKind::Read,
                4,
                atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
            ),
        );
        commit(
            &mut state,
            &batch(
                1,
                0,
                2,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );

        assert!(state
            .findings()
            .any(|finding| finding.overlap().byte_offset() == 0));
    }

    #[test]
    fn generic_to_async_global_access_requires_a_global_proxy_fence() {
        let run = |scope: Option<ProxyAsyncFenceScope>| {
            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let mut state = GlobalRaceState::new(Some(topology));
            commit(
                &mut state,
                &batch(
                    0,
                    0,
                    1,
                    PhysicalAccessKind::Write,
                    0,
                    MemoryAccessSemantics::plain(),
                ),
            );
            if let Some(scope) = scope {
                state
                    .proxy_async_fence(
                        &operation(0, 0, 2, OperationKind::Fence),
                        ProxyAsyncFenceEffect::new(scope, 0, 0, 0),
                    )
                    .unwrap();
            }
            let issue = operation(0, 0, 3, OperationKind::AsyncIssue);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let source = batch(
                0,
                0,
                3,
                PhysicalAccessKind::Read,
                0,
                MemoryAccessSemantics::async_proxy(),
            );
            state
                .begin_async_token(&token, &issue, std::slice::from_ref(&source), &[], true)
                .unwrap();
            let clean = state.findings().next().is_none();
            clean
        };

        assert!(!run(None));
        assert!(!run(Some(ProxyAsyncFenceScope::SharedCta)));
        assert!(run(Some(ProxyAsyncFenceScope::Global)));
        assert!(run(Some(ProxyAsyncFenceScope::All)));
    }

    #[test]
    fn global_proxy_fence_does_not_cover_a_later_generic_access() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        state
            .proxy_async_fence(
                &operation(0, 0, 1, OperationKind::Fence),
                ProxyAsyncFenceEffect::new(ProxyAsyncFenceScope::Global, 0, 0, 0),
            )
            .unwrap();
        commit(
            &mut state,
            &batch(
                0,
                0,
                2,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        let issue = operation(0, 0, 3, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let source = batch(
            0,
            0,
            3,
            PhysicalAccessKind::Read,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(&token, &issue, std::slice::from_ref(&source), &[], true)
            .unwrap();
        assert!(state.findings().next().is_some());
    }

    #[test]
    fn ordinary_global_barrier_does_not_replace_a_proxy_fence() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let mut state = GlobalRaceState::new(Some(topology));
        commit(
            &mut state,
            &batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::plain(),
            ),
        );
        let barrier = PhysicalBarrierId::new(9, 0, 0);
        let lane_zero = WarpMask::from_bits(1);
        state
            .physical_barrier_release(barrier, 0, 0, lane_zero)
            .unwrap();
        state
            .physical_barrier_acquire(None, barrier, 0, 1, lane_zero, true)
            .unwrap();

        let issue = operation(1, 0, 1, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let source = batch(
            1,
            0,
            1,
            PhysicalAccessKind::Read,
            0,
            MemoryAccessSemantics::async_proxy(),
        );
        state
            .begin_async_token(&token, &issue, std::slice::from_ref(&source), &[], true)
            .unwrap();

        assert!(state.findings().next().is_some());
    }

    #[test]
    fn global_proxy_fence_requires_completed_async_observation() {
        let run = |observe_completion: bool| {
            let topology = LaunchTopology::new(2, 1, 1).unwrap();
            let mut state = GlobalRaceState::new(Some(topology));
            let issue = operation(0, 0, 1, OperationKind::AsyncIssue);
            let token = AsyncTokenId::new(issue.id().clone(), 0);
            let destination = batch(
                0,
                0,
                1,
                PhysicalAccessKind::Write,
                0,
                MemoryAccessSemantics::async_proxy(),
            );
            state
                .begin_async_token(
                    &token,
                    &issue,
                    &[],
                    std::slice::from_ref(&destination),
                    false,
                )
                .unwrap();
            state
                .complete_async_token(
                    &token,
                    AsyncGroupMilestone::FullComplete,
                    std::slice::from_ref(&destination),
                )
                .unwrap();
            if observe_completion {
                let wait = operation(0, 0, 2, OperationKind::Barrier);
                state
                    .acquire_async_token(&wait, &token, AsyncGroupMilestone::FullComplete)
                    .unwrap();
            }
            let fence = operation(0, 0, 3, OperationKind::Fence);
            state
                .proxy_async_fence(
                    &fence,
                    ProxyAsyncFenceEffect::new(ProxyAsyncFenceScope::Global, 0, 0, 0),
                )
                .unwrap();
            commit(
                &mut state,
                &batch(
                    0,
                    0,
                    4,
                    PhysicalAccessKind::Write,
                    4,
                    atomic(MemoryOrder::Release, MemoryScope::Gpu),
                ),
            );
            commit(
                &mut state,
                &batch(
                    1,
                    0,
                    1,
                    PhysicalAccessKind::Read,
                    4,
                    atomic(MemoryOrder::Acquire, MemoryScope::Gpu),
                ),
            );
            commit(
                &mut state,
                &batch(
                    1,
                    0,
                    2,
                    PhysicalAccessKind::Read,
                    0,
                    MemoryAccessSemantics::plain(),
                ),
            );
            state.findings().count()
        };

        assert!(run(false) > 0);
        assert_eq!(run(true), 0);
    }
}
