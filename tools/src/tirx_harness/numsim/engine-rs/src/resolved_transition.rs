use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use crate::runtime::{TcgenLifecyclePlan, TcgenLifecycleResumePlan, TcgenWorkKind};
use crate::SyncVectorClock;
use crate::{
    AnalysisGapDomain, AnalysisGapEffect, AnalysisGapKind, AsyncGroupCommitOutcome,
    AsyncGroupCompletionAction, AsyncGroupCompletionOutcome, AsyncGroupDomain, AsyncGroupId,
    AsyncGroupMilestone, AsyncGroupWaitOutcome, AsyncPayloadEffect, AsyncTokenId, ClusterBarrierId,
    CompletionEffect, DeferredPayloadCompletionOutcome, DynamicOpId, MemoryAccessSemantics,
    NamedBarrierId, OperationContext, OperationEffect, OwnedOperationEffect, PhysicalAccessBatch,
    PhysicalAccessKind, PhysicalAccessSpace, PhysicalBarrierId, PhysicalByteSpan,
    PhysicalCompletionAction, PhysicalCompletionOutcome, ProxyAsyncFenceEffect,
    ProxyAsyncFenceScope, SetmaxnregAction, SetmaxnregBudgetDisposition,
    SetmaxnregCompletionAction, SetmaxnregCompletionOutcome, SetmaxnregPlan, SetmaxnregResource,
    SetmaxnregResumePlan, WarpMask, WarpSyncEffect, SETMAXNREG_WARPS_PER_GROUP,
};

/// Canonical physical-memory effect of one resolved dynamic operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedMemoryEffect {
    kind: PhysicalAccessKind,
    space: PhysicalAccessSpace,
    semantics: MemoryAccessSemantics,
    spans: Box<[PhysicalByteSpan]>,
}

impl ResolvedMemoryEffect {
    pub fn new(
        kind: PhysicalAccessKind,
        space: PhysicalAccessSpace,
        spans: impl IntoIterator<Item = PhysicalByteSpan>,
    ) -> Self {
        Self {
            kind,
            space,
            semantics: MemoryAccessSemantics::plain(),
            spans: canonicalize_spans(spans),
        }
    }

    pub fn from_batch(batch: &PhysicalAccessBatch) -> Self {
        let descriptor = batch.descriptor();
        Self {
            kind: descriptor.kind(),
            space: descriptor.space(),
            semantics: descriptor.memory_semantics(),
            spans: canonicalize_spans(
                batch
                    .lanes()
                    .iter()
                    .flat_map(|lane| lane.footprint().spans().iter().copied()),
            ),
        }
    }

    pub const fn kind(&self) -> PhysicalAccessKind {
        self.kind
    }

    pub const fn space(&self) -> PhysicalAccessSpace {
        self.space
    }

    pub fn spans(&self) -> &[PhysicalByteSpan] {
        &self.spans
    }
}

fn canonicalize_memory_effects(
    effects: impl IntoIterator<Item = ResolvedMemoryEffect>,
) -> Box<[ResolvedMemoryEffect]> {
    let mut grouped = BTreeMap::<
        (
            PhysicalAccessSpace,
            PhysicalAccessKind,
            MemoryAccessSemantics,
        ),
        Vec<PhysicalByteSpan>,
    >::new();
    for effect in effects {
        let ResolvedMemoryEffect {
            kind,
            space,
            semantics,
            spans,
        } = effect;
        if spans.is_empty() {
            continue;
        }
        grouped
            .entry((space, kind, semantics))
            .or_default()
            .extend(spans.into_vec());
    }
    grouped
        .into_iter()
        .map(|((space, kind, semantics), spans)| ResolvedMemoryEffect {
            kind,
            space,
            semantics,
            spans: canonicalize_spans(spans),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Conservative scheduler resource attached to an analysis-gap marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResolvedAnalysisResource {
    TcgenGlobal {
        kernel_index: usize,
    },
    TcgenCta {
        kernel_index: usize,
        global_cta_id: usize,
    },
    ClusterBarrierKernel {
        kernel_index: usize,
    },
    AtomicCta {
        kernel_index: usize,
        global_cta_id: usize,
    },
}

/// Replay-stable summary of one path-sensitive unmodeled analysis effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAnalysisGapEffect {
    kind: AnalysisGapKind,
    cta_group: Option<u32>,
    resources: Box<[ResolvedAnalysisResource]>,
}

impl ResolvedAnalysisGapEffect {
    pub fn from_effect(effect: AnalysisGapEffect) -> Self {
        let resources = match effect.domain() {
            AnalysisGapDomain::Tcgen => vec![
                ResolvedAnalysisResource::TcgenGlobal {
                    kernel_index: effect.kernel_index(),
                },
                ResolvedAnalysisResource::TcgenCta {
                    kernel_index: effect.kernel_index(),
                    global_cta_id: effect.global_cta_id(),
                },
            ],
            AnalysisGapDomain::ClusterBarrier => {
                vec![ResolvedAnalysisResource::ClusterBarrierKernel {
                    kernel_index: effect.kernel_index(),
                }]
            }
            AnalysisGapDomain::Atomic => vec![ResolvedAnalysisResource::AtomicCta {
                kernel_index: effect.kernel_index(),
                global_cta_id: effect.global_cta_id(),
            }],
        };
        Self {
            kind: effect.kind(),
            cta_group: effect.cta_group(),
            resources: resources.into_boxed_slice(),
        }
    }

    pub const fn kind(&self) -> AnalysisGapKind {
        self.kind
    }

    pub const fn cta_group(&self) -> Option<u32> {
        self.cta_group
    }

    pub fn resources(&self) -> &[ResolvedAnalysisResource] {
        &self.resources
    }
}

/// Analysis resource touched by a resolved synchronization transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResolvedSyncResourceKey {
    PhysicalMbarrier(PhysicalBarrierId),
    PhysicalMbarrierCompletionActionAllocator,
    NamedBarrier(NamedBarrierId),
    ClusterBarrier(ClusterBarrierId),
    AsyncGroup {
        global_warp_id: usize,
        lane: usize,
        domain: AsyncGroupDomain,
    },
    ProxyAsyncCta {
        kernel_index: usize,
        global_cta_id: usize,
    },
    ProxyAsyncCluster {
        kernel_index: usize,
        cluster_id: usize,
    },
    ProxyAsyncGlobal {
        kernel_index: usize,
    },
    TcgenLifecycleCta {
        kernel_index: usize,
        global_cta_id: usize,
    },
    TcgenCommitQueue {
        global_warp_id: usize,
        lane: usize,
        cta_group: u32,
    },
    TcgenTransferQueue {
        global_warp_id: usize,
        lane: usize,
        kind: TcgenWorkKind,
    },
    Setmaxnreg {
        kernel_index: usize,
        global_cta_id: usize,
        warpgroup_id: usize,
        ordinal: u64,
    },
    SetmaxnregPool {
        kernel_index: usize,
        global_cta_id: usize,
    },
}

/// One synchronization resource and its exact generation, when available.
///
/// A missing generation is a conservative wildcard for the resource. This is
/// required for mbarrier plans, which resolve the physical slot and parity but
/// do not expose the hub's current generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedSyncResource {
    key: ResolvedSyncResourceKey,
    generation: Option<u64>,
}

impl ResolvedSyncResource {
    pub const fn physical_mbarrier(barrier_id: PhysicalBarrierId, generation: Option<u64>) -> Self {
        Self {
            key: ResolvedSyncResourceKey::PhysicalMbarrier(barrier_id),
            generation,
        }
    }

    pub const fn key(self) -> ResolvedSyncResourceKey {
        self.key
    }

    pub const fn physical_mbarrier_completion_action_allocator() -> Self {
        Self {
            key: ResolvedSyncResourceKey::PhysicalMbarrierCompletionActionAllocator,
            generation: None,
        }
    }

    pub const fn named_barrier(barrier_id: NamedBarrierId, generation: Option<u64>) -> Self {
        Self {
            key: ResolvedSyncResourceKey::NamedBarrier(barrier_id),
            generation,
        }
    }

    pub const fn cluster_barrier(barrier_id: ClusterBarrierId, generation: Option<u64>) -> Self {
        Self {
            key: ResolvedSyncResourceKey::ClusterBarrier(barrier_id),
            generation,
        }
    }

    pub const fn async_group(
        global_warp_id: usize,
        lane: usize,
        domain: AsyncGroupDomain,
        group_ordinal: Option<u64>,
    ) -> Self {
        Self {
            key: ResolvedSyncResourceKey::AsyncGroup {
                global_warp_id,
                lane,
                domain,
            },
            generation: group_ordinal,
        }
    }

    pub const fn proxy_async(effect: ProxyAsyncFenceEffect) -> Self {
        let key = match effect.scope() {
            ProxyAsyncFenceScope::SharedCta => ResolvedSyncResourceKey::ProxyAsyncCta {
                kernel_index: effect.kernel_index(),
                global_cta_id: effect.global_cta_id(),
            },
            ProxyAsyncFenceScope::All | ProxyAsyncFenceScope::SharedCluster => {
                ResolvedSyncResourceKey::ProxyAsyncCluster {
                    kernel_index: effect.kernel_index(),
                    cluster_id: effect.cluster_id(),
                }
            }
            ProxyAsyncFenceScope::Global => ResolvedSyncResourceKey::ProxyAsyncGlobal {
                kernel_index: effect.kernel_index(),
            },
        };
        Self {
            key,
            generation: None,
        }
    }

    pub const fn tcgen_lifecycle_cta(kernel_index: usize, global_cta_id: usize) -> Self {
        Self {
            key: ResolvedSyncResourceKey::TcgenLifecycleCta {
                kernel_index,
                global_cta_id,
            },
            generation: None,
        }
    }

    pub const fn tcgen_work_queue(
        global_warp_id: usize,
        lane: usize,
        kind: TcgenWorkKind,
        cta_group: Option<u32>,
    ) -> Self {
        let key = if kind.uses_commit() {
            ResolvedSyncResourceKey::TcgenCommitQueue {
                global_warp_id,
                lane,
                cta_group: match cta_group {
                    Some(cta_group) => cta_group,
                    None => 0,
                },
            }
        } else {
            ResolvedSyncResourceKey::TcgenTransferQueue {
                global_warp_id,
                lane,
                kind,
            }
        };
        Self {
            key,
            generation: None,
        }
    }

    pub const fn setmaxnreg(plan: &SetmaxnregPlan) -> Self {
        let resource = plan.resource();
        Self {
            key: ResolvedSyncResourceKey::Setmaxnreg {
                kernel_index: resource.kernel_index(),
                global_cta_id: resource.global_cta_id(),
                warpgroup_id: resource.warpgroup_id(),
                ordinal: resource.ordinal(),
            },
            generation: None,
        }
    }

    pub const fn setmaxnreg_pool(kernel_index: usize, global_cta_id: usize) -> Self {
        Self {
            key: ResolvedSyncResourceKey::SetmaxnregPool {
                kernel_index,
                global_cta_id,
            },
            generation: None,
        }
    }

    pub const fn generation(self) -> Option<u64> {
        self.generation
    }
}

/// Immutable summary of one resolved synchronization operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedSynchronizationEffect {
    resources: ResolvedSyncResources,
    details: OwnedOperationEffect,
    mbarrier_wait_requests: Box<[(PhysicalBarrierId, u64, Option<u64>)]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResolvedSyncResources {
    Empty,
    One(ResolvedSyncResource),
    Many(Box<[ResolvedSyncResource]>),
}

impl ResolvedSyncResources {
    fn new(resources: impl IntoIterator<Item = ResolvedSyncResource>) -> Self {
        let mut resources = resources.into_iter();
        let Some(first) = resources.next() else {
            return Self::Empty;
        };
        let Some(second) = resources.next() else {
            return Self::One(first);
        };
        let mut many = Vec::with_capacity(resources.size_hint().0.saturating_add(2));
        many.extend([first, second]);
        many.extend(resources);
        many.sort_unstable();
        many.dedup();
        match many.as_slice() {
            [] => Self::Empty,
            [resource] => Self::One(*resource),
            _ => Self::Many(many.into_boxed_slice()),
        }
    }

    fn as_slice(&self) -> &[ResolvedSyncResource] {
        match self {
            Self::Empty => &[],
            Self::One(resource) => std::slice::from_ref(resource),
            Self::Many(resources) => resources,
        }
    }
}

impl ResolvedSynchronizationEffect {
    /// Resolve one synchronization-class effect into its replay-stable summary.
    ///
    /// The resource set is a derived index over the payload — the effect does
    /// not carry it — so it is computed here; the protocol arguments are the
    /// payload itself.
    pub fn from_effect(effect: OperationEffect<'_>) -> Self {
        let mbarrier_wait_requests = match effect {
            OperationEffect::MbarrierWait { plan, outcome } => Box::new([(
                plan.barrier_id(),
                plan.requested_phase(),
                outcome.and_then(crate::runtime::PhysicalMbarrierWaitOutcome::completed_generation),
            )])
                as Box<[(PhysicalBarrierId, u64, Option<u64>)]>,
            _ => Box::new([]),
        };
        Self {
            resources: ResolvedSyncResources::new(sync_resources(effect)),
            details: canonical_sync_payload(effect).to_owned_effect(),
            mbarrier_wait_requests,
        }
    }

    pub fn from_mbarrier_init_fence(barrier_ids: &[PhysicalBarrierId]) -> Self {
        Self::from_effect(OperationEffect::MbarrierInitFence { barrier_ids })
    }

    pub fn from_warp_sync(mask: WarpMask) -> Self {
        Self::from_effect(OperationEffect::WarpSync(WarpSyncEffect::new(mask)))
    }

    pub fn resources(&self) -> &[ResolvedSyncResource] {
        self.resources.as_slice()
    }

    /// The stored payload: the protocol arguments this operation resolved to.
    pub const fn details(&self) -> &OwnedOperationEffect {
        &self.details
    }

    pub fn mbarrier_wait_requests(&self) -> &[(PhysicalBarrierId, u64, Option<u64>)] {
        &self.mbarrier_wait_requests
    }
}

/// Fold two summaries one operation produced for different lanes.
///
/// A warp-wide wait registers once per lane, so the same `DynamicOpId` arrives
/// with a payload that differs by lane. Refusing that is what
/// `ConflictingOperation` is for -- two *different* operations at one position
/// -- not this, where the lanes are the one operation.
fn merge_lane_varying_summaries(
    existing: &ResolvedTransitionSummary,
    attempted: &ResolvedTransitionSummary,
) -> Option<ResolvedTransitionSummary> {
    merge_lane_varying_mbarrier_wait_summaries(existing, attempted)
        .or_else(|| merge_lane_varying_declared_word_wait_summaries(existing, attempted))
}

/// A declared word's wait, registered once per active lane.
///
/// Each lane polls its own address and accepts its own position in that word's
/// history, so the payloads differ -- but the wait holds no synchronization
/// resource (`sync_resources` gives it none: its subject is a memory address),
/// and the lane-varying part is therefore nothing the record is keeping. One
/// representative retains what the record does keep, the operation and its
/// warp, the same way the mbarrier fold above retains its common warp.
fn merge_lane_varying_declared_word_wait_summaries(
    existing: &ResolvedTransitionSummary,
    attempted: &ResolvedTransitionSummary,
) -> Option<ResolvedTransitionSummary> {
    let (
        ResolvedTransitionSummary::Synchronization(existing_effect),
        ResolvedTransitionSummary::Synchronization(attempted_effect),
    ) = (existing, attempted)
    else {
        return None;
    };
    let (
        OwnedOperationEffect::DeclaredWordWait {
            plan: existing_plan,
        },
        OwnedOperationEffect::DeclaredWordWait {
            plan: attempted_plan,
        },
    ) = (existing_effect.details(), attempted_effect.details())
    else {
        return None;
    };
    if existing_plan.warp_id() != attempted_plan.warp_id() {
        return None;
    }
    Some(existing.clone())
}

fn merge_lane_varying_mbarrier_wait_summaries(
    existing: &ResolvedTransitionSummary,
    attempted: &ResolvedTransitionSummary,
) -> Option<ResolvedTransitionSummary> {
    let (
        ResolvedTransitionSummary::Synchronization(existing),
        ResolvedTransitionSummary::Synchronization(attempted),
    ) = (existing, attempted)
    else {
        return None;
    };
    let (
        OwnedOperationEffect::MbarrierWait {
            plan: existing_plan,
            ..
        },
        OwnedOperationEffect::MbarrierWait {
            plan: attempted_plan,
            ..
        },
    ) = (existing.details(), attempted.details())
    else {
        return None;
    };
    if existing_plan.warp_id() != attempted_plan.warp_id()
        || existing_plan.is_conditional() != attempted_plan.is_conditional()
        || existing_plan.has_acquire() != attempted_plan.has_acquire()
    {
        return None;
    }

    let mut wait_requests = existing
        .mbarrier_wait_requests()
        .iter()
        .chain(attempted.mbarrier_wait_requests())
        .copied()
        .collect::<Vec<_>>();
    wait_requests.sort_unstable();
    wait_requests.dedup();

    Some(ResolvedTransitionSummary::Synchronization(
        ResolvedSynchronizationEffect {
            resources: ResolvedSyncResources::new(
                existing
                    .resources()
                    .iter()
                    .chain(attempted.resources())
                    .copied(),
            ),
            // One representative payload retains the common warp; exact
            // lane-varying barrier/phase requests are stored separately.
            details: existing.details.clone(),
            mbarrier_wait_requests: wait_requests.into_boxed_slice(),
        },
    ))
}

/// Fold one delivered synchronization effect onto the form that is stored.
///
/// A blocking barrier publishes its effect twice under one `DynamicOpId`: once
/// when it registers and once when it resumes. The retired projection erased
/// that distinction — both spellings resolved to the same protocol arguments —
/// so the second registration compared equal and returned `AlreadyRegistered`.
/// Storing the delivered variants verbatim would make the pair differ, and
/// every blocking barrier would fail as `ConflictingOperation`. That fold is
/// therefore kept here; it is the only part of the projection with semantics of
/// its own.
///
/// The register-side outcome is dropped for exactly the folded pairs. It is not
/// recoverable from the resume plan (`completed_now` differs by construction),
/// and nothing reads it from the summary: an outcome reaches the record only
/// through the derived resource generation, which `from_effect` computes from
/// the delivered effect before this fold runs. Every other variant is stored as
/// delivered.
///
/// Why the blocking waits that are NOT folded are nonetheless safe under
/// full-payload `Eq`: `MbarrierWait`, `AsyncGroupWait` and the mbarrier arrivals
/// are also delivered twice per operation — staged by `before_effect`, committed
/// by `after_effect` — and the two spellings genuinely differ (a staged arrival
/// carries no outcome; a staged `MbarrierCompletionIssue` carries no action
/// ids). They do not conflict only because **`before_effect` stages into
/// checker-private protocol state and never registers into the transition log**:
/// every registration site in `native_analysis/` is on an `after_effect` path,
/// so each of those operations is registered exactly once and its payload never
/// gets re-compared. A future refactor that registered staged effects would
/// conflict on every blocking wait immediately; it would have to extend the fold
/// below to staged/committed pairs, exactly as this one covers register/resume.
fn canonical_sync_payload(effect: OperationEffect<'_>) -> OperationEffect<'_> {
    match effect {
        OperationEffect::NamedBarrierSyncRegister { plan, .. } => {
            OperationEffect::NamedBarrierSyncRegister {
                plan,
                outcome: None,
            }
        }
        OperationEffect::NamedBarrierSyncResume(resume) => {
            OperationEffect::NamedBarrierSyncRegister {
                plan: resume.plan(),
                outcome: None,
            }
        }
        OperationEffect::ClusterBarrierWaitRegister { plan, .. } => {
            OperationEffect::ClusterBarrierWaitRegister {
                plan,
                outcome: None,
            }
        }
        OperationEffect::ClusterBarrierWaitResume(resume) => {
            OperationEffect::ClusterBarrierWaitRegister {
                plan: resume.plan(),
                outcome: None,
            }
        }
        OperationEffect::TcgenLifecycleResume(resume) => {
            OperationEffect::TcgenLifecycleRegister(resume.plan())
        }
        OperationEffect::SetmaxnregResume(resume) => {
            OperationEffect::SetmaxnregRegister(resume.plan())
        }
        effect => effect,
    }
}

/// The synchronization resources one effect touches, in payload order.
///
/// `ResolvedSyncResources::new` sorts and dedups, so the order here only has to
/// be deterministic.
fn sync_resources(effect: OperationEffect<'_>) -> Vec<ResolvedSyncResource> {
    match effect {
        // A declared word's wait holds no synchronization resource; its
        // subject is a memory address.
        OperationEffect::DeclaredWordWait { .. } => Vec::new(),
        OperationEffect::MbarrierInit(plan) => plan
            .barrier_ids()
            .iter()
            .copied()
            .map(|id| ResolvedSyncResource::physical_mbarrier(id, None))
            .collect(),
        OperationEffect::MbarrierInitFence { barrier_ids }
        | OperationEffect::MbarrierInvalidate { barrier_ids } => barrier_ids
            .iter()
            .copied()
            .map(|id| ResolvedSyncResource::physical_mbarrier(id, None))
            .collect(),
        OperationEffect::MbarrierExpectTx { plan, outcome } => {
            if let Some(outcome) = outcome {
                outcome
                    .generations()
                    .iter()
                    .map(|&(barrier_id, generation)| {
                        ResolvedSyncResource::physical_mbarrier(barrier_id, Some(generation))
                    })
                    .collect()
            } else {
                plan.entries()
                    .iter()
                    .map(|entry| ResolvedSyncResource::physical_mbarrier(entry.barrier_id(), None))
                    .collect()
            }
        }
        OperationEffect::MbarrierArrive { plan, outcome } => {
            vec![ResolvedSyncResource::physical_mbarrier(
                plan.barrier_id(),
                outcome.map(crate::PhysicalMbarrierArrivalOutcome::generation),
            )]
        }
        OperationEffect::MbarrierArriveBatch { plan, outcome } => {
            let generations = outcome.map(|outcome| {
                outcome
                    .outcomes()
                    .iter()
                    .copied()
                    .map(crate::PhysicalMbarrierArrivalOutcome::generation)
                    .collect::<Vec<_>>()
            });
            plan.entries()
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    ResolvedSyncResource::physical_mbarrier(
                        entry.plan().barrier_id(),
                        generations
                            .as_ref()
                            .and_then(|generations| generations.get(index).copied()),
                    )
                })
                .collect()
        }
        OperationEffect::MbarrierWait { plan, outcome } => {
            vec![ResolvedSyncResource::physical_mbarrier(
                plan.barrier_id(),
                outcome.and_then(crate::runtime::PhysicalMbarrierWaitOutcome::completed_generation),
            )]
        }
        OperationEffect::MbarrierCompletionIssue { plan, .. } => {
            std::iter::once(ResolvedSyncResource::physical_mbarrier_completion_action_allocator())
                .chain(plan.completions().iter().map(|&(barrier_id, _)| {
                    ResolvedSyncResource::physical_mbarrier(barrier_id, None)
                }))
                .collect()
        }
        OperationEffect::TcgenCommitIssue { plan, work, .. } => {
            std::iter::once(ResolvedSyncResource::physical_mbarrier_completion_action_allocator())
                .chain(
                    plan.barrier_ids().iter().copied().map(|barrier_id| {
                        ResolvedSyncResource::physical_mbarrier(barrier_id, None)
                    }),
                )
                .chain(work.lane_tokens().iter().map(|(lane, _)| {
                    ResolvedSyncResource::tcgen_work_queue(
                        work.global_warp_id(),
                        *lane,
                        work.kind(),
                        work.cta_group(),
                    )
                }))
                .collect()
        }
        OperationEffect::CpAsyncMbarrierArrive {
            plan,
            commit_plan,
            outcome,
            ..
        } => {
            let groups = outcome
                .map(AsyncGroupCommitOutcome::groups)
                .unwrap_or_default();
            std::iter::once(ResolvedSyncResource::physical_mbarrier_completion_action_allocator())
                .chain(plan.targets().map(|(_, barrier_id)| {
                    ResolvedSyncResource::physical_mbarrier(barrier_id, None)
                }))
                .chain(async_group_resources(
                    commit_plan.global_warp_id(),
                    commit_plan.domain(),
                    commit_plan.lanes(),
                    groups.iter().map(|group| group.id()),
                ))
                .collect()
        }
        OperationEffect::TcgenWorkIssue(issue) => issue
            .operation()
            .active_mask()
            .iter()
            .map(|lane| {
                ResolvedSyncResource::tcgen_work_queue(
                    issue.operation().id().global_warp_id(),
                    lane,
                    issue.kind(),
                    Some(issue.cta_group()),
                )
            })
            .collect(),
        OperationEffect::TcgenWait { work } => work
            .lane_tokens()
            .iter()
            .map(|(lane, _)| {
                ResolvedSyncResource::tcgen_work_queue(
                    work.global_warp_id(),
                    *lane,
                    work.kind(),
                    work.cta_group(),
                )
            })
            .collect(),
        OperationEffect::AsyncGroupIssue(effect) => effect
            .operation()
            .active_mask()
            .iter()
            .map(|lane| {
                ResolvedSyncResource::async_group(
                    effect.operation().id().global_warp_id(),
                    lane,
                    effect.domain(),
                    None,
                )
            })
            .collect(),
        OperationEffect::AsyncGroupIssueBatch(effect) => effect
            .members()
            .iter()
            .flat_map(|member| {
                member.operation().active_mask().iter().map(|lane| {
                    ResolvedSyncResource::async_group(
                        effect.operation().id().global_warp_id(),
                        lane,
                        effect.domain(),
                        None,
                    )
                })
            })
            .collect(),
        OperationEffect::AsyncGroupCommit { plan, outcome } => async_group_resources(
            plan.global_warp_id(),
            plan.domain(),
            plan.lanes(),
            outcome
                .map(AsyncGroupCommitOutcome::groups)
                .unwrap_or_default()
                .iter()
                .map(|group| group.id()),
        ),
        OperationEffect::AsyncGroupWait { plan, outcome } => async_group_resources(
            plan.global_warp_id(),
            plan.domain(),
            plan.lanes(),
            outcome
                .map(AsyncGroupWaitOutcome::groups)
                .unwrap_or_default()
                .iter()
                .map(|group| group.id()),
        ),
        OperationEffect::ProxyAsyncFence(effect) => {
            vec![ResolvedSyncResource::proxy_async(effect)]
        }
        OperationEffect::MemoryFence(_)
        | OperationEffect::TensorMap(_)
        | OperationEffect::TcgenFence(_)
        | OperationEffect::WarpSync(_) => Vec::new(),
        OperationEffect::NamedBarrierArrive { plan, outcome } => {
            vec![ResolvedSyncResource::named_barrier(
                plan.barrier_id(),
                outcome.map(|outcome| outcome.generation()),
            )]
        }
        OperationEffect::NamedBarrierSyncRegister { plan, outcome } => {
            vec![ResolvedSyncResource::named_barrier(
                plan.barrier_id(),
                outcome.map(|outcome| outcome.generation()),
            )]
        }
        OperationEffect::NamedBarrierSyncResume(plan) => {
            vec![ResolvedSyncResource::named_barrier(
                plan.plan().barrier_id(),
                Some(plan.generation()),
            )]
        }
        OperationEffect::ClusterBarrierArrive { plan, outcome } => {
            vec![ResolvedSyncResource::cluster_barrier(
                plan.barrier_id(),
                outcome.map(|outcome| outcome.generation()),
            )]
        }
        OperationEffect::ClusterBarrierWaitRegister { plan, outcome } => {
            vec![ResolvedSyncResource::cluster_barrier(
                plan.barrier_id(),
                outcome.map(|outcome| outcome.generation()),
            )]
        }
        OperationEffect::ClusterBarrierWaitResume(plan) => {
            vec![ResolvedSyncResource::cluster_barrier(
                plan.plan().barrier_id(),
                Some(plan.generation()),
            )]
        }
        OperationEffect::TcgenLifecycleRegister(plan) => tcgen_lifecycle_resources(plan),
        OperationEffect::TcgenLifecycleResume(plan) => tcgen_lifecycle_resources(plan.plan()),
        OperationEffect::SetmaxnregRegister(plan) => setmaxnreg_resources(plan),
        OperationEffect::SetmaxnregResume(plan) => setmaxnreg_resources(plan.plan()),
        OperationEffect::PhysicalAccess(_)
        | OperationEffect::AsyncPayload(_)
        | OperationEffect::AnalysisGap(_) => {
            debug_assert!(false, "effect is not synchronization-class");
            Vec::new()
        }
    }
}

/// Async-group lanes, keyed by resolved group ordinal once the hub reported one.
fn async_group_resources(
    global_warp_id: usize,
    domain: AsyncGroupDomain,
    lanes: impl Iterator<Item = usize>,
    groups: impl Iterator<Item = AsyncGroupId>,
) -> Vec<ResolvedSyncResource> {
    let resolved = groups
        .map(|id| {
            ResolvedSyncResource::async_group(
                id.global_warp_id(),
                id.lane(),
                id.domain(),
                Some(id.ordinal()),
            )
        })
        .collect::<Vec<_>>();
    if resolved.is_empty() {
        lanes
            .map(|lane| ResolvedSyncResource::async_group(global_warp_id, lane, domain, None))
            .collect()
    } else {
        resolved
    }
}

fn tcgen_lifecycle_resources(plan: &TcgenLifecyclePlan) -> Vec<ResolvedSyncResource> {
    plan.participant_ctas()
        .iter()
        .copied()
        .map(|global_cta_id| {
            ResolvedSyncResource::tcgen_lifecycle_cta(plan.kernel_index(), global_cta_id)
        })
        .collect()
}

fn setmaxnreg_resources(plan: &SetmaxnregPlan) -> Vec<ResolvedSyncResource> {
    let resource = plan.resource();
    vec![
        ResolvedSyncResource::setmaxnreg(plan),
        ResolvedSyncResource::setmaxnreg_pool(resource.kernel_index(), resource.global_cta_id()),
    ]
}

/// Exact memory and synchronization resources of one async payload issue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAsyncPayloadEffect {
    memory: Box<[ResolvedMemoryEffect]>,
    synchronization: ResolvedSynchronizationEffect,
}

impl ResolvedAsyncPayloadEffect {
    pub fn from_effect(effect: &AsyncPayloadEffect) -> Self {
        Self {
            memory: canonicalize_memory_effects(
                effect
                    .issue_accesses()
                    .iter()
                    .map(ResolvedMemoryEffect::from_batch),
            ),
            synchronization: ResolvedSynchronizationEffect::from_effect(
                OperationEffect::MbarrierCompletionIssue {
                    plan: effect.completion_plan(),
                    action_ids: effect.completion_action_ids(),
                },
            ),
        }
    }

    pub fn memory_effects(&self) -> &[ResolvedMemoryEffect] {
        &self.memory
    }

    pub const fn synchronization(&self) -> &ResolvedSynchronizationEffect {
        &self.synchronization
    }
}

/// Immutable identity and resource effect of one asynchronous completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedSetmaxnregGrantEffect {
    request: SetmaxnregResource,
    required_count: i64,
    available_count_before: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCompletionEffect {
    action_id: u64,
    token: Option<AsyncTokenId>,
    resources: Box<[ResolvedSyncResource]>,
    memory: Box<[ResolvedMemoryEffect]>,
    transactions: u64,
    setmaxnreg_grant: Option<ResolvedSetmaxnregGrantEffect>,
    // None is an issue-time hint. Some records the numeric owner's committed
    // primary completions and whether each also advanced the conditional phase.
    mbarrier_completions: Option<Box<[(PhysicalBarrierId, u64, bool)]>>,
}

impl ResolvedCompletionEffect {
    pub fn new(action_id: u64, resource: ResolvedSyncResource, transactions: u64) -> Self {
        Self {
            action_id,
            token: None,
            resources: Box::new([resource]),
            memory: Box::new([]),
            transactions,
            setmaxnreg_grant: None,
            mbarrier_completions: None,
        }
    }

    /// The log keeps completion summaries for their sync resources only
    /// (fixed-sync projections, re-registration checks); nothing reads a
    /// completion's memory effects back, and retaining a copy of every
    /// completed payload's footprint costs gigabytes on long launches.
    fn without_memory_effects(mut self) -> Self {
        self.memory = Box::new([]);
        self
    }

    pub fn new_deferred_payload(
        action_id: u64,
        token: AsyncTokenId,
        resources: impl IntoIterator<Item = ResolvedSyncResource>,
        memory: impl IntoIterator<Item = ResolvedMemoryEffect>,
        transactions: u64,
    ) -> Self {
        Self {
            action_id,
            token: Some(token),
            resources: resources.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            memory: canonicalize_memory_effects(memory),
            transactions,
            setmaxnreg_grant: None,
            mbarrier_completions: None,
        }
    }

    pub fn from_action(action: PhysicalCompletionAction) -> Self {
        Self::new(
            action.id().get(),
            ResolvedSyncResource::physical_mbarrier(action.barrier_id(), Some(action.generation())),
            action.transactions(),
        )
    }

    pub fn from_outcome(outcome: &PhysicalCompletionOutcome) -> Self {
        let mut effect = Self::from_action(outcome.action());
        effect.mbarrier_completions = Some(Self::committed_mbarrier_phases([outcome]));
        effect
    }

    fn committed_mbarrier_phases<'a>(
        outcomes: impl IntoIterator<Item = &'a PhysicalCompletionOutcome>,
    ) -> Box<[(PhysicalBarrierId, u64, bool)]> {
        outcomes
            .into_iter()
            .filter_map(|outcome| {
                outcome.completed_generation().map(|generation| {
                    (
                        outcome.action().barrier_id(),
                        generation,
                        outcome.conditional_completed_generation() == Some(generation),
                    )
                })
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn from_deferred_outcome(outcome: &DeferredPayloadCompletionOutcome) -> Self {
        let action = outcome.action();
        let mut effect = Self::new_deferred_payload(
            action.scheduler_action_id().get(),
            action.token().clone(),
            action.physical_actions().iter().map(|action| {
                ResolvedSyncResource::physical_mbarrier(
                    action.barrier_id(),
                    Some(action.generation()),
                )
            }),
            action
                .completion_accesses()
                .iter()
                .map(ResolvedMemoryEffect::from_batch),
            action
                .physical_actions()
                .iter()
                .map(|action| action.transactions())
                .sum(),
        );
        effect.mbarrier_completions =
            Some(Self::committed_mbarrier_phases(outcome.physical_outcomes()));
        effect
    }

    pub fn from_async_group_action(action: &AsyncGroupCompletionAction) -> Self {
        let id = action.group_id();
        let memory = action.members().iter().flat_map(|member| {
            let accesses = match action.milestone() {
                AsyncGroupMilestone::SourceReadComplete => member.source_accesses(),
                AsyncGroupMilestone::FullComplete => member.destination_accesses(),
            };
            accesses.iter().map(ResolvedMemoryEffect::from_batch)
        });
        let tokens = action
            .members()
            .iter()
            .map(|member| member.token().clone())
            .collect::<Vec<_>>();
        Self {
            action_id: action.id().get(),
            token: (tokens.len() == 1).then(|| tokens[0].clone()),
            resources: Box::new([ResolvedSyncResource::async_group(
                id.global_warp_id(),
                id.lane(),
                id.domain(),
                Some(id.ordinal()),
            )]),
            memory: canonicalize_memory_effects(memory),
            transactions: u64::try_from(action.members().len()).unwrap_or(u64::MAX),
            setmaxnreg_grant: None,
            mbarrier_completions: None,
        }
    }

    pub fn from_async_group_outcome(outcome: &AsyncGroupCompletionOutcome) -> Self {
        Self::from_async_group_action(outcome.action())
    }

    pub fn from_setmaxnreg_action(action: &SetmaxnregCompletionAction) -> Self {
        let resource = action.resource();
        let mut effect = Self::new(
            action.id().get(),
            ResolvedSyncResource::setmaxnreg_pool(
                resource.kernel_index(),
                resource.global_cta_id(),
            ),
            0,
        );
        effect.setmaxnreg_grant = Some(ResolvedSetmaxnregGrantEffect {
            request: resource,
            required_count: action.required_count(),
            available_count_before: action.available_count_before(),
        });
        effect
    }

    pub fn from_setmaxnreg_outcome(outcome: &SetmaxnregCompletionOutcome) -> Self {
        Self::from_setmaxnreg_action(outcome.action())
    }

    pub const fn action_id(&self) -> u64 {
        self.action_id
    }

    pub fn resource(&self) -> ResolvedSyncResource {
        *self
            .resources
            .first()
            .expect("resolved completion retains at least one sync resource")
    }

    pub fn resources(&self) -> &[ResolvedSyncResource] {
        &self.resources
    }

    pub fn memory_effects(&self) -> &[ResolvedMemoryEffect] {
        &self.memory
    }

    pub const fn token(&self) -> Option<&AsyncTokenId> {
        self.token.as_ref()
    }

    pub const fn transactions(&self) -> u64 {
        self.transactions
    }
}

/// Fully resolved, replay-stable semantic effect used by schedule reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedTransitionSummary {
    Memory(ResolvedMemoryEffect),
    Synchronization(ResolvedSynchronizationEffect),
    AsyncPayload(ResolvedAsyncPayloadEffect),
    AnalysisGap(ResolvedAnalysisGapEffect),
    Completion(ResolvedCompletionEffect),
    Unknown { reason: Box<str> },
    Unsupported { effect_name: Box<str> },
}

impl ResolvedTransitionSummary {
    pub fn from_operation_effect(effect: OperationEffect<'_>) -> Self {
        match effect {
            OperationEffect::PhysicalAccess(batch) => {
                Self::Memory(ResolvedMemoryEffect::from_batch(batch))
            }
            OperationEffect::AsyncPayload(effect) => {
                Self::AsyncPayload(ResolvedAsyncPayloadEffect::from_effect(effect))
            }
            OperationEffect::AnalysisGap(effect) => {
                Self::AnalysisGap(ResolvedAnalysisGapEffect::from_effect(effect))
            }
            effect => Self::Synchronization(ResolvedSynchronizationEffect::from_effect(effect)),
        }
    }

    pub fn from_completion_effect(effect: CompletionEffect<'_>) -> Self {
        match effect {
            CompletionEffect::PhysicalMbarrier(outcome) => {
                Self::Completion(ResolvedCompletionEffect::from_outcome(outcome))
            }
            CompletionEffect::DeferredPayload(outcome) => {
                Self::Completion(ResolvedCompletionEffect::from_deferred_outcome(outcome))
            }
            CompletionEffect::AsyncGroup(outcome) => {
                Self::Completion(ResolvedCompletionEffect::from_async_group_outcome(outcome))
            }
            CompletionEffect::Setmaxnreg(outcome) => {
                Self::Completion(ResolvedCompletionEffect::from_setmaxnreg_outcome(outcome))
            }
        }
    }

    pub fn unknown(reason: impl Into<Box<str>>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }

    pub fn unsupported(effect_name: impl Into<Box<str>>) -> Self {
        Self::Unsupported {
            effect_name: effect_name.into(),
        }
    }
}

/// Result of idempotently registering one replayed transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedTransitionRegistration {
    Inserted,
    Refined,
    AlreadyRegistered,
}

/// Stable-identity violation detected while accumulating replay effects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedTransitionLogError {
    OperationEffectMismatch {
        expected: DynamicOpId,
        actual: DynamicOpId,
    },
    CompletionIssueActionCountMismatch {
        operation: DynamicOpId,
        completion_count: usize,
        action_id_count: usize,
    },
    DuplicateCompletionIssueActionId {
        operation: DynamicOpId,
        action_id: u64,
    },
    ConflictingOperation {
        operation: DynamicOpId,
        existing: Box<ResolvedTransitionSummary>,
        attempted: Box<ResolvedTransitionSummary>,
    },
    ConflictingCompletion {
        action_id: u64,
        existing: Box<ResolvedTransitionSummary>,
        attempted: Box<ResolvedTransitionSummary>,
    },
    ConflictingOperationClock {
        operation: DynamicOpId,
        existing: SyncVectorClock,
        attempted: SyncVectorClock,
    },
    ConflictingCompletionSourceClock {
        action_id: u64,
        existing: SyncVectorClock,
        attempted: SyncVectorClock,
    },
}

impl fmt::Display for ResolvedTransitionLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperationEffectMismatch { expected, actual } => write!(
                f,
                "operation effect belongs to {actual}, not registered operation {expected}"
            ),
            Self::CompletionIssueActionCountMismatch {
                operation,
                completion_count,
                action_id_count,
            } => write!(
                f,
                "dynamic operation {operation} issued {completion_count} completions but returned {action_id_count} action IDs"
            ),
            Self::DuplicateCompletionIssueActionId {
                operation,
                action_id,
            } => write!(
                f,
                "dynamic operation {operation} returned duplicate completion action ID {action_id}"
            ),
            Self::ConflictingOperation { operation, .. } => write!(
                f,
                "dynamic operation {operation} resolved to conflicting semantic summaries"
            ),
            Self::ConflictingCompletion { action_id, .. } => write!(
                f,
                "completion action {action_id} resolved to conflicting semantic summaries"
            ),
            Self::ConflictingOperationClock { operation, .. } => write!(
                f,
                "dynamic operation {operation} resolved to conflicting causal clocks"
            ),
            Self::ConflictingCompletionSourceClock { action_id, .. } => write!(
                f,
                "completion action {action_id} resolved to conflicting causal source clocks"
            ),
        }
    }
}

impl Error for ResolvedTransitionLogError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedSetmaxnregRequestResolution {
    request: SetmaxnregResource,
    action: SetmaxnregAction,
    target_count: i64,
    current_count_before: i64,
    available_count_before: i64,
    disposition: SetmaxnregBudgetDisposition,
    warpgroup_count: usize,
}

impl ResolvedSetmaxnregRequestResolution {
    fn from_resume(resume: &SetmaxnregResumePlan) -> Option<Self> {
        let plan = resume.plan();
        let outcome = resume.outcome();
        let warpgroup_count = plan
            .context()
            .topology()
            .warps_per_cta()
            .div_ceil(SETMAXNREG_WARPS_PER_GROUP);
        (plan.resource() == outcome.resource()
            && plan.action() == outcome.action()
            && plan.count() == outcome.count())
        .then_some(Self {
            request: outcome.resource(),
            action: outcome.action(),
            target_count: outcome.count(),
            current_count_before: outcome.current_count_before(),
            available_count_before: outcome.available_count_before(),
            disposition: outcome.budget_disposition(),
            warpgroup_count,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedSetmaxnregRequestResolutionRecord {
    Known(ResolvedSetmaxnregRequestResolution),
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedSetmaxnregParticipant {
    request: SetmaxnregResource,
    plan_global_warp_id: usize,
    plan_global_cta_id: usize,
    plan_warp_id_in_cta: usize,
    plan_warps_per_cta: usize,
    operation_mask: WarpMask,
    plan_mask: WarpMask,
    action: SetmaxnregAction,
    count: i64,
}

impl ResolvedSetmaxnregParticipant {
    fn from_effect(operation: &OperationContext, effect: OperationEffect<'_>) -> Option<Self> {
        let plan = match effect {
            OperationEffect::SetmaxnregRegister(plan) => plan,
            OperationEffect::SetmaxnregResume(resume) => resume.plan(),
            _ => return None,
        };
        let context = plan.context();
        Some(Self {
            request: plan.resource(),
            plan_global_warp_id: context.global_warp_id(),
            plan_global_cta_id: context.global_cta_id(),
            plan_warp_id_in_cta: context.warp_id_in_cta(),
            plan_warps_per_cta: context.topology().warps_per_cta(),
            operation_mask: operation.active_mask(),
            plan_mask: context.active_mask(),
            action: plan.action(),
            count: plan.count(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedSetmaxnregParticipantRecord {
    Known(ResolvedSetmaxnregParticipant),
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedTcgenLifecycleResolutionRecord {
    Known(Option<(u32, usize)>),
    Ambiguous,
}

#[derive(Clone, Debug)]
pub(crate) struct FixedSyncSetmaxParticipantSnapshot {
    pub(crate) request: SetmaxnregResource,
    pub(crate) action: SetmaxnregAction,
    pub(crate) count: i64,
    pub(crate) operation_mask: WarpMask,
    pub(crate) plan_mask: WarpMask,
    pub(crate) plan_global_warp_id: usize,
    pub(crate) plan_global_cta_id: usize,
    pub(crate) plan_warp_id_in_cta: usize,
    pub(crate) plan_warps_per_cta: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedSyncSetmaxResolutionSnapshot {
    pub(crate) request: SetmaxnregResource,
    pub(crate) action: SetmaxnregAction,
    pub(crate) target_count: i64,
    pub(crate) warpgroup_count: usize,
    pub(crate) current_count_before: i64,
}

#[derive(Clone, Debug)]
pub(crate) enum FixedSyncSnapshotRecord<T> {
    Known(T),
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ResolvedOperationPosition {
    kernel_index: usize,
    global_warp_id: usize,
    per_warp_sequence: u64,
}

impl ResolvedOperationPosition {
    const fn from_operation(operation: &DynamicOpId) -> Self {
        Self {
            kernel_index: operation.kernel_index(),
            global_warp_id: operation.global_warp_id(),
            per_warp_sequence: operation.per_warp_sequence(),
        }
    }
}

#[derive(Debug)]
struct ResolvedOperationRecord {
    operation: DynamicOpId,
    summary: Option<ResolvedTransitionSummary>,
    initial_clock: Option<SyncVectorClock>,
    clock: Option<SyncVectorClock>,
}

#[derive(Debug)]
pub(crate) struct FixedSyncLogSnapshot<'a> {
    operations: Option<&'a BTreeMap<ResolvedOperationPosition, ResolvedOperationRecord>>,
    operation_shards: Vec<MutexGuard<'a, Vec<ResolvedOperationRecord>>>,
    pub(crate) completion_mbarrier_generations: BTreeMap<u64, Box<[(PhysicalBarrierId, u64)]>>,
    pub(crate) conditional_mbarrier_completions:
        BTreeMap<PhysicalBarrierId, BTreeMap<DynamicOpId, BTreeSet<u64>>>,
    pub(crate) setmax_participants:
        BTreeMap<DynamicOpId, FixedSyncSnapshotRecord<FixedSyncSetmaxParticipantSnapshot>>,
    pub(crate) setmax_resolutions:
        BTreeMap<SetmaxnregResource, FixedSyncSnapshotRecord<FixedSyncSetmaxResolutionSnapshot>>,
    pub(crate) tcgen_allocations:
        BTreeMap<DynamicOpId, FixedSyncSnapshotRecord<Option<(u32, usize)>>>,
}

impl FixedSyncLogSnapshot<'_> {
    fn operation_records(&self) -> impl Iterator<Item = &ResolvedOperationRecord> {
        self.operations
            .into_iter()
            .flat_map(BTreeMap::values)
            .chain(
                self.operation_shards
                    .iter()
                    .flat_map(|records| records.iter()),
            )
    }

    pub(crate) fn operations_with_clocks(
        &self,
    ) -> impl Iterator<
        Item = (
            &DynamicOpId,
            &ResolvedTransitionSummary,
            Option<&SyncVectorClock>,
            Option<&SyncVectorClock>,
        ),
    > {
        self.operation_records().filter_map(|record| {
            record.summary.as_ref().map(|summary| {
                (
                    &record.operation,
                    summary,
                    record.initial_clock.as_ref().or(record.clock.as_ref()),
                    record.clock.as_ref(),
                )
            })
        })
    }
}

#[derive(Debug, Default)]
struct ResolvedTransitionState {
    operations: BTreeMap<ResolvedOperationPosition, ResolvedOperationRecord>,
    completions: BTreeMap<u64, ResolvedTransitionSummary>,
    completion_source_clocks: BTreeMap<u64, SyncVectorClock>,
    setmaxnreg_participants: BTreeMap<DynamicOpId, ResolvedSetmaxnregParticipantRecord>,
    setmaxnreg_request_resolutions:
        BTreeMap<ResolvedSyncResourceKey, ResolvedSetmaxnregRequestResolutionRecord>,
    tcgen_lifecycle_resolutions: BTreeMap<DynamicOpId, ResolvedTcgenLifecycleResolutionRecord>,
}

/// Thread-safe current-replay effect history shared by one mode and explorer.
///
/// Warp effects use [`DynamicOpId`] as their stable replay identity. Completion
/// candidates have no `OperationContext`, so their current replay's
/// scheduler-visible action ID is retained in a separate index while the
/// summary stores the exact resource and generation. Call [`Self::begin_replay`]
/// before every stateless replay. One log supports only sequential replays:
/// the explorer must finish all dependency queries for a run before clearing
/// the log for the next run.
#[derive(Clone, Debug)]
pub struct ResolvedTransitionLog {
    state: Arc<RwLock<ResolvedTransitionState>>,
    operation_shards: Option<Arc<[Mutex<Vec<ResolvedOperationRecord>>]>>,
}

impl Default for ResolvedTransitionLog {
    fn default() -> Self {
        Self {
            state: Arc::new(RwLock::new(ResolvedTransitionState::default())),
            operation_shards: None,
        }
    }
}

impl ResolvedTransitionLog {
    /// Build the fixed-trace recorder used by one native launch phase.
    ///
    /// A warp executes its dynamic operations in increasing sequence order.
    /// Keeping one append-oriented shard per warp avoids a launch-wide lock
    /// and a `BTreeMap` node allocation for every fixed-verifier command while
    /// retaining the exact operation identity, summary, and causal clocks.
    pub(crate) fn with_warp_operation_shards(warp_count: usize) -> Self {
        Self {
            state: Arc::new(RwLock::new(ResolvedTransitionState::default())),
            operation_shards: Some(
                (0..warp_count)
                    .map(|_| Mutex::new(Vec::new()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
        }
    }

    /// Start one stateless replay and discard every prior resolved effect.
    ///
    /// A [`DynamicOpId`] identifies a dynamic control-flow position, but its
    /// concrete address or synchronization arguments may legitimately change
    /// under a different schedule. Raw completion action IDs are also
    /// launch-local allocator results. Dependence must therefore use only the
    /// current replay's summaries; duplicate validation applies within that
    /// replay, not across different schedules.
    pub fn begin_replay(&self) {
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        state.operations.clear();
        state.completions.clear();
        state.completion_source_clocks.clear();
        state.setmaxnreg_participants.clear();
        state.setmaxnreg_request_resolutions.clear();
        state.tcgen_lifecycle_resolutions.clear();
        drop(state);
        if let Some(shards) = &self.operation_shards {
            for shard in shards.iter() {
                shard
                    .lock()
                    .expect("resolved transition operation shard poisoned")
                    .clear();
            }
        }
    }

    pub fn register_operation(
        &self,
        operation: DynamicOpId,
        summary: ResolvedTransitionSummary,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        if let Some(shards) = &self.operation_shards {
            let mut records = operation_shard(shards, &operation)
                .lock()
                .expect("resolved transition operation shard poisoned");
            return register_operation_summary_in_records(&mut records, operation, summary);
        }
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        register_operation_summary_in_state(&mut state, operation, summary)
    }

    /// Register one resolved operation effect.
    ///
    /// A committed `MbarrierCompletionIssue` must include its returned action
    /// IDs. The staged `action_ids: None` form is a different, incomplete
    /// identity and must not be registered before the committed form.
    pub fn register_operation_effect(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        let (summary, completion_hints) = prepare_operation_effect(operation, effect)?;
        if let Some(shards) = &self.operation_shards {
            let registration = if completion_hints.is_empty() {
                let mut records = operation_shard(shards, operation.id())
                    .lock()
                    .expect("resolved transition operation shard poisoned");
                register_operation_summary_in_records(
                    &mut records,
                    operation.id().clone(),
                    summary,
                )?
            } else {
                // Keep completion hints and their issuing operation
                // transactional. The global state is always locked before an
                // operation shard when both are needed.
                let mut state = self
                    .state
                    .write()
                    .expect("resolved transition log poisoned");
                let mut records = operation_shard(shards, operation.id())
                    .lock()
                    .expect("resolved transition operation shard poisoned");
                register_prepared_operation_effect_in_records(
                    &mut state,
                    &mut records,
                    operation,
                    summary,
                    completion_hints,
                )?
            };
            self.record_setmaxnreg_participant(operation, effect);
            return Ok(registration);
        }
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        let registration =
            register_prepared_operation_effect(&mut state, operation, summary, completion_hints)?;
        drop(state);
        self.record_setmaxnreg_participant(operation, effect);
        Ok(registration)
    }

    pub fn register_operation_effect_and_clock(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
        clock: Option<SyncVectorClock>,
    ) -> Result<
        (
            ResolvedTransitionRegistration,
            Option<ResolvedTransitionRegistration>,
        ),
        ResolvedTransitionLogError,
    > {
        let (summary, completion_hints) = prepare_operation_effect(operation, effect)?;
        if let Some(shards) = &self.operation_shards {
            let (effect_registration, clock_registration) = if completion_hints.is_empty() {
                let mut records = operation_shard(shards, operation.id())
                    .lock()
                    .expect("resolved transition operation shard poisoned");
                register_operation_summary_and_clock_in_records(
                    &mut records,
                    operation.id().clone(),
                    summary,
                    clock,
                )?
            } else {
                let mut state = self
                    .state
                    .write()
                    .expect("resolved transition log poisoned");
                let mut records = operation_shard(shards, operation.id())
                    .lock()
                    .expect("resolved transition operation shard poisoned");
                let effect_registration = register_prepared_operation_effect_in_records(
                    &mut state,
                    &mut records,
                    operation,
                    summary,
                    completion_hints,
                )?;
                let clock_registration = clock
                    .map(|clock| {
                        register_operation_clock_in_records(
                            &mut records,
                            operation.id().clone(),
                            clock,
                        )
                    })
                    .transpose()?;
                (effect_registration, clock_registration)
            };
            self.record_setmaxnreg_participant(operation, effect);
            return Ok((effect_registration, clock_registration));
        }
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        let effect_registration =
            register_prepared_operation_effect(&mut state, operation, summary, completion_hints)?;
        let clock_registration = clock
            .map(|clock| {
                register_operation_clock_in_state(&mut state, operation.id().clone(), clock)
            })
            .transpose()?;
        drop(state);
        self.record_setmaxnreg_participant(operation, effect);
        Ok((effect_registration, clock_registration))
    }

    pub fn register_completion(
        &self,
        completion: ResolvedCompletionEffect,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        let action_id = completion.action_id();
        let summary = ResolvedTransitionSummary::Completion(completion.without_memory_effects());
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        let update = inspect_completion_registration(&state.completions, action_id, &summary)?;
        if let Some(summary) = update.summary {
            state.completions.insert(action_id, summary);
        }
        Ok(update.registration)
    }

    pub fn register_completion_effect(
        &self,
        effect: CompletionEffect<'_>,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        match ResolvedTransitionSummary::from_completion_effect(effect) {
            ResolvedTransitionSummary::Completion(completion) => {
                self.register_completion(completion)
            }
            _ => unreachable!("completion effects always produce completion summaries"),
        }
    }

    pub fn register_operation_clock(
        &self,
        operation: DynamicOpId,
        clock: SyncVectorClock,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        if let Some(shards) = &self.operation_shards {
            let mut records = operation_shard(shards, &operation)
                .lock()
                .expect("resolved transition operation shard poisoned");
            return register_operation_clock_in_records(&mut records, operation, clock);
        }
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        register_operation_clock_in_state(&mut state, operation, clock)
    }

    pub fn register_completion_source_clock(
        &self,
        action_id: u64,
        clock: SyncVectorClock,
    ) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        match state.completion_source_clocks.get(&action_id) {
            Some(existing) if existing == &clock => {
                Ok(ResolvedTransitionRegistration::AlreadyRegistered)
            }
            Some(existing) => Err(
                ResolvedTransitionLogError::ConflictingCompletionSourceClock {
                    action_id,
                    existing: existing.clone(),
                    attempted: clock,
                },
            ),
            None => {
                state.completion_source_clocks.insert(action_id, clock);
                Ok(ResolvedTransitionRegistration::Inserted)
            }
        }
    }

    pub fn operation_clock(&self, operation: &DynamicOpId) -> Option<SyncVectorClock> {
        if let Some(shards) = &self.operation_shards {
            let records = operation_shard(shards, operation)
                .lock()
                .expect("resolved transition operation shard poisoned");
            return operation_record_in_sorted_slice(
                &records,
                ResolvedOperationPosition::from_operation(operation),
            )
            .filter(|record| record.operation == *operation)
            .and_then(|record| record.clock.clone());
        }
        self.state
            .read()
            .expect("resolved transition log poisoned")
            .operations
            .get(&ResolvedOperationPosition::from_operation(operation))
            .filter(|record| record.operation == *operation)
            .and_then(|record| record.clock.clone())
    }

    pub fn operation_summary(&self, operation: &DynamicOpId) -> Option<ResolvedTransitionSummary> {
        if let Some(shards) = &self.operation_shards {
            let records = operation_shard(shards, operation)
                .lock()
                .expect("resolved transition operation shard poisoned");
            return operation_record_in_sorted_slice(
                &records,
                ResolvedOperationPosition::from_operation(operation),
            )
            .filter(|record| record.operation == *operation)
            .and_then(|record| record.summary.clone());
        }
        self.state
            .read()
            .expect("resolved transition log poisoned")
            .operations
            .get(&ResolvedOperationPosition::from_operation(operation))
            .filter(|record| record.operation == *operation)
            .and_then(|record| record.summary.clone())
    }

    pub fn completion_summary(&self, action_id: u64) -> Option<ResolvedTransitionSummary> {
        self.state
            .read()
            .expect("resolved transition log poisoned")
            .completions
            .get(&action_id)
            .cloned()
    }

    pub fn operation_count(&self) -> usize {
        if let Some(shards) = &self.operation_shards {
            return shards
                .iter()
                .map(|shard| {
                    shard
                        .lock()
                        .expect("resolved transition operation shard poisoned")
                        .iter()
                        .filter(|record| record.summary.is_some())
                        .count()
                })
                .sum();
        }
        self.state
            .read()
            .expect("resolved transition log poisoned")
            .operations
            .values()
            .filter(|record| record.summary.is_some())
            .count()
    }

    pub fn completion_count(&self) -> usize {
        self.state
            .read()
            .expect("resolved transition log poisoned")
            .completions
            .len()
    }

    pub(crate) fn with_fixed_sync_snapshot<R>(
        &self,
        use_snapshot: impl FnOnce(FixedSyncLogSnapshot<'_>) -> R,
    ) -> R {
        let state = self.state.read().expect("resolved transition log poisoned");
        let operation_shards = self
            .operation_shards
            .as_ref()
            .map(|shards| {
                shards
                    .iter()
                    .map(|shard| {
                        shard
                            .lock()
                            .expect("resolved transition operation shard poisoned")
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let completion_mbarrier_generations = state
            .completions
            .iter()
            .filter_map(|(&action_id, summary)| {
                let ResolvedTransitionSummary::Completion(completion) = summary else {
                    return None;
                };
                let generations = completion
                    .resources()
                    .iter()
                    .filter_map(|resource| match (resource.key(), resource.generation()) {
                        (
                            ResolvedSyncResourceKey::PhysicalMbarrier(barrier_id),
                            Some(generation),
                        ) => Some((barrier_id, generation)),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                (!generations.is_empty()).then_some((action_id, generations))
            })
            .collect();
        let setmax_participants = state
            .setmaxnreg_participants
            .iter()
            .map(|(operation, record)| {
                let record = match record {
                    ResolvedSetmaxnregParticipantRecord::Known(participant) => {
                        FixedSyncSnapshotRecord::Known(FixedSyncSetmaxParticipantSnapshot {
                            request: participant.request,
                            action: participant.action,
                            count: participant.count,
                            operation_mask: participant.operation_mask,
                            plan_mask: participant.plan_mask,
                            plan_global_warp_id: participant.plan_global_warp_id,
                            plan_global_cta_id: participant.plan_global_cta_id,
                            plan_warp_id_in_cta: participant.plan_warp_id_in_cta,
                            plan_warps_per_cta: participant.plan_warps_per_cta,
                        })
                    }
                    ResolvedSetmaxnregParticipantRecord::Ambiguous => {
                        FixedSyncSnapshotRecord::Ambiguous
                    }
                };
                (operation.clone(), record)
            })
            .collect();
        let setmax_resolutions = state
            .setmaxnreg_request_resolutions
            .iter()
            .filter_map(|(key, record)| {
                let ResolvedSyncResourceKey::Setmaxnreg {
                    kernel_index,
                    global_cta_id,
                    warpgroup_id,
                    ordinal,
                } = *key
                else {
                    return None;
                };
                let request =
                    SetmaxnregResource::new(kernel_index, global_cta_id, warpgroup_id, ordinal);
                let record = match record {
                    ResolvedSetmaxnregRequestResolutionRecord::Known(resolution) => {
                        FixedSyncSnapshotRecord::Known(FixedSyncSetmaxResolutionSnapshot {
                            request: resolution.request,
                            action: resolution.action,
                            target_count: resolution.target_count,
                            warpgroup_count: resolution.warpgroup_count,
                            current_count_before: resolution.current_count_before,
                        })
                    }
                    ResolvedSetmaxnregRequestResolutionRecord::Ambiguous => {
                        FixedSyncSnapshotRecord::Ambiguous
                    }
                };
                Some((request, record))
            })
            .collect();
        let tcgen_allocations = state
            .tcgen_lifecycle_resolutions
            .iter()
            .map(|(operation, record)| {
                let record = match record {
                    ResolvedTcgenLifecycleResolutionRecord::Known(allocation) => {
                        FixedSyncSnapshotRecord::Known(*allocation)
                    }
                    ResolvedTcgenLifecycleResolutionRecord::Ambiguous => {
                        FixedSyncSnapshotRecord::Ambiguous
                    }
                };
                (operation.clone(), record)
            })
            .collect();
        let mut conditional_mbarrier_completions = BTreeMap::<_, BTreeMap<_, BTreeSet<_>>>::new();
        let mut snapshot = FixedSyncLogSnapshot {
            operations: self.operation_shards.is_none().then_some(&state.operations),
            operation_shards,
            completion_mbarrier_generations,
            conditional_mbarrier_completions: BTreeMap::new(),
            setmax_participants,
            setmax_resolutions,
            tcgen_allocations,
        };
        let conditional_barriers = snapshot.operations_with_clocks().filter_map(|(_, summary, _, _)| {
            let sync = match summary {
                ResolvedTransitionSummary::Synchronization(sync) => sync,
                ResolvedTransitionSummary::AsyncPayload(payload) => payload.synchronization(),
                _ => return None,
            };
            matches!(sync.details(), OwnedOperationEffect::MbarrierWait { plan, .. } if plan.is_conditional())
                .then_some(sync)
        }).flat_map(|sync| sync.resources().iter().filter_map(|resource| {
            match resource.key() {
                ResolvedSyncResourceKey::PhysicalMbarrier(barrier) => Some(barrier),
                _ => None,
            }
        })).collect::<BTreeSet<_>>();
        if conditional_barriers.is_empty() {
            return use_snapshot(snapshot);
        }
        // Retain the issuer as well as the primary generation: generation zero
        // may recur after reinitialization. The fixed projection binds each
        // issuer to its causally preceding init, never to a guessed epoch.
        for (operation, summary, _, _) in snapshot.operations_with_clocks() {
            let sync = match summary {
                ResolvedTransitionSummary::Synchronization(sync) => sync,
                ResolvedTransitionSummary::AsyncPayload(payload) => payload.synchronization(),
                _ => continue,
            };
            let mut record = |barrier, generation| {
                if !conditional_barriers.contains(&barrier) {
                    return;
                }
                conditional_mbarrier_completions
                    .entry(barrier)
                    .or_default()
                    .entry(operation.clone())
                    .or_default()
                    .insert(generation);
            };
            let mut record_async = |action_id| {
                if let Some(ResolvedTransitionSummary::Completion(completion)) =
                    state.completions.get(&action_id)
                {
                    for &(barrier, generation, conditional) in completion
                        .mbarrier_completions
                        .iter()
                        .flat_map(|phases| phases.iter())
                    {
                        if conditional {
                            record(barrier, generation);
                        }
                    }
                }
            };
            match sync.details() {
                OwnedOperationEffect::MbarrierArrive {
                    plan,
                    outcome: Some(outcome),
                } if outcome.completed_now()
                    && outcome.conditional_completed_generation() == Some(outcome.generation()) =>
                {
                    record(plan.barrier_id(), outcome.generation())
                }
                OwnedOperationEffect::MbarrierArriveBatch {
                    plan,
                    outcome: Some(outcome),
                } => {
                    for (entry, outcome) in plan.entries().iter().zip(outcome.outcomes().iter()) {
                        if outcome.completed_now()
                            && outcome.conditional_completed_generation()
                                == Some(outcome.generation())
                        {
                            record(entry.plan().barrier_id(), outcome.generation());
                        }
                    }
                }
                OwnedOperationEffect::MbarrierCompletionIssue {
                    action_ids: Some(ids),
                    ..
                } => {
                    for id in ids {
                        record_async(id.get());
                    }
                }
                OwnedOperationEffect::TcgenCommitIssue {
                    actions: Some(actions),
                    ..
                }
                | OwnedOperationEffect::CpAsyncMbarrierArrive {
                    actions: Some(actions),
                    ..
                } => {
                    for action in actions {
                        record_async(action.id().get());
                    }
                }
                _ => {}
            }
        }
        snapshot.conditional_mbarrier_completions = conditional_mbarrier_completions;
        use_snapshot(snapshot)
    }

    fn record_setmaxnreg_participant(
        &self,
        operation: &OperationContext,
        effect: OperationEffect<'_>,
    ) {
        let Some(participant) = ResolvedSetmaxnregParticipant::from_effect(operation, effect)
        else {
            return;
        };
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        match state.setmaxnreg_participants.get(operation.id()) {
            Some(ResolvedSetmaxnregParticipantRecord::Known(existing))
                if existing == &participant => {}
            Some(_) => {
                state.setmaxnreg_participants.insert(
                    operation.id().clone(),
                    ResolvedSetmaxnregParticipantRecord::Ambiguous,
                );
            }
            None => {
                state.setmaxnreg_participants.insert(
                    operation.id().clone(),
                    ResolvedSetmaxnregParticipantRecord::Known(participant),
                );
            }
        }
    }

    pub(crate) fn record_setmaxnreg_resume(&self, resume: &SetmaxnregResumePlan) {
        let request = setmaxnreg_request_key(resume.plan().resource());
        let resolution = ResolvedSetmaxnregRequestResolution::from_resume(resume);
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        match (
            state.setmaxnreg_request_resolutions.get(&request),
            resolution,
        ) {
            (
                Some(ResolvedSetmaxnregRequestResolutionRecord::Known(existing)),
                Some(resolution),
            ) if existing == &resolution => {}
            (None, Some(resolution)) => {
                state.setmaxnreg_request_resolutions.insert(
                    request,
                    ResolvedSetmaxnregRequestResolutionRecord::Known(resolution),
                );
            }
            _ => {
                state.setmaxnreg_request_resolutions.insert(
                    request,
                    ResolvedSetmaxnregRequestResolutionRecord::Ambiguous,
                );
            }
        }
    }

    pub(crate) fn record_tcgen_lifecycle_resume(
        &self,
        operation: &DynamicOpId,
        resume: &TcgenLifecycleResumePlan,
    ) {
        let allocation = resume
            .allocation()
            .map(|allocation| (allocation.base_column, allocation.columns));
        let mut state = self
            .state
            .write()
            .expect("resolved transition log poisoned");
        match state.tcgen_lifecycle_resolutions.get(operation) {
            Some(ResolvedTcgenLifecycleResolutionRecord::Known(existing))
                if existing == &allocation => {}
            None => {
                state.tcgen_lifecycle_resolutions.insert(
                    operation.clone(),
                    ResolvedTcgenLifecycleResolutionRecord::Known(allocation),
                );
            }
            _ => {
                state.tcgen_lifecycle_resolutions.insert(
                    operation.clone(),
                    ResolvedTcgenLifecycleResolutionRecord::Ambiguous,
                );
            }
        }
    }
}

fn setmaxnreg_request_key(resource: SetmaxnregResource) -> ResolvedSyncResourceKey {
    ResolvedSyncResourceKey::Setmaxnreg {
        kernel_index: resource.kernel_index(),
        global_cta_id: resource.global_cta_id(),
        warpgroup_id: resource.warpgroup_id(),
        ordinal: resource.ordinal(),
    }
}

fn prepare_operation_effect(
    operation: &OperationContext,
    effect: OperationEffect<'_>,
) -> Result<
    (
        ResolvedTransitionSummary,
        BTreeMap<u64, ResolvedCompletionEffect>,
    ),
    ResolvedTransitionLogError,
> {
    let effect_operation = match effect {
        OperationEffect::PhysicalAccess(batch) => Some(batch.operation()),
        OperationEffect::AsyncPayload(payload) => Some(payload.operation()),
        OperationEffect::AsyncGroupIssue(issue) => Some(issue.operation()),
        OperationEffect::TcgenWorkIssue(issue) => Some(issue.operation()),
        _ => None,
    };
    if let Some(actual) = effect_operation {
        if actual.id() != operation.id() {
            return Err(ResolvedTransitionLogError::OperationEffectMismatch {
                expected: operation.id().clone(),
                actual: actual.id().clone(),
            });
        }
    }
    let completion_hints = completion_hints(operation, effect)?;
    let summary = ResolvedTransitionSummary::from_operation_effect(effect);
    Ok((summary, completion_hints))
}

fn operation_shard<'a>(
    shards: &'a [Mutex<Vec<ResolvedOperationRecord>>],
    operation: &DynamicOpId,
) -> &'a Mutex<Vec<ResolvedOperationRecord>> {
    shards
        .get(operation.global_warp_id())
        .expect("fixed-transition operation warp belongs to its launch topology")
}

fn operation_record_index(
    records: &[ResolvedOperationRecord],
    position: ResolvedOperationPosition,
) -> Result<usize, usize> {
    if records.last().is_none_or(|record| {
        ResolvedOperationPosition::from_operation(&record.operation) < position
    }) {
        return Err(records.len());
    }
    records.binary_search_by_key(&position, |record| {
        ResolvedOperationPosition::from_operation(&record.operation)
    })
}

fn operation_record_in_sorted_slice(
    records: &[ResolvedOperationRecord],
    position: ResolvedOperationPosition,
) -> Option<&ResolvedOperationRecord> {
    operation_record_index(records, position)
        .ok()
        .and_then(|index| records.get(index))
}

fn inspect_operation_registration_in_records(
    records: &[ResolvedOperationRecord],
    operation: &DynamicOpId,
    attempted: &ResolvedTransitionSummary,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let position = ResolvedOperationPosition::from_operation(operation);
    let Some(record) = operation_record_in_sorted_slice(records, position) else {
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    if record.operation != *operation {
        return Err(ResolvedTransitionLogError::ConflictingOperation {
            operation: operation.clone(),
            existing: Box::new(record.summary.clone().unwrap_or_else(|| {
                ResolvedTransitionSummary::unknown(
                    "the same kernel/warp/sequence position already has a different identity",
                )
            })),
            attempted: Box::new(attempted.clone()),
        });
    }
    let Some(existing) = record.summary.as_ref() else {
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    if existing == attempted {
        Ok(ResolvedTransitionRegistration::AlreadyRegistered)
    } else {
        Err(ResolvedTransitionLogError::ConflictingOperation {
            operation: operation.clone(),
            existing: Box::new(existing.clone()),
            attempted: Box::new(attempted.clone()),
        })
    }
}

fn insert_operation_summary_in_records(
    records: &mut Vec<ResolvedOperationRecord>,
    operation: DynamicOpId,
    summary: ResolvedTransitionSummary,
) {
    let position = ResolvedOperationPosition::from_operation(&operation);
    match operation_record_index(records, position) {
        Ok(index) => {
            let record = &mut records[index];
            debug_assert_eq!(record.operation, operation);
            debug_assert!(record.summary.is_none());
            record.summary = Some(summary);
        }
        Err(index) => records.insert(
            index,
            ResolvedOperationRecord {
                operation,
                summary: Some(summary),
                initial_clock: None,
                clock: None,
            },
        ),
    }
}

fn register_operation_summary_in_records(
    records: &mut Vec<ResolvedOperationRecord>,
    operation: DynamicOpId,
    summary: ResolvedTransitionSummary,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let position = ResolvedOperationPosition::from_operation(&operation);
    let Ok(index) = operation_record_index(records, position) else {
        insert_operation_summary_in_records(records, operation, summary);
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    let record = &mut records[index];
    if record.operation != operation {
        return Err(ResolvedTransitionLogError::ConflictingOperation {
            operation,
            existing: Box::new(record.summary.clone().unwrap_or_else(|| {
                ResolvedTransitionSummary::unknown(
                    "the same kernel/warp/sequence position already has a different identity",
                )
            })),
            attempted: Box::new(summary),
        });
    }
    match record.summary.as_ref() {
        Some(existing) if existing == &summary => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) => {
            let Some(merged) = merge_lane_varying_summaries(existing, &summary) else {
                return Err(ResolvedTransitionLogError::ConflictingOperation {
                    operation,
                    existing: Box::new(existing.clone()),
                    attempted: Box::new(summary),
                });
            };
            record.summary = Some(merged);
            Ok(ResolvedTransitionRegistration::Refined)
        }
        None => {
            record.summary = Some(summary);
            Ok(ResolvedTransitionRegistration::Inserted)
        }
    }
}

fn register_operation_summary_and_clock_in_records(
    records: &mut Vec<ResolvedOperationRecord>,
    operation: DynamicOpId,
    summary: ResolvedTransitionSummary,
    clock: Option<SyncVectorClock>,
) -> Result<
    (
        ResolvedTransitionRegistration,
        Option<ResolvedTransitionRegistration>,
    ),
    ResolvedTransitionLogError,
> {
    let position = ResolvedOperationPosition::from_operation(&operation);
    let record = match operation_record_index(records, position) {
        Ok(index) => &mut records[index],
        Err(index) => {
            let clock_registration = clock
                .as_ref()
                .map(|_| ResolvedTransitionRegistration::Inserted);
            records.insert(
                index,
                ResolvedOperationRecord {
                    operation,
                    summary: Some(summary),
                    initial_clock: None,
                    clock,
                },
            );
            return Ok((ResolvedTransitionRegistration::Inserted, clock_registration));
        }
    };
    if record.operation != operation {
        return Err(ResolvedTransitionLogError::ConflictingOperation {
            operation,
            existing: Box::new(record.summary.clone().unwrap_or_else(|| {
                ResolvedTransitionSummary::unknown(
                    "the same kernel/warp/sequence position already has a different identity",
                )
            })),
            attempted: Box::new(summary),
        });
    }
    let effect_registration = match record.summary.as_ref() {
        Some(existing) if existing == &summary => ResolvedTransitionRegistration::AlreadyRegistered,
        Some(existing) => match merge_lane_varying_summaries(existing, &summary) {
            Some(merged) => {
                record.summary = Some(merged);
                ResolvedTransitionRegistration::Refined
            }
            None => {
                return Err(ResolvedTransitionLogError::ConflictingOperation {
                    operation,
                    existing: Box::new(existing.clone()),
                    attempted: Box::new(summary),
                });
            }
        },
        None => {
            record.summary = Some(summary);
            ResolvedTransitionRegistration::Inserted
        }
    };
    let Some(clock) = clock else {
        return Ok((effect_registration, None));
    };
    let clock_registration = match record.clock.as_ref() {
        Some(existing) if existing == &clock => ResolvedTransitionRegistration::AlreadyRegistered,
        Some(existing) if existing.happens_before(&clock) => {
            if record.initial_clock.is_none() {
                record.initial_clock = Some(existing.clone());
            }
            record.clock = Some(clock);
            ResolvedTransitionRegistration::Refined
        }
        Some(existing) if clock.happens_before(existing) => {
            ResolvedTransitionRegistration::AlreadyRegistered
        }
        Some(existing) => {
            return Err(ResolvedTransitionLogError::ConflictingOperationClock {
                operation,
                existing: existing.clone(),
                attempted: clock,
            });
        }
        None => {
            record.initial_clock = None;
            record.clock = Some(clock);
            ResolvedTransitionRegistration::Inserted
        }
    };
    Ok((effect_registration, Some(clock_registration)))
}

fn register_operation_clock_in_records(
    records: &mut Vec<ResolvedOperationRecord>,
    operation: DynamicOpId,
    clock: SyncVectorClock,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let position = ResolvedOperationPosition::from_operation(&operation);
    let record = match operation_record_index(records, position) {
        Ok(index) => &mut records[index],
        Err(index) => {
            records.insert(
                index,
                ResolvedOperationRecord {
                    operation,
                    summary: None,
                    initial_clock: None,
                    clock: Some(clock),
                },
            );
            return Ok(ResolvedTransitionRegistration::Inserted);
        }
    };
    if record.operation != operation {
        return Err(ResolvedTransitionLogError::ConflictingOperationClock {
            operation,
            existing: record.clock.clone().unwrap_or_else(|| clock.clone()),
            attempted: clock,
        });
    }
    match record.clock.as_ref() {
        Some(existing) if existing == &clock => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) if existing.happens_before(&clock) => {
            if record.initial_clock.is_none() {
                record.initial_clock = Some(existing.clone());
            }
            record.clock = Some(clock);
            Ok(ResolvedTransitionRegistration::Refined)
        }
        Some(existing) if clock.happens_before(existing) => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) => Err(ResolvedTransitionLogError::ConflictingOperationClock {
            operation,
            existing: existing.clone(),
            attempted: clock,
        }),
        None => {
            record.initial_clock = None;
            record.clock = Some(clock);
            Ok(ResolvedTransitionRegistration::Inserted)
        }
    }
}

fn register_prepared_operation_effect_in_records(
    state: &mut ResolvedTransitionState,
    records: &mut Vec<ResolvedOperationRecord>,
    operation: &OperationContext,
    summary: ResolvedTransitionSummary,
    completion_hints: BTreeMap<u64, ResolvedCompletionEffect>,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let operation_registration =
        inspect_operation_registration_in_records(records, operation.id(), &summary)?;
    let mut completion_updates = Vec::with_capacity(completion_hints.len());
    for completion in completion_hints.into_values() {
        let action_id = completion.action_id();
        let attempted = ResolvedTransitionSummary::Completion(completion.without_memory_effects());
        completion_updates.push((
            action_id,
            inspect_completion_registration(&state.completions, action_id, &attempted)?,
        ));
    }

    if operation_registration == ResolvedTransitionRegistration::Inserted {
        insert_operation_summary_in_records(records, operation.id().clone(), summary);
    }
    let mut registration = operation_registration;
    for (action_id, update) in completion_updates {
        registration = combine_registration(registration, update.registration);
        if let Some(summary) = update.summary {
            state.completions.insert(action_id, summary);
        }
    }
    Ok(registration)
}

fn register_prepared_operation_effect(
    state: &mut ResolvedTransitionState,
    operation: &OperationContext,
    summary: ResolvedTransitionSummary,
    completion_hints: BTreeMap<u64, ResolvedCompletionEffect>,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    if completion_hints.is_empty() {
        return register_operation_summary_in_state(state, operation.id().clone(), summary);
    }

    let operation_registration =
        inspect_operation_registration(&state.operations, operation.id(), &summary)?;
    let mut completion_updates = Vec::with_capacity(completion_hints.len());
    for completion in completion_hints.into_values() {
        let action_id = completion.action_id();
        let attempted = ResolvedTransitionSummary::Completion(completion.without_memory_effects());
        completion_updates.push((
            action_id,
            inspect_completion_registration(&state.completions, action_id, &attempted)?,
        ));
    }

    if operation_registration == ResolvedTransitionRegistration::Inserted {
        insert_operation_summary(&mut state.operations, operation.id().clone(), summary);
    }
    let mut registration = operation_registration;
    for (action_id, update) in completion_updates {
        registration = combine_registration(registration, update.registration);
        if let Some(summary) = update.summary {
            state.completions.insert(action_id, summary);
        }
    }
    Ok(registration)
}

fn register_operation_clock_in_state(
    state: &mut ResolvedTransitionState,
    operation: DynamicOpId,
    clock: SyncVectorClock,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let position = ResolvedOperationPosition::from_operation(&operation);
    let Some(record) = state.operations.get_mut(&position) else {
        state.operations.insert(
            position,
            ResolvedOperationRecord {
                operation,
                summary: None,
                initial_clock: None,
                clock: Some(clock),
            },
        );
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    if record.operation != operation {
        return Err(ResolvedTransitionLogError::ConflictingOperationClock {
            operation,
            existing: record.clock.clone().unwrap_or_else(|| clock.clone()),
            attempted: clock,
        });
    }
    match record.clock.as_ref() {
        Some(existing) if existing == &clock => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) if existing.happens_before(&clock) => {
            if record.initial_clock.is_none() {
                record.initial_clock = Some(existing.clone());
            }
            record.clock = Some(clock);
            Ok(ResolvedTransitionRegistration::Refined)
        }
        Some(existing) if clock.happens_before(existing) => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) => Err(ResolvedTransitionLogError::ConflictingOperationClock {
            operation,
            existing: existing.clone(),
            attempted: clock,
        }),
        None => {
            record.initial_clock = None;
            record.clock = Some(clock);
            Ok(ResolvedTransitionRegistration::Inserted)
        }
    }
}

fn completion_hints(
    operation: &OperationContext,
    effect: OperationEffect<'_>,
) -> Result<BTreeMap<u64, ResolvedCompletionEffect>, ResolvedTransitionLogError> {
    if let OperationEffect::CpAsyncMbarrierArrive {
        outcome: Some(outcome),
        actions: Some(physical_actions),
        ..
    } = effect
    {
        let mut hints = BTreeMap::new();
        for group in outcome.groups() {
            for action in [group.source_read_action(), group.full_action()] {
                let completion = ResolvedCompletionEffect::from_async_group_action(&action);
                if hints.insert(action.id().get(), completion).is_some() {
                    return Err(
                        ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                            operation: operation.id().clone(),
                            action_id: action.id().get(),
                        },
                    );
                }
            }
        }
        for &action in physical_actions {
            let action_id = action.id().get();
            if hints
                .insert(action_id, ResolvedCompletionEffect::from_action(action))
                .is_some()
            {
                return Err(
                    ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                        operation: operation.id().clone(),
                        action_id,
                    },
                );
            }
        }
        return Ok(hints);
    }
    if let OperationEffect::TcgenCommitIssue {
        actions: Some(actions),
        ..
    } = effect
    {
        let mut hints = BTreeMap::new();
        for &action in actions {
            if hints
                .insert(
                    action.id().get(),
                    ResolvedCompletionEffect::from_action(action),
                )
                .is_some()
            {
                return Err(
                    ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                        operation: operation.id().clone(),
                        action_id: action.id().get(),
                    },
                );
            }
        }
        return Ok(hints);
    }
    if let OperationEffect::AsyncGroupCommit {
        outcome: Some(outcome),
        ..
    } = effect
    {
        let mut hints = BTreeMap::new();
        for group in outcome.groups() {
            for action in [group.source_read_action(), group.full_action()] {
                let completion = ResolvedCompletionEffect::from_async_group_action(&action);
                if hints.insert(action.id().get(), completion).is_some() {
                    return Err(
                        ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                            operation: operation.id().clone(),
                            action_id: action.id().get(),
                        },
                    );
                }
            }
        }
        return Ok(hints);
    }
    if let OperationEffect::AsyncPayload(payload) = effect {
        let Some(action_ids) = payload.completion_action_ids() else {
            return Ok(BTreeMap::new());
        };
        let plan = payload.completion_plan();
        if plan.completions().len() != action_ids.len() {
            return Err(
                ResolvedTransitionLogError::CompletionIssueActionCountMismatch {
                    operation: operation.id().clone(),
                    completion_count: plan.completions().len(),
                    action_id_count: action_ids.len(),
                },
            );
        }
        let targets = plan
            .completions()
            .iter()
            .zip(action_ids.iter())
            .filter(|((_, transactions), _)| *transactions != 0);
        let Some((_, scheduler_action_id)) = targets.clone().next() else {
            return Ok(BTreeMap::new());
        };
        let scheduler_action_id = scheduler_action_id.get();
        let resources = targets
            .clone()
            .map(|(&(barrier_id, _), _)| ResolvedSyncResource::physical_mbarrier(barrier_id, None))
            .collect::<Vec<_>>();
        let transactions = targets
            .map(|(&(.., transactions), _)| transactions)
            .sum::<u64>();
        let completion = ResolvedCompletionEffect::new_deferred_payload(
            scheduler_action_id,
            payload.token().clone(),
            resources,
            payload
                .completion_accesses()
                .iter()
                .map(ResolvedMemoryEffect::from_batch),
            transactions,
        );
        return Ok(BTreeMap::from([(scheduler_action_id, completion)]));
    }

    let (plan, action_ids) = match effect {
        OperationEffect::MbarrierCompletionIssue {
            plan,
            action_ids: Some(action_ids),
        } => (plan, action_ids),
        _ => return Ok(BTreeMap::new()),
    };
    if plan.completions().len() != action_ids.len() {
        return Err(
            ResolvedTransitionLogError::CompletionIssueActionCountMismatch {
                operation: operation.id().clone(),
                completion_count: plan.completions().len(),
                action_id_count: action_ids.len(),
            },
        );
    }

    let mut hints = BTreeMap::new();
    for (&(barrier_id, transactions), &action_id) in
        plan.completions().iter().zip(action_ids.iter())
    {
        if transactions == 0 {
            continue;
        }
        let action_id = action_id.get();
        let completion = ResolvedCompletionEffect::new(
            action_id,
            ResolvedSyncResource::physical_mbarrier(barrier_id, None),
            transactions,
        );
        if hints.insert(action_id, completion).is_some() {
            return Err(
                ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                    operation: operation.id().clone(),
                    action_id,
                },
            );
        }
    }
    Ok(hints)
}

fn inspect_operation_registration(
    entries: &BTreeMap<ResolvedOperationPosition, ResolvedOperationRecord>,
    operation: &DynamicOpId,
    attempted: &ResolvedTransitionSummary,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let Some(record) = entries.get(&ResolvedOperationPosition::from_operation(operation)) else {
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    if record.operation != *operation {
        return Err(ResolvedTransitionLogError::ConflictingOperation {
            operation: operation.clone(),
            existing: Box::new(record.summary.clone().unwrap_or_else(|| {
                ResolvedTransitionSummary::unknown(
                    "the same kernel/warp/sequence position already has a different identity",
                )
            })),
            attempted: Box::new(attempted.clone()),
        });
    }
    let Some(existing) = record.summary.as_ref() else {
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    match existing {
        existing if existing == attempted => Ok(ResolvedTransitionRegistration::AlreadyRegistered),
        existing => Err(ResolvedTransitionLogError::ConflictingOperation {
            operation: operation.clone(),
            existing: Box::new(existing.clone()),
            attempted: Box::new(attempted.clone()),
        }),
    }
}

fn insert_operation_summary(
    entries: &mut BTreeMap<ResolvedOperationPosition, ResolvedOperationRecord>,
    operation: DynamicOpId,
    summary: ResolvedTransitionSummary,
) {
    let position = ResolvedOperationPosition::from_operation(&operation);
    match entries.get_mut(&position) {
        Some(record) => {
            debug_assert_eq!(record.operation, operation);
            debug_assert!(record.summary.is_none());
            record.summary = Some(summary);
        }
        None => {
            entries.insert(
                position,
                ResolvedOperationRecord {
                    operation,
                    summary: Some(summary),
                    initial_clock: None,
                    clock: None,
                },
            );
        }
    }
}

fn register_operation_summary_in_state(
    state: &mut ResolvedTransitionState,
    operation: DynamicOpId,
    summary: ResolvedTransitionSummary,
) -> Result<ResolvedTransitionRegistration, ResolvedTransitionLogError> {
    let position = ResolvedOperationPosition::from_operation(&operation);
    let Some(record) = state.operations.get_mut(&position) else {
        insert_operation_summary(&mut state.operations, operation, summary);
        return Ok(ResolvedTransitionRegistration::Inserted);
    };
    if record.operation != operation {
        return Err(ResolvedTransitionLogError::ConflictingOperation {
            operation,
            existing: Box::new(record.summary.clone().unwrap_or_else(|| {
                ResolvedTransitionSummary::unknown(
                    "the same kernel/warp/sequence position already has a different identity",
                )
            })),
            attempted: Box::new(summary),
        });
    }
    match record.summary.as_ref() {
        Some(existing) if existing == &summary => {
            Ok(ResolvedTransitionRegistration::AlreadyRegistered)
        }
        Some(existing) => {
            let Some(merged) = merge_lane_varying_summaries(existing, &summary) else {
                return Err(ResolvedTransitionLogError::ConflictingOperation {
                    operation,
                    existing: Box::new(existing.clone()),
                    attempted: Box::new(summary),
                });
            };
            record.summary = Some(merged);
            Ok(ResolvedTransitionRegistration::Refined)
        }
        None => {
            record.summary = Some(summary);
            Ok(ResolvedTransitionRegistration::Inserted)
        }
    }
}

struct CompletionRegistrationUpdate {
    registration: ResolvedTransitionRegistration,
    summary: Option<ResolvedTransitionSummary>,
}

fn inspect_completion_registration(
    entries: &BTreeMap<u64, ResolvedTransitionSummary>,
    action_id: u64,
    attempted: &ResolvedTransitionSummary,
) -> Result<CompletionRegistrationUpdate, ResolvedTransitionLogError> {
    let Some(existing) = entries.get(&action_id) else {
        return Ok(CompletionRegistrationUpdate {
            registration: ResolvedTransitionRegistration::Inserted,
            summary: Some(attempted.clone()),
        });
    };
    if existing == attempted {
        return Ok(CompletionRegistrationUpdate {
            registration: ResolvedTransitionRegistration::AlreadyRegistered,
            summary: None,
        });
    }

    if let (
        ResolvedTransitionSummary::Completion(existing_completion),
        ResolvedTransitionSummary::Completion(attempted_completion),
    ) = (existing, attempted)
    {
        let same_action = existing_completion.action_id == attempted_completion.action_id;
        let same_token = existing_completion.token == attempted_completion.token;
        let same_memory = existing_completion.memory == attempted_completion.memory;
        let same_resource_shape = existing_completion.resources.len()
            == attempted_completion.resources.len()
            && existing_completion
                .resources
                .iter()
                .zip(attempted_completion.resources.iter())
                .all(|(existing, attempted)| existing.key == attempted.key);
        let same_transactions =
            existing_completion.transactions == attempted_completion.transactions;
        if same_action && same_token && same_memory && same_resource_shape && same_transactions {
            let existing_to_attempted = existing_completion
                .resources
                .iter()
                .zip(attempted_completion.resources.iter())
                .all(|(existing, attempted)| {
                    existing.generation == attempted.generation
                        || (existing.generation.is_none() && attempted.generation.is_some())
                })
                && (existing_completion.mbarrier_completions
                    == attempted_completion.mbarrier_completions
                    || existing_completion.mbarrier_completions.is_none());
            let attempted_to_existing = existing_completion
                .resources
                .iter()
                .zip(attempted_completion.resources.iter())
                .all(|(existing, attempted)| {
                    existing.generation == attempted.generation
                        || (existing.generation.is_some() && attempted.generation.is_none())
                })
                && (existing_completion.mbarrier_completions
                    == attempted_completion.mbarrier_completions
                    || attempted_completion.mbarrier_completions.is_none());
            if existing_to_attempted {
                return Ok(CompletionRegistrationUpdate {
                    registration: ResolvedTransitionRegistration::Refined,
                    summary: Some(attempted.clone()),
                });
            }
            if attempted_to_existing {
                return Ok(CompletionRegistrationUpdate {
                    registration: ResolvedTransitionRegistration::AlreadyRegistered,
                    summary: None,
                });
            }
        }
    }

    Err(ResolvedTransitionLogError::ConflictingCompletion {
        action_id,
        existing: Box::new(existing.clone()),
        attempted: Box::new(attempted.clone()),
    })
}

const fn combine_registration(
    left: ResolvedTransitionRegistration,
    right: ResolvedTransitionRegistration,
) -> ResolvedTransitionRegistration {
    use ResolvedTransitionRegistration::{AlreadyRegistered, Inserted, Refined};
    match (left, right) {
        (Inserted, _) | (_, Inserted) => Inserted,
        (Refined, _) | (_, Refined) => Refined,
        (AlreadyRegistered, AlreadyRegistered) => AlreadyRegistered,
    }
}

fn canonicalize_spans(
    spans: impl IntoIterator<Item = PhysicalByteSpan>,
) -> Box<[PhysicalByteSpan]> {
    let mut spans = spans.into_iter().collect::<Vec<_>>();
    spans.sort_unstable();
    let mut canonical = Vec::<PhysicalByteSpan>::with_capacity(spans.len());
    for span in spans {
        let Some(previous) = canonical.last_mut() else {
            canonical.push(span);
            continue;
        };
        if previous.allocation() != span.allocation() || previous.byte_end() < span.byte_offset() {
            canonical.push(span);
            continue;
        }
        let byte_end = previous.byte_end().max(span.byte_end());
        *previous = PhysicalByteSpan::new(
            previous.allocation(),
            previous.byte_offset(),
            byte_end - previous.byte_offset(),
        )
        .expect("the union of valid physical spans is valid");
    }
    canonical.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use crate::runtime::{
        plan_cluster_barrier_arrive, plan_named_barrier_arrive, plan_physical_mbarrier_arrive,
        plan_physical_mbarrier_arrive_lanes, plan_physical_mbarrier_init,
        plan_physical_mbarrier_wait, plan_tcgen_allocate, ClusterBarrierArrivalSemantics,
        PhysicalMbarrierCompletionIssuePlan, PhysicalMbarrierCompletionTargets, PhysicalPtr,
        RuntimeBuffer,
    };
    use crate::{
        AnalysisGapEffect, AnalysisGapKind, AsyncPayloadEffect, CtaId, LaunchTopology, LoopFrame,
        NamedBarrierArrivalOutcome, OperationKind, PhysicalAccessDescriptor, PhysicalAllocationId,
        PhysicalBarrierHub, PhysicalMemory, StaticOpId, WarpContext, WarpMask, WarpValue,
    };

    use super::*;

    fn operation(warp: usize, sequence: u64, kind: OperationKind) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(
                3,
                warp,
                sequence,
                StaticOpId::new(100 + sequence),
                [LoopFrame::new(StaticOpId::new(9), 2)],
            ),
            kind,
            WarpMask::from_lanes([0]).unwrap(),
        )
    }

    #[test]
    fn paired_cta_lifecycle_has_shared_resources() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = |global_cta_id| {
            topology
                .warp_contexts()
                .find(|context| context.global_cta_id() == global_cta_id)
                .unwrap()
        };
        let first = plan_tcgen_allocate(3, 7, [0], context(0), 64, 2).unwrap();
        let second = plan_tcgen_allocate(3, 7, [0], context(1), 64, 2).unwrap();
        let first = ResolvedTransitionSummary::from_operation_effect(
            OperationEffect::TcgenLifecycleRegister(&first),
        );
        let second = ResolvedTransitionSummary::from_operation_effect(
            OperationEffect::TcgenLifecycleRegister(&second),
        );

        let ResolvedTransitionSummary::Synchronization(second) = second else {
            panic!("TCGEN lifecycle must resolve as synchronization")
        };
        let ResolvedTransitionSummary::Synchronization(first) = first else {
            panic!("TCGEN lifecycle must resolve as synchronization")
        };
        assert_eq!(first.resources(), second.resources());
        assert_eq!(first.resources().len(), 2);
        assert_eq!(
            first.resources()[0].key(),
            ResolvedSyncResourceKey::TcgenLifecycleCta {
                kernel_index: 3,
                global_cta_id: 0,
            }
        );
        assert_eq!(
            first.resources()[1].key(),
            ResolvedSyncResourceKey::TcgenLifecycleCta {
                kernel_index: 3,
                global_cta_id: 1,
            }
        );
    }

    fn span(allocation: u64, offset: usize, len: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(PhysicalAllocationId::new(allocation), offset, len).unwrap()
    }

    fn memory_summary(
        kind: PhysicalAccessKind,
        allocation: u64,
        offset: usize,
        len: usize,
    ) -> ResolvedTransitionSummary {
        ResolvedTransitionSummary::Memory(ResolvedMemoryEffect::new(
            kind,
            PhysicalAccessSpace::Shared,
            [span(allocation, offset, len)],
        ))
    }

    #[test]
    fn tcgen_analysis_gap_exposes_global_and_cta_resources_and_fails_closed() {
        let summary =
            ResolvedTransitionSummary::from_operation_effect(OperationEffect::AnalysisGap(
                AnalysisGapEffect::new(AnalysisGapKind::TcgenMma, 3, 7, Some(2)),
            ));
        let ResolvedTransitionSummary::AnalysisGap(gap) = &summary else {
            panic!("analysis gap summary was not preserved")
        };
        assert_eq!(gap.kind(), AnalysisGapKind::TcgenMma);
        assert_eq!(gap.cta_group(), Some(2));
        assert_eq!(
            gap.resources(),
            [
                ResolvedAnalysisResource::TcgenGlobal { kernel_index: 3 },
                ResolvedAnalysisResource::TcgenCta {
                    kernel_index: 3,
                    global_cta_id: 7,
                },
            ]
        );
    }

    #[test]
    fn lane_varying_arrive_is_one_summary_with_every_target_resource() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = (0..2)
            .map(|global_cta_id| {
                physical
                    .shared()
                    .allocate_cta_zeroed(CtaId::new(topology, 0, global_cta_id).unwrap(), 8)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(allocations),
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            8,
        );
        let targets = WarpValue::from_fn(|lane| if lane == 0 { 0_i64 } else { 1_i64 });
        let plan = plan_physical_mbarrier_arrive_lanes(
            &context,
            &pointer,
            WarpMask::from_lanes([0, 1]).unwrap(),
            Some(&targets),
            None,
            None,
        )
        .unwrap();
        let summary = ResolvedTransitionSummary::from_operation_effect(
            OperationEffect::MbarrierArriveBatch {
                plan: &plan,
                outcome: None,
            },
        );
        let ResolvedTransitionSummary::Synchronization(sync) = summary else {
            panic!("batch arrival must remain one synchronization summary")
        };
        assert_eq!(sync.resources().len(), 2);
        assert!(matches!(
            sync.details(),
            OwnedOperationEffect::MbarrierArriveBatch { plan, .. }
                if plan.entries().len() == 2
        ));
    }

    #[test]
    fn named_barrier_arrive_summary_carries_exact_generation_and_resource() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let plan = plan_named_barrier_arrive(&context, 7, 32, WarpMask::FULL)
            .unwrap()
            .unwrap();
        let summary =
            ResolvedTransitionSummary::from_operation_effect(OperationEffect::NamedBarrierArrive {
                plan,
                outcome: Some(NamedBarrierArrivalOutcome::new(3, true)),
            });
        let ResolvedTransitionSummary::Synchronization(summary) = summary else {
            panic!("named bar.arrive must resolve as synchronization")
        };
        assert_eq!(
            summary.resources(),
            &[ResolvedSyncResource::named_barrier(
                NamedBarrierId::new(0, 7),
                Some(3)
            )]
        );
        assert!(matches!(
            summary.details(),
            OwnedOperationEffect::NamedBarrierArrive { plan, .. }
                if plan.expected_arrivals() == 32
                    && plan.warp_id() == 0
                    && plan.arrival_mask().bits() == u32::MAX
        ));
    }

    #[test]
    fn duplicate_replay_registration_requires_identical_summary() {
        let log = ResolvedTransitionLog::default();
        let first_identity = operation(0, 5, OperationKind::Load);
        let replay_identity = operation(0, 5, OperationKind::Load);
        let summary = memory_summary(PhysicalAccessKind::Read, 1, 0, 4);

        assert_eq!(
            log.register_operation(first_identity.id().clone(), summary.clone())
                .unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        assert_eq!(
            log.register_operation(replay_identity.id().clone(), summary)
                .unwrap(),
            ResolvedTransitionRegistration::AlreadyRegistered
        );
        assert!(matches!(
            log.register_operation(
                replay_identity.id().clone(),
                memory_summary(PhysicalAccessKind::Read, 1, 4, 4)
            ),
            Err(ResolvedTransitionLogError::ConflictingOperation { operation, .. })
                if operation == *replay_identity.id()
        ));
    }

    #[test]
    fn concurrent_identical_registration_is_idempotent() {
        let log = ResolvedTransitionLog::default();
        let operation = operation(0, 0, OperationKind::Load);
        let summary = memory_summary(PhysicalAccessKind::Read, 1, 0, 4);
        let threads = (0..8)
            .map(|_| {
                let log = log.clone();
                let id = operation.id().clone();
                let summary = summary.clone();
                thread::spawn(move || log.register_operation(id, summary).unwrap())
            })
            .collect::<Vec<_>>();
        let outcomes = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            outcomes
                .iter()
                .filter(|&&outcome| outcome == ResolvedTransitionRegistration::Inserted)
                .count(),
            1
        );
        assert_eq!(log.operation_count(), 1);
    }

    #[test]
    fn physical_access_batches_produce_canonical_exact_spans() {
        let operation = OperationContext::new(
            DynamicOpId::new(0, 0, 0, StaticOpId::new(1), []),
            OperationKind::Store,
            WarpMask::from_lanes([0, 1]).unwrap(),
        );
        let descriptor = PhysicalAccessDescriptor::new(
            PhysicalAccessKind::Write,
            PhysicalAccessSpace::Shared,
            4,
        )
        .unwrap();
        let batch = PhysicalAccessBatch::resolve(operation, descriptor, |provenance| {
            Ok::<_, ()>(vec![span(8, provenance.lane() * 4, 4)])
        })
        .unwrap();

        let summary = ResolvedMemoryEffect::from_batch(&batch);

        assert_eq!(summary.spans(), &[span(8, 0, 8)]);
    }

    #[test]
    fn canonical_memory_effects_group_by_space_and_access_kind() {
        let effects = canonicalize_memory_effects([
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                [span(8, 0, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                [span(8, 4, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Read,
                PhysicalAccessSpace::Shared,
                [span(8, 16, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Global,
                [span(9, 0, 4)],
            ),
            ResolvedMemoryEffect::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Shared, []),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::AtomicReadModifyWrite,
                PhysicalAccessSpace::Shared,
                [],
            ),
        ]);

        assert_eq!(effects.len(), 3);
        assert_eq!(effects[0].space(), PhysicalAccessSpace::Global);
        assert_eq!(effects[0].kind(), PhysicalAccessKind::Write);
        assert_eq!(effects[0].spans(), &[span(9, 0, 4)]);
        assert_eq!(effects[1].space(), PhysicalAccessSpace::Shared);
        assert_eq!(effects[1].kind(), PhysicalAccessKind::Read);
        assert_eq!(effects[1].spans(), &[span(8, 16, 4)]);
        assert_eq!(effects[2].space(), PhysicalAccessSpace::Shared);
        assert_eq!(effects[2].kind(), PhysicalAccessKind::Write);
        assert_eq!(effects[2].spans(), &[span(8, 0, 8)]);
    }

    #[test]
    fn canonical_completion_memory_ignores_order_fragmentation_and_empty_effects() {
        let log = ResolvedTransitionLog::default();
        let issue = operation(0, 0, OperationKind::AsyncIssue);
        let token = AsyncTokenId::new(issue.id().clone(), 0);
        let resource =
            ResolvedSyncResource::physical_mbarrier(PhysicalBarrierId::new(31, 0, 0), Some(0));
        let fragmented = ResolvedCompletionEffect::new_deferred_payload(
            200,
            token.clone(),
            [resource],
            [
                ResolvedMemoryEffect::new(
                    PhysicalAccessKind::Write,
                    PhysicalAccessSpace::Shared,
                    [span(11, 4, 4)],
                ),
                ResolvedMemoryEffect::new(
                    PhysicalAccessKind::Read,
                    PhysicalAccessSpace::Shared,
                    [],
                ),
                ResolvedMemoryEffect::new(
                    PhysicalAccessKind::Write,
                    PhysicalAccessSpace::Shared,
                    [span(11, 0, 4)],
                ),
            ],
            1,
        );
        let merged = ResolvedCompletionEffect::new_deferred_payload(
            200,
            token,
            [resource],
            [ResolvedMemoryEffect::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                [span(11, 0, 8)],
            )],
            1,
        );

        assert_eq!(fragmented, merged);
        assert_eq!(
            log.register_completion(fragmented).unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        assert_eq!(
            log.register_completion(merged).unwrap(),
            ResolvedTransitionRegistration::AlreadyRegistered
        );
    }

    #[test]
    fn canonical_memory_preserves_atomic_kind_and_space_boundaries() {
        let effects = canonicalize_memory_effects([
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::AtomicReadModifyWrite,
                PhysicalAccessSpace::Shared,
                [span(12, 0, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Write,
                PhysicalAccessSpace::Shared,
                [span(12, 0, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Read,
                PhysicalAccessSpace::Shared,
                [span(12, 0, 4)],
            ),
            ResolvedMemoryEffect::new(
                PhysicalAccessKind::Read,
                PhysicalAccessSpace::Global,
                [span(12, 0, 4)],
            ),
        ]);

        assert_eq!(effects.len(), 4);
        assert_eq!(effects[1].kind(), PhysicalAccessKind::Read);
        assert_eq!(effects[2].kind(), PhysicalAccessKind::Write);
        assert_eq!(effects[3].kind(), PhysicalAccessKind::AtomicReadModifyWrite);
        assert_eq!(effects[0].space(), PhysicalAccessSpace::Global);
        assert_eq!(effects[3].space(), PhysicalAccessSpace::Shared);
    }

    fn shared_pointer(indices: WarpValue<i64>) -> (PhysicalMemory, WarpContext, PhysicalPtr) {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical.shared().allocate_cta_zeroed(owner, 256).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: Arc::new(vec![allocation]),
                byte_offset: 0,
                byte_len: 256,
                backing_byte_len: 256,
                virtual_base: 0,
            },
            indices,
            8,
        );
        (physical, context, pointer)
    }

    #[test]
    fn mbarrier_plans_and_completion_outcomes_build_resource_summaries() {
        let (_physical, context, pointer) = shared_pointer(WarpValue::splat(2));
        let mask = WarpMask::from_lanes([0]).unwrap();
        let init = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        let arrive = plan_physical_mbarrier_arrive(&context, &pointer, mask, None, Some(16))
            .unwrap()
            .unwrap();
        let wait = plan_physical_mbarrier_wait(&context, &pointer, mask, 0)
            .unwrap()
            .unwrap();
        let init_summary =
            ResolvedSynchronizationEffect::from_effect(OperationEffect::MbarrierInit(&init));
        let arrive_summary =
            ResolvedSynchronizationEffect::from_effect(OperationEffect::MbarrierArrive {
                plan: arrive,
                outcome: None,
            });
        let wait_summary =
            ResolvedSynchronizationEffect::from_effect(OperationEffect::MbarrierWait {
                plan: wait,
                outcome: None,
            });
        assert_eq!(init_summary.resources(), arrive_summary.resources());
        assert_eq!(arrive_summary.resources(), wait_summary.resources());

        let hub = PhysicalBarrierHub::new();
        init.apply(&hub).unwrap();
        arrive.apply(&hub).unwrap();
        let action_id = hub
            .enqueue_transaction_completion(arrive.barrier_id(), 16)
            .unwrap();
        let outcome = hub.apply_completion_detailed(action_id).unwrap();
        let completion = ResolvedCompletionEffect::from_outcome(&outcome);

        assert_eq!(completion.action_id(), action_id.get());
        assert_eq!(completion.resource().generation(), Some(0));
        assert_eq!(
            arrive_summary.resources()[0].key(),
            completion.resource().key()
        );
        assert_eq!(
            completion.mbarrier_completions.as_deref(),
            Some([(arrive.barrier_id(), 0, true)].as_slice()),
        );
    }

    #[test]
    fn lane_varying_mbarrier_waits_merge_every_target_for_one_operation() {
        let indices = WarpValue::from_fn(|lane| if lane == 0 { 0_i64 } else { 1_i64 });
        let (_physical, context, pointer) = shared_pointer(indices);
        let lane0 = WarpMask::from_lanes([0]).unwrap();
        let lane1 = WarpMask::from_lanes([1]).unwrap();
        let wait0 = plan_physical_mbarrier_wait(&context, &pointer, lane0, 0)
            .unwrap()
            .unwrap();
        let wait1 = plan_physical_mbarrier_wait(&context, &pointer, lane1, 1)
            .unwrap()
            .unwrap();
        let operation = operation(0, 7, OperationKind::Barrier);
        let log = ResolvedTransitionLog::default();

        assert_eq!(
            log.register_operation_effect(
                &operation,
                OperationEffect::MbarrierWait {
                    plan: wait0,
                    outcome: None,
                },
            )
            .unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        assert!(matches!(
            log.register_operation_effect(
                &operation,
                OperationEffect::MbarrierWait {
                    plan: wait1.with_acquire(false),
                    outcome: None,
                },
            ),
            Err(ResolvedTransitionLogError::ConflictingOperation { .. })
        ));
        assert_eq!(
            log.register_operation_effect(
                &operation,
                OperationEffect::MbarrierWait {
                    plan: wait1,
                    outcome: None,
                },
            )
            .unwrap(),
            ResolvedTransitionRegistration::Refined
        );

        let ResolvedTransitionSummary::Synchronization(summary) =
            log.operation_summary(operation.id()).unwrap()
        else {
            panic!("mbarrier wait must resolve as synchronization")
        };
        assert_eq!(summary.resources().len(), 2);
        assert_eq!(
            summary.mbarrier_wait_requests(),
            [(wait0.barrier_id(), 0, None), (wait1.barrier_id(), 1, None)]
        );
        assert_eq!(
            summary.resources()[0].key(),
            ResolvedSyncResourceKey::PhysicalMbarrier(wait0.barrier_id())
        );
        assert_eq!(
            summary.resources()[1].key(),
            ResolvedSyncResourceKey::PhysicalMbarrier(wait1.barrier_id())
        );
    }

    #[test]
    fn completion_issue_indexes_actions_then_outcome_refines_generation() {
        let log = ResolvedTransitionLog::default();
        let barrier = PhysicalBarrierId::new(22, 8, 0);
        let hub = PhysicalBarrierHub::new();
        hub.init(barrier, 1).unwrap();
        let plan = PhysicalMbarrierCompletionIssuePlan::single(barrier, 32);
        let action_ids = plan.apply(&hub).unwrap();
        let issue = operation(0, 0, OperationKind::AsyncIssue);

        assert_eq!(
            log.register_operation_effect(
                &issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &plan,
                    action_ids: Some(&action_ids),
                },
            )
            .unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        let action_id = action_ids[0].get();
        let ResolvedTransitionSummary::Completion(hint) =
            log.completion_summary(action_id).unwrap()
        else {
            panic!("completion issue must create a completion summary");
        };
        assert_eq!(hint.resource().generation(), None);

        hub.arrive_expect_tx(barrier, 0, 1, 32).unwrap();
        let outcome = hub.apply_completion_detailed(action_ids[0]).unwrap();
        assert_eq!(
            log.register_completion_effect(CompletionEffect::PhysicalMbarrier(&outcome))
                .unwrap(),
            ResolvedTransitionRegistration::Refined
        );
        let ResolvedTransitionSummary::Completion(completion) =
            log.completion_summary(action_id).unwrap()
        else {
            panic!("completion outcome must retain a completion summary");
        };
        assert_eq!(completion.resource().generation(), Some(0));

        assert_eq!(
            log.register_operation_effect(
                &issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &plan,
                    action_ids: Some(&action_ids),
                },
            )
            .unwrap(),
            ResolvedTransitionRegistration::AlreadyRegistered
        );

        log.begin_replay();
        assert_eq!(log.operation_count(), 0);
        assert_eq!(log.completion_count(), 0);
        assert_eq!(
            log.register_operation_effect(
                &issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &plan,
                    action_ids: Some(&action_ids),
                },
            )
            .unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        let ResolvedTransitionSummary::Completion(replay_hint) =
            log.completion_summary(action_id).unwrap()
        else {
            panic!("replay issue must rebuild its completion summary");
        };
        assert_eq!(replay_hint.resource().generation(), None);
    }

    #[test]
    fn completion_issue_bindings_are_per_replay_and_strict_within_replay() {
        let log = ResolvedTransitionLog::default();
        let first_barrier = PhysicalBarrierId::new(44, 0, 0);
        let second_barrier = PhysicalBarrierId::new(44, 8, 0);
        let first_plan = PhysicalMbarrierCompletionIssuePlan::single(first_barrier, 8);
        let second_plan = PhysicalMbarrierCompletionIssuePlan::single(second_barrier, 8);
        let first_issue = operation(0, 0, OperationKind::AsyncIssue);
        let second_issue = operation(1, 0, OperationKind::AsyncIssue);

        let canonical_hub = PhysicalBarrierHub::new();
        canonical_hub.init(first_barrier, 1).unwrap();
        canonical_hub.init(second_barrier, 1).unwrap();
        let canonical_first_ids = first_plan.apply(&canonical_hub).unwrap();
        let canonical_second_ids = second_plan.apply(&canonical_hub).unwrap();
        log.register_operation_effect(
            &first_issue,
            OperationEffect::MbarrierCompletionIssue {
                plan: &first_plan,
                action_ids: Some(&canonical_first_ids),
            },
        )
        .unwrap();
        log.register_operation_effect(
            &second_issue,
            OperationEffect::MbarrierCompletionIssue {
                plan: &second_plan,
                action_ids: Some(&canonical_second_ids),
            },
        )
        .unwrap();

        log.begin_replay();
        let reversed_hub = PhysicalBarrierHub::new();
        reversed_hub.init(first_barrier, 1).unwrap();
        reversed_hub.init(second_barrier, 1).unwrap();
        let reversed_second_ids = second_plan.apply(&reversed_hub).unwrap();
        let reversed_first_ids = first_plan.apply(&reversed_hub).unwrap();
        log.register_operation_effect(
            &second_issue,
            OperationEffect::MbarrierCompletionIssue {
                plan: &second_plan,
                action_ids: Some(&reversed_second_ids),
            },
        )
        .unwrap();

        let error = log
            .register_operation_effect(
                &second_issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &second_plan,
                    action_ids: Some(&canonical_second_ids),
                },
            )
            .unwrap_err();

        assert!(matches!(
            error,
            ResolvedTransitionLogError::ConflictingOperation { operation, .. }
                if operation == *second_issue.id()
        ));
        log.register_operation_effect(
            &first_issue,
            OperationEffect::MbarrierCompletionIssue {
                plan: &first_plan,
                action_ids: Some(&reversed_first_ids),
            },
        )
        .unwrap();

        assert_eq!(log.operation_count(), 2);
        assert_eq!(log.completion_count(), 2);
        let ResolvedTransitionSummary::Completion(first_action) = log
            .completion_summary(reversed_second_ids[0].get())
            .unwrap()
        else {
            panic!("first reversed action must be indexed");
        };
        assert_eq!(
            first_action.resource().key(),
            ResolvedSyncResourceKey::PhysicalMbarrier(second_barrier)
        );
        let ResolvedTransitionSummary::Completion(second_action) =
            log.completion_summary(reversed_first_ids[0].get()).unwrap()
        else {
            panic!("second reversed action must be indexed");
        };
        assert_eq!(
            second_action.resource().key(),
            ResolvedSyncResourceKey::PhysicalMbarrier(first_barrier)
        );
    }

    #[test]
    fn zero_transaction_issue_has_no_schedulable_completion_summary() {
        let log = ResolvedTransitionLog::default();
        let barrier = PhysicalBarrierId::new(22, 16, 0);
        let plan = PhysicalMbarrierCompletionIssuePlan::single(barrier, 0);
        let hub = PhysicalBarrierHub::new();
        let action_ids = plan.apply(&hub).unwrap();
        let issue = operation(0, 0, OperationKind::AsyncIssue);

        log.register_operation_effect(
            &issue,
            OperationEffect::MbarrierCompletionIssue {
                plan: &plan,
                action_ids: Some(&action_ids),
            },
        )
        .unwrap();

        assert_eq!(log.operation_count(), 1);
        assert_eq!(log.completion_count(), 0);
        assert_eq!(log.completion_summary(action_ids[0].get()), None);
    }

    #[test]
    fn malformed_completion_issue_is_rejected_without_partial_registration() {
        let log = ResolvedTransitionLog::default();
        let first = PhysicalBarrierId::new(31, 0, 0);
        let second = PhysicalBarrierId::new(31, 8, 0);
        let hub = PhysicalBarrierHub::new();
        hub.init(first, 1).unwrap();
        hub.init(second, 1).unwrap();
        let one_target = PhysicalMbarrierCompletionIssuePlan::single(first, 8);
        let one_action_id = one_target.apply(&hub).unwrap();
        let two_targets =
            PhysicalMbarrierCompletionTargets::from_barrier_ids([first, second]).issue_plan(8);
        let issue = operation(0, 0, OperationKind::AsyncIssue);

        assert!(matches!(
            log.register_operation_effect(
                &issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &two_targets,
                    action_ids: Some(&one_action_id),
                },
            ),
            Err(
                ResolvedTransitionLogError::CompletionIssueActionCountMismatch {
                    completion_count: 2,
                    action_id_count: 1,
                    ..
                }
            )
        ));
        assert_eq!(log.operation_count(), 0);
        assert_eq!(log.completion_count(), 0);
    }

    #[test]
    fn duplicate_completion_issue_action_id_is_rejected() {
        let log = ResolvedTransitionLog::default();
        let first = PhysicalBarrierId::new(32, 0, 0);
        let second = PhysicalBarrierId::new(32, 8, 0);
        let hub = PhysicalBarrierHub::new();
        hub.init(first, 1).unwrap();
        let one_target = PhysicalMbarrierCompletionIssuePlan::single(first, 8);
        let action_ids = one_target.apply(&hub).unwrap();
        let duplicated_ids = Box::new([action_ids[0], action_ids[0]]);
        let two_targets =
            PhysicalMbarrierCompletionTargets::from_barrier_ids([first, second]).issue_plan(8);
        let issue = operation(0, 0, OperationKind::AsyncIssue);

        assert!(matches!(
            log.register_operation_effect(
                &issue,
                OperationEffect::MbarrierCompletionIssue {
                    plan: &two_targets,
                    action_ids: Some(&duplicated_ids[..]),
                },
            ),
            Err(ResolvedTransitionLogError::DuplicateCompletionIssueActionId {
                action_id,
                ..
            }) if action_id == action_ids[0].get()
        ));
        assert_eq!(log.operation_count(), 0);
        assert_eq!(log.completion_count(), 0);
    }

    #[test]
    fn operation_effect_registration_rejects_a_mismatched_batch_identity() {
        let log = ResolvedTransitionLog::default();
        let expected = operation(0, 0, OperationKind::Load);
        let actual = operation(1, 0, OperationKind::Load);
        let descriptor =
            PhysicalAccessDescriptor::new(PhysicalAccessKind::Read, PhysicalAccessSpace::Global, 4)
                .unwrap();
        let batch = PhysicalAccessBatch::resolve(actual.clone(), descriptor, |_| {
            Ok::<_, ()>(vec![span(1, 0, 4)])
        })
        .unwrap();

        assert!(matches!(
            log.register_operation_effect(&expected, OperationEffect::PhysicalAccess(&batch)),
            Err(ResolvedTransitionLogError::OperationEffectMismatch {
                expected: expected_id,
                actual: actual_id,
            }) if expected_id == *expected.id() && actual_id == *actual.id()
        ));
    }

    #[test]
    fn operation_effect_and_clock_share_one_consistent_registration() {
        let log = ResolvedTransitionLog::default();
        let (_physical, context, pointer) = shared_pointer(WarpValue::splat(2));
        let mask = WarpMask::from_lanes([0]).unwrap();
        let plan = plan_physical_mbarrier_init(&context, &pointer, mask, 1).unwrap();
        let operation = operation(0, 0, OperationKind::Barrier);
        let mut clock = SyncVectorClock::zero(1);
        clock.tick(0).unwrap();

        assert_eq!(
            log.register_operation_effect_and_clock(
                &operation,
                OperationEffect::MbarrierInit(&plan),
                Some(clock.clone()),
            )
            .unwrap(),
            (
                ResolvedTransitionRegistration::Inserted,
                Some(ResolvedTransitionRegistration::Inserted),
            )
        );
        assert_eq!(log.operation_clock(operation.id()), Some(clock.clone()));
        log.with_fixed_sync_snapshot(|snapshot| {
            let (_, _, initial_clock, current_clock) = snapshot
                .operations_with_clocks()
                .find(|(candidate, _, _, _)| *candidate == operation.id())
                .expect("registered operation is present in fixed-sync snapshot");
            assert_eq!(initial_clock, Some(&clock));
            assert_eq!(current_clock, Some(&clock));
        });

        assert_eq!(
            log.register_operation_effect_and_clock(
                &operation,
                OperationEffect::MbarrierInit(&plan),
                Some(clock.clone()),
            )
            .unwrap(),
            (
                ResolvedTransitionRegistration::AlreadyRegistered,
                Some(ResolvedTransitionRegistration::AlreadyRegistered),
            )
        );

        let mut refined = clock.clone();
        refined.tick(0).unwrap();
        assert_eq!(
            log.register_operation_clock(operation.id().clone(), refined.clone())
                .unwrap(),
            ResolvedTransitionRegistration::Refined
        );
        log.with_fixed_sync_snapshot(|snapshot| {
            let (_, _, initial_clock, current_clock) = snapshot
                .operations_with_clocks()
                .find(|(candidate, _, _, _)| *candidate == operation.id())
                .expect("registered operation is present in fixed-sync snapshot");
            assert_eq!(initial_clock, Some(&clock));
            assert_eq!(current_clock, Some(&refined));
        });
    }

    #[test]
    fn warp_sharded_operation_log_preserves_exact_sorted_evidence() {
        let log = ResolvedTransitionLog::with_warp_operation_shards(2);
        let warp1_late = operation(1, 9, OperationKind::Barrier);
        let warp0 = operation(0, 4, OperationKind::Barrier);
        let warp1_early = operation(1, 2, OperationKind::Barrier);
        let summary = ResolvedTransitionSummary::unknown("fixed evidence");

        for operation in [&warp1_late, &warp0, &warp1_early] {
            assert_eq!(
                log.register_operation(operation.id().clone(), summary.clone())
                    .unwrap(),
                ResolvedTransitionRegistration::Inserted
            );
        }
        let mut clock = SyncVectorClock::zero(2);
        clock.tick(1).unwrap();
        assert_eq!(
            log.register_operation_clock(warp1_early.id().clone(), clock.clone())
                .unwrap(),
            ResolvedTransitionRegistration::Inserted
        );
        assert_eq!(log.operation_count(), 3);
        assert_eq!(log.operation_clock(warp1_early.id()), Some(clock));
        assert_eq!(log.operation_summary(warp0.id()), Some(summary));
        log.with_fixed_sync_snapshot(|snapshot| {
            assert_eq!(
                snapshot
                    .operations_with_clocks()
                    .map(|(operation, _, _, _)| {
                        (operation.global_warp_id(), operation.per_warp_sequence())
                    })
                    .collect::<Vec<_>>(),
                [(0, 4), (1, 2), (1, 9)]
            );
        });

        log.begin_replay();
        assert_eq!(log.operation_count(), 0);
        log.with_fixed_sync_snapshot(|snapshot| {
            assert_eq!(snapshot.operations_with_clocks().count(), 0)
        });
    }
}
