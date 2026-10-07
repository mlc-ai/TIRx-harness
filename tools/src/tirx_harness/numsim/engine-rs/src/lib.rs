#![forbid(unsafe_code)]

pub mod abi;
mod analysis_search;
#[doc(hidden)]
pub mod artifact_support;
mod async_groups;
mod async_token;
#[path = "native_analysis/checker_launch.rs"]
mod checker_launch;
mod cluster_barriers;
mod collectives;
mod completion;
mod completion_action_id;
mod context;
mod deferred_drop;
mod deferred_payload;
mod effect;
mod engine_mode;
mod executor;
mod hardware_barriers;
mod instruction_codec;
#[cfg(feature = "profile")]
mod instruction_profile;
mod kernel_engine;
mod mask;
mod memory;
mod mode_completions;
mod numpy_backend;
mod operation;
mod ordering;
mod physical_access;
mod profile;
#[cfg(any(test, feature = "racecheck"))]
#[path = "native_analysis/racecheck/race_check.rs"]
mod race_check;
#[cfg(all(feature = "python", any(test, feature = "racecheck")))]
#[path = "native_analysis/racecheck/race_check_python.rs"]
mod race_check_python;
#[cfg(any(test, feature = "racecheck"))]
#[path = "native_analysis/racecheck/race_shadow.rs"]
mod race_shadow;
mod resolved_transition;
mod runtime;
mod scalar;
mod scheduling;
mod setmaxnreg;
#[path = "native_analysis/synccheck/setmaxnreg_verifier.rs"]
mod setmaxnreg_verifier;
mod spaces;
mod strict_cluster_barrier;
#[path = "native_analysis/synccheck/strict_mbarrier.rs"]
mod strict_mbarrier;
mod strict_named_barrier;
#[path = "native_analysis/sync_causality.rs"]
mod sync_causality;
#[path = "native_analysis/synccheck/sync_check.rs"]
mod sync_check;
#[cfg(feature = "python")]
#[path = "native_analysis/synccheck/sync_check_python.rs"]
mod sync_check_python;
#[path = "native_analysis/synccheck/sync_fixed_unified.rs"]
mod sync_fixed_unified;
#[path = "native_analysis/synccheck/sync_fixed_verifier.rs"]
mod sync_fixed_verifier;
#[path = "native_analysis/synccheck/sync_partial_order.rs"]
mod sync_partial_order;
mod tcgen;
mod topology;
#[cfg(any(test, feature = "racecheck"))]
#[path = "native_analysis/racecheck/transactional_interval_map.rs"]
mod transactional_interval_map;
mod typed_async;
mod typed_copy;
mod warp_value;
mod worker_affinity;

pub(crate) use analysis_search::{
    CoverageBounds, CoverageStatus, CoverageSummary, CoverageUsage, ResourceAmount,
    ResourceLimitHit, ResourceLimitKind, ResourceLimits, ResourceUsage, SearchTermination,
};
pub(crate) use async_groups::{
    AsyncGroupCommitOutcome, AsyncGroupCommitPlan, AsyncGroupCommittedGroup,
    AsyncGroupCompletionAction, AsyncGroupCompletionActionId, AsyncGroupCompletionOutcome,
    AsyncGroupDomain, AsyncGroupHub, AsyncGroupId, AsyncGroupIssueBatchEffect,
    AsyncGroupIssueEffect, AsyncGroupMilestone, AsyncGroupWaitOutcome, AsyncGroupWaitPlan,
    AsyncGroupWaitedGroup,
};
pub(crate) use async_token::AsyncTokenId;
pub(crate) use checker_launch::CheckerLaunchContext;
pub(crate) use cluster_barriers::{
    ClusterBarrierArrivalOutcome, ClusterBarrierHub, ClusterBarrierId, ClusterBarrierWait,
};
pub(crate) use collectives::{
    CollectiveHub, CollectiveWait, CtaReduceContribution, CtaReduceHub, CtaReduceOp, RendezvousHub,
};
pub(crate) use completion::AwaitedOperation;
pub(crate) use completion::{
    BlockedOperation, ClusterBarrierOperation, CompletionProgress, CompletionRegistry,
    CompletionRegistryError, CompletionSource, DiagnosticLabel, OccurrenceKey, ParticipantContract,
    ParticipantSet, ParticipantState, ScopeInstance, SynchronizationError,
};
pub(crate) use context::{ControlProvenance, WarpContext};
pub(crate) use deferred_drop::defer_drop;
pub(crate) use deferred_payload::{
    DeferredPayloadCompletionAction, DeferredPayloadCompletionOutcome, DeferredPayloadHub,
    MbarrierCompletionAction, MbarrierCompletionOutcome,
};
pub(crate) use effect::{
    AnalysisGapDomain, AnalysisGapEffect, AnalysisGapKind, AsyncPayloadEffect,
    AsyncPayloadEffectError, CompletionActionEffect, CompletionEffect, MemoryFenceEffect,
    OperationEffect, OwnedOperationEffect, ProxyAsyncFenceEffect, ProxyAsyncFenceScope,
    TcgenFenceKind, WarpSyncEffect,
};
pub(crate) use engine_mode::{EngineMode, EngineModeImpl, NumSimMode};
pub(crate) use executor::{
    engine_error_kind, synchronization_error_kind,
    ClusterBarrierParticipantExitEvidence, EngineError, EngineErrorKind, ExecutionReport,
    ExecutionStats, Executor, OutOfBoundsError, WarpTask, OUT_OF_BOUNDS_ERROR_PREFIX,
};
pub(crate) use hardware_barriers::{
    MbarrierInitFenceTracker, NamedBarrierArrivalOutcome, NamedBarrierHub, NamedBarrierId,
    NamedBarrierWait, PhysicalBarrierHub, PhysicalBarrierId, PhysicalCompletionAction,
    PhysicalCompletionActionError, PhysicalCompletionActionId, PhysicalCompletionBatchCommitError,
    PhysicalCompletionKind, PhysicalCompletionOutcome, PhysicalMbarrierArrivalOutcome,
    MAX_MBARRIER_EXPECTED_ARRIVALS,
};
pub(crate) use kernel_engine::KernelEngine;
pub(crate) use kernel_engine::WarpEngine;
pub(crate) use mask::{ActiveLanes, MaskError, WarpMask, WARP_SIZE};
pub(crate) use memory::{
    AllocationId, BufferView, DeferredGlobalReduction, DeferredGlobalWrite, GlobalMemory,
    MemoryError,
};
pub(crate) use numpy_backend::{
    bf16_bits_to_f32, f32_to_bf16_bits,
    f32_to_float8_e4m3fn_bits, f32_to_float8_e8m0fnu_bits, f32_to_float8_e8m0fnu_bits_rounded,
    f32_to_fp16_bits, f32_to_narrow_float_bits_rn_satfinite, f32_to_narrow_float_bits_rs,
    f32_to_tf32, float4_e2m1fn_bits_to_f32, float8_e4m3fn_bits_to_f32, float8_e8m0fnu_bits_to_f32,
    fp16_bits_to_f32, narrow_float_bits_to_f32_checked, NarrowFloatFormat, NarrowFloatSpecials,
    FLOAT4_E2M1, FLOAT6_E2M3, FLOAT6_E3M2, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT8_UE5M3,
};
pub(crate) use operation::{DynamicOpId, LoopFrame, OperationContext, OperationKind, StaticOpId};
pub(crate) use ordering::OrderingHub;
pub(crate) use ordering::TcgenTransferKind;
pub(crate) use physical_access::{
    InvalidLaneProvenance, LanePhysicalAccess, LaneProvenance,
    MemoryAccessClass, MemoryAccessSemantics, MemoryOrder, MemoryProxy, MemoryScope,
    PhysicalAccessBatch, PhysicalAccessBatchError,
    PhysicalAccessDescriptor, PhysicalAccessDescriptorError, PhysicalAccessKind,
    PhysicalAccessSpace, PhysicalAccessWidth, PhysicalAllocationId, PhysicalByteSpan,
    PhysicalFootprint, PhysicalFootprintError,
};
pub(crate) use profile::{
    profile_count, profile_count_by, profile_reset, profile_snapshot, ProfileKind, ProfileTimer,
};
#[cfg(any(test, feature = "racecheck"))]
pub(crate) use race_check::{
    AliasStaleReadAdvisory, DeclaredWordBypassDiagnostic, GlobalActorRelation,
    GlobalScopeMismatchDiagnostic, RaceCheckAccessRecord, RaceCheckIncompleteReason,
    RaceCheckLaunchState, RaceCheckMode, RaceCheckResult, RaceCheckStatus,
    UndeclaredProtocolWordDiagnostic,
};
#[cfg(all(feature = "python", any(test, feature = "racecheck")))]
pub(crate) use race_check_python::build_native_race_check_phase_result;
#[cfg(any(test, feature = "racecheck"))]
pub(crate) use race_shadow::{
    BarrierClockPayload, PhysicalRaceFinding, PhysicalRaceKind, PhysicalRaceOrderingDomain,
    PhysicalRaceOrderingFailure, PhysicalRaceProxyDomain, PhysicalRaceWitness, RaceBatchValidation,
    RaceShadow, RaceShadowError, RaceVectorClock,
};
pub(crate) use resolved_transition::{
    ResolvedAnalysisGapEffect, ResolvedAnalysisResource, ResolvedAsyncPayloadEffect,
    ResolvedCompletionEffect, ResolvedMemoryEffect, ResolvedSyncResource, ResolvedSyncResourceKey,
    ResolvedSynchronizationEffect, ResolvedTransitionLog, ResolvedTransitionLogError,
    ResolvedTransitionRegistration, ResolvedTransitionSummary,
};
pub(crate) use scalar::{
    add_f32, add_f32_ftz, cuda_f32_add, cuda_f32_max, cuda_f32_min, cuda_f64_add,
    cuda_f64_max, cuda_f64_min, cuda_reduce_bf16_add, cuda_reduce_bf16_max, cuda_reduce_bf16_min,
    cuda_reduce_fp16_add, cuda_reduce_fp16_max, cuda_reduce_fp16_min, div_f32_rn, float2_x,
    float2_y, floor_div_i64, floor_mod_i64, fma_f32, fma_f32_ftz, fma_f32_rn,
    fp8x4_e4m3_from_float4, hmax2_bf16, hmin2_bf16, make_float2, mul_f32, mul_f32_ftz,
    pack_bf16x2, ptx_exp2_approx_ftz_f32, ptx_fns_b32, ptx_lg2_approx_ftz_f32,
    ptx_max_f32, ptx_rcp_approx_ftz_f32, sub_f32, sub_f32_ftz, unpack_bf16x2, F32RoundingMode,
    F32x4, RuntimeScalar, U64x2,
};
pub(crate) use scheduling::{reschedule, Reschedule};
pub(crate) use setmaxnreg::{
    setmaxnreg_default_register_count, SetmaxnregAction, SetmaxnregAvailabilityProvenance,
    SetmaxnregBudgetDisposition, SetmaxnregCompletionAction, SetmaxnregCompletionActionId,
    SetmaxnregCompletionOutcome, SetmaxnregError, SetmaxnregErrorKind, SetmaxnregHub,
    SetmaxnregOutcome, SetmaxnregPlan, SetmaxnregRegistration, SetmaxnregReleaseProvenance,
    SetmaxnregResource, SetmaxnregResumePlan, SETMAXNREG_COUNT_GRANULARITY,
    SETMAXNREG_CTA_REGISTER_POOL, SETMAXNREG_MAX_COUNT, SETMAXNREG_MIN_COUNT,
    SETMAXNREG_WARPS_PER_GROUP,
};
pub(crate) use spaces::{
    AddressSpaceError, CtaId, LocalMemory, PhysicalMemory, PhysicalOwner,
    PhysicalUninitializedReadReview, RegisterMemory, SharedAllocation, SharedMemory, SharedView,
    TmemAllocation, TmemMemory, TmemRegion, TmemView, WarpId, WarpPrivateAllocation,
    WarpPrivateMemory, WarpPrivateView, TMEM_CELL_BYTES,
};
pub(crate) use strict_cluster_barrier::{
    StrictClusterBarrierError, StrictClusterBarrierOutcome, StrictClusterBarrierProtocol,
};
pub(crate) use strict_mbarrier::{
    StrictMbarrierCompletionToken, StrictMbarrierEffect, StrictMbarrierError,
    StrictMbarrierLifecycle, StrictMbarrierProtocol, StrictMbarrierSnapshot,
    StrictMbarrierWaitOutcome, StrictMbarrierWaiter,
};
pub(crate) use strict_named_barrier::{
    StrictNamedBarrierError, StrictNamedBarrierOperation, StrictNamedBarrierOutcome,
    StrictNamedBarrierProtocol, StrictNamedBarrierSnapshot, StrictNamedBarrierWaiter,
};
pub(crate) use sync_causality::{
    retire_named_barrier_generations, MbarrierCausalState, MbarrierCausalUse,
    MbarrierCompletionCausalToken, MbarrierCompletionCausalTokenId,
    MbarrierGenerationCausalState, SyncCausalityError, SyncCausalityTracker, SyncClockPayload,
    SyncVectorClock,
};
pub(crate) use sync_check::{
    SyncCheckEffectKind, SyncCheckEffectOutcome, SyncCheckEffectRecord, SyncCheckFinding,
    SyncCheckIncompleteReason, SyncCheckLaunchState, SyncCheckMode, SyncCheckProtocolError,
    SyncCheckResult, SyncCheckStatus, SyncCheckWaitState,
};
#[cfg(feature = "python")]
pub(crate) use sync_check_python::build_native_sync_check_phase_result;
pub(crate) use sync_fixed_unified::{
    FixedSyncCommandId, FixedSyncCompletionId, FixedSyncDeadlock, FixedSyncProgram,
    FixedSyncProgramBuildError, FixedSyncProgramError, FixedSyncProtocolKind, FixedSyncState,
    FixedSyncTransition,
};
pub(crate) use sync_fixed_verifier::{
    verify_fixed_sync_programs, FixedSyncTransitionEvidence, FixedSyncVerificationError,
    FixedSyncVerificationIncomplete, FixedSyncVerificationResult, FixedSyncVerificationStats,
};
pub(crate) use sync_partial_order::{
    explore_sync_states, SyncStateFailure, SyncStateSearchLimits, SyncStateSearchOptions,
    SyncStateSearchResult, SyncStateSearchTermination, SyncTransitionSystem,
};
pub(crate) use tcgen::{
    TcgenAllocation, TcgenCtaSnapshot, TcgenLifecycleAction, TcgenLifecycleError,
    TcgenLifecycleErrorKind, TcgenLifecycleHub, TcgenLifecycleResult, TcgenLifecycleWait,
    TmemAccessError, TmemAccessErrorKind, TmemAccessMode, TMEM_COLUMN_CAPACITY,
};
pub(crate) use topology::{LaunchTopology, TopologyError};
pub(crate) use warp_value::WarpValue;

pub(crate) use runtime::operand::PhysicalAddress;

/// Increment for an incompatible generated-code, binding, or result boundary change.
#[cfg(feature = "python")]
pub(crate) const NUMSIM_ABI_VERSION: u32 = 38;
