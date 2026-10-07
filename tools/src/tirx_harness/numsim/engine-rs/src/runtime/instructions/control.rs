//! Engine implementation of v2 control-flow and scheduling operations.

use std::convert::Infallible;

use super::EngineError;
use crate::abi::v2::transport::{engine, Address, ExecCtx, LaneMask, Shared, SiteId, R};
use crate::{OperationKind, WarpValue};

use super::instruction::{async_instruction, instruction_variant, sync_instruction};

async_instruction!(setmaxnreg_spec, SetmaxnregVariant, setmaxnreg);
sync_instruction!(
    clc_query_cancel_spec,
    ClcQueryCancelVariant,
    clc_query_cancel
);

/// Static forms of control instructions that actually have more than one
/// legal spelling.
pub mod variant {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ClcQueryRelaxed;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ClcQueryAcquire;

    /// `setmaxnreg.inc.sync.aligned.u32 COUNT`. `COUNT` is a plain immediate
    /// the protocol hub re-checks at runtime, so it rides in `Args`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetmaxnregIncrease;
    /// `setmaxnreg.dec.sync.aligned.u32 COUNT`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetmaxnregDecrease;
}

#[inline(never)]
async fn execute_setmaxnreg(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    increase: bool,
    count: u32,
) -> Result<(), EngineError> {
    // The PTX count whitelist used to be a compile-time trait bound on the
    // marker. With the count demoted to a runtime operand it is checked here,
    // ahead of the protocol hub, so every engine mode rejects the same set --
    // the hub's own check (setmaxnreg.rs) only runs when operations are
    // observed.
    let checked = i64::from(count);
    if !(crate::SETMAXNREG_MIN_COUNT..=crate::SETMAXNREG_MAX_COUNT).contains(&checked)
        || checked % crate::SETMAXNREG_COUNT_GRANULARITY != 0
    {
        return Err(EngineError::message(format!(
            "setmaxnreg register count must be in {}..={} and a multiple of {}, got {count}",
            crate::SETMAXNREG_MIN_COUNT,
            crate::SETMAXNREG_MAX_COUNT,
            crate::SETMAXNREG_COUNT_GRANULARITY,
        )));
    }
    let operation = engine(warp).begin_optional_operation(
        context.into_inner(),
        site.get(),
        OperationKind::Collective,
        true,
    )?;
    engine(warp)
        .setmaxnreg_instruction(operation.as_ref(), increase, count)
        .await?;
    engine(warp).finish_optional_operation(&operation)?;
    Ok(())
}

macro_rules! setmaxnreg_variant {
    ($marker:ident, $increase:expr) => {
        instruction_variant! {
            [impl] setmaxnreg_spec, variant::$marker,
            u32 => ();
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                count: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                execute_setmaxnreg(warp, context, site, $increase, count).await
            }
        }
    };
}

setmaxnreg_variant!(SetmaxnregIncrease, true);
setmaxnreg_variant!(SetmaxnregDecrease, false);

/// Both griddepcontrol actions are ready at the serialized launch boundary.
///
/// Artifact phases do not overlap, and external prerequisites are discharged
/// by the host contract. A launch hint is not a completion token: repeated
/// waits and waits without a preceding hint are valid. Neither action creates
/// synchronization between threads of the current grid.
#[inline(never)]
pub fn griddepcontrol(
    _warp: &mut super::Engine,
    _context: ExecCtx,
    _site: SiteId,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        _context,
        _site,
        std::any::type_name_of_val(&griddepcontrol),
    );
    Ok(())
}

/// Each issuing thread claims one logical cluster and owns its response payload.
#[inline(never)]
pub async fn clc_try_cancel<const MULTICAST: bool>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    response: Address<Shared>,
    completion_barrier: Address<Shared>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name_of_val(&clc_try_cancel::<MULTICAST>),
    );
    let transactions = WarpValue::splat(16_u64);
    let ctas = context.into_inner().topology().ctas_per_cluster();
    let cta_masks = if MULTICAST {
        if ctas >= i64::BITS as usize {
            return Err(EngineError::message(format!(
                "CLC cluster size {ctas} cannot be represented by its completion mask"
            )));
        }
        Some(WarpValue::splat((1_i64 << ctas) - 1))
    } else {
        None
    };
    for lane in context.active_mask().into_inner() {
        let issuing = context.with_active_mask(LaneMask::from_bits(1_u32 << lane));
        let next_task = engine(warp).kernel().services().clc_tasks().try_cancel()?;
        let mut response_bytes = [0_u8; 16];
        response_bytes[..4].copy_from_slice(&next_task.to_le_bytes());
        let operation = engine(warp).begin_optional_operation(
            issuing.into_inner(),
            site.get(),
            OperationKind::AsyncIssue,
            true,
        )?;
        engine(warp).mbarrier_completion_issue(
            operation.as_ref(),
            completion_barrier.inner(),
            issuing.active_mask().into_inner(),
            None,
            1,
            cta_masks.as_ref(),
            &transactions,
            Some((response.inner(), response_bytes)),
        )?;
        engine(warp).finish_optional_operation(&operation)?;
    }
    Ok(())
}

macro_rules! clc_query_variant {
    ($marker:ty, $semantics:ty) => {
        instruction_variant! {
            [impl] clc_query_cancel_spec, $marker,
            Address<Shared> => R<u32>;
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                _site: SiteId,
                response: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                super::mem::ld::<
                    super::mem::variant::Ld<super::reg::variant::U32, Shared, $semantics>,
                >(warp, context, _site, response)
            }
        }
    };
}

clc_query_variant!(variant::ClcQueryRelaxed, super::mem::variant::Plain);
clc_query_variant!(
    variant::ClcQueryAcquire,
    super::mem::variant::Acquire<super::mem::variant::Cluster>
);

/// Execute one `nanosleep` using NumSim's documented deterministic timing
/// representative. The runtime duration remains an operand even though it
/// cannot affect logical values or readiness.
#[inline(never)]
pub async fn nanosleep(
    _warp: &mut super::Engine,
    _context: ExecCtx,
    _site: SiteId,
    _duration: R<u32>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(_context, _site, std::any::type_name_of_val(&nanosleep));
    Ok(())
}

/// Terminate the simulated kernel at one executed PTX `trap` instruction.
/// `Infallible` makes a successful return unrepresentable while preserving a
/// structured [`EngineError`] for host and checker reporting.
#[inline(never)]
pub fn trap(
    _warp: &mut super::Engine,
    _context: ExecCtx,
    site: SiteId,
) -> Result<Infallible, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(_context, site, std::any::type_name_of_val(&trap));
    Err(EngineError::message(format!(
        "PTX trap executed at source site {}",
        site.get()
    )))
}

mod sealed {
    pub trait BranchVariant {
        const ELECT_SYNC: bool;
    }
}

/// Static specialization for an ordinary divergent branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ordinary;

/// Static specialization for a branch predicate produced by `elect.sync`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElectSync;

impl sealed::BranchVariant for Ordinary {
    const ELECT_SYNC: bool = false;
}

impl sealed::BranchVariant for ElectSync {
    const ELECT_SYNC: bool = true;
}

/// Closed set of branch-context specializations.
#[allow(private_bounds)]
pub trait BranchVariant: sealed::BranchVariant {}

impl<T: sealed::BranchVariant> BranchVariant for T {}

/// Select a structured child context while preserving engine-private control
/// provenance.
#[inline(always)]
pub fn branch_context<V: BranchVariant>(
    parent: ExecCtx,
    selected: LaneMask,
) -> Result<ExecCtx, EngineError> {
    branch_context_entry(parent, selected, <V as sealed::BranchVariant>::ELECT_SYNC)
}

#[inline(never)]
fn branch_context_entry(
    parent: ExecCtx,
    selected: LaneMask,
    elect_sync: bool,
) -> Result<ExecCtx, EngineError> {
    let parent = parent.into_inner();
    let selected = selected.into_inner();
    let outside = selected - parent.active_mask();
    if !outside.is_empty() {
        return Err(EngineError::message(format!(
            "branch selected lanes {:#010x} outside parent mask {:#010x}",
            outside.bits(),
            parent.active_mask().bits(),
        )));
    }
    let child = if elect_sync {
        parent.with_elect_sync_active_mask(parent.active_mask(), selected)
    } else {
        parent.with_active_mask(selected)
    };
    Ok(ExecCtx::from_inner(child))
}

/// Enter one dynamic native `for` body occurrence.  The body ordinal is a
/// runtime control-flow fact, not part of the static specialization.
#[inline(never)]
pub fn for_enter(
    warp: &mut super::Engine,
    site: SiteId,
    iteration_ordinal: i64,
) -> Result<(), EngineError> {
    engine(warp)
        .native_for_enter(site.get(), iteration_ordinal)
        .map_err(Into::into)
}

/// Finish one dynamic native `for` body occurrence. `next_iteration_ordinal`
/// is the ordinal the frontend would execute next; `next_live` determines
/// whether that occurrence exists.
#[inline(never)]
pub async fn for_exit(
    warp: &mut super::Engine,
    site: SiteId,
    next_iteration_ordinal: i64,
    next_live: LaneMask,
) -> Result<(), EngineError> {
    engine(warp)
        .native_for_exit(site.get(), next_iteration_ordinal, next_live.into_inner())
        .await
        .map_err(Into::into)
}

/// Enter one dynamic native `while` body occurrence.
#[inline(never)]
pub async fn while_enter(
    warp: &mut super::Engine,
    site: SiteId,
    live: LaneMask,
) -> Result<(), EngineError> {
    engine(warp)
        .native_while_enter(site.get(), live.into_inner())
        .await
        .map_err(Into::into)
}

/// Finish one dynamic native `while` body occurrence.
#[inline(never)]
pub async fn while_exit(
    warp: &mut super::Engine,
    site: SiteId,
    next_live: LaneMask,
) -> Result<(), EngineError> {
    engine(warp)
        .native_while_exit(site.get(), next_live.into_inner())
        .await
        .map_err(Into::into)
}

#[cfg(all(test, not(feature = "analysis-core")))]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::runtime::{run_kernel_engine_launch, ExecutionPolicy, LaunchSelection};
    use crate::{LaunchTopology, NumSimMode, OperationKind, PhysicalMemory, StaticOpId, WarpMask};

    fn root() -> ExecCtx {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        ExecCtx::from_inner(topology.warp_contexts().next().unwrap())
    }

    #[test]
    fn branch_context_rejects_lanes_not_active_in_parent() {
        let parent = branch_context::<Ordinary>(root(), LaneMask::from_bits(0x0f)).unwrap();
        let error = branch_context::<Ordinary>(parent, LaneMask::from_bits(0x11)).unwrap_err();
        assert!(error.to_string().contains("outside parent mask"));
    }

    #[test]
    fn ordinary_and_elect_branches_keep_the_same_visible_mask() {
        let selected = LaneMask::from_bits(1);
        let ordinary = branch_context::<Ordinary>(root(), selected).unwrap();
        let elect = branch_context::<ElectSync>(root(), selected).unwrap();
        assert_eq!(ordinary.active_mask(), selected);
        assert_eq!(elect.active_mask(), selected);
        assert_eq!(
            ordinary.into_inner().control_provenance(),
            Default::default()
        );
        assert_eq!(
            elect
                .into_inner()
                .control_provenance()
                .elect_sync_entry_mask(),
            Some(WarpMask::FULL)
        );
    }

    #[test]
    fn native_for_owns_iteration_budget_and_dynamic_occurrence_identity() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let occurrences = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&occurrences);
        let error = run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::new(2, 8).unwrap(),
            move |mut warp| {
                let observed = Arc::clone(&observed);
                async move {
                    let site = SiteId::new(7);
                    let live = LaneMask::FULL;
                    for iteration in 0..2 {
                        for_enter(&mut warp, site, iteration)?;
                        let operation = warp.begin_current_operation(
                            warp.context(),
                            StaticOpId::new(41),
                            OperationKind::Control,
                        )?;
                        observed.lock().unwrap().push(operation.to_string());
                        for_exit(&mut warp, site, iteration + 1, live).await?;
                    }
                    for_enter(&mut warp, site, 2).map_err(Into::into)
                }
            },
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("exceeded configured native loop iteration budget 2"),
            "{error}"
        );
        assert_eq!(
            occurrences.lock().unwrap().as_slice(),
            [
                "kernel:0/warp:0/seq:0/source:op:41/loops:[op:7@0] kind:control active_mask:0xffffffff",
                "kernel:0/warp:0/seq:1/source:op:41/loops:[op:7@1] kind:control active_mask:0xffffffff",
            ]
        );
    }

    #[test]
    fn native_for_reschedules_at_the_engine_quantum() {
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        let stats = run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::new(8, 1).unwrap(),
            move |mut warp| async move {
                let site = SiteId::new(9);
                for_enter(&mut warp, site, 0)?;
                for_exit(&mut warp, site, 1, LaneMask::FULL).await?;
                for_enter(&mut warp, site, 1)?;
                for_exit(&mut warp, site, 2, LaneMask::EMPTY)
                    .await
                    .map_err(Into::into)
            },
        )
        .unwrap();

        assert_eq!(stats.poll_order, vec![0, 1, 0, 1]);
    }

    #[test]
    fn numeric_native_while_reschedules_before_condition_recheck() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let stats = run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::new(8, 1).unwrap(),
            move |mut warp| async move {
                let site = SiteId::new(11);
                while_enter(&mut warp, site, LaneMask::FULL).await?;
                while_exit(&mut warp, site, LaneMask::FULL).await?;
                while_enter(&mut warp, site, LaneMask::FULL).await?;
                while_exit(&mut warp, site, LaneMask::EMPTY)
                    .await
                    .map_err(Into::into)
            },
        )
        .unwrap();

        assert_eq!(stats.poll_order, vec![0, 0]);
    }

    #[test]
    fn while_recheck_that_turns_false_closes_engine_owned_loop_state() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::new(8, 1).unwrap(),
            move |mut warp| async move {
                let site = SiteId::new(13);
                while_enter(&mut warp, site, LaneMask::FULL).await?;
                // The body leaves candidates for another condition recheck.
                while_exit(&mut warp, site, LaneMask::from_bits(0xffff)).await?;

                // The condition can then turn false. The empty enter closes
                // the prior occurrence even though no next body runs.
                while_enter(&mut warp, site, LaneMask::EMPTY).await?;

                // A fresh occurrence at the same source site proves that no
                // stale state or loop frame survived the close.
                while_enter(&mut warp, site, LaneMask::FULL).await?;
                while_exit(&mut warp, site, LaneMask::EMPTY)
                    .await
                    .map_err(Into::into)
            },
        )
        .unwrap();
    }
}
