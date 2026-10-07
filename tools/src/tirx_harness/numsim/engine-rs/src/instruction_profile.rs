//! Executed ABI specializations for a single profiled executor launch.
//!
//! Names are diagnostic Rust type names, not stable identities or cache keys.
//! A pair means an instruction was entered with at least one active lane; it
//! does not assert that an asynchronous operation completed successfully.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;

use crate::abi::v2::{ExecCtx, SiteId};

#[derive(Clone, Default)]
pub(crate) struct InstructionProfile(Arc<ThreadLocal<Mutex<BTreeSet<(u64, &'static str)>>>>);

thread_local! {
    static ACTIVE: RefCell<Option<InstructionProfile>> = const { RefCell::new(None) };
}

pub(crate) struct InstructionScope(Option<InstructionProfile>);

impl Drop for InstructionScope {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = self.0.take());
    }
}

impl InstructionProfile {
    pub(crate) fn current() -> Option<Self> {
        ACTIVE.with(|active| active.borrow().clone())
    }

    pub(crate) fn enter(profile: Option<Self>) -> InstructionScope {
        InstructionScope(ACTIVE.with(|active| active.replace(profile)))
    }

    pub(crate) fn snapshot(&self) -> Vec<(u64, &'static str)> {
        let mut pairs = BTreeSet::new();
        for worker in self.0.iter() {
            pairs.extend(worker.lock().unwrap().iter().copied());
        }
        pairs.into_iter().collect()
    }
}

pub(crate) fn record(context: ExecCtx, site: SiteId, variant: &'static str) {
    if context.active_mask().is_empty() {
        return;
    }
    ACTIVE.with(|active| {
        if let Some(profile) = active.borrow().as_ref() {
            profile
                .0
                .get_or_default()
                .lock()
                .unwrap()
                .insert((site.get(), variant));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::v2::{reg, LaneMask, R};
    use crate::{Executor, LaunchTopology, WarpTask};

    fn run(site: u64) -> Vec<(u64, &'static str)> {
        let topology = LaunchTopology::new(2, 1, 1).unwrap();
        let tasks = topology.warp_contexts().enumerate().map(|(warp, context)| {
            WarpTask::new(warp, async move {
                let context = ExecCtx::from_inner(context);
                for _ in 0..3 {
                    let value = reg::add::<reg::variant::I32>(
                        context,
                        SiteId::new(site),
                        (R::splat(2), R::splat(3)),
                    )?;
                    assert_eq!(value[0], 5);
                }
                reg::add::<reg::variant::U32>(
                    context,
                    SiteId::new(site),
                    (R::splat(2), R::splat(3)),
                )?;
                reg::add::<reg::variant::I32>(
                    context.with_active_mask(LaneMask::from_bits(0)),
                    SiteId::new(site + 100),
                    (R::splat(2), R::splat(3)),
                )?;
                Ok(())
            })
        });
        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();
        assert_eq!(stats.completed_task_count, 2);
        stats.executed_instruction_variants
    }

    fn expected(site: u64) -> Vec<(u64, &'static str)> {
        let mut result = vec![
            (site, std::any::type_name::<reg::variant::I32>()),
            (site, std::any::type_name::<reg::variant::U32>()),
        ];
        result.sort_unstable();
        result
    }

    #[test]
    fn worker_calls_record_actual_types_once_and_ignore_empty_masks() {
        assert_eq!(run(7), expected(7));
    }

    #[test]
    fn concurrent_and_repeated_launches_do_not_share_records() {
        let first = std::thread::spawn(|| run(11));
        let second = std::thread::spawn(|| run(29));
        assert_eq!(first.join().unwrap(), expected(11));
        assert_eq!(second.join().unwrap(), expected(29));
        assert_eq!(run(43), expected(43));
        assert!(InstructionProfile::current().is_none());
    }

    #[test]
    fn more_clusters_than_workers_preserve_all_sites() {
        let topology = LaunchTopology::new(7, 1, 1).unwrap();
        let tasks = topology.warp_contexts().enumerate().map(|(warp, context)| {
            WarpTask::new(warp, async move {
                reg::add::<reg::variant::U32>(
                    ExecCtx::from_inner(context),
                    SiteId::new(200 + warp as u64),
                    (R::splat(2), R::splat(3)),
                )?;
                Ok(())
            })
        });
        let stats = Executor::with_max_workers(2)
            .run_with_topology(tasks, topology)
            .unwrap();
        assert_eq!(stats.completed_task_count, 7);
        assert_eq!(stats.worker_count, 2);
        assert_eq!(
            stats.executed_instruction_variants,
            (200..207)
                .map(|site| (site, std::any::type_name::<reg::variant::U32>()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_entered_instruction_is_retained_when_execution_fails() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = ExecCtx::from_inner(topology.warp_contexts().next().unwrap());
        let task = WarpTask::new(0, async move {
            reg::div::<reg::variant::I32>(context, SiteId::new(17), (R::splat(1), R::splat(0)))?;
            Ok(())
        });
        let report = Executor::default().run_report([task]);
        assert!(!report.is_success());
        assert_eq!(
            report.stats.executed_instruction_variants,
            vec![(17, std::any::type_name::<reg::variant::I32>())]
        );
        assert!(InstructionProfile::current().is_none());
    }
}
