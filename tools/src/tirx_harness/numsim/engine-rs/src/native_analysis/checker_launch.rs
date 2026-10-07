//! Launch-level substrate shared by the peer native-analysis observers.
//!
//! Synccheck and racecheck are peers. A racecheck run reports both verdicts —
//! that is product behavior — but neither checker owns the other's state. What
//! they genuinely share is the per-launch analysis context:
//!
//! * the [`LaunchTopology`] the launch is partitioned by (synccheck shards its
//!   protocol state per cluster, racecheck shards its shadow per cluster),
//! * the [`ResolvedTransitionLog`] both register resolved effects into, and
//!   whether this launch records them at all,
//! * the global-write allocation set that selects compact memory analysis.
//!
//! Holding those behind one `Arc` both peers reference replaces racecheck
//! reading its own constructor inputs back out of the synccheck state it
//! embeds. The context is immutable for the life of a launch; the interior
//! mutability that matters (the transition log's own `Arc<RwLock<…>>`) lives
//! inside `ResolvedTransitionLog`.

use std::collections::BTreeSet;

use crate::{LaunchTopology, PhysicalAllocationId, ResolvedTransitionLog};

/// Immutable per-launch analysis context shared by the peer checkers.
pub(crate) struct CheckerLaunchContext {
    topology: Option<LaunchTopology>,
    transitions: ResolvedTransitionLog,
    global_write_allocations: Option<BTreeSet<PhysicalAllocationId>>,
    records_resolved_transitions: bool,
}

impl Default for CheckerLaunchContext {
    fn default() -> Self {
        Self {
            topology: None,
            transitions: ResolvedTransitionLog::default(),
            global_write_allocations: None,
            records_resolved_transitions: true,
        }
    }
}

impl CheckerLaunchContext {
    pub(crate) const fn new(
        topology: Option<LaunchTopology>,
        transitions: ResolvedTransitionLog,
        global_write_allocations: Option<BTreeSet<PhysicalAllocationId>>,
        records_resolved_transitions: bool,
    ) -> Self {
        Self {
            topology,
            transitions,
            global_write_allocations,
            records_resolved_transitions,
        }
    }

    pub(crate) const fn topology(&self) -> Option<LaunchTopology> {
        self.topology
    }

    /// Number of independent cluster shards, or zero when the launch was not
    /// given a topology. Both peers size their per-cluster state by this.
    pub(crate) fn cluster_shards(&self) -> usize {
        self.topology.map_or(0, |topology| topology.clusters())
    }

    pub(crate) const fn transition_log(&self) -> &ResolvedTransitionLog {
        &self.transitions
    }

    pub(crate) const fn records_resolved_transitions(&self) -> bool {
        self.records_resolved_transitions
    }

    pub(crate) const fn uses_compact_memory_analysis(&self) -> bool {
        self.global_write_allocations.is_some()
    }

    pub(crate) fn tracks_global_allocation(&self, allocation: PhysicalAllocationId) -> bool {
        self.global_write_allocations
            .as_ref()
            .is_none_or(|allocations| allocations.contains(&allocation))
    }
}
