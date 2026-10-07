use std::time::Duration;

/// Declared analysis coverage for one bounded search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverageBounds {
    pub max_warp_preemptions: u64,
    pub max_completion_schedule_deviations: u64,
}

impl CoverageBounds {
    pub const fn new(max_warp_preemptions: u64, max_completion_schedule_deviations: u64) -> Self {
        Self {
            max_warp_preemptions,
            max_completion_schedule_deviations,
        }
    }

    /// Return whether one concrete branch lies inside both declared bounds.
    pub const fn contains(self, usage: CoverageUsage) -> bool {
        usage.warp_preemptions <= self.max_warp_preemptions
            && usage.completion_schedule_deviations <= self.max_completion_schedule_deviations
    }
}

/// Coverage counters for one concrete schedule branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverageUsage {
    pub warp_preemptions: u64,
    pub completion_schedule_deviations: u64,
}

impl CoverageUsage {
    pub const fn within(self, bounds: CoverageBounds) -> bool {
        bounds.contains(self)
    }

    pub const fn componentwise_max(self, other: Self) -> Self {
        Self {
            warp_preemptions: if self.warp_preemptions > other.warp_preemptions {
                self.warp_preemptions
            } else {
                other.warp_preemptions
            },
            completion_schedule_deviations: if self.completion_schedule_deviations
                > other.completion_schedule_deviations
            {
                self.completion_schedule_deviations
            } else {
                other.completion_schedule_deviations
            },
        }
    }
}

/// Operational limits that cap analysis work without changing declared coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    pub max_schedules: u64,
    pub max_backtrack_nodes: u64,
    pub max_events_per_run: u64,
    pub max_total_events: u64,
    pub max_loop_steps: u64,
    pub max_wall_time: Duration,
    pub max_diagnostic_bytes: u64,
}

impl ResourceLimits {
    pub const fn unbounded() -> Self {
        Self {
            max_schedules: u64::MAX,
            max_backtrack_nodes: u64::MAX,
            max_events_per_run: u64::MAX,
            max_total_events: u64::MAX,
            max_loop_steps: u64::MAX,
            max_wall_time: Duration::MAX,
            max_diagnostic_bytes: u64::MAX,
        }
    }

    /// Describe the selected limit at the current usage. Controllers call this
    /// when that limit prevents the next unit of work.
    pub const fn hit(self, kind: ResourceLimitKind, usage: ResourceUsage) -> ResourceLimitHit {
        match kind {
            ResourceLimitKind::Schedules => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_schedules),
                usage: ResourceAmount::Count(usage.schedules),
            },
            ResourceLimitKind::BacktrackNodes => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_backtrack_nodes),
                usage: ResourceAmount::Count(usage.backtrack_nodes),
            },
            ResourceLimitKind::EventsPerRun => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_events_per_run),
                usage: ResourceAmount::Count(usage.events_in_current_run),
            },
            ResourceLimitKind::TotalEvents => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_total_events),
                usage: ResourceAmount::Count(usage.total_events),
            },
            ResourceLimitKind::LoopSteps => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_loop_steps),
                usage: ResourceAmount::Count(usage.loop_steps),
            },
            ResourceLimitKind::WallTime => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Time(self.max_wall_time),
                usage: ResourceAmount::Time(usage.wall_time),
            },
            ResourceLimitKind::DiagnosticBytes => ResourceLimitHit {
                kind,
                limit: ResourceAmount::Count(self.max_diagnostic_bytes),
                usage: ResourceAmount::Count(usage.diagnostic_bytes),
            },
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Work actually consumed by the analysis search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    pub schedules: u64,
    pub backtrack_nodes: u64,
    pub events_in_current_run: u64,
    pub total_events: u64,
    pub loop_steps: u64,
    pub wall_time: Duration,
    pub diagnostic_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceLimitKind {
    Schedules,
    BacktrackNodes,
    EventsPerRun,
    TotalEvents,
    LoopSteps,
    WallTime,
    DiagnosticBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceAmount {
    Count(u64),
    Time(Duration),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimitHit {
    pub kind: ResourceLimitKind,
    pub limit: ResourceAmount,
    pub usage: ResourceAmount,
}

/// Why schedule exploration stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchTermination {
    WorklistExhausted,
    Finding,
    ResourceLimit(ResourceLimitHit),
    Unsupported,
    Cancelled,
}

impl SearchTermination {
    pub const fn resource_limit_hit(self) -> Option<ResourceLimitHit> {
        match self {
            Self::ResourceLimit(hit) => Some(hit),
            _ => None,
        }
    }
}

/// Whether the explored search is sufficient to support a scoped clean verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageStatus {
    CompleteWithinBounds,
    Finding,
    Incomplete,
}

/// Compact report metadata shared by schedule-search analysis modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverageSummary {
    pub bounds: CoverageBounds,
    pub maximum_observed_usage: CoverageUsage,
    pub resource_limits: ResourceLimits,
    pub resource_usage: ResourceUsage,
    pub pending_work_items: u64,
    pub pending_backtracks: u64,
    pub termination: SearchTermination,
}

impl CoverageSummary {
    pub const fn status(self) -> CoverageStatus {
        if matches!(self.termination, SearchTermination::Finding) {
            CoverageStatus::Finding
        } else if self.eligible_for_clean() {
            CoverageStatus::CompleteWithinBounds
        } else {
            CoverageStatus::Incomplete
        }
    }

    /// A clean verdict is eligible only after the bounded worklist and all
    /// backtracks are exhausted without resource-limit termination.
    pub const fn eligible_for_clean(self) -> bool {
        matches!(self.termination, SearchTermination::WorklistExhausted)
            && self.pending_work_items == 0
            && self.pending_backtracks == 0
            && self.resource_limit_hit().is_none()
            && self.bounds.contains(self.maximum_observed_usage)
    }

    pub const fn resource_limit_hit(self) -> Option<ResourceLimitHit> {
        self.termination.resource_limit_hit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_bounds_accept_each_counter_at_the_boundary() {
        let bounds = CoverageBounds::new(1, 2);

        assert!(bounds.contains(CoverageUsage {
            warp_preemptions: 1,
            completion_schedule_deviations: 2,
        }));
        assert!(!bounds.contains(CoverageUsage {
            warp_preemptions: 2,
            completion_schedule_deviations: 2,
        }));
        assert!(!bounds.contains(CoverageUsage {
            warp_preemptions: 1,
            completion_schedule_deviations: 3,
        }));
    }

    #[test]
    fn clean_requires_exhausted_worklist_and_backtracks() {
        let mut summary = complete_summary();
        assert_eq!(summary.status(), CoverageStatus::CompleteWithinBounds);
        assert!(summary.eligible_for_clean());

        summary.pending_work_items = 1;
        assert_eq!(summary.status(), CoverageStatus::Incomplete);
        summary.pending_work_items = 0;
        summary.pending_backtracks = 1;
        assert_eq!(summary.status(), CoverageStatus::Incomplete);
    }

    #[test]
    fn finding_is_terminal_without_claiming_clean_coverage() {
        let mut summary = complete_summary();
        summary.termination = SearchTermination::Finding;

        assert_eq!(summary.status(), CoverageStatus::Finding);
        assert!(!summary.eligible_for_clean());
    }

    #[test]
    fn resource_limit_termination_carries_incomplete_metadata() {
        let mut summary = complete_summary();
        summary.resource_limits.max_schedules = 4;
        summary.resource_usage.schedules = 4;
        let hit = summary
            .resource_limits
            .hit(ResourceLimitKind::Schedules, summary.resource_usage);
        summary.termination = SearchTermination::ResourceLimit(hit);
        summary.pending_backtracks = 2;

        assert_eq!(summary.status(), CoverageStatus::Incomplete);
        assert!(!summary.eligible_for_clean());
        assert_eq!(summary.resource_limit_hit(), Some(hit));
        assert_eq!(hit.kind, ResourceLimitKind::Schedules);
        assert_eq!(hit.limit, ResourceAmount::Count(4));
        assert_eq!(hit.usage, ResourceAmount::Count(4));
    }

    #[test]
    fn reaching_usage_value_does_not_imply_limit_termination() {
        let mut summary = complete_summary();
        summary.resource_limits.max_schedules = 4;
        summary.resource_usage.schedules = 4;

        assert!(summary.eligible_for_clean());
        assert_eq!(summary.resource_limit_hit(), None);
    }

    fn complete_summary() -> CoverageSummary {
        CoverageSummary {
            bounds: CoverageBounds::new(1, 2),
            maximum_observed_usage: CoverageUsage {
                warp_preemptions: 1,
                completion_schedule_deviations: 2,
            },
            resource_limits: ResourceLimits::unbounded(),
            resource_usage: ResourceUsage {
                schedules: 3,
                ..ResourceUsage::default()
            },
            pending_work_items: 0,
            pending_backtracks: 0,
            termination: SearchTermination::WorklistExhausted,
        }
    }
}
