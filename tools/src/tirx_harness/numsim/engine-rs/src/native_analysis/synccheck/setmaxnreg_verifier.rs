//! Pure register-pool verifier state for synccheck's explicit-state layer.
//!
//! Relocated verbatim from `setmaxnreg.rs`: this core is synccheck-only by
//! its own contract (no waiters, no completion allocator, no runtime side
//! effects) and `sync_fixed_unified.rs` is its sole consumer.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::setmaxnreg::{
    SetmaxnregAction, SetmaxnregResource, SETMAXNREG_COUNT_GRANULARITY,
    SETMAXNREG_CTA_REGISTER_POOL, SETMAXNREG_MAX_COUNT, SETMAXNREG_MIN_COUNT,
};

/// Pure, cloneable register-pool state for explicit-state synchronization verification.
///
/// This core deliberately has no waiters, completion allocator, provenance, or
/// runtime side effects. A blocked increase retains its warpgroup's current
/// count and becomes runnable only through an explicit grant transition after
/// a later decrease releases enough capacity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SetmaxnregVerifierCore {
    kernel_index: usize,
    global_cta_id: usize,
    capacity: i64,
    available_count: i64,
    current_counts: Box<[i64]>,
    pending_increases: BTreeMap<SetmaxnregResource, SetmaxnregVerifierPendingIncrease>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SetmaxnregVerifierPendingIncrease {
    target_count: i64,
    required_count: i64,
}

impl SetmaxnregVerifierPendingIncrease {
    pub(crate) const fn target_count(self) -> i64 {
        self.target_count
    }

    pub(crate) const fn required_count(self) -> i64 {
        self.required_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SetmaxnregVerifierRequestDisposition {
    DecreaseApplied { released_count: i64 },
    IncreaseImmediate { required_count: i64 },
    IncreasePending { required_count: i64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SetmaxnregVerifierRequestOutcome {
    resource: SetmaxnregResource,
    current_count_before: i64,
    available_count_before: i64,
    disposition: SetmaxnregVerifierRequestDisposition,
}

impl SetmaxnregVerifierRequestOutcome {
    pub(crate) const fn resource(self) -> SetmaxnregResource {
        self.resource
    }

    pub(crate) const fn current_count_before(self) -> i64 {
        self.current_count_before
    }

    pub(crate) const fn available_count_before(self) -> i64 {
        self.available_count_before
    }

    pub(crate) const fn disposition(self) -> SetmaxnregVerifierRequestDisposition {
        self.disposition
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SetmaxnregVerifierGrantOutcome {
    resource: SetmaxnregResource,
    target_count: i64,
    required_count: i64,
    available_count_before: i64,
}

impl SetmaxnregVerifierGrantOutcome {
    pub(crate) const fn resource(self) -> SetmaxnregResource {
        self.resource
    }

    pub(crate) const fn target_count(self) -> i64 {
        self.target_count
    }

    pub(crate) const fn required_count(self) -> i64 {
        self.required_count
    }

    pub(crate) const fn available_count_before(self) -> i64 {
        self.available_count_before
    }
}

/// Progress available from the verifier core without executing another request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SetmaxnregVerifierGrantStatus {
    Quiescent,
    Grantable,
    Stalled,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SetmaxnregVerifierError {
    InvalidPoolInvariant {
        available_count: i64,
        current_total: i64,
    },
    InvalidInitialCount {
        warpgroup_id: usize,
        count: i64,
    },
    ContextMismatch {
        resource: SetmaxnregResource,
    },
    InvalidWarpgroup {
        resource: SetmaxnregResource,
        warpgroup_count: usize,
    },
    InvalidCount {
        resource: SetmaxnregResource,
        count: i64,
    },
    InvalidDirection {
        resource: SetmaxnregResource,
        action: SetmaxnregAction,
        current_count: i64,
        target_count: i64,
    },
    WarpgroupPending {
        resource: SetmaxnregResource,
    },
    ArithmeticOverflow,
    UnknownGrant {
        resource: SetmaxnregResource,
    },
    GrantNotEnabled {
        resource: SetmaxnregResource,
        required_count: i64,
        available_count: i64,
    },
}

impl fmt::Display for SetmaxnregVerifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPoolInvariant {
                available_count,
                current_total,
            } => write!(
                formatter,
                "setmaxnreg verifier pool has invalid available={available_count} and current total={current_total}; available must be nonnegative and combined capacity must not exceed the {SETMAXNREG_CTA_REGISTER_POOL}-register CTA limit",
            ),
            Self::InvalidInitialCount {
                warpgroup_id,
                count,
            } => write!(
                formatter,
                "setmaxnreg verifier warpgroup {warpgroup_id} has invalid initial count {count}",
            ),
            Self::ContextMismatch { resource } => {
                write!(formatter, "setmaxnreg verifier resource {resource} belongs to another CTA")
            }
            Self::InvalidWarpgroup {
                resource,
                warpgroup_count,
            } => write!(
                formatter,
                "setmaxnreg verifier resource {resource} names a warpgroup outside 0..{warpgroup_count}",
            ),
            Self::InvalidCount { resource, count } => write!(
                formatter,
                "setmaxnreg verifier request {resource} has invalid target count {count}",
            ),
            Self::InvalidDirection {
                resource,
                action,
                current_count,
                target_count,
            } => write!(
                formatter,
                "setmaxnreg verifier request {resource} cannot apply {} from {current_count} to {target_count}",
                action.label(),
            ),
            Self::WarpgroupPending { resource } => write!(
                formatter,
                "setmaxnreg verifier request {resource} belongs to a warpgroup with a pending increase",
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("setmaxnreg verifier arithmetic overflow")
            }
            Self::UnknownGrant { resource } => {
                write!(formatter, "setmaxnreg verifier grant {resource} is not pending")
            }
            Self::GrantNotEnabled {
                resource,
                required_count,
                available_count,
            } => write!(
                formatter,
                "setmaxnreg verifier grant {resource} requires {required_count} registers but only {available_count} are available",
            ),
        }
    }
}

impl Error for SetmaxnregVerifierError {}

impl SetmaxnregVerifierCore {
    pub(crate) fn from_parts(
        kernel_index: usize,
        global_cta_id: usize,
        available_count: i64,
        current_counts: impl IntoIterator<Item = i64>,
    ) -> Result<Self, SetmaxnregVerifierError> {
        let current_counts = current_counts.into_iter().collect::<Vec<_>>();
        for (warpgroup_id, &count) in current_counts.iter().enumerate() {
            if count <= 0 || count % SETMAXNREG_COUNT_GRANULARITY != 0 {
                return Err(SetmaxnregVerifierError::InvalidInitialCount {
                    warpgroup_id,
                    count,
                });
            }
        }
        let current_total = current_counts.iter().try_fold(0_i64, |total, count| {
            total
                .checked_add(*count)
                .ok_or(SetmaxnregVerifierError::ArithmeticOverflow)
        })?;
        let capacity = available_count
            .checked_add(current_total)
            .ok_or(SetmaxnregVerifierError::ArithmeticOverflow)?;
        if available_count < 0 || capacity > SETMAXNREG_CTA_REGISTER_POOL {
            return Err(SetmaxnregVerifierError::InvalidPoolInvariant {
                available_count,
                current_total,
            });
        }
        Ok(Self {
            kernel_index,
            global_cta_id,
            capacity,
            available_count,
            current_counts: current_counts.into_boxed_slice(),
            pending_increases: BTreeMap::new(),
        })
    }

    pub(crate) const fn available_count(&self) -> i64 {
        self.available_count
    }

    pub(crate) fn current_counts(&self) -> &[i64] {
        &self.current_counts
    }

    pub(crate) fn current_count(&self, warpgroup_id: usize) -> Option<i64> {
        self.current_counts.get(warpgroup_id).copied()
    }

    pub(crate) fn pending_increases(
        &self,
    ) -> &BTreeMap<SetmaxnregResource, SetmaxnregVerifierPendingIncrease> {
        &self.pending_increases
    }

    pub(crate) fn warpgroup_has_pending_increase(&self, warpgroup_id: usize) -> bool {
        self.pending_increases
            .keys()
            .any(|resource| resource.warpgroup_id() == warpgroup_id)
    }

    pub(crate) fn enabled_grants(&self) -> Vec<SetmaxnregResource> {
        self.pending_increases
            .iter()
            .filter_map(|(&resource, pending)| {
                (pending.required_count <= self.available_count).then_some(resource)
            })
            .collect()
    }

    pub(crate) fn grant_status(&self) -> SetmaxnregVerifierGrantStatus {
        if self.pending_increases.is_empty() {
            SetmaxnregVerifierGrantStatus::Quiescent
        } else if self
            .pending_increases
            .values()
            .any(|pending| pending.required_count <= self.available_count)
        {
            SetmaxnregVerifierGrantStatus::Grantable
        } else {
            SetmaxnregVerifierGrantStatus::Stalled
        }
    }

    pub(crate) fn is_quiescent(&self) -> bool {
        self.grant_status() == SetmaxnregVerifierGrantStatus::Quiescent
    }

    /// Whether the explicit grant relation has no enabled transition.
    ///
    /// A stalled state may still be changed by a future decrease request from
    /// the command graph; global deadlock is decided by the graph verifier.
    pub(crate) fn is_grant_terminal(&self) -> bool {
        self.grant_status() != SetmaxnregVerifierGrantStatus::Grantable
    }

    pub(crate) fn apply_request(
        &mut self,
        resource: SetmaxnregResource,
        action: SetmaxnregAction,
        target_count: i64,
    ) -> Result<SetmaxnregVerifierRequestOutcome, SetmaxnregVerifierError> {
        let warpgroup_id = self.validate_request(resource, action, target_count)?;
        let current_count_before = self.current_counts[warpgroup_id];
        let available_count_before = self.available_count;
        let disposition = match action {
            SetmaxnregAction::Decrease => {
                let released_count = current_count_before
                    .checked_sub(target_count)
                    .ok_or(SetmaxnregVerifierError::ArithmeticOverflow)?;
                self.current_counts[warpgroup_id] = target_count;
                self.available_count = self
                    .available_count
                    .checked_add(released_count)
                    .ok_or(SetmaxnregVerifierError::ArithmeticOverflow)?;
                SetmaxnregVerifierRequestDisposition::DecreaseApplied { released_count }
            }
            SetmaxnregAction::Increase => {
                let required_count = target_count
                    .checked_sub(current_count_before)
                    .ok_or(SetmaxnregVerifierError::ArithmeticOverflow)?;
                if required_count <= self.available_count {
                    self.available_count -= required_count;
                    self.current_counts[warpgroup_id] = target_count;
                    SetmaxnregVerifierRequestDisposition::IncreaseImmediate { required_count }
                } else {
                    self.pending_increases.insert(
                        resource,
                        SetmaxnregVerifierPendingIncrease {
                            target_count,
                            required_count,
                        },
                    );
                    SetmaxnregVerifierRequestDisposition::IncreasePending { required_count }
                }
            }
        };
        debug_assert_eq!(
            self.available_count + self.current_counts.iter().sum::<i64>(),
            self.capacity,
        );
        Ok(SetmaxnregVerifierRequestOutcome {
            resource,
            current_count_before,
            available_count_before,
            disposition,
        })
    }

    pub(crate) fn apply_grant(
        &mut self,
        resource: SetmaxnregResource,
    ) -> Result<SetmaxnregVerifierGrantOutcome, SetmaxnregVerifierError> {
        self.validate_resource(resource)?;
        let pending = self
            .pending_increases
            .get(&resource)
            .copied()
            .ok_or(SetmaxnregVerifierError::UnknownGrant { resource })?;
        if pending.required_count > self.available_count {
            return Err(SetmaxnregVerifierError::GrantNotEnabled {
                resource,
                required_count: pending.required_count,
                available_count: self.available_count,
            });
        }
        let available_count_before = self.available_count;
        self.available_count -= pending.required_count;
        self.current_counts[resource.warpgroup_id()] = pending.target_count;
        self.pending_increases.remove(&resource);
        debug_assert_eq!(
            self.available_count + self.current_counts.iter().sum::<i64>(),
            self.capacity,
        );
        Ok(SetmaxnregVerifierGrantOutcome {
            resource,
            target_count: pending.target_count,
            required_count: pending.required_count,
            available_count_before,
        })
    }

    fn validate_request(
        &self,
        resource: SetmaxnregResource,
        action: SetmaxnregAction,
        target_count: i64,
    ) -> Result<usize, SetmaxnregVerifierError> {
        let warpgroup_id = self.validate_resource(resource)?;
        if !(SETMAXNREG_MIN_COUNT..=SETMAXNREG_MAX_COUNT).contains(&target_count)
            || target_count % SETMAXNREG_COUNT_GRANULARITY != 0
        {
            return Err(SetmaxnregVerifierError::InvalidCount {
                resource,
                count: target_count,
            });
        }
        if self.warpgroup_has_pending_increase(warpgroup_id) {
            return Err(SetmaxnregVerifierError::WarpgroupPending { resource });
        }
        let current_count = self.current_counts[warpgroup_id];
        let valid_direction = match action {
            SetmaxnregAction::Decrease => target_count <= current_count,
            SetmaxnregAction::Increase => target_count >= current_count,
        };
        if !valid_direction {
            return Err(SetmaxnregVerifierError::InvalidDirection {
                resource,
                action,
                current_count,
                target_count,
            });
        }
        Ok(warpgroup_id)
    }

    fn validate_resource(
        &self,
        resource: SetmaxnregResource,
    ) -> Result<usize, SetmaxnregVerifierError> {
        if resource.kernel_index() != self.kernel_index
            || resource.global_cta_id() != self.global_cta_id
        {
            return Err(SetmaxnregVerifierError::ContextMismatch { resource });
        }
        let warpgroup_id = resource.warpgroup_id();
        if warpgroup_id >= self.current_counts.len() {
            return Err(SetmaxnregVerifierError::InvalidWarpgroup {
                resource,
                warpgroup_count: self.current_counts.len(),
            });
        }
        Ok(warpgroup_id)
    }
}

#[cfg(test)]
#[path = "setmaxnreg_verifier_tests.rs"]
mod tests;
