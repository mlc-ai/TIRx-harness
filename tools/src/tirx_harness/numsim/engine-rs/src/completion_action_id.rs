//! One identity space for every scheduler-visible completion action.
//!
//! Three hubs mint completion-action IDs into what is really a single 64-bit
//! space, but each hub used to state its own share of the partition privately:
//! `hardware_barriers.rs` implicitly, by starting a plain monotonic counter at
//! zero and never bounding it; `setmaxnreg.rs` and `async_groups.rs` explicitly,
//! through `1 << 62` / `1 << 63` literals declared next to their own
//! byte-identical copy of the Cantor pairing helper. The partition was real,
//! declared in three places, and enforced in only two — nothing checked that the
//! physical counter stays below the tagged ranges.
//!
//! This module states the partition once and enforces it uniformly. Every value
//! minted through it is bit-identical to what the per-hub code produced, so the
//! change is invisible to anything that observes an ID.
//!
//! Allocation *strategy* stays per-namespace on purpose, because the two
//! strategies are load-bearing in opposite directions:
//!
//! - `PhysicalBarrier` allocates monotonically because
//!   `PhysicalBarrierState::transaction_completions` keys its scheduler queue on
//!   the ID and depends on key order being issue order.
//! - `Setmaxnreg` and `AsyncGroup` *derive* the ID from the resource tuple so a
//!   waiter can recompute the ID it needs without consulting a side table.
//!   `async_groups::tests::completion_action_ids_do_not_depend_on_cross_warp_commit_order`
//!   pins that property, which a monotonic counter would break.

/// Disjoint sub-range of the shared completion-action identity space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum CompletionActionNamespace {
    /// Physical mbarrier transaction and deferred-arrival completions.
    PhysicalBarrier,
    /// `setmaxnreg` register-pool completions.
    Setmaxnreg,
    /// Async-group (`cp.async` / `cp.async.bulk`) milestones.
    AsyncGroup,
}

impl CompletionActionNamespace {
    /// Inclusive low bound of the namespace.
    pub(crate) const fn base(self) -> u64 {
        match self {
            Self::PhysicalBarrier => 0,
            Self::Setmaxnreg => 1_u64 << 62,
            Self::AsyncGroup => 1_u64 << 63,
        }
    }

    /// Number of distinct identities the namespace can hold.
    pub(crate) const fn capacity(self) -> u128 {
        match self {
            Self::PhysicalBarrier | Self::Setmaxnreg => 1_u128 << 62,
            Self::AsyncGroup => 1_u128 << 63,
        }
    }

    /// Tag `offset` into this namespace, rejecting offsets that would collide
    /// with a neighbouring namespace.
    pub(crate) const fn tag(self, offset: u128) -> Option<u64> {
        if offset >= self.capacity() {
            return None;
        }
        Some(self.base() | offset as u64)
    }
}

/// Cantor pairing: the injective fold the derived namespaces use to collapse a
/// resource tuple into one offset.
pub(crate) fn pair(left: u128, right: u128) -> Option<u128> {
    let sum = left.checked_add(right)?;
    sum.checked_mul(sum.checked_add(1)?)?
        .checked_div(2)?
        .checked_add(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_partition_the_identity_space_without_gaps_or_overlap() {
        let mut ranges = [
            CompletionActionNamespace::PhysicalBarrier,
            CompletionActionNamespace::Setmaxnreg,
            CompletionActionNamespace::AsyncGroup,
        ]
        .map(|namespace| {
            (
                u128::from(namespace.base()),
                u128::from(namespace.base()) + namespace.capacity(),
            )
        });
        ranges.sort_unstable();
        assert_eq!(ranges[0].0, 0, "the partition must start at zero");
        assert_eq!(
            ranges[2].1,
            1_u128 << 64,
            "the partition must cover the whole 64-bit space"
        );
        for window in ranges.windows(2) {
            assert_eq!(
                window[0].1, window[1].0,
                "namespaces must abut exactly: {window:?}"
            );
        }
    }

    #[test]
    fn tagging_rejects_offsets_that_would_reach_the_next_namespace() {
        for namespace in [
            CompletionActionNamespace::PhysicalBarrier,
            CompletionActionNamespace::Setmaxnreg,
            CompletionActionNamespace::AsyncGroup,
        ] {
            let last = namespace.capacity() - 1;
            assert_eq!(
                namespace.tag(last),
                Some(namespace.base() + last as u64),
                "the final in-range offset must tag"
            );
            assert_eq!(
                namespace.tag(namespace.capacity()),
                None,
                "the first out-of-range offset must be rejected"
            );
        }
    }

    #[test]
    fn tagged_values_reproduce_the_previous_per_hub_literals() {
        // The three hubs used to compute exactly these expressions inline.
        assert_eq!(CompletionActionNamespace::PhysicalBarrier.tag(7), Some(7));
        assert_eq!(
            CompletionActionNamespace::Setmaxnreg.tag(7),
            Some((1_u64 << 62) | 7)
        );
        assert_eq!(
            CompletionActionNamespace::AsyncGroup.tag(7),
            Some((1_u64 << 63) | 7)
        );
    }

    #[test]
    fn pairing_is_injective_over_a_dense_square() {
        let mut seen = std::collections::BTreeSet::new();
        for left in 0..64_u128 {
            for right in 0..64_u128 {
                assert!(
                    seen.insert(pair(left, right).expect("small operands cannot overflow")),
                    "Cantor pairing collided at ({left}, {right})"
                );
            }
        }
    }

    #[test]
    fn pairing_reports_overflow_instead_of_wrapping() {
        assert_eq!(pair(u128::MAX, 1), None);
    }
}
