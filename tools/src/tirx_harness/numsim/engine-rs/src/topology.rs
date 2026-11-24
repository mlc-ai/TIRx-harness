use std::error::Error;
use std::fmt;
use std::ops::Range;

use crate::{WarpContext, WarpMask, WARP_SIZE};

/// Current engine limit, matching CUDA's 1024-thread CTA limit for 32-lane warps.
pub const MAX_WARPS_PER_CTA: usize = 32;

/// CTA member masks in the engine are represented by one `u64`.
pub const MAX_CTAS_PER_CLUSTER: usize = 64;

/// Linearized cluster/CTA/warp launch dimensions.
///
/// A CTA has `threads_per_cta` threads in `warps_per_cta = ceil(threads / 32)`
/// warps formed from consecutive linear thread ids. When the thread count is
/// not a multiple of 32 the last warp of every CTA is a partial warp whose
/// upper lanes do not exist: they never execute and never arrive anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchTopology {
    clusters: usize,
    ctas_per_cluster: usize,
    warps_per_cta: usize,
    threads_per_cta: usize,
    warp_count: usize,
}

impl LaunchTopology {
    /// A launch of whole-warp CTAs: `warps_per_cta * 32` threads per CTA.
    pub fn new(
        clusters: usize,
        ctas_per_cluster: usize,
        warps_per_cta: usize,
    ) -> Result<Self, TopologyError> {
        let threads_per_cta = warps_per_cta
            .checked_mul(WARP_SIZE)
            .ok_or(TopologyError::WarpCountOverflow)?;
        Self::with_dimensions(clusters, ctas_per_cluster, warps_per_cta, threads_per_cta)
    }

    /// A launch whose CTAs have `threads_per_cta` threads, which need not be
    /// a multiple of 32: the last warp of each CTA is then partial.
    pub fn with_threads_per_cta(
        clusters: usize,
        ctas_per_cluster: usize,
        threads_per_cta: usize,
    ) -> Result<Self, TopologyError> {
        if threads_per_cta == 0 {
            return Err(TopologyError::ZeroDimension {
                name: "threads_per_cta",
            });
        }
        Self::with_dimensions(
            clusters,
            ctas_per_cluster,
            threads_per_cta.div_ceil(WARP_SIZE),
            threads_per_cta,
        )
    }

    fn with_dimensions(
        clusters: usize,
        ctas_per_cluster: usize,
        warps_per_cta: usize,
        threads_per_cta: usize,
    ) -> Result<Self, TopologyError> {
        for (name, value) in [
            ("clusters", clusters),
            ("ctas_per_cluster", ctas_per_cluster),
            ("warps_per_cta", warps_per_cta),
        ] {
            if value == 0 {
                return Err(TopologyError::ZeroDimension { name });
            }
        }
        if warps_per_cta > MAX_WARPS_PER_CTA {
            return Err(TopologyError::DimensionTooLarge {
                name: "warps_per_cta",
                value: warps_per_cta,
                maximum: MAX_WARPS_PER_CTA,
            });
        }
        if ctas_per_cluster > MAX_CTAS_PER_CLUSTER {
            return Err(TopologyError::DimensionTooLarge {
                name: "ctas_per_cluster",
                value: ctas_per_cluster,
                maximum: MAX_CTAS_PER_CLUSTER,
            });
        }
        let warp_count = clusters
            .checked_mul(ctas_per_cluster)
            .and_then(|count| count.checked_mul(warps_per_cta))
            .ok_or(TopologyError::WarpCountOverflow)?;
        Ok(Self {
            clusters,
            ctas_per_cluster,
            warps_per_cta,
            threads_per_cta,
            warp_count,
        })
    }

    pub const fn threads_per_cta(self) -> usize {
        self.threads_per_cta
    }

    /// Whether the last warp of each CTA has lanes that do not exist.
    pub const fn has_partial_warp(self) -> bool {
        self.threads_per_cta % WARP_SIZE != 0
    }

    /// The lanes that exist in warp `warp_id_in_cta` of a CTA: every lane
    /// except in a partial last warp.
    pub const fn lane_mask_for_warp_in_cta(self, warp_id_in_cta: usize) -> WarpMask {
        let remaining = self
            .threads_per_cta
            .saturating_sub(warp_id_in_cta * WARP_SIZE);
        if remaining >= WARP_SIZE {
            WarpMask::FULL
        } else {
            WarpMask::from_bits((1_u32 << remaining) - 1)
        }
    }

    /// The lanes that exist in the warp with global id `global_warp_id`.
    pub const fn lane_mask_for_warp(self, global_warp_id: usize) -> WarpMask {
        self.lane_mask_for_warp_in_cta(global_warp_id % self.warps_per_cta)
    }

    pub const fn clusters(self) -> usize {
        self.clusters
    }

    pub const fn ctas_per_cluster(self) -> usize {
        self.ctas_per_cluster
    }

    pub const fn warps_per_cta(self) -> usize {
        self.warps_per_cta
    }

    pub const fn cta_count(self) -> usize {
        self.clusters * self.ctas_per_cluster
    }

    pub const fn warp_count(self) -> usize {
        self.warp_count
    }

    /// Number of warps in one indivisible cluster scheduling domain.
    pub const fn warps_per_cluster(self) -> usize {
        self.ctas_per_cluster * self.warps_per_cta
    }

    /// Return the cluster scheduling domain containing a global warp ID.
    pub fn cluster_id_for_warp(self, warp_id: usize) -> Option<usize> {
        (warp_id < self.warp_count).then(|| warp_id / self.warps_per_cluster())
    }

    /// Return the half-open global warp-ID range owned by one cluster.
    pub fn cluster_warp_range(self, cluster_id: usize) -> Option<Range<usize>> {
        if cluster_id >= self.clusters {
            return None;
        }
        let start = cluster_id * self.warps_per_cluster();
        Some(start..start + self.warps_per_cluster())
    }

    pub fn warp_contexts(self) -> impl ExactSizeIterator<Item = WarpContext> {
        (0..self.warp_count).map(move |warp_id| WarpContext::from_topology(self, warp_id))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyError {
    ZeroDimension {
        name: &'static str,
    },
    DimensionTooLarge {
        name: &'static str,
        value: usize,
        maximum: usize,
    },
    WarpCountOverflow,
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { name } => write!(f, "launch dimension {name} must be non-zero"),
            Self::DimensionTooLarge {
                name,
                value,
                maximum,
            } => write!(
                f,
                "launch dimension {name}={value} exceeds the supported maximum {maximum}"
            ),
            Self::WarpCountOverflow => write!(f, "launch warp count overflows usize"),
        }
    }
}

impl Error for TopologyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_have_stable_linear_coordinates() {
        let topology = LaunchTopology::new(2, 3, 4).unwrap();
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        assert_eq!(contexts.len(), 24);
        assert_eq!(contexts[0].global_warp_id(), 0);
        assert_eq!(contexts[0].cluster_id(), 0);
        assert_eq!(contexts[0].cta_id_in_cluster(), 0);
        assert_eq!(contexts[0].warp_id_in_cta(), 0);

        assert_eq!(contexts[17].cluster_id(), 1);
        assert_eq!(contexts[17].cta_id_in_cluster(), 1);
        assert_eq!(contexts[17].global_cta_id(), 4);
        assert_eq!(contexts[17].warp_id_in_cta(), 1);
        assert!(contexts
            .iter()
            .all(|context| context.active_mask().is_full()));
        assert_eq!(topology.warps_per_cluster(), 12);
        assert_eq!(topology.cluster_id_for_warp(11), Some(0));
        assert_eq!(topology.cluster_id_for_warp(12), Some(1));
        assert_eq!(topology.cluster_id_for_warp(24), None);
        assert_eq!(topology.cluster_warp_range(1), Some(12..24));
        assert_eq!(topology.cluster_warp_range(2), None);
    }

    #[test]
    fn partial_warp_ctas_have_ceil_warps_and_launch_masks() {
        for (threads, warps, last) in [(1, 1, 0x1_u32), (48, 2, 0xffff), (64, 2, u32::MAX)] {
            let topology = LaunchTopology::with_threads_per_cta(2, 1, threads).unwrap();
            assert_eq!(topology.warps_per_cta(), warps);
            assert_eq!(topology.lane_mask_for_warp(2 * warps - 1).bits(), last);
            let contexts = topology.warp_contexts().collect::<Vec<_>>();
            assert_eq!(contexts[warps - 1].active_mask().bits(), last);
            assert_eq!(contexts[warps - 1].launch_mask().bits(), last);
        }
        assert_eq!(
            LaunchTopology::with_threads_per_cta(1, 1, 64).unwrap(),
            LaunchTopology::new(1, 1, 2).unwrap()
        );
    }

    #[test]
    fn dimensions_are_nonzero_and_multiplication_is_checked() {
        assert_eq!(
            LaunchTopology::new(1, 0, 1),
            Err(TopologyError::ZeroDimension {
                name: "ctas_per_cluster"
            })
        );
        assert_eq!(
            LaunchTopology::new(usize::MAX, 2, 1),
            Err(TopologyError::WarpCountOverflow)
        );
    }

    #[test]
    fn dimensions_respect_engine_representation_limits() {
        assert!(LaunchTopology::new(1, MAX_CTAS_PER_CLUSTER, MAX_WARPS_PER_CTA).is_ok());
        assert_eq!(
            LaunchTopology::new(1, MAX_CTAS_PER_CLUSTER + 1, 1),
            Err(TopologyError::DimensionTooLarge {
                name: "ctas_per_cluster",
                value: MAX_CTAS_PER_CLUSTER + 1,
                maximum: MAX_CTAS_PER_CLUSTER,
            })
        );
        assert_eq!(
            LaunchTopology::new(1, 1, MAX_WARPS_PER_CTA + 1),
            Err(TopologyError::DimensionTooLarge {
                name: "warps_per_cta",
                value: MAX_WARPS_PER_CTA + 1,
                maximum: MAX_WARPS_PER_CTA,
            })
        );
    }
}
