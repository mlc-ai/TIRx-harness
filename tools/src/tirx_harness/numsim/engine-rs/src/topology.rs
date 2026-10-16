use std::error::Error;
use std::fmt;
use std::ops::Range;

use crate::{WarpContext, WarpMask, WARP_SIZE};

/// Current engine limit, matching CUDA's 1024-thread CTA limit for 32-lane warps.
pub const MAX_WARPS_PER_CTA: usize = 32;

/// CTA member masks in the engine are represented by one `u64`.
pub const MAX_CTAS_PER_CLUSTER: usize = 64;

/// Current engine limit on simultaneously simulated devices.
pub const MAX_RANKS: usize = 64;

/// Linearized rank/cluster/CTA/warp launch dimensions.
///
/// A multi-rank launch models one identical grid per device. Engine-wide
/// cluster, CTA, and warp IDs stay linear across all ranks; each rank owns a
/// contiguous block of `clusters_per_rank()` clusters, and kernel-visible
/// grid coordinates are rank-local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchTopology {
    clusters: usize,
    ctas_per_cluster: usize,
    warps_per_cta: usize,
    threads_per_cta: usize,
    warp_count: usize,
    ranks: usize,
}

impl LaunchTopology {
    pub fn new(
        clusters: usize,
        ctas_per_cluster: usize,
        warps_per_cta: usize,
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
            threads_per_cta: warps_per_cta * WARP_SIZE,
            warp_count,
            ranks: 1,
        })
    }

    /// Replicate one rank's grid of `clusters_per_rank` clusters on `ranks`
    /// devices.
    pub fn with_ranks(
        clusters_per_rank: usize,
        ctas_per_cluster: usize,
        warps_per_cta: usize,
        ranks: usize,
    ) -> Result<Self, TopologyError> {
        if ranks == 0 {
            return Err(TopologyError::ZeroDimension { name: "ranks" });
        }
        if ranks > MAX_RANKS {
            return Err(TopologyError::DimensionTooLarge {
                name: "ranks",
                value: ranks,
                maximum: MAX_RANKS,
            });
        }
        let clusters = clusters_per_rank
            .checked_mul(ranks)
            .ok_or(TopologyError::WarpCountOverflow)?;
        let mut topology = Self::new(clusters, ctas_per_cluster, warps_per_cta)?;
        topology.ranks = ranks;
        Ok(topology)
    }

    /// Total clusters across every rank.
    pub const fn clusters(self) -> usize {
        self.clusters
    }

    pub const fn ranks(self) -> usize {
        self.ranks
    }

    pub const fn clusters_per_rank(self) -> usize {
        self.clusters / self.ranks
    }

    pub const fn ctas_per_rank(self) -> usize {
        self.clusters_per_rank() * self.ctas_per_cluster
    }

    pub const fn warps_per_rank(self) -> usize {
        self.ctas_per_rank() * self.warps_per_cta
    }

    pub const fn rank_of_cluster(self, cluster_id: usize) -> usize {
        cluster_id / self.clusters_per_rank()
    }

    pub const fn rank_of_warp(self, warp_id: usize) -> usize {
        warp_id / self.warps_per_rank()
    }

    /// One-rank topology with the same per-rank dimensions.
    pub fn rank_local(self) -> Self {
        Self {
            clusters: self.clusters_per_rank(),
            warp_count: self.warps_per_rank(),
            ranks: 1,
            ..self
        }
    }

    pub const fn ctas_per_cluster(self) -> usize {
        self.ctas_per_cluster
    }

    pub const fn warps_per_cta(self) -> usize {
        self.warps_per_cta
    }

    pub fn with_cta_threads(mut self, threads: usize) -> Result<Self, TopologyError> {
        if threads == 0 || threads.div_ceil(WARP_SIZE) != self.warps_per_cta {
            return Err(TopologyError::ThreadCountMismatch {
                threads,
                warps: self.warps_per_cta,
            });
        }
        self.threads_per_cta = threads;
        Ok(self)
    }

    pub const fn threads_per_cta(self) -> usize {
        self.threads_per_cta
    }

    pub const fn launched_lanes(self, warp_id_in_cta: usize) -> WarpMask {
        let remaining = self
            .threads_per_cta
            .saturating_sub(warp_id_in_cta * WARP_SIZE);
        if remaining >= WARP_SIZE {
            WarpMask::FULL
        } else {
            WarpMask::from_bits((1_u32 << remaining) - 1)
        }
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
    ThreadCountMismatch {
        threads: usize,
        warps: usize,
    },
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
            Self::ThreadCountMismatch { threads, warps } => write!(
                f,
                "CTA thread count {threads} does not occupy exactly {warps} warps"
            ),
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
    fn partial_ctas_preserve_only_launched_lanes_on_every_rank() {
        for threads in [1_usize, 17, 31, 32, 33, 63, 1023, 1024] {
            let topology = LaunchTopology::with_ranks(2, 2, threads.div_ceil(WARP_SIZE), 4)
                .unwrap()
                .with_cta_threads(threads)
                .unwrap();
            assert_eq!(topology.rank_local().threads_per_cta(), threads);
            for context in topology.warp_contexts() {
                let base = context.warp_id_in_cta() * WARP_SIZE;
                assert_eq!(context.active_mask().len(), (threads - base).min(WARP_SIZE));
                for lane in 0..WARP_SIZE {
                    assert_eq!(context.active_mask().contains(lane), base + lane < threads);
                }
                assert_eq!(
                    context.with_active_mask(WarpMask::FULL).active_mask(),
                    context.active_mask()
                );
            }
        }
        let topology = LaunchTopology::new(1, 1, 2).unwrap();
        for threads in [0, 1, 32, 65, usize::MAX] {
            assert!(topology.with_cta_threads(threads).is_err());
        }
    }

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
    fn ranks_partition_linear_clusters_into_rank_local_grids() {
        let topology = LaunchTopology::with_ranks(2, 2, 3, 4).unwrap();
        assert_eq!(topology.clusters(), 8);
        assert_eq!(topology.ranks(), 4);
        assert_eq!(topology.clusters_per_rank(), 2);
        assert_eq!(topology.warps_per_rank(), 12);
        let contexts = topology.warp_contexts().collect::<Vec<_>>();
        let warp = contexts[2 * 12 + 7];
        assert_eq!(warp.rank(), 2);
        assert_eq!(warp.cluster_id(), 5);
        assert_eq!(warp.kernel_cluster_id(), 1);
        assert_eq!(warp.global_cta_id(), 10);
        assert_eq!(warp.kernel_cta_id(), 2);
        assert_eq!(warp.kernel_topology().clusters(), 2);
        assert_eq!(warp.kernel_topology().cta_count(), 4);
        assert_eq!(topology.rank_of_cluster(7), 3);
        assert_eq!(
            LaunchTopology::with_ranks(1, 1, 1, 0),
            Err(TopologyError::ZeroDimension { name: "ranks" })
        );
        let single = LaunchTopology::new(3, 1, 1).unwrap();
        assert_eq!(single.ranks(), 1);
        assert_eq!(single.rank_local(), single);
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
