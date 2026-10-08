use crate::{LaunchTopology, WarpMask};

/// Typed source of structured lane divergence that affects operation legality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ControlProvenance {
    #[default]
    None,
    ElectSync {
        entry_mask: WarpMask,
    },
}

impl ControlProvenance {
    pub(crate) const fn elect_sync_entry_mask(self) -> Option<WarpMask> {
        match self {
            Self::None => None,
            Self::ElectSync { entry_mask } => Some(entry_mask),
        }
    }

    const fn enter_elect_sync(self, entry_mask: WarpMask, branch_mask: WarpMask) -> Self {
        if entry_mask.bits() == branch_mask.bits() {
            return self;
        }
        match self {
            Self::None => Self::ElectSync { entry_mask },
            Self::ElectSync { .. } => self,
        }
    }
}

/// Immutable launch coordinates plus the current structured-control lane mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WarpContext {
    topology: LaunchTopology,
    global_warp_id: usize,
    cluster_id: usize,
    cta_id_in_cluster: usize,
    warp_id_in_cta: usize,
    active_mask: WarpMask,
    control_provenance: ControlProvenance,
}

impl WarpContext {
    pub(crate) fn from_topology(topology: LaunchTopology, global_warp_id: usize) -> Self {
        debug_assert!(global_warp_id < topology.warp_count());
        let warps_per_cluster = topology.ctas_per_cluster() * topology.warps_per_cta();
        let cluster_id = global_warp_id / warps_per_cluster;
        let within_cluster = global_warp_id % warps_per_cluster;
        let cta_id_in_cluster = within_cluster / topology.warps_per_cta();
        let warp_id_in_cta = within_cluster % topology.warps_per_cta();
        Self {
            topology,
            global_warp_id,
            cluster_id,
            cta_id_in_cluster,
            warp_id_in_cta,
            active_mask: WarpMask::FULL,
            control_provenance: ControlProvenance::None,
        }
    }

    pub const fn topology(self) -> LaunchTopology {
        self.topology
    }

    pub const fn global_warp_id(self) -> usize {
        self.global_warp_id
    }

    pub const fn cluster_id(self) -> usize {
        self.cluster_id
    }

    pub const fn cta_id_in_cluster(self) -> usize {
        self.cta_id_in_cluster
    }

    pub const fn global_cta_id(self) -> usize {
        self.cluster_id * self.topology.ctas_per_cluster() + self.cta_id_in_cluster
    }

    pub const fn warp_id_in_cta(self) -> usize {
        self.warp_id_in_cta
    }

    pub const fn active_mask(self) -> WarpMask {
        self.active_mask
    }

    pub fn set_active_mask(&mut self, active_mask: WarpMask) {
        self.active_mask = active_mask;
    }

    pub const fn with_active_mask(mut self, active_mask: WarpMask) -> Self {
        self.active_mask = active_mask;
        self
    }

    pub(crate) const fn control_provenance(self) -> ControlProvenance {
        self.control_provenance
    }

    pub(crate) const fn with_control_provenance(mut self, provenance: ControlProvenance) -> Self {
        self.control_provenance = provenance;
        self
    }

    /// Enter one branch selected by a value derived from `elect_sync`.
    ///
    /// Native generated control flow knows the elect entry and selected branch
    /// masks. Keeping the provenance update atomic with the active-mask update
    /// avoids exposing the engine's provenance representation to artifacts.
    pub const fn with_elect_sync_active_mask(
        mut self,
        entry_mask: WarpMask,
        branch_mask: WarpMask,
    ) -> Self {
        self.active_mask = branch_mask;
        self.control_provenance = self
            .control_provenance
            .enter_elect_sync(entry_mask, branch_mask);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{ControlProvenance, WarpMask};

    #[test]
    fn elect_sync_control_preserves_outer_origin_and_ignores_uniform_results() {
        let entry = WarpMask::from_bits(0xff);
        let elected = WarpMask::from_bits(0x01);
        let provenance = ControlProvenance::None.enter_elect_sync(entry, elected);
        assert_eq!(provenance.elect_sync_entry_mask(), Some(entry));
        assert_eq!(
            provenance.enter_elect_sync(elected, WarpMask::from_bits(0x02)),
            provenance
        );
        assert_eq!(
            ControlProvenance::None.enter_elect_sync(entry, entry),
            ControlProvenance::None
        );
    }
}
