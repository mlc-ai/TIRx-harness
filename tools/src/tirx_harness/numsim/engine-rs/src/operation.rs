use std::fmt;
use std::sync::Arc;

use crate::{ControlProvenance, WarpMask};

/// Stable identifier assigned to an operation or control-flow site at transpile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct StaticOpId(u64);

impl StaticOpId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StaticOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op:{}", self.0)
    }
}

/// One active loop in the dynamic control-flow path of an operation.
///
/// `iteration_ordinal` counts executions of the loop body. It is deliberately
/// independent of the loop's induction value, which may be non-integral or
/// data-dependent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LoopFrame {
    loop_site_id: StaticOpId,
    iteration_ordinal: u64,
}

impl LoopFrame {
    pub(crate) const fn new(loop_site_id: StaticOpId, iteration_ordinal: u64) -> Self {
        Self {
            loop_site_id,
            iteration_ordinal,
        }
    }

    pub(crate) const fn loop_site_id(self) -> StaticOpId {
        self.loop_site_id
    }

    pub(crate) const fn iteration_ordinal(self) -> u64 {
        self.iteration_ordinal
    }
}

impl fmt::Display for LoopFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.loop_site_id, self.iteration_ordinal)
    }
}

/// Identity of one concrete execution of a transpiled semantic operation.
///
/// The per-warp sequence makes repeated executions unambiguous, while the
/// source operation and complete loop stack preserve useful source-level
/// identity for reports and cross-tool correlation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DynamicOpId {
    kernel_index: usize,
    global_warp_id: usize,
    per_warp_sequence: u64,
    source_op_id: StaticOpId,
    loop_frames: Arc<[LoopFrame]>,
}

impl DynamicOpId {
    pub(crate) fn new(
        kernel_index: usize,
        global_warp_id: usize,
        per_warp_sequence: u64,
        source_op_id: StaticOpId,
        loop_frames: impl Into<Arc<[LoopFrame]>>,
    ) -> Self {
        Self::new_shared(
            kernel_index,
            global_warp_id,
            per_warp_sequence,
            source_op_id,
            Arc::from(loop_frames.into()),
        )
    }

    pub(crate) const fn new_shared(
        kernel_index: usize,
        global_warp_id: usize,
        per_warp_sequence: u64,
        source_op_id: StaticOpId,
        loop_frames: Arc<[LoopFrame]>,
    ) -> Self {
        Self {
            kernel_index,
            global_warp_id,
            per_warp_sequence,
            source_op_id,
            loop_frames,
        }
    }

    pub(crate) const fn kernel_index(&self) -> usize {
        self.kernel_index
    }

    pub(crate) const fn global_warp_id(&self) -> usize {
        self.global_warp_id
    }

    pub(crate) const fn per_warp_sequence(&self) -> u64 {
        self.per_warp_sequence
    }

    pub(crate) const fn source_op_id(&self) -> StaticOpId {
        self.source_op_id
    }

    pub(crate) fn loop_frames(&self) -> &[LoopFrame] {
        &self.loop_frames
    }

    pub(crate) const fn shared_loop_frames(&self) -> &Arc<[LoopFrame]> {
        &self.loop_frames
    }

    /// Compare stable TIR call sites: kernel index and source operation only.
    /// Runtime loop iterations deliberately share one origin; final backend
    /// instruction duplication is outside this model.
    pub(crate) fn same_static_instruction(&self, other: &DynamicOpId) -> bool {
        self.kernel_index == other.kernel_index && self.source_op_id == other.source_op_id
    }
}

impl fmt::Display for DynamicOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kernel:{}/warp:{}/seq:{}/source:{}/loops:[",
            self.kernel_index, self.global_warp_id, self.per_warp_sequence, self.source_op_id
        )?;
        for (index, frame) in self.loop_frames.iter().enumerate() {
            if index != 0 {
                f.write_str(",")?;
            }
            write!(f, "{frame}")?;
        }
        f.write_str("]")
    }
}

/// Broad semantic category shared by the native analysis modes.
///
/// Explicit discriminants make the derived ordering independent of later
/// source reformatting. New variants must be appended rather than inserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OperationKind {
    Load = 0,
    Store = 1,
    Atomic = 2,
    AsyncIssue = 3,
    Barrier = 4,
    Fence = 5,
    Collective = 6,
    Control = 7,
    Lifecycle = 8,
    TcgenWork = 9,
    MbarrierInitFence = 10,
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Load => "load",
            Self::Store => "store",
            Self::Atomic => "atomic",
            Self::AsyncIssue => "async_issue",
            Self::Barrier => "barrier",
            Self::Fence => "fence",
            Self::Collective => "collective",
            Self::Control => "control",
            Self::Lifecycle => "lifecycle",
            Self::TcgenWork => "tcgen_work",
            Self::MbarrierInitFence => "mbarrier_init_fence",
        })
    }
}

/// Analysis-visible metadata for one concrete semantic operation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationContext {
    id: Arc<DynamicOpId>,
    kind: OperationKind,
    active_mask: WarpMask,
    control_provenance: ControlProvenance,
}

impl OperationContext {
    pub(crate) fn new(id: DynamicOpId, kind: OperationKind, active_mask: WarpMask) -> Self {
        Self {
            id: Arc::new(id),
            kind,
            active_mask,
            control_provenance: ControlProvenance::None,
        }
    }

    pub(crate) const fn with_control_provenance(mut self, provenance: ControlProvenance) -> Self {
        self.control_provenance = provenance;
        self
    }

    pub(crate) const fn with_active_mask(mut self, active_mask: WarpMask) -> Self {
        self.active_mask = active_mask;
        self
    }

    pub(crate) fn id(&self) -> &DynamicOpId {
        self.id.as_ref()
    }

    pub(crate) fn shared_id(&self) -> Arc<DynamicOpId> {
        Arc::clone(&self.id)
    }

    pub(crate) const fn kind(&self) -> OperationKind {
        self.kind
    }

    pub(crate) const fn active_mask(&self) -> WarpMask {
        self.active_mask
    }

    pub(crate) const fn control_provenance(&self) -> ControlProvenance {
        self.control_provenance
    }
}

impl fmt::Display for OperationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} kind:{} active_mask:{:#010x}",
            self.id,
            self.kind,
            self.active_mask.bits()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::*;

    fn id(
        kernel_index: usize,
        global_warp_id: usize,
        per_warp_sequence: u64,
        source_op_id: u64,
        loop_frames: &[(u64, u64)],
    ) -> DynamicOpId {
        DynamicOpId::new(
            kernel_index,
            global_warp_id,
            per_warp_sequence,
            StaticOpId::new(source_op_id),
            loop_frames
                .iter()
                .map(|&(site, iteration)| LoopFrame::new(StaticOpId::new(site), iteration))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn every_dynamic_coordinate_participates_in_identity() {
        let baseline = id(2, 7, 11, 19, &[(3, 5), (13, 17)]);
        let variants = [
            id(3, 7, 11, 19, &[(3, 5), (13, 17)]),
            id(2, 8, 11, 19, &[(3, 5), (13, 17)]),
            id(2, 7, 12, 19, &[(3, 5), (13, 17)]),
            id(2, 7, 11, 20, &[(3, 5), (13, 17)]),
            id(2, 7, 11, 19, &[(4, 5), (13, 17)]),
            id(2, 7, 11, 19, &[(3, 6), (13, 17)]),
            id(2, 7, 11, 19, &[(3, 5), (14, 17)]),
            id(2, 7, 11, 19, &[(3, 5), (13, 18)]),
            id(2, 7, 11, 19, &[(13, 17), (3, 5)]),
        ];

        let mut identities = HashSet::from([baseline]);
        identities.extend(variants);
        assert_eq!(identities.len(), 10);
    }

    #[test]
    fn operation_context_includes_kind_and_active_mask_in_identity() {
        let operation_id = id(0, 1, 2, 3, &[]);
        let contexts = HashSet::from([
            OperationContext::new(
                operation_id.clone(),
                OperationKind::Load,
                WarpMask::from_bits(0x0000_000f),
            ),
            OperationContext::new(
                operation_id.clone(),
                OperationKind::Store,
                WarpMask::from_bits(0x0000_000f),
            ),
            OperationContext::new(
                operation_id.clone(),
                OperationKind::Load,
                WarpMask::from_bits(0x0000_00f0),
            ),
            OperationContext::new(
                operation_id,
                OperationKind::Load,
                WarpMask::from_bits(0x0000_000f),
            )
            .with_control_provenance(ControlProvenance::ElectSync {
                entry_mask: WarpMask::FULL,
            }),
        ]);

        assert_eq!(contexts.len(), 4);
    }

    #[test]
    fn derived_order_is_execution_coordinate_order() {
        let expected = vec![
            id(0, 0, 0, 9, &[(8, 1)]),
            id(0, 0, 1, 2, &[(4, 7)]),
            id(0, 1, 0, 1, &[]),
            id(1, 0, 0, 1, &[]),
        ];
        let set = BTreeSet::from([
            expected[3].clone(),
            expected[1].clone(),
            expected[0].clone(),
            expected[2].clone(),
        ]);

        assert_eq!(set.into_iter().collect::<Vec<_>>(), expected);
        assert!(OperationKind::Load < OperationKind::Store);
        assert!(OperationKind::Store < OperationKind::Atomic);
        assert!(WarpMask::from_bits(1) < WarpMask::from_bits(2));
    }

    #[test]
    fn operation_kind_display_and_order_are_stable() {
        let kinds = [
            OperationKind::Load,
            OperationKind::Store,
            OperationKind::Atomic,
            OperationKind::AsyncIssue,
            OperationKind::Barrier,
            OperationKind::Fence,
            OperationKind::Collective,
            OperationKind::Control,
            OperationKind::Lifecycle,
            OperationKind::TcgenWork,
            OperationKind::MbarrierInitFence,
        ];
        let displays = [
            "load",
            "store",
            "atomic",
            "async_issue",
            "barrier",
            "fence",
            "collective",
            "control",
            "lifecycle",
            "tcgen_work",
            "mbarrier_init_fence",
        ];

        assert!(kinds.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            kinds.map(|kind| kind.to_string()),
            displays.map(str::to_owned)
        );
    }

    #[test]
    fn display_is_stable_and_includes_the_complete_loop_stack() {
        let operation_id = id(2, 7, 11, 19, &[(3, 5), (13, 17)]);
        assert_eq!(StaticOpId::new(19).to_string(), "op:19");
        assert_eq!(LoopFrame::new(StaticOpId::new(3), 5).to_string(), "op:3@5");
        assert_eq!(
            operation_id.to_string(),
            "kernel:2/warp:7/seq:11/source:op:19/loops:[op:3@5,op:13@17]"
        );

        let context = OperationContext::new(
            operation_id,
            OperationKind::AsyncIssue,
            WarpMask::from_bits(0x8000_0003),
        );
        assert_eq!(
            context.to_string(),
            "kernel:2/warp:7/seq:11/source:op:19/loops:[op:3@5,op:13@17] kind:async_issue active_mask:0x80000003"
        );
    }
}
