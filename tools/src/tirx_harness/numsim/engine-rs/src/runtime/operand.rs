use std::fmt;
use std::sync::Arc;

use crate::{
    BufferView, EngineError, PhysicalAccessSpace, PhysicalBarrierId, PhysicalMemory,
    SharedAllocation, TmemAllocation, WarpContext, WarpMask, WarpPrivateAllocation, WarpValue,
};

#[derive(Clone)]
// Remote pointers carry full per-lane coordinates; keeping them inline avoids
// heap allocation on every generated runtime operation.
#[allow(clippy::large_enum_variant)]
pub enum RuntimeBuffer {
    AccessView {
        buffer: Arc<RuntimeBuffer>,
        readable_lanes: WarpMask,
        writable_lanes: WarpMask,
    },
    LaneSelected {
        buffers: WarpValue<Arc<RuntimeBuffer>>,
    },
    Global(BufferView),
    Shared {
        allocations: Arc<Vec<SharedAllocation>>,
        byte_offset: usize,
        byte_len: usize,
        backing_byte_len: usize,
        virtual_base: usize,
    },
    RemoteShared {
        allocations: Arc<Vec<SharedAllocation>>,
        byte_offsets: WarpValue<i64>,
        byte_len: usize,
        target_cta_ids: WarpValue<i64>,
        virtual_base: usize,
    },
    Local {
        allocations: Arc<Vec<WarpPrivateAllocation>>,
        byte_offset: usize,
        byte_len: usize,
    },
    Register {
        allocations: Arc<Vec<WarpPrivateAllocation>>,
        byte_offset: usize,
        byte_len: usize,
    },
    Tmem {
        allocations: Arc<Vec<TmemAllocation>>,
        lane_span: usize,
        tcol_span_elements: usize,
        elem_offset: usize,
        itemsize: usize,
    },
}

/// What a walker is doing through a `DeclBuffer` view.
///
/// Selects the per-lane permission `RuntimeBuffer::AccessView` enforces as it
/// is peeled.
#[derive(Clone, Copy)]
pub enum ViewAccess {
    /// Reads the viewed bytes; the lane must be readable.
    Read,
    /// Writes the viewed bytes; the lane must be writable.
    Write,
    /// Invalidates the viewed bytes; the lane must be writable, and says so in
    /// its own words.
    Invalidate,
    /// Resolves an address without touching the bytes; no permission applies.
    Address,
    /// Whatever a resolved `PhysicalAccessKind` reads and/or writes.
    Kind { reads: bool, writes: bool },
}

/// The one walk over `RuntimeBuffer`'s two transparent wrapper variants.
///
/// `AccessView` and `LaneSelected` are not storage. Every physical-access
/// walker peels them the same way — enforce the view's per-lane permission,
/// then continue on the wrapped buffer, or on the buffer this lane selected —
/// and only the six leaf variants differ per walker. So the leaves stay with
/// their walker and the traversal lives here once.
///
/// Peeling iteratively is equivalent to the self-recursion the walkers used to
/// spell out as the first two arms of their own `match`: every one of them
/// passed `lane` through unchanged and re-entered with only the buffer
/// replaced.
pub fn peel_runtime_buffer_wrappers(
    mut buffer: &RuntimeBuffer,
    lane: usize,
    access: ViewAccess,
) -> Result<&RuntimeBuffer, EngineError> {
    loop {
        match buffer {
            RuntimeBuffer::AccessView {
                buffer: inner,
                readable_lanes,
                writable_lanes,
            } => {
                let (reads, writes) = match access {
                    ViewAccess::Read => (true, false),
                    ViewAccess::Write | ViewAccess::Invalidate => (false, true),
                    ViewAccess::Address => (false, false),
                    ViewAccess::Kind { reads, writes } => (reads, writes),
                };
                if reads && !readable_lanes.contains(lane) {
                    return Err(EngineError::message(format!(
                        "read through a non-readable DeclBuffer view on lane {lane}"
                    )));
                }
                if writes && !writable_lanes.contains(lane) {
                    let verb = match access {
                        ViewAccess::Invalidate => "invalidate",
                        _ => "write",
                    };
                    return Err(EngineError::message(format!(
                        "{verb} through a non-writable DeclBuffer view on lane {lane}"
                    )));
                }
                buffer = inner;
            }
            RuntimeBuffer::LaneSelected { buffers } => buffer = &buffers[lane],
            _ => return Ok(buffer),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PointerSpace {
    Global,
    Shared,
    Local,
    Register,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtxStateSpace {
    Generic,
    Global,
    Shared,
    SharedCta,
    SharedCluster,
    Local,
}

impl fmt::Display for PtxStateSpace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Generic => "generic",
            Self::Global => "global",
            Self::Shared => "shared",
            Self::SharedCta => "shared::cta",
            Self::SharedCluster => "shared::cluster",
            Self::Local => "local",
        })
    }
}

#[derive(Clone)]
struct PhysicalPtrState {
    buffer: RuntimeBuffer,
    element_indices: WarpValue<i64>,
    itemsize: usize,
    byte_offsets: WarpValue<i128>,
    // These lanes have integer bits but no allocation provenance. Their
    // byte_offsets are normalized u64 values, and element_indices are zero.
    integer_lanes: WarpMask,
    pointee_itemsize: usize,
    readable_lanes: WarpMask,
    writable_lanes: WarpMask,
    access_view_permissions: bool,
    bounded_lanes: WarpMask,
    range_starts: WarpValue<i128>,
    range_ends: WarpValue<i128>,
}

/// A lane-wise physical pointer with cheap value cloning.
///
/// Instruction operands are immutable views of a pointer.  Keep the sizeable
/// per-lane state behind one shared allocation so passing a reusable pointer
/// through the existing owned `Address` ABI only increments a refcount.  The
/// persistent pointer-derivation methods below materialize a private state
/// exactly when the derived value needs to change it; `PhysicalPtrSlot` does
/// the same for lane merges.
#[derive(Clone)]
pub struct PhysicalPtr {
    state: Arc<PhysicalPtrState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalAddress {
    allocation_id: u64,
    byte_offset: usize,
}

impl PhysicalAddress {
    pub(crate) const fn new(allocation_id: u64, byte_offset: usize) -> Self {
        Self {
            allocation_id,
            byte_offset,
        }
    }

    pub const fn allocation_id(self) -> u64 {
        self.allocation_id
    }

    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }
}

impl RuntimeBuffer {
    /// Return the physical space when every lane-selected leaf has the same
    /// statically visible space.
    ///
    /// Analysis modes use this cheap classification to bypass address
    /// resolution for spaces outside their conflict domain. A mixed
    /// lane-selected buffer returns `None` and retains the exact resolver.
    pub(crate) fn uniform_physical_space(&self) -> Option<PhysicalAccessSpace> {
        match self {
            Self::AccessView { buffer, .. } => buffer.uniform_physical_space(),
            Self::LaneSelected { buffers } => {
                let mut spaces = buffers
                    .lanes()
                    .iter()
                    .map(|buffer| buffer.uniform_physical_space());
                let first = spaces.next().flatten()?;
                spaces.all(|space| space == Some(first)).then_some(first)
            }
            Self::Global(_) => Some(PhysicalAccessSpace::Global),
            Self::Shared { .. } | Self::RemoteShared { .. } => Some(PhysicalAccessSpace::Shared),
            Self::Local { .. } => Some(PhysicalAccessSpace::Local),
            Self::Register { .. } => Some(PhysicalAccessSpace::Register),
            Self::Tmem { .. } => Some(PhysicalAccessSpace::Tmem),
        }
    }

    pub(crate) fn physical_address_at(
        &self,
        context: &WarpContext,
        lane: usize,
        byte_offset: usize,
        target_cta_id_in_cluster: Option<usize>,
    ) -> Result<PhysicalAddress, EngineError> {
        match peel_runtime_buffer_wrappers(self, lane, ViewAccess::Address)? {
            Self::AccessView { .. } | Self::LaneSelected { .. } => {
                unreachable!("runtime buffer wrappers are peeled before the leaf walk")
            }
            Self::Global(view) => Ok(PhysicalAddress::new(
                view.allocation().as_u64(),
                view.byte_offset().checked_add(byte_offset).ok_or_else(|| {
                    EngineError::out_of_bounds("global physical address overflow")
                })?,
            )),
            Self::Shared {
                allocations,
                byte_offset: view_offset,
                ..
            } => {
                let target_global = if let Some(target_local) = target_cta_id_in_cluster {
                    if target_local >= context.topology().ctas_per_cluster() {
                        return Err(EngineError::message(
                            "shared physical address target CTA is outside the cluster",
                        ));
                    }
                    context
                        .cluster_id()
                        .checked_mul(context.topology().ctas_per_cluster())
                        .and_then(|base| base.checked_add(target_local))
                        .ok_or_else(|| {
                            EngineError::message("shared physical address CTA overflow")
                        })?
                } else {
                    context.global_cta_id()
                };
                let allocation = allocations.get(target_global).ok_or_else(|| {
                    EngineError::message("shared physical address allocation is missing")
                })?;
                Ok(PhysicalAddress::new(
                    allocation.allocation().as_u64(),
                    view_offset.checked_add(byte_offset).ok_or_else(|| {
                        EngineError::out_of_bounds("shared physical address offset overflow")
                    })?,
                ))
            }
            Self::RemoteShared {
                allocations,
                byte_offsets,
                target_cta_ids,
                ..
            } => {
                if target_cta_id_in_cluster.is_some() {
                    return Err(EngineError::message(
                        "remote shared physical address cannot override its mapped CTA",
                    ));
                }
                let target_local = usize::try_from(target_cta_ids[lane])
                    .map_err(|_| EngineError::message("negative mapped shared CTA id"))?;
                if target_local >= context.topology().ctas_per_cluster() {
                    return Err(EngineError::message(
                        "mapped shared physical address target CTA is outside the cluster",
                    ));
                }
                let target_global = context
                    .cluster_id()
                    .checked_mul(context.topology().ctas_per_cluster())
                    .and_then(|base| base.checked_add(target_local))
                    .ok_or_else(|| {
                        EngineError::message("mapped shared physical address CTA overflow")
                    })?;
                let allocation = allocations.get(target_global).ok_or_else(|| {
                    EngineError::message("mapped shared physical address allocation is missing")
                })?;
                let view_offset = usize::try_from(byte_offsets[lane]).map_err(|_| {
                    EngineError::out_of_bounds("negative mapped shared byte offset")
                })?;
                Ok(PhysicalAddress::new(
                    allocation.allocation().as_u64(),
                    view_offset.checked_add(byte_offset).ok_or_else(|| {
                        EngineError::out_of_bounds("mapped shared physical address offset overflow")
                    })?,
                ))
            }
            Self::Local {
                allocations,
                byte_offset: view_offset,
                ..
            }
            | Self::Register {
                allocations,
                byte_offset: view_offset,
                ..
            } => {
                let allocation = allocations.get(context.global_warp_id()).ok_or_else(|| {
                    EngineError::message("warp-private physical address allocation is missing")
                })?;
                let lane_base = lane
                    .checked_mul(allocation.bytes_per_lane())
                    .ok_or_else(|| {
                        EngineError::out_of_bounds("warp-private lane offset overflow")
                    })?;
                Ok(PhysicalAddress::new(
                    allocation.allocation().as_u64(),
                    lane_base
                        .checked_add(*view_offset)
                        .and_then(|offset| offset.checked_add(byte_offset))
                        .ok_or_else(|| {
                            EngineError::out_of_bounds("warp-private physical address overflow")
                        })?,
                ))
            }
            Self::Tmem { .. } => Err(EngineError::message(
                "generic physical addresses cannot name TMEM cells",
            )),
        }
    }
}

impl PhysicalPtr {
    pub fn buffer(&self) -> &RuntimeBuffer {
        &self.state.buffer
    }

    pub fn element_indices(&self) -> &WarpValue<i64> {
        &self.state.element_indices
    }

    pub fn byte_offsets(&self) -> &WarpValue<i128> {
        &self.state.byte_offsets
    }

    pub fn pointee_itemsize(&self) -> usize {
        self.state.pointee_itemsize
    }

    pub fn new(buffer: RuntimeBuffer, element_indices: WarpValue<i64>, itemsize: usize) -> Self {
        let (buffer, readable_lanes, writable_lanes, restricted) = peel_runtime_access_view(buffer);
        let (buffer, byte_offsets, range_starts, range_ends) = canonicalize_pointer_buffer(buffer);
        Self {
            state: Arc::new(PhysicalPtrState {
                buffer,
                element_indices,
                itemsize,
                byte_offsets,
                integer_lanes: WarpMask::EMPTY,
                pointee_itemsize: itemsize,
                readable_lanes,
                writable_lanes,
                access_view_permissions: restricted,
                bounded_lanes: if restricted {
                    WarpMask::FULL
                } else {
                    WarpMask::EMPTY
                },
                range_starts,
                range_ends,
            }),
        }
    }

    /// Keep ordinary pointer arithmetic allocation-relative, but let a
    /// frontend mark a direct semantic Buffer access as bounded by the view
    /// from which this pointer was formed.
    pub(crate) fn bounded_to_initial_view(mut self) -> Self {
        Arc::make_mut(&mut self.state).bounded_lanes = WarpMask::FULL;
        self
    }

    pub fn null() -> Self {
        Self::integer(WarpValue::splat(0))
    }

    pub fn integer(bits: WarpValue<u64>) -> Self {
        // Integer bits carry no allocation. Resolve only at a memory access;
        // ordinary comparisons and arithmetic must not invent a backing.
        let mut pointer = Self::new(
            RuntimeBuffer::Local {
                allocations: Arc::new(Vec::new()),
                byte_offset: 0,
                byte_len: 0,
            },
            WarpValue::splat(0),
            1,
        );
        let state = Arc::make_mut(&mut pointer.state);
        state.integer_lanes = WarpMask::FULL;
        state.byte_offsets = WarpValue::from_fn(|lane| i128::from(bits[lane]));
        pointer
    }


    pub(crate) fn integer_lanes(&self) -> WarpMask {
        self.state.integer_lanes
    }

    fn unresolved_address_error(&self, lane: usize) -> EngineError {
        if self.state.byte_offsets[lane] == 0 {
            EngineError::out_of_bounds(format!("null pointer access on lane {lane}"))
        } else {
            EngineError::analysis_incomplete("integer_address_without_binding")
        }
    }


    pub fn with_pointee_itemsize(&self, pointee_itemsize: usize) -> Self {
        let mut result = self.clone();
        Arc::make_mut(&mut result.state).pointee_itemsize = pointee_itemsize;
        result
    }

    /// Numeric address transport erases the source element type, not its
    /// allocation or access contract. Use byte units so equivalent typed
    /// views have one identity; a later typed cast supplies its own width.
    /// A uint64 carrier wraps the complete byte offset modulo 2^64, including
    /// element indices. This never changes its original access contract.
    pub fn into_byte_address(mut self, mask: WarpMask) -> Result<Self, EngineError> {
        for lane in mask {
            let offset = self.raw_relative_byte_offset(lane)?;
            let state = Arc::make_mut(&mut self.state);
            state.element_indices[lane] = 0;
            state.byte_offsets[lane] = if state.integer_lanes.contains(lane) {
                offset as u64 as i128
            } else {
                offset as i64 as i128
            };
        }
        let state = Arc::make_mut(&mut self.state);
        state.itemsize = 1;
        state.pointee_itemsize = 1;
        Ok(self)
    }

    pub fn with_byte_storage_access_width(&self, access_width: usize) -> Result<Self, EngineError> {
        if access_width == 0 {
            return Err(EngineError::message(
                "raw byte-storage access width must be nonzero",
            ));
        }
        if self.state.itemsize == 1 {
            return Ok(self.with_pointee_itemsize(access_width));
        }
        Ok(self.clone())
    }

    pub fn with_byte_offset(
        &self,
        byte_offsets: &WarpValue<i64>,
        pointee_itemsize: usize,
        mask: WarpMask,
    ) -> Result<Self, EngineError> {
        if pointee_itemsize == 0 {
            return Err(EngineError::message(
                "pointer pointee width must be nonzero",
            ));
        }
        let mut combined = self.state.byte_offsets.clone();
        for lane in mask {
            if self.state.integer_lanes.contains(lane) {
                combined[lane] =
                    (combined[lane] as u64).wrapping_add(byte_offsets[lane] as u64) as i128;
                continue;
            }
            combined[lane] = combined[lane]
                .checked_add(i128::from(byte_offsets[lane]))
                .ok_or_else(|| {
                    EngineError::out_of_bounds(format!(
                        "pointer byte offset overflow on lane {lane}"
                    ))
                })?;
        }
        let buffer = match &self.state.buffer {
            RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                byte_len,
                backing_byte_len,
                virtual_base,
            } => {
                let physical_tail =
                    backing_byte_len.checked_sub(*byte_offset).ok_or_else(|| {
                        EngineError::message(
                            "shared-memory view starts beyond its physical backing",
                        )
                    })?;
                RuntimeBuffer::Shared {
                    allocations: allocations.clone(),
                    byte_offset: *byte_offset,
                    byte_len: (*byte_len).max(physical_tail),
                    backing_byte_len: *backing_byte_len,
                    virtual_base: *virtual_base,
                }
            }
            _ => self.state.buffer.clone(),
        };
        let mut result = self.clone();
        let state = Arc::make_mut(&mut result.state);
        state.buffer = buffer;
        state.byte_offsets = combined;
        state.pointee_itemsize = pointee_itemsize;
        for lane in mask - self.state.integer_lanes {
            // Pointer arithmetic carries the access contract; it does not
            // exercise it. Only check arithmetic overflow at this point.
            let _ = result.raw_relative_byte_offset(lane)?;
        }
        Ok(result)
    }

    pub fn with_element_offset_extent(
        &self,
        element_offsets: &WarpValue<i64>,
        element_extents: &WarpValue<i64>,
        pointee_itemsize: usize,
        mask: WarpMask,
        access_mask: u8,
        operation: &str,
    ) -> Result<Self, EngineError> {
        self.with_element_offset_extent_labeled(
            element_offsets,
            element_extents,
            pointee_itemsize,
            mask,
            access_mask,
            &crate::DiagnosticLabel::new(operation),
        )
    }

    pub(crate) fn with_element_offset_extent_labeled(
        &self,
        element_offsets: &WarpValue<i64>,
        element_extents: &WarpValue<i64>,
        pointee_itemsize: usize,
        mask: WarpMask,
        access_mask: u8,
        operation: &crate::DiagnosticLabel,
    ) -> Result<Self, EngineError> {
        if !(1_u8..=3_u8).contains(&access_mask) {
            return Err(operation.engine_error(format_args!(
                " access mask must be read=1, write=2, or both=3"
            )));
        }
        let itemsize = i64::try_from(pointee_itemsize)
            .map_err(|_| operation.engine_error(format_args!(" itemsize exceeds i64")))?;
        let mut byte_offsets = WarpValue::splat(0_i64);
        for lane in mask {
            if element_extents[lane] < 0 {
                return Err(operation.engine_error(format_args!(" extent is negative")));
            }
            byte_offsets[lane] = element_offsets[lane]
                .checked_mul(itemsize)
                .ok_or_else(|| operation.out_of_bounds(format_args!(" byte offset overflow")))?;
        }
        let mut result = self.with_byte_offset(&byte_offsets, pointee_itemsize, mask)?;
        for lane in mask {
            let requests_read = access_mask & 1 != 0;
            let requests_write = access_mask & 2 != 0;
            if requests_read && !self.state.readable_lanes.contains(lane) {
                return Err(operation.engine_error(format_args!(
                    " cannot add read access to a non-readable base pointer on lane {lane}"
                )));
            }
            if requests_write && !self.state.writable_lanes.contains(lane) {
                return Err(operation.engine_error(format_args!(
                    " cannot add write access to a non-writable base pointer on lane {lane}"
                )));
            }
            let extent_bytes = i128::from(element_extents[lane])
                .checked_mul(i128::from(itemsize))
                .ok_or_else(|| operation.out_of_bounds(format_args!(" byte extent overflow")))?;
            let extent_bytes = usize::try_from(extent_bytes)
                .map_err(|_| operation.out_of_bounds(format_args!(" byte extent exceeds usize")))?;
            let start = result.lane_address_byte_offset(lane, extent_bytes)? as i128;
            let end = start
                .checked_add(extent_bytes as i128)
                .ok_or_else(|| operation.out_of_bounds(format_args!(" byte range overflow")))?;
            let lane_mask = WarpMask::from_bits(1_u32 << lane);
            let state = Arc::make_mut(&mut result.state);
            state.readable_lanes = (state.readable_lanes - lane_mask)
                | if requests_read {
                    lane_mask
                } else {
                    WarpMask::EMPTY
                };
            state.writable_lanes = (state.writable_lanes - lane_mask)
                | if requests_write {
                    lane_mask
                } else {
                    WarpMask::EMPTY
                };
            state.bounded_lanes |= lane_mask;
            state.range_starts[lane] = start;
            state.range_ends[lane] = end;
        }
        Ok(result)
    }

    fn raw_relative_byte_offset(&self, lane: usize) -> Result<i128, EngineError> {
        let element_bytes = i128::from(self.state.element_indices[lane])
            .checked_mul(
                i128::try_from(self.state.itemsize)
                    .map_err(|_| EngineError::message("pointer itemsize exceeds i128"))?,
            )
            .ok_or_else(|| EngineError::out_of_bounds("pointer element byte offset overflow"))?;
        element_bytes
            .checked_add(self.state.byte_offsets[lane])
            .ok_or_else(|| EngineError::out_of_bounds("pointer relative byte offset overflow"))
    }

    fn lane_byte_offset_at(
        &self,
        lane: usize,
        byte_delta: usize,
        access_byte_len: usize,
        require_read: bool,
        require_write: bool,
    ) -> Result<usize, EngineError> {
        if self.state.integer_lanes.contains(lane) {
            return Err(self.unresolved_address_error(lane));
        }
        if require_read && !self.state.readable_lanes.contains(lane) {
            let subject = if self.state.access_view_permissions {
                "DeclBuffer view"
            } else {
                "physical pointer"
            };
            return Err(EngineError::message(format!(
                "read through a non-readable {subject} on lane {lane}"
            )));
        }
        if require_write && !self.state.writable_lanes.contains(lane) {
            let subject = if self.state.access_view_permissions {
                "DeclBuffer view"
            } else {
                "physical pointer"
            };
            return Err(EngineError::message(format!(
                "write through a non-writable {subject} on lane {lane}"
            )));
        }
        let relative = self
            .raw_relative_byte_offset(lane)?
            .checked_add(byte_delta as i128)
            .ok_or_else(|| EngineError::out_of_bounds("pointer relative byte offset overflow"))?;
        let end = relative
            .checked_add(access_byte_len as i128)
            .ok_or_else(|| EngineError::out_of_bounds("pointer access end overflow"))?;
        if self.state.bounded_lanes.contains(lane)
            && (relative < self.state.range_starts[lane] || end > self.state.range_ends[lane])
        {
            return Err(EngineError::out_of_bounds(format!(
                "pointer byte range [{relative}, {end}) is outside tvm_access_ptr range [{}, {}) on lane {lane}",
                self.state.range_starts[lane], self.state.range_ends[lane]
            )));
        }
        let relative_usize = usize::try_from(relative).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "negative physical pointer byte offset {relative} on lane {lane}"
            ))
        })?;
        let end_usize = usize::try_from(end)
            .map_err(|_| EngineError::out_of_bounds("pointer access end exceeds usize"))?;
        let byte_len = runtime_buffer_byte_len_at(&self.state.buffer, lane);
        if end_usize > byte_len {
            return Err(EngineError::out_of_bounds(format!(
                "pointer byte range [{relative_usize}, {end_usize}) exceeds {byte_len} bytes on lane {lane}"
            )));
        }
        Ok(relative_usize)
    }

    fn lane_shared_view_byte_offset_at(
        &self,
        lane: usize,
        byte_delta: usize,
        view_byte_len: usize,
    ) -> Result<usize, EngineError> {
        if self.state.bounded_lanes.contains(lane) {
            return self.lane_byte_offset_at(lane, byte_delta, view_byte_len, false, false);
        }
        let relative = self
            .raw_relative_byte_offset(lane)?
            .checked_add(byte_delta as i128)
            .ok_or_else(|| EngineError::out_of_bounds("shared view byte offset overflow"))?;
        let end = relative
            .checked_add(view_byte_len as i128)
            .ok_or_else(|| EngineError::out_of_bounds("shared view byte range overflow"))?;
        let relative = usize::try_from(relative).map_err(|_| {
            EngineError::out_of_bounds(format!(
                "negative shared view byte offset {relative} on lane {lane}"
            ))
        })?;
        let _ = usize::try_from(end)
            .map_err(|_| EngineError::out_of_bounds("shared view byte end exceeds usize"))?;
        Ok(relative)
    }

    pub fn lane_address_byte_offset(
        &self,
        lane: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, 0, access_byte_len, false, false)
    }

    pub fn lane_physical_byte_offset(
        &self,
        lane: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        let relative = self.lane_address_byte_offset(lane, access_byte_len)?;
        self.lane_physical_byte_offset_from_relative(lane, relative)
    }

    fn lane_physical_byte_offset_from_relative(
        &self,
        lane: usize,
        relative: usize,
    ) -> Result<usize, EngineError> {
        let view_offset = match &self.state.buffer {
            RuntimeBuffer::Global(view) => view.byte_offset(),
            RuntimeBuffer::Shared { byte_offset, .. }
            | RuntimeBuffer::Local { byte_offset, .. }
            | RuntimeBuffer::Register { byte_offset, .. } => *byte_offset,
            RuntimeBuffer::RemoteShared { byte_offsets, .. } => usize::try_from(byte_offsets[lane])
                .map_err(|_| {
                    EngineError::out_of_bounds(format!(
                        "negative remote shared byte offset on lane {lane}"
                    ))
                })?,
            RuntimeBuffer::AccessView { .. } => {
                return Err(EngineError::message(
                    "internal error: PhysicalPtr retained an unpeeled access view",
                ));
            }
            RuntimeBuffer::LaneSelected { buffers } => {
                return runtime_buffer_view_offset_at(&buffers[lane], lane, relative);
            }
            RuntimeBuffer::Tmem { .. } => {
                return Err(EngineError::message(
                    "generic physical pointers cannot address tensor memory",
                ));
            }
        };
        view_offset
            .checked_add(relative)
            .ok_or_else(|| EngineError::out_of_bounds("physical pointer byte offset overflow"))
    }

    /// Resolve one load's complete active-lane contract in a single pass.
    pub(crate) fn resolve_load_byte_offsets(
        &self,
        ptx_space: PtxStateSpace,
        mask: WarpMask,
        access_byte_len: usize,
    ) -> Result<WarpValue<usize>, EngineError> {
        if access_byte_len == 0 {
            return Err(EngineError::message("load access width must be nonzero"));
        }
        let mut byte_offsets = WarpValue::splat(0_usize);
        for lane in mask {
            let relative = self.lane_read_byte_offset(lane, access_byte_len)?;
            if !runtime_buffer_matches_ptx_space_at(&self.state.buffer, lane, ptx_space) {
                return Err(EngineError::message(format!(
                    "resolved address on active lane {lane} does not match PTX state space {ptx_space}"
                )));
            }
            let register = self.pointer_space_at_lane(lane)? == PointerSpace::Register;
            if !register
                && self.lane_physical_byte_offset_from_relative(lane, relative)? % access_byte_len
                    != 0
            {
                return Err(EngineError::message(format!(
                    "load requires {access_byte_len}-byte alignment on lane {lane}"
                )));
            }
            byte_offsets[lane] = relative;
        }
        Ok(byte_offsets)
    }

    pub fn lane_read_byte_offset(
        &self,
        lane: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, 0, access_byte_len, true, false)
    }

    pub fn lane_write_byte_offset(
        &self,
        lane: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, 0, access_byte_len, false, true)
    }

    pub fn lane_read_write_byte_offset(
        &self,
        lane: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, 0, access_byte_len, true, true)
    }

    pub fn lane_read_byte_offset_at(
        &self,
        lane: usize,
        byte_delta: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, byte_delta, access_byte_len, true, false)
    }

    pub fn lane_write_byte_offset_at(
        &self,
        lane: usize,
        byte_delta: usize,
        access_byte_len: usize,
    ) -> Result<usize, EngineError> {
        self.lane_byte_offset_at(lane, byte_delta, access_byte_len, false, true)
    }

    pub fn require_readable(&self, mask: WarpMask, operation: &str) -> Result<(), EngineError> {
        self.require_readable_labeled(mask, &crate::DiagnosticLabel::new(operation))
    }

    pub(crate) fn require_readable_labeled(
        &self,
        mask: WarpMask,
        operation: &crate::DiagnosticLabel,
    ) -> Result<(), EngineError> {
        for lane in mask {
            self.lane_read_byte_offset(lane, 0)
                .map_err(|error| operation.wrap_engine_error(error))?;
        }
        Ok(())
    }

    /// The space this pointer resolves to on one lane.
    ///
    /// Lane-selected pointers may branch into different spaces, which is legal
    /// for a generic access. Callers that need a per-lane decision use this;
    /// `pointer_space_for_mask` is the stricter "all active lanes agree" form.
    pub fn pointer_space_at_lane(&self, lane: usize) -> Result<PointerSpace, EngineError> {
        if self.state.integer_lanes.contains(lane) {
            return Ok(PointerSpace::Global);
        }
        runtime_buffer_pointer_space_at(&self.state.buffer, lane)
    }

    pub(crate) fn is_in_ptx_address_space(
        &self,
        context: &WarpContext,
        lane: usize,
        space: PtxStateSpace,
    ) -> Result<bool, EngineError> {
        if self.state.integer_lanes.contains(lane) {
            return Err(self.unresolved_address_error(lane));
        }
        // A mapped address can still name this CTA. Instruction compatibility
        // distinguishes mapped pointers, but isspacep asks about the owner of
        // the variable, not the instruction used to obtain its address.
        if matches!(space, PtxStateSpace::Shared | PtxStateSpace::SharedCta) {
            if let RuntimeBuffer::RemoteShared { target_cta_ids, .. } =
                peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Address)?
            {
                return Ok(target_cta_ids[lane] == context.cta_id_in_cluster() as i64);
            }
        }
        Ok(runtime_buffer_matches_ptx_space_at(
            &self.state.buffer,
            lane,
            space,
        ))
    }

    pub fn pointer_space_for_mask(&self, mask: WarpMask) -> Result<PointerSpace, EngineError> {
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("cannot determine pointer space for an empty active mask")
        })?;
        let first = self.pointer_space_at_lane(first_lane)?;
        for lane in mask {
            if self.pointer_space_at_lane(lane)? != first {
                return Err(EngineError::message(format!(
                    "active lane-selected pointer branches use different memory spaces: lane {first_lane} is {first:?}, lane {lane} is {:?}",
                    self.pointer_space_at_lane(lane)?,
                )));
            }
        }
        Ok(first)
    }

    pub fn pointer_space(&self) -> Result<PointerSpace, EngineError> {
        self.pointer_space_for_mask(WarpMask::FULL)
    }

    pub fn require_ptx_space_for_mask(
        &self,
        space: PtxStateSpace,
        mask: WarpMask,
    ) -> Result<(), EngineError> {
        for lane in mask {
            self.require_ptx_address_at_lane(space, lane)?;
        }
        Ok(())
    }

    fn require_ptx_address_at_lane(
        &self,
        space: PtxStateSpace,
        lane: usize,
    ) -> Result<(), EngineError> {
        if self.state.integer_lanes.contains(lane) {
            return Err(self.unresolved_address_error(lane));
        }
        if !runtime_buffer_matches_ptx_space_at(&self.state.buffer, lane, space) {
            return Err(EngineError::message(format!(
                "active lane-selected buffers use different physical spaces: physical pointer provenance on active lane {lane} does not match PTX state space {space}"
            )));
        }
        Ok(())
    }

    pub fn require_ptx_space(&self, space: PtxStateSpace) -> Result<(), EngineError> {
        self.require_ptx_space_for_mask(space, WarpMask::FULL)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn runtime_view(
        &self,
        physical: &PhysicalMemory,
        expected_space: PointerSpace,
        view_byte_offset: usize,
        view_byte_len: usize,
        view_itemsize: usize,
        _context: &WarpContext,
        mask: WarpMask,
    ) -> Result<RuntimeBuffer, EngineError> {
        if self.state.pointee_itemsize != view_itemsize {
            return Err(EngineError::message(format!(
                "pointer pointee width {} does not match DeclBuffer itemsize {view_itemsize}",
                self.state.pointee_itemsize
            )));
        }
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("pointer-derived DeclBuffer has an empty active mask")
        })?;
        let relative_for = |lane: usize| -> Result<usize, EngineError> {
            self.lane_byte_offset_at(lane, view_byte_offset, view_byte_len, false, false)
        };
        let view = match (&self.state.buffer, expected_space) {
            (RuntimeBuffer::LaneSelected { .. }, _) => {
                return Err(EngineError::message(
                    "lane-selected pointer cannot define one uniform DeclBuffer view",
                ));
            }
            (RuntimeBuffer::Global(view), PointerSpace::Global) => {
                let relative = relative_for(first_lane)?;
                for lane in mask {
                    if relative_for(lane)? != relative {
                        return Err(EngineError::message(
                            "global DeclBuffer data pointer is lane-varying",
                        ));
                    }
                }
                Ok(RuntimeBuffer::Global(physical.global().subview(
                    view,
                    relative,
                    view_byte_len,
                )?))
            }
            (
                RuntimeBuffer::Shared {
                    allocations,
                    byte_offset,
                    backing_byte_len,
                    virtual_base,
                    ..
                },
                PointerSpace::Shared,
            ) => {
                let relative = self.lane_shared_view_byte_offset_at(
                    first_lane,
                    view_byte_offset,
                    view_byte_len,
                )?;
                for lane in mask {
                    if self.lane_shared_view_byte_offset_at(
                        lane,
                        view_byte_offset,
                        view_byte_len,
                    )? != relative
                    {
                        return Err(EngineError::message(
                            "local shared DeclBuffer data pointer is lane-varying",
                        ));
                    }
                }
                let absolute = byte_offset.checked_add(relative).ok_or_else(|| {
                    EngineError::out_of_bounds("shared DeclBuffer view offset overflow")
                })?;
                Ok(RuntimeBuffer::Shared {
                    allocations: allocations.clone(),
                    byte_offset: absolute,
                    byte_len: view_byte_len,
                    backing_byte_len: *backing_byte_len,
                    virtual_base: virtual_base.checked_add(relative).ok_or_else(|| {
                        EngineError::message("shared DeclBuffer virtual address overflow")
                    })?,
                })
            }
            (
                RuntimeBuffer::RemoteShared {
                    allocations,
                    byte_offsets,
                    target_cta_ids,
                    virtual_base,
                    ..
                },
                PointerSpace::Shared,
            ) => {
                let mut view_offsets = WarpValue::splat(0_i64);
                for lane in mask {
                    let relative = relative_for(lane)?;
                    let absolute = usize::try_from(byte_offsets[lane])
                        .map_err(|_| {
                            EngineError::out_of_bounds("negative mapped shared byte offset")
                        })?
                        .checked_add(relative)
                        .ok_or_else(|| {
                            EngineError::out_of_bounds("remote shared DeclBuffer offset overflow")
                        })?;
                    view_offsets[lane] = i64::try_from(absolute).map_err(|_| {
                        EngineError::out_of_bounds("remote shared DeclBuffer offset exceeds i64")
                    })?;
                }
                Ok(RuntimeBuffer::RemoteShared {
                    allocations: allocations.clone(),
                    byte_offsets: view_offsets,
                    byte_len: view_byte_len,
                    target_cta_ids: target_cta_ids.clone(),
                    virtual_base: *virtual_base,
                })
            }
            (
                RuntimeBuffer::Local {
                    allocations,
                    byte_offset,
                    ..
                },
                PointerSpace::Local,
            ) => {
                let relative = relative_for(first_lane)?;
                for lane in mask {
                    if relative_for(lane)? != relative {
                        return Err(EngineError::message(
                            "local DeclBuffer data pointer is lane-varying",
                        ));
                    }
                }
                Ok(RuntimeBuffer::Local {
                    allocations: allocations.clone(),
                    byte_offset: byte_offset.checked_add(relative).ok_or_else(|| {
                        EngineError::out_of_bounds("local DeclBuffer view offset overflow")
                    })?,
                    byte_len: view_byte_len,
                })
            }
            (
                RuntimeBuffer::Register {
                    allocations,
                    byte_offset,
                    ..
                },
                PointerSpace::Register,
            ) => {
                let relative = relative_for(first_lane)?;
                for lane in mask {
                    if relative_for(lane)? != relative {
                        return Err(EngineError::message(
                            "register DeclBuffer data pointer is lane-varying",
                        ));
                    }
                }
                Ok(RuntimeBuffer::Register {
                    allocations: allocations.clone(),
                    byte_offset: byte_offset.checked_add(relative).ok_or_else(|| {
                        EngineError::out_of_bounds("register DeclBuffer view offset overflow")
                    })?,
                    byte_len: view_byte_len,
                })
            }
            _ => Err(EngineError::message(
                "DeclBuffer memory space disagrees with its resolved address",
            )),
        }?;
        Ok(RuntimeBuffer::AccessView {
            buffer: Arc::new(view),
            readable_lanes: self.state.readable_lanes & mask,
            writable_lanes: self.state.writable_lanes & mask,
        })
    }

    /// Observe a bound global address as an ordinary integer value.
    pub fn global_addresses_u64(&self, mask: WarpMask) -> Result<WarpValue<u64>, EngineError> {
        let mut addresses = WarpValue::splat(0_u64);
        for lane in mask {
            if self.state.integer_lanes.contains(lane) {
                addresses[lane] = self.state.byte_offsets[lane] as u64;
                continue;
            }
            let RuntimeBuffer::Global(view) =
                peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Address)?
            else {
                return Err(EngineError::message(
                    "integer address observation requires a global pointer",
                ));
            };
            let base = view.observed_allocation_address().ok_or_else(|| {
                EngineError::analysis_incomplete("global_address_without_binding")
            })?;
            let relative = self.raw_relative_byte_offset(lane)?;
            addresses[lane] = base
                .wrapping_add(view.byte_offset() as u64)
                .wrapping_add(relative as u64);
        }
        Ok(addresses)
    }

    pub fn shared_byte_addresses_u32(
        &self,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<WarpValue<u32>, EngineError> {
        let mut addresses = WarpValue::splat(0_u32);
        for lane in mask {
            addresses[lane] = self.shared_byte_address_u32_at(context, lane)?;
        }
        Ok(addresses)
    }

    fn shared_byte_address_u32_at(
        &self,
        context: &WarpContext,
        lane: usize,
    ) -> Result<u32, EngineError> {
        let leaf = peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Address)?;
        let (address_base, rank) = match leaf {
            RuntimeBuffer::Shared { virtual_base, .. } => {
                (*virtual_base, context.cta_id_in_cluster())
            }
            RuntimeBuffer::RemoteShared {
                virtual_base,
                byte_offsets,
                target_cta_ids,
                ..
            } => (
                virtual_base
                    .checked_add(usize::try_from(byte_offsets[lane]).map_err(|_| {
                        EngineError::out_of_bounds(format!(
                            "negative remote shared byte offset on lane {lane}"
                        ))
                    })?)
                    .ok_or_else(|| {
                        EngineError::message("remote shared virtual address overflow")
                    })?,
                usize::try_from(target_cta_ids[lane]).map_err(|_| {
                    EngineError::message(format!("negative cluster shared CTA rank on lane {lane}"))
                })?,
            ),
            _ => {
                return Err(EngineError::message(
                    "shared address conversion requires a physical shared-memory pointer",
                ));
            }
        };
        if rank >= context.topology().ctas_per_cluster() {
            return Err(EngineError::message(format!(
                "cluster shared CTA rank {rank} is outside cluster size {} on lane {lane}",
                context.topology().ctas_per_cluster()
            )));
        }
        let byte_offset = u32::try_from(address_base)
            .map_err(|_| EngineError::message("shared address byte offset exceeds uint32"))?;
        let encoded = crate::instruction_codec::encode_shared_address(
            byte_offset,
            u32::try_from(rank)
                .map_err(|_| EngineError::message("shared CTA rank exceeds uint32"))?,
        )
        .ok_or_else(|| {
            EngineError::message(format!(
                "shared address byte offset {byte_offset:#x} or CTA rank {rank} exceeds the modeled cluster address fields"
            ))
        })?;
        Ok(encoded.wrapping_add(self.raw_relative_byte_offset(lane)? as u32))
    }

    /// Observe a CUDA generic address value. Shared-memory addresses use the
    /// hardware generic shared window; global addresses retain their bound
    /// invocation address. No allocation identity accompanies the result.
    pub fn generic_addresses_u64(
        &self,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<WarpValue<u64>, EngineError> {
        let mut addresses = WarpValue::splat(0_u64);
        for lane in mask {
            if self.state.integer_lanes.contains(lane) {
                addresses[lane] = self.raw_relative_byte_offset(lane)? as u64;
                continue;
            }
            let leaf = peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Address)?;
            match leaf {
                RuntimeBuffer::Shared { .. } | RuntimeBuffer::RemoteShared { .. } => {
                    addresses[lane] = crate::instruction_codec::generic_shared_address(
                        self.shared_byte_address_u32_at(context, lane)?,
                    );
                }
                RuntimeBuffer::Global(view) => {
                    let base = view.observed_allocation_address().ok_or_else(|| {
                        EngineError::message("global pointer has no bound invocation address")
                    })?;
                    let relative = self.raw_relative_byte_offset(lane)?;
                    addresses[lane] = base
                        .wrapping_add(view.byte_offset() as u64)
                        .wrapping_add(relative as u64);
                }
                RuntimeBuffer::Local { .. } | RuntimeBuffer::Register { .. } => {
                    return Err(EngineError::analysis_incomplete(
                        "generic_local_address_not_modeled",
                    ));
                }
                RuntimeBuffer::Tmem { .. }
                | RuntimeBuffer::AccessView { .. }
                | RuntimeBuffer::LaneSelected { .. } => {
                    return Err(EngineError::message(
                        "generic addresses cannot name this physical memory space",
                    ));
                }
            }
        }
        Ok(addresses)
    }

    pub fn resolve_uniform_shared_address_u32(
        &self,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<u32, EngineError> {
        let addresses = self.shared_byte_addresses_u32(context, mask)?;
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("cannot resolve a shared address for an empty mask")
        })?;
        let address = addresses[first_lane];
        for lane in mask {
            if addresses[lane] != address {
                return Err(EngineError::message(format!(
                    "shared address is lane-varying: lane {first_lane} has {address}, lane {lane} has {}",
                    addresses[lane]
                )));
            }
        }
        Ok(address)
    }

    pub fn map_shared_rank(
        &self,
        context: &WarpContext,
        ranks: &WarpValue<i64>,
        mask: WarpMask,
    ) -> Result<Self, EngineError> {
        let RuntimeBuffer::Shared {
            allocations,
            backing_byte_len,
            virtual_base,
            ..
        } = &self.state.buffer
        else {
            return Err(EngineError::message(
                "ptx.mapa source must address local physical shared memory",
            ));
        };
        let mut pointer_offsets = WarpValue::splat(0_i128);
        for lane in mask {
            let target = usize::try_from(ranks[lane]).map_err(|_| {
                EngineError::message(format!(
                    "negative mapa CTA rank {} on lane {lane}",
                    ranks[lane]
                ))
            })?;
            if target >= context.topology().ctas_per_cluster() {
                return Err(EngineError::message(format!(
                    "mapa CTA rank {target} is outside cluster size {} on lane {lane}",
                    context.topology().ctas_per_cluster()
                )));
            }
            pointer_offsets[lane] = self.lane_address_byte_offset(lane, 0)? as i128;
        }
        Ok(Self {
            state: Arc::new(PhysicalPtrState {
                buffer: RuntimeBuffer::RemoteShared {
                    allocations: allocations.clone(),
                    byte_offsets: WarpValue::splat(0_i64),
                    byte_len: *backing_byte_len,
                    target_cta_ids: ranks.clone(),
                    virtual_base: *virtual_base,
                },
                element_indices: WarpValue::splat(0_i64),
                itemsize: 1,
                byte_offsets: pointer_offsets,
                integer_lanes: self.state.integer_lanes,
                pointee_itemsize: self.state.pointee_itemsize,
                readable_lanes: self.state.readable_lanes,
                writable_lanes: self.state.writable_lanes,
                access_view_permissions: self.state.access_view_permissions,
                bounded_lanes: self.state.bounded_lanes,
                range_starts: self.state.range_starts.clone(),
                range_ends: self.state.range_ends.clone(),
            }),
        })
    }

    /// Return the physical cluster-CTA target selected by each active lane.
    ///
    /// Cluster-scoped mbarrier instructions carry only an address operand;
    /// `mapa` has already encoded its target in the resulting physical
    /// pointer.  Recover that target at the engine ABI instead of requiring
    /// generated code to preserve a second copy of the rank alongside every
    /// pointer-derived view.
    pub fn shared_target_cta_ranks(
        &self,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<WarpValue<i64>, EngineError> {
        let local_rank = i64::try_from(context.cta_id_in_cluster())
            .map_err(|_| EngineError::message("cluster CTA rank exceeds i64"))?;
        let mut ranks = WarpValue::splat(local_rank);
        for lane in mask {
            match peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Address)? {
                RuntimeBuffer::Shared { .. } => {}
                RuntimeBuffer::RemoteShared { target_cta_ids, .. } => {
                    ranks[lane] = target_cta_ids[lane];
                }
                _ => {
                    return Err(EngineError::message(
                        "cluster shared target requires a physical shared-memory pointer",
                    ));
                }
            }
        }
        Ok(ranks)
    }

    pub fn resolve_uniform(
        &self,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<PhysicalAddress, EngineError> {
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("cannot resolve a physical pointer for an empty active mask")
        })?;
        let relative = self.lane_address_byte_offset(first_lane, 0)?;
        for lane in mask {
            let lane_relative = self.lane_address_byte_offset(lane, 0)?;
            if lane_relative != relative {
                return Err(EngineError::message(format!(
                    "physical pointer is lane-varying: lane {first_lane} has byte offset {relative}, lane {lane} has {lane_relative}"
                )));
            }
        }
        match &self.state.buffer {
            RuntimeBuffer::LaneSelected { .. } => {
                let first = self
                    .state
                    .buffer
                    .physical_address_at(context, first_lane, relative, None)?;
                for lane in mask {
                    let lane_relative = self.lane_address_byte_offset(lane, 0)?;
                    let address = self.state.buffer.physical_address_at(
                        context,
                        lane,
                        lane_relative,
                        None,
                    )?;
                    if address != first {
                        return Err(EngineError::message(
                            "lane-selected physical pointer is not uniform",
                        ));
                    }
                }
                Ok(first)
            }
            RuntimeBuffer::AccessView { .. } => Err(EngineError::message(
                "internal error: PhysicalPtr retained an unpeeled access view",
            )),
            RuntimeBuffer::Global(view) => Ok(PhysicalAddress {
                allocation_id: view.allocation().as_u64(),
                byte_offset: view.byte_offset().checked_add(relative).ok_or_else(|| {
                    EngineError::message("global physical pointer offset overflow")
                })?,
            }),
            RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                ..
            } => {
                let allocation = allocations.get(context.global_cta_id()).ok_or_else(|| {
                    EngineError::message("shared-memory CTA allocation is missing")
                })?;
                Ok(PhysicalAddress {
                    allocation_id: allocation.allocation().as_u64(),
                    byte_offset: byte_offset.checked_add(relative).ok_or_else(|| {
                        EngineError::message("shared physical pointer offset overflow")
                    })?,
                })
            }
            RuntimeBuffer::RemoteShared {
                allocations,
                byte_offsets,
                target_cta_ids,
                ..
            } => {
                let target_local = usize::try_from(target_cta_ids[first_lane])
                    .map_err(|_| EngineError::message("negative mapped shared CTA id"))?;
                let base_offset = byte_offsets[first_lane];
                for lane in mask {
                    if target_cta_ids[lane] != target_cta_ids[first_lane]
                        || byte_offsets[lane] != base_offset
                    {
                        return Err(EngineError::message(
                            "mapped physical pointer is lane-varying",
                        ));
                    }
                }
                let topology = context.topology();
                if target_local >= topology.ctas_per_cluster() {
                    return Err(EngineError::message(
                        "mapped shared CTA id is outside cluster",
                    ));
                }
                let target_global = context
                    .cluster_id()
                    .checked_mul(topology.ctas_per_cluster())
                    .and_then(|base| base.checked_add(target_local))
                    .ok_or_else(|| EngineError::message("mapped shared CTA index overflow"))?;
                let allocation = allocations.get(target_global).ok_or_else(|| {
                    EngineError::message("mapped shared-memory CTA allocation is missing")
                })?;
                let absolute = usize::try_from(base_offset)
                    .map_err(|_| EngineError::out_of_bounds("negative mapped shared byte offset"))?
                    .checked_add(relative)
                    .ok_or_else(|| {
                        EngineError::out_of_bounds("mapped shared pointer offset overflow")
                    })?;
                Ok(PhysicalAddress {
                    allocation_id: allocation.allocation().as_u64(),
                    byte_offset: absolute,
                })
            }
            RuntimeBuffer::Local {
                allocations,
                byte_offset,
                ..
            }
            | RuntimeBuffer::Register {
                allocations,
                byte_offset,
                ..
            } => {
                if mask.len() != 1 {
                    return Err(EngineError::message(
                        "warp-private pointer identity requires exactly one active lane",
                    ));
                }
                let allocation = allocations
                    .get(context.global_warp_id())
                    .ok_or_else(|| EngineError::message("warp-private allocation is missing"))?;
                let lane_base = first_lane
                    .checked_mul(allocation.bytes_per_lane())
                    .ok_or_else(|| EngineError::message("lane pointer offset overflow"))?;
                let byte_offset = lane_base
                    .checked_add(*byte_offset)
                    .and_then(|value| value.checked_add(relative))
                    .ok_or_else(|| EngineError::message("warp-private pointer offset overflow"))?;
                Ok(PhysicalAddress {
                    allocation_id: allocation.allocation().as_u64(),
                    byte_offset,
                })
            }
            RuntimeBuffer::Tmem { .. } => Err(EngineError::message(
                "generic physical pointers cannot address TLane/TCol TMEM views",
            )),
        }
    }

    pub(crate) fn resolve_lane(
        &self,
        context: &WarpContext,
        lane: usize,
    ) -> Result<PhysicalAddress, EngineError> {
        let relative = self.lane_address_byte_offset(lane, 0)?;
        self.state
            .buffer
            .physical_address_at(context, lane, relative, None)
    }

    pub(crate) fn broadcast_lane(&self, lane: usize) -> Self {
        Self {
            state: Arc::new(PhysicalPtrState {
                buffer: runtime_buffer_broadcast_lane(&self.state.buffer, lane),
                element_indices: WarpValue::splat(self.state.element_indices[lane]),
                itemsize: self.state.itemsize,
                byte_offsets: WarpValue::splat(self.state.byte_offsets[lane]),
                integer_lanes: if self.state.integer_lanes.contains(lane) {
                    WarpMask::FULL
                } else {
                    WarpMask::EMPTY
                },
                pointee_itemsize: self.state.pointee_itemsize,
                readable_lanes: if self.state.readable_lanes.contains(lane) {
                    WarpMask::FULL
                } else {
                    WarpMask::EMPTY
                },
                writable_lanes: if self.state.writable_lanes.contains(lane) {
                    WarpMask::FULL
                } else {
                    WarpMask::EMPTY
                },
                access_view_permissions: self.state.access_view_permissions,
                bounded_lanes: if self.state.bounded_lanes.contains(lane) {
                    WarpMask::FULL
                } else {
                    WarpMask::EMPTY
                },
                range_starts: WarpValue::splat(self.state.range_starts[lane]),
                range_ends: WarpValue::splat(self.state.range_ends[lane]),
            }),
        }
    }

    pub(crate) fn observe_readonly_proxy(
        &self,
        physical: &PhysicalMemory,
        mask: WarpMask,
        byte_len: usize,
    ) -> Result<(), EngineError> {
        let offsets = self.resolve_load_byte_offsets(PtxStateSpace::Global, mask, byte_len)?;
        for lane in mask {
            let leaf = peel_runtime_buffer_wrappers(&self.state.buffer, lane, ViewAccess::Read)?;
            let RuntimeBuffer::Global(view) = leaf else {
                return Err(EngineError::message(
                    "readonly-proxy load requires global memory",
                ));
            };
            physical
                .global()
                .observe_readonly_proxy(view, offsets[lane], byte_len)?;
        }
        Ok(())
    }

    pub fn resolve_uniform_global_remainder_view(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
    ) -> Result<BufferView, EngineError> {
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("cannot resolve a global pointer for an empty active mask")
        })?;
        let address = self.resolve_uniform(context, mask)?;
        let leaf =
            peel_runtime_buffer_wrappers(&self.state.buffer, first_lane, ViewAccess::Address)?;
        let RuntimeBuffer::Global(view) = leaf else {
            return Err(EngineError::message(
                "TensorMap global-address replacement requires a global physical pointer",
            ));
        };
        if view.allocation().as_u64() != address.allocation_id() {
            return Err(EngineError::message(
                "global pointer allocation identity changed during TensorMap replacement",
            ));
        }
        let full = view.full_allocation_view();
        let byte_len = full
            .byte_len()
            .checked_sub(address.byte_offset())
            .ok_or_else(|| {
                EngineError::out_of_bounds("TensorMap global address is out of bounds")
            })?;
        physical
            .global()
            .subview(&full, address.byte_offset(), byte_len)
            .map_err(EngineError::from)
    }

    pub fn resolve_shared_barrier(
        &self,
        context: &WarpContext,
        mask: WarpMask,
        target_cta_id_in_cluster: Option<usize>,
    ) -> Result<PhysicalBarrierId, EngineError> {
        self.resolve_shared_barrier_impl(context, mask, target_cta_id_in_cluster, false, true, true)
    }

    pub fn resolve_shared_barrier_read(
        &self,
        context: &WarpContext,
        mask: WarpMask,
        target_cta_id_in_cluster: Option<usize>,
    ) -> Result<PhysicalBarrierId, EngineError> {
        self.resolve_shared_barrier_impl(
            context,
            mask,
            target_cta_id_in_cluster,
            false,
            true,
            false,
        )
    }

    pub fn resolve_shared_barrier_write(
        &self,
        context: &WarpContext,
        mask: WarpMask,
        target_cta_id_in_cluster: Option<usize>,
    ) -> Result<PhysicalBarrierId, EngineError> {
        self.resolve_shared_barrier_impl(
            context,
            mask,
            target_cta_id_in_cluster,
            false,
            false,
            true,
        )
    }

    pub fn resolve_shared_barrier_multicast(
        &self,
        context: &WarpContext,
        mask: WarpMask,
        target_cta_id_in_cluster: usize,
    ) -> Result<PhysicalBarrierId, EngineError> {
        self.resolve_shared_barrier_impl(
            context,
            mask,
            Some(target_cta_id_in_cluster),
            true,
            true,
            true,
        )
    }

    fn resolve_shared_barrier_impl(
        &self,
        context: &WarpContext,
        mask: WarpMask,
        target_cta_id_in_cluster: Option<usize>,
        multicast_same_offset: bool,
        require_read: bool,
        require_write: bool,
    ) -> Result<PhysicalBarrierId, EngineError> {
        let first_lane = mask.first_active().ok_or_else(|| {
            EngineError::message("cannot resolve an mbarrier pointer for an empty active mask")
        })?;
        let relative = self.lane_byte_offset_at(first_lane, 0, 8, require_read, require_write)?;
        for lane in mask {
            let lane_relative =
                self.lane_byte_offset_at(lane, 0, 8, require_read, require_write)?;
            if lane_relative != relative {
                return Err(EngineError::message(format!(
                    "mbarrier pointer is lane-varying: lane {first_lane} has byte offset {relative}, lane {lane} has {lane_relative}"
                )));
            }
        }
        let topology = context.topology();
        let (allocations, base_offset, mapped_target) = match &self.state.buffer {
            RuntimeBuffer::LaneSelected { .. } => {
                return Err(EngineError::message(
                    "lane-selected pointer cannot name one physical mbarrier",
                ));
            }
            RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                ..
            } => (allocations, *byte_offset, None),
            RuntimeBuffer::RemoteShared {
                allocations,
                byte_offsets,
                target_cta_ids,
                ..
            } => {
                let base = usize::try_from(byte_offsets[first_lane])
                    .map_err(|_| EngineError::out_of_bounds("negative mapped mbarrier offset"))?;
                let target = usize::try_from(target_cta_ids[first_lane])
                    .map_err(|_| EngineError::message("negative mapped mbarrier CTA id"))?;
                for lane in mask {
                    if byte_offsets[lane] != byte_offsets[first_lane]
                        || target_cta_ids[lane] != target_cta_ids[first_lane]
                    {
                        return Err(EngineError::message(
                            "mapped mbarrier pointer is lane-varying",
                        ));
                    }
                }
                (allocations, base, Some(target))
            }
            _ => {
                return Err(EngineError::message(
                    "mbarrier pointer must address physical shared memory",
                ));
            }
        };
        if !multicast_same_offset
            && mapped_target.is_some()
            && target_cta_id_in_cluster.is_some()
            && mapped_target != target_cta_id_in_cluster
        {
            return Err(EngineError::message(
                "mbarrier pointer target disagrees with explicit remote CTA id",
            ));
        }
        let target_local = if multicast_same_offset {
            target_cta_id_in_cluster.unwrap_or(context.cta_id_in_cluster())
        } else {
            mapped_target
                .or(target_cta_id_in_cluster)
                .unwrap_or(context.cta_id_in_cluster())
        };
        if target_local >= topology.ctas_per_cluster() {
            return Err(EngineError::message(format!(
                "remote mbarrier CTA {target_local} is outside cluster size {}",
                topology.ctas_per_cluster()
            )));
        }
        let target_global = context
            .cluster_id()
            .checked_mul(topology.ctas_per_cluster())
            .and_then(|base| base.checked_add(target_local))
            .ok_or_else(|| EngineError::message("remote mbarrier CTA index overflow"))?;
        let allocation = allocations.get(target_global).ok_or_else(|| {
            EngineError::message("target shared-memory CTA allocation is missing")
        })?;
        let absolute = base_offset
            .checked_add(relative)
            .ok_or_else(|| EngineError::message("mbarrier pointer offset overflow"))?;
        if absolute % 8 != 0 {
            return Err(EngineError::message(format!(
                "mbarrier pointer byte offset {absolute} is not 8-byte aligned"
            )));
        }
        Ok(PhysicalBarrierId::new(
            allocation.allocation().as_u64(),
            absolute,
            target_global,
        ))
    }

    fn same_pointer_view(&self, other: &Self) -> bool {
        let same_buffer = match (&self.state.buffer, &other.state.buffer) {
            (RuntimeBuffer::Global(left), RuntimeBuffer::Global(right)) => left == right,
            (
                RuntimeBuffer::Shared {
                    allocations: left_allocations,
                    byte_offset: left_offset,
                    byte_len: left_len,
                    virtual_base: left_virtual,
                    ..
                },
                RuntimeBuffer::Shared {
                    allocations: right_allocations,
                    byte_offset: right_offset,
                    byte_len: right_len,
                    virtual_base: right_virtual,
                    ..
                },
            ) => {
                Arc::ptr_eq(left_allocations, right_allocations)
                    && left_offset == right_offset
                    && left_len == right_len
                    && left_virtual == right_virtual
            }
            (
                RuntimeBuffer::RemoteShared {
                    allocations: left_allocations,
                    byte_len: left_len,
                    virtual_base: left_virtual_base,
                    ..
                },
                RuntimeBuffer::RemoteShared {
                    allocations: right_allocations,
                    byte_len: right_len,
                    virtual_base: right_virtual_base,
                    ..
                },
            ) => {
                Arc::ptr_eq(left_allocations, right_allocations)
                    && left_len == right_len
                    && left_virtual_base == right_virtual_base
            }
            (
                RuntimeBuffer::Local {
                    allocations: left_allocations,
                    byte_offset: left_offset,
                    byte_len: left_len,
                },
                RuntimeBuffer::Local {
                    allocations: right_allocations,
                    byte_offset: right_offset,
                    byte_len: right_len,
                },
            )
            | (
                RuntimeBuffer::Register {
                    allocations: left_allocations,
                    byte_offset: left_offset,
                    byte_len: left_len,
                },
                RuntimeBuffer::Register {
                    allocations: right_allocations,
                    byte_offset: right_offset,
                    byte_len: right_len,
                },
            ) => {
                Arc::ptr_eq(left_allocations, right_allocations)
                    && left_offset == right_offset
                    && left_len == right_len
            }
            _ => false,
        };
        same_buffer
            && self.state.itemsize == other.state.itemsize
            && self.state.pointee_itemsize == other.state.pointee_itemsize
    }
}

pub struct PhysicalPtrSlot {
    value: Option<PhysicalPtr>,
    initialized: WarpMask,
}

impl PhysicalPtrSlot {
    pub fn new() -> Self {
        Self {
            value: None,
            initialized: WarpMask::EMPTY,
        }
    }

    pub fn store(&mut self, value: &PhysicalPtr, mask: WarpMask) -> Result<(), EngineError> {
        if mask.is_empty() {
            return Ok(());
        }
        // A complete overwrite of the initialized lanes does not merge views.
        // In particular, raw byte addresses and typed pointers may reuse the
        // same PTX address register without retaining the previous width.
        if (self.initialized - mask).is_empty() {
            self.value = Some(value.clone());
            self.initialized = mask;
            return Ok(());
        }
        match &mut self.value {
            None => self.value = Some(value.clone()),
            Some(current) => {
                let incoming_integer = (mask - value.state.integer_lanes).is_empty();
                let retained_integer =
                    (self.initialized - mask - current.state.integer_lanes).is_empty();
                if !incoming_integer && retained_integer {
                    let state = Arc::make_mut(&mut current.state);
                    state.itemsize = value.state.itemsize;
                    state.pointee_itemsize = value.state.pointee_itemsize;
                }
                if !incoming_integer
                    && (current.state.itemsize != value.state.itemsize
                        || current.state.pointee_itemsize != value.state.pointee_itemsize)
                {
                    return Err(EngineError::message(
                        "PhysicalPtrSlot cannot merge pointers with different element widths",
                    ));
                }
                let same_pointer_view = current.same_pointer_view(value);
                let current_state = Arc::make_mut(&mut current.state);
                if !same_pointer_view {
                    let mut selected = runtime_buffer_lane_selection(&current_state.buffer);
                    let incoming = runtime_buffer_lane_selection(&value.state.buffer);
                    selected.masked_assign(mask, &incoming);
                    current_state.buffer = RuntimeBuffer::LaneSelected { buffers: selected };
                }
                current_state
                    .element_indices
                    .masked_assign(mask, &value.state.element_indices);
                current_state
                    .byte_offsets
                    .masked_assign(mask, &value.state.byte_offsets);
                if let (
                    RuntimeBuffer::RemoteShared {
                        byte_offsets: current_offsets,
                        target_cta_ids: current_targets,
                        ..
                    },
                    RuntimeBuffer::RemoteShared {
                        byte_offsets: value_offsets,
                        target_cta_ids: value_targets,
                        ..
                    },
                ) = (&mut current_state.buffer, &value.state.buffer)
                {
                    current_offsets.masked_assign(mask, value_offsets);
                    current_targets.masked_assign(mask, value_targets);
                }
                current_state.integer_lanes =
                    (current_state.integer_lanes - mask) | (value.state.integer_lanes & mask);
                current_state.readable_lanes =
                    (current_state.readable_lanes - mask) | (value.state.readable_lanes & mask);
                current_state.writable_lanes =
                    (current_state.writable_lanes - mask) | (value.state.writable_lanes & mask);
                current_state.bounded_lanes =
                    (current_state.bounded_lanes - mask) | (value.state.bounded_lanes & mask);
                current_state
                    .range_starts
                    .masked_assign(mask, &value.state.range_starts);
                current_state
                    .range_ends
                    .masked_assign(mask, &value.state.range_ends);
            }
        }
        self.initialized |= mask;
        Ok(())
    }

    pub fn load(&self, mask: WarpMask) -> Result<PhysicalPtr, EngineError> {
        let missing = mask - self.initialized;
        if !missing.is_empty() {
            return Err(EngineError::message(format!(
                "PhysicalPtrSlot load uses uninitialized lanes {:?}",
                missing.iter().collect::<Vec<_>>()
            )));
        }
        self.value
            .clone()
            .ok_or_else(|| EngineError::message("PhysicalPtrSlot is uninitialized"))
    }
}

impl Default for PhysicalPtrSlot {
    fn default() -> Self {
        Self::new()
    }
}

pub fn runtime_buffer_byte_len(buffer: &RuntimeBuffer) -> usize {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => runtime_buffer_byte_len(buffer),
        RuntimeBuffer::LaneSelected { buffers } => buffers
            .lanes()
            .iter()
            .map(|buffer| runtime_buffer_byte_len(buffer))
            .min()
            .unwrap_or(0),
        RuntimeBuffer::Global(view) => view.byte_len(),
        RuntimeBuffer::Shared { byte_len, .. }
        | RuntimeBuffer::RemoteShared { byte_len, .. }
        | RuntimeBuffer::Local { byte_len, .. }
        | RuntimeBuffer::Register { byte_len, .. } => *byte_len,
        RuntimeBuffer::Tmem {
            lane_span,
            tcol_span_elements,
            itemsize,
            ..
        } => lane_span
            .saturating_mul(*tcol_span_elements)
            .saturating_mul(*itemsize),
    }
}

pub fn physical_ptr_from_shared_addresses_u32(
    candidates: &[RuntimeBuffer],
    addresses: &WarpValue<u32>,
    context: &WarpContext,
    mask: WarpMask,
) -> Result<PhysicalPtr, EngineError> {
    if mask.is_empty() {
        let candidate = candidates
            .iter()
            .find(|candidate| {
                matches!(
                    runtime_buffer_base(candidate),
                    RuntimeBuffer::Shared { .. } | RuntimeBuffer::RemoteShared { .. }
                )
            })
            .ok_or_else(|| {
                EngineError::message(
                    "raw shared address conversion has no declared shared-memory candidate",
                )
            })?;
        return Ok(PhysicalPtr::new(
            runtime_buffer_base(candidate).clone(),
            WarpValue::splat(0_i64),
            1,
        ));
    }
    let mut selected: Option<(Arc<Vec<SharedAllocation>>, usize, usize)> = None;
    let mut byte_offsets = WarpValue::splat(0_i64);
    let mut target_cta_ids = WarpValue::splat(0_i64);

    for lane in mask {
        let target_rank = usize::try_from(crate::instruction_codec::shared_address_cta_rank(
            addresses[lane],
        ))
        .map_err(|_| EngineError::message("raw shared CTA rank conversion failed"))?;
        if target_rank >= context.topology().ctas_per_cluster() {
            return Err(EngineError::message(format!(
                "raw shared address {:#010x} targets CTA rank {target_rank}, outside cluster size {} on lane {lane}",
                addresses[lane],
                context.topology().ctas_per_cluster(),
            )));
        }
        target_cta_ids[lane] = i64::try_from(target_rank)
            .map_err(|_| EngineError::message("raw shared CTA rank exceeds int64"))?;
        let address = usize::try_from(crate::instruction_codec::shared_address_byte_offset(
            addresses[lane],
        ))
        .map_err(|_| EngineError::message("raw shared address conversion failed"))?;
        let mut matched = false;
        for candidate in candidates {
            let RuntimeBuffer::Shared {
                allocations,
                byte_offset,
                byte_len,
                backing_byte_len,
                virtual_base,
            } = runtime_buffer_base(candidate)
            else {
                continue;
            };
            let view_end = virtual_base
                .checked_add(*byte_len)
                .ok_or_else(|| EngineError::message("raw shared view range overflow"))?;
            if address < *virtual_base || address >= view_end {
                continue;
            }
            let backing_virtual_base = virtual_base.checked_sub(*byte_offset).ok_or_else(|| {
                EngineError::message("raw shared view precedes its backing virtual base")
            })?;
            if let Some((root_allocations, root_base, root_len)) = &selected {
                if !Arc::ptr_eq(root_allocations, allocations)
                    || *root_base != backing_virtual_base
                    || *root_len != *backing_byte_len
                {
                    return Err(EngineError::message(format!(
                        "raw shared address {address} ambiguously names multiple backings"
                    )));
                }
            } else {
                selected = Some((
                    Arc::clone(allocations),
                    backing_virtual_base,
                    *backing_byte_len,
                ));
            }
            let relative = address.checked_sub(backing_virtual_base).ok_or_else(|| {
                EngineError::message("raw shared address precedes its backing")
            })?;
            byte_offsets[lane] = i64::try_from(relative)
                .map_err(|_| EngineError::message("raw shared address offset exceeds int64"))?;
            matched = true;
        }
        if !matched {
            return Err(EngineError::out_of_bounds(format!(
                "raw shared address {address} exceeds allocation or does not name a declared shared-memory view"
            )));
        }
    }

    let Some((allocations, virtual_base, byte_len)) = selected else {
        return Err(EngineError::message(
            "raw shared address conversion has no active lanes",
        ));
    };
    let all_local = mask.into_iter().all(|lane| {
        usize::try_from(target_cta_ids[lane]).ok() == Some(context.cta_id_in_cluster())
    });
    let (backing, indices) = if all_local {
        (
            RuntimeBuffer::Shared {
                allocations,
                byte_offset: 0,
                byte_len,
                backing_byte_len: byte_len,
                virtual_base,
            },
            byte_offsets,
        )
    } else {
        (
            RuntimeBuffer::RemoteShared {
                allocations,
                byte_offsets,
                byte_len,
                target_cta_ids,
                virtual_base,
            },
            WarpValue::splat(0_i64),
        )
    };
    // Integer address bits do not carry a source view's bounds or access
    // permissions. The final instruction validates its complete access
    // against the resolved physical backing.
    Ok(PhysicalPtr::new(backing, indices, 1))
}

/// Resolve ordinary CUDA generic address values against the runtime memory
/// map. Global bindings are owned by the global arena; shared addresses use
/// the engine's invocation-local virtual codec.
/// No producer/provenance metadata participates in resolution.
pub fn physical_ptr_from_generic_addresses_u64(
    physical: &PhysicalMemory,
    candidates: &[RuntimeBuffer],
    addresses: &WarpValue<u64>,
    context: &WarpContext,
    mask: WarpMask,
) -> Result<PhysicalPtr, EngineError> {
    physical_ptr_from_addresses_u64(physical, candidates, addresses, context, mask, false)
}

/// Integer TIR views also accept zero-extended shared offsets. A nominal
/// DeclBuffer scope must not override an address already bound to an owner.
pub fn physical_ptr_from_view_addresses_u64(
    physical: &PhysicalMemory,
    candidates: &[RuntimeBuffer],
    addresses: &WarpValue<u64>,
    context: &WarpContext,
    mask: WarpMask,
) -> Result<PhysicalPtr, EngineError> {
    physical_ptr_from_addresses_u64(physical, candidates, addresses, context, mask, true)
}

fn physical_ptr_from_addresses_u64(
    physical: &PhysicalMemory,
    candidates: &[RuntimeBuffer],
    addresses: &WarpValue<u64>,
    context: &WarpContext,
    mask: WarpMask,
    allow_shared_offsets: bool,
) -> Result<PhysicalPtr, EngineError> {
    if mask.is_empty() {
        let candidate = candidates.first().ok_or_else(|| {
            EngineError::message("raw generic address conversion has no declared memory candidate")
        })?;
        return Ok(PhysicalPtr::new(
            runtime_buffer_base(candidate).clone(),
            WarpValue::splat(0_i64),
            1,
        ));
    }
    let mut shared_addresses = WarpValue::splat(0_u32);
    let all_shared = mask.into_iter().all(|lane| {
        let Some(shared) = crate::instruction_codec::decode_generic_shared_address(addresses[lane])
        else {
            return false;
        };
        shared_addresses[lane] = shared;
        true
    });
    if all_shared {
        return physical_ptr_from_shared_addresses_u32(
            candidates,
            &shared_addresses,
            context,
            mask,
        );
    }

    let mut merged = PhysicalPtrSlot::new();
    for lane in mask {
        let lane_mask = WarpMask::from_bits(1_u32 << lane);
        if addresses[lane] == 0 && !allow_shared_offsets {
            // Carry null to the consuming instruction, which owns the access
            // error and its source site. Zero is not a generic shared address.
            merged.store(&PhysicalPtr::null(), lane_mask)?;
            continue;
        }
        let pointer = if let Some(shared) =
            crate::instruction_codec::decode_generic_shared_address(addresses[lane])
        {
            let shared_addresses = WarpValue::splat(shared);
            physical_ptr_from_shared_addresses_u32(
                candidates,
                &shared_addresses,
                context,
                lane_mask,
            )?
        } else {
            let address = addresses[lane];
            if let Some((view, offset)) = physical.global().observed_address_owner(address)? {
                let relative = i64::try_from(offset).map_err(|_| {
                    EngineError::out_of_bounds("global address offset exceeds int64")
                })?;
                PhysicalPtr::new(
                    RuntimeBuffer::Global(view),
                    WarpValue::splat(relative),
                    1,
                )
            } else if allow_shared_offsets && address <= u64::from(u32::MAX) {
                physical_ptr_from_shared_addresses_u32(
                    candidates, &WarpValue::splat(address as u32), context, lane_mask,
                )?
            } else {
                // A missing global binding is not proof of an invalid address.
                // Preserve the bits; the consuming instruction owns the
                // incomplete finding and its source site, as it does for null.
                PhysicalPtr::integer(WarpValue::splat(address))
            }
        };
        merged.store(&pointer, lane_mask)?;
    }
    merged.load(mask)
}

pub fn runtime_buffer_base(buffer: &RuntimeBuffer) -> &RuntimeBuffer {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => runtime_buffer_base(buffer),
        RuntimeBuffer::LaneSelected { .. } => buffer,
        _ => buffer,
    }
}

pub fn runtime_buffer_base_at(buffer: &RuntimeBuffer, lane: usize) -> &RuntimeBuffer {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => runtime_buffer_base_at(buffer, lane),
        RuntimeBuffer::LaneSelected { buffers } => runtime_buffer_base_at(&buffers[lane], lane),
        _ => buffer,
    }
}

pub fn runtime_buffer_readable_lanes(buffer: &RuntimeBuffer) -> WarpMask {
    match buffer {
        RuntimeBuffer::AccessView {
            buffer,
            readable_lanes,
            ..
        } => *readable_lanes & runtime_buffer_readable_lanes(buffer),
        RuntimeBuffer::LaneSelected { buffers } => {
            WarpMask::from_bits(buffers.lanes().iter().enumerate().fold(
                0_u32,
                |bits, (lane, buffer)| {
                    bits | if runtime_buffer_readable_lanes(buffer).contains(lane) {
                        1_u32 << lane
                    } else {
                        0
                    }
                },
            ))
        }
        _ => WarpMask::FULL,
    }
}

fn canonicalize_pointer_buffer(
    mut buffer: RuntimeBuffer,
) -> (
    RuntimeBuffer,
    WarpValue<i128>,
    WarpValue<i128>,
    WarpValue<i128>,
) {
    let (starts, ends) = match &mut buffer {
        RuntimeBuffer::Global(view) => {
            let start = view.byte_offset() as i128;
            let end = start + view.byte_len() as i128;
            *view = view.full_allocation_view();
            (WarpValue::splat(start), WarpValue::splat(end))
        }
        RuntimeBuffer::Shared {
            byte_offset,
            byte_len,
            backing_byte_len,
            virtual_base,
            ..
        } => {
            let start = *byte_offset as i128;
            let end = start + *byte_len as i128;
            *virtual_base = virtual_base.saturating_sub(*byte_offset);
            *byte_offset = 0;
            *byte_len = *backing_byte_len;
            (WarpValue::splat(start), WarpValue::splat(end))
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len,
            ..
        } => {
            let starts = WarpValue::from_fn(|lane| i128::from(byte_offsets[lane]));
            let ends = WarpValue::from_fn(|lane| starts[lane] + *byte_len as i128);
            *byte_offsets = WarpValue::splat(0_i64);
            *byte_len = allocations
                .iter()
                .map(|allocation| allocation.byte_len())
                .min()
                .unwrap_or(*byte_len);
            (starts, ends)
        }
        RuntimeBuffer::Local {
            allocations,
            byte_offset,
            byte_len,
        }
        | RuntimeBuffer::Register {
            allocations,
            byte_offset,
            byte_len,
        } => {
            let start = *byte_offset as i128;
            let end = start + *byte_len as i128;
            *byte_offset = 0;
            *byte_len = allocations
                .iter()
                .map(|allocation| allocation.bytes_per_lane())
                .min()
                .unwrap_or(*byte_len);
            (WarpValue::splat(start), WarpValue::splat(end))
        }
        RuntimeBuffer::Tmem { .. } => {
            let end = runtime_buffer_byte_len(&buffer) as i128;
            (WarpValue::splat(0_i128), WarpValue::splat(end))
        }
        RuntimeBuffer::AccessView { .. } => {
            unreachable!("access views are peeled before pointer canonicalization")
        }
        RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("lane selections are created only after pointer canonicalization")
        }
    };
    (buffer, starts.clone(), starts, ends)
}

fn runtime_buffer_lane_selection(buffer: &RuntimeBuffer) -> WarpValue<Arc<RuntimeBuffer>> {
    match buffer {
        RuntimeBuffer::LaneSelected { buffers } => buffers.clone(),
        _ => WarpValue::splat(Arc::new(buffer.clone())),
    }
}

fn runtime_buffer_broadcast_lane(buffer: &RuntimeBuffer, lane: usize) -> RuntimeBuffer {
    match buffer {
        RuntimeBuffer::AccessView {
            buffer,
            readable_lanes,
            writable_lanes,
        } => RuntimeBuffer::AccessView {
            buffer: Arc::new(runtime_buffer_broadcast_lane(buffer, lane)),
            readable_lanes: if readable_lanes.contains(lane) {
                WarpMask::FULL
            } else {
                WarpMask::EMPTY
            },
            writable_lanes: if writable_lanes.contains(lane) {
                WarpMask::FULL
            } else {
                WarpMask::EMPTY
            },
        },
        RuntimeBuffer::LaneSelected { buffers } => {
            runtime_buffer_broadcast_lane(&buffers[lane], lane)
        }
        RuntimeBuffer::RemoteShared {
            allocations,
            byte_offsets,
            byte_len,
            target_cta_ids,
            virtual_base,
        } => RuntimeBuffer::RemoteShared {
            allocations: allocations.clone(),
            byte_offsets: WarpValue::splat(byte_offsets[lane]),
            byte_len: *byte_len,
            target_cta_ids: WarpValue::splat(target_cta_ids[lane]),
            virtual_base: *virtual_base,
        },
        _ => buffer.clone(),
    }
}

fn runtime_buffer_byte_len_at(buffer: &RuntimeBuffer, lane: usize) -> usize {
    match buffer {
        RuntimeBuffer::LaneSelected { buffers } => runtime_buffer_byte_len_at(&buffers[lane], lane),
        _ => runtime_buffer_byte_len(buffer),
    }
}

fn runtime_buffer_view_offset_at(
    buffer: &RuntimeBuffer,
    lane: usize,
    relative: usize,
) -> Result<usize, EngineError> {
    let view_offset = match buffer {
        RuntimeBuffer::AccessView { .. } => {
            return Err(EngineError::message(
                "internal error: selected pointer retained an access view",
            ));
        }
        RuntimeBuffer::LaneSelected { buffers } => {
            return runtime_buffer_view_offset_at(&buffers[lane], lane, relative);
        }
        RuntimeBuffer::Global(view) => view.byte_offset(),
        RuntimeBuffer::Shared { byte_offset, .. }
        | RuntimeBuffer::Local { byte_offset, .. }
        | RuntimeBuffer::Register { byte_offset, .. } => *byte_offset,
        RuntimeBuffer::RemoteShared { byte_offsets, .. } => usize::try_from(byte_offsets[lane])
            .map_err(|_| EngineError::out_of_bounds("negative remote shared byte offset"))?,
        RuntimeBuffer::Tmem { .. } => {
            return Err(EngineError::message(
                "generic physical pointers cannot address tensor memory",
            ));
        }
    };
    view_offset
        .checked_add(relative)
        .ok_or_else(|| EngineError::out_of_bounds("physical pointer byte offset overflow"))
}

fn runtime_buffer_pointer_space_at(
    buffer: &RuntimeBuffer,
    lane: usize,
) -> Result<PointerSpace, EngineError> {
    match peel_runtime_buffer_wrappers(buffer, lane, ViewAccess::Address)? {
        RuntimeBuffer::AccessView { .. } | RuntimeBuffer::LaneSelected { .. } => {
            unreachable!("runtime buffer wrappers are peeled before the leaf walk")
        }
        RuntimeBuffer::Global(_) => Ok(PointerSpace::Global),
        RuntimeBuffer::Shared { .. } | RuntimeBuffer::RemoteShared { .. } => {
            Ok(PointerSpace::Shared)
        }
        RuntimeBuffer::Local { .. } => Ok(PointerSpace::Local),
        RuntimeBuffer::Register { .. } => Ok(PointerSpace::Register),
        RuntimeBuffer::Tmem { .. } => Err(EngineError::message(
            "generic physical pointers cannot address tensor memory",
        )),
    }
}

fn runtime_buffer_matches_ptx_space_at(
    buffer: &RuntimeBuffer,
    lane: usize,
    space: PtxStateSpace,
) -> bool {
    match buffer {
        RuntimeBuffer::AccessView { buffer, .. } => {
            runtime_buffer_matches_ptx_space_at(buffer, lane, space)
        }
        RuntimeBuffer::LaneSelected { buffers } => {
            runtime_buffer_matches_ptx_space_at(&buffers[lane], lane, space)
        }
        RuntimeBuffer::Global(_) => matches!(space, PtxStateSpace::Generic | PtxStateSpace::Global),
        RuntimeBuffer::Shared { .. } => matches!(
            space,
            PtxStateSpace::Generic
                | PtxStateSpace::Shared
                | PtxStateSpace::SharedCta
                | PtxStateSpace::SharedCluster
        ),
        RuntimeBuffer::RemoteShared { .. } => {
            matches!(space, PtxStateSpace::Generic | PtxStateSpace::SharedCluster)
        }
        RuntimeBuffer::Local { .. } | RuntimeBuffer::Register { .. } => {
            matches!(space, PtxStateSpace::Generic | PtxStateSpace::Local)
        }
        RuntimeBuffer::Tmem { .. } => false,
    }
}

fn peel_runtime_access_view(buffer: RuntimeBuffer) -> (RuntimeBuffer, WarpMask, WarpMask, bool) {
    let mut buffer = buffer;
    let mut readable_lanes = WarpMask::FULL;
    let mut writable_lanes = WarpMask::FULL;
    let mut restricted = false;
    loop {
        match buffer {
            RuntimeBuffer::AccessView {
                buffer: inner,
                readable_lanes: inner_readable,
                writable_lanes: inner_writable,
            } => {
                restricted = true;
                readable_lanes &= inner_readable;
                writable_lanes &= inner_writable;
                buffer = (*inner).clone();
            }
            _ => return (buffer, readable_lanes, writable_lanes, restricted),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::runtime::launch::allocate_cta_shared;
    use crate::{GlobalMemory, LaunchTopology, PhysicalMemory, WarpMask, WarpValue};

    use super::{PhysicalPtr, PhysicalPtrSlot, PointerSpace, PtxStateSpace, RuntimeBuffer};
    use crate::runtime::{read_runtime_bytes, write_runtime_bytes};


    fn global_pointer(byte_len: usize) -> (GlobalMemory, PhysicalPtr) {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(byte_len).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(global.full_view(allocation).unwrap()),
            WarpValue::splat(0_i64),
            4,
        );
        (global, pointer)
    }

    #[test]
    fn physical_pointer_clone_shares_state_until_derivation() {
        let (_global, base) = global_pointer(64);
        let clone = base.clone();

        assert!(Arc::ptr_eq(&base.state, &clone.state));

        let derived = base.with_pointee_itemsize(8);
        assert!(!Arc::ptr_eq(&base.state, &derived.state));
        assert_eq!(base.pointee_itemsize(), 4);
        assert_eq!(derived.pointee_itemsize(), 8);
    }

    #[test]
    fn load_offset_resolution_enforces_the_complete_lane_contract() {
        let (_global, base) = global_pointer(64);
        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        let pointer = PhysicalPtr::new(
            base.buffer().clone(),
            WarpValue::from_fn(|lane| lane as i64),
            4,
        );

        let offsets = pointer
            .resolve_load_byte_offsets(PtxStateSpace::Global, mask, 4)
            .unwrap();
        assert_eq!((offsets[0], offsets[1]), (0, 4));
        assert!(pointer
            .resolve_load_byte_offsets(PtxStateSpace::Shared, mask, 4)
            .unwrap_err()
            .to_string()
            .contains("does not match PTX state space"));

        let misaligned = pointer
            .with_byte_offset(&WarpValue::splat(2_i64), 4, mask)
            .unwrap();
        assert!(misaligned
            .resolve_load_byte_offsets(PtxStateSpace::Global, mask, 4)
            .unwrap_err()
            .to_string()
            .contains("requires 4-byte alignment"));
    }

    #[test]
    fn access_ptr_contract_preserves_permissions_and_extent_through_arithmetic() {
        let (_global, base) = global_pointer(64);
        let lane = WarpMask::from_lanes([0]).unwrap();
        let pointer = base
            .with_element_offset_extent(
                &WarpValue::splat(2_i64),
                &WarpValue::splat(2_i64),
                4,
                lane,
                1,
                "tvm_access_ptr",
            )
            .unwrap();

        assert_eq!(pointer.lane_read_byte_offset(0, 8).unwrap(), 8);
        assert!(pointer
            .lane_write_byte_offset(0, 4)
            .unwrap_err()
            .to_string()
            .contains("non-writable"));
        let access_range = pointer.lane_read_byte_offset_at(0, 8, 1).unwrap_err();
        assert!(access_range.is_out_of_bounds());
        assert!(access_range
            .to_string()
            .contains("outside tvm_access_ptr range"));

        let advanced = pointer
            .with_byte_offset(&WarpValue::splat(4_i64), 4, lane)
            .unwrap();
        assert_eq!(advanced.lane_read_byte_offset(0, 4).unwrap(), 12);
        let advanced_range = advanced.lane_read_byte_offset_at(0, 4, 1).unwrap_err();
        assert!(advanced_range.is_out_of_bounds());
        assert!(advanced_range
            .to_string()
            .contains("outside tvm_access_ptr range"));
        let negative = base
            .with_byte_offset(&WarpValue::splat(-4_i64), 4, lane)
            .unwrap();
        let restored = negative
            .with_byte_offset(&WarpValue::splat(4_i64), 4, lane)
            .unwrap();
        assert_eq!(restored.lane_read_byte_offset(0, 4).unwrap(), 0);
        let negative = negative.lane_read_byte_offset(0, 4).unwrap_err();
        assert!(negative.is_out_of_bounds());
        assert!(negative
            .to_string()
            .contains("negative physical pointer byte offset"));
        assert!(pointer
            .with_element_offset_extent(
                &WarpValue::splat(0_i64),
                &WarpValue::splat(1_i64),
                4,
                lane,
                2,
                "nested_tvm_access_ptr",
            )
            .err()
            .unwrap()
            .to_string()
            .contains("cannot add write access"));
    }

    #[test]
    fn pointer_slot_replaces_width_only_when_all_initialized_lanes_are_overwritten() {
        let (_global, base) = global_pointer(64);
        let wider = base.with_pointee_itemsize(8);
        let lane0 = WarpMask::from_lanes([0]).unwrap();
        let lane1 = WarpMask::from_lanes([1]).unwrap();
        let mut slot = PhysicalPtrSlot::new();
        slot.store(&base, lane0).unwrap();
        assert!(slot.store(&wider, lane1).is_err());
        assert_eq!(slot.load(lane0).unwrap().pointee_itemsize(), 4);
        slot.store(&wider, lane0).unwrap();
        assert_eq!(slot.load(lane0).unwrap().pointee_itemsize(), 8);
        assert!(slot.load(lane1).is_err());
    }

    #[test]
    fn pointer_slot_merges_lane_local_contracts_without_widening_them() {
        let (_global, base) = global_pointer(64);
        let lane0 = WarpMask::from_lanes([0]).unwrap();
        let lane1 = WarpMask::from_lanes([1]).unwrap();
        let both = lane0 | lane1;
        let read_pointer = base
            .with_element_offset_extent(
                &WarpValue::splat(0_i64),
                &WarpValue::splat(1_i64),
                4,
                lane0,
                1,
                "read_access",
            )
            .unwrap();
        let write_pointer = base
            .with_element_offset_extent(
                &WarpValue::splat(1_i64),
                &WarpValue::splat(1_i64),
                4,
                lane1,
                2,
                "write_access",
            )
            .unwrap();
        let mut slot = PhysicalPtrSlot::new();
        slot.store(&read_pointer, lane0).unwrap();
        slot.store(&write_pointer, lane1).unwrap();
        let merged = slot.load(both).unwrap();

        assert_eq!(merged.lane_read_byte_offset(0, 4).unwrap(), 0);
        assert!(merged.lane_write_byte_offset(0, 4).is_err());
        assert_eq!(merged.lane_write_byte_offset(1, 4).unwrap(), 4);
        assert!(merged.lane_read_byte_offset(1, 4).is_err());
    }

    #[test]
    fn pointer_slot_merges_distinct_views_of_one_physical_backing() {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(32).unwrap();
        let full = global.full_view(allocation).unwrap();
        let low = PhysicalPtr::new(
            RuntimeBuffer::Global(global.subview(&full, 0, 8).unwrap()),
            WarpValue::splat(0_i64),
            4,
        );
        let high = PhysicalPtr::new(
            RuntimeBuffer::Global(global.subview(&full, 8, 8).unwrap()),
            WarpValue::splat(0_i64),
            4,
        );
        let lane0 = WarpMask::from_lanes([0]).unwrap();
        let lane1 = WarpMask::from_lanes([1]).unwrap();
        let mut slot = PhysicalPtrSlot::new();
        slot.store(&low, lane0).unwrap();
        slot.store(&high, lane1).unwrap();
        let merged = slot.load(lane0 | lane1).unwrap();

        assert_eq!(merged.lane_read_byte_offset(0, 4).unwrap(), 0);
        assert_eq!(merged.lane_read_byte_offset(1, 4).unwrap(), 8);
    }

    #[test]
    fn pointer_slot_selects_distinct_global_backings_per_lane() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let global = GlobalMemory::new();
        let left_allocation = global.allocate_from_bytes([0x35_u8; 32]).unwrap();
        let right_allocation = global.allocate_from_bytes([0xa7_u8; 32]).unwrap();
        let left = PhysicalPtr::new(
            RuntimeBuffer::Global(global.full_view(left_allocation).unwrap()),
            WarpValue::from_fn(|lane| lane as i64),
            1,
        );
        let right = PhysicalPtr::new(
            RuntimeBuffer::Global(global.full_view(right_allocation).unwrap()),
            WarpValue::from_fn(|lane| lane as i64),
            1,
        );
        let even = WarpMask::from_bits(0x5555_5555);
        let odd = WarpMask::from_bits(0xaaaa_aaaa);
        let mut slot = PhysicalPtrSlot::new();
        slot.store(&left, even).unwrap();
        slot.store(&right, odd).unwrap();
        assert!(matches!(left.buffer(), RuntimeBuffer::Global(_)));
        assert!(matches!(right.buffer(), RuntimeBuffer::Global(_)));
        let selected = slot.load(WarpMask::FULL).unwrap();
        let physical = PhysicalMemory::with_global(topology, global);

        for lane in WarpMask::FULL {
            let offset = selected.lane_read_byte_offset(lane, 1).unwrap();
            let value = read_runtime_bytes(&physical, &context, selected.buffer(), lane, offset, 1)
                .unwrap();
            assert_eq!(value, vec![if lane % 2 == 0 { 0x35 } else { 0xa7 }]);
            write_runtime_bytes(
                &physical,
                &context,
                selected.buffer(),
                lane,
                offset,
                &[lane as u8],
            )
            .unwrap();
        }
        for lane in WarpMask::FULL {
            let offset = selected.lane_read_byte_offset(lane, 1).unwrap();
            assert_eq!(
                read_runtime_bytes(&physical, &context, selected.buffer(), lane, offset, 1,)
                    .unwrap(),
                vec![lane as u8]
            );
        }
    }

    #[test]
    fn lane_selected_pointer_space_validation_uses_the_operation_mask() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let global_allocation = physical.global().allocate_zeroed(8).unwrap();
        let global = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(global_allocation).unwrap()),
            WarpValue::splat(0_i64),
            1,
        );
        let shared_allocations = Arc::new(allocate_cta_shared(&physical, topology, 8).unwrap());
        let shared = PhysicalPtr::new(
            RuntimeBuffer::Shared {
                allocations: shared_allocations,
                byte_offset: 0,
                byte_len: 8,
                backing_byte_len: 8,
                virtual_base: 0,
            },
            WarpValue::splat(0_i64),
            1,
        );
        let mut slot = PhysicalPtrSlot::new();
        slot.store(&global, WarpMask::from_lanes([0]).unwrap())
            .unwrap();
        slot.store(&shared, WarpMask::from_lanes([1]).unwrap())
            .unwrap();
        let selected = slot.load(WarpMask::from_lanes([0, 1]).unwrap()).unwrap();

        assert!(selected
            .pointer_space()
            .unwrap_err()
            .to_string()
            .contains("different memory spaces"));
        let global_lane = WarpMask::from_lanes([0]).unwrap();
        let shared_lane = WarpMask::from_lanes([1]).unwrap();
        let mixed_lanes = global_lane | shared_lane;
        assert_eq!(
            selected.pointer_space_for_mask(global_lane).unwrap(),
            PointerSpace::Global
        );
        assert_eq!(
            selected.pointer_space_for_mask(shared_lane).unwrap(),
            PointerSpace::Shared
        );
        assert!(selected
            .require_ptx_space_for_mask(PtxStateSpace::Global, global_lane)
            .is_ok());
        assert!(selected
            .require_ptx_space_for_mask(PtxStateSpace::Shared, shared_lane)
            .is_ok());
        assert!(selected
            .require_ptx_space_for_mask(PtxStateSpace::Generic, mixed_lanes)
            .is_ok());
        assert!(selected
            .require_ptx_space_for_mask(PtxStateSpace::Global, mixed_lanes)
            .is_err());
    }

    #[test]
    fn decl_buffer_runtime_view_keeps_pointer_permissions() {
        let (global, base) = global_pointer(64);
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::with_global(topology, global);
        let lane = WarpMask::from_lanes([0]).unwrap();
        let read_pointer = base
            .with_element_offset_extent(
                &WarpValue::splat(1_i64),
                &WarpValue::splat(2_i64),
                4,
                lane,
                1,
                "read_access",
            )
            .unwrap();
        let view = read_pointer
            .runtime_view(&physical, PointerSpace::Global, 0, 8, 4, &context, lane)
            .unwrap();

        assert!(
            write_runtime_bytes(&physical, &context, &view, 0, 0, &1_u32.to_le_bytes(),)
                .unwrap_err()
                .to_string()
                .contains("non-writable DeclBuffer")
        );
        assert!(read_runtime_bytes(&physical, &context, &view, 0, 0, 4).is_ok());
    }

    #[test]
    fn remote_shared_pointer_requires_the_cluster_address_window() {
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let context = topology.warp_contexts().next().unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocations = Arc::new(allocate_cta_shared(&physical, topology, 16).unwrap());
        let mask = WarpMask::from_lanes([0, 1]).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::RemoteShared {
                allocations, byte_offsets: WarpValue::splat(0),
                byte_len: 16, target_cta_ids: WarpValue::splat(1), virtual_base: 0,
            },
            WarpValue::splat(1), 4,
        );
        pointer.require_ptx_space(PtxStateSpace::SharedCluster).unwrap();
        assert!(pointer.require_ptx_space(PtxStateSpace::SharedCta).is_err());
        assert!(pointer.require_ptx_space(PtxStateSpace::Shared).is_err());
        let observed = pointer.shared_byte_addresses_u32(&context, mask).unwrap();
        assert_eq!(observed[0], 0x01000004);
        assert_eq!(observed[1], 0x01000004);
        let outside = pointer.with_byte_offset(&WarpValue::splat(32), 4, mask).unwrap();
        // Forming an address does not dereference or retain a producer token.
        assert_eq!(outside.shared_byte_addresses_u32(&context, mask).unwrap()[0], 0x01000024);
        assert!(outside.lane_physical_byte_offset(0, 4).is_err());
    }
}
