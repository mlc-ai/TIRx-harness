use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::abi::v2::{ExecCtx, SiteId, WarpHandle};
use crate::effect::TensorMapObservation;
use crate::{
    BufferView, EngineError, GlobalMemory, MemoryScope, OperationKind, PhysicalAddress,
    PhysicalAllocationId, PhysicalByteSpan, PhysicalMemory, WarpContext, WarpMask,
};

use super::tensor_map::{RuntimeTensorMapImage, TENSOR_MAP_DESCRIPTOR_BYTES};
use super::{read_runtime_bytes, write_runtime_bytes, PhysicalPtr, PointerSpace, RuntimeTensorMap};

#[derive(Clone)]
pub struct RuntimeTensorMapRegistry {
    state: Arc<Mutex<RuntimeTensorMapRegistryState>>,
}

#[derive(Default)]
struct RuntimeTensorMapRegistryState {
    entries: BTreeMap<PhysicalAddress, TensorMapOrderingEntry>,
    parameter_addresses: BTreeMap<String, PhysicalPtr>,
    generic_release_generation: u64,
}

struct TensorMapOrderingEntry {
    published_generation: u64,
    // Necessary runtime precondition, not a grant of lane visibility. An
    // eligible acquire must occur in the consuming CTA for this generation;
    // Racecheck's lane clocks verify the actual acquire-to-consume handoff.
    acquired_ctas: BTreeSet<usize>,
    dirty_owner: Option<usize>,
    initial_host_visibility: bool,
    // Release occurrences are a necessary scope precondition. Whether their
    // snapshots actually cover each write is checked by Racecheck.
    publishers: BTreeMap<usize, MemoryScope>,
}

impl TensorMapOrderingEntry {
    fn bound(initial_host_visibility: bool) -> Self {
        Self {
            published_generation: 1,
            acquired_ctas: BTreeSet::new(),
            dirty_owner: None,
            initial_host_visibility,
            publishers: BTreeMap::new(),
        }
    }

    fn unbound() -> Self {
        Self {
            published_generation: 0,
            acquired_ctas: BTreeSet::new(),
            dirty_owner: None,
            initial_host_visibility: false,
            publishers: BTreeMap::new(),
        }
    }
}

impl Default for RuntimeTensorMapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeTensorMapRegistry {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(RuntimeTensorMapRegistryState::default())),
        }
    }

    fn state(&self) -> MutexGuard<'_, RuntimeTensorMapRegistryState> {
        self.state
            .lock()
            .expect("runtime TensorMap registry mutex poisoned")
    }

    fn descriptor_address(view: &BufferView) -> Result<PhysicalAddress, EngineError> {
        if view.byte_len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Err(EngineError::out_of_bounds(format!(
                "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {} remain",
                view.byte_len()
            )));
        }
        if !view
            .byte_offset()
            .is_multiple_of(TENSOR_MAP_DESCRIPTOR_BYTES)
        {
            return Err(EngineError::message(format!(
                "TensorMap descriptor address {}:{} must be 128-byte aligned",
                view.allocation(),
                view.byte_offset()
            )));
        }
        Ok(PhysicalAddress::new(
            view.allocation().as_u64(),
            view.byte_offset(),
        ))
    }

    fn resolve_descriptor(
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
        descriptor: &PhysicalPtr,
    ) -> Result<(PhysicalAddress, BufferView), EngineError> {
        let view = descriptor.resolve_uniform_global_remainder_view(physical, context, mask)?;
        let address = Self::descriptor_address(&view)?;
        Ok((address, view))
    }

    fn bind(
        &self,
        memory: &GlobalMemory,
        descriptor: BufferView,
        tensor_map: &RuntimeTensorMap,
        initial_host_visibility: bool,
    ) -> Result<(), EngineError> {
        let address = Self::descriptor_address(&descriptor)?;
        RuntimeTensorMapImage::from_tensor_map(tensor_map).write(memory, &descriptor)?;
        let mut state = self.state();
        if state
            .entries
            .insert(
                address,
                TensorMapOrderingEntry::bound(initial_host_visibility),
            )
            .is_some()
        {
            return Err(EngineError::message(format!(
                "TensorMap descriptor {}:{} is already bound",
                address.allocation_id(),
                address.byte_offset()
            )));
        }
        Ok(())
    }

    pub(crate) fn bind_host(
        &self,
        memory: &GlobalMemory,
        descriptor: BufferView,
        tensor_map: &RuntimeTensorMap,
    ) -> Result<(), EngineError> {
        self.bind(memory, descriptor, tensor_map, false)
    }

    pub(crate) fn bind_parameter(
        &self,
        name: &str,
        memory: &GlobalMemory,
        descriptor: BufferView,
        pointer: PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
    ) -> Result<(), EngineError> {
        self.bind(memory, descriptor, tensor_map, true)?;
        let mut state = self.state();
        if state
            .parameter_addresses
            .insert(name.to_owned(), pointer)
            .is_some()
        {
            return Err(EngineError::message(format!(
                "TensorMap parameter address {name:?} is already bound"
            )));
        }
        Ok(())
    }

    pub fn parameter_address(&self, name: &str) -> Result<PhysicalPtr, EngineError> {
        self.state()
            .parameter_addresses
            .get(name)
            .cloned()
            .ok_or_else(|| {
                EngineError::message(format!("TensorMap parameter address {name:?} is not bound"))
            })
    }

    fn mark_dirty(
        state: &mut RuntimeTensorMapRegistryState,
        address: PhysicalAddress,
        owner: usize,
    ) -> Result<(), EngineError> {
        let entry = state
            .entries
            .entry(address)
            .or_insert_with(TensorMapOrderingEntry::unbound);
        entry.dirty_owner = Some(owner);
        entry.initial_host_visibility = false;
        entry.acquired_ctas.clear();
        entry.publishers.clear();
        Ok(())
    }

    pub fn replace_global_address(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
        descriptor: &PhysicalPtr,
        address: &PhysicalPtr,
    ) -> Result<(), EngineError> {
        let replacement = address.resolve_uniform_global_remainder_view(physical, context, mask)?;
        self.replace_image(physical, context, mask, descriptor, |image| {
            image.replace_global_address(replacement);
            Ok(())
        })
    }

    pub fn replace_field(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
        descriptor: &PhysicalPtr,
        field: &str,
        index: Option<usize>,
        value: usize,
    ) -> Result<(), EngineError> {
        self.replace_image(physical, context, mask, descriptor, |image| {
            image.replace_field(field, index, value)
        })
    }

    fn replace_image(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
        descriptor: &PhysicalPtr,
        update: impl FnOnce(&mut RuntimeTensorMapImage) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let Some(lane) = mask.first_active() else {
            return Ok(());
        };
        let space = descriptor.pointer_space_for_mask(mask)?;
        if !matches!(space, PointerSpace::Global | PointerSpace::Shared) {
            return Err(EngineError::message(
                "tensormap.replace requires global or local-CTA shared memory",
            ));
        }
        if space == PointerSpace::Shared {
            descriptor.require_ptx_space_for_mask(super::PtxStateSpace::SharedCta, mask)?;
        }
        let address = descriptor.resolve_uniform(context, mask)?;
        if address.byte_offset() % TENSOR_MAP_DESCRIPTOR_BYTES != 0 {
            return Err(EngineError::message(
                "tensormap.replace requires 128-byte alignment",
            ));
        }
        for active in mask {
            descriptor.lane_read_byte_offset(active, TENSOR_MAP_DESCRIPTOR_BYTES)?;
            descriptor.lane_write_byte_offset(active, TENSOR_MAP_DESCRIPTOR_BYTES)?;
        }
        let offset = descriptor.lane_read_byte_offset(lane, TENSOR_MAP_DESCRIPTOR_BYTES)?;
        let mut image = RuntimeTensorMapImage::read_payload(|delta, count| {
            read_runtime_bytes(
                physical,
                context,
                descriptor.buffer(),
                lane,
                offset + delta,
                count,
            )
        })?;
        update(&mut image)?;
        let bytes = image.encode()?;
        // Shared descriptors are ordinary shared bytes. Publication happens
        // when cp_fenceproxy copies them into a global descriptor, not here.
        let mut state = self.state();
        if space == PointerSpace::Global {
            Self::mark_dirty(&mut state, address, context.global_warp_id())?;
        }
        write_runtime_bytes(physical, context, descriptor.buffer(), lane, offset, &bytes)
    }

    pub fn release(
        &self,
        warp: &mut impl WarpHandle,
        context: ExecCtx,
        site: SiteId,
        scope: MemoryScope,
    ) -> Result<(), EngineError> {
        let context = context.into_inner();
        if context.active_mask().is_empty() {
            return Ok(());
        }
        let warp = super::abi_transport::engine(warp);
        let operation =
            warp.begin_optional_operation(context, site.get(), OperationKind::Fence, false)?;
        warp.observe_tensor_map(operation.as_ref(), TensorMapObservation::Release { scope })?;
        self.release_entries(&context, scope)?;
        warp.finish_optional_operation(&operation)
    }

    fn release_entries(
        &self,
        context: &WarpContext,
        scope: MemoryScope,
    ) -> Result<(), EngineError> {
        let owner = context.global_warp_id();
        let mut state = self.state();
        let generation = state
            .generic_release_generation
            .checked_add(1)
            .ok_or_else(|| EngineError::message("TensorMap generation overflow"))?;
        state.generic_release_generation = generation;
        for entry in state.entries.values_mut() {
            if entry.dirty_owner.is_some() {
                entry.published_generation = generation;
                entry.dirty_owner = None;
                entry.acquired_ctas.clear();
            }
            entry
                .publishers
                .entry(owner)
                .and_modify(|current| *current = (*current).max(scope))
                .or_insert(scope);
        }
        Ok(())
    }

    /// Publish only the descriptor copied by cp_fenceproxy. A release does
    /// not acquire it, nor publish unrelated dirty descriptors.
    pub(crate) fn publish_copy(
        &self,
        warp: &mut impl WarpHandle,
        context: ExecCtx,
        site: SiteId,
        destination: &PhysicalPtr,
        scope: MemoryScope,
    ) -> Result<(), EngineError> {
        let context = context.into_inner();
        let warp = super::abi_transport::engine(warp);
        let (address, _) = Self::resolve_descriptor(
            warp.kernel().physical(),
            &context,
            context.active_mask(),
            destination,
        )?;
        let operation =
            warp.begin_optional_operation(context, site.get(), OperationKind::Fence, false)?;
        warp.observe_tensor_map(
            operation.as_ref(),
            TensorMapObservation::CopyRelease {
                descriptor: descriptor_span(address)?,
                scope,
            },
        )?;
        self.publish_copy_descriptor(warp.kernel().physical(), &context, destination, scope)?;
        warp.finish_optional_operation(&operation)
    }

    fn publish_copy_descriptor(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        destination: &PhysicalPtr,
        scope: MemoryScope,
    ) -> Result<(), EngineError> {
        let (address, _) =
            Self::resolve_descriptor(physical, context, context.active_mask(), destination)?;
        let mut state = self.state();
        Self::mark_dirty(&mut state, address, context.global_warp_id())?;
        let generation = state
            .generic_release_generation
            .checked_add(1)
            .ok_or_else(|| EngineError::message("TensorMap generation overflow"))?;
        state.generic_release_generation = generation;
        let entry = state.entries.get_mut(&address).expect("marked descriptor");
        entry.published_generation = generation;
        entry.dirty_owner = None;
        entry.publishers.insert(context.global_warp_id(), scope);
        Ok(())
    }

    pub fn acquire(
        &self,
        warp: &mut impl WarpHandle,
        context: ExecCtx,
        site: SiteId,
        descriptor: &PhysicalPtr,
        scope: MemoryScope,
    ) -> Result<(), EngineError> {
        let context = context.into_inner();
        let mask = context.active_mask();
        let warp = super::abi_transport::engine(warp);
        for lane in mask {
            let lane_mask = WarpMask::from_bits(1 << lane);
            if let Some((address, generation)) = self.acquire_descriptor(
                warp.kernel().physical(),
                &context,
                lane_mask,
                descriptor,
                scope,
            )? {
                // A proxy acquire is thread-local, including when one source
                // instruction addresses different descriptors across lanes.
                let operation = warp.begin_optional_operation(
                    context.with_active_mask(lane_mask),
                    site.get(),
                    OperationKind::Fence,
                    false,
                )?;
                warp.observe_tensor_map(
                    operation.as_ref(),
                    TensorMapObservation::Acquire {
                        descriptor: descriptor_span(address)?,
                        generation,
                        lane,
                        cta: context.global_cta_id(),
                        scope,
                    },
                )?;
                warp.finish_optional_operation(&operation)?;
            }
        }
        Ok(())
    }

    fn acquire_descriptor(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        mask: WarpMask,
        descriptor: &PhysicalPtr,
        scope: MemoryScope,
    ) -> Result<Option<(PhysicalAddress, u64)>, EngineError> {
        let (address, _view) = Self::resolve_descriptor(physical, context, mask, descriptor)?;
        let mut state = self.state();
        // A kernel launch makes prior generic-proxy writes visible.  The
        // explicit acquire bridges that launch-visible state into the
        // tensormap proxy even when the descriptor was copied as ordinary
        // bytes rather than supplied as a host-bound TensorMap slot.
        let fallback_generation = state.generic_release_generation.max(1);
        let entry = match state.entries.entry(address) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(TensorMapOrderingEntry {
                    published_generation: fallback_generation,
                    acquired_ctas: BTreeSet::new(),
                    dirty_owner: None,
                    initial_host_visibility: false,
                    publishers: BTreeMap::new(),
                })
            }
        };
        if let Some(owner) = entry.dirty_owner {
            return Err(EngineError::message(format!(
                "TensorMap descriptor is dirty in warp {owner} and cannot be acquired"
            )));
        }
        if entry.published_generation == 0 {
            return Err(EngineError::message(
                "TensorMap descriptor has not been published",
            ));
        }
        if !entry.publishers.is_empty()
            && !entry
                .publishers
                .iter()
                .any(|(&publisher, &published_scope)| {
                    let required = MemoryScope::required_between_warps(
                        Some(context.topology()),
                        publisher,
                        context.global_warp_id(),
                    );
                    published_scope >= required && scope >= required
                })
        {
            // A narrow acquire is legal but supplies no descriptor visibility.
            return Ok(None);
        }
        entry.acquired_ctas.insert(context.global_cta_id());
        Ok(Some((address, entry.published_generation)))
    }

    pub fn lookup(
        &self,
        warp: &mut impl WarpHandle,
        context: ExecCtx,
        site: SiteId,
        descriptor: &PhysicalPtr,
        expected_rank: usize,
    ) -> Result<RuntimeTensorMap, EngineError> {
        let context = context.into_inner();
        let mask = context.active_mask();
        let warp = super::abi_transport::engine(warp);
        let operation =
            warp.begin_optional_operation(context, site.get(), OperationKind::Load, false)?;
        let result = (|| {
            let physical = warp.kernel().physical();
            let (address, descriptor_view) =
                Self::resolve_descriptor(physical, &context, mask, descriptor)?;
            let observation = {
                let state = self.state();
                let entry = state.entries.get(&address).ok_or_else(|| {
                    EngineError::message(format!(
                        "TensorMap descriptor {}:{} is not bound",
                        address.allocation_id(),
                        address.byte_offset()
                    ))
                })?;
                if let Some(owner) = entry.dirty_owner {
                    return Err(EngineError::message(format!(
                        "TensorMap descriptor is dirty in warp {owner}"
                    )));
                }
                if entry.published_generation == 0 {
                    return Err(EngineError::message("TensorMap descriptor is unpublished"));
                }
                let initially_visible =
                    entry.initial_host_visibility && entry.published_generation == 1;
                if !initially_visible && !entry.acquired_ctas.contains(&context.global_cta_id()) {
                    return Err(EngineError::message(
                        "TensorMap descriptor latest published generation is not acquired within this CTA",
                    ));
                }
                (!initially_visible).then_some(TensorMapObservation::Consume {
                    descriptor: descriptor_span(address)?,
                    generation: entry.published_generation,
                    cta: context.global_cta_id(),
                })
            };
            if let Some(observation) = observation {
                warp.observe_tensor_map(operation.as_ref(), observation)?;
            }
            RuntimeTensorMapImage::read(physical.global(), &descriptor_view)?
                .materialize(physical.global(), expected_rank)
        })()
        .map_err(|error: EngineError| match operation.as_ref() {
            Some(operation) => error.with_operation_context(operation),
            None => error,
        })?;
        warp.finish_optional_operation(&operation)?;
        Ok(result)
    }
}

fn descriptor_span(address: PhysicalAddress) -> Result<PhysicalByteSpan, EngineError> {
    PhysicalByteSpan::new(
        PhysicalAllocationId::new(address.allocation_id()),
        address.byte_offset(),
        TENSOR_MAP_DESCRIPTOR_BYTES,
    )
    .map_err(|error| EngineError::message(error.to_string()))
}
