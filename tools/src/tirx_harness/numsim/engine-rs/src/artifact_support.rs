//! Generated-artifact transport and launch support.
//!
//! This module is not an instruction ABI.  It contains the Rust/Python launch
//! boundary, opaque runtime storage used by generated functions, and the
//! one-way adapters that wrap those storage values in `abi::v2` capabilities.
//! Instruction, memory, synchronization, and checker effects must enter through
//! `abi::v2`. Pure frontend expressions over register storage have no engine
//! handle and are implemented in `frontend_expr`.

/// Concrete pure frontend expressions over generated register storage.
/// These have no instruction, memory, or checker effects.
pub mod frontend_expr;

use std::future::Future;
use std::sync::Arc;

pub use crate::worker_affinity::prime_host_load_sample;

pub use crate::context::WarpContext;
pub use crate::engine_mode::EngineMode;
pub use crate::executor::{EngineError, ExecutionStats};
pub use crate::instruction_codec::{
    decode_generic_shared_address, encode_shared_address, generic_shared_address,
    replace_generic_shared_address_cta_rank, replace_shared_address_cta_rank,
    shared_address_byte_offset, shared_address_cta_rank,
};
pub use crate::kernel_engine::WarpEngine;
pub use crate::mask::{WarpMask, WARP_SIZE};
pub use crate::memory::{AllocationId, GlobalMemory};
pub use crate::numpy_backend::{
    bf16_bits_to_f32, decoded_bf16_to_bits, decoded_fp16_to_bits, f32_to_bf16_bits,
    f32_to_float8_e4m3fn_bits, f32_to_float8_e8m0fnu_bits, f32_to_fp16_bits,
    float4_e2m1fn_bits_to_f32, float8_e4m3fn_bits_to_f32, float8_e8m0fnu_bits_to_f32,
    fp16_bits_to_f32,
};
pub use crate::physical_access::MemoryScope;
pub use crate::runtime::launch::{
    allocate_cta_shared, allocate_cta_tmem, allocate_warp_private, ExecutionPolicy,
    LaunchSelection, DEFAULT_NATIVE_LOOP_ITERATION_BUDGET, DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM,
};
pub use crate::runtime::operand::{
    physical_ptr_from_generic_addresses_u64, physical_ptr_from_shared_addresses_u32,
    physical_ptr_from_view_addresses_u64,
    runtime_buffer_byte_len, PhysicalPtr, PointerSpace, PtxStateSpace, RuntimeBuffer,
};
pub use crate::runtime::tensor_map::RuntimeTensorMap;
pub use crate::runtime::tensor_map_registry::RuntimeTensorMapRegistry;
pub use crate::runtime::tmem::get_tmem_addr;
pub use crate::runtime::warp_ops::require_uniform_i64;
pub use crate::scalar::{
    cuda_canonicalize_nan_f32, cuda_f32_max, cuda_f32_min, cuda_f32_to_fp16_bits,
    cuda_fp16_bits_to_f32, float2_x, float2_y, floor_div_i64, floor_mod_i64,
    fp8x4_e4m3_from_float4, make_float2, pack_bf16x2, ptx_exp2_approx_ftz_f32,
    ptx_lg2_approx_ftz_f32, ptx_tanh_approx_f32, unpack_bf16x2, F32x4, U64x2,
};
pub use crate::spaces::{PhysicalMemory, SharedAllocation, TmemAllocation, WarpPrivateAllocation};
pub use crate::topology::LaunchTopology;
pub use crate::warp_value::WarpValue;

/// Type-erased pure element map used by generated frontends.
///
/// This is an artifact carrier, not an instruction ABI: the frontend still
/// supplies the complete layout program and this type has no engine handle.
#[doc(hidden)]
pub struct FrontendFnMap<S: crate::abi::v2::MemorySpace> {
    mapping: Box<
        dyn for<'a> Fn(
                crate::abi::v2::LogicalCoord<'a>,
                crate::abi::v2::LaneId,
            ) -> Result<crate::abi::v2::ElementRef<S>, EngineError>
            + Send
            + Sync,
    >,
    owners: FrontendOwners,
}

enum FrontendOwners {
    AllLanes,
    Dynamic(
        Box<
            dyn for<'a> Fn(
                    crate::abi::v2::LogicalCoord<'a>,
                ) -> Result<crate::abi::v2::LaneMask, EngineError>
                + Send
                + Sync,
        >,
    ),
}

impl<S: crate::abi::v2::MemorySpace> FrontendFnMap<S> {
    pub fn new<F, O>(mapping: F, owners: O) -> Self
    where
        F: for<'a> Fn(
                crate::abi::v2::LogicalCoord<'a>,
                crate::abi::v2::LaneId,
            ) -> Result<crate::abi::v2::ElementRef<S>, EngineError>
            + Send
            + Sync
            + 'static,
        O: for<'a> Fn(
                crate::abi::v2::LogicalCoord<'a>,
            ) -> Result<crate::abi::v2::LaneMask, EngineError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            mapping: Box::new(mapping),
            owners: FrontendOwners::Dynamic(Box::new(owners)),
        }
    }

    pub fn all_lanes<F>(mapping: F) -> Self
    where
        F: for<'a> Fn(
                crate::abi::v2::LogicalCoord<'a>,
                crate::abi::v2::LaneId,
            ) -> Result<crate::abi::v2::ElementRef<S>, EngineError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            mapping: Box::new(mapping),
            owners: FrontendOwners::AllLanes,
        }
    }
}

impl<S: crate::abi::v2::MemorySpace> crate::abi::v2::ElementMap<S> for FrontendFnMap<S> {
    type Runtime = ();

    fn owners(
        &self,
        logical: crate::abi::v2::LogicalCoord<'_>,
        (): &Self::Runtime,
    ) -> Result<crate::abi::v2::LaneMask, crate::abi::v2::MapError> {
        match &self.owners {
            FrontendOwners::AllLanes => Ok(crate::abi::v2::LaneMask::FULL),
            FrontendOwners::Dynamic(owners) => {
                (owners)(logical).map_err(|error| crate::abi::v2::MapError::new(error.to_string()))
            }
        }
    }

    fn map(
        &self,
        logical: crate::abi::v2::LogicalCoord<'_>,
        lane: crate::abi::v2::LaneId,
        (): &Self::Runtime,
    ) -> Result<crate::abi::v2::ElementRef<S>, crate::abi::v2::MapError> {
        (self.mapping)(logical, lane)
            .map_err(|error| crate::abi::v2::MapError::new(error.to_string()))
    }
}

/// Table-backed pure element map used by generated frontends.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrontendTableMap<S: crate::abi::v2::MemorySpace>(std::marker::PhantomData<fn() -> S>);

impl<S: crate::abi::v2::MemorySpace> FrontendTableMap<S> {
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<S: crate::abi::v2::MemorySpace> crate::abi::v2::ElementMap<S> for FrontendTableMap<S> {
    type Runtime = (Vec<usize>, Vec<crate::abi::v2::ElementRef<S>>);

    fn map(
        &self,
        logical: crate::abi::v2::LogicalCoord<'_>,
        lane: crate::abi::v2::LaneId,
        runtime: &Self::Runtime,
    ) -> Result<crate::abi::v2::ElementRef<S>, crate::abi::v2::MapError> {
        let (shape, elements) = runtime;
        if logical.dimensions().len() != shape.len() {
            return Err(crate::abi::v2::MapError::new(
                "frontend element map rank mismatch",
            ));
        }
        let mut linear = 0_usize;
        for (&coordinate, &extent) in logical.dimensions().iter().zip(shape) {
            let coordinate = usize::try_from(coordinate)
                .map_err(|_| crate::abi::v2::MapError::new("negative frontend tile coordinate"))?;
            if coordinate >= extent {
                return Err(crate::abi::v2::MapError::new(
                    "frontend tile coordinate is out of bounds",
                ));
            }
            linear = linear
                .checked_mul(extent)
                .and_then(|value| value.checked_add(coordinate))
                .ok_or_else(|| crate::abi::v2::MapError::new("frontend tile index overflow"))?;
        }
        let index = linear
            .checked_mul(crate::WARP_SIZE)
            .and_then(|value| value.checked_add(lane.index()))
            .ok_or_else(|| {
                crate::abi::v2::MapError::new("frontend element table index overflow")
            })?;
        elements
            .get(index)
            .copied()
            .ok_or_else(|| crate::abi::v2::MapError::new("frontend element table is incomplete"))
    }
}

#[cfg(feature = "python")]
pub use crate::runtime::python::{
    build_artifact_metadata_with_build_identity, build_run_result, extract_allocations,
    extract_buffer, extract_buffer_alias, extract_or_build_implicit_tensor_map,
    extract_output_allocations, extract_phase_selection, extract_pointer, extract_rank_inputs,
    prepare_rank_buffers, extract_scalar_bool,
    extract_scalar_f32, extract_scalar_f64, extract_scalar_i16, extract_scalar_i32,
    extract_scalar_i64, extract_scalar_i8, extract_scalar_u16, extract_scalar_u32,
    extract_scalar_u64, extract_scalar_u8, extract_selection, extract_shape_extent,
    extract_tensor_map, extract_tensor_map_descriptor_registry, required_item,
    validate_shape_scalar, RunResultBuilder,
};

/// Concrete engine type used by ordinary NumSim artifacts.
///
/// Generated code names this alias instead of carrying an `impl EngineMode`
/// parameter through every helper. Instruction specializations can therefore
/// be compiled in this crate once and called through concrete entry points.
pub type NumSimWarpEngine = WarpEngine<crate::engine_mode::NumSimMode>;
pub type SyncCheckWarpEngine = WarpEngine<crate::sync_check::SyncCheckMode>;
#[cfg(feature = "racecheck")]
pub type RaceCheckWarpEngine = WarpEngine<crate::race_check::RaceCheckMode>;

/// Frontend storage/layout code needs launch coordinates before it can form
/// v2 operands.  Keep that transport read here so `WarpEngine` exposes no
/// operation methods across the generated-artifact boundary.
#[doc(hidden)]
pub fn artifact_warp_context<M: crate::engine_mode::EngineMode>(
    warp: &WarpEngine<M>,
) -> WarpContext {
    warp.context()
}

/// Materialize a frontend-owned index list without unrolling every buffer clone.
#[inline(never)]
pub fn select_runtime_buffers(buffers: &[RuntimeBuffer], indices: &[usize]) -> Vec<RuntimeBuffer> {
    indices.iter().map(|&index| buffers[index].clone()).collect()
}

pub fn runtime_buffer_global(view: crate::BufferView) -> RuntimeBuffer {
    RuntimeBuffer::Global(view)
}

/// Detach immutable handle tables at a scheduling-domain boundary, not storage.
/// All aliases must be passed together so raw-address ambiguity checks retain
/// the same table identity. Hot pointer clones then avoid a launch-wide refcount.
pub fn localize_shared_buffer_handles<'a>(
    buffers: impl Iterator<Item = &'a mut RuntimeBuffer>,
) {
    let mut tables = std::collections::BTreeMap::new();
    for buffer in buffers {
        if let RuntimeBuffer::Shared { allocations, .. }
        | RuntimeBuffer::RemoteShared { allocations, .. } = buffer
        {
            let local = tables.entry(Arc::as_ptr(allocations) as usize)
                .or_insert_with(|| Arc::new(allocations.as_ref().clone()));
            *allocations = Arc::clone(local);
        }
    }
}

pub fn runtime_buffer_shared(
    allocations: Arc<Vec<SharedAllocation>>,
    byte_offset: usize,
    byte_len: usize,
    backing_byte_len: usize,
    virtual_base: usize,
) -> RuntimeBuffer {
    RuntimeBuffer::Shared {
        allocations,
        byte_offset,
        byte_len,
        backing_byte_len,
        virtual_base,
    }
}

pub fn runtime_buffer_local(
    allocations: Arc<Vec<WarpPrivateAllocation>>,
    byte_offset: usize,
    byte_len: usize,
) -> RuntimeBuffer {
    RuntimeBuffer::Local {
        allocations,
        byte_offset,
        byte_len,
    }
}

pub fn runtime_buffer_register(
    allocations: Arc<Vec<WarpPrivateAllocation>>,
    byte_offset: usize,
    byte_len: usize,
) -> RuntimeBuffer {
    RuntimeBuffer::Register {
        allocations,
        byte_offset,
        byte_len,
    }
}

pub fn runtime_buffer_tmem(
    allocations: Arc<Vec<TmemAllocation>>,
    lane_span: usize,
    tcol_span_elements: usize,
    elem_offset: usize,
    itemsize: usize,
) -> RuntimeBuffer {
    RuntimeBuffer::Tmem {
        allocations,
        lane_span,
        tcol_span_elements,
        elem_offset,
        itemsize,
    }
}

/// Pure CUDA-source token conversion; this is not a PTX instruction effect.
pub const fn shared_address_from_u64(address: u64) -> u32 {
    crate::instruction_codec::shared_address(address)
}

/// Pure SM100 CUDA-source encoding used before a TMA mbarrier operand reaches
/// its instruction ABI.
pub const fn sm100_tma_2sm_mbarrier_address_from_u64(address: u64) -> u32 {
    crate::instruction_codec::sm100_tma_2sm_mbarrier_address(address)
}

/// Pure descriptor-field replacement used by the typed DPS frontend before
/// the descriptor reaches a raw tcgen05 instruction boundary.
pub const fn tcgen_runtime_instruction_descriptor(bits: u32, sf_id: u32) -> u32 {
    crate::instruction_codec::tcgen_runtime_instruction_descriptor(bits, sf_id)
}

pub fn physical_ptr_byte_offset(
    pointer: &PhysicalPtr,
    offsets: &WarpValue<i64>,
    pointee_itemsize: usize,
    mask: WarpMask,
) -> Result<PhysicalPtr, EngineError> {
    pointer.with_byte_offset(offsets, pointee_itemsize, mask)
}

#[allow(clippy::too_many_arguments)]
pub fn physical_ptr_access_view(
    pointer: &PhysicalPtr,
    element_offsets: &WarpValue<i64>,
    element_extents: &WarpValue<i64>,
    itemsize: usize,
    mask: WarpMask,
    access_mask: u8,
    operation: &str,
) -> Result<PhysicalPtr, EngineError> {
    pointer.with_element_offset_extent_labeled(
        element_offsets,
        element_extents,
        itemsize,
        mask,
        access_mask,
        &crate::DiagnosticLabel::new(operation),
    )
}

/// Opaque services shared by all generated warps in one launch.
#[derive(Clone)]
pub struct KernelRuntimeServices(crate::runtime::launch::LaunchRuntimeServices);

impl KernelRuntimeServices {
    fn from_runtime(services: crate::runtime::launch::LaunchRuntimeServices) -> Self {
        Self(services)
    }

    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.0.execution_policy()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_kernel_launch_ordered<MakeWarp, WarpFuture>(
    physical: &PhysicalMemory,
    kernel_index: usize,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    mut make_warp: MakeWarp,
) -> Result<ExecutionStats, EngineError>
where
    MakeWarp:
        FnMut(WarpEngine<crate::engine_mode::NumSimMode>, KernelRuntimeServices) -> WarpFuture,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    crate::runtime::launch::execute_kernel_launch_ordered(
        physical.topology(),
        selection,
        max_workers,
        execution_policy,
        physical.ordering(),
        move |context, services| {
            let artifact_services = KernelRuntimeServices::from_runtime(services.clone());
            let kernel = crate::kernel_engine::KernelEngine::new(
                kernel_index,
                physical.clone(),
                services,
                Arc::new(()),
            );
            make_warp(WarpEngine::new(context, kernel), artifact_services)
        },
    )
}

#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
pub fn run_synccheck_analysis_phase<MakeWarp, WarpFuture>(
    py: pyo3::Python<'_>,
    phase_index: usize,
    phase_name: &str,
    physical: PhysicalMemory,
    inputs: &pyo3::Bound<'_, pyo3::types::PyDict>,
    allocation_ids: &[AllocationId],
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    fixed_trace_eligible: bool,
    max_warp_preemptions: u64,
    max_completion_schedule_deviations: u64,
    max_schedules: u64,
    max_backtrack_nodes: u64,
    max_events_per_run: u64,
    max_total_events: u64,
    max_loop_steps: u64,
    max_wall_time_ms: u64,
    max_diagnostic_bytes: u64,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    make_warp: MakeWarp,
) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>>
where
    MakeWarp: FnMut(SyncCheckWarpEngine, PhysicalMemory, KernelRuntimeServices) -> WarpFuture + Send,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    let topology = physical.topology();
    let options = crate::sync_check_python::NativeSyncCheckOptions::new(
        max_warp_preemptions,
        max_completion_schedule_deviations,
        max_schedules,
        max_backtrack_nodes,
        max_events_per_run,
        max_total_events,
        max_loop_steps,
        max_wall_time_ms,
        max_diagnostic_bytes,
        max_polls,
        max_transitions,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let callback_physical = physical.clone();
    let review_physical = physical.clone();
    let mut make_warp = make_warp;
    let result = crate::sync_check_python::run_native_sync_check_phase(
        py,
        phase_index,
        phase_name,
        topology,
        physical,
        inputs,
        allocation_ids,
        selection,
        max_workers,
        execution_policy,
        fixed_trace_eligible,
        options,
        move |warp| {
            let services = KernelRuntimeServices::from_runtime(warp.kernel().services().clone());
            make_warp(warp, callback_physical.clone(), services)
        },
    )?;
    crate::runtime::python::annotate_analysis_uninitialized_read_reviews(
        py,
        result,
        review_physical.take_uninitialized_read_reviews(),
    )
}

#[cfg(all(feature = "python", feature = "racecheck"))]
pub use crate::race_check_python::with_complete_global_write_seed;

#[cfg(all(feature = "python", feature = "racecheck"))]
#[allow(clippy::too_many_arguments)]
pub fn run_racecheck_analysis_phase<MakeWarp, WarpFuture>(
    py: pyo3::Python<'_>,
    phase_index: usize,
    phase_name: &str,
    physical: PhysicalMemory,
    global_write_allocations: Vec<AllocationId>,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    inspect_accesses: bool,
    global_memory_model_enabled: bool,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    make_warp: MakeWarp,
) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>>
where
    MakeWarp:
        FnMut(RaceCheckWarpEngine, PhysicalMemory, KernelRuntimeServices) -> WarpFuture + Send,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    let topology = physical.topology();
    let callback_physical = physical.clone();
    let review_physical = physical.clone();
    let mut make_warp = make_warp;
    let result = crate::race_check_python::run_native_race_check_phase(
        py,
        phase_index,
        phase_name,
        topology,
        physical,
        global_write_allocations.into_iter().map(Into::into).collect(),
        selection,
        max_workers,
        execution_policy,
        inspect_accesses,
        global_memory_model_enabled,
        max_polls,
        max_transitions,
        move |warp| {
            let services = KernelRuntimeServices::from_runtime(warp.kernel().services().clone());
            make_warp(warp, callback_physical.clone(), services)
        },
    )?;
    crate::runtime::python::annotate_analysis_uninitialized_read_reviews(
        py,
        result,
        review_physical.take_uninitialized_read_reviews(),
    )
}

use crate::abi::v2::{
    Address, BufferHandle, DescriptorDomain, ExecCtx, LaneMask, MemorySpace, TensorMapHandle, R,
};

#[doc(hidden)]
#[inline(always)]
pub fn v2_context(value: crate::WarpContext) -> ExecCtx {
    ExecCtx::from_inner(value)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_context_out(value: ExecCtx) -> crate::WarpContext {
    value.into_inner()
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_mask(value: crate::WarpMask) -> LaneMask {
    LaneMask::from_inner(value)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_address<S: MemorySpace>(value: crate::runtime::PhysicalPtr) -> Address<S> {
    Address::from_inner(value)
}

/// Form an address whose physical pointer was recovered from a named TIR
/// buffer expression. The name is checker/diagnostic metadata only; bounds and
/// state-space validation remain properties of the physical pointer.
#[doc(hidden)]
#[inline(always)]
pub fn v2_named_address<S: MemorySpace>(
    value: crate::runtime::PhysicalPtr,
    logical_buffer: &'static str,
) -> Address<S> {
    Address::from_logical_buffer(value, logical_buffer)
}

/// Form a direct TIR Buffer address. Unlike an arbitrary pointer value, this
/// capability remains bounded to the declared Buffer view and carries its
/// logical identity only as checker/diagnostic metadata.
#[doc(hidden)]
#[inline(always)]
pub fn v2_buffer_address<'a, S: MemorySpace>(
    value: &'a crate::runtime::RuntimeBuffer,
    indices: &'a crate::WarpValue<i64>,
    itemsize: usize,
    logical_buffer: &'static str,
) -> crate::abi::v2::DirectAddress<'a, S> {
    crate::abi::v2::DirectAddress::new(value, indices, itemsize, logical_buffer)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_address_out<S: MemorySpace>(value: Address<S>) -> crate::runtime::PhysicalPtr {
    value.into_inner()
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_buffer<S: MemorySpace>(value: crate::runtime::RuntimeBuffer) -> BufferHandle<S> {
    BufferHandle::from_inner(value)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_named_buffer<S: MemorySpace>(
    value: crate::runtime::RuntimeBuffer,
    logical_buffer: &'static str,
) -> BufferHandle<S> {
    BufferHandle::from_logical_buffer(value, logical_buffer)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_descriptor_domain<S: MemorySpace>(
    values: Vec<crate::runtime::RuntimeBuffer>,
) -> DescriptorDomain<S> {
    DescriptorDomain::from_buffers(values)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_tensor_map(value: crate::runtime::RuntimeTensorMap) -> TensorMapHandle {
    TensorMapHandle::from_inner(std::sync::Arc::new(value))
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_register<T>(value: crate::WarpValue<T>) -> R<T> {
    R::from_inner(value)
}

#[doc(hidden)]
#[inline(always)]
pub fn v2_register_out<T>(value: R<T>) -> crate::WarpValue<T> {
    value.into_inner()
}

#[doc(hidden)]
/// Read an instruction operand materialized in frontend local/register
/// storage. This is register transport, not a memory instruction.
#[doc(hidden)]
pub fn read_frontend_register_operand<T: crate::RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: ExecCtx,
    source: &crate::runtime::PhysicalPtr,
) -> Result<R<T>, crate::abi::v2::EngineError> {
    let context = context.into_inner();
    let mask = context.active_mask();
    if !matches!(
        source.pointer_space_for_mask(mask)?,
        crate::runtime::PointerSpace::Local | crate::runtime::PointerSpace::Register
    ) {
        return Err(crate::abi::v2::EngineError::message(
            "frontend register operand must address local or register storage",
        ));
    }
    let source = source.with_byte_storage_access_width(T::BYTE_LEN)?;
    Ok(R::from_inner(crate::runtime::load_physical_ptr_warp::<T>(
        physical, &context, &source, mask,
    )?))
}

/// Gather one source-level tile operand from its frontend-selected owner
/// thread. PTX has no cross-warp register-read instruction; the frontend owns
/// this layout choice, while this adapter enforces that the carrier is only
/// simulated local/register storage.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn read_frontend_register_element_at_thread<T: crate::RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: ExecCtx,
    buffer: &crate::runtime::RuntimeBuffer,
    target_warp_id_in_cta: usize,
    index: i64,
    lane: usize,
) -> Result<T, crate::abi::v2::EngineError> {
    if !matches!(
        buffer.uniform_physical_space(),
        Some(crate::PhysicalAccessSpace::Local | crate::PhysicalAccessSpace::Register)
    ) {
        return Err(crate::abi::v2::EngineError::message(
            "frontend register gather must address local or register storage",
        ));
    }
    let context = context.into_inner();
    Ok(crate::runtime::load_warp_private_scalar_at_thread::<T>(
        physical,
        &context,
        buffer,
        target_warp_id_in_cta,
        index,
        lane,
    )?)
}

/// Materialize an instruction result in frontend local/register storage. This
/// adapter cannot commit a global/shared/TMEM memory effect.
#[doc(hidden)]
pub fn write_frontend_register_result<T: crate::RuntimeScalar>(
    physical: &crate::PhysicalMemory,
    context: ExecCtx,
    destination: &crate::runtime::PhysicalPtr,
    value: R<T>,
) -> Result<(), crate::abi::v2::EngineError> {
    let context = context.into_inner();
    let mask = context.active_mask();
    if !matches!(
        destination.pointer_space_for_mask(mask)?,
        crate::runtime::PointerSpace::Local | crate::runtime::PointerSpace::Register
    ) {
        return Err(crate::abi::v2::EngineError::message(
            "frontend register result must address local or register storage",
        ));
    }
    let destination = destination.with_byte_storage_access_width(T::BYTE_LEN)?;
    Ok(crate::runtime::store_physical_ptr_warp(
        physical,
        &context,
        &destination,
        value.inner(),
        mask,
    )?)
}
