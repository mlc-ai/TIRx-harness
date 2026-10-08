use super::launch::LaunchSelection;
use super::operand::{PhysicalPtr, RuntimeBuffer};
use super::tensor_map::{
    tensor_map_has_magic, Fp4SharedLayout, RuntimeTensorMap, RuntimeTensorMapImage,
    TensorMapElementType, TensorMapFillMode, TENSOR_MAP_DESCRIPTOR_BYTES,
};
use super::tensor_map_registry::RuntimeTensorMapRegistry;
use crate::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32, profile_reset,
    profile_snapshot, AllocationId, BufferView, ExecutionStats, GlobalMemory, LaunchTopology,
    PhysicalAllocationId, PhysicalUninitializedReadReview, WarpValue, NUMSIM_ABI_VERSION,
};
use numsim_host_buffer::{HostByteBuffer, HostByteSource};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBool, PyBytes, PyBytesMethods, PyDict, PyFloat, PyInt, PyList, PyModule};
use std::fmt::Display;

type ProfileSnapshot = Vec<(&'static str, u64, u64)>;

struct PhaseRunStats {
    kernel_index: usize,
    name: String,
    topology: LaunchTopology,
    stats: ExecutionStats,
    profile: ProfileSnapshot,
}

struct PhaseUninitializedReadReview {
    kernel_index: usize,
    kernel_name: String,
    review: PhysicalUninitializedReadReview,
}

#[derive(Default)]
pub struct RunResultBuilder {
    aggregate: ExecutionStats,
    phases: Vec<PhaseRunStats>,
    uninitialized_read_reviews: Vec<PhaseUninitializedReadReview>,
}

impl RunResultBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_phase(
        &mut self,
        kernel_index: usize,
        name: &str,
        topology: LaunchTopology,
        stats: ExecutionStats,
        profile: ProfileSnapshot,
    ) {
        self.aggregate.task_count += stats.task_count;
        self.aggregate.completed_task_count += stats.completed_task_count;
        self.aggregate.poll_count += stats.poll_count;
        self.aggregate.normal_poll_count += stats.normal_poll_count;
        self.aggregate.poll_recheck_poll_count += stats.poll_recheck_poll_count;
        self.aggregate
            .poll_order
            .extend(stats.poll_order.iter().copied());
        self.aggregate.completion_pump_count += stats.completion_pump_count;
        self.aggregate.completion_operation_count += stats.completion_operation_count;
        self.aggregate.worker_count = self.aggregate.worker_count.max(stats.worker_count);
        self.aggregate.scheduling_domain_count += stats.scheduling_domain_count;
        self.phases.push(PhaseRunStats {
            kernel_index,
            name: name.to_string(),
            topology,
            stats,
            profile,
        });
    }

    pub fn record_uninitialized_read_reviews(
        &mut self,
        kernel_index: usize,
        kernel_name: &str,
        reviews: Vec<PhysicalUninitializedReadReview>,
    ) {
        self.uninitialized_read_reviews
            .extend(
                reviews
                    .into_iter()
                    .map(|review| PhaseUninitializedReadReview {
                        kernel_index,
                        kernel_name: kernel_name.to_string(),
                        review,
                    }),
            );
    }

    pub fn run_phase<F>(
        &mut self,
        py: Python<'_>,
        kernel_index: usize,
        name: &str,
        topology: LaunchTopology,
        run: F,
    ) -> PyResult<()>
    where
        F: Ungil + FnOnce() -> Result<ExecutionStats, String>,
    {
        profile_reset();
        let stats = py.detach(run).map_err(|error| {
            PyRuntimeError::new_err(format!(
                "NumSim kernel phase {kernel_index} failed: {error}"
            ))
        })?;
        self.record_phase(kernel_index, name, topology, stats, profile_snapshot());
        Ok(())
    }
}

pub fn required_item<'py>(inputs: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    inputs
        .get_item(name)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing NumSim binding field '{name}'")))
}

fn extract_scalar_value<'py>(
    inputs: &Bound<'py, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let scalars = required_item(inputs, "scalars")?;
    let descriptor = scalars
        .get_item(name)
        .map_err(|_| PyKeyError::new_err(format!("missing NumSim scalar binding '{name}'")))?;
    let dtype = descriptor.get_item("dtype")?.extract::<String>()?;
    if dtype != expected_dtype {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' requires dtype {expected_dtype}, got {dtype}"
        )));
    }
    descriptor.get_item("value")
}

pub fn extract_scalar_bool(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<bool> {
    let value = extract_scalar_value(inputs, name, expected_dtype)?;
    if !value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' with dtype {expected_dtype} requires a bool value"
        )));
    }
    value.extract::<bool>()
}

fn extract_signed_scalar(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<i64> {
    let value = extract_scalar_value(inputs, name, expected_dtype)?;
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' with dtype {expected_dtype} requires an integer value"
        )));
    }
    value.extract::<i64>().map_err(|_| {
        PyValueError::new_err(format!(
            "NumSim scalar '{name}' value is outside signed 64-bit range"
        ))
    })
}

fn extract_unsigned_scalar(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<u64> {
    let value = extract_scalar_value(inputs, name, expected_dtype)?;
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' with dtype {expected_dtype} requires an integer value"
        )));
    }
    value.extract::<u64>().map_err(|_| {
        PyValueError::new_err(format!(
            "NumSim scalar '{name}' value is outside unsigned 64-bit range"
        ))
    })
}

macro_rules! impl_signed_scalar_extractor {
    ($function:ident, $scalar_type:ty) => {
        pub fn $function(
            inputs: &Bound<'_, PyDict>,
            name: &str,
            expected_dtype: &str,
        ) -> PyResult<$scalar_type> {
            let value = extract_signed_scalar(inputs, name, expected_dtype)?;
            <$scalar_type>::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "NumSim scalar '{name}' value {value} is outside {} range",
                    expected_dtype,
                ))
            })
        }
    };
}

macro_rules! impl_unsigned_scalar_extractor {
    ($function:ident, $scalar_type:ty) => {
        pub fn $function(
            inputs: &Bound<'_, PyDict>,
            name: &str,
            expected_dtype: &str,
        ) -> PyResult<$scalar_type> {
            let value = extract_unsigned_scalar(inputs, name, expected_dtype)?;
            <$scalar_type>::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "NumSim scalar '{name}' value {value} is outside {} range",
                    expected_dtype,
                ))
            })
        }
    };
}

impl_signed_scalar_extractor!(extract_scalar_i8, i8);
impl_signed_scalar_extractor!(extract_scalar_i16, i16);
impl_signed_scalar_extractor!(extract_scalar_i32, i32);
impl_signed_scalar_extractor!(extract_scalar_i64, i64);
impl_unsigned_scalar_extractor!(extract_scalar_u8, u8);
impl_unsigned_scalar_extractor!(extract_scalar_u16, u16);
impl_unsigned_scalar_extractor!(extract_scalar_u32, u32);
impl_unsigned_scalar_extractor!(extract_scalar_u64, u64);

pub fn extract_scalar_f32(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<f32> {
    let value = extract_scalar_value(inputs, name, expected_dtype)?;
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
    {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' with dtype {expected_dtype} requires a numeric value"
        )));
    }
    let value = value.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!(
            "NumSim scalar '{name}' value cannot be represented as {expected_dtype}"
        ))
    })?;
    let maximum = match expected_dtype {
        "float16" => 65_504.0_f64,
        "bfloat16" | "float32" => f32::MAX as f64,
        _ => {
            return Err(PyValueError::new_err(format!(
                "NumSim scalar '{name}' has unsupported floating dtype {expected_dtype}"
            )));
        }
    };
    if value.is_finite() && value.abs() > maximum {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' value {value} is outside finite {expected_dtype} range"
        )));
    }
    let value = value as f32;
    Ok(match expected_dtype {
        "float16" => fp16_bits_to_f32(f32_to_fp16_bits(value)),
        "bfloat16" => bf16_bits_to_f32(f32_to_bf16_bits(value)),
        "float32" => value,
        _ => unreachable!(),
    })
}

pub fn extract_scalar_f64(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
) -> PyResult<f64> {
    if expected_dtype != "float64" {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' has unsupported floating dtype {expected_dtype}"
        )));
    }
    let value = extract_scalar_value(inputs, name, expected_dtype)?;
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
    {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' with dtype float64 requires a numeric value"
        )));
    }
    value.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!(
            "NumSim scalar '{name}' value cannot be represented as float64"
        ))
    })
}

pub fn extract_allocations(
    inputs: &Bound<'_, PyDict>,
    memory: &GlobalMemory,
    expected_abi_version: u32,
) -> PyResult<Vec<AllocationId>> {
    let abi_version = required_item(inputs, "numsim_abi_version")?.extract::<u32>()?;
    if abi_version != expected_abi_version {
        return Err(PyValueError::new_err(format!(
            "NumSim ABI {abi_version} does not match artifact ABI {expected_abi_version}"
        )));
    }
    let entries = required_item(inputs, "allocations")?;
    let allocation_count = entries.len()?;
    extract_output_allocations(inputs, allocation_count)?;
    let mut allocation_ids = Vec::with_capacity(allocation_count);
    for index in 0..allocation_count {
        let entry = entries.get_item(index)?;
        let byte_len = entry.get_item("byte_len")?.extract::<usize>()?;
        let write_through = entry.get_item("write_through")?.extract::<bool>()?;
        let validity_value = entry.get_item("validity")?;
        if write_through {
            let validity = validity_value.cast::<PyBytes>().map_err(|_| {
                PyValueError::new_err("write-through NumSim validity must be bytes")
            })?;
            if !validity.as_bytes().is_empty() {
                return Err(PyValueError::new_err(
                    "write-through NumSim allocations must be all-valid",
                ));
            }
            let buffer_value = entry.get_item("host_buffer")?;
            let buffer = PyBuffer::<u8>::get(&buffer_value)?;
            let owner = entry.get_item("host_owner")?.unbind();
            let host = HostByteBuffer::new(buffer, owner).map_err(PyValueError::new_err)?;
            if host.byte_len() != byte_len {
                return Err(PyValueError::new_err(format!(
                    "NumSim write-through allocation {index} declares {byte_len} bytes but its host buffer carries {}",
                    host.byte_len(),
                )));
            }
            let allocation = memory
                .allocate_from_host_bytes_all_valid(host)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            allocation_ids.push(allocation);
            continue;
        }
        let host_readonly = entry
            .cast::<PyDict>()
            .ok()
            .and_then(|entry| entry.get_item("host_readonly").ok().flatten())
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(false);
        if host_readonly {
            // A borrowed host array used as the allocation's immutable initial
            // bytes (stripes copy on write): no snapshot for an analysis.
            let validity = validity_value.cast::<PyBytes>().map_err(|_| {
                PyValueError::new_err("read-only host NumSim validity must be bytes")
            })?;
            if !validity.as_bytes().is_empty() {
                return Err(PyValueError::new_err(
                    "read-only host NumSim allocations must be all-valid",
                ));
            }
            let buffer_value = entry.get_item("host_buffer")?;
            let buffer = PyBuffer::<u8>::get(&buffer_value)?;
            let owner = entry.get_item("host_owner")?.unbind();
            let source = HostByteSource::new(buffer, owner).map_err(PyValueError::new_err)?;
            if source.byte_len() != byte_len {
                return Err(PyValueError::new_err(format!(
                    "NumSim read-only host allocation {index} declares {byte_len} bytes but its host buffer carries {}",
                    source.byte_len(),
                )));
            }
            let allocation = memory
                .allocate_from_immutable_bytes_all_valid(source)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            allocation_ids.push(allocation);
            continue;
        }
        let data = entry.get_item("data")?.extract::<PyBackedBytes>()?;
        if data.len() != byte_len {
            return Err(PyValueError::new_err(format!(
                "NumSim allocation {index} declares {byte_len} bytes but carries {}",
                data.len(),
            )));
        }
        let all_valid = validity_value
            .cast::<PyBytes>()
            .ok()
            .is_some_and(|validity| {
                validity.as_bytes().is_empty()
                    || (validity.as_bytes().len() == data.len()
                        && validity.as_bytes().iter().all(|value| *value == 1))
            });
        let allocation = if all_valid {
            memory.allocate_from_immutable_bytes_all_valid(data)
        } else {
            let validity = validity_value.extract::<Vec<u8>>()?;
            memory.allocate_from_bytes_with_validity(data.as_ref().to_vec(), validity)
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        allocation_ids.push(allocation);
    }
    for (index, &allocation) in allocation_ids.iter().enumerate() {
        let address = entries
            .get_item(index)?
            .get_item("host_address")?
            .extract::<u64>()?;
        memory
            .full_view(allocation)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .bind_observed_allocation_address(address)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    }
    rewrite_tensor_map_addresses(inputs, memory, &allocation_ids, false)?;
    Ok(allocation_ids)
}

fn rewrite_tensor_map_addresses(
    inputs: &Bound<'_, PyDict>,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
    restore_host: bool,
) -> PyResult<()> {
    let buffers = required_item(inputs, "buffers")?;
    let descriptor_names = buffers
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("NumSim buffers must be a mapping"))?
        .keys()
        .extract::<Vec<String>>()?;
    if descriptor_names.is_empty() {
        return Ok(());
    }
    let entries = required_item(inputs, "allocations")?;
    let mut host_ranges = Vec::with_capacity(allocation_ids.len());
    for index in 0..allocation_ids.len() {
        let entry = entries.get_item(index)?;
        let address = entry.get_item("host_address")?.extract::<u64>()?;
        let byte_len = entry.get_item("byte_len")?.extract::<usize>()?;
        host_ranges.push((address, byte_len));
    }

    for name in descriptor_names {
        let descriptor = buffers.get_item(&name).map_err(|_| {
            PyKeyError::new_err(format!(
                "missing NumSim TensorMap descriptor buffer '{name}'"
            ))
        })?;
        let allocation_index = descriptor.get_item("allocation")?.extract::<usize>()?;
        let allocation = allocation_ids.get(allocation_index).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "TensorMap descriptor buffer '{name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
        let data_offset = descriptor.get_item("data_offset")?.extract::<usize>()?;
        let shape = descriptor.get_item("shape")?.extract::<Vec<usize>>()?;
        let dtype = descriptor.get_item("dtype")?.extract::<String>()?;
        let itemsize = descriptor.get_item("itemsize")?.extract::<usize>()?;
        let byte_strides = descriptor
            .get_item("byte_strides")?
            .extract::<Vec<isize>>()?;
        if dtype == "float4_e2m1fn" {
            if !byte_strides.is_empty() {
                continue;
            }
        } else if byte_strides != contiguous_byte_strides(&shape, itemsize)? {
            continue;
        }
        let byte_len = logical_buffer_byte_len(
            &dtype,
            itemsize,
            &shape,
            "TensorMap descriptor candidate buffer",
        )?;
        if byte_len < TENSOR_MAP_DESCRIPTOR_BYTES {
            continue;
        }
        for byte_offset in (0..=byte_len - TENSOR_MAP_DESCRIPTOR_BYTES).step_by(128) {
            let view = memory
                .view(
                    allocation,
                    data_offset.checked_add(byte_offset).ok_or_else(|| {
                        PyValueError::new_err("TensorMap descriptor offset overflow")
                    })?,
                    TENSOR_MAP_DESCRIPTOR_BYTES,
                )
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            if !tensor_map_has_magic(
                memory
                    .read_bytes(&view, 63, 1)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?[0],
            ) {
                continue;
            }
            let mut image = match RuntimeTensorMapImage::read_candidate(memory, &view)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
            {
                Some(image) => image,
                None => continue,
            };
            if restore_host {
                if image.is_host_address() {
                    continue;
                }
                let runtime_allocation = AllocationId::from_u64(image.address_token());
                let runtime_index = allocation_ids
                    .iter()
                    .position(|candidate| *candidate == runtime_allocation)
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "TensorMap descriptor references an unknown runtime allocation",
                        )
                    })?;
                image
                    .restore_host_address(host_ranges[runtime_index].0)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
            } else {
                if !image.is_host_address() {
                    continue;
                }
                let address = image
                    .address_token()
                    .checked_add(u64::try_from(image.base_byte_offset()).map_err(|_| {
                        PyValueError::new_err("TensorMap base offset does not fit u64")
                    })?)
                    .ok_or_else(|| PyValueError::new_err("TensorMap host address overflow"))?;
                let (runtime_index, byte_offset) = host_ranges
                    .iter()
                    .enumerate()
                    .find_map(|(index, (start, byte_len))| {
                        let end = start.checked_add(u64::try_from(*byte_len).ok()?)?;
                        (address >= *start && address < end)
                            .then(|| (index, usize::try_from(address - *start).ok()))
                            .and_then(|(index, offset)| offset.map(|offset| (index, offset)))
                    })
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "TensorMap host address {address:#x} is absent from prepared allocations"
                        ))
                    })?;
                image.relocate(allocation_ids[runtime_index], byte_offset);
            }
            image
                .write(memory, &view)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
    }
    Ok(())
}

pub fn extract_shape_extent(
    inputs: &Bound<'_, PyDict>,
    scalar_name: &str,
    sources: &[(&str, usize)],
) -> PyResult<usize> {
    let descriptors = required_item(inputs, "buffers")?;
    let mut resolved = None;
    for &(buffer_name, axis) in sources {
        let descriptor = descriptors.get_item(buffer_name).map_err(|_| {
            PyKeyError::new_err(format!(
                "missing NumSim buffer descriptor '{buffer_name}' for runtime shape '{scalar_name}'"
            ))
        })?;
        let shape = descriptor.get_item("shape")?.extract::<Vec<usize>>()?;
        let extent = shape.get(axis).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "buffer '{buffer_name}' has no axis {axis} for runtime shape '{scalar_name}'"
            ))
        })?;
        if let Some(previous) = resolved {
            if previous != extent {
                return Err(PyValueError::new_err(format!(
                    "runtime shape '{scalar_name}' disagrees across bindings: {previous} != {extent} at '{buffer_name}'[{axis}]"
                )));
            }
        } else {
            resolved = Some(extent);
        }
    }
    resolved.ok_or_else(|| {
        PyValueError::new_err(format!(
            "runtime shape '{scalar_name}' has no buffer descriptor source"
        ))
    })
}

pub fn validate_shape_scalar<T>(name: &str, value: T, extent: usize) -> PyResult<()>
where
    T: Copy + Display + TryInto<usize>,
    T::Error: Display,
{
    let scalar_extent = value.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "NumSim scalar '{name}' value {value} cannot be used as a buffer extent"
        ))
    })?;
    if scalar_extent != extent {
        return Err(PyValueError::new_err(format!(
            "NumSim scalar '{name}' value {value} disagrees with bound buffer extent {extent}"
        )));
    }
    Ok(())
}

pub fn contiguous_byte_strides(shape: &[usize], itemsize: usize) -> PyResult<Vec<isize>> {
    let mut result = vec![0_isize; shape.len()];
    let mut stride = itemsize;
    for axis in (0..shape.len()).rev() {
        result[axis] = isize::try_from(stride)
            .map_err(|_| PyValueError::new_err("buffer byte stride exceeds isize"))?;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| PyValueError::new_err("buffer byte stride overflow"))?;
    }
    Ok(result)
}

fn logical_buffer_byte_len(
    dtype: &str,
    itemsize: usize,
    shape: &[usize],
    label: &str,
) -> PyResult<usize> {
    let elements = shape.iter().try_fold(1_usize, |product, extent| {
        product
            .checked_mul(*extent)
            .ok_or_else(|| PyValueError::new_err(format!("{label} element count overflow")))
    })?;
    if dtype == "float4_e2m1fn" {
        return elements
            .checked_add(1)
            .map(|value| value / 2)
            .ok_or_else(|| PyValueError::new_err(format!("{label} byte length overflow")));
    }
    elements
        .checked_mul(itemsize)
        .ok_or_else(|| PyValueError::new_err(format!("{label} byte length overflow")))
}

pub fn extract_pointer(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
    expected_itemsize: usize,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<PhysicalPtr> {
    let descriptors = required_item(inputs, "buffers")?;
    let descriptor = descriptors.get_item(name).map_err(|_| {
        PyKeyError::new_err(format!("missing NumSim pointer buffer descriptor '{name}'"))
    })?;
    let allocation_index = descriptor.get_item("allocation")?.extract::<usize>()?;
    let allocation = allocation_ids
        .get(allocation_index)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "pointer '{name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
    let data_offset = descriptor.get_item("data_offset")?.extract::<usize>()?;
    let dtype = descriptor.get_item("dtype")?.extract::<String>()?;
    let itemsize = descriptor.get_item("itemsize")?.extract::<usize>()?;
    let shape = descriptor.get_item("shape")?.extract::<Vec<usize>>()?;
    let byte_strides = descriptor
        .get_item("byte_strides")?
        .extract::<Vec<isize>>()?;
    let expected_byte_strides = contiguous_byte_strides(&shape, expected_itemsize)?;
    if dtype != expected_dtype
        || itemsize != expected_itemsize
        || byte_strides != expected_byte_strides.as_slice()
    {
        return Err(PyValueError::new_err(format!(
            "pointer '{name}' requires contiguous {expected_dtype} elements of {expected_itemsize} bytes, got dtype={dtype}, itemsize={itemsize}, shape={shape:?}, byte_strides={byte_strides:?}"
        )));
    }
    let byte_len = logical_buffer_byte_len(&dtype, expected_itemsize, &shape, "pointer")?;
    let view = memory
        .view(allocation, data_offset, byte_len)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PhysicalPtr::new(
        RuntimeBuffer::Global(view),
        WarpValue::splat(0_i64),
        expected_itemsize,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn extract_buffer(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    expected_dtype: &str,
    expected_itemsize: usize,
    expected_shape: &[usize],
    view_byte_offset: usize,
    physical_byte_len: Option<usize>,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<BufferView> {
    let descriptors = required_item(inputs, "buffers")?;
    let descriptor = descriptors
        .get_item(name)
        .map_err(|_| PyKeyError::new_err(format!("missing NumSim buffer descriptor '{name}'")))?;
    let allocation_index = descriptor.get_item("allocation")?.extract::<usize>()?;
    let allocation = allocation_ids
        .get(allocation_index)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "buffer '{name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
    let data_offset = descriptor.get_item("data_offset")?.extract::<usize>()?;
    let dtype = descriptor.get_item("dtype")?.extract::<String>()?;
    let itemsize = descriptor.get_item("itemsize")?.extract::<usize>()?;
    let shape = descriptor.get_item("shape")?.extract::<Vec<usize>>()?;
    let byte_strides = descriptor
        .get_item("byte_strides")?
        .extract::<Vec<isize>>()?;
    let packed_float4 = expected_dtype == "float4_e2m1fn";
    let expected_byte_strides = if packed_float4 {
        Vec::new()
    } else {
        contiguous_byte_strides(expected_shape, expected_itemsize)?
    };
    let descriptor_matches = if packed_float4 {
        dtype == expected_dtype && itemsize == expected_itemsize && byte_strides.is_empty()
    } else {
        dtype == expected_dtype
            && itemsize == expected_itemsize
            && shape == expected_shape
            && byte_strides == expected_byte_strides.as_slice()
    };
    if !descriptor_matches {
        return Err(PyValueError::new_err(format!(
            "buffer '{name}' requires contiguous {expected_dtype} ({expected_itemsize} bytes) shape {expected_shape:?} with byte strides {expected_byte_strides:?}, got dtype={dtype}, itemsize={itemsize}, shape={shape:?}, byte_strides={byte_strides:?}"
        )));
    }
    let logical_byte_len =
        logical_buffer_byte_len(expected_dtype, expected_itemsize, expected_shape, "buffer")?;
    let byte_len = physical_byte_len.unwrap_or(logical_byte_len);
    let data_offset = data_offset
        .checked_add(view_byte_offset)
        .ok_or_else(|| PyValueError::new_err("buffer data offset overflow"))?;
    memory
        .view(allocation, data_offset, byte_len)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub fn extract_buffer_alias(
    inputs: &Bound<'_, PyDict>,
    base_name: &str,
    view_byte_offset: usize,
    physical_byte_len: Option<usize>,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<BufferView> {
    let descriptors = required_item(inputs, "buffers")?;
    let descriptor = descriptors.get_item(base_name).map_err(|_| {
        PyKeyError::new_err(format!(
            "missing NumSim base buffer descriptor '{base_name}' for global alias"
        ))
    })?;
    let allocation_index = descriptor.get_item("allocation")?.extract::<usize>()?;
    let allocation = allocation_ids
        .get(allocation_index)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "global alias base '{base_name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
    let data_offset = descriptor.get_item("data_offset")?.extract::<usize>()?;
    let dtype = descriptor.get_item("dtype")?.extract::<String>()?;
    let itemsize = descriptor.get_item("itemsize")?.extract::<usize>()?;
    let shape = descriptor.get_item("shape")?.extract::<Vec<usize>>()?;
    let physical_byte_len = match physical_byte_len {
        Some(byte_len) => byte_len,
        None => logical_buffer_byte_len(
            &dtype,
            itemsize,
            &shape,
            &format!("global alias base '{base_name}'"),
        )?,
    };
    let data_offset = data_offset
        .checked_add(view_byte_offset)
        .ok_or_else(|| PyValueError::new_err("global alias data offset overflow"))?;
    memory
        .view(allocation, data_offset, physical_byte_len)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
pub fn extract_tensor_map(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<RuntimeTensorMap> {
    let buffers = required_item(inputs, "buffers")?;
    let descriptor = buffers.get_item(name).map_err(|_| {
        PyKeyError::new_err(format!(
            "missing NumSim TensorMap descriptor buffer '{name}'"
        ))
    })?;
    let allocation_index = descriptor.get_item("allocation")?.extract::<usize>()?;
    let allocation = allocation_ids
        .get(allocation_index)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "TensorMap '{name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
    let data_offset = descriptor.get_item("data_offset")?.extract::<usize>()?;
    let view = memory
        .view(allocation, data_offset, TENSOR_MAP_DESCRIPTOR_BYTES)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    RuntimeTensorMapImage::read(memory, &view)
        .and_then(|image| {
            image.materialize(
                memory,
                usize::from(memory.read_bytes(&view, 59, 1)?[0] & 0b111),
            )
        })
        .map_err(|error| PyValueError::new_err(format!("TensorMap '{name}': {error}")))
}
#[allow(clippy::too_many_arguments)]
pub fn extract_or_build_implicit_tensor_map(
    inputs: &Bound<'_, PyDict>,
    name: &str,
    base_name: &str,
    base_byte_offset: i128,
    dtype: &str,
    tma_dtype: Option<&str>,
    fp4_shared_layout: Option<&str>,
    global_shape: &[usize],
    global_strides: &[usize],
    box_shape: &[usize],
    element_strides: &[usize],
    interleave: Option<&str>,
    swizzle: Option<&str>,
    fill_mode: Option<&str>,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<RuntimeTensorMap> {
    let interleave_bytes = match interleave {
        None | Some("none") => None,
        Some("16B") => Some(16),
        Some("32B") => Some(32),
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "unsupported TensorMap interleave '{value}'"
            )))
        }
    };

    let buffers = required_item(inputs, "buffers")?;
    if buffers.contains(name)? {
        return extract_tensor_map(inputs, name, memory, allocation_ids);
    }
    let base = buffers.get_item(base_name).map_err(|_| {
        PyKeyError::new_err(format!(
            "implicit TensorMap '{name}' references missing base buffer '{base_name}'"
        ))
    })?;
    let allocation_index = base.get_item("allocation")?.extract::<usize>()?;
    let allocation = allocation_ids
        .get(allocation_index)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "implicit TensorMap '{name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
    let base_data_offset = base.get_item("data_offset")?.extract::<usize>()?;
    let magnitude = usize::try_from(base_byte_offset.unsigned_abs()).map_err(|_| {
        PyValueError::new_err(format!(
            "implicit TensorMap '{name}' base byte offset {base_byte_offset} is outside usize"
        ))
    })?;
    let data_offset = if base_byte_offset >= 0 {
        base_data_offset.checked_add(magnitude)
    } else {
        base_data_offset.checked_sub(magnitude)
    }
    .ok_or_else(|| {
        PyValueError::new_err(format!(
            "implicit TensorMap '{name}' base byte offset {base_byte_offset} escapes its allocation"
        ))
    })?;
    let allocation_len = memory
        .allocation_len(allocation)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let byte_len = allocation_len.checked_sub(data_offset).ok_or_else(|| {
        PyValueError::new_err(format!(
            "implicit TensorMap '{name}' base byte offset {data_offset} exceeds its {allocation_len}-byte allocation"
        ))
    })?;
    let view = memory
        .view(allocation, data_offset, byte_len)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let base_type = tensor_map_element_type(dtype).ok_or_else(|| {
        PyValueError::new_err(format!(
            "implicit TensorMap '{name}' has unsupported dtype '{dtype}'"
        ))
    })?;
    let element_type = match tma_dtype {
        None | Some("none") => base_type,
        Some("tf32" | "tfloat32") if base_type == TensorMapElementType::F32 => {
            TensorMapElementType::Tf32
        }
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' has invalid TMA dtype '{value}' for base dtype '{dtype}'"
            )));
        }
    };
    let fp4_shared_layout = match (element_type, fp4_shared_layout) {
        (TensorMapElementType::Float4E2M1Fn, Some("align8_packed")) => {
            Some(Fp4SharedLayout::Align8Packed)
        }
        (TensorMapElementType::Float4E2M1Fn, Some("align16_padded")) => {
            Some(Fp4SharedLayout::Align16Padded)
        }
        (TensorMapElementType::Float4E2M1Fn, Some(value)) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' has unsupported FP4 shared layout '{value}'"
            )));
        }
        (TensorMapElementType::Float4E2M1Fn, None) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' is missing its FP4 shared layout"
            )));
        }
        (_, Some(value)) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' has FP4 shared layout '{value}' for non-FP4 elements"
            )));
        }
        (_, None) => None,
    };
    let swizzle_bytes = match swizzle {
        None | Some("none") => None,
        Some("32B") => Some(32_usize),
        Some("64B") => Some(64_usize),
        Some("96B") => Some(96_usize),
        Some("128B") => Some(128_usize),
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' has unsupported swizzle '{value}'"
            )));
        }
    };
    let fill_mode = match fill_mode {
        None | Some("none" | "zero") => TensorMapFillMode::Zero,
        Some("nan") => TensorMapFillMode::OobNan,
        Some(value) => {
            return Err(PyValueError::new_err(format!(
                "implicit TensorMap '{name}' has unsupported fill mode '{value}'"
            )));
        }
    };
    RuntimeTensorMap::new_with_interleave(
        view,
        global_shape.to_vec(),
        global_strides.to_vec(),
        box_shape.to_vec(),
        element_strides.to_vec(),
        element_type.bits(),
        element_type,
        fp4_shared_layout,
        swizzle_bytes,
        fill_mode,
        interleave_bytes,
    )
    .map_err(|error| PyValueError::new_err(format!("implicit TensorMap '{name}': {error}")))
}

pub fn extract_tensor_map_descriptor_registry(
    inputs: &Bound<'_, PyDict>,
    memory: &GlobalMemory,
    allocation_ids: &[AllocationId],
    descriptor_storage_names: &[&str],
    parameter_tensor_maps: &[(&str, RuntimeTensorMap)],
) -> PyResult<RuntimeTensorMapRegistry> {
    let registry = RuntimeTensorMapRegistry::new();
    let buffers = required_item(inputs, "buffers")?;
    for &storage_name in descriptor_storage_names {
        if !buffers.contains(storage_name)? {
            continue;
        }
        let storage = buffers.get_item(storage_name)?;
        let itemsize = storage.get_item("itemsize")?.extract::<usize>()?;
        let shape = storage.get_item("shape")?.extract::<Vec<usize>>()?;
        let byte_strides = storage.get_item("byte_strides")?.extract::<Vec<isize>>()?;
        let contiguous_stride = isize::try_from(itemsize)
            .map_err(|_| PyValueError::new_err("TensorMap descriptor itemsize is too large"))?;
        if shape.len() != 1 || byte_strides != [contiguous_stride] {
            continue;
        }
        let allocation_index = storage.get_item("allocation")?.extract::<usize>()?;
        let allocation = allocation_ids.get(allocation_index).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "TensorMap descriptor storage '{storage_name}' references allocation {allocation_index}, but only {} exist",
                allocation_ids.len()
            ))
        })?;
        let data_offset = storage.get_item("data_offset")?.extract::<usize>()?;
        let storage_byte_len = shape[0]
            .checked_mul(itemsize)
            .ok_or_else(|| PyValueError::new_err("TensorMap descriptor storage size overflow"))?;
        if storage_byte_len < TENSOR_MAP_DESCRIPTOR_BYTES {
            continue;
        }
        for byte_offset in (0..=storage_byte_len - TENSOR_MAP_DESCRIPTOR_BYTES).step_by(128) {
            let absolute_offset = data_offset.checked_add(byte_offset).ok_or_else(|| {
                PyValueError::new_err("TensorMap descriptor data offset overflow")
            })?;
            let view = memory
                .view(allocation, absolute_offset, TENSOR_MAP_DESCRIPTOR_BYTES)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            if !tensor_map_has_magic(
                memory
                    .read_bytes(&view, 63, 1)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?[0],
            ) {
                continue;
            }
            let image = match RuntimeTensorMapImage::read_candidate(memory, &view)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
            {
                Some(image) => image,
                None => continue,
            };
            let rank = usize::from(
                memory
                    .read_bytes(&view, 59, 1)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?[0]
                    & 0b111,
            );
            let tensor_map = match image.materialize(memory, rank) {
                Ok(tensor_map) => tensor_map,
                Err(_) => continue,
            };
            registry
                .bind_host(memory, view, &tensor_map)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
    }

    for &(name, ref tensor_map) in parameter_tensor_maps {
        let allocation = memory
            .allocate_zeroed(TENSOR_MAP_DESCRIPTOR_BYTES)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let view = memory
            .view(allocation, 0, TENSOR_MAP_DESCRIPTOR_BYTES)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        // TensorMap parameters are materialized into simulator-owned global
        // storage. Give that storage an invocation-local virtual address so
        // address_of transports ordinary bits that the memory map can resolve.
        let virtual_address = 0xffff_8000_0000_0000_u64
            .checked_add(allocation.as_u64().checked_mul(0x1000).ok_or_else(|| {
                PyValueError::new_err("TensorMap parameter virtual address overflow")
            })?)
            .ok_or_else(|| PyValueError::new_err("TensorMap parameter virtual address overflow"))?;
        view.bind_observed_allocation_address(virtual_address)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(view.clone()),
            WarpValue::splat(0_i64),
            1,
        );
        registry
            .bind_parameter(name, memory, view, pointer, tensor_map)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    }

    Ok(registry)
}

fn tensor_map_element_type(dtype: &str) -> Option<TensorMapElementType> {
    Some(match dtype {
        "float4_e2m1fn" => TensorMapElementType::Float4E2M1Fn,
        "bool" => TensorMapElementType::Bool,
        "int8" => TensorMapElementType::I8,
        "uint8" => TensorMapElementType::U8,
        "float8_e4m3fn" => TensorMapElementType::Float8E4M3Fn,
        "float8_e8m0fnu" => TensorMapElementType::Float8E8M0Fnu,
        "int16" => TensorMapElementType::I16,
        "uint16" => TensorMapElementType::U16,
        "float16" => TensorMapElementType::F16,
        "bfloat16" => TensorMapElementType::Bf16,
        "int32" => TensorMapElementType::I32,
        "uint32" => TensorMapElementType::U32,
        "float32" => TensorMapElementType::F32,
        "tf32" => TensorMapElementType::Tf32,
        "float32_ftz" => TensorMapElementType::F32Ftz,
        "tf32_ftz" => TensorMapElementType::Tf32Ftz,
        "float64" => TensorMapElementType::F64,
        "int64" => TensorMapElementType::I64,
        "uint64" => TensorMapElementType::U64,
        "uint32x2" => TensorMapElementType::U32x2,
        _ => return None,
    })
}

pub fn extract_optional_ids(
    subset: &Bound<'_, PyAny>,
    key: &str,
    upper_bound: usize,
) -> PyResult<Option<Vec<usize>>> {
    let value = subset.get_item(key)?;
    if value.is_none() {
        return Ok(None);
    }
    let mut ids = value.extract::<Vec<usize>>()?;
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PyValueError::new_err(format!(
            "NumSim subset '{key}' contains duplicate IDs"
        )));
    }
    if let Some(id) = ids.iter().find(|&&id| id >= upper_bound) {
        return Err(PyValueError::new_err(format!(
            "NumSim subset '{key}' ID {id} is outside [0, {upper_bound})"
        )));
    }
    Ok(Some(ids))
}

pub fn extract_selection(
    subset: Option<&Bound<'_, PyAny>>,
    topology: LaunchTopology,
) -> PyResult<LaunchSelection> {
    let Some(subset) = subset else {
        return Ok(LaunchSelection::default());
    };
    let selection = LaunchSelection::new(
        extract_optional_ids(subset, "cluster_ids", topology.clusters())?,
        extract_optional_ids(subset, "cta_ids", topology.cta_count())?,
    );
    selection
        .validate(topology)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(selection)
}

pub fn extract_phase_selection(
    subset: Option<&Bound<'_, PyAny>>,
    kernel_index: usize,
    kernel_count: usize,
    topology: LaunchTopology,
) -> PyResult<LaunchSelection> {
    if kernel_count == 0 || kernel_index >= kernel_count {
        return Err(PyValueError::new_err(format!(
            "NumSim kernel phase {kernel_index} is outside [0, {kernel_count})"
        )));
    }
    let Some(subset) = subset else {
        return Ok(LaunchSelection::default());
    };
    if let Ok(phases) = subset.cast::<PyList>() {
        if phases.len() != kernel_count {
            return Err(PyValueError::new_err(format!(
                "NumSim phase subset list has length {}, expected {kernel_count}",
                phases.len()
            )));
        }
        let phase = phases.get_item(kernel_index)?;
        return if phase.is_none() {
            Ok(LaunchSelection::default())
        } else {
            extract_selection(Some(&phase), topology)
        };
    }
    if kernel_count != 1 {
        return Err(PyValueError::new_err(
            "NumSim multi-kernel launches require one subset entry per phase",
        ));
    }
    extract_selection(Some(subset), topology)
}

pub fn extract_output_allocations(
    inputs: &Bound<'_, PyDict>,
    allocation_count: usize,
) -> PyResult<Vec<usize>> {
    let mut output_allocations =
        required_item(inputs, "output_allocations")?.extract::<Vec<usize>>()?;
    output_allocations.sort_unstable();
    output_allocations.dedup();
    if let Some(index) = output_allocations
        .iter()
        .find(|&&index| index >= allocation_count)
    {
        return Err(PyValueError::new_err(format!(
            "NumSim output allocation {index} is outside [0, {allocation_count})"
        )));
    }
    Ok(output_allocations)
}

/// The allocations the kernel may write (`written_allocations`: the returned
/// buffers and the bases of the returned TensorMaps), for an analysis that
/// shadows written global memory. `output_allocations` also names every
/// TensorMap descriptor's base and storage, which execution hands back to the
/// host; a payload without the narrower key falls back to it.
pub fn extract_written_allocations(
    inputs: &Bound<'_, PyDict>,
    allocation_count: usize,
) -> PyResult<Vec<usize>> {
    let Some(written) = inputs.get_item("written_allocations")? else {
        return extract_output_allocations(inputs, allocation_count);
    };
    let mut written = written.extract::<Vec<usize>>()?;
    written.sort_unstable();
    written.dedup();
    if let Some(index) = written.iter().find(|&&index| index >= allocation_count) {
        return Err(PyValueError::new_err(format!(
            "NumSim written allocation {index} is outside [0, {allocation_count})"
        )));
    }
    Ok(written)
}

pub fn extract_written_allocation_ids(
    inputs: &Bound<'_, PyDict>,
    allocation_ids: &[AllocationId],
) -> PyResult<Vec<PhysicalAllocationId>> {
    extract_written_allocations(inputs, allocation_ids.len()).map(|indices| {
        indices
            .into_iter()
            .map(|index| allocation_ids[index].into())
            .collect()
    })
}

fn topology_to_py<'py>(py: Python<'py>, topology: LaunchTopology) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("clusters", topology.clusters())?;
    result.set_item("ctas_per_cluster", topology.ctas_per_cluster())?;
    result.set_item("warps_per_cta", topology.warps_per_cta())?;
    result.set_item("warp_count", topology.warp_count())?;
    Ok(result)
}

fn profile_to_py<'py>(py: Python<'py>, profile: &ProfileSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    for &(name, count, nanos) in profile {
        let entry = PyDict::new(py);
        entry.set_item("count", count)?;
        entry.set_item("nanos", nanos)?;
        result.set_item(name, entry)?;
    }
    Ok(result)
}

fn execution_stats_to_py<'py>(
    py: Python<'py>,
    stats: &ExecutionStats,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("task_count", stats.task_count)?;
    result.set_item("completed_task_count", stats.completed_task_count)?;
    result.set_item("poll_count", stats.poll_count)?;
    result.set_item("normal_poll_count", stats.normal_poll_count)?;
    result.set_item("poll_recheck_poll_count", stats.poll_recheck_poll_count)?;
    result.set_item("poll_order", &stats.poll_order)?;
    result.set_item("completion_pump_count", stats.completion_pump_count)?;
    result.set_item(
        "completion_operation_count",
        stats.completion_operation_count,
    )?;
    result.set_item("worker_count", stats.worker_count)?;
    result.set_item("scheduling_domain_count", stats.scheduling_domain_count)?;
    Ok(result)
}

fn phase_stats_to_py<'py>(py: Python<'py>, phase: &PhaseRunStats) -> PyResult<Bound<'py, PyDict>> {
    let result = execution_stats_to_py(py, &phase.stats)?;
    result.set_item("kernel_index", phase.kernel_index)?;
    result.set_item("name", &phase.name)?;
    result.set_item("topology", topology_to_py(py, phase.topology)?)?;
    result.set_item("profile", profile_to_py(py, &phase.profile)?)?;
    #[cfg(feature = "profile")]
    result.set_item(
        "executed_instruction_variants",
        instruction_variants_to_py(py, &phase.stats)?,
    )?;
    Ok(result)
}

#[cfg(feature = "profile")]
pub(crate) fn instruction_variants_to_py<'py>(
    py: Python<'py>,
    stats: &ExecutionStats,
) -> PyResult<Bound<'py, PyList>> {
    let result = PyList::empty(py);
    for &(site_id, variant) in &stats.executed_instruction_variants {
        let entry = PyDict::new(py);
        entry.set_item("site_id", site_id)?;
        entry.set_item("variant", variant)?;
        result.append(entry)?;
    }
    Ok(result)
}

fn aggregate_stats_to_py<'py>(
    py: Python<'py>,
    builder: &RunResultBuilder,
) -> PyResult<Bound<'py, PyDict>> {
    let result = execution_stats_to_py(py, &builder.aggregate)?;
    let phases = PyList::empty(py);
    for phase in &builder.phases {
        phases.append(phase_stats_to_py(py, phase)?)?;
    }
    result.set_item("kernels", phases)?;
    Ok(result)
}

fn diagnostics_to_py<'py>(
    py: Python<'py>,
    builder: &RunResultBuilder,
) -> PyResult<Bound<'py, PyList>> {
    let diagnostics = PyList::empty(py);
    for phase in &builder.uninitialized_read_reviews {
        let physical = phase.review;
        let item = uninitialized_read_review_to_py(py, physical)?;
        item.set_item("kernel_index", phase.kernel_index)?;
        item.set_item("kernel_name", &phase.kernel_name)?;
        diagnostics.append(item)?;
    }
    Ok(diagnostics)
}

fn uninitialized_read_review_to_py<'py>(
    py: Python<'py>,
    physical: PhysicalUninitializedReadReview,
) -> PyResult<Bound<'py, PyDict>> {
    let review = physical.review();
    let item = PyDict::new(py);
    if let Some(source) = review.source() {
        item.set_item("kernel_index", source.kernel_index)?;
        item.set_item("source_op_id", source.source_op_id)?;
        item.set_item("global_warp_id", source.global_warp_id)?;
    }
    item.set_item("status", "review")?;
    item.set_item("kind", "uninitialized_read")?;
    item.set_item(
        "message",
        format!(
            "{} memory {} was materialized with zero",
            physical.space(),
            review,
        ),
    )?;
    item.set_item("space", physical.space())?;
    item.set_item("allocation", review.allocation().as_u64())?;
    item.set_item("byte_offset", review.byte_offset())?;
    item.set_item("byte_len", review.byte_len())?;
    item.set_item(
        "first_uninitialized_byte",
        review.first_uninitialized_byte(),
    )?;
    Ok(item)
}

pub(crate) fn annotate_analysis_uninitialized_read_reviews(
    py: Python<'_>,
    result: Py<PyAny>,
    reviews: Vec<PhysicalUninitializedReadReview>,
) -> PyResult<Py<PyAny>> {
    if reviews.is_empty() {
        return Ok(result);
    }
    let payload = result.bind(py).cast::<PyDict>()?;
    let advisories = if let Some(value) = payload.get_item("advisories")? {
        value.cast::<PyList>()?.clone()
    } else {
        let value = PyList::empty(py);
        payload.set_item("advisories", &value)?;
        value
    };
    for review in reviews {
        advisories.append(uninitialized_read_review_to_py(py, review)?)?;
    }
    let verdict = payload
        .get_item("verdict")?
        .expect("native analysis payload has a verdict")
        .extract::<String>()?;
    if verdict == "clean" {
        payload.set_item("verdict", "review")?;
    }
    Ok(result)
}

fn output_allocation_bytes<'py>(
    py: Python<'py>,
    global: &GlobalMemory,
    allocation_ids: &[AllocationId],
    output_allocations: &[usize],
) -> PyResult<Bound<'py, PyList>> {
    let result = PyList::empty(py);
    for (index, allocation) in allocation_ids.iter().copied().enumerate() {
        if output_allocations.binary_search(&index).is_ok() {
            if global
                .allocation_is_write_through(allocation)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            {
                result.append(py.None())?;
                continue;
            }
            let bytes = global
                .snapshot_allocation_bytes(allocation)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            result.append(PyBytes::new(py, &bytes))?;
        } else {
            result.append(py.None())?;
        }
    }
    Ok(result)
}

fn allocation_state_to_py<'py>(
    py: Python<'py>,
    global: &GlobalMemory,
    allocation_ids: &[AllocationId],
) -> PyResult<Bound<'py, PyList>> {
    let result = PyList::empty(py);
    for allocation in allocation_ids.iter().copied() {
        let state = PyDict::new(py);
        let data = global
            .snapshot_allocation_bytes(allocation)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let validity = global
            .snapshot_allocation_validity(allocation)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        state.set_item("data", PyBytes::new(py, &data))?;
        state.set_item("validity", PyBytes::new(py, &validity))?;
        result.append(state)?;
    }
    Ok(result)
}

pub fn build_run_result(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    builder: RunResultBuilder,
    global: &GlobalMemory,
    allocation_ids: &[AllocationId],
    output_allocations: &[usize],
    include_allocation_state: bool,
) -> PyResult<Py<PyAny>> {
    rewrite_tensor_map_addresses(inputs, global, allocation_ids, true)?;
    let result = PyDict::new(py);
    result.set_item("outputs", PyDict::new(py))?;
    result.set_item(
        "allocation_bytes",
        output_allocation_bytes(py, global, allocation_ids, output_allocations)?,
    )?;
    result.set_item("diagnostics", diagnostics_to_py(py, &builder)?)?;
    result.set_item("stats", aggregate_stats_to_py(py, &builder)?)?;
    if include_allocation_state {
        result.set_item(
            "allocation_state",
            allocation_state_to_py(py, global, allocation_ids)?,
        )?;
    }
    Ok(result.into_any().unbind())
}

#[allow(clippy::too_many_arguments)]
pub fn build_artifact_metadata_with_build_identity(
    py: Python<'_>,
    cache_key: &str,
    abi_version: u32,
    engine_hash: &str,
    build_identity_json: &str,
    kernel_names: &[&str],
    topology_dimensions: &[(usize, usize, usize)],
) -> PyResult<Py<PyAny>> {
    if abi_version != NUMSIM_ABI_VERSION {
        return Err(PyValueError::new_err(format!(
            "generated NumSim ABI {abi_version} does not match engine ABI {NUMSIM_ABI_VERSION}"
        )));
    }
    if kernel_names.len() != topology_dimensions.len() {
        return Err(PyValueError::new_err(format!(
            "NumSim metadata has {} kernel names but {} topologies",
            kernel_names.len(),
            topology_dimensions.len(),
        )));
    }

    let topologies = PyList::empty(py);
    let mut warp_counts = Vec::with_capacity(topology_dimensions.len());
    let mut total_warp_count = 0_usize;
    for &(clusters, ctas_per_cluster, warps_per_cta) in topology_dimensions {
        let topology = LaunchTopology::new(clusters, ctas_per_cluster, warps_per_cta)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        warp_counts.push(topology.warp_count());
        total_warp_count = total_warp_count
            .checked_add(topology.warp_count())
            .ok_or_else(|| PyValueError::new_err("NumSim metadata warp count overflow"))?;

        let value = PyDict::new(py);
        value.set_item("clusters", clusters)?;
        value.set_item("ctas_per_cluster", ctas_per_cluster)?;
        value.set_item("warps_per_cta", warps_per_cta)?;
        topologies.append(value)?;
    }

    let result = PyDict::new(py);
    result.set_item("cache_key", cache_key)?;
    result.set_item("numsim_abi_version", NUMSIM_ABI_VERSION)?;
    let parsed_identity = PyModule::import(py, "json")?
        .getattr("loads")?
        .call1((build_identity_json,))?;
    if !parsed_identity.is_instance_of::<PyDict>() {
        return Err(PyValueError::new_err(
            "NumSim build identity JSON must decode to an object",
        ));
    }
    let identity_engine_hash = parsed_identity
        .get_item("engine_hash")?
        .extract::<String>()
        .map_err(|_| {
            PyValueError::new_err("NumSim build identity must contain a string engine_hash")
        })?;
    if identity_engine_hash != engine_hash {
        return Err(PyValueError::new_err(format!(
            "NumSim engine_hash disagrees with build_identity.engine_hash: \
             engine_hash={engine_hash:?}, build_identity={identity_engine_hash:?}",
        )));
    }
    result.set_item("engine_hash", engine_hash)?;
    result.set_item("build_identity", parsed_identity)?;
    result.set_item("kernel_count", kernel_names.len())?;
    result.set_item("warp_count", total_warp_count)?;
    result.set_item("warp_counts", warp_counts)?;
    result.set_item("kernel_names", kernel_names)?;
    result.set_item("topologies", topologies)?;
    Ok(result.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_task_stats() -> ExecutionStats {
        ExecutionStats {
            task_count: 1,
            completed_task_count: 1,
            poll_count: 1,
            normal_poll_count: 1,
            poll_recheck_poll_count: 0,
            poll_order: vec![0],
            completion_pump_count: 0,
            completion_operation_count: 0,
            worker_count: 1,
            scheduling_domain_count: 1,
            #[cfg(feature = "profile")]
            executed_instruction_variants: Vec::new(),
        }
    }

    #[test]
    fn output_allocation_extraction_sorts_deduplicates_and_validates() {
        Python::attach(|py| {
            let inputs = PyDict::new(py);
            inputs
                .set_item("output_allocations", vec![2_usize, 0, 2])
                .unwrap();
            assert_eq!(extract_output_allocations(&inputs, 3).unwrap(), vec![0, 2]);

            inputs
                .set_item("output_allocations", vec![3_usize])
                .unwrap();
            let error = extract_output_allocations(&inputs, 3).unwrap_err();
            assert!(error
                .to_string()
                .contains("output allocation 3 is outside [0, 3)"));
        });
    }

    #[test]
    fn written_allocation_extraction_narrows_outputs_and_falls_back() {
        Python::attach(|py| {
            let inputs = PyDict::new(py);
            inputs
                .set_item("output_allocations", vec![0_usize, 1, 2])
                .unwrap();
            // Without the narrower key the outputs are the written set.
            assert_eq!(extract_written_allocations(&inputs, 3).unwrap(), vec![0, 1, 2]);
            inputs
                .set_item("written_allocations", vec![2_usize, 2, 0])
                .unwrap();
            assert_eq!(extract_written_allocations(&inputs, 3).unwrap(), vec![0, 2]);
            inputs
                .set_item("written_allocations", vec![3_usize])
                .unwrap();
            let error = extract_written_allocations(&inputs, 3).unwrap_err();
            assert!(error
                .to_string()
                .contains("written allocation 3 is outside [0, 3)"));
        });
    }

    #[test]
    fn phase_selection_uses_only_the_matching_phase_entry() {
        Python::attach(|py| {
            let first = PyDict::new(py);
            first.set_item("cluster_ids", vec![1_usize]).unwrap();
            first.set_item("cta_ids", py.None()).unwrap();
            let phases = PyList::new(py, [py.None(), first.into_any().unbind()]).unwrap();
            let topology = LaunchTopology::new(2, 1, 1).unwrap();

            let phase0 = extract_phase_selection(Some(phases.as_any()), 0, 2, topology).unwrap();
            let phase1 = extract_phase_selection(Some(phases.as_any()), 1, 2, topology).unwrap();
            let contexts = topology.warp_contexts().collect::<Vec<_>>();

            assert!(phase0.includes(contexts[0]));
            assert!(phase0.includes(contexts[1]));
            assert!(!phase1.includes(contexts[0]));
            assert!(phase1.includes(contexts[1]));

            let direct = PyDict::new(py);
            direct.set_item("cluster_ids", vec![0_usize]).unwrap();
            direct.set_item("cta_ids", py.None()).unwrap();
            let error = extract_phase_selection(Some(direct.as_any()), 0, 2, topology).unwrap_err();
            assert!(error.to_string().contains("one subset entry per phase"));
        });
    }

    #[test]
    fn run_result_builder_owns_phase_execution_and_error_context() {
        Python::attach(|py| {
            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let mut builder = RunResultBuilder::new();
            builder
                .run_phase(py, 3, "phase", topology, || Ok(one_task_stats()))
                .unwrap();
            assert_eq!(builder.aggregate.task_count, 1);
            assert_eq!(builder.phases.len(), 1);
            assert_eq!(builder.phases[0].kernel_index, 3);
            assert_eq!(builder.phases[0].name, "phase");

            let error = builder
                .run_phase(py, 4, "broken", topology, || Err("failure".to_string()))
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("NumSim kernel phase 4 failed: failure"));
        });
    }

    #[test]
    fn shape_scalar_validation_is_engine_owned_and_typed() {
        validate_shape_scalar("n", 7_i32, 7).unwrap();
        let mismatch = validate_shape_scalar("n", 8_u64, 7).unwrap_err();
        assert!(mismatch
            .to_string()
            .contains("value 8 disagrees with bound buffer extent 7"));
        let negative = validate_shape_scalar("n", -1_i64, 7).unwrap_err();
        assert!(negative
            .to_string()
            .contains("value -1 cannot be used as a buffer extent"));
    }

    #[test]
    fn artifact_metadata_adapter_owns_python_container_construction() {
        Python::attach(|py| {
            let metadata = build_artifact_metadata_with_build_identity(
                py,
                "cache",
                NUMSIM_ABI_VERSION,
                "engine-digest",
                r#"{"engine_hash":"engine-digest"}"#,
                &["first", "second"],
                &[(1, 1, 2), (2, 1, 3)],
            )
            .unwrap();
            let metadata = metadata.bind(py);
            assert_eq!(
                metadata
                    .get_item("warp_counts")
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                vec![2, 6],
            );
            assert_eq!(
                metadata
                    .get_item("warp_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                8,
            );
            assert_eq!(
                metadata
                    .get_item("numsim_abi_version")
                    .unwrap()
                    .extract::<u32>()
                    .unwrap(),
                NUMSIM_ABI_VERSION,
            );
        });
    }

    #[test]
    fn artifact_metadata_records_and_validates_structured_build_identity() {
        Python::attach(|py| {
            let metadata = build_artifact_metadata_with_build_identity(
                py,
                "cache",
                NUMSIM_ABI_VERSION,
                "engine-digest",
                r#"{"engine_hash":"engine-digest","nested":{"target":"test"}}"#,
                &["kernel"],
                &[(1, 1, 1)],
            )
            .unwrap();
            let metadata = metadata.bind(py);
            assert_eq!(
                metadata
                    .get_item("engine_hash")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "engine-digest",
            );
            let identity = metadata.get_item("build_identity").unwrap();
            assert_eq!(
                identity
                    .get_item("engine_hash")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "engine-digest",
            );
            assert_eq!(
                identity
                    .get_item("nested")
                    .unwrap()
                    .get_item("target")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "test",
            );

            let mismatch = build_artifact_metadata_with_build_identity(
                py,
                "cache",
                NUMSIM_ABI_VERSION,
                "engine-digest",
                r#"{"engine_hash":"other"}"#,
                &["kernel"],
                &[(1, 1, 1)],
            )
            .unwrap_err();
            assert!(mismatch
                .to_string()
                .contains("engine_hash disagrees with build_identity.engine_hash"));
        });
    }

    #[test]
    fn run_result_builder_aggregates_phases_and_snapshots_only_outputs() {
        Python::attach(|py| {
            let topology0 = LaunchTopology::new(1, 1, 1).unwrap();
            let topology1 = LaunchTopology::new(2, 1, 1).unwrap();
            let mut builder = RunResultBuilder::new();
            builder.record_phase(
                0,
                "first",
                topology0,
                ExecutionStats {
                    task_count: 1,
                    completed_task_count: 1,
                    poll_count: 2,
                    normal_poll_count: 1,
                    poll_recheck_poll_count: 1,
                    poll_order: vec![0, 0],
                    completion_pump_count: 1,
                    completion_operation_count: 2,
                    worker_count: 1,
                    scheduling_domain_count: 1,
                    #[cfg(feature = "profile")]
                    executed_instruction_variants: Vec::new(),
                },
                vec![("gmem_read", 3, 5)],
            );
            builder.record_phase(
                1,
                "second",
                topology1,
                ExecutionStats {
                    task_count: 2,
                    completed_task_count: 2,
                    poll_count: 2,
                    normal_poll_count: 2,
                    poll_recheck_poll_count: 0,
                    poll_order: vec![1, 2],
                    completion_pump_count: 3,
                    completion_operation_count: 4,
                    worker_count: 2,
                    scheduling_domain_count: 2,
                    #[cfg(feature = "profile")]
                    executed_instruction_variants: Vec::new(),
                },
                Vec::new(),
            );

            let global = GlobalMemory::new();
            let first = global.allocate_from_bytes([1_u8, 2]).unwrap();
            let second = global.allocate_from_bytes([3_u8, 4]).unwrap();
            let result = {
                let inputs = PyDict::new(py);
                inputs
                    .set_item("allocations", Vec::<Py<PyAny>>::new())
                    .unwrap();
                inputs.set_item("buffers", PyDict::new(py)).unwrap();
                build_run_result(py, &inputs, builder, &global, &[first, second], &[1], false)
                    .unwrap()
            };
            let result = result.bind(py);
            let stats = result.get_item("stats").unwrap();
            assert_eq!(
                stats
                    .get_item("task_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );
            assert_eq!(
                stats
                    .get_item("completion_operation_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                6
            );
            assert_eq!(
                stats
                    .get_item("worker_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
            let phases = stats.get_item("kernels").unwrap();
            assert_eq!(phases.len().unwrap(), 2);
            assert_eq!(
                phases
                    .get_item(1)
                    .unwrap()
                    .get_item("name")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "second"
            );
            assert_eq!(
                phases
                    .get_item(1)
                    .unwrap()
                    .get_item("topology")
                    .unwrap()
                    .get_item("clusters")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );

            let allocation_bytes = result.get_item("allocation_bytes").unwrap();
            assert!(allocation_bytes.get_item(0).unwrap().is_none());
            assert_eq!(
                allocation_bytes
                    .get_item(1)
                    .unwrap()
                    .extract::<Vec<u8>>()
                    .unwrap(),
                vec![3, 4]
            );
        });
    }
}
