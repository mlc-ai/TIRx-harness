//! Launch-resource limits a CUDA launch must satisfy on its target
//! architecture.
//!
//! A launch the driver rejects never executes, so these checks do not change
//! NumSim's execution model; they prove a kernel invalid before it is
//! simulated. Only rules backed by a measured or documented device limit are
//! checked; an architecture without data is not checked at all.

use tvm_tirx::ir::{RegionStmtObj, Stmt};
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::ObjectRefCore;

use super::buffers::BufferBindings;
use super::cuda_arch::{cuda_arch, CUDA_ARCH_FIELD};
use super::memory::{bind_map, dynamic_shared_capacity, referenced_buffers};
use super::topology::{is_launch_entry, launch_values};
use super::util::{
    buffer_name, buffer_scope, int_imm, node_span, oref, same, span_json, AResult, Failure,
};
use super::Ctx;
use crate::decode::{tensor_decl, TensorDecl};

/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`: the portable per-block
/// limit on every architecture this table lists.
const PORTABLE_SHARED_BYTES: i64 = 49152;
/// The largest cluster `CU_LAUNCH_PORTABLE_CLUSTER_MODE_REQUIRE_PORTABLE`
/// accepts; CUDA guarantees portability of cluster sizes up to 8 CTAs.
const PORTABLE_CLUSTER_CTAS: i64 = 8;

/// Per-block shared-memory limits of one architecture family.
struct SharedMemoryLimits {
    family: &'static str,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    optin: i64,
    /// `CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK`, or `None`
    /// when the architecture has no oversized configuration.
    oversized: Option<i64>,
}

/// SM107: measured on VR200 (CUDA 13.5, `cuDeviceGetAttribute`): opt-in
/// 232448 B and oversized 334848 B; a 327 KiB dynamic launch with an
/// oversized mode runs and a 328 KiB one fails with CUDA_ERROR_INVALID_VALUE.
///
/// SM100/SM103: the documented 227 KiB opt-in limit. Oversized shared memory
/// was not measured on these parts and is treated as unavailable, so their
/// ceiling is the opt-in limit.
const SHARED_MEMORY_LIMITS: &[(&str, SharedMemoryLimits)] = &[
    (
        "100",
        SharedMemoryLimits {
            family: "SM100",
            optin: 232448,
            oversized: None,
        },
    ),
    (
        "103",
        SharedMemoryLimits {
            family: "SM103",
            optin: 232448,
            oversized: None,
        },
    ),
    (
        "107",
        SharedMemoryLimits {
            family: "SM107",
            optin: 232448,
            oversized: Some(334848),
        },
    ),
];

/// `CUsharedMemoryMode` values `T.cuda.LaunchConfig(shared_memory_mode=...)`
/// records as integers.
const SHARED_MEMORY_MODES: &[&str] = &[
    "default",
    "require_portable",
    "allow_non_portable",
    "allow_oversized",
    "prefer_oversized",
];
/// `CUlaunchAttributePortableClusterMode` values.
const PORTABLE_CLUSTER_MODES: &[&str] = &["default", "require_portable", "allow_non_portable"];

/// The limits of `sm_<NNN>`, `sm_<NNN>a` or `sm_<NNN>f`, when known.
fn limits_for_arch(arch: &str) -> Option<&'static SharedMemoryLimits> {
    let number = arch.strip_prefix("sm_")?;
    let number = number.strip_suffix(['a', 'f']).unwrap_or(number);
    SHARED_MEMORY_LIMITS
        .iter()
        .find(|(candidate, _)| *candidate == number)
        .map(|(_, limits)| limits)
}

/// `Some(name)` of a constant enum launch field in `0..names.len()`;
/// `None` when it is absent or not a known constant.
fn enum_field(
    launch: &[(String, tvm_tirx::ir::Expr)],
    field: &str,
    names: &'static [&'static str],
) -> Option<&'static str> {
    let (_, value) = launch.iter().find(|(name, _)| name == field)?;
    let value = int_imm(&oref(value.clone()))?;
    usize::try_from(value)
        .ok()
        .and_then(|index| names.get(index).copied())
}

/// Bytes of the statically sized `scope="shared"` allocations the kernel
/// references; CUDA places them beside the dynamic pool in each block.
///
/// This is a lower bound: an allocation whose size is not static, or that no
/// access reaches (the compiler may drop it), is not counted, and alignment
/// padding between allocations is ignored.
fn static_shared_bytes(
    ctx: &Ctx,
    statements: &[Stmt],
    nodes: &[ObjectRef],
    buffer_bindings: &BufferBindings,
) -> (i64, Vec<String>) {
    // Missing facts only lower the bound; the analyses that own them report
    // any unsupported form.
    let Ok(referenced) = referenced_buffers(nodes, true) else {
        return (0, Vec::new());
    };
    let referenced_roots = referenced
        .iter()
        .filter_map(|buffer| buffer_bindings.storage_key(buffer).ok())
        .collect::<Vec<_>>();
    let bindings = bind_map(statements);
    let mut total = 0i64;
    let mut names = Vec::new();
    for statement in statements {
        let Ok(Some((buffer, TensorDecl::Alloc))) = tensor_decl(statement) else {
            continue;
        };
        if buffer_scope(&buffer) != "shared" {
            continue;
        }
        let Ok(root) = buffer_bindings.storage_key(&buffer) else {
            continue;
        };
        if !referenced_roots
            .iter()
            .any(|candidate| same(candidate, &root))
        {
            continue;
        }
        let Ok(bytes) = ctx
            .inspect_layout(&buffer, &bindings)
            .and_then(|info| info.byte_end())
        else {
            continue;
        };
        if bytes > 0 {
            total += bytes;
            names.push(format!("{}={bytes}", buffer_name(&buffer)));
        }
    }
    (total, names)
}

/// The launch region's source span carries the evidence: every operand it
/// names is a field of that one `T.device_entry(launch=...)`.
fn invalid_launch<T>(region: &Stmt, message: String) -> AResult<T> {
    Err(Failure::InvalidLaunch {
        message,
        span: span_json(node_span(&oref(region.clone())).as_ref()),
    })
}

/// Fail with `Failure::InvalidLaunch` when the kernel's launch can never
/// succeed on its authored CUDA architecture.
pub fn check_launch_limits(
    ctx: &Ctx,
    func: &Function,
    statements: &[Stmt],
    nodes: &[ObjectRef],
    buffer_bindings: &BufferBindings,
) -> AResult<()> {
    let Some((region, entry)) = statements.iter().find_map(|statement| {
        statement
            .downcast_ref::<RegionStmtObj>()
            .filter(|entry| is_launch_entry(entry))
            .map(|entry| (statement, entry))
    }) else {
        return Ok(());
    };
    // A malformed launch region or architecture is reported by the analyses
    // that own it; this check only proves invalidity from well-formed facts.
    let Ok(launch) = launch_values(entry) else {
        return Ok(());
    };
    let Ok(Some(arch)) = cuda_arch(func) else {
        return Ok(());
    };
    let cluster = ["x", "y", "z"].iter().try_fold(1i64, |product, axis| {
        match launch
            .iter()
            .find(|(field, _)| *field == format!("cluster.{axis}"))
        {
            None => Some(product),
            Some((_, value)) => int_imm(&oref(value.clone())).map(|extent| product * extent),
        }
    });
    let cluster_mode = enum_field(
        &launch,
        "portable_cluster_size_mode",
        PORTABLE_CLUSTER_MODES,
    );
    if let (Some("require_portable"), Some(ctas)) = (cluster_mode, cluster) {
        if ctas > PORTABLE_CLUSTER_CTAS {
            return invalid_launch(
                region,
                format!(
                    "launch cluster of {ctas} CTAs exceeds the portable cluster size \
                     {PORTABLE_CLUSTER_CTAS} that portable_cluster_size_mode=\"require_portable\" \
                     requires; CUDA rejects the launch"
                ),
            );
        }
    }

    let Some(limits) = limits_for_arch(&arch) else {
        return Ok(());
    };
    let Ok(dynamic) = dynamic_shared_capacity(statements) else {
        return Ok(());
    };
    let dynamic = dynamic.unwrap_or(0);
    let (static_bytes, static_names) = static_shared_bytes(ctx, statements, nodes, buffer_bindings);
    let total = dynamic + static_bytes;
    let mode = enum_field(&launch, "shared_memory_mode", SHARED_MEMORY_MODES);
    let family = limits.family;
    let footprint = if static_bytes == 0 {
        format!("{total} bytes of dynamic shared memory per block")
    } else {
        format!(
            "{total} bytes of shared memory per block ({dynamic} dynamic + {static_bytes} static: {})",
            static_names.join(", ")
        )
    };
    let arch_note = format!("{CUDA_ARCH_FIELD}={arch:?}");

    // The device ceiling, whatever the mode: oversized shared memory where the
    // architecture has it, else the opt-in limit.
    let ceiling = limits.oversized.unwrap_or(limits.optin);
    if total > ceiling {
        let which = if limits.oversized.is_some() {
            "maximum oversized shared memory per block"
        } else {
            "maximum opt-in shared memory per block (oversized shared memory is not available)"
        };
        return invalid_launch(
            region,
            format!(
                "launch requests {footprint}, exceeding the {family} {which} of {ceiling} bytes \
                 ({arch_note}); CUDA rejects the launch"
            ),
        );
    }
    match mode {
        // `default` sizes the launch by CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        // which CUDA caps at the opt-in limit minus static shared memory and
        // which cannot reach oversized shared memory. Whether TIRx raises the
        // attribute below that cap when a mode is given is not modeled, so
        // only the cap itself is proof.
        Some("default") if total > limits.optin => invalid_launch(
            region,
            format!(
                "launch requests {footprint} with shared_memory_mode=\"default\", exceeding the \
                 {family} opt-in limit of {} bytes ({arch_note}); only an oversized \
                 shared_memory_mode reaches more, so CUDA rejects the launch",
                limits.optin
            ),
        ),
        // CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE bounds the requested dynamic
        // size by CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK.
        Some("require_portable") if dynamic > PORTABLE_SHARED_BYTES => invalid_launch(
            region,
            format!(
                "launch requests {dynamic} bytes of dynamic shared memory with \
                 shared_memory_mode=\"require_portable\", exceeding the portable per-block \
                 limit of {PORTABLE_SHARED_BYTES} bytes ({arch_note}); CUDA rejects the launch"
            ),
        ),
        // CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE bounds the requested
        // dynamic size by CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
        Some("allow_non_portable") if dynamic > limits.optin => invalid_launch(
            region,
            format!(
                "launch requests {dynamic} bytes of dynamic shared memory with \
                 shared_memory_mode=\"allow_non_portable\", exceeding the {family} opt-in \
                 limit of {} bytes ({arch_note}); only an oversized shared_memory_mode \
                 reaches more, so CUDA rejects the launch",
                limits.optin
            ),
        ),
        // An unset mode lets TIRx select one (an oversized mode above the
        // opt-in limit); the oversized modes accept anything up to the ceiling.
        _ => Ok(()),
    }
}
