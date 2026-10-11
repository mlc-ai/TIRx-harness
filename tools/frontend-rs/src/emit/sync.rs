//! Validation and emission of the sync instruction family.

use crate::analyze::util::{
    dtype_of, expr_type, is_pointer_type, oref, prim, prim_dtype, unsupported, AResult,
};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::{shared_pointer_source, SharedAddressForms};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::ir::StringImmObj;
use tvm::ir::{CallObj, TensorLoadObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

#[derive(Default)]
pub struct State {
    pub uses_setmaxnreg: bool,
    pub max_valid_increase_target: Option<i64>,
}

impl State {
    /// The SETMAXNREG calling initial count of the execution policy.
    pub fn calling_initial_count(
        &self,
        topology: &crate::analyze::topology::LaunchTopology,
    ) -> AResult<Option<i64>> {
        if !self.uses_setmaxnreg {
            return Ok(None);
        }
        let warpgroup_count = (topology.warps_per_cta + 3) / 4;
        let default_count = ((512 / warpgroup_count) / 8) * 8;
        // [VERIFY] This source-only caller base models CUDA 13.1 ptxas allocation.
        let compiler_register_cap = self.max_valid_increase_target.unwrap_or(256);
        let min_blocks = topology.min_blocks_per_sm;
        if min_blocks < 1 {
            return unsupported("min_blocks_per_sm must be positive");
        }
        // [VERIFY] 512 = the SM100 65536-register file / 128 threads per warpgroup. Launch
        // bounds cap the initial per-thread allocation, while an increase can later
        // consume registers released by another warpgroup.
        let resident_count = ((512 / (warpgroup_count * min_blocks)) / 8) * 8;
        if resident_count < 24 {
            return unsupported(
                "launch bounds cannot provide the minimum 24 registers per thread required by setmaxnreg",
            );
        }
        Ok(Some(
            default_count.min(compiler_register_cap).min(resident_count),
        ))
    }
}

/// `SyncCallKind`: the stable manifest category consumed by native Synccheck.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyncCallKind {
    MbarrierInit,
    MbarrierInval,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierExpectTx,
    MbarrierArriveNoComplete,
    MbarrierCompleteTx,
    MbarrierTryWait,
    ClcTryCancel,
    BarArrive,
    BarSync,
    WarpSync,
    WarpgroupSync,
    CtaSync,
    ClusterSync,
    FenceProxyAsync,
    FenceMbarrierInit,
    GriddepcontrolWait,
    ClusterArrive,
    ClusterWait,
    GridSync,
    OrderingMarker,
    DebugPrintf,
    Setmaxnreg,
}

impl SyncCallKind {
    /// The kind of an mbarrier statement function.
    fn from_function(function: &str) -> SyncCallKind {
        match function {
            "mbarrier_arrive" => SyncCallKind::MbarrierArrive,
            "mbarrier_arrive_expect_tx" => SyncCallKind::MbarrierArriveExpectTx,
            "mbarrier_expect_tx" => SyncCallKind::MbarrierExpectTx,
            "mbarrier_complete_tx" => SyncCallKind::MbarrierCompleteTx,
            _ => unreachable!("validated mbarrier statement function"),
        }
    }
}

const MBARRIER_NO_COMPLETE_CALLS: [&str; 4] = [
    "tirx.ptx.mbarrier_arrive_no_complete",
    "tirx.ptx.mbarrier_arrive_no_complete_sink",
    "tirx.ptx.mbarrier_arrive_drop_no_complete",
    "tirx.ptx.mbarrier_arrive_drop_no_complete_sink",
];
const MBARRIER_DROP_BASES: [&str; 8] = [
    "tirx.ptx.mbarrier_arrive_nocount",
    "tirx.ptx.mbarrier_arrive",
    "tirx.ptx.mbarrier_arrive_count_state",
    "tirx.ptx.mbarrier_arrive_state",
    "tirx.ptx.mbarrier_arrive_expect_tx",
    "tirx.ptx.mbarrier_arrive_expect_tx_state",
    "tirx.ptx.mbarrier_arrive_no_complete",
    "tirx.ptx.mbarrier_arrive_no_complete_sink",
];
/// (multicast op name, base op name).
const MBARRIER_MULTICAST_BASE: [(&str, &str); 8] = [
    (
        "tirx.ptx.mbarrier_expect_tx_multicast32",
        "tirx.ptx.mbarrier_expect_tx",
    ),
    (
        "tirx.ptx.mbarrier_complete_tx_multicast32",
        "tirx.ptx.mbarrier_complete_tx",
    ),
    (
        "tirx.ptx.mbarrier_arrive_multicast32",
        "tirx.ptx.mbarrier_arrive",
    ),
    (
        "tirx.ptx.mbarrier_arrive_multicast32_nocount",
        "tirx.ptx.mbarrier_arrive_nocount",
    ),
    (
        "tirx.ptx.mbarrier_arrive_drop_multicast32",
        "tirx.ptx.mbarrier_arrive_drop",
    ),
    (
        "tirx.ptx.mbarrier_arrive_drop_multicast32_nocount",
        "tirx.ptx.mbarrier_arrive_drop_nocount",
    ),
    (
        "tirx.ptx.mbarrier_arrive_expect_tx_multicast32",
        "tirx.ptx.mbarrier_arrive_expect_tx",
    ),
    (
        "tirx.ptx.mbarrier_arrive_drop_expect_tx_multicast32",
        "tirx.ptx.mbarrier_arrive_drop_expect_tx",
    ),
];
const NAMED_BARRIER_CALLS: [&str; 7] = [
    "tirx.ptx.bar_sync",
    "tirx.ptx.bar_sync_count",
    "tirx.ptx.bar_arrive",
    "tirx.ptx.barrier_sync",
    "tirx.ptx.barrier_sync_count",
    "tirx.ptx.barrier_arrive",
    "tirx.ptx.bar_warp_sync",
];
const MBARRIER_STATE_QUERY_CALLS: [&str; 3] = [
    "tirx.ptx.mbarrier_try_wait",
    "tirx.ptx.mbarrier_try_wait_hint",
    "tirx.ptx.mbarrier_test_wait",
];
const MBARRIER_QUERY_CALLS: [&str; 6] = [
    "tirx.ptx.mbarrier_try_wait",
    "tirx.ptx.mbarrier_try_wait_hint",
    "tirx.ptx.mbarrier_test_wait",
    "tirx.ptx.mbarrier_test_wait_parity",
    "tirx.ptx.mbarrier_try_wait_parity",
    "tirx.ptx.mbarrier_try_wait_parity_no_hint",
];
const MBARRIER_STATEMENT_BASES: [&str; 11] = [
    "tirx.ptx.mbarrier_arrive_nocount",
    "tirx.ptx.mbarrier_arrive",
    "tirx.ptx.mbarrier_arrive_count_state",
    "tirx.ptx.mbarrier_arrive_state",
    "tirx.ptx.mbarrier_arrive_expect_tx",
    "tirx.ptx.mbarrier_arrive_expect_tx_state",
    "tirx.ptx.mbarrier_expect_tx",
    "tirx.ptx.mbarrier_arrive_no_complete",
    "tirx.ptx.mbarrier_arrive_no_complete_sink",
    "tirx.ptx.mbarrier_complete_tx",
    "tirx.ptx.mbarrier_complete_tx",
];

/// The report queries: `(op name, (try_wait, state_query))`.
const MBARRIER_REPORT_QUERIES: [(&str, (bool, bool)); 12] = [
    ("tirx.ptx.mbarrier_test_wait_report", (false, true)),
    ("tirx.ptx.mbarrier_test_wait_report_value", (false, true)),
    ("tirx.ptx.mbarrier_try_wait_report", (true, true)),
    ("tirx.ptx.mbarrier_try_wait_report_value", (true, true)),
    ("tirx.ptx.mbarrier_try_wait_report_hint", (true, true)),
    ("tirx.ptx.mbarrier_try_wait_report_value_hint", (true, true)),
    ("tirx.ptx.mbarrier_test_wait_parity_report", (false, false)),
    (
        "tirx.ptx.mbarrier_test_wait_parity_report_value",
        (false, false),
    ),
    ("tirx.ptx.mbarrier_try_wait_parity_report", (true, false)),
    (
        "tirx.ptx.mbarrier_try_wait_parity_report_value",
        (true, false),
    ),
    (
        "tirx.ptx.mbarrier_try_wait_parity_report_hint",
        (true, false),
    ),
    (
        "tirx.ptx.mbarrier_try_wait_parity_report_value_hint",
        (true, false),
    ),
];

fn report_query(name: &str) -> Option<(bool, bool)> {
    MBARRIER_REPORT_QUERIES
        .iter()
        .find(|(query, _)| *query == name)
        .map(|(_, flags)| *flags)
}

/// The mbarrier queries, including the report queries.
fn is_mbarrier_query_call(name: &str) -> bool {
    MBARRIER_QUERY_CALLS.contains(&name) || report_query(name).is_some()
}

/// Whether an mbarrier query is one of the report queries.
pub fn is_report_query(name: &str) -> bool {
    report_query(name).is_some()
}

fn drop_name(base: &str) -> String {
    base.replacen("mbarrier_arrive", "mbarrier_arrive_drop", 1)
}

pub fn is_bar_reduce_call(name: &str) -> bool {
    for mnemonic in ["bar", "barrier"] {
        for kind in ["popc", "pred"] {
            for suffix in ["", "_count"] {
                if name == format!("tirx.ptx.{mnemonic}_red_{kind}{suffix}") {
                    return true;
                }
            }
        }
    }
    false
}

fn multicast_base(name: &str) -> Option<&'static str> {
    MBARRIER_MULTICAST_BASE
        .iter()
        .find(|(multicast, _)| *multicast == name)
        .map(|(_, base)| *base)
}

fn is_drop_call(name: &str) -> bool {
    MBARRIER_DROP_BASES
        .iter()
        .any(|base| drop_name(base) == name)
}

fn is_mbarrier_statement_call(name: &str) -> bool {
    multicast_base(name).is_some() || is_drop_call(name) || MBARRIER_STATEMENT_BASES.contains(&name)
}

/// A validated non-PTX synchronization call. Its Synccheck category, arities
/// and lowering all derive from the variant.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawSyncCall {
    MbarrierWait,
    WarpSync,
    WarpgroupSync,
    CtaSync,
    StorageSync,
    ClusterSync,
    GridSync,
    ThreadFence,
    NanoSleep,
    Printf,
}

impl RawSyncCall {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "tirx.cuda.mbarrier_wait" | "tirx.cuda.mbarrier_wait_acquire_cluster" => {
                Self::MbarrierWait
            }
            "tirx.cuda.warp_sync" => Self::WarpSync,
            "tirx.cuda.warpgroup_sync" => Self::WarpgroupSync,
            "tirx.cuda.cta_sync" => Self::CtaSync,
            "tirx.tvm_storage_sync" => Self::StorageSync,
            "tirx.cuda.cluster_sync" => Self::ClusterSync,
            "tirx.cuda.grid_sync" => Self::GridSync,
            "tirx.cuda.thread_fence" => Self::ThreadFence,
            "tirx.cuda.nano_sleep" => Self::NanoSleep,
            "tirx.cuda.printf" => Self::Printf,
            _ => return None,
        })
    }

    /// `(kind, arities)`.
    pub fn signature(self) -> (SyncCallKind, &'static [usize]) {
        match self {
            Self::MbarrierWait => (SyncCallKind::MbarrierTryWait, &[2]),
            Self::WarpSync => (SyncCallKind::WarpSync, &[0]),
            Self::WarpgroupSync => (SyncCallKind::WarpgroupSync, &[1]),
            Self::CtaSync => (SyncCallKind::CtaSync, &[0]),
            Self::StorageSync => (SyncCallKind::CtaSync, &[3]),
            Self::ClusterSync => (SyncCallKind::ClusterSync, &[0]),
            Self::GridSync => (SyncCallKind::GridSync, &[0]),
            Self::ThreadFence => (SyncCallKind::OrderingMarker, &[0]),
            Self::NanoSleep => (SyncCallKind::OrderingMarker, &[1]),
            Self::Printf => (SyncCallKind::DebugPrintf, &[]),
        }
    }

    /// The engine functions (below `v2::`) the lowering calls, and the
    /// specialization of a single call. `printf` calls none; it lowers to the
    /// host-side argument evaluation.
    pub fn engine_calls(self) -> (&'static [&'static str], Option<&'static str>) {
        match self {
            Self::MbarrierWait => (&["sync::mbarrier_wait_until"], Some(WAIT_PARITY)),
            Self::WarpSync => (&[BAR_WARP_SYNC], None),
            Self::WarpgroupSync | Self::CtaSync | Self::StorageSync => (&[BAR_SYNC], None),
            Self::ClusterSync => (&[BARRIER_CLUSTER_ARRIVE, BARRIER_CLUSTER_WAIT], None),
            Self::GridSync => (&["collective::grid_sync"], None),
            Self::ThreadFence => (&[FENCE], Some(THREAD_FENCE)),
            Self::NanoSleep => (&["control::nanosleep"], None),
            Self::Printf => (&[], None),
        }
    }
}

pub const WAIT_PARITY: &str = "v2::sync::variant::WaitUntilParity";
/// TVM emits `__threadfence()`: a sequentially consistent device-scope fence
/// (`fence_variant("sc", "gpu")`).
pub const THREAD_FENCE: &str =
    "v2::sync::variant::Fence<v2::sync::variant::Gpu, v2::sync::variant::Sc>";

/// The engine functions (below `v2::`) the lowerings call.
pub const BAR_REDUCE: &str = "collective::bar_reduce";
pub const BAR_SYNC: &str = "sync::bar_sync";
pub const BAR_WARP_SYNC: &str = "warp::bar_warp_sync";
pub const BARRIER_CLUSTER_ARRIVE: &str = "sync::barrier_cluster_arrive";
pub const BARRIER_CLUSTER_WAIT: &str = "sync::barrier_cluster_wait";
pub const CLC_TRY_CANCEL: &str = "control::clc_try_cancel";
pub const FENCE: &str = "sync::fence";
pub const FENCE_MBARRIER_INIT: &str = "sync::fence_mbarrier_init";
pub const FENCE_PROXY_ASYNC: &str = "sync::fence_proxy_async";
pub const GRIDDEPCONTROL: &str = "control::griddepcontrol";
pub const MBARRIER_CHECK_LAYOUT: &str = "sync::mbarrier_check_layout";
pub const MBARRIER_INIT: &str = "sync::mbarrier_init";
pub const MBARRIER_INVAL: &str = "sync::mbarrier_inval";
pub const MBARRIER_PENDING_COUNT: &str = "sync::mbarrier_pending_count";
pub const SETMAXNREG: &str = "control::setmaxnreg";

/// The engine function of one mbarrier query form.
pub fn mbarrier_query_function(try_wait: bool) -> &'static str {
    if try_wait {
        "sync::mbarrier_try_wait"
    } else {
        "sync::mbarrier_test_wait"
    }
}

/// The variant of one mbarrier statement or query marker.
pub fn sync_variant(marker: &str) -> String {
    format!("v2::sync::variant::{marker}")
}

/// The cluster-arrive variant of one validated semantic.
pub fn cluster_arrive_variant(sem: &str, aligned: bool) -> String {
    format!(
        "v2::sync::variant::ClusterArrive<v2::sync::variant::{}, {aligned}>",
        cluster_arrive_semantics(sem)
    )
}

/// The cluster-wait variant.
pub fn cluster_wait_variant(aligned: bool) -> String {
    format!("v2::sync::variant::ClusterWait<{aligned}>")
}

/// The `mbarrier.init` variant: `layout::v1` selects the versioned layout.
pub fn mbarrier_init_variant(decoded: &DecodedPtx) -> &'static str {
    if decoded.modifier_or_empty("layout") == "layout::v1" {
        "v2::sync::variant::MbarrierInit<true>"
    } else {
        "v2::sync::variant::MbarrierInit"
    }
}

/// The `fence.proxy.async` variant of one validated space.
pub fn proxy_async_variant(space: &str) -> String {
    format!(
        "v2::sync::variant::ProxyAsync<v2::sync::variant::{}>",
        proxy_marker(space)
    )
}

/// The `setmaxnreg` variant of an increase or a decrease.
pub fn setmaxnreg_variant(increase: bool) -> String {
    let action = if increase { "Increase" } else { "Decrease" };
    format!("v2::control::variant::Setmaxnreg{action}")
}

/// `"handle"` for pointers, the dtype text, or `""`.
fn sync_dtype(value: &ObjectRef) -> String {
    let Some(ty) = expr_type(value) else {
        return String::new();
    };
    if is_pointer_type(&ty) {
        return "handle".to_owned();
    }
    prim_dtype(&ty)
        .map(crate::analyze::util::dtype_text)
        .unwrap_or_default()
}

/// `allowed` must be sorted lexicographically.
fn require_dtype(value: &ObjectRef, allowed: &[&str], field: &str) -> AResult<()> {
    let actual = sync_dtype(value);
    if !allowed.contains(&actual.as_str()) {
        return unsupported(format!(
            "{field} must have dtype in {:?}, got {:?}",
            &allowed, &actual
        ));
    }
    Ok(())
}

/// `(increase, count)`. `ptx_sync_kind` checks the count against the engine's
/// register range.
pub fn decoded_setmaxnreg_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<(bool, i64)> {
    let op_name = decoded.op_name.as_str();
    for (name, value) in [("sync", "sync"), ("aligned", "aligned"), ("type", "u32")] {
        if decoded.modifier(name)? != value {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                value,
                decoded.modifier(name)?
            ));
        }
    }
    let action = decoded.modifier("action")?;
    if action != "inc" && action != "dec" {
        return unsupported(format!("{op_name} has invalid action {:?}", action));
    }
    let count_expr = prim(&decoded.scalar_operand("nreg")?)?;
    let simplified = crate::analyze::util::simplify(&ctx.analyzer, &count_expr)?;
    let Some(count) = crate::analyze::util::int_imm_expr(&simplified) else {
        return unsupported(format!("{op_name}.register_count must be a static integer"));
    };
    Ok((action == "inc", count))
}

/// Validates a raw (non-PTX) synchronization call.
pub fn validate_raw_sync_call(node: &ObjectRef, op_name: &str) -> AResult<RawSyncCall> {
    let Some(call) = node.as_node::<CallObj>() else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error("sync lowerer requires a TIRx Call"),
        ));
    };
    let Some(raw) = RawSyncCall::from_name(op_name) else {
        return Err(crate::analyze::util::Failure::Ffi(
            crate::analyze::util::ffi_error(&format!("sync lowerer received {op_name}")),
        ));
    };
    let args: Vec<ObjectRef> = call.args.iter().map(oref).collect();
    if !sync_dtype(node).is_empty() {
        return unsupported(format!("{op_name} must have a void result"));
    }
    if raw == RawSyncCall::Printf {
        if args.is_empty() || args[0].as_node::<StringImmObj>().is_none() {
            return unsupported(format!("{op_name} requires a static format string"));
        }
        let supported = [
            "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
            "float32", "float64", "handle",
        ];
        let bad: Vec<String> = args[1..]
            .iter()
            .map(sync_dtype)
            .filter(|dtype| !supported.contains(&dtype.as_str()))
            .map(|dtype| (&dtype).to_string())
            .collect();
        if !bad.is_empty() {
            return unsupported(format!("{op_name} unsupported argument dtypes {:?}", &bad));
        }
        return Ok(raw);
    }
    let (_, arities) = raw.signature();
    if !arities.contains(&args.len()) {
        let mut sorted: Vec<usize> = arities.to_vec();
        sorted.sort_unstable();
        let expected: Vec<String> = sorted.iter().map(|value| value.to_string()).collect();
        return unsupported(format!(
            "{op_name} expects {} arguments, got {}",
            expected.join("/"),
            args.len()
        ));
    }
    match raw {
        RawSyncCall::MbarrierWait => {
            require_dtype(
                &args[0],
                &["handle", "uint32"],
                &format!("{op_name}.barrier"),
            )?;
            require_dtype(
                &args[1],
                crate::dtypes::integer_dtypes(),
                &format!("{op_name}.phase"),
            )?;
        }
        RawSyncCall::WarpgroupSync => {
            require_dtype(
                &args[0],
                crate::dtypes::integer_dtypes(),
                &format!("{op_name}.barrier_id"),
            )?;
        }
        RawSyncCall::NanoSleep => {
            require_dtype(
                &args[0],
                crate::dtypes::integer_dtypes(),
                &format!("{op_name}.duration"),
            )?;
        }
        _ => {}
    }
    Ok(raw)
}

pub struct MbarrierInitParts {
    /// The shared source `addr` retains.
    pub address: ObjectRef,
    pub count: ObjectRef,
    pub predicate: Option<ObjectRef>,
}

pub fn decoded_mbarrier_init_parts(decoded: &DecodedPtx) -> AResult<MbarrierInitParts> {
    let op_name = decoded.op_name.as_str();
    if decoded.modifier("action")? != "init" || decoded.modifier("type")? != "b64" {
        return unsupported(format!("{op_name} requires init.b64"));
    }
    let layout = decoded.modifier_or_empty("layout");
    if !["", "layout::v0", "layout::v1"].contains(&layout) {
        return unsupported(format!(
            "{op_name} requires layout::v0 or layout::v1, got {:?}",
            layout
        ));
    }
    let space = decoded.modifier("space")?.to_owned();
    if !["", "shared", "shared::cta"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    let address = decoded.scalar_operand("addr")?;
    let count = decoded.scalar_operand("count")?;
    let predicate = decoded.predicate.clone();
    let (address, _) = shared_pointer_source(
        &address,
        &format!("{op_name}.addr"),
        SharedAddressForms::SYNC,
    )?;
    require_dtype(
        &count,
        crate::dtypes::integer_dtypes(),
        &format!("{op_name}.count"),
    )?;
    if let Some(predicate) = &predicate {
        require_dtype(
            predicate,
            crate::dtypes::predicate_dtypes(),
            &format!("{op_name}.predicate"),
        )?;
    }
    Ok(MbarrierInitParts {
        address,
        count,
        predicate,
    })
}

pub fn decoded_proxy_space(decoded: &DecodedPtx) -> AResult<String> {
    let op_name = decoded.op_name.as_str();
    if decoded.modifier("proxy")? != "proxy" {
        return unsupported(format!("{op_name} requires the proxy modifier"));
    }
    let proxykind = decoded.modifier("proxykind")?;
    let space = decoded.modifier("space")?.to_owned();
    if proxykind == "alias" {
        if !space.is_empty() {
            return unsupported(format!("{op_name} fence.proxy.alias takes no state space"));
        }
        return Ok("alias".to_owned());
    }
    if proxykind != "async" {
        return unsupported(format!(
            "{op_name} proxy kind {:?} is not modeled; expected 'async' or 'alias'",
            proxykind
        ));
    }
    if !["", "global", "shared::cta", "shared::cluster"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    Ok(space)
}

/// `(sem, scope)`.
pub fn decoded_fence_parts(decoded: &DecodedPtx) -> AResult<(String, String)> {
    let op_name = decoded.op_name.as_str();
    let sem = decoded.modifier("sem")?.to_owned();
    let scope = decoded.modifier("scope")?.to_owned();
    if !["", "sc", "acq_rel", "acquire", "release"].contains(&sem.as_str()) {
        return unsupported(format!("{op_name} has unsupported semantic {:?}", &sem));
    }
    if !["cta", "cluster", "gpu", "sys"].contains(&scope.as_str()) {
        return unsupported(format!("{op_name} has unsupported scope {:?}", &scope));
    }
    Ok((sem, scope))
}

pub fn decoded_grid_action(decoded: &DecodedPtx) -> AResult<String> {
    let action = decoded.modifier("action")?.to_owned();
    if action != "launch_dependents" && action != "wait" {
        return unsupported(format!(
            "{} has unsupported action {:?}",
            decoded.op_name, &action
        ));
    }
    Ok(action)
}

fn require_decoded_modifiers(decoded: &DecodedPtx, allowed: &[(&str, &[&str])]) -> AResult<()> {
    for (name, choices) in allowed {
        let value = decoded.modifier(name)?;
        if !choices.contains(&value) {
            return unsupported(format!(
                "{} modifier {:?} must be one of {:?}, got {:?}",
                decoded.op_name, name, &choices, value
            ));
        }
    }
    Ok(())
}

pub struct NamedBarrierParts {
    /// `"warp"`, `"bar_arrive"`, `"barrier_sync"` or `"bar_sync"`.
    pub function: String,
    pub barrier_id: ObjectRef,
    pub count: Option<ObjectRef>,
}

pub fn decoded_named_barrier_parts(decoded: &DecodedPtx) -> AResult<NamedBarrierParts> {
    let op_name = decoded.op_name.as_str();
    if op_name == "tirx.ptx.bar_warp_sync" {
        require_decoded_modifiers(decoded, &[("warp", &["warp"]), ("action", &["sync"])])?;
        let membermask = decoded.scalar_operand("membermask")?;
        require_dtype(
            &membermask,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.membermask"),
        )?;
        return Ok(NamedBarrierParts {
            function: "warp".to_owned(),
            barrier_id: membermask,
            count: None,
        });
    }
    require_decoded_modifiers(decoded, &[("cta", &["", "cta"])])?;
    let action = decoded.modifier("action")?.to_owned();
    if action != "sync" && action != "arrive" {
        return unsupported(format!("{op_name} has unsupported action {:?}", &action));
    }
    let mut aligned = false;
    if op_name.starts_with("tirx.ptx.barrier_") {
        require_decoded_modifiers(decoded, &[("aligned", &["", "aligned"])])?;
        aligned = decoded.modifier("aligned")? == "aligned";
    }
    let barrier_id = decoded.scalar_operand("a")?;
    require_dtype(
        &barrier_id,
        crate::dtypes::integer_dtypes(),
        &format!("{op_name}.barrier_id"),
    )?;
    let count = if decoded.has_operand("b") {
        Some(decoded.scalar_operand("b")?)
    } else {
        None
    };
    if let Some(count) = &count {
        require_dtype(
            count,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.thread_count"),
        )?;
    }
    if action == "arrive" && count.is_none() {
        return unsupported(format!("{op_name} requires an explicit thread count"));
    }
    let function = if action == "arrive" {
        "bar_arrive"
    } else if (op_name == "tirx.ptx.barrier_sync" || op_name == "tirx.ptx.barrier_sync_count")
        && !aligned
    {
        "barrier_sync"
    } else {
        "bar_sync"
    };
    Ok(NamedBarrierParts {
        function: function.to_owned(),
        barrier_id,
        count,
    })
}

pub struct BarReduceParts {
    pub destination: ObjectRef,
    pub barrier_id: ObjectRef,
    pub count: Option<ObjectRef>,
    pub predicate: ObjectRef,
    pub variant: String,
}

pub fn decoded_bar_reduce_parts(decoded: &DecodedPtx) -> AResult<BarReduceParts> {
    let op_name = decoded.op_name.as_str();
    require_decoded_modifiers(decoded, &[("cta", &["", "cta"]), ("action", &["red"])])?;
    let op = decoded.modifier("op")?;
    let marker = match op {
        "popc" => Some("Sum"),
        "and" => Some("All"),
        "or" => Some("Any"),
        _ => None,
    };
    let expected_type = if op == "popc" { "u32" } else { "pred" };
    if marker.is_none() || decoded.modifier("type")? != expected_type {
        return unsupported(format!("{op_name} has unsupported reduction/type"));
    }
    let aligned =
        op_name.starts_with("tirx.ptx.bar_") || decoded.modifier_or_empty("aligned") == "aligned";
    let destination = decoded.scalar_operand("d")?;
    if destination.as_node::<TensorLoadObj>().is_none() || sync_dtype(&destination) != "uint32" {
        return unsupported(format!("{op_name} destination must be a uint32 TensorLoad"));
    }
    let barrier_id = decoded.scalar_operand("a")?;
    let count = if decoded.has_operand("b") {
        Some(decoded.scalar_operand("b")?)
    } else {
        None
    };
    let predicate = decoded.scalar_operand("c")?;
    require_dtype(
        &barrier_id,
        crate::dtypes::integer_dtypes(),
        &format!("{op_name}.barrier_id"),
    )?;
    if let Some(count) = &count {
        require_dtype(
            count,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.count"),
        )?;
    }
    require_dtype(
        &predicate,
        crate::dtypes::predicate_dtypes(),
        &format!("{op_name}.predicate"),
    )?;
    let variant = format!(
        "v2::collective::variant::BarReduce<v2::collective::variant::{}, {aligned}>",
        marker.expect("validated marker")
    );
    Ok(BarReduceParts {
        destination,
        barrier_id,
        count,
        predicate,
        variant,
    })
}

/// `(action, sem, aligned)`.
pub fn decoded_cluster_barrier_parts(decoded: &DecodedPtx) -> AResult<(String, String, bool)> {
    let op_name = decoded.op_name.as_str();
    require_decoded_modifiers(
        decoded,
        &[("cluster", &["cluster"]), ("aligned", &["", "aligned"])],
    )?;
    let action = decoded.modifier("action")?.to_owned();
    let expected_action = if op_name == "tirx.ptx.barrier_cluster_arrive" {
        "arrive"
    } else {
        "wait"
    };
    if action != expected_action {
        return unsupported(format!(
            "{op_name} requires action {:?}, got {:?}",
            expected_action, &action
        ));
    }
    let sem = decoded.modifier("sem")?.to_owned();
    let allowed: &[&str] = if action == "arrive" {
        &["", "release", "relaxed"]
    } else {
        &["", "acquire"]
    };
    if !allowed.contains(&sem.as_str()) {
        return unsupported(format!("{op_name} has unsupported semantic {:?}", &sem));
    }
    Ok((action, sem, decoded.modifier("aligned")? == "aligned"))
}

pub fn cluster_arrive_semantics(sem: &str) -> &'static str {
    match sem {
        "" => "DefaultRelease",
        "release" => "Release",
        "relaxed" => "Relaxed",
        _ => unreachable!("validated cluster arrive semantic"),
    }
}

pub struct MbarrierStatementParts {
    pub function: String,
    pub marker: String,
    pub address: ObjectRef,
    /// `None`, an operand, or the implied count `IntImm("int64", 1)`.
    pub amount: Option<AmountOperand>,
    pub predicate: Option<ObjectRef>,
    pub state_destination: Option<ObjectRef>,
    pub space: String,
}

/// An mbarrier amount: an IR operand or the implicit literal one.
#[derive(Clone)]
pub enum AmountOperand {
    Expression(ObjectRef),
    One,
}

pub fn decoded_mbarrier_statement_parts(decoded: &DecodedPtx) -> AResult<MbarrierStatementParts> {
    let op_name = decoded.op_name.as_str();
    let mut base_name = multicast_base(op_name).unwrap_or(op_name).to_owned();
    let drop = is_drop_call(&base_name);
    base_name = base_name.replace("mbarrier_arrive_drop", "mbarrier_arrive");
    let action = if drop { "arrive_drop" } else { "arrive" };
    let multicast = multicast_base(op_name).is_some();
    if multicast {
        require_decoded_modifiers(
            decoded,
            &[
                ("multicast", &["multicast::cluster::32b"]),
                ("space", &["shared::cluster"]),
            ],
        )?;
        require_dtype(
            &decoded.scalar_operand("cta_mask")?,
            &["uint32"],
            &format!("{op_name}.cta_mask"),
        )?;
    }
    require_decoded_modifiers(decoded, &[("type", &["b64"])])?;
    let sem = decoded.modifier("sem")?.to_owned();
    let scope = decoded.modifier("scope")?.to_owned();
    let space = decoded.modifier("space")?.to_owned();
    if !["", "shared", "shared::cta", "shared::cluster"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} has unsupported space {:?}", &space));
    }
    if sem.is_empty() != scope.is_empty() {
        return unsupported(format!(
            "{op_name} requires semantic and scope to be both present or both omitted"
        ));
    }
    let predicate = decoded.predicate.clone();
    if let Some(predicate) = &predicate {
        require_dtype(
            predicate,
            crate::dtypes::predicate_dtypes(),
            &format!("{op_name}.predicate"),
        )?;
    }
    let address = decoded.scalar_operand("addr")?;
    let mut state_destination: Option<ObjectRef> = None;
    if matches!(
        base_name.as_str(),
        "tirx.ptx.mbarrier_arrive_state"
            | "tirx.ptx.mbarrier_arrive_expect_tx_state"
            | "tirx.ptx.mbarrier_arrive_count_state"
    ) {
        let state = decoded.scalar_operand("state")?;
        if state.as_node::<TensorLoadObj>().is_none() || sync_dtype(&state) != "uint64" {
            return unsupported(format!(
                "{op_name}.state must be a uint64 TensorLoad lvalue"
            ));
        }
        state_destination = Some(state);
    }
    let amount: Option<AmountOperand>;
    let mut marker: String;
    let function: &str;
    if base_name == "tirx.ptx.mbarrier_expect_tx" {
        require_decoded_modifiers(
            decoded,
            &[
                ("action", &["expect_tx"]),
                ("sem", &["", "relaxed"]),
                ("scope", &["", "cta", "cluster"]),
            ],
        )?;
        amount = Some(AmountOperand::Expression(
            decoded.scalar_operand("tx_count")?,
        ));
        marker = "ExpectTx".to_owned();
        function = "mbarrier_expect_tx";
    } else if base_name == "tirx.ptx.mbarrier_complete_tx" {
        require_decoded_modifiers(decoded, &[("action", &["complete_tx"])])?;
        let pair = (sem.as_str(), scope.as_str());
        let allowed = pair == ("relaxed", "cta")
            || pair == ("relaxed", "cluster")
            || (multicast && pair == ("", ""));
        if !allowed {
            return unsupported(format!(
                "{op_name} requires relaxed semantics at cta or cluster scope"
            ));
        }
        amount = Some(AmountOperand::Expression(
            decoded.scalar_operand("tx_count")?,
        ));
        marker = "CompleteTxLocal".to_owned();
        function = "mbarrier_complete_tx";
    } else if base_name == "tirx.ptx.mbarrier_arrive_expect_tx"
        || base_name == "tirx.ptx.mbarrier_arrive_expect_tx_state"
    {
        require_decoded_modifiers(
            decoded,
            &[("action", &[action]), ("expect_tx", &["expect_tx"])],
        )?;
        let field = if state_destination.is_some() {
            "count"
        } else {
            "tx_count"
        };
        amount = Some(AmountOperand::Expression(decoded.scalar_operand(field)?));
        marker = format!(
            "ArriveExpectTxLocal{}",
            if state_destination.is_some() {
                "State"
            } else {
                ""
            }
        );
        function = "mbarrier_arrive_expect_tx";
    } else {
        require_decoded_modifiers(decoded, &[("action", &[action])])?;
        if MBARRIER_NO_COMPLETE_CALLS.contains(&op_name) {
            require_decoded_modifiers(
                decoded,
                &[
                    ("nocomplete", &["noComplete"]),
                    ("sem", &["", "release"]),
                    ("scope", &["", "cta"]),
                ],
            )?;
            if base_name == "tirx.ptx.mbarrier_arrive_no_complete" {
                let state = decoded.scalar_operand("state")?;
                require_dtype(&state, &["uint64"], &format!("{op_name}.state"))?;
                if state.as_node::<TensorLoadObj>().is_none() {
                    return unsupported(format!("{op_name}.state must be a TensorLoad lvalue"));
                }
                state_destination = Some(state);
            }
        }
        let mut chosen = if matches!(
            base_name.as_str(),
            "tirx.ptx.mbarrier_arrive"
                | "tirx.ptx.mbarrier_arrive_count_state"
                | "tirx.ptx.mbarrier_arrive_no_complete"
                | "tirx.ptx.mbarrier_arrive_no_complete_sink"
        ) {
            Some(AmountOperand::Expression(decoded.scalar_operand("count")?))
        } else {
            None
        };
        if base_name == "tirx.ptx.mbarrier_arrive_state" {
            chosen = Some(AmountOperand::One);
        }
        marker = format!("ArriveLocal{}", if chosen.is_some() { "Count" } else { "" });
        if state_destination.is_some() {
            marker.push_str("State");
        }
        if MBARRIER_NO_COMPLETE_CALLS.contains(&op_name) {
            marker = "ArriveLocalCountState<true>".to_owned();
        }
        amount = chosen;
        function = "mbarrier_arrive";
    }
    if space == "shared::cluster"
        && (function == "mbarrier_arrive" || function == "mbarrier_arrive_expect_tx")
    {
        if state_destination.is_some() {
            return unsupported(format!(
                "{op_name} cannot return state from a cluster shared barrier"
            ));
        }
        marker = marker.replacen("Local", "Remote", 1);
    }
    let mut amount = amount;
    if let Some(AmountOperand::Expression(value)) = &amount {
        let amount_field = if matches!(
            base_name.as_str(),
            "tirx.ptx.mbarrier_expect_tx"
                | "tirx.ptx.mbarrier_arrive_expect_tx"
                | "tirx.ptx.mbarrier_complete_tx"
        ) {
            "tx_count"
        } else {
            "count"
        };
        require_dtype(
            value,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.{amount_field}"),
        )?;
    }
    if multicast && amount.is_none() {
        amount = Some(AmountOperand::One);
        marker.push_str("Count");
    }
    if predicate.is_some()
        && state_destination.is_none()
        && function != "mbarrier_expect_tx"
        && !MBARRIER_NO_COMPLETE_CALLS.contains(&op_name)
        && !multicast
    {
        marker.push_str("Predicated");
    }
    if drop {
        if marker.contains('<') {
            marker.pop();
            marker.push_str(", true>");
        } else if marker == "ArriveLocalCountState" {
            marker.push_str("<false, true>");
        } else {
            marker.push_str("<true>");
        }
    }
    if sem == "relaxed"
        && (function == "mbarrier_arrive" || function == "mbarrier_arrive_expect_tx")
    {
        if marker.contains('<') {
            marker.pop();
            marker.push_str(", false>");
        } else if marker == "ArriveLocalCountState" {
            marker.push_str("<false, false, false>");
        } else {
            marker.push_str("<false, false>");
        }
    }
    if multicast {
        marker = format!("Multicast<v2::sync::variant::{marker}>");
    }
    Ok(MbarrierStatementParts {
        function: function.to_owned(),
        marker,
        address,
        amount,
        predicate,
        state_destination,
        space,
    })
}

pub struct MbarrierQueryParts {
    pub destination: ObjectRef,
    /// The shared source `addr` retains.
    pub address: ObjectRef,
    pub state_or_phase: ObjectRef,
    pub sem: String,
    pub try_wait: bool,
    pub state_query: bool,
}

pub fn decoded_mbarrier_query_parts(decoded: &DecodedPtx) -> AResult<MbarrierQueryParts> {
    let op_name = decoded.op_name.as_str();
    let mut try_wait = matches!(
        op_name,
        "tirx.ptx.mbarrier_try_wait"
            | "tirx.ptx.mbarrier_try_wait_hint"
            | "tirx.ptx.mbarrier_try_wait_parity"
            | "tirx.ptx.mbarrier_try_wait_parity_no_hint"
    );
    let mut state_query = MBARRIER_STATE_QUERY_CALLS.contains(&op_name);
    let reporting = report_query(op_name);
    if let Some(flags) = reporting {
        (try_wait, state_query) = flags;
        for (name, dtype) in [("report_predicate", "uint32"), ("report_value", "uint8")] {
            if !decoded.has_operand(name) {
                continue;
            }
            let output = decoded.scalar_operand(name)?;
            if output.as_node::<TensorLoadObj>().is_none() || sync_dtype(&output) != dtype {
                return unsupported(format!(
                    "{op_name}.{name} must be a {dtype} TensorLoad lvalue"
                ));
            }
        }
    }
    let action: &[&str] = if try_wait {
        &["try_wait"]
    } else {
        &["test_wait"]
    };
    let mut modifiers: Vec<(&str, &[&str])> = vec![
        ("action", action),
        ("sem", &["", "acquire", "relaxed"]),
        ("scope", &["", "cta", "cluster"]),
        ("space", &["", "shared", "shared::cta"]),
        ("type", &["b64"]),
    ];
    if !state_query {
        modifiers.push(("parity", &["parity"]));
    }
    if reporting.is_some() {
        modifiers.push(("phase_type", &["phase_type::primary"]));
    } else if !state_query
        && decoded
            .modifiers
            .iter()
            .any(|(slot, _)| slot == "phase_type")
    {
        modifiers.push((
            "phase_type",
            &["", "phase_type::primary", "phase_type::conditional"],
        ));
    }
    require_decoded_modifiers(decoded, &modifiers)?;
    let sem = decoded.modifier("sem")?.to_owned();
    let scope = decoded.modifier("scope")?.to_owned();
    if sem.is_empty() != scope.is_empty() {
        return unsupported(format!(
            "{op_name} requires semantic and scope to be both present or both omitted"
        ));
    }
    let destination = decoded.scalar_operand("wait_complete")?;
    if destination.as_node::<TensorLoadObj>().is_none() || sync_dtype(&destination) != "uint32" {
        return unsupported(format!(
            "{op_name}.wait_complete must be a uint32 TensorLoad lvalue"
        ));
    }
    let (address, _) = shared_pointer_source(
        &decoded.scalar_operand("addr")?,
        &format!("{op_name}.addr"),
        SharedAddressForms::SYNC,
    )?;
    let operand = if state_query && reporting.is_none() {
        "state"
    } else {
        "phase"
    };
    let state_or_phase = decoded.scalar_operand(operand)?;
    let field = if state_query { "state" } else { "phase" };
    if state_query {
        require_dtype(&state_or_phase, &["uint64"], &format!("{op_name}.{field}"))?;
    } else {
        require_dtype(
            &state_or_phase,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.{field}"),
        )?;
    }
    let time_hint = if try_wait && decoded.has_operand("time_hint") {
        Some(decoded.scalar_operand("time_hint")?)
    } else {
        None
    };
    if let Some(hint) = &time_hint {
        require_dtype(
            hint,
            crate::dtypes::integer_dtypes(),
            &format!("{op_name}.time_hint"),
        )?;
    }
    Ok(MbarrierQueryParts {
        destination,
        address,
        state_or_phase,
        sem,
        try_wait,
        state_query,
    })
}

pub fn mbarrier_query_marker(
    decoded: &DecodedPtx,
    sem: &str,
    try_wait: bool,
    state_query: bool,
) -> String {
    let mut marker = if try_wait { "TryWait" } else { "TestWait" }.to_owned();
    marker.push_str(if state_query { "State" } else { "Parity" });
    if sem == "relaxed" {
        marker.push_str("Relaxed");
    }
    if decoded.modifier_or_empty("phase_type") == "phase_type::conditional" {
        marker.push_str("<true>");
    }
    if is_report_query(&decoded.op_name) {
        marker = format!("Report<v2::sync::variant::{marker}>");
    }
    marker
}

/// `(response, barrier, space)`; the addresses are the shared sources the
/// operands retain.
pub fn decoded_clc_try_cancel_parts(
    decoded: &DecodedPtx,
) -> AResult<(ObjectRef, ObjectRef, String)> {
    let op_name = decoded.op_name.as_str();
    require_decoded_modifiers(
        decoded,
        &[
            ("action", &["try_cancel"]),
            ("async_", &["async"]),
            ("space", &["", "shared::cta"]),
            ("completion", &["mbarrier::complete_tx::bytes"]),
            ("multicast", &["", "multicast::cluster::all"]),
            ("type", &["b128"]),
        ],
    )?;
    let response = decoded.scalar_operand("addr")?;
    let barrier = decoded.scalar_operand("mbar")?;
    let (response, _) = shared_pointer_source(
        &response,
        &format!("{op_name}.addr"),
        SharedAddressForms::SYNC,
    )?;
    let (barrier, _) = shared_pointer_source(
        &barrier,
        &format!("{op_name}.mbar"),
        SharedAddressForms::SYNC,
    )?;
    Ok((response, barrier, decoded.modifier("space")?.to_owned()))
}

pub struct CheckLayoutParts {
    pub destination: ObjectRef,
    /// The shared source `addr` retains.
    pub address: ObjectRef,
    pub layout: i64,
}

pub fn check_layout_parts(decoded: &DecodedPtx) -> AResult<CheckLayoutParts> {
    let op_name = decoded.op_name.as_str();
    let destination = decoded.scalar_operand("matches")?;
    if destination.as_node::<TensorLoadObj>().is_none()
        || !matches!(
            sync_dtype(&destination).as_str(),
            "bool" | "uint32" | "int32"
        )
    {
        return unsupported(format!("{op_name} requires a predicate destination lvalue"));
    }
    let (address, _) = shared_pointer_source(
        &decoded.scalar_operand("addr")?,
        &format!("{op_name}.addr"),
        SharedAddressForms::SYNC,
    )?;
    let layout = decoded
        .modifier("layout")?
        .chars()
        .last()
        .and_then(|digit| digit.to_digit(10))
        .map(i64::from);
    let Some(layout) = layout else {
        return crate::analyze::util::not_covered(format!(
            "{op_name} layout does not end in a digit"
        ));
    };
    Ok(CheckLayoutParts {
        destination,
        address,
        layout,
    })
}

/// `(destination, state)`.
pub fn pending_count_parts(decoded: &DecodedPtx) -> AResult<(ObjectRef, ObjectRef)> {
    let op_name = decoded.op_name.as_str();
    let destination = decoded.scalar_operand("count")?;
    let state = decoded.scalar_operand("state")?;
    if destination.as_node::<TensorLoadObj>().is_none() || sync_dtype(&destination) != "uint32" {
        return unsupported(format!(
            "{op_name}.count requires a uint32 TensorLoad lvalue"
        ));
    }
    require_dtype(&state, &["int64", "uint64"], &format!("{op_name}.state"))?;
    Ok((destination, state))
}

/// The fence scope marker and its optional ordering.
pub fn fence_variant(sem: &str, scope: &str) -> String {
    let order = match sem {
        "acquire" => ", v2::sync::variant::Acquire",
        "release" => ", v2::sync::variant::Release",
        "sc" => ", v2::sync::variant::Sc",
        _ => "",
    };
    format!(
        "v2::sync::variant::Fence<v2::sync::variant::{}{order}>",
        crate::tables::scope_marker(scope)
    )
}

pub fn proxy_marker(space: &str) -> &'static str {
    match space {
        "" => "All",
        "global" => "Global",
        "shared::cta" => "SharedCta",
        "shared::cluster" => "SharedCluster",
        "alias" => "Alias",
        _ => unreachable!("validated proxy space"),
    }
}

/// The parsed parts of one PTX synchronization instruction.
pub enum SyncParts {
    CheckLayout(CheckLayoutParts),
    /// The shared source the barrier address retains.
    Inval(ObjectRef),
    /// `(destination, state)`.
    PendingCount(ObjectRef, ObjectRef),
    BarReduce(BarReduceParts),
    NamedBarrier(NamedBarrierParts),
    /// `(action, sem, aligned)`.
    ClusterBarrier(String, String, bool),
    MbarrierStatement(MbarrierStatementParts),
    MbarrierQuery(MbarrierQueryParts),
    /// `(response, barrier)` shared sources.
    ClcTryCancel(ObjectRef, ObjectRef),
    MbarrierInit(MbarrierInitParts),
    /// The proxy space.
    FenceProxy(String),
    FenceMbarrierInit,
    /// Whether the action is `wait`.
    Griddepcontrol(bool),
    /// `(sem, scope)`.
    Fence(String, String),
    /// `(increase, count)`.
    Setmaxnreg(bool, i64),
}

/// The parsed parts of one PTX synchronization call.
pub fn ptx_sync_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<SyncParts> {
    decoded.require_void_result_type()?;
    let op_name = decoded.op_name.as_str();
    if op_name == "tirx.ptx.mbarrier_check_layout" {
        let parts = check_layout_parts(decoded)?;
        return Ok(SyncParts::CheckLayout(parts));
    }
    if op_name == "tirx.ptx.mbarrier_inval" {
        let (address, _) = shared_pointer_source(
            &decoded.scalar_operand("addr")?,
            &format!("{op_name}.addr"),
            SharedAddressForms::SYNC,
        )?;
        return Ok(SyncParts::Inval(address));
    }
    if op_name == "tirx.ptx.mbarrier_pending_count" {
        let (destination, state) = pending_count_parts(decoded)?;
        return Ok(SyncParts::PendingCount(destination, state));
    }
    if is_bar_reduce_call(op_name) {
        let parts = decoded_bar_reduce_parts(decoded)?;
        return Ok(SyncParts::BarReduce(parts));
    }
    if NAMED_BARRIER_CALLS.contains(&op_name) {
        let parts = decoded_named_barrier_parts(decoded)?;
        return Ok(SyncParts::NamedBarrier(parts));
    }
    if op_name == "tirx.ptx.barrier_cluster_arrive" || op_name == "tirx.ptx.barrier_cluster_wait" {
        let (action, sem, aligned) = decoded_cluster_barrier_parts(decoded)?;
        return Ok(SyncParts::ClusterBarrier(action, sem, aligned));
    }
    if is_mbarrier_statement_call(op_name) {
        let parts = decoded_mbarrier_statement_parts(decoded)?;
        return Ok(SyncParts::MbarrierStatement(parts));
    }
    if is_mbarrier_query_call(op_name) {
        let parts = decoded_mbarrier_query_parts(decoded)?;
        return Ok(SyncParts::MbarrierQuery(parts));
    }
    if op_name == "tirx.ptx.clusterlaunchcontrol_try_cancel" {
        let (response, barrier, _space) = decoded_clc_try_cancel_parts(decoded)?;
        return Ok(SyncParts::ClcTryCancel(response, barrier));
    }
    if op_name == "tirx.ptx.mbarrier_init" {
        let parts = decoded_mbarrier_init_parts(decoded)?;
        return Ok(SyncParts::MbarrierInit(parts));
    }
    if op_name == "tirx.ptx.fence_proxy" {
        let space = decoded_proxy_space(decoded)?;
        return Ok(SyncParts::FenceProxy(space));
    }
    if op_name == "tirx.ptx.fence_mbarrier_init" {
        for (name, value) in [
            ("op_restrict", "mbarrier_init"),
            ("sem", "release"),
            ("scope", "cluster"),
        ] {
            if decoded.modifier(name)? != value {
                return unsupported(format!(
                    "{op_name} requires {name}={:?}, got {:?}",
                    value,
                    decoded.modifier(name)?
                ));
            }
        }
        return Ok(SyncParts::FenceMbarrierInit);
    }
    if op_name == "tirx.ptx.griddepcontrol" {
        let action = decoded_grid_action(decoded)?;
        return Ok(SyncParts::Griddepcontrol(action == "wait"));
    }
    if op_name == "tirx.ptx.fence" {
        let (sem, scope) = decoded_fence_parts(decoded)?;
        return Ok(SyncParts::Fence(sem, scope));
    }
    if op_name == "tirx.ptx.setmaxnreg" {
        let (increase, count) = decoded_setmaxnreg_parts(ctx, decoded)?;
        return Ok(SyncParts::Setmaxnreg(increase, count));
    }
    Err(crate::analyze::util::Failure::Ffi(
        crate::analyze::util::ffi_error(&format!("unhandled PTX sync call {op_name}")),
    ))
}

/// The Synccheck category of a decoded sync call, from its parsed parts.
/// The parts admit any static `setmaxnreg` count;
/// without `allow_invalid_setmaxnreg` the count must be in the engine's range.
pub fn ptx_sync_kind(
    op_name: &str,
    parts: &SyncParts,
    allow_invalid_setmaxnreg: bool,
) -> AResult<SyncCallKind> {
    Ok(match parts {
        SyncParts::CheckLayout(_) | SyncParts::PendingCount(..) | SyncParts::Fence(..) => {
            SyncCallKind::OrderingMarker
        }
        SyncParts::Inval(_) => SyncCallKind::MbarrierInval,
        SyncParts::BarReduce(_) => SyncCallKind::BarSync,
        SyncParts::NamedBarrier(_) => match op_name {
            "tirx.ptx.bar_arrive" | "tirx.ptx.barrier_arrive" => SyncCallKind::BarArrive,
            "tirx.ptx.bar_warp_sync" => SyncCallKind::WarpSync,
            _ => SyncCallKind::BarSync,
        },
        SyncParts::ClusterBarrier(..) => {
            if op_name == "tirx.ptx.barrier_cluster_arrive" {
                SyncCallKind::ClusterArrive
            } else {
                SyncCallKind::ClusterWait
            }
        }
        SyncParts::MbarrierStatement(parts) => {
            if MBARRIER_NO_COMPLETE_CALLS.contains(&op_name) {
                SyncCallKind::MbarrierArriveNoComplete
            } else {
                SyncCallKind::from_function(&parts.function)
            }
        }
        SyncParts::MbarrierQuery(_) => SyncCallKind::MbarrierTryWait,
        SyncParts::ClcTryCancel(..) => SyncCallKind::ClcTryCancel,
        SyncParts::MbarrierInit(_) => SyncCallKind::MbarrierInit,
        SyncParts::FenceProxy(_) => SyncCallKind::FenceProxyAsync,
        SyncParts::FenceMbarrierInit => SyncCallKind::FenceMbarrierInit,
        SyncParts::Griddepcontrol(wait) => {
            if *wait {
                SyncCallKind::GriddepcontrolWait
            } else {
                SyncCallKind::OrderingMarker
            }
        }
        SyncParts::Setmaxnreg(_, count) => {
            if !allow_invalid_setmaxnreg && (!(24..=256).contains(count) || count % 8 != 0) {
                return unsupported(format!(
                    "{op_name}.register_count must be a multiple of 8 in 24..=256, got {count}"
                ));
            }
            SyncCallKind::Setmaxnreg
        }
    })
}

/// The mbarrier multicast base op names (emission needs the membership).
pub fn is_multicast_call(name: &str) -> bool {
    multicast_base(name).is_some()
}

pub fn is_no_complete_call(name: &str) -> bool {
    MBARRIER_NO_COMPLETE_CALLS.contains(&name)
}

/// `bar.warp.sync` takes an explicit member mask; every lane.
fn all_lanes() -> String {
    abi::splat("u32::MAX")
}

/// An mbarrier operand always lives in shared space.
fn shared_address(pointer: &RustValue) -> String {
    abi::address("v2::Shared", &abi::cloned(&pointer.code), None)
}

impl<'a> Emitter<'a> {
    pub fn sync_mask(&mut self, predicate: Option<&ObjectRef>) -> AResult<String> {
        let Some(predicate) = predicate else {
            return Ok("ctx.active_mask()".to_owned());
        };
        let value = self.emit_expr(predicate)?;
        if value.rust_type != "bool" {
            return unsupported("mbarrier remote predicate must be boolean");
        }
        let name = self.control_name("sync_mask");
        if value.uniformity == Uniformity::Uniform {
            self.emit_line(&format!(
                "let {name} = if {} {{ ctx.active_mask() }} else {{ WarpMask::EMPTY }};",
                value.code
            ));
        } else {
            let value = if value.is_mask {
                value
            } else {
                let lane = self.boolean_lane(&value)?;
                self.emit_varying_mask(&lane, crate::emit::ControlProvenance::None)
            };
            self.emit_line(&format!("let {name} = ctx.active_mask() & {};", value.code));
        }
        Ok(name)
    }

    pub fn v2_register_operand(
        &mut self,
        expr: &ObjectRef,
        rust_type: &str,
        prefix: &str,
    ) -> AResult<String> {
        let value = self.emit_expr(expr)?;
        let value = self.coerce_value(value, rust_type, prefix)?;
        let value = self.as_warp_value(value);
        Ok(abi::register(&value.code))
    }

    /// The register operand of an amount; a default amount is `IntImm("int64", 1)`.
    fn v2_register_amount(
        &mut self,
        amount: &AmountOperand,
        rust_type: &str,
        prefix: &str,
    ) -> AResult<String> {
        match amount {
            AmountOperand::Expression(expr) => self.v2_register_operand(expr, rust_type, prefix),
            AmountOperand::One => {
                let value = RustValue::new("1_i64", "i64", Uniformity::Uniform);
                let value = self.coerce_value(value, rust_type, prefix)?;
                let value = self.as_warp_value(value);
                Ok(abi::register(&value.code))
            }
        }
    }

    /// An unspecialized `(warp, ctx, site)` call.
    fn plain_sync_call(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &[String],
    ) -> String {
        let site = self.v2_site(Some(source_op_id));
        abi::warp_call(function, &site, arguments, None, None, false, true)
    }

    /// The suspending form, terminated as a statement.
    fn awaited_sync_call(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &[String],
    ) -> String {
        let site = self.v2_site(Some(source_op_id));
        format!(
            "{};",
            abi::warp_call(function, &site, arguments, None, None, true, true)
        )
    }

    fn emit_sync_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &[String],
        variant: Option<&str>,
        context: Option<&str>,
    ) {
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(function, &site, arguments, variant, context, false, true);
        self.emit_line(&format!("{call};"));
    }

    /// The pointer of a shared source the analysis retained.
    fn emit_sync_shared_pointer(&mut self, source: &ObjectRef) -> AResult<RustValue> {
        self.emit_address_pointer(source, "shared", None, "ctx.active_mask()")
    }

    pub(super) fn emit_scope_barrier(&mut self, scope: &str, source_op_id: i64) -> AResult<()> {
        let mask = self.sync_mask(None)?;
        self.emit_line(&format!("if !{mask}.is_empty() {{"));
        self.indent += 1;
        let (barrier_id, expected) = match scope {
            "cta" => (
                "0_i64".to_owned(),
                format!("{}_i64", self.warps_per_cta * 32),
            ),
            "warpgroup" => (
                "8_i64".to_owned(),
                format!("{}_i64", self.threads_per_warpgroup),
            ),
            other => return unsupported(format!("unsupported source scope barrier {:?}", other)),
        };
        let line = self.awaited_sync_call(BAR_SYNC, source_op_id, &[barrier_id, expected]);
        self.emit_suspend_line(&line);
        self.indent -= 1;
        self.emit_line("}");
        Ok(())
    }

    fn emit_setmaxnreg(&mut self, increase: bool, count: i64, source_op_id: i64) -> AResult<()> {
        self.sync.uses_setmaxnreg = true;
        if increase && (24..=256).contains(&count) && count % 8 == 0 {
            let current = self.sync.max_valid_increase_target.unwrap_or(count);
            self.sync.max_valid_increase_target = Some(count.max(current));
        }
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            SETMAXNREG,
            &site,
            &[format!("{count}_u32")],
            Some(&setmaxnreg_variant(increase)),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("{call};"));
        Ok(())
    }

    fn emit_fence_mbarrier_init(&mut self, source_op_id: i64) {
        let call = self.plain_sync_call(FENCE_MBARRIER_INIT, source_op_id, &[]);
        self.emit_line(&format!("{call};"));
    }

    fn emit_griddepcontrol(&mut self, source_op_id: i64) {
        let call = self.plain_sync_call(GRIDDEPCONTROL, source_op_id, &[]);
        self.emit_line(&format!("{call};"));
    }

    /// Non-PTX `tirx.cuda.mbarrier_wait*`.
    fn emit_mbarrier_wait(
        &mut self,
        args: &[ObjectRef],
        function: &str,
        specialization: Option<&str>,
        source_op_id: i64,
    ) -> AResult<()> {
        let mask = self.sync_mask(None)?;
        self.emit_line(&format!("if !{mask}.is_empty() {{"));
        self.indent += 1;
        let barrier = &args[0];
        let pointer = if dtype_of(barrier)? == "uint32" {
            self.emit_raw_shared_pointer(barrier, None, "ctx.active_mask()")?
        } else {
            self.shared_pointer(barrier, "mbarrier.try_wait pointer")?
        };
        let phase = self.emit_expr(&args[1])?;
        let phase = self.as_i64(phase)?;
        let phase = self.as_warp_value(phase);
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            function,
            &site,
            &[format!(
                "({}, {})",
                shared_address(&pointer),
                abi::register(&phase.code)
            )],
            specialization,
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("{call};"));
        self.indent -= 1;
        self.emit_line("}");
        Ok(())
    }

    /// `tirx.cuda.warpgroup_sync`.
    fn emit_warpgroup_sync(
        &mut self,
        args: &[ObjectRef],
        function: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let mask = self.sync_mask(None)?;
        self.emit_line(&format!("if !{mask}.is_empty() {{"));
        self.indent += 1;
        let barrier_id = self.uniform_i64(&args[0], "warpgroup_sync barrier id")?;
        let expected = format!("{}_i64", self.threads_per_warpgroup);
        let line = self.awaited_sync_call(function, source_op_id, &[barrier_id.code, expected]);
        self.emit_suspend_line(&line);
        self.indent -= 1;
        self.emit_line("}");
        Ok(())
    }

    /// `functions` are the arrive and wait calls.
    fn emit_cluster_sync(&mut self, functions: &[&str], source_op_id: i64) {
        self.emit_sync_stateful(
            functions[0],
            source_op_id,
            &["()".to_owned()],
            Some("v2::sync::variant::ClusterArrive<v2::sync::variant::DefaultRelease, true>"),
            None,
        );
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            functions[1],
            &site,
            &["()".to_owned()],
            Some("v2::sync::variant::ClusterWait<true>"),
            None,
            true,
            true,
        );
        self.emit_suspend_line(&format!("{call};"));
    }

    fn emit_debug_printf(&mut self, args: &[ObjectRef], source_op_id: i64) -> AResult<()> {
        for argument in args {
            if argument.as_node::<StringImmObj>().is_some() {
                continue;
            }
            let value = self.emit_expr(argument)?;
            let marker_arg = self.control_name("ordering_arg");
            self.emit_line(&format!("let {marker_arg} = &({});", value.code));
            self.emit_line(&format!("let _ = {marker_arg};"));
        }
        let marker = self.control_name("debug_printf");
        self.emit_line(&format!("let {marker}: u64 = {source_op_id}_u64;"));
        Ok(())
    }

    /// Non-PTX synchronization calls.
    fn emit_raw_sync(
        &mut self,
        expr: &ObjectRef,
        raw: RawSyncCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let args: Vec<ObjectRef> = expr
            .as_node::<CallObj>()
            .map(|call| call.args.iter().map(oref).collect())
            .unwrap_or_default();
        let (functions, specialization) = raw.engine_calls();
        match raw {
            RawSyncCall::MbarrierWait => {
                self.emit_mbarrier_wait(&args, functions[0], specialization, source_op_id)
            }
            RawSyncCall::WarpSync => {
                let line = self.awaited_sync_call(functions[0], source_op_id, &[all_lanes()]);
                self.emit_suspend_line(&line);
                Ok(())
            }
            RawSyncCall::WarpgroupSync => {
                self.emit_warpgroup_sync(&args, functions[0], source_op_id)
            }
            RawSyncCall::CtaSync | RawSyncCall::StorageSync => {
                self.emit_scope_barrier("cta", source_op_id)
            }
            RawSyncCall::ClusterSync => {
                self.emit_cluster_sync(functions, source_op_id);
                Ok(())
            }
            RawSyncCall::GridSync => {
                let line = self.awaited_sync_call(functions[0], source_op_id, &[]);
                self.emit_suspend_line(&line);
                Ok(())
            }
            RawSyncCall::ThreadFence => {
                self.emit_sync_stateful(
                    functions[0],
                    source_op_id,
                    &["()".to_owned()],
                    specialization,
                    None,
                );
                Ok(())
            }
            RawSyncCall::NanoSleep => {
                let duration = self.v2_register_operand(&args[0], "u32", "nanosleep_duration")?;
                let line = self.awaited_sync_call(functions[0], source_op_id, &[duration]);
                self.emit_suspend_line(&line);
                Ok(())
            }
            RawSyncCall::Printf => self.emit_debug_printf(&args, source_op_id),
        }
    }

    /// `emit_ptx_sync`.
    fn emit_ptx_sync(
        &mut self,
        decoded: &DecodedPtx,
        parts: &SyncParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let predicate = decoded.predicate.as_ref();
        match parts {
            SyncParts::CheckLayout(parts) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "mbarrier_layout",
                    "mbarrier.check_layout predicate must be bool or integer",
                )?;
                let mask = region.mask.clone();
                let body = self.emit_mbarrier_check_layout(
                    parts,
                    region.context.as_deref(),
                    &mask,
                    source_op_id,
                );
                self.close_predicated_region(region);
                body?;
                let dtype = dtype_of(&parts.destination)?;
                self.finish_predicated_destinations(
                    decoded,
                    &[Some(parts.destination.clone())],
                    &dtype,
                    &mask,
                    source_op_id,
                    true,
                )
            }
            SyncParts::Inval(address) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "mbarrier_inval",
                    "mbarrier.inval predicate must be bool or integer",
                )?;
                let body = self.emit_sync_shared_pointer(address).map(|pointer| {
                    self.emit_sync_stateful(
                        MBARRIER_INVAL,
                        source_op_id,
                        &[shared_address(&pointer)],
                        None,
                        region.context.as_deref(),
                    )
                });
                self.close_predicated_region(region);
                body
            }
            SyncParts::PendingCount(destination, state) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "mbarrier_pending_count",
                    "mbarrier.pending_count predicate must be bool or integer",
                )?;
                let mask = region.mask.clone();
                let body = self.emit_mbarrier_pending_count(
                    destination,
                    state,
                    region.context.as_deref(),
                    &mask,
                    source_op_id,
                );
                self.close_predicated_region(region);
                body?;
                self.finish_predicated_destinations(
                    decoded,
                    &[Some(destination.clone())],
                    "uint32",
                    &mask,
                    source_op_id,
                    false,
                )
            }
            SyncParts::BarReduce(parts) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "barrier_red_instruction",
                    "barrier.red instruction predicate must be bool or integer",
                )?;
                let mask = region.mask.clone();
                let body =
                    self.emit_bar_reduce(parts, region.context.as_deref(), &mask, source_op_id);
                self.close_predicated_region(region);
                body
            }
            SyncParts::NamedBarrier(parts) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "barrier_instruction",
                    "barrier instruction predicate must be bool or integer",
                )?;
                let body = self.emit_named_barrier(op_name, parts, source_op_id);
                self.close_predicated_region(region);
                body
            }
            SyncParts::ClusterBarrier(action, sem, aligned) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "cluster_barrier_instruction",
                    "cluster barrier instruction predicate must be bool or integer",
                )?;
                self.emit_cluster_barrier(
                    action,
                    sem,
                    *aligned,
                    region.context.as_deref(),
                    source_op_id,
                );
                self.close_predicated_region(region);
                Ok(())
            }
            SyncParts::MbarrierStatement(parts) => {
                let region = self.open_shadow_predicated_region(
                    parts.predicate.as_ref(),
                    "mbarrier_instruction",
                    "mbarrier predicate must be bool or integer",
                )?;
                let mask = region.mask.clone();
                let body = self.emit_mbarrier_statement(
                    decoded,
                    parts,
                    region.context.as_deref(),
                    &mask,
                    source_op_id,
                );
                self.close_predicated_region(region);
                body
            }
            SyncParts::MbarrierQuery(parts) => {
                self.emit_mbarrier_query(decoded, parts, source_op_id)
            }
            SyncParts::ClcTryCancel(response, barrier) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "clc_try_cancel",
                    "clusterlaunchcontrol.try_cancel predicate must be bool or integer",
                )?;
                let body = self.emit_clc_try_cancel(
                    decoded,
                    response,
                    barrier,
                    region.context.as_deref(),
                    source_op_id,
                );
                self.close_predicated_region(region);
                body
            }
            SyncParts::MbarrierInit(parts) => {
                let region = self.open_shadow_predicated_region(
                    parts.predicate.as_ref(),
                    "mbarrier_init",
                    "mbarrier.init predicate must be bool or integer",
                )?;
                let body = self.emit_mbarrier_init(
                    decoded,
                    parts,
                    region.context.as_deref(),
                    source_op_id,
                );
                self.close_predicated_region(region);
                body
            }
            SyncParts::FenceProxy(_)
            | SyncParts::FenceMbarrierInit
            | SyncParts::Griddepcontrol(_)
            | SyncParts::Fence(..) => {
                let region = self.open_shadow_predicated_region(
                    predicate,
                    "ordering",
                    &format!("{op_name} predicate must be bool or integer"),
                )?;
                self.emit_ordering(parts, source_op_id);
                self.close_predicated_region(region);
                Ok(())
            }
            SyncParts::Setmaxnreg(increase, count) => {
                self.emit_setmaxnreg(*increase, *count, source_op_id)
            }
        }
    }

    /// The `mbarrier.check_layout` region body of `emit_ptx_sync`.
    fn emit_mbarrier_check_layout(
        &mut self,
        parts: &CheckLayoutParts,
        context: Option<&str>,
        instruction_mask: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let pointer = self.emit_sync_shared_pointer(&parts.address)?;
        let result = self.control_name("mbarrier_layout");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            MBARRIER_CHECK_LAYOUT,
            &site,
            &[shared_address(&pointer)],
            Some(&parts.layout.to_string()),
            context,
            false,
            true,
        );
        self.emit_line(&format!("let {result} = v2_register_out({invocation});"));
        let dtype = dtype_of(&parts.destination)?;
        let mut value = RustValue::new(result.clone(), "bool", Uniformity::Varying);
        if dtype != "bool" {
            let rust_type = if dtype == "uint32" { "u32" } else { "i32" };
            let converted = self.control_name("mbarrier_layout_predicate");
            self.emit_line(&format!(
                "let {converted} = {result}.map(|_, value| value as {rust_type});"
            ));
            value = RustValue::new(converted, rust_type, Uniformity::Varying);
        }
        self.emit_explicit_buffer_store(
            &parts.destination,
            value,
            source_op_id,
            None,
            Some(instruction_mask),
            None,
        )
    }

    /// The `mbarrier.pending_count` region body of `emit_ptx_sync`.
    fn emit_mbarrier_pending_count(
        &mut self,
        destination: &ObjectRef,
        state: &ObjectRef,
        context: Option<&str>,
        instruction_mask: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let value = self.v2_register_operand(state, "u64", "mbarrier_state")?;
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            MBARRIER_PENDING_COUNT,
            &site,
            &[value],
            None,
            context,
            false,
            true,
        );
        let result = self.control_name("mbarrier_pending_count");
        self.emit_line(&format!("let {result} = v2_register_out({invocation});"));
        self.emit_explicit_buffer_store(
            destination,
            RustValue::new(result, "u32", Uniformity::Varying),
            source_op_id,
            None,
            Some(instruction_mask),
            None,
        )
    }

    /// The `barrier.red` region body of `emit_ptx_sync`.
    fn emit_bar_reduce(
        &mut self,
        parts: &BarReduceParts,
        context: Option<&str>,
        instruction_mask: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let barrier_id = self.uniform_i64(&parts.barrier_id, "barrier.red id")?.code;
        let count = match &parts.count {
            Some(count) => self.uniform_i64(count, "barrier.red thread count")?.code,
            None => format!("{}_i64", self.warps_per_cta * 32),
        };
        let predicate =
            self.v2_register_operand(&parts.predicate, "bool", "barrier_red_predicate")?;
        let result = self.control_name("barrier_red_result");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            BAR_REDUCE,
            &site,
            &[format!("({barrier_id}, {count}, {predicate})")],
            Some(&parts.variant),
            context,
            true,
            true,
        );
        self.emit_suspend_line(&format!("let {result} = v2_register_out({invocation});"));
        self.emit_explicit_buffer_store(
            &parts.destination,
            RustValue::new(result, "u32", Uniformity::Varying),
            source_op_id,
            None,
            Some(instruction_mask),
            None,
        )
    }

    /// The named-barrier region body of `emit_ptx_sync`.
    fn emit_named_barrier(
        &mut self,
        op_name: &str,
        parts: &NamedBarrierParts,
        source_op_id: i64,
    ) -> AResult<()> {
        if parts.function == "warp" {
            let membermask =
                self.v2_register_operand(&parts.barrier_id, "u32", "bar.warp.sync membermask")?;
            let line = self.awaited_sync_call(BAR_WARP_SYNC, source_op_id, &[membermask]);
            self.emit_suspend_line(&line);
            return Ok(());
        }
        let mask = self.sync_mask(None)?;
        self.emit_line(&format!("if !{mask}.is_empty() {{"));
        self.indent += 1;
        let barrier_id = self.uniform_i64(&parts.barrier_id, &format!("{op_name} barrier id"))?;
        let expected = match &parts.count {
            Some(count) => {
                self.uniform_i64(count, &format!("{op_name} thread count"))?
                    .code
            }
            None => format!("{}_i64", self.warps_per_cta * 32),
        };
        if parts.function == "bar_arrive" {
            let call =
                self.plain_sync_call("sync::bar_arrive", source_op_id, &[barrier_id.code, expected]);
            self.emit_line(&format!("{call};"));
        } else {
            let line = self.awaited_sync_call(
                &format!("sync::{}", parts.function),
                source_op_id,
                &[barrier_id.code, expected],
            );
            self.emit_suspend_line(&line);
        }
        self.indent -= 1;
        self.emit_line("}");
        Ok(())
    }

    /// The cluster-barrier region body of `emit_ptx_sync`.
    fn emit_cluster_barrier(
        &mut self,
        action: &str,
        sem: &str,
        aligned: bool,
        context: Option<&str>,
        source_op_id: i64,
    ) {
        if action == "arrive" {
            self.emit_sync_stateful(
                BARRIER_CLUSTER_ARRIVE,
                source_op_id,
                &["()".to_owned()],
                Some(&cluster_arrive_variant(sem, aligned)),
                context,
            );
        } else {
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                BARRIER_CLUSTER_WAIT,
                &site,
                &["()".to_owned()],
                Some(&cluster_wait_variant(aligned)),
                context,
                true,
                true,
            );
            self.emit_suspend_line(&format!("{call};"));
        }
    }

    /// The mbarrier statement region body of `emit_ptx_sync`.
    fn emit_mbarrier_statement(
        &mut self,
        decoded: &DecodedPtx,
        parts: &MbarrierStatementParts,
        context: Option<&str>,
        instruction_mask: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let contextual = parts.state_destination.is_some()
            || parts.function == "mbarrier_expect_tx"
            || is_no_complete_call(op_name)
            || is_multicast_call(op_name);
        let (pointer_source, _) = shared_pointer_source(
            &parts.address,
            &format!("{op_name}.addr"),
            SharedAddressForms::SYNC,
        )?;
        let address_value = self.emit_expr(&pointer_source)?;
        let integer_address = address_value.rust_type != "PhysicalPtr";
        let pointer = self.emit_address_pointer(
            &pointer_source,
            "shared",
            Some(address_value),
            "ctx.active_mask()",
        )?;
        let mut marker = parts.marker.clone();
        if integer_address
            && (parts.function == "mbarrier_arrive"
                || parts.function == "mbarrier_arrive_expect_tx")
            && parts.state_destination.is_none()
            && (parts.space.is_empty() || parts.space == "shared::cluster")
            && marker.contains("Local")
        {
            marker = marker.replacen("Local", "Remote", 1);
        }
        let mut remote_rank: Option<RustValue> = None;
        if marker.contains("Remote") {
            let rank_name = self.control_name("mbarrier_target_rank");
            self.emit_line(&format!(
                "let {rank_name} = ({}).shared_target_cta_ranks(&ctx, ctx.active_mask())?;",
                pointer.code
            ));
            remote_rank = Some(RustValue::new(rank_name, "i64", Uniformity::Varying));
        }
        let mut operands = vec![shared_address(&pointer)];
        if let Some(amount) = &parts.amount {
            operands.push(self.v2_register_amount(
                amount,
                "i64",
                &format!("{}_count", parts.function),
            )?);
        }
        if is_multicast_call(op_name) {
            let cta_mask = decoded.scalar_operand("cta_mask")?;
            operands.push(self.v2_register_operand(&cta_mask, "u32", "mbarrier_multicast_mask")?);
        }
        if parts.function != "mbarrier_arrive" && parts.function != "mbarrier_arrive_expect_tx" {
            if let Some(rank) = &remote_rank {
                operands.push(abi::register(&rank.code));
            }
        }
        if parts.predicate.is_some() && !contextual {
            operands.push(abi::splat("true"));
        }
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            &format!("sync::{}", parts.function),
            &site,
            &[format!("({})", operands.join(", "))],
            Some(&sync_variant(&marker)),
            context,
            false,
            true,
        );
        match &parts.state_destination {
            None => {
                self.emit_line(&format!("{invocation};"));
                Ok(())
            }
            Some(state_destination) => {
                let result = self.control_name("mbarrier_state");
                self.emit_line(&format!("let {result} = v2_register_out({invocation});"));
                self.emit_explicit_buffer_store(
                    state_destination,
                    RustValue::new(result, "u64", Uniformity::Varying),
                    source_op_id,
                    None,
                    Some(instruction_mask),
                    None,
                )
            }
        }
    }

    /// The mbarrier query branch of `emit_ptx_sync`.
    fn emit_mbarrier_query(
        &mut self,
        decoded: &DecodedPtx,
        parts: &MbarrierQueryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let pointer = self.emit_sync_shared_pointer(&parts.address)?;
        let mut state = self.emit_expr(&parts.state_or_phase)?;
        if !parts.state_query {
            state = self.as_i64(state)?;
        }
        let state = self.as_warp_value(state);
        let marker = mbarrier_query_marker(decoded, &parts.sem, parts.try_wait, parts.state_query);
        let result = self.control_name("mbarrier_query");
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            mbarrier_query_function(parts.try_wait),
            &site,
            &[format!(
                "({}, {})",
                shared_address(&pointer),
                abi::register(&state.code)
            )],
            Some(&sync_variant(&marker)),
            None,
            parts.try_wait,
            true,
        );
        let line = format!("let {result} = {invocation};");
        if parts.try_wait {
            self.emit_suspend_line(&line);
        } else {
            self.emit_line(&line);
        }
        if is_report_query(op_name) {
            for (index, name) in [(1, "report_predicate"), (2, "report_value")] {
                if !decoded.has_operand(name) {
                    continue;
                }
                let dtype = if index == 1 { "u32" } else { "u8" };
                let register = format!("{result}_report_{index}");
                let mut value = format!("{result}.{index}");
                if index == 1 {
                    value.push_str(".map(|_, value| u32::from(value))");
                }
                self.emit_line(&format!("let {register} = v2_register_out({value});"));
                self.emit_explicit_buffer_store(
                    &decoded.scalar_operand(name)?,
                    RustValue::new(register, dtype, Uniformity::Varying),
                    source_op_id,
                    None,
                    None,
                    None,
                )?;
            }
            self.emit_line(&format!("let {result} = {result}.0;"));
        }
        self.emit_line(&format!(
            "let {result} = v2_register_out({result}.map(|_, ready| u32::from(ready)));"
        ));
        self.emit_explicit_buffer_store(
            &parts.destination,
            RustValue::new(result, "u32", Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }

    /// The `clusterlaunchcontrol.try_cancel` region body of `emit_ptx_sync`.
    fn emit_clc_try_cancel(
        &mut self,
        decoded: &DecodedPtx,
        response: &ObjectRef,
        barrier: &ObjectRef,
        context: Option<&str>,
        source_op_id: i64,
    ) -> AResult<()> {
        let response = self.emit_sync_shared_pointer(response)?;
        let barrier = self.emit_sync_shared_pointer(barrier)?;
        let multicast = if decoded.modifier("multicast")?.is_empty() {
            "false"
        } else {
            "true"
        };
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            CLC_TRY_CANCEL,
            &site,
            &[shared_address(&response), shared_address(&barrier)],
            Some(multicast),
            context,
            true,
            true,
        );
        self.emit_suspend_line(&format!("{call};"));
        Ok(())
    }

    /// The `mbarrier.init` region body of `emit_ptx_sync`.
    fn emit_mbarrier_init(
        &mut self,
        decoded: &DecodedPtx,
        parts: &MbarrierInitParts,
        context: Option<&str>,
        source_op_id: i64,
    ) -> AResult<()> {
        let pointer = self.emit_sync_shared_pointer(&parts.address)?;
        let expected = self.v2_register_operand(&parts.count, "i64", "mbarrier_init_count")?;
        self.emit_sync_stateful(
            MBARRIER_INIT,
            source_op_id,
            &[format!("({}, {expected})", shared_address(&pointer))],
            Some(mbarrier_init_variant(decoded)),
            context,
        );
        Ok(())
    }

    /// The `ordering` region body of `emit_ptx_sync` (fence, fence.proxy,
    /// fence.mbarrier_init and griddepcontrol).
    fn emit_ordering(&mut self, parts: &SyncParts, source_op_id: i64) {
        match parts {
            SyncParts::FenceProxy(space) => self.emit_sync_stateful(
                FENCE_PROXY_ASYNC,
                source_op_id,
                &["()".to_owned()],
                Some(&proxy_async_variant(space)),
                None,
            ),
            SyncParts::FenceMbarrierInit => self.emit_fence_mbarrier_init(source_op_id),
            SyncParts::Griddepcontrol(_) => self.emit_griddepcontrol(source_op_id),
            SyncParts::Fence(sem, scope) => self.emit_sync_stateful(
                FENCE,
                source_op_id,
                &["()".to_owned()],
                Some(&fence_variant(sem, scope)),
                None,
            ),
            _ => unreachable!("ordering instruction"),
        }
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = ptx_sync_parts(emitter.ctx, decoded)?;
    ptx_sync_kind(&decoded.op_name, &parts, false)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_sync(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_legacy(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = validate_raw_sync_call(call.node, &call.op_name)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_raw_sync(call.node, parts, source_op_id)?;
    Ok(None)
}

pub const EXTERNAL_GRID_DEPENDENCY_REQUIREMENT: &str = "external_grid_dependency_satisfied";

pub fn external_grid_dependency(ctx: &Ctx, nodes: &[ObjectRef]) -> AResult<bool> {
    for node in nodes {
        if crate::decode::call_name(node)?.as_deref() != Some("tirx.ptx.griddepcontrol") {
            continue;
        }
        let decoded = match ctx.decoding.ptx_decoded(node) {
            Ok(decoded) => decoded,
            Err(crate::analyze::util::Failure::Unsupported { .. }) => continue,
            Err(error) => return Err(error),
        };
        if decoded.modifier_or_empty("action") == "wait" {
            return Ok(true);
        }
    }
    Ok(false)
}
