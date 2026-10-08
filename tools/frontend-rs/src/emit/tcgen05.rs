//! Validation and emission of the tcgen05 instruction family.

use crate::analyze::util::{dtype_of, int_imm_expr, prim, simplify, unsupported, AResult};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::{shared_pointer_source, SharedAddressForms};
use crate::emit::{abi, Emitter, RustValue};
use tvm::tvm_ffi::object::ObjectRef;

pub const TCGEN_CONTROL_CALLS: [&str; 12] = [
    "tirx.ptx.tcgen05_alloc",
    "tirx.ptx.tcgen05_alloc_exclusive",
    "tirx.ptx.tcgen05_commit",
    "tirx.ptx.tcgen05_commit_multicast",
    "tirx.ptx.tcgen05_commit_multicast_width",
    "tirx.ptx.tcgen05_commit_sync_restrict",
    "tirx.ptx.tcgen05_commit_sync_restrict_multicast",
    "tirx.ptx.tcgen05_dealloc",
    "tirx.ptx.tcgen05_dealloc_exclusive",
    "tirx.ptx.tcgen05_fence",
    "tirx.ptx.tcgen05_relinquish_alloc_permit",
    "tirx.ptx.tcgen05_wait",
];

fn require_modifiers(decoded: &DecodedPtx, expected: &[(&str, &[&str])]) -> AResult<()> {
    for (name, allowed) in expected {
        let value = decoded.modifier(name)?;
        if !allowed.contains(&value) {
            return unsupported(format!(
                "{} modifier {:?} must be one of {:?}, got {:?}",
                decoded.op_name, name, &allowed, value
            ));
        }
    }
    Ok(())
}

pub struct AllocationParts {
    pub address: ObjectRef,
    /// The runtime column count operand.
    pub columns: ObjectRef,
    pub group: i64,
}

pub fn allocation_parts(decoded: &DecodedPtx, op_name: &str) -> AResult<AllocationParts> {
    decoded.require_void_result_type()?;
    if decoded.op_name.ends_with("_exclusive") {
        require_modifiers(decoded, &[("exclusive", &["exclusive"])])?;
    }
    let address = if op_name == "tirx.ptx.tcgen05_alloc" {
        require_modifiers(
            decoded,
            &[
                ("action", &["alloc"]),
                ("sync", &["sync"]),
                ("aligned", &["aligned"]),
                ("space", &["", "shared::cta"]),
                ("type", &["b32"]),
            ],
        )?;
        decoded.scalar_operand("dst")?
    } else {
        require_modifiers(
            decoded,
            &[
                ("action", &["dealloc"]),
                ("sync", &["sync"]),
                ("aligned", &["aligned"]),
                ("type", &["b32"]),
            ],
        )?;
        decoded.scalar_operand("taddr")?
    };
    let columns = decoded.scalar_operand("ncols")?;
    Ok(AllocationParts {
        address,
        columns,
        group: decoded.cta_group(None)?,
    })
}

pub fn allocation_variant(decoded: &DecodedPtx, name: &str) -> String {
    let suffix = if decoded.op_name.ends_with("_exclusive") {
        "<true>"
    } else {
        ""
    };
    format!("v2::tcgen05::variant::{name}{suffix}")
}

/// The engine function (below `v2::`) and the variant of one parsed control
/// form, which the lowering calls.
pub fn engine_call(decoded: &DecodedPtx, control: &TcgenControl) -> (String, Option<String>) {
    match control {
        TcgenControl::Alloc(_) => (
            "tcgen05::alloc".to_owned(),
            Some(allocation_variant(decoded, "Alloc")),
        ),
        TcgenControl::Commit(parts) => ("tcgen05::commit".to_owned(), Some(commit_variant(parts))),
        TcgenControl::Dealloc(_) => (
            "tcgen05::dealloc".to_owned(),
            Some(allocation_variant(decoded, "Dealloc")),
        ),
        TcgenControl::Fence(marker) => (
            "tcgen05::fence".to_owned(),
            Some(format!("v2::tcgen05::variant::{marker}")),
        ),
        TcgenControl::Relinquish(_) => (
            "tcgen05::relinquish_alloc_permit".to_owned(),
            Some("v2::tcgen05::variant::Relinquish".to_owned()),
        ),
        TcgenControl::Wait(action) => (
            format!(
                "tcgen05::wait_{}",
                action.strip_prefix("wait::").unwrap_or(action)
            ),
            None,
        ),
    }
}

pub struct CommitParts {
    pub barrier: ObjectRef,
    pub cta_mask: Option<ObjectRef>,
    pub group: i64,
    /// The `sync_restrict::shared::read::mma::a` commit.
    pub shared_a: bool,
}

pub fn commit_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<CommitParts> {
    decoded.require_void_result_type()?;
    let op_name = decoded.op_name.as_str();
    let mut expected: Vec<(&str, &[&str])> = vec![
        ("action", &["commit"]),
        ("completion", &["mbarrier::arrive::one"]),
        ("space", &["", "shared::cluster"]),
        ("type", &["b64"]),
    ];
    if op_name == "tirx.ptx.tcgen05_commit_multicast" {
        expected.push(("multicast", &["multicast::cluster"]));
    } else if op_name == "tirx.ptx.tcgen05_commit_multicast_width" {
        expected.push((
            "multicast",
            &["multicast::cluster::16b", "multicast::cluster::32b"],
        ));
    } else if op_name == "tirx.ptx.tcgen05_commit_sync_restrict_multicast" {
        expected.push((
            "multicast",
            &[
                "multicast::cluster",
                "multicast::cluster::16b",
                "multicast::cluster::32b",
            ],
        ));
    }
    let shared_a = op_name == "tirx.ptx.tcgen05_commit_sync_restrict"
        || op_name == "tirx.ptx.tcgen05_commit_sync_restrict_multicast";
    if shared_a {
        expected.push(("sync_restrict", &["sync_restrict::shared::read::mma::a"]));
    }
    require_modifiers(decoded, &expected)?;
    let barrier = decoded.scalar_operand("mbar")?;
    let mut cta_mask = if op_name == "tirx.ptx.tcgen05_commit_multicast" {
        Some(decoded.scalar_operand("mask")?)
    } else {
        None
    };
    if op_name == "tirx.ptx.tcgen05_commit_multicast_width"
        || op_name == "tirx.ptx.tcgen05_commit_sync_restrict_multicast"
    {
        cta_mask = Some(decoded.scalar_operand("cta_mask")?);
    }
    if let Some(mask) = &cta_mask {
        let simplified = simplify(&ctx.analyzer, &prim(mask)?)?;
        if int_imm_expr(&simplified).is_some_and(|value| value < 0) {
            return unsupported(format!("{op_name} mask must be nonnegative"));
        }
    }
    Ok(CommitParts {
        barrier,
        cta_mask,
        group: decoded.cta_group(None)?,
        shared_a,
    })
}

pub fn commit_variant(parts: &CommitParts) -> String {
    let marker = if parts.shared_a {
        "CommitSharedA"
    } else {
        "Commit"
    };
    let multicast = if parts.cta_mask.is_some() {
        "Multicast"
    } else {
        ""
    };
    format!("v2::tcgen05::variant::{marker}{multicast}")
}

/// `"wait::ld"` or `"wait::st"`.
pub fn wait_action(decoded: &DecodedPtx) -> AResult<String> {
    decoded.require_void_result_type()?;
    require_modifiers(decoded, &[("sync", &["sync"]), ("aligned", &["aligned"])])?;
    let action = decoded.modifier("action")?;
    if action != "wait::ld" && action != "wait::st" {
        return unsupported(format!(
            "{} has unsupported action {:?}",
            "tirx.ptx.tcgen05_wait", action
        ));
    }
    Ok(action.to_owned())
}

/// The variant marker of the fence action.
pub fn fence_marker(decoded: &DecodedPtx) -> AResult<&'static str> {
    decoded.require_void_result_type()?;
    let action = decoded.modifier("action")?;
    match action {
        "fence::before_thread_sync" => Ok("FenceBeforeThreadSync"),
        "fence::after_thread_sync" => Ok("FenceAfterThreadSync"),
        _ => unsupported(format!(
            "{} has unsupported action {:?}",
            "tirx.ptx.tcgen05_fence", action
        )),
    }
}

/// The parsed parts of one TCGEN control instruction.
pub enum TcgenControl {
    Alloc(AllocationParts),
    Dealloc(AllocationParts),
    Commit(CommitParts),
    /// The CTA group.
    Relinquish(i64),
    /// `"wait::ld"` or `"wait::st"`.
    Wait(String),
    /// The fence variant marker.
    Fence(&'static str),
}

/// The parsed control form.
fn control_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<TcgenControl> {
    decoded.require_void_result_type()?;
    let control = match decoded.op_name.as_str() {
        "tirx.ptx.tcgen05_alloc" | "tirx.ptx.tcgen05_alloc_exclusive" => {
            TcgenControl::Alloc(allocation_parts(decoded, "tirx.ptx.tcgen05_alloc")?)
        }
        "tirx.ptx.tcgen05_commit"
        | "tirx.ptx.tcgen05_commit_multicast"
        | "tirx.ptx.tcgen05_commit_multicast_width"
        | "tirx.ptx.tcgen05_commit_sync_restrict"
        | "tirx.ptx.tcgen05_commit_sync_restrict_multicast" => {
            TcgenControl::Commit(commit_parts(ctx, decoded)?)
        }
        "tirx.ptx.tcgen05_dealloc" | "tirx.ptx.tcgen05_dealloc_exclusive" => {
            TcgenControl::Dealloc(allocation_parts(decoded, "tirx.ptx.tcgen05_dealloc")?)
        }
        "tirx.ptx.tcgen05_fence" => TcgenControl::Fence(fence_marker(decoded)?),
        "tirx.ptx.tcgen05_relinquish_alloc_permit" => {
            require_modifiers(
                decoded,
                &[
                    ("action", &["relinquish_alloc_permit"]),
                    ("sync", &["sync"]),
                    ("aligned", &["aligned"]),
                ],
            )?;
            TcgenControl::Relinquish(decoded.cta_group(None)?)
        }
        "tirx.ptx.tcgen05_wait" => TcgenControl::Wait(wait_action(decoded)?),
        other => {
            return Err(crate::analyze::util::Failure::Ffi(
                crate::analyze::util::ffi_error(&format!(
                    "TCGEN control lowering expects DecodedPtxCall({}), got {other}",
                    TCGEN_CONTROL_CALLS.join(", ")
                )),
            ))
        }
    };
    Ok(control)
}

impl<'a> Emitter<'a> {
    /// The shared-pointer operand of alloc/commit (`emit_raw_shared_pointer`
    /// for the public `uint32` carrier, `_pointer` otherwise).
    fn tcgen_shared_pointer(&mut self, source: &ObjectRef, label: &str) -> AResult<RustValue> {
        if dtype_of(source)? == "uint32" {
            self.emit_raw_shared_pointer(source, None, "ctx.active_mask()")
        } else {
            self.pointer(source, label)
        }
    }

    /// A stateful ABI call terminated as a statement, awaited
    /// through `emit_suspend_line` when requested.
    fn emit_tcgen_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &[String],
        variant: Option<&str>,
        await_result: bool,
    ) {
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            function,
            &site,
            arguments,
            variant,
            None,
            await_result,
            true,
        );
        if await_result {
            self.emit_suspend_line(&format!("{invocation};"));
        } else {
            self.emit_line(&format!("{invocation};"));
        }
    }

    fn tcgen_alloc_operand(&mut self, parts: &AllocationParts) -> AResult<String> {
        let columns = self.uniform_i64(&parts.columns, "tcgen05.alloc ncols")?;
        let (pointer_source, _) = shared_pointer_source(
            &parts.address,
            &format!("{}.dst", "tirx.ptx.tcgen05_alloc"),
            SharedAddressForms::TCGEN,
        )?;
        let pointer = self.tcgen_shared_pointer(&pointer_source, "tcgen05.alloc destination")?;
        let address = abi::address("v2::Shared", &abi::cloned(&pointer.code), None);
        Ok(format!(
            "({address}, {} as usize, {}_usize)",
            columns.code, parts.group
        ))
    }

    fn tcgen_dealloc_operand(&mut self, parts: &AllocationParts) -> AResult<String> {
        let columns = self.uniform_i64(&parts.columns, "tcgen05.dealloc ncols")?;
        let address = self.uniform_i64(&parts.address, "tcgen05.dealloc address")?;
        let operand = abi::splat(&format!(
            "u32::try_from({}).map_err(|_| EngineError::message(\"tcgen05.dealloc address is outside uint32\"))?",
            address.code
        ));
        Ok(format!(
            "({operand}, {} as usize, {}_usize)",
            columns.code, parts.group
        ))
    }

    fn tcgen_commit_operand(
        &mut self,
        decoded: &DecodedPtx,
        parts: &CommitParts,
    ) -> AResult<String> {
        let (pointer_source, _) = shared_pointer_source(
            &parts.barrier,
            &format!("{}.mbar", decoded.op_name),
            SharedAddressForms::TCGEN,
        )?;
        let pointer = self.tcgen_shared_pointer(&pointer_source, "tcgen05.commit barrier")?;
        let cta_masks = match &parts.cta_mask {
            None => None,
            Some(mask) => {
                let values = self.emit_expr(mask)?;
                let values = self.as_i64(values)?;
                let values = self.as_warp_value(values);
                Some(abi::register(&values.code))
            }
        };
        let address = abi::address("v2::Shared", &abi::cloned(&pointer.code), None);
        Ok(match &cta_masks {
            None => format!("({address}, {}_u32)", parts.group),
            Some(masks) => format!("({address}, {masks}, {}_u32)", parts.group),
        })
    }

    /// The operands of one control form, then its engine call.
    fn emit_tcgen_call(
        &mut self,
        decoded: &DecodedPtx,
        control: &TcgenControl,
        source_op_id: i64,
    ) -> AResult<()> {
        let operands = match control {
            TcgenControl::Alloc(parts) => vec![self.tcgen_alloc_operand(parts)?],
            TcgenControl::Commit(parts) => vec![self.tcgen_commit_operand(decoded, parts)?],
            TcgenControl::Dealloc(parts) => vec![self.tcgen_dealloc_operand(parts)?],
            TcgenControl::Fence(_) => vec!["()".to_owned()],
            TcgenControl::Relinquish(group) => vec![format!("{group}_usize")],
            TcgenControl::Wait(_) => Vec::new(),
        };
        let (function, variant) = engine_call(decoded, control);
        // The lifecycle calls and the waits suspend; commit and fence do not.
        let await_result = !matches!(control, TcgenControl::Commit(_) | TcgenControl::Fence(_));
        self.emit_tcgen_stateful(
            &function,
            source_op_id,
            &operands,
            variant.as_deref(),
            await_result,
        );
        Ok(())
    }

    /// `emit_tcgen_control`.
    fn emit_tcgen_control(
        &mut self,
        decoded: &DecodedPtx,
        control: &TcgenControl,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "tcgen_control",
            "TCGEN control predicate must lower to bool or integer",
        )?;
        let result = self.emit_tcgen_call(decoded, control, source_op_id);
        self.close_predicated_region(region);
        result
    }
}

pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = control_parts(emitter.ctx, decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_tcgen_control(decoded, &parts, source_op_id)?;
    Ok(None)
}

#[derive(Default)]
pub struct TmemRequirements {
    pub uses_tmem: bool,
    pub dynamic_lifecycle: bool,
}

pub fn scan_tmem_requirements(nodes: &[ObjectRef]) -> AResult<TmemRequirements> {
    let mut requirements = TmemRequirements::default();
    for node in nodes {
        let Some(name) = crate::decode::call_name(node)? else {
            continue;
        };
        requirements.uses_tmem |= name.starts_with("tirx.ptx.tcgen05_")
            || matches!(
                name.as_str(),
                "tirx.cuda.tcgen05_encode_instr_descriptor"
                    | "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled"
                    | "tirx.cuda.tcgen05_encode_matrix_descriptor"
            );
        requirements.dynamic_lifecycle |= matches!(
            name.as_str(),
            "tirx.ptx.tcgen05_alloc" | "tirx.ptx.tcgen05_alloc_exclusive"
        );
    }
    Ok(requirements)
}
