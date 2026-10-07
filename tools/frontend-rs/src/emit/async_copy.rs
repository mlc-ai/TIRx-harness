//! Validation and emission of the async_copy instruction family.

use crate::analyze::util::{dtype_of, prim, unsupported, AResult};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::atomic_bulk::ptx_atomic_signature;
use crate::emit::register_call::require_register_call;
use crate::emit::{abi, Emitter, RustValue};
use crate::tables::is_integer_dtype;
use tvm::tvm_ffi::object::ObjectRef;

/// The engine functions (below `v2::`) the lowerings call.
pub const CP_ASYNC_COMMIT_GROUP: &str = "async_copy::cp_async_commit_group";
pub const RED_ASYNC: &str = "async_copy::red_async";
pub const ST_ASYNC: &str = "async_copy::st_async";

/// The validated parts of one `st.async`/`red.async` release instruction.
pub struct AsyncReleaseParts {
    pub function: &'static str,
    pub marker: String,
    pub bits: i64,
}

pub fn release_parts(decoded: &DecodedPtx) -> AResult<AsyncReleaseParts> {
    let op_name = decoded.op_name.as_str();
    let reduction = op_name == "tirx.ptx.red_async_release";
    let scope = decoded.modifier("scope")?;
    let mmio = decoded.modifier("mmio")?;
    let suffix = decoded.modifier("type")?;
    if decoded.modifier("sem")? != "release"
        || !(scope == "gpu" || scope == "sys")
        || !["", "global"].contains(&decoded.modifier("space")?)
        || !(mmio.is_empty() || mmio == "mmio")
        || (!mmio.is_empty() && scope != "sys")
    {
        return unsupported(format!(
            "{op_name} requires global release.gpu/sys (MMIO: sys)"
        ));
    }
    let allowed = if reduction {
        ["u32", "s32", "u64", "s64"].contains(&suffix)
    } else {
        ["f32", "f64"].contains(&suffix)
            || (suffix.len() > 1
                && "bus".contains(&suffix[..1])
                && ["8", "16", "32", "64"].contains(&&suffix[1..]))
    };
    if !allowed || (reduction && decoded.modifier("op")? != "add") {
        return unsupported(format!("{op_name} unsupported type/operation"));
    }
    let bits: i64 = suffix[1..].parse().map_err(|_| {
        crate::analyze::util::Failure::Ffi(crate::analyze::util::ffi_error("release type width"))
    })?;
    let marker = if reduction { "RedRelease" } else { "StRelease" };
    let mut scope_marker = scope[..1].to_uppercase();
    scope_marker.push_str(&scope[1..]);
    Ok(AsyncReleaseParts {
        function: if reduction { RED_ASYNC } else { ST_ASYNC },
        marker: format!(
            "v2::async_copy::variant::{marker}<v2::mem::variant::{scope_marker}, {bits}, {}>",
            !mmio.is_empty()
        ),
        bits,
    })
}

/// The validated parts of one `red.async` instruction.
pub struct RedAsyncParts {
    pub value: ObjectRef,
    pub variant: String,
}

pub fn red_async_parts(decoded: &DecodedPtx) -> AResult<RedAsyncParts> {
    let op_name = decoded.op_name.as_str();
    let Some((operation, dtype)) =
        ptx_atomic_signature(decoded.modifier("op")?, decoded.modifier("type")?)
    else {
        return unsupported(format!("{op_name} unsupported reduction type"));
    };
    let value = decoded.scalar_operand("value")?;
    if dtype_of(&value)? != dtype {
        return unsupported(format!("{op_name} value must be {dtype}"));
    }
    let scalar = match dtype {
        "uint32" => "u32",
        "int32" => "i32",
        "uint64" => "u64",
        other => {
            return Err(crate::analyze::util::Failure::Ffi(
                crate::analyze::util::ffi_error(&format!(
                    "red.async has no scalar carrier for {other}"
                )),
            ))
        }
    };
    Ok(RedAsyncParts {
        value,
        variant: format!(
            "v2::async_copy::variant::RedAsync<{scalar}, {}>",
            operation.marker()
        ),
    })
}

/// The validated parts.

fn wait_marker(bulk: bool, read_only: bool) -> &'static str {
    if read_only {
        "v2::async_copy::variant::BulkWaitGroupRead"
    } else if bulk {
        "v2::async_copy::variant::BulkWaitGroup"
    } else {
        "v2::async_copy::variant::CpAsyncWaitGroup"
    }
}

pub struct GroupParts {
    pub function: &'static str,
    pub marker: Option<&'static str>,
    pub pending: Option<i64>,
}

fn static_group(ctx: &Ctx, value: &ObjectRef, op_name: &str) -> AResult<i64> {
    let dtype = dtype_of(value)?;
    if !is_integer_dtype(&dtype) {
        return unsupported(format!(
            "{op_name}.pending_group_count must be an integer, got {dtype}"
        ));
    }
    let simplified = crate::analyze::util::simplify(&ctx.analyzer, &prim(value)?)?;
    let Some(pending) = crate::analyze::util::int_imm_expr(&simplified) else {
        return unsupported(format!("{op_name}.pending_group_count must be static"));
    };
    if pending < 0 {
        return unsupported(format!(
            "{op_name}.pending_group_count must be a non-negative int64, got {pending}"
        ));
    }
    Ok(pending)
}

pub fn decoded_group_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<GroupParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    if let Some(predicate) = &decoded.predicate {
        let dtype = dtype_of(predicate)?;
        if !(is_integer_dtype(&dtype) || dtype == "bool") {
            return unsupported(format!("{op_name} predicate must lower to bool or integer"));
        }
    }
    if decoded.modifier("api")? != "async" {
        return unsupported(format!("{op_name} requires api='async'"));
    }
    let bulk = op_name == "tirx.ptx.cp_async_bulk_commit_group"
        || op_name == "tirx.ptx.cp_async_bulk_wait_group";
    if bulk && decoded.modifier("kind")? != "bulk" {
        return unsupported(format!("{op_name} requires kind='bulk'"));
    }
    let wait = matches!(
        op_name,
        "tirx.ptx.cp_async_wait_group"
            | "tirx.ptx.cp_async_wait_all"
            | "tirx.ptx.cp_async_bulk_wait_group"
    );
    let expected_action = if op_name == "tirx.ptx.cp_async_wait_all" {
        "wait_all"
    } else if wait {
        "wait_group"
    } else {
        "commit_group"
    };
    if decoded.modifier("action")? != expected_action {
        return unsupported(format!("{op_name} requires action={:?}", expected_action));
    }
    if wait {
        let pending = if op_name == "tirx.ptx.cp_async_wait_all" {
            0
        } else {
            static_group(ctx, &decoded.scalar_operand("group")?, op_name)?
        };
        let read_only = bulk && decoded.modifier("read")? == "read";
        let function = if bulk {
            "async_copy::cp_async_bulk_wait_group"
        } else {
            "async_copy::cp_async_wait_group"
        };
        return Ok(GroupParts {
            function,
            marker: Some(wait_marker(bulk, read_only)),
            pending: Some(pending),
        });
    }
    Ok(GroupParts {
        function: if bulk {
            "async_copy::cp_async_bulk_commit_group"
        } else {
            CP_ASYNC_COMMIT_GROUP
        },
        marker: None,
        pending: None,
    })
}

/// The validated parts.

pub struct StAsyncParts {
    pub destination: ObjectRef,
    pub values: Vec<ObjectRef>,
    pub barrier: ObjectRef,
    pub bits: i64,
}

pub fn decoded_st_async_parts(decoded: &DecodedPtx) -> AResult<StAsyncParts> {
    let op_name = decoded.op_name.as_str();
    require_register_call(decoded, false)?;
    let expected: [(&str, &[&str]); 3] = [
        ("weak", &["", "weak"]),
        ("space", &["", "shared::cluster"]),
        ("completion", &["mbarrier::complete_tx::bytes"]),
    ];
    for (name, allowed) in expected {
        let actual = decoded.modifier(name)?;
        if !allowed.contains(&actual) {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                &allowed, actual
            ));
        }
    }
    let destination = decoded.scalar_operand("addr")?;
    let lanes: Vec<Option<ObjectRef>> = decoded.operand("b")?.to_vec();
    let barrier = decoded.scalar_operand("mbar")?;
    let bits: i64 = decoded.modifier("type")?[1..].parse().map_err(|_| {
        crate::analyze::util::Failure::Ffi(crate::analyze::util::ffi_error("st.async type width"))
    })?;
    let count = lanes.len() as i64;
    if ![32, 64, 128].contains(&bits) || ![32, 64, 128].contains(&(bits * count)) {
        return unsupported(format!("{op_name} requires a 4, 8, or 16-byte payload"));
    }
    let mut values = Vec::new();
    for lane in lanes {
        match lane {
            Some(value) => values.push(value),
            None => {
                return Err(crate::analyze::util::Failure::Ffi(
                    crate::analyze::util::ffi_error("sunk lane in a st.async payload"),
                ))
            }
        }
    }
    Ok(StAsyncParts {
        destination,
        values,
        barrier,
        bits,
    })
}

/// The `st.async` variant: one 32-bit word per payload register.
pub fn st_async_variant(parts: &StAsyncParts) -> String {
    format!(
        "v2::async_copy::variant::StAsyncClusterMappedCompleteTxBytes<{}>",
        parts.values.len() as i64 * parts.bits / 32
    )
}

/// The validated parts.

impl<'a> Emitter<'a> {
    pub fn emit_async_release(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AsyncReleaseParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "async_release",
            "async release predicate must be bool or integer",
        )?;
        let pointer = self.emit_address_pointer(
            &decoded.scalar_operand("addr")?,
            "global",
            None,
            "ctx.active_mask()",
        )?;
        let value = self.emit_as_unsigned_bits(
            &decoded.scalar_operand(if decoded.op_name == "tirx.ptx.red_async_release" {
                "value"
            } else {
                "b"
            })?,
            parts.bits,
            &decoded.op_name,
            "async_release_bits",
            None,
        )?;
        let operand = abi::register(&format!(
            "WarpValue::from_fn(|lane| {}[lane] as u64)",
            value.code
        ));
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            parts.function,
            &site,
            &[format!(
                "({}, {operand})",
                abi::address("v2::Global", &abi::cloned(&pointer.code), None)
            )],
            Some(&parts.marker),
            region.context.as_deref(),
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        self.close_predicated_region(region);
        Ok(())
    }

    pub fn emit_red_async(
        &mut self,
        decoded: &DecodedPtx,
        parts: &RedAsyncParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "red_async",
            "red.async predicate must be bool or integer",
        )?;
        let mut addresses = Vec::new();
        for field in ["addr", "mbar"] {
            let pointer = self.emit_raw_shared_pointer(
                &decoded.scalar_operand(field)?,
                None,
                "ctx.active_mask()",
            )?;
            addresses.push(abi::address(
                "v2::Shared",
                &abi::cloned(&pointer.code),
                None,
            ));
        }
        let value = self.emit_expr(&parts.value)?;
        let value = self.as_warp_value(value);
        let operand = abi::register(&value.code);
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            RED_ASYNC,
            &site,
            &[format!("({}, {operand}, {})", addresses[0], addresses[1])],
            Some(&parts.variant),
            region.context.as_deref(),
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        self.close_predicated_region(region);
        Ok(())
    }

    pub fn emit_async_group(
        &mut self,
        decoded: &DecodedPtx,
        parts: &GroupParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "async_group",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let context = region.context.clone();
        if decoded.op_name == "tirx.ptx.cp_async_wait_all" {
            // The implicit commit and wait belong to the same issuing lanes.
            let site = self.v2_site(Some(source_op_id));
            let call = abi::warp_call(
                CP_ASYNC_COMMIT_GROUP,
                &site,
                &[],
                None,
                context.as_deref(),
                false,
                true,
            );
            self.emit_line(&format!("{call};"));
        }
        let site = self.v2_site(Some(source_op_id));
        match parts.pending {
            None => {
                let call = abi::warp_call(
                    parts.function,
                    &site,
                    &[],
                    None,
                    context.as_deref(),
                    false,
                    true,
                );
                self.emit_line(&format!("{call};"));
            }
            Some(pending) => {
                let call = abi::warp_call(
                    parts.function,
                    &site,
                    &[format!("{pending}_i64")],
                    Some(parts.marker.expect("wait marker")),
                    context.as_deref(),
                    true,
                    true,
                );
                self.emit_suspend_line(&format!("{call};"));
            }
        }
        self.close_predicated_region(region);
        Ok(())
    }

    pub fn emit_st_async(
        &mut self,
        decoded: &DecodedPtx,
        parts: &StAsyncParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "st_async",
            "st.async predicate must be bool or integer",
        )?;
        let destination =
            self.emit_raw_shared_pointer(&parts.destination, None, "ctx.active_mask()")?;
        let barrier = self.emit_raw_shared_pointer(&parts.barrier, None, "ctx.active_mask()")?;
        let mut words = Vec::new();
        for (index, expression) in parts.values.iter().enumerate() {
            let value = self.emit_as_unsigned_bits(
                expression,
                parts.bits,
                &decoded.op_name,
                &format!("st_async_bits_{index}"),
                None,
            )?;
            for word in 0..(parts.bits / 32) {
                let payload = format!(
                    "{}[lane]{}",
                    value.code,
                    if parts.bits == 128 {
                        format!("[{}]", word / 2)
                    } else {
                        String::new()
                    }
                );
                let shift = (word % 2) * 32;
                let name = self.control_name("st_async_word");
                self.emit_line(&format!(
                    "let {name} = WarpValue::from_fn(|lane| ({payload} >> {shift}) as u32);"
                ));
                words.push(abi::register(&name));
            }
        }
        let site = self.v2_site(Some(source_op_id));
        let call = abi::warp_call(
            ST_ASYNC,
            &site,
            &[format!(
                "({}, {}, [{}])",
                abi::address("v2::Shared", &abi::cloned(&destination.code), None),
                abi::address("v2::Shared", &abi::cloned(&barrier.code), None),
                words.join(", ")
            )],
            Some(&st_async_variant(parts)),
            region.context.as_deref(),
            false,
            true,
        );
        self.emit_line(&format!("{call};"));
        self.close_predicated_region(region);
        Ok(())
    }
}

pub fn emit_group(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = decoded_group_parts(emitter.ctx, decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_async_group(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_store(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = decoded_st_async_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_st_async(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_release(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = release_parts(decoded)?;
    emitter.record_pointer_write(&decoded.scalar_operand("addr")?, "global")?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_async_release(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_reduce(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = red_async_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_red_async(decoded, &parts, source_op_id)?;
    Ok(None)
}
