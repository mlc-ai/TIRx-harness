//! Validation and emission of the ptx_warp instruction family.

use crate::analyze::util::{dtype_of, unsupported, upper_first, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::{require_register_call, table_marker, PTX_TYPE_MARKERS};
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

/// The engine functions (below `v2::`) the lowerings call.
pub const ACTIVEMASK: &str = "warp::activemask";
pub const ELECT_SYNC: &str = "warp::elect_sync";
pub const MATCH_SYNC: &str = "warp::match_sync";
pub const MOVMATRIX: &str = "warp::movmatrix";
pub const REDUX_SYNC: &str = "warp::redux_sync";
pub const SHFL_SYNC: &str = "warp::shfl_sync";
pub const VOTE_SYNC: &str = "warp::vote_sync";
/// The `movmatrix` variant: the engine models the b16 transpose alone.
pub const MOVMATRIX_B16: &str = "v2::warp::variant::MovMatrixB16";

/// `(expected mode, mode marker)`.
pub fn match_variant_parts(op_name: &str) -> Option<(&'static str, &'static str)> {
    match op_name {
        "tirx.ptx.match_any_sync" => Some(("any", "Any")),
        "tirx.ptx.match_all_sync" | "tirx.ptx.match_all_sync_p" => Some(("all", "All")),
        _ => None,
    }
}

/// `(mode marker, source-mask function)`.
pub fn shfl_mode(mode: &str) -> Option<(&'static str, &'static str)> {
    match mode {
        "idx" => Some(("Index", "warp::shfl_source_mask_idx")),
        "up" => Some(("Up", "warp::shfl_source_mask_up")),
        "down" => Some(("Down", "warp::shfl_source_mask_down")),
        "bfly" => Some(("Butterfly", "warp::shfl_source_mask_bfly")),
        _ => None,
    }
}

/// The parsed parts of one raw PTX warp call: its validated register
/// destinations, and the specialization of the two specialized families.
pub struct WarpParts {
    pub destinations: Vec<ObjectRef>,
    pub specialization: Option<WarpSpecialization>,
}

/// One `match.sync` or `shfl.sync` specialization: the variant the lowering
/// passes, whether a predicate is written alongside the
/// datum, and the bit width of the source carrier.
pub struct WarpSpecialization {
    pub variant: String,
    pub writes_predicate: bool,
    pub source_bits: i64,
}

/// The specialization of one validated call; `activemask`, `movmatrix`,
/// `redux`, `vote` and `elect` have none.
fn warp_specialization(decoded: &DecodedPtx) -> AResult<Option<WarpSpecialization>> {
    let op_name = decoded.op_name.as_str();
    let (destination, writes_predicate) = match op_name {
        "tirx.ptx.match_all_sync_p" | "tirx.ptx.shfl_sync_p" => ("DataAndPredicate", true),
        _ => ("DataOnly", false),
    };
    let (family, carrier, mode, bits) = if let Some((_, mode)) = match_variant_parts(op_name) {
        let ptx_type = decoded.modifier("type")?;
        let carrier = table_marker(PTX_TYPE_MARKERS, ptx_type).expect("validated match type");
        let bits = if ptx_type == "b32" { 32 } else { 64 };
        ("Match", carrier, mode, bits)
    } else if op_name == "tirx.ptx.shfl_sync" || op_name == "tirx.ptx.shfl_sync_p" {
        let (mode, _) = shfl_mode(decoded.modifier("mode")?).expect("validated shfl mode");
        ("Shfl", "U32", mode, 32)
    } else {
        return Ok(None);
    };
    Ok(Some(WarpSpecialization {
        variant: format!(
            "v2::warp::variant::{family}<v2::reg::variant::{carrier}, v2::warp::variant::{mode}, v2::warp::variant::{destination}>"
        ),
        writes_predicate,
        source_bits: bits,
    }))
}

/// `resolve_ptx_warp`: validate once and return the register destinations.
pub fn resolve_ptx_warp(decoded: &DecodedPtx) -> AResult<Vec<ObjectRef>> {
    let operands = require_register_call(decoded, false)?;
    let destinations = operands.destinations;
    let op_name = decoded.op_name.as_str();
    if op_name == "tirx.ptx.movmatrix" {
        let expected: [(&str, &str); 5] = [
            ("sync", "sync"),
            ("aligned", "aligned"),
            ("shape", "m8n8"),
            ("trans", "trans"),
            ("type", "b16"),
        ];
        let mut actual: Vec<String> = Vec::new();
        let mut matches = true;
        for (name, value) in expected {
            let token = decoded.modifier(name)?;
            if token != value {
                matches = false;
            }
            actual.push(format!("{:?}: {:?}", name, token));
        }
        if !matches {
            return unsupported(format!(
                "{op_name} has unsupported modifiers {{{}}}",
                actual.join(", ")
            ));
        }
        return Ok(destinations);
    }
    if op_name == "tirx.ptx.activemask" {
        if decoded.modifier("type")? != "b32" {
            return unsupported(format!("{op_name} requires the b32 instruction type"));
        }
        return Ok(destinations);
    }
    if let Some((expected_mode, _)) = match_variant_parts(op_name) {
        let ptx_type = decoded.modifier("type")?;
        if !(ptx_type == "b32" || ptx_type == "b64") {
            return unsupported(format!(
                "{op_name} has unsupported comparison type {:?}",
                ptx_type
            ));
        }
        if decoded.modifier("sync")? != "sync" {
            return unsupported(format!("{op_name} requires sync"));
        }
        if decoded.modifier("mode")? != expected_mode {
            return unsupported(format!("{op_name} requires mode={:?}", expected_mode));
        }
        return Ok(destinations);
    }
    if op_name == "tirx.ptx.redux_sync_bitwise" {
        if decoded.modifier("type")? != "b32"
            || !matches!(decoded.modifier("op")?, "and" | "or" | "xor")
        {
            return unsupported(format!("{op_name} requires and/or/xor.b32"));
        }
        return Ok(destinations);
    }
    if op_name == "tirx.ptx.redux_sync" || op_name == "tirx.ptx.redux_sync_f32" {
        let ptx_type = decoded.modifier("type")?;
        if !matches!(ptx_type, "f32" | "s32" | "u32") {
            return unsupported(format!(
                "{op_name} has unsupported reduction type {:?}",
                ptx_type
            ));
        }
        if op_name == "tirx.ptx.redux_sync_f32" {
            if !matches!(decoded.modifier("op")?, "min" | "max") {
                return unsupported(format!("{op_name} requires min/max.f32"));
            }
            let nan = decoded.modifier("nan")?;
            if !(nan.is_empty() || nan == "NaN") {
                return unsupported(format!("{op_name} has unsupported NaN modifier {:?}", nan));
            }
        } else if !matches!(decoded.modifier("op")?, "add" | "min" | "max") {
            return unsupported(format!("{op_name} requires add/min/max.{ptx_type}"));
        }
        return Ok(destinations);
    }
    if op_name == "tirx.ptx.vote_sync" || op_name == "tirx.ptx.vote_sync_ballot" {
        let mode = decoded.modifier("mode")?;
        let ptx_type = decoded.modifier("type")?;
        if op_name == "tirx.ptx.vote_sync"
            && (!matches!(mode, "all" | "any" | "uni") || ptx_type != "pred")
        {
            return unsupported(format!("{op_name} requires all/any/uni.pred"));
        }
        if op_name == "tirx.ptx.vote_sync_ballot" && (mode != "ballot" || ptx_type != "b32") {
            return unsupported(format!("{op_name} requires ballot.b32"));
        }
        return Ok(destinations);
    }
    if op_name == "tirx.ptx.shfl_sync" || op_name == "tirx.ptx.shfl_sync_p" {
        if decoded.modifier("type")? != "b32" {
            return unsupported(format!("{op_name} requires the b32 instruction type"));
        }
        let mode = decoded.modifier("mode")?;
        if shfl_mode(mode).is_none() {
            return unsupported(format!("{op_name} has unsupported mode {:?}", mode));
        }
    }
    Ok(destinations)
}

/// The validated parts.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = WarpParts {
        destinations: resolve_ptx_warp(decoded)?,
        specialization: warp_specialization(decoded)?,
    };
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_warp(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    fn ptx_warp_u32_operand(&mut self, expression: &ObjectRef, label: &str) -> AResult<RustValue> {
        let value = self.emit_expr(expression)?;
        let value = self.coerce_value(value, "u32", label)?;
        Ok(self.as_warp_value(value))
    }

    fn ptx_warp_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: &[String],
        variant: Option<&str>,
    ) -> String {
        let site = self.v2_site(Some(source_op_id));
        abi::warp_call(function, &site, arguments, variant, None, false, true)
    }

    fn ptx_warp_store_predicate(
        &mut self,
        destination: &ObjectRef,
        predicate: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let value = self.control_name("ptx_warp_predicate_u32");
        self.emit_line(&format!(
            "let {value} = WarpValue::from_fn(|lane| u32::from({predicate}[lane]));"
        ));
        self.emit_explicit_buffer_store(
            destination,
            RustValue::new(value, "u32", Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }

    fn ptx_warp_store_u32_bits(
        &mut self,
        destination: &ObjectRef,
        bits: &str,
        op_name: &str,
        prefix: &str,
        source_op_id: i64,
    ) -> AResult<()> {
        let result = self.emit_from_unsigned_bits(
            RustValue::new(bits, "u32", Uniformity::Varying),
            &dtype_of(destination)?,
            32,
            op_name,
            prefix,
        )?;
        self.emit_explicit_buffer_store(destination, result, source_op_id, None, None, None)
    }

    /// `emit_ptx_warp`.
    pub fn emit_ptx_warp(
        &mut self,
        decoded: &DecodedPtx,
        parts: &WarpParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "ptx_warp",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        let mask = region.mask.clone();
        self.emit_ptx_warp_body(decoded, parts, source_op_id)?;
        self.close_predicated_region(region);
        for slot in &decoded.operands {
            if slot.rw != "w" && slot.rw != "rw" {
                continue;
            }
            for destination in &slot.values {
                let Some(destination) = destination else {
                    return Err(crate::analyze::util::Failure::Ffi(
                        crate::analyze::util::ffi_error("sunk lane in a warp destination"),
                    ));
                };
                self.finish_predicated_destinations(
                    decoded,
                    &[Some(destination.clone())],
                    &dtype_of(destination)?,
                    &mask,
                    source_op_id,
                    slot.operand_type == "pred",
                )?;
            }
        }
        Ok(())
    }

    fn emit_ptx_warp_body(
        &mut self,
        decoded: &DecodedPtx,
        parts: &WarpParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let destinations = &parts.destinations;
        let op_name = decoded.op_name.clone();
        if op_name == "tirx.ptx.activemask" {
            let raw = self.control_name("ptx_activemask_raw");
            let bits = self.control_name("ptx_activemask_bits");
            let call = self.ptx_warp_stateful(ACTIVEMASK, source_op_id, &[], None);
            self.emit_line(&format!("let {raw} = {call};"));
            self.emit_line(&format!("let {bits} = v2_register_out({raw});"));
            return self.ptx_warp_store_u32_bits(
                &destinations[0],
                &bits,
                &op_name,
                "ptx_activemask_result",
                source_op_id,
            );
        }
        if op_name == "tirx.ptx.movmatrix" {
            let source = self.emit_as_unsigned_bits(
                &decoded.scalar_operand("a")?,
                32,
                &op_name,
                "ptx_movmatrix_source_bits",
                None,
            )?;
            let raw = self.control_name("ptx_movmatrix_raw");
            let bits = self.control_name("ptx_movmatrix_bits");
            let call = self.ptx_warp_stateful(
                MOVMATRIX,
                source_op_id,
                &[abi::register(&source.code)],
                Some(MOVMATRIX_B16),
            );
            self.emit_line(&format!("let {raw} = {call};"));
            self.emit_line(&format!("let {bits} = v2_register_out({raw});"));
            return self.ptx_warp_store_u32_bits(
                &destinations[0],
                &bits,
                &op_name,
                "ptx_movmatrix_result",
                source_op_id,
            );
        }

        let membermask = self.ptx_warp_u32_operand(
            &decoded.scalar_operand("membermask")?,
            &format!("{op_name} membermask"),
        )?;

        if match_variant_parts(&op_name).is_some() {
            let specialization = parts.specialization.as_ref().expect("match specialization");
            let source = self.emit_as_unsigned_bits(
                &decoded.scalar_operand("a")?,
                specialization.source_bits,
                &op_name,
                "ptx_match_source_bits",
                None,
            )?;
            let raw = self.control_name("ptx_match_raw");
            let call = self.ptx_warp_stateful(
                MATCH_SYNC,
                source_op_id,
                &[format!(
                    "({}, {})",
                    abi::register(&membermask.code),
                    abi::register(&source.code)
                )],
                Some(&specialization.variant),
            );
            self.emit_line(&format!("let {raw} = {call};"));
            let (raw_mask, raw_predicate) = if specialization.writes_predicate {
                let raw_mask = self.control_name("ptx_match_mask_raw");
                let raw_predicate = self.control_name("ptx_match_predicate_raw");
                self.emit_line(&format!("let ({raw_mask}, {raw_predicate}) = {raw};"));
                (raw_mask, Some(raw_predicate))
            } else {
                (raw, None)
            };
            let mask = self.control_name("ptx_match_mask");
            self.emit_line(&format!("let {mask} = v2_register_out({raw_mask});"));
            self.ptx_warp_store_u32_bits(
                &destinations[0],
                &mask,
                &op_name,
                "ptx_match_result",
                source_op_id,
            )?;
            if let Some(raw_predicate) = raw_predicate {
                let predicate = self.control_name("ptx_match_predicate");
                self.emit_line(&format!(
                    "let {predicate} = v2_register_out({raw_predicate});"
                ));
                self.ptx_warp_store_predicate(&destinations[1], &predicate, source_op_id)?;
            }
            return Ok(());
        }

        if matches!(
            op_name.as_str(),
            "tirx.ptx.redux_sync" | "tirx.ptx.redux_sync_bitwise" | "tirx.ptx.redux_sync_f32"
        ) {
            let source = if op_name == "tirx.ptx.redux_sync_bitwise" {
                self.emit_as_unsigned_bits(
                    &decoded.scalar_operand("a")?,
                    32,
                    &op_name,
                    "ptx_redux_source_bits",
                    None,
                )?
            } else {
                let value = self.emit_expr(&decoded.scalar_operand("a")?)?;
                self.as_warp_value(value)
            };
            let ptx_type = decoded.modifier("type")?;
            let (rust_type, type_variant) = match ptx_type {
                "f32" => ("f32", "F32"),
                "s32" => ("i32", "I32"),
                "u32" => ("u32", "U32"),
                "b32" => ("u32", "B32"),
                _ => unreachable!("validated redux type"),
            };
            if source.rust_type != rust_type {
                return unsupported(format!(
                    "{op_name}.a lowered to {}, expected {rust_type}",
                    source.rust_type
                ));
            }
            let mut source_code = source.code.clone();
            if op_name == "tirx.ptx.redux_sync_f32" && decoded.modifier("abs")? == "abs" {
                source_code = self.control_name("ptx_redux_absolute");
                self.emit_line(&format!(
                    "let {source_code} = WarpValue::from_fn(|lane| {}[lane].abs());",
                    source.code
                ));
            }
            let mut operation = upper_first(decoded.modifier("op")?);
            if op_name == "tirx.ptx.redux_sync_f32" && decoded.modifier("nan")? == "NaN" {
                operation.push_str("Nan");
            }
            let raw = self.control_name("ptx_redux_raw");
            let result = self.control_name("ptx_redux");
            let call = self.ptx_warp_stateful(
                REDUX_SYNC,
                source_op_id,
                &[format!(
                    "({}, {})",
                    abi::register(&membermask.code),
                    abi::register(&source_code)
                )],
                Some(&format!(
                    "v2::warp::variant::Redux<v2::reg::variant::{type_variant}, v2::warp::variant::{operation}>"
                )),
            );
            self.emit_line(&format!("let {raw} = {call};"));
            self.emit_line(&format!("let {result} = v2_register_out({raw});"));
            if op_name == "tirx.ptx.redux_sync_bitwise" {
                self.ptx_warp_store_u32_bits(
                    &destinations[0],
                    &result,
                    &op_name,
                    "ptx_redux_result",
                    source_op_id,
                )?;
            } else {
                self.emit_explicit_buffer_store(
                    &destinations[0],
                    RustValue::new(result, rust_type, Uniformity::Varying),
                    source_op_id,
                    None,
                    None,
                    None,
                )?;
            }
            return Ok(());
        }

        if op_name == "tirx.ptx.vote_sync" || op_name == "tirx.ptx.vote_sync_ballot" {
            let value = self.emit_expr(&decoded.scalar_operand("a")?)?;
            let value = self.coerce_value(value, "bool", "ptx_vote_predicate")?;
            let source = self.as_warp_value(value);
            let mode_variant = match decoded.modifier("mode")? {
                "all" => "All",
                "any" => "Any",
                "uni" => "Uniform",
                "ballot" => "Ballot",
                _ => unreachable!("validated vote mode"),
            };
            let raw = self.control_name("ptx_vote_raw");
            let result = self.control_name("ptx_vote");
            let call = self.ptx_warp_stateful(
                VOTE_SYNC,
                source_op_id,
                &[format!(
                    "({}, {})",
                    abi::register(&membermask.code),
                    abi::register(&source.code)
                )],
                Some(&format!("v2::warp::variant::{mode_variant}")),
            );
            self.emit_line(&format!("let {raw} = {call};"));
            self.emit_line(&format!("let {result} = v2_register_out({raw});"));
            if op_name == "tirx.ptx.vote_sync" {
                self.ptx_warp_store_predicate(&destinations[0], &result, source_op_id)?;
            } else {
                self.ptx_warp_store_u32_bits(
                    &destinations[0],
                    &result,
                    &op_name,
                    "ptx_vote_ballot_result",
                    source_op_id,
                )?;
            }
            return Ok(());
        }

        if op_name == "tirx.ptx.shfl_sync" || op_name == "tirx.ptx.shfl_sync_p" {
            let specialization = parts.specialization.as_ref().expect("shfl specialization");
            let selector =
                self.ptx_warp_u32_operand(&decoded.scalar_operand("b")?, &format!("{op_name}.b"))?;
            let control =
                self.ptx_warp_u32_operand(&decoded.scalar_operand("c")?, &format!("{op_name}.c"))?;
            let (_, source_mask_function) =
                shfl_mode(decoded.modifier("mode")?).expect("validated shfl mode");
            let source_lane_mask = self.control_name("ptx_shfl_source_lane_mask");
            let source_access_mask = self.control_name("ptx_shfl_source_access_mask");
            let call = abi::call(
                source_mask_function,
                &[
                    abi::context("ctx"),
                    abi::register(&membermask.code),
                    abi::register(&selector.code),
                    abi::register(&control.code),
                ],
                &[],
                true,
                false,
            );
            self.emit_line(&format!("let {source_lane_mask} = {call};"));
            self.emit_line(&format!(
                "let {source_access_mask} = WarpMask::from_bits({source_lane_mask}.bits());"
            ));
            let source = self.emit_as_unsigned_bits(
                &decoded.scalar_operand("a")?,
                specialization.source_bits,
                &op_name,
                "ptx_shfl_source_bits",
                Some(&source_access_mask),
            )?;
            let raw = self.control_name("ptx_shfl_raw");
            let call = self.ptx_warp_stateful(
                SHFL_SYNC,
                source_op_id,
                &[format!(
                    "({}, {}, {}, {})",
                    abi::register(&membermask.code),
                    abi::register(&source.code),
                    abi::register(&selector.code),
                    abi::register(&control.code)
                )],
                Some(&specialization.variant),
            );
            self.emit_line(&format!("let {raw} = {call};"));
            let (raw_bits, raw_predicate) = if specialization.writes_predicate {
                let raw_bits = self.control_name("ptx_shfl_bits_raw");
                let raw_predicate = self.control_name("ptx_shfl_predicate_raw");
                self.emit_line(&format!("let ({raw_bits}, {raw_predicate}) = {raw};"));
                (raw_bits, Some(raw_predicate))
            } else {
                (raw, None)
            };
            let bits = self.control_name("ptx_shfl_bits");
            self.emit_line(&format!("let {bits} = v2_register_out({raw_bits});"));
            self.ptx_warp_store_u32_bits(
                &destinations[0],
                &bits,
                &op_name,
                "ptx_shfl_result",
                source_op_id,
            )?;
            if let Some(raw_predicate) = raw_predicate {
                let predicate = self.control_name("ptx_shfl_predicate");
                self.emit_line(&format!(
                    "let {predicate} = v2_register_out({raw_predicate});"
                ));
                self.ptx_warp_store_predicate(&destinations[1], &predicate, source_op_id)?;
            }
            return Ok(());
        }

        let raw = self.control_name("ptx_elect_raw");
        let raw_lane = self.control_name("ptx_elect_lane_raw");
        let raw_predicate = self.control_name("ptx_elect_predicate_raw");
        let lane = self.control_name("ptx_elect_lane");
        let predicate = self.control_name("ptx_elect_predicate");
        let call = self.ptx_warp_stateful(
            ELECT_SYNC,
            source_op_id,
            &[abi::register(&membermask.code)],
            None,
        );
        self.emit_line(&format!("let {raw} = {call};"));
        self.emit_line(&format!("let ({raw_lane}, {raw_predicate}) = {raw};"));
        self.emit_line(&format!("let {lane} = v2_register_out({raw_lane});"));
        self.emit_line(&format!(
            "let {predicate} = v2_register_out({raw_predicate});"
        ));
        self.ptx_warp_store_u32_bits(
            &destinations[0],
            &lane,
            &op_name,
            "ptx_elect_result",
            source_op_id,
        )?;
        self.ptx_warp_store_predicate(&destinations[1], &predicate, source_op_id)
    }
}
