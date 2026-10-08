//! Validation and emission of the clc instruction family.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::register_call::require_register_call;
use crate::emit::{Emitter, RustValue, Uniformity};
use tvm::tvm_ffi::object::ObjectRef;

pub struct ClcQueryParts {
    /// Register destinations in operand order; `None` is a sunk lane.
    pub destinations: Vec<Option<ObjectRef>>,
    pub response: ObjectRef,
    pub predicate: Option<ObjectRef>,
}

/// The table-validated register operands of one CLC query.
pub fn clc_query_parts(decoded: &DecodedPtx) -> AResult<ClcQueryParts> {
    let op_name = decoded.op_name.as_str();
    require_register_call(decoded, true)?;
    let destinations: Vec<Option<ObjectRef>> =
        if op_name == "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled" {
            vec![Some(decoded.scalar_operand("p")?)]
        } else {
            let destinations = decoded.operand("d")?.to_vec();
            if op_name == "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid_v4"
                && destinations.get(3).map_or(true, Option::is_some)
            {
                return unsupported(format!(
                    "{op_name} requires v4 with its unspecified fourth result sunk"
                ));
            }
            destinations
        };
    let response = decoded.scalar_operand("response")?;
    Ok(ClcQueryParts {
        destinations,
        response,
        predicate: decoded.predicate.clone(),
    })
}

/// The validated parts.

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = clc_query_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_clc_query(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    /// `emit_ptx_clc_query`.
    pub fn emit_ptx_clc_query(
        &mut self,
        decoded: &DecodedPtx,
        parts: &ClcQueryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            parts.predicate.as_ref(),
            "clc_query",
            "CLC query predicate must lower to bool or integer",
        )?;
        let mask = region.mask.clone();
        let body = self.emit_clc_query_body(decoded, parts, source_op_id);
        self.close_predicated_region(region);
        body?;
        if decoded.op_name == "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled" {
            let destination = parts.destinations[0]
                .as_ref()
                .expect("is_canceled destination");
            let dtype = dtype_of(destination)?;
            self.finish_predicated_destinations(
                decoded,
                &parts.destinations,
                &dtype,
                &mask,
                source_op_id,
                true,
            )?;
        }
        // get_first_ctaid destinations are intrinsically read-write in TVM's
        // table: inactive lanes retain their bits even without preserve_dst.
        Ok(())
    }

    fn emit_clc_query_body(
        &mut self,
        decoded: &DecodedPtx,
        parts: &ClcQueryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let response =
            self.emit_as_unsigned_bits(&parts.response, 128, op_name, "clc_response", None)?;
        let first_ctaid = self.control_name("clc_first_ctaid");
        self.emit_line(&format!(
            "let {first_ctaid} = WarpValue::from_fn(|lane| ({}[lane][0] & 0xffff_ffff_u64) as u32);",
            response.code
        ));
        if op_name == "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled" {
            let result = self.control_name("clc_is_canceled");
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| u32::from({first_ctaid}[lane] != u32::MAX));"
            ));
            return self.emit_explicit_buffer_store(
                parts.destinations[0]
                    .as_ref()
                    .expect("is_canceled destination"),
                RustValue::new(result, "u32", Uniformity::Varying),
                source_op_id,
                None,
                None,
                None,
            );
        }
        for (axis, destination) in parts.destinations.iter().enumerate() {
            let Some(destination) = destination else {
                continue;
            };
            let mut coordinate = first_ctaid.clone();
            let is_x = if op_name == "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid_v4"
            {
                axis == 0
            } else {
                decoded.modifier("query")? == "get_first_ctaid::x"
            };
            if !is_x {
                // NumSim linearizes the launch CTA domain.  The CLC payload stores
                // that linear base CTA id, so x is the represented coordinate and
                // y/z are zero in the current one-dimensional launch model.
                coordinate = self.control_name("clc_zero_coordinate");
                self.emit_line(&format!("let {coordinate} = WarpValue::splat(0_u32);"));
            }
            let value = self.emit_from_unsigned_bits(
                RustValue::new(coordinate, "u32", Uniformity::Varying),
                &dtype_of(destination)?,
                32,
                op_name,
                "clc_coordinate",
            )?;
            self.emit_explicit_buffer_store(destination, value, source_op_id, None, None, None)?;
        }
        Ok(())
    }
}
