//! Validation and emission of the ptx_address instruction family.

use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::memory_support::v2_memory_space_rust;
use crate::emit::{abi, Emitter, RustValue, Uniformity};
use crate::tables::is_integer_dtype;
use tvm::ir::TensorLoadObj;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

/// The engine address-conversion function (below `v2::`) the `cvta` lowerings call.
pub const CVTA: &str = "addr::cvta";

pub const ADDRESS_QUERIES: [&str; 3] = [
    "tirx.ptx.isspacep",
    "tirx.ptx.getctarank",
    "tirx.ptx.getctarank_generic",
];

pub fn is_address_query(name: &str) -> bool {
    ADDRESS_QUERIES.contains(&name)
}

fn is_tensor_load(value: &ObjectRef) -> bool {
    value.as_node::<TensorLoadObj>().is_some()
}

pub struct MapaParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub rank: ObjectRef,
    pub space: String,
    pub address_bits: i64,
}

pub fn mapa_parts(decoded: &DecodedPtx) -> AResult<MapaParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let space = decoded.modifier_or_empty("space").to_owned();
    let address_bits: i64 = if op_name == "tirx.ptx.mapa_u32" {
        32
    } else {
        64
    };
    let expected_type = format!("u{address_bits}");
    let space_allowed = space == "shared::cluster" || (address_bits == 64 && space.is_empty());
    if !space_allowed || decoded.modifier("type")? != expected_type {
        let allowed = if address_bits == 32 {
            "['shared::cluster']"
        } else {
            "['', 'shared::cluster']"
        };
        return unsupported(format!(
            "{op_name} requires space in {allowed} and type {expected_type}"
        ));
    }
    let destination = decoded.scalar_operand("d")?;
    let source = decoded.scalar_operand("a")?;
    let rank = decoded.scalar_operand("b")?;
    let destination_dtype = format!("uint{address_bits}");
    if !is_tensor_load(&destination) || dtype_of(&destination)? != destination_dtype {
        return unsupported(format!(
            "{op_name}.d must be a {destination_dtype} TensorLoad lvalue"
        ));
    }
    let expected_source = if address_bits == 32 {
        "uint32"
    } else if op_name == "tirx.ptx.mapa" {
        "handle"
    } else {
        "uint64"
    };
    if dtype_of(&source)? != expected_source {
        return unsupported(format!("{op_name}.a must have {expected_source} type"));
    }
    if !is_integer_dtype(&dtype_of(&rank)?) {
        return unsupported(format!("{op_name}.b must be an integer CTA rank"));
    }
    Ok(MapaParts {
        destination,
        source,
        rank,
        space,
        address_bits,
    })
}

pub struct CvtaParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub space: String,
}

pub fn cvta_parts(decoded: &DecodedPtx) -> AResult<CvtaParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let dir = decoded.modifier("dir")?;
    let space = decoded.modifier("space")?;
    let ptx_type = decoded.modifier("type")?;
    if dir != "to"
        || ptx_type != "u64"
        || !["shared", "shared::cta", "shared::cluster", "global"].contains(&space)
    {
        return unsupported(format!(
            "{op_name} requires a modeled cvta.to space and u64, got {{'dir': {:?}, 'space': {:?}, 'type': {:?}}}",
            dir,
            space,
            ptx_type));
    }
    let destination = decoded.scalar_operand("d")?;
    let source = decoded.scalar_operand("ptr")?;
    if !is_tensor_load(&destination) || dtype_of(&destination)? != "uint64" {
        return unsupported(format!("{op_name}.d must be a uint64 TensorLoad lvalue"));
    }
    if dtype_of(&source)? != "handle" {
        return unsupported(format!("{op_name}.ptr must have handle type"));
    }
    Ok(CvtaParts {
        destination,
        source,
        space: space.to_owned(),
    })
}

pub struct QueryParts {
    pub destination: ObjectRef,
    pub source: ObjectRef,
    pub space: String,
}

pub fn query_parts(decoded: &DecodedPtx) -> AResult<QueryParts> {
    let op_name = decoded.op_name.as_str();
    if !decoded.result_type.is_empty() {
        return unsupported(format!("{op_name} requires a void call"));
    }
    let space = decoded.modifier_or_empty("space").to_owned();
    let destination = decoded.scalar_operand(if op_name == "tirx.ptx.isspacep" {
        "p"
    } else {
        "d"
    })?;
    if !is_tensor_load(&destination) || dtype_of(&destination)? != "uint32" {
        return unsupported(format!("{op_name} destination must be a uint32 TensorLoad"));
    }
    if op_name == "tirx.ptx.isspacep"
        && !matches!(
            space.as_str(),
            "global" | "shared" | "shared::cta" | "shared::cluster" | "local"
        )
    {
        return unsupported(format!("{op_name} does not model {space} address space"));
    }
    Ok(QueryParts {
        destination,
        source: decoded.scalar_operand("a")?,
        space,
    })
}

pub fn cvta_generic_parts(decoded: &DecodedPtx) -> AResult<QueryParts> {
    let op_name = decoded.op_name.as_str();
    if !decoded.result_type.is_empty() {
        return unsupported(format!("{op_name} requires a void call"));
    }
    let space = decoded.modifier("space")?.to_owned();
    if !["global", "shared", "shared::cta", "shared::cluster"].contains(&space.as_str()) {
        return unsupported(format!("{op_name} does not model {space} address space"));
    }
    let destination = decoded.scalar_operand("d")?;
    if !is_tensor_load(&destination) || dtype_of(&destination)? != "uint64" {
        return unsupported(format!("{op_name}.d must be a uint64 TensorLoad"));
    }
    Ok(QueryParts {
        destination,
        source: decoded.scalar_operand("a")?,
        space,
    })
}

/// The validated parts of one address instruction.
pub enum AddressParts {
    CvtaGeneric(QueryParts),
    Query(QueryParts),
    Cvta(CvtaParts),
    Mapa(MapaParts),
}

/// The validated parts.
fn address_parts(decoded: &DecodedPtx) -> AResult<AddressParts> {
    let op_name = decoded.op_name.as_str();
    let parts = if op_name == "tirx.ptx.cvta_generic" {
        AddressParts::CvtaGeneric(cvta_generic_parts(decoded)?)
    } else if is_address_query(op_name) {
        AddressParts::Query(query_parts(decoded)?)
    } else if op_name == "tirx.ptx.cvta" {
        AddressParts::Cvta(cvta_parts(decoded)?)
    } else {
        AddressParts::Mapa(mapa_parts(decoded)?)
    };
    Ok(parts)
}

/// Validate and emit one instruction through the registry callback.
pub fn emit(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = address_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_address(decoded, &parts, source_op_id)?;
    Ok(None)
}

impl<'a> Emitter<'a> {
    fn emit_ptx_address_query(
        &mut self,
        decoded: &DecodedPtx,
        parts: &QueryParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.as_str();
        let value = self.emit_expr(&parts.source)?;
        let result = self.control_name("address_query");
        if op_name == "tirx.ptx.isspacep" {
            let pointer = if value.rust_type == "u32" {
                self.emit_raw_shared_pointer(&parts.source, Some(value), "ctx.active_mask()")?
            } else {
                self.emit_raw_generic_pointer(value, "ctx.active_mask()")?
            };
            let invocation = abi::warp_call(
                "addr::isspacep",
                &abi::site(source_op_id as u64),
                &[abi::address(
                    "v2::Generic",
                    &abi::cloned(&pointer.code),
                    None,
                )],
                Some(v2_memory_space_rust(&parts.space)?),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {result} = v2_register_out({invocation});"));
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| u32::from({result}[lane]));"
            ));
        } else {
            let shared = if value.rust_type == "PhysicalPtr" {
                let shared = self.control_name("getctarank_shared_address");
                self.emit_line(&format!(
                    "let {shared} = ({}).shared_byte_addresses_u32(&ctx, ctx.active_mask())?;",
                    value.code
                ));
                shared
            } else {
                let value = self.as_warp_value(value);
                if value.rust_type == "u32" {
                    value.code.clone()
                } else if value.rust_type == "u64" {
                    let shared = self.control_name("getctarank_shared_address");
                    if op_name == "tirx.ptx.getctarank_generic" {
                        self.emit_line(&format!(
                            "if ctx.active_mask().into_iter().any(|lane| decode_generic_shared_address({}[lane]).is_none()) {{ return Err(EngineError::message(\"getctarank requires a generic shared address\")); }}",
                            value.code
                        ));
                        self.emit_line(&format!(
                            "let {shared} = WarpValue::from_fn(|lane| decode_generic_shared_address({}[lane]).unwrap_or(0_u32));",
                            value.code
                        ));
                    } else {
                        self.emit_line(&format!(
                            "if ctx.active_mask().into_iter().any(|lane| {}[lane] > u64::from(u32::MAX)) {{ return Err(EngineError::message(\"getctarank shared address exceeds 32 bits\")); }}",
                            value.code
                        ));
                        self.emit_line(&format!(
                            "let {shared} = WarpValue::from_fn(|lane| {}[lane] as u32);",
                            value.code
                        ));
                    }
                    shared
                } else {
                    return unsupported(format!(
                        "{op_name} source lowered to {}, expected u32 or u64",
                        value.rust_type
                    ));
                }
            };
            self.emit_line(&format!(
                "let {result} = WarpValue::from_fn(|lane| shared_address_cta_rank({shared}[lane]));"
            ));
        }
        self.emit_explicit_buffer_store(
            &parts.destination,
            RustValue::new(result, "u32", Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }

    /// `emit_ptx_address`: gate source evaluation and pointer/scalar writeback
    /// with one lane mask.
    pub fn emit_ptx_address(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AddressParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            decoded.predicate.as_ref(),
            "ptx_address",
            &format!(
                "{} predicate must lower to bool or integer",
                decoded.op_name
            ),
        )?;
        self.emit_ptx_address_body(decoded, parts, source_op_id)?;
        self.close_predicated_region(region);
        Ok(())
    }

    fn emit_ptx_address_body(
        &mut self,
        decoded: &DecodedPtx,
        parts: &AddressParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let conversion = match parts {
            AddressParts::CvtaGeneric(parts) => {
                Some((&parts.destination, &parts.source, &parts.space, true))
            }
            AddressParts::Cvta(parts) => {
                Some((&parts.destination, &parts.source, &parts.space, false))
            }
            _ => None,
        };
        if let Some((destination, source, space, generic)) = conversion {
            let bits = self.emit_expr(source)?;
            let bits = self.observe_pointer_bits(bits);
            let bits = self.as_warp_value(bits);
            let result = self.control_name("cvta_address");
            let invocation = abi::warp_call(
                CVTA,
                &abi::site(source_op_id as u64),
                &[abi::register(&bits.code)],
                Some(&format!(
                    "v2::addr::variant::Convert<{}, {generic}>",
                    v2_memory_space_rust(space)?
                )),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {result} = v2_register_out({invocation});"));
            return self.emit_explicit_buffer_store(
                destination,
                RustValue::new(result, "u64", Uniformity::Varying),
                source_op_id,
                None,
                None,
                None,
            );
        }
        if let AddressParts::Query(parts) = parts {
            return self.emit_ptx_address_query(decoded, parts, source_op_id);
        }

        let AddressParts::Mapa(parts) = parts else {
            unreachable!("query and cvta parts are lowered above");
        };
        let rank = self.emit_expr(&parts.rank)?;
        let rank = self.as_i64(rank)?;
        let rank = self.as_warp_value(rank);
        self.emit_line(&format!(
            "if ctx.active_mask().into_iter().any(|lane| {}[lane] < 0_i64 || usize::try_from({}[lane]).ok().map_or(true, |rank| rank >= ctx.topology().ctas_per_cluster())) {{ return Err(EngineError::message(\"mapa CTA rank is outside the active cluster\")); }}",
            rank.code, rank.code
        ));
        if parts.address_bits == 32 {
            let source = self.emit_expr(&parts.source)?;
            let source = self.as_warp_value(source);
            if source.rust_type != "u32" {
                return unsupported(format!(
                    "{op_name} source lowered to {}, expected u32",
                    source.rust_type
                ));
            }
            let mapped = self.control_name("mapped_shared_u32_address");
            self.emit_line(&format!(
                "let {mapped} = WarpValue::from_fn(|lane| replace_shared_address_cta_rank({}[lane], {}[lane] as u32).unwrap());",
                source.code, rank.code
            ));
            return self.emit_explicit_buffer_store(
                &parts.destination,
                RustValue::new(mapped, "u32", Uniformity::Varying),
                source_op_id,
                None,
                None,
                None,
            );
        }

        let source = self.emit_expr(&parts.source)?;
        let source = if source.rust_type == "PhysicalPtr" {
            let source_bits = self.control_name("mapa_generic_source");
            self.emit_line(&format!(
                "let {source_bits} = ({}).generic_addresses_u64(&ctx, ctx.active_mask())?;",
                source.code
            ));
            RustValue::new(source_bits, "u64", Uniformity::Varying)
        } else {
            self.as_warp_value(source)
        };
        if source.rust_type != "u64" {
            return unsupported(format!(
                "{op_name} source lowered to {}, expected u64",
                source.rust_type
            ));
        }
        let mapped = self.control_name("mapped_shared_u64_address");
        if parts.space == "shared::cluster" {
            let shared_source = self.control_name("mapa_shared_u64_source");
            self.emit_line(&format!(
                "if ctx.active_mask().into_iter().any(|lane| decode_generic_shared_address({}[lane]).is_none() && {}[lane] > u64::from(u32::MAX)) {{ return Err(EngineError::message(\"mapa.shared::cluster.u64 source is not a shared address\")); }}",
                source.code, source.code
            ));
            self.emit_line(&format!(
                "let {shared_source} = WarpValue::from_fn(|lane| decode_generic_shared_address({}[lane]).unwrap_or({}[lane] as u32));",
                source.code, source.code
            ));
            self.emit_line(&format!(
                "let {mapped} = WarpValue::from_fn(|lane| u64::from(replace_shared_address_cta_rank({shared_source}[lane], {}[lane] as u32).unwrap()));",
                rank.code
            ));
        } else {
            self.emit_line(&format!(
                "if ctx.active_mask().into_iter().any(|lane| replace_generic_shared_address_cta_rank({}[lane], {}[lane] as u32).is_none()) {{ return Err(EngineError::message(\"mapa.u64 requires a generic shared address\")); }}",
                source.code, rank.code
            ));
            self.emit_line(&format!(
                "let {mapped} = WarpValue::from_fn(|lane| replace_generic_shared_address_cta_rank({}[lane], {}[lane] as u32).unwrap_or(0_u64));",
                source.code, rank.code
            ));
        }
        self.emit_explicit_buffer_store(
            &parts.destination,
            RustValue::new(mapped, "u64", Uniformity::Varying),
            source_op_id,
            None,
            None,
            None,
        )
    }
}
