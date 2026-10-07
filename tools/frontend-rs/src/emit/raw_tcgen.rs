//! Validation and emission of the raw_tcgen instruction family.

use crate::analyze::memory::MemorySpace;
use crate::analyze::util::{
    as_buffer, dtype_of, ffi_error, not_covered, oref, prim, repr_text, unmodeled, unsupported,
    upper_first, AResult, Failure,
};
use crate::analyze::Ctx;
use crate::decode::ptx::DecodedPtx;
use crate::decode::Decoded;
use crate::emit::abi::named_buffer;
use crate::emit::cuda_helper::MMA_MXF4_BLOCK32_SS_VARIANT;
use crate::emit::ptx_address::CVTA;
use crate::emit::pure::reinterpret_atom;
use crate::emit::tcgen_descriptor::{parse_tcgen_descriptor_call, TcgenDescriptorCall};
use crate::emit::tcgen_descriptor::{
    require_tcgen_descriptor_layout, tcgen_descriptor_variant, TCGEN_DESCRIPTOR_LAYOUT,
};
use crate::emit::{abi, Emitter, RustValue};
use crate::tables::MEM_ST;
use crate::tvm_compat::int_value;
use tvm::ir::{IntImmObj, PrimExpr, TensorLoadObj};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

#[derive(Clone, Default)]
pub struct State {
    /// `(storage indices, field)` in insertion order.
    pub shared_descriptor_domains: Vec<(Vec<usize>, String)>,
}

/// The engine functions (below `v2::`) the raw TCGEN lowerings call.
pub const CP: &str = "tcgen05::cp";
pub const MMA: &str = "tcgen05::mma";

fn ldst_shape(shape: &str) -> Option<(&'static str, i64)> {
    Some(match shape {
        "16x32bx2" => ("Shape16x32bx2", 1),
        "16x64b" => ("Shape16x64b", 1),
        "16x128b" => ("Shape16x128b", 2),
        "32x32b" => ("Shape32x32b", 1),
        "16x256b" => ("Shape16x256b", 4),
        _ => return None,
    })
}

pub fn cp_shape_variant(code: usize) -> &'static str {
    [
        "Cp32x128bWarpx4",
        "Cp64x128bWarpx2_02_13",
        "Cp128x128b",
        "Cp128x256b",
        "Cp4x256b",
        "Cp64x128bWarpx2_01_23",
    ][code]
}

pub fn cp_decompress_variant(code: usize) -> &'static str {
    ["NoDecompress", "DecompressB4", "DecompressB6"][code]
}

fn cp_form_code(shape: &str, multicast: &str) -> Option<usize> {
    Some(match (shape, multicast) {
        ("32x128b", "warpx4") => 0,
        ("64x128b", "warpx2::02_13") => 1,
        ("128x128b", "") => 2,
        ("128x256b", "") => 3,
        ("4x256b", "") => 4,
        ("64x128b", "warpx2::01_23") => 5,
        _ => return None,
    })
}

fn cp_decompress_code(dst_fmt: &str, src_fmt: &str) -> Option<usize> {
    Some(match (dst_fmt, src_fmt) {
        ("", "") => 0,
        ("b8x16", "b4x16_p64") => 1,
        ("b8x16", "b6x16_p32") => 2,
        _ => return None,
    })
}

/// The `UnmodeledTIRxFormError` target of one call.
pub fn unmodeled_target(op_name: &str) -> String {
    format!("call:{op_name}")
}

/// `allowed` is listed sorted.
fn require_operand_dtype(
    decoded: &DecodedPtx,
    name: &str,
    value: &ObjectRef,
    allowed: &[&str],
) -> AResult<()> {
    let actual = dtype_of(value)?;
    if !allowed.contains(&actual.as_str()) {
        return unsupported(format!(
            "{}.{name} must have dtype in {:?}, got {actual}",
            decoded.op_name, &allowed
        ));
    }
    Ok(())
}

/// A sunk lane has no dtype: an internal error.
fn sink_lane(decoded: &DecodedPtx, name: &str) -> Failure {
    Failure::Ffi(ffi_error(&format!(
        "{}.{name} sunk lane has no dtype",
        decoded.op_name
    )))
}

/// The optional scalar operand of one named slot.
fn optional_operand(decoded: &DecodedPtx, name: &str) -> AResult<Option<ObjectRef>> {
    if decoded.has_operand(name) {
        Ok(Some(decoded.scalar_operand(name)?))
    } else {
        Ok(None)
    }
}

// ----------------------------------------------------------------------
// tcgen05.ld / tcgen05.st
// ----------------------------------------------------------------------

pub fn ldst_reduction(decoded: &DecodedPtx) -> AResult<Option<String>> {
    let op_name = decoded.op_name.as_str();
    if op_name == "tirx.ptx.tcgen05_ld_spcompress"
        || op_name == "tirx.ptx.tcgen05_ld_red_spcompress"
    {
        let maximum = decoded.modifier("rowop")? == "max";
        let absolute = decoded.modifier("abs")? == "abs";
        let mut reduction = "NoReduction".to_owned();
        if op_name == "tirx.ptx.tcgen05_ld_red_spcompress" {
            let nan = decoded.modifier("nan")? == "NaN";
            reduction = format!("ReduceF32<{maximum}, {absolute}, {nan}>");
        }
        return Ok(Some(format!(
            "Compress<{maximum}, {absolute}, v2::tcgen05::variant::{reduction}>"
        )));
    }
    if op_name != "tirx.ptx.tcgen05_ld_red" && op_name != "tirx.ptx.tcgen05_ld_red_split" {
        return Ok(None);
    }
    let dtype = decoded.modifier("type")?;
    let redop = decoded.modifier("redop")?;
    let abs = decoded.modifier("abs")?;
    let nan = decoded.modifier("nan")?;
    let maximum = redop == "max";
    let absolute = abs == "abs";
    let is_nan = nan == "NaN";
    if decoded.modifier("red")? != "red"
        || !(redop == "min" || redop == "max")
        || !matches!(dtype, "f32" | "u32" | "s32")
        || !(abs.is_empty() || abs == "abs")
        || !(nan.is_empty() || nan == "NaN")
        || (dtype != "f32" && (absolute || is_nan))
    {
        return unsupported(format!("{op_name} has unsupported reduction modifiers"));
    }
    let mut args = vec![maximum.to_string()];
    if dtype == "f32" {
        args.push(absolute.to_string());
        args.push(is_nan.to_string());
    }
    let marker = match dtype {
        "f32" => "F32",
        "u32" => "U32",
        _ => "I32",
    };
    Ok(Some(format!("Reduce{marker}<{}>", args.join(", "))))
}

pub struct LdstParts {
    pub instruction: &'static str,
    pub shape_variant: String,
    pub num: i64,
    pub packed: bool,
    pub address: ObjectRef,
    /// Load destinations (`TensorLoad`s) or store input values.
    pub registers: Vec<ObjectRef>,
    pub reduction: Option<String>,
}

pub fn decoded_ldst_parts(ctx: &Ctx, decoded: &DecodedPtx) -> AResult<LdstParts> {
    let op_name = decoded.op_name.as_str();
    decoded.require_void()?;
    let instruction = if matches!(op_name, "tirx.ptx.tcgen05_st" | "tirx.ptx.tcgen05_st_split") {
        "st"
    } else {
        "ld"
    };
    let compressed = op_name == "tirx.ptx.tcgen05_ld_spcompress"
        || op_name == "tirx.ptx.tcgen05_ld_red_spcompress";
    let reduction = ldst_reduction(decoded)?;
    let expected_type = if reduction.is_some() {
        decoded.modifier("type")?.to_owned()
    } else {
        "b32".to_owned()
    };
    let expected: [(&str, &str); 4] = [
        ("action", instruction),
        ("sync", "sync"),
        ("aligned", "aligned"),
        ("type", expected_type.as_str()),
    ];
    for (name, required) in expected {
        let actual = decoded.modifier(name)?;
        if actual != required {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                required, actual
            ));
        }
    }
    let shape = decoded.modifier("shape")?;
    let split = matches!(
        op_name,
        "tirx.ptx.tcgen05_ld_split" | "tirx.ptx.tcgen05_ld_red_split" | "tirx.ptx.tcgen05_st_split"
    );
    if split != (shape == "16x32bx2") {
        let expected_shape = if split {
            "16x32bx2"
        } else {
            "one of 16x64b/16x128b/16x256b/32x32b"
        };
        return unsupported(format!(
            "{op_name} requires shape {expected_shape}, got {:?}",
            shape
        ));
    }
    let Some((shape_marker, registers_per_num)) = ldst_shape(shape) else {
        return unsupported(format!("{op_name} has unsupported shape {:?}", shape));
    };
    let num_token = decoded.modifier("num")?;
    let num_digits = num_token.strip_prefix('x');
    if num_token.len() < 2
        || num_digits.map_or(true, |digits| {
            digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit())
        })
    {
        return unsupported(format!(
            "{op_name} has malformed num modifier {:?}",
            num_token
        ));
    }
    let num: i64 = num_digits
        .expect("validated num")
        .parse()
        .map_err(|_| Failure::Ffi(ffi_error("num digits overflow")))?;
    if reduction.is_some() && (num < 2 || !(shape == "32x32b" || shape == "16x32bx2")) {
        return unsupported(format!(
            "{op_name} requires x2 or larger and a reduction shape"
        ));
    }
    let qualifier = if reduction.is_some() {
        ""
    } else {
        decoded.modifier(if instruction == "ld" {
            "pack"
        } else {
            "unpack"
        })?
    };
    let packed = !qualifier.is_empty();

    let address = decoded.scalar_operand("taddr")?;
    let address_dtype = dtype_of(&address)?;
    if address_dtype != "int32" && address_dtype != "uint32" {
        return unsupported(format!(
            "{op_name}.taddr must be int32 or uint32, got {:?}",
            &address_dtype
        ));
    }
    let mut registers: Vec<Option<ObjectRef>> = if compressed {
        let mut lanes = decoded.operand("mdata")?.to_vec();
        lanes.extend(decoded.operand("cdata")?.iter().cloned());
        lanes
    } else {
        decoded.operand("r")?.to_vec()
    };
    let expected_registers = if compressed {
        (num + 31) / 32 + num / 2
    } else {
        registers_per_num * num
    };
    if registers.len() as i64 != expected_registers {
        return unsupported(format!(
            "{op_name} shape={shape} num={num} expects {expected_registers} registers, got {}",
            registers.len()
        ));
    }
    if reduction.is_some() && op_name != "tirx.ptx.tcgen05_ld_spcompress" {
        registers.push(Some(decoded.scalar_operand("redval")?));
    }
    let role = if instruction == "ld" {
        "TensorLoad destination"
    } else {
        "input value"
    };
    let mut validated: Vec<ObjectRef> = Vec::new();
    for (index, register) in registers.iter().enumerate() {
        let Some(register) = register else {
            if instruction != "ld" {
                return Err(sink_lane(decoded, &format!("r[{index}]")));
            }
            return unsupported(format!(
                "{op_name}.r[{index}] must be a 32-bit {role}, got PTX_SINK"
            ));
        };
        if (instruction == "ld" && register.as_node::<TensorLoadObj>().is_none())
            || !matches!(dtype_of(register)?.as_str(), "int32" | "uint32" | "float32")
        {
            return unsupported(format!(
                "{op_name}.r[{index}] must be a 32-bit {role}, got {}",
                repr_text(register)?
            ));
        }
        validated.push(register.clone());
    }
    let mut shape_variant = shape_marker.to_owned();
    if split {
        let half_splitoff = crate::analyze::util::static_int(
            &ctx.analyzer,
            &prim(&decoded.scalar_operand("imm_half_splitoff")?)?,
            &format!("{op_name}.imm_half_splitoff"),
            "must be a compile-time integer",
        )?;
        if !(0..(1 << 16)).contains(&half_splitoff) {
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} requires a 16-bit nonnegative column split offset, got {half_splitoff}"
                ),
            );
        }
        shape_variant = format!("{shape_variant}<{half_splitoff}>");
    }
    Ok(LdstParts {
        instruction,
        shape_variant,
        num,
        packed,
        address,
        registers: validated,
        reduction,
    })
}

/// The `emit_ldst_operands` variant spelling.
pub fn ldst_variant(parts: &LdstParts, access: &str) -> String {
    let reduction = match &parts.reduction {
        Some(reduction) => format!(", v2::tcgen05::variant::{reduction}"),
        None => String::new(),
    };
    format!(
        "v2::tcgen05::variant::{}<v2::tcgen05::variant::{}, v2::tcgen05::variant::Num<{}>, {}, {access}{reduction}>",
        upper_first(parts.instruction),
        parts.shape_variant,
        parts.num,
        parts.packed
    )
}

/// The parsed parts.

// ----------------------------------------------------------------------
// tcgen05.cp
// ----------------------------------------------------------------------

pub struct CpParts {
    pub address: ObjectRef,
    pub descriptor: ObjectRef,
    pub shape_code: usize,
    pub decompress_code: usize,
    pub cta_group: i64,
    pub predicate: Option<ObjectRef>,
}

pub fn decoded_tcgen_cp_parts(decoded: &DecodedPtx) -> AResult<CpParts> {
    decoded.require_void()?;
    let op_name = decoded.op_name.as_str();
    let action = decoded.modifier("action")?;
    if action != "cp" {
        return unsupported(format!("{op_name} requires action='cp', got {:?}", action));
    }
    let cta_group = decoded.cta_group(None)?;
    let shape = decoded.modifier("shape")?;
    let multicast = decoded.modifier("multicast")?;
    let Some(shape_code) = cp_form_code(shape, multicast) else {
        return unmodeled(
            unmodeled_target(op_name),
            format!(
                "{op_name} shape={:?} multicast={:?} has no engine ABI",
                shape, multicast
            ),
        );
    };
    let dst_fmt = decoded.modifier("dst_fmt")?;
    let src_fmt = decoded.modifier("src_fmt")?;
    let Some(decompress_code) = cp_decompress_code(dst_fmt, src_fmt) else {
        return unmodeled(
            unmodeled_target(op_name),
            format!(
                "{op_name} decompression format {:?} has no engine ABI",
                &[dst_fmt, src_fmt]
            ),
        );
    };
    let address = decoded.scalar_operand("taddr")?;
    let descriptor = decoded.scalar_operand("s_desc")?;
    require_operand_dtype(decoded, "taddr", &address, &["int32", "uint32"])?;
    require_operand_dtype(decoded, "s_desc", &descriptor, &["uint64"])?;
    if let Some(predicate) = &decoded.predicate {
        require_operand_dtype(
            decoded,
            "predicate",
            predicate,
            crate::dtypes::predicate_dtypes(),
        )?;
    }
    Ok(CpParts {
        address,
        descriptor,
        shape_code,
        decompress_code,
        cta_group,
        predicate: decoded.predicate.clone(),
    })
}

/// The parsed parts.

// ----------------------------------------------------------------------
// tcgen05.mma
// ----------------------------------------------------------------------

const DENSE_NARROW_MARKERS: [(i128, &str); 5] = [
    (0, "E4m3"),
    (1, "E5m2"),
    (3, "E2m3"),
    (4, "E3m2"),
    (5, "E2m1"),
];

fn dense_narrow_marker(format: i128) -> Option<&'static str> {
    DENSE_NARROW_MARKERS
        .iter()
        .find(|(code, _)| *code == format)
        .map(|(_, marker)| *marker)
}

pub fn dense_mma_variant(
    op_name: &str,
    kind: &str,
    mode: &str,
    cta_group: i64,
    predicated: bool,
    descriptor: i128,
) -> AResult<String> {
    let mut suffix = if predicated { "Pred" } else { "" };
    let weight_stationary = op_name.contains("_ws_");
    let sparse = op_name.contains("_sp_");
    let ws_suffix = if weight_stationary { "Ws" } else { "" };
    // Dense narrow forms exist on SM100 with K=32 and on SM107 with K=32/K=64.
    // Architecture is a PrimFunc-level fact and is specialized during artifact
    // emission, so leave bit 29 available here.
    let f8_cta2 = kind == "kind::f8f6f4" && cta_group == 2;
    let mut reserved_mask: i128 = if weight_stationary {
        0x2080004F
    } else {
        0xE080004F
    };
    if kind == "kind::f8f6f4" {
        reserved_mask &= !(1 << 29);
    }
    if sparse {
        reserved_mask &= !7;
    }
    if kind == "kind::i8" {
        reserved_mask = (reserved_mask & !8) | (1 << 13) | (1 << 14);
    }
    if descriptor & reserved_mask != 0 {
        return unsupported(format!(
            "{op_name} instruction descriptor {} sets unsupported dense reserved/sparse/saturate fields",
            hex8(descriptor)
        ));
    }
    let d_format = (descriptor >> 4) & 0x3;
    let a_format = (descriptor >> 7) & 0x7;
    let b_format = (descriptor >> 10) & 0x7;
    let mode_marker = upper_first(mode);
    let e4m3_or_e5m2 = |format: i128| format == 0 || format == 1;
    if kind == "kind::f8f6f4"
        && !sparse
        && descriptor & (1 << 29) != 0
        && (!e4m3_or_e5m2(a_format) || !e4m3_or_e5m2(b_format))
    {
        return unsupported(format!("{op_name} dense K=64 requires E4M3/E5M2 operands"));
    }

    if kind == "kind::ti16" || kind == "kind::i8" {
        if kind == "kind::ti16" && (d_format, a_format, b_format) != (2, 3, 3) {
            return unsupported(format!("{op_name} must encode S32/S1Z4M11/S1Z4M11"));
        }
        if kind == "kind::i8"
            && (d_format != 2
                || !e4m3_or_e5m2(a_format)
                || !e4m3_or_e5m2(b_format)
                || (sparse && descriptor & 3 != 0))
        {
            return unsupported(format!(
                "{op_name} must encode S32 with U8/S8 operands and sparsity selector zero"
            ));
        }
        if weight_stationary {
            suffix = "Ws";
        } else if sparse {
            suffix = "";
        }
        if (descriptor & 4 != 0) != sparse {
            return unsupported(format!(
                "{op_name} descriptor sparsity does not match the instruction"
            ));
        }
        let integer_kind = if kind == "kind::ti16" { "Ti16" } else { "I8" };
        let family = if sparse { "Sparse" } else { "Integer" };
        return Ok(format!(
            "v2::tcgen05::variant::Mma{family}{mode_marker}Cta{cta_group}{suffix}<v2::tcgen05::variant::{integer_kind}, <artifact-tmem-mode>>"
        ));
    }

    if kind == "kind::f16" {
        if !(0..=1).contains(&d_format)
            || !(0..=1).contains(&a_format)
            || !(0..=1).contains(&b_format)
        {
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} descriptor {} must encode F16/F32 with F16/BF16 operands",
                    hex8(descriptor)
                ),
            );
        }
        if d_format == 0 && (a_format != 0 || b_format != 0) {
            // PTX's kind::f16 type/shape table permits BF16 only with F32 D.
            return unsupported(format!(
                "{op_name}: F16 accumulation requires F16 operands; BF16 requires F32"
            ));
        }
        let a_marker = if a_format == 1 { "Bf16" } else { "Fp16" };
        let b_marker = if b_format == 1 { "Bf16" } else { "Fp16" };
        if sparse {
            if descriptor & 4 == 0 || descriptor & 2 != 0 {
                return unsupported(format!(
                    "{op_name} descriptor requires sparsity and selector 0 or 1"
                ));
            }
            return Ok(format!(
                "v2::tcgen05::variant::MmaSparse{mode_marker}Cta{cta_group}{ws_suffix}<(v2::tcgen05::variant::{a_marker}, v2::tcgen05::variant::{b_marker}), <artifact-tmem-mode>>"
            ));
        }
        if weight_stationary {
            if cta_group != 1 {
                return unmodeled(
                    unmodeled_target(op_name),
                    format!("{op_name} weight-stationary execution requires cta_group=1"),
                );
            }
            suffix = "Ws";
        }
        return Ok(format!(
            "v2::tcgen05::variant::MmaF16{mode_marker}Cta{cta_group}{suffix}<v2::tcgen05::variant::{a_marker}, v2::tcgen05::variant::{b_marker}, v2::tcgen05::variant::Scale<0>, <artifact-tmem-mode>>"
        ));
    }

    if weight_stationary && kind != "kind::tf32" && kind != "kind::f8f6f4" {
        return unmodeled(
            unmodeled_target(op_name),
            format!(
                "{op_name} weight-stationary {kind} semantics have no exact NumSim engine variant"
            ),
        );
    }

    if kind == "kind::f8f6f4" && sparse {
        if descriptor & 15 != 4 {
            return unsupported(format!("{op_name} sparse F8F6F4 requires selector zero"));
        }
        let (Some(a_marker), Some(b_marker)) =
            (dense_narrow_marker(a_format), dense_narrow_marker(b_format))
        else {
            return unmodeled(
                unmodeled_target(op_name),
                "unmodeled sparse narrow operand format",
            );
        };
        if !(0..=1).contains(&d_format) {
            return unmodeled(
                unmodeled_target(op_name),
                "unmodeled sparse narrow operand format",
            );
        }
        return Ok(format!(
            "v2::tcgen05::variant::MmaSparse{mode_marker}Cta{cta_group}{ws_suffix}<(v2::tcgen05::variant::{a_marker}, v2::tcgen05::variant::{b_marker}, {TCGEN_DESCRIPTOR_LAYOUT}), <artifact-tmem-mode>>"
        ));
    }

    if kind == "kind::f8f6f4" && cta_group == 1 {
        let (true, Some(a_marker), Some(b_marker)) = (
            (0..=1).contains(&d_format),
            dense_narrow_marker(a_format),
            dense_narrow_marker(b_format),
        ) else {
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} descriptor {} requires modeled F16/F32 and E4M3/E5M2/E2M3/E3M2/E2M1 formats",
                    hex8(descriptor)
                ),
            );
        };
        let accumulator = if d_format == 0 { "F16" } else { "F32" };
        let suffix = if weight_stationary { "Ws" } else { suffix };
        return Ok(format!(
            "v2::tcgen05::variant::MmaF8f6f4{accumulator}{mode_marker}Cta1{suffix}<v2::tcgen05::variant::{a_marker}, v2::tcgen05::variant::{b_marker}, {TCGEN_DESCRIPTOR_LAYOUT}, <artifact-tmem-mode>>"
        ));
    }

    if f8_cta2 {
        if !(0..=1).contains(&d_format) {
            return unmodeled(
                unmodeled_target(op_name),
                format!("{op_name} cta_group=2 requires an F16 or F32 destination"),
            );
        }
        let (Some(a_marker), Some(b_marker)) =
            (dense_narrow_marker(a_format), dense_narrow_marker(b_format))
        else {
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} descriptor {} requires modeled narrow-float operands",
                    hex8(descriptor)
                ),
            );
        };
        let m = ((descriptor >> 24) & 0x1F) * 16;
        let n = ((descriptor >> 17) & 0x3F) * 8;
        let b_mn_major = descriptor & (1 << 16) != 0;
        let n_granularity: i128 = if b_mn_major { 32 } else { 16 };
        if !(m == 128 || m == 256) || !(n_granularity <= n && n <= 256) || n % n_granularity != 0 {
            let b_major = if b_mn_major { "MN-major" } else { "K-major" };
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} cta_group=2 requires M in {{128, 256}} and N in {n_granularity}..=256 by {n_granularity} for {b_major} B, got M={m}, N={n}"
                ),
            );
        }
        let accumulator = if d_format == 0 { "F16" } else { "F32" };
        return Ok(format!(
            "v2::tcgen05::variant::MmaF8f6f4{accumulator}{mode_marker}Cta2{suffix}<v2::tcgen05::variant::{a_marker}, v2::tcgen05::variant::{b_marker}, {TCGEN_DESCRIPTOR_LAYOUT}, <artifact-tmem-mode>>"
        ));
    }

    if kind == "kind::tf32" {
        if (d_format, a_format, b_format) != (1, 2, 2) {
            return unmodeled(
                unmodeled_target(op_name),
                format!(
                    "{op_name} descriptor {} must encode F32/TF32/TF32",
                    hex8(descriptor)
                ),
            );
        }
        if (descriptor & 4 != 0) != sparse || (sparse && descriptor & 2 != 0) {
            return unsupported(format!(
                "{op_name} descriptor requires matching sparsity and selector 0 or 1"
            ));
        }
        if sparse {
            return Ok(format!(
                "v2::tcgen05::variant::MmaSparse{mode_marker}Cta{cta_group}{ws_suffix}<v2::reg::variant::Tf32, <artifact-tmem-mode>>"
            ));
        }
        let suffix = if weight_stationary { "Ws" } else { suffix };
        return Ok(format!(
            "v2::tcgen05::variant::MmaTf32{mode_marker}Cta{cta_group}{suffix}<v2::tcgen05::variant::Scale<0>, <artifact-tmem-mode>>"
        ));
    }

    unmodeled(
        unmodeled_target(op_name),
        format!("{op_name} {kind} {mode} cta_group={cta_group} has no exact NumSim engine variant"),
    )
}

/// `f"0x{value:08x}"`.
pub fn hex8(value: i128) -> String {
    if value < 0 {
        format!("0x-{:0>7x}", -value)
    } else {
        format!("0x{value:08x}")
    }
}

/// `(descriptor, variant)` in descriptor order.
pub fn dense_mma_candidate_variants(
    op_name: &str,
    kind: &str,
    mode: &str,
    cta_group: i64,
    predicated: bool,
) -> AResult<Vec<(i128, String)>> {
    let sparse_bit: i128 = if op_name.contains("_sp_") { 4 } else { 0 };
    let cta = i128::from(cta_group);
    let mut descriptors: Vec<i128> = Vec::new();
    if kind == "kind::i8" {
        descriptors.push((2 << 4) | sparse_bit);
    } else if kind == "kind::ti16" {
        descriptors.push((2 << 4) | (3 << 7) | (3 << 10) | sparse_bit);
    } else if kind == "kind::f16" {
        for (a_format, b_format) in [(0_i128, 0_i128), (1, 1)] {
            descriptors.push((1 << 4) | (a_format << 7) | (b_format << 10) | sparse_bit);
        }
    } else if kind == "kind::f8f6f4" && op_name.contains("_sp_") {
        for (a_format, _) in DENSE_NARROW_MARKERS {
            for (b_format, _) in DENSE_NARROW_MARKERS {
                descriptors.push(
                    4 | (1 << 4)
                        | (a_format << 7)
                        | (b_format << 10)
                        | ((128 >> 3) << 17)
                        | (((128 * cta) >> 4) << 24),
                );
            }
        }
    } else if kind == "kind::f8f6f4" && cta_group == 1 {
        for d_format in 0..2_i128 {
            for (a_format, _) in DENSE_NARROW_MARKERS {
                for (b_format, _) in DENSE_NARROW_MARKERS {
                    descriptors.push((d_format << 4) | (a_format << 7) | (b_format << 10));
                }
            }
        }
    } else if kind == "kind::f8f6f4" && cta_group == 2 {
        // The shape fields do not affect the engine type specialization. Use
        // one legal M=256, N=128, K=32 image while enumerating the independently
        // specialized A/B formats; the runtime decoder validates actual values.
        for d_format in 0..2_i128 {
            for (a_format, _) in DENSE_NARROW_MARKERS {
                for (b_format, _) in DENSE_NARROW_MARKERS {
                    descriptors.push(
                        (d_format << 4)
                            | (a_format << 7)
                            | (b_format << 10)
                            | ((128 >> 3) << 17)
                            | ((256 >> 4) << 24),
                    );
                }
            }
        }
    } else if kind == "kind::tf32" {
        descriptors.push((1 << 4) | (2 << 7) | (2 << 10) | sparse_bit);
    } else {
        return unmodeled(
            unmodeled_target(op_name),
            format!(
                "{op_name} {kind} {mode} cta_group={cta_group} has no exact NumSim engine variant"
            ),
        );
    }
    let mut variants = Vec::new();
    for descriptor in descriptors {
        variants.push((
            descriptor,
            dense_mma_variant(op_name, kind, mode, cta_group, predicated, descriptor)?,
        ));
    }
    Ok(variants)
}

/// Decode collector actions once into fill,
/// require-valid and discard sets.
pub fn collector_masks(decoded: &DecodedPtx) -> AResult<(i64, i64, i64)> {
    let op_name = decoded.op_name.as_str();
    let ws = op_name.contains("_ws_");
    let modifiers: &[(&str, &str)] = if ws {
        &[("collector_b", "b0")]
    } else {
        &[("collector_a", "a"), ("collector_b", "b")]
    };
    let (mut fill, mut required, mut discard) = (0_i64, 0_i64, 0_i64);
    for (name, default) in modifiers {
        let token = decoded.modifier_or_empty(name);
        let text = if token.is_empty() {
            format!("collector::{default}::discard")
        } else {
            token.to_owned()
        };
        let fields: Vec<&str> = text.split("::").collect();
        if fields.len() != 3 || fields[0] != "collector" {
            return unsupported(format!("{op_name} malformed collector {:?}", &text));
        }
        let (slot, action) = (fields[1], fields[2]);
        let slot_bit = if ws {
            match slot {
                "b0" => Some(1),
                "b1" => Some(2),
                "b2" => Some(3),
                "b3" => Some(4),
                _ => None,
            }
        } else {
            match slot {
                "a" => Some(0),
                "b" => Some(1),
                _ => None,
            }
        };
        let Some(slot_bit) = slot_bit.filter(|_| ws || slot == *default) else {
            return unsupported(format!("{op_name} invalid collector slot {:?}", slot));
        };
        let bit = 1_i64 << slot_bit;
        match action {
            "fill" => fill |= bit,
            "use" => required |= bit,
            "lastuse" => {
                required |= bit;
                discard |= bit;
            }
            "discard" => discard |= bit,
            _ => return unsupported(format!("{op_name} invalid collector action {:?}", action)),
        }
    }
    Ok((fill, required, discard))
}

pub fn mma_effect_variant(variant: &str, decoded: &DecodedPtx) -> String {
    if !decoded.modifier_or_empty("ashift").is_empty() {
        return format!("v2::tcgen05::variant::MmaAshift<{variant}>");
    }
    if decoded.has_operand("b_decompress_metadata") {
        return format!("v2::tcgen05::variant::MmaLutB<{variant}>");
    }
    variant.to_owned()
}

pub fn collector_variant(variant: &str, masks: (i64, i64, i64)) -> String {
    let (fill, required, discard) = masks;
    format!("v2::tcgen05::variant::MmaCollectors<{variant}, {fill}, {required}, {discard}>")
}

pub struct DenseMmaParts {
    pub destination: ObjectRef,
    pub a_operand: ObjectRef,
    pub a_type: &'static str,
    pub b_descriptor: ObjectRef,
    pub instruction: ObjectRef,
    pub enable_input_d: ObjectRef,
    pub masks: Vec<Option<ObjectRef>>,
    pub predicate: Option<ObjectRef>,
    pub kind: String,
    pub mode: &'static str,
    pub cta_group: i64,
    /// `dense_mma_candidate_variants` of this instruction.
    pub candidate_variants: Vec<(i128, String)>,
    pub collector_masks: (i64, i64, i64),
    /// The WS zero-column mask operand.
    pub ws_mask: Option<ObjectRef>,
    /// The sparse metadata TMEM address.
    pub metadata: Option<ObjectRef>,
    /// The LUT-B decompression metadata address.
    pub lut_b: Option<ObjectRef>,
}

pub fn decoded_tcgen_dense_mma_parts(decoded: &DecodedPtx) -> AResult<DenseMmaParts> {
    decoded.require_void()?;
    let op_name = decoded.op_name.as_str();
    if op_name.contains("block_scale") {
        return Err(Failure::Ffi(ffi_error(&format!(
            "dense PTX TCGEN MMA emitter received {op_name}"
        ))));
    }
    let action = decoded.modifier("action")?;
    if action != "mma" {
        return unsupported(format!("{op_name} requires action='mma', got {:?}", action));
    }
    let weight_stationary = op_name.contains("_ws_");
    if weight_stationary {
        let ws = decoded.modifier("ws")?;
        if ws != "ws" {
            return unsupported(format!("{op_name} requires ws='ws', got {:?}", ws));
        }
    }
    let cta_group = decoded.cta_group(None)?;
    if weight_stationary && cta_group != 1 {
        return unsupported(format!("{op_name} requires cta_group=1"));
    }
    let mode = if decoded.has_operand("a_desc") {
        "ss"
    } else {
        "ts"
    };
    let kind = decoded.modifier("kind")?.to_owned();
    let collectors = collector_masks(decoded)?;
    if op_name.contains("_sp_")
        && !matches!(
            kind.as_str(),
            "kind::ti16" | "kind::tf32" | "kind::f16" | "kind::f8f6f4"
        )
    {
        return unmodeled(
            unmodeled_target(op_name),
            format!("sparse {kind} has no exact NumSim engine variant"),
        );
    }
    let candidate_variants =
        dense_mma_candidate_variants(op_name, &kind, mode, cta_group, decoded.predicate.is_some())?;
    let destination = decoded.scalar_operand("d_tmem")?;
    let a_name = if mode == "ss" { "a_desc" } else { "a_tmem" };
    let a_operand = decoded.scalar_operand(a_name)?;
    let b_name = if decoded.has_operand("b_decompress_metadata") {
        "b_compressed_desc"
    } else {
        "b_desc"
    };
    let b_descriptor = decoded.scalar_operand(b_name)?;
    let instruction = decoded.scalar_operand("idesc")?;
    let enable_input_d = decoded.scalar_operand("enable_input_d")?;
    let mut ws_mask = None;
    let masks: Vec<Option<ObjectRef>> = if weight_stationary {
        ws_mask = optional_operand(decoded, "zero_col_mask")?;
        if let Some(zero_column_mask) = &ws_mask {
            require_operand_dtype(decoded, "zero_col_mask", zero_column_mask, &["uint64"])?;
        }
        // WS has a separate column mask, not a disable-output-lane mask.
        vec![None, None, None, None]
    } else {
        let masks = decoded.operand("disable_output_lane")?.to_vec();
        let expected_mask_count = if cta_group == 2 { 8 } else { 4 };
        if masks.len() != expected_mask_count {
            return unsupported(format!(
                "{op_name} cta_group={cta_group} expects {expected_mask_count} disable-output masks, got {}",
                masks.len()
            ));
        }
        masks
    };
    require_operand_dtype(decoded, "d_tmem", &destination, &["int32", "uint32"])?;
    if mode == "ss" {
        require_operand_dtype(decoded, a_name, &a_operand, &["uint64"])?;
    } else {
        require_operand_dtype(decoded, a_name, &a_operand, &["int32", "uint32"])?;
    }
    require_operand_dtype(decoded, b_name, &b_descriptor, &["uint64"])?;
    let lut_b = optional_operand(decoded, "b_decompress_metadata")?;
    if let Some(lut_b) = &lut_b {
        require_operand_dtype(
            decoded,
            "b_decompress_metadata",
            lut_b,
            &["int32", "uint32"],
        )?;
    }
    require_operand_dtype(decoded, "idesc", &instruction, &["uint32"])?;
    let metadata = optional_operand(decoded, "sp_meta_tmem")?;
    if let Some(metadata) = &metadata {
        require_operand_dtype(decoded, "sp_meta_tmem", metadata, &["int32", "uint32"])?;
    }
    require_operand_dtype(
        decoded,
        "enable_input_d",
        &enable_input_d,
        &["bool", "int32", "uint32"],
    )?;
    if !weight_stationary {
        for (index, mask) in masks.iter().enumerate() {
            let Some(mask) = mask else {
                return Err(sink_lane(decoded, &format!("disable_output_lane[{index}]")));
            };
            require_operand_dtype(
                decoded,
                &format!("disable_output_lane[{index}]"),
                mask,
                &["int32", "uint32"],
            )?;
        }
    }
    Ok(DenseMmaParts {
        destination,
        a_operand,
        a_type: if mode == "ss" { "u64" } else { "u32" },
        b_descriptor,
        instruction,
        enable_input_d,
        masks,
        predicate: decoded.predicate.clone(),
        kind,
        mode,
        cta_group,
        candidate_variants,
        collector_masks: collectors,
        ws_mask,
        metadata,
        lut_b,
    })
}

pub struct BlockMmaParts {
    pub destination: ObjectRef,
    pub a_operand: ObjectRef,
    pub a_type: &'static str,
    pub b_descriptor: ObjectRef,
    pub sfa: ObjectRef,
    pub sfb: ObjectRef,
    pub instruction: ObjectRef,
    pub enable_input_d: ObjectRef,
    pub predicate: Option<ObjectRef>,
    pub variant: String,
    pub collector_masks: (i64, i64, i64),
    /// The sparsity metadata TMEM address.
    pub metadata: Option<ObjectRef>,
    /// The LUT-B decompression metadata address.
    pub lut_b: Option<ObjectRef>,
}

pub fn decoded_tcgen_block_mma_parts(decoded: &DecodedPtx) -> AResult<BlockMmaParts> {
    decoded.require_void()?;
    let op_name = decoded.op_name.as_str();
    if !op_name.contains("block_scale") {
        return Err(Failure::Ffi(ffi_error(&format!(
            "block-scaled PTX TCGEN MMA emitter received {op_name}"
        ))));
    }
    let collectors = collector_masks(decoded)?;
    for (name, required) in [("action", "mma"), ("block_scale", "block_scale")] {
        let actual = decoded.modifier(name)?;
        if actual != required {
            return unsupported(format!(
                "{op_name} requires {name}={:?}, got {:?}",
                required, actual
            ));
        }
    }
    let cta_group = decoded.cta_group(None)?;
    let mode = if op_name.ends_with("_ss") { "ss" } else { "ts" };
    let mode_marker = upper_first(mode);
    let kind = decoded.modifier("kind")?;
    let block_form = op_name.ends_with("_block_ss") || op_name.ends_with("_block_ts");
    // The SM107 collector entry omits the scale-size slot. Its MXF4 default
    // and MXF8F6F4's single supported block size use the existing block32 path.
    let has_lut_b = decoded.has_operand("b_decompress_metadata");
    let scale = if op_name.contains("_collector_ab_") || has_lut_b {
        "block32"
    } else {
        decoded.modifier(if block_form {
            "block_size"
        } else {
            "scale_vec"
        })?
    };
    let mut variant = if kind == "kind::mxf4" && (scale == "scale_vec::2X" || scale == "block32") {
        format!("v2::tcgen05::variant::MmaBlockMxf4E8m0{mode_marker}Cta{cta_group}<<artifact-tmem-mode>>")
    } else if kind == "kind::mxf4nvf4" && (scale == "scale_vec::4X" || scale == "block16") {
        format!("v2::tcgen05::variant::MmaBlockMxf4nvf4E2m1{mode_marker}Cta{cta_group}<<artifact-tmem-mode>>")
    } else if kind == "kind::mxf4nvf4" && (scale == "scale_vec::2X" || scale == "block32") {
        format!("v2::tcgen05::variant::MmaBlockMxf4nvf4Vec2{mode_marker}Cta{cta_group}<<artifact-tmem-mode>>")
    } else if kind == "kind::mxf8f6f4" && (scale == "scale_vec::1X" || scale == "block32") {
        format!("v2::tcgen05::variant::MmaBlockMxf8f6f4E8m0{mode_marker}Cta{cta_group}<<artifact-tmem-mode>>")
    } else {
        return unmodeled(
            unmodeled_target(op_name),
            format!(
                "{op_name} cta_group={cta_group} {kind} {scale} has no exact NumSim engine variant"
            ),
        );
    };
    if scale.starts_with("scale_vec::") && kind != "kind::mxf8f6f4" {
        variant = variant.replace(
            "<artifact-tmem-mode>>",
            &format!("<artifact-tmem-mode>, {TCGEN_DESCRIPTOR_LAYOUT}, true>"),
        );
    }
    let destination = decoded.scalar_operand("d_tmem")?;
    let a_name = if mode == "ss" { "a_desc" } else { "a_tmem" };
    let a_operand = decoded.scalar_operand(a_name)?;
    let b_name = if has_lut_b {
        "b_compressed_desc"
    } else {
        "b_desc"
    };
    let b_descriptor = decoded.scalar_operand(b_name)?;
    let lut_b = optional_operand(decoded, "b_decompress_metadata")?;
    if let Some(lut_b) = &lut_b {
        require_operand_dtype(
            decoded,
            "b_decompress_metadata",
            lut_b,
            &["int32", "uint32"],
        )?;
        variant = mma_effect_variant(&variant, decoded);
    }
    let metadata = optional_operand(decoded, "sp_meta_tmem")?;
    if let Some(metadata) = &metadata {
        require_operand_dtype(decoded, "sp_meta_tmem", metadata, &["int32", "uint32"])?;
        variant = format!("v2::tcgen05::variant::MmaSparseBlock<{variant}>");
    }
    let instruction = decoded.scalar_operand("idesc")?;
    let sfa = decoded.scalar_operand("sfa_tmem")?;
    let sfb = decoded.scalar_operand("sfb_tmem")?;
    let enable_input_d = decoded.scalar_operand("enable_input_d")?;
    require_operand_dtype(decoded, "d_tmem", &destination, &["int32", "uint32"])?;
    if mode == "ss" {
        require_operand_dtype(decoded, a_name, &a_operand, &["uint64"])?;
    } else {
        require_operand_dtype(decoded, a_name, &a_operand, &["int32", "uint32"])?;
    }
    require_operand_dtype(decoded, b_name, &b_descriptor, &["uint64"])?;
    require_operand_dtype(decoded, "idesc", &instruction, &["uint32"])?;
    require_operand_dtype(decoded, "sfa_tmem", &sfa, &["int32", "uint32"])?;
    require_operand_dtype(decoded, "sfb_tmem", &sfb, &["int32", "uint32"])?;
    require_operand_dtype(
        decoded,
        "enable_input_d",
        &enable_input_d,
        &["bool", "int32", "uint32"],
    )?;
    if let Some(predicate) = &decoded.predicate {
        require_operand_dtype(
            decoded,
            "predicate",
            predicate,
            crate::dtypes::predicate_dtypes(),
        )?;
    }
    Ok(BlockMmaParts {
        destination,
        a_operand,
        a_type: if mode == "ss" { "u64" } else { "u32" },
        b_descriptor,
        sfa,
        sfb,
        instruction,
        enable_input_d,
        predicate: decoded.predicate.clone(),
        variant,
        collector_masks: collectors,
        metadata,
        lut_b,
    })
}

/// The parsed parts of one raw TCGEN MMA call.
pub enum TcgenMmaParts {
    Block(BlockMmaParts),
    Dense(DenseMmaParts),
}

/// The parsed parts.

// ----------------------------------------------------------------------
// Descriptor encoders.
// ----------------------------------------------------------------------

pub fn parse_descriptor(
    ctx: &Ctx,
    node: &ObjectRef,
    op_name: &str,
) -> AResult<TcgenDescriptorCall> {
    match parse_tcgen_descriptor_call(ctx, node)? {
        Some(parsed) => Ok(parsed),
        None => Err(Failure::Ffi(ffi_error(&format!(
            "TCGEN descriptor lowerer received {op_name}"
        )))),
    }
}

/// The parsed descriptor call.

// ----------------------------------------------------------------------
// Family dispatch.
// ----------------------------------------------------------------------

/// Resolution of the `raw_tcgen` kind.

fn v2_register_marker(rust_type: &str) -> Option<&'static str> {
    Some(match rust_type {
        "i8" => "I8",
        "i16" => "I16",
        "i32" => "I32",
        "i64" => "I64",
        "u8" => "U8",
        "u16" => "U16",
        "u32" => "U32",
        "u64" => "U64",
        _ => return None,
    })
}

impl<'a> Emitter<'a> {
    fn raw_tcgen_physical_pointer(
        &mut self,
        expression: &ObjectRef,
        label: &str,
    ) -> AResult<RustValue> {
        let mut value = self.emit_expr(expression)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_generic_pointer(value, "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} must resolve to a physical address"));
        }
        Ok(value)
    }

    /// Lower one runtime operand to the exact register type
    /// consumed by PTX.
    fn raw_tcgen_v2_register(
        &mut self,
        expression: &ObjectRef,
        target_type: &str,
        label: &str,
        source_op_id: i64,
    ) -> AResult<String> {
        let value = self.emit_expr(expression)?;
        let source_type = value.rust_type.clone();
        if source_type != "bool" && v2_register_marker(&source_type).is_none() {
            return unsupported(format!(
                "{label} lowered to {source_type}, expected a scalar integer or bool"
            ));
        }
        let value = self.as_warp_value(value);
        let source = abi::register(&value.code);
        let result = self.control_name("raw_tcgen_operand");
        let site = self.v2_site(Some(source_op_id));
        if source_type == target_type {
            self.emit_line(&format!("let {result} = {source};"));
            return Ok(result);
        }
        if target_type == "bool" {
            if source_type == "bool" {
                self.emit_line(&format!("let {result} = {source};"));
                return Ok(result);
            }
            let marker = v2_register_marker(&source_type).expect("integer source");
            let call = abi::lane_call(
                "reg::setp",
                &site,
                &[format!(
                    "({source}, {})",
                    abi::splat(&format!("0_{source_type}"))
                )],
                Some(&format!(
                    "v2::reg::variant::Setp<v2::reg::variant::{marker}, v2::reg::variant::Ne>"
                )),
            );
            self.emit_line(&format!("let {result} = {call};"));
            return Ok(result);
        }
        let Some(target_marker) = v2_register_marker(target_type) else {
            return unsupported(format!(
                "{label} cannot convert {source_type} to {target_type}"
            ));
        };
        if source_type == "bool" {
            let call = abi::lane_call(
                "reg::selp",
                &site,
                &[format!(
                    "({source}, {}, {})",
                    abi::splat(&format!("1_{target_type}")),
                    abi::splat(&format!("0_{target_type}"))
                )],
                Some(&format!("v2::reg::variant::{target_marker}")),
            );
            self.emit_line(&format!("let {result} = {call};"));
            return Ok(result);
        }
        let source_marker = v2_register_marker(&source_type).expect("integer source");
        let call = abi::lane_call(
            "reg::cvt",
            &site,
            &[source],
            Some(&format!(
                "v2::reg::variant::Cvt<v2::reg::variant::{source_marker}, v2::reg::variant::{target_marker}>"
            )),
        );
        self.emit_line(&format!("let {result} = {call};"));
        Ok(result)
    }

    /// `(anchor, access marker, logical name)`.
    fn tmem_anchor(&mut self) -> AResult<(String, String, String)> {
        let mut anchor: Option<String> = None;
        let mut logical_name: Option<String> = None;
        let mut has_dynamic_address = self.tmem.dynamic_lifecycle;
        let mut backing_indices: Vec<usize> = Vec::new();
        let candidates: Vec<(BufferVar, Option<usize>, bool)> = self
            .buffers
            .iter()
            .filter_map(|(buffer, code)| {
                let plan = self.plan_of(code);
                if plan.space == MemorySpace::Tmem && plan.dynamic_data_var.is_none() {
                    Some((
                        buffer.clone(),
                        plan.backing_index,
                        plan.layout.allocated_addr_static.is_none(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        for (buffer, backing_index, dynamic_address) in candidates {
            let Some(backing_index) = backing_index else {
                return unsupported("raw TCGEN TMEM view has no physical backing");
            };
            if anchor.is_none() {
                anchor = Some(self.buffer_ref(&buffer)?);
                logical_name = Some(self.logical_buffer_name(&buffer)?);
            }
            has_dynamic_address |= dynamic_address;
            if !backing_indices.contains(&backing_index) {
                backing_indices.push(backing_index);
            }
        }
        if self.tmem.implicit {
            if anchor.is_some() {
                return unsupported(
                    "raw TCGEN cannot mix implicit TMEM with declared physical views",
                );
            }
            anchor = Some("buffers.implicit_tmem".to_owned());
            logical_name = Some("implicit_tmem".to_owned());
            has_dynamic_address = true;
        }
        let Some(anchor) = anchor else {
            return unsupported("raw TCGEN transfer requires a declared TMEM physical view");
        };
        if backing_indices.len() > 1 {
            return unsupported("raw TCGEN TMEM views span multiple physical backings");
        }
        let mode = if has_dynamic_address {
            "DynamicTmem"
        } else {
            "StaticTmem"
        };
        Ok((
            anchor,
            format!("v2::tcgen05::variant::{mode}"),
            logical_name.expect("logical name"),
        ))
    }

    /// Persistent candidates share one descriptor
    /// domain field per storage-index set.
    fn raw_tcgen_shared_candidates(&mut self) -> AResult<String> {
        let mut selected: Vec<((usize, i64, i64), BufferVar)> = Vec::new();
        for (buffer, code) in &self.buffers {
            let plan = self.plan_of(code);
            if plan.space != MemorySpace::Shared || plan.dynamic_data_var.is_some() {
                continue;
            }
            let Some(backing_index) = plan.backing_index else {
                return Err(Failure::Ffi(ffi_error(
                    "shared physical view has no backing index",
                )));
            };
            let Some(element_count) = plan.layout.element_count else {
                return Err(Failure::Ffi(ffi_error(
                    "shared physical view has no static element count",
                )));
            };
            let key = (
                backing_index,
                plan.layout.elem_offset * plan.layout.itemsize,
                element_count * plan.layout.itemsize,
            );
            if !selected.iter().any(|(existing, _)| *existing == key) {
                selected.push((key, buffer.clone()));
            }
        }
        if selected.is_empty() {
            return unsupported(
                "raw TCGEN descriptor requires a declared physical shared-memory backing",
            );
        }
        selected.sort_by(|left, right| left.0.cmp(&right.0));
        let mut indices: Option<Vec<usize>> = Some(Vec::new());
        for (_, buffer) in &selected {
            match self.buffer_code(buffer)?.storage_index {
                Some(storage_index) => indices
                    .as_mut()
                    .expect("persistent candidates")
                    .push(storage_index),
                None => {
                    indices = None;
                    break;
                }
            }
        }
        if let Some(indices) = indices {
            let existing = self
                .raw_tcgen
                .shared_descriptor_domains
                .iter()
                .find(|(key, _)| *key == indices)
                .map(|(_, field)| field.clone());
            let field = match existing {
                Some(field) => field,
                None => {
                    let field = format!(
                        "shared_descriptor_domain_{}",
                        self.raw_tcgen.shared_descriptor_domains.len()
                    );
                    self.raw_tcgen
                        .shared_descriptor_domains
                        .push((indices, field.clone()));
                    field
                }
            };
            return Ok(format!("(buffers.{field}).clone()"));
        }
        let mut refs = Vec::new();
        for (_, buffer) in &selected {
            refs.push(format!("({}).clone()", self.buffer_ref(buffer)?));
        }
        Ok(format!(
            "v2_descriptor_domain::<v2::Shared>(vec![{}])",
            refs.join(", ")
        ))
    }

    /// A stateful ABI call for this family.
    fn emit_raw_tcgen_stateful(
        &mut self,
        function: &str,
        source_op_id: i64,
        arguments: String,
        variant: &str,
        context: Option<&str>,
    ) {
        let site = self.v2_site(Some(source_op_id));
        let invocation = abi::warp_call(
            function,
            &site,
            &[arguments],
            Some(variant),
            context,
            false,
            true,
        );
        self.emit_line(&format!("{invocation};"));
    }

    fn emit_matrix_descriptor(
        &mut self,
        call: &TcgenDescriptorCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let site = self.v2_site(Some(source_op_id));
        let destination =
            self.raw_tcgen_physical_pointer(&call.args[0], "matrix descriptor destination")?;
        let destination_name = self.control_name("raw_tcgen_matrix_destination");
        self.emit_line(&format!(
            "let {destination_name} = {};",
            abi::address("v2::Generic", &abi::cloned(&destination.code), None)
        ));
        let addresses = if call.matrix_source_is_null {
            let addresses = self.control_name("raw_tcgen_null_shared_addresses");
            self.emit_line(&format!("let {addresses} = {};", abi::splat("0_u32")));
            addresses
        } else {
            let source = self.emit_raw_shared_pointer(&call.args[1], None, "ctx.active_mask()")?;
            let addresses = self.control_name("raw_tcgen_shared_addresses");
            let cvta = abi::warp_call(
                CVTA,
                &site,
                &[abi::address(
                    "v2::Generic",
                    &abi::cloned(&source.code),
                    None,
                )],
                Some("v2::addr::variant::GenericToSharedU32<false>"),
                None,
                false,
                true,
            );
            self.emit_line(&format!("let {addresses} = {cvta};"));
            addresses
        };
        let values = self.control_name("raw_tcgen_matrix_values");
        let mut fields: Vec<RustValue> = Vec::new();
        for argument in &call.args[2..] {
            let value = self.emit_expr(argument)?;
            fields.push(self.as_warp_value(value));
        }
        let (ldo, sdo, swizzle) = (&fields[0], &fields[1], &fields[2]);
        // TVM's pure-C SmemDescriptor helper truncates unsigned bit fields. It
        // neither reads the source allocation nor validates a future TCGEN access.
        // A swizzle outside its switch cases leaves the zero-initialized layout.
        self.emit_line(&format!("let mut {values} = {};", abi::splat("0_u64")));
        self.emit_line(&format!(
            "for lane in {}.active_mask() {{",
            abi::context("ctx")
        ));
        self.indent += 1;
        let address = self.control_name("raw_tcgen_matrix_address");
        self.emit_line(&format!(
            "let {address} = shared_address_byte_offset({addresses}[lane]);"
        ));
        let layout_type = self.control_name("raw_tcgen_matrix_layout");
        self.emit_line(&format!(
            "let {layout_type}: u64 = match {}[lane] {{ 1 => 6, 2 => 4, 3 => 2, 4 => 1, _ => 0 }};",
            swizzle.code
        ));
        self.emit_line(&format!(
            "{values}[lane] = u64::from(({addresses}[lane] >> 4) & 0x3fff_u32) | (({}[lane] as u64 & 0x3fff_u64) << 16) | (({}[lane] as u64 & 0x3fff_u64) << 32) | (1_u64 << 46) | ({layout_type} << 61);",
            ldo.code, sdo.code
        ));
        self.indent -= 1;
        self.emit_line("}");
        let store = abi::warp_call(
            MEM_ST,
            &site,
            &[format!("({destination_name}, {values})")],
            Some("v2::mem::variant::St<v2::reg::variant::U64, v2::Generic>"),
            None,
            false,
            true,
        );
        self.emit_line(&format!("{store};"));
        Ok(())
    }

    fn emit_instruction_descriptor(
        &mut self,
        call: &TcgenDescriptorCall,
        source_op_id: i64,
    ) -> AResult<()> {
        let destination =
            self.raw_tcgen_physical_pointer(&call.args[0], "instruction descriptor destination")?;
        let destination_name = self.control_name("raw_tcgen_instr_destination");
        let encoded = call.encoded_u32.expect("encoded descriptor");
        self.emit_line(&format!(
            "let {destination_name} = {};",
            abi::address("v2::Generic", &abi::cloned(&destination.code), None)
        ));
        self.emit_raw_tcgen_stateful(
            MEM_ST,
            source_op_id,
            format!(
                "({destination_name}, {})",
                abi::splat(&format!("{encoded}_u32"))
            ),
            "v2::mem::variant::St<v2::reg::variant::U32, v2::Generic>",
            None,
        );
        Ok(())
    }

    /// Row/column are never given by the raw family. The instruction
    /// predicate gates operand evaluation and the transfer together.
    #[allow(clippy::too_many_arguments)]
    fn emit_ldst_operands(
        &mut self,
        op_name: &str,
        address_expression: &ObjectRef,
        registers: &[ObjectRef],
        variant_of: impl Fn(&str) -> String,
        instruction: &str,
        predicate: Option<&ObjectRef>,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            predicate,
            "raw_tcgen_transfer",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let address = self.raw_tcgen_v2_register(
            address_expression,
            "u32",
            &format!("{op_name}.address"),
            source_op_id,
        )?;
        let row = abi::splat("0_i64");
        let col = abi::splat("0_i64");
        let (anchor, access, logical_name) = self.tmem_anchor()?;
        let mut register_arguments: Vec<String> = Vec::new();
        for register in registers {
            if instruction == "st" {
                let value = self.emit_expr(register)?;
                let value = self.as_warp_value(value);
                let bits =
                    reinterpret_atom(&format!("{}[lane]", value.code), &value.rust_type, "u32");
                let name = self.control_name("raw_tcgen_st_value");
                self.emit_line(&format!("let {name} = {};", abi::per_lane(&bits)));
                register_arguments.push(name);
                continue;
            }
            let load = register
                .as_node::<TensorLoadObj>()
                .expect("validated TensorLoad");
            let Some(buffer) = as_buffer(&oref(load.source.clone())) else {
                return not_covered("tcgen05 register source is not a typed buffer");
            };
            self.record_global_write(&buffer)?;
            let indices: Vec<PrimExpr> = load.indices.iter().collect();
            let pointer = self.emit_buffer_address(&buffer, &indices)?;
            let pointer_name = self.control_name(&format!("raw_tcgen_{instruction}_register"));
            self.emit_line(&format!("let {pointer_name} = {};", pointer.code));
            register_arguments.push(abi::address(
                "v2::Register",
                &abi::cloned(&pointer_name),
                None,
            ));
        }
        self.emit_raw_tcgen_stateful(
            &format!("tcgen05::{instruction}"),
            source_op_id,
            format!(
                "({}, {address}, {row}, {col}, vec![{}])",
                named_buffer("v2::Tmem", &anchor, &logical_name),
                register_arguments.join(", ")
            ),
            &variant_of(&access),
            region.context.as_deref(),
        );
        self.close_predicated_region(region);
        Ok(())
    }

    /// `emit_ptx_tcgen_ldst`.
    pub fn emit_ptx_tcgen_ldst(
        &mut self,
        decoded: &DecodedPtx,
        parts: &LdstParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let instruction = parts.instruction;
        let address = parts.address.clone();
        let registers = parts.registers.clone();
        self.emit_ldst_operands(
            &decoded.op_name,
            &address,
            &registers,
            |access| ldst_variant(parts, access),
            instruction,
            decoded.predicate.as_ref(),
            source_op_id,
        )
    }

    /// `variants` maps each engine variant to the
    /// masked descriptor values that select it.
    #[allow(clippy::too_many_arguments)]
    fn emit_mma_operands(
        &mut self,
        op_name: &str,
        destination_expression: &ObjectRef,
        a_expression: &ObjectRef,
        a_type: &str,
        b_descriptor_expression: &ObjectRef,
        instruction_expression: &ObjectRef,
        enable_input_d_expression: &ObjectRef,
        mask_expressions: &[Option<ObjectRef>],
        predicate_expression: Option<&ObjectRef>,
        variants: &[(String, Vec<i128>)],
        descriptor_mask: i128,
        source_op_id: i64,
        ws_mask: Option<&ObjectRef>,
        weight_stationary: bool,
        metadata: Option<&ObjectRef>,
        lut_b: Option<&ObjectRef>,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            predicate_expression,
            "tcgen_mma",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let destination = self.raw_tcgen_v2_register(
            destination_expression,
            "u32",
            &format!("{op_name} destination address"),
            source_op_id,
        )?;
        let a_operand = self.raw_tcgen_v2_register(
            a_expression,
            a_type,
            &format!("{op_name} A operand"),
            source_op_id,
        )?;
        let b_descriptor = self.raw_tcgen_v2_register(
            b_descriptor_expression,
            "u64",
            &format!("{op_name} B descriptor"),
            source_op_id,
        )?;
        let instruction = self.raw_tcgen_v2_register(
            instruction_expression,
            "u32",
            &format!("{op_name} instruction descriptor"),
            source_op_id,
        )?;
        let enable_input_d = self.raw_tcgen_v2_register(
            enable_input_d_expression,
            "bool",
            &format!("{op_name} enable_input_d"),
            source_op_id,
        )?;
        let mut masks: Vec<String> = Vec::new();
        for (index, mask) in mask_expressions.iter().enumerate() {
            masks.push(match mask {
                None => abi::splat("0_u32"),
                Some(mask) => self.raw_tcgen_v2_register(
                    mask,
                    "u32",
                    &format!("{op_name} disable_output_lane[{index}]"),
                    source_op_id,
                )?,
            });
        }
        // The enclosing context has already evaluated the predicate and
        // selected its lanes. Re-evaluating a collective such as elect.sync
        // here would execute it under that narrowed mask.
        let predicate = if predicate_expression.is_none() || weight_stationary || metadata.is_some()
        {
            None
        } else {
            Some(abi::splat("1_u32"))
        };
        let (anchor, access, logical_name) = self.tmem_anchor()?;
        let mut runtime_args = vec![
            named_buffer("v2::Tmem", &anchor, &logical_name),
            self.raw_tcgen_shared_candidates()?,
            destination,
            a_operand,
            b_descriptor,
            instruction.clone(),
            enable_input_d,
            format!("[{}]", masks.join(", ")),
        ];
        if weight_stationary || metadata.is_some() {
            let zero_mask = match ws_mask {
                None => abi::splat("0_u64"),
                Some(mask) => self.raw_tcgen_v2_register(
                    mask,
                    "u64",
                    &format!("{op_name} zero-column mask"),
                    source_op_id,
                )?,
            };
            match metadata {
                None => *runtime_args.last_mut().expect("disable-output masks") = zero_mask,
                Some(metadata) => {
                    let sparse_metadata = self.raw_tcgen_v2_register(
                        metadata,
                        "u32",
                        &format!("{op_name} sparse metadata"),
                        source_op_id,
                    )?;
                    runtime_args.push(zero_mask);
                    runtime_args.push(sparse_metadata);
                }
            }
        }
        if let Some(predicate) = predicate {
            runtime_args.push(predicate);
        }
        let mut arguments = format!("({})", runtime_args.join(", "));
        if let Some(lut_b) = lut_b {
            let lookup = self.raw_tcgen_v2_register(
                lut_b,
                "u32",
                &format!("{op_name} LUT-B address"),
                source_op_id,
            )?;
            arguments = format!("({arguments}, {lookup})");
        }
        let context = region.context.clone();
        let emit = |emitter: &mut Self, variant: &str| {
            emitter.emit_raw_tcgen_stateful(
                MMA,
                source_op_id,
                arguments.clone(),
                &variant.replace("<artifact-tmem-mode>", &access),
                context.as_deref(),
            );
        };
        if variants.len() == 1 {
            emit(self, &variants[0].0);
        } else {
            // Dispatch the issuing lane, but preserve the complete context:
            // the existing MMA entry must still reject multiple issuers.
            let lane = self.control_name("tcgen_mma_dispatch_lane");
            let context_code = context.clone().unwrap_or_else(|| abi::context("ctx"));
            self.emit_line(&format!(
                "if let Some({lane}) = {context_code}.active_mask().first_active() {{"
            ));
            self.indent += 1;
            self.emit_line(&format!(
                "match {instruction}[{lane}] & {descriptor_mask}_u32 {{"
            ));
            self.indent += 1;
            for (variant, descriptors) in variants {
                let patterns: Vec<String> = descriptors
                    .iter()
                    .map(|value| format!("{value}_u32"))
                    .collect();
                self.emit_line(&format!("{} => {{", patterns.join(" | ")));
                self.indent += 1;
                emit(self, variant);
                self.indent -= 1;
                self.emit_line("}");
            }
            if descriptor_mask == 0xFFFF_FFFF {
                self.emit_line(
                    "_ => return Err(EngineError::message(\"TCGEN descriptor escaped its proven finite set\")),",
                );
            } else {
                // Keep malformed-descriptor checks owned by the existing
                // decoder, including its issuing-lane and predication rules.
                self.emit_line("_ => {");
                self.indent += 1;
                emit(self, &variants[0].0);
                self.indent -= 1;
                self.emit_line("}");
            }
            self.indent -= 1;
            self.emit_line("}");
            self.indent -= 1;
            self.emit_line("}");
        }
        self.close_predicated_region(region);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_mma_block_scale_operands(
        &mut self,
        op_name: &str,
        destination_expression: &ObjectRef,
        a_expression: &ObjectRef,
        a_type: &str,
        b_descriptor_expression: &ObjectRef,
        sfa_expression: &ObjectRef,
        sfb_expression: &ObjectRef,
        instruction_expression: &ObjectRef,
        enable_input_d_expression: &ObjectRef,
        predicate_expression: Option<&ObjectRef>,
        variant: &str,
        source_op_id: i64,
        collector_masks: (i64, i64, i64),
        lut_b: Option<&ObjectRef>,
        metadata: Option<&ObjectRef>,
    ) -> AResult<()> {
        let marker = tcgen_descriptor_variant(self.func)?;
        let variant = variant.replace(
            "<artifact-tmem-mode>>",
            &format!("<artifact-tmem-mode>, {marker}>"),
        );
        let variant = variant.replace(TCGEN_DESCRIPTOR_LAYOUT, &marker);
        let region = self.open_shadow_predicated_region(
            predicate_expression,
            "raw_tcgen_block_mma_issue",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let destination = self.raw_tcgen_v2_register(
            destination_expression,
            "u32",
            &format!("{op_name} destination address"),
            source_op_id,
        )?;
        let names = [
            self.raw_tcgen_v2_register(
                a_expression,
                a_type,
                &format!("{op_name} A operand"),
                source_op_id,
            )?,
            self.raw_tcgen_v2_register(
                b_descriptor_expression,
                "u64",
                &format!("{op_name} B descriptor"),
                source_op_id,
            )?,
            self.raw_tcgen_v2_register(
                sfa_expression,
                "u32",
                &format!("{op_name} SFA address"),
                source_op_id,
            )?,
            self.raw_tcgen_v2_register(
                sfb_expression,
                "u32",
                &format!("{op_name} SFB address"),
                source_op_id,
            )?,
            self.raw_tcgen_v2_register(
                instruction_expression,
                "u32",
                &format!("{op_name} instruction descriptor"),
                source_op_id,
            )?,
            self.raw_tcgen_v2_register(
                enable_input_d_expression,
                "bool",
                &format!("{op_name} enable_input_d"),
                source_op_id,
            )?,
        ];
        let (anchor, access, logical_name) = self.tmem_anchor()?;
        let candidates = self.raw_tcgen_shared_candidates()?;
        let mut arguments = format!(
            "({}, {candidates}, {destination}, {})",
            named_buffer("v2::Tmem", &anchor, &logical_name),
            names.join(", ")
        );
        if let Some(metadata) = metadata {
            let address = self.raw_tcgen_v2_register(
                metadata,
                "u32",
                &format!("{op_name} sparsity metadata"),
                source_op_id,
            )?;
            arguments = format!("({arguments}, {address})");
        }
        if let Some(lut_b) = lut_b {
            let lookup = self.raw_tcgen_v2_register(
                lut_b,
                "u32",
                &format!("{op_name} LUT-B address"),
                source_op_id,
            )?;
            arguments = format!("({arguments}, {lookup})");
        }
        self.emit_raw_tcgen_stateful(
            MMA,
            source_op_id,
            arguments,
            &collector_variant(
                &variant.replace("<artifact-tmem-mode>", &access),
                collector_masks,
            ),
            region.context.as_deref(),
        );
        self.close_predicated_region(region);
        Ok(())
    }

    /// `emit_known_cuda_tcgen05_mxf4_block32_ss`: lower the validated DeepGEMM
    /// helper through the raw TCGEN path.
    pub fn emit_known_cuda_tcgen05_mxf4_block32_ss(
        &mut self,
        arguments: &[ObjectRef],
        source_op_id: i64,
    ) -> AResult<()> {
        if arguments.len() != 7 {
            return unsupported("tvm_builtin_tcgen05_mma_mxf4_block32_ss expects seven operands");
        }
        self.emit_mma_block_scale_operands(
            "tvm_builtin_tcgen05_mma_mxf4_block32_ss",
            &arguments[0],
            &arguments[1],
            "u64",
            &arguments[2],
            &arguments[5],
            &arguments[6],
            &arguments[3],
            &arguments[4],
            None,
            MMA_MXF4_BLOCK32_SS_VARIANT,
            source_op_id,
            (0, 0, 3),
            None,
            None,
        )
    }

    /// Row/column are never given by the raw family. The instruction
    /// predicate gates operand evaluation and the copy issue together.
    #[allow(clippy::too_many_arguments)]
    fn emit_cp_operands(
        &mut self,
        op_name: &str,
        address_expression: &ObjectRef,
        descriptor_expression: &ObjectRef,
        shape_code: usize,
        decompress_code: usize,
        cta_group: i64,
        predicate_expression: Option<&ObjectRef>,
        source_op_id: i64,
    ) -> AResult<()> {
        let region = self.open_shadow_predicated_region(
            predicate_expression,
            "raw_tcgen_cp_issue",
            &format!("{op_name} predicate must lower to bool or integer"),
        )?;
        let address = self.raw_tcgen_v2_register(
            address_expression,
            "u32",
            &format!("{op_name} address"),
            source_op_id,
        )?;
        let descriptor = self.raw_tcgen_v2_register(
            descriptor_expression,
            "u64",
            &format!("{op_name} descriptor"),
            source_op_id,
        )?;
        let row = abi::splat("0_i64");
        let col = abi::splat("0_i64");
        let (anchor, access, logical_name) = self.tmem_anchor()?;
        let shape = cp_shape_variant(shape_code);
        let decompress = cp_decompress_variant(decompress_code);
        let candidates = self.raw_tcgen_shared_candidates()?;
        let descriptor_marker = tcgen_descriptor_variant(self.func)?;
        self.emit_raw_tcgen_stateful(
            CP,
            source_op_id,
            format!(
                "({}, {candidates}, {address}, {descriptor}, {row}, {col})",
                named_buffer("v2::Tmem", &anchor, &logical_name)
            ),
            &format!(
                "v2::tcgen05::variant::Cp<v2::tcgen05::variant::{shape}, v2::tcgen05::variant::{decompress}, {cta_group}, {access}, {descriptor_marker}>"
            ),
            region.context.as_deref(),
        );
        self.close_predicated_region(region);
        Ok(())
    }

    /// `emit_ptx_tcgen_cp`.
    pub fn emit_ptx_tcgen_cp(
        &mut self,
        decoded: &DecodedPtx,
        parts: &CpParts,
        source_op_id: i64,
    ) -> AResult<()> {
        self.emit_cp_operands(
            &decoded.op_name,
            &parts.address,
            &parts.descriptor,
            parts.shape_code,
            parts.decompress_code,
            parts.cta_group,
            parts.predicate.as_ref(),
            source_op_id,
        )
    }

    /// Replace the artifact layout
    /// marker using kernel-static architecture.
    fn specialize_f8f6f4_descriptor_layout(
        &self,
        variant: &str,
        cta_group: i64,
    ) -> AResult<String> {
        if !variant.contains(TCGEN_DESCRIPTOR_LAYOUT) {
            return Ok(variant.to_owned());
        }
        if cta_group == 2 {
            require_tcgen_descriptor_layout(self.func)?;
        }
        Ok(variant.replace(
            TCGEN_DESCRIPTOR_LAYOUT,
            &tcgen_descriptor_variant(self.func)?,
        ))
    }

    /// `emit_ptx_tcgen_mma`.
    pub fn emit_ptx_tcgen_mma(
        &mut self,
        decoded: &DecodedPtx,
        parts: &TcgenMmaParts,
        source_op_id: i64,
    ) -> AResult<()> {
        let op_name = decoded.op_name.clone();
        let parts = match parts {
            TcgenMmaParts::Block(parts) => {
                return self.emit_mma_block_scale_operands(
                    &op_name,
                    &parts.destination,
                    &parts.a_operand,
                    parts.a_type,
                    &parts.b_descriptor,
                    &parts.sfa,
                    &parts.sfb,
                    &parts.instruction,
                    &parts.enable_input_d,
                    parts.predicate.as_ref(),
                    &parts.variant,
                    source_op_id,
                    parts.collector_masks,
                    parts.lut_b.as_ref(),
                    parts.metadata.as_ref(),
                );
            }
            TcgenMmaParts::Dense(parts) => parts,
        };
        // Only codec fields select a Rust type. Shapes, signs, transpose,
        // sparsity and reserved bits remain in the original runtime operand
        // and are checked by the existing native descriptor decoder.
        let mut descriptor_mask: i128 = (7 << 7) | (7 << 10);
        if parts.kind == "kind::f8f6f4" && !op_name.contains("_sp_") {
            descriptor_mask |= 3 << 4;
        }
        let mut variants: Vec<(String, Vec<i128>)> = Vec::new();
        for (descriptor, variant) in parts.candidate_variants.clone() {
            let variant = self.specialize_f8f6f4_descriptor_layout(&variant, parts.cta_group)?;
            let variant = collector_variant(
                &mma_effect_variant(&variant, decoded),
                parts.collector_masks,
            );
            match variants
                .iter_mut()
                .find(|(existing, _)| *existing == variant)
            {
                Some((_, descriptors)) => descriptors.push(descriptor & descriptor_mask),
                None => variants.push((variant, vec![descriptor & descriptor_mask])),
            }
        }
        self.emit_mma_operands(
            &op_name,
            &parts.destination,
            &parts.a_operand,
            parts.a_type,
            &parts.b_descriptor,
            &parts.instruction,
            &parts.enable_input_d,
            &parts.masks,
            parts.predicate.as_ref(),
            &variants,
            descriptor_mask,
            source_op_id,
            parts.ws_mask.as_ref(),
            op_name.contains("_ws_"),
            parts.metadata.as_ref(),
            parts.lut_b.as_ref(),
        )
    }
}

pub fn emit_ldst(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = decoded_ldst_parts(emitter.ctx, decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_tcgen_ldst(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_copy(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = decoded_tcgen_cp_parts(decoded)?;
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_tcgen_cp(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_mma(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let decoded = call.table()?;
    let parts = if decoded.op_name.contains("block_scale") {
        TcgenMmaParts::Block(decoded_tcgen_block_mma_parts(decoded)?)
    } else {
        let parts = decoded_tcgen_dense_mma_parts(decoded)?;
        if let Some(imm) = parts.instruction.as_node::<IntImmObj>() {
            dense_mma_variant(
                &decoded.op_name,
                &parts.kind,
                parts.mode,
                parts.cta_group,
                parts.predicate.is_some(),
                i128::from(int_value(imm)?),
            )?;
        }
        TcgenMmaParts::Dense(parts)
    };
    let source_op_id = call.source_op_id(emitter)?;
    emitter.emit_ptx_tcgen_mma(decoded, &parts, source_op_id)?;
    Ok(None)
}

pub fn emit_descriptor(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let parts = parse_descriptor(emitter.ctx, call.node, &call.op_name)?;
    emitter.record_pointer_write(&parts.args[0], "")?;
    let source_op_id = call.source_op_id(emitter)?;
    if call.op_name == "tirx.cuda.tcgen05_encode_matrix_descriptor" {
        emitter.emit_matrix_descriptor(&parts, source_op_id)?;
    } else {
        emitter.emit_instruction_descriptor(&parts, source_op_id)?;
    }
    Ok(None)
}
