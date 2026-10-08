//! Operand-carrier and marker mappings for the engine conversion ABI.
//! The target decoder owns PTX grammar; the engine owns variant implementations.

use crate::analyze::util::capitalize;

/// PTX `scaled` attrs modeled by the packed conversion catalogue.
pub const SCALED_UE8M0_N1: &str = "n1::ue8m0";
pub const SCALED_UE8M0_N2: &str = "n2::ue8m0";

const SCALAR_INTEGER_TYPES: [&str; 8] = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64"];
/// The decoded spelling shared by capability checks and ABI specialization.
pub(super) struct Modifiers<'a> {
    pub destination: &'a str,
    pub source: &'a str,
    pub rounding: &'a str,
    pub ftz: bool,
    pub sat: bool,
    pub relu: bool,
    pub satfinite: bool,
    pub pzo: bool,
    pub scaled: &'a str,
}

/// Scalar conversions represented by the engine's PackedMode rather than CvtMode.
pub(super) fn scalar_uses_packed_mode(m: &Modifiers) -> bool {
    matches!(
        (m.destination, m.source, m.rounding, m.ftz, m.sat),
        ("f16" | "bf16" | "tf32", "f32", "rn" | "rz", false, false)
            | ("tf32", "f32", "rna", false, false)
    )
}

/// The engine uses the Exact marker for these integer-to-float ABI entries.
const EXACT_INTEGER_FLOAT: &[(&str, &str)] = &[
    ("f16", "u8"),
    ("f16", "s8"),
    ("bf16", "u8"),
    ("bf16", "s8"),
    ("f32", "u8"),
    ("f32", "s8"),
    ("f32", "u16"),
    ("f32", "s16"),
    ("f64", "u8"),
    ("f64", "s8"),
    ("f64", "u16"),
    ("f64", "s16"),
    ("f64", "u32"),
    ("f64", "s32"),
];

pub fn scalar_cvt_rounding_is_inert(destination: &str, source: &str, sat: bool) -> bool {
    EXACT_INTEGER_FLOAT.contains(&(destination, source))
        || (sat && scalar_cvt_flush_is_inert(destination, source))
}

/// Integer-to-float engine entries use PreserveSubnormal for their FTZ axis.
pub fn scalar_cvt_flush_is_inert(destination: &str, source: &str) -> bool {
    SCALAR_INTEGER_TYPES.contains(&source) && ["f16", "bf16", "f32", "f64"].contains(&destination)
}

/// The ABI signature of one accepted packed spelling.
pub struct PackedCvtVariant {
    pub result_dtype: &'static str,
    pub operand_dtypes: Vec<&'static str>,
    pub specialization: String,
}

fn packed_type_marker(name: &str) -> &'static str {
    match name {
        "f32" => "F32",
        "f16x2" => "F16x2",
        "bf16x2" => "Bf16x2",
        "e4m3x2" => "E4m3x2",
        "e5m2x2" => "E5m2x2",
        "e2m1x2" => "E2m1x2",
        "e2m3x2" => "E2m3x2",
        "e3m2x2" => "E3m2x2",
        "ue5m3x2" => "Ue5m3x2",
        "e4m3x4" => "E4m3x4",
        "e5m2x4" => "E5m2x4",
        "e2m1x4" => "E2m1x4",
        "e2m3x4" => "E2m3x4",
        "e3m2x4" => "E3m2x4",
        "ue8m0x2" => "Ue8m0x2",
        "s2f6x2" => "S2f6x2",
        other => unreachable!("packed cvt marker for {other}"),
    }
}

/// Canonical engine carriers, shared by input and output positions.
fn carrier(name: &str) -> &'static str {
    match name {
        "f32" => "float32",
        "e2m1x2" => "uint8",
        "s2f6x2" | "e2m3x2" | "e3m2x2" | "ue5m3x2" | "e4m3x2" | "e5m2x2" | "e2m1x4" | "ue8m0x2" => {
            "uint16"
        }
        "e4m3x4" | "e5m2x4" | "e2m3x4" | "e3m2x4" | "f16x2" | "bf16x2" => "uint32",
        other => unreachable!("packed cvt carrier for {other}"),
    }
}

/// The `rbits` operand is a `.b32`; n1 and n2 scale factors are `.b8` and `.b16`.
const RBITS_CARRIER: &str = "uint32";

fn scale_carrier(scaled: &str) -> &'static str {
    match scaled {
        SCALED_UE8M0_N1 => "uint8",
        SCALED_UE8M0_N2 => "uint16",
        other => unreachable!("packed cvt scale carrier for {other}"),
    }
}

fn marker(name: &str) -> String {
    format!("v2::reg::variant::{}", packed_type_marker(name))
}

/// `packed_mode_marker`: the `PackedMode` specialization one narrow-float
/// spelling selects.
pub fn packed_mode_marker(
    rounding: &str,
    satfinite: bool,
    relu: bool,
    pzo: bool,
    scaled: &str,
) -> String {
    let mut parameters = vec![
        format!("v2::reg::variant::{}", capitalize(rounding)),
        format!(
            "v2::reg::variant::{}",
            if satfinite {
                "SatFinite"
            } else {
                "NoSatFinite"
            }
        ),
        format!("v2::reg::variant::{}", if relu { "Relu" } else { "NoRelu" }),
    ];
    if !scaled.is_empty() {
        let scale_marker = match scaled {
            SCALED_UE8M0_N1 => "ScaledUe8m0N1",
            SCALED_UE8M0_N2 => "ScaledUe8m0N2",
            other => unreachable!("packed cvt scale marker for {other}"),
        };
        parameters.push(format!("v2::reg::variant::{scale_marker}"));
    } else if pzo {
        parameters.push("v2::reg::variant::NoScale".to_owned());
    }
    if pzo {
        parameters.push("v2::reg::variant::Pzo".to_owned());
    }
    format!("v2::reg::variant::PackedMode<{}>", parameters.join(", "))
}

/// Destination/source pairs with packed engine operand signatures.
const PACKED_TYPE_PAIRS: &[(&str, &str)] = &[
    ("f16x2", "f32"),
    ("bf16x2", "f32"),
    ("e4m3x2", "f32"),
    ("e4m3x2", "f16x2"),
    ("e4m3x2", "bf16x2"),
    ("e5m2x2", "f32"),
    ("e5m2x2", "f16x2"),
    ("e5m2x2", "bf16x2"),
    ("e2m1x2", "f32"),
    ("e2m1x2", "f16x2"),
    ("e2m1x2", "bf16x2"),
    ("e2m3x2", "f32"),
    ("e2m3x2", "f16x2"),
    ("e2m3x2", "bf16x2"),
    ("e3m2x2", "f32"),
    ("e3m2x2", "f16x2"),
    ("e3m2x2", "bf16x2"),
    ("e2m1x4", "f32"),
    ("e4m3x4", "f32"),
    ("e5m2x4", "f32"),
    ("e2m3x4", "f32"),
    ("e3m2x4", "f32"),
    ("f16x2", "e4m3x2"),
    ("f16x2", "e5m2x2"),
    ("f16x2", "e2m1x2"),
    ("f16x2", "e2m3x2"),
    ("f16x2", "e3m2x2"),
    ("f16x2", "ue5m3x2"),
    ("bf16x2", "e4m3x2"),
    ("bf16x2", "e5m2x2"),
    ("bf16x2", "e2m1x2"),
    ("bf16x2", "e2m3x2"),
    ("bf16x2", "e3m2x2"),
    ("bf16x2", "ue5m3x2"),
    ("ue8m0x2", "f32"),
    ("ue8m0x2", "bf16x2"),
    ("bf16x2", "ue8m0x2"),
    ("s2f6x2", "f32"),
    ("s2f6x2", "bf16x2"),
    ("bf16x2", "s2f6x2"),
    ("ue5m3x2", "f32"),
    ("ue5m3x2", "f16x2"),
    ("ue5m3x2", "bf16x2"),
];

/// Map a decoded type pair and its operand modifiers to the engine signature.
pub(super) fn lookup_packed_cvt_variant(m: &Modifiers) -> Option<PackedCvtVariant> {
    if !PACKED_TYPE_PAIRS.contains(&(m.destination, m.source)) {
        return None;
    }
    // Scalar F32 sources supply each packed component separately. Other
    // sources already contain a pair. PTX appends rbits, then scale.
    let primary_count = if m.source != "f32" {
        1
    } else if m.destination.ends_with("x4") {
        4
    } else {
        2
    };
    let mut operand_dtypes = vec![carrier(m.source); primary_count];
    if m.rounding == "rs" {
        operand_dtypes.push(RBITS_CARRIER);
    }
    if !m.scaled.is_empty() {
        operand_dtypes.push(scale_carrier(m.scaled));
    }
    // Preserve the existing unqualified half RN artifact specialization.
    let mode = if ["f16x2", "bf16x2"].contains(&m.destination)
        && m.source == "f32"
        && m.rounding == "rn"
        && !(m.relu || m.satfinite || m.pzo)
    {
        "v2::reg::variant::Rn".to_owned()
    } else {
        packed_mode_marker(m.rounding, m.satfinite, m.relu, m.pzo, m.scaled)
    };
    Some(PackedCvtVariant {
        result_dtype: carrier(m.destination),
        operand_dtypes,
        specialization: format!(
            "v2::reg::variant::Cvt<{}, {}, {mode}>",
            marker(m.source),
            marker(m.destination)
        ),
    })
}
