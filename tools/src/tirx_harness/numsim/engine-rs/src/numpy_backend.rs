//! Synchronous NumPy payload execution and explicit low-precision codecs.
//!
//! The engine owns all operation inputs and outputs as Rust values. With the
//! `python` feature enabled, each backend call attaches to Python only for the
//! duration of one eager NumPy operation. No GIL token, borrowed array, or
//! Python-owned reference is stored in a backend or operation value.

/// Encode one `f32` as IEEE 754 binary16 using round-to-nearest, ties-to-even.
pub fn f32_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        let payload = ((fraction >> 13) as u16) | 0x0200;
        return sign | 0x7c00 | payload;
    }

    // Every binary32 subnormal is smaller than half of the least binary16
    // subnormal, so it rounds to signed zero.
    if exponent == 0 {
        return sign;
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = fraction | 0x0080_0000;
        let rounded = round_shift_right_even(significand, (14 - half_exponent) as u32);
        return sign | rounded as u16;
    }

    let rounded_fraction = round_shift_right_even(fraction, 13);
    let encoded = ((half_exponent as u32) << 10) + rounded_fraction;
    if encoded >= 0x7c00 {
        sign | 0x7c00
    } else {
        sign | encoded as u16
    }
}

/// Decode one IEEE 754 binary16 payload into `f32`.
pub fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;

    let decoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading_zeros = fraction.leading_zeros() - (u32::BITS - 10);
            let normalized_fraction = (fraction << (leading_zeros + 1)) & 0x03ff;
            let decoded_exponent = (127 - 15 - leading_zeros) << 23;
            sign | decoded_exponent | (normalized_fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(decoded)
}

/// Recover the exact IEEE 754 binary16 storage payload from a decoded value.
///
/// NumSim represents logical `float16` scalars as `f32`.  Every value loaded
/// from binary16 storage is therefore exactly representable, including the
/// payload of a signalling NaN.  The ordinary narrowing codec intentionally
/// quiets NaNs, which is correct for numerical conversion but not for a PTX
/// `.b16` instruction that treats the register as uninterpreted bits.  This
/// inverse keeps that distinction explicit.
pub fn decoded_fp16_to_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && fraction != 0 {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7c00 | ((fraction >> 13) as u16);
    }
    f32_to_fp16_bits(value)
}

/// Encode one `f32` as bfloat16 using round-to-nearest, ties-to-even.
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && fraction != 0 {
        // Preserve sign and the high payload while ensuring the narrowed value
        // remains a quiet NaN rather than becoming infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let tie = (bits >> 16) & 1;
    bits.wrapping_add(0x7fff + tie).wrapping_shr(16) as u16
}

/// Decode one bfloat16 payload into `f32`.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Recover the exact bfloat16 storage payload from its decoded `f32` value.
pub fn decoded_bf16_to_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Round one IEEE 754 binary32 value to TensorFloat-32 precision.
///
/// TF32 keeps the binary32 sign and exponent with a 10-bit fraction. Finite
/// values use round-to-nearest, ties-to-even; infinities and NaNs retain their
/// original payload because they do not participate in finite significand
/// rounding.
pub fn f32_to_tf32(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return value;
    }
    let tie = (bits >> 13) & 1;
    f32::from_bits(bits.wrapping_add(0x0fff + tie) & 0xffff_e000)
}

/// Decode one finite E4M3FN payload into `f32`.
pub fn float8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exponent = ((bits >> 3) & 0x0f) as i32;
    let mantissa = (bits & 0x07) as f32;
    if exponent == 0x0f && bits & 0x07 == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        (mantissa / 8.0_f32) * 2.0_f32.powi(-6)
    } else {
        (1.0_f32 + mantissa / 8.0_f32) * 2.0_f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// Encode `f32` with round-to-nearest-even and finite saturation to E4M3FN.
pub fn f32_to_float8_e4m3fn_bits(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    let mut best = 0_u8;
    let mut best_distance = f32::INFINITY;
    for code in 0_u8..=0x7e {
        let candidate = float8_e4m3fn_bits_to_f32(code);
        let distance = (candidate - magnitude).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

/// How a narrow float's greatest exponent encodes non-finite values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarrowFloatSpecials {
    /// IEEE shape: the all-ones exponent is an infinity or, with a non-zero
    /// significand, a NaN. `.e5m2`.
    Ieee,
    /// Finite-NaN shape: only the all-ones *magnitude* (signed E4M3) or code
    /// (unsigned UE5M3) is a NaN, and the format has no infinities.
    NanOnly,
    /// Every encoding is finite. `.e2m1`, `.e2m3`, and `.e3m2`.
    Finite,
}

/// Field layout of one packed PTX narrow float.
///
/// This is a configuration value, not a behaviour switch: every field is a
/// static property of the PTX type named by a `cvt` specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NarrowFloatFormat {
    /// Width of the encoded value, including a sign bit when present.
    pub width_bits: u32,
    /// Width of the value's field in a packed register.
    ///
    /// Six-bit `.e2m3` and `.e3m2` values occupy byte fields whose two most
    /// significant bits are padding; the other modeled formats are packed
    /// without padding.
    pub storage_bits: u32,
    /// Whether the encoding owns a sign bit above the exponent.
    pub signed: bool,
    /// Stored significand width.
    pub mantissa_bits: u32,
    /// Exponent bias.
    pub exponent_bias: i32,
    /// Greatest-magnitude finite encoding, without the sign bit.
    pub max_finite_code: u8,
    /// The encoding every NaN *input* converts to, always without a sign.
    ///
    /// A format with no NaN encoding names its greatest finite code here. For
    /// the pre-PTX-9.4 formats, this conversion behavior is measured on an
    /// NVIDIA B200 (driver 595.58.03, CUDA 13.2).
    pub nan_code: u8,
    /// Which encodings are non-finite.
    pub specials: NarrowFloatSpecials,
}

impl NarrowFloatFormat {
    /// Sign bit of one packed element.
    pub(crate) const fn sign_mask(self) -> u8 {
        if self.signed {
            1 << (self.width_bits - 1)
        } else {
            0
        }
    }

    const fn exponent_bits(self) -> u32 {
        self.width_bits - self.mantissa_bits - self.signed as u32
    }
}

/// `.e4m3` element format: bias 7, no infinities, `0x7f` is the only NaN.
pub const FLOAT8_E4M3: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 8,
    storage_bits: 8,
    signed: true,
    mantissa_bits: 3,
    exponent_bias: 7,
    max_finite_code: 0x7e,
    nan_code: 0x7f,
    specials: NarrowFloatSpecials::NanOnly,
};

/// `.e5m2` element format: bias 15, IEEE-shaped infinities and NaNs.
pub const FLOAT8_E5M2: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 8,
    storage_bits: 8,
    signed: true,
    mantissa_bits: 2,
    exponent_bias: 15,
    max_finite_code: 0x7b,
    nan_code: 0x7f,
    specials: NarrowFloatSpecials::Ieee,
};

/// `.e2m1` element format: four bits, bias 1, every encoding finite.
///
/// With no NaN encoding to reach, a NaN input lands on the greatest finite
/// code `0x7` (`+6.0`), which is the measured hardware result.
pub const FLOAT4_E2M1: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 4,
    storage_bits: 4,
    signed: true,
    mantissa_bits: 1,
    exponent_bias: 1,
    max_finite_code: 0x07,
    nan_code: 0x07,
    specials: NarrowFloatSpecials::Finite,
};

/// `.e2m3` element format: six encoded bits in a padded byte field.
pub const FLOAT6_E2M3: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 6,
    storage_bits: 8,
    signed: true,
    mantissa_bits: 3,
    exponent_bias: 1,
    max_finite_code: 0x1f,
    nan_code: 0x1f,
    specials: NarrowFloatSpecials::Finite,
};

/// `.e3m2` element format: six encoded bits in a padded byte field.
pub const FLOAT6_E3M2: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 6,
    storage_bits: 8,
    signed: true,
    mantissa_bits: 2,
    exponent_bias: 3,
    max_finite_code: 0x1f,
    nan_code: 0x1f,
    specials: NarrowFloatSpecials::Finite,
};

/// `.ue5m3` element format: unsigned E5M3 with one canonical NaN.
pub const FLOAT8_UE5M3: NarrowFloatFormat = NarrowFloatFormat {
    width_bits: 8,
    storage_bits: 8,
    signed: false,
    mantissa_bits: 3,
    exponent_bias: 15,
    max_finite_code: 0xfe,
    nan_code: 0xff,
    specials: NarrowFloatSpecials::NanOnly,
};

/// Encode one `f32` with PTX `cvt.rn.satfinite` semantics for `format`.
///
/// Round-to-nearest-even on the reduced significand; every NaN becomes
/// [`NarrowFloatFormat::nan_code`]; infinities and finite overflow clamp to the
/// greatest finite magnitude of the input's sign; binary32 subnormals are
/// below half of the least narrow-float subnormal and become a signed zero.
pub fn f32_to_narrow_float_bits_rn_satfinite(value: f32, format: NarrowFloatFormat) -> u8 {
    let (sign, exponent, fraction) = split_binary32(value, format);

    if exponent == 0xff {
        if fraction != 0 {
            return format.nan_code;
        }
        return sign | format.max_finite_code;
    }
    if exponent == 0 {
        return sign;
    }

    let target_exponent = exponent - 127 + format.exponent_bias;
    if target_exponent <= 0 {
        // Subnormal in the target format: one ulp is 2^(1 - bias - mantissa).
        let shift = 24 - format.mantissa_bits as i32 - target_exponent;
        if shift >= 25 {
            return sign;
        }
        let code = round_shift_right_even(fraction | 0x0080_0000, shift as u32);
        return sign | code as u8;
    }

    let code = ((target_exponent as u32) << format.mantissa_bits)
        + round_shift_right_even(fraction, 23 - format.mantissa_bits);
    sign | code.min(u32::from(format.max_finite_code)) as u8
}

/// Encode one `f32` with PTX `cvt.rs.satfinite` stochastic-rounding semantics.
///
/// `random` is the element's sixteen stochastic-rounding bits. They are added
/// at the top of the discarded significand field and the sum is truncated
/// toward zero, so the conversion is a deterministic function of the pair.
/// The significand bits below the random field cannot carry on their own —
/// their greatest value is one short of the field — so truncating them and
/// keeping them are the same function. Non-finite and overflowing inputs are
/// unaffected by `random` and follow the `.satfinite` rules of the `.rn` form.
pub fn f32_to_narrow_float_bits_rs(value: f32, random: u16, format: NarrowFloatFormat) -> u8 {
    let (sign, exponent, fraction) = split_binary32(value, format);

    if exponent == 0xff {
        if fraction != 0 {
            return format.nan_code;
        }
        return sign | format.max_finite_code;
    }
    if exponent == 0 {
        return sign;
    }

    let random = u32::from(random);
    let target_exponent = exponent - 127 + format.exponent_bias;
    if target_exponent <= 0 {
        let shift = 24 - format.mantissa_bits as i32 - target_exponent;
        // Beyond this the random field alone cannot reach the kept part, and
        // neither can it plus the whole significand.
        if shift >= 40 {
            return sign;
        }
        let significand = u64::from(fraction | 0x0080_0000);
        let biased = significand + (u64::from(random) << (shift - 16));
        return sign | (biased >> shift) as u8;
    }

    let shift = 23 - format.mantissa_bits;
    let code = ((target_exponent as u32) << format.mantissa_bits)
        + ((fraction + (random << (shift - 16))) >> shift);
    sign | code.min(u32::from(format.max_finite_code)) as u8
}

/// Split one `f32` into the packed sign of `format` plus exponent and fraction.
fn split_binary32(value: f32, format: NarrowFloatFormat) -> (u8, i32, u32) {
    let bits = value.to_bits();
    let sign = if bits & 0x8000_0000 != 0 {
        format.sign_mask()
    } else {
        0
    };
    (sign, ((bits >> 23) & 0xff) as i32, bits & 0x007f_ffff)
}

/// Decode one packed narrow-float payload into `f32`, or `None` for a NaN.
///
/// Every finite value of these formats is exactly representable in binary16,
/// bfloat16, and binary32, so the caller re-encodes without a second rounding
/// step.
pub fn narrow_float_bits_to_f32_checked(bits: u8, format: NarrowFloatFormat) -> Option<f32> {
    let mantissa_bits = format.mantissa_bits;
    let exponent_mask = (1_u8 << format.exponent_bits()) - 1;
    let fraction_mask = (1_u8 << mantissa_bits) - 1;
    let sign = (u32::from(bits & format.sign_mask())) << (32 - format.width_bits);
    let exponent = (bits >> mantissa_bits) & exponent_mask;
    let fraction = bits & fraction_mask;

    if exponent == exponent_mask {
        match format.specials {
            NarrowFloatSpecials::Ieee => {
                if fraction == 0 {
                    return Some(f32::from_bits(sign | 0x7f80_0000));
                }
                return None;
            }
            NarrowFloatSpecials::NanOnly if fraction == fraction_mask => return None,
            _ => {}
        }
    }
    if exponent == 0 {
        if fraction == 0 {
            return Some(f32::from_bits(sign));
        }
        let leading = u8::BITS - fraction.leading_zeros();
        let shift = mantissa_bits + 1 - leading;
        let normalized = (u32::from(fraction) << shift) & u32::from(fraction_mask);
        let biased = (127 - format.exponent_bias + 1 - shift as i32) as u32;
        return Some(f32::from_bits(
            sign | (biased << 23) | (normalized << (23 - mantissa_bits)),
        ));
    }
    let biased = (i32::from(exponent) + 127 - format.exponent_bias) as u32;
    Some(f32::from_bits(
        sign | (biased << 23) | (u32::from(fraction) << (23 - mantissa_bits)),
    ))
}

/// Decode one unsigned E8M0 exponent payload into `f32`.
pub fn float8_e8m0fnu_bits_to_f32(bits: u8) -> f32 {
    if bits == 0xff {
        f32::NAN
    } else {
        2.0_f32.powi(bits as i32 - 127)
    }
}

/// Decode one E2M1 nibble into `f32`.
pub fn float4_e2m1fn_bits_to_f32(bits: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let bits = bits & 0x0f;
    let magnitude = VALUES[usize::from(bits & 0x07)];
    if bits & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Encode a positive scale as the nearest finite E8M0 power of two.
pub fn f32_to_float8_e8m0fnu_bits(value: f32) -> u8 {
    if value.is_nan() {
        return 0xff;
    }
    if value <= 0.0_f32 {
        return 0;
    }
    let exponent = value.log2().round_ties_even().clamp(-127.0, 127.0) as i32;
    (exponent + 127) as u8
}

/// Encode one `f32` exponent with PTX `cvt.{rz,rp}{.satfinite}.ue8m0` semantics.
///
/// `ROUND_UP` selects `.rp` over `.rz`; the result is the exponent of the
/// smallest power of two at or above (`.rp`) or the largest at or below
/// (`.rz`) the magnitude, biased by 127 and clamped into the encoding.
/// `SATURATE` is `.satfinite`, which lowers the ceiling from the NaN encoding
/// `0xff` to the greatest finite code `0xfe` — that is observable for an
/// infinity and, under `.rp`, for any finite magnitude above `2^127`. A NaN
/// input is `0xff` either way.
///
/// The exponent is read from the bit pattern rather than through `log2`, which
/// rounds a magnitude just below a power of two up to the exact power and then
/// reports an exponent one too high.
pub fn f32_to_float8_e8m0fnu_bits_rounded<const ROUND_UP: bool, const SATURATE: bool>(
    value: f32,
) -> u8 {
    let ceiling: i32 = if SATURATE { 0xfe } else { 0xff };
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        return if fraction != 0 { 0xff } else { ceiling as u8 };
    }
    if exponent == 0 {
        if fraction == 0 {
            return 0;
        }
        // Subnormal: the magnitude is `fraction * 2^-149`.
        let floor_log2 = (32 - fraction.leading_zeros()) as i32 - 1 - 149;
        let inexact = i32::from(ROUND_UP && !fraction.is_power_of_two());
        return (floor_log2 + 127 + inexact).clamp(0, ceiling) as u8;
    }
    // Normal: `floor(log2(|v|)) + 127` is the biased exponent field itself.
    let inexact = i32::from(ROUND_UP && fraction != 0);
    (exponent + inexact).clamp(0, ceiling) as u8
}

fn round_shift_right_even(value: u32, shift: u32) -> u32 {
    debug_assert!((1..u32::BITS).contains(&shift));
    let quotient = value >> shift;
    let remainder_mask = (1_u32 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1_u32 << (shift - 1);
    quotient + u32::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0))
}

#[cfg(feature = "python")]
mod python {
    use std::error::Error;
    use std::fmt;

    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyModule};

    use crate::{ProfileKind, ProfileTimer};

    /// A dense row-major matrix owned entirely by Rust.
    #[derive(Clone, Debug, PartialEq)]
    pub struct F32Matrix {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    }

    impl F32Matrix {
        pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, NumpyBackendError> {
            let expected = rows
                .checked_mul(cols)
                .ok_or(NumpyBackendError::ShapeOverflow { rows, cols })?;
            if values.len() != expected {
                return Err(NumpyBackendError::LengthMismatch {
                    label: "matrix",
                    expected,
                    actual: values.len(),
                });
            }
            Ok(Self { rows, cols, values })
        }

        pub const fn rows(&self) -> usize {
            self.rows
        }

        pub const fn cols(&self) -> usize {
            self.cols
        }

        pub fn into_values(self) -> Vec<f32> {
            self.values
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum NumpyBackendError {
        ShapeOverflow {
            rows: usize,
            cols: usize,
        },
        LengthMismatch {
            label: &'static str,
            expected: usize,
            actual: usize,
        },
        InnerDimensionMismatch {
            a_cols: usize,
            b_cols: usize,
        },
        Python(String),
    }

    impl fmt::Display for NumpyBackendError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ShapeOverflow { rows, cols } => {
                    write!(f, "matrix shape [{rows}, {cols}] overflows usize")
                }
                Self::LengthMismatch {
                    label,
                    expected,
                    actual,
                } => write!(
                    f,
                    "{label} payload has {actual} values, expected {expected}"
                ),
                Self::InnerDimensionMismatch { a_cols, b_cols } => write!(
                    f,
                    "A @ B^T requires equal inner dimensions, got {a_cols} and {b_cols}"
                ),
                Self::Python(message) => write!(f, "NumPy operation failed: {message}"),
            }
        }
    }

    impl Error for NumpyBackendError {}

    /// Eager `A @ B^T`, returning only Rust-owned values after releasing Python.
    pub fn matmul_f32_abt(a: F32Matrix, b: F32Matrix) -> Result<F32Matrix, NumpyBackendError> {
        if a.cols != b.cols {
            return Err(NumpyBackendError::InnerDimensionMismatch {
                a_cols: a.cols,
                b_cols: b.cols,
            });
        }
        let _profile_timer = ProfileTimer::new(ProfileKind::NumpyTotal);
        let rows = a.rows;
        let cols = b.rows;
        let output_len = rows
            .checked_mul(cols)
            .ok_or(NumpyBackendError::ShapeOverflow { rows, cols })?;
        let output_values = with_numpy(|py, numpy| {
            let a = numpy_f32_matrix(py, numpy, &a)?;
            let b = numpy_f32_matrix(py, numpy, &b)?;
            let b_transpose = b.getattr("T")?;
            let result = {
                let _profile_timer = ProfileTimer::new(ProfileKind::NumpyMatmul);
                numpy.call_method1("matmul", (&a, &b_transpose))?
            };
            owned_rust_f32_values(py, numpy, &result, output_len)
        })?;
        F32Matrix::new(rows, cols, output_values)
    }

    fn with_numpy<T>(
        operation: impl for<'py> FnOnce(Python<'py>, &Bound<'py, PyModule>) -> PyResult<T>,
    ) -> Result<T, NumpyBackendError> {
        // Convert PyErr to a Rust-owned string before detaching from Python.
        // Thus even the error path retains no Python reference.
        Python::attach(|py| {
            let numpy = py.import("numpy").map_err(|error| error.to_string())?;
            operation(py, &numpy).map_err(|error| error.to_string())
        })
        .map_err(NumpyBackendError::Python)
    }

    fn numpy_f32_matrix<'py>(
        py: Python<'py>,
        numpy: &Bound<'py, PyModule>,
        matrix: &F32Matrix,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _profile_timer = ProfileTimer::new(ProfileKind::NumpyInput);
        let bytes = f32_values_to_le_bytes(&matrix.values);
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "<f4")?;
        let flat = numpy.call_method("frombuffer", (PyBytes::new(py, &bytes),), Some(&kwargs))?;
        flat.call_method1("reshape", ((matrix.rows, matrix.cols),))
    }

    fn owned_rust_f32_values(
        py: Python<'_>,
        numpy: &Bound<'_, PyModule>,
        value: &Bound<'_, PyAny>,
        expected: usize,
    ) -> PyResult<Vec<f32>> {
        let _profile_timer = ProfileTimer::new(ProfileKind::NumpyOutput);
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "<f4")?;
        let contiguous = numpy.call_method("ascontiguousarray", (value,), Some(&kwargs))?;
        let buffer = PyBuffer::<f32>::get(&contiguous)?;
        if buffer.item_count() != expected {
            return Err(PyValueError::new_err(format!(
                "NumPy returned {} values, expected {expected}",
                buffer.item_count()
            )));
        }
        buffer.to_vec(py)
    }

    fn f32_values_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(feature = "python")]
pub(crate) use python::{matmul_f32_abt, F32Matrix};

#[cfg(test)]
mod codec_tests {
    use super::*;

    #[test]
    fn fp16_known_values_and_rounding_edges() {
        assert_eq!(f32_to_fp16_bits(0.0), 0x0000);
        assert_eq!(f32_to_fp16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_fp16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_fp16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_fp16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_fp16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_fp16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_fp16_bits(2.0_f32.powi(-14)), 0x0400);

        let one = fp16_bits_to_f32(0x3c00);
        let next = fp16_bits_to_f32(0x3c01);
        let halfway = one + (next - one) * 0.5;
        assert_eq!(f32_to_fp16_bits(halfway), 0x3c00);
        let next_even = fp16_bits_to_f32(0x3c02);
        let odd_halfway = next + (next_even - next) * 0.5;
        assert_eq!(f32_to_fp16_bits(odd_halfway), 0x3c02);
    }

    #[test]
    fn every_non_nan_fp16_value_round_trips_exactly() {
        for bits in 0_u16..=u16::MAX {
            let exponent = bits & 0x7c00;
            let fraction = bits & 0x03ff;
            if exponent != 0x7c00 || fraction == 0 {
                assert_eq!(f32_to_fp16_bits(fp16_bits_to_f32(bits)), bits);
            }
        }

    }

    #[test]
    fn decoded_low_precision_payloads_round_trip_all_bit_patterns() {
        for bits in 0_u16..=u16::MAX {
            assert_eq!(decoded_fp16_to_bits(fp16_bits_to_f32(bits)), bits);
            assert_eq!(decoded_bf16_to_bits(bf16_bits_to_f32(bits)), bits);
        }
    }

    #[test]
    fn float8_codecs_cover_known_values_and_finite_saturation() {
        assert_eq!(float8_e4m3fn_bits_to_f32(0x00), 0.0);
        assert_eq!(float8_e4m3fn_bits_to_f32(0x38), 1.0);
        assert_eq!(float8_e4m3fn_bits_to_f32(0xb8), -1.0);
        assert!(float8_e4m3fn_bits_to_f32(0x7f).is_nan());
        assert_eq!(f32_to_float8_e4m3fn_bits(1.0), 0x38);
        assert_eq!(f32_to_float8_e4m3fn_bits(-1.0), 0xb8);
        assert_eq!(f32_to_float8_e4m3fn_bits(f32::INFINITY), 0x7e);

        assert_eq!(float8_e8m0fnu_bits_to_f32(127), 1.0);
        assert_eq!(float8_e8m0fnu_bits_to_f32(128), 2.0);
        assert!(float8_e8m0fnu_bits_to_f32(0xff).is_nan());
        assert_eq!(f32_to_float8_e8m0fnu_bits(1.0), 127);
        assert_eq!(f32_to_float8_e8m0fnu_bits(2.0), 128);

        assert_eq!(float4_e2m1fn_bits_to_f32(0x0), 0.0);
        assert_eq!(float4_e2m1fn_bits_to_f32(0x7), 6.0);
        assert_eq!(float4_e2m1fn_bits_to_f32(0xf), -6.0);
    }

    /// Directed cases from an NVIDIA B200 (driver 595.58.03, CUDA 13.2); the
    /// exhaustive tables live in
    /// `tests/numsim/microtests/cases/ptx_cvt_fp8_goldens.py`.
    #[test]
    fn packed_float8_rn_satfinite_encode_matches_hardware() {
        // Signed zeros and every binary32 subnormal round to a signed zero.
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(0.0, FLOAT8_E4M3),
            0x00
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(-0.0, FLOAT8_E4M3),
            0x80
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::from_bits(0x0000_0001), FLOAT8_E5M2),
            0x00
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::from_bits(0x8000_0001), FLOAT8_E5M2),
            0x80
        );

        // Round-to-nearest-even on the reduced significand.
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(1.0, FLOAT8_E4M3),
            0x38
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(1.0625, FLOAT8_E4M3),
            0x38
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(1.1875, FLOAT8_E4M3),
            0x3a
        );

        // Subnormals are preserved, and half of the least one ties onto zero.
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(2.0_f32.powi(-9), FLOAT8_E4M3),
            0x01
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(2.0_f32.powi(-10), FLOAT8_E4M3),
            0x00
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(2.0_f32.powi(-16), FLOAT8_E5M2),
            0x01
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(2.0_f32.powi(-17), FLOAT8_E5M2),
            0x00
        );

        // .satfinite clamps overflow and both infinities onto the greatest
        // finite magnitude of the input's sign.
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(448.0, FLOAT8_E4M3),
            0x7e
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(464.0, FLOAT8_E4M3),
            0x7e
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(-512.0, FLOAT8_E4M3),
            0xfe
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::INFINITY, FLOAT8_E4M3),
            0x7e
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(57344.0, FLOAT8_E5M2),
            0x7b
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(61440.0, FLOAT8_E5M2),
            0x7b
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::NEG_INFINITY, FLOAT8_E5M2),
            0xfb
        );

        // Every NaN, of either sign, becomes the canonical positive NaN code.
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::NAN, FLOAT8_E4M3),
            0x7f
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(-f32::NAN, FLOAT8_E5M2),
            0x7f
        );
        assert_eq!(
            f32_to_narrow_float_bits_rn_satfinite(f32::from_bits(0x7f80_0001), FLOAT8_E4M3),
            0x7f
        );
    }

    #[test]
    fn packed_narrow_float_decode_round_trips_every_finite_encoding() {
        for format in [
            FLOAT8_E4M3,
            FLOAT8_E5M2,
            FLOAT4_E2M1,
            FLOAT6_E2M3,
            FLOAT6_E3M2,
            FLOAT8_UE5M3,
        ] {
            let last = (1_u16 << format.width_bits) - 1;
            for code in (0..=last).map(|code| code as u8) {
                match narrow_float_bits_to_f32_checked(code, format) {
                    None => {}
                    Some(value) if value.is_infinite() => {
                        assert_eq!(format.specials, NarrowFloatSpecials::Ieee);
                        assert_eq!(code & 0x7f, 0x7c);
                    }
                    Some(value) => {
                        assert_eq!(
                            f32_to_narrow_float_bits_rn_satfinite(value, format),
                            code,
                            "code {code:#04x} did not round-trip"
                        );
                    }
                }
            }
        }
        // Signed e4m3 has two NaNs, e5m2 six, UE5M3 one, and the finite-only
        // four- and six-bit formats have none.
        let nan_count = |format: NarrowFloatFormat| {
            (0..(1_u16 << format.width_bits))
                .filter(|code| narrow_float_bits_to_f32_checked(*code as u8, format).is_none())
                .count()
        };
        assert_eq!(
            (
                nan_count(FLOAT8_E4M3),
                nan_count(FLOAT8_E5M2),
                nan_count(FLOAT4_E2M1),
                nan_count(FLOAT6_E2M3),
                nan_count(FLOAT6_E3M2),
                nan_count(FLOAT8_UE5M3)
            ),
            (2, 6, 0, 0, 0, 1)
        );
        // Every `.e2m1` code decodes to its documented grid value.
        let grid: Vec<f32> = (0_u8..16)
            .map(|code| narrow_float_bits_to_f32_checked(code, FLOAT4_E2M1).unwrap())
            .collect();
        assert_eq!(
            grid,
            vec![
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );

        // Format-defined anchors independently pin the two padded FP6
        // encodings and unsigned UE5M3.  In particular, width and byte-field
        // stride are deliberately different facts for FP6.
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x01, FLOAT6_E2M3),
            Some(0.125)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x1f, FLOAT6_E2M3),
            Some(7.5)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x21, FLOAT6_E2M3),
            Some(-0.125)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x01, FLOAT6_E3M2),
            Some(0.0625)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x1f, FLOAT6_E3M2),
            Some(28.0)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x01, FLOAT8_UE5M3),
            Some(2.0_f32.powi(-17))
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0x78, FLOAT8_UE5M3),
            Some(1.0)
        );
        assert_eq!(
            narrow_float_bits_to_f32_checked(0xfe, FLOAT8_UE5M3),
            Some(114_688.0)
        );
        assert_eq!(narrow_float_bits_to_f32_checked(0xff, FLOAT8_UE5M3), None);
    }

    /// Directed cases from an NVIDIA B200 (driver 595.58.03, CUDA 13.2); the
    /// exhaustive tables live in
    /// `tests/numsim/microtests/cases/ptx_cvt_narrow_goldens.py`.
    #[test]
    fn e8m0_exponent_rounding_matches_hardware() {
        let rz = f32_to_float8_e8m0fnu_bits_rounded::<false, false>;
        let rp = f32_to_float8_e8m0fnu_bits_rounded::<true, false>;
        let rz_sat = f32_to_float8_e8m0fnu_bits_rounded::<false, true>;
        let rp_sat = f32_to_float8_e8m0fnu_bits_rounded::<true, true>;

        // Exact powers of two are unchanged by either rounding.
        assert_eq!((rz(1.0), rp(1.0)), (127, 127));
        assert_eq!((rz(2.0), rp(2.0)), (128, 128));
        // Inexact magnitudes floor under `.rz` and ceil under `.rp`, on the
        // magnitude, so the sign is irrelevant.
        assert_eq!((rz(1.5), rp(1.5)), (127, 128));
        assert_eq!((rz(-3.0), rp(-3.0)), (128, 129));
        assert_eq!((rz(0.75), rp(0.75)), (126, 127));
        // Signed zeros, and every magnitude at or below 2^-127, encode as 0.
        assert_eq!((rz(0.0), rp(-0.0)), (0, 0));
        assert_eq!(
            (
                rz(f32::from_bits(0x0040_0000)),
                rp(f32::from_bits(0x0040_0000))
            ),
            (0, 0)
        );
        // The largest binary32 subnormal is just under 2^-126: this is the case
        // an `f32::log2().floor()` encoder gets wrong, reporting 1 for `.rz`.
        let largest_subnormal = f32::from_bits(0x007f_ffff);
        assert_eq!((rz(largest_subnormal), rp(largest_subnormal)), (0, 1));
        // `.satfinite` lowers the ceiling from the NaN encoding to 0xfe, which
        // is observable for an infinity and for a `.rp` overflow.
        assert_eq!((rz(f32::INFINITY), rz_sat(f32::INFINITY)), (0xff, 0xfe));
        assert_eq!(
            (rp(f32::NEG_INFINITY), rp_sat(f32::NEG_INFINITY)),
            (0xff, 0xfe)
        );
        assert_eq!((rp(f32::MAX), rp_sat(f32::MAX)), (0xff, 0xfe));
        assert_eq!((rz(f32::MAX), rz_sat(f32::MAX)), (0xfe, 0xfe));
        // A NaN stays the NaN encoding under `.satfinite`.
        assert_eq!((rz(f32::NAN), rz_sat(-f32::NAN)), (0xff, 0xff));
    }

    /// Why `.scaled` may narrow through binary32 rather than binary64.
    ///
    /// Every product the form can reach is a narrow-float element — at most
    /// four significant bits — times an exact power of two in `2^-127 ..=
    /// 2^127`, so the product itself has at most four significant bits and a
    /// magnitude in `2^-144 ..= 1.75*2^143`. Binary32 subnormals reach down to
    /// `2^-149`, leaving five bits of significand at `2^-144`: every product
    /// within binary32's range is exact, while every binary32 overflow already
    /// exceeds bfloat16's range. Thus the binary32 intermediate cannot cause a
    /// double-rounding difference. This test pins that invariant over the whole
    /// reachable set.
    #[test]
    fn every_reachable_scaled_product_is_exact_or_already_overflows_bf16() {
        let greatest_finite_bf16 = f64::from(crate::bf16_bits_to_f32(0x7f7f));
        let mut checked = 0_u32;
        let mut infinite_elements = 0_u32;
        let mut smallest = f64::INFINITY;
        let mut largest = 0.0_f64;
        for format in [FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT8_UE5M3] {
            let last = (1_u16 << format.width_bits) - 1;
            for code in (0..=last).map(|code| code as u8) {
                let Some(element) = narrow_float_bits_to_f32_checked(code, format) else {
                    continue;
                };
                // The `0xff` scale byte is the E8M0 NaN and never reaches the
                // multiply, so the reachable scale codes are `0..=0xfe`.
                for scale_code in 0_u8..=0xfe {
                    let factor = float8_e8m0fnu_bits_to_f32(scale_code);
                    let exact = f64::from(element) * f64::from(factor);
                    let narrowed = element * factor;
                    checked += 1;
                    if !element.is_finite() {
                        // `.e5m2` has infinities, and scaling one by a finite
                        // positive power of two leaves it infinite in both
                        // widths. `.satfinite` is what clamps these.
                        infinite_elements += 1;
                        assert!(narrowed.is_infinite() && exact.is_infinite());
                        assert_eq!(narrowed.is_sign_negative(), exact.is_sign_negative());
                        continue;
                    }
                    let magnitude = exact.abs();
                    if magnitude > 0.0 {
                        smallest = smallest.min(magnitude);
                        largest = largest.max(magnitude);
                    }
                    if narrowed.is_finite() {
                        assert_eq!(
                            f64::from(narrowed),
                            exact,
                            "binary32 lost the product of {code:#04x} and scale {scale_code}"
                        );
                    } else {
                        // A binary32 overflow is only sound because the exact
                        // product also exceeds bfloat16, which rounds to the
                        // same infinity.
                        assert!(magnitude > greatest_finite_bf16);
                    }
                }
            }
        }
        assert_eq!(checked, 197_625);
        // Two `.e5m2` infinity encodings across 255 scale bytes.
        assert_eq!(infinite_elements, 2 * 255);
        // The least is UE5M3's smallest subnormal 2^-17 scaled by 2^-127.
        // The greatest is its greatest finite 1.75*2^16 scaled by 2^127;
        // binary32 overflows there, but so does the final bfloat16 result.
        assert_eq!(smallest, 2.0_f64.powi(-144));
        assert_eq!(largest, 1.75 * 2.0_f64.powi(143));
    }

    /// The same argument for `cvt.rn.bf16x2.ue8m0x2`, whose 255 finite decodes
    /// are exact powers of two from `2^-127` to `2^127`.
    #[test]
    fn every_e8m0_decode_is_exact_in_binary32() {
        for code in 0_u8..=0xfe {
            let value = float8_e8m0fnu_bits_to_f32(code);
            assert!(value.is_finite() && value > 0.0);
            assert_eq!(f64::from(value), 2.0_f64.powi(i32::from(code) - 127));
        }
        // The least one lands mid-subnormal in bfloat16 and still narrows
        // exactly, which is the case a naive encoder gets wrong.
        assert_eq!(f32_to_bf16_bits(float8_e8m0fnu_bits_to_f32(0)), 0x0040);
        assert_eq!(f32_to_bf16_bits(float8_e8m0fnu_bits_to_f32(127)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(float8_e8m0fnu_bits_to_f32(254)), 0x7f00);
    }

    #[test]
    fn stochastic_rounding_truncates_the_biased_significand() {
        // 1.0 plus m/65536 of an e4m3 ulp, with the element's sixteen random
        // bits: the pair carries exactly when it reaches one whole ulp.
        let residue = |m: u32| f32::from_bits(0x3f80_0000 | (m << 4));
        for (m, random, expected) in [
            (0, 0xffff, 0x38),
            (1, 0xfffe, 0x38),
            (1, 0xffff, 0x39),
            (0x8000, 0x7fff, 0x38),
            (0x8000, 0x8000, 0x39),
            (0xffff, 0x0001, 0x39),
            (0xffff, 0x0000, 0x38),
        ] {
            assert_eq!(
                f32_to_narrow_float_bits_rs(residue(m), random, FLOAT8_E4M3),
                expected,
                "m={m} random={random:#06x}"
            );
        }
        // Exactly representable values never move, whatever the random bits.
        for random in [0_u16, 1, 0x8000, 0xffff] {
            assert_eq!(f32_to_narrow_float_bits_rs(1.0, random, FLOAT8_E4M3), 0x38);
            assert_eq!(f32_to_narrow_float_bits_rs(-6.0, random, FLOAT4_E2M1), 0x0f);
            // Non-finite and overflowing inputs ignore the random bits.
            assert_eq!(
                f32_to_narrow_float_bits_rs(f32::NAN, random, FLOAT8_E5M2),
                0x7f
            );
            assert_eq!(
                f32_to_narrow_float_bits_rs(-f32::NAN, random, FLOAT4_E2M1),
                0x07
            );
            assert_eq!(
                f32_to_narrow_float_bits_rs(f32::NEG_INFINITY, random, FLOAT8_E4M3),
                0xfe
            );
            assert_eq!(f32_to_narrow_float_bits_rs(1e30, random, FLOAT4_E2M1), 0x07);
            assert_eq!(f32_to_narrow_float_bits_rs(-0.0, random, FLOAT8_E4M3), 0x80);
        }
        // The target-subnormal path uses the same rule: 0.75 is half an e2m1
        // ulp above 0.5, so it carries at exactly half the random range.
        assert_eq!(f32_to_narrow_float_bits_rs(0.75, 0x7fff, FLOAT4_E2M1), 0x01);
        assert_eq!(f32_to_narrow_float_bits_rs(0.75, 0x8000, FLOAT4_E2M1), 0x02);
    }

    #[test]
    fn fp16_nan_stays_nan_and_known_payloads_round_trip() {
        let encoded_nan = f32_to_fp16_bits(f32::from_bits(0xff80_0001));
        assert_eq!(encoded_nan & 0x8000, 0x8000);
        assert_eq!(encoded_nan & 0x7c00, 0x7c00);
        assert_ne!(encoded_nan & 0x03ff, 0);
        assert!(fp16_bits_to_f32(encoded_nan).is_nan());

        let source = vec![1.0, -2.0, 0.5];
        let encoded = source.iter().copied().map(f32_to_fp16_bits).collect::<Vec<_>>();
        let decoded = encoded.iter().copied().map(fp16_bits_to_f32).collect::<Vec<_>>();
        assert_eq!(encoded, vec![0x3c00, 0xc000, 0x3800]);
        assert_eq!(decoded, source);
    }

    #[test]
    fn bf16_known_values_and_rounding_edges() {
        assert_eq!(f32_to_bf16_bits(0.0), 0x0000);
        assert_eq!(f32_to_bf16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xff80);

        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn tf32_rounds_to_nearest_even_and_preserves_special_values() {
        assert_eq!(f32_to_tf32(0.0).to_bits(), 0x0000_0000);
        assert_eq!(f32_to_tf32(-0.0).to_bits(), 0x8000_0000);
        assert_eq!(f32_to_tf32(f32::INFINITY), f32::INFINITY);
        assert_eq!(f32_to_tf32(f32::NEG_INFINITY), f32::NEG_INFINITY);

        let one = 1.0_f32;
        let tf32_ulp = 2.0_f32.powi(-10);
        assert_eq!(f32_to_tf32(one + tf32_ulp / 2.0), one);
        assert_eq!(
            f32_to_tf32(one + tf32_ulp + tf32_ulp / 2.0),
            one + 2.0 * tf32_ulp
        );
        assert_eq!(f32_to_tf32(-(one + tf32_ulp / 2.0)), -one);

        let nan = f32::from_bits(0x7fc1_2345);
        assert_eq!(f32_to_tf32(nan).to_bits(), nan.to_bits());
    }

    #[test]
    fn bf16_nan_stays_nan_and_known_payloads_round_trip() {
        let encoded_nan = f32_to_bf16_bits(f32::from_bits(0x7f80_0001));
        assert_eq!(encoded_nan & 0x7f80, 0x7f80);
        assert_ne!(encoded_nan & 0x007f, 0);
        assert!(bf16_bits_to_f32(encoded_nan).is_nan());

        let source = vec![1.0, -2.0, 0.5];
        let encoded = source.iter().copied().map(f32_to_bf16_bits).collect::<Vec<_>>();
        let decoded = encoded.iter().copied().map(bf16_bits_to_f32).collect::<Vec<_>>();
        assert_eq!(encoded, vec![0x3f80, 0xc000, 0x3f00]);
        assert_eq!(decoded, source);
    }

    #[test]
    fn every_non_nan_bf16_value_round_trips_exactly() {
        for bits in 0_u16..=u16::MAX {
            let exponent = bits & 0x7f80;
            let fraction = bits & 0x007f;
            if exponent != 0x7f80 || fraction == 0 {
                assert_eq!(f32_to_bf16_bits(bf16_bits_to_f32(bits)), bits);
            }
        }

    }
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use super::python::*;

    fn matrix(rows: usize, cols: usize, values: &[f32]) -> F32Matrix {
        F32Matrix::new(rows, cols, values.to_vec()).unwrap()
    }

    #[test]
    fn matmul_uses_b_transpose_and_returns_rust_owned_values() {
        let a = matrix(2, 3, &[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]);
        let b = matrix(
            4,
            3,
            &[2.0, 0.0, 1.0, 1.0, 1.0, 1.0, -2.0, 3.0, 0.5, 0.0, -1.0, 2.0],
        );
        let expected = vec![5.0, 6.0, 5.5, 4.0, 2.0, 3.5, 5.5, 7.5];

        let output = matmul_f32_abt(a, b).unwrap();
        assert_eq!((output.rows(), output.cols()), (2, 4));
        assert_eq!(output.into_values(), expected);
    }

    #[test]
    fn matrices_and_results_hold_no_python_lifetime() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        fn assert_send_static_future<T: std::future::Future + Send + 'static>(_: T) {}
        assert_send_sync_static::<F32Matrix>();

        assert_send_static_future(async move {
            let input = matrix(1, 3, &[1.0, 2.0, 4.0]);
            let output = matmul_f32_abt(input, matrix(1, 3, &[1.0; 3])).unwrap();
            std::future::ready(()).await;
            output
        });
        let output = std::thread::spawn(|| {
            let input = matrix(1, 3, &[1.0, 2.0, 4.0]);
            matmul_f32_abt(input, matrix(1, 3, &[1.0; 3])).unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(output.into_values(), vec![7.0]);
    }

    #[test]
    fn shape_contracts_fail_before_numpy_execution() {
        assert_eq!(
            F32Matrix::new(2, 3, vec![0.0; 5]).unwrap_err(),
            NumpyBackendError::LengthMismatch {
                label: "matrix",
                expected: 6,
                actual: 5,
            }
        );
        let a = matrix(1, 2, &[1.0, 2.0]);
        let b = matrix(1, 3, &[1.0, 2.0, 3.0]);
        assert_eq!(
            matmul_f32_abt(a, b).unwrap_err(),
            NumpyBackendError::InnerDimensionMismatch {
                a_cols: 2,
                b_cols: 3,
            }
        );
    }

}
