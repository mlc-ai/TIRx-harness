use crate::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_float8_e4m3fn_bits, EngineError, NarrowFloatFormat,
};
use std::cmp::Ordering;

pub type F32x4 = [f32; 4];
pub type U64x2 = [u64; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32RoundingMode {
    Nearest,
    Down,
    Up,
    Zero,
}

pub trait RuntimeScalar: Copy + Clone {
    const BYTE_LEN: usize;
    const HIGH_FORMAT: u8 = 0;

    fn high_value(self) -> f64 {
        unreachable!("native scalar has no high precision value")
    }

    fn with_high_value(self, _value: f64) -> Self {
        unreachable!("native scalar has no high precision value")
    }

    fn zero() -> Self;
    fn decode_le(bytes: &[u8]) -> Result<Self, EngineError>;
    fn encode_le_into(self, target: &mut [u8]);

    fn encode_le(self) -> Vec<u8> {
        let mut bytes = vec![0; Self::BYTE_LEN];
        self.encode_le_into(&mut bytes);
        bytes
    }
}

macro_rules! impl_runtime_scalar {
    ($rust_type:ty, $byte_len:expr) => {
        impl RuntimeScalar for $rust_type {
            const BYTE_LEN: usize = $byte_len;

            fn zero() -> Self {
                0 as $rust_type
            }

            fn decode_le(bytes: &[u8]) -> Result<Self, EngineError> {
                let encoded: [u8; $byte_len] = bytes.try_into().map_err(|_| {
                    EngineError::message(format!(
                        "expected {} bytes for {}",
                        $byte_len,
                        stringify!($rust_type)
                    ))
                })?;
                Ok(<$rust_type>::from_le_bytes(encoded))
            }

            fn encode_le_into(self, target: &mut [u8]) {
                target.copy_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl_runtime_scalar!(i8, 1);
impl_runtime_scalar!(i16, 2);
impl_runtime_scalar!(i32, 4);
impl_runtime_scalar!(i64, 8);
impl_runtime_scalar!(u8, 1);
impl_runtime_scalar!(u16, 2);
impl_runtime_scalar!(u32, 4);
impl_runtime_scalar!(u64, 8);
impl_runtime_scalar!(f32, 4);
impl_runtime_scalar!(f64, 8);

impl RuntimeScalar for F32x4 {
    const BYTE_LEN: usize = 16;

    fn zero() -> Self {
        [0.0_f32; 4]
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(EngineError::message(format!(
                "expected {} bytes for F32x4, got {}",
                Self::BYTE_LEN,
                bytes.len(),
            )));
        }
        let mut values = [0.0_f32; 4];
        for (index, value) in values.iter_mut().enumerate() {
            let start = index * 4;
            *value = f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ]);
        }
        Ok(values)
    }

    fn encode_le_into(self, target: &mut [u8]) {
        for (index, value) in self.into_iter().enumerate() {
            let start = index * 4;
            target[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl RuntimeScalar for U64x2 {
    const BYTE_LEN: usize = 16;

    fn zero() -> Self {
        [0_u64; 2]
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(EngineError::message(format!(
                "expected {} bytes for U64x2, got {}",
                Self::BYTE_LEN,
                bytes.len(),
            )));
        }
        let mut values = [0_u64; 2];
        for (index, value) in values.iter_mut().enumerate() {
            let start = index * 8;
            *value = u64::from_le_bytes(
                bytes[start..start + 8]
                    .try_into()
                    .expect("U64x2 component width was checked"),
            );
        }
        Ok(values)
    }

    fn encode_le_into(self, target: &mut [u8]) {
        for (index, value) in self.into_iter().enumerate() {
            let start = index * 8;
            target[start..start + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl RuntimeScalar for bool {
    const BYTE_LEN: usize = 1;

    fn zero() -> Self {
        false
    }

    fn decode_le(bytes: &[u8]) -> Result<Self, EngineError> {
        match bytes {
            [0] => Ok(false),
            [1] => Ok(true),
            [value] => Err(EngineError::message(format!(
                "invalid physical bool byte {value}"
            ))),
            _ => Err(EngineError::message("expected one byte for bool")),
        }
    }

    fn encode_le_into(self, target: &mut [u8]) {
        target[0] = u8::from(self);
    }
}

#[inline]
pub fn floor_div_i64(lhs: i64, rhs: i64) -> Result<i64, EngineError> {
    let quotient = lhs
        .checked_div(rhs)
        .ok_or_else(|| EngineError::message(format!("invalid floor division: {lhs} // {rhs}")))?;
    let remainder = lhs
        .checked_rem(rhs)
        .ok_or_else(|| EngineError::message(format!("invalid floor remainder: {lhs} % {rhs}")))?;
    Ok(if remainder != 0 && ((remainder < 0) != (rhs < 0)) {
        quotient - 1
    } else {
        quotient
    })
}

#[inline]
pub fn floor_mod_i64(lhs: i64, rhs: i64) -> Result<i64, EngineError> {
    let quotient = floor_div_i64(lhs, rhs)?;
    Ok(lhs - quotient * rhs)
}

pub fn ptx_fns_b32(mask: u32, base: u32, offset: i32) -> u32 {
    debug_assert!(base < 32);
    if offset == 0 {
        return if mask & (1_u32 << base) != 0 {
            base
        } else {
            u32::MAX
        };
    }
    let mut position = base as i32;
    let mut remaining = offset.unsigned_abs() - 1;
    let increment = if offset > 0 { 1_i32 } else { -1_i32 };
    while (0..32).contains(&position) {
        if mask & (1_u32 << position) != 0 {
            if remaining == 0 {
                return position as u32;
            }
            remaining -= 1;
        }
        position += increment;
    }
    u32::MAX
}

pub(crate) fn flush_subnormal_f32(value: f32) -> f32 {
    if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}

pub(crate) const fn is_subnormal_f16_bits(bits: u16) -> bool {
    bits & 0x7c00 == 0 && bits & 0x03ff != 0
}

pub(crate) const fn flush_subnormal_f16_bits(bits: u16) -> u16 {
    if is_subnormal_f16_bits(bits) {
        bits & 0x8000
    } else {
        bits
    }
}

fn cuda_canonical_nan_f32() -> f32 {
    f32::from_bits(0x7fff_ffff)
}

pub fn cuda_canonicalize_nan_f32(value: f32) -> f32 {
    if value.is_nan() {
        cuda_canonical_nan_f32()
    } else {
        value
    }
}

pub fn cuda_f32_to_fp16_bits(value: f32) -> u16 {
    if value.is_nan() {
        0x7fff
    } else {
        crate::f32_to_fp16_bits(value)
    }
}

pub fn cuda_fp16_bits_to_f32(bits: u16) -> f32 {
    cuda_canonicalize_nan_f32(crate::fp16_bits_to_f32(bits))
}

pub fn cuda_f32_add(lhs: f32, rhs: f32) -> f32 {
    let result = lhs + rhs;
    if result.is_nan() {
        cuda_canonical_nan_f32()
    } else {
        result
    }
}

fn cuda_round_fp16(value: f32) -> f32 {
    cuda_fp16_bits_to_f32(cuda_f32_to_fp16_bits(value))
}

fn cuda_round_bf16(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(cuda_canonicalize_nan_f32(value)))
}

pub fn cuda_reduce_fp16_add(lhs: f32, rhs: f32) -> f32 {
    cuda_round_fp16(cuda_f32_add(cuda_round_fp16(lhs), cuda_round_fp16(rhs)))
}

pub fn cuda_reduce_fp16_max(lhs: f32, rhs: f32) -> f32 {
    let lhs = cuda_round_fp16(lhs);
    let rhs = cuda_round_fp16(rhs);
    cuda_round_fp16(if lhs > rhs { lhs } else { rhs })
}

pub fn cuda_reduce_fp16_min(lhs: f32, rhs: f32) -> f32 {
    let lhs = cuda_round_fp16(lhs);
    let rhs = cuda_round_fp16(rhs);
    cuda_round_fp16(if lhs < rhs { lhs } else { rhs })
}

pub fn cuda_reduce_bf16_add(lhs: f32, rhs: f32) -> f32 {
    cuda_round_bf16(cuda_f32_add(cuda_round_bf16(lhs), cuda_round_bf16(rhs)))
}

pub fn cuda_reduce_bf16_max(lhs: f32, rhs: f32) -> f32 {
    let lhs = cuda_round_bf16(lhs);
    let rhs = cuda_round_bf16(rhs);
    cuda_round_bf16(if lhs > rhs { lhs } else { rhs })
}

pub fn cuda_reduce_bf16_min(lhs: f32, rhs: f32) -> f32 {
    let lhs = cuda_round_bf16(lhs);
    let rhs = cuda_round_bf16(rhs);
    cuda_round_bf16(if lhs < rhs { lhs } else { rhs })
}

pub fn cuda_f32_max(lhs: f32, rhs: f32) -> f32 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => cuda_canonical_nan_f32(),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_positive() || rhs.is_sign_positive() {
                0.0
            } else {
                -0.0
            }
        }
        (false, false) if lhs > rhs => lhs,
        (false, false) => rhs,
    }
}

pub fn cuda_f32_min(lhs: f32, rhs: f32) -> f32 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => cuda_canonical_nan_f32(),
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_negative() || rhs.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        }
        (false, false) if lhs < rhs => lhs,
        (false, false) => rhs,
    }
}

fn quiet_f64_nan(value: f64) -> f64 {
    f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000)
}

pub fn cuda_f64_add(lhs: f64, rhs: f64) -> f64 {
    if rhs.is_nan() {
        quiet_f64_nan(rhs)
    } else if lhs.is_nan() {
        quiet_f64_nan(lhs)
    } else if lhs.is_infinite()
        && rhs.is_infinite()
        && lhs.is_sign_negative() != rhs.is_sign_negative()
    {
        f64::from_bits(0xfff8_0000_0000_0000)
    } else {
        lhs + rhs
    }
}

pub fn cuda_f64_max(lhs: f64, rhs: f64) -> f64 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => rhs,
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_positive() || rhs.is_sign_positive() {
                0.0
            } else {
                -0.0
            }
        }
        (false, false) if lhs > rhs => lhs,
        (false, false) => rhs,
    }
}

pub fn cuda_f64_min(lhs: f64, rhs: f64) -> f64 {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => rhs,
        (true, false) => rhs,
        (false, true) => lhs,
        (false, false) if lhs == 0.0 && rhs == 0.0 => {
            if lhs.is_sign_negative() || rhs.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        }
        (false, false) if lhs < rhs => lhs,
        (false, false) => rhs,
    }
}

pub fn ptx_exp2_approx_f32(value: f32) -> f32 {
    // Use the software f64 implementation as a target-independent canonical
    // representative, then round once to binary32. This deliberately does not
    // replay any GPU architecture's approximation polynomial.
    libm::exp2(value as f64) as f32
}

pub fn ptx_exp2_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(ptx_exp2_approx_f32(value))
}

pub fn ptx_sin_approx_f32(value: f32, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    // A high-accuracy software sine is a stable representative inside PTX's
    // architecture-dependent approximation bound.
    let result = libm::sin(value as f64) as f32;
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

pub fn ptx_cos_approx_f32(value: f32, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    // As with sine, do not pretend to reproduce one GPU's approximation
    // polynomial; keep one target-independent value within the PTX contract.
    let result = libm::cos(value as f64) as f32;
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

pub fn ptx_exp2_approx_ftz_bf16x2(value: u32) -> u32 {
    u32::from(ptx_exp2_approx_ftz_bf16(value as u16))
        | (u32::from(ptx_exp2_approx_ftz_bf16((value >> 16) as u16)) << 16)
}

pub(crate) fn ptx_exp2_approx_ftz_bf16(value: u16) -> u16 {
    let result = ptx_exp2_approx_ftz_f32(crate::bf16_bits_to_f32(value));
    let encoded = crate::f32_to_bf16_bits(cuda_canonicalize_nan_f32(result));
    if encoded & 0x7f80 == 0 {
        encoded & 0x8000
    } else {
        encoded
    }
}

pub(crate) fn ptx_exp2_approx_f16(value: u16) -> u16 {
    cuda_f32_to_fp16_bits(ptx_exp2_approx_f32(cuda_fp16_bits_to_f32(value)))
}

pub(crate) fn ptx_exp2_approx_f16x2(value: u32) -> u32 {
    u32::from(ptx_exp2_approx_f16(value as u16))
        | (u32::from(ptx_exp2_approx_f16((value >> 16) as u16)) << 16)
}

pub fn ptx_lg2_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    // As with the exp2 adapter, use a target-independent software value as the
    // canonical representative instead of replaying an architecture-specific
    // approximation polynomial.
    flush_subnormal_f32(libm::log2(value as f64) as f32)
}

pub fn ptx_tanh_approx_f32(value: f32) -> f32 {
    // Use one target-independent software representative rather than replaying
    // an architecture-specific tanh.approx polynomial.
    libm::tanh(value as f64) as f32
}

pub fn ptx_tanh_approx_f16(value: u16) -> u16 {
    cuda_f32_to_fp16_bits(ptx_tanh_approx_f32(cuda_fp16_bits_to_f32(value)))
}

pub fn ptx_tanh_approx_f16x2(value: u32) -> u32 {
    u32::from(ptx_tanh_approx_f16(value as u16))
        | (u32::from(ptx_tanh_approx_f16((value >> 16) as u16)) << 16)
}

pub fn ptx_tanh_approx_bf16(value: u16) -> u16 {
    crate::f32_to_bf16_bits(cuda_canonicalize_nan_f32(ptx_tanh_approx_f32(
        crate::bf16_bits_to_f32(value),
    )))
}

pub fn ptx_tanh_approx_bf16x2(value: u32) -> u32 {
    u32::from(ptx_tanh_approx_bf16(value as u16))
        | (u32::from(ptx_tanh_approx_bf16((value >> 16) as u16)) << 16)
}

pub fn ptx_rsqrt_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(1.0_f32 / value.sqrt())
}

pub fn ptx_rsqrt_approx_f32(value: f32) -> f32 {
    1.0_f32 / value.sqrt()
}

pub fn ptx_rcp_approx_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(1.0_f32 / value)
}

pub fn ptx_rcp_approx_f32(value: f32) -> f32 {
    1.0_f32 / value
}

pub fn ptx_max_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if propagate_nan && (lhs.is_nan() || rhs.is_nan()) {
        cuda_canonical_nan_f32()
    } else {
        cuda_f32_max(lhs, rhs)
    };
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

pub(crate) fn ptx_min_f32(lhs: f32, rhs: f32, ftz: bool, propagate_nan: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if propagate_nan && (lhs.is_nan() || rhs.is_nan()) {
        cuda_canonical_nan_f32()
    } else {
        cuda_f32_min(lhs, rhs)
    };
    if ftz {
        flush_subnormal_f32(result)
    } else {
        result
    }
}

fn next_f32(value: f32, upward: bool) -> f32 {
    let terminal = if upward {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    if value.is_nan() || value == terminal {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(if upward { 1 } else { 0x8000_0001 });
    }
    let bits = value.to_bits();
    let next = if (value > 0.0) == upward {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

fn round_f32_from_exact(rounded: f32, exact: f64, mode: F32RoundingMode) -> f32 {
    match mode {
        F32RoundingMode::Nearest => rounded,
        F32RoundingMode::Down if (rounded as f64) > exact => next_f32(rounded, false),
        F32RoundingMode::Up if (rounded as f64) < exact => next_f32(rounded, true),
        F32RoundingMode::Zero if exact > 0.0 && (rounded as f64) > exact => {
            next_f32(rounded, false)
        }
        F32RoundingMode::Zero if exact < 0.0 && (rounded as f64) < exact => next_f32(rounded, true),
        F32RoundingMode::Down | F32RoundingMode::Up | F32RoundingMode::Zero => rounded,
    }
}

// Every finite f32 is an integer multiple of 2^-149. Five words cover the
// largest exact sum: two 24-bit significands shifted by at most 253 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactF32Magnitude([u64; 5]);

impl ExactF32Magnitude {
    fn from_f32(value: f32) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits() & 0x7fff_ffff;
        let exponent = ((bits >> 23) & 0xff) as usize;
        let fraction = bits & 0x007f_ffff;
        let significand = if exponent == 0 {
            fraction
        } else {
            (1 << 23) | fraction
        } as u64;
        let shift = if exponent == 0 { 0 } else { exponent - 1 };
        let word_index = shift / 64;
        let bit_index = shift % 64;
        let mut words = [0_u64; 5];
        words[word_index] = significand << bit_index;
        if bit_index != 0 {
            words[word_index + 1] = significand >> (64 - bit_index);
        }
        Self(words)
    }

    fn is_zero(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    fn compare(self, rhs: Self) -> Ordering {
        for index in (0..self.0.len()).rev() {
            match self.0[index].cmp(&rhs.0[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }

    fn add(self, rhs: Self) -> Self {
        let mut words = [0_u64; 5];
        let mut carry = 0_u128;
        for (index, word) in words.iter_mut().enumerate() {
            let sum = self.0[index] as u128 + rhs.0[index] as u128 + carry;
            *word = sum as u64;
            carry = sum >> 64;
        }
        debug_assert_eq!(carry, 0);
        Self(words)
    }

    fn subtract(self, rhs: Self) -> Self {
        debug_assert!(self.compare(rhs) != Ordering::Less);
        let mut words = [0_u64; 5];
        let mut borrow = false;
        for (index, word) in words.iter_mut().enumerate() {
            let (difference, rhs_borrow) = self.0[index].overflowing_sub(rhs.0[index]);
            let (difference, carry_borrow) = difference.overflowing_sub(u64::from(borrow));
            *word = difference;
            borrow = rhs_borrow || carry_borrow;
        }
        debug_assert!(!borrow);
        Self(words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactF32Value {
    negative: bool,
    magnitude: ExactF32Magnitude,
}

const EXACT_FMA_WORDS: usize = 10;
// Binary64 products are multiples of 2^-2148 and smaller than 2^2048.
// The same fixed-word arithmetic therefore needs ceil(4196 / 64) words.
const EXACT_F64_FMA_WORDS: usize = 66;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactFmaMagnitude<const WORDS: usize = EXACT_FMA_WORDS>([u64; WORDS]);

impl<const WORDS: usize> ExactFmaMagnitude<WORDS> {
    fn from_significand(significand: u64, shift: usize) -> Self {
        let mut words = [0_u64; WORDS];
        if significand == 0 {
            return Self(words);
        }
        let word_index = shift / 64;
        let bit_index = shift % 64;
        debug_assert!(word_index < WORDS);
        words[word_index] = significand << bit_index;
        if bit_index != 0 {
            debug_assert!(word_index + 1 < WORDS);
            words[word_index + 1] = significand >> (64 - bit_index);
        }
        Self(words)
    }
}

impl ExactFmaMagnitude {
    fn f32_parts(value: f32) -> (u64, usize) {
        debug_assert!(value.is_finite());
        let bits = value.to_bits() & 0x7fff_ffff;
        let exponent = ((bits >> 23) & 0xff) as usize;
        let fraction = bits & 0x007f_ffff;
        let significand = if exponent == 0 {
            fraction
        } else {
            (1 << 23) | fraction
        } as u64;
        let shift = if exponent == 0 { 0 } else { exponent - 1 };
        (significand, shift)
    }

    fn from_f32_addend(value: f32) -> Self {
        let (significand, shift) = Self::f32_parts(value);
        Self::from_significand(significand, shift + 149)
    }

    fn from_f32_product(lhs: f32, rhs: f32) -> Self {
        let (lhs_significand, lhs_shift) = Self::f32_parts(lhs);
        let (rhs_significand, rhs_shift) = Self::f32_parts(rhs);
        Self::from_significand(lhs_significand * rhs_significand, lhs_shift + rhs_shift)
    }
}

impl<const WORDS: usize> ExactFmaMagnitude<WORDS> {
    fn is_zero(self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    fn compare(self, rhs: Self) -> Ordering {
        for index in (0..WORDS).rev() {
            match self.0[index].cmp(&rhs.0[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }

    fn add(self, rhs: Self) -> Self {
        let mut words = [0_u64; WORDS];
        let mut carry = 0_u128;
        for (index, word) in words.iter_mut().enumerate() {
            let sum = self.0[index] as u128 + rhs.0[index] as u128 + carry;
            *word = sum as u64;
            carry = sum >> 64;
        }
        debug_assert_eq!(carry, 0);
        Self(words)
    }

    fn subtract(self, rhs: Self) -> Self {
        debug_assert!(self.compare(rhs) != Ordering::Less);
        let mut words = [0_u64; WORDS];
        let mut borrow = false;
        for (index, word) in words.iter_mut().enumerate() {
            let (difference, rhs_borrow) = self.0[index].overflowing_sub(rhs.0[index]);
            let (difference, carry_borrow) = difference.overflowing_sub(u64::from(borrow));
            *word = difference;
            borrow = rhs_borrow || carry_borrow;
        }
        debug_assert!(!borrow);
        Self(words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactFmaValue<const WORDS: usize = EXACT_FMA_WORDS> {
    negative: bool,
    magnitude: ExactFmaMagnitude<WORDS>,
}

impl ExactFmaValue {
    fn product(lhs: f32, rhs: f32) -> Self {
        Self {
            negative: lhs.is_sign_negative() != rhs.is_sign_negative(),
            magnitude: ExactFmaMagnitude::from_f32_product(lhs, rhs),
        }
    }

    fn addend(value: f32) -> Self {
        Self {
            negative: value.is_sign_negative(),
            magnitude: ExactFmaMagnitude::from_f32_addend(value),
        }
    }
}

impl<const WORDS: usize> ExactFmaValue<WORDS> {
    fn is_zero(self) -> bool {
        self.magnitude.is_zero()
    }

    fn compare(self, rhs: Self) -> Ordering {
        let lhs_negative = self.negative && !self.is_zero();
        let rhs_negative = rhs.negative && !rhs.is_zero();
        if lhs_negative != rhs_negative {
            return if lhs_negative {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        let magnitude_order = self.magnitude.compare(rhs.magnitude);
        if lhs_negative {
            magnitude_order.reverse()
        } else {
            magnitude_order
        }
    }
}

impl ExactFmaValue<EXACT_F64_FMA_WORDS> {
    fn product_f64(lhs: f64, rhs: f64) -> Self {
        let magnitude = if lhs == 0.0 || rhs == 0.0 {
            ExactFmaMagnitude([0; EXACT_F64_FMA_WORDS])
        } else {
            let (a, a_exponent) = positive_f64_dyadic(lhs.abs());
            let (b, b_exponent) = positive_f64_dyadic(rhs.abs());
            let product = a * b;
            let shift = (a_exponent + b_exponent + 2148) as usize;
            ExactFmaMagnitude::from_significand(product as u64, shift).add(
                ExactFmaMagnitude::from_significand((product >> 64) as u64, shift + 64),
            )
        };
        Self {
            negative: lhs.is_sign_negative() != rhs.is_sign_negative(),
            magnitude,
        }
    }

    fn addend_f64(value: f64) -> Self {
        let magnitude = if value == 0.0 {
            ExactFmaMagnitude([0; EXACT_F64_FMA_WORDS])
        } else {
            let (significand, exponent) = positive_f64_dyadic(value.abs());
            ExactFmaMagnitude::from_significand(significand as u64, (exponent + 2148) as usize)
        };
        Self {
            negative: value.is_sign_negative(),
            magnitude,
        }
    }
}

impl ExactF32Value {
    fn from_f32(value: f32) -> Self {
        Self {
            negative: value.is_sign_negative(),
            magnitude: ExactF32Magnitude::from_f32(value),
        }
    }

    fn is_zero(self) -> bool {
        self.magnitude.is_zero()
    }
}

fn exact_f32_sum(lhs: ExactF32Value, rhs: ExactF32Value) -> ExactF32Value {
    if lhs.is_zero() {
        return rhs;
    }
    if rhs.is_zero() {
        return lhs;
    }
    if lhs.negative == rhs.negative {
        return ExactF32Value {
            negative: lhs.negative,
            magnitude: lhs.magnitude.add(rhs.magnitude),
        };
    }
    match lhs.magnitude.compare(rhs.magnitude) {
        Ordering::Greater => ExactF32Value {
            negative: lhs.negative,
            magnitude: lhs.magnitude.subtract(rhs.magnitude),
        },
        Ordering::Less => ExactF32Value {
            negative: rhs.negative,
            magnitude: rhs.magnitude.subtract(lhs.magnitude),
        },
        Ordering::Equal => ExactF32Value {
            negative: false,
            magnitude: ExactF32Magnitude([0; 5]),
        },
    }
}

fn exact_fma_sum<const WORDS: usize>(
    lhs: ExactFmaValue<WORDS>,
    rhs: ExactFmaValue<WORDS>,
) -> ExactFmaValue<WORDS> {
    if lhs.is_zero() {
        return rhs;
    }
    if rhs.is_zero() {
        return lhs;
    }
    if lhs.negative == rhs.negative {
        return ExactFmaValue {
            negative: lhs.negative,
            magnitude: lhs.magnitude.add(rhs.magnitude),
        };
    }
    match lhs.magnitude.compare(rhs.magnitude) {
        Ordering::Greater => ExactFmaValue {
            negative: lhs.negative,
            magnitude: lhs.magnitude.subtract(rhs.magnitude),
        },
        Ordering::Less => ExactFmaValue {
            negative: rhs.negative,
            magnitude: rhs.magnitude.subtract(lhs.magnitude),
        },
        Ordering::Equal => ExactFmaValue {
            negative: false,
            magnitude: ExactFmaMagnitude([0; WORDS]),
        },
    }
}

#[derive(Clone, Copy)]
pub(crate) enum LowPrecisionFormat {
    F16,
    Bf16,
}

impl LowPrecisionFormat {
    const fn fraction_bits(self) -> usize {
        match self {
            Self::F16 => 10,
            Self::Bf16 => 7,
        }
    }

    const fn minimum_normal_exponent(self) -> i32 {
        match self {
            Self::F16 => -14,
            Self::Bf16 => -126,
        }
    }

    const fn maximum_normal_exponent(self) -> i32 {
        match self {
            Self::F16 => 15,
            Self::Bf16 => 127,
        }
    }

    const fn exponent_bias(self) -> i32 {
        match self {
            Self::F16 => 15,
            Self::Bf16 => 127,
        }
    }

    pub(crate) const fn infinity(self) -> u16 {
        match self {
            Self::F16 => 0x7c00,
            Self::Bf16 => 0x7f80,
        }
    }

    fn encode_host_result(self, value: f32) -> u16 {
        match self {
            Self::F16 => cuda_f32_to_fp16_bits(value),
            Self::Bf16 => f32_to_bf16_bits(cuda_canonicalize_nan_f32(value)),
        }
    }
}

fn highest_set_bit(words: &[u64]) -> Option<usize> {
    words.iter().rposition(|word| *word != 0).map(|index| {
        index * u64::BITS as usize + (u64::BITS - 1 - words[index].leading_zeros()) as usize
    })
}

fn bit_is_set(words: &[u64], bit: usize) -> bool {
    words
        .get(bit / u64::BITS as usize)
        .is_some_and(|word| word & (1_u64 << (bit % u64::BITS as usize)) != 0)
}

fn any_bit_below(words: &[u64], bit: usize) -> bool {
    let whole_words = bit / u64::BITS as usize;
    if words.iter().take(whole_words).any(|word| *word != 0) {
        return true;
    }
    let partial = bit % u64::BITS as usize;
    partial != 0
        && words
            .get(whole_words)
            .is_some_and(|word| word & ((1_u64 << partial) - 1) != 0)
}

fn low_u64_after_shift(words: &[u64], shift: usize) -> u64 {
    let word_index = shift / u64::BITS as usize;
    let bit_index = shift % u64::BITS as usize;
    let low = words.get(word_index).copied().unwrap_or(0) >> bit_index;
    if bit_index == 0 {
        low
    } else {
        low | words
            .get(word_index + 1)
            .copied()
            .unwrap_or(0)
            .wrapping_shl((u64::BITS as usize - bit_index) as u32)
    }
}

fn round_words_right_even(words: &[u64], shift: usize) -> u64 {
    if shift == 0 {
        return words.first().copied().unwrap_or(0);
    }
    let truncated = low_u64_after_shift(words, shift);
    let halfway = bit_is_set(words, shift - 1);
    let below_halfway = any_bit_below(words, shift - 1);
    truncated + u64::from(halfway && (below_halfway || truncated & 1 != 0))
}

/// Round an exact signed dyadic integer to one scalar f16/bf16 payload.
/// `words` is a little-endian magnitude in units of `2^unit_exponent`.
fn encode_exact_low(
    negative: bool,
    words: &[u64],
    unit_exponent: i32,
    format: LowPrecisionFormat,
    ftz: bool,
) -> u16 {
    let sign = if negative { 0x8000 } else { 0 };
    let Some(highest) = highest_set_bit(words) else {
        return sign;
    };
    let fraction_bits = format.fraction_bits();
    let mut exponent = unit_exponent + highest as i32;
    // Half FTZ detects tininess before rounding, including values that
    // would otherwise round up to the smallest normal.
    if ftz && exponent < format.minimum_normal_exponent() {
        return sign;
    }
    if exponent > format.maximum_normal_exponent() {
        return sign | format.infinity();
    }

    if exponent >= format.minimum_normal_exponent() {
        let mut significand = if highest >= fraction_bits {
            round_words_right_even(words, highest - fraction_bits)
        } else {
            // Exact inputs such as small integers may need normalization to
            // the left, with no discarded bits and therefore no rounding.
            words[0] << (fraction_bits - highest)
        };
        if significand == 1_u64 << (fraction_bits + 1) {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > format.maximum_normal_exponent() {
            return sign | format.infinity();
        }
        let exponent_field = (exponent + format.exponent_bias()) as u16;
        let fraction_mask = (1_u16 << fraction_bits) - 1;
        return sign | (exponent_field << fraction_bits) | (significand as u16 & fraction_mask);
    }

    let quantum_exponent = format.minimum_normal_exponent() - fraction_bits as i32;
    let shift = (quantum_exponent - unit_exponent) as usize;
    let subnormal = round_words_right_even(words, shift);
    sign | subnormal as u16
}

pub(crate) fn decode_low(bits: u16, format: LowPrecisionFormat) -> f32 {
    match format {
        LowPrecisionFormat::F16 => cuda_fp16_bits_to_f32(bits),
        LowPrecisionFormat::Bf16 => bf16_bits_to_f32(bits),
    }
}

pub(crate) fn low_add_rn(
    lhs: u16,
    rhs: u16,
    format: LowPrecisionFormat,
    subtract: bool,
    ftz: bool,
) -> u16 {
    let (lhs, rhs) = if ftz {
        (flush_subnormal_f16_bits(lhs), flush_subnormal_f16_bits(rhs))
    } else {
        (lhs, rhs)
    };
    let lhs = decode_low(lhs, format);
    let rhs = decode_low(rhs, format);
    let host = if subtract { lhs - rhs } else { lhs + rhs };
    if !lhs.is_finite() || !rhs.is_finite() {
        return format.encode_host_result(host);
    }
    let rhs = if subtract { -rhs } else { rhs };
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    if exact.is_zero() {
        return format.encode_host_result(host);
    }
    encode_exact_low(exact.negative, &exact.magnitude.0, -149, format, ftz)
}

pub(crate) fn low_fma_rn(
    lhs: u16,
    rhs: u16,
    addend: u16,
    format: LowPrecisionFormat,
    ftz: bool,
) -> u16 {
    let (lhs, rhs, addend) = if ftz {
        (
            flush_subnormal_f16_bits(lhs),
            flush_subnormal_f16_bits(rhs),
            flush_subnormal_f16_bits(addend),
        )
    } else {
        (lhs, rhs, addend)
    };
    let lhs = decode_low(lhs, format);
    let rhs = decode_low(rhs, format);
    let addend = decode_low(addend, format);
    let host = lhs.mul_add(rhs, addend);
    if !lhs.is_finite() || !rhs.is_finite() || !addend.is_finite() {
        return format.encode_host_result(host);
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product(lhs, rhs),
        ExactFmaValue::addend(addend),
    );
    if exact.is_zero() {
        return format.encode_host_result(host);
    }
    encode_exact_low(exact.negative, &exact.magnitude.0, -298, format, ftz)
}

pub(crate) fn sub_f16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::F16, true, false)
}

pub(crate) fn sub_f16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(sub_f16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(sub_f16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

pub(crate) fn mul_f16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_mul_rn(lhs, rhs, LowPrecisionFormat::F16, false)
}

pub(crate) fn low_mul_rn(lhs: u16, rhs: u16, format: LowPrecisionFormat, ftz: bool) -> u16 {
    // A same-sign zero addend preserves multiplication's signed zero.
    low_fma_rn(lhs, rhs, (lhs ^ rhs) & 0x8000, format, ftz)
}

/// TMA's 16-bit OOB-NaN payload; FMA tests its magnitude, not its sign.
pub(crate) const PTX_OOB_NAN: u16 = 0x7ff7;

pub(crate) fn fma_f16_bits_rn(lhs: u16, rhs: u16, addend: u16) -> u16 {
    low_fma_rn(lhs, rhs, addend, LowPrecisionFormat::F16, false)
}

pub(crate) fn add_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::Bf16, false, false)
}

pub(crate) fn add_bf16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(add_bf16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(add_bf16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

pub(crate) fn sub_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_add_rn(lhs, rhs, LowPrecisionFormat::Bf16, true, false)
}

pub(crate) fn sub_bf16x2_bits_rn(lhs: u32, rhs: u32) -> u32 {
    u32::from(sub_bf16_bits_rn(lhs as u16, rhs as u16))
        | (u32::from(sub_bf16_bits_rn((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

pub(crate) fn mul_bf16_bits_rn(lhs: u16, rhs: u16) -> u16 {
    low_mul_rn(lhs, rhs, LowPrecisionFormat::Bf16, false)
}

pub(crate) fn fma_bf16_bits_rn(lhs: u16, rhs: u16, addend: u16) -> u16 {
    low_fma_rn(lhs, rhs, addend, LowPrecisionFormat::Bf16, false)
}

fn compare_f32_to_exact_fma(rounded: f32, exact: ExactFmaValue) -> Ordering {
    debug_assert!(!rounded.is_nan());
    if rounded == f32::INFINITY {
        return Ordering::Greater;
    }
    if rounded == f32::NEG_INFINITY {
        return Ordering::Less;
    }
    ExactFmaValue::addend(rounded).compare(exact)
}

fn round_exact_zero_sum_f32(rounded: f32, lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    debug_assert_eq!(rounded, 0.0);
    let both_positive_zero = lhs.to_bits() == 0 && rhs.to_bits() == 0;
    if mode == F32RoundingMode::Down && !both_positive_zero {
        -0.0
    } else {
        rounded
    }
}

fn round_f32_from_exact_fma(
    rounded: f32,
    exact: ExactFmaValue,
    lhs: f32,
    rhs: f32,
    mode: F32RoundingMode,
) -> f32 {
    if exact.is_zero() {
        return round_exact_zero_sum_f32(rounded, lhs, rhs, mode);
    }
    let comparison = compare_f32_to_exact_fma(rounded, exact);
    round_f32_candidate(rounded, comparison, exact.negative, mode)
}

fn compare_f32_to_exact(rounded: f32, exact: ExactF32Value) -> Ordering {
    debug_assert!(!rounded.is_nan());
    if rounded == f32::INFINITY {
        return Ordering::Greater;
    }
    if rounded == f32::NEG_INFINITY {
        return Ordering::Less;
    }
    if exact.is_zero() {
        return if rounded == 0.0 {
            Ordering::Equal
        } else if rounded.is_sign_negative() {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if rounded == 0.0 {
        return if exact.negative {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }
    let rounded = ExactF32Value::from_f32(rounded);
    if rounded.negative != exact.negative {
        return if rounded.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitude_order = rounded.magnitude.compare(exact.magnitude);
    if rounded.negative {
        magnitude_order.reverse()
    } else {
        magnitude_order
    }
}

fn round_f32_from_exact_sum(
    rounded: f32,
    exact: ExactF32Value,
    lhs: f32,
    rhs: f32,
    mode: F32RoundingMode,
) -> f32 {
    if exact.is_zero() {
        return round_exact_zero_sum_f32(rounded, lhs, rhs, mode);
    }
    let comparison = compare_f32_to_exact(rounded, exact);
    round_f32_candidate(rounded, comparison, exact.negative, mode)
}

fn round_f32_candidate(
    rounded: f32,
    comparison: Ordering,
    negative: bool,
    mode: F32RoundingMode,
) -> f32 {
    match rounding_adjustment(comparison, negative, mode) {
        Ordering::Less => next_f32(rounded, false),
        Ordering::Greater => next_f32(rounded, true),
        Ordering::Equal => rounded,
    }
}

pub fn add_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let rounded = lhs + rhs;
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        return rounded;
    }
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    round_f32_from_exact_sum(rounded, exact, lhs, rhs, mode)
}

pub fn sub_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let rounded = lhs - rhs;
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        return rounded;
    }
    let rhs = -rhs;
    let exact = exact_f32_sum(ExactF32Value::from_f32(lhs), ExactF32Value::from_f32(rhs));
    round_f32_from_exact_sum(rounded, exact, lhs, rhs, mode)
}

pub fn mul_f32(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    round_f32_from_exact(lhs * rhs, (lhs as f64) * (rhs as f64), mode)
}

pub fn add_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    flush_subnormal_f32(add_f32(
        flush_subnormal_f32(lhs),
        flush_subnormal_f32(rhs),
        mode,
    ))
}

pub fn sub_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    flush_subnormal_f32(sub_f32(
        flush_subnormal_f32(lhs),
        flush_subnormal_f32(rhs),
        mode,
    ))
}

pub fn mul_f32_ftz(lhs: f32, rhs: f32, mode: F32RoundingMode) -> f32 {
    let lhs = flush_subnormal_f32(lhs);
    let rhs = flush_subnormal_f32(rhs);
    flush_f32_result(mul_f32(lhs, rhs, mode), || {
        (f64::from(lhs) * f64::from(rhs)).abs() < f64::from(f32::MIN_POSITIVE)
    })
}

/// At the smallest normal result, use the instruction's tininess rule;
/// ordinary subnormal outputs are identified by their rounded representation.
fn flush_f32_result(rounded: f32, is_tiny: impl FnOnce() -> bool) -> f32 {
    if rounded.abs() == f32::MIN_POSITIVE && is_tiny() {
        0.0_f32.copysign(rounded)
    } else {
        flush_subnormal_f32(rounded)
    }
}

pub fn div_f32_rn(lhs: f32, rhs: f32) -> f32 {
    lhs / rhs
}

pub(crate) fn ptx_div_f32(lhs: f32, rhs: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let nearest = div_f32_rn(lhs, rhs);
    let rounded = if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || lhs == 0.0
        || rhs == 0.0
    {
        nearest
    } else {
        let comparison =
            compare_division_candidate(f64::from(nearest), f64::from(lhs), f64::from(rhs));
        round_f32_candidate(
            nearest,
            comparison,
            lhs.is_sign_negative() != rhs.is_sign_negative(),
            mode,
        )
    };
    if ftz {
        flush_f32_result(rounded, || {
            compare_division_candidate(
                f64::from(f32::MIN_POSITIVE),
                f64::from(lhs.abs()),
                f64::from(rhs.abs()),
            ) == Ordering::Greater
        })
    } else {
        rounded
    }
}

/// PTX full-range and limited-range division approximations. In the normal
/// bounded domain the nearest quotient is a stable in-bound representative.
/// The limited-range instruction additionally has prescribed large-divisor
/// zeros/NaNs. It does not require a rounded FP32 reciprocal intermediate.
pub(crate) fn ptx_div_approx_f32(lhs: f32, rhs: f32, ftz: bool, full: bool) -> f32 {
    let lhs = if ftz { flush_subnormal_f32(lhs) } else { lhs };
    let rhs = if ftz { flush_subnormal_f32(rhs) } else { rhs };
    let result = if !full && rhs.is_finite() && rhs.abs() > f32::from_bits(0x7e80_0000) {
        lhs * 0.0_f32.copysign(rhs)
    } else {
        div_f32_rn(lhs, rhs)
    };
    if ftz {
        flush_f32_result(result, || {
            compare_division_candidate(
                f64::from(f32::MIN_POSITIVE),
                f64::from(lhs.abs()),
                f64::from(rhs.abs()),
            ) == Ordering::Greater
        })
    } else {
        result
    }
}

/// PTX 1.11.20 gross reciprocal: ignore both low words, honor canonical NaN
/// and FTZ, and choose nearest-even at the reduced precision for finite values.
pub(crate) fn ptx_rcp_approx_ftz_f64(value: f64) -> f64 {
    gross_f64_approx(value, |value| 1.0 / value)
}

pub(crate) fn ptx_rsqrt_approx_ftz_f64(value: f64) -> f64 {
    gross_f64_approx(value, |value| 1.0 / value.sqrt())
}

/// Shared input/output contract of the two PTX 1.11.20 approximations.
fn gross_f64_approx(value: f64, operation: impl FnOnce(f64) -> f64) -> f64 {
    let value = f64::from_bits(value.to_bits() & 0xffff_ffff_0000_0000);
    if value.is_nan() {
        return f64::from_bits(0x7fff_ffff_0000_0000);
    }
    let value = if value.is_subnormal() {
        0.0_f64.copysign(value)
    } else {
        value
    };
    let reciprocal = operation(value);
    if reciprocal.is_nan() {
        return f64::from_bits(0x7fff_ffff_0000_0000);
    }
    let bits = reciprocal.to_bits();
    let upper = bits >> 32;
    let discarded = bits & 0xffff_ffff;
    let increment = discarded > 0x8000_0000 || (discarded == 0x8000_0000 && upper & 1 != 0);
    let rounded = f64::from_bits((upper + u64::from(increment)) << 32);
    if rounded.is_subnormal() {
        0.0_f64.copysign(rounded)
    } else {
        rounded
    }
}

pub fn ptx_neg_ftz_f32(value: f32) -> f32 {
    let value = flush_subnormal_f32(value);
    flush_subnormal_f32(f32::from_bits(value.to_bits() ^ 0x8000_0000))
}

pub(crate) fn ptx_neg_f16_bits(value: u16, ftz: bool) -> u16 {
    let value = if ftz {
        flush_subnormal_f16_bits(value)
    } else {
        value
    };
    value ^ 0x8000
}

pub(crate) fn ptx_neg_f16x2_bits(value: u32, ftz: bool) -> u32 {
    u32::from(ptx_neg_f16_bits(value as u16, ftz))
        | (u32::from(ptx_neg_f16_bits((value >> 16) as u16, ftz)) << 16)
}

pub fn ptx_sqrt_f32(value: f32, mode: F32RoundingMode, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    let nearest = value.sqrt();
    let rounded = if mode == F32RoundingMode::Nearest
        || !value.is_finite()
        || value <= 0.0
        || nearest.is_nan()
    {
        nearest
    } else {
        // A binary32 square has at most 48 significant bits, so binary64
        // compares the rounded candidate's square with the input exactly.
        let comparison = ((nearest as f64) * (nearest as f64)).total_cmp(&(value as f64));
        match mode {
            F32RoundingMode::Down | F32RoundingMode::Zero if comparison == Ordering::Greater => {
                next_f32(nearest, false)
            }
            F32RoundingMode::Up if comparison == Ordering::Less => next_f32(nearest, true),
            F32RoundingMode::Nearest
            | F32RoundingMode::Down
            | F32RoundingMode::Up
            | F32RoundingMode::Zero => nearest,
        }
    };
    if ftz {
        flush_subnormal_f32(rounded)
    } else {
        rounded
    }
}

fn positive_f64_dyadic(value: f64) -> (u128, i32) {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let (significand, exponent) = if exponent_field == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_field - 1023 - 52)
    };
    let trailing = significand.trailing_zeros();
    (
        u128::from(significand >> trailing),
        exponent + trailing as i32,
    )
}

fn compare_positive_dyadics(
    lhs_significand: u128,
    lhs_exponent: i32,
    rhs_significand: u128,
    rhs_exponent: i32,
) -> Ordering {
    let lhs_bits = (u128::BITS - lhs_significand.leading_zeros()) as i32;
    let rhs_bits = (u128::BITS - rhs_significand.leading_zeros()) as i32;
    match (lhs_bits + lhs_exponent).cmp(&(rhs_bits + rhs_exponent)) {
        Ordering::Equal => {}
        order => return order,
    }
    if lhs_exponent >= rhs_exponent {
        let shift = u32::try_from(lhs_exponent - rhs_exponent).unwrap();
        lhs_significand
            .checked_shl(shift)
            .expect("equal-magnitude dyadics fit after alignment")
            .cmp(&rhs_significand)
    } else {
        let shift = u32::try_from(rhs_exponent - lhs_exponent).unwrap();
        lhs_significand.cmp(
            &rhs_significand
                .checked_shl(shift)
                .expect("equal-magnitude dyadics fit after alignment"),
        )
    }
}

fn compare_f64_square_to_input(root: f64, input: f64) -> Ordering {
    let (root_significand, root_exponent) = positive_f64_dyadic(root);
    let square = root_significand * root_significand;
    let trailing = square.trailing_zeros();
    let square = square >> trailing;
    let square_exponent = 2 * root_exponent + trailing as i32;
    let (input_significand, input_exponent) = positive_f64_dyadic(input);
    compare_positive_dyadics(square, square_exponent, input_significand, input_exponent)
}

pub fn ptx_sqrt_f64(value: f64, mode: F32RoundingMode) -> f64 {
    let nearest = value.sqrt();
    if mode == F32RoundingMode::Nearest || !value.is_finite() || value <= 0.0 || nearest.is_nan() {
        return nearest;
    }
    match (mode, compare_f64_square_to_input(nearest, value)) {
        (F32RoundingMode::Down | F32RoundingMode::Zero, Ordering::Greater) => nearest.next_down(),
        (F32RoundingMode::Up, Ordering::Less) => nearest.next_up(),
        (
            F32RoundingMode::Nearest
            | F32RoundingMode::Down
            | F32RoundingMode::Up
            | F32RoundingMode::Zero,
            _,
        ) => nearest,
    }
}

pub fn fma_f32_rn(lhs: f32, rhs: f32, addend: f32) -> f32 {
    lhs.mul_add(rhs, addend)
}

/// Correct a nearest binary64 FMA by comparison with its exact dyadic result.
/// No thread-local rounding environment is changed, including on overflow.
pub(crate) fn fma_f64(lhs: f64, rhs: f64, addend: f64, mode: F32RoundingMode) -> f64 {
    let rounded = lhs.mul_add(rhs, addend);
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || !addend.is_finite()
    {
        return rounded;
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product_f64(lhs, rhs),
        ExactFmaValue::addend_f64(addend),
    );
    if exact.is_zero() {
        let positive_zero_product =
            (lhs == 0.0 || rhs == 0.0) && lhs.is_sign_negative() == rhs.is_sign_negative();
        return if mode == F32RoundingMode::Down && !(positive_zero_product && addend.to_bits() == 0)
        {
            -0.0
        } else {
            rounded
        };
    }
    let comparison = if rounded == f64::INFINITY {
        Ordering::Greater
    } else if rounded == f64::NEG_INFINITY {
        Ordering::Less
    } else {
        ExactFmaValue::addend_f64(rounded).compare(exact)
    };
    round_f64_candidate(rounded, comparison, exact.negative, mode)
}

fn round_f64_candidate(
    rounded: f64,
    comparison: Ordering,
    negative: bool,
    mode: F32RoundingMode,
) -> f64 {
    match rounding_adjustment(comparison, negative, mode) {
        Ordering::Less => rounded.next_down(),
        Ordering::Greater => rounded.next_up(),
        Ordering::Equal => rounded,
    }
}

fn rounding_adjustment(comparison: Ordering, negative: bool, mode: F32RoundingMode) -> Ordering {
    match mode {
        F32RoundingMode::Down if comparison == Ordering::Greater => Ordering::Less,
        F32RoundingMode::Up if comparison == Ordering::Less => Ordering::Greater,
        F32RoundingMode::Zero if !negative && comparison == Ordering::Greater => Ordering::Less,
        F32RoundingMode::Zero if negative && comparison == Ordering::Less => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

pub(crate) fn add_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        cuda_f64_add(lhs, rhs)
    } else {
        fma_f64(lhs, 1.0, rhs, mode)
    }
}

pub(crate) fn sub_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        lhs - rhs
    } else {
        fma_f64(lhs, 1.0, -rhs, mode)
    }
}

pub(crate) fn mul_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    if mode == F32RoundingMode::Nearest || !lhs.is_finite() || !rhs.is_finite() {
        lhs * rhs
    } else {
        // Same-signed zero addition preserves the product's sign, including -0.
        let zero = f64::from_bits((lhs.to_bits() ^ rhs.to_bits()) & (1_u64 << 63));
        fma_f64(lhs, rhs, zero, mode)
    }
}

pub(crate) fn div_f64(lhs: f64, rhs: f64, mode: F32RoundingMode) -> f64 {
    let rounded = lhs / rhs;
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || lhs == 0.0
        || rhs == 0.0
    {
        return rounded;
    }
    let comparison = compare_division_candidate(rounded, lhs, rhs);
    round_f64_candidate(
        rounded,
        comparison,
        lhs.is_sign_negative() != rhs.is_sign_negative(),
        mode,
    )
}

fn compare_division_candidate(rounded: f64, lhs: f64, rhs: f64) -> Ordering {
    debug_assert!(lhs.is_finite() && rhs.is_finite() && lhs != 0.0 && rhs != 0.0);
    // Compare |candidate| * |denominator| with |numerator| exactly. The
    // product has at most 106 significand bits; no floating division residual
    // or thread-local rounding state is needed, even at exponent extremes.
    let magnitude_comparison = if rounded.is_infinite() {
        Ordering::Greater
    } else if rounded == 0.0 {
        Ordering::Less
    } else {
        let (candidate, candidate_exp) = positive_f64_dyadic(rounded.abs());
        let (denominator, denominator_exp) = positive_f64_dyadic(rhs.abs());
        let (numerator, numerator_exp) = positive_f64_dyadic(lhs.abs());
        compare_positive_dyadics(
            candidate * denominator,
            candidate_exp + denominator_exp,
            numerator,
            numerator_exp,
        )
    };
    let negative = lhs.is_sign_negative() != rhs.is_sign_negative();
    if negative {
        magnitude_comparison.reverse()
    } else {
        magnitude_comparison
    }
}

pub fn fma_f32(lhs: f32, rhs: f32, addend: f32, mode: F32RoundingMode) -> f32 {
    let rounded = lhs.mul_add(rhs, addend);
    if mode == F32RoundingMode::Nearest
        || !lhs.is_finite()
        || !rhs.is_finite()
        || !addend.is_finite()
    {
        return rounded;
    }
    let exact = exact_fma_sum(
        ExactFmaValue::product(lhs, rhs),
        ExactFmaValue::addend(addend),
    );
    round_f32_from_exact_fma(rounded, exact, lhs * rhs, addend, mode)
}

pub fn fma_f32_ftz(lhs: f32, rhs: f32, addend: f32, mode: F32RoundingMode) -> f32 {
    let lhs = flush_subnormal_f32(lhs);
    let rhs = flush_subnormal_f32(rhs);
    let addend = flush_subnormal_f32(addend);
    flush_f32_result(fma_f32(lhs, rhs, addend, mode), || {
        let exact = exact_fma_sum(
            ExactFmaValue::product(lhs, rhs),
            ExactFmaValue::addend(addend),
        );
        exact
            .magnitude
            .compare(ExactFmaValue::addend(f32::MIN_POSITIVE).magnitude)
            == Ordering::Less
    })
}

fn saturate_float<T: PartialOrd + From<u8>>(value: T) -> T {
    // PTX saturation maps NaNs and both signs of zero to positive zero.
    if value.partial_cmp(&T::from(0)) != Some(Ordering::Greater) {
        T::from(0)
    } else if value > T::from(1) {
        T::from(1)
    } else {
        value
    }
}

pub fn ptx_saturate_f32(value: f32) -> f32 {
    saturate_float(value)
}

pub(crate) fn ptx_saturate_f64(value: f64) -> f64 {
    saturate_float(value)
}

pub fn make_float2(x: f32, y: f32) -> u64 {
    x.to_bits() as u64 | ((y.to_bits() as u64) << 32)
}

pub fn float2_x(value: u64) -> f32 {
    f32::from_bits(value as u32)
}

pub fn float2_y(value: u64) -> f32 {
    f32::from_bits((value >> 32) as u32)
}

/// PTX vector `mov.b64`: the first b16 lane occupies the least-significant bits.
pub fn ptx_mov_pack_b16x4(values: [u16; 4]) -> u64 {
    values
        .into_iter()
        .enumerate()
        .fold(0_u64, |packed, (lane, value)| {
            packed | (u64::from(value) << (lane * 16))
        })
}

/// Inverse of [`ptx_mov_pack_b16x4`] in PTX register-group order.
pub fn ptx_mov_unpack_b16x4(value: u64) -> [u16; 4] {
    std::array::from_fn(|lane| (value >> (lane * 16)) as u16)
}

/// PTX vector `mov.b128`: four b32 lanes ordered from least to most significant.
pub fn ptx_mov_pack_b32x4(values: [u32; 4]) -> U64x2 {
    [
        u64::from(values[0]) | (u64::from(values[1]) << 32),
        u64::from(values[2]) | (u64::from(values[3]) << 32),
    ]
}

/// Inverse of [`ptx_mov_pack_b32x4`] in PTX register-group order.
pub fn ptx_mov_unpack_b32x4(value: U64x2) -> [u32; 4] {
    [
        value[0] as u32,
        (value[0] >> 32) as u32,
        value[1] as u32,
        (value[1] >> 32) as u32,
    ]
}

/// PTX vector `mov.b128` split into its low then high b64 lane.
pub fn ptx_mov_unpack_b64x2(value: U64x2) -> [u64; 2] {
    value
}

/// PTX `cvt.pack.sat`: saturate two s32 values and pack `b` below `a`.
///
/// For 2/4/8-bit fields, the remaining high bits come from the low bits of
/// `c`. The closed v2 variants instantiate only the four widths admitted by
/// the ISA, with `c = 0` for the 16-bit syntax line that has no c operand.
pub fn ptx_cvt_pack<const BITS: u32, const SIGNED: bool>(a: i32, b: i32, c: u32) -> u32 {
    debug_assert!(matches!(BITS, 2 | 4 | 8 | 16));
    let (minimum, maximum) = if SIGNED {
        (-(1_i32 << (BITS - 1)), (1_i32 << (BITS - 1)) - 1)
    } else {
        (0, ((1_u32 << BITS) - 1) as i32)
    };
    let mask = (1_u32 << BITS) - 1;
    let field = |value: i32| value.clamp(minimum, maximum) as u32 & mask;
    let packed = field(b) | (field(a) << BITS);
    if BITS == 16 {
        packed
    } else {
        packed | (c << (2 * BITS))
    }
}

pub fn pack_bf16x2(lhs: f32, rhs: f32) -> u32 {
    let encode = |value: f32| {
        if value.is_nan() {
            0x7fff_u16
        } else {
            f32_to_bf16_bits(value)
        }
    };
    encode(lhs) as u32 | ((encode(rhs) as u32) << 16)
}

pub fn unpack_bf16x2(value: u32) -> u64 {
    make_float2(
        bf16_bits_to_f32(value as u16),
        bf16_bits_to_f32((value >> 16) as u16),
    )
}

/// `.relu` clamps a negative operand — including negative zero — to `+0` before
/// conversion; it deliberately leaves NaN alone, which is the measured hardware
/// behaviour rather than a "NaN is not positive" reading.
fn clamp_negative<const CLAMP: bool>(value: f32) -> f32 {
    if CLAMP && value.is_sign_negative() && !value.is_nan() {
        0.0_f32
    } else {
        value
    }
}

/// PTX `cvt.rn.satfinite{.relu}.<narrow>x2.{f32,f16x2,bf16x2}` element packing.
///
/// `high` becomes the upper element and `low` the lower one, as the ISA
/// specifies for both the two-`f32` and the one-packed-source spellings.
pub fn ptx_cvt_pack_narrow_x2<const CLAMP_NEGATIVE: bool>(
    high: f32,
    low: f32,
    format: NarrowFloatFormat,
) -> u16 {
    let encode = |value: f32| {
        crate::f32_to_narrow_float_bits_rn_satfinite(
            clamp_negative::<CLAMP_NEGATIVE>(value),
            format,
        )
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// PTX `cvt.{rn,rz,rp}.satfinite{.relu}.<narrow>x2` element packing.
///
/// The unsigned UE5M3 destination converts a source's magnitude, matching the
/// ISA's unsigned narrow-float encoding.  Directed modes are derived from the
/// reviewed nearest-even encoder and the exactly decoded adjacent code, so
/// normal/subnormal boundaries and ties have one canonical implementation.
pub fn ptx_cvt_pack_narrow_x2_rounded<const CLAMP_NEGATIVE: bool>(
    high: f32,
    low: f32,
    rounding: PtxFloatRounding,
    format: NarrowFloatFormat,
) -> u16 {
    let encode = |value: f32| {
        ptx_cvt_narrow_bits_satfinite(clamp_negative::<CLAMP_NEGATIVE>(value), rounding, format)
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// UE5M3 without saturation, following CUDA 13.4's `cuda_fp8` contract.
/// Sign is discarded before rounding. NaN/Inf become NaN; finite overflow
/// rounds to MAX_NORM for RZ, or to NaN for RN/RP when rounding crosses MAX_NORM.
/// `scale` is the UE8M0 divisor (127 means one). Scaled callers first flush
/// source subnormals using their source format's existing n1 preprocessing.
pub fn ptx_cvt_pack_ue5m3x2_unsaturated(
    high: f32,
    low: f32,
    rounding: PtxFloatRounding,
    scale: u8,
) -> u16 {
    let format = crate::FLOAT8_UE5M3;
    let decode = |code| f64::from(crate::narrow_float_bits_to_f32_checked(code, format).unwrap());
    let max = decode(format.max_finite_code);
    let midpoint = max + (max - decode(format.max_finite_code - 1)) / 2.0;
    let divisor = f64::from(crate::float8_e8m0fnu_bits_to_f32(scale));
    let encode = |value: f32| {
        // Power-of-two scaling is exact in f64. Do not turn finite scaled
        // overflow into an artificial f32 infinity: RZ must still produce MAX.
        let magnitude = f64::from(value).abs() / divisor;
        if !magnitude.is_finite()
            || matches!(rounding, PtxFloatRounding::PositiveInfinity) && magnitude > max
            || matches!(rounding, PtxFloatRounding::NearestEven) && magnitude > midpoint
        {
            format.nan_code
        } else if magnitude > max {
            // The highest finite code is even, so RN's exact midpoint stays here.
            format.max_finite_code
        } else {
            ptx_cvt_narrow_bits_satfinite(magnitude as f32, rounding, format)
        }
    };
    (u16::from(encode(high)) << format.storage_bits) | u16::from(encode(low))
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one binary32 primary.
///
/// The n1 grammar uses one scale for both destination elements and flushes a
/// source subnormal to positive zero before division by that power of two.
pub fn ptx_cvt_scaled_n1_f32(value: f32, scale: u8) -> f32 {
    let value = if value.is_subnormal() { 0.0 } else { value };
    value / crate::float8_e8m0fnu_bits_to_f32(scale)
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one binary16 primary.
pub fn ptx_cvt_scaled_n1_f16(bits: u16, scale: u8) -> f32 {
    let flushed = if is_subnormal_f16_bits(bits) { 0 } else { bits };
    crate::fp16_bits_to_f32(flushed) / crate::float8_e8m0fnu_bits_to_f32(scale)
}

/// Apply PTX `.scaled::n1::ue8m0` preprocessing to one bfloat16 primary.
pub fn ptx_cvt_scaled_n1_bf16(bits: u16, scale: u8) -> f32 {
    let flushed = if bits & 0x7f80 == 0 && bits & 0x007f != 0 {
        0
    } else {
        bits
    };
    bf16_bits_to_f32(flushed) / crate::float8_e8m0fnu_bits_to_f32(scale)
}

fn ptx_cvt_narrow_bits_satfinite(
    value: f32,
    rounding: PtxFloatRounding,
    format: NarrowFloatFormat,
) -> u8 {
    let nearest = crate::f32_to_narrow_float_bits_rn_satfinite(value, format);
    if matches!(rounding, PtxFloatRounding::NearestEven) || !value.is_finite() || value == 0.0 {
        return nearest;
    }

    let sign_mask = format.sign_mask();
    let sign = nearest & sign_mask;
    let nearest_magnitude = nearest & !sign_mask;
    let magnitude = value.abs();
    let nearest_value = crate::narrow_float_bits_to_f32_checked(nearest_magnitude, format)
        .expect("satfinite narrow conversion never selects a NaN code");
    let toward_zero = if nearest_value > magnitude {
        nearest_magnitude.saturating_sub(1)
    } else {
        nearest_magnitude
    };

    match rounding {
        PtxFloatRounding::Zero => sign | toward_zero,
        PtxFloatRounding::PositiveInfinity if format.signed && value.is_sign_negative() => {
            sign | toward_zero
        }
        PtxFloatRounding::PositiveInfinity => {
            let truncated = crate::narrow_float_bits_to_f32_checked(toward_zero, format)
                .expect("a finite magnitude's predecessor is finite");
            let upward = if truncated < magnitude {
                toward_zero.saturating_add(1).min(format.max_finite_code)
            } else {
                toward_zero
            };
            sign | upward
        }
        PtxFloatRounding::NearestEven => nearest,
        PtxFloatRounding::NearestAway | PtxFloatRounding::NegativeInfinity => {
            unreachable!("unsupported packed narrow-float rounding mode")
        }
    }
}

/// PTX `cvt.rs{.relu}.satfinite.<narrow>x4.f32` element packing.
///
/// `values` are the four primaries in PTX operand order, the first in the most
/// significant field, each paired with its own sixteen stochastic-rounding bits
/// from [`ptx_cvt_rs_randoms`].
pub fn ptx_cvt_pack_narrow_x4<const CLAMP_NEGATIVE: bool>(
    values: [f32; 4],
    randoms: [u16; 4],
    format: NarrowFloatFormat,
) -> u32 {
    let mut packed = 0_u32;
    for (index, (value, random)) in values.into_iter().zip(randoms).enumerate() {
        let code = crate::f32_to_narrow_float_bits_rs(
            clamp_negative::<CLAMP_NEGATIVE>(value),
            random,
            format,
        );
        packed |= u32::from(code) << (format.storage_bits * (3 - index as u32));
    }
    packed
}

/// Split one `rbits` operand into the four elements' sixteen random bits.
///
/// Established on an NVIDIA B200: each *pair* of primaries shares one sixteen
/// bit field, the first primary of the pair reading it bit-reversed and the
/// second as written. The eight-bit destinations take the two contiguous
/// halfwords — `(a, b)` from `rbits[31:16]` and `(e, f)` from `rbits[15:0]`.
/// `.e2m1x4`, whose result is itself only sixteen bits wide, instead gathers
/// each field from one byte of each halfword: `(a, b)` from bytes 3 and 1,
/// `(e, f)` from bytes 2 and 0.
pub fn ptx_cvt_rs_randoms(rbits: u32, format: NarrowFloatFormat) -> [u16; 4] {
    let (high, low) = if format.storage_bits == 8 {
        ((rbits >> 16) as u16, rbits as u16)
    } else {
        (
            ((rbits >> 8) & 0xff) as u16 | ((((rbits >> 24) & 0xff) as u16) << 8),
            (rbits & 0xff) as u16 | ((((rbits >> 16) & 0xff) as u16) << 8),
        )
    };
    [high.reverse_bits(), high, low.reverse_bits(), low]
}

/// PTX `cvt.rn{.relu}.f16x2.<narrow>x2`.
///
/// Every NaN encoding widens to the canonical `0x7fff` payload with the sign
/// dropped, and `.relu` does not zero it.
pub fn ptx_cvt_unpack_narrow_x2_f16x2<const CLAMP_NEGATIVE: bool>(
    value: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8| -> u16 {
        match crate::narrow_float_bits_to_f32_checked(code, format) {
            None => WIDE_CANONICAL_NAN,
            Some(decoded) if CLAMP_NEGATIVE && decoded.is_sign_negative() => 0,
            Some(decoded) => crate::f32_to_fp16_bits(decoded),
        }
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high)) << 16) | u32::from(convert(low))
}

/// PTX `cvt.rn{.relu}{.satfinite}.bf16x2.<narrow>x2`.
///
/// `.satfinite` is observable only for `.e5m2x2`, whose infinities become the
/// signed greatest finite bfloat16; `.e4m3x2` and `.e2m1x2` have no infinities,
/// so the qualifier is numerically inert there.
pub fn ptx_cvt_unpack_narrow_x2_bf16x2<const CLAMP_NEGATIVE: bool, const SATURATE_FINITE: bool>(
    value: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8| -> u16 {
        match crate::narrow_float_bits_to_f32_checked(code, format) {
            None => WIDE_CANONICAL_NAN,
            Some(decoded) if CLAMP_NEGATIVE && decoded.is_sign_negative() => 0,
            Some(decoded) if SATURATE_FINITE && decoded.is_infinite() => {
                saturate_bf16(decoded.is_sign_negative())
            }
            Some(decoded) => crate::f32_to_bf16_bits(decoded),
        }
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high)) << 16) | u32::from(convert(low))
}

/// PTX `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.<narrow>x2`.
///
/// The scale operand is a `ue8m0x2`: each result half is its own element
/// multiplied by the power of two named by the *corresponding* scale byte.
/// A `0xff` scale byte is the E8M0 NaN and makes that half a NaN whatever the
/// element is; `.relu` does not zero it. `.satfinite` clamps a half that
/// *rounds* to an infinity, so it also catches a finite product that overflows
/// bfloat16, not only an infinite element.
///
/// The product is formed in binary32 and narrowed once. That is exact, not a
/// convenience: an element carries at most four significant bits and the
/// factor is an exact power of two in `2^-127 ..= 2^127`. Every product within
/// binary32's range is exact; a product outside it has already overflowed
/// bfloat16. `every_reachable_scaled_product_is_exact_or_already_overflows_bf16`
/// pins that over all 197625 reachable pairs.
pub fn ptx_cvt_unpack_scaled_bf16x2<const CLAMP_NEGATIVE: bool, const SATURATE_FINITE: bool>(
    value: u16,
    scale: u16,
    format: NarrowFloatFormat,
) -> u32 {
    let convert = |code: u8, scale_code: u8| -> u16 {
        let Some(decoded) = crate::narrow_float_bits_to_f32_checked(code, format) else {
            return WIDE_CANONICAL_NAN;
        };
        if scale_code == E8M0_NAN_CODE {
            return WIDE_CANONICAL_NAN;
        }
        let scaled = decoded * crate::float8_e8m0fnu_bits_to_f32(scale_code);
        if CLAMP_NEGATIVE && scaled.is_sign_negative() {
            return 0;
        }
        let rounded = f32_to_bf16_bits(scaled);
        if SATURATE_FINITE && rounded & 0x7fff == BF16_INFINITY {
            return saturate_bf16(rounded & 0x8000 != 0);
        }
        rounded
    };
    let (high, low) = split_packed_pair(value, format);
    (u32::from(convert(high, (scale >> 8) as u8)) << 16) | u32::from(convert(low, scale as u8))
}

/// S2F6 is a signed two's-complement byte in units of 1/64, not FP6.
/// Each input has its own UE8M0 scale; NaN maps to positive MAX_NORM.
pub fn ptx_cvt_pack_s2f6x2(high: f32, low: f32, scale: u16, relu: bool) -> u16 {
    let convert = |value: f32, scale: u8| {
        let value = value / crate::float8_e8m0fnu_bits_to_f32(scale);
        if value.is_nan() {
            return 127_u8;
        }
        let value = if relu { value.max(0.0) } else { value };
        // Power-of-two scaling is exact near every fixed-point midpoint;
        // values overflowing binary32 already require fixed-point saturation.
        (value * 64.0).round_ties_even().clamp(-128.0, 127.0) as i8 as u8
    };
    (u16::from(convert(high, (scale >> 8) as u8)) << 8) | u16::from(convert(low, scale as u8))
}

/// Decode each signed byte exactly, scale it, then use the shared BF16 codec.
pub fn ptx_cvt_unpack_s2f6x2(value: u16, scale: u16, relu: bool, satfinite: bool) -> u32 {
    let convert = |code: u8, scale: u8| {
        let decoded = f32::from(code as i8) / 64.0;
        let scaled = decoded * crate::float8_e8m0fnu_bits_to_f32(scale);
        ptx_cvt_f32_to_bf16(scaled, PtxFloatRounding::NearestEven, relu, satfinite)
    };
    (u32::from(convert((value >> 8) as u8, (scale >> 8) as u8)) << 16)
        | u32::from(convert(value as u8, scale as u8))
}

/// PTX `cvt.{rz,rp}{.satfinite}.ue8m0x2.f32`, one exponent per primary.
pub fn ptx_cvt_pack_e8m0x2_f32<const ROUND_UP: bool, const SATURATE: bool>(
    high: f32,
    low: f32,
) -> u16 {
    let encode = crate::f32_to_float8_e8m0fnu_bits_rounded::<ROUND_UP, SATURATE>;
    (u16::from(encode(high)) << 8) | u16::from(encode(low))
}

/// PTX `cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2`.
///
/// Widening bfloat16 to binary32 is exact, subnormals included, so this reuses
/// the binary32 exponent encoder rather than repeating it.
pub fn ptx_cvt_pack_e8m0x2_bf16x2<const ROUND_UP: bool, const SATURATE: bool>(value: u32) -> u16 {
    ptx_cvt_pack_e8m0x2_f32::<ROUND_UP, SATURATE>(
        bf16_bits_to_f32((value >> 16) as u16),
        bf16_bits_to_f32(value as u16),
    )
}

/// PTX `cvt.rn.bf16x2.ue8m0x2`, widening each exponent to a bfloat16.
///
/// Every finite decode is an exact power of two in `2^-127 ..= 2^127`, so the
/// binary32 intermediate is exact and the narrowing rounds once — the least of
/// them lands mid-subnormal in bfloat16 and still converts exactly.
pub fn ptx_cvt_unpack_e8m0x2_bf16x2(value: u16) -> u32 {
    let convert = |code: u8| -> u16 {
        if code == E8M0_NAN_CODE {
            WIDE_CANONICAL_NAN
        } else {
            f32_to_bf16_bits(crate::float8_e8m0fnu_bits_to_f32(code))
        }
    };
    (u32::from(convert((value >> 8) as u8)) << 16) | u32::from(convert(value as u8))
}

/// Split one packed pair into its upper and lower element codes.
fn split_packed_pair(value: u16, format: NarrowFloatFormat) -> (u8, u8) {
    let mask = (1_u16 << format.width_bits) - 1;
    (
        ((value >> format.storage_bits) & mask) as u8,
        (value & mask) as u8,
    )
}

fn saturate_bf16(negative: bool) -> u16 {
    if negative {
        0x8000 | BF16_MAX_FINITE
    } else {
        BF16_MAX_FINITE
    }
}

/// The binary16 and bfloat16 payload every narrow-float NaN widens to.
const WIDE_CANONICAL_NAN: u16 = 0x7fff;

/// Greatest finite bfloat16 magnitude, without the sign bit.
const BF16_MAX_FINITE: u16 = 0x7f7f;

/// bfloat16 infinity, without the sign bit.
const BF16_INFINITY: u16 = 0x7f80;

/// The single E8M0 NaN encoding.
const E8M0_NAN_CODE: u8 = 0xff;

pub fn hmin2_bf16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::Bf16, false)
}

pub fn hmin2_f16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::F16, false)
}

pub fn hmax2_bf16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::Bf16, true)
}

pub fn hmax2_f16(lhs: u32, rhs: u32) -> u32 {
    low_minmax2(lhs, rhs, LowPrecisionFormat::F16, true)
}

fn low_minmax2(lhs: u32, rhs: u32, format: LowPrecisionFormat, maximum: bool) -> u32 {
    let component = |shift| {
        u32::from(low_minmax(
            (lhs >> shift) as u16,
            (rhs >> shift) as u16,
            format,
            false,
            false,
            false,
            maximum,
        ))
    };
    component(0) | (component(16) << 16)
}

/// Min/max selects an existing half value; comparing ordered encodings avoids
/// host floating-point conversion or rounding, including for BF16 subnormals.
pub(crate) fn low_minmax(
    lhs: u16,
    rhs: u16,
    format: LowPrecisionFormat,
    ftz: bool,
    propagate_nan: bool,
    xor_sign: bool,
    maximum: bool,
) -> u16 {
    let sign = (lhs ^ rhs) & 0x8000;
    let prepare = |bits| {
        let bits = if ftz {
            flush_subnormal_f16_bits(bits)
        } else {
            bits
        };
        if xor_sign {
            bits & 0x7fff
        } else {
            bits
        }
    };
    let (lhs, rhs) = (prepare(lhs), prepare(rhs));
    let (lhs_nan, rhs_nan) = (
        lhs & 0x7fff > format.infinity(),
        rhs & 0x7fff > format.infinity(),
    );
    if (lhs_nan && rhs_nan) || (propagate_nan && (lhs_nan || rhs_nan)) {
        return WIDE_CANONICAL_NAN;
    }
    let order = |bits: u16| {
        if bits & 0x8000 != 0 {
            !bits
        } else {
            bits ^ 0x8000
        }
    };
    let result = if lhs_nan {
        rhs
    } else if rhs_nan {
        lhs
    } else if (order(lhs) > order(rhs)) == maximum {
        lhs
    } else {
        rhs
    };
    if xor_sign {
        (result & 0x7fff) | sign
    } else {
        result
    }
}

pub fn fp8x4_e4m3_from_float4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    f32_to_float8_e4m3fn_bits(x) as u32
        | ((f32_to_float8_e4m3fn_bits(y) as u32) << 8)
        | ((f32_to_float8_e4m3fn_bits(z) as u32) << 16)
        | ((f32_to_float8_e4m3fn_bits(w) as u32) << 24)
}

fn pack_f32x2(low: f32, high: f32) -> u64 {
    u64::from(low.to_bits()) | (u64::from(high.to_bits()) << 32)
}

fn pack_f32x2_result(low: f32, high: f32) -> u64 {
    pack_f32x2(
        cuda_canonicalize_nan_f32(low),
        cuda_canonicalize_nan_f32(high),
    )
}

pub fn add_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_x = f32::from_bits(lhs as u32);
    let lhs_y = f32::from_bits((lhs >> 32) as u32);
    let rhs_x = f32::from_bits(rhs as u32);
    let rhs_y = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { add_f32_ftz } else { add_f32 };
    pack_f32x2_result(operation(lhs_x, rhs_x, mode), operation(lhs_y, rhs_y, mode))
}

pub fn fma_f32x2(lhs: u64, rhs: u64, addend: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_x = f32::from_bits(lhs as u32);
    let lhs_y = f32::from_bits((lhs >> 32) as u32);
    let rhs_x = f32::from_bits(rhs as u32);
    let rhs_y = f32::from_bits((rhs >> 32) as u32);
    let addend_x = f32::from_bits(addend as u32);
    let addend_y = f32::from_bits((addend >> 32) as u32);
    let operation = if ftz { fma_f32_ftz } else { fma_f32 };
    pack_f32x2_result(
        operation(lhs_x, rhs_x, addend_x, mode),
        operation(lhs_y, rhs_y, addend_y, mode),
    )
}

pub fn sub_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_low = f32::from_bits(lhs as u32);
    let lhs_high = f32::from_bits((lhs >> 32) as u32);
    let rhs_low = f32::from_bits(rhs as u32);
    let rhs_high = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { sub_f32_ftz } else { sub_f32 };
    pack_f32x2_result(
        operation(lhs_low, rhs_low, mode),
        operation(lhs_high, rhs_high, mode),
    )
}

pub fn mul_f32x2(lhs: u64, rhs: u64, mode: F32RoundingMode, ftz: bool) -> u64 {
    let lhs_low = f32::from_bits(lhs as u32);
    let lhs_high = f32::from_bits((lhs >> 32) as u32);
    let rhs_low = f32::from_bits(rhs as u32);
    let rhs_high = f32::from_bits((rhs >> 32) as u32);
    let operation = if ftz { mul_f32_ftz } else { mul_f32 };
    pack_f32x2_result(
        operation(lhs_low, rhs_low, mode),
        operation(lhs_high, rhs_high, mode),
    )
}

// ---------------------------------------------------------------------------
// Scalar PTX `cvt` numeric cores.
//
// Rules follow the PTX ISA section "Data Movement and Conversion Instructions:
// cvt". Comments distinguish measured NaN/FTZ behavior where the specification
// leaves representation details open; compiler defects are not ISA semantics.
// ---------------------------------------------------------------------------

/// PTX `.irnd` integer rounding modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtxIntegerRounding {
    /// `.rni` — nearest integer, ties to even.
    NearestEven,
    /// `.rzi` — nearest integer toward zero.
    Zero,
    /// `.rmi` — nearest integer toward negative infinity.
    NegativeInfinity,
    /// `.rpi` — nearest integer toward positive infinity.
    PositiveInfinity,
}

/// PTX `.frnd` / `.frnd2` floating-point rounding modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtxFloatRounding {
    /// `.rn` — nearest, ties to even.
    NearestEven,
    /// `.rna` — nearest, ties away from zero.  Only `cvt.rna.tf32.f32` spells it.
    NearestAway,
    /// `.rz` — toward zero.
    Zero,
    /// `.rm` — toward negative infinity.
    NegativeInfinity,
    /// `.rp` — toward positive infinity.
    PositiveInfinity,
}

/// The `f32` NaN every measured `cvt` form that canonicalizes produces.
///
/// This is the same all-ones payload `cuda_canonical_nan_f32` already uses.
const PTX_CVT_CANONICAL_NAN_F32: u32 = 0x7fff_ffff;

/// The `.f16` / `.bf16` NaN every measured narrowing `cvt` form produces.
///
/// Hardware drops the sign and the payload: `cvt.rn.bf16.f32`,
/// `cvt.rz.bf16.f32`, and the `.f16` forms all answer `0x7fff` for `+NaN`,
/// `-NaN` and signalling NaN, with or without `.relu` / `.satfinite`.
const PTX_CVT_CANONICAL_NAN_NARROW: u16 = 0x7fff;

/// One tf32 unit in the last place, in `f32` bit positions.
const TF32_ULP: u32 = 0x2000;

/// Round `value` to an integral `f32`, applying `.ftz` to the source first.
///
/// `.ftz` is observable here: without it `cvt.rmi.f32.f32` of the smallest
/// negative subnormal answers `-1.0` and `cvt.rpi.f32.f32` of the smallest
/// positive subnormal answers `1.0`; with it both answer signed zero.
pub fn ptx_cvt_integral_f32(value: f32, rounding: PtxIntegerRounding, ftz: bool) -> f32 {
    let value = if ftz {
        flush_subnormal_f32(value)
    } else {
        value
    };
    match rounding {
        PtxIntegerRounding::NearestEven => value.round_ties_even(),
        PtxIntegerRounding::Zero => value.trunc(),
        PtxIntegerRounding::NegativeInfinity => value.floor(),
        PtxIntegerRounding::PositiveInfinity => value.ceil(),
    }
}

/// `cvt.irnd{.ftz}.f32.f32`: integral rounding inside `f32`.
///
/// Measured: every NaN input answers `0x7fffffff`, with and without `.ftz`.
pub fn ptx_cvt_integral_f32_to_f32(value: f32, rounding: PtxIntegerRounding, ftz: bool) -> f32 {
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    ptx_cvt_integral_f32(value, rounding, ftz)
}

/// Round `value` to an integral `f64`.  `.ftz` is illegal without an `f32`.
pub fn ptx_cvt_integral_f64(value: f64, rounding: PtxIntegerRounding) -> f64 {
    match rounding {
        PtxIntegerRounding::NearestEven => value.round_ties_even(),
        PtxIntegerRounding::Zero => value.trunc(),
        PtxIntegerRounding::NegativeInfinity => value.floor(),
        PtxIntegerRounding::PositiveInfinity => value.ceil(),
    }
}

/// `cvt.irnd.f64.f64`: integral rounding inside `f64`.
///
/// Measured: unlike the `f32` form this preserves the NaN sign and payload and
/// only quiets a signalling NaN.
pub fn ptx_cvt_integral_f64_to_f64(value: f64, rounding: PtxIntegerRounding) -> f64 {
    if value.is_nan() {
        return f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000);
    }
    ptx_cvt_integral_f64(value, rounding)
}

/// `cvt{.ftz}.f32.f32`: the identity conversion, which `.ftz` makes observable.
///
/// Measured: `.ftz` flushes a subnormal source to signed zero *and*
/// canonicalizes NaN; without it the value passes through bit-for-bit.
pub fn ptx_cvt_f32_to_f32(value: f32, ftz: bool) -> f32 {
    if !ftz {
        return value;
    }
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    flush_subnormal_f32(value)
}

/// Round the exact integer `magnitude` to `mantissa_bits` of significand.
///
/// Returns `(head, exponent)` with `magnitude ~ head << exponent` and
/// `head < 1 << mantissa_bits`, so the caller can rebuild the value exactly.
/// `away_from_zero` selects between truncation and the next magnitude up,
/// which is how `.rz` / `.rm` / `.rp` differ for `u64`/`s64` sources whose
/// value needs more bits than the destination format has.
fn round_integer_magnitude(magnitude: u64, mantissa_bits: u32, away_from_zero: bool) -> (u64, u32) {
    if magnitude == 0 {
        return (0, 0);
    }
    let significant = u64::BITS - magnitude.leading_zeros();
    if significant <= mantissa_bits {
        return (magnitude, 0);
    }
    let mut exponent = significant - mantissa_bits;
    let mut head = magnitude >> exponent;
    if away_from_zero && magnitude & ((1u64 << exponent) - 1) != 0 {
        head += 1;
        if head == 1u64 << mantissa_bits {
            head >>= 1;
            exponent += 1;
        }
    }
    (head, exponent)
}

fn rounds_away_from_zero(rounding: PtxFloatRounding, negative: bool) -> bool {
    match rounding {
        PtxFloatRounding::Zero => false,
        PtxFloatRounding::NegativeInfinity => negative,
        PtxFloatRounding::PositiveInfinity => !negative,
        // The nearest modes never reach here; their callers answer first.
        PtxFloatRounding::NearestEven | PtxFloatRounding::NearestAway => {
            unreachable!("nearest rounding does not select a directed magnitude")
        }
    }
}

/// `cvt.frnd.f32.{u,s}*`: integer to `f32` with a directed rounding modifier.
///
/// `.rn` is Rust's `as`, which is round-to-nearest-even.  The directed modes
/// are computed from the exact integer magnitude rather than from the
/// nearest-rounded value, because for a 64-bit source the two differ by more
/// than one `f32` ulp.
pub fn ptx_cvt_integer_to_f32(magnitude: u64, negative: bool, rounding: PtxFloatRounding) -> f32 {
    // `.frnd` is {rn, rz, rm, rp}; `.rna` has no integer-to-float spelling, so
    // it fails closed here rather than silently aliasing onto `.rn`.
    assert!(
        rounding != PtxFloatRounding::NearestAway,
        "PTX has no cvt.rna integer-to-f32 form"
    );
    if rounding == PtxFloatRounding::NearestEven {
        let nearest = magnitude as f32;
        return if negative { -nearest } else { nearest };
    }
    let (head, exponent) = round_integer_magnitude(
        magnitude,
        f32::MANTISSA_DIGITS,
        rounds_away_from_zero(rounding, negative),
    );
    let scale = f32::from_bits((127 + exponent) << 23);
    let result = (head as f32) * scale;
    if negative {
        -result
    } else {
        result
    }
}

/// `cvt.frnd.f64.{u,s}*`: integer to `f64` with a directed rounding modifier.
pub fn ptx_cvt_integer_to_f64(magnitude: u64, negative: bool, rounding: PtxFloatRounding) -> f64 {
    assert!(
        rounding != PtxFloatRounding::NearestAway,
        "PTX has no cvt.rna integer-to-f64 form"
    );
    if rounding == PtxFloatRounding::NearestEven {
        let nearest = magnitude as f64;
        return if negative { -nearest } else { nearest };
    }
    let (head, exponent) = round_integer_magnitude(
        magnitude,
        f64::MANTISSA_DIGITS,
        rounds_away_from_zero(rounding, negative),
    );
    let scale = f64::from_bits(((1023 + exponent) as u64) << 52);
    let result = (head as f64) * scale;
    if negative {
        -result
    } else {
        result
    }
}

fn ptx_cvt_integer_to_low(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
    format: LowPrecisionFormat,
) -> u16 {
    if rounding == PtxFloatRounding::NearestEven {
        return encode_exact_low(negative, &[magnitude], 0, format, false);
    }
    let away = rounds_away_from_zero(rounding, negative);
    let (head, exponent) =
        round_integer_magnitude(magnitude, format.fraction_bits() as u32 + 1, away);
    // The directed result already has the target precision. Encoding it is
    // exact except for F16 exponent overflow, which still needs its direction.
    let bits = encode_exact_low(negative, &[head], exponent as i32, format, false);
    if !away && narrow_is_infinite(bits, format.infinity()) {
        narrow_step_toward_zero(bits)
    } else {
        bits
    }
}

pub(crate) fn ptx_cvt_integer_to_f16(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
) -> u16 {
    ptx_cvt_integer_to_low(magnitude, negative, rounding, LowPrecisionFormat::F16)
}

pub(crate) fn ptx_cvt_integer_to_bf16(
    magnitude: u64,
    negative: bool,
    rounding: PtxFloatRounding,
) -> u16 {
    ptx_cvt_integer_to_low(magnitude, negative, rounding, LowPrecisionFormat::Bf16)
}

/// `cvt.frnd{.ftz}.f32.f64`: narrowing with a directed rounding modifier.
///
/// Measured: a NaN source keeps its sign, is quieted, and carries its high
/// payload bits down. At the normal/subnormal boundary, measured `.ftz`
/// behavior detects tininess after rounding to 24-bit precision without
/// restricting the exponent, not after gradual-underflow rounding.
pub fn ptx_cvt_f64_to_f32(value: f64, rounding: PtxFloatRounding, ftz: bool) -> f32 {
    if value.is_nan() {
        let bits = value.to_bits();
        let sign = ((bits >> 32) as u32) & 0x8000_0000;
        let payload = ((bits >> 29) as u32) & 0x003f_ffff;
        return f32::from_bits(sign | 0x7fc0_0000 | payload);
    }
    let nearest = value as f32;
    let exact = f64::from(nearest) == value;
    let rounded = if exact {
        nearest
    } else {
        match rounding {
            PtxFloatRounding::NearestEven | PtxFloatRounding::NearestAway => nearest,
            PtxFloatRounding::Zero => {
                if value > 0.0 {
                    if f64::from(nearest) > value {
                        nearest.next_down()
                    } else {
                        nearest
                    }
                } else if f64::from(nearest) < value {
                    nearest.next_up()
                } else {
                    nearest
                }
            }
            PtxFloatRounding::NegativeInfinity => {
                if f64::from(nearest) > value {
                    nearest.next_down()
                } else {
                    nearest
                }
            }
            PtxFloatRounding::PositiveInfinity => {
                if f64::from(nearest) < value {
                    nearest.next_up()
                } else {
                    nearest
                }
            }
        }
    };
    if ftz {
        flush_f32_result(rounded, || {
            // Only the MIN_NORMAL boundary enters here. Exact scaling moves
            // it into the normal range so the existing converter retains all
            // 24 significand bits before the tininess comparison.
            ptx_cvt_f64_to_f32(value * 2.0, rounding, false).abs() < 2.0 * f32::MIN_POSITIVE
        })
    } else {
        rounded
    }
}

/// `cvt{.ftz}.f32.f16` / `cvt{.ftz}.f32.bf16`: widening with the `.ftz` axis.
///
/// Measured: `.ftz` on these forms canonicalizes NaN to `0x7fffffff` and
/// flushes a subnormal `f32` result, which only bf16 sources can produce.
pub fn ptx_cvt_widen_to_f32(value: f32, ftz: bool) -> f32 {
    if !ftz {
        return value;
    }
    if value.is_nan() {
        return f32::from_bits(PTX_CVT_CANONICAL_NAN_F32);
    }
    flush_subnormal_f32(value)
}

/// `cvt{.ftz}.f64.f32`: widening, where `.ftz` still applies to the source.
pub fn ptx_cvt_f32_to_f64(value: f32, ftz: bool) -> f64 {
    f64::from(ptx_cvt_widen_to_f32(value, ftz))
}

/// One step of the `.f16` / `.bf16` magnitude toward zero.
fn narrow_step_toward_zero(bits: u16) -> u16 {
    let sign = bits & 0x8000;
    let magnitude = bits & 0x7fff;
    if magnitude == 0 {
        bits
    } else {
        sign | (magnitude - 1)
    }
}

/// Round to `.f16` or `.bf16`. Directed rounding adjusts the nearest
/// encoding by at most one magnitude ULP, including zero and finite overflow.
/// The comparison retains the source precision; the nearest payload must have
/// been rounded directly from that source, not through a narrower intermediate.
fn ptx_cvt_narrow_rounded(
    value: f64,
    rounding: PtxFloatRounding,
    nearest: u16,
    decode: fn(u16) -> f32,
) -> u16 {
    match rounding {
        PtxFloatRounding::NearestEven => return nearest,
        PtxFloatRounding::Zero
        | PtxFloatRounding::NegativeInfinity
        | PtxFloatRounding::PositiveInfinity => {}
        PtxFloatRounding::NearestAway => {
            unreachable!("half CVT has no .rna spelling")
        }
    }
    let decoded = f64::from(decode(nearest));
    let away = rounds_away_from_zero(rounding, value.is_sign_negative());
    if away && decoded.abs() < value.abs() {
        nearest + 1
    } else if !away && decoded.abs() > value.abs() {
        narrow_step_toward_zero(nearest)
    } else {
        nearest
    }
}

/// FP64 narrowing preserves the NaN sign/high payload and quiets it (B200).
/// Finite inputs reuse the exact dyadic codec, avoiding FP32 double rounding.
pub(crate) fn ptx_cvt_f64_to_low(
    value: f64,
    rounding: PtxFloatRounding,
    format: LowPrecisionFormat,
) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) as u16) & 0x8000;
    let fraction_bits = format.fraction_bits();
    if value.is_nan() {
        let payload = ((bits >> (52 - fraction_bits)) as u16) & ((1 << fraction_bits) - 1);
        return sign | format.infinity() | payload | (1 << (fraction_bits - 1));
    }
    if value.is_infinite() {
        return sign | format.infinity();
    }
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let significand = (bits & ((1_u64 << 52) - 1)) | (u64::from(exponent != 0) << 52);
    let nearest = encode_exact_low(
        sign != 0,
        &[significand],
        exponent.max(1) - 1023 - 52,
        format,
        false,
    );
    let decode = match format {
        LowPrecisionFormat::F16 => crate::fp16_bits_to_f32,
        LowPrecisionFormat::Bf16 => bf16_bits_to_f32,
    };
    ptx_cvt_narrow_rounded(value, rounding, nearest, decode)
}

/// Half-to-FP64 is exact; NaNs retain sign/payload and become quiet (B200).
pub(crate) fn ptx_cvt_low_to_f64(bits: u16, format: LowPrecisionFormat) -> f64 {
    if bits & 0x7fff > format.infinity() {
        let fraction_bits = format.fraction_bits();
        let payload = u64::from(bits & ((1 << fraction_bits) - 1)) << (52 - fraction_bits);
        return f64::from_bits((u64::from(bits & 0x8000) << 48) | 0x7ff8_0000_0000_0000 | payload);
    }
    f64::from(decode_low(bits, format))
}

fn narrow_is_infinite(bits: u16, exponent_mask: u16) -> bool {
    bits & 0x7fff == exponent_mask
}

/// Shared body of `cvt.frnd2{.relu}{.satfinite}.{f16,bf16}.f32`.
fn ptx_cvt_narrow(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
    encode: fn(f32) -> u16,
    decode: fn(u16) -> f32,
    infinity: u16,
) -> u16 {
    if value.is_nan() {
        // Measured: `.relu` and `.satfinite` both leave this canonical NaN.
        return PTX_CVT_CANONICAL_NAN_NARROW;
    }
    let mut bits = ptx_cvt_narrow_rounded(f64::from(value), rounding, encode(value), decode);
    if satfinite && narrow_is_infinite(bits, infinity) {
        bits = narrow_step_toward_zero(bits);
    }
    if relu && bits & 0x8000 != 0 {
        bits = 0;
    }
    bits
}

/// `cvt.frnd2{.relu}{.satfinite}.f16.f32`.
pub fn ptx_cvt_f32_to_f16(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u16 {
    ptx_cvt_narrow(
        value,
        rounding,
        relu,
        satfinite,
        crate::f32_to_fp16_bits,
        crate::fp16_bits_to_f32,
        0x7c00,
    )
}

/// `cvt.frnd2{.relu}{.satfinite}.bf16.f32`.
///
/// This is a separate specialization from `Cvt<F32, Bf16, Rn>`, which the tile
/// lowering emits for bf16 elementwise and reduction rounding, but the two
/// agree on NaN: `encode_bf16` canonicalizes to `0x7fffffff` before narrowing,
/// so it also answers the `0x7fff` this family measured on hardware.
pub fn ptx_cvt_f32_to_bf16(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u16 {
    ptx_cvt_narrow(
        value,
        rounding,
        relu,
        satfinite,
        f32_to_bf16_bits,
        bf16_bits_to_f32,
        0x7f80,
    )
}

/// Half stochastic rounding reuses the RZ codec and its adjacent value.
///
/// Comparing the discarded fraction plus the supplied random fraction with
/// one is exactly the PTX carry test. All operands are binary32 values or
/// powers of two; binary64 keeps this comparison exact even at half subnormal
/// boundaries. There is no random generator or mutable rounding state.
pub fn ptx_cvt_half_rs<const RANDOM_BITS: u32>(
    value: f32,
    random: u16,
    relu: bool,
    satfinite: bool,
    convert: fn(f32, PtxFloatRounding, bool, bool) -> u16,
    decode: fn(u16) -> f32,
) -> u16 {
    let truncated = convert(value, PtxFloatRounding::Zero, relu, satfinite);
    if !value.is_finite() || (relu && value.is_sign_negative()) {
        return truncated;
    }
    let magnitude = truncated & 0x7fff;
    let base = f64::from(decode(magnitude));
    let next = f64::from(decode(magnitude + 1));
    if satfinite && next.is_infinite() {
        return truncated;
    }
    let step = if next.is_infinite() {
        // Infinity's rounding boundary is the next value at this precision,
        // not an infinite distance from MAX_NORM.
        base - f64::from(decode(magnitude - 1))
    } else {
        next - base
    };
    let units = f64::from(1_u32 << RANDOM_BITS);
    let carry = (f64::from(value.abs()) - base) * units >= step * (units - f64::from(random));
    truncated + u16::from(carry)
}

/// PTX 9.4 `.pzo` post-processing for an f16/bf16 conversion result.
///
/// The qualifier acts after conversion: only a negative-zero result loses its
/// sign. Applying this to the input would miss negative values that round to
/// zero, so callers normalize the encoded destination instead.
pub fn ptx_cvt_pzo_u16(bits: u16) -> u16 {
    if bits & 0x7fff == 0 {
        0
    } else {
        bits
    }
}

/// PTX 9.4 `.pzo` post-processing for a tf32 conversion result.
pub fn ptx_cvt_pzo_u32(bits: u32) -> u32 {
    if bits & 0x7fff_ffff == 0 {
        0
    } else {
        bits
    }
}

/// PTX 9.4 `.pzo` post-processing for two packed signed narrow floats.
///
/// Six-bit formats occupy padded byte fields while E2M1 uses packed nibbles,
/// so the format owns both the sign-bit position and the field stride. Only a
/// field whose encoded magnitude is zero loses its sign.
pub fn ptx_cvt_pzo_narrow_x2(bits: u16, format: NarrowFloatFormat) -> u16 {
    debug_assert!(format.signed);
    let field_mask = (1_u16 << format.storage_bits) - 1;
    let value_mask = (1_u16 << format.width_bits) - 1;
    let sign_mask = u16::from(format.sign_mask());
    let normalize = |field: u16| {
        if field & value_mask == sign_mask {
            field & !sign_mask
        } else {
            field
        }
    };
    let low = normalize(bits & field_mask);
    let high = normalize((bits >> format.storage_bits) & field_mask);
    (high << format.storage_bits) | low
}

/// `cvt.rna{.satfinite}.tf32.f32` and `cvt.frnd2{.satfinite}{.relu}.tf32.f32`.
///
/// The destination is a `.b32` register holding the source's sign and exponent
/// with a 10-bit fraction; the low 13 bits are always zero.  Measured
/// behaviour that the ISA prose does not state:
///
/// * `.rn` and `.rz` answer the canonical tf32 NaN `0x7fffe000` for every NaN
///   input, dropping the sign.
/// * `.rna` instead rounds a NaN arithmetically, so a signalling NaN whose
///   payload lives in the discarded bits becomes infinity; the carry saturates
///   at `0x7fffe000` rather than reaching the sign bit.
/// * `.satfinite` clamps by stepping one tf32 ulp down from any result whose
///   exponent field is all ones, which is why `.rna.satfinite` of a quiet NaN
///   is `0x7fbfe000` rather than a NaN-class constant.
pub fn ptx_cvt_f32_to_tf32(
    value: f32,
    rounding: PtxFloatRounding,
    relu: bool,
    satfinite: bool,
) -> u32 {
    let source = value.to_bits();
    let sign = source & 0x8000_0000;
    let magnitude = source & 0x7fff_ffff;
    if rounding != PtxFloatRounding::NearestAway && value.is_nan() {
        // Measured: `.relu` and `.satfinite` both leave this canonical NaN.
        return 0x7fff_e000;
    }
    let mut bits = match rounding {
        PtxFloatRounding::NearestAway => {
            let carried = (magnitude + 0x1000) & !(TF32_ULP - 1);
            sign | carried.min(0x7fff_e000)
        }
        PtxFloatRounding::Zero => sign | (magnitude & !(TF32_ULP - 1)),
        PtxFloatRounding::NearestEven => crate::f32_to_tf32(value).to_bits(),
        // The tf32 line admits only {.rna, .rn, .rz}; the directed modes fail
        // closed rather than silently answering `.rn`.
        PtxFloatRounding::NegativeInfinity | PtxFloatRounding::PositiveInfinity => {
            unreachable!("PTX cvt.tf32.f32 admits only .rna, .rn and .rz")
        }
    };
    if satfinite && bits & 0x7f80_0000 == 0x7f80_0000 {
        bits = (bits & 0x8000_0000) | ((bits & 0x7fff_ffff) - TF32_ULP);
    }
    if relu && bits & 0x8000_0000 != 0 {
        bits = 0;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_division_matches_python_sign_rules() {
        assert_eq!(floor_div_i64(7, 3).unwrap(), 2);
        assert_eq!(floor_div_i64(-7, 3).unwrap(), -3);
        assert_eq!(floor_div_i64(7, -3).unwrap(), -3);
        assert_eq!(floor_mod_i64(-7, 3).unwrap(), 2);
        assert_eq!(floor_mod_i64(7, -3).unwrap(), -2);
    }

    #[test]
    fn fns_walks_set_bits_in_both_directions() {
        let mask = 0b10110_u32;
        assert_eq!(ptx_fns_b32(mask, 1, 1), 1);
        assert_eq!(ptx_fns_b32(mask, 1, 2), 2);
        assert_eq!(ptx_fns_b32(mask, 4, -2), 2);
        assert_eq!(ptx_fns_b32(mask, 0, -1), u32::MAX);
    }

    #[test]
    fn packed_f32_operations_preserve_lane_order() {
        let lhs = 2.0_f32.to_bits() as u64 | ((3.0_f32.to_bits() as u64) << 32);
        let rhs = 4.0_f32.to_bits() as u64 | ((5.0_f32.to_bits() as u64) << 32);
        let addend = 1.0_f32.to_bits() as u64 | ((2.0_f32.to_bits() as u64) << 32);
        let product = mul_f32x2(lhs, rhs, F32RoundingMode::Nearest, true);
        assert_eq!(f32::from_bits(product as u32), 8.0);
        assert_eq!(f32::from_bits((product >> 32) as u32), 15.0);
        let sum = add_f32x2(lhs, rhs, F32RoundingMode::Nearest, false);
        assert_eq!(f32::from_bits(sum as u32), 6.0);
        assert_eq!(f32::from_bits((sum >> 32) as u32), 8.0);
        let fused = fma_f32x2(lhs, rhs, addend, F32RoundingMode::Nearest, false);
        assert_eq!(f32::from_bits(fused as u32), 9.0);
        assert_eq!(f32::from_bits((fused >> 32) as u32), 17.0);
    }

    #[test]
    fn ptx_move_vector_layout_is_low_lane_first_and_bit_exact() {
        let b16 = [0x0123_u16, 0x4567, 0x89ab, 0xcdef];
        assert_eq!(ptx_mov_pack_b16x4(b16), 0xcdef_89ab_4567_0123);
        assert_eq!(ptx_mov_unpack_b16x4(0xcdef_89ab_4567_0123), b16);

        let b32 = [0x0123_4567_u32, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
        let packed = [0x89ab_cdef_0123_4567, 0x7654_3210_fedc_ba98];
        assert_eq!(ptx_mov_pack_b32x4(b32), packed);
        assert_eq!(ptx_mov_unpack_b32x4(packed), b32);
        assert_eq!(ptx_mov_unpack_b64x2(packed), packed);
    }

    #[test]
    fn ptx_cvt_pack_saturates_fields_places_b_low_and_preserves_c_high_bits() {
        assert_eq!(ptx_cvt_pack::<16, false>(70_000, -3, 0), 0xffff_0000);
        assert_eq!(ptx_cvt_pack::<16, true>(40_000, -40_000, 0), 0x7fff_8000);
        assert_eq!(ptx_cvt_pack::<8, false>(300, -1, 0x89ab_cdef), 0xcdef_ff00);
        assert_eq!(ptx_cvt_pack::<8, true>(200, -200, 0x0123_4567), 0x4567_7f80);
        assert_eq!(ptx_cvt_pack::<4, false>(17, -1, 0xdead_beef), 0xad_beef_f0);
        assert_eq!(ptx_cvt_pack::<4, true>(8, -9, 0x1234_5678), 0x34_5678_78);
        assert_eq!(ptx_cvt_pack::<2, false>(4, -1, 0xfedc_ba98), 0xedcb_a98c);
        assert_eq!(ptx_cvt_pack::<2, true>(2, -3, 0x0123_4567), 0x123_45676);
    }

    #[test]
    fn low_precision_rounding_uses_the_exact_result_not_an_f32_intermediate() {
        // Both values are just above an exact target-format midpoint.  An
        // intermediate f32 loses the final bit and would tie-to-even downward.
        let f16_exact = [(1_u64 << 31) | (1_u64 << 20) | 1];
        assert_eq!(
            encode_exact_low(false, &f16_exact, -30, LowPrecisionFormat::F16, false),
            0x4001
        );
        assert_eq!(crate::f32_to_fp16_bits(2.0 + 2.0_f32.powi(-10)), 0x4000);

        let bf16_exact = [(1_u64 << 60) | (1_u64 << 52) | 1];
        assert_eq!(
            encode_exact_low(false, &bf16_exact, -60, LowPrecisionFormat::Bf16, false),
            0x3f81
        );
        assert_eq!(crate::f32_to_bf16_bits(1.0 + 2.0_f32.powi(-8)), 0x3f80);
    }

    #[test]
    fn ptx_ftz_helpers_flush_inputs_and_outputs_with_sign() {
        let positive_subnormal = f32::from_bits(0x0000_0001);
        let negative_subnormal = f32::from_bits(0x8000_0001);
        let largest_normal = f32::MAX;

        assert_eq!(
            add_f32_ftz(positive_subnormal, positive_subnormal, F32RoundingMode::Nearest).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            add_f32_ftz(negative_subnormal, negative_subnormal, F32RoundingMode::Nearest).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            ptx_exp2_approx_ftz_f32(negative_subnormal).to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(ptx_exp2_approx_ftz_f32(-149.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(ptx_exp2_approx_f32(-149.0).to_bits(), 1);
        assert_eq!(
            ptx_rcp_approx_ftz_f32(positive_subnormal).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            ptx_rcp_approx_ftz_f32(negative_subnormal).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            ptx_rsqrt_approx_ftz_f32(positive_subnormal).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            ptx_rsqrt_approx_ftz_f32(negative_subnormal).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            ptx_rcp_approx_ftz_f32(largest_normal).to_bits(),
            0.0_f32.to_bits()
        );
    }

    #[test]
    fn ptx_f16x2_exp2_preserves_component_order_and_subnormal_results() {
        assert_eq!(ptx_exp2_approx_f16x2(0x3c00_0000), 0x4000_3c00);
        assert_eq!(ptx_exp2_approx_f16x2(0xce00_cb00), 0x0001_0400);
        assert_eq!(ptx_exp2_approx_f16x2(0x0001_fc00), 0x3c00_0000);
    }

    #[test]
    fn packed_f32_arithmetic_applies_directed_rounding_and_ftz() {
        let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
        let three_quarter_ulp = 3.0_f32 * 2.0_f32.powi(-25);
        let tiny_increment = 2.0_f32.powi(-100);
        assert_eq!(
            fma_f32(1.0, 1.0, three_quarter_ulp, F32RoundingMode::Nearest),
            one_up
        );
        assert_eq!(
            fma_f32(1.0, 1.0, three_quarter_ulp, F32RoundingMode::Zero),
            1.0
        );
        assert_eq!(
            fma_f32(1.0, 1.0, tiny_increment, F32RoundingMode::Up),
            one_up
        );

        let smallest_normal = f32::from_bits(0x0080_0000);
        let smallest_subnormal = f32::from_bits(1);
        assert_eq!(
            add_f32_ftz(smallest_normal, smallest_subnormal, F32RoundingMode::Zero).to_bits(),
            smallest_normal.to_bits()
        );
        assert_eq!(
            mul_f32_ftz(smallest_normal, 0.5, F32RoundingMode::Zero).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            fma_f32_ftz(
                smallest_subnormal,
                2.0_f32.powi(126),
                0.0,
                F32RoundingMode::Zero
            )
            .to_bits(),
            0.0_f32.to_bits()
        );
    }

    #[test]
    fn cuda_f32_oracles_canonicalize_nan_and_signed_zero() {
        let nan_a = f32::from_bits(0x7fc0_0001);
        let nan_b = f32::from_bits(0xffc0_0002);
        let canonical_nan = 0x7fff_ffff;

        assert_eq!(cuda_f32_add(nan_a, 1.0).to_bits(), canonical_nan);
        assert_eq!(cuda_f32_max(nan_a, nan_b).to_bits(), canonical_nan);
        assert_eq!(cuda_f32_min(nan_a, nan_b).to_bits(), canonical_nan);
        assert_eq!(cuda_f32_max(nan_a, 3.0), 3.0);
        assert_eq!(cuda_f32_max(3.0, nan_a), 3.0);
        assert_eq!(cuda_f32_min(nan_a, -3.0), -3.0);
        assert_eq!(cuda_f32_min(-3.0, nan_a), -3.0);
        assert_eq!(cuda_f32_max(-0.0, 0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(cuda_f32_max(0.0, -0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(cuda_f32_min(-0.0, 0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(cuda_f32_min(0.0, -0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(cuda_f32_max(-0.0, -0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(cuda_f32_min(0.0, 0.0).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn cuda_f64_minmax_match_nan_and_signed_zero_rules() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan_b = f64::from_bits(0xfff8_0000_0000_0002);

        assert_eq!(cuda_f64_max(nan_a, nan_b).to_bits(), nan_b.to_bits());
        assert_eq!(cuda_f64_min(nan_a, nan_b).to_bits(), nan_b.to_bits());
        assert_eq!(cuda_f64_max(nan_a, 3.0), 3.0);
        assert_eq!(cuda_f64_max(3.0, nan_a), 3.0);
        assert_eq!(cuda_f64_min(nan_a, -3.0), -3.0);
        assert_eq!(cuda_f64_min(-3.0, nan_a), -3.0);
        assert_eq!(cuda_f64_max(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(cuda_f64_max(0.0, -0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(cuda_f64_min(-0.0, 0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cuda_f64_min(0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cuda_f64_max(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cuda_f64_min(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn cuda_f64_add_matches_nan_selection_and_invalid_infinity() {
        let nan_a = f64::from_bits(0x7ff0_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
        let quiet_nan_a = f64::from_bits(0x7ff8_0000_0000_1234);

        assert_eq!(cuda_f64_add(nan_a, 1.0).to_bits(), quiet_nan_a.to_bits());
        assert_eq!(cuda_f64_add(1.0, nan_b).to_bits(), nan_b.to_bits());
        assert_eq!(cuda_f64_add(nan_a, nan_b).to_bits(), nan_b.to_bits());
        assert_eq!(
            cuda_f64_add(f64::INFINITY, f64::NEG_INFINITY).to_bits(),
            0xfff8_0000_0000_0000
        );
    }

    #[test]
    fn cuda_packed_ops_canonicalize_nan_components() {
        let lhs = 0x7fe1_2345_ffc5_4321_u64;
        let rhs = 0xffd2_468a_7fa1_3579_u64;
        let canonical_pair = 0x7fff_ffff_7fff_ffff_u64;
        assert_eq!(add_f32x2(lhs, rhs, F32RoundingMode::Nearest, true), canonical_pair);
        assert_eq!(mul_f32x2(lhs, rhs, F32RoundingMode::Nearest, true), canonical_pair);
        assert_eq!(pack_bf16x2(f32::NAN, -f32::NAN), 0x7fff_7fff);
        assert_eq!(hmin2_bf16(0xffc2_7fc1, 0x7fc4_7fe3), 0x7fff_7fff);
        assert_eq!(hmax2_bf16(0xffc2_7fc1, 0x7fc4_7fe3), 0x7fff_7fff);

        let smallest_normal = f32::from_bits(0x0080_0000);
        let negative_half_normal = f32::from_bits(0x8040_0000);
        assert_eq!(
            add_f32x2(
                pack_f32x2(smallest_normal, smallest_normal),
                pack_f32x2(negative_half_normal, negative_half_normal),
                F32RoundingMode::Nearest,
                true,
            ),
            pack_f32x2(smallest_normal, smallest_normal),
        );
        assert_eq!(
            mul_f32x2(
                pack_f32x2(smallest_normal, smallest_normal),
                pack_f32x2(0.5, 0.5),
                F32RoundingMode::Nearest,
                true,
            ),
            pack_f32x2(0.0, 0.0),
        );

        let positive_zero_pair = pack_bf16x2(0.0, 0.0);
        let negative_zero_pair = pack_bf16x2(-0.0, -0.0);
        assert_eq!(
            hmin2_bf16(positive_zero_pair, negative_zero_pair),
            negative_zero_pair
        );
        assert_eq!(
            hmin2_bf16(negative_zero_pair, positive_zero_pair),
            negative_zero_pair
        );
        assert_eq!(
            hmax2_bf16(positive_zero_pair, negative_zero_pair),
            positive_zero_pair
        );
        assert_eq!(
            hmax2_bf16(negative_zero_pair, positive_zero_pair),
            positive_zero_pair
        );
    }

    #[test]
    fn ptx_approximate_helpers_have_stable_canonical_representatives() {
        let inputs = [0.1_f32, -3.75_f32, 17.125_f32, 1.234_567_f32];
        let exp2_bits = [0x3f89_2fdf, 0x3d98_37f0, 0x480b_95c2, 0x4016_994f];
        let reciprocal_bits = [0x4120_0000, 0xbe88_8889, 0x3d6f_2eb7, 0x3f4f_5c32];

        for ((input, expected_exp2), expected_reciprocal) in
            inputs.into_iter().zip(exp2_bits).zip(reciprocal_bits)
        {
            assert_eq!(ptx_exp2_approx_ftz_f32(input).to_bits(), expected_exp2);
            assert_eq!(ptx_rcp_approx_ftz_f32(input).to_bits(), expected_reciprocal);
        }
        assert_eq!(ptx_rsqrt_approx_ftz_f32(0.25).to_bits(), 2.0_f32.to_bits());
        assert_eq!(ptx_rsqrt_approx_ftz_f32(4.0).to_bits(), 0.5_f32.to_bits());
        assert_eq!(
            ptx_tanh_approx_f32(1.0).to_bits(),
            0.761_594_2_f32.to_bits()
        );
        assert_eq!(
            ptx_tanh_approx_f32(-1.0).to_bits(),
            (-0.761_594_2_f32).to_bits()
        );
    }

    #[test]
    fn packed_scalar_abi_helpers_preserve_component_order() {
        use crate::abi::v2::{reg, ExecCtx, SiteId, R};

        let pair = make_float2(1.25, -2.5);
        assert_eq!(float2_x(pair), 1.25);
        assert_eq!(float2_y(pair), -2.5);

        let bf16_pair = pack_bf16x2(1.0, 2.0);
        let unpacked = unpack_bf16x2(bf16_pair);
        assert_eq!(float2_x(unpacked), 1.0);
        assert_eq!(float2_y(unpacked), 2.0);

        let other = pack_bf16x2(3.0, -1.0);
        let minimum = unpack_bf16x2(hmin2_bf16(bf16_pair, other));
        let maximum = unpack_bf16x2(hmax2_bf16(bf16_pair, other));
        assert_eq!((float2_x(minimum), float2_y(minimum)), (1.0, -1.0));
        assert_eq!((float2_x(maximum), float2_y(maximum)), (3.0, 2.0));

        let context = ExecCtx::from_inner(crate::WarpContext::from_topology(
            crate::LaunchTopology::new(1, 1, 1).unwrap(), 0,
        ));
        let mixed = reg::add::<reg::variant::MixedF32<reg::variant::Bf16>>(
            context,
            SiteId::new(0),
            (R::splat(f32_to_bf16_bits(0.5)), R::splat(4.0)),
        ).unwrap();
        assert_eq!(mixed[0], 4.5);
        assert_eq!(
            fp8x4_e4m3_from_float4(1.0, 2.0, 3.0, 4.0),
            f32_to_float8_e4m3fn_bits(1.0) as u32
                | ((f32_to_float8_e4m3fn_bits(2.0) as u32) << 8)
                | ((f32_to_float8_e4m3fn_bits(3.0) as u32) << 16)
                | ((f32_to_float8_e4m3fn_bits(4.0) as u32) << 24)
        );
    }

    #[test]
    fn directed_f32_rounding_handles_ties_signs_and_overflow() {
        let half_ulp_at_one = f32::from_bits(0x3380_0000);
        let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
        assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Nearest), 1.0);
        assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Down), 1.0);
        assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Up), one_up);
        assert_eq!(add_f32(1.0, half_ulp_at_one, F32RoundingMode::Zero), 1.0);

        let minus_one_down = f32::from_bits((-1.0_f32).to_bits() + 1);
        assert_eq!(
            add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Down),
            minus_one_down
        );
        assert_eq!(add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Up), -1.0);
        assert_eq!(add_f32(-1.0, -half_ulp_at_one, F32RoundingMode::Zero), -1.0);

        assert_eq!(mul_f32(f32::MAX, 2.0, F32RoundingMode::Down), f32::MAX);
        assert_eq!(mul_f32(f32::MAX, 2.0, F32RoundingMode::Up), f32::INFINITY);
        assert_eq!(sub_f32(one_up, half_ulp_at_one, F32RoundingMode::Down), 1.0);
    }

    #[test]
    fn directed_add_sub_detect_increments_below_f64_precision() {
        let two_to_minus_100 = f32::from_bits(0x0d80_0000);
        let one_up = f32::from_bits(1.0_f32.to_bits() + 1);
        let one_down = f32::from_bits(1.0_f32.to_bits() - 1);
        let minus_one_down = f32::from_bits((-1.0_f32).to_bits() + 1);

        assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Up), one_up);
        assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Down), 1.0);
        assert_eq!(add_f32(1.0, two_to_minus_100, F32RoundingMode::Zero), 1.0);

        assert_eq!(
            add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Down),
            minus_one_down
        );
        assert_eq!(add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Up), -1.0);
        assert_eq!(
            add_f32(-1.0, -two_to_minus_100, F32RoundingMode::Zero),
            -1.0
        );

        assert_eq!(
            sub_f32(1.0, two_to_minus_100, F32RoundingMode::Down),
            one_down
        );
        assert_eq!(sub_f32(1.0, two_to_minus_100, F32RoundingMode::Up), 1.0);
        assert_eq!(
            sub_f32(1.0, two_to_minus_100, F32RoundingMode::Zero),
            one_down
        );
    }

    #[test]
    fn directed_add_sub_preserve_cancellation_and_subnormal_results() {
        let smallest_normal = f32::from_bits(0x0080_0000);
        let largest_subnormal = f32::from_bits(0x007f_ffff);
        let smallest_subnormal = f32::from_bits(1);
        let modes = [
            F32RoundingMode::Nearest,
            F32RoundingMode::Down,
            F32RoundingMode::Up,
            F32RoundingMode::Zero,
        ];

        for mode in modes {
            let cancellation_zero = if mode == F32RoundingMode::Down {
                -0.0_f32
            } else {
                0.0_f32
            };
            assert_eq!(
                sub_f32(smallest_normal, largest_subnormal, mode),
                smallest_subnormal
            );
            assert_eq!(
                sub_f32(largest_subnormal, smallest_normal, mode),
                -smallest_subnormal
            );
            assert_eq!(
                add_f32(1.0, -1.0, mode).to_bits(),
                cancellation_zero.to_bits()
            );
            assert_eq!(
                sub_f32(1.0, 1.0, mode).to_bits(),
                cancellation_zero.to_bits()
            );
            assert_eq!(
                fma_f32(1.0, 1.0, -1.0, mode).to_bits(),
                cancellation_zero.to_bits()
            );
            assert_eq!(add_f32(0.0, 0.0, mode).to_bits(), 0.0_f32.to_bits());
            assert_eq!(add_f32(-0.0, -0.0, mode).to_bits(), (-0.0_f32).to_bits());
            assert_eq!(sub_f32(-0.0, 0.0, mode).to_bits(), (-0.0_f32).to_bits());
        }
    }

    #[test]
    fn directed_add_sub_preserve_special_values_and_overflow() {
        assert_eq!(add_f32(f32::MAX, f32::MAX, F32RoundingMode::Down), f32::MAX);
        assert_eq!(add_f32(f32::MAX, f32::MAX, F32RoundingMode::Zero), f32::MAX);
        assert_eq!(
            add_f32(f32::MAX, f32::MAX, F32RoundingMode::Up),
            f32::INFINITY
        );
        assert_eq!(
            add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Down),
            f32::NEG_INFINITY
        );
        assert_eq!(
            add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Up),
            -f32::MAX
        );
        assert_eq!(
            add_f32(-f32::MAX, -f32::MAX, F32RoundingMode::Zero),
            -f32::MAX
        );

        for mode in [
            F32RoundingMode::Nearest,
            F32RoundingMode::Down,
            F32RoundingMode::Up,
            F32RoundingMode::Zero,
        ] {
            assert_eq!(add_f32(f32::INFINITY, 1.0, mode), f32::INFINITY);
            assert_eq!(sub_f32(f32::NEG_INFINITY, 1.0, mode), f32::NEG_INFINITY);
            assert!(add_f32(f32::NAN, 1.0, mode).is_nan());
            assert!(add_f32(f32::INFINITY, f32::NEG_INFINITY, mode).is_nan());
        }
    }

    #[test]
    fn nearest_division_and_fma_use_native_f32_operations() {
        assert_eq!(div_f32_rn(7.0, 2.0), 3.5);
        assert_eq!(fma_f32_rn(2.0, 4.0, 1.0), 9.0);
    }

    #[test]
    fn runtime_scalars_round_trip_little_endian_bytes() {
        assert_eq!(u32::decode_le(&42_u32.encode_le()).unwrap(), 42);
        assert!(bool::decode_le(&true.encode_le()).unwrap());
        let vector = [1.0_f32, -2.0, 3.5, 0.0];
        assert_eq!(F32x4::decode_le(&vector.encode_le()).unwrap(), vector);
    }

    #[test]
    fn packed_float8_pack_places_the_first_operand_in_the_upper_byte() {
        assert_eq!(
            ptx_cvt_pack_narrow_x2::<false>(1.0, 2.0, crate::FLOAT8_E4M3),
            0x3840
        );
        // .relu clamps negatives, negative zero, and -inf, but not NaN.
        assert_eq!(
            ptx_cvt_pack_narrow_x2::<false>(-0.0, f32::NEG_INFINITY, crate::FLOAT8_E4M3),
            0x80fe
        );
        assert_eq!(
            ptx_cvt_pack_narrow_x2::<true>(-0.0, f32::NEG_INFINITY, crate::FLOAT8_E4M3),
            0x0000
        );
        assert_eq!(
            ptx_cvt_pack_narrow_x2::<true>(f32::NAN, -f32::NAN, crate::FLOAT8_E5M2),
            0x7f7f
        );
    }

    #[test]
    fn ptx94_narrow_rounding_and_padded_lane_placement_are_bit_exact() {
        let pack = |high, low, rounding, format| {
            ptx_cvt_pack_narrow_x2_rounded::<false>(high, low, rounding, format)
        };

        // E2M3 uses six payload bits in each byte.  1.0625 is exactly halfway
        // between codes 0x08 and 0x09; RN chooses even and RZ truncates.
        assert_eq!(
            pack(
                1.0625,
                -1.1875,
                PtxFloatRounding::NearestEven,
                crate::FLOAT6_E2M3,
            ),
            0x082a
        );
        assert_eq!(
            pack(1.0625, -1.1875, PtxFloatRounding::Zero, crate::FLOAT6_E2M3,),
            0x0829
        );
        assert_eq!(
            pack(1.0, 28.0, PtxFloatRounding::NearestEven, crate::FLOAT6_E3M2,),
            0x0c1f
        );

        // UE5M3 has no sign bit.  Its code 0x78 is 1.0 and 0x79 is 1.125.
        assert_eq!(
            pack(
                1.0625,
                1.1875,
                PtxFloatRounding::NearestEven,
                crate::FLOAT8_UE5M3,
            ),
            0x787a
        );
        assert_eq!(
            pack(1.0625, 1.1875, PtxFloatRounding::Zero, crate::FLOAT8_UE5M3,),
            0x7879
        );
        assert_eq!(
            pack(
                1.0625,
                1.1875,
                PtxFloatRounding::PositiveInfinity,
                crate::FLOAT8_UE5M3,
            ),
            0x797a
        );
        assert_eq!(
            pack(
                f32::NAN,
                f32::INFINITY,
                PtxFloatRounding::Zero,
                crate::FLOAT8_UE5M3,
            ),
            0xfffe
        );
    }

    #[test]
    fn ptx94_n1_scaling_flushes_source_subnormals_to_positive_zero() {
        let negative_f32_subnormal = f32::from_bits(0x8000_0001);
        assert_eq!(
            ptx_cvt_scaled_n1_f32(negative_f32_subnormal, 127).to_bits(),
            0
        );
        assert_eq!(ptx_cvt_scaled_n1_f16(0x8001, 127).to_bits(), 0);
        assert_eq!(ptx_cvt_scaled_n1_bf16(0x8001, 127).to_bits(), 0);

        // UE8M0 code 128 is 2 and code 126 is 1/2.
        assert_eq!(ptx_cvt_scaled_n1_f32(2.0, 128), 1.0);
        assert_eq!(ptx_cvt_scaled_n1_f16(0x3c00, 126), 2.0);
        assert_eq!(ptx_cvt_scaled_n1_bf16(0x3f80, 126), 2.0);
        assert!(ptx_cvt_scaled_n1_f32(1.0, 0xff).is_nan());
    }

    /// The `.rs` random-bit split, pinned directly rather than only through
    /// the goldens.
    ///
    /// This is the one place the wave's least obvious fact lives, and the
    /// checked-in golden vectors separate it from the plausible alternatives
    /// on only a handful of values — 13 of the 1536 `.rs` entries, and none at
    /// all for `e2m1x4.f32.rs.relu`. The vectors below were measured
    /// independently on an NVIDIA B200 through an exact carry-condition
    /// read-out, and they discriminate every alternative reading by
    /// construction: an eight-bit field, unreversed pairs, and a low-order
    /// placement each produce different values here.
    #[test]
    fn stochastic_rounding_splits_rbits_into_four_sixteen_bit_fields() {
        // Eight-bit destinations take the two contiguous halfwords: (a, b)
        // from rbits[31:16] and (e, f) from rbits[15:0], the first of each
        // pair bit-reversed.
        assert_eq!(
            ptx_cvt_rs_randoms(0x1234_5678, crate::FLOAT8_E4M3),
            [0x2c48, 0x1234, 0x1e6a, 0x5678]
        );
        assert_eq!(
            ptx_cvt_rs_randoms(0x1234_5678, crate::FLOAT8_E5M2),
            [0x2c48, 0x1234, 0x1e6a, 0x5678]
        );
        // `.e2m1x4`, whose result is itself only sixteen bits wide, instead
        // gathers each field from one byte of each halfword: (a, b) from bytes
        // 3 and 1, (e, f) from bytes 2 and 0.
        assert_eq!(
            ptx_cvt_rs_randoms(0x1234_5678, crate::FLOAT4_E2M1),
            [0x6a48, 0x1256, 0x1e2c, 0x3478]
        );
        // All-ones and all-zeros are fixed points of every candidate rule, so
        // they only guard against a field being dropped entirely.
        for format in [crate::FLOAT8_E4M3, crate::FLOAT4_E2M1] {
            assert_eq!(ptx_cvt_rs_randoms(0, format), [0, 0, 0, 0]);
            assert_eq!(
                ptx_cvt_rs_randoms(u32::MAX, format),
                [0xffff, 0xffff, 0xffff, 0xffff]
            );
        }
        // One bit at a time: bit 31 is the most significant bit of `b`'s field
        // for an eight-bit destination and the least significant bit of `a`'s
        // reversed field, and both formats agree on that pair.
        assert_eq!(
            ptx_cvt_rs_randoms(1 << 31, crate::FLOAT8_E4M3),
            [0x0001, 0x8000, 0, 0]
        );
        assert_eq!(
            ptx_cvt_rs_randoms(1 << 31, crate::FLOAT4_E2M1),
            [0x0001, 0x8000, 0, 0]
        );
        // Bit 8 separates the two formats: it belongs to (e, f) for an
        // eight-bit destination and to (a, b) for `.e2m1x4`.
        assert_eq!(
            ptx_cvt_rs_randoms(1 << 8, crate::FLOAT8_E4M3),
            [0, 0, 0x0080, 0x0100]
        );
        assert_eq!(
            ptx_cvt_rs_randoms(1 << 8, crate::FLOAT4_E2M1),
            [0x8000, 0x0001, 0, 0]
        );
    }

    // The expectations below are B200 measurements (sm_100a, driver 595.58.03,
    // CUDA 13.2); the Python microtests carry the full recorded tables.

    #[test]
    fn ptx_cvt_integer_rounding_follows_the_named_direction() {
        let minus_subnormal = f32::from_bits(0x8000_0001);
        assert_eq!(
            ptx_cvt_integral_f32(2.5, PtxIntegerRounding::NearestEven, false),
            2.0
        );
        assert_eq!(
            ptx_cvt_integral_f32(1.5, PtxIntegerRounding::NearestEven, false),
            2.0
        );
        assert_eq!(
            ptx_cvt_integral_f32(1.5, PtxIntegerRounding::Zero, false),
            1.0
        );
        assert_eq!(
            ptx_cvt_integral_f32(-0.5, PtxIntegerRounding::NegativeInfinity, false),
            -1.0
        );
        assert_eq!(
            ptx_cvt_integral_f32(0.5, PtxIntegerRounding::PositiveInfinity, false),
            1.0
        );
        // `.ftz` is what makes a subnormal round to signed zero instead of -1.
        assert_eq!(
            ptx_cvt_integral_f32(minus_subnormal, PtxIntegerRounding::NegativeInfinity, false),
            -1.0
        );
        assert_eq!(
            ptx_cvt_integral_f32(minus_subnormal, PtxIntegerRounding::NegativeInfinity, true)
                .to_bits(),
            0x8000_0000
        );
        // Same-size float-to-float rounding canonicalizes NaN in f32 but keeps
        // the payload in f64.
        assert_eq!(
            ptx_cvt_integral_f32_to_f32(f32::NAN, PtxIntegerRounding::Zero, false).to_bits(),
            0x7fff_ffff
        );
        assert_eq!(
            ptx_cvt_integral_f64_to_f64(
                f64::from_bits(0x7ff0_0000_0000_0001),
                PtxIntegerRounding::Zero
            )
            .to_bits(),
            0x7ff8_0000_0000_0001
        );
    }

    #[test]
    fn ptx_cvt_float_to_integer_nan_depends_on_both_widths() {
        use crate::abi::v2::{reg, ExecCtx, SiteId, R};
        use reg::variant as v;

        let context = ExecCtx::from_inner(crate::WarpContext::from_topology(
            crate::LaunchTopology::new(1, 1, 1).unwrap(), 0,
        ));
        // NaN results now belong to the closed CVT variants, not a scalar facade.
        macro_rules! check {
            ($source:ident, $destination:ident, $input:expr, $expected:expr) => {
                assert_eq!(reg::cvt::<v::Cvt<v::$source, v::$destination, v::CvtMode<v::Rzi>>>(
                    context, SiteId::new(0), R::splat($input),
                ).unwrap()[0], $expected);
            };
        }
        check!(F32, U8, f32::NAN, 0);
        check!(F32, U32, f32::NAN, 0);
        check!(F32, U64, f32::NAN, 1 << 63);
        check!(F64, U8, f64::NAN, 0x80);
        check!(F64, U32, f64::NAN, 0x8000_0000);
    }

    #[test]
    fn ptx_cvt_integer_to_float_rounds_from_the_exact_magnitude() {
        for rounding in [
            PtxFloatRounding::NearestEven,
            PtxFloatRounding::Zero,
            PtxFloatRounding::NegativeInfinity,
            PtxFloatRounding::PositiveInfinity,
        ] {
            assert_eq!(ptx_cvt_integer_to_f16(0, false, rounding), 0);
            assert_eq!(ptx_cvt_integer_to_f16(1, true, rounding), 0xbc00);
            assert_eq!(ptx_cvt_integer_to_bf16(1, false, rounding), 0x3f80);
        }
        assert_eq!(
            ptx_cvt_integer_to_bf16(
                (1_u64 << 60) + (1_u64 << 52) + 1,
                false,
                PtxFloatRounding::NearestEven
            ),
            0x5d81
        );
        assert_eq!(
            ptx_cvt_integer_to_f16(u64::MAX, true, PtxFloatRounding::PositiveInfinity),
            0xfbff
        );
        assert_eq!(
            ptx_cvt_integer_to_f16(u64::MAX, true, PtxFloatRounding::NegativeInfinity),
            0xfc00
        );
        // 2^24 + 1 needs 25 significand bits, so the directed modes differ.
        assert_eq!(
            ptx_cvt_integer_to_f32(16_777_217, false, PtxFloatRounding::Zero),
            16_777_216.0
        );
        assert_eq!(
            ptx_cvt_integer_to_f32(16_777_217, false, PtxFloatRounding::PositiveInfinity),
            16_777_218.0
        );
        assert_eq!(
            ptx_cvt_integer_to_f32(16_777_217, true, PtxFloatRounding::NegativeInfinity),
            -16_777_218.0
        );
        assert_eq!(
            ptx_cvt_integer_to_f32(u64::MAX, false, PtxFloatRounding::Zero),
            18_446_742_974_197_923_840.0
        );
        assert_eq!(
            ptx_cvt_integer_to_f32(u64::MAX, false, PtxFloatRounding::PositiveInfinity),
            18_446_744_073_709_551_616.0
        );
        // i64::MIN is exactly representable, so every mode agrees.
        for rounding in [
            PtxFloatRounding::Zero,
            PtxFloatRounding::NegativeInfinity,
            PtxFloatRounding::PositiveInfinity,
        ] {
            assert_eq!(
                ptx_cvt_integer_to_f64(1u64 << 63, true, rounding),
                -9_223_372_036_854_775_808.0
            );
        }
        assert_eq!(
            ptx_cvt_integer_to_f64((1u64 << 53) + 1, false, PtxFloatRounding::Zero),
            9_007_199_254_740_992.0
        );
        assert_eq!(
            ptx_cvt_integer_to_f64((1u64 << 53) + 1, false, PtxFloatRounding::PositiveInfinity),
            9_007_199_254_740_994.0
        );
    }

    #[test]
    fn packed_float8_unpack_matches_hardware_special_values() {
        // Upper byte to upper half; every NaN widens to 0x7fff, sign dropped.
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_f16x2::<false>(0x7f38, crate::FLOAT8_E4M3),
            0x7fff_3c00
        );
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_f16x2::<true>(0xff80, crate::FLOAT8_E4M3),
            0x7fff_0000
        );
        // e5m2 infinities widen exactly; .relu still zeroes the negative one.
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_f16x2::<false>(0x7cfc, crate::FLOAT8_E5M2),
            0x7c00_fc00
        );
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_f16x2::<true>(0x7cfc, crate::FLOAT8_E5M2),
            0x7c00_0000
        );
        // .satfinite is observable only where the source has infinities.
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_bf16x2::<false, false>(0x7cfc, crate::FLOAT8_E5M2),
            0x7f80_ff80
        );
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_bf16x2::<false, true>(0x7cfc, crate::FLOAT8_E5M2),
            0x7f7f_ff7f
        );
        assert_eq!(
            ptx_cvt_unpack_narrow_x2_bf16x2::<false, false>(0x7efe, crate::FLOAT8_E4M3),
            ptx_cvt_unpack_narrow_x2_bf16x2::<false, true>(0x7efe, crate::FLOAT8_E4M3),
        );
    }

    #[test]
    fn ptx_cvt_f64_to_f32_saturates_and_flushes_per_modifier() {
        use PtxFloatRounding::{NearestEven, NegativeInfinity, PositiveInfinity, Zero};

        // Full-precision rounding below MIN_NORMAL has half the spacing of
        // F32 gradual underflow. Cover both ties and the immediate neighbors.
        let normal = 0x3810_0000_0000_0000_u64;
        for (distance, nearest, away) in [
            (0, true, true),
            (1, true, true),
            ((1 << 28) - 1, true, true),
            (1 << 28, true, true),
            ((1 << 28) + 1, false, true),
            ((1 << 29) - 1, false, true),
            (1 << 29, false, false),
            ((1 << 29) + 1, false, false),
        ] {
            for negative in [false, true] {
                let sign = if negative { 0x8000_0000 } else { 0 };
                let value = f64::from_bits(normal - distance) * if negative { -1.0 } else { 1.0 };
                for (mode, keep) in [
                    (NearestEven, nearest),
                    (Zero, distance == 0),
                    (
                        NegativeInfinity,
                        if negative { away } else { distance == 0 },
                    ),
                    (
                        PositiveInfinity,
                        if negative { distance == 0 } else { away },
                    ),
                ] {
                    assert_eq!(
                        ptx_cvt_f64_to_f32(value, mode, true).to_bits(),
                        sign | if keep { 0x0080_0000 } else { 0 },
                        "distance={distance}, negative={negative}, mode={mode:?}"
                    );
                }
            }
        }
        let tiny = f64::from_bits(0x0000_0000_0000_0001);
        assert_eq!(
            ptx_cvt_f64_to_f32(tiny, PtxFloatRounding::PositiveInfinity, false).to_bits(),
            0x0000_0001
        );
        assert_eq!(
            ptx_cvt_f64_to_f32(tiny, PtxFloatRounding::PositiveInfinity, true).to_bits(),
            0x0000_0000
        );
        assert_eq!(
            ptx_cvt_f64_to_f32(f64::MAX, PtxFloatRounding::Zero, false),
            f32::MAX
        );
        assert_eq!(
            ptx_cvt_f64_to_f32(f64::MAX, PtxFloatRounding::PositiveInfinity, false),
            f32::INFINITY
        );
        assert_eq!(
            ptx_cvt_f64_to_f32(f64::NAN, PtxFloatRounding::NearestEven, false).to_bits(),
            0x7fc0_0000
        );
    }

    #[test]
    fn ptx_cvt_narrowing_applies_satfinite_then_relu() {
        // Without `.satfinite` an overflowing `.rn` result is infinity; with it
        // the result is the destination's largest finite value.
        assert_eq!(
            ptx_cvt_f32_to_f16(f32::MAX, PtxFloatRounding::NearestEven, false, false),
            0x7c00
        );
        assert_eq!(
            ptx_cvt_f32_to_f16(f32::MAX, PtxFloatRounding::NearestEven, false, true),
            0x7bff
        );
        // `.rz` truncates, so 65520 lands on MAX_NORM instead of infinity.
        assert_eq!(
            ptx_cvt_f32_to_f16(65_520.0, PtxFloatRounding::NearestEven, false, false),
            0x7c00
        );
        assert_eq!(
            ptx_cvt_f32_to_f16(65_520.0, PtxFloatRounding::Zero, false, false),
            0x7bff
        );
        assert_eq!(
            ptx_cvt_f32_to_f16(
                f32::NEG_INFINITY,
                PtxFloatRounding::NearestEven,
                true,
                false
            ),
            0
        );
        // Every NaN answers the canonical narrow NaN, `.relu` included.
        assert_eq!(
            ptx_cvt_f32_to_f16(f32::NAN, PtxFloatRounding::NearestEven, true, true),
            0x7fff
        );
        assert_eq!(
            ptx_cvt_f32_to_bf16(f32::NAN, PtxFloatRounding::NearestEven, false, false),
            0x7fff
        );
        assert_eq!(
            ptx_cvt_f32_to_bf16(f32::MAX, PtxFloatRounding::Zero, false, false),
            0x7f7f
        );
        assert_eq!(
            ptx_cvt_f32_to_bf16(f32::MAX, PtxFloatRounding::NearestEven, false, false),
            0x7f80
        );
    }
    #[test]
    fn ptx_cvt_pzo_changes_only_negative_zero_results() {
        assert_eq!(ptx_cvt_pzo_u16(0x8000), 0);
        assert_eq!(ptx_cvt_pzo_u16(0), 0);
        assert_eq!(ptx_cvt_pzo_u16(0xbc00), 0xbc00);
        assert_eq!(ptx_cvt_pzo_u16(0xffff), 0xffff);

        assert_eq!(ptx_cvt_pzo_u32(0x8000_0000), 0);
        assert_eq!(ptx_cvt_pzo_u32(0), 0);
        assert_eq!(ptx_cvt_pzo_u32(0xbf80_0000), 0xbf80_0000);
        assert_eq!(ptx_cvt_pzo_u32(0xffff_e000), 0xffff_e000);

        // `.pzo` is post-conversion: a negative value that narrows to -0 is
        // normalized even though its f32 input was not itself zero.
        let tiny_f16 = ptx_cvt_f32_to_f16(-f32::from_bits(1), PtxFloatRounding::Zero, false, false);
        assert_eq!(tiny_f16, 0x8000);
        assert_eq!(ptx_cvt_pzo_u16(tiny_f16), 0);
        let tiny_tf32 =
            ptx_cvt_f32_to_tf32(-f32::from_bits(1), PtxFloatRounding::Zero, false, false);
        assert_eq!(tiny_tf32, 0x8000_0000);
        assert_eq!(ptx_cvt_pzo_u32(tiny_tf32), 0);

        assert_eq!(ptx_cvt_pzo_narrow_x2(0x8080, crate::FLOAT8_E4M3), 0);
        assert_eq!(ptx_cvt_pzo_narrow_x2(0x807f, crate::FLOAT8_E4M3), 0x007f);
        assert_eq!(ptx_cvt_pzo_narrow_x2(0x2020, crate::FLOAT6_E2M3), 0);
        assert_eq!(ptx_cvt_pzo_narrow_x2(0x0088, crate::FLOAT4_E2M1), 0);
    }

    #[test]
    fn ptx_cvt_tf32_rounds_ties_away_only_for_rna() {
        let tie = f32::from_bits(0x3f80_1000);
        assert_eq!(
            ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::NearestEven, false, false),
            0x3f80_0000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::NearestAway, false, false),
            0x3f80_2000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(tie, PtxFloatRounding::Zero, false, false),
            0x3f80_0000
        );
        // `.rn`/`.rz` canonicalize NaN; `.rna` rounds it arithmetically, which
        // turns a payload-in-the-discarded-bits NaN into infinity.
        assert_eq!(
            ptx_cvt_f32_to_tf32(f32::NAN, PtxFloatRounding::Zero, false, true),
            0x7fff_e000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(
                f32::from_bits(0x7f80_0001),
                PtxFloatRounding::NearestAway,
                false,
                false
            ),
            0x7f80_0000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(
                f32::from_bits(0x7fc0_0000),
                PtxFloatRounding::NearestAway,
                false,
                true
            ),
            0x7fbf_e000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(f32::MAX, PtxFloatRounding::NearestEven, false, true),
            0x7f7f_e000
        );
        assert_eq!(
            ptx_cvt_f32_to_tf32(-f32::MAX, PtxFloatRounding::NearestEven, true, false),
            0
        );
    }


}
