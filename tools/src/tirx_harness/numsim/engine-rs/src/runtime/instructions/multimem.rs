//! `multimem.ld_reduce`, `multimem.st`, and `multimem.red` (PTX ISA 9.7.10.15).
//!
//! A multimem address lies in a multicast window whose bytes live in one
//! bound replica per rank. The engine executes each instruction as one access
//! per replica, in rank order, through `MemoryProxy::MulticastAlias`; ordering
//! those accesses against unicast accesses of the same bytes therefore needs
//! `fence.proxy.alias` on the synchronization path (PTX ISA 8.6).
//!
//! The ISA leaves the order and precision in which `ld_reduce` combines the
//! replicas unspecified. On GB200 NVLink SHARP a float sum is accumulated in a
//! fixed-point window anchored at the largest input exponent (see
//! `accumulator_window`) and rounded once; NumSim reproduces f32/f64 bit for
//! bit. `.acc::f32` does not change any f16/bf16 hardware result. The
//! switch's half rounding is not RNE and no input-only rule reproduces it;
//! NumSim rounds halves to nearest even, within one ulp of the hardware.
//! `red` reduces at each replica like a unicast atomic (f32 `add` flushes
//! subnormals).

use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::reg;
use super::{
    engine, execute_atomic_access, execute_physical_load, execute_physical_store, variant,
    Address, ExecCtx, Global, SiteId, R,
};
use crate::memory::BufferView;
use crate::runtime::{raw_atomic_update_physical_ptr_warp, PhysicalPtr, PtxStateSpace, RuntimeBuffer};
use crate::scalar::{add_f32_ftz, cuda_f64_add, ptx_max_f32, ptx_min_f32, F32RoundingMode};
use crate::{
    bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32, EngineError,
    MemoryAccessClass, MemoryAccessSemantics, MemoryOrder, MemoryProxy, MemoryScope,
    RuntimeScalar, WarpMask, WarpValue,
};

type Engine = super::super::Engine;
type AbiError = super::super::EngineError;
type Bytes = [u8; 16];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    LdReduce,
    St,
    Red,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Element {
    U32,
    S32,
    U64,
    S64,
    B32,
    B64,
    F16,
    Bf16,
    F32,
    F64,
}

impl Element {
    const fn byte_len(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::S32 | Self::B32 | Self::F32 => 4,
            Self::U64 | Self::S64 | Self::B64 | Self::F64 => 8,
        }
    }

    const fn is_half(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reduction {
    None,
    Add,
    AddAccF32,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// One decoded multimem specialization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MultimemForm {
    kind: Kind,
    element: Element,
    reduction: Reduction,
    byte_width: usize,
    semantics: MemoryAccessSemantics,
    atomic_coherent: bool,
}

impl MultimemForm {
    /// Decode the const codes the frontend writes into
    /// `variant::Multimem<KIND, TYPE, OP, VEC, SEM, SCOPE>`:
    ///
    /// - KIND: 0 ld_reduce, 1 st, 2 red
    /// - TYPE: 0 u32, 1 s32, 2 u64, 3 s64, 4 b32, 5 b64, 6 f16, 7 f16x2,
    ///   8 bf16, 9 bf16x2, 10 f32, 11 f64
    /// - OP: 0 none, 1 add, 2 add with `.acc::f32`, 3 min, 4 max, 5 and, 6 or, 7 xor
    /// - VEC: register count (1 for the scalar line)
    /// - SEM: 0 weak, 1 relaxed, 2 acquire, 3 release
    /// - SCOPE: 0 cta, 1 cluster, 2 gpu, 3 sys (ignored for weak)
    pub(crate) fn decode(
        kind: u8,
        ty: u8,
        op: u8,
        vec: u8,
        sem: u8,
        scope: u8,
    ) -> Result<Self, EngineError> {
        let invalid = |what: &str| {
            Err(EngineError::message(format!(
                "invalid multimem specialization: {what} (kind {kind}, type {ty}, op {op}, vec {vec}, sem {sem}, scope {scope})"
            )))
        };
        let kind = match kind {
            0 => Kind::LdReduce,
            1 => Kind::St,
            2 => Kind::Red,
            _ => return invalid("kind"),
        };
        // Packed half registers are two half elements; the element, not the
        // register, is the unit of reduction.
        let (element, register_bytes) = match ty {
            0 => (Element::U32, 4),
            1 => (Element::S32, 4),
            2 => (Element::U64, 8),
            3 => (Element::S64, 8),
            4 => (Element::B32, 4),
            5 => (Element::B64, 8),
            6 => (Element::F16, 2),
            7 => (Element::F16, 4),
            8 => (Element::Bf16, 2),
            9 => (Element::Bf16, 4),
            10 => (Element::F32, 4),
            11 => (Element::F64, 8),
            _ => return invalid("type"),
        };
        let reduction = match op {
            0 => Reduction::None,
            1 => Reduction::Add,
            2 => Reduction::AddAccF32,
            3 => Reduction::Min,
            4 => Reduction::Max,
            5 => Reduction::And,
            6 => Reduction::Or,
            7 => Reduction::Xor,
            _ => return invalid("op"),
        };
        if !matches!(vec, 1 | 2 | 4 | 8) {
            return invalid("vector width");
        }
        let byte_width = register_bytes * usize::from(vec);
        if !matches!(byte_width, 4 | 8 | 16) {
            return invalid("access width must be 32, 64, or 128 bits");
        }
        match (kind, reduction) {
            (Kind::St, Reduction::None) => {}
            (Kind::St, _) | (Kind::LdReduce | Kind::Red, Reduction::None) => {
                return invalid("operation")
            }
            (Kind::Red, Reduction::AddAccF32) => return invalid(".acc::f32 on red"),
            _ => {}
        }
        let reduction_fits = match reduction {
            Reduction::None => true,
            Reduction::Add => !matches!(element, Element::B32 | Element::B64),
            Reduction::AddAccF32 => element.is_half(),
            Reduction::Min | Reduction::Max => !matches!(
                element,
                Element::B32 | Element::B64 | Element::F32 | Element::F64
            ),
            Reduction::And | Reduction::Or | Reduction::Xor => {
                matches!(element, Element::B32 | Element::B64)
            }
        };
        if !reduction_fits {
            return invalid("operation does not apply to the element type");
        }
        let order = match (kind, sem) {
            (Kind::LdReduce | Kind::St, 0) => None,
            (_, 1) => Some(MemoryOrder::Relaxed),
            (Kind::LdReduce, 2) => Some(MemoryOrder::Acquire),
            (Kind::St | Kind::Red, 3) => Some(MemoryOrder::Release),
            _ => return invalid("memory semantic"),
        };
        let semantics = match order {
            None => MemoryAccessSemantics::plain().with_proxy(MemoryProxy::MulticastAlias),
            Some(order) => {
                let scope = match scope {
                    0 => MemoryScope::Cta,
                    1 => MemoryScope::Cluster,
                    2 => MemoryScope::Gpu,
                    3 => MemoryScope::Sys,
                    _ => return invalid("scope"),
                };
                let semantics = MemoryAccessSemantics::scoped(
                    order,
                    scope,
                    MemoryProxy::MulticastAlias,
                    MemoryAccessClass::Atomic,
                );
                if kind == Kind::Red {
                    semantics.as_reduction()
                } else {
                    semantics
                }
            }
        };
        Ok(Self {
            kind,
            element,
            reduction,
            byte_width,
            semantics,
            atomic_coherent: order.is_some(),
        })
    }

    fn element_count(&self) -> usize {
        self.byte_width / self.element.byte_len()
    }
}

fn words_to_bytes(words: [u32; 4]) -> Bytes {
    let mut bytes = [0_u8; 16];
    for (index, word) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn bytes_to_words(bytes: &Bytes) -> [u32; 4] {
    std::array::from_fn(|index| {
        u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    })
}

fn scalar_bytes<T: RuntimeScalar>(value: T) -> Bytes {
    let mut bytes = [0_u8; 16];
    value.encode_le_into(&mut bytes[..T::BYTE_LEN]);
    bytes
}

fn half_decode(element: Element, bits: u16) -> f32 {
    if element == Element::Bf16 {
        bf16_bits_to_f32(bits)
    } else {
        fp16_bits_to_f32(bits)
    }
}

fn half_encode(element: Element, value: f32) -> u16 {
    if element == Element::Bf16 {
        f32_to_bf16_bits(value)
    } else {
        f32_to_fp16_bits(value)
    }
}

/// Combine one element of `incoming` into `current`, rounding to the element
/// type. `.acc::f32` is handled by the caller, which keeps wide accumulators.
fn combine_element(
    element: Element,
    reduction: Reduction,
    current: &mut [u8],
    incoming: &[u8],
) -> Result<(), EngineError> {
    macro_rules! int {
        ($ty:ty) => {{
            let a = <$ty>::from_le_bytes(current.try_into().unwrap());
            let b = <$ty>::from_le_bytes(incoming.try_into().unwrap());
            let value = match reduction {
                Reduction::Add => a.wrapping_add(b),
                Reduction::Min => a.min(b),
                Reduction::Max => a.max(b),
                Reduction::And => a & b,
                Reduction::Or => a | b,
                Reduction::Xor => a ^ b,
                _ => unreachable!("validated multimem integer reduction"),
            };
            current.copy_from_slice(&value.to_le_bytes());
        }};
    }
    match element {
        Element::U32 | Element::B32 => int!(u32),
        Element::S32 => int!(i32),
        Element::U64 | Element::B64 => int!(u64),
        Element::S64 => int!(i64),
        Element::F32 => {
            let a = f32::from_le_bytes(current.try_into().unwrap());
            let b = f32::from_le_bytes(incoming.try_into().unwrap());
            current.copy_from_slice(&add_f32_ftz(a, b, F32RoundingMode::Nearest).to_le_bytes());
        }
        Element::F64 => {
            let a = f64::from_le_bytes(current.try_into().unwrap());
            let b = f64::from_le_bytes(incoming.try_into().unwrap());
            current.copy_from_slice(&cuda_f64_add(a, b).to_le_bytes());
        }
        Element::F16 | Element::Bf16 => {
            let a = half_decode(element, u16::from_le_bytes(current.try_into().unwrap()));
            let b = half_decode(element, u16::from_le_bytes(incoming.try_into().unwrap()));
            let value = match reduction {
                Reduction::Add => a + b,
                Reduction::Min => ptx_min_f32(a, b, false, false),
                Reduction::Max => ptx_max_f32(a, b, false, false),
                _ => {
                    return Err(EngineError::message(
                        "half-precision multimem supports only add/min/max",
                    ))
                }
            };
            current.copy_from_slice(&half_encode(element, value).to_le_bytes());
        }
    }
    Ok(())
}

fn combine(form: &MultimemForm, current: &mut Bytes, incoming: &Bytes) -> Result<(), EngineError> {
    let size = form.element.byte_len();
    for index in 0..form.element_count() {
        let range = index * size..index * size + size;
        combine_element(
            form.element,
            form.reduction,
            &mut current[range.clone()],
            &incoming[range],
        )?;
    }
    Ok(())
}

struct WindowGroup {
    window: BufferView,
    mask: WarpMask,
}

/// Group the active lanes by the multicast window they address and record
/// each lane's byte offset into that window.
fn resolve_windows(
    physical: &crate::PhysicalMemory,
    pointer: &PhysicalPtr,
    mask: WarpMask,
    byte_width: usize,
) -> Result<(Vec<WindowGroup>, WarpValue<i64>), EngineError> {
    let mut groups: BTreeMap<u64, WindowGroup> = BTreeMap::new();
    let mut offsets = WarpValue::splat(0_i64);
    for lane in mask {
        let (window, offset) = pointer.lane_global_allocation_offset(physical, lane, byte_width)?;
        if window.multicast_replicas().is_none() {
            return Err(EngineError::message(format!(
                "multimem address on lane {lane} is not in a multicast window (PTX ISA 9.7.10.15)"
            )));
        }
        if offset % byte_width != 0 {
            return Err(EngineError::message(format!(
                "multimem requires {byte_width}-byte alignment on lane {lane}"
            )));
        }
        if offset
            .checked_add(byte_width)
            .map_or(true, |end| end > window.byte_len())
        {
            return Err(EngineError::out_of_bounds(format!(
                "multimem access on lane {lane} leaves its {}-byte multicast window",
                window.byte_len()
            )));
        }
        offsets[lane] = i64::try_from(offset)
            .map_err(|_| EngineError::out_of_bounds("multimem window offset exceeds i64"))?;
        let group = groups
            .entry(window.allocation().as_u64())
            .or_insert_with(|| WindowGroup {
                window,
                mask: WarpMask::EMPTY,
            });
        group.mask = group.mask | WarpMask::from_bits(1 << lane);
    }
    Ok((groups.into_values().collect(), offsets))
}

fn replica_pointer(replica: &BufferView, offsets: &WarpValue<i64>, byte_width: usize) -> PhysicalPtr {
    PhysicalPtr::new(RuntimeBuffer::Global(replica.clone()), offsets.clone(), 1)
        .with_pointee_itemsize(byte_width)
}

/// Dispatch on the 32/64/128-bit access width with the matching identity
/// memory carrier.
macro_rules! by_width {
    ($width:expr, $marker:ident, $scalar:ident, $body:block) => {
        match $width {
            4 => {
                #[allow(dead_code)]
                type $marker = reg::variant::U32;
                type $scalar = u32;
                $body
            }
            8 => {
                #[allow(dead_code)]
                type $marker = reg::variant::U64;
                type $scalar = u64;
                $body
            }
            16 => {
                #[allow(dead_code)]
                type $marker = variant::U64x2;
                type $scalar = crate::scalar::U64x2;
                $body
            }
            _ => unreachable!("validated multimem access width"),
        }
    };
}

fn load_replica(
    warp: &mut Engine,
    context: crate::WarpContext,
    site: SiteId,
    form: &MultimemForm,
    pointer: PhysicalPtr,
    label: Arc<str>,
) -> Result<WarpValue<Bytes>, AbiError> {
    by_width!(form.byte_width, Marker, Scalar, {
        let loaded = execute_physical_load::<Marker, Engine>(
            engine(warp),
            context,
            site,
            pointer,
            Some(label),
            PtxStateSpace::Global,
            form.semantics,
            form.atomic_coherent,
            false,
        )?;
        let loaded: &WarpValue<Scalar> = loaded.inner();
        Ok(WarpValue::from_fn(|lane| scalar_bytes(loaded[lane])))
    })
}

fn store_replica(
    warp: &mut Engine,
    context: crate::WarpContext,
    site: SiteId,
    form: &MultimemForm,
    pointer: PhysicalPtr,
    label: Arc<str>,
    values: &WarpValue<Bytes>,
) -> Result<(), AbiError> {
    by_width!(form.byte_width, Marker, Scalar, {
        let mut decoded = WarpValue::splat(<Scalar as RuntimeScalar>::zero());
        for lane in context.active_mask() {
            decoded[lane] = Scalar::decode_le(&values[lane][..form.byte_width])?;
        }
        execute_physical_store::<Marker, Engine>(
            engine(warp),
            context,
            site,
            pointer,
            Some(label),
            R::from_inner(decoded),
            PtxStateSpace::Global,
            form.semantics,
        )
    })
}

async fn reduce_replica(
    warp: &mut Engine,
    context: ExecCtx,
    site: SiteId,
    form: &MultimemForm,
    pointer: PhysicalPtr,
    label: Arc<str>,
    values: &WarpValue<Bytes>,
) -> Result<(), AbiError> {
    let address = Address::<super::Generic>::from_logical_buffer(pointer, label);
    by_width!(form.byte_width, Marker, Scalar, {
        let mut operands = WarpValue::splat(<Scalar as RuntimeScalar>::zero());
        for lane in context.active_mask() {
            operands[lane] = Scalar::decode_le(&values[lane][..form.byte_width])?;
        }
        let form = *form;
        execute_atomic_access::<Scalar>(
            warp,
            context,
            site,
            address,
            form.byte_width,
            form.semantics,
            PtxStateSpace::Global,
            false,
            move |physical, context, pointer, mask, ptx_space| {
                raw_atomic_update_physical_ptr_warp::<Scalar>(
                    physical,
                    context,
                    pointer,
                    &operands,
                    mask,
                    ptx_space,
                    |_lane, old, operand| {
                        let mut current = scalar_bytes(old);
                        combine(&form, &mut current, &scalar_bytes(operand))?;
                        Scalar::decode_le(&current[..form.byte_width])
                    },
                )
            },
        )
        .await
        .map(|_| ())
    })
}

/// A two's-complement fixed-point integer in units of 2^-1074, wide enough to
/// hold any sum of up to 2^60 finite f64 values exactly.
struct ExactSum {
    words: [u64; 34],
}

impl ExactSum {
    const QUANTUM_EXPONENT: i32 = -1074;

    fn new() -> Self {
        Self { words: [0; 34] }
    }

    fn add(&mut self, value: f64) {
        let bits = value.to_bits();
        let field = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, shift) = if field == 0 {
            (fraction, 0)
        } else {
            (fraction | (1_u64 << 52), field - 1)
        };
        if mantissa == 0 {
            return;
        }
        let word = shift / 64;
        let offset = shift % 64;
        let wide = u128::from(mantissa) << offset;
        let parts = [wide as u64, (wide >> 64) as u64];
        let negative = value.is_sign_negative();
        let mut carry = false;
        for (index, slot) in self.words.iter_mut().enumerate().skip(word) {
            let part = parts.get(index - word).copied().unwrap_or(0);
            if part == 0 && !carry {
                if index > word + 1 {
                    break;
                }
                continue;
            }
            if negative {
                let (difference, borrow_a) = slot.overflowing_sub(part);
                let (difference, borrow_b) = difference.overflowing_sub(u64::from(carry));
                *slot = difference;
                carry = borrow_a || borrow_b;
            } else {
                let (sum, carry_a) = slot.overflowing_add(part);
                let (sum, carry_b) = sum.overflowing_add(u64::from(carry));
                *slot = sum;
                carry = carry_a || carry_b;
            }
        }
    }

    fn magnitude(&self) -> (bool, [u64; 34]) {
        let negative = self.words[33] >> 63 == 1;
        let mut magnitude = self.words;
        if negative {
            let mut carry = true;
            for word in &mut magnitude {
                let (value, overflow) = (!*word).overflowing_add(u64::from(carry));
                *word = value;
                carry = overflow;
            }
        }
        (negative, magnitude)
    }

    fn most_significant_bit(magnitude: &[u64; 34]) -> Option<usize> {
        let top = (0..magnitude.len()).rev().find(|&index| magnitude[index] != 0)?;
        Some(top * 64 + 63 - magnitude[top].leading_zeros() as usize)
    }

    /// Whether `|sum| < 2^exponent`.
    fn below(&self, exponent: i32) -> bool {
        Self::most_significant_bit(&self.magnitude().1)
            .is_none_or(|msb| (msb as i32 + Self::QUANTUM_EXPONENT) < exponent)
    }

    /// Round to a binary format with `precision` significand bits and
    /// minimum normal exponent `min_exponent`, ties to even. The result is
    /// returned as the f64 holding that exactly representable value; callers
    /// convert to the element type, which overflows to infinity as RN does.
    fn round(&self, precision: u32, min_exponent: i32) -> f64 {
        let (negative, magnitude) = self.magnitude();
        let Some(msb) = Self::most_significant_bit(&magnitude) else {
            return 0.0;
        };
        let bit = |index: usize| magnitude[index / 64] >> (index % 64) & 1 == 1;
        let exponent = (msb as i32 + Self::QUANTUM_EXPONENT).max(min_exponent);
        let ulp = exponent - (precision as i32 - 1) - Self::QUANTUM_EXPONENT;
        let mut mantissa: u64 = 0;
        if ulp <= 0 {
            for index in (0..=msb).rev() {
                mantissa = mantissa << 1 | u64::from(bit(index));
            }
            mantissa <<= -ulp;
        } else {
            let ulp = ulp as usize;
            for index in (ulp..=msb.max(ulp)).rev() {
                mantissa = mantissa << 1 | u64::from(index <= msb && bit(index));
            }
            let half = bit(ulp - 1);
            let below = ulp - 1;
            let sticky = magnitude[..below / 64].iter().any(|word| *word != 0)
                || magnitude[below / 64] & ((1_u64 << (below % 64)) - 1) != 0;
            if half && (sticky || mantissa & 1 == 1) {
                mantissa += 1;
            }
        }
        let scale = exponent - (precision as i32 - 1);
        let value = mantissa as f64 * 2_f64.powi(scale.max(-1022)) * 2_f64.powi((scale + 1022).min(0));
        if negative {
            -value
        } else {
            value
        }
    }
}

/// `(precision, min_exponent)` of the element's binary format.
fn float_format(element: Element) -> (u32, i32) {
    match element {
        Element::F16 => (11, -14),
        Element::Bf16 => (8, -126),
        Element::F32 => (24, -126),
        Element::F64 => (53, -1022),
        _ => unreachable!("integer multimem elements have no float format"),
    }
}

fn float_element(element: Element, bytes: &[u8]) -> f64 {
    match element {
        Element::F16 | Element::Bf16 => {
            f64::from(half_decode(element, u16::from_le_bytes(bytes.try_into().unwrap())))
        }
        Element::F32 => f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
        Element::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
        _ => unreachable!("integer multimem elements have no float value"),
    }
}

fn store_float_element(element: Element, value: f64, bytes: &mut [u8]) {
    match element {
        // The value is already rounded to the element format, so these
        // conversions are exact (or overflow to infinity).
        Element::F16 | Element::Bf16 => {
            bytes.copy_from_slice(&half_encode(element, value as f32).to_le_bytes())
        }
        Element::F32 => bytes.copy_from_slice(&(value as f32).to_le_bytes()),
        Element::F64 => bytes.copy_from_slice(&value.to_le_bytes()),
        _ => unreachable!("integer multimem elements have no float value"),
    }
}

/// The accumulator window, measured on GB200: it is anchored at
/// `16 * ceil(e / 16)` for the largest input exponent `e` (subnormals count as
/// the minimum normal exponent), and a sum below `2^(anchor - result_bits)`
/// reads as +0. f32/f64 inputs are also truncated toward zero below
/// `anchor - term_bits`; for halves that bit lies below the format's
/// precision wherever it was probed, so it is not modeled.
const ANCHOR_GRANULE: i32 = 16;

/// `(term_bits, result_bits)` of the element's accumulator window.
fn accumulator_window(element: Element) -> (Option<i32>, i32) {
    match element {
        Element::F32 | Element::F64 => (Some(95), 86),
        _ => (None, 38),
    }
}

/// Unbiased exponent of a finite nonzero f64, with f64 subnormals at -1022.
fn f64_exponent(value: f64) -> i32 {
    (((value.to_bits() >> 52) & 0x7ff) as i32).max(1) - 1023
}

fn truncate_below(value: f64, lsb: i32) -> f64 {
    let dropped = lsb - (f64_exponent(value) - 52);
    if dropped <= 0 {
        value
    } else if dropped >= 53 {
        0.0
    } else {
        f64::from_bits(value.to_bits() & !((1_u64 << dropped) - 1))
    }
}

/// One `ld_reduce.add` element over every replica; a zero sum is +0 even
/// when every input is -0. Infinities and NaNs propagate as an IEEE fold
/// would.
fn exact_float_sum(element: Element, values: &[f64]) -> f64 {
    if values.iter().any(|value| !value.is_finite()) {
        return values.iter().sum();
    }
    let (precision, min_exponent) = float_format(element);
    let Some(largest) = values
        .iter()
        .filter(|value| **value != 0.0)
        .map(|value| f64_exponent(*value).max(min_exponent))
        .max()
    else {
        return 0.0;
    };
    let anchor = -(-largest).div_euclid(ANCHOR_GRANULE) * ANCHOR_GRANULE;
    let (term_bits, result_bits) = accumulator_window(element);
    let mut sum = ExactSum::new();
    for &value in values {
        sum.add(match term_bits {
            Some(bits) => truncate_below(value, anchor - bits),
            None => value,
        });
    }
    if sum.below(anchor - result_bits) {
        return 0.0;
    }
    sum.round(precision, min_exponent)
}

fn ld_reduce_fold(
    form: &MultimemForm,
    replicas: &[WarpValue<Bytes>],
    mask: WarpMask,
) -> Result<WarpValue<Bytes>, EngineError> {
    let mut result = WarpValue::splat([0_u8; 16]);
    let float_add = matches!(form.reduction, Reduction::Add | Reduction::AddAccF32)
        && matches!(
            form.element,
            Element::F16 | Element::Bf16 | Element::F32 | Element::F64
        );
    for lane in mask {
        if float_add {
            let size = form.element.byte_len();
            let mut values = Vec::with_capacity(replicas.len());
            for index in 0..form.element_count() {
                let range = index * size..index * size + size;
                values.clear();
                values.extend(
                    replicas
                        .iter()
                        .map(|replica| float_element(form.element, &replica[lane][range.clone()])),
                );
                let sum = exact_float_sum(form.element, &values);
                store_float_element(form.element, sum, &mut result[lane][range]);
            }
            continue;
        }
        let mut current = replicas[0][lane];
        for replica in &replicas[1..] {
            combine(form, &mut current, &replica[lane])?;
        }
        result[lane] = current;
    }
    Ok(result)
}

pub(super) async fn execute(
    warp: &mut Engine,
    context: ExecCtx,
    site: SiteId,
    form: MultimemForm,
    address: Address<Global>,
    operand: R<[u32; 4]>,
) -> Result<R<[u32; 4]>, AbiError> {
    let (pointer, logical_buffer) = address.into_parts();
    let warp_context = context.into_inner();
    let physical = engine(warp).kernel().physical().clone();
    let (groups, offsets) =
        resolve_windows(&physical, &pointer, warp_context.active_mask(), form.byte_width)?;
    let values = WarpValue::from_fn(|lane| words_to_bytes(operand.inner()[lane]));
    let base = logical_buffer.as_deref().unwrap_or("multimem").to_owned();
    let mut result = WarpValue::splat([0_u32; 4]);
    for group in groups {
        let replicas = group
            .window
            .multicast_replicas()
            .expect("resolved window has replicas")
            .clone();
        let group_context = warp_context.with_active_mask(group.mask);
        let mut loaded = Vec::with_capacity(replicas.len());
        for (rank, replica) in replicas.iter().enumerate() {
            let pointer = replica_pointer(replica, &offsets, form.byte_width);
            let label: Arc<str> = Arc::from(format!("{base}@rank{rank}"));
            match form.kind {
                Kind::LdReduce => loaded.push(load_replica(
                    warp,
                    group_context,
                    site,
                    &form,
                    pointer,
                    label,
                )?),
                Kind::St => {
                    store_replica(warp, group_context, site, &form, pointer, label, &values)?
                }
                Kind::Red => {
                    reduce_replica(
                        warp,
                        ExecCtx::from_inner(group_context),
                        site,
                        &form,
                        pointer,
                        label,
                        &values,
                    )
                    .await?
                }
            }
        }
        if form.kind == Kind::LdReduce {
            let folded = ld_reduce_fold(&form, &loaded, group.mask)?;
            for lane in group.mask {
                result[lane] = bytes_to_words(&folded[lane]);
            }
        }
    }
    Ok(R::from_inner(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(kind: u8, ty: u8, op: u8, vec: u8) -> MultimemForm {
        MultimemForm::decode(kind, ty, op, vec, if kind == 2 { 1 } else { 0 }, 3).unwrap()
    }

    #[test]
    fn decode_rejects_forms_outside_the_isa_table() {
        // f16 scalar is 16 bits; s64 add is fine for the engine but not
        // `.and` on u32; `.acc::f32` needs a half type; weak red is illegal.
        assert!(MultimemForm::decode(0, 6, 1, 1, 0, 3).is_err());
        assert!(MultimemForm::decode(0, 0, 5, 1, 0, 3).is_err());
        assert!(MultimemForm::decode(0, 10, 2, 1, 0, 3).is_err());
        assert!(MultimemForm::decode(2, 0, 1, 1, 0, 3).is_err());
        assert!(MultimemForm::decode(1, 0, 1, 1, 0, 3).is_err());
        assert!(MultimemForm::decode(0, 10, 3, 1, 0, 3).is_err());
    }

    #[test]
    fn ld_reduce_folds_replicas_in_rank_order() {
        let add_f32x4 = form(0, 10, 1, 4);
        let replica = |base: f32| {
            let mut bytes = [0_u8; 16];
            for index in 0..4 {
                bytes[index * 4..index * 4 + 4]
                    .copy_from_slice(&(base + index as f32).to_le_bytes());
            }
            WarpValue::splat(bytes)
        };
        let folded = ld_reduce_fold(
            &add_f32x4,
            &[replica(1.0), replica(10.0), replica(100.0)],
            WarpMask::from_bits(1),
        )
        .unwrap();
        let words = bytes_to_words(&folded[0]);
        let values: Vec<f32> = words.iter().map(|word| f32::from_bits(*word)).collect();
        assert_eq!(values, vec![111.0, 114.0, 117.0, 120.0]);
    }

    #[test]
    fn half_add_rounds_the_exact_sum_once_with_or_without_acc_f32() {
        // 2048 + 1 + 1 in binary16: rounding each step would stay at 2048.
        let replica = |value: f32| {
            let mut bytes = [0_u8; 16];
            bytes[..2].copy_from_slice(&f32_to_fp16_bits(value).to_le_bytes());
            bytes[2..4].copy_from_slice(&f32_to_fp16_bits(-0.0).to_le_bytes());
            WarpValue::splat(bytes)
        };
        let replicas = [replica(2048.0), replica(1.0), replica(1.0)];
        let mask = WarpMask::from_bits(1);
        let plain = ld_reduce_fold(&form(0, 7, 1, 1), &replicas, mask).unwrap();
        let widened = ld_reduce_fold(&form(0, 7, 2, 1), &replicas, mask).unwrap();
        let half = |bytes: &Bytes, index: usize| {
            fp16_bits_to_f32(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]))
        };
        assert_eq!(half(&plain[0], 0), 2050.0);
        assert_eq!(half(&widened[0], 0), 2050.0);
        // The switch accumulator has no -0.
        assert!(half(&plain[0], 1) == 0.0 && half(&plain[0], 1).is_sign_positive());
        // bf16 MAX - MAX anchors the window at 2^128: the rest is flushed.
        let p = |exponent: i32| 2_f64.powi(exponent);
        let max = f64::from(bf16_bits_to_f32(0x7f7f));
        assert_eq!(exact_float_sum(Element::Bf16, &[max, -max, 7.0, -1.0]), 0.0);
        assert_eq!(exact_float_sum(Element::Bf16, &[1.0, -1.0, 1.75 * p(-38)]), 1.75 * p(-38));
        assert_eq!(exact_float_sum(Element::Bf16, &[1.0, -1.0, p(-39)]), 0.0);
        assert_eq!(exact_float_sum(Element::F16, &[p(1), -p(1), p(-22)]), p(-22));
        assert_eq!(exact_float_sum(Element::F16, &[p(1), -p(1), p(-23)]), 0.0);
    }

    #[test]
    fn float_add_survives_cancellation_and_rounds_ties_to_even() {
        // 2^24 + 1 - 2^24 + 1: any f32 fold loses a 1; the exact sum is 2.
        assert_eq!(exact_float_sum(Element::F32, &[16777216.0, 1.0, -16777216.0, 1.0]), 2.0);
        // 1 + 2^-24 is a binary32 tie: even stays at 1.
        assert_eq!(exact_float_sum(Element::F32, &[1.0, 2_f64.powi(-24)]), 1.0);
        let p = |exponent: i32| 2_f64.powi(exponent);
        let next = f64::from(1.0_f32 + f32::EPSILON);
        for element in [Element::F32, Element::F64] {
            let tie = if element == Element::F32 { p(-24) } else { p(-53) };
            let next = if element == Element::F32 { next } else { 1.0 + f64::EPSILON };
            // Anchor 0: a term at 2^-95 still breaks the tie, 2^-96 is truncated away.
            assert_eq!(exact_float_sum(element, &[1.0, tie, p(-95)]), next);
            assert_eq!(exact_float_sum(element, &[1.0, tie, p(-96)]), 1.0);
            assert_eq!(exact_float_sum(element, &[1.0, 3.0 * tie, -p(-96)]), 1.0 + 4.0 * tie);
            // A sum below 2^(anchor - 86) reads as +0, whatever its sign.
            assert_eq!(exact_float_sum(element, &[1.0, -1.0, 1.75 * p(-86)]), 1.75 * p(-86));
            assert_eq!(
                exact_float_sum(element, &[1.0, -1.0, -1.75 * p(-87)]).to_bits(),
                0.0_f64.to_bits()
            );
            // The anchor moves in steps of 16 exponents: 2^16 and 2^17 differ.
            assert_eq!(exact_float_sum(element, &[p(16), -p(16), p(-70)]), p(-70));
            assert_eq!(exact_float_sum(element, &[p(17), -p(17), p(-70)]), 0.0);
        }
        // An f32 subnormal is far below an anchor set by normal inputs.
        assert_eq!(
            exact_float_sum(Element::F32, &[1.0, p(-24), p(-149)]),
            1.0
        );
        assert_eq!(exact_float_sum(Element::F32, &[p(-149), p(-149)]), p(-148));
        assert_eq!(exact_float_sum(Element::F64, &[-0.0, -0.0]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(exact_float_sum(Element::F64, &[f64::MAX, f64::MAX]), f64::INFINITY);
        assert_eq!(
            exact_float_sum(Element::F64, &[5e-324, 5e-324, -1e-323]).to_bits(),
            0.0_f64.to_bits()
        );
        // binary16 overflow: 65519 rounds to MAX, the 65520 tie to 2^16 (stored as inf).
        assert_eq!(exact_float_sum(Element::F16, &[65504.0, 15.0]), 65504.0);
        assert_eq!(exact_float_sum(Element::F16, &[65504.0, 16.0]), 65536.0);
        assert!(exact_float_sum(Element::F32, &[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }

    #[test]
    fn integer_reductions_respect_signedness() {
        let replica = |value: i32| WarpValue::splat(scalar_bytes(value as u32));
        let replicas = [replica(-1), replica(3)];
        let mask = WarpMask::from_bits(1);
        let signed = ld_reduce_fold(&form(0, 1, 3, 1), &replicas, mask).unwrap();
        let unsigned = ld_reduce_fold(&form(0, 0, 3, 1), &replicas, mask).unwrap();
        assert_eq!(bytes_to_words(&signed[0])[0], (-1_i32) as u32);
        assert_eq!(bytes_to_words(&unsigned[0])[0], 3);
    }

    #[test]
    fn ordered_forms_use_the_multicast_alias_proxy() {
        let red = MultimemForm::decode(2, 0, 1, 1, 3, 3).unwrap();
        assert_eq!(red.semantics.proxy(), MemoryProxy::MulticastAlias);
        assert_eq!(red.semantics.scope(), Some(MemoryScope::Sys));
        let weak = MultimemForm::decode(0, 0, 1, 1, 0, 0).unwrap();
        assert_eq!(weak.semantics.proxy(), MemoryProxy::MulticastAlias);
        assert!(!weak.atomic_coherent);
    }
}
