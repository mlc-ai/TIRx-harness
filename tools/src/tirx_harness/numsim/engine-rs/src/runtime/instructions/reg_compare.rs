//! PTX 9.4 register-only comparison, classification, and sign selection.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum CompareAtom {
    Bits(u64),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

#[derive(Clone, Copy)]
pub(super) enum CompareKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,
}

pub(super) fn compare_atom(kind: CompareKind, lhs: CompareAtom, rhs: CompareAtom) -> bool {
    match (lhs, rhs) {
        (CompareAtom::Bits(lhs), CompareAtom::Bits(rhs)) => match kind {
            CompareKind::Eq => lhs == rhs,
            CompareKind::Ne => lhs != rhs,
            _ => unreachable!("closed variants exclude ordered bit-size comparisons"),
        },
        (CompareAtom::Signed(lhs), CompareAtom::Signed(rhs)) => match kind {
            CompareKind::Eq => lhs == rhs,
            CompareKind::Ne => lhs != rhs,
            CompareKind::Lt => lhs < rhs,
            CompareKind::Le => lhs <= rhs,
            CompareKind::Gt => lhs > rhs,
            CompareKind::Ge => lhs >= rhs,
            _ => unreachable!("closed variants exclude unordered integer comparisons"),
        },
        (CompareAtom::Unsigned(lhs), CompareAtom::Unsigned(rhs)) => match kind {
            CompareKind::Eq => lhs == rhs,
            CompareKind::Ne => lhs != rhs,
            CompareKind::Lt => lhs < rhs,
            CompareKind::Le => lhs <= rhs,
            CompareKind::Gt => lhs > rhs,
            CompareKind::Ge => lhs >= rhs,
            _ => unreachable!("closed variants exclude unordered integer comparisons"),
        },
        (CompareAtom::Float(lhs), CompareAtom::Float(rhs)) => {
            let unordered = lhs.is_nan() || rhs.is_nan();
            match kind {
                CompareKind::Eq => !unordered && lhs == rhs,
                CompareKind::Ne => !unordered && lhs != rhs,
                CompareKind::Lt => !unordered && lhs < rhs,
                CompareKind::Le => !unordered && lhs <= rhs,
                CompareKind::Gt => !unordered && lhs > rhs,
                CompareKind::Ge => !unordered && lhs >= rhs,
                CompareKind::Equ => unordered || lhs == rhs,
                CompareKind::Neu => unordered || lhs != rhs,
                CompareKind::Ltu => unordered || lhs < rhs,
                CompareKind::Leu => unordered || lhs <= rhs,
                CompareKind::Gtu => unordered || lhs > rhs,
                CompareKind::Geu => unordered || lhs >= rhs,
                CompareKind::Num => !unordered,
                CompareKind::Nan => unordered,
            }
        }
        _ => unreachable!("one specialization always compares one PTX source type"),
    }
}

pub(super) trait CompareFormat: RegisterType
where
    Self::Scalar: Copy,
{
    const LANES: usize;
    fn compare(
        kind: CompareKind,
        lhs: Self::Scalar,
        rhs: Self::Scalar,
        flush_subnormal: bool,
    ) -> u8;
}

pub(super) trait ScalarCompareFormat: RegisterType
where
    Self::Scalar: Copy,
{
    fn atom(value: Self::Scalar, flush_subnormal: bool) -> CompareAtom;
}

impl<Src> CompareFormat for Src
where
    Src: ScalarCompareFormat,
    Src::Scalar: Copy,
{
    const LANES: usize = 1;

    fn compare(
        kind: CompareKind,
        lhs: Self::Scalar,
        rhs: Self::Scalar,
        flush_subnormal: bool,
    ) -> u8 {
        u8::from(compare_atom(
            kind,
            Self::atom(lhs, flush_subnormal),
            Self::atom(rhs, flush_subnormal),
        ))
    }
}

macro_rules! scalar_compare_format {
    ($marker:ty, $scalar:ty, $atom:ident) => {
        impl ScalarCompareFormat for $marker {
            fn atom(value: $scalar, _flush_subnormal: bool) -> CompareAtom {
                CompareAtom::$atom(value as _)
            }
        }
    };
}

scalar_compare_format!(variant::I8, i8, Signed);
scalar_compare_format!(variant::B16, u16, Bits);
scalar_compare_format!(variant::B32, u32, Bits);
scalar_compare_format!(variant::B64, u64, Bits);
scalar_compare_format!(variant::I16, i16, Signed);
scalar_compare_format!(variant::I32, i32, Signed);
scalar_compare_format!(variant::I64, i64, Signed);
scalar_compare_format!(variant::U8, u8, Unsigned);
scalar_compare_format!(variant::U16, u16, Unsigned);
scalar_compare_format!(variant::U32, u32, Unsigned);
scalar_compare_format!(variant::U64, u64, Unsigned);

impl ScalarCompareFormat for variant::F32 {
    fn atom(value: f32, flush_subnormal: bool) -> CompareAtom {
        let value = if flush_subnormal {
            crate::scalar::flush_subnormal_f32(value)
        } else {
            value
        };
        CompareAtom::Float(value.into())
    }
}

impl ScalarCompareFormat for variant::F64 {
    fn atom(value: f64, _flush_subnormal: bool) -> CompareAtom {
        CompareAtom::Float(value)
    }
}

impl ScalarCompareFormat for variant::F16 {
    fn atom(value: u16, flush_subnormal: bool) -> CompareAtom {
        let bits = if flush_subnormal {
            crate::scalar::flush_subnormal_f16_bits(value)
        } else {
            value
        };
        let value = f64::from(decode_f16(bits));
        CompareAtom::Float(value)
    }
}

impl ScalarCompareFormat for variant::Bf16 {
    fn atom(value: u16, _flush_subnormal: bool) -> CompareAtom {
        let value = f64::from(decode_bf16(value));
        CompareAtom::Float(value)
    }
}

impl CompareFormat for variant::F16x2 {
    const LANES: usize = 2;
    fn compare(kind: CompareKind, lhs: u32, rhs: u32, flush_subnormal: bool) -> u8 {
        let atom = |bits| {
            let bits = if flush_subnormal {
                crate::scalar::flush_subnormal_f16_bits(bits)
            } else {
                bits
            };
            CompareAtom::Float(f64::from(decode_f16(bits)))
        };
        u8::from(compare_atom(kind, atom(lhs as u16), atom(rhs as u16)))
            | (u8::from(compare_atom(
                kind,
                atom((lhs >> 16) as u16),
                atom((rhs >> 16) as u16),
            )) << 1)
    }
}

impl CompareFormat for variant::Bf16x2 {
    const LANES: usize = 2;
    fn compare(kind: CompareKind, lhs: u32, rhs: u32, _flush_subnormal: bool) -> u8 {
        let atom = |bits| CompareAtom::Float(f64::from(decode_bf16(bits)));
        u8::from(compare_atom(kind, atom(lhs as u16), atom(rhs as u16)))
            | (u8::from(compare_atom(
                kind,
                atom((lhs >> 16) as u16),
                atom((rhs >> 16) as u16),
            )) << 1)
    }
}

pub(super) trait SubnormalMode<Src: CompareFormat>
where
    Src::Scalar: Copy,
{
    const FLUSH: bool;
}

impl<Src> SubnormalMode<Src> for variant::PreserveSubnormal
where
    Src: CompareFormat,
    Src::Scalar: Copy,
{
    const FLUSH: bool = false;
}

impl SubnormalMode<variant::F32> for variant::Ftz {
    const FLUSH: bool = true;
}
impl SubnormalMode<variant::F16> for variant::Ftz {
    const FLUSH: bool = true;
}
impl SubnormalMode<variant::F16x2> for variant::Ftz {
    const FLUSH: bool = true;
}

pub(super) trait ValidComparison<Src: CompareFormat>
where
    Src::Scalar: Copy,
{
    const KIND: CompareKind;
}

macro_rules! comparisons_for {
    ($compare:ty, $kind:ident; $($source:ty),+ $(,)?) => {
        $(
            impl ValidComparison<$source> for $compare {
                const KIND: CompareKind = CompareKind::$kind;
            }
        )+
    };
}

comparisons_for!(variant::Eq, Eq; variant::B16, variant::B32, variant::B64,
    variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64,
    variant::F16, variant::Bf16, variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);
comparisons_for!(variant::Ne, Ne; variant::B16, variant::B32, variant::B64,
    variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64,
    variant::F16, variant::Bf16, variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);
comparisons_for!(variant::Lt, Lt; variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64, variant::F16, variant::Bf16,
    variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);
comparisons_for!(variant::Le, Le; variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64, variant::F16, variant::Bf16,
    variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);
comparisons_for!(variant::Gt, Gt; variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64, variant::F16, variant::Bf16,
    variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);
comparisons_for!(variant::Ge, Ge; variant::I8, variant::I16, variant::I32, variant::I64,
    variant::U8, variant::U16, variant::U32, variant::U64, variant::F16, variant::Bf16,
    variant::F16x2, variant::Bf16x2, variant::F32, variant::F64);

macro_rules! float_comparison {
    ($compare:ty, $kind:ident) => {
        comparisons_for!($compare, $kind; variant::F16, variant::Bf16, variant::F16x2,
            variant::Bf16x2, variant::F32, variant::F64);
    };
}

float_comparison!(variant::Equ, Equ);
float_comparison!(variant::Neu, Neu);
float_comparison!(variant::Ltu, Ltu);
float_comparison!(variant::Leu, Leu);
float_comparison!(variant::Gtu, Gtu);
float_comparison!(variant::Geu, Geu);
float_comparison!(variant::Num, Num);
float_comparison!(variant::Nan, Nan);

pub(super) fn compare_masks<Src, Compare, Subnormal>(
    lhs: &R<Src::Scalar>,
    rhs: &R<Src::Scalar>,
) -> R<u8>
where
    Src: CompareFormat,
    Src::Scalar: Copy,
    Compare: ValidComparison<Src>,
    Subnormal: SubnormalMode<Src>,
{
    lhs.zip_map(rhs, |_lane, lhs, rhs| {
        Src::compare(Compare::KIND, *lhs, *rhs, Subnormal::FLUSH)
    })
}

trait BooleanOp {
    fn apply(value: bool, predicate: bool) -> bool;
}

impl BooleanOp for variant::BoolAnd {
    fn apply(value: bool, predicate: bool) -> bool {
        value && predicate
    }
}
impl BooleanOp for variant::BoolOr {
    fn apply(value: bool, predicate: bool) -> bool {
        value || predicate
    }
}
impl BooleanOp for variant::BoolXor {
    fn apply(value: bool, predicate: bool) -> bool {
        value ^ predicate
    }
}

fn combine_masks<BoolOp: BooleanOp>(masks: R<u8>, predicate: &R<bool>) -> R<u8> {
    masks.zip_map(predicate, |_lane, mask, predicate| {
        u8::from(BoolOp::apply(mask & 1 != 0, *predicate))
            | (u8::from(BoolOp::apply(mask & 2 != 0, *predicate)) << 1)
    })
}

trait SetDestination<Src: CompareFormat>: RegisterType
where
    Src::Scalar: Copy,
    Self::Scalar: Copy,
{
    fn encode(mask: u8) -> Self::Scalar;
}

macro_rules! set_destination {
    ($dst:ty, $scalar:ty, $encode:expr; $($src:ty),+ $(,)?) => {
        $(
            impl SetDestination<$src> for $dst {
                fn encode(mask: u8) -> $scalar {
                    ($encode)(mask, <$src as CompareFormat>::LANES)
                }
            }
        )+
    };
}

set_destination!(variant::U32, u32,
    |mask: u8, lanes: usize| if lanes == 2 {
        u32::from(if mask & 1 != 0 { u16::MAX } else { 0 })
            | (u32::from(if mask & 2 != 0 { u16::MAX } else { 0 }) << 16)
    } else if mask & 1 != 0 { u32::MAX } else { 0 };
    variant::B16, variant::B32, variant::B64, variant::I16, variant::I32, variant::I64,
    variant::U16, variant::U32, variant::U64, variant::F32, variant::F64,
    variant::F16, variant::Bf16, variant::F16x2, variant::Bf16x2);
set_destination!(variant::I32, i32,
    |mask: u8, lanes: usize| if lanes == 2 {
        (u32::from(if mask & 1 != 0 { u16::MAX } else { 0 })
            | (u32::from(if mask & 2 != 0 { u16::MAX } else { 0 }) << 16)) as i32
    } else if mask & 1 != 0 { -1 } else { 0 };
    variant::B16, variant::B32, variant::B64, variant::I16, variant::I32, variant::I64,
    variant::U16, variant::U32, variant::U64, variant::F32, variant::F64,
    variant::F16, variant::Bf16, variant::F16x2, variant::Bf16x2);
set_destination!(variant::F32, f32,
    |mask: u8, _lanes: usize| if mask & 1 != 0 { 1.0 } else { 0.0 };
    variant::B16, variant::B32, variant::B64, variant::I16, variant::I32, variant::I64,
    variant::U16, variant::U32, variant::U64, variant::F32, variant::F64);
set_destination!(variant::U16, u16,
    |mask: u8, _lanes: usize| if mask & 1 != 0 { u16::MAX } else { 0 };
    variant::F16, variant::Bf16);
set_destination!(variant::I16, i16,
    |mask: u8, _lanes: usize| if mask & 1 != 0 { -1 } else { 0 };
    variant::F16, variant::Bf16);
set_destination!(variant::F16, u16,
    |mask: u8, _lanes: usize| if mask & 1 != 0 { 0x3c00 } else { 0 };
    variant::B16, variant::B32, variant::B64, variant::I16, variant::I32, variant::I64,
    variant::U16, variant::U32, variant::U64, variant::F16, variant::F32, variant::F64);
set_destination!(variant::Bf16, u16,
    |mask: u8, _lanes: usize| if mask & 1 != 0 { 0x3f80 } else { 0 };
    variant::B16, variant::B32, variant::B64, variant::I16, variant::I32, variant::I64,
    variant::U16, variant::U32, variant::U64, variant::F16, variant::F32, variant::F64);
set_destination!(variant::F16x2, u32,
    |mask: u8, _lanes: usize| u32::from(if mask & 1 != 0 { 0x3c00_u16 } else { 0 })
        | (u32::from(if mask & 2 != 0 { 0x3c00_u16 } else { 0 }) << 16);
    variant::F16x2);
set_destination!(variant::Bf16x2, u32,
    |mask: u8, _lanes: usize| u32::from(if mask & 1 != 0 { 0x3f80_u16 } else { 0 })
        | (u32::from(if mask & 2 != 0 { 0x3f80_u16 } else { 0 }) << 16);
    variant::Bf16x2);

register_variant! {
    [impl<Src, Compare, Dst, Subnormal>] set_spec, variant::Set<Src, Compare, Dst, Subnormal>
    where [
        Src: CompareFormat,
        Src::Scalar: Copy,
        Compare: ValidComparison<Src>,
        Dst: SetDestination<Src>,
        Dst::Scalar: Copy,
        Subnormal: SubnormalMode<Src>,
    ],
    (R<Src::Scalar>, R<Src::Scalar>) => R<Dst::Scalar>;
    |_context, _site, (lhs, rhs)| {
        Ok(compare_masks::<Src, Compare, Subnormal>(&lhs, &rhs).map(|_, mask| Dst::encode(mask)))
    }
}

register_variant! {
    [impl<Src, Compare, Dst, BoolOp, Subnormal>] set_spec, variant::SetBool<Src, Compare, Dst, BoolOp, Subnormal>
    where [
        Src: CompareFormat,
        Src::Scalar: Copy,
        Compare: ValidComparison<Src>,
        Dst: SetDestination<Src>,
        Dst::Scalar: Copy,
        BoolOp: BooleanOp,
        Subnormal: SubnormalMode<Src>,
    ],
    (R<Src::Scalar>, R<Src::Scalar>, R<bool>) => R<Dst::Scalar>;
    |_context, _site, (lhs, rhs, predicate)| {
        let masks = combine_masks::<BoolOp>(
            compare_masks::<Src, Compare, Subnormal>(&lhs, &rhs),
            &predicate,
        );
        Ok(masks.map(|_, mask| Dst::encode(mask)))
    }
}

register_variant! {
    [impl<Src, Compare, BoolOp, Subnormal>] setp_spec, variant::SetpBool<Src, Compare, BoolOp, Subnormal>
    where [
        Src: ScalarCompareFormat,
        Src::Scalar: Copy,
        Compare: ValidComparison<Src>,
        BoolOp: BooleanOp,
        Subnormal: SubnormalMode<Src>,
    ],
    (R<Src::Scalar>, R<Src::Scalar>, R<bool>) => R<bool>;
    |_context, _site, (lhs, rhs, predicate)| {
        Ok(compare_masks::<Src, Compare, Subnormal>(&lhs, &rhs)
            .zip_map(&predicate, |_lane, mask, predicate| {
                BoolOp::apply(mask & 1 != 0, *predicate)
            }))
    }
}

fn predicate_pair<Src: CompareFormat>(mask: u8) -> (bool, bool)
where
    Src::Scalar: Copy,
{
    let first = mask & 1 != 0;
    (
        first,
        if Src::LANES == 1 {
            !first
        } else {
            mask & 2 != 0
        },
    )
}

register_variant! {
    [impl<Src, Compare, Subnormal>] setp_spec, variant::SetpPair<Src, Compare, Subnormal>
    where [
        Src: CompareFormat,
        Src::Scalar: Copy,
        Compare: ValidComparison<Src>,
        Subnormal: SubnormalMode<Src>,
    ],
    (R<Src::Scalar>, R<Src::Scalar>) => (R<bool>, R<bool>);
    |_context, _site, (lhs, rhs)| {
        let masks = compare_masks::<Src, Compare, Subnormal>(&lhs, &rhs);
        Ok((
            masks.clone().map(|_, mask| predicate_pair::<Src>(mask).0),
            masks.map(|_, mask| predicate_pair::<Src>(mask).1),
        ))
    }
}

register_variant! {
    [impl<Src, Compare, BoolOp, Subnormal>] setp_spec, variant::SetpPairBool<Src, Compare, BoolOp, Subnormal>
    where [
        Src: CompareFormat,
        Src::Scalar: Copy,
        Compare: ValidComparison<Src>,
        BoolOp: BooleanOp,
        Subnormal: SubnormalMode<Src>,
    ],
    (R<Src::Scalar>, R<Src::Scalar>, R<bool>) => (R<bool>, R<bool>);
    |_context, _site, (lhs, rhs, predicate)| {
        let masks = compare_masks::<Src, Compare, Subnormal>(&lhs, &rhs);
        Ok((
            masks.clone().zip_map(&predicate, |_lane, mask, predicate| {
                BoolOp::apply(predicate_pair::<Src>(*mask).0, *predicate)
            }),
            masks.zip_map(&predicate, |_lane, mask, predicate| {
                BoolOp::apply(predicate_pair::<Src>(*mask).1, *predicate)
            }),
        ))
    }
}

trait SlctValue: RegisterType
where
    Self::Scalar: Copy,
{
}
impl SlctValue for variant::B16 {}
impl SlctValue for variant::B32 {}
impl SlctValue for variant::B64 {}

trait SlctSelector<Subnormal>: RegisterType
where
    Self::Scalar: Copy,
{
    fn select_first(value: Self::Scalar) -> bool;
}
impl SlctSelector<variant::PreserveSubnormal> for variant::I32 {
    fn select_first(value: i32) -> bool {
        value >= 0
    }
}
impl SlctSelector<variant::PreserveSubnormal> for variant::F32 {
    fn select_first(value: f32) -> bool {
        value >= 0.0
    }
}
impl SlctSelector<variant::Ftz> for variant::F32 {
    fn select_first(value: f32) -> bool {
        crate::scalar::flush_subnormal_f32(value) >= 0.0
    }
}

register_variant! {
    [impl<Value, Selector, Subnormal>] slct_spec, variant::Slct<Value, Selector, Subnormal>
    where [
        Value: SlctValue,
        Value::Scalar: Copy,
        Selector: SlctSelector<Subnormal>,
        Selector::Scalar: Copy,
    ],
    (R<Value::Scalar>, R<Value::Scalar>, R<Selector::Scalar>) => R<Value::Scalar>;
    |_context, _site, (a, b, c)| {
        Ok(R::from_fn(|lane| {
            if Selector::select_first(c[lane]) {
                a[lane]
            } else {
                b[lane]
            }
        }))
    }
}

trait TestpFormat: RegisterType
where
    Self::Scalar: Copy,
{
}
impl TestpFormat for variant::F32 {}
impl TestpFormat for variant::F64 {}

trait TestpClass<T> {
    fn classify(value: T) -> bool;
}

macro_rules! testp_classes {
    ($float:ty) => {
        impl TestpClass<$float> for variant::Finite {
            fn classify(value: $float) -> bool {
                value.is_finite()
            }
        }
        impl TestpClass<$float> for variant::Infinite {
            fn classify(value: $float) -> bool {
                value.is_infinite()
            }
        }
        impl TestpClass<$float> for variant::Number {
            fn classify(value: $float) -> bool {
                !value.is_nan()
            }
        }
        impl TestpClass<$float> for variant::NotANumber {
            fn classify(value: $float) -> bool {
                value.is_nan()
            }
        }
        impl TestpClass<$float> for variant::Normal {
            fn classify(value: $float) -> bool {
                value == 0.0 || value.is_normal()
            }
        }
        impl TestpClass<$float> for variant::Subnormal {
            fn classify(value: $float) -> bool {
                value.is_subnormal()
            }
        }
    };
}

testp_classes!(f32);
testp_classes!(f64);

register_variant! {
    [impl<T, Class>] testp_spec, variant::Testp<T, Class>
    where [
        T: TestpFormat,
        T::Scalar: Copy,
        Class: TestpClass<T::Scalar>,
    ],
    R<T::Scalar> => R<bool>;
    |_context, _site, source| {
        Ok(source.map(|_, value| Class::classify(value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_float_ne_is_false_for_nan_but_unordered_ne_is_true() {
        let nan = CompareAtom::Float(f64::NAN);
        let one = CompareAtom::Float(1.0);
        assert!(!compare_atom(CompareKind::Ne, nan, one));
        assert!(compare_atom(CompareKind::Neu, nan, one));
        assert!(!compare_atom(CompareKind::Num, nan, one));
        assert!(compare_atom(CompareKind::Nan, nan, one));
    }

    #[test]
    fn half_ftz_is_sign_preserving_and_testp_zero_is_normal() {
        assert_eq!(crate::scalar::flush_subnormal_f16_bits(0x0001), 0x0000);
        assert_eq!(crate::scalar::flush_subnormal_f16_bits(0x8001), 0x8000);
        assert!(<variant::Normal as TestpClass<f32>>::classify(-0.0));
        assert!(!<variant::Subnormal as TestpClass<f32>>::classify(-0.0));
    }
}
