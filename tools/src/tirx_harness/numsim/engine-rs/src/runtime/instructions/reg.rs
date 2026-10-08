//! Engine implementation of v2 register-instruction specializations.

use std::marker::PhantomData;

use super::instruction::{instruction_variant, register_instruction};
use super::EngineError;
use super::{ExecCtx, SiteId, R};
use crate::scalar::{PtxFloatRounding, PtxIntegerRounding, U64x2};

// One declaration owns each register variant's ABI and execution body.
macro_rules! register_variant {
    ([$($implementation:tt)*] $spec:ident, $variant:ty $(where [$($bounds:tt)*])?,
     $args:ty => $output:ty;
     |$context:ident, $site:ident, $values:pat_param| $body:block) => {
        instruction_variant! {
            [$($implementation)*] $spec, $variant $(where [$($bounds)*])?,
            $args => $output;
            fn execute(
                $context: ExecCtx, $site: SiteId, $values: Self::Args,
            ) -> Result<Self::Output, EngineError> $body
        }
    };
}

register_instruction!(mov_spec, MovVariant, mov);
register_instruction!(mov_pack_spec, MovPackVariant, mov_pack);
register_instruction!(mov_unpack_spec, MovUnpackVariant, mov_unpack);
register_instruction!(cvt_pack_spec, CvtPackVariant, cvt_pack);
register_instruction!(add_spec, AddVariant, add);
register_instruction!(sub_spec, SubVariant, sub);
register_instruction!(mul_spec, MulVariant, mul);
register_instruction!(copysign_spec, CopysignVariant, copysign);
register_instruction!(mad_spec, MadVariant, mad);
register_instruction!(mul24_spec, Mul24Variant, mul24);
register_instruction!(mad24_spec, Mad24Variant, mad24);
register_instruction!(sad_spec, SadVariant, sad);
register_instruction!(clmad_spec, ClmadVariant, clmad);
register_instruction!(bfe_spec, BfeVariant, bfe);
register_instruction!(bfi_spec, BfiVariant, bfi);
register_instruction!(bfind_spec, BfindVariant, bfind);
register_instruction!(bmsk_spec, BmskVariant, bmsk);
register_instruction!(brev_spec, BrevVariant, brev);
register_instruction!(clz_spec, ClzVariant, clz);
register_instruction!(cnot_spec, CnotVariant, cnot);
register_instruction!(popc_spec, PopcVariant, popc);
register_instruction!(shf_spec, ShfVariant, shf);
register_instruction!(szext_spec, SzextVariant, szext);
register_instruction!(spdecompress_spec, SpDecompressVariant, spdecompress);
register_instruction!(spcompress_spec, SpCompressVariant, spcompress);
register_instruction!(fma_spec, FmaVariant, fma);
register_instruction!(div_spec, DivVariant, div);
register_instruction!(rem_spec, RemVariant, rem);
register_instruction!(dp2a_spec, Dp2aVariant, dp2a);
register_instruction!(dp4a_spec, Dp4aVariant, dp4a);
register_instruction!(max_spec, MaxVariant, max);
register_instruction!(min_spec, MinVariant, min);
register_instruction!(rcp_spec, RcpVariant, rcp);
register_instruction!(sqrt_spec, SqrtVariant, sqrt);
register_instruction!(sin_spec, SinVariant, sin);
register_instruction!(cos_spec, CosVariant, cos);
register_instruction!(exp2_spec, Exp2Variant, exp2);
register_instruction!(lg2_spec, Lg2Variant, lg2);
register_instruction!(rsqrt_spec, RsqrtVariant, rsqrt);
register_instruction!(tanh_spec, TanhVariant, tanh);
register_instruction!(neg_spec, NegVariant, neg);
register_instruction!(abs_spec, AbsVariant, abs);
register_instruction!(and_spec, AndVariant, and);
register_instruction!(or_spec, OrVariant, or);
register_instruction!(xor_spec, XorVariant, xor);
register_instruction!(not_spec, NotVariant, not);
register_instruction!(lop3_spec, Lop3Variant, lop3);
register_instruction!(shl_spec, ShlVariant, shl);
register_instruction!(shr_spec, ShrVariant, shr);
register_instruction!(setp_spec, SetpVariant, setp);
register_instruction!(selp_spec, SelpVariant, selp);
register_instruction!(set_spec, SetVariant, set);
register_instruction!(slct_spec, SlctVariant, slct);
register_instruction!(testp_spec, TestpVariant, testp);
register_instruction!(cvt_spec, CvtVariant, cvt);
register_instruction!(fns_spec, FnsVariant, fns);
register_instruction!(createpolicy_spec, CreatePolicyVariant, createpolicy);

include!("mov_unpack.rs");
include!("sparse_compress.rs");

// PTX policy bits are opaque; every valid policy is equivalent in a model
// without cache residency. Zero is our private representative, not a hardware
// encoding. Validate the input contract without reading or ordering memory.
register_variant! {
    [impl] createpolicy_spec, variant::PolicyFraction,
    R<f32> => R<u64>;
    |context, _site, fraction| {
        for lane in context.active_mask() {
            if !(fraction[lane] > 0.0 && fraction[lane] <= 1.0) {
                return Err(EngineError::message(format!(
                    "createpolicy fraction must be in (0, 1] on lane {lane}"
                )));
            }
        }
        Ok(R::splat(0))
    }
}
register_variant! {
    [impl] createpolicy_spec, variant::PolicyConvert,
    R<u64> => R<u64>;
    |_context, _site, _property| {
        Ok(R::splat(0))
    }
}
register_variant! {
    [impl] createpolicy_spec, variant::PolicyRange,
    (super::Address<super::Global>, R<u32>, R<u32>) => R<u64>;
    |context, _site, args| {
        let (address, primary, total) = args;
        address.inner().require_ptx_space_for_mask(
            crate::runtime::PtxStateSpace::Global,
            context.active_mask().into_inner(),
        )?;
        for lane in context.active_mask() {
            if primary[lane] > total[lane] {
                return Err(EngineError::message(format!(
                    "createpolicy primary size exceeds total size on lane {lane}"
                )));
            }
        }
        Ok(R::splat(0))
    }
}

/// Closed compile-time PTX variants shared by register instructions.
///
/// A marker is meaningful only together with the instruction function.  For
/// example, `add::<I32>` is `add.s32`, while `mul::<I32>` is `mul.lo.s32`.
pub mod variant {
    use super::PhantomData;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SpCompress<const ELEM_BITS: usize, const INDEX_BITS: usize, const NUM: usize>;

    macro_rules! scalar_markers {
        ($($name:ident),+ $(,)?) => {
            $(
                #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
                pub struct $name;
            )+
        };
    }

    scalar_markers!(I8, I16, I32, I64, U8, U16, U32, U64, B16, B32, B64, B128, Pred);
    scalar_markers!(I16x2, U16x2);
    scalar_markers!(B16x4, B32x4, B64x2);
    scalar_markers!(PolicyRange, PolicyFraction, PolicyConvert);
    scalar_markers!(ClmadHi, ClmadLo, Hi, Lo);
    scalar_markers!(
        F16, F16Ftz, F16x2, F16x2Ftz, Bf16, Bf16x2, F32, F32Ftz, F32Rn, F32Rm, F32RnFtz, F64
    );
    scalar_markers!(Eq, Ne, Lt, Le, Gt, Ge, Equ, Neu, Ltu, Leu, Gtu, Geu, Num, Nan);
    scalar_markers!(Unmodified, Rn, Rz, Rm, Rp, Rmi, Rs);
    scalar_markers!(Approx, Full);
    // PTX `cvt` rounding spellings the generic and tf32 grammar lines add to
    // the set above: the remaining three `.irnd` modes and `.rna`.
    scalar_markers!(Rni, Rzi, Rpi, Rna);
    /// The rounding position of a conversion that cannot round.
    ///
    /// Used where every source value is exactly representable in the
    /// destination, so all four `.frnd` spellings are one function and the
    /// rounding axis does not exist for that pair.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Exact;
    scalar_markers!(PreserveSubnormal, Ftz, NoSat, Sat);
    /// TensorFloat-32, a `cvt` destination type with no arithmetic mnemonic.
    ///
    /// The register carrier is `.b32`: sign and exponent in place, a 10-bit
    /// fraction, and thirteen zero low bits.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Tf32;
    scalar_markers!(IgnoreNan, PropagateNan, BoolAnd, BoolOr, BoolXor);
    scalar_markers!(Finite, Infinite, Number, NotANumber, Normal, Subnormal);
    scalar_markers!(LaneId, WarpIdInCta, CtaId, ClusterId, Clock64);
    scalar_markers!(
        E4m3x2, E5m2x2, E2m1x2, E2m3x2, E3m2x2, Ue5m3x2, E4m3x4, E5m2x4, E2m1x4, E2m3x4, E3m2x4,
        Ue8m0x2, S2f6x2
    );
    scalar_markers!(NoSatFinite, SatFinite, NoRelu, Relu, PreserveZero, Pzo);
    /// Same-precision scalar or packed half arithmetic, always round-to-nearest-even.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct HalfArithmetic<T, Subnormal = PreserveSubnormal, Clamp = NoSat, const OOB: bool = false>(
        PhantomData<fn(T, Subnormal, Clamp)>,
    );
    pub type F16Rn = HalfArithmetic<F16>;
    pub type Bf16Rn = HalfArithmetic<Bf16>;
    scalar_markers!(NoScale, ScaledUe8m0N1, ScaledUe8m0N2);
    scalar_markers!(Clamp, Wrap, Left, Right, BitPosition, ShiftAmount);
    scalar_markers!(F4e, B4e, Rc8, Ecl, Ecr, Rc16);

    /// Byte-selection mode of `prmt.b32`; the unmodified form uses `B32`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Prmt<Mode>(PhantomData<fn(Mode)>);

    /// Exact signedness/width and result mode of `bfind`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bfind<T, Mode>(PhantomData<fn(T, Mode)>);

    /// Exact `.clamp`/`.wrap` specialization of `bmsk.b32`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bmsk<Mode>(PhantomData<fn(Mode)>);

    /// Exact direction and shift-bound mode of `shf.b32`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Shf<Direction, Mode>(PhantomData<fn(Direction, Mode)>);

    /// Exact signedness and width mode of `szext`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Szext<T, Mode>(PhantomData<fn(T, Mode)>);

    /// Exact source/destination scalar types and modifier of one `cvt` form.
    ///
    /// `Unmodified` is the default so existing two-parameter specializations
    /// keep spelling an unmodified instruction.  A narrowing conversion such
    /// as `cvt.rn.f16.f32` must name `Rn` explicitly.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cvt<Src, Dst, Mode = Unmodified>(PhantomData<fn(Src) -> (Dst, Mode)>);

    /// One exact `cvt.pack.sat` field type.
    ///
    /// Only the eight `(BITS, SIGNED)` combinations implemented by the
    /// instruction trait are valid, so generated artifacts cannot request an
    /// invented field width through a runtime flag.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CvtPack<const BITS: u32, const SIGNED: bool>;

    /// Modifier product of one narrow-float `cvt` form, packed or scalar.
    ///
    /// The `.satfinite` and `.relu` qualifiers change the numeric result, so
    /// each combination is its own closed specialization rather than a runtime
    /// flag on a shared one.  `.satfinite` is mandatory in the FP8 destination
    /// direction and optional in the `.bf16x2` source direction, which is why
    /// both axes are spelled out instead of being folded into the rounding
    /// marker.  `Scale` names the `.scaled::{n1,n2}::ue8m0` qualifiers, which
    /// add a runtime scale operand and so change `Args` as well as the numerics.
    /// `Zero` names PTX 9.4's `.pzo` result post-processing. Both trailing
    /// parameters default so older spellings keep the same specialization text.
    ///
    /// The scalar `.f16`, `.bf16` and `.tf32` destinations share this type
    /// rather than owning a second one: the ISA gives `.f16` and `.f16x2` the
    /// same `cvt.frnd2{.relu}{.satfinite}` grammar line, differing only in
    /// destination arity, so splitting the modifier product by arity would be
    /// arbitrary and would leave two axis orders for one set of qualifiers.
    /// The name is therefore broader than "packed"; renaming it is a
    /// follow-up, not a behaviour change.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PackedMode<
        Round,
        Saturate = NoSatFinite,
        Activation = NoRelu,
        Scale = NoScale,
        Zero = PreserveZero,
    >(PhantomData<fn(Round, Saturate, Activation, Scale, Zero)>);

    /// Exact modifier spelling of one generic-grammar `cvt` form.
    ///
    /// `Round` is the PTX rounding modifier (`Unmodified` when the form takes
    /// none), `Subnormal` is `.ftz`, and `Clamp` is the result clamp used by
    /// same-width floating conversion. Float-to-integer `.sat` remains
    /// redundant and continues to share its plain specialization.
    ///
    /// A bare rounding marker in `Mode` position, as in `Cvt<F64, F32, Rn>`,
    /// is *not* a member of this family.  Those are the older frontend entry
    /// points; a `cvt` written in TIRx source always resolves here.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CvtMode<Round = Unmodified, Subnormal = PreserveSubnormal, Clamp = NoSat>(
        PhantomData<fn(Round, Subnormal, Clamp)>,
    );

    /// Exact operand type and comparison relation of one `setp` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Setp<T, Compare, Subnormal = PreserveSubnormal>(
        PhantomData<fn(T, Compare, Subnormal)>,
    );

    /// Exact packed integer lane type and relation of one PTX 9.4 `set` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetPacked<T, Compare>(PhantomData<fn(T, Compare)>);

    /// One value-producing PTX `set` specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Set<Src, Compare, Dst, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Src, Compare, Subnormal) -> Dst>,
    );

    /// One value-producing PTX `set.CmpOp.BoolOp` specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetBool<Src, Compare, Dst, BoolOp, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Src, Compare, BoolOp, Subnormal) -> Dst>,
    );

    /// One single-predicate PTX `setp.CmpOp.BoolOp` specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetpBool<Src, Compare, BoolOp, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Src, Compare, BoolOp, Subnormal)>,
    );

    /// One two-predicate PTX `setp` specialization.
    ///
    /// Scalar sources return the comparison and its complement. Packed-half
    /// sources return the low- and high-lane comparisons respectively.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetpPair<Src, Compare, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Src, Compare, Subnormal)>,
    );

    /// The Boolean-combining sibling of [`SetpPair`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SetpPairBool<Src, Compare, BoolOp, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Src, Compare, BoolOp, Subnormal)>,
    );

    /// One numeric-sign PTX `slct` specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Slct<ValueBits, Selector, Subnormal = PreserveSubnormal>(
        PhantomData<fn(ValueBits, Selector, Subnormal)>,
    );

    /// One PTX floating-point classification specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Testp<T, Class>(PhantomData<fn(T, Class)>);

    /// One PTX `lop3.b32` truth table.  The `u8` type is the ISA's complete
    /// immediate domain, so no invalid table value can enter the ABI.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Lop3<const LUT: u8>;

    /// One PTX `lop3.BoolOp.b32` truth table and predicate combiner.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Lop3Bool<const LUT: u8, BoolOp>(PhantomData<fn() -> BoolOp>);

    /// Exact static shape of one PTX sparse decompression instruction.
    ///
    /// Vector register counts are deliberately derived by the implementation:
    /// they are consequences of these five PTX spelling facts, not additional
    /// specialization axes that generated artifacts could state inconsistently.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SpDecompress<
        const ELEM_BITS: usize,
        const INDEX_BITS: usize,
        const SRC: usize,
        const DST: usize,
        const NUM: usize,
    >;

    /// Exact signedness and result slice of one PTX `mul24` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Mul24<T, Mode>(PhantomData<fn(T, Mode)>);

    /// Exact signedness, result slice, and saturation of one PTX `mad24` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Mad24<T, Mode, Clamp = NoSat>(PhantomData<fn(T, Mode, Clamp)>);

    /// Exact input type of one double-width PTX `mul.wide` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MulWide<T>(PhantomData<fn(T)>);

    /// Exact narrow multiplicand type of one double-width PTX `mad.wide` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MadWide<T>(PhantomData<fn(T)>);

    /// Integer high-product and saturation modifiers; plain wrapping forms
    /// continue using their existing scalar markers.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct IntegerArithmetic<T, Mode, Clamp = NoSat>(PhantomData<fn(T, Mode, Clamp)>);
    pub type MulHiU32 = IntegerArithmetic<U32, Hi>;

    /// Exact input signednesses and byte-pair selection of one PTX `dp2a` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Dp2a<A, B, Mode>(PhantomData<fn(A, B, Mode)>);

    /// Exact input signednesses of one PTX `dp4a` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Dp4a<A, B>(PhantomData<fn(A, B)>);

    /// Scalar `f32` arithmetic modifiers. Division additionally accepts Approx
    /// and Full in the mode position, but never a saturation modifier.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F32Arithmetic<Round = Rn, Subnormal = PreserveSubnormal, Clamp = NoSat>(
        PhantomData<fn(Round, Subnormal, Clamp)>,
    );

    /// Scalar binary64 arithmetic mode, with no independent FTZ or saturation
    /// axis. Reciprocal and reciprocal square root additionally accept Approx
    /// (the PTX 1.11.20 forms whose FTZ is mandatory).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F64Arithmetic<Round = Rn>(PhantomData<fn(Round)>);

    pub type F64Rn = F64Arithmetic<Rn>;
    pub type F32ApproxFtz = F32Arithmetic<Approx, Ftz>;

    /// Exact packed `.f32x2` arithmetic modifiers. Packed arithmetic is one
    /// PTX instruction, not two scalar ABI calls.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F32x2Arithmetic<Round = Rn, Subnormal = PreserveSubnormal>(
        PhantomData<fn(Round, Subnormal)>,
    );

    /// Exact modifiers and operand count of `min/max.f32`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F32MinMax<
        Subnormal = PreserveSubnormal,
        Nan = IgnoreNan,
        const ABS: bool = false,
        const XORSIGN: bool = false,
        const N: usize = 2,
    >(PhantomData<fn(Subnormal, Nan)>);
    pub type F32ThreeSource = F32MinMax<PreserveSubnormal, IgnoreNan, false, false, 3>;
    pub type F32ThreeSourceNan = F32MinMax<PreserveSubnormal, PropagateNan, false, false, 3>;

    /// Scalar and packed half min/max share the same per-component semantics.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct HalfMinMax<
        T,
        Subnormal = PreserveSubnormal,
        Nan = IgnoreNan,
        const XORSIGN: bool = false,
    >(PhantomData<fn(T, Subnormal, Nan)>);

    /// Integer min/max's optional negative-result clamp.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct IntegerMinMax<T, Activation>(PhantomData<fn(T, Activation)>);

    /// IEEE-rounded scalar square root. `T`, `Round`, and `Subnormal` are the
    /// complete PTX spelling axes; approximate forms use the scalar markers.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Sqrt<T, Round, Subnormal = PreserveSubnormal>(PhantomData<fn(T, Round, Subnormal)>);

    /// Mixed-precision `add/sub/fma.rnd.f32.atype` specialization.
    ///
    /// `Src` names the low-precision `atype`; binary operands follow PTX order
    /// `(atype, f32)`, while FMA uses `(atype, atype, f32)`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MixedF32<Src, Round = Rn, Clamp = NoSat>(PhantomData<fn(Src, Round, Clamp)>);

    /// PTX 9.4 mixed packed-low / `.f32x2` arithmetic.
    ///
    /// `Src` is exactly `F16x2` or `Bf16x2`; `Round` is the final f32
    /// rounding mode.  The same closed product serves binary add/sub and the
    /// three-source fused multiply-add because their carrier ABI is fixed by
    /// the instruction trait.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MixedF32x2<Src, Round>(PhantomData<fn(Src, Round)>);

    /// PTX 9.4 `.f32x2` arithmetic narrowed with mandatory `.rz`.
    ///
    /// Only the two destination markers implemented below are legal:
    /// `F16x2` also carries the syntax-mandated `.ftz`, while `Bf16x2`
    /// preserves subnormals.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MixedF32x2Down<Dst>(PhantomData<fn() -> Dst>);

    /// PTX 9.4 packed low-precision multiply with unlike source formats.
    ///
    /// Implementations exist only for `(Bf16x2, F16x2)` and
    /// `(F16x2, Bf16x2)`, keeping the source-conversion direction explicit.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MixedLowMul<Dst, Src>(PhantomData<fn(Src) -> Dst>);
}

macro_rules! binary_variant {
    ($spec:ident, $variant:ty, $scalar:ty, $operation:expr) => {
        register_variant! {
            [impl] $spec, $variant,
            (R<$scalar>, R<$scalar>) => R<$scalar>;
            |_context, _site, (lhs, rhs)| {
                let operation = $operation;
                Ok(lhs.zip_map(&rhs, |_lane, lhs, rhs| operation(*lhs, *rhs)))
            }
        }
    };
}

macro_rules! unary_variant {
    ($spec:ident, $variant:ty, $scalar:ty, $operation:expr) => {
        register_variant! {
            [impl] $spec, $variant,
            R<$scalar> => R<$scalar>;
            |_context, _site, source| {
                let operation = $operation;
                Ok(source.map(|_lane, value| operation(value)))
            }
        }
    };
}

macro_rules! ternary_variant {
    ($spec:ident, $variant:ty, $scalar:ty, $operation:expr) => {
        register_variant! {
            [impl] $spec, $variant,
            (R<$scalar>, R<$scalar>, R<$scalar>) => R<$scalar>;
            |_context, _site, (first, second, third)| {
                let operation = $operation;
                Ok(R::from_fn(|lane| {
                    operation(first[lane], second[lane], third[lane])
                }))
            }
        }
    };
}

trait HighProduct: RegisterType {
    fn high(lhs: Self::Scalar, rhs: Self::Scalar) -> Self::Scalar;
}

macro_rules! integer_arithmetic_variants {
    ($marker:ty, $scalar:ty, $wide:ty) => {
        binary_variant!(add_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            .wrapping_add(rhs));
        binary_variant!(sub_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            .wrapping_sub(rhs));
        binary_variant!(mul_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            .wrapping_mul(rhs));
        ternary_variant!(mad_spec, $marker, $scalar, |a: $scalar, b: $scalar, c: $scalar| {
            a.wrapping_mul(b).wrapping_add(c)
        });
        impl HighProduct for $marker {
            fn high(lhs: $scalar, rhs: $scalar) -> $scalar {
                ((lhs as $wide * rhs as $wide) >> <$scalar>::BITS) as $scalar
            }
        }
        binary_variant!(mul_spec, variant::IntegerArithmetic<$marker, variant::Hi>, $scalar,
            <$marker as HighProduct>::high);
        ternary_variant!(mad_spec, variant::IntegerArithmetic<$marker, variant::Hi>, $scalar,
            |a: $scalar, b: $scalar, c: $scalar| <$marker as HighProduct>::high(a, b).wrapping_add(c));
        binary_variant!(max_spec, $marker, $scalar, <$scalar>::max);
        binary_variant!(min_spec, $marker, $scalar, <$scalar>::min);
    };
}

integer_arithmetic_variants!(variant::I16, i16, i32);
integer_arithmetic_variants!(variant::I32, i32, i64);
integer_arithmetic_variants!(variant::I64, i64, i128);
integer_arithmetic_variants!(variant::U16, u16, u32);
integer_arithmetic_variants!(variant::U32, u32, u64);
integer_arithmetic_variants!(variant::U64, u64, u128);
binary_variant!(add_spec, variant::IntegerArithmetic<variant::I32, variant::Lo, variant::Sat>,
    i32, i32::saturating_add);
binary_variant!(sub_spec, variant::IntegerArithmetic<variant::I32, variant::Lo, variant::Sat>,
    i32, i32::saturating_sub);
ternary_variant!(mad_spec, variant::IntegerArithmetic<variant::I32, variant::Hi, variant::Sat>,
    i32, |a: i32, b: i32, c: i32| <variant::I32 as HighProduct>::high(a, b).saturating_add(c));

fn add_16x2(lhs: u32, rhs: u32) -> u32 {
    // Signed and unsigned packed addition have identical wrapping bits.
    u32::from((lhs as u16).wrapping_add(rhs as u16))
        | (u32::from(((lhs >> 16) as u16).wrapping_add((rhs >> 16) as u16)) << 16)
}
binary_variant!(add_spec, variant::I16x2, u32, add_16x2);
binary_variant!(add_spec, variant::U16x2, u32, add_16x2);

fn minmax_16x2<const SIGNED: bool, const RELU: bool, const MAX: bool>(a: u32, b: u32) -> u32 {
    let mut result = 0;
    for shift in [0, 16] {
        let decode = |bits: u32| {
            if SIGNED {
                (bits >> shift) as i16 as i32
            } else {
                (bits >> shift) as u16 as i32
            }
        };
        let (a, b) = (decode(a), decode(b));
        let value = if MAX { a.max(b) } else { a.min(b) };
        let value = if RELU { value.max(0) } else { value };
        result |= u32::from(value as u16) << shift;
    }
    result
}
binary_variant!(
    min_spec,
    variant::I16x2,
    u32,
    minmax_16x2::<true, false, false>
);
binary_variant!(
    max_spec,
    variant::I16x2,
    u32,
    minmax_16x2::<true, false, true>
);
binary_variant!(
    min_spec,
    variant::U16x2,
    u32,
    minmax_16x2::<false, false, false>
);
binary_variant!(
    max_spec,
    variant::U16x2,
    u32,
    minmax_16x2::<false, false, true>
);
binary_variant!(min_spec, variant::IntegerMinMax<variant::I16x2, variant::Relu>, u32,
    minmax_16x2::<true, true, false>);
binary_variant!(max_spec, variant::IntegerMinMax<variant::I16x2, variant::Relu>, u32,
    minmax_16x2::<true, true, true>);
binary_variant!(min_spec, variant::IntegerMinMax<variant::I32, variant::Relu>, i32,
    |a: i32, b: i32| a.min(b).max(0));
binary_variant!(max_spec, variant::IntegerMinMax<variant::I32, variant::Relu>, i32,
    |a: i32, b: i32| a.max(b).max(0));

macro_rules! sad_variant {
    ($marker:ty, $scalar:ty) => {
        ternary_variant!(
            sad_spec,
            $marker,
            $scalar,
            |a: $scalar, b: $scalar, c: $scalar| {
                let difference = if a < b {
                    b.wrapping_sub(a)
                } else {
                    a.wrapping_sub(b)
                };
                c.wrapping_add(difference)
            }
        );
    };
}

sad_variant!(variant::I16, i16);
sad_variant!(variant::I32, i32);
sad_variant!(variant::I64, i64);
sad_variant!(variant::U16, u16);
sad_variant!(variant::U32, u32);
sad_variant!(variant::U64, u64);

macro_rules! mul_wide_variant {
    ($marker:ty, $input:ty, $output:ty, $operation:expr) => {
        register_variant! {
            [impl] mul_spec, variant::MulWide<$marker>,
            (R<$input>, R<$input>) => R<$output>;
            |_context, _site, (lhs, rhs)| {
                let operation = $operation;
                Ok(lhs.zip_map(&rhs, |_lane, lhs, rhs| operation(*lhs, *rhs)))
            }
        }
    };
}

mul_wide_variant!(variant::I16, i16, i32, |lhs: i16, rhs: i16| {
    i32::from(lhs) * i32::from(rhs)
});
mul_wide_variant!(variant::U16, u16, u32, |lhs: u16, rhs: u16| {
    u32::from(lhs) * u32::from(rhs)
});
mul_wide_variant!(variant::I32, i32, i64, |lhs: i32, rhs: i32| {
    i64::from(lhs) * i64::from(rhs)
});
mul_wide_variant!(variant::U32, u32, u64, |lhs: u32, rhs: u32| {
    u64::from(lhs) * u64::from(rhs)
});

macro_rules! mad_wide_variant {
    ($marker:ty, $input:ty, $output:ty, $operation:expr) => {
        register_variant! {
            [impl] mad_spec, variant::MadWide<$marker>,
            (R<$input>, R<$input>, R<$output>) => R<$output>;
            |_context, _site, (lhs, rhs, addend)| {
                let operation = $operation;
                Ok(R::from_fn(|lane| {
                    operation(lhs[lane], rhs[lane], addend[lane])
                }))
            }
        }
    };
}

mad_wide_variant!(variant::I16, i16, i32, |lhs: i16, rhs: i16, addend: i32| {
    (i32::from(lhs) * i32::from(rhs)).wrapping_add(addend)
});
mad_wide_variant!(variant::U16, u16, u32, |lhs: u16, rhs: u16, addend: u32| {
    (u32::from(lhs) * u32::from(rhs)).wrapping_add(addend)
});
mad_wide_variant!(variant::I32, i32, i64, |lhs: i32, rhs: i32, addend: i64| {
    (i64::from(lhs) * i64::from(rhs)).wrapping_add(addend)
});
mad_wide_variant!(variant::U32, u32, u64, |lhs: u32, rhs: u32, addend: u64| {
    (u64::from(lhs) * u64::from(rhs)).wrapping_add(addend)
});

fn signed_24(value: i32) -> i64 {
    i64::from(((value as u32 & 0x00ff_ffff) << 8) as i32 >> 8)
}

fn mul24_s32(lhs: i32, rhs: i32, high: bool) -> i32 {
    let product = signed_24(lhs) * signed_24(rhs);
    if high {
        (product >> 16) as i32
    } else {
        product as i32
    }
}

fn mul24_u32(lhs: u32, rhs: u32, high: bool) -> u32 {
    let product = u64::from(lhs & 0x00ff_ffff) * u64::from(rhs & 0x00ff_ffff);
    if high {
        (product >> 16) as u32
    } else {
        product as u32
    }
}

binary_variant!(mul24_spec, variant::Mul24<variant::I32, variant::Lo>, i32, |lhs, rhs| {
    mul24_s32(lhs, rhs, false)
});
binary_variant!(mul24_spec, variant::Mul24<variant::I32, variant::Hi>, i32, |lhs, rhs| {
    mul24_s32(lhs, rhs, true)
});
binary_variant!(mul24_spec, variant::Mul24<variant::U32, variant::Lo>, u32, |lhs, rhs| {
    mul24_u32(lhs, rhs, false)
});
binary_variant!(mul24_spec, variant::Mul24<variant::U32, variant::Hi>, u32, |lhs, rhs| {
    mul24_u32(lhs, rhs, true)
});

ternary_variant!(
    mad24_spec,
    variant::Mad24<variant::I32, variant::Lo, variant::NoSat>,
    i32,
    |lhs, rhs, addend| mul24_s32(lhs, rhs, false).wrapping_add(addend)
);
ternary_variant!(
    mad24_spec,
    variant::Mad24<variant::I32, variant::Hi, variant::NoSat>,
    i32,
    |lhs, rhs, addend| mul24_s32(lhs, rhs, true).wrapping_add(addend)
);
ternary_variant!(
    mad24_spec,
    variant::Mad24<variant::U32, variant::Lo, variant::NoSat>,
    u32,
    |lhs, rhs, addend| mul24_u32(lhs, rhs, false).wrapping_add(addend)
);
ternary_variant!(
    mad24_spec,
    variant::Mad24<variant::U32, variant::Hi, variant::NoSat>,
    u32,
    |lhs, rhs, addend| mul24_u32(lhs, rhs, true).wrapping_add(addend)
);
ternary_variant!(
    mad24_spec,
    variant::Mad24<variant::I32, variant::Hi, variant::Sat>,
    i32,
    |lhs, rhs, addend| (i64::from(mul24_s32(lhs, rhs, true)) + i64::from(addend))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
);

fn packed_element(word: u32, index: usize, width: usize, signed: bool) -> i64 {
    let mask = (1_u32 << width) - 1;
    let raw = (word >> (index * width)) & mask;
    if signed && raw & (1_u32 << (width - 1)) != 0 {
        i64::from(raw) - (1_i64 << width)
    } else {
        i64::from(raw)
    }
}

fn dp4a_bits(a: u32, b: u32, c: u32, signed_a: bool, signed_b: bool) -> u32 {
    let accumulator = if signed_a || signed_b {
        i64::from(c as i32)
    } else {
        i64::from(c)
    };
    (0..4).fold(accumulator, |sum, index| {
        sum + packed_element(a, index, 8, signed_a) * packed_element(b, index, 8, signed_b)
    }) as u32
}

fn dp2a_bits(a: u32, b: u32, c: u32, signed_a: bool, signed_b: bool, high: bool) -> u32 {
    let accumulator = if signed_a || signed_b {
        i64::from(c as i32)
    } else {
        i64::from(c)
    };
    let first_byte = if high { 2 } else { 0 };
    (0..2).fold(accumulator, |sum, index| {
        sum + packed_element(a, index, 16, signed_a)
            * packed_element(b, first_byte + index, 8, signed_b)
    }) as u32
}

macro_rules! dp4a_variant {
    ($a_marker:ty, $b_marker:ty, $a:ty, $b:ty, $acc:ty, $signed_a:expr, $signed_b:expr) => {
        register_variant! {
            [impl] dp4a_spec, variant::Dp4a<$a_marker, $b_marker>,
            (R<$a>, R<$b>, R<$acc>) => R<$acc>;
            |_context, _site, (a, b, c)| {
                Ok(R::from_fn(|lane| {
                    dp4a_bits(
                        a[lane] as u32,
                        b[lane] as u32,
                        c[lane] as u32,
                        $signed_a,
                        $signed_b,
                    ) as $acc
                }))
            }
        }
    };
}

dp4a_variant!(variant::U32, variant::U32, u32, u32, u32, false, false);
dp4a_variant!(variant::U32, variant::I32, u32, i32, i32, false, true);
dp4a_variant!(variant::I32, variant::U32, i32, u32, i32, true, false);
dp4a_variant!(variant::I32, variant::I32, i32, i32, i32, true, true);

macro_rules! dp2a_variant {
    (
        $a_marker:ty, $b_marker:ty, $mode:ty, $a:ty, $b:ty, $acc:ty,
        $signed_a:expr, $signed_b:expr, $high:expr
    ) => {
        register_variant! {
            [impl] dp2a_spec, variant::Dp2a<$a_marker, $b_marker, $mode>,
            (R<$a>, R<$b>, R<$acc>) => R<$acc>;
            |_context, _site, (a, b, c)| {
                Ok(R::from_fn(|lane| {
                    dp2a_bits(
                        a[lane] as u32,
                        b[lane] as u32,
                        c[lane] as u32,
                        $signed_a,
                        $signed_b,
                        $high,
                    ) as $acc
                }))
            }
        }
    };
}

dp2a_variant!(
    variant::U32,
    variant::U32,
    variant::Lo,
    u32,
    u32,
    u32,
    false,
    false,
    false
);
dp2a_variant!(
    variant::U32,
    variant::I32,
    variant::Lo,
    u32,
    i32,
    i32,
    false,
    true,
    false
);
dp2a_variant!(
    variant::I32,
    variant::U32,
    variant::Lo,
    i32,
    u32,
    i32,
    true,
    false,
    false
);
dp2a_variant!(
    variant::I32,
    variant::I32,
    variant::Lo,
    i32,
    i32,
    i32,
    true,
    true,
    false
);
dp2a_variant!(
    variant::U32,
    variant::U32,
    variant::Hi,
    u32,
    u32,
    u32,
    false,
    false,
    true
);
dp2a_variant!(
    variant::U32,
    variant::I32,
    variant::Hi,
    u32,
    i32,
    i32,
    false,
    true,
    true
);
dp2a_variant!(
    variant::I32,
    variant::U32,
    variant::Hi,
    i32,
    u32,
    i32,
    true,
    false,
    true
);
dp2a_variant!(
    variant::I32,
    variant::I32,
    variant::Hi,
    i32,
    i32,
    i32,
    true,
    true,
    true
);

fn carryless_product_u64(lhs: u64, rhs: u64) -> u128 {
    let mut product = 0_u128;
    for bit in 0..64 {
        if lhs & (1_u64 << bit) != 0 {
            product ^= u128::from(rhs) << bit;
        }
    }
    product
}

ternary_variant!(
    clmad_spec,
    variant::ClmadLo,
    u64,
    |lhs, rhs, addend| (carryless_product_u64(lhs, rhs) as u64) ^ addend
);
ternary_variant!(
    clmad_spec,
    variant::ClmadHi,
    u64,
    |lhs, rhs, addend| ((carryless_product_u64(lhs, rhs) >> 64) as u64) ^ addend
);

fn low_mask_u64(width: u32) -> u64 {
    if width >= u64::BITS {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}

fn check_bit_field_controls(
    context: ExecCtx,
    site: SiteId,
    instruction: &str,
    position: &R<u32>,
    length: &R<u32>,
) -> Result<(), EngineError> {
    for lane in context.active_mask() {
        if position[lane] > 255 || length[lane] > 255 {
            return Err(EngineError::message(format!(
                "{instruction} position/length is outside the defined 0..255 range at site {} lane {lane}: {}/{}",
                site.get(), position[lane], length[lane]
            )));
        }
    }
    Ok(())
}

fn bit_field_extract(value: u64, position: u32, length: u32, width: u32, signed: bool) -> u64 {
    // Active controls were checked at the instruction boundary. Bound the
    // ignored lanes too, so their placeholders cannot overflow host arithmetic.
    let position = position & 0xff;
    let length = length & 0xff;
    if length == 0 {
        return 0;
    }

    let copied = if position >= width {
        0
    } else {
        length.min(width - position)
    };
    let copied_mask = low_mask_u64(copied);
    let mut result = if copied == 0 {
        0
    } else {
        (value >> position) & copied_mask
    };

    if signed {
        let sign_position = (position + length - 1).min(width - 1);
        if value & (1_u64 << sign_position) != 0 {
            result |= !copied_mask;
        }
    }
    result & low_mask_u64(width)
}

macro_rules! bfe_variant {
    ($marker:ty, $scalar:ty, $width:expr, $signed:expr) => {
        register_variant! {
            [impl] bfe_spec, $marker,
            (R<$scalar>, R<u32>, R<u32>) => R<$scalar>;
            |context, site, (value, position, length)| {
                check_bit_field_controls(context, site, "bfe", &position, &length)?;
                Ok(R::from_fn(|lane| {
                    bit_field_extract(
                        value[lane] as u64,
                        position[lane],
                        length[lane],
                        $width,
                        $signed,
                    ) as $scalar
                }))
            }
        }
    };
}

bfe_variant!(variant::U32, u32, 32, false);
bfe_variant!(variant::U64, u64, 64, false);
bfe_variant!(variant::I32, i32, 32, true);
bfe_variant!(variant::I64, i64, 64, true);

fn bit_field_insert(source: u64, base: u64, position: u32, length: u32, width: u32) -> u64 {
    let position = position & 0xff;
    let length = length & 0xff;
    let copied = if position >= width {
        0
    } else {
        length.min(width - position)
    };
    if copied == 0 {
        return base & low_mask_u64(width);
    }
    let source_mask = low_mask_u64(copied);
    let insertion_mask = source_mask << position;
    ((base & !insertion_mask) | ((source & source_mask) << position)) & low_mask_u64(width)
}

macro_rules! bfi_variant {
    ($marker:ty, $scalar:ty, $width:expr) => {
        register_variant! {
            [impl] bfi_spec, $marker,
            (R<$scalar>, R<$scalar>, R<u32>, R<u32>) => R<$scalar>;
            |context, site, (source, base, position, length)| {
                check_bit_field_controls(context, site, "bfi", &position, &length)?;
                Ok(R::from_fn(|lane| {
                    bit_field_insert(
                        source[lane] as u64,
                        base[lane] as u64,
                        position[lane],
                        length[lane],
                        $width,
                    ) as $scalar
                }))
            }
        }
    };
}

bfi_variant!(variant::B32, u32, 32);
bfi_variant!(variant::B64, u64, 64);

fn most_significant_non_sign_bit(value: u64, width: u32, signed: bool) -> u32 {
    let width_mask = low_mask_u64(width);
    let mut bits = value & width_mask;
    if signed && bits & (1_u64 << (width - 1)) != 0 {
        bits = !bits & width_mask;
    }
    if bits == 0 {
        u32::MAX
    } else {
        u64::BITS - 1 - bits.leading_zeros()
    }
}

macro_rules! bfind_variant {
    ($type_marker:ty, $mode_marker:ty, $scalar:ty, $width:expr, $signed:expr, $shift:expr) => {
        register_variant! {
            [impl] bfind_spec, variant::Bfind<$type_marker, $mode_marker>,
            R<$scalar> => R<u32>;
            |_context, _site, value| {
                Ok(value.map(|_lane, value| {
                    let position = most_significant_non_sign_bit(value as u64, $width, $signed);
                    if $shift && position != u32::MAX {
                        $width - 1 - position
                    } else {
                        position
                    }
                }))
            }
        }
    };
}

macro_rules! bfind_type_variants {
    ($type_marker:ty, $scalar:ty, $width:expr, $signed:expr) => {
        bfind_variant!(
            $type_marker,
            variant::BitPosition,
            $scalar,
            $width,
            $signed,
            false
        );
        bfind_variant!(
            $type_marker,
            variant::ShiftAmount,
            $scalar,
            $width,
            $signed,
            true
        );
    };
}

bfind_type_variants!(variant::U32, u32, 32, false);
bfind_type_variants!(variant::U64, u64, 64, false);
bfind_type_variants!(variant::I32, i32, 32, true);
bfind_type_variants!(variant::I64, i64, 64, true);

fn bit_mask(position: u32, width: u32, clamp: bool) -> u32 {
    let position = if clamp {
        if position >= 32 {
            return 0;
        }
        position
    } else {
        position & 0x1f
    };
    let width = if clamp { width.min(32) } else { width & 0x1f };
    if width == 0 {
        return 0;
    }
    let width = width.min(32 - position);
    (low_mask_u64(width) as u32) << position
}

binary_variant!(
    bmsk_spec,
    variant::Bmsk<variant::Clamp>,
    u32,
    |position, width| bit_mask(position, width, true)
);
binary_variant!(
    bmsk_spec,
    variant::Bmsk<variant::Wrap>,
    u32,
    |position, width| bit_mask(position, width, false)
);

macro_rules! unary_to_u32_variant {
    ($spec:ident, $marker:ty, $scalar:ty, $operation:expr) => {
        register_variant! {
            [impl] $spec, $marker,
            R<$scalar> => R<u32>;
            |_context, _site, source| {
                let operation: fn($scalar) -> u32 = $operation;
                Ok(source.map(|_lane, value| operation(value)))
            }
        }
    };
}

unary_variant!(brev_spec, variant::B32, u32, u32::reverse_bits);
unary_variant!(brev_spec, variant::B64, u64, u64::reverse_bits);
unary_to_u32_variant!(clz_spec, variant::B32, u32, u32::leading_zeros);
unary_to_u32_variant!(clz_spec, variant::B64, u64, u64::leading_zeros);
unary_variant!(cnot_spec, variant::B16, u16, |value: u16| u16::from(
    value == 0
));
unary_variant!(cnot_spec, variant::B32, u32, |value: u32| u32::from(
    value == 0
));
unary_variant!(cnot_spec, variant::B64, u64, |value: u64| u64::from(
    value == 0
));
unary_to_u32_variant!(popc_spec, variant::B32, u32, u32::count_ones);
unary_to_u32_variant!(popc_spec, variant::B64, u64, u64::count_ones);

fn funnel_shift(low: u32, high: u32, shift: u32, left: bool, clamp: bool) -> u32 {
    let shift = if clamp { shift.min(32) } else { shift & 0x1f };
    let joined = (u64::from(high) << 32) | u64::from(low);
    if left {
        ((joined << shift) >> 32) as u32
    } else {
        (joined >> shift) as u32
    }
}

ternary_variant!(
    shf_spec,
    variant::Shf<variant::Left, variant::Clamp>,
    u32,
    |low, high, shift| funnel_shift(low, high, shift, true, true)
);
ternary_variant!(
    shf_spec,
    variant::Shf<variant::Left, variant::Wrap>,
    u32,
    |low, high, shift| funnel_shift(low, high, shift, true, false)
);
ternary_variant!(
    shf_spec,
    variant::Shf<variant::Right, variant::Clamp>,
    u32,
    |low, high, shift| funnel_shift(low, high, shift, false, true)
);
ternary_variant!(
    shf_spec,
    variant::Shf<variant::Right, variant::Wrap>,
    u32,
    |low, high, shift| funnel_shift(low, high, shift, false, false)
);

fn zero_extend_32(value: u32, width: u32, clamp: bool) -> u32 {
    if clamp && width >= 32 {
        return value;
    }
    value & (low_mask_u64(width & 0x1f) as u32)
}

fn sign_extend_32(value: i32, width: u32, clamp: bool) -> i32 {
    if clamp && width >= 32 {
        return value;
    }
    let width = width & 0x1f;
    if width == 0 {
        return 0;
    }
    let mask = low_mask_u64(width) as u32;
    let bits = value as u32 & mask;
    if bits & (1_u32 << (width - 1)) != 0 {
        (bits | !mask) as i32
    } else {
        bits as i32
    }
}

macro_rules! szext_variant {
    ($type_marker:ty, $mode:ty, $scalar:ty, $operation:expr, $clamp:expr) => {
        register_variant! {
            [impl] szext_spec, variant::Szext<$type_marker, $mode>,
            (R<$scalar>, R<u32>) => R<$scalar>;
            |_context, _site, (value, width)| {
                let operation: fn($scalar, u32, bool) -> $scalar = $operation;
                Ok(value.zip_map(&width, |_lane, value, width| {
                    operation(*value, *width, $clamp)
                }))
            }
        }
    };
}

szext_variant!(variant::U32, variant::Clamp, u32, zero_extend_32, true);
szext_variant!(variant::U32, variant::Wrap, u32, zero_extend_32, false);
szext_variant!(variant::I32, variant::Clamp, i32, sign_extend_32, true);
szext_variant!(variant::I32, variant::Wrap, i32, sign_extend_32, false);

trait ValidSpDecompressShape {
    const METADATA_REGISTERS: usize;
    const COMPRESSED_REGISTERS: usize;
    const DATA_REGISTERS: usize;
}

macro_rules! valid_spdecompress_shapes {
    ($elem:literal, $index:literal, $src:literal, $dst:literal; $($num:literal),+ $(,)?) => {
        $(
            impl ValidSpDecompressShape
                for variant::SpDecompress<$elem, $index, $src, $dst, $num>
            {
                const METADATA_REGISTERS: usize =
                    (($src as usize) * ($index as usize) * ($num as usize)).div_ceil(32);
                const COMPRESSED_REGISTERS: usize =
                    (($src as usize) * ($elem as usize) * ($num as usize)).div_ceil(32);
                const DATA_REGISTERS: usize =
                    (($dst as usize) * ($elem as usize) * ($num as usize)).div_ceil(32);
            }
        )+
    };
}

// The closed PTX 9.4 domain after applying all five vector-size constraints.
valid_spdecompress_shapes!(8, 2, 1, 2; 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 2, 1, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 2, 2, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 1, 2; 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 1, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 1, 8; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 1, 16; 1, 2, 4, 8, 16, 32);
valid_spdecompress_shapes!(8, 4, 2, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 2, 8; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 2, 16; 1, 2, 4, 8, 16, 32);
valid_spdecompress_shapes!(8, 4, 4, 8; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(8, 4, 4, 16; 1, 2, 4, 8, 16, 32);
valid_spdecompress_shapes!(16, 2, 1, 2; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 2, 1, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 2, 2, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 4, 1, 2; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 4, 1, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 4, 1, 8; 1, 2, 4, 8, 16, 32);
valid_spdecompress_shapes!(16, 4, 1, 16; 1, 2, 4, 8, 16);
valid_spdecompress_shapes!(16, 4, 2, 4; 1, 2, 4, 8, 16, 32, 64);
valid_spdecompress_shapes!(16, 4, 2, 8; 1, 2, 4, 8, 16, 32);
valid_spdecompress_shapes!(16, 4, 2, 16; 1, 2, 4, 8, 16);

register_variant! {
    [impl<
        const ELEM_BITS: usize,
        const INDEX_BITS: usize,
        const SRC: usize,
        const DST: usize,
        const NUM: usize,
    >] spdecompress_spec, variant::SpDecompress<ELEM_BITS, INDEX_BITS, SRC, DST, NUM>
    where [
        variant::SpDecompress<ELEM_BITS, INDEX_BITS, SRC, DST, NUM>: ValidSpDecompressShape,
    ],
    (Vec<R<u32>>, Vec<R<u32>>) => Vec<R<u32>>;
    |context, site, (metadata, compressed)| {
        let metadata_registers = <Self as ValidSpDecompressShape>::METADATA_REGISTERS;
        let compressed_registers = <Self as ValidSpDecompressShape>::COMPRESSED_REGISTERS;
        let data_registers = <Self as ValidSpDecompressShape>::DATA_REGISTERS;
        if metadata.len() != metadata_registers || compressed.len() != compressed_registers {
            return Err(EngineError::message(format!(
                "spdecompress operand-vector length mismatch at site {}: expected ({metadata_registers}, {compressed_registers}), got ({}, {})",
                site.get(),
                metadata.len(),
                compressed.len()
            )));
        }

        // PTX initializes the entire dense vector to zero. Each compressed
        // element then overwrites the indexed destination; this also gives
        // the specified last-source-wins behaviour for duplicate indices.
        let mut data = vec![R::splat(0_u32); data_registers];
        let index_mask = (1_u32 << INDEX_BITS) - 1;
        let element_mask = (1_u32 << ELEM_BITS) - 1;
        for lane in context.active_mask() {
            for repetition in 0..NUM {
                for source in 0..SRC {
                    let packed_source = repetition * SRC + source;
                    let metadata_bit = packed_source * INDEX_BITS;
                    let destination = ((metadata[metadata_bit / 32][lane] >> (metadata_bit % 32))
                        & index_mask) as usize;
                    if destination >= DST {
                        return Err(EngineError::message(format!(
                            "spdecompress metadata index {destination} is outside 0..{DST} at site {} lane {lane}, repetition {repetition}, source {source}",
                            site.get()
                        )));
                    }

                    let compressed_bit = packed_source * ELEM_BITS;
                    let value = (compressed[compressed_bit / 32][lane] >> (compressed_bit % 32))
                        & element_mask;
                    let data_bit = (repetition * DST + destination) * ELEM_BITS;
                    let register = &mut data[data_bit / 32][lane];
                    let shift = data_bit % 32;
                    *register = (*register & !(element_mask << shift)) | (value << shift);
                }
            }
        }
        Ok(data)
    }
}

register_variant! {
    [impl] mov_pack_spec, variant::B32,
    (R<u16>, R<u16>) => R<u32>;
    |_context, _site, (low, high)| {
        Ok(low.zip_map(&high, |_lane, low, high| {
            u32::from(*low) | (u32::from(*high) << 16)
        }))
    }
}

register_variant! {
    [impl] mov_pack_spec, variant::B16x4,
    (R<u16>, R<u16>, R<u16>, R<u16>) => R<u64>;
    |_context, _site, (x, y, z, w)| {
        Ok(R::from_fn(|lane| {
            crate::scalar::ptx_mov_pack_b16x4([x[lane], y[lane], z[lane], w[lane]])
        }))
    }
}

register_variant! {
    [impl] mov_unpack_spec, variant::B16x4,
    R<u64> => (R<u16>, R<u16>, R<u16>, R<u16>);
    |_context, _site, source| {
        let lanes = std::array::from_fn(|index| {
            source
                .clone()
                .map(|_lane, value| crate::scalar::ptx_mov_unpack_b16x4(value)[index])
        });
        let [x, y, z, w] = lanes;
        Ok((x, y, z, w))
    }
}

// Scalar PTX `neg` has signed integer spellings only.  Unsigned and scalar
// 8-bit markers remain usable by storage/conversion forms but deliberately do
// not implement arithmetic variants that PTX does not define.
unary_variant!(neg_spec, variant::I16, i16, i16::wrapping_neg);
unary_variant!(neg_spec, variant::I32, i32, i32::wrapping_neg);
unary_variant!(neg_spec, variant::I64, i64, i64::wrapping_neg);
unary_variant!(abs_spec, variant::I16, i16, i16::wrapping_abs);
unary_variant!(abs_spec, variant::I32, i32, i32::wrapping_abs);
unary_variant!(abs_spec, variant::I64, i64, i64::wrapping_abs);

macro_rules! bit_variants {
    ($marker:ty, $scalar:ty) => {
        binary_variant!(and_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            & rhs);
        binary_variant!(or_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            | rhs);
        binary_variant!(xor_spec, $marker, $scalar, |lhs: $scalar, rhs: $scalar| lhs
            ^ rhs);
        unary_variant!(not_spec, $marker, $scalar, |value: $scalar| !value);
    };
}

bit_variants!(variant::B16, u16);
bit_variants!(variant::B32, u32);
bit_variants!(variant::B64, u64);
bit_variants!(variant::I32, i32);
bit_variants!(variant::I64, i64);
bit_variants!(variant::U32, u32);
bit_variants!(variant::U64, u64);
bit_variants!(variant::Pred, bool);

fn lop3_word<const LUT: u8>(a: u32, b: u32, c: u32) -> u32 {
    let mut result = 0_u32;
    for row in 0_u8..8 {
        if LUT & (1_u8 << row) == 0 {
            continue;
        }
        let a_mask = if row & 0b100 == 0 { !a } else { a };
        let b_mask = if row & 0b010 == 0 { !b } else { b };
        let c_mask = if row & 0b001 == 0 { !c } else { c };
        result |= a_mask & b_mask & c_mask;
    }
    result
}

register_variant! {
    [impl<const LUT: u8>] lop3_spec, variant::Lop3<LUT>,
    (R<u32>, R<u32>, R<u32>) => R<u32>;
    |_context, _site, (a, b, c)| {
        Ok(R::from_fn(|lane| {
            lop3_word::<LUT>(a[lane], b[lane], c[lane])
        }))
    }
}

macro_rules! lop3_bool_variant {
    ($bool_op:ty, $combine:expr) => {
        register_variant! {
            [impl<const LUT: u8>] lop3_spec, variant::Lop3Bool<LUT, $bool_op>,
            (R<u32>, R<u32>, R<u32>, R<bool>) => (R<u32>, R<bool>);
            |_context, _site, (a, b, c, q)| {
                let data = R::from_fn(|lane| lop3_word::<LUT>(a[lane], b[lane], c[lane]));
                let combine = $combine;
                let predicate = data.zip_map(&q, |_lane, data, q| combine(*data != 0, *q));
                Ok((data, predicate))
            }
        }
    };
}

lop3_bool_variant!(variant::BoolAnd, |data: bool, q: bool| data && q);
lop3_bool_variant!(variant::BoolOr, |data: bool, q: bool| data || q);

register_variant! {
    [impl] mov_pack_spec, variant::B64,
    (R<u32>, R<u32>) => R<u64>;
    |_context, _site, (low, high)| {
        Ok(low.zip_map(&high, |_lane, low, high| {
            u64::from(*low) | (u64::from(*high) << 32)
        }))
    }
}

register_variant! {
    [impl] mov_pack_spec, variant::B128,
    (R<u64>, R<u64>) => R<U64x2>;
    |_context, _site, (low, high)| {
        Ok(low.zip_map(&high, |_lane, low, high| [*low, *high]))
    }
}

register_variant! {
    [impl] mov_pack_spec, variant::B32x4,
    (R<u32>, R<u32>, R<u32>, R<u32>) => R<U64x2>;
    |_context, _site, (x, y, z, w)| {
        Ok(R::from_fn(|lane| {
            crate::scalar::ptx_mov_pack_b32x4([x[lane], y[lane], z[lane], w[lane]])
        }))
    }
}

register_variant! {
    [impl] mov_unpack_spec, variant::B32x4,
    R<U64x2> => (R<u32>, R<u32>, R<u32>, R<u32>);
    |_context, _site, source| {
        let lanes = std::array::from_fn(|index| {
            source
                .clone()
                .map(|_lane, value| crate::scalar::ptx_mov_unpack_b32x4(value)[index])
        });
        let [x, y, z, w] = lanes;
        Ok((x, y, z, w))
    }
}

register_variant! {
    [impl] mov_unpack_spec, variant::B64x2,
    R<U64x2> => (R<u64>, R<u64>);
    |_context, _site, source| {
        let low = source
            .clone()
            .map(|_lane, value| crate::scalar::ptx_mov_unpack_b64x2(value)[0]);
        let high = source.map(|_lane, value| crate::scalar::ptx_mov_unpack_b64x2(value)[1]);
        Ok((low, high))
    }
}

register_variant! {
    [impl] mov_unpack_spec, variant::B64,
    R<u64> => (R<u32>, R<u32>);
    |_context, _site, source| {
        let low = source.clone().map(|_lane, value| value as u32);
        let high = source.map(|_lane, value| (value >> 32) as u32);
        Ok((low, high))
    }
}

macro_rules! cvt_pack_variant {
    ($bits:literal, $signed:literal, without_c) => {
        register_variant! {
            [impl] cvt_pack_spec, variant::CvtPack<$bits, $signed>,
            (R<i32>, R<i32>) => R<u32>;
            |_context, _site, (a, b)| {
                Ok(a.zip_map(&b, |_lane, a, b| {
                    crate::scalar::ptx_cvt_pack::<$bits, $signed>(*a, *b, 0)
                }))
            }
        }
    };
    ($bits:literal, $signed:literal, with_c) => {
        register_variant! {
            [impl] cvt_pack_spec, variant::CvtPack<$bits, $signed>,
            (R<i32>, R<i32>, R<u32>) => R<u32>;
            |_context, _site, (a, b, c)| {
                Ok(R::from_fn(|lane| {
                    crate::scalar::ptx_cvt_pack::<$bits, $signed>(a[lane], b[lane], c[lane])
                }))
            }
        }
    };
}

cvt_pack_variant!(16, false, without_c);
cvt_pack_variant!(16, true, without_c);
cvt_pack_variant!(8, false, with_c);
cvt_pack_variant!(8, true, with_c);
cvt_pack_variant!(4, false, with_c);
cvt_pack_variant!(4, true, with_c);
cvt_pack_variant!(2, false, with_c);
cvt_pack_variant!(2, true, with_c);

/// One `shl`/`shr` specialization: carrier plus the value an out-of-range
/// shift amount leaves behind.
///
/// PTX states for both mnemonics that "Shift amounts greater than the register
/// width N are clamped to N" (PTX ISA 9.7.8.8 `shl`, 9.7.8.9 `shr`), so an
/// amount of N or more produces the fully shifted-out result. Rust's
/// `wrapping_sh*` instead reduces the amount modulo N, which would return the
/// operand unchanged for a shift of exactly N; `checked_sh*` plus an explicit
/// clamped fill reproduces the PTX result.
///
/// `shr` fill follows PTX's own split: "Signed shifts fill with the sign bit,
/// unsigned and untyped shifts fill with 0". The bit-size types are untyped --
/// PTX notes "Bit-size types are included for symmetry with shl" -- so
/// `shr.b32`/`shr.b64` are logical shifts and bind the unsigned carrier and the
/// zero-fill arm rather than aliasing onto the signed rows.
macro_rules! shift_variant {
    ($spec:ident, $marker:ty, $scalar:ty, $method:ident, zero_fill) => {
        shift_variant!(@impl $spec, $marker, $scalar, $method, |_value: $scalar| 0);
    };
    ($spec:ident, $marker:ty, $scalar:ty, $method:ident, sign_fill) => {
        shift_variant!(@impl $spec, $marker, $scalar, $method,
            |value: $scalar| value >> (<$scalar>::BITS - 1));
    };
    (@impl $spec:ident, $marker:ty, $scalar:ty, $method:ident, $clamped:expr) => {
        register_variant! {
            [impl] $spec, $marker,
            (R<$scalar>, R<u32>) => R<$scalar>;
            |_context, _site, (value, shift)| {
                let clamped: fn($scalar) -> $scalar = $clamped;
                Ok(value.zip_map(&shift, |_lane, value, shift| {
                    value.$method(*shift).unwrap_or_else(|| clamped(*value))
                }))
            }
        }
    };
}

shift_variant!(shl_spec, variant::B16, u16, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::B32, u32, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::B64, u64, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::I32, i32, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::I64, i64, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::U32, u32, checked_shl, zero_fill);
shift_variant!(shl_spec, variant::U64, u64, checked_shl, zero_fill);
shift_variant!(shr_spec, variant::B16, u16, checked_shr, zero_fill);
shift_variant!(shr_spec, variant::U16, u16, checked_shr, zero_fill);
shift_variant!(shr_spec, variant::I16, i16, checked_shr, sign_fill);
shift_variant!(shr_spec, variant::B32, u32, checked_shr, zero_fill);
shift_variant!(shr_spec, variant::B64, u64, checked_shr, zero_fill);
shift_variant!(shr_spec, variant::I32, i32, checked_shr, sign_fill);
shift_variant!(shr_spec, variant::I64, i64, checked_shr, sign_fill);
shift_variant!(shr_spec, variant::U32, u32, checked_shr, zero_fill);
shift_variant!(shr_spec, variant::U64, u64, checked_shr, zero_fill);

binary_variant!(add_spec, variant::F32Rn, f32, |lhs: f32, rhs: f32| {
    crate::scalar::add_f32(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
binary_variant!(add_spec, variant::F32Rm, f32, |lhs: f32, rhs: f32| {
    crate::scalar::add_f32(lhs, rhs, crate::scalar::F32RoundingMode::Down)
});
binary_variant!(add_spec, variant::F32RnFtz, f32, |lhs: f32, rhs: f32| {
    crate::scalar::add_f32_ftz(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
binary_variant!(sub_spec, variant::F32Rn, f32, |lhs: f32, rhs: f32| {
    crate::scalar::sub_f32(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
binary_variant!(sub_spec, variant::F32Rm, f32, |lhs: f32, rhs: f32| {
    crate::scalar::sub_f32(lhs, rhs, crate::scalar::F32RoundingMode::Down)
});
binary_variant!(sub_spec, variant::F32RnFtz, f32, |lhs: f32, rhs: f32| {
    crate::scalar::sub_f32_ftz(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
binary_variant!(mul_spec, variant::F32Rn, f32, |lhs: f32, rhs: f32| {
    crate::scalar::mul_f32(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
binary_variant!(
    copysign_spec,
    variant::F32,
    f32,
    |sign: f32, magnitude: f32| {
        f32::from_bits((magnitude.to_bits() & 0x7fff_ffff) | (sign.to_bits() & 0x8000_0000))
    }
);
binary_variant!(
    copysign_spec,
    variant::F64,
    f64,
    |sign: f64, magnitude: f64| {
        f64::from_bits(
            (magnitude.to_bits() & 0x7fff_ffff_ffff_ffff)
                | (sign.to_bits() & 0x8000_0000_0000_0000),
        )
    }
);
binary_variant!(mul_spec, variant::F32RnFtz, f32, |lhs: f32, rhs: f32| {
    crate::scalar::mul_f32_ftz(lhs, rhs, crate::scalar::F32RoundingMode::Nearest)
});
ternary_variant!(fma_spec, variant::F32Rn, f32, crate::scalar::fma_f32_rn);
ternary_variant!(
    fma_spec,
    variant::F32RnFtz,
    f32,
    |a: f32, b: f32, c: f32| {
        crate::scalar::fma_f32_ftz(a, b, c, crate::scalar::F32RoundingMode::Nearest)
    }
);
binary_variant!(div_spec, variant::F32Rn, f32, crate::scalar::div_f32_rn);

pub(super) trait FloatRound {
    const MODE: crate::scalar::F32RoundingMode;
}

impl FloatRound for variant::Rn {
    const MODE: crate::scalar::F32RoundingMode = crate::scalar::F32RoundingMode::Nearest;
}
impl FloatRound for variant::Rz {
    const MODE: crate::scalar::F32RoundingMode = crate::scalar::F32RoundingMode::Zero;
}
impl FloatRound for variant::Rm {
    const MODE: crate::scalar::F32RoundingMode = crate::scalar::F32RoundingMode::Down;
}
impl FloatRound for variant::Rp {
    const MODE: crate::scalar::F32RoundingMode = crate::scalar::F32RoundingMode::Up;
}

trait F32DivideMode {
    fn divide(lhs: f32, rhs: f32, ftz: bool) -> f32;
}

impl<Round: FloatRound> F32DivideMode for Round {
    fn divide(lhs: f32, rhs: f32, ftz: bool) -> f32 {
        crate::scalar::ptx_div_f32(lhs, rhs, Round::MODE, ftz)
    }
}

impl F32DivideMode for variant::Approx {
    fn divide(lhs: f32, rhs: f32, ftz: bool) -> f32 {
        crate::scalar::ptx_div_approx_f32(lhs, rhs, ftz, false)
    }
}

impl F32DivideMode for variant::Full {
    fn divide(lhs: f32, rhs: f32, ftz: bool) -> f32 {
        crate::scalar::ptx_div_approx_f32(lhs, rhs, ftz, true)
    }
}

trait SubnormalMode {
    const FTZ: bool;
}

impl SubnormalMode for variant::PreserveSubnormal {
    const FTZ: bool = false;
}
impl SubnormalMode for variant::Ftz {
    const FTZ: bool = true;
}

trait F32Nan {
    const PROPAGATE: bool;
}

impl F32Nan for variant::IgnoreNan {
    const PROPAGATE: bool = false;
}

impl F32Nan for variant::PropagateNan {
    const PROPAGATE: bool = true;
}

trait F32Clamp {
    const SATURATE: bool;
}

impl F32Clamp for variant::NoSat {
    const SATURATE: bool = false;
}
impl F32Clamp for variant::Sat {
    const SATURATE: bool = true;
}

#[inline(always)]
fn apply_f32_clamp(value: f32, saturate: bool) -> f32 {
    if saturate {
        crate::scalar::ptx_saturate_f32(value)
    } else {
        value
    }
}

macro_rules! f32_arithmetic_variant {
    ($spec:ident, $arity:ident, $entry:ident) => {
        register_variant! {
            [impl<Round, Subnormal, Clamp>] $spec, variant::F32Arithmetic<Round, Subnormal, Clamp>
            where [
                Round: FloatRound,
                Subnormal: SubnormalMode,
                Clamp: F32Clamp,
            ],
            $arity<R<f32>> => R<f32>;
            |_context, _site, args| {
                Ok($entry(args, Round::MODE, Subnormal::FTZ, Clamp::SATURATE))
            }
        }
    };
}

type Pair<T> = (T, T);
type Triple<T> = (T, T, T);

#[inline(never)]
fn f32_add_entry(
    (lhs, rhs): (R<f32>, R<f32>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
    saturate: bool,
) -> R<f32> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        let value = if ftz {
            crate::scalar::add_f32_ftz(*lhs, *rhs, round)
        } else {
            crate::scalar::add_f32(*lhs, *rhs, round)
        };
        apply_f32_clamp(value, saturate)
    })
}

#[inline(never)]
fn f32_sub_entry(
    (lhs, rhs): (R<f32>, R<f32>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
    saturate: bool,
) -> R<f32> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        let value = if ftz {
            crate::scalar::sub_f32_ftz(*lhs, *rhs, round)
        } else {
            crate::scalar::sub_f32(*lhs, *rhs, round)
        };
        apply_f32_clamp(value, saturate)
    })
}

#[inline(never)]
fn f32_mul_entry(
    (lhs, rhs): (R<f32>, R<f32>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
    saturate: bool,
) -> R<f32> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        let value = if ftz {
            crate::scalar::mul_f32_ftz(*lhs, *rhs, round)
        } else {
            crate::scalar::mul_f32(*lhs, *rhs, round)
        };
        apply_f32_clamp(value, saturate)
    })
}

#[inline(never)]
fn f32_fma_entry(
    (lhs, rhs, addend): (R<f32>, R<f32>, R<f32>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
    saturate: bool,
) -> R<f32> {
    R::from_fn(|lane| {
        let value = if ftz {
            crate::scalar::fma_f32_ftz(lhs[lane], rhs[lane], addend[lane], round)
        } else {
            crate::scalar::fma_f32(lhs[lane], rhs[lane], addend[lane], round)
        };
        apply_f32_clamp(value, saturate)
    })
}

f32_arithmetic_variant!(add_spec, Pair, f32_add_entry);
f32_arithmetic_variant!(sub_spec, Pair, f32_sub_entry);
f32_arithmetic_variant!(mul_spec, Pair, f32_mul_entry);
f32_arithmetic_variant!(mad_spec, Triple, f32_fma_entry);
f32_arithmetic_variant!(fma_spec, Triple, f32_fma_entry);

register_variant! {
    [impl<Mode: F32DivideMode, Subnormal: SubnormalMode>] div_spec, variant::F32Arithmetic<Mode, Subnormal, variant::NoSat>,
    (R<f32>, R<f32>) => R<f32>;
    |_context, _site, (lhs, rhs)| {
        Ok(lhs.zip_map(&rhs, |_lane, a, b| Mode::divide(*a, *b, Subnormal::FTZ)))
    }
}

register_variant! {
    [impl<Round: FloatRound, Subnormal: SubnormalMode>] rcp_spec, variant::F32Arithmetic<Round, Subnormal, variant::NoSat>,
    R<f32> => R<f32>;
    |_context, _site, source| {
        Ok(source.map(|_lane, value| {
            crate::scalar::ptx_div_f32(1.0, value, Round::MODE, Subnormal::FTZ)
        }))
    }
}

register_variant! {
    [impl<Round: FloatRound>] rcp_spec, variant::F64Arithmetic<Round>,
    R<f64> => R<f64>;
    |_context, _site, source| {
        Ok(source.map(|_lane, value| crate::scalar::div_f64(1.0, value, Round::MODE)))
    }
}

unary_variant!(
    rcp_spec,
    variant::F64Arithmetic<variant::Approx>,
    f64,
    crate::scalar::ptx_rcp_approx_ftz_f64
);

macro_rules! f32_minmax_variant {
    ($spec:ident, $arity:ident, $n:literal, ($($arg:ident),+), $maximum:literal) => {
        register_variant! {
            [impl<S: SubnormalMode, Nan: F32Nan, const ABS: bool, const XOR: bool>] $spec, variant::F32MinMax<S, Nan, ABS, XOR, $n>,
            $arity<R<f32>> => R<f32>;
            |_context, _site, ($($arg),+)| {
                Ok(f32_minmax_entry([$($arg),+], S::FTZ, Nan::PROPAGATE, ABS, XOR, $maximum))
            }
        }
    };
}

f32_minmax_variant!(max_spec, Pair, 2, (a, b), true);
f32_minmax_variant!(min_spec, Pair, 2, (a, b), false);
f32_minmax_variant!(max_spec, Triple, 3, (a, b, c), true);
f32_minmax_variant!(min_spec, Triple, 3, (a, b, c), false);

#[inline(never)]
fn f32_minmax_entry<const N: usize>(
    args: [R<f32>; N],
    ftz: bool,
    propagate_nan: bool,
    absolute: bool,
    xor_sign: bool,
    maximum: bool,
) -> R<f32> {
    let operation = if maximum {
        crate::scalar::ptx_max_f32
    } else {
        crate::scalar::ptx_min_f32
    };
    R::from_fn(|lane| {
        let sign = (args[0][lane].to_bits() ^ args[1][lane].to_bits()) & 0x8000_0000;
        let values = std::array::from_fn::<_, N, _>(|index| {
            let value = args[index][lane];
            if absolute {
                value.abs()
            } else {
                value
            }
        });
        let result = values[1..].iter().fold(values[0], |acc, &next| {
            operation(acc, next, ftz, propagate_nan)
        });
        if xor_sign && !result.is_nan() {
            f32::from_bits((result.to_bits() & 0x7fff_ffff) | sign)
        } else {
            result
        }
    })
}

impl<Src, Round, Clamp> add_spec::sealed::Sealed for variant::MixedF32<Src, Round, Clamp>
where
    Src: HalfRegister<Scalar = u16>,
    Round: FloatRound,
    Clamp: F32Clamp,
{
}

register_variant! {
    [impl<Src, Round, Clamp>] sub_spec, variant::MixedF32<Src, Round, Clamp>
    where [
        Src: HalfRegister<Scalar = u16>,
        Round: FloatRound,
        Clamp: F32Clamp,
    ],
    (R<u16>, R<f32>) => R<f32>;
    |_context, _site, (low_precision, subtrahend)| {
        Ok(
            low_precision.zip_map(&subtrahend, |_lane, value, subtrahend| {
                apply_f32_clamp(
                    crate::scalar::sub_f32(
                        crate::scalar::decode_low(*value, Src::FORMAT),
                        *subtrahend,
                        Round::MODE,
                    ),
                    Clamp::SATURATE,
                )
            }),
        )
    }
}

register_variant! {
    [impl<Src, Round, Clamp>] fma_spec, variant::MixedF32<Src, Round, Clamp>
    where [
        Src: HalfRegister<Scalar = u16>,
        Round: FloatRound,
        Clamp: F32Clamp,
    ],
    (R<u16>, R<u16>, R<f32>) => R<f32>;
    |_context, _site, (lhs, rhs, addend)| {
        Ok(R::from_fn(|lane| {
            apply_f32_clamp(
                crate::scalar::fma_f32(
                    crate::scalar::decode_low(lhs[lane], Src::FORMAT),
                    crate::scalar::decode_low(rhs[lane], Src::FORMAT),
                    addend[lane],
                    Round::MODE,
                ),
                Clamp::SATURATE,
            )
        }))
    }
}

impl<Src, Round, Clamp> add_spec::Variant for variant::MixedF32<Src, Round, Clamp>
where
    Src: HalfRegister<Scalar = u16>,
    Round: FloatRound,
    Clamp: F32Clamp,
{
    type Args = (R<u16>, R<f32>);
    type Output = R<f32>;
}

impl<Src, Round, Clamp> add_spec::sealed::Execute for variant::MixedF32<Src, Round, Clamp>
where
    Src: HalfRegister<Scalar = u16>,
    Round: FloatRound,
    Clamp: F32Clamp,
{
    fn execute(
        _context: ExecCtx,
        _site: SiteId,
        (low_precision, addend): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Ok(mixed_f32_add_entry(
            (low_precision, addend),
            Src::FORMAT,
            Round::MODE,
            Clamp::SATURATE,
        ))
    }
}

#[inline(never)]
fn mixed_f32_add_entry(
    (low_precision, addend): (R<u16>, R<f32>),
    format: crate::scalar::LowPrecisionFormat,
    round: crate::scalar::F32RoundingMode,
    saturate: bool,
) -> R<f32> {
    low_precision.zip_map(&addend, |_lane, value, addend| {
        apply_f32_clamp(
            crate::scalar::add_f32(crate::scalar::decode_low(*value, format), *addend, round),
            saturate,
        )
    })
}

macro_rules! f32x2_arithmetic_variant {
    ($spec:ident, $arity:ident, $entry:ident) => {
        register_variant! {
            [impl<Round, Subnormal>] $spec, variant::F32x2Arithmetic<Round, Subnormal>
            where [
                Round: FloatRound,
                Subnormal: SubnormalMode,
            ],
            $arity<R<u64>> => R<u64>;
            |_context, _site, args| {
                Ok($entry(args, Round::MODE, Subnormal::FTZ))
            }
        }
    };
}

#[inline(never)]
fn f32x2_add_entry(
    (lhs, rhs): (R<u64>, R<u64>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
) -> R<u64> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        crate::scalar::add_f32x2(*lhs, *rhs, round, ftz)
    })
}

#[inline(never)]
fn f32x2_sub_entry(
    (lhs, rhs): (R<u64>, R<u64>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
) -> R<u64> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        crate::scalar::sub_f32x2(*lhs, *rhs, round, ftz)
    })
}

#[inline(never)]
fn f32x2_mul_entry(
    (lhs, rhs): (R<u64>, R<u64>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
) -> R<u64> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        crate::scalar::mul_f32x2(*lhs, *rhs, round, ftz)
    })
}

#[inline(never)]
fn f32x2_fma_entry(
    (lhs, rhs, addend): (R<u64>, R<u64>, R<u64>),
    round: crate::scalar::F32RoundingMode,
    ftz: bool,
) -> R<u64> {
    R::from_fn(|lane| crate::scalar::fma_f32x2(lhs[lane], rhs[lane], addend[lane], round, ftz))
}

f32x2_arithmetic_variant!(add_spec, Pair, f32x2_add_entry);
f32x2_arithmetic_variant!(sub_spec, Pair, f32x2_sub_entry);
f32x2_arithmetic_variant!(mul_spec, Pair, f32x2_mul_entry);
f32x2_arithmetic_variant!(fma_spec, Triple, f32x2_fma_entry);

trait MixedF32x2Source {
    fn decode(bits: u16) -> f32;
}

impl MixedF32x2Source for variant::F16x2 {
    fn decode(bits: u16) -> f32 {
        crate::scalar::cuda_fp16_bits_to_f32(bits)
    }
}

impl MixedF32x2Source for variant::Bf16x2 {
    fn decode(bits: u16) -> f32 {
        crate::bf16_bits_to_f32(bits)
    }
}

fn mixed_f32x2_source<Src: MixedF32x2Source>(packed: u32) -> (f32, f32) {
    (
        Src::decode(packed as u16),
        Src::decode((packed >> 16) as u16),
    )
}

fn mixed_f32x2_result(low: f32, high: f32) -> u64 {
    crate::scalar::make_float2(
        crate::scalar::cuda_canonicalize_nan_f32(low),
        crate::scalar::cuda_canonicalize_nan_f32(high),
    )
}

macro_rules! mixed_f32x2_binary_variant {
    ($spec:ident, $entry:ident) => {
        register_variant! {
            [impl<Src, Round>] $spec, variant::MixedF32x2<Src, Round>
            where [
                Src: MixedF32x2Source,
                Round: FloatRound,
            ],
            (R<u32>, R<u64>) => R<u64>;
            |_context, _site, args| {
                Ok($entry::<Src>(args, Round::MODE))
            }
        }
    };
}

#[inline(never)]
fn mixed_f32x2_add_entry<Src: MixedF32x2Source>(
    (packed, addend): (R<u32>, R<u64>),
    round: crate::scalar::F32RoundingMode,
) -> R<u64> {
    R::from_fn(|lane| {
        let (low, high) = mixed_f32x2_source::<Src>(packed[lane]);
        mixed_f32x2_result(
            crate::scalar::add_f32(low, crate::scalar::float2_x(addend[lane]), round),
            crate::scalar::add_f32(high, crate::scalar::float2_y(addend[lane]), round),
        )
    })
}

#[inline(never)]
fn mixed_f32x2_sub_entry<Src: MixedF32x2Source>(
    (packed, subtrahend): (R<u32>, R<u64>),
    round: crate::scalar::F32RoundingMode,
) -> R<u64> {
    R::from_fn(|lane| {
        let (low, high) = mixed_f32x2_source::<Src>(packed[lane]);
        mixed_f32x2_result(
            crate::scalar::sub_f32(low, crate::scalar::float2_x(subtrahend[lane]), round),
            crate::scalar::sub_f32(high, crate::scalar::float2_y(subtrahend[lane]), round),
        )
    })
}

mixed_f32x2_binary_variant!(add_spec, mixed_f32x2_add_entry);
mixed_f32x2_binary_variant!(sub_spec, mixed_f32x2_sub_entry);

register_variant! {
    [impl<Src, Round>] fma_spec, variant::MixedF32x2<Src, Round>
    where [
        Src: MixedF32x2Source,
        Round: FloatRound,
    ],
    (R<u32>, R<u64>, R<u64>) => R<u64>;
    |_context, _site, (packed, multiplier, addend)| {
        Ok(R::from_fn(|lane| {
            let (low, high) = mixed_f32x2_source::<Src>(packed[lane]);
            mixed_f32x2_result(
                crate::scalar::fma_f32(
                    low,
                    crate::scalar::float2_x(multiplier[lane]),
                    crate::scalar::float2_x(addend[lane]),
                    Round::MODE,
                ),
                crate::scalar::fma_f32(
                    high,
                    crate::scalar::float2_y(multiplier[lane]),
                    crate::scalar::float2_y(addend[lane]),
                    Round::MODE,
                ),
            )
        }))
    }
}

trait MixedF32x2Destination {
    const FTZ: bool;

    fn encode(value: f32) -> u16;
    fn flush_output(bits: u16) -> u16;
}

impl MixedF32x2Destination for variant::F16x2 {
    const FTZ: bool = true;

    fn encode(value: f32) -> u16 {
        crate::scalar::ptx_cvt_f32_to_f16(value, PtxFloatRounding::Zero, false, false)
    }

    fn flush_output(bits: u16) -> u16 {
        crate::scalar::flush_subnormal_f16_bits(bits)
    }
}

impl MixedF32x2Destination for variant::Bf16x2 {
    const FTZ: bool = false;

    fn encode(value: f32) -> u16 {
        crate::scalar::ptx_cvt_f32_to_bf16(value, PtxFloatRounding::Zero, false, false)
    }

    fn flush_output(bits: u16) -> u16 {
        bits
    }
}

fn mixed_f32x2_down_lane<Dst: MixedF32x2Destination>(
    lhs: f32,
    rhs: f32,
    operation: fn(f32, f32, crate::scalar::F32RoundingMode) -> f32,
) -> u16 {
    // PTX 9.4 requires `.ftz` on the f16x2 destination spelling.  SM107
    // hardware was unavailable while this model was added, so this follows
    // PTX's general `.ftz` contract: flush subnormal inputs and result, while
    // retaining the sign of zero.  A future SM107 oracle should lock this down.
    let lhs = if Dst::FTZ {
        crate::scalar::flush_subnormal_f32(lhs)
    } else {
        lhs
    };
    let rhs = if Dst::FTZ {
        crate::scalar::flush_subnormal_f32(rhs)
    } else {
        rhs
    };
    let value = operation(lhs, rhs, crate::scalar::F32RoundingMode::Zero);
    // The destination formats are subsets of f32.  Rounding the exact
    // operation toward zero to f32 and then toward zero again therefore
    // equals one direct destination-format RZ conversion; unlike RN, there is
    // no double-rounding ambiguity.
    let value = if Dst::FTZ {
        crate::scalar::flush_subnormal_f32(value)
    } else {
        value
    };
    Dst::flush_output(Dst::encode(value))
}

fn mixed_f32x2_down_entry<Dst: MixedF32x2Destination>(
    (lhs, rhs): (R<u64>, R<u64>),
    operation: fn(f32, f32, crate::scalar::F32RoundingMode) -> f32,
) -> R<u32> {
    lhs.zip_map(&rhs, |_lane, lhs, rhs| {
        u32::from(mixed_f32x2_down_lane::<Dst>(
            crate::scalar::float2_x(*lhs),
            crate::scalar::float2_x(*rhs),
            operation,
        )) | (u32::from(mixed_f32x2_down_lane::<Dst>(
            crate::scalar::float2_y(*lhs),
            crate::scalar::float2_y(*rhs),
            operation,
        )) << 16)
    })
}

macro_rules! mixed_f32x2_down_variant {
    ($spec:ident, $operation:path) => {
        impl<Dst> $spec::sealed::Sealed for variant::MixedF32x2Down<Dst> where
            Dst: MixedF32x2Destination
        {
        }

        impl<Dst> $spec::Variant for variant::MixedF32x2Down<Dst>
        where
            Dst: MixedF32x2Destination,
        {
            type Args = (R<u64>, R<u64>);
            type Output = R<u32>;
        }

        impl<Dst> $spec::sealed::Execute for variant::MixedF32x2Down<Dst>
        where
            Dst: MixedF32x2Destination,
        {
            fn execute(
                _context: ExecCtx,
                _site: SiteId,
                args: Self::Args,
            ) -> Result<Self::Output, EngineError> {
                Ok(mixed_f32x2_down_entry::<Dst>(args, $operation))
            }
        }
    };
}

mixed_f32x2_down_variant!(add_spec, crate::scalar::add_f32);
mixed_f32x2_down_variant!(sub_spec, crate::scalar::sub_f32);
mixed_f32x2_down_variant!(mul_spec, crate::scalar::mul_f32);

fn mixed_low_mul_word(
    lhs: u32,
    rhs: u32,
    convert_rhs: fn(u16) -> u16,
    multiply: fn(u16, u16) -> u16,
) -> u32 {
    u32::from(multiply(lhs as u16, convert_rhs(rhs as u16)))
        | (u32::from(multiply(
            (lhs >> 16) as u16,
            convert_rhs((rhs >> 16) as u16),
        )) << 16)
}

binary_variant!(
    mul_spec,
    variant::MixedLowMul<variant::Bf16x2, variant::F16x2>,
    u32,
    |lhs, rhs| mixed_low_mul_word(
        lhs,
        rhs,
        |bits| encode_bf16(decode_f16(bits)),
        crate::scalar::mul_bf16_bits_rn,
    )
);
binary_variant!(
    mul_spec,
    variant::MixedLowMul<variant::F16x2, variant::Bf16x2>,
    u32,
    |lhs, rhs| mixed_low_mul_word(
        lhs,
        rhs,
        |bits| encode_f16(decode_bf16(bits)),
        crate::scalar::mul_f16_bits_rn,
    )
);

macro_rules! integer_div_variant {
    ($marker:ty, $scalar:ty) => {
        register_variant! {
            [impl] div_spec, $marker,
            (R<$scalar>, R<$scalar>) => R<$scalar>;
            |context, site, (lhs, rhs)| {
                let mut result = R::splat(0 as $scalar);
                for lane in context.active_mask().iter() {
                    result[lane] = lhs[lane].checked_div(rhs[lane]).ok_or_else(|| {
                        EngineError::message(format!(
                            "integer div has an undefined operand at site {} lane {lane}: {} / {}",
                            site.get(),
                            lhs[lane],
                            rhs[lane],
                        ))
                    })?;
                }
                Ok(result)
            }
        }
    };
}

// PTX integer div is a distinct spelling of the existing `div` mnemonic.  The
// table-driven frontend exposes all six scalar integer types, including the
// 16-bit forms that core Nymph scalar expressions do not otherwise require.
integer_div_variant!(variant::I16, i16);
integer_div_variant!(variant::U16, u16);
integer_div_variant!(variant::I32, i32);
integer_div_variant!(variant::U32, u32);
integer_div_variant!(variant::I64, i64);
integer_div_variant!(variant::U64, u64);

macro_rules! integer_rem_variant {
    ($marker:ty, $scalar:ty) => {
        register_variant! {
            [impl] rem_spec, $marker,
            (R<$scalar>, R<$scalar>) => R<$scalar>;
            |context, site, (lhs, rhs)| {
                let mut result = R::splat(0 as $scalar);
                for lane in context.active_mask().iter() {
                    result[lane] = lhs[lane].checked_rem(rhs[lane]).ok_or_else(|| {
                        EngineError::message(format!(
                            "integer rem has an undefined operand at site {} lane {lane}: {} % {}",
                            site.get(),
                            lhs[lane],
                            rhs[lane],
                        ))
                    })?;
                }
                Ok(result)
            }
        }
    };
}

macro_rules! signed_integer_rem_variant {
    ($marker:ty, $scalar:ty) => {
        register_variant! {
            [impl] rem_spec, $marker,
            (R<$scalar>, R<$scalar>) => R<$scalar>;
            |context, site, (lhs, rhs)| {
                let mut result = R::splat(0 as $scalar);
                for lane in context.active_mask().iter() {
                    let remainder = lhs[lane].checked_rem(rhs[lane]).ok_or_else(|| {
                        EngineError::message(format!(
                            "integer rem has an undefined operand at site {} lane {lane}: {} % {}",
                            site.get(),
                            lhs[lane],
                            rhs[lane],
                        ))
                    })?;
                    // PTX 9.4 explicitly leaves negative remainder machine-specific:
                    // the quotient may round either toward zero or toward negative
                    // infinity.  Those rules differ exactly for a non-integral,
                    // negative quotient, so do not silently choose Rust's rule.
                    if remainder != 0 && (lhs[lane] < 0) != (rhs[lane] < 0) {
                        return Err(EngineError::message(format!(
                            "integer rem has a machine-specific negative operand at site {} lane {lane}: {} % {}",
                            site.get(),
                            lhs[lane],
                            rhs[lane],
                        )));
                    }
                    result[lane] = remainder;
                }
                Ok(result)
            }
        }
    };
}

signed_integer_rem_variant!(variant::I16, i16);
integer_rem_variant!(variant::U16, u16);
signed_integer_rem_variant!(variant::I32, i32);
integer_rem_variant!(variant::U32, u32);
signed_integer_rem_variant!(variant::I64, i64);
integer_rem_variant!(variant::U64, u64);
binary_variant!(max_spec, variant::F32, f32, crate::scalar::cuda_f32_max);
binary_variant!(min_spec, variant::F32, f32, crate::scalar::cuda_f32_min);
unary_variant!(rcp_spec, variant::F32RnFtz, f32, |value: f32| {
    crate::scalar::ptx_rcp_approx_ftz_f32(value)
});
unary_variant!(rcp_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_rcp_approx_f32(value)
});
unary_variant!(rcp_spec, variant::F32Rn, f32, |value: f32| {
    crate::scalar::div_f32_rn(1.0, value)
});
unary_variant!(sqrt_spec, variant::F32Rn, f32, f32::sqrt);
unary_variant!(sqrt_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_sqrt_f32(value, crate::scalar::F32RoundingMode::Nearest, false)
});
unary_variant!(sqrt_spec, variant::F32Ftz, f32, |value: f32| {
    crate::scalar::ptx_sqrt_f32(value, crate::scalar::F32RoundingMode::Nearest, true)
});
unary_variant!(sin_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_sin_approx_f32(value, false)
});
unary_variant!(sin_spec, variant::F32Ftz, f32, |value: f32| {
    crate::scalar::ptx_sin_approx_f32(value, true)
});
unary_variant!(cos_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_cos_approx_f32(value, false)
});
unary_variant!(cos_spec, variant::F32Ftz, f32, |value: f32| {
    crate::scalar::ptx_cos_approx_f32(value, true)
});

register_variant! {
    [impl<Round, Subnormal>] sqrt_spec, variant::Sqrt<variant::F32, Round, Subnormal>
    where [
        Round: FloatRound,
        Subnormal: SubnormalMode,
    ],
    R<f32> => R<f32>;
    |_context, _site, source| {
        Ok(source
            .map(|_lane, value| crate::scalar::ptx_sqrt_f32(value, Round::MODE, Subnormal::FTZ)))
    }
}

register_variant! {
    [impl<Round>] sqrt_spec, variant::Sqrt<variant::F64, Round, variant::PreserveSubnormal>
    where [
        Round: FloatRound,
    ],
    R<f64> => R<f64>;
    |_context, _site, source| {
        Ok(source.map(|_lane, value| crate::scalar::ptx_sqrt_f64(value, Round::MODE)))
    }
}
unary_variant!(exp2_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_exp2_approx_f32(value)
});
unary_variant!(exp2_spec, variant::F32RnFtz, f32, |value: f32| {
    crate::scalar::ptx_exp2_approx_ftz_f32(value)
});
unary_variant!(exp2_spec, variant::Bf16x2, u32, |value: u32| {
    crate::scalar::ptx_exp2_approx_ftz_bf16x2(value)
});
unary_variant!(exp2_spec, variant::Bf16, u16, |value: u16| {
    crate::scalar::ptx_exp2_approx_ftz_bf16(value)
});
unary_variant!(exp2_spec, variant::F16, u16, |value: u16| {
    crate::scalar::ptx_exp2_approx_f16(value)
});
unary_variant!(exp2_spec, variant::F16x2, u32, |value: u32| {
    crate::scalar::ptx_exp2_approx_f16x2(value)
});
unary_variant!(lg2_spec, variant::F32, f32, f32::log2);
unary_variant!(lg2_spec, variant::F32RnFtz, f32, |value: f32| {
    crate::scalar::ptx_lg2_approx_ftz_f32(value)
});
unary_variant!(rsqrt_spec, variant::F32RnFtz, f32, |value: f32| {
    crate::scalar::ptx_rsqrt_approx_ftz_f32(value)
});
unary_variant!(rsqrt_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_rsqrt_approx_f32(value)
});
unary_variant!(rsqrt_spec, variant::F64, f64, |value: f64| {
    1.0 / value.sqrt()
});
unary_variant!(
    rsqrt_spec,
    variant::F64Arithmetic<variant::Approx>,
    f64,
    crate::scalar::ptx_rsqrt_approx_ftz_f64
);
unary_variant!(tanh_spec, variant::F32, f32, |value: f32| {
    crate::scalar::ptx_tanh_approx_f32(value)
});
unary_variant!(tanh_spec, variant::F16, u16, |value: u16| {
    crate::scalar::ptx_tanh_approx_f16(value)
});
unary_variant!(tanh_spec, variant::F16x2, u32, |value: u32| {
    crate::scalar::ptx_tanh_approx_f16x2(value)
});
unary_variant!(tanh_spec, variant::Bf16, u16, |value: u16| {
    crate::scalar::ptx_tanh_approx_bf16(value)
});
unary_variant!(tanh_spec, variant::Bf16x2, u32, |value: u32| {
    crate::scalar::ptx_tanh_approx_bf16x2(value)
});
unary_variant!(neg_spec, variant::F32, f32, |value: f32| -value);
unary_variant!(neg_spec, variant::F32Ftz, f32, |value: f32| {
    crate::scalar::ptx_neg_ftz_f32(value)
});

fn decode_f16(bits: u16) -> f32 {
    crate::scalar::cuda_fp16_bits_to_f32(bits)
}

fn encode_f16(value: f32) -> u16 {
    crate::scalar::cuda_f32_to_fp16_bits(value)
}

fn decode_bf16(bits: u16) -> f32 {
    crate::bf16_bits_to_f32(bits)
}

fn encode_bf16(value: f32) -> u16 {
    crate::f32_to_bf16_bits(crate::scalar::cuda_canonicalize_nan_f32(value))
}

macro_rules! low_precision_binary_variant {
    ($spec:ident, $variant:ty, $decode:ident, $encode:ident, $operation:expr) => {
        binary_variant!($spec, $variant, u16, |lhs: u16, rhs: u16| {
            $encode(($operation)($decode(lhs), $decode(rhs)))
        });
    };
}

// Legacy packed markers remain used by ordinary TIR expressions. Their
// arithmetic delegates to the same scalar operations as HalfArithmetic.
binary_variant!(
    sub_spec,
    variant::F16x2,
    u32,
    crate::scalar::sub_f16x2_bits_rn
);
binary_variant!(
    add_spec,
    variant::Bf16x2,
    u32,
    crate::scalar::add_bf16x2_bits_rn
);
binary_variant!(
    sub_spec,
    variant::Bf16x2,
    u32,
    crate::scalar::sub_bf16x2_bits_rn
);

trait HalfRegister: RegisterType {
    const FORMAT: crate::scalar::LowPrecisionFormat;
    fn map<const N: usize>(
        args: [Self::Scalar; N],
        operation: impl Fn([u16; N]) -> u16,
    ) -> Self::Scalar;
}

macro_rules! half_register {
    ($marker:ident, $format:ident, $components:expr) => {
        impl HalfRegister for variant::$marker {
            const FORMAT: crate::scalar::LowPrecisionFormat =
                crate::scalar::LowPrecisionFormat::$format;
            fn map<const N: usize>(
                args: [Self::Scalar; N],
                operation: impl Fn([u16; N]) -> u16,
            ) -> Self::Scalar {
                let mut result = 0;
                for component in 0..$components {
                    result |= (operation(args.map(|x| (x >> (component * 16)) as u16))
                        as Self::Scalar)
                        << (component * 16);
                }
                result
            }
        }
    };
}
half_register!(F16, F16, 1);
half_register!(F16x2, F16, 2);
half_register!(Bf16, Bf16, 1);
half_register!(Bf16x2, Bf16, 2);

macro_rules! half_minmax_variant {
    ($spec:ident, $maximum:literal) => {
        register_variant! {
            [impl<T: HalfRegister, S: SubnormalMode, Nan: F32Nan, const XOR: bool>] $spec, variant::HalfMinMax<T, S, Nan, XOR>,
            Pair<R<T::Scalar>> => R<T::Scalar>;
            |_context, _site, (a, b)| {
                Ok(R::from_fn(|lane| {
                    T::map([a[lane].clone(), b[lane].clone()], |[a, b]| {
                        crate::scalar::low_minmax(
                            a,
                            b,
                            T::FORMAT,
                            S::FTZ,
                            Nan::PROPAGATE,
                            XOR,
                            $maximum,
                        )
                    })
                }))
            }
        }
    };
}
half_minmax_variant!(min_spec, false);
half_minmax_variant!(max_spec, true);

trait HalfClamp {
    fn apply(bits: u16, format: crate::scalar::LowPrecisionFormat) -> u16;
}
impl HalfClamp for variant::NoSat {
    fn apply(bits: u16, _format: crate::scalar::LowPrecisionFormat) -> u16 {
        bits
    }
}
impl HalfClamp for variant::Sat {
    fn apply(bits: u16, format: crate::scalar::LowPrecisionFormat) -> u16 {
        if bits & 0x7fff > format.infinity() || bits & 0x8000 != 0 {
            0
        } else {
            bits.min(0x3c00)
        }
    }
}
impl HalfClamp for variant::Relu {
    fn apply(bits: u16, format: crate::scalar::LowPrecisionFormat) -> u16 {
        if bits & 0x7fff > format.infinity() {
            0x7fff
        } else if bits & 0x8000 != 0 {
            0
        } else {
            bits
        }
    }
}

macro_rules! half_arithmetic_variant {
    ($spec:ident, $arity:ident, ($($arg:ident),+), $operation:expr) => {
        register_variant! {
            [impl<T: HalfRegister, S: SubnormalMode, C: HalfClamp, const OOB: bool>] $spec, variant::HalfArithmetic<T, S, C, OOB>,
            $arity<R<T::Scalar>> => R<T::Scalar>;
            |_context, _site, ($($arg),+)| {
                Ok(R::from_fn(|lane| T::map([$($arg[lane].clone()),+], |[$($arg),+]| {
                    C::apply(($operation)($($arg),+, T::FORMAT, S::FTZ, OOB), T::FORMAT)
                })))
            }
        }
    };
}
half_arithmetic_variant!(add_spec, Pair, (a, b), |a, b, format, ftz, _oob| {
    crate::scalar::low_add_rn(a, b, format, false, ftz)
});
half_arithmetic_variant!(sub_spec, Pair, (a, b), |a, b, format, ftz, _oob| {
    crate::scalar::low_add_rn(a, b, format, true, ftz)
});
half_arithmetic_variant!(mul_spec, Pair, (a, b), |a, b, format, ftz, _oob| {
    crate::scalar::low_mul_rn(a, b, format, ftz)
});
half_arithmetic_variant!(fma_spec, Triple, (a, b, c), |a, b, c, format, ftz, oob| {
    // Only the two multiplicands are OOB-sensitive; a NaN addend remains NaN.
    if oob
        && [a, b]
            .iter()
            .any(|x| x & 0x7fff == crate::scalar::PTX_OOB_NAN)
    {
        0
    } else {
        crate::scalar::low_fma_rn(a, b, c, format, ftz)
    }
});

low_precision_binary_variant!(
    max_spec,
    variant::F16,
    decode_f16,
    encode_f16,
    crate::scalar::cuda_f32_max
);
low_precision_binary_variant!(
    min_spec,
    variant::F16,
    decode_f16,
    encode_f16,
    crate::scalar::cuda_f32_min
);
low_precision_binary_variant!(
    max_spec,
    variant::Bf16,
    decode_bf16,
    encode_bf16,
    crate::scalar::cuda_f32_max
);
low_precision_binary_variant!(
    min_spec,
    variant::Bf16,
    decode_bf16,
    encode_bf16,
    crate::scalar::cuda_f32_min
);
binary_variant!(max_spec, variant::Bf16x2, u32, crate::scalar::hmax2_bf16);
binary_variant!(min_spec, variant::Bf16x2, u32, crate::scalar::hmin2_bf16);
binary_variant!(max_spec, variant::F16x2, u32, crate::scalar::hmax2_f16);
binary_variant!(min_spec, variant::F16x2, u32, crate::scalar::hmin2_f16);
unary_variant!(neg_spec, variant::F16, u16, |value: u16| { value ^ 0x8000 });
unary_variant!(neg_spec, variant::F16Ftz, u16, |value: u16| {
    crate::scalar::ptx_neg_f16_bits(value, true)
});
unary_variant!(neg_spec, variant::F16x2, u32, |value: u32| {
    value ^ 0x8000_8000
});
unary_variant!(neg_spec, variant::F16x2Ftz, u32, |value: u32| {
    crate::scalar::ptx_neg_f16x2_bits(value, true)
});
unary_variant!(neg_spec, variant::Bf16, u16, |value: u16| {
    value ^ 0x8000
});
unary_variant!(neg_spec, variant::Bf16x2, u32, |value: u32| {
    value ^ 0x8000_8000
});

macro_rules! f64_arithmetic_variant {
    ($spec:ident, $arity:ident, ($($arg:ident),+), $operation:path) => {
        register_variant! {
            [impl<Round: FloatRound>] $spec, variant::F64Arithmetic<Round>,
            $arity<R<f64>> => R<f64>;
            |_context, _site, ($($arg),+)| {
                Ok(R::from_fn(|lane| $operation($($arg[lane]),+, Round::MODE)))
            }
        }
    };
}

f64_arithmetic_variant!(add_spec, Pair, (lhs, rhs), crate::scalar::add_f64);
f64_arithmetic_variant!(sub_spec, Pair, (lhs, rhs), crate::scalar::sub_f64);
f64_arithmetic_variant!(mul_spec, Pair, (lhs, rhs), crate::scalar::mul_f64);
f64_arithmetic_variant!(mad_spec, Triple, (lhs, rhs, addend), crate::scalar::fma_f64);
f64_arithmetic_variant!(fma_spec, Triple, (lhs, rhs, addend), crate::scalar::fma_f64);
f64_arithmetic_variant!(div_spec, Pair, (lhs, rhs), crate::scalar::div_f64);
binary_variant!(max_spec, variant::F64, f64, crate::scalar::cuda_f64_max);
binary_variant!(min_spec, variant::F64, f64, crate::scalar::cuda_f64_min);
unary_variant!(sqrt_spec, variant::F64Rn, f64, f64::sqrt);
unary_variant!(lg2_spec, variant::F64, f64, f64::log2);
unary_variant!(neg_spec, variant::F64, f64, |value: f64| -value);
unary_variant!(abs_spec, variant::F32, f32, f32::abs);
unary_variant!(abs_spec, variant::F32Ftz, f32, |value: f32| {
    crate::scalar::flush_subnormal_f32(value).abs()
});
unary_variant!(abs_spec, variant::F64, f64, |value: f64| {
    // PTX abs.f64 passes NaNs through unchanged, unlike its f32/half forms.
    if value.is_nan() {
        value
    } else {
        value.abs()
    }
});
unary_variant!(abs_spec, variant::F16, u16, |value: u16| value & 0x7fff);
unary_variant!(abs_spec, variant::F16Ftz, u16, |value: u16| {
    crate::scalar::flush_subnormal_f16_bits(value) & 0x7fff
});
unary_variant!(abs_spec, variant::F16x2Ftz, u32, |value: u32| {
    u32::from(crate::scalar::flush_subnormal_f16_bits(value as u16) & 0x7fff)
        | (u32::from(crate::scalar::flush_subnormal_f16_bits((value >> 16) as u16) & 0x7fff) << 16)
});
unary_variant!(abs_spec, variant::Bf16, u16, |value: u16| value & 0x7fff);
unary_variant!(abs_spec, variant::Bf16x2, u32, |value: u32| value
    & 0x7fff_7fff);
unary_variant!(abs_spec, variant::F16x2, u32, |value: u32| {
    value & 0x7fff_7fff
});

macro_rules! move_variant {
    ($marker:ty, $scalar:ty) => {
        register_variant! {
            [impl] mov_spec, $marker,
            R<$scalar> => R<$scalar>;
            |_context, _site, value| {
                Ok(value)
            }
        }
    };
}

move_variant!(variant::I8, i8);
move_variant!(variant::I16, i16);
move_variant!(variant::I32, i32);
move_variant!(variant::I64, i64);
move_variant!(variant::U8, u8);
move_variant!(variant::U16, u16);
move_variant!(variant::U32, u32);
move_variant!(variant::U64, u64);
move_variant!(variant::F16, u16);
move_variant!(variant::Bf16, u16);
move_variant!(variant::F32, f32);
move_variant!(variant::F64, f64);
move_variant!(variant::Pred, bool);

mod register_type_sealed {
    pub trait Sealed {}
}

/// Compile-time PTX register dtype carried by a specialization marker.
#[allow(private_bounds)]
pub trait RegisterType: register_type_sealed::Sealed {
    type Scalar: Clone + Send + Sync + 'static;
}

macro_rules! register_types {
    ($($marker:ty => $scalar:ty),+ $(,)?) => {
        $(
            impl register_type_sealed::Sealed for $marker {}

            impl RegisterType for $marker {
                type Scalar = $scalar;
            }
        )+
    };
}

register_types!(
    variant::I16x2 => u32,
    variant::U16x2 => u32,
    variant::I8 => i8,
    variant::I16 => i16,
    variant::I32 => i32,
    variant::I64 => i64,
    variant::U8 => u8,
    variant::U16 => u16,
    variant::U32 => u32,
    variant::U64 => u64,
    variant::B16 => u16,
    variant::B32 => u32,
    variant::B64 => u64,
    variant::F16 => u16,
    variant::F16x2 => u32,
    variant::Bf16 => u16,
    variant::Bf16x2 => u32,
    variant::F32 => f32,
    variant::F64 => f64,
);

register_variant! {
    [impl<T, Compare, Subnormal>] setp_spec, variant::Setp<T, Compare, Subnormal>
    where [
        T: reg_compare::ScalarCompareFormat,
        <T as RegisterType>::Scalar: Copy,
        Compare: reg_compare::ValidComparison<T>,
        Subnormal: reg_compare::SubnormalMode<T>,
    ],
    (
        R<<T as RegisterType>::Scalar>,
        R<<T as RegisterType>::Scalar>,
    ) => R<bool>;
    |_context, _site, (lhs, rhs)| {
        Ok(
            reg_compare::compare_masks::<T, Compare, Subnormal>(&lhs, &rhs)
                .map(|_lane, mask| mask & 1 != 0),
        )
    }
}

trait PackedSetMarker: reg_compare::ScalarCompareFormat
where
    Self::Scalar: Copy,
{
    const WIDTH: u32;
    fn decode_lane(value: u32) -> Self::Scalar;
}

#[inline(always)]
fn signed_packed_lane(value: u32, width: u32) -> i32 {
    let shift = 32 - width;
    ((value << shift) as i32) >> shift
}

#[inline(always)]
fn compare_packed_words<T, Compare>(lhs: u32, rhs: u32) -> u32
where
    T: PackedSetMarker,
    T::Scalar: Copy,
    Compare: reg_compare::ValidComparison<T>,
{
    debug_assert!(matches!(T::WIDTH, 8 | 16));
    let lane_mask = (1_u32 << T::WIDTH) - 1;
    let mut packed = 0_u32;
    for lane in 0..(32 / T::WIDTH) {
        let shift = lane * T::WIDTH;
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        if reg_compare::compare_atom(
            <Compare as reg_compare::ValidComparison<T>>::KIND,
            <T as reg_compare::ScalarCompareFormat>::atom(T::decode_lane(lhs_lane), false),
            <T as reg_compare::ScalarCompareFormat>::atom(T::decode_lane(rhs_lane), false),
        ) {
            packed |= lane_mask << shift;
        }
    }
    packed
}

macro_rules! packed_set_marker {
    ($marker:ty, $scalar:ty, $width:expr, $decode:expr) => {
        impl PackedSetMarker for $marker {
            const WIDTH: u32 = $width;

            fn decode_lane(value: u32) -> $scalar {
                ($decode)(value)
            }
        }
    };
}

packed_set_marker!(variant::U8, u8, 8, |value| value as u8);
packed_set_marker!(variant::I8, i8, 8, |value| signed_packed_lane(value, 8)
    as i8);
packed_set_marker!(variant::U16, u16, 16, |value| value as u16);
packed_set_marker!(variant::I16, i16, 16, |value| signed_packed_lane(value, 16)
    as i16);

register_variant! {
    [impl<T, Compare>] set_spec, variant::SetPacked<T, Compare>
    where [
        T: PackedSetMarker,
        T::Scalar: Copy,
        Compare: reg_compare::ValidComparison<T>,
    ],
    (R<u32>, R<u32>) => R<u32>;
    |_context, _site, (lhs, rhs)| {
        Ok(lhs.zip_map(&rhs, |_lane, lhs, rhs| {
            compare_packed_words::<T, Compare>(*lhs, *rhs)
        }))
    }
}

trait SelectMarker: RegisterType {
    fn execute_select(
        predicate: R<bool>,
        on_true: R<Self::Scalar>,
        on_false: R<Self::Scalar>,
    ) -> R<Self::Scalar>;
}

macro_rules! select_markers {
    ($($marker:ty => $scalar:ty),+ $(,)?) => {
        $(
            impl SelectMarker for $marker {
                #[inline(never)]
                fn execute_select(
                    predicate: R<bool>,
                    on_true: R<$scalar>,
                    on_false: R<$scalar>,
                ) -> R<$scalar> {
                    R::from_fn(|lane| {
                        if predicate[lane] {
                            on_true[lane].clone()
                        } else {
                            on_false[lane].clone()
                        }
                    })
                }
            }
        )+
    };
}

select_markers!(
    variant::I8 => i8,
    variant::I16 => i16,
    variant::I32 => i32,
    variant::I64 => i64,
    variant::U8 => u8,
    variant::U16 => u16,
    variant::U32 => u32,
    variant::U64 => u64,
    variant::B16 => u16,
    variant::B32 => u32,
    variant::B64 => u64,
    variant::F16 => u16,
    variant::Bf16 => u16,
    variant::F32 => f32,
    variant::F64 => f64,
);

register_variant! {
    [impl<T: SelectMarker>] selp_spec, T,
    (
        R<bool>,
        R<<T as RegisterType>::Scalar>,
        R<<T as RegisterType>::Scalar>,
    ) => R<<T as RegisterType>::Scalar>;
    |_context, _site, (predicate, on_true, on_false)| {
        Ok(T::execute_select(predicate, on_true, on_false))
    }
}

#[path = "reg_compare.rs"]
mod reg_compare;

macro_rules! cvt_variant {
    ($source_marker:ty => $destination_marker:ty, $source:ty => $destination:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<$source_marker, $destination_marker, variant::Unmodified>,
            R<$source> => R<$destination>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| value as $destination))
            }
        }
    };
}

macro_rules! exact_cvt_variant {
    ($source_marker:ty => $destination_marker:ty, $mode:ty, $source:ty => $destination:ty, $convert:expr) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<$source_marker, $destination_marker, $mode>,
            R<$source> => R<$destination>;
            |_context, _site, source| {
                let convert = $convert;
                Ok(source.map(|_lane, value| convert(value)))
            }
        }
    };
}

macro_rules! integer_cvt_variant {
    ($source_marker:ty => $destination_marker:ty, $source:ty => $destination:ty) => {
        cvt_variant!($source_marker => $destination_marker, $source => $destination);
        exact_cvt_variant!(
            $source_marker => $destination_marker,
            variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
            $source => $destination,
            // Every signed/unsigned PTX integer fits in i128. Clamp before
            // narrowing so negative sources and u64::MAX retain their value.
            |value| (value as i128).clamp(<$destination>::MIN as i128, <$destination>::MAX as i128)
                as $destination
        );
    };
}

macro_rules! integer_cvt_destinations {
    ($source_marker:ty, $source:ty) => {
        integer_cvt_variant!($source_marker => variant::I8, $source => i8);
        integer_cvt_variant!($source_marker => variant::I16, $source => i16);
        integer_cvt_variant!($source_marker => variant::I32, $source => i32);
        integer_cvt_variant!($source_marker => variant::I64, $source => i64);
        integer_cvt_variant!($source_marker => variant::U8, $source => u8);
        integer_cvt_variant!($source_marker => variant::U16, $source => u16);
        integer_cvt_variant!($source_marker => variant::U32, $source => u32);
        integer_cvt_variant!($source_marker => variant::U64, $source => u64);
        exact_cvt_variant!(
            $source_marker => variant::F32,
            variant::Rn,
            $source => f32,
            |value| value as f32
        );
        exact_cvt_variant!(
            $source_marker => variant::F64,
            variant::Rn,
            $source => f64,
            |value| value as f64
        );
    };
}

integer_cvt_destinations!(variant::I8, i8);
integer_cvt_destinations!(variant::I16, i16);
integer_cvt_destinations!(variant::I32, i32);
integer_cvt_destinations!(variant::I64, i64);
integer_cvt_destinations!(variant::U8, u8);
integer_cvt_destinations!(variant::U16, u16);
integer_cvt_destinations!(variant::U32, u32);
integer_cvt_destinations!(variant::U64, u64);
cvt_variant!(variant::F32 => variant::F32, f32 => f32);
cvt_variant!(variant::F32 => variant::F64, f32 => f64);
exact_cvt_variant!(
    variant::F64 => variant::F32,
    variant::Rn,
    f64 => f32,
    |value| value as f32
);
cvt_variant!(variant::F64 => variant::F64, f64 => f64);
// A same-type half conversion still canonicalizes NaNs. Reuse the scalar
// codecs: every finite half value widens exactly and rounds back unchanged.
exact_cvt_variant!(
    variant::F16 => variant::F16,
    variant::Unmodified,
    u16 => u16,
    |value| crate::scalar::ptx_cvt_f32_to_f16(
        decode_f16(value), crate::scalar::PtxFloatRounding::NearestEven, false, false
    )
);
exact_cvt_variant!(
    variant::Bf16 => variant::Bf16,
    variant::Unmodified,
    u16 => u16,
    |value| crate::scalar::ptx_cvt_f32_to_bf16(
        decode_bf16(value), crate::scalar::PtxFloatRounding::NearestEven, false, false
    )
);
exact_cvt_variant!(
    variant::F16 => variant::F32,
    variant::Unmodified,
    u16 => f32,
    decode_f16
);
exact_cvt_variant!(
    variant::Bf16 => variant::F32,
    variant::Unmodified,
    u16 => f32,
    decode_bf16
);
exact_cvt_variant!(
    variant::F16 => variant::F64,
    variant::Unmodified,
    u16 => f64,
    |value| crate::scalar::ptx_cvt_low_to_f64(value, crate::scalar::LowPrecisionFormat::F16)
);
exact_cvt_variant!(
    variant::Bf16 => variant::F64,
    variant::Unmodified,
    u16 => f64,
    |value| crate::scalar::ptx_cvt_low_to_f64(value, crate::scalar::LowPrecisionFormat::Bf16)
);
exact_cvt_variant!(
    variant::F32 => variant::F16,
    variant::Rn,
    f32 => u16,
    encode_f16
);

register_variant! {
    [impl] cvt_spec, variant::Cvt<variant::F32, variant::F16x2, variant::Rn>,
    (R<f32>, R<f32>) => R<u32>;
    |_context, _site, (high, low)| {
        Ok(high.zip_map(&low, |_lane, high, low| {
            (u32::from(encode_f16(*high)) << 16) | u32::from(encode_f16(*low))
        }))
    }
}

register_variant! {
    [impl] cvt_spec, variant::Cvt<variant::F32, variant::Bf16x2, variant::Rn>,
    (R<f32>, R<f32>) => R<u32>;
    |_context, _site, (high, low)| {
        Ok(high.zip_map(&low, |_lane, high, low| {
            (u32::from(encode_bf16(*high)) << 16) | u32::from(encode_bf16(*low))
        }))
    }
}

exact_cvt_variant!(
    variant::F32 => variant::Bf16,
    variant::Rn,
    f32 => u16,
    encode_bf16
);
// The Nymph floor entry point.  A `cvt.rmi.f32.f32` written in TIRx source
// resolves to `CvtMode<Rmi>` below instead, which also canonicalizes NaN the
// way the measured instruction does; this two-parameter marker keeps the plain
// `f32::floor` a Nymph frontend already names.
exact_cvt_variant!(
    variant::F32 => variant::F32,
    variant::Rmi,
    f32 => f32,
    f32::floor
);

// --- packed narrow-float cvt ----------------------------------------------
//
// Packing, widening, rounding, saturation, ReLU and scaling reuse the numeric
// bodies in `crate::scalar`. These specializations bind format/modifier markers
// and adapt operands; the frontend's target table owns instruction legality.

/// Element format named by one packed narrow-float marker.
trait PackedNarrowFormat {
    const FORMAT: crate::NarrowFloatFormat;
}

impl PackedNarrowFormat for variant::E4m3x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT8_E4M3;
}

impl PackedNarrowFormat for variant::E5m2x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT8_E5M2;
}

impl PackedNarrowFormat for variant::E2m1x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT4_E2M1;
}

impl PackedNarrowFormat for variant::E2m3x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT6_E2M3;
}

impl PackedNarrowFormat for variant::E3m2x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT6_E3M2;
}

impl PackedNarrowFormat for variant::Ue5m3x2 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT8_UE5M3;
}

impl PackedNarrowFormat for variant::E4m3x4 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT8_E4M3;
}

impl PackedNarrowFormat for variant::E5m2x4 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT8_E5M2;
}

impl PackedNarrowFormat for variant::E2m1x4 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT4_E2M1;
}

impl PackedNarrowFormat for variant::E2m3x4 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT6_E2M3;
}

impl PackedNarrowFormat for variant::E3m2x4 {
    const FORMAT: crate::NarrowFloatFormat = crate::FLOAT6_E3M2;
}

/// `.rz` versus `.rp` on a `ue8m0x2` destination, as a marker axis.
trait RoundExponentUp {
    const ROUNDS_UP: bool;
}

impl RoundExponentUp for variant::Rz {
    const ROUNDS_UP: bool = false;
}

impl RoundExponentUp for variant::Rp {
    const ROUNDS_UP: bool = true;
}

/// `.satfinite` presence, as a marker axis rather than a runtime flag.
trait SaturateFinite {
    const SATURATES: bool;
}

impl SaturateFinite for variant::NoSatFinite {
    const SATURATES: bool = false;
}

impl SaturateFinite for variant::SatFinite {
    const SATURATES: bool = true;
}

/// `.relu` presence, as a marker axis rather than a runtime flag.
trait ClampNegative {
    const CLAMPS: bool;
}

impl ClampNegative for variant::NoRelu {
    const CLAMPS: bool = false;
}

impl ClampNegative for variant::Relu {
    const CLAMPS: bool = true;
}

/// Post-conversion handling of negative-zero fields in a packed narrow result.
trait CvtNarrowZero {
    fn apply(bits: u16, format: crate::NarrowFloatFormat) -> u16;
    fn apply_half(bits: u16) -> u16;
}

impl CvtNarrowZero for variant::PreserveZero {
    fn apply(bits: u16, _format: crate::NarrowFloatFormat) -> u16 {
        bits
    }
    fn apply_half(bits: u16) -> u16 {
        bits
    }
}

impl CvtNarrowZero for variant::Pzo {
    fn apply(bits: u16, format: crate::NarrowFloatFormat) -> u16 {
        crate::scalar::ptx_cvt_pzo_narrow_x2(bits, format)
    }
    fn apply_half(bits: u16) -> u16 {
        crate::scalar::ptx_cvt_pzo_u16(bits)
    }
}

/// The two floating rounding modes admitted by PTX's `.frnd2` grammar.
trait CvtFrnd2 {
    const MODE: PtxFloatRounding;
}

impl CvtFrnd2 for variant::Rn {
    const MODE: PtxFloatRounding = PtxFloatRounding::NearestEven;
}

impl CvtFrnd2 for variant::Rz {
    const MODE: PtxFloatRounding = PtxFloatRounding::Zero;
}

/// Numerical conversion and output shape, independent of the zero policy.
trait F32CvtDestination<Zero> {
    type Args;
    type Output;

    fn convert(
        args: Self::Args,
        rounding: PtxFloatRounding,
        relu: bool,
        satfinite: bool,
    ) -> Self::Output;
}

macro_rules! scalar_pzo_destination {
    ($marker:ty, $output:ty, $convert:path, $pzo:path) => {
        impl F32CvtDestination<variant::Pzo> for $marker {
            type Args = R<f32>;
            type Output = R<$output>;

            fn convert(
                source: Self::Args,
                rounding: PtxFloatRounding,
                relu: bool,
                satfinite: bool,
            ) -> Self::Output {
                source.map(|_lane, value| $pzo($convert(value, rounding, relu, satfinite)))
            }
        }
    };
}

scalar_pzo_destination!(
    variant::F16,
    u16,
    crate::scalar::ptx_cvt_f32_to_f16,
    crate::scalar::ptx_cvt_pzo_u16
);
scalar_pzo_destination!(
    variant::Bf16,
    u16,
    crate::scalar::ptx_cvt_f32_to_bf16,
    crate::scalar::ptx_cvt_pzo_u16
);
scalar_pzo_destination!(
    variant::Tf32,
    u32,
    crate::scalar::ptx_cvt_f32_to_tf32,
    crate::scalar::ptx_cvt_pzo_u32
);

macro_rules! packed_half_destination {
    ($marker:ty, $convert:path) => {
        impl<Zero: CvtNarrowZero> F32CvtDestination<Zero> for $marker {
            type Args = (R<f32>, R<f32>);
            type Output = R<u32>;

            fn convert(
                (high, low): Self::Args,
                rounding: PtxFloatRounding,
                relu: bool,
                satfinite: bool,
            ) -> Self::Output {
                high.zip_map(&low, |_lane, high, low| {
                    let convert =
                        |value| Zero::apply_half($convert(value, rounding, relu, satfinite));
                    (u32::from(convert(*high)) << 16) | u32::from(convert(*low))
                })
            }
        }
    };
}

/// Rounding direction of a PTX 9.4 packed narrow-float conversion.
trait PackedNarrowRounding {
    const ROUNDING: PtxFloatRounding;
}

impl PackedNarrowRounding for variant::Rn {
    const ROUNDING: PtxFloatRounding = PtxFloatRounding::NearestEven;
}

impl PackedNarrowRounding for variant::Rz {
    const ROUNDING: PtxFloatRounding = PtxFloatRounding::Zero;
}

impl PackedNarrowRounding for variant::Rp {
    const ROUNDING: PtxFloatRounding = PtxFloatRounding::PositiveInfinity;
}

/// PTX 9.4 directed or newly introduced narrow-format conversion from f32.
macro_rules! packed_narrow_rounded_from_f32_variant {
    ($narrow:ty, $carrier:ty, $round:ty, $relu:ty) => {
        packed_narrow_rounded_from_f32_variant!(
            $narrow,
            $carrier,
            $round,
            $relu,
            variant::PreserveZero
        );
    };
    ($narrow:ty, $carrier:ty, $round:ty, $relu:ty, $zero:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::F32,
                $narrow,
                variant::PackedMode<$round, variant::SatFinite, $relu, variant::NoScale, $zero>,
            >,
            (R<f32>, R<f32>) => R<$carrier>;
            |_context, _site, (high, low)| {
                Ok(high.zip_map(&low, |_lane, high, low| {
                    let bits = crate::scalar::ptx_cvt_pack_narrow_x2_rounded::<
                        { <$relu as ClampNegative>::CLAMPS },
                    >(
                        *high,
                        *low,
                        <$round as PackedNarrowRounding>::ROUNDING,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    );
                    <$zero as CvtNarrowZero>::apply(bits, <$narrow as PackedNarrowFormat>::FORMAT)
                        as $carrier
                }))
            }
        }
    };
}

packed_half_destination!(variant::F16x2, crate::scalar::ptx_cvt_f32_to_f16);
packed_half_destination!(variant::Bf16x2, crate::scalar::ptx_cvt_f32_to_bf16);

impl<Dst, Round, Saturate, Activation, Zero> cvt_spec::sealed::Sealed
    for variant::Cvt<
        variant::F32,
        Dst,
        variant::PackedMode<Round, Saturate, Activation, variant::NoScale, Zero>,
    >
where
    Dst: F32CvtDestination<Zero>,
    Round: CvtFrnd2,
    Saturate: SaturateFinite,
    Activation: ClampNegative,
{
}

impl<Dst, Round, Saturate, Activation, Zero, Args, Output> cvt_spec::Variant
    for variant::Cvt<
        variant::F32,
        Dst,
        variant::PackedMode<Round, Saturate, Activation, variant::NoScale, Zero>,
    >
where
    Dst: F32CvtDestination<Zero, Args = Args, Output = Output>,
    Round: CvtFrnd2,
    Saturate: SaturateFinite,
    Activation: ClampNegative,
{
    type Args = Args;
    type Output = Output;
}

impl<Dst, Round, Saturate, Activation, Zero, Args, Output> cvt_spec::sealed::Execute
    for variant::Cvt<
        variant::F32,
        Dst,
        variant::PackedMode<Round, Saturate, Activation, variant::NoScale, Zero>,
    >
where
    Dst: F32CvtDestination<Zero, Args = Args, Output = Output>,
    Round: CvtFrnd2,
    Saturate: SaturateFinite,
    Activation: ClampNegative,
{
    fn execute(
        _context: ExecCtx,
        _site: SiteId,
        args: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Ok(Dst::convert(
            args,
            Round::MODE,
            Activation::CLAMPS,
            Saturate::SATURATES,
        ))
    }
}

/// PTX 9.4 directed or newly introduced narrow-format conversion from a pair.
macro_rules! packed_narrow_rounded_from_packed_variant {
    ($wide:ty, $decode:ident, $narrow:ty, $carrier:ty, $round:ty, $relu:ty) => {
        packed_narrow_rounded_from_packed_variant!(
            $wide,
            $decode,
            $narrow,
            $carrier,
            $round,
            $relu,
            variant::PreserveZero
        );
    };
    ($wide:ty, $decode:ident, $narrow:ty, $carrier:ty, $round:ty, $relu:ty, $zero:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $wide,
                $narrow,
                variant::PackedMode<$round, variant::SatFinite, $relu, variant::NoScale, $zero>,
            >,
            R<u32> => R<$carrier>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| {
                    let bits = crate::scalar::ptx_cvt_pack_narrow_x2_rounded::<
                        { <$relu as ClampNegative>::CLAMPS },
                    >(
                        $decode((value >> 16) as u16),
                        $decode(value as u16),
                        <$round as PackedNarrowRounding>::ROUNDING,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    );
                    <$zero as CvtNarrowZero>::apply(bits, <$narrow as PackedNarrowFormat>::FORMAT)
                        as $carrier
                }))
            }
        }
    };
}

/// PTX 9.4 n1-scaled conversion from two binary32 primaries.
macro_rules! packed_narrow_scaled_n1_from_f32_variant {
    ($narrow:ty, $carrier:ty, $round:ty, $relu:ty) => {
        packed_narrow_scaled_n1_from_f32_variant!(
            $narrow,
            $carrier,
            $round,
            $relu,
            variant::PreserveZero
        );
    };
    ($narrow:ty, $carrier:ty, $round:ty, $relu:ty, $zero:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::F32,
                $narrow,
                variant::PackedMode<
                    $round,
                    variant::SatFinite,
                    $relu,
                    variant::ScaledUe8m0N1,
                    $zero,
                >,
            >,
            (R<f32>, R<f32>, R<u8>) => R<$carrier>;
            |_context, _site, (high, low, scale)| {
                Ok(R::from_fn(|lane| {
                    let bits = crate::scalar::ptx_cvt_pack_narrow_x2_rounded::<
                        { <$relu as ClampNegative>::CLAMPS },
                    >(
                        crate::scalar::ptx_cvt_scaled_n1_f32(high[lane], scale[lane]),
                        crate::scalar::ptx_cvt_scaled_n1_f32(low[lane], scale[lane]),
                        <$round as PackedNarrowRounding>::ROUNDING,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    );
                    <$zero as CvtNarrowZero>::apply(bits, <$narrow as PackedNarrowFormat>::FORMAT)
                        as $carrier
                }))
            }
        }
    };
}

/// PTX 9.4 n1-scaled conversion from one binary16 or bfloat16 pair.
macro_rules! packed_narrow_scaled_n1_from_packed_variant {
    ($wide:ty, $scale:ident, $narrow:ty, $carrier:ty, $round:ty, $relu:ty) => {
        packed_narrow_scaled_n1_from_packed_variant!(
            $wide,
            $scale,
            $narrow,
            $carrier,
            $round,
            $relu,
            variant::PreserveZero
        );
    };
    ($wide:ty, $scale:ident, $narrow:ty, $carrier:ty, $round:ty, $relu:ty, $zero:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $wide,
                $narrow,
                variant::PackedMode<
                    $round,
                    variant::SatFinite,
                    $relu,
                    variant::ScaledUe8m0N1,
                    $zero,
                >,
            >,
            (R<u32>, R<u8>) => R<$carrier>;
            |_context, _site, (source, scale)| {
                Ok(source.zip_map(&scale, |_lane, value, scale| {
                    let bits = crate::scalar::ptx_cvt_pack_narrow_x2_rounded::<
                        { <$relu as ClampNegative>::CLAMPS },
                    >(
                        crate::scalar::$scale((*value >> 16) as u16, *scale),
                        crate::scalar::$scale(*value as u16, *scale),
                        <$round as PackedNarrowRounding>::ROUNDING,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    );
                    <$zero as CvtNarrowZero>::apply(bits, <$narrow as PackedNarrowFormat>::FORMAT)
                        as $carrier
                }))
            }
        }
    };
}

/// `cvt.rn.satfinite{.relu}.<narrow>x2.f32` — two `f32` primaries, packed pair.
macro_rules! packed_narrow_from_f32_variant {
    ($narrow:ty, $carrier:ty, $relu:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::F32,
                $narrow,
                variant::PackedMode<variant::Rn, variant::SatFinite, $relu>,
            >,
            (R<f32>, R<f32>) => R<$carrier>;
            |_context, _site, (high, low)| {
                Ok(high.zip_map(&low, |_lane, high, low| {
                    crate::scalar::ptx_cvt_pack_narrow_x2::<{ <$relu as ClampNegative>::CLAMPS }>(
                        *high,
                        *low,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    ) as $carrier
                }))
            }
        }
    };
}

/// `cvt.rn.satfinite{.relu}.<narrow>x2.fp16x2` — one packed source, packed pair.
macro_rules! packed_narrow_from_packed_variant {
    ($wide:ty, $decode:ident, $narrow:ty, $carrier:ty, $relu:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $wide,
                $narrow,
                variant::PackedMode<variant::Rn, variant::SatFinite, $relu>,
            >,
            R<u32> => R<$carrier>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| {
                    crate::scalar::ptx_cvt_pack_narrow_x2::<{ <$relu as ClampNegative>::CLAMPS }>(
                        $decode((value >> 16) as u16),
                        $decode(value as u16),
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    ) as $carrier
                }))
            }
        }
    };
}

/// Half pairs use one independent random halfword per input, unlike FP4/6/8 x4.
macro_rules! packed_half_rs_variant {
    ($wide:ty, $random_bits:expr, $convert:path, $decode:path) => {
        register_variant! {
            [impl<S: SaturateFinite, A: ClampNegative>] cvt_spec, variant::Cvt<variant::F32, $wide, variant::PackedMode<variant::Rs, S, A>>,
            (R<f32>, R<f32>, R<u32>) => R<u32>;
            |context, site, (high, low, rbits)| {
                let mask = (1_u32 << $random_bits) - 1;
                let mut output = R::splat(0);
                for lane in context.active_mask() {
                    if rbits[lane] & !(mask | (mask << 16)) != 0 {
                        return Err(EngineError::message(format!(
                            "cvt.rs.f16x2 requires zero reserved rbits at site {} lane {lane}",
                            site.get(),
                        )));
                    }
                    let convert = |value, random| {
                        crate::scalar::ptx_cvt_half_rs::<$random_bits>(
                            value,
                            random,
                            A::CLAMPS,
                            S::SATURATES,
                            $convert,
                            $decode,
                        )
                    };
                    output[lane] = (u32::from(convert(high[lane], (rbits[lane] >> 16) as u16))
                        << 16)
                        | u32::from(convert(low[lane], rbits[lane] as u16));
                }
                Ok(output)
            }
        }
    };
}

packed_half_rs_variant!(
    variant::F16x2,
    13,
    crate::scalar::ptx_cvt_f32_to_f16,
    crate::fp16_bits_to_f32
);
packed_half_rs_variant!(
    variant::Bf16x2,
    16,
    crate::scalar::ptx_cvt_f32_to_bf16,
    crate::bf16_bits_to_f32
);

/// `cvt.rs{.relu}.satfinite.<narrow>x4.f32` — four primaries plus `rbits`.
macro_rules! packed_narrow_rs_variant {
    ($narrow:ty, $carrier:ty, $relu:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::F32,
                $narrow,
                variant::PackedMode<variant::Rs, variant::SatFinite, $relu>,
            >,
            (R<f32>, R<f32>, R<f32>, R<f32>, R<u32>) => R<$carrier>;
            |_context, _site, (a, b, e, f, rbits)| {
                const FORMAT: crate::NarrowFloatFormat = <$narrow as PackedNarrowFormat>::FORMAT;
                Ok(R::from_fn(|lane| {
                    crate::scalar::ptx_cvt_pack_narrow_x4::<{ <$relu as ClampNegative>::CLAMPS }>(
                        [a[lane], b[lane], e[lane], f[lane]],
                        crate::scalar::ptx_cvt_rs_randoms(rbits[lane], FORMAT),
                        FORMAT,
                    ) as $carrier
                }))
            }
        }
    };
}

/// `cvt.rn{.relu}.f16x2.<narrow>x2` — packed pair widened to `f16x2`.
macro_rules! packed_narrow_to_f16x2_variant {
    ($narrow:ty, $carrier:ty, $relu:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $narrow,
                variant::F16x2,
                variant::PackedMode<variant::Rn, variant::NoSatFinite, $relu>,
            >,
            R<$carrier> => R<u32>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| {
                    crate::scalar::ptx_cvt_unpack_narrow_x2_f16x2::<
                        { <$relu as ClampNegative>::CLAMPS },
                    >(u16::from(value), <$narrow as PackedNarrowFormat>::FORMAT)
                }))
            }
        }
    };
}

/// `cvt.rn{.relu}{.satfinite}.bf16x2.<narrow>x2` — packed pair widened.
macro_rules! packed_narrow_to_bf16x2_variant {
    ($narrow:ty, $carrier:ty, $relu:ty, $satfinite:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $narrow,
                variant::Bf16x2,
                variant::PackedMode<variant::Rn, $satfinite, $relu>,
            >,
            R<$carrier> => R<u32>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| {
                    crate::scalar::ptx_cvt_unpack_narrow_x2_bf16x2::<
                        { <$relu as ClampNegative>::CLAMPS },
                        { <$satfinite as SaturateFinite>::SATURATES },
                    >(u16::from(value), <$narrow as PackedNarrowFormat>::FORMAT)
                }))
            }
        }
    };
}

/// `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.<narrow>x2`.
macro_rules! packed_narrow_scaled_variant {
    ($narrow:ty, $carrier:ty, $relu:ty, $satfinite:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                $narrow,
                variant::Bf16x2,
                variant::PackedMode<variant::Rn, $satfinite, $relu, variant::ScaledUe8m0N2>,
            >,
            (R<$carrier>, R<u16>) => R<u32>;
            |_context, _site, (source, scale)| {
                Ok(source.zip_map(&scale, |_lane, value, scale| {
                    crate::scalar::ptx_cvt_unpack_scaled_bf16x2::<
                        { <$relu as ClampNegative>::CLAMPS },
                        { <$satfinite as SaturateFinite>::SATURATES },
                    >(
                        u16::from(*value),
                        *scale,
                        <$narrow as PackedNarrowFormat>::FORMAT,
                    )
                }))
            }
        }
    };
}

// S2F6 has the same modifier and lane contracts as CVT, but its signed fixed-
// point encoding is not a NarrowFloatFormat. Only the six actual operand
// shapes need adapters; both scaled and unscaled forms use one numerical body.
macro_rules! s2f6_cvt_variant {
    ([$($parameters:tt)*] $source:ty => $destination:ty,
     $satfinite:ty, $scale:ty, $args:ty => $output:ty, $body:expr) => {
        register_variant! {
            [impl<$($parameters)*>] cvt_spec, variant::Cvt<
            $source, $destination, variant::PackedMode<variant::Rn, $satfinite, A, $scale>
        >,
            $args => $output;
            |_context, _site, args| {
                Ok(($body)(args))
            }
        }
    };
}

s2f6_cvt_variant!([A: ClampNegative] variant::F32 => variant::S2f6x2,
    variant::SatFinite, variant::NoScale, (R<f32>, R<f32>) => R<u16>,
    |(high, low): (R<f32>, R<f32>)| high.zip_map(&low, |_lane, high, low|
        crate::scalar::ptx_cvt_pack_s2f6x2(*high, *low, 0x7f7f, A::CLAMPS)));
s2f6_cvt_variant!([A: ClampNegative] variant::F32 => variant::S2f6x2,
    variant::SatFinite, variant::ScaledUe8m0N2, (R<f32>, R<f32>, R<u16>) => R<u16>,
    |(high, low, scale): (R<f32>, R<f32>, R<u16>)| R::from_fn(|lane|
        crate::scalar::ptx_cvt_pack_s2f6x2(high[lane], low[lane], scale[lane], A::CLAMPS)));
s2f6_cvt_variant!([A: ClampNegative] variant::Bf16x2 => variant::S2f6x2,
    variant::SatFinite, variant::NoScale, R<u32> => R<u16>,
    |source: R<u32>| source.map(|_lane, value|
        crate::scalar::ptx_cvt_pack_s2f6x2(
            decode_bf16((value >> 16) as u16), decode_bf16(value as u16), 0x7f7f, A::CLAMPS)));
s2f6_cvt_variant!([A: ClampNegative] variant::Bf16x2 => variant::S2f6x2,
    variant::SatFinite, variant::ScaledUe8m0N2, (R<u32>, R<u16>) => R<u16>,
    |(source, scale): (R<u32>, R<u16>)| source.zip_map(&scale, |_lane, value, scale|
        crate::scalar::ptx_cvt_pack_s2f6x2(
            decode_bf16((value >> 16) as u16), decode_bf16(*value as u16), *scale, A::CLAMPS)));
s2f6_cvt_variant!([A: ClampNegative, S: SaturateFinite] variant::S2f6x2 => variant::Bf16x2,
    S, variant::NoScale, R<u16> => R<u32>,
    |source: R<u16>| source.map(|_lane, value|
        crate::scalar::ptx_cvt_unpack_s2f6x2(value, 0x7f7f, A::CLAMPS, S::SATURATES)));
s2f6_cvt_variant!([A: ClampNegative, S: SaturateFinite] variant::S2f6x2 => variant::Bf16x2,
    S, variant::ScaledUe8m0N2, (R<u16>, R<u16>) => R<u32>,
    |(source, scale): (R<u16>, R<u16>)| source.zip_map(&scale, |_lane, value, scale|
        crate::scalar::ptx_cvt_unpack_s2f6x2(*value, *scale, A::CLAMPS, S::SATURATES)));

/// `cvt.{rz,rp}{.satfinite}.ue8m0x2.f32` — two `f32` primaries, packed exponents.
macro_rules! packed_e8m0_from_f32_variant {
    ($round:ty, $satfinite:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::F32,
                variant::Ue8m0x2,
                variant::PackedMode<$round, $satfinite, variant::NoRelu>,
            >,
            (R<f32>, R<f32>) => R<u16>;
            |_context, _site, (high, low)| {
                Ok(high.zip_map(&low, |_lane, high, low| {
                    crate::scalar::ptx_cvt_pack_e8m0x2_f32::<
                        { <$round as RoundExponentUp>::ROUNDS_UP },
                        { <$satfinite as SaturateFinite>::SATURATES },
                    >(*high, *low)
                }))
            }
        }
    };
}

/// `cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2` — one packed source.
macro_rules! packed_e8m0_from_bf16x2_variant {
    ($round:ty, $satfinite:ty) => {
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
                variant::Bf16x2,
                variant::Ue8m0x2,
                variant::PackedMode<$round, $satfinite, variant::NoRelu>,
            >,
            R<u32> => R<u16>;
            |_context, _site, source| {
                Ok(source.map(|_lane, value| {
                    crate::scalar::ptx_cvt_pack_e8m0x2_bf16x2::<
                        { <$round as RoundExponentUp>::ROUNDS_UP },
                        { <$satfinite as SaturateFinite>::SATURATES },
                    >(value)
                }))
            }
        }
    };
}

impl cvt_spec::sealed::Sealed
    for variant::Cvt<
        variant::Ue8m0x2,
        variant::Bf16x2,
        variant::PackedMode<variant::Rn, variant::NoSatFinite, variant::NoRelu>,
    >
{
}

impl cvt_spec::Variant
    for variant::Cvt<
        variant::Ue8m0x2,
        variant::Bf16x2,
        variant::PackedMode<variant::Rn, variant::NoSatFinite, variant::NoRelu>,
    >
{
    type Args = R<u16>;
    type Output = R<u32>;
}

/// `cvt.rn.bf16x2.ue8m0x2` — its one spelling carries no modifier axis.
impl cvt_spec::sealed::Execute
    for variant::Cvt<
        variant::Ue8m0x2,
        variant::Bf16x2,
        variant::PackedMode<variant::Rn, variant::NoSatFinite, variant::NoRelu>,
    >
{
    fn execute(
        _context: ExecCtx,
        _site: SiteId,
        source: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        Ok(source.map(|_lane, value| crate::scalar::ptx_cvt_unpack_e8m0x2_bf16x2(value)))
    }
}

packed_narrow_from_f32_variant!(variant::E4m3x2, u16, variant::NoRelu);
packed_narrow_from_f32_variant!(variant::E4m3x2, u16, variant::Relu);
packed_narrow_from_f32_variant!(variant::E5m2x2, u16, variant::NoRelu);
packed_narrow_from_f32_variant!(variant::E5m2x2, u16, variant::Relu);
packed_narrow_from_f32_variant!(variant::E2m1x2, u8, variant::NoRelu);
packed_narrow_from_f32_variant!(variant::E2m1x2, u8, variant::Relu);

packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E4m3x2,
    u16,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E4m3x2,
    u16,
    variant::Relu
);
packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E5m2x2,
    u16,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E5m2x2,
    u16,
    variant::Relu
);
packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E2m1x2,
    u8,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::E2m1x2,
    u8,
    variant::Relu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E4m3x2,
    u16,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E4m3x2,
    u16,
    variant::Relu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E5m2x2,
    u16,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E5m2x2,
    u16,
    variant::Relu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E2m1x2,
    u8,
    variant::NoRelu
);
packed_narrow_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::E2m1x2,
    u8,
    variant::Relu
);

// PTX 9.4 adds `.rz` and `.scaled::n1::ue8m0` to the five signed packed
// destinations. Bare `.rn` spellings belong to pre-9.4 schema families;
// the older three formats use their nearest-even specializations above,
// and F6 reuses the rounded conversion body below.
macro_rules! ptx94_narrow_unscaled_round {
    ($narrow:ty, $carrier:ty, $round:ty, $zero:ty) => {
        packed_narrow_rounded_from_f32_variant!($narrow, $carrier, $round, variant::NoRelu, $zero);
        packed_narrow_rounded_from_f32_variant!($narrow, $carrier, $round, variant::Relu, $zero);
        packed_narrow_rounded_from_packed_variant!(
            variant::F16x2,
            decode_f16,
            $narrow,
            $carrier,
            $round,
            variant::NoRelu,
            $zero
        );
        packed_narrow_rounded_from_packed_variant!(
            variant::F16x2,
            decode_f16,
            $narrow,
            $carrier,
            $round,
            variant::Relu,
            $zero
        );
        packed_narrow_rounded_from_packed_variant!(
            variant::Bf16x2,
            decode_bf16,
            $narrow,
            $carrier,
            $round,
            variant::NoRelu,
            $zero
        );
        packed_narrow_rounded_from_packed_variant!(
            variant::Bf16x2,
            decode_bf16,
            $narrow,
            $carrier,
            $round,
            variant::Relu,
            $zero
        );
    };
}

macro_rules! ptx94_narrow_scaled_round {
    ($narrow:ty, $carrier:ty, $round:ty, $zero:ty) => {
        packed_narrow_scaled_n1_from_f32_variant!(
            $narrow,
            $carrier,
            $round,
            variant::NoRelu,
            $zero
        );
        packed_narrow_scaled_n1_from_f32_variant!($narrow, $carrier, $round, variant::Relu, $zero);
        packed_narrow_scaled_n1_from_packed_variant!(
            variant::F16x2,
            ptx_cvt_scaled_n1_f16,
            $narrow,
            $carrier,
            $round,
            variant::NoRelu,
            $zero
        );
        packed_narrow_scaled_n1_from_packed_variant!(
            variant::F16x2,
            ptx_cvt_scaled_n1_f16,
            $narrow,
            $carrier,
            $round,
            variant::Relu,
            $zero
        );
        packed_narrow_scaled_n1_from_packed_variant!(
            variant::Bf16x2,
            ptx_cvt_scaled_n1_bf16,
            $narrow,
            $carrier,
            $round,
            variant::NoRelu,
            $zero
        );
        packed_narrow_scaled_n1_from_packed_variant!(
            variant::Bf16x2,
            ptx_cvt_scaled_n1_bf16,
            $narrow,
            $carrier,
            $round,
            variant::Relu,
            $zero
        );
    };
}

macro_rules! ptx94_narrow_format {
    ($narrow:ty, $carrier:ty) => {
        ptx94_narrow_unscaled_round!($narrow, $carrier, variant::Rz, variant::PreserveZero);
        ptx94_narrow_unscaled_round!($narrow, $carrier, variant::Rn, variant::Pzo);
        ptx94_narrow_unscaled_round!($narrow, $carrier, variant::Rz, variant::Pzo);
        ptx94_narrow_scaled_round!($narrow, $carrier, variant::Rn, variant::PreserveZero);
        ptx94_narrow_scaled_round!($narrow, $carrier, variant::Rz, variant::PreserveZero);
        ptx94_narrow_scaled_round!($narrow, $carrier, variant::Rn, variant::Pzo);
        ptx94_narrow_scaled_round!($narrow, $carrier, variant::Rz, variant::Pzo);
    };
}

ptx94_narrow_format!(variant::E4m3x2, u16);
ptx94_narrow_format!(variant::E5m2x2, u16);
ptx94_narrow_format!(variant::E2m1x2, u8);
ptx94_narrow_format!(variant::E2m3x2, u16);
ptx94_narrow_format!(variant::E3m2x2, u16);
ptx94_narrow_unscaled_round!(variant::E2m3x2, u16, variant::Rn, variant::PreserveZero);
ptx94_narrow_unscaled_round!(variant::E3m2x2, u16, variant::Rn, variant::PreserveZero);

macro_rules! ue5m3_unsaturated_variant {
    ($source:ty, $scale:ty, $args:ident: $args_type:ty, $lane:ident => $values:expr;
        $($round:ty),+ $(,)?) => {$(
        register_variant! {
            [impl] cvt_spec, variant::Cvt<
            $source, variant::Ue5m3x2,
            variant::PackedMode<$round, variant::NoSatFinite, variant::NoRelu, $scale>,
        >,
            $args_type => R<u16>;
            |context, _site, $args| {
                let mut result = R::splat(0);
                for $lane in context.active_mask().iter() {
                    let (high, low, scale) = $values;
                    result[$lane] = crate::scalar::ptx_cvt_pack_ue5m3x2_unsaturated(
                        high, low, <$round as PackedNarrowRounding>::ROUNDING, scale,
                    );
                }
                Ok(result)
            }
        }
    )+};
}

ue5m3_unsaturated_variant!(
    variant::F32, variant::NoScale, args: (R<f32>, R<f32>), lane => (args.0[lane], args.1[lane], 127);
    variant::Rn, variant::Rz, variant::Rp
);
ue5m3_unsaturated_variant!(
    variant::F16x2, variant::NoScale, source: R<u32>, lane =>
        (decode_f16((source[lane] >> 16) as u16), decode_f16(source[lane] as u16), 127);
    variant::Rn, variant::Rz, variant::Rp
);
ue5m3_unsaturated_variant!(
    variant::Bf16x2, variant::NoScale, source: R<u32>, lane =>
        (decode_bf16((source[lane] >> 16) as u16), decode_bf16(source[lane] as u16), 127);
    variant::Rn, variant::Rz, variant::Rp
);
ue5m3_unsaturated_variant!(
    variant::F32, variant::ScaledUe8m0N1, args: (R<f32>, R<f32>, R<u8>), lane => (
        crate::scalar::ptx_cvt_scaled_n1_f32(args.0[lane], 127),
        crate::scalar::ptx_cvt_scaled_n1_f32(args.1[lane], 127), args.2[lane]
    ); variant::Rn, variant::Rz
);
ue5m3_unsaturated_variant!(
    variant::F16x2, variant::ScaledUe8m0N1, args: (R<u32>, R<u8>), lane => (
        crate::scalar::ptx_cvt_scaled_n1_f16((args.0[lane] >> 16) as u16, 127),
        crate::scalar::ptx_cvt_scaled_n1_f16(args.0[lane] as u16, 127), args.1[lane]
    ); variant::Rn, variant::Rz
);
ue5m3_unsaturated_variant!(
    variant::Bf16x2, variant::ScaledUe8m0N1, args: (R<u32>, R<u8>), lane => (
        crate::scalar::ptx_cvt_scaled_n1_bf16((args.0[lane] >> 16) as u16, 127),
        crate::scalar::ptx_cvt_scaled_n1_bf16(args.0[lane] as u16, 127), args.1[lane]
    ); variant::Rn, variant::Rz
);

packed_narrow_rounded_from_f32_variant!(variant::Ue5m3x2, u16, variant::Rn, variant::NoRelu);
packed_narrow_rounded_from_f32_variant!(variant::Ue5m3x2, u16, variant::Rz, variant::NoRelu);
packed_narrow_rounded_from_f32_variant!(variant::Ue5m3x2, u16, variant::Rp, variant::NoRelu);
packed_narrow_rounded_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::Ue5m3x2,
    u16,
    variant::Rn,
    variant::NoRelu
);
packed_narrow_rounded_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::Ue5m3x2,
    u16,
    variant::Rz,
    variant::NoRelu
);
packed_narrow_rounded_from_packed_variant!(
    variant::F16x2,
    decode_f16,
    variant::Ue5m3x2,
    u16,
    variant::Rp,
    variant::NoRelu
);
packed_narrow_rounded_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::Ue5m3x2,
    u16,
    variant::Rn,
    variant::NoRelu
);
packed_narrow_rounded_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::Ue5m3x2,
    u16,
    variant::Rz,
    variant::NoRelu
);
packed_narrow_rounded_from_packed_variant!(
    variant::Bf16x2,
    decode_bf16,
    variant::Ue5m3x2,
    u16,
    variant::Rp,
    variant::NoRelu
);
packed_narrow_scaled_n1_from_f32_variant!(variant::Ue5m3x2, u16, variant::Rn, variant::NoRelu);
packed_narrow_scaled_n1_from_f32_variant!(variant::Ue5m3x2, u16, variant::Rz, variant::NoRelu);
packed_narrow_scaled_n1_from_packed_variant!(
    variant::F16x2,
    ptx_cvt_scaled_n1_f16,
    variant::Ue5m3x2,
    u16,
    variant::Rn,
    variant::NoRelu
);
packed_narrow_scaled_n1_from_packed_variant!(
    variant::F16x2,
    ptx_cvt_scaled_n1_f16,
    variant::Ue5m3x2,
    u16,
    variant::Rz,
    variant::NoRelu
);
packed_narrow_scaled_n1_from_packed_variant!(
    variant::Bf16x2,
    ptx_cvt_scaled_n1_bf16,
    variant::Ue5m3x2,
    u16,
    variant::Rn,
    variant::NoRelu
);
packed_narrow_scaled_n1_from_packed_variant!(
    variant::Bf16x2,
    ptx_cvt_scaled_n1_bf16,
    variant::Ue5m3x2,
    u16,
    variant::Rz,
    variant::NoRelu
);

packed_narrow_rs_variant!(variant::E4m3x4, u32, variant::NoRelu);
packed_narrow_rs_variant!(variant::E4m3x4, u32, variant::Relu);
packed_narrow_rs_variant!(variant::E5m2x4, u32, variant::NoRelu);
packed_narrow_rs_variant!(variant::E5m2x4, u32, variant::Relu);
packed_narrow_rs_variant!(variant::E2m1x4, u16, variant::NoRelu);
packed_narrow_rs_variant!(variant::E2m1x4, u16, variant::Relu);
packed_narrow_rs_variant!(variant::E2m3x4, u32, variant::NoRelu);
packed_narrow_rs_variant!(variant::E2m3x4, u32, variant::Relu);
packed_narrow_rs_variant!(variant::E3m2x4, u32, variant::NoRelu);
packed_narrow_rs_variant!(variant::E3m2x4, u32, variant::Relu);

macro_rules! packed_narrow_widenings {
    ($narrow:ty, $carrier:ty) => {
        packed_narrow_to_f16x2_variant!($narrow, $carrier, variant::NoRelu);
        packed_narrow_to_f16x2_variant!($narrow, $carrier, variant::Relu);
        packed_narrow_to_bf16x2_variant!($narrow, $carrier, variant::NoRelu, variant::NoSatFinite);
        packed_narrow_to_bf16x2_variant!($narrow, $carrier, variant::NoRelu, variant::SatFinite);
        packed_narrow_to_bf16x2_variant!($narrow, $carrier, variant::Relu, variant::NoSatFinite);
        packed_narrow_to_bf16x2_variant!($narrow, $carrier, variant::Relu, variant::SatFinite);
        packed_narrow_scaled_variant!($narrow, $carrier, variant::NoRelu, variant::NoSatFinite);
        packed_narrow_scaled_variant!($narrow, $carrier, variant::NoRelu, variant::SatFinite);
        packed_narrow_scaled_variant!($narrow, $carrier, variant::Relu, variant::NoSatFinite);
        packed_narrow_scaled_variant!($narrow, $carrier, variant::Relu, variant::SatFinite);
    };
}

packed_narrow_widenings!(variant::E4m3x2, u16);
packed_narrow_widenings!(variant::E5m2x2, u16);
packed_narrow_widenings!(variant::E2m1x2, u8);
packed_narrow_widenings!(variant::E2m3x2, u16);
packed_narrow_widenings!(variant::E3m2x2, u16);

packed_narrow_to_f16x2_variant!(variant::Ue5m3x2, u16, variant::NoRelu);
packed_narrow_to_bf16x2_variant!(variant::Ue5m3x2, u16, variant::NoRelu, variant::NoSatFinite);
packed_narrow_to_bf16x2_variant!(variant::Ue5m3x2, u16, variant::NoRelu, variant::SatFinite);
packed_narrow_scaled_variant!(variant::Ue5m3x2, u16, variant::NoRelu, variant::NoSatFinite);
packed_narrow_scaled_variant!(variant::Ue5m3x2, u16, variant::NoRelu, variant::SatFinite);

packed_e8m0_from_f32_variant!(variant::Rz, variant::NoSatFinite);
packed_e8m0_from_f32_variant!(variant::Rz, variant::SatFinite);
packed_e8m0_from_f32_variant!(variant::Rp, variant::NoSatFinite);
packed_e8m0_from_f32_variant!(variant::Rp, variant::SatFinite);
packed_e8m0_from_bf16x2_variant!(variant::Rz, variant::NoSatFinite);
packed_e8m0_from_bf16x2_variant!(variant::Rz, variant::SatFinite);
packed_e8m0_from_bf16x2_variant!(variant::Rp, variant::NoSatFinite);
packed_e8m0_from_bf16x2_variant!(variant::Rp, variant::SatFinite);

// ---------------------------------------------------------------------------
// Scalar PTX `cvt` grammar forms.
//
// Numeric bodies live in `crate::scalar`; everything below is marker plumbing
// that binds one PTX spelling to one of them.  The legality of each row was
// checked against ptxas for `sm_100a` and the results against a B200.
// ---------------------------------------------------------------------------

/// One `cvt.irnd{.ftz}{.sat}.<int>.<float>` specialization.
macro_rules! cvt_float_to_int_form {
    (
        $src_marker:ty, $src:ty, $widen:expr, $round:path, $ftz:expr, $subnormal:ty,
        $dst_marker:ty, $dst:ty, $nan:expr, $round_marker:ty, $mode:expr
    ) => {
        exact_cvt_variant!(
            $src_marker => $dst_marker,
            variant::CvtMode<$round_marker, $subnormal>,
            $src => $dst,
            |value: $src| {
                let widened = $widen(value);
                if widened.is_nan() {
                    $nan
                } else {
                    $round(widened, $mode, $ftz) as $dst
                }
            }
        );
    };
}

/// The four `.irnd` spellings of one `(source, destination)` pair.
macro_rules! cvt_float_to_int_roundings {
    (
        $src_marker:ty, $src:ty, $widen:expr, $round:path, $ftz:expr, $subnormal:ty,
        $( $dst_marker:ty => $dst:ty, $nan:expr );+ $(;)?
    ) => {
        $(
            cvt_float_to_int_form!($src_marker, $src, $widen, $round, $ftz, $subnormal,
                $dst_marker, $dst, $nan, variant::Rni, PtxIntegerRounding::NearestEven);
            cvt_float_to_int_form!($src_marker, $src, $widen, $round, $ftz, $subnormal,
                $dst_marker, $dst, $nan, variant::Rzi, PtxIntegerRounding::Zero);
            cvt_float_to_int_form!($src_marker, $src, $widen, $round, $ftz, $subnormal,
                $dst_marker, $dst, $nan, variant::Rmi, PtxIntegerRounding::NegativeInfinity);
            cvt_float_to_int_form!($src_marker, $src, $widen, $round, $ftz, $subnormal,
                $dst_marker, $dst, $nan, variant::Rpi, PtxIntegerRounding::PositiveInfinity);
        )+
    };
}

/// Destinations reachable from a non-`.f64` float, with the ISA's NaN result:
/// zero unless the destination is 64-bit.
macro_rules! cvt_narrow_source_int_destinations {
    ($src_marker:ty, $src:ty, $widen:expr, $ftz:expr, $subnormal:ty) => {
        cvt_float_to_int_roundings!(
            $src_marker, $src, $widen, crate::scalar::ptx_cvt_integral_f32, $ftz, $subnormal,
            variant::U8 => u8, 0;
            variant::I8 => i8, 0;
            variant::U16 => u16, 0;
            variant::I16 => i16, 0;
            variant::U32 => u32, 0;
            variant::I32 => i32, 0;
            variant::U64 => u64, 1u64 << 63;
            variant::I64 => i64, i64::MIN
        );
    };
}

/// `.bf16` cannot reach an 8-bit integer destination; ptxas rejects the pair.
macro_rules! cvt_bf16_source_int_destinations {
    ($subnormal:ty) => {
        cvt_float_to_int_roundings!(
            variant::Bf16, u16, decode_bf16, crate::scalar::ptx_cvt_integral_f32, false, $subnormal,
            variant::U16 => u16, 0;
            variant::I16 => i16, 0;
            variant::U32 => u32, 0;
            variant::I32 => i32, 0;
            variant::U64 => u64, 1u64 << 63;
            variant::I64 => i64, i64::MIN
        );
    };
}

cvt_narrow_source_int_destinations!(
    variant::F32,
    f32,
    core::convert::identity,
    false,
    variant::PreserveSubnormal
);
cvt_narrow_source_int_destinations!(
    variant::F32,
    f32,
    core::convert::identity,
    true,
    variant::Ftz
);
cvt_narrow_source_int_destinations!(
    variant::F16,
    u16,
    decode_f16,
    false,
    variant::PreserveSubnormal
);
cvt_bf16_source_int_destinations!(variant::PreserveSubnormal);

// An `.f64` source promotes every NaN answer to `1 << (BitWidth(dst) - 1)`.
cvt_float_to_int_roundings!(
    variant::F64, f64, core::convert::identity, ptx_cvt_integral_f64_ignoring_ftz, false,
    variant::PreserveSubnormal,
    variant::U8 => u8, 0x80;
    variant::I8 => i8, i8::MIN;
    variant::U16 => u16, 0x8000;
    variant::I16 => i16, i16::MIN;
    variant::U32 => u32, 0x8000_0000;
    variant::I32 => i32, i32::MIN;
    variant::U64 => u64, 1u64 << 63;
    variant::I64 => i64, i64::MIN
);

/// `.ftz` is illegal without an `.f32` operand, so the `f64` rows ignore it.
fn ptx_cvt_integral_f64_ignoring_ftz(value: f64, rounding: PtxIntegerRounding, _ftz: bool) -> f64 {
    crate::scalar::ptx_cvt_integral_f64(value, rounding)
}

/// One `cvt.frnd{.ftz}.<float>.<int>` specialization.
macro_rules! cvt_int_to_float_form {
    (
        $src_marker:ty, $src:ty, $magnitude:expr, $negative:expr,
        $dst_marker:ty, $dst:ty, $convert:path, $round_marker:ty, $mode:expr
    ) => {
        exact_cvt_variant!(
            $src_marker => $dst_marker,
            variant::CvtMode<$round_marker>,
            $src => $dst,
            |value: $src| $convert($magnitude(value), $negative(value), $mode)
        );
    };
}

/// The four `.frnd` spellings of one integer source into one float
/// destination, for the pairs where the conversion can actually round.
macro_rules! cvt_int_to_float_directed {
    (
        $src_marker:ty, $src:ty, $magnitude:expr, $negative:expr,
        $dst_marker:ty, $dst:ty, $convert:path
    ) => {
        cvt_int_to_float_form!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            $dst_marker,
            $dst,
            $convert,
            variant::Rn,
            PtxFloatRounding::NearestEven
        );
        cvt_int_to_float_form!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            $dst_marker,
            $dst,
            $convert,
            variant::Rz,
            PtxFloatRounding::Zero
        );
        cvt_int_to_float_form!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            $dst_marker,
            $dst,
            $convert,
            variant::Rm,
            PtxFloatRounding::NegativeInfinity
        );
        cvt_int_to_float_form!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            $dst_marker,
            $dst,
            $convert,
            variant::Rp,
            PtxFloatRounding::PositiveInfinity
        );
    };
}

/// The one specialization every `.frnd` spelling of an *exact* integer-to-float
/// pair binds to.
///
/// When the source's bit width is at most the destination's significand width,
/// every value of the source type is exactly representable, so no rounding ever
/// occurs.  An integer source also has no NaN, no infinity and no signed zero,
/// so there is no corner where the modes could diverge: `.rn`, `.rz`, `.rm` and
/// `.rp` are one function.  The rounding axis therefore does not exist for
/// these pairs, and `Exact` names that rather than picking an arbitrary mode.
/// The paired integer-conversion tests exercise the boundary on each side.
macro_rules! cvt_int_to_float_exact {
    (
        $src_marker:ty, $src:ty, $magnitude:expr, $negative:expr,
        $dst_marker:ty, $dst:ty, $convert:path
    ) => {
        cvt_int_to_float_form!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            $dst_marker,
            $dst,
            $convert,
            variant::Exact,
            PtxFloatRounding::NearestEven
        );
    };
}

// `.ftz` is legal on the `.f32` destination but inert: an integer magnitude
// never rounds to a subnormal, so the flush has nothing to do.  Every
// `cvt.frnd.ftz.f32.<int>` spelling was measured bit-identical to its plain
// twin, so the lowering binds both to the `PreserveSubnormal` entry and no
// `Ftz` instantiation exists here -- the same treatment `.sat` gets on the
// float-to-integer rows.
macro_rules! cvt_int_to_float_roundings {
    (
        $src_marker:ty, $src:ty, $magnitude:expr, $negative:expr,
        $f32_rounding:ident, $f64_rounding:ident
        $(; $half_marker:ty, $half_convert:path, $half_rounding:ident)*
    ) => {
        $f32_rounding!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            variant::F32,
            f32,
            crate::scalar::ptx_cvt_integer_to_f32
        );
        $f64_rounding!(
            $src_marker,
            $src,
            $magnitude,
            $negative,
            variant::F64,
            f64,
            crate::scalar::ptx_cvt_integer_to_f64
        );
        $(
            $half_rounding!(
                $src_marker, $src, $magnitude, $negative,
                $half_marker, u16, $half_convert
            );
        )*
        // For integer sources, post-conversion [0, 1] saturation is exactly
        // this predicate, even when ordinary conversion would round. Share
        // one Exact specialization across every rounding/FTZ spelling.
        exact_cvt_variant!(
            $src_marker => variant::F16,
            variant::CvtMode<variant::Exact, variant::PreserveSubnormal, variant::Sat>,
            $src => u16,
            |value| u16::from(value > 0) * 0x3c00
        );
        exact_cvt_variant!(
            $src_marker => variant::F32,
            variant::CvtMode<variant::Exact, variant::PreserveSubnormal, variant::Sat>,
            $src => f32,
            |value| f32::from(u8::from(value > 0))
        );
        exact_cvt_variant!(
            $src_marker => variant::F64,
            variant::CvtMode<variant::Exact, variant::PreserveSubnormal, variant::Sat>,
            $src => f64,
            |value| f64::from(u8::from(value > 0))
        );
    };
}

// Exact pairs omit the rounding axis; other pairs retain all four modes.
cvt_int_to_float_roundings!(
    variant::U8,
    u8,
    |value: u8| u64::from(value),
    |_| false,
    cvt_int_to_float_exact,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_exact
);
cvt_int_to_float_roundings!(
    variant::U16,
    u16,
    |value: u16| u64::from(value),
    |_| false,
    cvt_int_to_float_exact,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);
cvt_int_to_float_roundings!(
    variant::U32,
    u32,
    |value: u32| u64::from(value),
    |_| false,
    cvt_int_to_float_directed,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);
cvt_int_to_float_roundings!(
    variant::U64,
    u64,
    |value: u64| value,
    |_| false,
    cvt_int_to_float_directed,
    cvt_int_to_float_directed
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);
cvt_int_to_float_roundings!(
    variant::I8,
    i8,
    |value: i8| i64::from(value).unsigned_abs(),
    |value: i8| value < 0,
    cvt_int_to_float_exact,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_exact
);
cvt_int_to_float_roundings!(
    variant::I16,
    i16,
    |value: i16| i64::from(value).unsigned_abs(),
    |value: i16| value < 0,
    cvt_int_to_float_exact,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);
cvt_int_to_float_roundings!(
    variant::I32,
    i32,
    |value: i32| i64::from(value).unsigned_abs(),
    |value: i32| value < 0,
    cvt_int_to_float_directed,
    cvt_int_to_float_exact
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);
cvt_int_to_float_roundings!(
    variant::I64,
    i64,
    |value: i64| value.unsigned_abs(),
    |value: i64| value < 0,
    cvt_int_to_float_directed,
    cvt_int_to_float_directed
    ; variant::F16, crate::scalar::ptx_cvt_integer_to_f16, cvt_int_to_float_directed
    ; variant::Bf16, crate::scalar::ptx_cvt_integer_to_bf16, cvt_int_to_float_directed
);

/// `cvt.frnd{.ftz}{.sat}.f32.f64`: saturation follows the shared conversion.
macro_rules! cvt_f64_to_f32_rounding {
    ($round_marker:ty, $mode:expr) => {
        exact_cvt_variant!(
            variant::F64 => variant::F32,
            variant::CvtMode<$round_marker, variant::PreserveSubnormal, variant::Sat>,
            f64 => f32,
            |value: f64| crate::scalar::ptx_saturate_f32(
                crate::scalar::ptx_cvt_f64_to_f32(value, $mode, false)
            )
        );
        exact_cvt_variant!(
            variant::F64 => variant::F32,
            variant::CvtMode<$round_marker, variant::Ftz, variant::Sat>,
            f64 => f32,
            |value: f64| crate::scalar::ptx_saturate_f32(
                crate::scalar::ptx_cvt_f64_to_f32(value, $mode, true)
            )
        );
        exact_cvt_variant!(
            variant::F64 => variant::F32,
            variant::CvtMode<$round_marker>,
            f64 => f32,
            |value: f64| crate::scalar::ptx_cvt_f64_to_f32(value, $mode, false)
        );
        exact_cvt_variant!(
            variant::F64 => variant::F32,
            variant::CvtMode<$round_marker, variant::Ftz>,
            f64 => f32,
            |value: f64| crate::scalar::ptx_cvt_f64_to_f32(value, $mode, true)
        );
    };
}

cvt_f64_to_f32_rounding!(variant::Rn, PtxFloatRounding::NearestEven);
cvt_f64_to_f32_rounding!(variant::Rz, PtxFloatRounding::Zero);
cvt_f64_to_f32_rounding!(variant::Rm, PtxFloatRounding::NegativeInfinity);
cvt_f64_to_f32_rounding!(variant::Rp, PtxFloatRounding::PositiveInfinity);

/// Same-size float conversions: integral rounding, then optional saturation.
macro_rules! cvt_same_size_integral {
    ($round_marker:ty, $mode:expr) => {
        // Half inputs widen exactly. Their rounded integers remain exactly
        // representable in the original format, so encoding cannot round twice.
        exact_cvt_variant!(
            variant::F16 => variant::F16,
            variant::CvtMode<$round_marker>,
            u16 => u16,
            |value: u16| encode_f16(crate::scalar::ptx_cvt_integral_f32(decode_f16(value), $mode, false))
        );
        exact_cvt_variant!(
            variant::Bf16 => variant::Bf16,
            variant::CvtMode<$round_marker>,
            u16 => u16,
            |value: u16| encode_bf16(crate::scalar::ptx_cvt_integral_f32(decode_bf16(value), $mode, false))
        );
        exact_cvt_variant!(
            variant::F16 => variant::F16,
            variant::CvtMode<$round_marker, variant::PreserveSubnormal, variant::Sat>,
            u16 => u16,
            |value: u16| encode_f16(crate::scalar::ptx_saturate_f32(
                crate::scalar::ptx_cvt_integral_f32(decode_f16(value), $mode, false)
            ))
        );
        exact_cvt_variant!(
            variant::F32 => variant::F32,
            variant::CvtMode<$round_marker, variant::PreserveSubnormal, variant::Sat>,
            f32 => f32,
            |value: f32| crate::scalar::ptx_saturate_f32(
                crate::scalar::ptx_cvt_integral_f32_to_f32(value, $mode, false)
            )
        );
        exact_cvt_variant!(
            variant::F32 => variant::F32,
            variant::CvtMode<$round_marker, variant::Ftz, variant::Sat>,
            f32 => f32,
            |value: f32| crate::scalar::ptx_saturate_f32(
                crate::scalar::ptx_cvt_integral_f32_to_f32(value, $mode, true)
            )
        );
        exact_cvt_variant!(
            variant::F64 => variant::F64,
            variant::CvtMode<$round_marker, variant::PreserveSubnormal, variant::Sat>,
            f64 => f64,
            |value: f64| crate::scalar::ptx_saturate_f64(
                crate::scalar::ptx_cvt_integral_f64_to_f64(value, $mode)
            )
        );
        exact_cvt_variant!(
            variant::F32 => variant::F32,
            variant::CvtMode<$round_marker>,
            f32 => f32,
            |value: f32| crate::scalar::ptx_cvt_integral_f32_to_f32(value, $mode, false)
        );
        exact_cvt_variant!(
            variant::F32 => variant::F32,
            variant::CvtMode<$round_marker, variant::Ftz>,
            f32 => f32,
            |value: f32| crate::scalar::ptx_cvt_integral_f32_to_f32(value, $mode, true)
        );
        exact_cvt_variant!(
            variant::F64 => variant::F64,
            variant::CvtMode<$round_marker>,
            f64 => f64,
            |value: f64| crate::scalar::ptx_cvt_integral_f64_to_f64(value, $mode)
        );
    };
}

cvt_same_size_integral!(variant::Rni, PtxIntegerRounding::NearestEven);
cvt_same_size_integral!(variant::Rzi, PtxIntegerRounding::Zero);
cvt_same_size_integral!(variant::Rmi, PtxIntegerRounding::NegativeInfinity);
cvt_same_size_integral!(variant::Rpi, PtxIntegerRounding::PositiveInfinity);

// `.ftz` / `.sat` spellings of unmodified float conversions. The plain forms
// keep their existing two-parameter markers.
exact_cvt_variant!(
    variant::F16 => variant::F16,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    u16 => u16,
    |value: u16| encode_f16(crate::scalar::ptx_saturate_f32(decode_f16(value)))
);
exact_cvt_variant!(
    variant::F64 => variant::F64,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    f64 => f64,
    crate::scalar::ptx_saturate_f64
);
exact_cvt_variant!(
    variant::F32 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::Ftz, variant::Sat>,
    f32 => f32,
    |value: f32| crate::scalar::ptx_saturate_f32(crate::scalar::ptx_cvt_f32_to_f32(value, true))
);
exact_cvt_variant!(
    variant::F32 => variant::F64,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    f32 => f64,
    |value: f32| crate::scalar::ptx_saturate_f64(f64::from(value))
);
exact_cvt_variant!(
    variant::F32 => variant::F64,
    variant::CvtMode<variant::Unmodified, variant::Ftz, variant::Sat>,
    f32 => f64,
    |value: f32| crate::scalar::ptx_saturate_f64(crate::scalar::ptx_cvt_f32_to_f64(value, true))
);
exact_cvt_variant!(
    variant::F16 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    u16 => f32,
    |value: u16| crate::scalar::ptx_saturate_f32(decode_f16(value))
);
exact_cvt_variant!(
    variant::F16 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::Ftz, variant::Sat>,
    u16 => f32,
    |value: u16| crate::scalar::ptx_saturate_f32(
        crate::scalar::ptx_cvt_widen_to_f32(decode_f16(value), true)
    )
);
exact_cvt_variant!(
    variant::F16 => variant::F64,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    u16 => f64,
    |value: u16| crate::scalar::ptx_saturate_f64(f64::from(decode_f16(value)))
);
exact_cvt_variant!(
    variant::F32 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::Ftz>,
    f32 => f32,
    |value: f32| crate::scalar::ptx_cvt_f32_to_f32(value, true)
);
exact_cvt_variant!(
    variant::F32 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::PreserveSubnormal, variant::Sat>,
    f32 => f32,
    crate::scalar::ptx_saturate_f32
);
exact_cvt_variant!(
    variant::F16 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::Ftz>,
    u16 => f32,
    |value: u16| crate::scalar::ptx_cvt_widen_to_f32(decode_f16(value), true)
);
exact_cvt_variant!(
    variant::Bf16 => variant::F32,
    variant::CvtMode<variant::Unmodified, variant::Ftz>,
    u16 => f32,
    |value: u16| crate::scalar::ptx_cvt_widen_to_f32(decode_bf16(value), true)
);
exact_cvt_variant!(
    variant::F32 => variant::F64,
    variant::CvtMode<variant::Unmodified, variant::Ftz>,
    f32 => f64,
    |value: f32| crate::scalar::ptx_cvt_f32_to_f64(value, true)
);

// The generic half grammar adds FTZ, SAT and RM/RP. Its modifiers are the
// existing scalar CvtMode axes, not PackedMode's ReLU/satfinite product.
macro_rules! cvt_f32_half_forms {
    ($dst:ty, $convert:path, $encode:ident, $decode:ident;
        $($round:ty, $mode:expr => [$($flush:ty, $clamp:ty);+]);+ $(;)?) => {
        $($(exact_cvt_variant!(
            variant::F32 => $dst,
            variant::CvtMode<$round, $flush, $clamp>,
            f32 => u16,
            |value: f32| {
                let input = if <$flush as SubnormalMode>::FTZ {
                    crate::scalar::flush_subnormal_f32(value)
                } else {
                    value
                };
                let result = $convert(input, $mode, false, false);
                if <$clamp as F32Clamp>::SATURATE {
                    $encode(crate::scalar::ptx_saturate_f32($decode(result)))
                } else {
                    result
                }
            }
        );)+)+
    };
}

cvt_f32_half_forms!(
    variant::F16, crate::scalar::ptx_cvt_f32_to_f16, encode_f16, decode_f16;
    variant::Rn, PtxFloatRounding::NearestEven => [
        variant::Ftz, variant::NoSat;
        variant::PreserveSubnormal, variant::Sat;
        variant::Ftz, variant::Sat];
    variant::Rz, PtxFloatRounding::Zero => [
        variant::Ftz, variant::NoSat;
        variant::PreserveSubnormal, variant::Sat;
        variant::Ftz, variant::Sat];
    variant::Rm, PtxFloatRounding::NegativeInfinity => [
        variant::PreserveSubnormal, variant::NoSat;
        variant::Ftz, variant::NoSat;
        variant::PreserveSubnormal, variant::Sat;
        variant::Ftz, variant::Sat];
    variant::Rp, PtxFloatRounding::PositiveInfinity => [
        variant::PreserveSubnormal, variant::NoSat;
        variant::Ftz, variant::NoSat;
        variant::PreserveSubnormal, variant::Sat;
        variant::Ftz, variant::Sat];
);
cvt_f32_half_forms!(
    variant::Bf16, crate::scalar::ptx_cvt_f32_to_bf16, encode_bf16, decode_bf16;
    variant::Rn, PtxFloatRounding::NearestEven => [variant::Ftz, variant::NoSat];
    variant::Rz, PtxFloatRounding::Zero => [variant::Ftz, variant::NoSat];
    variant::Rm, PtxFloatRounding::NegativeInfinity => [
        variant::PreserveSubnormal, variant::NoSat; variant::Ftz, variant::NoSat];
    variant::Rp, PtxFloatRounding::PositiveInfinity => [
        variant::PreserveSubnormal, variant::NoSat; variant::Ftz, variant::NoSat];
);

// FP64 inputs round directly from their full significand. Half-cross inputs
// widen exactly to FP32, so the existing converter performs the only rounding.
macro_rules! cvt_half_cross_rounding {
    ($round:ty, $mode:expr) => {
        exact_cvt_variant!(
            variant::F64 => variant::F16,
            variant::CvtMode<$round>,
            f64 => u16,
            |value| crate::scalar::ptx_cvt_f64_to_low(value, $mode, crate::scalar::LowPrecisionFormat::F16)
        );
        exact_cvt_variant!(
            variant::F64 => variant::F16,
            variant::CvtMode<$round, variant::PreserveSubnormal, variant::Sat>,
            f64 => u16,
            |value| encode_f16(crate::scalar::ptx_saturate_f32(decode_f16(
                crate::scalar::ptx_cvt_f64_to_low(value, $mode, crate::scalar::LowPrecisionFormat::F16)
            )))
        );
        exact_cvt_variant!(
            variant::F64 => variant::Bf16,
            variant::CvtMode<$round>,
            f64 => u16,
            |value| crate::scalar::ptx_cvt_f64_to_low(value, $mode, crate::scalar::LowPrecisionFormat::Bf16)
        );
        exact_cvt_variant!(
            variant::F16 => variant::Bf16,
            variant::CvtMode<$round>,
            u16 => u16,
            |value| crate::scalar::ptx_cvt_f32_to_bf16(decode_f16(value), $mode, false, false)
        );
        exact_cvt_variant!(
            variant::Bf16 => variant::F16,
            variant::CvtMode<$round>,
            u16 => u16,
            |value| crate::scalar::ptx_cvt_f32_to_f16(decode_bf16(value), $mode, false, false)
        );
    };
}

cvt_half_cross_rounding!(variant::Rn, PtxFloatRounding::NearestEven);
cvt_half_cross_rounding!(variant::Rz, PtxFloatRounding::Zero);
cvt_half_cross_rounding!(variant::Rm, PtxFloatRounding::NegativeInfinity);
cvt_half_cross_rounding!(variant::Rp, PtxFloatRounding::PositiveInfinity);

/// One `cvt.frnd2{.relu}{.satfinite}.{f16,bf16}.f32` specialization.
///
/// These share `PackedMode` with the packed narrow-float rows above: the ISA
/// gives `.f16` and `.f16x2` one grammar line, so the `.satfinite` / `.relu`
/// product is the same and is spelled in the same order.  `Scale` stays at its
/// `NoScale` default because no scalar narrowing form takes a scale operand.
macro_rules! cvt_narrow_modifiers {
    ($dst_marker:ty, $dst:ty, $convert:path, $round_marker:ty, $mode:expr) => {
        exact_cvt_variant!(
            variant::F32 => $dst_marker,
            variant::PackedMode<$round_marker, variant::NoSatFinite, variant::NoRelu>,
            f32 => $dst,
            |value: f32| $convert(value, $mode, false, false)
        );
        exact_cvt_variant!(
            variant::F32 => $dst_marker,
            variant::PackedMode<$round_marker, variant::SatFinite, variant::NoRelu>,
            f32 => $dst,
            |value: f32| $convert(value, $mode, false, true)
        );
        exact_cvt_variant!(
            variant::F32 => $dst_marker,
            variant::PackedMode<$round_marker, variant::NoSatFinite, variant::Relu>,
            f32 => $dst,
            |value: f32| $convert(value, $mode, true, false)
        );
        exact_cvt_variant!(
            variant::F32 => $dst_marker,
            variant::PackedMode<$round_marker, variant::SatFinite, variant::Relu>,
            f32 => $dst,
            |value: f32| $convert(value, $mode, true, true)
        );
    };
}

cvt_narrow_modifiers!(
    variant::F16,
    u16,
    crate::scalar::ptx_cvt_f32_to_f16,
    variant::Rn,
    PtxFloatRounding::NearestEven
);
cvt_narrow_modifiers!(
    variant::F16,
    u16,
    crate::scalar::ptx_cvt_f32_to_f16,
    variant::Rz,
    PtxFloatRounding::Zero
);
cvt_narrow_modifiers!(
    variant::Bf16,
    u16,
    crate::scalar::ptx_cvt_f32_to_bf16,
    variant::Rn,
    PtxFloatRounding::NearestEven
);
cvt_narrow_modifiers!(
    variant::Bf16,
    u16,
    crate::scalar::ptx_cvt_f32_to_bf16,
    variant::Rz,
    PtxFloatRounding::Zero
);
cvt_narrow_modifiers!(
    variant::Tf32,
    u32,
    crate::scalar::ptx_cvt_f32_to_tf32,
    variant::Rn,
    PtxFloatRounding::NearestEven
);
cvt_narrow_modifiers!(
    variant::Tf32,
    u32,
    crate::scalar::ptx_cvt_f32_to_tf32,
    variant::Rz,
    PtxFloatRounding::Zero
);

// `cvt.rna{.satfinite}.tf32.f32` has no `.relu` spelling, so only the
// `NoRelu` half of the axis exists for it.
exact_cvt_variant!(
    variant::F32 => variant::Tf32,
    variant::PackedMode<variant::Rna, variant::NoSatFinite, variant::NoRelu>,
    f32 => u32,
    |value: f32| crate::scalar::ptx_cvt_f32_to_tf32(value, PtxFloatRounding::NearestAway, false, false)
);
exact_cvt_variant!(
    variant::F32 => variant::Tf32,
    variant::PackedMode<variant::Rna, variant::SatFinite, variant::NoRelu>,
    f32 => u32,
    |value: f32| crate::scalar::ptx_cvt_f32_to_tf32(value, PtxFloatRounding::NearestAway, false, true)
);

macro_rules! special_register_variant {
    ($marker:ty, $scalar:ty, $value:expr) => {
        register_variant! {
            [impl] mov_spec, $marker,
            () => R<$scalar>;
            |context, _site, ()| {
                let context = context.into_inner();
                Ok(R::from_fn(|lane| $value(&context, lane)))
            }
        }
    };
}

special_register_variant!(
    variant::LaneId,
    u32,
    |_context: &crate::WarpContext, lane| lane as u32
);
special_register_variant!(
    variant::WarpIdInCta,
    u32,
    |context: &crate::WarpContext, _lane| { context.warp_id_in_cta() as u32 }
);
special_register_variant!(
    variant::CtaId,
    u32,
    |context: &crate::WarpContext, _lane| { context.global_cta_id() as u32 }
);
special_register_variant!(
    variant::ClusterId,
    u32,
    |context: &crate::WarpContext, _lane| { context.cluster_id() as u32 }
);
special_register_variant!(
    variant::Clock64,
    u64,
    |_context: &crate::WarpContext, _lane| 0_u64
);

register_variant! {
    [impl] fns_spec, variant::B32,
    (R<u32>, R<u32>, R<i32>) => R<u32>;
    |context, _site, (mask, base, offset)| {
        let result = crate::runtime::warp_ops::warp_fns_b32(
            context.active_mask().into_inner(),
            mask.inner(),
            base.inner(),
            offset.inner(),
        )?;
        Ok(R::from_inner(result))
    }
}

include!("packed_mul.rs");

include!("packed_fma.rs");

include!("prmt.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::v2::LaneMask;
    use crate::{LaunchTopology, NumSimMode, PhysicalMemory};
    use std::sync::Arc;

    use crate::runtime::{run_kernel_engine_launch, ExecutionPolicy, LaunchSelection};

    #[test]
    fn lop3_truth_table_uses_ptx_a_b_c_row_order() {
        let a = 0xf0f0_0f0f_u32;
        let b = 0xcccc_3333_u32;
        let c = 0xaaaa_5555_u32;
        assert_eq!(lop3_word::<0x00>(a, b, c), 0);
        assert_eq!(lop3_word::<0xff>(a, b, c), u32::MAX);
        assert_eq!(lop3_word::<0x80>(a, b, c), a & b & c);
        assert_eq!(lop3_word::<0xfe>(a, b, c), a | b | c);
        assert_eq!(lop3_word::<0x40>(a, b, c), a & b & !c);
        assert_eq!(lop3_word::<0x1a>(a, b, c), ((a & b) | c) ^ a);
    }

    #[test]
    fn packed_comparison_reuses_scalar_signedness_semantics() {
        let lhs = 0x807f_ff00;
        let rhs = 0x0080_00ff;
        assert_eq!(
            compare_packed_words::<variant::I8, variant::Lt>(lhs, rhs),
            0xff00_ff00
        );
        assert_eq!(
            compare_packed_words::<variant::U8, variant::Lt>(lhs, rhs),
            0x00ff_00ff
        );
    }

    #[test]
    fn ptx_94_bit_helpers_cover_control_boundaries() {
        assert_eq!(bit_field_extract(0xffff_ffff, 0, 0, 32, true), 0);
        assert_eq!(
            bit_field_extract(0x8000_0000, 31, 1, 32, true),
            u32::MAX as u64
        );
        assert_eq!(
            bit_field_extract(0x8000_0000, 32, 1, 32, true),
            u32::MAX as u64
        );
        assert_eq!(bit_field_extract(1, 255, 255, 32, false), 0);

        assert_eq!(bit_field_insert(0, 0xdead_beef, 32, 1, 32), 0xdead_beef);
        assert_eq!(bit_field_insert(1, 0, 255, 255, 32), 0);

        assert_eq!(most_significant_non_sign_bit(0, 32, false), u32::MAX);
        assert_eq!(
            most_significant_non_sign_bit(u32::MAX as u64, 32, true),
            u32::MAX
        );
        assert_eq!(most_significant_non_sign_bit(0xffff_fffe, 32, true), 0);
        assert_eq!(most_significant_non_sign_bit(0x8000_0000, 32, false), 31);

        assert_eq!(bit_mask(32, 1, true), 0);
        assert_eq!(bit_mask(31, u32::MAX, true), 0x8000_0000);
        assert_eq!(bit_mask(33, 34, false), 0x0000_0006);
        assert_eq!(bit_mask(0, 32, false), 0);

        let low = 0x0123_4567;
        let high = 0x89ab_cdef;
        assert_eq!(funnel_shift(low, high, 0, true, true), high);
        assert_eq!(funnel_shift(low, high, 0, false, true), low);
        assert_eq!(funnel_shift(low, high, 32, true, true), low);
        assert_eq!(funnel_shift(low, high, 32, false, true), high);
        assert_eq!(funnel_shift(low, high, 32, true, false), high);
        assert_eq!(funnel_shift(low, high, 32, false, false), low);

        assert_eq!(zero_extend_32(u32::MAX, 0, true), 0);
        assert_eq!(zero_extend_32(u32::MAX, 33, true), u32::MAX);
        assert_eq!(zero_extend_32(u32::MAX, 33, false), 1);
        assert_eq!(sign_extend_32(0b1000, 4, true), -8);
        assert_eq!(sign_extend_32(-1, 0, false), 0);
        assert_eq!(sign_extend_32(i32::MIN, 32, true), i32::MIN);
        assert_eq!(sign_extend_32(0b10, 33, false), 0);
    }

    #[test]
    fn spdecompress_matches_the_ptx_low_bit_first_example() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let output = spdecompress::<variant::SpDecompress<8, 4, 2, 4, 2>>(
                    ExecCtx::from_inner(warp.context()),
                    SiteId::new(1),
                    (vec![R::splat(0x3121)], vec![R::splat(0x0605_03f9)]),
                )?;
                assert_eq!(output.len(), 2);
                assert_eq!(output[0][0], 0x0003_f900);
                assert_eq!(output[1][0], 0x0600_0500);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn spdecompress_supports_the_128_register_output_boundary() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let output = spdecompress::<variant::SpDecompress<8, 4, 4, 16, 32>>(
                    ExecCtx::from_inner(warp.context()),
                    SiteId::new(2),
                    (vec![R::splat(0); 16], vec![R::splat(0xa5a5_a5a5); 32]),
                )?;
                assert_eq!(output.len(), 128);
                assert_eq!(output[0][0], 0xa5);
                assert_eq!(output[124][0], 0xa5);
                assert_eq!(output[127][0], 0);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn spdecompress_rejects_out_of_range_metadata_only_for_active_lanes() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context())
                    .with_active_mask(LaneMask::from_bits(1_u32));
                let inactive_invalid = R::from_fn(|lane| if lane == 0 { 0 } else { 2 });
                spdecompress::<variant::SpDecompress<8, 4, 1, 2, 2>>(
                    context,
                    SiteId::new(3),
                    (vec![inactive_invalid], vec![R::splat(0x2211)]),
                )?;

                let error = spdecompress::<variant::SpDecompress<8, 4, 1, 2, 2>>(
                    context,
                    SiteId::new(4),
                    (vec![R::splat(2)], vec![R::splat(0x2211)]),
                )
                .unwrap_err();
                assert!(error
                    .to_string()
                    .contains("metadata index 2 is outside 0..2"));
                assert!(error.to_string().contains("site 4 lane 0"));
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn static_register_variants_execute_one_lane_wise_instruction() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());
                let lhs = R::from_fn(|lane| i32::MAX - lane as i32);
                let rhs = R::splat(2_i32);
                let sum = add::<variant::I32>(context, SiteId::new(1), (lhs, rhs))?;
                assert_eq!(sum[0], i32::MIN + 1);

                let bits = xor::<variant::B32>(
                    context,
                    SiteId::new(2),
                    (R::splat(0xf0_u32), R::splat(0x33_u32)),
                )?;
                assert_eq!(bits[0], 0xc3);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn ptx_integer_arithmetic_variants_match_instruction_width_semantics() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());

                assert_eq!(
                    div::<variant::I32>(context, SiteId::new(50), (R::splat(-7), R::splat(3)),)?[0],
                    -2
                );
                assert_eq!(
                    rem::<variant::I32>(context, SiteId::new(51), (R::splat(-7), R::splat(-3)),)?
                        [0],
                    -1
                );
                assert_eq!(
                    sad::<variant::I32>(
                        context,
                        SiteId::new(52),
                        (R::splat(i32::MIN), R::splat(i32::MAX), R::splat(7)),
                    )?[0],
                    6
                );
                assert_eq!(
                    mul::<variant::MulWide<variant::I16>>(
                        context,
                        SiteId::new(53),
                        (R::splat(-30_000), R::splat(2)),
                    )?[0],
                    -60_000
                );
                assert_eq!(
                    mad::<variant::MadWide<variant::U16>>(
                        context,
                        SiteId::new(54),
                        (R::splat(u16::MAX), R::splat(u16::MAX), R::splat(u32::MAX)),
                    )?[0],
                    4_294_836_224
                );

                let signed_low = mul24::<variant::Mul24<variant::I32, variant::Lo>>(
                    context,
                    SiteId::new(55),
                    (R::splat(0x0080_0000), R::splat(2)),
                )?;
                let signed_high = mul24::<variant::Mul24<variant::I32, variant::Hi>>(
                    context,
                    SiteId::new(56),
                    (R::splat(0x0080_0000), R::splat(2)),
                )?;
                assert_eq!(signed_low[0], -16_777_216);
                assert_eq!(signed_high[0], -256);
                assert_eq!(
                    mul24::<variant::Mul24<variant::U32, variant::Hi>>(
                        context,
                        SiteId::new(57),
                        (R::splat(0xffff_ffff), R::splat(2)),
                    )?[0],
                    0x1ff
                );
                assert_eq!(
                    mad24::<variant::Mad24<variant::I32, variant::Hi, variant::Sat>>(
                        context,
                        SiteId::new(58),
                        (
                            R::splat(0x007f_ffff),
                            R::splat(0x007f_ffff),
                            R::splat(i32::MAX),
                        ),
                    )?[0],
                    i32::MAX
                );

                assert_eq!(
                    dp4a::<variant::Dp4a<variant::U32, variant::U32>>(
                        context,
                        SiteId::new(59),
                        (R::splat(0x0403_0201), R::splat(0x0403_0201), R::splat(5),),
                    )?[0],
                    35
                );
                assert_eq!(
                    dp4a::<variant::Dp4a<variant::U32, variant::I32>>(
                        context,
                        SiteId::new(60),
                        (
                            R::splat(0x0403_0201),
                            R::splat(0xfc03_fe01_u32 as i32),
                            R::splat(5),
                        ),
                    )?[0],
                    -5
                );
                assert_eq!(
                    dp2a::<variant::Dp2a<variant::I32, variant::I32, variant::Lo>>(
                        context,
                        SiteId::new(61),
                        (
                            R::splat(0xfffd_0002_u32 as i32),
                            R::splat(0x07fa_0504),
                            R::splat(10),
                        ),
                    )?[0],
                    3
                );
                assert_eq!(
                    dp2a::<variant::Dp2a<variant::I32, variant::I32, variant::Hi>>(
                        context,
                        SiteId::new(62),
                        (
                            R::splat(0xfffd_0002_u32 as i32),
                            R::splat(0x07fa_0504),
                            R::splat(10),
                        ),
                    )?[0],
                    -23
                );
                assert_eq!(
                    neg::<variant::I16>(context, SiteId::new(63), R::splat(i16::MIN))?[0],
                    i16::MIN
                );
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn integer_division_fail_closed_checks_only_active_lanes() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context())
                    .with_active_mask(LaneMask::from_bits(1_u32));
                let inactive_zero = R::from_fn(|lane| if lane == 0 { 2_i16 } else { 0 });
                let quotient =
                    div::<variant::I16>(context, SiteId::new(70), (R::splat(8), inactive_zero))?;
                assert_eq!(quotient[0], 4);

                let zero =
                    div::<variant::U32>(context, SiteId::new(71), (R::splat(1), R::splat(0)))
                        .unwrap_err();
                assert!(zero.to_string().contains("site 71 lane 0: 1 / 0"));

                let overflow = rem::<variant::I64>(
                    context,
                    SiteId::new(72),
                    (R::splat(i64::MIN), R::splat(-1)),
                )
                .unwrap_err();
                assert!(overflow
                    .to_string()
                    .contains("site 72 lane 0: -9223372036854775808 % -1"));

                let machine_specific =
                    rem::<variant::I32>(context, SiteId::new(73), (R::splat(-5), R::splat(2)))
                        .unwrap_err();
                assert!(machine_specific
                    .to_string()
                    .contains("machine-specific negative operand at site 73 lane 0"));
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn bit_size_shifts_are_logical_and_clamp_out_of_range_amounts() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());

                // Zero fill, not sign fill: the same bit pattern read as i32
                // would keep its leading ones.
                let logical32 = shr::<variant::B32>(
                    context,
                    SiteId::new(1),
                    (R::splat(0x8000_0000_u32), R::splat(4_u32)),
                )?;
                assert_eq!(logical32[0], 0x0800_0000);
                let arithmetic32 = shr::<variant::I32>(
                    context,
                    SiteId::new(2),
                    (R::splat(i32::MIN), R::splat(4_u32)),
                )?;
                assert_eq!(arithmetic32[0], -0x0800_0000);

                let logical64 = shr::<variant::B64>(
                    context,
                    SiteId::new(3),
                    (R::splat(0x8000_0000_0000_0000_u64), R::splat(4_u32)),
                )?;
                assert_eq!(logical64[0], 0x0800_0000_0000_0000);

                // Clamped, not wrapped: a modulo-N amount would return the
                // operand unchanged for a shift of exactly N.
                let clamped_b32 = shr::<variant::B32>(
                    context,
                    SiteId::new(4),
                    (R::splat(0xdead_beef_u32), R::splat(32_u32)),
                )?;
                assert_eq!(clamped_b32[0], 0);
                let clamped_b64 = shr::<variant::B64>(
                    context,
                    SiteId::new(5),
                    (R::splat(u64::MAX), R::splat(100_u32)),
                )?;
                assert_eq!(clamped_b64[0], 0);
                let clamped_shl = shl::<variant::B32>(
                    context,
                    SiteId::new(6),
                    (R::splat(0xdead_beef_u32), R::splat(32_u32)),
                )?;
                assert_eq!(clamped_shl[0], 0);
                let clamped_signed = shr::<variant::I32>(
                    context,
                    SiteId::new(7),
                    (R::splat(-3_i32), R::splat(64_u32)),
                )?;
                assert_eq!(clamped_signed[0], -1);
                let clamped_signed_positive = shr::<variant::I64>(
                    context,
                    SiteId::new(8),
                    (R::splat(3_i64), R::splat(64_u32)),
                )?;
                assert_eq!(clamped_signed_positive[0], 0);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn nymph_low_precision_forms_have_ptx_shaped_numeric_semantics() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());

                // 1 + 2^-11 is exactly halfway between two f16 values; RN-even
                // returns 1.  The bf16 case is the analogous 1 + 2^-8 tie.
                let f16_sum = add::<variant::F16Rn>(
                    context,
                    SiteId::new(20),
                    (R::splat(0x3c00_u16), R::splat(0x1000_u16)),
                )?;
                assert_eq!(f16_sum[0], 0x3c00);
                let bf16_sum = add::<variant::Bf16Rn>(
                    context,
                    SiteId::new(21),
                    (R::splat(0x3f80_u16), R::splat(0x3b80_u16)),
                )?;
                assert_eq!(bf16_sum[0], 0x3f80);

                let f16_fma = fma::<variant::F16Rn>(
                    context,
                    SiteId::new(22),
                    (
                        R::splat(0x3e00_u16),
                        R::splat(0x4000_u16),
                        R::splat(0x3800_u16),
                    ),
                )?;
                assert_eq!(f16_fma[0], 0x4300); // 1.5 * 2 + 0.5 = 3.5

                let f16 = cvt::<variant::Cvt<variant::F32, variant::F16, variant::Rn>>(
                    context,
                    SiteId::new(23),
                    R::splat(1.000_488_281_25_f32),
                )?;
                assert_eq!(f16[0], 0x3c00);
                let widened =
                    cvt::<variant::Cvt<variant::F16, variant::F32>>(context, SiteId::new(24), f16)?;
                assert_eq!(widened[0].to_bits(), 1.0_f32.to_bits());

                let packed = cvt::<variant::Cvt<variant::F32, variant::F16x2, variant::Rn>>(
                    context,
                    SiteId::new(27),
                    (R::splat(2.0_f32), R::splat(1.0_f32)),
                )?;
                assert_eq!(packed[0], 0x4000_3c00);

                let ordered = setp::<variant::Setp<variant::F32, variant::Lt>>(
                    context,
                    SiteId::new(25),
                    (R::splat(-0.0_f32), R::splat(1.0_f32)),
                )?;
                assert!(ordered[0]);
                let ordered_nan = setp::<variant::Setp<variant::Bf16, variant::Ne>>(
                    context,
                    SiteId::new(26),
                    (R::splat(0x7fc1_u16), R::splat(0x3f80_u16)),
                )?;
                assert!(!ordered_nan[0]);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn ptx_94_mixed_vector_arithmetic_is_bit_exact_and_lane_ordered() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());
                let f16_one_negative_one = R::splat(0xbc00_3c00_u32);
                let half_ulp = R::splat(crate::scalar::make_float2(
                    2.0_f32.powi(-24),
                    -2.0_f32.powi(-24),
                ));

                // The two packed lanes deliberately have opposite signs.  A
                // halfway addition distinguishes all four f32 rounding modes.
                let rn = add::<variant::MixedF32x2<variant::F16x2, variant::Rn>>(
                    context,
                    SiteId::new(30),
                    (f16_one_negative_one.clone(), half_ulp.clone()),
                )?;
                let rz = add::<variant::MixedF32x2<variant::F16x2, variant::Rz>>(
                    context,
                    SiteId::new(31),
                    (f16_one_negative_one.clone(), half_ulp.clone()),
                )?;
                let rm = add::<variant::MixedF32x2<variant::F16x2, variant::Rm>>(
                    context,
                    SiteId::new(32),
                    (f16_one_negative_one.clone(), half_ulp.clone()),
                )?;
                let rp = add::<variant::MixedF32x2<variant::F16x2, variant::Rp>>(
                    context,
                    SiteId::new(33),
                    (f16_one_negative_one, half_ulp),
                )?;
                assert_eq!(rn[0], 0xbf80_0000_3f80_0000);
                assert_eq!(rz[0], 0xbf80_0000_3f80_0000);
                assert_eq!(rm[0], 0xbf80_0001_3f80_0000);
                assert_eq!(rp[0], 0xbf80_0000_3f80_0001);

                let exact_bf16_sub = sub::<variant::MixedF32x2<variant::Bf16x2, variant::Rn>>(
                    context,
                    SiteId::new(34),
                    (
                        R::splat(0x4000_3f80),
                        R::splat(crate::scalar::make_float2(0.5, 1.0)),
                    ),
                )?;
                assert_eq!(exact_bf16_sub[0], crate::scalar::make_float2(0.5, 1.0));

                // 3 * (1 + 2^-23) - 3 is 3 * 2^-23.  Rounding the product
                // before the subtraction would instead produce 4 * 2^-23.
                let fused = fma::<variant::MixedF32x2<variant::F16x2, variant::Rn>>(
                    context,
                    SiteId::new(35),
                    (
                        R::splat(0x4200_4200),
                        R::splat(crate::scalar::make_float2(
                            1.0 + f32::EPSILON,
                            1.0 + f32::EPSILON,
                        )),
                        R::splat(crate::scalar::make_float2(-3.0, -3.0)),
                    ),
                )?;
                assert_eq!(fused[0], 0x34c0_0000_34c0_0000);
                let fused_bf16 = fma::<variant::MixedF32x2<variant::Bf16x2, variant::Rz>>(
                    context,
                    SiteId::new(36),
                    (
                        R::splat(0x4040_4040),
                        R::splat(crate::scalar::make_float2(2.0, 4.0)),
                        R::splat(crate::scalar::make_float2(1.0, 1.0)),
                    ),
                )?;
                assert_eq!(fused_bf16[0], crate::scalar::make_float2(7.0, 13.0));

                let wide_lhs = R::splat(crate::scalar::make_float2(4.0, 8.0));
                let wide_rhs = R::splat(crate::scalar::make_float2(0.5, 1.0));
                let down_add_f16 = add::<variant::MixedF32x2Down<variant::F16x2>>(
                    context,
                    SiteId::new(37),
                    (wide_lhs.clone(), wide_rhs.clone()),
                )?;
                let down_add_bf16 = add::<variant::MixedF32x2Down<variant::Bf16x2>>(
                    context,
                    SiteId::new(38),
                    (wide_lhs.clone(), wide_rhs.clone()),
                )?;
                let down_sub_f16 = sub::<variant::MixedF32x2Down<variant::F16x2>>(
                    context,
                    SiteId::new(39),
                    (wide_lhs.clone(), wide_rhs.clone()),
                )?;
                let down_sub_bf16 = sub::<variant::MixedF32x2Down<variant::Bf16x2>>(
                    context,
                    SiteId::new(40),
                    (wide_lhs.clone(), wide_rhs.clone()),
                )?;
                let down_mul_f16 = mul::<variant::MixedF32x2Down<variant::F16x2>>(
                    context,
                    SiteId::new(41),
                    (wide_lhs.clone(), wide_rhs.clone()),
                )?;
                let down_mul_bf16 = mul::<variant::MixedF32x2Down<variant::Bf16x2>>(
                    context,
                    SiteId::new(42),
                    (wide_lhs, wide_rhs),
                )?;
                assert_eq!(down_add_f16[0], 0x4880_4480);
                assert_eq!(down_add_bf16[0], 0x4110_4090);
                assert_eq!(down_sub_f16[0], 0x4700_4300);
                assert_eq!(down_sub_bf16[0], 0x40e0_4060);
                assert_eq!(down_mul_f16[0], 0x4800_4000);
                assert_eq!(down_mul_bf16[0], 0x4100_4000);

                // `.rz.ftz.f16x2` flushes low-precision subnormal results to
                // signed zero.  The bf16 destination spelling has no `.ftz`
                // and preserves both signs of its smallest subnormal.
                let f16_subnormal = add::<variant::MixedF32x2Down<variant::F16x2>>(
                    context,
                    SiteId::new(43),
                    (
                        R::splat(crate::scalar::make_float2(
                            2.0_f32.powi(-24),
                            -2.0_f32.powi(-24),
                        )),
                        R::splat(0),
                    ),
                )?;
                assert_eq!(f16_subnormal[0], 0x8000_0000);
                let bf16_subnormal = add::<variant::MixedF32x2Down<variant::Bf16x2>>(
                    context,
                    SiteId::new(44),
                    (R::splat(0x8001_0000_0001_0000), R::splat(0)),
                )?;
                assert_eq!(bf16_subnormal[0], 0x8001_0001);

                // The unlike-format multiply first converts the right input
                // to the destination format, then performs low-precision RN
                // multiplication.  These cases differ from one wide product
                // followed by one narrowing conversion.
                let bf16_from_f16 = mul::<variant::MixedLowMul<variant::Bf16x2, variant::F16x2>>(
                    context,
                    SiteId::new(45),
                    (R::splat(0x3dcd_3dcd), R::splat(0x3003_3003)),
                )?;
                assert_eq!(bf16_from_f16[0], 0x3c4d_3c4d);
                let f16_from_bf16 = mul::<variant::MixedLowMul<variant::F16x2, variant::Bf16x2>>(
                    context,
                    SiteId::new(46),
                    (R::splat(0x0e6b_0e6b), R::splat(0x4d08_4d08)),
                )?;
                assert_eq!(f16_from_bf16[0], 0x7c00_7c00);
                Ok(())
            },
        )
        .unwrap();
    }

    /// The tile lowering's bf16 narrowing marker and the PTX-spelled
    /// `cvt.frnd2` family must not disagree on NaN.
    ///
    /// `encode_bf16` is the function `Cvt<F32, Bf16, Rn>` binds; it
    /// canonicalizes to `0x7fffffff` before narrowing, so it lands on the same
    /// `0x7fff` that `cvt.rn.bf16.f32` was measured to produce on a B200 for
    /// every NaN encoding.
    #[test]
    fn ptx94_packed_cvt_specializations_preserve_bits_and_operand_order() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        run_kernel_engine_launch::<NumSimMode, _, _>(
            PhysicalMemory::new(topology),
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |warp| async move {
                let context = ExecCtx::from_inner(warp.context());
                type E2m3ScaledRz = variant::Cvt<
                    variant::F32,
                    variant::E2m3x2,
                    variant::PackedMode<
                        variant::Rz,
                        variant::SatFinite,
                        variant::NoRelu,
                        variant::ScaledUe8m0N1,
                    >,
                >;
                let e2m3 = cvt::<E2m3ScaledRz>(
                    context,
                    SiteId::new(30),
                    (R::splat(2.125_f32), R::splat(2.375_f32), R::splat(128_u8)),
                )?;
                // Divide by two first: RZ maps 1.0625 -> 1.0 (0x08) and
                // 1.1875 -> 1.125 (0x09), in padded upper/lower byte fields.
                assert_eq!(e2m3[0], 0x0809);

                type Ue5PackedScaledRz = variant::Cvt<
                    variant::F16x2,
                    variant::Ue5m3x2,
                    variant::PackedMode<
                        variant::Rz,
                        variant::SatFinite,
                        variant::NoRelu,
                        variant::ScaledUe8m0N1,
                    >,
                >;
                let ue5 = cvt::<Ue5PackedScaledRz>(
                    context,
                    SiteId::new(31),
                    (R::splat(0x4040_40c0_u32), R::splat(128_u8)),
                )?;
                assert_eq!(ue5[0], 0x7879);

                type Ue5ToF16 = variant::Cvt<
                    variant::Ue5m3x2,
                    variant::F16x2,
                    variant::PackedMode<variant::Rn, variant::NoSatFinite, variant::NoRelu>,
                >;
                let widened = cvt::<Ue5ToF16>(context, SiteId::new(32), R::splat(0x78fe_u16))?;
                assert_eq!(widened[0], 0x3c00_7c00);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn bf16_narrowing_markers_agree_on_every_nan_encoding() {
        for bits in [
            0x7fc0_0000_u32, // quiet NaN
            0xffc0_0000,     // negative quiet NaN
            0x7f80_0001,     // signalling NaN
            0xff80_0001,     // negative signalling NaN
            0x7fff_ffff,     // all-ones payload
        ] {
            let value = f32::from_bits(bits);
            assert_eq!(encode_bf16(value), 0x7fff, "encode_bf16({bits:#010x})");
            assert_eq!(
                encode_bf16(value),
                crate::scalar::ptx_cvt_f32_to_bf16(
                    value,
                    crate::scalar::PtxFloatRounding::NearestEven,
                    false,
                    false,
                ),
                "bf16 narrowing markers disagree on {bits:#010x}"
            );
        }
    }
}
