//! Engine implementation of v2 register-fragment matrix instructions.
//!
//! The array lengths below are operand widths of one PTX instruction.  They
//! are not a generic repetition facility: issuing several instructions stays
//! an ordinary frontend loop.

use std::marker::PhantomData;

use super::instruction::{instruction_variant, sync_instruction};
use super::reg::{variant as reg_variant, FloatRound};
use super::transport::engine;
use super::{Address, EngineError, ExecCtx, LaneMask, Register, SiteId, R};
use crate::runtime::matrix_ops::{
    raw_mma_sp_sync, raw_mma_sync_f16_f16, raw_mma_sync_f16_f8, raw_mma_sync_f32_b16,
    raw_mma_sync_f32_f8, raw_mma_sync_f32_tf32, raw_mma_sync_f64, raw_mma_sync_m8n8k4_f16,
    raw_mma_sync_packed_integer, MatrixAccumulatorType, MatrixB16Type, MatrixBitOp, MatrixF8Type,
    MatrixLayout, MatrixPackedIntType, MatrixSparseAccumulatorType, MatrixSparseOperandType,
};
use crate::runtime::PhysicalPtr;

sync_instruction!(mma_sync_spec, MmaSyncVariant, mma_sync);
sync_instruction!(mma_sp_sync_spec, MmaSpSyncVariant, mma_sp_sync);

/// Compile-time variants of `mma.sync` and `mma.sp.sync`.
pub mod variant {
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Fp16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Bf16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Tf32;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Fp32;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct I8;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct U8;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct I4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct U4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct E4M3;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct E5M2;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Row;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Col;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Xor;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct And;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct NoC;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct WithC;

    /// `mma.sync.aligned.m16n8k{8,16}.row.col.f32.{f16,bf16}`.
    pub struct F32B16<const K: usize, T, C>(PhantomData<fn() -> (T, C)>);
    /// `mma.sync.aligned.m16n8k{8,16}.row.col.f16.f16`.
    pub struct F16<const K: usize, C>(PhantomData<fn() -> C>);
    /// `mma.sync.aligned.m16n8k{4,8}.row.col.f32.tf32`.
    pub struct F32Tf32<const K: usize, C>(PhantomData<fn() -> C>);
    /// `mma.sync.aligned.m16n8k{16,32}.row.col.{f16,f32}.f8`.
    pub struct F8<const K: usize, Acc, A, B, C>(PhantomData<fn() -> (Acc, A, B, C)>);
    /// Dense signed/unsigned 8-bit integer forms.
    pub struct PackedI8<const M: usize, const K: usize, A, B, C, const SATURATE: bool>(
        PhantomData<fn() -> (A, B, C)>,
    );
    /// Dense signed/unsigned 4-bit integer forms.
    pub struct PackedI4<const M: usize, const K: usize, A, B, C, const SATURATE: bool>(
        PhantomData<fn() -> (A, B, C)>,
    );
    /// Dense binary forms; `OP` is `Xor` or `And`.
    pub struct Binary<const M: usize, const K: usize, Op, C>(PhantomData<fn() -> (Op, C)>);
    /// `mma.sync` f64 forms. Legal `(M,K)` pairs are sealed by implementations.
    pub struct F64<const M: usize, const K: usize, C, Round = reg_variant::Rn>(
        PhantomData<fn() -> (C, Round)>,
    );

    pub struct M8N8K4F16F16<ALayout, BLayout, C>(PhantomData<fn() -> (ALayout, BLayout, C)>);
    pub struct M8N8K4F32F16<ALayout, BLayout, C>(PhantomData<fn() -> (ALayout, BLayout, C)>);
    pub struct M8N8K4F32F32<ALayout, BLayout, C>(PhantomData<fn() -> (ALayout, BLayout, C)>);

    /// Sparse f16 forms; `Acc` is `Fp16` or `Fp32`.
    pub struct SparseF16<const K: usize, Acc, const ORDERED: bool = false>(
        PhantomData<fn() -> Acc>,
    );
    /// Sparse bf16-to-f32 forms.
    pub struct SparseBf16<const K: usize, const ORDERED: bool = false>;
    /// Sparse TF32-to-f32 forms.
    pub struct SparseTf32<const K: usize, const ORDERED: bool = false>;
    /// Sparse signed/unsigned 8-bit integer forms.
    pub struct SparseI8<const K: usize, A, B, const SATURATE: bool, const ORDERED: bool = false>(
        PhantomData<fn() -> (A, B)>,
    );
    /// Sparse signed/unsigned 4-bit integer forms.
    pub struct SparseI4<const K: usize, A, B, const SATURATE: bool, const ORDERED: bool = false>(
        PhantomData<fn() -> (A, B)>,
    );
    /// Sparse f8-to-f32 form.
    pub struct SparseF8<A, B, const ORDERED: bool = false>(PhantomData<fn() -> (A, B)>);
}

fn physical<const N: usize>(fragment: &[Address<Register>; N]) -> [PhysicalPtr; N] {
    std::array::from_fn(|index| fragment[index].inner().clone())
}

trait B16Type {
    const VALUE: MatrixB16Type;
}

impl B16Type for variant::Fp16 {
    const VALUE: MatrixB16Type = MatrixB16Type::Fp16;
}

impl B16Type for variant::Bf16 {
    const VALUE: MatrixB16Type = MatrixB16Type::Bf16;
}

trait F8Type {
    const VALUE: MatrixF8Type;
}

impl F8Type for variant::E4M3 {
    const VALUE: MatrixF8Type = MatrixF8Type::E4M3;
}

impl F8Type for variant::E5M2 {
    const VALUE: MatrixF8Type = MatrixF8Type::E5M2;
}

trait PackedI8Type {
    const VALUE: MatrixPackedIntType;
    const SPARSE: MatrixSparseOperandType;
}

impl PackedI8Type for variant::I8 {
    const VALUE: MatrixPackedIntType = MatrixPackedIntType::I8;
    const SPARSE: MatrixSparseOperandType = MatrixSparseOperandType::I8;
}

impl PackedI8Type for variant::U8 {
    const VALUE: MatrixPackedIntType = MatrixPackedIntType::U8;
    const SPARSE: MatrixSparseOperandType = MatrixSparseOperandType::U8;
}

trait PackedI4Type {
    const VALUE: MatrixPackedIntType;
    const SPARSE: MatrixSparseOperandType;
}

impl PackedI4Type for variant::I4 {
    const VALUE: MatrixPackedIntType = MatrixPackedIntType::I4;
    const SPARSE: MatrixSparseOperandType = MatrixSparseOperandType::I4;
}

impl PackedI4Type for variant::U4 {
    const VALUE: MatrixPackedIntType = MatrixPackedIntType::U4;
    const SPARSE: MatrixSparseOperandType = MatrixSparseOperandType::U4;
}

trait BitOperation {
    const VALUE: MatrixBitOp;
}

impl BitOperation for variant::Xor {
    const VALUE: MatrixBitOp = MatrixBitOp::Xor;
}

impl BitOperation for variant::And {
    const VALUE: MatrixBitOp = MatrixBitOp::And;
}

trait Layout {
    const VALUE: MatrixLayout;
}

impl Layout for variant::Row {
    const VALUE: MatrixLayout = MatrixLayout::Row;
}

impl Layout for variant::Col {
    const VALUE: MatrixLayout = MatrixLayout::Col;
}

macro_rules! impl_dense {
    (
        impl ($($generic:tt)*) $variant:ty;
        counts=($d:literal, $a:literal, $b:literal); word=$word:ty;
        call=|$physical:ident, $context:ident, $d_ptrs:ident, $a_ptrs:ident, $b_ptrs:ident, $c_ptrs:ident| $call:expr;
        where $($bounds:tt)*
    ) => {
        instruction_variant! {
            [impl $($generic)*] mma_sync_spec, $variant
            where [
                $($bounds)*
            ],
            (
                [Address<Register>; $d],
                [R<$word>; $a],
                [R<$word>; $b],
            ) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                _site: SiteId,
                (d, a, b): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let $d_ptrs = physical(&d);
                let $a_ptrs = a.map(R::into_inner);
                let $b_ptrs = b.map(R::into_inner);
                let $c_ptrs = None;
                let $context = context.into_inner();
                let $physical = engine(warp).kernel().physical();
                $call.map_err(Into::into)
            }
        }
    };
    (
        with_c impl ($($generic:tt)*) $variant:ty;
        counts=($d:literal, $a:literal, $b:literal, $c:literal); word=$word:ty;
        call=|$physical:ident, $context:ident, $d_ptrs:ident, $a_ptrs:ident, $b_ptrs:ident, $c_ptrs:ident| $call:expr;
        where $($bounds:tt)*
    ) => {
        instruction_variant! {
            [impl $($generic)*] mma_sync_spec, $variant
            where [
                $($bounds)*
            ],
            (
                [Address<Register>; $d],
                [R<$word>; $a],
                [R<$word>; $b],
                [R<$word>; $c],
            ) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                _site: SiteId,
                (d, a, b, c): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let $d_ptrs = physical(&d);
                let $a_ptrs = a.map(R::into_inner);
                let $b_ptrs = b.map(R::into_inner);
                let c_storage = c.map(R::into_inner);
                let $c_ptrs = Some(&c_storage[..]);
                let $context = context.into_inner();
                let $physical = engine(warp).kernel().physical();
                $call.map_err(Into::into)
            }
        }
    };
}

macro_rules! f32_b16 {
    ($k:literal, $a:literal, $b:literal, $ty:ty) => {
        impl_dense!(
            impl () variant::F32B16<$k, $ty, variant::NoC>;
            counts=(4, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f32_b16(
                physical, &context, &d, &a, &b, c, $k, <$ty as B16Type>::VALUE
            );
            where
        );
        impl_dense!(
            with_c impl () variant::F32B16<$k, $ty, variant::WithC>;
            counts=(4, $a, $b, 4); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f32_b16(
                physical, &context, &d, &a, &b, c, $k, <$ty as B16Type>::VALUE
            );
            where
        );
    };
}

f32_b16!(8, 2, 1, variant::Fp16);
f32_b16!(8, 2, 1, variant::Bf16);
f32_b16!(16, 4, 2, variant::Fp16);
f32_b16!(16, 4, 2, variant::Bf16);

macro_rules! f16 {
    ($k:literal, $a:literal, $b:literal) => {
        impl_dense!(
            impl () variant::F16<$k, variant::NoC>;
            counts=(2, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f16_f16(
                physical, &context, &d, &a, &b, c, $k
            );
            where
        );
        impl_dense!(
            with_c impl () variant::F16<$k, variant::WithC>;
            counts=(2, $a, $b, 2); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f16_f16(
                physical, &context, &d, &a, &b, c, $k
            );
            where
        );
    };
}

f16!(8, 2, 1);
f16!(16, 4, 2);

macro_rules! f32_tf32 {
    ($k:literal, $a:literal, $b:literal) => {
        impl_dense!(
            impl () variant::F32Tf32<$k, variant::NoC>;
            counts=(4, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f32_tf32(
                physical, &context, &d, &a, &b, c, $k
            );
            where
        );
        impl_dense!(
            with_c impl () variant::F32Tf32<$k, variant::WithC>;
            counts=(4, $a, $b, 4); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_f32_tf32(
                physical, &context, &d, &a, &b, c, $k
            );
            where
        );
    };
}

f32_tf32!(4, 2, 1);
f32_tf32!(8, 4, 2);

macro_rules! f8 {
    ($k:literal, $a:literal, $b:literal, $acc:ty, $d:literal, $function:ident, $aty:ty, $bty:ty) => {
        impl_dense!(
            impl () variant::F8<$k, $acc, $aty, $bty, variant::NoC>;
            counts=($d, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| $function(
                physical, &context, &d, &a, &b, c, $k,
                <$aty as F8Type>::VALUE, <$bty as F8Type>::VALUE
            );
            where
        );
        impl_dense!(
            with_c impl () variant::F8<$k, $acc, $aty, $bty, variant::WithC>;
            counts=($d, $a, $b, $d); word=u32;
            call=|physical, context, d, a, b, c| $function(
                physical, &context, &d, &a, &b, c, $k,
                <$aty as F8Type>::VALUE, <$bty as F8Type>::VALUE
            );
            where
        );
    };
}

macro_rules! f8_types {
    ($k:literal, $a:literal, $b:literal, $acc:ty, $d:literal, $function:ident) => {
        f8!(
            $k,
            $a,
            $b,
            $acc,
            $d,
            $function,
            variant::E4M3,
            variant::E4M3
        );
        f8!(
            $k,
            $a,
            $b,
            $acc,
            $d,
            $function,
            variant::E4M3,
            variant::E5M2
        );
        f8!(
            $k,
            $a,
            $b,
            $acc,
            $d,
            $function,
            variant::E5M2,
            variant::E4M3
        );
        f8!(
            $k,
            $a,
            $b,
            $acc,
            $d,
            $function,
            variant::E5M2,
            variant::E5M2
        );
    };
}

f8_types!(16, 2, 1, variant::Fp16, 2, raw_mma_sync_f16_f8);
f8_types!(32, 4, 2, variant::Fp16, 2, raw_mma_sync_f16_f8);
f8_types!(16, 2, 1, variant::Fp32, 4, raw_mma_sync_f32_f8);
f8_types!(32, 4, 2, variant::Fp32, 4, raw_mma_sync_f32_f8);

macro_rules! packed {
    ($family:ident, $trait:ident, $m:literal, $k:literal, $d:literal, $a:literal, $b:literal, $aty:ty, $bty:ty, $sat:literal) => {
        impl_dense!(
            impl () variant::$family<$m, $k, $aty, $bty, variant::NoC, $sat>;
            counts=($d, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_packed_integer(
                physical, &context, &d, &a, &b, c, $m, $k,
                <$aty as $trait>::VALUE, <$bty as $trait>::VALUE, $sat, None
            );
            where
        );
        impl_dense!(
            with_c impl () variant::$family<$m, $k, $aty, $bty, variant::WithC, $sat>;
            counts=($d, $a, $b, $d); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_packed_integer(
                physical, &context, &d, &a, &b, c, $m, $k,
                <$aty as $trait>::VALUE, <$bty as $trait>::VALUE, $sat, None
            );
            where
        );
    };
}

macro_rules! packed_types {
    ($family:ident, $trait:ident, $signed:ty, $unsigned:ty, $m:literal, $k:literal, $d:literal, $a:literal, $b:literal) => {
        packed!($family, $trait, $m, $k, $d, $a, $b, $signed, $signed, false);
        packed!($family, $trait, $m, $k, $d, $a, $b, $signed, $signed, true);
        packed!($family, $trait, $m, $k, $d, $a, $b, $signed, $unsigned, false);
        packed!($family, $trait, $m, $k, $d, $a, $b, $signed, $unsigned, true);
        packed!($family, $trait, $m, $k, $d, $a, $b, $unsigned, $signed, false);
        packed!($family, $trait, $m, $k, $d, $a, $b, $unsigned, $signed, true);
        packed!($family, $trait, $m, $k, $d, $a, $b, $unsigned, $unsigned, false);
        packed!($family, $trait, $m, $k, $d, $a, $b, $unsigned, $unsigned, true);
    };
}

packed_types!(
    PackedI8,
    PackedI8Type,
    variant::I8,
    variant::U8,
    8,
    16,
    2,
    1,
    1
);
packed_types!(
    PackedI8,
    PackedI8Type,
    variant::I8,
    variant::U8,
    16,
    16,
    4,
    2,
    1
);
packed_types!(
    PackedI8,
    PackedI8Type,
    variant::I8,
    variant::U8,
    16,
    32,
    4,
    4,
    2
);
packed_types!(
    PackedI4,
    PackedI4Type,
    variant::I4,
    variant::U4,
    8,
    32,
    2,
    1,
    1
);
packed_types!(
    PackedI4,
    PackedI4Type,
    variant::I4,
    variant::U4,
    16,
    32,
    4,
    2,
    1
);
packed_types!(
    PackedI4,
    PackedI4Type,
    variant::I4,
    variant::U4,
    16,
    64,
    4,
    4,
    2
);

macro_rules! binary {
    ($m:literal, $k:literal, $d:literal, $a:literal, $b:literal, $op:ty) => {
        impl_dense!(
            impl () variant::Binary<$m, $k, $op, variant::NoC>;
            counts=($d, $a, $b); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_packed_integer(
                physical, &context, &d, &a, &b, c, $m, $k,
                MatrixPackedIntType::B1, MatrixPackedIntType::B1, false,
                Some(<$op as BitOperation>::VALUE)
            );
            where
        );
        impl_dense!(
            with_c impl () variant::Binary<$m, $k, $op, variant::WithC>;
            counts=($d, $a, $b, $d); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_packed_integer(
                physical, &context, &d, &a, &b, c, $m, $k,
                MatrixPackedIntType::B1, MatrixPackedIntType::B1, false,
                Some(<$op as BitOperation>::VALUE)
            );
            where
        );
    };
}

macro_rules! binary_ops {
    ($m:literal, $k:literal, $d:literal, $a:literal, $b:literal) => {
        binary!($m, $k, $d, $a, $b, variant::Xor);
        binary!($m, $k, $d, $a, $b, variant::And);
    };
}

binary_ops!(8, 128, 2, 1, 1);
binary_ops!(16, 128, 4, 2, 1);
binary_ops!(16, 256, 4, 4, 2);

macro_rules! f64_form {
    ($m:literal, $k:literal, $d:literal, $a:literal, $b:literal) => {
        impl_dense!(
            impl (<Round>) variant::F64<$m, $k, variant::NoC, Round>;
            counts=($d, $a, $b); word=f64;
            call=|physical, context, d, a, b, c| raw_mma_sync_f64(
                physical, &context, &d, &a, &b, c, $m, $k, Round::MODE
            );
            where Round: FloatRound,
        );
        impl_dense!(
            with_c impl (<Round>) variant::F64<$m, $k, variant::WithC, Round>;
            counts=($d, $a, $b, $d); word=f64;
            call=|physical, context, d, a, b, c| raw_mma_sync_f64(
                physical, &context, &d, &a, &b, c, $m, $k, Round::MODE
            );
            where Round: FloatRound,
        );
    };
}

f64_form!(8, 4, 2, 1, 1);
f64_form!(16, 4, 4, 2, 1);
f64_form!(16, 8, 4, 4, 2);
f64_form!(16, 16, 4, 8, 4);

macro_rules! m8n8k4 {
    ($name:ident, $d:literal, $c:literal, $d_dtype:ident, $c_dtype:ident, $al:ty, $bl:ty) => {
        impl_dense!(
            impl () variant::$name<$al, $bl, variant::NoC>;
            counts=($d, 2, 2); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_m8n8k4_f16(
                physical, &context, &d, &a, &b, c,
                <$al as Layout>::VALUE, <$bl as Layout>::VALUE,
                MatrixAccumulatorType::$d_dtype, MatrixAccumulatorType::$c_dtype
            );
            where
        );
        impl_dense!(
            with_c impl () variant::$name<$al, $bl, variant::WithC>;
            counts=($d, 2, 2, $c); word=u32;
            call=|physical, context, d, a, b, c| raw_mma_sync_m8n8k4_f16(
                physical, &context, &d, &a, &b, c,
                <$al as Layout>::VALUE, <$bl as Layout>::VALUE,
                MatrixAccumulatorType::$d_dtype, MatrixAccumulatorType::$c_dtype
            );
            where
        );
    };
}

macro_rules! m8n8k4_layouts {
    ($name:ident, $d:literal, $c:literal, $d_dtype:ident, $c_dtype:ident) => {
        m8n8k4!(
            $name,
            $d,
            $c,
            $d_dtype,
            $c_dtype,
            variant::Row,
            variant::Row
        );
        m8n8k4!(
            $name,
            $d,
            $c,
            $d_dtype,
            $c_dtype,
            variant::Row,
            variant::Col
        );
        m8n8k4!(
            $name,
            $d,
            $c,
            $d_dtype,
            $c_dtype,
            variant::Col,
            variant::Row
        );
        m8n8k4!(
            $name,
            $d,
            $c,
            $d_dtype,
            $c_dtype,
            variant::Col,
            variant::Col
        );
    };
}

m8n8k4_layouts!(M8N8K4F16F16, 4, 4, Fp16, Fp16);
m8n8k4_layouts!(M8N8K4F32F16, 8, 4, Fp32, Fp16);
m8n8k4_layouts!(M8N8K4F32F32, 8, 8, Fp32, Fp32);

macro_rules! sparse_variant {
    (
        impl ($($generic:tt)*) $variant:ty;
        counts=($a:literal, $b:literal, $c:literal);
        values=($k:literal, $a_type:expr, $b_type:expr, $acc_type:expr, $sat:expr, $ordered:expr);
        where $($bounds:tt)*
    ) => {
        impl $($generic)* $variant
        where
            $($bounds)*
        {
            /// Lanes whose metadata registers this exact MMA form consumes.
            pub fn metadata_source_mask(selector: usize) -> Result<LaneMask, EngineError> {
                let mask = crate::runtime::matrix_ops::sparse_metadata_source_mask(
                    $k, $a_type, selector,
                )?;
                Ok(LaneMask::from_bits(mask.bits()))
            }
        }

        instruction_variant! {
            [impl $($generic)*] mma_sp_sync_spec, $variant
            where [
                $($bounds)*
            ],
            (
                [Address<Register>; $c],
                [R<u32>; $a],
                [R<u32>; $b],
                [R<u32>; $c],
                R<u32>,
                usize,
            ) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                _site: SiteId,
                (d, a, b, accumulator, metadata, selector): Self::Args,
            ) -> Result<Self::Output, EngineError> {
                let d = physical(&d);
                let a = a.map(R::into_inner);
                let b = b.map(R::into_inner);
                let accumulator = accumulator.map(R::into_inner);
                let context = context.into_inner();
                raw_mma_sp_sync(
                    engine(warp).kernel().physical(),
                    &context,
                    &d,
                    &a,
                    &b,
                    &accumulator,
                    metadata.inner(),
                    selector,
                    $k,
                    $a_type,
                    $b_type,
                    $acc_type,
                    $sat,
                    $ordered,
                )
                .map_err(Into::into)
            }
        }
    };
}

trait SparseAccumulator {
    const VALUE: MatrixSparseAccumulatorType;
}

impl SparseAccumulator for variant::Fp16 {
    const VALUE: MatrixSparseAccumulatorType = MatrixSparseAccumulatorType::Fp16;
}

impl SparseAccumulator for variant::Fp32 {
    const VALUE: MatrixSparseAccumulatorType = MatrixSparseAccumulatorType::Fp32;
}

macro_rules! sparse_f16 {
    ($k:literal, $count:literal, $acc:ty, $c:literal) => {
        sparse_variant!(
            impl (<const ORDERED: bool>) variant::SparseF16<$k, $acc, ORDERED>;
            counts=($count, $count, $c);
            values=(
                $k,
                MatrixSparseOperandType::Fp16,
                MatrixSparseOperandType::Fp16,
                <$acc as SparseAccumulator>::VALUE,
                false,
                ORDERED
            );
            where
        );
    };
}

sparse_f16!(16, 2, variant::Fp16, 2);
sparse_f16!(16, 2, variant::Fp32, 4);
sparse_f16!(32, 4, variant::Fp16, 2);
sparse_f16!(32, 4, variant::Fp32, 4);

macro_rules! sparse_bf16 {
    ($k:literal, $count:literal) => {
        sparse_variant!(
            impl (<const ORDERED: bool>) variant::SparseBf16<$k, ORDERED>;
            counts=($count, $count, 4);
            values=(
                $k,
                MatrixSparseOperandType::Bf16,
                MatrixSparseOperandType::Bf16,
                MatrixSparseAccumulatorType::Fp32,
                false,
                ORDERED
            );
            where
        );
    };
}

sparse_bf16!(16, 2);
sparse_bf16!(32, 4);

macro_rules! sparse_tf32 {
    ($k:literal, $count:literal) => {
        sparse_variant!(
            impl (<const ORDERED: bool>) variant::SparseTf32<$k, ORDERED>;
            counts=($count, $count, 4);
            values=(
                $k,
                MatrixSparseOperandType::Tf32,
                MatrixSparseOperandType::Tf32,
                MatrixSparseAccumulatorType::Fp32,
                false,
                ORDERED
            );
            where
        );
    };
}

sparse_tf32!(8, 2);
sparse_tf32!(16, 4);

macro_rules! sparse_integer {
    ($family:ident, $trait:ident, $k:literal, $count:literal, $aty:ty, $bty:ty, $sat:literal) => {
        sparse_variant!(
            impl (<const ORDERED: bool>) variant::$family<$k, $aty, $bty, $sat, ORDERED>;
            counts=($count, $count, 4);
            values=(
                $k,
                <$aty as $trait>::SPARSE,
                <$bty as $trait>::SPARSE,
                MatrixSparseAccumulatorType::I32,
                $sat,
                ORDERED
            );
            where
        );
    };
}

macro_rules! sparse_integer_types {
    ($family:ident, $trait:ident, $signed:ty, $unsigned:ty, $k:literal, $count:literal) => {
        sparse_integer!($family, $trait, $k, $count, $signed, $signed, false);
        sparse_integer!($family, $trait, $k, $count, $signed, $signed, true);
        sparse_integer!($family, $trait, $k, $count, $signed, $unsigned, false);
        sparse_integer!($family, $trait, $k, $count, $signed, $unsigned, true);
        sparse_integer!($family, $trait, $k, $count, $unsigned, $signed, false);
        sparse_integer!($family, $trait, $k, $count, $unsigned, $signed, true);
        sparse_integer!($family, $trait, $k, $count, $unsigned, $unsigned, false);
        sparse_integer!($family, $trait, $k, $count, $unsigned, $unsigned, true);
    };
}

sparse_integer_types!(SparseI8, PackedI8Type, variant::I8, variant::U8, 32, 2);
sparse_integer_types!(SparseI8, PackedI8Type, variant::I8, variant::U8, 64, 4);
sparse_integer_types!(SparseI4, PackedI4Type, variant::I4, variant::U4, 64, 2);
sparse_integer_types!(SparseI4, PackedI4Type, variant::I4, variant::U4, 128, 4);

macro_rules! sparse_f8 {
    ($aty:ty, $bty:ty) => {
        sparse_variant!(
    impl (<const ORDERED: bool>) variant::SparseF8<$aty, $bty, ORDERED>;
    counts=(4, 4, 4);
    values=(
        64,
        match <$aty as F8Type>::VALUE {
            MatrixF8Type::E4M3 => MatrixSparseOperandType::E4M3,
            MatrixF8Type::E5M2 => MatrixSparseOperandType::E5M2,
        },
        match <$bty as F8Type>::VALUE {
            MatrixF8Type::E4M3 => MatrixSparseOperandType::E4M3,
            MatrixF8Type::E5M2 => MatrixSparseOperandType::E5M2,
        },
        MatrixSparseAccumulatorType::Fp32,
        false,
        ORDERED
    );
    where
        );
    };
}

sparse_f8!(variant::E4M3, variant::E4M3);
sparse_f8!(variant::E4M3, variant::E5M2);
sparse_f8!(variant::E5M2, variant::E4M3);
sparse_f8!(variant::E5M2, variant::E5M2);

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dense_variant<V: MmaSyncVariant>() {}
    fn assert_sparse_variant<V: MmaSpSyncVariant>() {}

    #[test]
    fn supported_forms_are_statically_callable() {
        assert_dense_variant::<variant::F32B16<8, variant::Fp16, variant::WithC>>();
        assert_dense_variant::<variant::F32B16<16, variant::Bf16, variant::NoC>>();
        assert_dense_variant::<
            variant::PackedI8<16, 32, variant::I8, variant::U8, variant::WithC, true>,
        >();
        assert_dense_variant::<variant::Binary<16, 256, variant::Xor, variant::WithC>>();
        assert_dense_variant::<variant::M8N8K4F32F16<variant::Col, variant::Row, variant::NoC>>();
        assert_sparse_variant::<variant::SparseI4<128, variant::I4, variant::U4, true>>();
        assert_sparse_variant::<variant::SparseF8<variant::E4M3, variant::E5M2>>();
    }
}
