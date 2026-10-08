//! Register-instruction ABI. Implementations live in `runtime/instructions`.

pub use crate::runtime::instructions::reg::{
    abs, add, and, bfe, bfi, bfind, bmsk, brev, clmad, clz, cnot, copysign, cos, createpolicy, cvt,
    cvt_pack, div, dp2a, dp4a, exp2, fma, fns, lg2, lop3, mad, mad24, max, min, mov, mov_pack,
    mov_unpack, mul, mul24, neg, not, or, popc, prmt, rcp, rem, rsqrt, sad, selp, set, setp, shf,
    shl, shr, sin, slct, spcompress, spdecompress, sqrt, sub, szext, tanh, testp, variant, xor,
    AbsVariant, AddVariant, AndVariant, BfeVariant, BfiVariant, BfindVariant, BmskVariant,
    BrevVariant, ClmadVariant, ClzVariant, CnotVariant, CopysignVariant, CosVariant,
    CreatePolicyVariant, CvtPackVariant, CvtVariant, DivVariant, Dp2aVariant, Dp4aVariant,
    Exp2Variant, FmaVariant, FnsVariant, Lg2Variant, Lop3Variant, Mad24Variant, MadVariant,
    MaxVariant, MinVariant, MovPackVariant, MovUnpackVariant, MovVariant, Mul24Variant, MulVariant,
    NegVariant, NotVariant, OrVariant, PopcVariant, PrmtVariant, RcpVariant, RegisterType,
    RemVariant, RsqrtVariant, SadVariant, SelpVariant, SetVariant, SetpVariant, ShfVariant,
    ShlVariant, ShrVariant, SinVariant, SlctVariant, SpCompressVariant, SpDecompressVariant,
    SqrtVariant, SubVariant, SzextVariant, TanhVariant, TestpVariant, XorVariant,
};
