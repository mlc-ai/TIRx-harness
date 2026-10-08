// Packed half-precision multiply register variants.

fn packed_mul(lhs: u32, rhs: u32, scalar: fn(u16, u16) -> u16) -> u32 {
    u32::from(scalar(lhs as u16, rhs as u16))
        | (u32::from(scalar((lhs >> 16) as u16, (rhs >> 16) as u16)) << 16)
}

macro_rules! packed_mul_variant {
    ($marker:ty, $scalar:path) => {
        register_variant! {
            [impl] mul_spec, $marker,
            (R<u32>, R<u32>) => R<u32>;
            |_context, _site, (lhs, rhs)| {
                Ok(R::from_fn(|lane| packed_mul(lhs[lane], rhs[lane], $scalar)))
            }
        }
    };
}

packed_mul_variant!(variant::F16x2, crate::scalar::mul_f16_bits_rn);
packed_mul_variant!(variant::Bf16x2, crate::scalar::mul_bf16_bits_rn);
