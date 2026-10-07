// Packed half-precision fused multiply-add register variants.

fn packed_fma(lhs: u32, rhs: u32, addend: u32, scalar: fn(u16, u16, u16) -> u16) -> u32 {
    u32::from(scalar(lhs as u16, rhs as u16, addend as u16))
        | (u32::from(scalar(
            (lhs >> 16) as u16,
            (rhs >> 16) as u16,
            (addend >> 16) as u16,
        )) << 16)
}

macro_rules! packed_fma_variant {
    ($marker:ty, $scalar:path) => {
        register_variant! {
            [impl] fma_spec, $marker,
            (R<u32>, R<u32>, R<u32>) => R<u32>;
            |_context, _site, (lhs, rhs, addend)| {
                Ok(R::from_fn(|lane| {
                    packed_fma(lhs[lane], rhs[lane], addend[lane], $scalar)
                }))
            }
        }
    };
}

packed_fma_variant!(variant::F16x2, crate::scalar::fma_f16_bits_rn);
packed_fma_variant!(variant::Bf16x2, crate::scalar::fma_bf16_bits_rn);
