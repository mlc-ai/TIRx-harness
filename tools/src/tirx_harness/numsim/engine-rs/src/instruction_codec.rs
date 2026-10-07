//! Engine-private codecs for CUDA-source tokens and instruction descriptors.
//!
//! v2 raw operations accept runtime descriptor/address bits and decode them at
//! the instruction boundary.  Typed operations consume mapped views directly.
//! These pure bit transformations therefore are not part of the v2
//! instruction ABI; generated artifacts reach them through the hidden
//! artifact-support adapter.

pub(crate) const fn tcgen_runtime_instruction_descriptor(bits: u32, sf_id: u32) -> u32 {
    (bits & !0x6000_0030_u32) | sf_id.wrapping_shl(29) | sf_id.wrapping_shl(4)
}

pub(crate) const fn sm100_tma_2sm_mbarrier_address(address: u64) -> u32 {
    (address as u32) & 0xfeff_ffff_u32
}

// GB200/sm_100a device probes show that cluster shared addresses use the high
// byte for the CTA rank and the low 24 bits for the byte address. NumSim uses
// one fixed virtual generic-shared prefix for its integer address model.
pub const SHARED_CTA_RANK_SHIFT: u32 = 24;
pub const SHARED_BYTE_OFFSET_MASK: u32 = 0x00ff_ffff;
pub const DEFAULT_GENERIC_SHARED_PREFIX: u64 = 0x0000_fffe_0000_0000;
pub const GENERIC_ADDRESS_PREFIX_MASK: u64 = 0xffff_ffff_0000_0000;

pub const fn encode_shared_address(byte_offset: u32, cta_rank: u32) -> Option<u32> {
    if byte_offset > SHARED_BYTE_OFFSET_MASK || cta_rank > u8::MAX as u32 {
        return None;
    }
    Some(byte_offset | (cta_rank << SHARED_CTA_RANK_SHIFT))
}

pub const fn shared_address_byte_offset(address: u32) -> u32 {
    address & SHARED_BYTE_OFFSET_MASK
}

pub const fn shared_address_cta_rank(address: u32) -> u32 {
    address >> SHARED_CTA_RANK_SHIFT
}

pub const fn generic_shared_address(address: u32) -> u64 {
    DEFAULT_GENERIC_SHARED_PREFIX | address as u64
}

pub const fn decode_generic_shared_address(address: u64) -> Option<u32> {
    if address & GENERIC_ADDRESS_PREFIX_MASK == DEFAULT_GENERIC_SHARED_PREFIX {
        Some(address as u32)
    } else {
        None
    }
}

pub const fn replace_shared_address_cta_rank(address: u32, cta_rank: u32) -> Option<u32> {
    encode_shared_address(shared_address_byte_offset(address), cta_rank)
}

pub const fn replace_generic_shared_address_cta_rank(
    address: u64,
    cta_rank: u32,
) -> Option<u64> {
    let shared = match decode_generic_shared_address(address) {
        Some(value) => value,
        None => return None,
    };
    match replace_shared_address_cta_rank(shared, cta_rank) {
        Some(value) => Some(generic_shared_address(value)),
        None => None,
    }
}

pub(crate) const fn shared_address(address: u64) -> u32 {
    address as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcgen_runtime_descriptor_replaces_only_the_two_scale_factor_fields() {
        let original = 0xa5a5_5a5a;
        for scale_factor_id in 0..=3 {
            let encoded = tcgen_runtime_instruction_descriptor(original, scale_factor_id);
            assert_eq!(encoded & !0x6000_0030, original & !0x6000_0030);
            assert_eq!((encoded >> 29) & 0x3, scale_factor_id);
            assert_eq!((encoded >> 4) & 0x3, scale_factor_id);
        }
    }

    #[test]
    fn sm100_two_cta_mbarrier_address_clears_the_remote_rank_bit() {
        assert_eq!(sm100_tma_2sm_mbarrier_address(0xffff_ffff), 0xfeff_ffff);
        assert_eq!(sm100_tma_2sm_mbarrier_address(0x0100_0000), 0);
        assert_eq!(sm100_tma_2sm_mbarrier_address(0x1_1234_5678), 0x1234_5678);
    }

    #[test]
    fn shared_address_codec_matches_the_sm100_cluster_layout() {
        let address = encode_shared_address(0x1234, 3).unwrap();
        assert_eq!(address, 0x0300_1234);
        assert_eq!(shared_address_byte_offset(address), 0x1234);
        assert_eq!(shared_address_cta_rank(address), 3);
        assert_eq!(generic_shared_address(address), 0x0000_fffe_0300_1234);
        assert_eq!(decode_generic_shared_address(generic_shared_address(address)), Some(address));
        assert_eq!(decode_generic_shared_address(0x0000_fffd_0300_1234), None);
        assert_eq!(replace_shared_address_cta_rank(address, 0), Some(0x0000_1234));
        assert_eq!(replace_shared_address_cta_rank(address, 7), Some(0x0700_1234));
        assert_eq!(
            replace_generic_shared_address_cta_rank(generic_shared_address(address), 1),
            Some(0x0000_fffe_0100_1234),
        );
        assert_eq!(replace_generic_shared_address_cta_rank(0x1234, 1), None);
    }
}
