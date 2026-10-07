// The B32 implementation unpacks two 16-bit values from a 32-bit carrier.

register_variant! {
    [impl] mov_unpack_spec, variant::B32,
    R<u32> => (R<u16>, R<u16>);
    |_context, _site, source| {
        let low = source.clone().map(|_lane, value| value as u16);
        let high = source.map(|_lane, value| (value >> 16) as u16);
        Ok((low, high))
    }
}
