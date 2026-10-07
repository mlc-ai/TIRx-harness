// All PRMT modes select bytes from the same register pair. Specialized modes
// derive selectors from c[1:0]; only the generic form can replicate a sign.
register_instruction!(prmt_spec, PrmtVariant, prmt);

macro_rules! prmt_variant {
    ($marker:ty, $select:expr) => {
        register_variant! {
            [impl] prmt_spec, $marker,
            (R<u32>, R<u32>, R<u32>) => R<u32>;
            |_context, _site, (a, b, control)| {
                let select: fn(u32, u32) -> u32 = $select;
                Ok(R::from_fn(|lane| {
                    let source = (u64::from(b[lane]) << 32) | u64::from(a[lane]);
                    let mut output = 0_u32;
                    for byte in 0..4 {
                        let selector = select(control[lane], byte);
                        let value = ((source >> ((selector & 7) * 8)) & 0xff) as u32;
                        let value = if selector & 8 != 0 {
                            if value & 0x80 != 0 { 0xff } else { 0 }
                        } else {
                            value
                        };
                        output |= value << (byte * 8);
                    }
                    output
                }))
            }
        }
    };
}

prmt_variant!(variant::B32, |c, byte| (c >> (4 * byte)) & 15);
prmt_variant!(variant::Prmt<variant::F4e>, |c, byte| (c & 3) + byte);
prmt_variant!(variant::Prmt<variant::B4e>, |c, byte| ((c & 3) + 8 - byte) & 7);
prmt_variant!(variant::Prmt<variant::Rc8>, |c, _byte| c & 3);
prmt_variant!(variant::Prmt<variant::Ecl>, |c, byte| byte.max(c & 3));
prmt_variant!(variant::Prmt<variant::Ecr>, |c, byte| byte.min(c & 3));
prmt_variant!(variant::Prmt<variant::Rc16>, |c, byte| ((c & 1) * 2) + (byte & 1));
