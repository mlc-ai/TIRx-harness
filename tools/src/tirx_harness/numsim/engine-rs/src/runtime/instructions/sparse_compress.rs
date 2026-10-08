// Shared numerical primitive for register and TMEM sparse compression.
// Metadata is packed low-bit first, in source-index order. PTX permits any
// choice among tied candidates; the model deterministically picks lower indices.
pub(crate) fn compress_sparse_vector(
    data: &[u32], elem_bits: usize, index_bits: usize, descriptor: u32,
) -> Result<(Vec<u32>, Vec<u32>), EngineError> {
    if !matches!(elem_bits, 8 | 16) || !matches!(index_bits, 2 | 4)
        || data.is_empty() || data.len() % 2 != 0 || data.len() > 128
    {
        return Err(EngineError::message("spcompress invalid vector shape"));
    }
    let dtype = (descriptor >> 2) & 7;
    if descriptor >> 5 != 0 || dtype > 5 || (elem_bits == 16 && dtype > 1) {
        return Err(EngineError::message("spcompress invalid sparsity descriptor"));
    }
    let operation = descriptor & 3;
    let groups = data.len() * 32 / elem_bits / 4;
    let mut metadata = vec![0; (groups * 2 * index_bits).div_ceil(32)];
    let mut compressed = vec![0; data.len() / 2];
    let decode = |bits: u32| -> Result<f32, EngineError> {
        Ok(match (elem_bits, dtype) {
            (16, 0) => crate::fp16_bits_to_f32(bits as u16),
            (16, 1) => crate::bf16_bits_to_f32(bits as u16),
            (8, 0) => bits as f32,
            (8, 1) => bits as u8 as i8 as f32,
            (8, 2..=5) => crate::narrow_float_bits_to_f32_checked(bits as u8, match dtype {
                2 => crate::FLOAT8_E5M2,
                3 => crate::FLOAT8_E4M3,
                4 => crate::FLOAT6_E3M2,
                _ => crate::FLOAT6_E2M3,
            }).ok_or_else(|| EngineError::message("spcompress invalid narrow-float encoding"))?,
            _ => unreachable!(),
        })
    };
    let elem_mask = (1_u32 << elem_bits) - 1;
    for group in 0..groups {
        let mut bits = [0_u32; 4];
        let mut values = [0_f32; 4];
        for index in 0..4 {
            let bit = (group * 4 + index) * elem_bits;
            bits[index] = (data[bit / 32] >> (bit % 32)) & elem_mask;
            values[index] = decode(bits[index])?;
            if operation & 1 != 0 { values[index] = values[index].abs(); }
        }
        let indices = sparse_pair_indices(values, operation & 2 == 0);
        for (out, &index) in indices.iter().enumerate() {
            let elem = group * 2 + out;
            let mb = elem * index_bits;
            metadata[mb / 32] |= (index as u32) << (mb % 32);
            let cb = elem * elem_bits;
            compressed[cb / 32] |= bits[index] << (cb % 32);
        }
    }
    Ok((metadata, compressed))
}

pub(crate) fn sparse_pair_indices(values: [f32; 4], maximum: bool) -> [usize; 2] {
    let mut indices = [0_usize, 1, 2, 3];
    indices.sort_by(|&a, &b| {
            match (values[a].is_nan(), values[b].is_nan()) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (true, true) => a.cmp(&b),
                _ => {
                    let order = values[a].total_cmp(&values[b]);
                    (if maximum { order.reverse() } else { order }).then(a.cmp(&b))
                }
            }
        });
    indices[..2].sort_unstable();
    [indices[0], indices[1]]
}

register_variant! {
    [impl<const E: usize, const I: usize, const N: usize>] spcompress_spec, variant::SpCompress<E, I, N>,
    (Vec<R<u32>>, R<u32>) => Vec<R<u32>>;
    |context, _site, (data, descriptor)| {
        if !matches!(E, 8 | 16) || !matches!(I, 2 | 4)
            || !matches!(N, 1 | 2 | 4 | 8 | 16 | 32 | 64) || data.len() != 2 * N {
            return Err(EngineError::message("spcompress invalid vector shape"));
        }
        let mut result = vec![R::splat(0_u32); (N * I).div_ceil(E) + N];
        for lane in context.active_mask() {
            let input: Vec<_> = data.iter().map(|word| word[lane]).collect();
            let (metadata, compressed) = compress_sparse_vector(&input, E, I, descriptor[lane])?;
            for (destination, word) in result.iter_mut().zip(metadata.into_iter().chain(compressed)) {
                destination[lane] = word;
            }
        }
        Ok(result)
    }
}
