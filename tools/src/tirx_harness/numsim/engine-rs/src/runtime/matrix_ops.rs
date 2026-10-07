use crate::{
    bf16_bits_to_f32, f32_to_fp16_bits, f32_to_tf32, float8_e4m3fn_bits_to_f32, fp16_bits_to_f32,
    narrow_float_bits_to_f32_checked, CtaId, EngineError, PhysicalMemory, RuntimeScalar,
    TcgenLifecycleHub, TmemAccessMode, WarpContext, WarpValue, FLOAT8_E5M2,
};

use super::{write_runtime_bytes, PhysicalPtr, RuntimeBuffer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixB16Type {
    Fp16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayout {
    Row,
    Col,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixAccumulatorType {
    Fp16,
    Fp32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixPackedIntType {
    I8,
    U8,
    I4,
    U4,
    B1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixBitOp {
    Xor,
    And,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixF8Type {
    E4M3,
    E5M2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixSparseOperandType {
    Fp16,
    Bf16,
    Tf32,
    I8,
    U8,
    I4,
    U4,
    E4M3,
    E5M2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixSparseAccumulatorType {
    Fp16,
    Fp32,
    I32,
}

fn decode_b16(dtype: MatrixB16Type, bits: u16) -> f32 {
    match dtype {
        MatrixB16Type::Fp16 => fp16_bits_to_f32(bits),
        MatrixB16Type::Bf16 => bf16_bits_to_f32(bits),
    }
}

fn decode_f8(dtype: MatrixF8Type, bits: u8) -> f32 {
    match dtype {
        MatrixF8Type::E4M3 => float8_e4m3fn_bits_to_f32(bits),
        // The shared codec reports the NaN encodings as `None`; matrix operands
        // carry them through as a quiet NaN, which is what the hand-written
        // E5M2 decoder this replaced did for all 256 payloads.
        MatrixF8Type::E5M2 => {
            narrow_float_bits_to_f32_checked(bits, FLOAT8_E5M2).unwrap_or(f32::NAN)
        }
    }
}

fn matrix_packed_int_bits(dtype: MatrixPackedIntType) -> usize {
    match dtype {
        MatrixPackedIntType::I8 | MatrixPackedIntType::U8 => 8,
        MatrixPackedIntType::I4 | MatrixPackedIntType::U4 => 4,
        MatrixPackedIntType::B1 => 1,
    }
}

fn decode_fragment<T: Copy>(
    registers: &[WarpValue<u32>],
    decode: impl Fn(u32) -> T,
) -> Vec<WarpValue<T>> {
    registers
        .iter()
        .map(|register| WarpValue::from_fn(|lane| decode(register[lane])))
        .collect()
}

fn packed_b16_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixB16Type,
) -> Result<f32, EngineError> {
    let word = registers
        .get(element_slot / 2)
        .ok_or_else(|| EngineError::message("matrix fragment register index is outside operand"))?
        [lane];
    Ok(decode_b16(
        dtype,
        ((word >> (16 * (element_slot % 2))) & 0xffff) as u16,
    ))
}

fn packed_f8_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixF8Type,
) -> Result<f32, EngineError> {
    let word = registers
        .get(element_slot / 4)
        .ok_or_else(|| EngineError::message("matrix fragment register index is outside operand"))?
        [lane];
    Ok(decode_f8(
        dtype,
        ((word >> (8 * (element_slot % 4))) & 0xff) as u8,
    ))
}

fn packed_integer_value(
    registers: &[WarpValue<u32>],
    lane: usize,
    register_slot: usize,
    element_in_register: usize,
    dtype: MatrixPackedIntType,
) -> Result<i32, EngineError> {
    let bits = matrix_packed_int_bits(dtype);
    let word = registers
        .get(register_slot)
        .ok_or_else(|| EngineError::message("matrix fragment register index is outside operand"))?
        [lane];
    let mask = (1_u32 << bits) - 1;
    let raw = (word >> (bits * element_in_register)) & mask;
    Ok(match dtype {
        MatrixPackedIntType::I8 => i32::from(raw as u8 as i8),
        MatrixPackedIntType::U8 => raw as i32,
        MatrixPackedIntType::I4 => {
            if raw & 0x8 != 0 {
                raw as i32 - 16
            } else {
                raw as i32
            }
        }
        MatrixPackedIntType::U4 | MatrixPackedIntType::B1 => raw as i32,
    })
}

fn gather_matrix<T>(
    rows: usize,
    columns: usize,
    mut value_at: impl FnMut(usize, usize) -> Result<T, EngineError>,
) -> Result<Vec<T>, EngineError> {
    let mut values = Vec::with_capacity(rows.saturating_mul(columns));
    for row in 0..rows {
        for column in 0..columns {
            values.push(value_at(row, column)?);
        }
    }
    Ok(values)
}

fn gather_mma_accumulator<T: Copy>(m: usize, registers: &[WarpValue<T>]) -> Vec<T> {
    let mut values = Vec::with_capacity(m * 8);
    for row in 0..m {
        for column in 0..8 {
            let (lane, slot) = mma_output_owner_for_m(m, row, column);
            values.push(registers[slot][lane]);
        }
    }
    values
}

fn gather_packed_f16_accumulator(
    m: usize,
    registers: &[WarpValue<u32>],
) -> Result<Vec<f32>, EngineError> {
    gather_matrix(m, 8, |row, column| {
        let (lane, slot) = mma_output_owner_for_m(m, row, column);
        packed_b16_value(registers, lane, slot, MatrixB16Type::Fp16)
    })
}

fn matrix_output_registers<T: Copy>(
    m: usize,
    register_count: usize,
    zero: T,
    values: &[T],
) -> Result<Vec<WarpValue<T>>, EngineError> {
    if values.len() != m * 8 {
        return Err(EngineError::message(
            "matrix numeric core returned the wrong output shape",
        ));
    }
    let mut registers = vec![WarpValue::splat(zero); register_count];
    for row in 0..m {
        for column in 0..8 {
            let (lane, slot) = mma_output_owner_for_m(m, row, column);
            registers[slot][lane] = values[row * 8 + column];
        }
    }
    Ok(registers)
}

fn store_mma_output<T: RuntimeScalar + Copy>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointers: &[PhysicalPtr],
    m: usize,
    zero: T,
    values: &[T],
) -> Result<(), EngineError> {
    let registers = matrix_output_registers(m, pointers.len(), zero, values)?;
    for (pointer, register) in pointers.iter().zip(&registers) {
        super::store_physical_ptr_warp::<T>(
            physical,
            context,
            pointer,
            register,
            context.active_mask(),
        )?;
    }
    Ok(())
}

fn mma_f32(
    m: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<&[f32]>,
) -> Result<Vec<f32>, EngineError> {
    super::mma_f32_abt_increasing_k(
        m,
        8,
        k,
        a_values,
        b_values,
        input_d.map(|values| (values, 1.0_f32)),
    )
}

fn mma_f64(
    m: usize,
    k: usize,
    a_values: &[f64],
    b_values: &[f64],
    input_d: Option<&[f64]>,
    rounding: crate::scalar::F32RoundingMode,
) -> Result<Vec<f64>, EngineError> {
    let mut output = input_d
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![0.0_f64; m * 8]);
    if rounding != crate::scalar::F32RoundingMode::Nearest {
        for row in 0..m {
            for column in 0..8 {
                let accumulator = &mut output[row * 8 + column];
                for inner in 0..k {
                    *accumulator = crate::scalar::fma_f64(
                        a_values[row * k + inner],
                        b_values[column * k + inner],
                        *accumulator,
                        rounding,
                    );
                }
            }
        }
        return Ok(output);
    }
    let mut b_transposed = vec![0.0_f64; b_values.len()];
    for column in 0..8 {
        for inner in 0..k {
            b_transposed[inner * 8 + column] = b_values[column * k + inner];
        }
    }
    numsim_fp_env::fma_f64_abt_increasing_k(m, 8, k, a_values, &b_transposed, &mut output)
        .map_err(|error| EngineError::message(format!("raw MMA SIMD shape error: {error}")))?;
    Ok(output)
}

fn mma_output_owner(row: usize, col: usize) -> (usize, usize) {
    (
        4 * (row % 8) + (col % 8) / 2,
        4 * (col / 8) + 2 * (row / 8) + col % 2,
    )
}

fn mma_output_owner_for_m(m: usize, row: usize, col: usize) -> (usize, usize) {
    if m == 8 {
        (4 * row + col / 2, col % 2)
    } else {
        mma_output_owner(row, col)
    }
}

fn mma_packed_a_owner(m: usize, row: usize, inner: usize, bits: usize) -> (usize, usize, usize) {
    let elements_per_register = 32 / bits;
    if m == 8 {
        (
            4 * row + inner / elements_per_register,
            0,
            inner % elements_per_register,
        )
    } else {
        let k_group = 4 * elements_per_register;
        (
            4 * (row % 8) + (inner % k_group) / elements_per_register,
            2 * (inner / k_group) + row / 8,
            inner % elements_per_register,
        )
    }
}

fn mma_packed_b_owner(col: usize, inner: usize, bits: usize) -> (usize, usize, usize) {
    let elements_per_register = 32 / bits;
    let k_group = 4 * elements_per_register;
    (
        4 * col + (inner % k_group) / elements_per_register,
        inner / k_group,
        inner % elements_per_register,
    )
}

fn store_packed_f16_registers(
    physical: &PhysicalMemory,
    context: &WarpContext,
    pointers: &[PhysicalPtr],
    values: &[WarpValue<f32>],
) -> Result<(), EngineError> {
    if values.len() != pointers.len() * 2 {
        return Err(EngineError::message(
            "packed f16 matrix output has the wrong logical register count",
        ));
    }
    for (register, pointer) in pointers.iter().enumerate() {
        if pointer.pointee_itemsize() != 2 && pointer.pointee_itemsize() != 4 {
            return Err(EngineError::message(format!(
                "matrix f16 output pointer width {} does not match f16 elements or a b32 backing",
                pointer.pointee_itemsize()
            )));
        }
        for lane in 0..32 {
            let word = u32::from(f32_to_fp16_bits(values[2 * register][lane]))
                | (u32::from(f32_to_fp16_bits(values[2 * register + 1][lane])) << 16);
            let offset = pointer.lane_write_byte_offset(lane, 4)?;
            write_runtime_bytes(
                physical,
                context,
                pointer.buffer(),
                lane,
                offset,
                &word.to_le_bytes(),
            )?;
        }
    }
    Ok(())
}

fn mma_a_b16_owner(row: usize, col: usize, k: usize) -> (usize, usize) {
    if k == 8 {
        (4 * (row % 8) + col / 2, 2 * (row / 8) + col % 2)
    } else {
        (
            4 * (row % 8) + (col % 8) / 2,
            4 * (col / 8) + 2 * (row / 8) + col % 2,
        )
    }
}

fn mma_b_b16_owner(row: usize, col: usize) -> (usize, usize) {
    (4 * col + (row % 8) / 2, 2 * (row / 8) + row % 2)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f32_b16(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    dtype: MatrixB16Type,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m16n8")?;
    if !matches!(k, 8 | 16) || d.len() != 4 || a.len() != k / 4 || b.len() != k / 8 {
        return Err(EngineError::message(format!(
            "mma.sync f32/b16 fragment counts do not match m16n8k{k}"
        )));
    }
    if let Some(c) = c {
        if c.len() != 4 {
            return Err(EngineError::message(
                "mma.sync C fragment must contain four f32 registers",
            ));
        }
    }

    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, slot) = mma_a_b16_owner(row, inner, k);
        packed_b16_value(a, lane, slot, dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_b16_owner(inner, column);
        packed_b16_value(b, lane, slot, dtype)
    })?;
    let c_registers = c.map(|registers| decode_fragment(registers, f32::from_bits));
    let input_d = c_registers
        .as_ref()
        .map(|registers| gather_mma_accumulator(16, registers));
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    store_mma_output(physical, context, d, 16, 0.0_f32, &output)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f32_tf32(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m16n8 tf32")?;
    if !matches!(k, 4 | 8)
        || d.len() != 4
        || a.len() != k / 2
        || b.len() != k / 4
        || c.is_some_and(|value| value.len() != 4)
    {
        return Err(EngineError::message(
            "mma.sync TF32 fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let lane = 4 * (row % 8) + inner % 4;
        let slot = 2 * (inner / 4) + row / 8;
        Ok(f32_to_tf32(f32::from_bits(a[slot][lane])))
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let lane = 4 * column + inner % 4;
        let slot = inner / 4;
        Ok(f32_to_tf32(f32::from_bits(b[slot][lane])))
    })?;
    let c_registers = c.map(|registers| decode_fragment(registers, f32::from_bits));
    let input_d = c_registers
        .as_ref()
        .map(|registers| gather_mma_accumulator(16, registers));
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    store_mma_output(physical, context, d, 16, 0.0_f32, &output)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f16_f16(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m16n8 f16 accumulator")?;
    if !matches!(k, 8 | 16)
        || d.len() != 2
        || a.len() != k / 4
        || b.len() != k / 8
        || c.is_some_and(|value| value.len() != 2)
    {
        return Err(EngineError::message(
            "mma.sync f16-accumulator fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, slot) = mma_a_b16_owner(row, inner, k);
        packed_b16_value(a, lane, slot, MatrixB16Type::Fp16)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, slot) = mma_b_b16_owner(inner, column);
        packed_b16_value(b, lane, slot, MatrixB16Type::Fp16)
    })?;
    let input_d = c
        .map(|registers| gather_packed_f16_accumulator(16, registers))
        .transpose()?;
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    let output_registers = matrix_output_registers(16, d.len() * 2, 0.0_f32, &output)?;
    store_packed_f16_registers(physical, context, d, &output_registers)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_packed_integer(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    m: usize,
    k: usize,
    a_dtype: MatrixPackedIntType,
    b_dtype: MatrixPackedIntType,
    saturate: bool,
    bit_op: Option<MatrixBitOp>,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync packed integer")?;
    if !matches!(m, 8 | 16) {
        return Err(EngineError::message(
            "packed integer MMA requires M=8 or M=16",
        ));
    }
    let a_bits = matrix_packed_int_bits(a_dtype);
    let b_bits = matrix_packed_int_bits(b_dtype);
    if a_bits != b_bits {
        return Err(EngineError::message(
            "packed integer MMA multiplicands must have the same element width",
        ));
    }
    let expected_d = m / 4;
    let expected_a = m * k * a_bits / (32 * 32);
    let expected_b = k * 8 * b_bits / (32 * 32);
    if d.len() != expected_d
        || a.len() != expected_a
        || b.len() != expected_b
        || c.is_some_and(|value| value.len() != expected_d)
    {
        return Err(EngineError::message(
            "packed integer MMA fragment counts are invalid",
        ));
    }
    let is_b1 = a_dtype == MatrixPackedIntType::B1 && b_dtype == MatrixPackedIntType::B1;
    if is_b1 != bit_op.is_some() {
        return Err(EngineError::message(
            "b1 MMA requires a bit operation and integer MMA forbids one",
        ));
    }
    if saturate && (is_b1 || a_bits == 1) {
        return Err(EngineError::message("b1 MMA does not support saturation"));
    }

    let mut a_values = gather_matrix(m, k, |row, inner| {
        let (lane, register, element) = mma_packed_a_owner(m, row, inner, a_bits);
        packed_integer_value(a, lane, register, element, a_dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, register, element) = mma_packed_b_owner(column, inner, b_bits);
        packed_integer_value(b, lane, register, element, b_dtype)
    })?;
    let c_registers = c.map(|registers| decode_fragment(registers, |bits| bits as i32));
    let mut output = c_registers
        .as_ref()
        .map(|registers| {
            gather_mma_accumulator(m, registers)
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0_i64; m * 8]);

    if bit_op == Some(MatrixBitOp::Xor) {
        for row in 0..m {
            let a_row = &mut a_values[row * k..(row + 1) * k];
            let true_count = a_row
                .iter()
                .map(|&value| i64::from(value != 0))
                .sum::<i64>();
            for accumulator in &mut output[row * 8..(row + 1) * 8] {
                *accumulator += true_count;
            }
            for value in a_row {
                *value = 1 - 2 * i32::from(*value != 0);
            }
        }
    }
    numsim_fp_env::multiply_accumulate_i32_abt(m, 8, k, &a_values, &b_values, &mut output)
        .map_err(|error| EngineError::message(format!("raw MMA SIMD shape error: {error}")))?;

    let output = output
        .into_iter()
        .map(|value| {
            if saturate {
                value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            } else {
                value as i32
            }
        })
        .collect::<Vec<_>>();
    store_mma_output(physical, context, d, m, 0_i32, &output)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f32_f8(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    a_dtype: MatrixF8Type,
    b_dtype: MatrixF8Type,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m16n8 f8")?;
    if !matches!(k, 16 | 32)
        || d.len() != 4
        || a.len() != k / 8
        || b.len() != k / 16
        || c.is_some_and(|value| value.len() != 4)
    {
        return Err(EngineError::message(
            "mma.sync f8 fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, register, element) = mma_packed_a_owner(16, row, inner, 8);
        packed_f8_value(&a[register..=register], lane, element, a_dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, register, element) = mma_packed_b_owner(column, inner, 8);
        packed_f8_value(&b[register..=register], lane, element, b_dtype)
    })?;
    let c_registers = c.map(|registers| decode_fragment(registers, f32::from_bits));
    let input_d = c_registers
        .as_ref()
        .map(|registers| gather_mma_accumulator(16, registers));
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    store_mma_output(physical, context, d, 16, 0.0_f32, &output)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f16_f8(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    k: usize,
    a_dtype: MatrixF8Type,
    b_dtype: MatrixF8Type,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m16n8 f8/f16 accumulator")?;
    if !matches!(k, 16 | 32)
        || d.len() != 2
        || a.len() != k / 8
        || b.len() != k / 16
        || c.is_some_and(|value| value.len() != 2)
    {
        return Err(EngineError::message(
            "mma.sync f8/f16-accumulator fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(16, k, |row, inner| {
        let (lane, register, element) = mma_packed_a_owner(16, row, inner, 8);
        packed_f8_value(&a[register..=register], lane, element, a_dtype)
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let (lane, register, element) = mma_packed_b_owner(column, inner, 8);
        packed_f8_value(&b[register..=register], lane, element, b_dtype)
    })?;
    let input_d = c
        .map(|registers| gather_packed_f16_accumulator(16, registers))
        .transpose()?;
    let output = mma_f32(16, k, &a_values, &b_values, input_d.as_deref())?;
    let output_registers = matrix_output_registers(16, d.len() * 2, 0.0_f32, &output)?;
    store_packed_f16_registers(physical, context, d, &output_registers)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_f64(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<f64>],
    b: &[WarpValue<f64>],
    c: Option<&[WarpValue<f64>]>,
    m: usize,
    k: usize,
    rounding: crate::scalar::F32RoundingMode,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync f64")?;
    let valid_shape = (m, k) == (8, 4) || (m == 16 && matches!(k, 4 | 8 | 16));
    let expected_d = m / 4;
    let expected_a = m * k / 32;
    let expected_b = k / 4;
    if !valid_shape
        || d.len() != expected_d
        || a.len() != expected_a
        || b.len() != expected_b
        || c.is_some_and(|value| value.len() != expected_d)
    {
        return Err(EngineError::message(
            "mma.sync f64 fragment counts are invalid",
        ));
    }
    let a_values = gather_matrix(m, k, |row, inner| {
        let (lane, slot) = if m == 8 {
            (4 * row + inner, 0)
        } else {
            (4 * (row % 8) + inner % 4, 2 * (inner / 4) + row / 8)
        };
        Ok(a[slot][lane])
    })?;
    let b_values = gather_matrix(8, k, |column, inner| {
        let lane = 4 * column + inner % 4;
        let slot = inner / 4;
        Ok(b[slot][lane])
    })?;
    let input_d = c.map(|registers| gather_mma_accumulator(m, registers));
    let output = mma_f64(m, k, &a_values, &b_values, input_d.as_deref(), rounding)?;
    store_mma_output(physical, context, d, m, 0.0_f64, &output)
}

fn m8n8k4_f16_a_owner(
    computation: usize,
    row: usize,
    inner: usize,
    layout: MatrixLayout,
) -> (usize, usize) {
    let high = row / 4;
    match layout {
        MatrixLayout::Row => (16 * high + 4 * computation + row % 4, inner),
        MatrixLayout::Col => (16 * high + 4 * computation + inner, row % 4),
    }
}

fn m8n8k4_f16_b_owner(
    computation: usize,
    inner: usize,
    col: usize,
    layout: MatrixLayout,
) -> (usize, usize) {
    let high = col / 4;
    match layout {
        MatrixLayout::Row => (16 * high + 4 * computation + inner, col % 4),
        MatrixLayout::Col => (16 * high + 4 * computation + col % 4, inner),
    }
}

fn m8n8k4_f16_acc_owner(computation: usize, row: usize, col: usize) -> (usize, usize) {
    (16 * (row / 4) + 4 * computation + row % 4, col)
}

fn m8n8k4_f32_acc_owner(
    computation: usize,
    row: usize,
    col: usize,
) -> Result<(usize, usize), EngineError> {
    let high = row / 4;
    for thread in 0..4 {
        for slot in 0..8 {
            let mapped_row = (thread & 1) + (slot & 2) + 4 * high;
            let mapped_col = (slot & 4) + (thread & 2) + (slot & 1);
            if mapped_row == row && mapped_col == col {
                return Ok((16 * high + 4 * computation + thread, slot));
            }
        }
    }
    Err(EngineError::message(
        "m8n8k4 f32 accumulator coordinate has no register owner",
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sync_m8n8k4_f16(
    physical: &PhysicalMemory,
    context: &WarpContext,
    d: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    c: Option<&[WarpValue<u32>]>,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    d_dtype: MatrixAccumulatorType,
    c_dtype: MatrixAccumulatorType,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sync m8n8k4 f16")?;
    let d_count = match d_dtype {
        MatrixAccumulatorType::Fp16 => 4,
        MatrixAccumulatorType::Fp32 => 8,
    };
    let c_count = match c_dtype {
        MatrixAccumulatorType::Fp16 => 4,
        MatrixAccumulatorType::Fp32 => 8,
    };
    if d.len() != d_count
        || a.len() != 2
        || b.len() != 2
        || c.is_some_and(|value| value.len() != c_count)
    {
        return Err(EngineError::message(
            "mma.sync m8n8k4 f16 fragment counts are invalid",
        ));
    }
    let c_f16 = if c_dtype == MatrixAccumulatorType::Fp16 {
        c
    } else {
        None
    };
    let c_f32 = if c_dtype == MatrixAccumulatorType::Fp32 {
        c.map(|registers| decode_fragment(registers, f32::from_bits))
    } else {
        None
    };
    let mut output = vec![WarpValue::splat(0.0_f32); 8];
    for computation in 0..4 {
        let a_values = gather_matrix(8, 4, |row, inner| {
            let (lane, slot) = m8n8k4_f16_a_owner(computation, row, inner, a_layout);
            packed_b16_value(a, lane, slot, MatrixB16Type::Fp16)
        })?;
        let b_values = gather_matrix(8, 4, |column, inner| {
            let (lane, slot) = m8n8k4_f16_b_owner(computation, inner, column, b_layout);
            packed_b16_value(b, lane, slot, MatrixB16Type::Fp16)
        })?;
        let input_d = if let Some(registers) = &c_f16 {
            Some(gather_matrix(8, 8, |row, column| {
                let (lane, slot) = m8n8k4_f16_acc_owner(computation, row, column);
                packed_b16_value(registers, lane, slot, MatrixB16Type::Fp16)
            })?)
        } else if let Some(registers) = &c_f32 {
            Some(gather_matrix(8, 8, |row, column| {
                let (lane, slot) = m8n8k4_f32_acc_owner(computation, row, column)?;
                Ok(registers[slot][lane])
            })?)
        } else {
            None
        };
        let values = mma_f32(8, 4, &a_values, &b_values, input_d.as_deref())?;
        for row in 0..8 {
            for column in 0..8 {
                let (lane, slot) = match d_dtype {
                    MatrixAccumulatorType::Fp16 => m8n8k4_f16_acc_owner(computation, row, column),
                    MatrixAccumulatorType::Fp32 => m8n8k4_f32_acc_owner(computation, row, column)?,
                };
                output[slot][lane] = values[row * 8 + column];
            }
        }
    }
    match d_dtype {
        MatrixAccumulatorType::Fp16 => store_packed_f16_registers(physical, context, d, &output),
        MatrixAccumulatorType::Fp32 => {
            for (pointer, values) in d.iter().zip(output.iter()) {
                super::store_physical_ptr_warp::<f32>(
                    physical,
                    context,
                    pointer,
                    values,
                    context.active_mask(),
                )?;
            }
            Ok(())
        }
    }
}

fn sparse_chunk_width(dtype: MatrixSparseOperandType) -> usize {
    match dtype {
        MatrixSparseOperandType::Tf32 => 2,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => 8,
        _ => 4,
    }
}

fn sparse_stored_per_chunk(dtype: MatrixSparseOperandType) -> usize {
    match dtype {
        MatrixSparseOperandType::Tf32 => 1,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => 4,
        _ => 2,
    }
}

fn sparse_fragment_counts(
    k: usize,
    dtype: MatrixSparseOperandType,
) -> Result<(usize, usize), EngineError> {
    let counts = match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            if matches!(k, 16 | 32) {
                (k / 8, k / 8)
            } else {
                return Err(EngineError::message("sparse b16 MMA requires K=16 or K=32"));
            }
        }
        MatrixSparseOperandType::Tf32 => {
            if matches!(k, 8 | 16) {
                (k / 4, k / 4)
            } else {
                return Err(EngineError::message("sparse TF32 MMA requires K=8 or K=16"));
            }
        }
        MatrixSparseOperandType::I8 | MatrixSparseOperandType::U8 => {
            if matches!(k, 32 | 64) {
                (k / 16, k / 16)
            } else {
                return Err(EngineError::message(
                    "sparse int8 MMA requires K=32 or K=64",
                ));
            }
        }
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => {
            if matches!(k, 64 | 128) {
                (k / 32, k / 32)
            } else {
                return Err(EngineError::message(
                    "sparse int4 MMA requires K=64 or K=128",
                ));
            }
        }
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => {
            if k == 64 {
                (4, 4)
            } else {
                return Err(EngineError::message("sparse FP8 MMA requires K=64"));
            }
        }
    };
    Ok(counts)
}

fn validate_sparse_type_pair(
    a_dtype: MatrixSparseOperandType,
    b_dtype: MatrixSparseOperandType,
    accumulator_dtype: MatrixSparseAccumulatorType,
    saturate: bool,
) -> Result<(), EngineError> {
    let valid = match (a_dtype, b_dtype, accumulator_dtype) {
        (
            MatrixSparseOperandType::Fp16,
            MatrixSparseOperandType::Fp16,
            MatrixSparseAccumulatorType::Fp16 | MatrixSparseAccumulatorType::Fp32,
        ) => true,
        (
            MatrixSparseOperandType::Bf16,
            MatrixSparseOperandType::Bf16,
            MatrixSparseAccumulatorType::Fp32,
        )
        | (
            MatrixSparseOperandType::Tf32,
            MatrixSparseOperandType::Tf32,
            MatrixSparseAccumulatorType::Fp32,
        ) => true,
        (
            MatrixSparseOperandType::I8 | MatrixSparseOperandType::U8,
            MatrixSparseOperandType::I8 | MatrixSparseOperandType::U8,
            MatrixSparseAccumulatorType::I32,
        )
        | (
            MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4,
            MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4,
            MatrixSparseAccumulatorType::I32,
        ) => true,
        (
            MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2,
            MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2,
            MatrixSparseAccumulatorType::Fp32,
        ) => true,
        _ => false,
    };
    if !valid {
        return Err(EngineError::message(
            "sparse MMA operand and accumulator types are incompatible",
        ));
    }
    if saturate
        && !matches!(
            a_dtype,
            MatrixSparseOperandType::I8
                | MatrixSparseOperandType::U8
                | MatrixSparseOperandType::I4
                | MatrixSparseOperandType::U4
        )
    {
        return Err(EngineError::message(
            "sparse MMA saturation requires integer multiplicands",
        ));
    }
    Ok(())
}

fn sparse_a_owner(
    row: usize,
    chunk: usize,
    packed: usize,
    dtype: MatrixSparseOperandType,
) -> (usize, usize) {
    let group = row % 8;
    let row_high = row / 8;
    match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            let block = chunk / 4;
            (4 * group + chunk % 4, 4 * block + 2 * row_high + packed)
        }
        MatrixSparseOperandType::Tf32 => {
            let block = chunk / 4;
            (4 * group + chunk % 4, 2 * block + row_high)
        }
        MatrixSparseOperandType::I8
        | MatrixSparseOperandType::U8
        | MatrixSparseOperandType::E4M3
        | MatrixSparseOperandType::E5M2 => {
            let block = chunk / 8;
            let within = chunk % 8;
            (
                4 * group + within / 2,
                8 * block + 4 * row_high + 2 * (within % 2) + packed,
            )
        }
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4 => {
            let block = chunk / 8;
            let within = chunk % 8;
            (
                4 * group + within / 2,
                16 * block + 8 * row_high + 4 * (within % 2) + packed,
            )
        }
    }
}

fn read_sparse_operand(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixSparseOperandType,
) -> Result<f32, EngineError> {
    match dtype {
        MatrixSparseOperandType::Fp16 => {
            packed_b16_value(registers, lane, element_slot, MatrixB16Type::Fp16)
        }
        MatrixSparseOperandType::Bf16 => {
            packed_b16_value(registers, lane, element_slot, MatrixB16Type::Bf16)
        }
        MatrixSparseOperandType::Tf32 => {
            let register = registers.get(element_slot).ok_or_else(|| {
                EngineError::message("sparse TF32 fragment register index is outside operand")
            })?;
            Ok(f32_to_tf32(f32::from_bits(register[lane])))
        }
        MatrixSparseOperandType::I8
        | MatrixSparseOperandType::U8
        | MatrixSparseOperandType::I4
        | MatrixSparseOperandType::U4 => Err(EngineError::message(
            "internal error: integer sparse operand requested as float",
        )),
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => packed_f8_value(
            registers,
            lane,
            element_slot,
            if dtype == MatrixSparseOperandType::E4M3 {
                MatrixF8Type::E4M3
            } else {
                MatrixF8Type::E5M2
            },
        ),
    }
}

fn sparse_packed_int_type(
    dtype: MatrixSparseOperandType,
) -> Result<MatrixPackedIntType, EngineError> {
    match dtype {
        MatrixSparseOperandType::I8 => Ok(MatrixPackedIntType::I8),
        MatrixSparseOperandType::U8 => Ok(MatrixPackedIntType::U8),
        MatrixSparseOperandType::I4 => Ok(MatrixPackedIntType::I4),
        MatrixSparseOperandType::U4 => Ok(MatrixPackedIntType::U4),
        _ => Err(EngineError::message(
            "internal error: non-integer sparse operand requested as integer",
        )),
    }
}

fn read_sparse_integer_operand(
    registers: &[WarpValue<u32>],
    lane: usize,
    element_slot: usize,
    dtype: MatrixSparseOperandType,
) -> Result<i32, EngineError> {
    let packed_dtype = sparse_packed_int_type(dtype)?;
    let bits = matrix_packed_int_bits(packed_dtype);
    let elements_per_register = 32 / bits;
    packed_integer_value(
        registers,
        lane,
        element_slot / elements_per_register,
        element_slot % elements_per_register,
        packed_dtype,
    )
}

fn sparse_b_float(
    registers: &[WarpValue<u32>],
    inner: usize,
    col: usize,
    dtype: MatrixSparseOperandType,
) -> Result<f32, EngineError> {
    match dtype {
        MatrixSparseOperandType::Fp16 | MatrixSparseOperandType::Bf16 => {
            let (lane, slot) = mma_b_b16_owner(inner, col);
            read_sparse_operand(registers, lane, slot, dtype)
        }
        MatrixSparseOperandType::Tf32 => {
            let lane = 4 * col + inner % 4;
            let slot = inner / 4;
            read_sparse_operand(registers, lane, slot, dtype)
        }
        MatrixSparseOperandType::E4M3 | MatrixSparseOperandType::E5M2 => {
            let (lane, register, element) = mma_packed_b_owner(col, inner, 8);
            packed_f8_value(
                &registers[register..=register],
                lane,
                element,
                if dtype == MatrixSparseOperandType::E4M3 {
                    MatrixF8Type::E4M3
                } else {
                    MatrixF8Type::E5M2
                },
            )
        }
        _ => Err(EngineError::message(
            "internal error: integer sparse B requested as float",
        )),
    }
}

fn sparse_b_integer(
    registers: &[WarpValue<u32>],
    inner: usize,
    col: usize,
    dtype: MatrixSparseOperandType,
) -> Result<i32, EngineError> {
    let packed_dtype = sparse_packed_int_type(dtype)?;
    let bits = matrix_packed_int_bits(packed_dtype);
    let (lane, register, element) = mma_packed_b_owner(col, inner, bits);
    packed_integer_value(registers, lane, register, element, packed_dtype)
}

fn sparse_metadata_location(
    selector: usize,
    row: usize,
    chunk: usize,
    chunks_per_row: usize,
) -> Result<(usize, usize), EngineError> {
    let group = row % 8;
    let row_high = row / 8;
    let (lane, code_index) = match chunks_per_row {
        4 => {
            if selector > 3 {
                return Err(EngineError::message(
                    "sparse MMA single-thread selector must be in 0..=3",
                ));
            }
            (4 * group + selector, 4 * row_high + chunk)
        }
        8 => {
            if selector > 1 {
                return Err(EngineError::message(
                    "sparse MMA thread-pair selector must be 0 or 1",
                ));
            }
            (
                4 * group + 2 * selector + chunk / 4,
                4 * row_high + chunk % 4,
            )
        }
        16 => {
            if selector != 0 {
                return Err(EngineError::message(
                    "sparse MMA all-thread metadata requires selector 0",
                ));
            }
            (4 * group + chunk / 4, 4 * row_high + chunk % 4)
        }
        _ => {
            return Err(EngineError::message(
                "sparse MMA metadata chunk geometry is unsupported",
            ));
        }
    };
    Ok((lane, code_index))
}

pub(super) fn sparse_metadata_source_mask(
    k: usize,
    dtype: MatrixSparseOperandType,
    selector: usize,
) -> Result<crate::WarpMask, EngineError> {
    let chunks_per_row = k / sparse_chunk_width(dtype);
    let mut bits = 0;
    for row in 0..16 {
        for chunk in 0..chunks_per_row {
            let (lane, _) = sparse_metadata_location(selector, row, chunk, chunks_per_row)?;
            bits |= 1 << lane;
        }
    }
    Ok(crate::WarpMask::from_bits(bits))
}

fn sparse_metadata_code(
    metadata: &WarpValue<u32>,
    selector: usize,
    row: usize,
    chunk: usize,
    chunks_per_row: usize,
) -> Result<u8, EngineError> {
    let (lane, code_index) = sparse_metadata_location(selector, row, chunk, chunks_per_row)?;
    let word = metadata[lane];
    Ok(((word >> (4 * code_index)) & 0xf) as u8)
}

fn sparse_dense_position(
    dtype: MatrixSparseOperandType,
    code: u8,
    packed: usize,
    ordered_metadata: bool,
) -> Result<usize, EngineError> {
    if dtype == MatrixSparseOperandType::Tf32 {
        return match code {
            0x4 => Ok(0),
            0xe => Ok(1),
            _ => Err(EngineError::message(format!(
                "sparse TF32 metadata code 0x{code:x} is invalid"
            ))),
        };
    }
    let first = usize::from(code & 0x3);
    let second = usize::from((code >> 2) & 0x3);
    if first == second {
        return Err(EngineError::message(format!(
            "sparse MMA metadata code 0x{code:x} repeats one position"
        )));
    }
    if ordered_metadata && first > second {
        return Err(EngineError::message(format!(
            "ordered sparse MMA metadata code 0x{code:x} has descending indices"
        )));
    }
    if matches!(
        dtype,
        MatrixSparseOperandType::I4 | MatrixSparseOperandType::U4
    ) {
        let pair = if packed < 2 { first } else { second };
        Ok(2 * pair + packed % 2)
    } else {
        Ok(if packed == 0 { first } else { second })
    }
}


#[allow(clippy::too_many_arguments)]
pub fn raw_mma_sp_sync(
    physical: &PhysicalMemory,
    context: &WarpContext,
    destination: &[PhysicalPtr],
    a: &[WarpValue<u32>],
    b: &[WarpValue<u32>],
    accumulator: &[WarpValue<u32>],
    metadata: &WarpValue<u32>,
    selector: usize,
    k: usize,
    a_dtype: MatrixSparseOperandType,
    b_dtype: MatrixSparseOperandType,
    accumulator_dtype: MatrixSparseAccumulatorType,
    saturate: bool,
    ordered_metadata: bool,
) -> Result<(), EngineError> {
    super::require_full_warp_sync(context.active_mask(), "mma.sp.sync m16n8")?;
    validate_sparse_type_pair(a_dtype, b_dtype, accumulator_dtype, saturate)?;
    let (expected_a, expected_b) = sparse_fragment_counts(k, a_dtype)?;
    let (_, expected_b_for_type) = sparse_fragment_counts(k, b_dtype)?;
    let expected_c = match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => 2,
        MatrixSparseAccumulatorType::Fp32 | MatrixSparseAccumulatorType::I32 => 4,
    };
    if a.len() != expected_a
        || b.len() != expected_b
        || expected_b != expected_b_for_type
        || accumulator.len() != expected_c
        || destination.len() != expected_c
    {
        return Err(EngineError::message(
            "sparse MMA fragment register counts are invalid",
        ));
    }
    let chunk_width = sparse_chunk_width(a_dtype);
    let chunks_per_row = k / chunk_width;
    let stored_per_chunk = sparse_stored_per_chunk(a_dtype);
    let compressed_k = chunks_per_row * stored_per_chunk;
    if accumulator_dtype == MatrixSparseAccumulatorType::I32 {
        let accumulator_registers = decode_fragment(accumulator, |bits| bits as i32);
        let mut output = gather_mma_accumulator(16, &accumulator_registers)
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        for row in 0..16 {
            let mut a_values = Vec::with_capacity(compressed_k);
            let mut dense_inner = Vec::with_capacity(compressed_k);
            for chunk in 0..chunks_per_row {
                let code = sparse_metadata_code(metadata, selector, row, chunk, chunks_per_row)?;
                for packed in 0..stored_per_chunk {
                    let (lane, slot) = sparse_a_owner(row, chunk, packed, a_dtype);
                    let position = sparse_dense_position(a_dtype, code, packed, ordered_metadata)?;
                    a_values.push(read_sparse_integer_operand(a, lane, slot, a_dtype)?);
                    dense_inner.push(chunk * chunk_width + position);
                }
            }
            let b_values = gather_matrix(8, compressed_k, |column, term| {
                sparse_b_integer(b, dense_inner[term], column, b_dtype)
            })?;
            numsim_fp_env::multiply_accumulate_i32_abt(
                1,
                8,
                compressed_k,
                &a_values,
                &b_values,
                &mut output[row * 8..(row + 1) * 8],
            )
            .map_err(|error| {
                EngineError::message(format!("raw sparse MMA SIMD shape error: {error}"))
            })?;
        }
        let output = output
            .into_iter()
            .map(|value| {
                if saturate {
                    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
                } else {
                    value as i32
                }
            })
            .collect::<Vec<_>>();
        return store_mma_output(physical, context, destination, 16, 0_i32, &output);
    }

    let mut output = match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => gather_packed_f16_accumulator(16, accumulator)?,
        MatrixSparseAccumulatorType::Fp32 => {
            let registers = decode_fragment(accumulator, f32::from_bits);
            gather_mma_accumulator(16, &registers)
        }
        MatrixSparseAccumulatorType::I32 => unreachable!(),
    };
    for row in 0..16 {
        let mut a_values = Vec::with_capacity(compressed_k);
        let mut dense_inner = Vec::with_capacity(compressed_k);
        for chunk in 0..chunks_per_row {
            let code = sparse_metadata_code(metadata, selector, row, chunk, chunks_per_row)?;
            for packed in 0..stored_per_chunk {
                let (lane, slot) = sparse_a_owner(row, chunk, packed, a_dtype);
                let position = sparse_dense_position(a_dtype, code, packed, ordered_metadata)?;
                a_values.push(read_sparse_operand(a, lane, slot, a_dtype)?);
                dense_inner.push(chunk * chunk_width + position);
            }
        }
        let b_values = gather_matrix(8, compressed_k, |column, term| {
            sparse_b_float(b, dense_inner[term], column, b_dtype)
        })?;
        let values = mma_f32(
            1,
            compressed_k,
            &a_values,
            &b_values,
            Some(&output[row * 8..(row + 1) * 8]),
        )?;
        output[row * 8..(row + 1) * 8].copy_from_slice(&values);
    }
    match accumulator_dtype {
        MatrixSparseAccumulatorType::Fp16 => {
            let registers = matrix_output_registers(16, destination.len() * 2, 0.0_f32, &output)?;
            store_packed_f16_registers(physical, context, destination, &registers)
        }
        MatrixSparseAccumulatorType::Fp32 => {
            store_mma_output(physical, context, destination, 16, 0.0_f32, &output)
        }
        MatrixSparseAccumulatorType::I32 => unreachable!(),
    }
}

fn tmem_full_view(
    physical: &PhysicalMemory,
    context: &WarpContext,
    anchor: &RuntimeBuffer,
    target_cta: usize,
) -> Result<crate::TmemView, EngineError> {
    let RuntimeBuffer::Tmem { allocations, .. } = anchor else {
        return Err(EngineError::message("tcgen05.shift requires a TMEM anchor"));
    };
    let owner = CtaId::new(context.topology(), context.cluster_id(), target_cta)?;
    let allocation = allocations
        .get(owner.global_cta_id(context.topology())?)
        .ok_or_else(|| EngineError::message("tcgen05.shift target TMEM is missing"))?;
    physical
        .tmem()
        .full_view(owner, allocation)
        .map_err(EngineError::from)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_tcgen05_shift<const COLUMNS: usize>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    lifecycle: &TcgenLifecycleHub,
    access_mode: TmemAccessMode,
    anchor: &RuntimeBuffer,
    address: u32,
    cta_group: usize,
    issuing_lane: usize,
) -> Result<(), EngineError> {
    if context.active_mask().len() != 1 || !context.active_mask().contains(issuing_lane) {
        return Err(EngineError::message(
            "tcgen05.shift requires one issuing lane",
        ));
    }
    let base_row = ((address >> 16) & 0xffff) as usize;
    let base_col = (address & 0xffff) as usize;
    if base_row % 32 != 0 || base_row >= 128 {
        return Err(EngineError::message(
            "tcgen05.shift row address must name one aligned warp",
        ));
    }
    let pair_base = context.cta_id_in_cluster() & !1_usize;
    let targets: Vec<usize> = if cta_group == 1 {
        vec![context.cta_id_in_cluster()]
    } else if cta_group == 2 {
        vec![pair_base, pair_base + 1]
    } else {
        return Err(EngineError::message(
            "tcgen05.shift cta_group must be 1 or 2",
        ));
    };
    for target in targets {
        lifecycle.validate_access_range_at_cta(*context, target, base_col, COLUMNS, access_mode)?;
        let view = tmem_full_view(physical, context, anchor, target)?;
        let lane_count = view.region().lane_count;
        if lane_count == 0 || lane_count > 128 || lane_count % 32 != 0 {
            return Err(EngineError::message(format!(
                "tcgen05.shift requires 1-4 complete 32-row TMEM partitions, got {lane_count} rows"
            )));
        }
        for partition_base in (0..lane_count).step_by(32) {
            let mut rows = vec![[([0_u8; 4], [false; 4]); COLUMNS]; 31];
            for destination_row in 0..31 {
                for column in 0..COLUMNS {
                    rows[destination_row][column] =
                        physical.tmem().snapshot_cell_preserving_validity(
                            &view,
                            partition_base + destination_row + 1,
                            base_col + column,
                        )?;
                }
            }
            for destination_row in 0..31 {
                for column in 0..COLUMNS {
                    let (bytes, validity) = &rows[destination_row][column];
                    physical.tmem().write_cell_snapshot(
                        &view,
                        partition_base + destination_row,
                        base_col + column,
                        bytes,
                        validity,
                    )?;
                }
            }
        }
    }
    Ok(())
}
