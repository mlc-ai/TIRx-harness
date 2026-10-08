#![deny(unsafe_op_in_unsafe_fn)]

use core::ffi::c_int;
use std::error::Error;
use std::fmt;

#[cfg(target_os = "linux")]
// Linux libc ABIs define FE_TONEAREST as zero.
const FE_TONEAREST: c_int = 0;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fesetround(round: c_int) -> c_int;
    fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: *const u64) -> c_int;
    fn sched_getaffinity(pid: c_int, cpusetsize: usize, mask: *mut u64) -> c_int;
    fn malloc_trim(pad: usize) -> c_int;
}

/// Return freed heap pages to the OS so peak RSS tracks live allocations.
/// Returns whether anything was released.
#[cfg(target_os = "linux")]
pub fn release_free_heap() -> bool {
    // SAFETY: malloc_trim takes no pointer arguments.
    unsafe { malloc_trim(0) == 1 }
}

#[cfg(not(target_os = "linux"))]
pub fn release_free_heap() -> bool {
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatEnvironmentError {
    UnsupportedPlatform,
    SetRoundingFailed(c_int),
}

impl fmt::Display for FloatEnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                formatter.write_str("floating-point environment setup is unsupported")
            }
            Self::SetRoundingFailed(status) => {
                write!(formatter, "fesetround returned status {status}")
            }
        }
    }
}

impl Error for FloatEnvironmentError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FmaShapeError {
    operand: &'static str,
    actual: usize,
    expected: Option<usize>,
}

impl fmt::Display for FmaShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expected {
            Some(expected) => write!(
                formatter,
                "{} has {} elements, expected {expected}",
                self.operand, self.actual
            ),
            None => write!(formatter, "{} shape overflows usize", self.operand),
        }
    }
}

impl Error for FmaShapeError {}

/// Select IEEE round-to-nearest-even for the calling thread.
#[cfg(target_os = "linux")]
pub fn set_round_to_nearest() -> Result<(), FloatEnvironmentError> {
    // SAFETY: fesetround has no pointer arguments and FE_TONEAREST is a valid
    // Linux libc rounding-mode constant. The floating-point environment is local to
    // the calling worker thread.
    let status = unsafe { fesetround(FE_TONEAREST) };
    if status == 0 {
        Ok(())
    } else {
        Err(FloatEnvironmentError::SetRoundingFailed(status))
    }
}

/// Fail closed when the host has no audited thread-local fenv implementation.
#[cfg(not(target_os = "linux"))]
pub fn set_round_to_nearest() -> Result<(), FloatEnvironmentError> {
    Err(FloatEnvironmentError::UnsupportedPlatform)
}

fn validate_abt_shapes<A, B, O>(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[A],
    b_transposed: &[B],
    output: &[O],
) -> Result<(), FmaShapeError> {
    for (operand, actual, shape) in [
        ("A", a_values.len(), (m, k)),
        ("transposed B", b_transposed.len(), (k, n)),
        ("output", output.len(), (m, n)),
    ] {
        let Some(expected) = shape.0.checked_mul(shape.1) else {
            return Err(FmaShapeError {
                operand,
                actual,
                expected: None,
            });
        };
        if actual != expected {
            return Err(FmaShapeError {
                operand,
                actual,
                expected: Some(expected),
            });
        }
    }
    Ok(())
}

fn fma_f32_abt_increasing_k_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

/// One register tile of `R` rows by `C` 16-lane column vectors.
///
/// Every accumulator stays in a register across the whole K loop, so each
/// output element's increasing-K FMA chain is unchanged — only its
/// intermediate values move from memory round-trips into registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_tile_avx512<const R: usize, const C: usize>(
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
    row: usize,
    column: usize,
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    const LANES: usize = 16;
    // SAFETY: the caller guarantees `row + R <= m` and `column + C * 16 <= n`
    // over shape-validated slices.
    unsafe {
        let mut accumulators = [[_mm512_setzero_ps(); C]; R];
        for r in 0..R {
            for c in 0..C {
                accumulators[r][c] =
                    _mm512_loadu_ps(output.as_ptr().add((row + r) * n + column + c * LANES));
            }
        }
        for inner in 0..k {
            let mut b_lanes = [_mm512_setzero_ps(); C];
            for (c, lanes) in b_lanes.iter_mut().enumerate() {
                *lanes = _mm512_loadu_ps(b_transposed.as_ptr().add(inner * n + column + c * LANES));
            }
            for (r, row_accumulators) in accumulators.iter_mut().enumerate() {
                let a_lanes = _mm512_set1_ps(*a_values.get_unchecked((row + r) * k + inner));
                for (accumulator, &b) in row_accumulators.iter_mut().zip(&b_lanes) {
                    *accumulator = _mm512_fmadd_ps(a_lanes, b, *accumulator);
                }
            }
        }
        for r in 0..R {
            for c in 0..C {
                _mm512_storeu_ps(
                    output.as_mut_ptr().add((row + r) * n + column + c * LANES),
                    accumulators[r][c],
                );
            }
        }
    }
}

/// Scalar columns keep the same register-resident chain: one accumulator per
/// element carried across the whole K loop.
fn fma_scalar_columns(
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) {
    for row in rows {
        for column in columns.clone() {
            let mut accumulator = output[row * n + column];
            for inner in 0..k {
                accumulator = a_values[row * k + inner]
                    .mul_add(b_transposed[inner * n + column], accumulator);
            }
            output[row * n + column] = accumulator;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_f32_abt_increasing_k_avx512(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    const LANES: usize = 16;
    const ROW_TILE: usize = 4;
    const COL_VECTORS: usize = 4;
    const COL_TILE: usize = COL_VECTORS * LANES;

    // SAFETY: shape validation proves the slice extents; every tile call stays
    // inside `row + rows <= m` and `column + width <= n`.
    unsafe {
        let mut row = 0;
        while row + ROW_TILE <= m {
            let mut column = 0;
            while column + COL_TILE <= n {
                fma_tile_avx512::<ROW_TILE, COL_VECTORS>(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row,
                    column,
                );
                column += COL_TILE;
            }
            while column + LANES <= n {
                fma_tile_avx512::<ROW_TILE, 1>(n, k, a_values, b_transposed, output, row, column);
                column += LANES;
            }
            if column < n {
                fma_scalar_columns(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row..row + ROW_TILE,
                    column..n,
                );
            }
            row += ROW_TILE;
        }
        while row < m {
            let mut column = 0;
            while column + COL_TILE <= n {
                fma_tile_avx512::<1, COL_VECTORS>(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row,
                    column,
                );
                column += COL_TILE;
            }
            while column + LANES <= n {
                fma_tile_avx512::<1, 1>(n, k, a_values, b_transposed, output, row, column);
                column += LANES;
            }
            if column < n {
                fma_scalar_columns(
                    n,
                    k,
                    a_values,
                    b_transposed,
                    output,
                    row..row + 1,
                    column..n,
                );
            }
            row += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_f32_abt_increasing_k_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // SAFETY: shape validation proves both row slices have n elements,
            // and the loop admits only complete 8-element vectors.
            unsafe {
                let a_lanes = _mm256_set1_ps(a);
                while column + 8 <= n {
                    let b_lanes = _mm256_loadu_ps(b_row.as_ptr().add(column));
                    let accumulators = _mm256_loadu_ps(output_row.as_ptr().add(column));
                    _mm256_storeu_ps(
                        output_row.as_mut_ptr().add(column),
                        _mm256_fmadd_ps(a_lanes, b_lanes, accumulators),
                    );
                    column += 8;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

/// Apply `output = A * B^T + output` while preserving increasing-K binary32
/// FMA order independently for every output element.
///
/// `b_transposed` is laid out as `[k, n]`. Runtime SIMD dispatch changes only
/// which independent output columns advance together; it never reassociates an
/// individual output element's FMA chain.
pub fn fma_f32_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_transposed: &[f32],
    output: &mut [f32],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_transposed, output)?;

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 first: every AVX-512F host also reports AVX2, so the
        // narrower path must not shadow the wider one.
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime detection establishes AVX-512F support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f32_abt_increasing_k_avx512(m, n, k, a_values, b_transposed, output);
            }
            return Ok(());
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection establishes AVX2/FMA support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f32_abt_increasing_k_avx2(m, n, k, a_values, b_transposed, output);
            }
            return Ok(());
        }
    }

    fma_f32_abt_increasing_k_scalar(m, n, k, a_values, b_transposed, output);
    Ok(())
}

fn fma_f64_abt_increasing_k_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_f64_abt_increasing_k_avx512(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // SAFETY: shape validation proves both row slices have n elements,
            // and the loop admits only complete eight-element vectors.
            unsafe {
                let a_lanes = _mm512_set1_pd(a);
                while column + 8 <= n {
                    let b_lanes = _mm512_loadu_pd(b_row.as_ptr().add(column));
                    let accumulators = _mm512_loadu_pd(output_row.as_ptr().add(column));
                    _mm512_storeu_pd(
                        output_row.as_mut_ptr().add(column),
                        _mm512_fmadd_pd(a_lanes, b_lanes, accumulators),
                    );
                    column += 8;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_f64_abt_increasing_k_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // SAFETY: shape validation proves both row slices have n elements,
            // and the loop admits only complete four-element vectors.
            unsafe {
                let a_lanes = _mm256_set1_pd(a);
                while column + 4 <= n {
                    let b_lanes = _mm256_loadu_pd(b_row.as_ptr().add(column));
                    let accumulators = _mm256_loadu_pd(output_row.as_ptr().add(column));
                    _mm256_storeu_pd(
                        output_row.as_mut_ptr().add(column),
                        _mm256_fmadd_pd(a_lanes, b_lanes, accumulators),
                    );
                    column += 4;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator = a.mul_add(b, *accumulator);
            }
        }
    }
}

/// Apply `output = A * B^T + output` with one binary64 FMA chain per output.
pub fn fma_f64_abt_increasing_k(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f64],
    b_transposed: &[f64],
    output: &mut [f64],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_transposed, output)?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection establishes AVX2/FMA support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f64_abt_increasing_k_avx2(m, n, k, a_values, b_transposed, output);
            }
            return Ok(());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime detection establishes AVX-512F support and the
            // validated slices satisfy the implementation's bounds contract.
            unsafe {
                fma_f64_abt_increasing_k_avx512(m, n, k, a_values, b_transposed, output);
            }
            return Ok(());
        }
    }

    fma_f64_abt_increasing_k_scalar(m, n, k, a_values, b_transposed, output);
    Ok(())
}

fn multiply_accumulate_i32_abt_scalar(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_transposed: &[i64],
    output: &mut [i64],
) {
    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            for (accumulator, &b) in output_row.iter_mut().zip(b_row) {
                *accumulator += i64::from(a) * b;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_accumulate_i32_abt_avx2(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_transposed: &[i64],
    output: &mut [i64],
) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi64x,
        _mm256_storeu_si256,
    };

    for row in 0..m {
        let a_row = &a_values[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, &a) in a_row.iter().enumerate() {
            let b_row = &b_transposed[inner * n..(inner + 1) * n];
            let mut column = 0;
            // Each i64 B lane contains one sign-extended i32 operand. AVX2
            // `mul_epi32` multiplies the low i32 of each i64 lane and returns
            // four exact i64 products.
            unsafe {
                let a_lanes = _mm256_set1_epi64x(i64::from(a));
                while column + 4 <= n {
                    let b_lanes = _mm256_loadu_si256(b_row.as_ptr().add(column).cast::<__m256i>());
                    let accumulators =
                        _mm256_loadu_si256(output_row.as_ptr().add(column).cast::<__m256i>());
                    let result = _mm256_add_epi64(accumulators, _mm256_mul_epi32(a_lanes, b_lanes));
                    _mm256_storeu_si256(
                        output_row.as_mut_ptr().add(column).cast::<__m256i>(),
                        result,
                    );
                    column += 4;
                }
            }
            for (accumulator, &b) in output_row[column..].iter_mut().zip(&b_row[column..]) {
                *accumulator += i64::from(a) * b;
            }
        }
    }
}

/// Apply exact signed-i32 products into i64 accumulators.
///
/// `b_values` is row-major `[n, k]`; this routine performs the transpose once
/// so independent output columns can advance together under SIMD.
pub fn multiply_accumulate_i32_abt(
    m: usize,
    n: usize,
    k: usize,
    a_values: &[i32],
    b_values: &[i32],
    output: &mut [i64],
) -> Result<(), FmaShapeError> {
    validate_abt_shapes(m, n, k, a_values, b_values, output)?;
    let mut b_transposed = vec![0_i64; b_values.len()];
    for column in 0..n {
        for inner in 0..k {
            b_transposed[inner * n + column] = i64::from(b_values[column * k + inner]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection establishes AVX2 support and validation
        // proves every vector load/store stays within its row slice.
        unsafe {
            multiply_accumulate_i32_abt_avx2(m, n, k, a_values, &b_transposed, output);
        }
        return Ok(());
    }

    multiply_accumulate_i32_abt_scalar(m, n, k, a_values, &b_transposed, output);
    Ok(())
}

/// Largest CPU index the affinity mask can address (glibc's `cpu_set_t` is
/// 1024 bits wide).
pub const MAX_AFFINITY_CPUS: usize = 1024;
const AFFINITY_MASK_WORDS: usize = MAX_AFFINITY_CPUS / 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadAffinityError {
    UnsupportedPlatform,
    CpuOutOfRange(usize),
    EmptyMask,
    SyscallFailed(c_int),
}

impl fmt::Display for ThreadAffinityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => formatter.write_str("thread affinity is unsupported"),
            Self::CpuOutOfRange(cpu) => {
                write!(formatter, "cpu {cpu} exceeds the {MAX_AFFINITY_CPUS}-bit affinity mask")
            }
            Self::EmptyMask => formatter.write_str("affinity mask selects no cpu"),
            Self::SyscallFailed(status) => {
                write!(formatter, "affinity syscall returned status {status}")
            }
        }
    }
}

impl Error for ThreadAffinityError {}

/// Restrict the calling thread to `cpus`. Threads spawned afterwards inherit
/// the mask.
#[cfg(target_os = "linux")]
pub fn set_current_thread_allowed_cpus(cpus: &[usize]) -> Result<(), ThreadAffinityError> {
    let mut mask = [0u64; AFFINITY_MASK_WORDS];
    for &cpu in cpus {
        if cpu >= MAX_AFFINITY_CPUS {
            return Err(ThreadAffinityError::CpuOutOfRange(cpu));
        }
        mask[cpu / 64] |= 1u64 << (cpu % 64);
    }
    if mask.iter().all(|word| *word == 0) {
        return Err(ThreadAffinityError::EmptyMask);
    }
    // SAFETY: `mask` is an initialized buffer of exactly `size_of_val(&mask)`
    // bytes that outlives the call, and pid 0 addresses the calling thread.
    let status = unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) };
    if status == 0 {
        Ok(())
    } else {
        Err(ThreadAffinityError::SyscallFailed(status))
    }
}

/// Logical CPUs the calling thread may currently run on, ascending.
#[cfg(target_os = "linux")]
pub fn current_thread_allowed_cpus() -> Result<Vec<usize>, ThreadAffinityError> {
    let mut mask = [0u64; AFFINITY_MASK_WORDS];
    // SAFETY: `mask` is a writable buffer of exactly `size_of_val(&mask)` bytes
    // that outlives the call, and pid 0 addresses the calling thread.
    let status =
        unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) };
    if status != 0 {
        return Err(ThreadAffinityError::SyscallFailed(status));
    }
    Ok((0..MAX_AFFINITY_CPUS)
        .filter(|cpu| (mask[cpu / 64] >> (cpu % 64)) & 1 == 1)
        .collect())
}

#[cfg(not(target_os = "linux"))]
pub fn set_current_thread_allowed_cpus(_cpus: &[usize]) -> Result<(), ThreadAffinityError> {
    Err(ThreadAffinityError::UnsupportedPlatform)
}

#[cfg(not(target_os = "linux"))]
pub fn current_thread_allowed_cpus() -> Result<Vec<usize>, ThreadAffinityError> {
    Err(ThreadAffinityError::UnsupportedPlatform)
}



#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn supported_host_accepts_round_to_nearest() {
        assert_eq!(set_round_to_nearest(), Ok(()));
    }

    #[test]
    fn simd_fma_is_bitwise_equal_to_scalar_increasing_k() {
        let (m, n, k) = (3_usize, 19_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f32 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| ((index as f32 + 3.0) * -0.21875).cos())
            .collect::<Vec<_>>();
        let mut b_transposed = vec![0.0_f32; n * k];
        for column in 0..n {
            for inner in 0..k {
                b_transposed[inner * n + column] = b_values[column * k + inner];
            }
        }
        let initial = (0..m * n)
            .map(|index| (index as f32 - 6.0) * -0.046875)
            .collect::<Vec<_>>();
        let mut simd = initial.clone();
        let mut scalar = initial;

        fma_f32_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut simd).unwrap();
        fma_f32_abt_increasing_k_scalar(m, n, k, &a_values, &b_transposed, &mut scalar);

        assert_eq!(
            simd.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn simd_fma_matches_scalar_bitwise_across_tile_shapes() {
        // Sweep every microkernel path: 4-row and leftover-row tiles, 64- and
        // 16-column vectors, and scalar column tails.
        for &m in &[1_usize, 2, 3, 4, 5, 8, 9] {
            for &n in &[1_usize, 7, 15, 16, 17, 63, 64, 65, 80, 130] {
                for &k in &[1_usize, 2, 7, 32] {
                    let a_values = (0..m * k)
                        .map(|index| ((index as f32 - 11.0) * 0.317).sin() * 3.0)
                        .collect::<Vec<_>>();
                    let b_transposed = (0..n * k)
                        .map(|index| ((index as f32 + 5.0) * -0.213).cos() * 2.0)
                        .collect::<Vec<_>>();
                    let initial = (0..m * n)
                        .map(|index| (index as f32 - 6.0) * -0.047)
                        .collect::<Vec<_>>();
                    let mut simd = initial.clone();
                    let mut scalar = initial;

                    fma_f32_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut simd).unwrap();
                    fma_f32_abt_increasing_k_scalar(m, n, k, &a_values, &b_transposed, &mut scalar);

                    assert_eq!(
                        simd.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        scalar
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "bitwise mismatch at m={m} n={n} k={k}",
                    );
                }
            }
        }
    }

    #[test]
    fn simd_fma_rejects_malformed_shapes() {
        let error =
            fma_f32_abt_increasing_k(2, 3, 4, &[0.0; 7], &[0.0; 12], &mut [0.0; 6]).unwrap_err();
        assert_eq!(error.to_string(), "A has 7 elements, expected 8");
    }

    #[test]
    fn f64_simd_matches_independent_increasing_k_fma_chains() {
        let (m, n, k) = (3_usize, 11_usize, 7_usize);
        let a_values = (0..m * k)
            .map(|index| ((index as f64 - 9.0) * 0.3125).sin())
            .collect::<Vec<_>>();
        let mut b_transposed = vec![0.0_f64; n * k];
        for inner in 0..k {
            for column in 0..n {
                b_transposed[inner * n + column] =
                    (((column * k + inner) as f64 + 3.0) * -0.21875).cos();
            }
        }
        let initial = (0..m * n)
            .map(|index| (index as f64 - 6.0) * -0.046875)
            .collect::<Vec<_>>();
        let mut actual = initial.clone();

        fma_f64_abt_increasing_k(m, n, k, &a_values, &b_transposed, &mut actual).unwrap();

        let mut expected = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut accumulator = initial[row * n + column];
                for inner in 0..k {
                    accumulator = a_values[row * k + inner]
                        .mul_add(b_transposed[inner * n + column], accumulator);
                }
                expected.push(accumulator.to_bits());
            }
        }
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
        );
    }

    #[test]
    fn integer_simd_matches_independent_dot_products() {
        let (m, n, k) = (3_usize, 11_usize, 17_usize);
        let a_values = (0..m * k)
            .map(|index| (index as i32 % 31) - 15)
            .collect::<Vec<_>>();
        let b_values = (0..n * k)
            .map(|index| 12 - (index as i32 % 29))
            .collect::<Vec<_>>();
        let initial = (0..m * n)
            .map(|index| i64::from(index as i32 - 20) * 1_000_003)
            .collect::<Vec<_>>();
        let mut actual = initial.clone();

        multiply_accumulate_i32_abt(m, n, k, &a_values, &b_values, &mut actual).unwrap();

        let expected = (0..m)
            .flat_map(|row| {
                let initial = initial.clone();
                let a_values = &a_values;
                let b_values = &b_values;
                (0..n).map(move |column| {
                    (0..k).fold(initial[row * n + column], |sum, inner| {
                        sum + i64::from(a_values[row * k + inner])
                            * i64::from(b_values[column * k + inner])
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
