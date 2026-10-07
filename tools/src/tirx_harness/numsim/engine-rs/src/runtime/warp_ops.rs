use crate::{
    cuda_f32_add, cuda_f32_max, cuda_f32_min, cuda_f64_add, cuda_f64_max, cuda_f64_min,
    cuda_reduce_bf16_add, cuda_reduce_bf16_max, cuda_reduce_bf16_min, cuda_reduce_fp16_add,
    cuda_reduce_fp16_max, cuda_reduce_fp16_min, ptx_fns_b32, EngineError, RuntimeScalar, WarpMask,
    WarpValue, WARP_SIZE,
};

use super::require_full_warp_sync;

#[derive(Clone, Copy)]
pub(crate) enum WarpShuffleMode {
    Index,
    Up,
    Down,
    Xor,
}

/// Execute the lane-selection semantics of one PTX `shfl.sync` instruction.
///
/// `selectors` is operand `b`; `controls` is packed operand `c` (clamp in
/// bits 4:0 and segment mask in bits 12:8).  The returned predicate is the
/// optional PTX destination predicate.  Synchronization scheduling and source
/// occurrence accounting stay in the instruction wrapper.
fn resolve_warp_shuffle_sources(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: WarpShuffleMode,
) -> Result<(WarpValue<usize>, WarpValue<bool>), EngineError> {
    let participant_mask = validate_warp_collective_participants(
        active_mask,
        participant_masks,
        &crate::DiagnosticLabel::new("shfl.sync"),
    )?;
    let mut source_lanes = WarpValue::splat(0_usize);
    let mut in_range = WarpValue::splat(false);
    for lane in active_mask {
        let lane5 = lane & 0x1f;
        let selector = selectors[lane] as usize & 0x1f;
        let control = controls[lane] as usize;
        let clamp = control & 0x1f;
        let segment_mask = (control >> 8) & 0x1f;
        let maximum = (lane5 & segment_mask) | (clamp & !segment_mask & 0x1f);
        let minimum = lane5 & segment_mask;
        let (candidate, valid) = match mode {
            WarpShuffleMode::Up => {
                let candidate = lane5.checked_sub(selector);
                (
                    candidate.unwrap_or(lane5),
                    candidate.is_some_and(|candidate| candidate >= maximum),
                )
            }
            WarpShuffleMode::Down => {
                let candidate = lane5 + selector;
                (candidate, candidate <= maximum)
            }
            WarpShuffleMode::Xor => {
                let candidate = lane5 ^ selector;
                (candidate, candidate <= maximum)
            }
            WarpShuffleMode::Index => {
                let candidate = minimum | (selector & !segment_mask & 0x1f);
                (candidate, candidate <= maximum)
            }
        };
        let source_lane = if valid { candidate } else { lane };
        if participant_mask & (1_u32 << source_lane) == 0 || !active_mask.contains(source_lane) {
            return Err(EngineError::message(
                "warp shuffle reads a non-participant lane",
            ));
        }
        source_lanes[lane] = source_lane;
        in_range[lane] = valid;
    }
    Ok((source_lanes, in_range))
}

/// Return the lanes whose register operands are actually selected by one PTX
/// `shfl.sync` instruction.  The frontend uses this mask to avoid materializing
/// register values from lanes that the instruction cannot observe.
pub(crate) fn warp_shuffle_source_mask(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: WarpShuffleMode,
) -> Result<WarpMask, EngineError> {
    let (source_lanes, _in_range) =
        resolve_warp_shuffle_sources(active_mask, participant_masks, selectors, controls, mode)?;
    let mut source_mask = WarpMask::EMPTY;
    for lane in active_mask {
        source_mask |= WarpMask::from_bits(1_u32 << source_lanes[lane]);
    }
    Ok(source_mask)
}

pub(crate) fn warp_shuffle_ptx<T: RuntimeScalar>(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    values: &WarpValue<T>,
    selectors: &WarpValue<u32>,
    controls: &WarpValue<u32>,
    mode: WarpShuffleMode,
) -> Result<(WarpValue<T>, WarpValue<bool>), EngineError> {
    let (source_lanes, in_range) =
        resolve_warp_shuffle_sources(active_mask, participant_masks, selectors, controls, mode)?;
    let mut result = WarpValue::splat(T::zero());
    for lane in active_mask {
        result[lane] = values[source_lanes[lane]];
    }
    Ok((result, in_range))
}

pub(crate) fn validate_warp_collective_participants(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    operation: &crate::DiagnosticLabel,
) -> Result<u32, EngineError> {
    let first_lane = active_mask
        .first_active()
        .ok_or_else(|| operation.engine_error(format_args!(" has no active lane")))?;
    let expected = participant_masks[first_lane];
    if expected == 0 {
        return Err(operation.engine_error(format_args!(" participant mask must not be zero")));
    }
    if expected & !active_mask.bits() != 0 {
        return Err(operation.warp_collective_divergence_with_detail(
            active_mask,
            format_args!(" participant mask names an inactive lane"),
        ));
    }
    for lane in active_mask {
        let actual = participant_masks[lane];
        if actual != expected {
            return Err(operation.warp_collective_divergence_with_detail(
                active_mask,
                format_args!(
                    " participant masks disagree: lane {lane} has 0x{actual:08x}, expected 0x{expected:08x}"
                ),
            ));
        }
        if actual & (1_u32 << lane) == 0 {
            return Err(operation.warp_collective_divergence_with_detail(
                active_mask,
                format_args!(" executing lane {lane} is absent from its participant mask"),
            ));
        }
    }
    Ok(expected)
}

pub fn require_uniform_i64(
    values: &WarpValue<i64>,
    mask: WarpMask,
    label: &str,
) -> Result<i64, EngineError> {
    let first_lane = mask
        .first_active()
        .ok_or_else(|| EngineError::message(format!("{label} has no active lane")))?;
    let expected = values[first_lane];
    for lane in mask {
        if values[lane] != expected {
            return Err(EngineError::message(format!(
                "{label} must agree across active lanes: lane {lane} has {}, expected {expected}",
                values[lane]
            )));
        }
    }
    Ok(expected)
}

pub fn warp_fns_b32(
    active_mask: WarpMask,
    masks: &WarpValue<u32>,
    bases: &WarpValue<u32>,
    offsets: &WarpValue<i32>,
) -> Result<WarpValue<u32>, EngineError> {
    let mut result = WarpValue::splat(0_u32);
    for lane in active_mask {
        if bases[lane] >= 32 {
            return Err(EngineError::message(
                "fns.b32 base is outside the defined 0..31 range",
            ));
        }
        result[lane] = ptx_fns_b32(masks[lane], bases[lane], offsets[lane]);
    }
    Ok(result)
}

pub fn warp_ballot_sync(
    active_mask: WarpMask,
    participant_masks: &WarpValue<u32>,
    predicates: &WarpValue<bool>,
) -> Result<WarpValue<u32>, EngineError> {
    let participant_mask = validate_warp_collective_participants(
        active_mask,
        participant_masks,
        &crate::DiagnosticLabel::new("warp ballot"),
    )?;
    let mut result = WarpValue::splat(0_u32);
    for lane in active_mask {
        let mut bits = 0_u32;
        for source_lane in 0..WARP_SIZE {
            if participant_mask & (1_u32 << source_lane) != 0 && predicates[source_lane] {
                bits |= 1_u32 << source_lane;
            }
        }
        result[lane] = bits;
    }
    Ok(result)
}

pub trait WarpReduceElement: RuntimeScalar {
    fn reduce_sum(self, other: Self) -> Self;
    fn reduce_max(self, other: Self) -> Self;
    fn reduce_min(self, other: Self) -> Self;
}

macro_rules! impl_integer_warp_reduce_element {
    ($($rust_type:ty),+ $(,)?) => {
        $(
            impl WarpReduceElement for $rust_type {
                fn reduce_sum(self, other: Self) -> Self {
                    self.wrapping_add(other)
                }

                fn reduce_max(self, other: Self) -> Self {
                    self.max(other)
                }

                fn reduce_min(self, other: Self) -> Self {
                    self.min(other)
                }
            }
        )+
    };
}

impl_integer_warp_reduce_element!(i8, i16, i32, i64, u8, u16, u32, u64);

impl WarpReduceElement for f32 {
    fn reduce_sum(self, other: Self) -> Self {
        cuda_f32_add(self, other)
    }

    fn reduce_max(self, other: Self) -> Self {
        cuda_f32_max(self, other)
    }

    fn reduce_min(self, other: Self) -> Self {
        cuda_f32_min(self, other)
    }
}

impl WarpReduceElement for f64 {
    fn reduce_sum(self, other: Self) -> Self {
        cuda_f64_add(self, other)
    }

    fn reduce_max(self, other: Self) -> Self {
        cuda_f64_max(self, other)
    }

    fn reduce_min(self, other: Self) -> Self {
        cuda_f64_min(self, other)
    }
}

fn warp_reduce<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
    combine: impl Fn(T, T) -> T,
) -> Result<WarpValue<T>, EngineError> {
    require_full_warp_sync(active_mask, "cuda_warp_reduce")?;
    if width == 0 || width > WARP_SIZE || !width.is_power_of_two() {
        return Err(EngineError::message(format!(
            "cuda_warp_reduce width must be a power of two in 1..={WARP_SIZE}, got {width}"
        )));
    }

    let mut result = values.clone();
    let mut delta = width / 2;
    while delta > 0 {
        let previous = result.clone();
        for lane in active_mask {
            let group_base = (lane / width) * width;
            let source_lane = group_base + ((lane - group_base) ^ delta);
            result[lane] = combine(previous[lane], previous[source_lane]);
        }
        delta >>= 1;
    }
    Ok(result)
}

pub fn warp_reduce_sum<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> Result<WarpValue<T>, EngineError> {
    warp_reduce(active_mask, values, width, T::reduce_sum)
}

pub fn warp_reduce_max<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> Result<WarpValue<T>, EngineError> {
    warp_reduce(active_mask, values, width, T::reduce_max)
}

pub fn warp_reduce_min<T: WarpReduceElement>(
    active_mask: WarpMask,
    values: &WarpValue<T>,
    width: usize,
) -> Result<WarpValue<T>, EngineError> {
    warp_reduce(active_mask, values, width, T::reduce_min)
}

pub fn warp_reduce_sum_fp16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_fp16_add)
}

pub fn warp_reduce_max_fp16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_fp16_max)
}

pub fn warp_reduce_min_fp16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_fp16_min)
}

pub fn warp_reduce_sum_bf16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_bf16_add)
}

pub fn warp_reduce_max_bf16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_bf16_max)
}

pub fn warp_reduce_min_bf16(
    active_mask: WarpMask,
    values: &WarpValue<f32>,
    width: usize,
) -> Result<WarpValue<f32>, EngineError> {
    warp_reduce(active_mask, values, width, cuda_reduce_bf16_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_and_shuffle_xor_select_lanes_inside_width_groups() {
        let participants = WarpValue::splat(u32::MAX);
        let values = WarpValue::from_fn(|lane| lane as u32);
        let (shuffled, _) = warp_shuffle_ptx(
            WarpMask::FULL,
            &participants,
            &values,
            &WarpValue::from_fn(|lane| 31 - lane as u32),
            &WarpValue::splat(31),
            WarpShuffleMode::Index,
        )
        .unwrap();
        let (xored, _) = warp_shuffle_ptx(
            WarpMask::FULL,
            &participants,
            &values,
            &WarpValue::splat(1),
            &WarpValue::splat(31),
            WarpShuffleMode::Xor,
        )
        .unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(shuffled[lane], (31 - lane) as u32);
            assert_eq!(xored[lane], (lane ^ 1) as u32);
        }
    }

    #[test]
    fn shuffle_xor_width_allows_later_groups_to_read_earlier_groups() {
        let (result, _) = warp_shuffle_ptx(
            WarpMask::FULL,
            &WarpValue::splat(u32::MAX),
            &WarpValue::from_fn(|lane| lane as u32),
            &WarpValue::splat(16),
            &WarpValue::splat(0x100f), // CUDA width 16 lowered to PTX segment/clamp.
            WarpShuffleMode::Xor,
        )
        .unwrap();
        for lane in 0..16 {
            assert_eq!(result[lane], lane as u32);
            assert_eq!(result[lane + 16], lane as u32);
        }
    }

    #[test]
    fn shuffle_rejects_nonparticipant_sources() {
        let active = WarpMask::from_lanes(0..16).unwrap();
        let error = warp_shuffle_ptx(
            active,
            &WarpValue::splat(active.bits()),
            &WarpValue::from_fn(|lane| lane as u32),
            &WarpValue::splat(31),
            &WarpValue::splat(31),
            WarpShuffleMode::Index,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("warp shuffle reads a non-participant lane"));
    }

    #[test]
    fn masked_collectives_reject_invalid_participants() {
        let label = crate::DiagnosticLabel::new("collective");
        let active = WarpMask::from_lanes(0..16).unwrap();
        assert!(
            validate_warp_collective_participants(active, &WarpValue::splat(u32::MAX), &label,)
                .is_err()
        );
        assert!(validate_warp_collective_participants(
            WarpMask::FULL,
            &WarpValue::splat(0),
            &label,
        )
        .is_err());
        let missing_self = WarpValue::splat(!(1_u32 << 7));
        let error = validate_warp_collective_participants(WarpMask::FULL, &missing_self, &label)
            .unwrap_err();
        assert!(error.to_string().contains("executing lane 7 is absent"));
        let inconsistent =
            WarpValue::from_fn(|lane| if lane == 7 { !(1_u32 << 0) } else { u32::MAX });
        let error = validate_warp_collective_participants(WarpMask::FULL, &inconsistent, &label)
            .unwrap_err();
        assert!(error.to_string().contains("participant masks disagree"));
    }

    #[test]
    fn uniform_values_preserve_diagnostics() {
        let uniform = WarpValue::splat(7_i64);
        assert_eq!(
            require_uniform_i64(&uniform, WarpMask::FULL, "field").unwrap(),
            7
        );
        let error = require_uniform_i64(&uniform, WarpMask::EMPTY, "field").unwrap_err();
        assert_eq!(error.to_string(), "field has no active lane");
        let disagreeing = WarpValue::from_fn(|lane| if lane == 3 { 9 } else { 7 });
        let error = require_uniform_i64(&disagreeing, WarpMask::FULL, "field").unwrap_err();
        assert_eq!(
            error.to_string(),
            "field must agree across active lanes: lane 3 has 9, expected 7"
        );
    }

    #[test]
    fn float64_warp_reductions_preserve_cuda_zero_and_nan_selection() {
        let nan_a = f64::from_bits(0x7ff8_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
        let values = WarpValue::from_fn(|lane| match lane {
            0 => 0.0,
            1 => -0.0,
            4 => nan_a,
            5 => nan_b,
            _ => lane as f64,
        });

        let maximum = warp_reduce_max(WarpMask::FULL, &values, 2).unwrap();
        let minimum = warp_reduce_min(WarpMask::FULL, &values, 2).unwrap();

        assert_eq!(maximum[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(maximum[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(minimum[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(minimum[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(maximum[4].to_bits(), nan_b.to_bits());
        assert_eq!(maximum[5].to_bits(), nan_a.to_bits());
        assert_eq!(minimum[4].to_bits(), nan_b.to_bits());
        assert_eq!(minimum[5].to_bits(), nan_a.to_bits());
    }

    #[test]
    fn checked_fns_applies_lane_values_and_rejects_invalid_bases() {
        let masks = WarpValue::splat(0b10110_u32);
        let bases = WarpValue::splat(1_u32);
        let offsets = WarpValue::from_fn(|lane| if lane % 2 == 0 { 1 } else { 2 });
        let result = warp_fns_b32(WarpMask::FULL, &masks, &bases, &offsets).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(result[lane], if lane % 2 == 0 { 1 } else { 2 });
        }

        let invalid_bases = WarpValue::from_fn(|lane| if lane == 7 { 32 } else { 0 });
        let error = warp_fns_b32(WarpMask::FULL, &masks, &invalid_bases, &offsets).unwrap_err();
        assert_eq!(
            error.to_string(),
            "fns.b32 base is outside the defined 0..31 range"
        );
    }

    #[test]
    fn butterfly_reductions_support_integer_and_float_groups() {
        let integers = WarpValue::from_fn(|lane| lane as u32);
        let sums = warp_reduce_sum(WarpMask::FULL, &integers, 8).unwrap();
        let maxima = warp_reduce_max(WarpMask::FULL, &integers, 8).unwrap();
        let minima = warp_reduce_min(WarpMask::FULL, &integers, 8).unwrap();
        for lane in 0..WARP_SIZE {
            let group = lane / 8;
            assert_eq!(sums[lane], (group * 64 + 28) as u32);
            assert_eq!(maxima[lane], (group * 8 + 7) as u32);
            assert_eq!(minima[lane], (group * 8) as u32);
        }

        let floats = WarpValue::from_fn(|lane| lane as f32 - 16.0);
        let sums = warp_reduce_sum(WarpMask::FULL, &floats, 32).unwrap();
        let maxima = warp_reduce_max(WarpMask::FULL, &floats, 32).unwrap();
        let minima = warp_reduce_min(WarpMask::FULL, &floats, 32).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(sums[lane], -16.0);
            assert_eq!(maxima[lane], 15.0);
            assert_eq!(minima[lane], -16.0);
        }

        let doubles = WarpValue::from_fn(|lane| lane as f64 * 0.5 - 8.0);
        let sums = warp_reduce_sum(WarpMask::FULL, &doubles, 32).unwrap();
        let maxima = warp_reduce_max(WarpMask::FULL, &doubles, 32).unwrap();
        let minima = warp_reduce_min(WarpMask::FULL, &doubles, 32).unwrap();
        for lane in 0..WARP_SIZE {
            assert_eq!(sums[lane], -8.0);
            assert_eq!(maxima[lane], 7.5);
            assert_eq!(minima[lane], -8.0);
        }
    }

    #[test]
    fn butterfly_reductions_require_full_warps_and_power_of_two_widths() {
        let values = WarpValue::splat(1_u32);
        let partial = WarpMask::from_lanes(0..16).unwrap();
        let error = warp_reduce_sum(partial, &values, 8).unwrap_err();
        assert!(matches!(
            error.kind(),
            crate::EngineErrorKind::WarpCollectiveDivergence {
                operation,
                active_mask: 0x0000_ffff,
            } if operation == "cuda_warp_reduce"
        ));
        assert!(error
            .to_string()
            .contains("cuda_warp_reduce requires all 32 lanes"));

        let error = warp_reduce_sum(WarpMask::FULL, &values, 3).unwrap_err();
        assert!(error.to_string().contains("width must be a power of two"));
    }
}
