use std::ops::{Index, IndexMut};

use crate::{WarpMask, WARP_SIZE};

/// One value per lane in a warp execution unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarpValue<T> {
    lanes: [T; WARP_SIZE],
}

impl<T> WarpValue<T> {
    pub const fn from_lanes(lanes: [T; WARP_SIZE]) -> Self {
        Self { lanes }
    }

    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Self {
            lanes: std::array::from_fn(f),
        }
    }

    /// Construct copyable lane values without the generic machinery used by
    /// `std::array::from_fn`.
    pub fn from_fn_copy(mut f: impl FnMut(usize) -> T) -> Self
    where
        T: Copy,
    {
        let first = f(0);
        let mut lanes = [first; WARP_SIZE];
        let mut lane = 1;
        while lane < WARP_SIZE {
            lanes[lane] = f(lane);
            lane += 1;
        }
        Self { lanes }
    }

    pub const fn lanes(&self) -> &[T; WARP_SIZE] {
        &self.lanes
    }

    pub fn lanes_mut(&mut self) -> &mut [T; WARP_SIZE] {
        &mut self.lanes
    }

    pub fn masked_assign(&mut self, mask: WarpMask, source: &Self)
    where
        T: Clone,
    {
        for lane in mask {
            self.lanes[lane] = source.lanes[lane].clone();
        }
    }

    pub fn masked_fill(&mut self, mask: WarpMask, value: T)
    where
        T: Clone,
    {
        for lane in mask {
            self.lanes[lane] = value.clone();
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(usize, T) -> U) -> WarpValue<U> {
        let mut lanes = self.lanes.into_iter().enumerate();
        WarpValue::from_fn(|_| {
            let (lane, value) = lanes.next().expect("warp lane count is fixed");
            f(lane, value)
        })
    }

    pub fn zip_map<U, V>(
        &self,
        other: &WarpValue<U>,
        mut f: impl FnMut(usize, &T, &U) -> V,
    ) -> WarpValue<V> {
        WarpValue::from_fn(|lane| f(lane, &self.lanes[lane], &other.lanes[lane]))
    }

    pub fn to_mask(&self, mut predicate: impl FnMut(usize, &T) -> bool) -> WarpMask {
        let mut bits = 0_u32;
        for lane in 0..WARP_SIZE {
            if predicate(lane, &self.lanes[lane]) {
                bits |= 1_u32 << lane;
            }
        }
        WarpMask::from_bits(bits)
    }
}

impl<T: Clone> WarpValue<T> {
    pub fn splat(value: T) -> Self {
        Self::from_fn(|_| value.clone())
    }
}

impl<T> Index<usize> for WarpValue<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.lanes[index]
    }
}

impl<T> IndexMut<usize> for WarpValue<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.lanes[index]
    }
}

impl<T> From<[T; WARP_SIZE]> for WarpValue<T> {
    fn from(value: [T; WARP_SIZE]) -> Self {
        Self::from_lanes(value)
    }
}

impl<T> From<WarpValue<T>> for [T; WARP_SIZE] {
    fn from(value: WarpValue<T>) -> Self {
        value.lanes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_assignment_changes_only_active_lanes() {
        let mut destination = WarpValue::from_fn(|lane| lane as i32);
        let source = WarpValue::from_fn(|lane| 1000 + lane as i32);
        let mask = WarpMask::from_lanes([0, 3, 17, 31]).unwrap();

        destination.masked_assign(mask, &source);

        for lane in 0..WARP_SIZE {
            let expected = if mask.contains(lane) {
                1000 + lane as i32
            } else {
                lane as i32
            };
            assert_eq!(destination[lane], expected, "lane {lane}");
        }
    }

    #[test]
    fn masked_fill_and_map_preserve_lane_identity() {
        let mut value = WarpValue::splat(0_i32);
        value.masked_fill(WarpMask::from_lanes([1, 2, 31]).unwrap(), 5);
        let mapped = value.map(|lane, item| item + lane as i32);
        assert_eq!(mapped[0], 0);
        assert_eq!(mapped[1], 6);
        assert_eq!(mapped[2], 7);
        assert_eq!(mapped[31], 36);
    }

    #[test]
    fn zip_map_and_to_mask_support_lane_wise_codegen() {
        let left = WarpValue::from_fn(|lane| lane as i32);
        let right = WarpValue::from_fn(|lane| (lane * 2) as i32);
        let sum = left.zip_map(&right, |_, lhs, rhs| lhs + rhs);
        assert_eq!(sum[0], 0);
        assert_eq!(sum[7], 21);
        assert_eq!(sum[31], 93);

        let mask = sum.to_mask(|lane, value| lane % 2 == 0 && *value >= 48);
        assert_eq!(
            mask.iter().collect::<Vec<_>>(),
            vec![16, 18, 20, 22, 24, 26, 28, 30]
        );
    }
}
