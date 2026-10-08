use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub};

pub const WARP_SIZE: usize = 32;

/// Active lanes for one simulated GPU warp.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WarpMask(u32);

impl WarpMask {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u32::MAX);

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub fn from_lanes(lanes: impl IntoIterator<Item = usize>) -> Result<Self, MaskError> {
        let mut mask = Self::EMPTY;
        for lane in lanes {
            mask.insert(lane)?;
        }
        Ok(mask)
    }

    pub fn from_predicate(mut predicate: impl FnMut(usize) -> bool) -> Self {
        let mut bits = 0_u32;
        let mut lane = 0_usize;
        while lane < WARP_SIZE {
            if predicate(lane) {
                bits |= 1_u32 << lane;
            }
            lane += 1;
        }
        Self(bits)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_full(self) -> bool {
        self.0 == u32::MAX
    }

    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub const fn contains(self, lane: usize) -> bool {
        lane < WARP_SIZE && (self.0 & (1_u32 << lane)) != 0
    }

    pub fn insert(&mut self, lane: usize) -> Result<bool, MaskError> {
        let bit = lane_bit(lane)?;
        let was_present = self.0 & bit != 0;
        self.0 |= bit;
        Ok(!was_present)
    }

    pub fn remove(&mut self, lane: usize) -> Result<bool, MaskError> {
        let bit = lane_bit(lane)?;
        let was_present = self.0 & bit != 0;
        self.0 &= !bit;
        Ok(was_present)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn first_active(self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }

    pub const fn iter(self) -> ActiveLanes {
        ActiveLanes { remaining: self.0 }
    }
}

impl fmt::Debug for WarpMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WarpMask({:#034b})", self.0)
    }
}

impl BitAnd for WarpMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl BitAndAssign for WarpMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for WarpMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitOrAssign for WarpMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for WarpMask {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for WarpMask {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for WarpMask {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Sub for WarpMask {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.difference(rhs)
    }
}

impl IntoIterator for WarpMask {
    type Item = usize;
    type IntoIter = ActiveLanes;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveLanes {
    remaining: u32,
}

impl Iterator for ActiveLanes {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let lane = self.remaining.trailing_zeros() as usize;
        self.remaining &= self.remaining - 1;
        Some(lane)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining.count_ones() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for ActiveLanes {}
impl FusedIterator for ActiveLanes {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskError {
    LaneOutOfRange { lane: usize },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaneOutOfRange { lane } => {
                write!(f, "lane {lane} is outside a {WARP_SIZE}-lane warp")
            }
        }
    }
}

impl Error for MaskError {}

fn lane_bit(lane: usize) -> Result<u32, MaskError> {
    if lane >= WARP_SIZE {
        return Err(MaskError::LaneOutOfRange { lane });
    }
    Ok(1_u32 << lane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_construction_and_iteration_are_ordered() {
        let mask = WarpMask::from_lanes([31, 0, 7, 7]).unwrap();
        assert_eq!(mask.bits(), (1 << 31) | (1 << 7) | 1);
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.first_active(), Some(0));
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 7, 31]);
        assert!(mask.contains(31));
        assert!(!mask.contains(30));
        assert!(!mask.contains(32));
    }

    #[test]
    fn mask_algebra_matches_set_algebra() {
        let even = WarpMask::from_lanes((0..WARP_SIZE).step_by(2)).unwrap();
        let low = WarpMask::from_bits(0xff);
        assert_eq!((even & low).iter().collect::<Vec<_>>(), vec![0, 2, 4, 6]);
        assert_eq!((low - even).iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
        assert_eq!(even | !even, WarpMask::FULL);
        assert_eq!(even ^ even, WarpMask::EMPTY);
    }

    #[test]
    fn insert_remove_and_bounds_are_checked() {
        let mut mask = WarpMask::EMPTY;
        assert!(mask.insert(3).unwrap());
        assert!(!mask.insert(3).unwrap());
        assert!(mask.remove(3).unwrap());
        assert!(!mask.remove(3).unwrap());
        assert_eq!(
            mask.insert(WARP_SIZE),
            Err(MaskError::LaneOutOfRange { lane: WARP_SIZE })
        );
    }
}
