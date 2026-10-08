//! Precompiled lane-wise frontend expressions. Keep concrete call boundaries
//! so generated kernels do not instantiate a closure for every expression.
use crate::warp_value::WarpValue;

macro_rules! binary {
    ($name:ident, $lhs:ty, $rhs:ty, $out:ty, |$a:ident, $b:ident| $body:expr) => {
        pub mod $name {
            use super::*;
            #[inline]
            pub fn vv(lhs: &WarpValue<$lhs>, rhs: &WarpValue<$rhs>) -> WarpValue<$out> {
                WarpValue::from_fn_copy(|lane| {
                    let $a = lhs[lane];
                    let $b = rhs[lane];
                    $body
                })
            }
            #[inline]
            pub fn vs(lhs: &WarpValue<$lhs>, rhs: $rhs) -> WarpValue<$out> {
                WarpValue::from_fn_copy(|lane| {
                    let $a = lhs[lane];
                    let $b = rhs;
                    $body
                })
            }
            #[inline]
            pub fn sv(lhs: $lhs, rhs: &WarpValue<$rhs>) -> WarpValue<$out> {
                WarpValue::from_fn_copy(|lane| {
                    let $a = lhs;
                    let $b = rhs[lane];
                    $body
                })
            }
        }
    };
}
macro_rules! unary {
    ($name:ident, $source:ty, $target:ty, |$a:ident| $body:expr) => {
        #[inline]
        pub fn $name(value: &WarpValue<$source>) -> WarpValue<$target> {
            WarpValue::from_fn_copy(|lane| {
                let $a = value[lane];
                $body
            })
        }
    };
}
macro_rules! integers {
    ($($ty:ident),*) => { $(
        pub mod $ty {
            use super::*;
            binary!(add, $ty, $ty, $ty, |a, b| a.wrapping_add(b));
            binary!(sub, $ty, $ty, $ty, |a, b| a.wrapping_sub(b));
            binary!(mul, $ty, $ty, $ty, |a, b| a.wrapping_mul(b));
            binary!(min, $ty, $ty, $ty, |a, b| a.min(b));
            binary!(max, $ty, $ty, $ty, |a, b| a.max(b));
            binary!(bitwise_and, $ty, $ty, $ty, |a, b| a & b);
            binary!(bitwise_or, $ty, $ty, $ty, |a, b| a | b);
            binary!(bitwise_xor, $ty, $ty, $ty, |a, b| a ^ b);
            binary!(shift_left, $ty, u32, $ty, |a, b| a.wrapping_shl(b));
            binary!(shift_right, $ty, u32, $ty, |a, b| a.wrapping_shr(b));
            pub mod cast {
                use super::*;
                unary!(i8, $ty, i8, |a| a as i8);
                unary!(i16, $ty, i16, |a| a as i16);
                unary!(i32, $ty, i32, |a| a as i32);
                unary!(i64, $ty, i64, |a| a as i64);
                unary!(u8, $ty, u8, |a| a as u8);
                unary!(u16, $ty, u16, |a| a as u16);
                unary!(u32, $ty, u32, |a| a as u32);
                unary!(u64, $ty, u64, |a| a as u64);
            }
            pub mod reinterpret {
                use super::*;
                unary!(f32, $ty, f32, |a| <f32>::from_bits(a as u32));
                unary!(f64, $ty, f64, |a| <f64>::from_bits(a as u64));
            }
        }
    )* };
}
integers!(i8, i16, i32, i64, u8, u16, u32, u64);
macro_rules! floats {
    ($($ty:ident),*) => { $(
        pub mod $ty {
            use super::*;
            binary!(add, $ty, $ty, $ty, |a, b| a + b);
            binary!(sub, $ty, $ty, $ty, |a, b| a - b);
            binary!(mul, $ty, $ty, $ty, |a, b| a * b);
            binary!(div, $ty, $ty, $ty, |a, b| a / b);
            pub mod reinterpret {
                use super::*;
                unary!(i8, $ty, i8, |a| a.to_bits() as i8);
                unary!(i16, $ty, i16, |a| a.to_bits() as i16);
                unary!(i32, $ty, i32, |a| a.to_bits() as i32);
                unary!(i64, $ty, i64, |a| a.to_bits() as i64);
                unary!(u8, $ty, u8, |a| a.to_bits() as u8);
                unary!(u16, $ty, u16, |a| a.to_bits() as u16);
                unary!(u32, $ty, u32, |a| a.to_bits() as u32);
                unary!(u64, $ty, u64, |a| a.to_bits() as u64);
            }
        }
    )* };
}
floats!(f32, f64);
binary!(make_float2, f32, f32, u64, |a, b| {
    crate::scalar::make_float2(a, b)
});
unary!(float2_x, u64, f32, |a| crate::scalar::float2_x(a));
unary!(float2_y, u64, f32, |a| crate::scalar::float2_y(a));
unary!(float_as_uint, f32, u32, |a| a.to_bits());
unary!(uint_as_float, u32, f32, |a| <f32>::from_bits(a));
unary!(fp16_bits_to_f32, u16, f32, |a| {
    crate::numpy_backend::fp16_bits_to_f32(a)
});
unary!(bf16_bits_to_f32, u16, f32, |a| {
    crate::numpy_backend::bf16_bits_to_f32(a)
});
unary!(f32_to_fp16_bits, f32, u16, |a| {
    crate::numpy_backend::f32_to_fp16_bits(a)
});
unary!(f32_to_bf16_bits, f32, u16, |a| {
    crate::numpy_backend::f32_to_bf16_bits(a)
});
