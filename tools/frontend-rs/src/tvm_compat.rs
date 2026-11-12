//! Checked access to fields whose native representation changed in Apache TVM.

use tvm_tirx::ir::IntImmObj;
use tvm_tirx::tvm_ffi::{BigInt, Error, Result, VALUE_ERROR};

fn integer_value(value: &BigInt) -> Result<i128> {
    value.to_i128().ok_or_else(|| {
        Error::new(
            VALUE_ERROR,
            "integer literal exceeds the frontend's 128-bit conversion range",
        )
    })
}

pub fn int_value(value: &IntImmObj) -> Result<i64> {
    integer_value(&value.value)?.try_into().map_err(|_| {
        Error::new(
            VALUE_ERROR,
            "integer literal does not fit the frontend's signed 64-bit contract",
        )
    })
}

pub fn int_bits(value: &IntImmObj) -> Result<u64> {
    let value = integer_value(&value.value)?;
    if (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(&value) {
        Ok(value as u64)
    } else {
        Err(Error::new(
            VALUE_ERROR,
            "integer literal does not fit a 64-bit word",
        ))
    }
}

/// Checked object-handle conversion through the FFI dynamic type system.
pub trait TryCast: tvm_tirx::tvm_ffi::ObjectRefCore {
    fn try_cast<T: TryFrom<tvm_tirx::tvm_ffi::Any, Error = Error>>(self) -> Result<T> {
        T::try_from(tvm_tirx::tvm_ffi::Any::from(
            tvm_tirx::tvm_ffi::AnyView::from(&self),
        ))
    }
}
impl<T: tvm_tirx::tvm_ffi::ObjectRefCore> TryCast for T {}
