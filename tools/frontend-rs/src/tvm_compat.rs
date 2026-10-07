//! Checked access to fields whose native representation changed in Apache TVM.

use std::slice;

use tvm::ir::IntImmObj;
use tvm::tvm_ffi::{Any, Error, Result, VALUE_ERROR};

#[repr(C)]
struct ByteArray {
    data: *const i64,
    size: usize,
}

extern "C" {
    fn TVMFFIBigIntGetContentByteArray(value: *const Any) -> ByteArray;
}

fn integer_value(value: &Any) -> Result<i128> {
    let bytes = unsafe { TVMFFIBigIntGetContentByteArray(value) };
    if bytes.size % std::mem::size_of::<i64>() != 0 {
        return Err(Error::new(
            VALUE_ERROR,
            "BigInt content is not a whole number of 64-bit words",
            "",
        ));
    }
    let word_count = bytes.size / std::mem::size_of::<i64>();
    if word_count > 2 {
        return Err(Error::new(
            VALUE_ERROR,
            "integer literal exceeds the frontend's 128-bit conversion range",
            "",
        ));
    }
    let words = if word_count == 0 {
        &[]
    } else {
        if bytes.data.is_null() {
            return Err(Error::new(VALUE_ERROR, "null BigInt content", ""));
        }
        unsafe { slice::from_raw_parts(bytes.data, word_count) }
    };
    Ok(match words {
        [] => 0,
        [low] => i128::from(*low),
        [low, high] => (i128::from(*high) << 64) | i128::from(*low as u64),
        _ => unreachable!(),
    })
}

pub fn int_value(value: &IntImmObj) -> Result<i64> {
    integer_value(&value.value)?.try_into().map_err(|_| {
        Error::new(
            VALUE_ERROR,
            "integer literal does not fit the frontend's signed 64-bit contract",
            "",
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
            "",
        ))
    }
}

const _: () = {
    assert!(std::mem::size_of::<Any>() == 16);
};
