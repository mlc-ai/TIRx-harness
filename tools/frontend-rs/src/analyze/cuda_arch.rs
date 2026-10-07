//! The authored CUDA target attribute.

use tvm::tirx::PrimFunc;
use tvm::tvm_ffi::String as FfiString;

use super::util::{ffi_text, not_covered, AResult};

pub const CUDA_ARCH_ATTR: &str = "tirx.cuda_arch";
/// `cuda_arch`: the exact authored CUDA architecture retained on a PrimFunc.
pub fn cuda_arch(func: &PrimFunc) -> AResult<Option<String>> {
    match func.attrs.dict.get(&FfiString::from(CUDA_ARCH_ATTR))? {
        None => Ok(None),
        Some(value) => match FfiString::try_from(value) {
            Ok(text) => Ok(Some(ffi_text(&text))),
            Err(_) => not_covered("tirx.cuda_arch attribute is not a string"),
        },
    }
}
