//! The authored CUDA architecture.

use tvm_tirx::ir::RegionStmtObj;
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::{Any, ObjectRefCore, String as FfiString};

use super::topology::is_launch_entry;
use super::util::{ffi_text, not_covered, unsupported, AResult};

/// Where the architecture is authored: `backend_config["cuda"]["arch"]`.
pub const CUDA_ARCH_FIELD: &str = "backend_config[\"cuda\"][\"arch\"]";

/// `cuda_arch`: the CUDA architecture the `backend_config` of the kernel's
/// device entry, or else of the function, selects.
pub fn cuda_arch(func: &Function) -> AResult<Option<String>> {
    let key = FfiString::from("backend_config");
    let mut configs: Vec<Any> = Vec::new();
    if let Some(body) = &func.body {
        for statement in body.seq.iter() {
            if let Some(region) = statement.as_node::<RegionStmtObj>() {
                if is_launch_entry(region) {
                    configs.extend(region.attrs.dict.get(&key)?);
                }
            }
        }
    }
    configs.extend(func.attrs.dict.get(&key)?);
    for config in configs {
        let Ok(text) = FfiString::try_from(config) else {
            return not_covered("backend_config attribute is not a string");
        };
        let json: serde_json::Value = match serde_json::from_str(&ffi_text(&text)) {
            Ok(json) => json,
            Err(error) => return unsupported(format!("backend_config is not JSON: {error}")),
        };
        match json.get("cuda").and_then(|cuda| cuda.get("arch")) {
            None => continue,
            Some(serde_json::Value::String(arch)) => return Ok(Some(arch.clone())),
            Some(_) => return unsupported(format!("{CUDA_ARCH_FIELD} must be a string")),
        }
    }
    Ok(None)
}
