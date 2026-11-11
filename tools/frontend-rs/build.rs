//! Record the locked tvm-tirx version for the frontend identity.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let lock = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = fs::read_to_string(&lock).expect("Cargo.lock is required to build the frontend");
    let package = text
        .split("[[package]]")
        .find(|package| package.contains("\nname = \"tvm-tirx\"\n"))
        .expect("Cargo.lock does not lock tvm-tirx");
    let field = |name: &str| {
        package.lines().find_map(|line| {
            line.strip_prefix(&format!("{name} = "))
                .map(|value| value.trim_matches('"').to_string())
        })
    };
    let version = field("version").expect("tvm-tirx lock entry has no version");
    println!("cargo:rustc-env=NUMSIM_TVM_TIRX_VERSION={version}");
}
