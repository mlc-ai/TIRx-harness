//! Record the locked tvm-tirx version, which the frontend identity reports.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let lock = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = fs::read_to_string(&lock).expect("Cargo.lock is required to build the frontend");
    let version = text
        .split("[[package]]")
        .find(|package| package.contains("\nname = \"tvm-tirx\"\n"))
        .and_then(|package| package.lines().find_map(|line| line.strip_prefix("version = ")))
        .map(|version| version.trim_matches('"').to_string())
        .expect("Cargo.lock does not lock tvm-tirx");
    println!("cargo:rustc-env=NUMSIM_TVM_TIRX_VERSION={version}");
}
