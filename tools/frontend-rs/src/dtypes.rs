//! Scalar storage widths and frontend capabilities from one packaged registry.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::analyze::util::Json;

struct ScalarDType {
    name: String,
    bits: i64,
    capabilities: HashSet<String>,
}

fn registry() -> &'static [ScalarDType] {
    static REGISTRY: OnceLock<Vec<ScalarDType>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let data: Json = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dtype_registry.json"
        )))
        .expect("packaged dtype registry is valid JSON");
        data["types"]
            .as_object()
            .expect("dtype registry types")
            .iter()
            .map(|(name, entry)| {
                let class = entry["class"].as_str().expect("dtype class");
                let capabilities = data["classes"][class]
                    .as_array()
                    .expect("defined dtype class")
                    .iter()
                    .map(|value| value.as_str().expect("dtype capability").to_owned())
                    .collect();
                ScalarDType {
                    name: name.clone(),
                    bits: entry["bits"].as_i64().expect("dtype bit width"),
                    capabilities,
                }
            })
            .collect()
    })
}

fn names(capability: &str) -> Vec<&'static str> {
    let mut names: Vec<_> = registry()
        .iter()
        .filter(|entry| entry.capabilities.contains(capability))
        .map(|entry| entry.name.as_str())
        .collect();
    names.sort_unstable();
    names
}

pub fn with_capability(capability: &str) -> HashSet<String> {
    names(capability).into_iter().map(str::to_owned).collect()
}

/// Ordinary byte-addressed storage excludes sub-byte scalar formats.
pub fn scalar_bits() -> HashMap<String, i64> {
    registry()
        .iter()
        .filter(|entry| entry.bits >= 8)
        .map(|entry| (entry.name.clone(), entry.bits))
        .collect()
}

pub fn integer_dtypes() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| names("integer"))
}

pub fn predicate_dtypes() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| names("predicate"))
}
