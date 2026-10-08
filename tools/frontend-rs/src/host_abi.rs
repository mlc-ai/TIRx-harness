//! `host_abi.HostAbiContract`: the public host binding names of one module.
//!
//! A kernel manifest names each parameter the way its own PrimFunc spells it.
//! This module is the one place that turns those local names into the public
//! keys the host ABI exposes, and rejects a module no such ABI can describe.

use crate::analyze::util::{json_object, json_strings};
use std::collections::{BTreeMap, HashMap};

use crate::analyze::util::{sorted_unique, Json};
use crate::tables::phase_binding_name;

/// One public binding slot, as `host_abi.HostBindingSlot` records it.
struct Slot {
    kernel_index: i64,
    /// The Python `BindingKind`.
    kind: &'static str,
    local_name: String,
    canonical_name: String,
    local_aliases: Vec<String>,
    dtype: Option<String>,
}

impl Slot {
    fn json(&self) -> Json {
        json_object(vec![
            ("kernel_index", Json::from(self.kernel_index)),
            ("kind", Json::from(self.kind)),
            ("local_name", Json::String(self.local_name.clone())),
            ("canonical_name", Json::String(self.canonical_name.clone())),
            ("local_aliases", json_strings(self.local_aliases.clone())),
            ("dtype", self.dtype.clone().map_or(Json::Null, Json::String)),
        ])
    }
}

/// The slot table of one module, keyed the three ways the contract reads it.
struct Slots {
    kernel_count: i64,
    slots: Vec<Slot>,
    by_canonical: HashMap<String, usize>,
    phase_aliases: BTreeMap<(i64, String), Vec<String>>,
}

impl Slots {
    /// Record one slot, or reject the canonical name two slots would share.
    fn add(&mut self, slot: Slot) -> Result<(), String> {
        if let Some(previous) = self.by_canonical.get(&slot.canonical_name) {
            let previous_kind = self.slots[*previous].kind;
            let name = format!("{:?}", &slot.canonical_name);
            return Err(if previous_kind == slot.kind {
                format!(
                    "host binding {name} identifies multiple {} slots",
                    slot.kind
                )
            } else {
                format!(
                    "host binding {name} identifies both {previous_kind} and {} slots",
                    slot.kind
                )
            });
        }
        for alias in &slot.local_aliases {
            self.phase_aliases
                .entry((slot.kernel_index, alias.clone()))
                .or_default()
                .push(slot.canonical_name.clone());
        }
        self.by_canonical
            .insert(slot.canonical_name.clone(), self.slots.len());
        self.slots.push(slot);
        Ok(())
    }

    /// The slots one kernel-local alias names, sorted, keeping those `keep` accepts.
    fn targets(&self, kernel_index: i64, alias: &str, keep: impl Fn(&Slot) -> bool) -> Vec<String> {
        let mut targets: Vec<String> = self
            .phase_aliases
            .get(&(kernel_index, alias.to_owned()))
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter(|target| keep(&self.slots[self.by_canonical[target.as_str()]]))
            .cloned()
            .collect();
        targets.sort();
        targets
    }
}

/// The host ABI one module manifest determines.
pub struct Contract {
    kernel_count: i64,
    slots: Vec<Slot>,
    /// Each implicit TensorMap binding and the host buffer binding behind it.
    implicit_bases: Vec<(String, String)>,
    alias_targets: BTreeMap<String, Vec<String>>,
}

impl Contract {
    /// The facts `host_abi.HostAbiContract` reads.
    fn facts(&self) -> Json {
        json_object(vec![
            ("kernel_count", Json::from(self.kernel_count)),
            (
                "slots",
                Json::Array(self.slots.iter().map(Slot::json).collect()),
            ),
            (
                "implicit_tensor_maps",
                Json::Array(
                    self.implicit_bases
                        .iter()
                        .map(|(canonical_name, base)| {
                            json_object(vec![
                                ("canonical_name", Json::String(canonical_name.clone())),
                                ("base_canonical_name", Json::String(base.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "alias_targets",
                string_map(&self.alias_targets, json_strings),
            ),
        ])
    }
}

/// The payload element: one module's host binding facts, or its rejection.
pub fn facts_json(contract: &Result<Contract, String>) -> String {
    match contract {
        Ok(contract) => contract.facts().to_string(),
        Err(message) => json_object(vec![("error", Json::String(message.clone()))]).to_string(),
    }
}

/// The contract of `manifest`, or the message of the `HostAbiError` rejecting it.
pub fn contract(manifest: &Json) -> Result<Contract, String> {
    let kernels = manifest
        .get("kernels")
        .and_then(Json::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if kernels.is_empty() {
        return Err("NumSim host ABI requires at least one kernel".to_owned());
    }
    let mut table = Slots {
        kernel_count: kernels.len() as i64,
        slots: Vec::new(),
        by_canonical: HashMap::new(),
        phase_aliases: BTreeMap::new(),
    };
    for (index, kernel) in kernels.iter().enumerate() {
        add_kernel_slots(&mut table, index as i64, kernel)?;
    }
    let implicit_bases = implicit_bases(&table, kernels)?;
    let alias_targets = alias_targets(&table);
    // A canonical key must always identify exactly one slot. An alias
    // collision inside a phase has no unambiguous spelling in this ABI.
    for slot in &table.slots {
        let targets = alias_targets
            .get(&slot.canonical_name)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if targets != [slot.canonical_name.as_str()] {
            return Err(format!(
                "canonical host binding {:?} is ambiguous across {:?}",
                &slot.canonical_name, targets
            ));
        }
    }
    Ok(Contract {
        kernel_count: table.kernel_count,
        slots: table.slots,
        implicit_bases,
        alias_targets,
    })
}

/// The buffer, pointer, scalar and TensorMap parameters of one kernel, in the
/// order the host ABI lists them.
fn add_kernel_slots(table: &mut Slots, kernel_index: i64, kernel: &Json) -> Result<(), String> {
    // A TensorMap binds a descriptor rather than an element type, so unlike the
    // other kinds it carries no host dtype.
    for (key, kind, typed) in [
        ("buffers", "buffer", true),
        ("pointers", "pointer", true),
        ("scalars", "scalar", true),
        ("tensor_maps", "tensor_map", false),
    ] {
        for entry in kernel
            .get(key)
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let local_name = text(entry, "name");
            // A buffer parameter's manifest `parameter` is a second local alias.
            let parameter = entry.get("parameter").and_then(Json::as_str);
            let mut local_aliases = vec![local_name.clone()];
            if let Some(parameter) = parameter.filter(|alias| *alias != local_name) {
                local_aliases.push(parameter.to_owned());
            }
            table.add(Slot {
                kernel_index,
                kind,
                canonical_name: phase_binding_name(kernel_index, &local_name, table.kernel_count),
                local_name,
                local_aliases,
                dtype: typed
                    .then(|| entry.get("dtype").and_then(Json::as_str))
                    .flatten()
                    .map(str::to_owned),
            })?;
        }
    }
    Ok(())
}

/// Every implicit TensorMap binding and the one host buffer binding it wraps.
fn implicit_bases(table: &Slots, kernels: &[Json]) -> Result<Vec<(String, String)>, String> {
    let mut bases = Vec::new();
    for (index, kernel) in kernels.iter().enumerate() {
        let kernel_index = index as i64;
        for entry in kernel
            .get("tensor_maps")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(base_buffer) = entry.get("base_buffer").and_then(Json::as_str) else {
                continue;
            };
            let name = text(entry, "name");
            let targets = table.targets(kernel_index, base_buffer, |slot| {
                matches!(slot.kind, "buffer" | "pointer")
            });
            if targets.len() != 1 {
                return Err(format!(
                    "implicit TensorMap {:?} base {:?} resolves to {:?}, expected one host buffer",
                    &name, base_buffer, &targets
                ));
            }
            bases.push((
                phase_binding_name(kernel_index, &name, table.kernel_count),
                targets[0].clone(),
            ));
        }
    }
    Ok(bases)
}

/// Every public spelling of a slot: its phase-qualified alias, and in a
/// multi-kernel module the bare local alias too.
fn alias_targets(table: &Slots) -> BTreeMap<String, Vec<String>> {
    let mut targets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for slot in &table.slots {
        for alias in &slot.local_aliases {
            let public = phase_binding_name(slot.kernel_index, alias, table.kernel_count);
            targets
                .entry(public)
                .or_default()
                .push(slot.canonical_name.clone());
            if table.kernel_count > 1 {
                targets
                    .entry(alias.clone())
                    .or_default()
                    .push(slot.canonical_name.clone());
            }
        }
    }
    for names in targets.values_mut() {
        *names = sorted_unique(names.clone());
    }
    targets
}

fn string_map(
    values: &BTreeMap<String, Vec<String>>,
    render: impl Fn(Vec<String>) -> Json,
) -> Json {
    Json::Object(
        values
            .iter()
            .map(|(key, value)| (key.clone(), render(value.clone())))
            .collect(),
    )
}

fn text(entry: &Json, key: &str) -> String {
    entry
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_owned()
}
