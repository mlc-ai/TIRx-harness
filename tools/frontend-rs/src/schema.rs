//! Compile-time tables: the closed tables only the frontend reads, the tables
//! Python shares with it (exported once per process), and the operation
//! registry (`registry.rs`) whose coverage they require.

use crate::analyze::util::{json_object, json_strings};
use std::collections::{HashMap, HashSet};

use tvm::tvm_ffi::{Any, Array, Map, Result, String as FfiString};

use crate::analyze::util::{ffi_error, Json};
use crate::registry::{self, OpRow};

/// One registry row as the Python registry export spells it.
fn row_json(ir_name: &str, row: &OpRow) -> Json {
    json_object(vec![
        ("ir_name", Json::from(ir_name)),
        ("family", Json::from(row.family)),
        ("support", Json::from(row.support)),
        ("reason", Json::from(row.reason)),
        ("suspends", Json::Bool(row.suspends)),
    ])
}

#[derive(Default)]
pub struct Schema {
    pub high_precision: bool,
    pub registered_ops: HashMap<String, OpRow>,
    pub supported_cast_dtypes: HashSet<String>,
    pub supported_buffer_dtypes: HashSet<String>,
    pub supported_parameter_scalar_dtypes: HashSet<String>,
    pub scalar_dtype_bits: HashMap<String, i64>,
    pub numerically_irrelevant_attrs: HashSet<String>,
    pub semantic_attrs: HashSet<String>,
    pub serial_loop_annotations: HashSet<String>,
    pub runtime_frontend_node_kinds: HashSet<String>,
    pub compile_only_frontend_node_kinds: HashSet<String>,
    pub integer_binary_node_kinds: HashSet<String>,
    pub register_owner_axes: HashSet<String>,
    pub register_scopes: HashSet<String>,
    pub extractable_vector_dtypes: HashSet<String>,
    /// `dtype_abi.vector_dtype_abis`: dtype -> (element dtype, lanes, element
    /// bits, total bits) for every vector dtype NumSim can store.
    pub vector_dtype_abis: HashMap<String, (String, i64, i64, i64)>,
    pub reinterpret_raw_scalars: HashSet<String>,
    pub reinterpret_identity_scalars: HashSet<String>,
    pub reinterpret_numeric_decode_pairs: HashSet<(String, String)>,
    pub call_scalars: HashSet<String>,
}

fn strings(map: &Map<FfiString, Any>, key: &str) -> Result<Vec<String>> {
    let value = map
        .get(&FfiString::from(key))?
        .ok_or_else(|| ffi_error(&format!("compile schema is missing {key:?}")))?;
    let items = Array::<FfiString>::try_from(value)?;
    Ok(items.iter().map(|item| item.as_str().to_owned()).collect())
}

fn vector_dtype_abis(
    map: &Map<FfiString, Any>,
    key: &str,
) -> Result<HashMap<String, (String, i64, i64, i64)>> {
    let value = map
        .get(&FfiString::from(key))?
        .ok_or_else(|| ffi_error(&format!("compile schema is missing {key:?}")))?;
    let items = Map::<FfiString, Array<Any>>::try_from(value)?;
    let mut abis = HashMap::new();
    for (dtype, abi) in items.iter() {
        let element_dtype = FfiString::try_from(abi.get(0)?)?;
        abis.insert(
            dtype.as_str().to_owned(),
            (
                element_dtype.as_str().to_owned(),
                i64::try_from(abi.get(1)?)?,
                i64::try_from(abi.get(2)?)?,
                i64::try_from(abi.get(3)?)?,
            ),
        );
    }
    Ok(abis)
}

// Closed tables the frontend owns.
const REGISTER_OWNER_AXES: &[&str] = &["laneid", "wid_in_wg", "tid_in_wg"];
const REGISTER_SCOPES: &[&str] = &["local", "local_scalar", "register", "reg"];
const EXTRACTABLE_VECTOR_DTYPES: &[&str] = &[
    "uint32x2",
    "float16x2",
    "bfloat16x2",
    "float32x2",
    "uint64x2",
    "float32x4",
];

const REINTERPRET_NUMERIC_DECODE_PAIRS: &[(&str, &str)] = &[("uint16", "float16")];

const NUMERICALLY_IRRELEVANT_ATTRS: &[&str] = &[
    "tirx.device_entry",
    // Register limits constrain GPU occupancy only; they do not alter the
    // scalar or memory semantics that NumSim executes.
    "tirx.max_registers",
    "tirx.launch_bounds_min_blocks_per_sm",
    "tirx.launch_bounds_max_blocks_per_cluster",
    // The required block size is a launch contract, not a numerical
    // operation; NumSim executes the declared thread extent directly.
    "tirx.required_block_size",
];
const SEMANTIC_ATTRS: &[&str] = &[
    "thread_extent",
    "tirx.dyn_smem_bytes",
    "tirx.pool_max_bytes",
];
const SERIAL_LOOP_ANNOTATIONS: &[&str] = &["pragma_unroll", "disable_unroll"];
pub const INTEGER_BINARY_NODE_KINDS: &[&str] = &[
    "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "Min", "Max",
];

/// Runtime node kinds besides the shared `integer_binary_node_kinds`.
const RUNTIME_FRONTEND_NODE_KINDS: &[&str] = &[
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitwiseNot",
    "LShift",
    "RShift",
    "And",
    "AssertStmt",
    "Bind",
    "Break",
    "TensorLoad",
    "BufferStore",
    "Call",
    "Cast",
    "Continue",
    "EQ",
    "Evaluate",
    "FloatImm",
    "For",
    "GE",
    "GT",
    "IfThenElse",
    "IntImm",
    "LE",
    "LT",
    "NE",
    "Not",
    "Or",
    "Return",
    "Select",
    "Shuffle",
    "TilePrimitiveCall",
    "Var",
    "While",
];
const COMPILE_ONLY_FRONTEND_NODE_KINDS: &[&str] = &[
    "AllocBuffer",
    "AttrStmt",
    // Operand view; its footprint and effects belong to the consuming tile op.
    "TensorRegion",
    "DeclBuffer",
    "Ramp",
    "ScopeIdDefStmt",
    "SeqStmt",
    "StringImm",
];

fn owned_set(values: &[&str]) -> HashSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

impl Schema {
    pub fn parse(map: &Map<FfiString, Any>) -> Result<Schema> {
        let integer_binary_node_kinds = owned_set(INTEGER_BINARY_NODE_KINDS);
        let ptx_table_names = strings(map, "ptx_table_names")?;
        let mut registered_ops = HashMap::new();
        for row in registry::OPS {
            if registered_ops
                .insert(row.ir_name.to_owned(), *row)
                .is_some()
            {
                return Err(ffi_error(&format!(
                    "duplicate registry operation {:?}",
                    row.ir_name
                )));
            }
        }
        for name in &ptx_table_names {
            if !registered_ops.contains_key(name) {
                return Err(ffi_error(&format!(
                    "target PTX operation {name:?} has no reviewed registry row"
                )));
            }
        }
        Ok(Schema {
            high_precision: false,
            registered_ops,
            supported_cast_dtypes: crate::dtypes::with_capability("cast"),
            supported_buffer_dtypes: crate::dtypes::with_capability("buffer"),
            supported_parameter_scalar_dtypes: crate::dtypes::with_capability("scalar"),
            scalar_dtype_bits: crate::dtypes::scalar_bits(),
            numerically_irrelevant_attrs: owned_set(NUMERICALLY_IRRELEVANT_ATTRS),
            semantic_attrs: owned_set(SEMANTIC_ATTRS),
            serial_loop_annotations: owned_set(SERIAL_LOOP_ANNOTATIONS),
            runtime_frontend_node_kinds: integer_binary_node_kinds
                .iter()
                .cloned()
                .chain(owned_set(RUNTIME_FRONTEND_NODE_KINDS))
                .collect(),
            compile_only_frontend_node_kinds: owned_set(COMPILE_ONLY_FRONTEND_NODE_KINDS),
            integer_binary_node_kinds,
            register_owner_axes: owned_set(REGISTER_OWNER_AXES),
            register_scopes: owned_set(REGISTER_SCOPES),
            extractable_vector_dtypes: owned_set(EXTRACTABLE_VECTOR_DTYPES),
            vector_dtype_abis: vector_dtype_abis(map, "vector_dtype_abis")?,
            reinterpret_raw_scalars: crate::dtypes::with_capability("reinterpret_raw"),
            reinterpret_identity_scalars: crate::dtypes::with_capability("scalar"),
            reinterpret_numeric_decode_pairs: REINTERPRET_NUMERIC_DECODE_PAIRS
                .iter()
                .map(|(source, target)| ((*source).to_owned(), (*target).to_owned()))
                .collect(),
            call_scalars: crate::dtypes::with_capability("scalar"),
        })
    }

    /// Registry rows in name order, the public helper policy, contextual
    /// paths and tile operation names.
    pub fn registry_json(&self) -> Json {
        let mut names: Vec<&String> = self.registered_ops.keys().collect();
        names.sort();
        json_object(vec![
            (
                "ops",
                Json::Array(
                    names
                        .iter()
                        .map(|name| row_json(name, &self.registered_ops[*name]))
                        .collect(),
                ),
            ),
            (
                "contextual_op_paths",
                Json::Array(
                    registry::CONTEXTUAL_OP_PATHS
                        .iter()
                        .map(|(name, row)| {
                            json_object(vec![
                                ("name", Json::from(*name)),
                                ("op", row_json(row.ir_name, row)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "tile_ops",
                json_strings(
                    crate::analyze::tile_forms::TILE_OPS
                        .iter()
                        .map(|kind| kind.op_name()),
                ),
            ),
            (
                "external_grid_dependency_requirement",
                Json::from(crate::analyze::frontend::EXTERNAL_GRID_DEPENDENCY_REQUIREMENT),
            ),
        ])
    }

    /// `dtype_abi.vector_dtype_abi`: (element dtype, lanes, element bits, total bits).
    pub fn vector_dtype_abi(&self, dtype: &str) -> Option<(String, i64, i64, i64)> {
        self.vector_dtype_abis.get(dtype).cloned()
    }

    pub fn is_supported_buffer_dtype_name(&self, dtype: &str) -> bool {
        self.supported_buffer_dtypes.contains(dtype) || self.vector_dtype_abi(dtype).is_some()
    }

    pub fn call_dtype_bits(&self, dtype: &str) -> Option<i64> {
        crate::tables::dtype_itemsize(self, dtype).map(|itemsize| itemsize * 8)
    }

    /// `(bits, signed)` of an ordinary integer dtype.
    pub fn integer_dtype_bits(&self, dtype: &str) -> Option<(i64, bool)> {
        if !crate::tables::is_integer_dtype(dtype) {
            return None;
        }
        let bits = *self.scalar_dtype_bits.get(dtype)?;
        Some((bits, dtype.starts_with("int")))
    }
}
