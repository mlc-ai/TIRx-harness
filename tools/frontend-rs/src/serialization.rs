//! Span-insensitive TIR graph serialization shared by normalization and caches.

use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{Any, Array, Map, Result, String as FfiString};

fn replace(map: Map<FfiString, Any>, key: &str, value: Any) -> Map<FfiString, Any> {
    map.iter()
        .map(|(name, old)| {
            (
                name.clone(),
                if name.as_str() == key {
                    value.clone()
                } else {
                    old
                },
            )
        })
        .collect()
}

pub fn semantic_ir_json(value: ObjectRef, tvm_version: FfiString) -> Result<FfiString> {
    let metadata: Map<FfiString, Any> = [(FfiString::from("tvm_version"), tvm_version.into())]
        .into_iter()
        .collect();
    // Work on TVM's graph values directly. Its JSON text permits non-finite
    // floats and its text parser can reject representable f64 subnormals.
    let graph: Map<FfiString, Any> = tvm::tvm_ffi::cached_global_func!("ffi.ToJSONGraph")
        .call_tuple((Any::from(value), &metadata))?
        .try_into()?;
    let nodes =
        Array::<Any>::try_from(graph.get(&FfiString::from("nodes"))?.expect("graph nodes"))?;
    // Node indices follow traversal order; zero need not be null, and a graph
    // need not contain null at all. Append our own without shifting any edges.
    let null_index = nodes.len() as i64;
    let mut stripped_nodes = Vec::with_capacity(nodes.len() + 1);
    for node in nodes.iter() {
        let mut stripped = node.clone();
        if let Ok(fields) = Map::<FfiString, Any>::try_from(node) {
            if let Some(data) = fields.get(&FfiString::from("data"))? {
                if let Ok(data) = Map::<FfiString, Any>::try_from(data) {
                    if data.get(&FfiString::from("span"))?.is_some() {
                        stripped = replace(
                            fields,
                            "data",
                            replace(data, "span", null_index.into()).into(),
                        )
                        .into();
                    }
                }
            }
        }
        stripped_nodes.push(stripped);
    }
    let null_node: Map<FfiString, Any> =
        [(FfiString::from("type"), FfiString::from("None").into())]
            .into_iter()
            .collect();
    stripped_nodes.push(null_node.into());
    let graph = replace(graph, "nodes", Array::new(stripped_nodes).into());
    let stripped: Any =
        tvm::tvm_ffi::cached_global_func!("ffi.FromJSONGraph").call_tuple((graph,))?;
    // Loading and saving removes Span/SourceName nodes disconnected above.
    tvm::tvm_ffi::cached_global_func!("ffi.ToJSONGraphString")
        .call_tuple((stripped, metadata))?
        .try_into()
}
