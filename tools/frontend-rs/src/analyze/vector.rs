//! Closed vector IR form classification.

use tvm::ir::{PrimExpr, TensorLoadObj};
use tvm::prim::{RampObj, ShuffleObj};
use tvm::tirx::BufferVar;
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::ObjectRefCore;

use super::util::{buffer_dtype, dtype_of, oref, static_int, unsupported, AResult};
use super::Ctx;

fn static_field(ctx: &Ctx, value: &PrimExpr, field: &str) -> AResult<i64> {
    static_int(&ctx.analyzer, value, field, "must be a static integer")
}

pub fn classify_contiguous_ramp(
    ctx: &Ctx,
    node: &ObjectRef,
    ramp: &RampObj,
    expected_lanes: Option<i64>,
) -> AResult<(ObjectRef, i64)> {
    let lanes = static_field(ctx, &ramp.lanes, "Ramp.lanes")?;
    if lanes != 2 && lanes != 4 {
        return unsupported(format!("Ramp lane count must be 2 or 4, got {lanes}"));
    }
    if let Some(expected) = expected_lanes {
        if lanes != expected {
            return unsupported(format!(
                "Ramp lane count {lanes} does not match vector width {expected}"
            ));
        }
    }
    let stride = static_field(ctx, &ramp.stride, "Ramp.stride")?;
    if stride != 1 {
        return unsupported(format!("Ramp stride must be contiguous 1, got {stride}"));
    }
    if dtype_of(&oref(ramp.base_.clone()))? != "int32"
        || dtype_of(&oref(ramp.stride.clone()))? != "int32"
    {
        return unsupported("Ramp base and stride must both use int32");
    }
    let dtype = dtype_of(node)?;
    if dtype != format!("int32x{lanes}") {
        return unsupported(format!(
            "Ramp result must be int32x{lanes}, got {:?}",
            &dtype
        ));
    }
    Ok((oref(ramp.base_.clone()), lanes))
}

/// `Some(result_dtype)` for a vector load form, `None` for scalar loads.
pub fn classify_vector_buffer_load(
    ctx: &Ctx,
    node: &ObjectRef,
    load: &TensorLoadObj,
    source: &BufferVar,
) -> AResult<Option<String>> {
    let result_dtype = dtype_of(node)?;
    let Some((element_dtype, lanes, _, _)) = ctx.schema.vector_dtype_abi(&result_dtype) else {
        return Ok(None);
    };
    let buffer_dtype = buffer_dtype(source);
    let indices: Vec<PrimExpr> = load.indices.iter().collect();
    let ramp_axes: Vec<usize> = indices
        .iter()
        .enumerate()
        .filter(|(_, index)| oref((*index).clone()).as_node::<RampObj>().is_some())
        .map(|(axis, _)| axis)
        .collect();
    if buffer_dtype == result_dtype {
        if !ramp_axes.is_empty() {
            return unsupported(format!(
                "vector buffer {result_dtype} must use a scalar element index"
            ));
        }
        return Ok(Some(result_dtype));
    }
    if buffer_dtype != element_dtype {
        return unsupported(format!(
            "{result_dtype} vload requires {element_dtype} elements, got {buffer_dtype}"
        ));
    }
    if ramp_axes != vec![indices.len() - 1] {
        return unsupported(format!(
            "{result_dtype} vload requires exactly one Ramp on the final axis"
        ));
    }
    let last = oref(indices[indices.len() - 1].clone());
    let ramp = last.as_node::<RampObj>().expect("ramp");
    classify_contiguous_ramp(ctx, &last, ramp, Some(lanes))?;
    Ok(Some(result_dtype))
}

pub enum VectorForm {
    Extract {
        vector: PrimExpr,
        vector_dtype: String,
        result_dtype: String,
        index: i64,
    },
    Construct {
        vectors: Vec<PrimExpr>,
        vector_dtype: String,
        element_dtype: String,
    },
}

pub fn classify_vector_extract(
    ctx: &Ctx,
    node: &ObjectRef,
    shuffle: &ShuffleObj,
) -> AResult<VectorForm> {
    let vectors: Vec<PrimExpr> = shuffle.vectors.iter().collect();
    let indices: Vec<PrimExpr> = shuffle.indices.iter().collect();
    if vectors.len() == 1 && indices.len() == 1 {
        let vector = oref(vectors[0].clone());
        let vector_dtype = dtype_of(&vector)?;
        if !ctx.schema.extractable_vector_dtypes.contains(&vector_dtype) {
            return unsupported(format!(
                "Shuffle source dtype {:?} has only a packed storage ABI",
                &vector_dtype
            ));
        }
        let (result_dtype, lanes, _, _) = ctx
            .schema
            .vector_dtype_abi(&vector_dtype)
            .expect("extractable vector dtype");
        let node_dtype = dtype_of(node)?;
        if node_dtype != result_dtype {
            return unsupported(format!(
                "Shuffle from {vector_dtype} must return {result_dtype}, got {node_dtype}"
            ));
        }
        let index = static_field(ctx, &indices[0], "Shuffle index")?;
        if index < 0 || index >= lanes {
            return unsupported(format!(
                "Shuffle index {index} is outside {vector_dtype} lane range [0, {lanes})"
            ));
        }
        return Ok(VectorForm::Extract {
            vector: vectors[0].clone(),
            vector_dtype,
            result_dtype,
            index,
        });
    }
    let result_dtype = dtype_of(node)?;
    let result_abi = ctx.schema.vector_dtype_abi(&result_dtype);
    if result_abi.is_none() || (result_dtype != "bfloat16x2" && result_dtype != "uint32x2") {
        return unsupported(
            "Shuffle construction is implemented only for bfloat16x2 and uint32x2 forms",
        );
    }
    let (element_dtype, lanes, _, _) = result_abi.unwrap();
    let mut static_indices = Vec::new();
    for index in &indices {
        static_indices.push(static_field(ctx, index, "Shuffle index")?);
    }
    let expected: Vec<i64> = (0..lanes).collect();
    if vectors.len() as i64 != lanes || static_indices != expected {
        let rendered: Vec<String> = static_indices
            .iter()
            .map(|value| value.to_string())
            .collect();
        return unsupported(format!(
            "Shuffle construction of {result_dtype} requires {lanes} scalar sources selected in order, got {} sources and indices {:?}",
            vectors.len(),
            &rendered));
    }
    let mut source_dtypes = Vec::new();
    for vector in &vectors {
        source_dtypes.push(dtype_of(&oref(vector.clone()))?);
    }
    if source_dtypes.iter().any(|dtype| *dtype != element_dtype) {
        return unsupported(format!(
            "Shuffle construction of {result_dtype} requires {element_dtype} sources, got {:?}",
            &source_dtypes
        ));
    }
    Ok(VectorForm::Construct {
        vectors,
        vector_dtype: result_dtype,
        element_dtype,
    })
}
