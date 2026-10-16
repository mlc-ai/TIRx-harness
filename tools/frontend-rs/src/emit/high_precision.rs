use crate::analyze::util::{dtype_of, unsupported, AResult};
use crate::decode::Decoded;
use crate::emit::{Emitter, RustValue};

pub fn validate_dtype(dtype: &str) -> AResult<()> {
    if (dtype.starts_with("float") || dtype.starts_with("bfloat"))
        && !crate::tables::is_promoted_float(dtype)
    {
        return unsupported(format!(
            "high precision does not model packed or vector floating dtype {dtype}"
        ));
    }
    Ok(())
}

pub fn validate_memory(
    emitter: &Emitter,
    dtype: &str,
    space: crate::analyze::memory::MemorySpace,
    zero_fill: bool,
) -> AResult<()> {
    validate_dtype(dtype)?;
    if space == crate::analyze::memory::MemorySpace::Tmem || zero_fill {
        return unsupported("high precision does not model TMEM or zero-filled scalar accesses");
    }
    if emitter.ctx.schema.vector_dtype_abi(dtype).is_some() {
        return unsupported("high precision does not model vector memory accesses");
    }
    Ok(())
}

pub fn emit_float(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let operation = call.op_name.as_str();
    if !matches!(
        operation,
        "tirx.exp"
            | "tirx.fabs"
            | "tirx.log"
            | "tirx.log1p"
            | "prim.log2"
            | "tirx.rsqrt"
            | "tirx.sigmoid"
            | "tirx.fma"
            | "tirx.cuda.fdividef"
    ) {
        return Ok(None);
    }
    let arguments = emitter.emit_arguments(call.node)?;
    if arguments.iter().any(|value| value.rust_type != "f64") {
        return unsupported(format!(
            "high precision {operation} requires floating operands"
        ));
    }
    emitter
        .emit_call_atom("high_float", &arguments, "f64", |codes| match operation {
            "tirx.exp" => format!("({}).exp()", codes[0]),
            "tirx.fabs" => format!("({}).abs()", codes[0]),
            "tirx.log" => format!("({}).ln()", codes[0]),
            "tirx.log1p" => format!("({}).ln_1p()", codes[0]),
            "prim.log2" => format!("({}).log2()", codes[0]),
            "tirx.rsqrt" => format!("1.0_f64 / ({}).sqrt()", codes[0]),
            "tirx.sigmoid" => format!("1.0_f64 / (1.0_f64 + (-({})).exp())", codes[0]),
            "tirx.fma" => format!("({}).mul_add({}, {})", codes[0], codes[1], codes[2]),
            "tirx.cuda.fdividef" => format!("({}) / ({})", codes[0], codes[1]),
            _ => unreachable!(),
        })
        .map(Some)
}

pub fn validate_call(call: &Decoded) -> AResult<()> {
    let operation = call.op_name.as_str();
    if matches!(
        operation,
        "tirx.break_loop"
            | "tirx.continue_loop"
            | "tirx.cuda.cta_sync"
            | "tirx.cuda.warp_sync"
            | "tirx.cuda.warpgroup_sync"
            | "tirx.cuda.cluster_sync"
            | "tirx.cuda.grid_sync"
            | "tirx.cuda.thread_fence"
            | "tirx.cuda.nano_sleep"
            | "tirx.ptx.bar_sync"
            | "tirx.ptx.bar_sync_count"
            | "tirx.ptx.bar_warp_sync"
    ) {
        return Ok(());
    }
    if matches!(
        operation,
        "prim.if_then_else"
            | "tirx.address_of"
            | "tirx.isnullptr"
            | "tirx.cuda.__activemask"
            | "tirx.cuda.__shfl_sync"
            | "tirx.cuda.__shfl_down_sync"
            | "tirx.cuda.__shfl_up_sync"
            | "tirx.cuda.__shfl_xor_sync"
            | "tirx.tvm_warp_activemask"
            | "tirx.tvm_warp_shuffle"
            | "tirx.tvm_warp_shuffle_down"
            | "tirx.tvm_warp_shuffle_up"
            | "tirx.tvm_warp_shuffle_xor"
            | "tirx.cuda.warp_reduce"
            | "tirx.cuda.elect_sync"
            | "tirx.cuda.thread_rank"
            | "tirx.cuda.ballot_sync"
            | "tirx.cuda.any_sync"
            | "tirx.cuda.ffs_u32"
            | "tirx.popcount"
    ) {
        return Ok(());
    }
    if operation == "tirx.reinterpret"
        && call.args.iter().all(|value| {
            dtype_of(value).is_ok_and(|dtype| !dtype.starts_with("float") && dtype != "bfloat16")
        })
        && !crate::tables::is_promoted_float(&dtype_of(call.node)?)
    {
        return Ok(());
    }
    unsupported(format!(
        "high precision does not model {operation}; byte reinterpretation, packed arithmetic, and unported instructions cannot certify a numerical result"
    ))
}
