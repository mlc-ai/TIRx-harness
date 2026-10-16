"""The global shadow seed is independent of returned buffers and respects aliases."""

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tirx_harness.numsim.transpiler.frontend import analyze


@T.prim_func
def structured_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                     output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias = T.decl_buffer((1,), "int32", data=target.data)
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            alias[0] = 7


@T.prim_func
def raw_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
              output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            T.ptx.st.global_.s32(target.ptr_to([0]), T.int32(7))


@T.prim_func
def tile_write(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    Tx.copy(output[:], source[:])


def race_source(func):
    return emit_rust_module(analyze(func), func, analysis_capable=True, analysis_checker="racecheck")


def _seed(indices):
    """The emitted seed: each rank's bindings contribute these buffers' allocations."""
    return (
        "let global_write_allocations = { let mut rank_writes = Vec::new(); "
        f"for buffers in rank_buffers.iter() {{ rank_writes.extend([{indices}].map"
    )


def test_seed_tracks_declared_alias_destinations_but_not_readonly_inputs():
    source = race_source(structured_write)
    assert _seed("2_usize, 3_usize") in source
    assert "view.allocation()" in source
    assert "inputs.copy()" not in source
    assert 'set_item("written_allocations"' not in source


def test_tile_destination_is_a_write_without_a_buffer_store():
    assert _seed("1_usize") in race_source(tile_write)


def test_raw_store_tracks_its_destination():
    assert _seed("1_usize, 2_usize") in race_source(raw_write)


def test_phase_names_are_qualified_in_multi_kernel_seed():
    source = race_source((structured_write, structured_write))
    assert source.count(_seed("2_usize, 3_usize")) == 2
    assert '"k0:output"' in source
    assert '"k1:output"' in source


@pytest.mark.parametrize("func", [structured_write, raw_write])
@pytest.mark.parametrize("alias_inputs", [False, True])
@pytest.mark.parametrize("inspect_accesses", [False, True])
def test_read_before_write_is_still_checked_for_host_aliases(tmp_path, func, alias_inputs, inspect_accesses):
    module = numsim.transpile(func, cache_dir=tmp_path, _analysis_checker="racecheck")
    source = np.zeros(1, dtype=np.int32)
    target = source if alias_inputs else np.zeros(1, dtype=np.int32)
    report = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"source": source, "target": target, "output": np.zeros(1, dtype=np.int32)},
        inspect_accesses=inspect_accesses,
    ).to_dict()
    assert not report["incomplete"]
    assert report["verdict"] == ("error" if alias_inputs else "clean")
    if alias_inputs:
        assert any(finding["access_pair"] in {"read_write", "write_read"}
                   for finding in report["findings"])


@T.prim_func
def register_result_to_global(source: T.Buffer((1,), "float32"),
                              output: T.Buffer((2,), "float32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.ex2.approx.ftz.f32(output[cta], source[0])


def test_register_instruction_buffer_destinations_are_not_assumed_readonly():
    # The existing register-result store owns the destination information.
    assert _seed("1_usize") in race_source(
        register_result_to_global
    )


@T.prim_func
def pointer_view_write(pointer: T.handle("int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    view = T.decl_buffer((1,), "int32", data=pointer)
    if lane == 0:
        view[0] = 7


def test_bound_pointer_view_uses_the_physical_parameter(tmp_path):
    source = race_source(pointer_view_write)
    assert "let global_write_allocations = allocation_ids.to_vec();" not in source
    assert "buffers.pointer_0.buffer()" in source
    module = numsim.transpile(pointer_view_write, cache_dir=tmp_path, _analysis_checker="racecheck")
    report = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"pointer": np.zeros(1, dtype=np.int32)},
    ).to_dict()
    assert report["verdict"] == "clean"
    assert report["incomplete"] == []


@T.prim_func
def readonly(source: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    T.evaluate(source[0])


def test_readonly_kernel_needs_no_global_write_history(tmp_path):
    module = numsim.transpile(readonly, cache_dir=tmp_path, _analysis_checker="racecheck")
    report = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"source": np.zeros(1, dtype=np.int32)}, inspect_accesses=True,
    ).to_dict()
    assert report["verdict"] == "clean"
    assert not report["incomplete"]


@pytest.mark.parametrize("alias_inputs", [False, True])
@pytest.mark.parametrize("inspect_accesses", [False, True])
def test_register_result_writes_preserve_host_alias_races(tmp_path, alias_inputs, inspect_accesses):
    module = numsim.transpile(
        register_result_to_global, cache_dir=tmp_path, _analysis_checker="racecheck",
    )
    output = np.zeros(2, dtype=np.float32)
    source = output[:1] if alias_inputs else np.zeros(1, dtype=np.float32)
    report = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"source": source, "output": output}, inspect_accesses=inspect_accesses,
    ).to_dict()
    assert not report["incomplete"]
    assert report["verdict"] == ("error" if alias_inputs else "clean")
    if alias_inputs:
        assert any(finding["access_pair"] in {"read_write", "write_read"}
                   for finding in report["findings"])


@T.prim_func
def selected_pointer_write(source: T.Buffer((3,), "int32"),
                           a: T.Buffer((3,), "int32"), b: T.Buffer((3,), "int32"),
                           output: T.Buffer((1,), "int32"), select_b: T.int32):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if lane == 0:
        if cta == 0:
            output[0] = source[2]
        else:
            T.ptx.mov.b64(pointer[0], T.reinterpret("uint64", a.ptr_to([0])))
            T.ptx.selp.u64(pointer[0], T.reinterpret("uint64", b.ptr_to([0])),
                          pointer[0], T.ptx.pred(select_b != 0))
            for _ in T.serial(2):
                pointer[0] = pointer[0] + T.uint64(4)
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


@pytest.mark.parametrize("select_b", [0, 1])
@pytest.mark.parametrize("alias", ["none", "selected", "unselected"])
def test_loop_carried_selected_pointer_preserves_compact_alias_races(tmp_path, select_b, alias):
    module = numsim.transpile(selected_pointer_write, cache_dir=tmp_path,
                              _analysis_checker="racecheck")
    assert "let global_write_allocations = allocation_ids.to_vec();" not in module.rust_source
    a, b = np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.int32)
    source = (b if select_b else a) if alias == "selected" else (
        (a if select_b else b) if alias == "unselected" else np.zeros(3, dtype=np.int32)
    )
    report = numsim.Engine(max_workers=1).run_racecheck_phase(module, {
        "source": source, "a": a, "b": b, "output": np.zeros(1, dtype=np.int32),
        "select_b": select_b,
    }).to_dict()
    assert report["incomplete"] == []
    assert report["execution_error"] is None
    assert report["verdict"] == ("error" if alias == "selected" else "clean")
    if alias == "selected":
        assert any(f["access_pair"] in {"read_write", "write_read"} for f in report["findings"])


@T.prim_func
def clobbered_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                            pointer_bits: T.Buffer((1,), "uint64"),
                            output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            pointer[0] = T.reinterpret("uint64", target.ptr_to([0]))
            T.ptx.xor.b64(pointer[0], pointer_bits[0], T.uint64(0))
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


@T.prim_func
def copied_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                         pointer_bits: T.Buffer((1,), "uint64"),
                         output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if cta == 0:
        if lane == 0:
            output[0] = source[0]
    else:
        pointer[0] = T.reinterpret("uint64", target.ptr_to([0]))
        Tx.copy(pointer[:], pointer_bits[:])
        if lane == 0:
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


@pytest.mark.parametrize("func", [clobbered_pointer_write, copied_pointer_write])
def test_unknown_register_overwrite_keeps_read_before_write_race(tmp_path, func):
    module = numsim.transpile(func, cache_dir=tmp_path,
                              _analysis_checker="racecheck")
    assert "let global_write_allocations = allocation_ids.to_vec();" in module.rust_source
    source = np.zeros(1, dtype=np.int32)
    report = numsim.Engine(max_workers=1).run_racecheck_phase(module, {
        "source": source, "target": np.zeros(1, dtype=np.int32),
        "pointer_bits": np.array([source.ctypes.data], dtype=np.uint64),
        "output": np.zeros(1, dtype=np.int32),
    }).to_dict()
    assert report["incomplete"] == []
    assert report["verdict"] == "error"
    assert any(f["access_pair"] in {"read_write", "write_read"} for f in report["findings"])


def tensor_map_writer(replace):
    from tvm.script import from_source

    # Typed TensorMap parameters are grid-constant. Mutate a global descriptor
    # buffer instead, just as the raw descriptor runtime cases do.
    map_type = 'T.Buffer((128,), "uint8")' if replace else "T.TensorMap()"
    descriptor = "target_map.ptr_to([0])" if replace else "T.address_of(target_map)"
    return from_source(f'''
@T.prim_func
def kernel(source: T.Buffer((32,), "int32"), replacement: T.Buffer((32,), "int32"),
           output: T.Buffer((1,), "int32"), target_map: {map_type}):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    if cta == 0:
        if lane == 0:
            output[0] = source[0]
    else:
        shared[lane] = 7
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            if {replace}:
                T.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
                    {descriptor}, T.reinterpret("uint64", replacement.ptr_to([0])))
                T.ptx.fence.proxy.tensormap__generic.release.gpu()
                T.ptx.fence.proxy.tensormap__generic.acquire.gpu({descriptor})
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                {descriptor}, 0, shared.ptr_to([0]))
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
''')


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("alias_inputs", [False, True])
def test_tensor_map_initial_and_replaced_bases_keep_compact_alias_races(tmp_path, replace, alias_inputs):
    module = numsim.transpile(tensor_map_writer(replace), cache_dir=tmp_path,
                              _analysis_checker="racecheck")
    fallback = "let global_write_allocations = allocation_ids.to_vec();" in module.rust_source
    assert fallback == replace
    initial, replacement = np.zeros(32, dtype=np.int32), np.zeros(32, dtype=np.int32)
    source = (replacement if replace else initial) if alias_inputs else np.zeros(32, dtype=np.int32)
    descriptor = numsim.TensorMap(
        initial, global_shape=(32,), global_strides=(), box_shape=(32,), element_strides=(1,),
    )
    report = numsim.Engine(max_workers=1).run_racecheck_phase(module, {
        "source": source, "replacement": replacement, "output": np.zeros(1, dtype=np.int32),
        "target_map": descriptor.numpy(),
    }).to_dict()
    assert report["incomplete"] == []
    assert report["execution_error"] is None
    assert report["verdict"] == ("error" if alias_inputs else "clean"), report
    if alias_inputs:
        assert any(f["access_pair"] in {"read_write", "write_read"} for f in report["findings"])


@T.prim_func
def discard_after_read(source: T.Buffer((128,), "uint8"), target: T.Buffer((128,), "uint8"),
                       output: T.Buffer((1,), "uint8")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            T.ptx.discard.global_.L2(target.ptr_to([0]))


@pytest.mark.parametrize("alias_inputs", [False, True])
def test_discard_is_a_write_for_compact_alias_races(tmp_path, alias_inputs):
    module = numsim.transpile(discard_after_read, cache_dir=tmp_path, _analysis_checker="racecheck")
    assert "let global_write_allocations = allocation_ids.to_vec();" not in module.rust_source
    storage = np.zeros(256, dtype=np.uint8)
    offset = -storage.ctypes.data % 128
    target = storage[offset:offset + 128]
    source = target if alias_inputs else np.zeros(128, dtype=np.uint8)
    report = numsim.Engine(max_workers=1).run_racecheck_phase(module, {
        "source": source, "target": target, "output": np.zeros(1, dtype=np.uint8),
    }).to_dict()
    assert report["incomplete"] == []
    assert report["verdict"] == ("error" if alias_inputs else "clean"), report


@T.prim_func
def offset_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                         redirected: T.Buffer((1,), "int32"), offset: T.Buffer((1,), "uint64"),
                         output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            if target[0] == 0:
                T.ptx.st.global_.s32(T.reinterpret("handle",
                    T.reinterpret("uint64", target.ptr_to([0])) + offset[0]), 7)
                target[0] = 1


@pytest.mark.parametrize("alias_inputs", [False, True])
@pytest.mark.parametrize("inspect_accesses", [False, True])
def test_raw_offset_escaping_seed_restarts_from_original_inputs(tmp_path, alias_inputs, inspect_accesses):
    module = numsim.transpile(offset_pointer_write, cache_dir=tmp_path, _analysis_checker="racecheck")
    assert "let global_write_allocations = allocation_ids.to_vec();" not in module.rust_source
    target, redirected = sorted((np.zeros(1, np.int32), np.zeros(1, np.int32)),
                                key=lambda value: value.ctypes.data)
    source = redirected if alias_inputs else np.zeros(1, np.int32)
    inputs = dict(source=source, target=target, redirected=redirected,
                  offset=np.array([redirected.ctypes.data - target.ctypes.data], np.uint64),
                  output=np.zeros(1, np.int32))
    report = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, inputs, inspect_accesses=inspect_accesses,
    ).to_dict()
    assert report["incomplete"] == []
    assert report["execution_error"] is None
    assert report["stats"]["global_write_seed_replays"] == 1
    assert report["verdict"] == ("error" if alias_inputs else "clean"), report
    if alias_inputs:
        assert any(f["access_pair"] in {"read_write", "write_read"} for f in report["findings"])
    # Reusing speculative memory would leave target[0] == 1 and skip the
    # conflicting write on replay. Neither attempt may alter caller bindings.
    for name in ("source", "target", "redirected", "output"):
        np.testing.assert_array_equal(inputs[name], np.zeros(1, np.int32))
