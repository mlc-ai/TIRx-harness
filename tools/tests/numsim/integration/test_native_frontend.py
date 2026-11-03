"""Production native services validate their build and preserve layout coordinates."""

import pytest
import tvm_ffi

import tvm_tirx
from tirx_harness.numsim.errors import NumSimBuildError
from tirx_harness.numsim.transpiler import native_frontend
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout


def test_native_frontend_rejects_a_mismatched_build(monkeypatch):
    native_frontend._library.cache_clear()
    monkeypatch.setattr(native_frontend, "_expected_identity", lambda: "incompatible-build")
    try:
        with pytest.raises(NumSimBuildError, match="frontend mismatch"):
            native_frontend._library()
    finally:
        native_frontend._library.cache_clear()


def test_native_frontend_reports_callback_registration_failure_at_load(monkeypatch):
    callback = "numsim.frontend.decode_ptx_call"
    register = tvm_ffi.register_global_func

    def fail_registration(name, function, **kwargs):
        if name == callback:
            raise RuntimeError(f"cannot register {name}")
        return register(name, function, **kwargs)

    native_frontend._library.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(tvm_ffi, "register_global_func", fail_registration)
            with pytest.raises(NumSimBuildError, match="callback registration failed") as caught:
                native_frontend._library()
            assert callback in str(caught.value.__cause__)
        # A failed load must not poison the cache; restoring registration is sufficient.
        loaded = native_frontend._library()
        assert str(loaded["numsim_frontend_identity"]()) == native_frontend._expected_identity()
    finally:
        native_frontend._library.cache_clear()


@pytest.mark.parametrize(
    "layout, atom_offsets, repeat_stride",
    [
        (TileLayout(S[(2, 4) : (8, 1)]), [0, 1, 2, 3, 8, 9, 10, 11], 16),
        (TileLayout(S[(2, 4) : (8, 1)] + 4), [4, 5, 6, 7, 12, 13, 14, 15], 16),
        (
            ComposeLayout(0, 1, 2, TileLayout(S[(2, 4) : (8, 1)] + 4)),
            [5, 4, 7, 6, 13, 12, 15, 14],
            16,
        ),
        (
            ComposeLayout(1, 1, 2, TileLayout(S[(16,)])),
            [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9, 14, 15, 12, 13],
            16,
        ),
        (
            ComposeLayout(1, 1, 2, TileLayout(S[(16,)]), swizzle_inner=False),
            [0, 1, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7],
            16,
        ),
        (ComposeLayout(0, 0, 2, TileLayout(S[(2, 4) : (8, 1)])), [0, 1, 2, 3, 8, 9, 10, 11], 16),
    ],
)
def test_linear_mapping_and_full_domain_span_preserve_physical_coordinates(
    layout,
    atom_offsets,
    repeat_stride,
):
    expected = [offset + repeat * repeat_stride for repeat in range(3) for offset in atom_offsets]
    before = tvm_tirx.ir.save_json(layout)
    assert list(map(int, native_frontend.layout_linear_offsets(layout, len(expected)))) == expected
    assert tvm_tirx.ir.save_json(layout) == before
