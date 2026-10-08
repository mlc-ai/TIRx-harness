from __future__ import annotations

import ast
from pathlib import Path

from tests.numsim.support.runtime_cases import RUNTIME_CASES
from tirx_harness.numsim.transpiler import native_frontend


_TEST_ROOT = Path(__file__).parents[1]


def _registered_targets() -> set[str]:
    registry = native_frontend.registry()
    targets = {
        f"call:{row['ir_name']}"
        for row in native_frontend.registry_ops()
        if row["support"] != "rejected"
    }
    targets.update(
        f"call:{path['op']['ir_name']}"
        for path in registry["contextual_op_paths"]
        if path["op"]["support"] != "rejected"
    )
    for ir_name in registry["tile_ops"]:
        target_id = f"tile:{ir_name}"
        assert target_id not in targets
        targets.add(target_id)
    return targets


def _selector_function(selector: str) -> tuple[Path, str]:
    relative_path, separator, test_id = selector.partition("::")
    assert separator and test_id, f"invalid runtime-case selector {selector!r}"
    path = _TEST_ROOT / relative_path
    function_name = test_id.split("[", 1)[0]
    return path, function_name


def test_each_registered_operation_has_a_handwritten_runtime_case():
    registered = _registered_targets()
    declared_targets = {target_id for case in RUNTIME_CASES for target_id in case.target_ids}
    assert declared_targets == registered


def test_runtime_case_selectors_name_existing_tests():
    functions_by_path: dict[Path, set[str]] = {}
    for case in RUNTIME_CASES:
        assert case.target_ids
        assert case.tests
        for selector in case.tests:
            path, function_name = _selector_function(selector)
            assert path.is_file(), selector
            functions = functions_by_path.get(path)
            if functions is None:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                functions = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                functions_by_path[path] = functions
            assert function_name in functions, selector
