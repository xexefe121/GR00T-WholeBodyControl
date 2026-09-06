"""Source-binding closure tests; AST parsing never imports tested modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.utils.g1_true23_generalist_source_closure import (
    GENERALIST_ROOTS,
    collect_local_source_closure,
    generalist_source_files,
)
from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest


def source(root: Path, name: str, text: str = "") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def paths(closure, root):
    return set(closure.as_source_files(root))


def test_transitive_imports_package_initializers_relative_imports_and_cycles(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py", "raise RuntimeError('must never import')\n")
    source(tmp_path, "gear_sonic/entry.py", "from gear_sonic.group import child\n")
    source(tmp_path, "gear_sonic/group/__init__.py", "from . import sibling\n")
    source(tmp_path, "gear_sonic/group/child.py", "from ..other import helper\n")
    source(tmp_path, "gear_sonic/group/sibling.py", "import gear_sonic.entry\n")
    source(tmp_path, "gear_sonic/other/__init__.py")
    source(tmp_path, "gear_sonic/other/helper.py", "import torch\n")
    source(tmp_path, "gear_sonic/unrelated.py")
    closure = collect_local_source_closure(tmp_path, ["gear_sonic/entry.py"])
    assert paths(closure, tmp_path) == {
        "gear_sonic/__init__.py",
        "gear_sonic/entry.py",
        "gear_sonic/group/__init__.py",
        "gear_sonic/group/child.py",
        "gear_sonic/group/sibling.py",
        "gear_sonic/other/__init__.py",
        "gear_sonic/other/helper.py",
    }
    assert len(closure.files) == len(set(closure.files))
    assert not closure.unresolved_dynamic_imports
    assert not closure.report(tmp_path)["source_imported_or_executed"]


def test_literal_import_module_and_class_targets(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    source(
        tmp_path,
        "gear_sonic/entry.py",
        """
import importlib as imports
from importlib import import_module as load
imports.import_module('gear_sonic.dynamic')
load('.relative', package='gear_sonic.group')
__import__('gear_sonic.builtin')
class_name = 'gear_sonic.class_module:Actor'
class_type = 'gear_sonic.dotted.Actor'
""",
    )
    for name in ("dynamic", "builtin", "class_module", "dotted", "explicit"):
        source(tmp_path, f"gear_sonic/{name}.py")
    source(tmp_path, "gear_sonic/group/relative.py")
    closure = collect_local_source_closure(
        tmp_path, ["gear_sonic/entry.py"], class_targets=["gear_sonic.explicit:Model"]
    )
    assert {
        f"gear_sonic/{name}.py" for name in ("dynamic", "builtin", "class_module", "dotted", "explicit")
    } <= paths(closure, tmp_path)
    assert "gear_sonic/group/relative.py" in paths(closure, tmp_path)
    assert not closure.unresolved_dynamic_imports


def test_conditional_imports_and_package_all_are_conservatively_bound(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    source(tmp_path, "gear_sonic/entry.py", "if False:\n    from gear_sonic.group import *\n")
    source(tmp_path, "gear_sonic/group/__init__.py", "__all__ = ['lazy_child']\n")
    source(tmp_path, "gear_sonic/group/lazy_child.py")
    assert "gear_sonic/group/lazy_child.py" in paths(
        collect_local_source_closure(tmp_path, ["gear_sonic/entry.py"]), tmp_path
    )


def test_unresolved_aliased_import_sweeps_model_namespaces_not_unrelated_scripts(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    source(
        tmp_path,
        "gear_sonic/entry.py",
        "from importlib import import_module as load\ndef build(name):\n    return load(name)\n",
    )
    source(tmp_path, "gear_sonic/trl/unreferenced_actor.py")
    source(tmp_path, "gear_sonic/envs/unreferenced_env.py")
    source(tmp_path, "gear_sonic/utils/unreferenced_helper.py")
    source(tmp_path, "gear_sonic/scripts/unrelated_robot_control.py")
    source(tmp_path, "gear_sonic/trl/tests/test_unused.py")
    closure = collect_local_source_closure(tmp_path, ["gear_sonic/entry.py"])
    assert len(closure.unresolved_dynamic_imports) == 1
    assert closure.conservative_namespaces == ("gear_sonic.envs", "gear_sonic.trl", "gear_sonic.utils")
    assert "gear_sonic/trl/unreferenced_actor.py" in paths(closure, tmp_path)
    assert "gear_sonic/envs/unreferenced_env.py" in paths(closure, tmp_path)
    assert "gear_sonic/utils/unreferenced_helper.py" in paths(closure, tmp_path)
    assert not any("unrelated_robot" in p or "tests/" in p for p in paths(closure, tmp_path))


def test_both_initializer_paths_retained_and_helper_edit_changes_manifest(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    source(tmp_path, "gear_sonic/entry.py", "from gear_sonic.a import helper\nimport gear_sonic.b\n")
    source(tmp_path, "gear_sonic/a/__init__.py")
    source(tmp_path, "gear_sonic/b/__init__.py")
    helper = source(tmp_path, "gear_sonic/a/helper.py", "GAIN = 1\n")
    before = collect_local_source_closure(tmp_path, ["gear_sonic/entry.py"]).as_source_files(tmp_path)
    first = build_file_manifest(before, kind="source_files")
    helper.write_text("GAIN = 2\n")
    second = build_file_manifest(before, kind="source_files")
    assert first != second
    assert "gear_sonic/a/__init__.py" in before and "gear_sonic/b/__init__.py" in before


def test_order_is_deterministic_and_no_duplicate_import_entries(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    source(tmp_path, "gear_sonic/a.py", "import gear_sonic.b\n")
    source(tmp_path, "gear_sonic/b.py", "import gear_sonic.a\n")
    a = collect_local_source_closure(tmp_path, ["gear_sonic/a.py", "gear_sonic/b.py"])
    b = collect_local_source_closure(tmp_path, ["gear_sonic/b.py", "gear_sonic/a.py"])
    assert a == b
    assert len(a.files) == 3


@pytest.mark.parametrize("bad", ["../outside.py", "gear_sonic/tests/test_bad.py", "gear_sonic/not_python.json"])
def test_roots_cannot_escape_or_include_tests_nonpython(tmp_path, bad):
    source(tmp_path, "gear_sonic/__init__.py")
    source(tmp_path, bad)
    with pytest.raises(ValueError):
        collect_local_source_closure(tmp_path, [bad])


def test_missing_direct_import_and_file_budget_fail_closed(tmp_path):
    source(tmp_path, "gear_sonic/__init__.py")
    entry = source(tmp_path, "gear_sonic/entry.py", "import gear_sonic.missing\n")
    with pytest.raises(ValueError, match="module is missing"):
        collect_local_source_closure(tmp_path, [entry])
    entry.write_text("import gear_sonic\n")
    with pytest.raises(ValueError, match="maximum_files"):
        collect_local_source_closure(tmp_path, [entry], maximum_files=1)


def test_real_generalist_closure_covers_reviewed_live_helpers_and_itself():
    root = Path(__file__).resolve().parents[2]
    files = generalist_source_files(root)
    required = {
        "gear_sonic/trl/mjlab/frozen_platform_lora_actor.py",
        "gear_sonic/trl/mjlab/frozen_platform_lora_runner.py",
        "gear_sonic/envs/mjlab/sonic_true23_causal_history_safe_target_v11.py",
        "gear_sonic/utils/g1_23dof_safe_target_transform.py",
        "gear_sonic/trl/mjlab/sonic_task_space_ppo_runner.py",
        "gear_sonic/envs/mjlab/native124_selected_v2_ankle_task.py",
        "gear_sonic/utils/g1_true23_generalist_source_closure.py",
        "gear_sonic/__init__.py",
        "gear_sonic/trl/mjlab/__init__.py",
    }
    assert required <= set(files)
    assert set(GENERALIST_ROOTS) <= set(files)
    assert len(files) > len(GENERALIST_ROOTS)
    assert not any("/tests/" in name for name in files)


def test_launcher_binds_closure_with_full_paths_not_basename_append():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "gear_sonic/scripts/train_g1_true23_generalist.py").read_text()
    assert "result.update(generalist_source_files(ROOT))" in launcher
    assert "base._source_files = source_files" in launcher
    assert "base.CAUSAL_SOURCE_FILES +=" not in launcher
