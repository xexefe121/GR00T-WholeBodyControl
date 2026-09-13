"""Retain the prior 511 checks and add standing-retention regressions."""

import ast
import json
from pathlib import Path

import pytest

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

ROOT = Path.cwd()
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OUTPUT = Path(__file__).resolve().parent
prior = OUTPUT.parent / "standing_motion_ppo_20260906_v1/run_regressions.py"
base = OUTPUT.parent / "original29_cpp_observation_replay_20260906_v1/run_regressions.py"
tree = ast.parse(base.read_text())
expression = next(
    node.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "MODULES" for target in node.targets)
)
assert isinstance(expression, ast.Call) and expression.func.attr == "split"
prior_tree = ast.parse(prior.read_text())
prior_expression = next(
    node.value
    for node in prior_tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "MODULES" for target in node.targets)
)
assert isinstance(prior_expression, ast.BinOp)
MODULES = (
    ast.literal_eval(expression.func.value).split()
    + ast.literal_eval(prior_expression.right)
    + [
        "test_g1_true23_standing_retention",
        "test_standing_retention_ppo",
        "test_train_g1_true23_standing_retention",
    ]
)
paths = [ROOT / "gear_sonic/tests" / f"{name}.py" for name in MODULES]
inputs = {str(path): file_sha256(path) for path in [*paths, prior, base, Path(__file__)]}


class OriginalAssetRoot:
    def pytest_collection_modifyitems(self, items):
        seen = set()
        for item in items:
            module = item.module
            if module in seen or not module.__file__.endswith(
                ("test_g1_true23_clean_mujoco_teleop.py", "test_g1_true23_pico_fullbody_motion.py")
            ):
                continue
            seen.add(module)
            for name, value in list(vars(module).items()):
                if isinstance(value, Path) and value.is_relative_to(ROOT):
                    setattr(module, name, ASSETS / value.relative_to(ROOT))


code = int(
    pytest.main([*map(str, paths), "-q", f"--junitxml={OUTPUT / 'regression.xml'}"], plugins=[OriginalAssetRoot()])
)
for path, digest in inputs.items():
    if file_sha256(path) != digest:
        raise ValueError(f"retention regression input changed: {path}")
inputs[str(OUTPUT / "regression.xml")] = file_sha256(OUTPUT / "regression.xml")
with (OUTPUT / "regression_report.json").open("x") as stream:
    json.dump(
        dict(
            exit_code=code,
            modules=MODULES,
            inputs=inputs,
            substituted_module_asset_roots=[
                "test_g1_true23_clean_mujoco_teleop",
                "test_g1_true23_pico_fullbody_motion",
            ],
            hardware_authorized=False,
            deployment_ready=False,
        ),
        stream,
        indent=2,
    )
raise SystemExit(code)
