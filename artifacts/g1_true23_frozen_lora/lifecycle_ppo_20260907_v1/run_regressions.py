"""Retain the previous focused suite and add the actual lifecycle PPO tests."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path.cwd()
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
HERE = Path(__file__).resolve().parent
prior = HERE.parent / "reset_curriculum_20260906_v1/regression_report.json"
previous = json.loads(prior.read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert previous["exit_code"] == 0
for path, expected in previous["inputs"].items():
    assert digest(path) == expected, path
modules = previous["modules"] + ["test_g1_true23_lifecycle_ppo"]
assert len(set(modules)) == len(modules) == 52
paths = [ROOT / "gear_sonic/tests" / f"{name}.py" for name in modules]
sources = [ROOT / "gear_sonic" / name for name in (
    "scripts/train_g1_true23_lifecycle_ppo.py", "utils/g1_true23_lifecycle_ppo.py",
    "utils/g1_true23_lifecycle_rollout.py",
)]
inputs = {str(path): digest(path) for path in [*paths, *sources, prior, Path(__file__)]}


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


code = int(pytest.main([*map(str, paths), "-q", f"--junitxml={HERE / 'regression.xml'}"],
                      plugins=[OriginalAssetRoot()]))
for path, expected in inputs.items():
    assert digest(path) == expected, path
inputs[str(HERE / "regression.xml")] = digest(HERE / "regression.xml")
counts = [element.attrib for element in ET.parse(HERE / "regression.xml").getroot().iter("testsuite")]
with (HERE / "regression_report.json").open("x") as stream:
    json.dump(dict(modules=modules, inputs=inputs, exit_code=code, junit_suites=counts,
        substituted_module_asset_roots=previous["substituted_module_asset_roots"],
        hardware_authorized=False, deployment_ready=False), stream, indent=2)
raise SystemExit(code)
