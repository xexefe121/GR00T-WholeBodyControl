"""Run ten actual library controls with passive policy/physics recording."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from gear_sonic.utils import g1_true23_sonic_library_replay as library


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


args = argparse.ArgumentParser()
args.add_argument("--output", type=Path, required=True)
args = args.parse_args()
root = Path("/mnt/z/codex/GR00T-WholeBodyControl")
motion_path = root / "artifacts/g1_true23/internet_pico_twist2_v1/canonical_safe9_001/0807_yanjie_walk_001_50hz.canonical.safe9.npz"
decoder = root / "artifacts/g1_true23/pico_internet_fullbody_v14_100_eval/model_100.decoder.onnx"
policy_class = library.ExactHashSonicPolicy
controller_class = library.LibraryMotionTrue23Controller
calls, controllers = [], []
physics = {key: [] for key in ("pre_qpos", "post_qpos", "pre_qvel", "post_qvel", "command", "time")}


class RecordedPolicy:
    def __init__(self, *args, **kwargs):
        self.original = policy_class(*args, **kwargs)

    def infer(self, encoder267, history930):
        record = {"encoder267": encoder267.copy(), "history930": history930.copy()}
        raw, decoder994 = self.original.infer(encoder267, history930)
        record.update(raw23=raw.copy(), decoder994=decoder994.copy())
        calls.append(record)
        return raw, decoder994


class RecordedModule:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def mj_step(self, model, data):
        physics["pre_qpos"].append(data.qpos.copy())
        physics["pre_qvel"].append(data.qvel.copy())
        physics["command"].append(data.ctrl.copy())
        before = float(data.time)
        self.original.mj_step(model, data)
        physics["post_qpos"].append(data.qpos.copy())
        physics["post_qvel"].append(data.qvel.copy())
        physics["time"].append([before, float(data.time)])


def make_controller(**kwargs):
    controller = controller_class(**kwargs)
    controller.module = RecordedModule(controller.module)
    controllers.append(controller)
    return controller


library.ExactHashSonicPolicy = RecordedPolicy
library.LibraryMotionTrue23Controller = make_controller
report, original_arrays = library.run_library_motion_replay(
    repository_root=root,
    motion_path=motion_path,
    maximum_steps=10,
    decoder_path=decoder,
    expected_decoder_sha256="f66408ae9a10720a3aff717269d0e2a4e07ab471e449a6fe8f5bae5e8607ef63",
    gain_profile="released_retained",
)
if len(controllers) != 1:
    raise ValueError("capture unexpectedly constructed multiple controllers")
controller = controllers[0]
arrays = {key: np.asarray([row[key] for row in calls]) for key in ("encoder267", "history930", "raw23", "decoder994")}
arrays.update({"physics_" + key: np.asarray(values) for key, values in physics.items()})
arrays.update({"original_" + key: value for key, value in original_arrays.items()})
inputs = {str(path.resolve()): sha(path) for path in (motion_path, decoder, Path(__file__))}
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
        inputs[str(Path(path).resolve())] = sha(path)
args.output.mkdir(parents=True, exist_ok=False)
trace_path = args.output / "trace.npz"
with trace_path.open("xb") as stream:
    np.savez_compressed(stream, **arrays)
receipt = dict(
    kind="actual_library_first10_control_capture_v1", report=report,
    controller_source=str(Path(inspect.getfile(controller_class)).resolve()),
    kp=controller.physics.kp.tolist(), kd=controller.physics.kd.tolist(), effort=controller.physics.effort.tolist(),
    policy_calls=len(calls), physics_calls=len(physics["time"]),
    inputs=inputs, trace_path=str(trace_path), trace_sha256=sha(trace_path),
    hardware_authorized=False, deployment_ready=False,
)
with (args.output / "report.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({"output": str(args.output), "controller_source": receipt["controller_source"], "policy_calls": len(calls), "completed": report["completed_transitions"], "failure": report["failure"]}))
