"""Run real reset-curriculum smoke, then one 500-update training experiment."""

import json
from pathlib import Path
import subprocess
import sys
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "ieee_motion_ppo_20260906_v2"
previous = json.loads((PREVIOUS / "experiment_report.json").read_text())
checkpoint = PREVIOUS / "breadth_serial/checkpoints/frozen_lora_model_100.pt"
checkpoint_sha = "f20f82385dd7c652a7a74b6103a4753f0317b6fb10055452862d647e7dc14de5"
assert file_sha256(checkpoint) == checkpoint_sha
inputs = {str(checkpoint): checkpoint_sha, str(Path(__file__)): file_sha256(Path(__file__))}
for path in (PREVIOUS / "experiment_report.json", HERE.parent / "sensor_exploration_20260906_v1/reset_controls_report.json",
             ROOT / "gear_sonic/scripts/train_g1_true23_reset_curriculum.py",
             ROOT / "gear_sonic/utils/g1_true23_reset_curriculum.py",
             ROOT / "gear_sonic/tests/test_g1_true23_reset_curriculum.py"):
    inputs[str(path)] = file_sha256(path)
base = list(previous["stages"][-1]["command"])
assert base[2] == "train" and base[base.index("--num-envs")+1] == "32"
base[0] = sys.executable
base[1] = str(ROOT / "gear_sonic/scripts/train_g1_true23_reset_curriculum.py")
base += ["--curriculum-actor-checkpoint", str(checkpoint), "--expected-curriculum-actor-sha256", checkpoint_sha]
plan = [dict(name="smoke", updates=3, envs=4, iterations=4, save=1, warm=8, end=16),
        dict(name="train500", updates=500, envs=32, iterations=500, save=100, warm=1600, end=6400)]
dump(HERE / "started.json", dict(inputs=inputs, plan=plan, hardware_authorized=False, deployment_ready=False))
stages = []
for entry in plan:
    command = list(base)
    command[2] = "smoke" if entry["name"] == "smoke" else "train"
    for flag, value in (("--run-dir", HERE / entry["name"]), ("--num-envs", entry["envs"]),
                        ("--iterations", entry["iterations"]), ("--session-updates", entry["updates"]),
                        ("--save-interval", entry["save"])):
        command[command.index(flag)+1] = str(value)
    command += ["--reset-warmup-controls", str(entry["warm"]), "--reset-ramp-end-controls", str(entry["end"])]
    print(json.dumps(dict(starting_stage=entry["name"], new_updates=entry["updates"])), flush=True)
    start = time.monotonic()
    log = HERE / f"{entry['name']}.log"
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=entry["name"], command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
               log=str(log), log_sha256=file_sha256(log))
    dump(HERE / f"{entry['name']}.stage.json", row)
    stages.append(row)
    print(json.dumps(dict(completed_stage=entry["name"], return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    runtime = HERE / entry["name"] / f"reset_curriculum_{entry['updates']}.json"
    receipt = json.loads(runtime.read_text())
    assert receipt["new_update_count"] == entry["updates"] and receipt["action_std_unchanged"]
    assert receipt["summary"]["maximum_scale"] == 1 and receipt["summary"]["remaining_detected_floor_penetrations"] == 0
    assert receipt["summary"]["full_disturbance_reset_rows"] > 0
    inputs[str(runtime)] = file_sha256(runtime)
    inputs[str(log)] = row["log_sha256"]
    if entry["name"] == "smoke":
        import torch
        initial = torch.load(HERE / "smoke/checkpoints/frozen_lora_model_0.pt", map_location="cpu", weights_only=True)
        final = torch.load(HERE / "smoke/checkpoints/frozen_lora_model_3.pt", map_location="cpu", weights_only=True)
        prior = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert initial["adapter_state_sha256"] == prior["adapter_state_sha256"]
        assert initial["merged_true23_policy_sha256"] == prior["merged_true23_policy_sha256"]
        assert initial["update_count"] == 0 and initial["optimizer_state_dict"]["state"] == {}
        assert final["update_count"] == 3 and final["adapter_state_sha256"] != initial["adapter_state_sha256"]
        assert final["critic_state_sha256"] != initial["critic_state_sha256"]
        dump(HERE / "smoke_verified.json", dict(
            initial_actor_exactly_matches_current_lora100=True, prior_updates_counted_as_new=False,
            new_updates=3, fresh_optimizer_verified=True, actor_and_critic_changed=True,
            actual_full_amplitude_resets=receipt["summary"]["full_disturbance_reset_rows"],
            hardware_authorized=False, deployment_ready=False))
        del initial, final, prior
        import gc
        gc.collect()
for path, digest in inputs.items():
    assert file_sha256(path) == digest
dump(HERE / "experiment_report.json", dict(
    kind="g1_true23_reset_curriculum_training_experiment_v1", inputs=inputs, plan=plan, stages=stages,
    new_training_updates=500, new_training_transitions=500*32*16, smoke_updates=3, smoke_transitions=3*4*8,
    full_motion_qualified=False, hardware_authorized=False, deployment_ready=False))
