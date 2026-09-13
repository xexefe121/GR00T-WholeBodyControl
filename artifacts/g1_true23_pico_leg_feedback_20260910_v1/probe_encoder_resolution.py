"""Fixed-state source-only leg-reference probe, no dynamics or controller changes."""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_normal_lora_replay import NormalLoraPolicy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PREVIOUS = ROOT / "artifacts/g1_true23_pico_training_20260910_v1"


def main():
    output = HERE / "encoder_resolution_probe.json"
    if output.exists():
        raise FileExistsError("encoder probe refuses overwrite")
    pins = {str(Path(__file__)): sha256_file(Path(__file__))}
    checkpoint = PREVIOUS / "train500_v1/checkpoints/normal_lora_model_500.pt"
    assert sha256_file(checkpoint) == "047b7387f8268d2ff8032c8b74764de6a3f0321f4245814c0f5a909e805dc065"
    torch.set_num_threads(1)
    policy = NormalLoraPolicy(
        checkpoint,
        warm_start_path=ASSETS / "sonic_release/g1_23dof_rev_1_0_init.pt",
        source_checkpoint_path=ASSETS / "sonic_release/last.pt",
    )
    identity = policy.identity()
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        directory = PREVIOUS / "eval500_v1" / name
        path = directory / "report.json"
        pins[str(path)] = sha256_file(path)
        report = json.loads(path.read_text())
        path = directory / "attempts.npz"
        assert sha256_file(path) == report["attempts_sha256"]
        pins[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as z:
            semantic = z["encoder267"].copy()
        source = next(p for p in report["timeline"]["phases"] if p["name"] == "source_motion")
        end = min(len(semantic), source["control_stop"])
        for offset in (1, 5):
            indices = np.unique(np.linspace(source["control_start"], end - offset - 1, 256).astype(np.int64))
            before = torch.from_numpy(semantic[indices].copy())
            after = before.clone()
            # Only swap complete lower-body horizon with another actually saved
            # horizon. Original VR/orientation, measured state and Root9 stay fixed.
            after[:, :240] = torch.from_numpy(semantic[indices + offset, :240])
            with torch.inference_mode():
                a, b = policy.actor.core.encode(before), policy.actor.core.encode(after)
            changed = (a != b).sum(-1).cpu().numpy()
            current_delta = np.sqrt(np.mean((after[:, :12].numpy() - before[:, :12].numpy()) ** 2, axis=-1))
            invisible = changed == 0
            row = dict(
                name=name,
                offset_controls=offset,
                cases=len(indices),
                invisible_reference_change_cases=int(invisible.sum()),
                invisible_current_leg_delta_max_rad=float(current_delta[invisible].max())
                if invisible.any()
                else 0,
                current_leg_delta_max_rad=float(current_delta.max()),
                token_coordinates_changed_mean=float(changed.mean()),
                token_coordinates_changed_min=int(changed.min()),
                counterfactual_scope="fixed_other_inputs_actual_recorded_lower_horizon_only",
                proves_encoder_information_sufficiency=False,
                controller_or_physics_run=False,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert policy.identity() == identity
    with output.open("x") as stream:
        json.dump(
            dict(cases=rows, identity=identity, inputs=pins, deployment_ready=False, hardware_authorized=False),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
