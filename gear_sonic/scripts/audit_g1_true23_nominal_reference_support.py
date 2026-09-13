"""Full-frame support hypothesis using actual nominal23 effort limits.

Unlike the historical quarter-effort gantry support audit, this uses the same
nominal actuation/model as the independent CPU policy referee. It is an offline
necessary-condition diagnostic, never controller or deployment qualification.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    assets = args.asset_root.resolve(strict=True)
    paths = (
        args.motion.resolve(strict=True),
        assets / MODEL,
        root / PHYSICS,
        Path(__file__),
        root / "gear_sonic/utils/g1_true23_reference_support.py",
        root / "gear_sonic/utils/g1_true23_native_model_actuation.py",
    )
    inputs = {str(path): sha256_file(path) for path in paths}
    with np.load(args.motion, allow_pickle=False) as archive:
        if "joint_names" in archive and tuple(archive["joint_names"].tolist()) != tuple(HARDWARE_23_JOINT_NAMES):
            raise ValueError("support source joint names differ from the physical native23 order")
        # Retarget artifacts may also carry bound provenance/diagnostic arrays.
        # Consume all seven physical channels without resampling any source frame.
        motion = {
            key: archive[key].copy()
            for key in (
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    profile = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    evidence = audit_reference_support(model, motion, profile.effort, reference_dynamics=True)
    if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("nominal support diagnostic input changed during execution")
    report = dict(
        kind="native23_nominal_reference_support_hypothesis_v1",
        reference_support=evidence,
        inputs=inputs,
        actual_nominal_effort_limits_not_quarter_effort_gantry=True,
        production_limits_changed=False,
        physics_integration_steps=0,
        **FLAGS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
