"""Cache full dynamics and locate rejected offline feedback trials, SIM only."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import assert_bindings, load_arrays
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant, state_difference
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import (
    MotionObjective,
    backward_pass,
    canonical_target,
    feedback_trial,
    linearize_trajectory,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


class FailureLocatedPlant(PdShootingPlant):
    def scratch(self, state):
        self.control_index, self.last = -1, None
        return super().scratch(state)

    def integrate_control(self, data, target, records=None):
        self.control_index += 1
        super().integrate_control(data, target, records)
        q = np.asarray(records["physics_post_qpos"][-10:])[:, 7:]
        v = np.asarray(records["physics_post_qvel"][-10:])[:, 6:]
        excess = np.maximum(self.model.jnt_range[1:, 0] - q, q - self.model.jnt_range[1:, 1])
        substep, joint = np.unravel_index(np.argmax(excess), excess.shape)
        self.last = dict(
            completed_controls=self.control_index + 1,
            physical_substep=self.control_index * 10 + int(substep),
            worst_joint=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, 1 + int(joint)),
            joint_limit_excess_rad=float(excess[substep, joint]),
            joint_position_rad=float(q[substep, joint]),
            joint_bounds_rad=self.model.jnt_range[1 + joint].tolist(),
            motor_velocity_ratio_max=float(np.max(np.abs(v) / np.asarray(self.profile.velocity))),
            root_height_m=float(data.qpos[2]),
            target23=np.asarray(target).tolist(),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root, start = Path(__file__).resolve().parents[2], time.monotonic()
    reproduction = json.loads(args.reproduction_report.read_text())
    assert_bindings(reproduction["inputs"])
    parent_path, trace_path = Path(reproduction["parent_evaluation"]), Path(reproduction["trace_path"])
    if sha256_file(trace_path) != reproduction["trace_sha256"]:
        raise ValueError("feedback diagnosis seed changed")
    parent = json.loads(parent_path.read_text())
    row = next(item for item in parent["records"] if item["case"] == "nominal")
    reference_path = Path(row["result"]["motion_path"])
    if sha256_file(reference_path) != row["result"]["motion_sha256"]:
        raise ValueError("feedback diagnosis original motion changed")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant = FailureLocatedPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != reproduction["reproduced_physics_sha256"]:
        raise ValueError("feedback diagnosis changed native23 physics")
    seed = load_arrays(trace_path)
    controls = seed["target23"]
    objective = MotionObjective(plant, load_arrays(reference_path), parent["timeline"])
    paths = [
        args.reproduction_report,
        parent_path,
        trace_path,
        reference_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    args.output_directory.mkdir(parents=True, exist_ok=False)

    def progress(event):
        message = json.dumps(dict(elapsed_s=round(time.monotonic() - start, 3), **event), allow_nan=False)
        with (args.output_directory / "events.jsonl").open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    local = linearize_trajectory(plant, objective, seed, controls, progress)
    cache = args.output_directory / "full_linearization.npz"
    with cache.open("xb") as stream:
        np.savez_compressed(stream, **local)
    derivative_checks, rng = [], np.random.default_rng(55182)
    for i in (0, 230, 250, 350, 500, 619, len(controls) - 1):
        dx, du, eps = rng.normal(size=58), rng.normal(size=23), 1e-6
        du[(controls[i] - plant.lower < 1e-5) | (plant.upper - controls[i] < 1e-5)] = 0
        states = []
        for sign in (-1, 1):
            data = plant.scratch(seed["integration_state"][i])
            mujoco.mj_integratePos(model, data.qpos, dx[:29], sign * eps)
            data.qvel[:] += sign * eps * dx[29:]
            PdShootingPlant.integrate_control(plant, data, controls[i] + sign * eps * du)
            states.append((data.qpos.copy(), data.qvel.copy()))
        finite = state_difference(model, *states[0], *states[1]) / (2 * eps)
        predicted = local["a"][i, :58, :58] @ dx + local["b"][i, :58] @ du
        check = dict(
            control_index=i,
            directional_error_norm=float(np.linalg.norm(finite - predicted)),
            finite_derivative_norm=float(np.linalg.norm(finite)),
        )
        derivative_checks.append(check)
        progress(dict(stage="derivative_check", **check))
    trials = []
    for regularization in (1.0, 10.0, 100.0):
        increments, feedback = backward_pass(plant, local, controls, controls, regularization)
        for alpha in (0.0, 0.03125, 0.0625):
            trial = dict(
                regularization=regularization,
                alpha=alpha,
                maximum_feedback_absolute=float(np.max(np.abs(feedback))),
            )
            try:
                candidate, targets = feedback_trial(
                    plant, seed["integration_state"][0], seed, controls, increments, feedback, alpha
                )
                trial.update(
                    complete_controls=len(targets), full_cost=objective.cost(candidate, targets, controls)
                )
                if alpha == 0:
                    trial["zero_alpha_physical_arrays_bit_exact"] = all(
                        value.tobytes() == seed[key].tobytes() for key, value in candidate.items()
                    )
            except (ValueError, RuntimeError) as exc:
                trial.update(error=str(exc), failure_location=plant.last)
            trials.append(trial)
            progress(dict(stage="diagnostic_trial", **trial))
    assert_bindings(bindings)
    report = dict(
        kind="g1_true23_full_horizon_pd_feedback_step_diagnosis_v1",
        inputs=bindings,
        complete_requested_controls=objective.count,
        linearization_path=str(cache.resolve()),
        linearization_sha256=sha256_file(cache),
        source_codec_recanonicalization_max_rad=float(
            np.max(np.abs(np.asarray([canonical_target(target) for target in controls]) - controls))
        ),
        derivative_checks=derivative_checks,
        trials=trials,
        not_a_policy_or_training_parent=True,
        full_future_motion_used_offline=True,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    progress(dict(stage="complete", deployment_ready=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
