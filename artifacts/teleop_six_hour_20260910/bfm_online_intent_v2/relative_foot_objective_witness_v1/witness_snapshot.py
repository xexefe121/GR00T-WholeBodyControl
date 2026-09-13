"""Independent native FK/Jacobian witness for an optional relative-foot cost."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, load_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model
from gear_sonic.utils.g1_true23_relative_foot_cost import relative_foot_residual


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(plan, output):
    output.mkdir(parents=True, exist_ok=False)
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    motion, timeline, motion_path = load_motion("walk002")
    names = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")
    ids = [model.body(name).id for name in names]
    with np.load(plan / "trace.npz", allow_pickle=False) as archive:
        qpos = archive["qpos"].copy()
    data = mujoco.MjData(model)
    def features(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return data.xpos[ids].copy()
    witnesses = []
    for control in (0, 400, 650, 873):
        frame = control + 11
        pose = qpos[control + 1]
        positions = features(pose)
        reference = motion["body_pos_w"][frame, np.asarray(ids)-1]
        # MuJoCo's point Jacobians provide an independent analytic derivative.
        jac = []
        for body in ids[:4]:
            j = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, j, None, data.xpos[body], body)
            jac.append(j)
        expected = 20 * np.concatenate((jac[2]-jac[0], jac[3]-jac[0]))
        observed = np.empty_like(expected)
        epsilon = 1e-6
        for axis in range(model.nv):
            direction = np.eye(model.nv)[axis]
            plus, minus = pose.copy(), pose.copy()
            mujoco.mj_integratePos(model, plus, direction, epsilon)
            mujoco.mj_integratePos(model, minus, direction, -epsilon)
            observed[:, axis] = (relative_foot_residual(features(plus), reference)
                                 - relative_foot_residual(features(minus), reference)) / (2*epsilon)
        residual = relative_foot_residual(positions, reference)
        changed = pose.copy()
        changed[:3] += [.17, -.31, .08]
        translated = relative_foot_residual(features(changed), reference)
        explicit_cost = 400 * sum(np.dot((positions[i]-positions[0])-(reference[i]-reference[0]),
                                        (positions[i]-positions[0])-(reference[i]-reference[0])) for i in (2, 3))
        witness = dict(control=control, frame=frame, tangent_jacobian_max_error=float(np.max(np.abs(observed-expected))),
                       translation_invariance_max_error=float(np.max(np.abs(residual-translated))),
                       explicit_objective_error=abs(float(residual @ residual)-explicit_cost),
                       analytic_root_translation_jacobian_max=float(np.max(np.abs(expected[:, :3]))))
        if witness["tangent_jacobian_max_error"] > 1e-6 or witness["translation_invariance_max_error"] > 1e-12 or witness["explicit_objective_error"] > 1e-10:
            raise AssertionError(witness)
        witnesses.append(witness)
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    costs = []
    for control in range(phase["control_start"], phase["control_stop"]):
        frame = control + 11
        actual = features(qpos[control+1])
        reference = motion["body_pos_w"][frame, np.asarray(ids)-1]
        absolute = 400 * np.sum((actual[2:4]-reference[2:4])**2)
        relative = np.sum(relative_foot_residual(actual, reference)**2)
        root = np.sum((actual[0]-reference[0])**2 * [200, 200, 1000])
        costs.append([absolute, relative, root])
    helper = ROOT / "gear_sonic/utils/g1_true23_relative_foot_cost.py"
    report = dict(kind="independent_relative_foot_objective_witness", mujoco=mujoco.__version__, weight=400,
                  all_witnesses_pass=True, witnesses=witnesses, source_controls=len(costs),
                  measured_cost_columns=["existing_absolute_feet", "proposed_relative_feet", "existing_root"],
                  measured_cost_p50_p95_max=np.percentile(costs, [50, 95, 100], axis=0).tolist(),
                  trace_sha256=sha(plan / "trace.npz"), motion_sha256=sha(motion_path), helper_sha256=sha(helper),
                  witness_sha256=sha(__file__), controller_modified=False, physical_rollout_performed=False,
                  controller_quality_improvement_proven=False, hardware_authorized=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "helper_snapshot.py").write_bytes(helper.read_bytes())
    (output / "witness_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
