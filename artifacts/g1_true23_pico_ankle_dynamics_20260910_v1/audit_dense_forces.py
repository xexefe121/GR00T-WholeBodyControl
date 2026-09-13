"""Independent dense constraint-Jacobian arithmetic; no force observer import."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
CAPTURE = HERE / "capture_v2"
OUT = CAPTURE / "independent_dense_audit.json"


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    if OUT.exists():
        raise FileExistsError("dense force audit refuses overwrite")
    report = json.loads((CAPTURE / "report.json").read_text())
    for path, digest in report["inputs"].items():
        assert sha256_file(Path(path)) == digest, path
    rows = []
    for name, family, run in (
        ("parent500", "g1_true23_pico_training_20260910_v1", "eval500_v1"),
        ("candidate1000", "g1_true23_pico_foot_precision_20260910_v1", "eval1000_v1"),
    ):
        directory = ROOT / "artifacts" / family / run / "pico"
        original_report = json.loads((directory / "report.json").read_text())
        assert sha256_file(directory / "report.json") == report["source_reports"][name]
        assert sha256_file(directory / "trace.npz") == original_report["trace_sha256"]
        assert sha256_file(CAPTURE / name / "forces.npz") == report["cases"][name]["arrays_sha256"]
        original, saved = read(directory / "trace.npz"), read(CAPTURE / name / "forces.npz")
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        assert not mujoco.mj_isSparse(c.model), "audit deliberately requires original dense Jacobian"
        q, v = original["qpos"][0], original["qvel"][0]
        c.reset(
            base_position=q[:3],
            base_quaternion_wxyz=q[3:7],
            joint_position_hardware=q[7:],
            root_velocity=v[:6],
            joint_velocity_hardware=v[6:],
        )
        feet = [c.model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
        floor = c.model.geom("floor").id
        errors = dict(
            generalized_constraint=0.0, all_components=0.0, mass=0.0, acceleration=0.0, foot_normal_load=0.0
        )
        observed = 0
        for step in range(18800):
            np.testing.assert_array_equal(c.data.qpos, original["physics_pre_qpos"][step])
            np.testing.assert_array_equal(c.data.qvel, original["physics_pre_qvel"][step])
            c.data.ctrl[:] = original["applied_torque23"][step]
            mujoco.mj_step(c.model, c.data)
            np.testing.assert_array_equal(c.data.qpos, original["physics_post_qpos"][step])
            np.testing.assert_array_equal(c.data.qvel, original["physics_post_qvel"][step])
            if step < 17800:
                continue
            k = step - 17800
            jac = c.data.efc_J.reshape(c.data.nefc, c.model.nv)
            types, ids, solved = c.data.efc_type, c.data.efc_id, c.data.efc_force
            masks = np.zeros((5, c.data.nefc), dtype=bool)
            is_contact = np.isin(
                types,
                [
                    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
                ],
            )
            loads = np.zeros(2)
            for i in range(c.data.ncon):
                con = c.data.contact[i]
                if con.efc_address < 0:
                    continue
                other = con.geom2 if con.geom1 == floor else con.geom1 if con.geom2 == floor else None
                if other is None or int(c.model.geom_bodyid[other]) not in feet:
                    continue
                f = feet.index(int(c.model.geom_bodyid[other]))
                masks[f] |= is_contact & (ids == i)
                # Independent normal load from solved cone coordinates, without
                # mj_contactForce or the original observer's contact records.
                if con.dim == 1 or c.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
                    loads[f] += solved[con.efc_address]
                else:
                    loads[f] += solved[con.efc_address : con.efc_address + 2 * (con.dim - 1)].sum()
            masks[2] = is_contact & ~masks[:2].any(0)
            masks[3] = types == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
            masks[4] = ~masks[:4].any(0)
            assert np.all(masks.sum(0) == 1)
            forces = np.zeros((9, c.model.nv))
            forces[:4] = c.data.qfrc_actuator, c.data.qfrc_passive, -c.data.qfrc_bias, c.data.qfrc_applied
            forces[4:] = np.asarray([jac.T @ (solved * mask) for mask in masks])
            mass = np.empty((c.model.nv, c.model.nv))
            mujoco.mj_fullM(c.model, mass, c.data.qM)
            differences = dict(
                generalized_constraint=np.max(np.abs(jac.T @ solved - saved["qfrc_constraint"][k])),
                all_components=np.max(np.abs(forces - saved["force_components"][k])),
                mass=np.max(np.abs(mass - saved["mass"][k])),
                acceleration=np.max(
                    np.abs(np.linalg.solve(mass, forces.T).T - saved["acceleration_components"][k])
                ),
                foot_normal_load=np.max(np.abs(loads - saved["foot_normal_load_n"][k])),
            )
            for key, value in differences.items():
                errors[key] = max(errors[key], float(value))
            observed += 1
        assert observed == 1000
        assert max(errors.values()) < 1e-7, errors
        row = dict(name=name, exact_prefix_substeps=18800, dense_force_samples=observed, maximum_errors=errors)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with OUT.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                observer_imported=False,
                mj_mulJacTVec_or_mj_contactForce_used=False,
                report_sha256=sha256_file(CAPTURE / "report.json"),
                auditor_sha256=sha256_file(Path(__file__)),
                evidence_consistency_passed=True,
                rejected_preview_forces_reaudited=False,
                controller_qualification_claimed=False,
                deployment_ready=False,
                hardware_authorized=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
