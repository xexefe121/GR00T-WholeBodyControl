"""Conservative support-patch experiment on one stationary pose, not a full clip."""

import inspect
import json
from pathlib import Path
import runpy

import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.utils import g1_true23_contact_patch
from gear_sonic.utils.g1_true23_contact_patch import FrozenFloorSupportPatch
from gear_sonic.utils.g1_true23_force_restoration import ForceRestorationConfig
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "report.json"
    if output.exists() or (HERE / "raw_probe.json").exists():
        raise FileExistsError(output)
    backend_path = HERE.parent / "force_trajectory_static_crouch_clarabel_20260906_v1/probe.py"
    backend = runpy.run_path(str(backend_path))

    class ConstantPoseBackend(backend["ClarabelAdapter"]):
        def setup(self, **kwargs):
            expansion = sparse.block_diag((
                sparse.vstack([sparse.eye(26)] * 3, format="csc"),
                sparse.eye(len(kwargs["q"]) - 78),
            ), format="csc")
            kwargs["P"] = (expansion.T @ kwargs["P"] @ expansion).tocsc()
            kwargs["q"] = np.asarray(expansion.T @ kwargs["q"])
            matrix = (kwargs["A"] @ expansion).tocsc()
            matrix.eliminate_zeros()
            nonzero = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
            if np.any(kwargs["l"][~nonzero] > 0) or np.any(kwargs["u"][~nonzero] < 0):
                raise ValueError("exact constant-pose fixture has contradictory zero rows")
            kwargs["A"], kwargs["l"], kwargs["u"] = matrix[nonzero], kwargs["l"][nonzero], kwargs["u"][nonzero]
            self.pose_expansion = expansion
            super().setup(**kwargs)

        def solve(self, **kwargs):
            result = super().solve(**kwargs)
            result.x = np.asarray(self.pose_expansion @ result.x)
            return result

    backend["ClarabelAdapter"] = ConstantPoseBackend

    source = HERE.parent / "force_trajectory_static_crouch_20260906_v1/probe.py"
    fixture = runpy.run_path(str(source))
    seed_path = HERE.parent / "force_trajectory_exact_static_crouch_patch_20260906_v1/report.json"
    seed = json.loads(seed_path.read_text())
    if seed["original_source_frame"] != 512 or seed["full_clip_test"] is not False:
        raise ValueError("expected the rejected single-pose diagnostic seed")
    original_restore = fixture["restore_force_trajectory"]
    original_step = fixture["g1_true23_force_restoration"].force_restoration_step

    def seeded_restore(_desired, *args, **kwargs):
        return original_restore(np.tile(seed["static_path"], (3, 1)), *args, **kwargs)

    def minimum_step(current, _desired, *args, **kwargs):
        return original_step(current, current, *args, **kwargs)

    fixture["main"].__globals__["restore_force_trajectory"] = seeded_restore
    fixture["g1_true23_force_restoration"].force_restoration_step = minimum_step

    fixture["main"].__globals__["OUTPUT"] = HERE / "raw_probe.json"
    original_force = fixture["ForceLinearization"].evaluate
    patch_rows, force_evaluations, patch_audits = {}, [], []
    config = ForceRestorationConfig(maximum_iterations=64, root_trust_m=0.015 / 16, joint_trust_rad=0.12 / 16)
    fixture["main"].__globals__["ForceRestorationConfig"] = lambda: config

    def force_with_patch(self, path, **kwargs):
        result = original_force(self, path, **kwargs)
        force_evaluations.append({
            "linearizing": kwargs.get("jacobian", True),
            "l1_residual": float(np.abs(result["normalized_residual"]).sum()),
            "no_contact_frames": [row["frames_without_candidate_contact"] for row in result["models"]],
        })
        if kwargs.get("jacobian", True):
            values, matrices = [], []
            for inverse, model, data, plane in self.models.values():
                poses, _, _ = inverse.state(path)
                frame_matrices = []
                for qpos in poses:
                    data.qpos[:] = qpos
                    mujoco.mj_fwdPosition(model, data)
                    _, contacts = floor_contact_map(model, data, plane, self.gap)
                    patch = FrozenFloorSupportPatch(model, data, contacts, plane, gap_m=self.gap)
                    value, jacobian = patch.evaluate(qpos)
                    values.extend(value)
                    frame_matrices.append(sparse.csc_matrix(jacobian))
                matrices.append(sparse.block_diag(frame_matrices, format="csc"))
            scale = np.tile(np.r_[np.full(3, config.root_trust_m), np.full(23, config.joint_trust_rad)], len(path))
            patch_rows["values"] = np.asarray(values) / 0.001
            patch_rows["matrix"] = sparse.vstack(matrices, format="csc") @ sparse.diags(scale) / 0.001
        return result

    class PatchBackend(backend["ClarabelAdapter"]):
        def setup(self, **kwargs):
            matrix = patch_rows["matrix"]
            extra = sparse.hstack((matrix, sparse.csc_matrix((matrix.shape[0], len(kwargs["q"]) - matrix.shape[1]))), format="csc")
            kwargs["A"] = sparse.vstack((kwargs["A"], extra), format="csc")
            kwargs["l"] = np.r_[kwargs["l"], -patch_rows["values"]]
            kwargs["u"] = np.r_[kwargs["u"], np.full(extra.shape[0], np.inf)]
            self.patch_matrix, self.patch_lower = extra, -patch_rows["values"]
            super().setup(**kwargs)

        def solve(self, **kwargs):
            result = super().solve(**kwargs)
            violation = float(np.maximum(self.patch_lower - self.patch_matrix @ result.x, 0).max(initial=0))
            patch_audits.append({"rows": len(self.patch_lower), "maximum_scaled_linear_violation": violation})
            if violation > 1e-8 or not np.isfinite(violation):
                result.info.status_val = 0
                result.info.status = "independent_patch_linear_audit_failed"
            return result

    fixture["ForceLinearization"].evaluate = force_with_patch
    fixture["g1_true23_force_restoration"].osqp.OSQP = PatchBackend
    fixture["main"]()
    result = json.loads((HERE / "raw_probe.json").read_text())
    result["kind"] = "g1_true23_minimum_step_stationary_crouch_diagnostic_v1"
    result["continued_from_rejected_static_seed"] = {"path": str(seed_path), "sha256": backend["digest"](seed_path)}
    result["qp_path_cost_centered_on_current_iterate_for_feasibility_restoration"] = True
    result["shared_pose_variables"] = 26
    result["repeated_pose_variables_eliminated"] = 52
    result["exact_stationarity_is_a_fixture_constraint_not_a_whole_clip_change"] = True
    result["candidate_point_guard_m"] = 0.00005
    result["candidate_point_constraints_are_hard_linearized_trial_rows"] = True
    result["nonlinear_actual_contact_and_force_fits_recomputed_for_every_trial"] = True
    result["force_evaluations_in_call_order"] = force_evaluations
    result["patch_linear_audits"] = patch_audits
    result["backend_solves"] = backend["SOLVES"]
    result["files"].update({str(path.resolve()): backend["digest"](path) for path in (
        Path(__file__), backend_path, seed_path, HERE / "raw_probe.json", Path(inspect.getfile(g1_true23_contact_patch)),
    )})
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": backend["digest"](output), "solver_steps": len(patch_audits), "failure": result["solver"]["failure"], "last_force": force_evaluations[-1]}), flush=True)


if __name__ == "__main__":
    main()
