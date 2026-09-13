"""Explain the rejected stationary QP step; no changed limits or motion qualification."""

import json
from pathlib import Path
import runpy

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "report.json"
    if output.exists() or (HERE / "raw_probe.json").exists():
        raise FileExistsError(output)
    backend_path = HERE.parent / "force_trajectory_static_crouch_clarabel_20260906_v1/probe.py"
    backend = runpy.run_path(str(backend_path))
    source = HERE.parent / "force_trajectory_static_crouch_20260906_v1/probe.py"
    fixture = runpy.run_path(str(source))
    fixture["main"].__globals__["OUTPUT"] = HERE / "raw_probe.json"
    restoration = fixture["g1_true23_force_restoration"]
    restoration.osqp.OSQP = backend["ClarabelAdapter"]
    original_force = fixture["ForceLinearization"].evaluate
    original_contacts = fixture["ContactLinearization"].audit
    original_step = restoration.force_restoration_step
    diagnostics, geometry, paths = [], [], {}

    def audit_force(self, path, **kwargs):
        value = original_force(self, path, **kwargs)
        # Exact L1 residual fit on this three-frame diagnostic only. It does
        # not change the optimizer or its actual line-search merit.
        mapping = value["force_map"].toarray() / value["scale"][:, None]
        required = value["required"] / value["scale"]
        nf, nr = mapping.shape[1], len(required)
        fit = linprog(
            np.r_[np.zeros(nf), np.ones(nr)],
            A_ub=np.vstack((np.column_stack((mapping, -np.eye(nr))), np.column_stack((-mapping, -np.eye(nr))))),
            b_ub=np.r_[required, -required],
            bounds=list(zip(value["lower"], value["upper"])) + [(0, None)] * nr,
            method="highs", options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        diagnostics.append({
            "linearizing": kwargs.get("jacobian", True),
            "l2_seed_l1_residual": float(np.abs(value["normalized_residual"]).sum()),
            "l2_seed_l2_squared": value["summed_normalized_force_residual_squared"],
            "optimal_l1_fit_success": bool(fit.success),
            "optimal_l1_residual": float(fit.fun) if fit.success else None,
            "max_physical_residual": value["maximum_absolute_generalized_force_residual"],
            "no_contact_frames": [row["frames_without_candidate_contact"] for row in value["models"]],
            "force_variables": nf,
        })
        return value

    def audit_contacts(self, path):
        value = original_contacts(self, path)
        geometry.append({key: value[key] for key in ("passed", "violated_frames", "summed_violation_m", "maximum_violation_m")})
        return value

    def audit_step(*args, **kwargs):
        value, report = original_step(*args, **kwargs)
        paths["current"] = args[0]
        if value is not None:
            paths["candidate"] = value
        return value, report

    fixture["ForceLinearization"].evaluate = audit_force
    fixture["ContactLinearization"].audit = audit_contacts
    restoration.force_restoration_step = audit_step
    fixture["main"]()
    path_file = HERE / "qp_paths.npz"
    with path_file.open("xb") as stream:
        np.savez_compressed(stream, **paths)
    result = json.loads((HERE / "raw_probe.json").read_text())
    result["kind"] = "g1_true23_stationary_crouch_linesearch_diagnostic_v1"
    result["force_evaluations_in_call_order"] = diagnostics
    result["contact_evaluations_in_call_order"] = geometry
    result["backend_solves"] = backend["SOLVES"]
    result["files"].update({str(path.resolve()): backend["digest"](path) for path in (
        Path(__file__), backend_path, HERE / "raw_probe.json", path_file,
    )})
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "force": diagnostics, "geometry": geometry}), flush=True)


if __name__ == "__main__":
    main()
