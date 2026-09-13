"""Same stationary numerical fixture and QP, using an explicit Clarabel adapter.

No production solver or running full-corpus process is modified. This wrapper
changes only its own Python process. Neither fixture replaces a full clip.
"""

import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import clarabel
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "force_trajectory_static_crouch_20260906_v1"
SOLVES = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClarabelAdapter:
    def setup(self, **kwargs):
        matrix, lower, upper = kwargs["A"], kwargs["l"], kwargs["u"]
        equality = np.isfinite(lower) & (lower == upper)
        finite_upper = np.isfinite(upper) & ~equality
        finite_lower = np.isfinite(lower) & ~equality
        cone_matrix = sparse.vstack((matrix[equality], matrix[finite_upper], -matrix[finite_lower]), format="csc")
        rhs = np.r_[lower[equality], upper[finite_upper], -lower[finite_lower]]
        cones = [clarabel.ZeroConeT(int(equality.sum())), clarabel.NonnegativeConeT(int((~equality & np.isfinite(upper)).sum() + finite_lower.sum()))]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.direct_solve_method = "qdldl"
        settings.max_iter = 200
        settings.tol_feas = 1e-9
        settings.tol_gap_abs = 1e-9
        settings.tol_gap_rel = 1e-9
        self.solver = clarabel.DefaultSolver(kwargs["P"], kwargs["q"], cone_matrix, rhs, cones, settings)

    def warm_start(self, **_kwargs):
        # Clarabel initializes its own interior-point state. No seed is claimed.
        pass

    def solve(self, **_kwargs):
        result = self.solver.solve()
        SOLVES.append({
            "backend": "clarabel", "version": clarabel.__version__, "status": str(result.status),
            "iterations": result.iterations, "solve_time_s": result.solve_time,
            "primal_residual": result.r_prim, "dual_residual": result.r_dual,
        })
        return SimpleNamespace(
            x=np.asarray(result.x, dtype=float),
            info=SimpleNamespace(
                status="clarabel_" + str(result.status), status_val=1 if result.status == clarabel.SolverStatus.Solved else 0,
                iter=result.iterations, run_time=result.solve_time, prim_res=result.r_prim, dual_res=result.r_dual,
            ),
        )


def main():
    output = HERE / "report.json"
    if output.exists() or (HERE / "raw_probe.json").exists():
        raise FileExistsError(output)
    previous = json.loads((PREVIOUS / "report.json").read_text())
    source = PREVIOUS / "probe.py"
    assert digest(source) == previous["files"][str(source)]
    wrapper_hash = digest(Path(__file__))
    fixture = runpy.run_path(str(source))
    fixture["main"].__globals__["OUTPUT"] = HERE / "raw_probe.json"
    fixture["g1_true23_force_restoration"].osqp.OSQP = ClarabelAdapter
    fixture["main"]()
    result = json.loads((HERE / "raw_probe.json").read_text())
    result["kind"] = "g1_true23_single_stationary_crouch_clarabel_probe_v1"
    result["solver_backend_override"] = "clarabel==0.11.1, this diagnostic process only"
    result["same_quadratic_objective_and_all_original_linear_bounds"] = True
    result["independent_original_linear_audit_tolerance"] = 1e-8
    result["solver_settings"] = {
        "max_iter": 200, "max_threads": 1, "direct_solve_method": "qdldl",
        "tol_feas": 1e-9, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9,
        "initial_force_seed_used_by_backend": False,
    }
    result["backend_solves"] = SOLVES
    result["files"].update({str(Path(__file__).resolve()): wrapper_hash, str(HERE / "raw_probe.json"): digest(HERE / "raw_probe.json")})
    assert digest(Path(__file__)) == wrapper_hash
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output), "solves": SOLVES}), flush=True)


if __name__ == "__main__":
    main()
