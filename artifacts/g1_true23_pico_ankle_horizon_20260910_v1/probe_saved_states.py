"""Counterfactual fixed final1000 states, never a resumed actual rollout."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_ankle_horizon_preview import AnkleHorizonNative23RangePreview
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = ROOT / "artifacts/g1_true23_pico_foot_precision_20260910_v1/eval1000_v1/pico"
OUT = HERE / "saved_states_v1.json"


def main():
    if OUT.exists():
        raise FileExistsError("saved-state ankle probes refuse overwrite")
    report = json.loads((BASE / "report.json").read_text())
    assert sha256_file(BASE / "report.json") == "7da5b318d6d30aca2c1469a5abdf1b587623932d6ac7c7e432b7046734420884"
    for path, expected in report["inputs"].items():
        assert sha256_file(Path(path)) == expected, path
    assert sha256_file(BASE / "attempts.npz") == report["attempts_sha256"]
    with np.load(BASE / "attempts.npz", allow_pickle=False) as data:
        attempts = {key: data[key].copy() for key in data.files}
    rows = []
    for i in range(1830, 1881):
        preview = AnkleHorizonNative23RangePreview(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS)
        raw, q, v = (attempts[key][i].copy() for key in ("inverse23", "measured_qpos", "measured_qvel"))
        before = (raw.copy(), q.copy(), v.copy())
        try:
            safe = preview.filter(raw, q, v)
            row = dict(control_index=i, accepted=True, record=preview.records[-1], selected_raw23=safe.tolist())
        except ValueError as error:
            row = dict(control_index=i, accepted=False, failure=str(error), record=preview.failed_search)
        for original, actual in zip(before, (raw, q, v), strict=True):
            np.testing.assert_array_equal(original, actual)
        rows.append(row)
    inputs = {
        str(path): sha256_file(path)
        for path in (
            Path(__file__),
            HERE / "EXPERIMENT.md",
            BASE / "report.json",
            BASE / "attempts.npz",
            ROOT / "gear_sonic/utils/g1_true23_ankle_horizon_preview.py",
            ROOT / "gear_sonic/utils/g1_true23_range_preview.py",
        )
    }
    result = dict(
        inputs=inputs,
        rows=rows,
        state_copies_unchanged=True,
        applied_to_actual_rollout=False,
        counterfactual_constant_target_not_future_policy=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                probes=len(rows),
                accepted=sum(row["accepted"] for row in rows),
                interventions=sum(row["accepted"] and row["record"]["intervened"] for row in rows),
                rejected_indices=[row["control_index"] for row in rows if not row["accepted"]],
                result_sha256=sha256_file(OUT),
                deployment_ready=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
