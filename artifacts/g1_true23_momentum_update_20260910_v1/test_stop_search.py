"""One bounded coordinated-target search at the saved rejected native state."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_coordinated_range_preview import CoordinatedNative23RangePreview
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS


def main():
    here = Path(__file__).resolve().parent
    output = here / "coordinated_stop_v1"
    if output.exists():
        raise FileExistsError("coordinated stop probe refuses overwrite")
    output.mkdir()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("coordinated stop input changed")
        inputs[str(path)] = digest
        return path

    bind(__file__)
    import gear_sonic.utils.g1_true23_coordinated_range_preview as implementation

    bind(implementation.__file__)
    report = json.loads(
        bind(
            here / "native_actual_v1/report.json",
            "2294d1381ca34ae9470731d27ecdfd7e74c60c19e37f8256ff97e3836df78f6b",
        ).read_text()
    )
    a_path = here / "native_actual_v1/attempts.npz"
    with np.load(bind(a_path, report["inputs"][str(a_path)]), allow_pickle=False) as z:
        q, v, raw = (z[k][-1].copy() for k in ("measured_qpos", "measured_qvel", "inverse23"))
    before_q, before_v = q.copy(), v.copy()
    guard = CoordinatedNative23RangePreview(
        model_path=bind(Path("/mnt/z/codex/GR00T-WholeBodyControl") / MODEL),
        physics_path=bind(here.parents[1] / PHYSICS),
    )
    found, failure = None, None
    try:
        found = guard.filter(raw, q, v)
    except ValueError as error:
        failure = str(error)
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(v, before_v)
    result = dict(
        kind="native23_saved_stop_coordinated_leg_search_v1",
        inputs=inputs,
        found_verified_next_control_target=found is not None,
        failure=failure,
        raw23=None if found is None else found.tolist(),
        guard=guard.contract(),
        saved_control_index=1980,
        new_closed_loop_policy_trials=0,
        full_source_tracking_or_timing_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
