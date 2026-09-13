"""Causal source-model prediction, with no native23 policy rollout."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW, MISSING_HW
from gear_sonic.utils.g1_true23_current_momentum_observer import CurrentMomentumSourceModel as VirtualSourceModel

HERE = Path(__file__).parent
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def main():
    output = HERE / "physics_preflight_v1"
    if output.exists():
        raise FileExistsError("prediction preflight refuses overwrite")
    output.mkdir()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("source pin changed: " + str(path))
        inputs[str(path)] = digest
        return path

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    diagnostic = json.loads(
        bind(HERE.parent / "g1_true23_momentum_update_20260910_v1/diagnostic_v1/report.json").read_text()
    )
    assert diagnostic["correction_hypothesis_preflight_justified"]
    for path, digest in diagnostic["inputs"].items():
        bind(path, digest)
    import gear_sonic.utils.g1_true23_current_momentum_observer as corrected_implementation
    import gear_sonic.utils.g1_true23_momentum_source_model as momentum_implementation

    bind(momentum_implementation.__file__)

    bind(corrected_implementation.__file__)
    import gear_sonic.utils.g1_true23_virtual_source_model as implementation

    bind(implementation.__file__)
    records = {}
    cases = {
        "pico": BASE / "normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico",
        **{n: BASE / "original29_normal_v1/actual_v1" / n for n in ("walk002", "walk003", "walk008")},
    }
    for name, directory in cases.items():
        source = json.loads(bind(directory / "report.json").read_text())
        trace_path = (directory / "trace.npz").resolve(strict=True)
        trace_sha = source.get("trace_sha256") or source["inputs"][str(trace_path)]
        with np.load(bind(trace_path, trace_sha), allow_pickle=False) as z:
            # Truth is for comparison ONLY, never passed to the predictor.
            qpos = z["attempt_qpos"].copy()
            qvel = z["attempt_qvel"].copy()
            actions = z["attempt_raw29"].copy()
            truth_next_qpos = z["qpos"][1:].copy()
            truth_next_qvel = z["qvel"][1:].copy()
        parameters = CppParameters(json.loads(bind(source["parameter_path"]).read_text()))
        effort = source["actuation"]["effective_sim_effort29"] if name == "pico" else stock.EFFORT_LIMITS
        predictor = VirtualSourceModel(
            bind(source["model_path"], source["model_sha256"]), parameters, source_effort29=effort
        )
        np.testing.assert_array_equal(qpos[0, 7 + MISSING_HW], parameters.default_angles[MISSING_HW])
        np.testing.assert_array_equal(qvel[0, 6 + MISSING_HW], np.zeros(6))
        predicted_q, predicted_dq = [], []
        for i in range(len(actions)):
            q23 = np.r_[qpos[i, :7], qpos[i, 7 + KEEP_HW]]
            dq23 = np.r_[qvel[i, :6], qvel[i, 6 + KEEP_HW]]
            before_q, before_dq = q23.copy(), dq23.copy()
            predictor.advance(q23, dq23, actions[i])
            np.testing.assert_array_equal(q23, before_q)
            np.testing.assert_array_equal(dq23, before_dq)
            predicted_q.append(predictor.data.qpos[7 + MISSING_HW].copy())
            predicted_dq.append(predictor.data.qvel[6 + MISSING_HW].copy())
        predicted_q, predicted_dq = np.asarray(predicted_q), np.asarray(predicted_dq)
        q_error = float(np.max(np.abs(predicted_q - truth_next_qpos[:, 7 + MISSING_HW])))
        dq_error = float(np.max(np.abs(predicted_dq - truth_next_qvel[:, 6 + MISSING_HW])))
        descriptor = predictor.descriptor()
        records[name] = dict(
            controls=len(actions),
            position_max_abs_rad=q_error,
            velocity_max_abs_rad_s=dq_error,
            passed=q_error <= 1e-5
            and dq_error <= 1e-5
            and descriptor["maximum_predicted_missing_joint_range_excess_rad"] <= 1e-6,
            predictor=descriptor,
            maximum_internal_velocity_assimilation_rad_s=descriptor["maximum_missing_velocity_correction_rad_s"],
            native23_dynamics_performed=False,
            original_source_tracking_qualified=False,
        )
        with (output / (name + ".npz")).open("xb") as stream:
            np.savez_compressed(
                stream,
                predicted_missing_q=predicted_q,
                predicted_missing_dq=predicted_dq,
                reference_missing_q=truth_next_qpos[:, 7 + MISSING_HW],
                reference_missing_dq=truth_next_qvel[:, 6 + MISSING_HW],
            )
        bind(output / (name + ".npz"))
        print(json.dumps(dict(name=name, **records[name])), flush=True)
    report = dict(
        kind="current_measurement_momentum_source29_prediction_preflight_v1",
        records=records,
        inputs=inputs,
        passed=all(r["passed"] for r in records.values()),
        hypothetical_simulator_predictions_not_physical_sensors=True,
        native23_dynamics_performed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=report["passed"], native23_dynamics_performed=False)), flush=True)


if __name__ == "__main__":
    main()
