"""Measure old observer's implicit impulse without changing its predictions."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_momentum_source_model import MissingMomentumAssimilation
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW, VirtualSourceModel

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "g1_true23_virtual_state_20260910_v1/native_actual_v1"
SOURCE = Path(
    "/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico"
)


def main():
    output = HERE / "diagnostic_v1"
    if output.exists():
        raise FileExistsError("diagnostic refuses overwrite")
    output.mkdir()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("diagnostic input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    import gear_sonic.utils.g1_true23_momentum_source_model as implementation
    import gear_sonic.utils.g1_true23_virtual_source_model as previous

    bind(implementation.__file__)
    bind(previous.__file__)
    old = json.loads(
        bind(OLD / "report.json", "363d7d9f9b091cbe0f60fec5aed69f14934804230db6b4245f3c5373b9a5649f").read_text()
    )
    source = json.loads(bind(SOURCE / "report.json").read_text())
    p = CppParameters(json.loads(bind(source["parameter_path"]).read_text()))
    with np.load(bind(OLD / "attempts.npz", old["inputs"][str(OLD / "attempts.npz")]), allow_pickle=False) as z:
        native_q, native_v, native_raw = (
            z[k].copy() for k in ("measured_qpos", "measured_qvel", "released_raw29")
        )
        expected = z["predicted_next_missing_state12"].copy()
    with np.load(bind(SOURCE / "trace.npz", source["trace_sha256"]), allow_pickle=False) as z:
        source_q, source_v, source_raw = (z[k].copy() for k in ("attempt_qpos", "attempt_qvel", "attempt_raw29"))
    count = len(native_raw)
    records = {}
    for name in ("native_copy_update", "matched_original_source"):
        predictor = VirtualSourceModel(
            bind(source["model_path"], source["model_sha256"]),
            p,
            source_effort29=source["actuation"]["effective_sim_effort29"],
        )
        proposal = MissingMomentumAssimilation(predictor.model)
        rows = []
        for i in range(count):
            if name == "native_copy_update":
                q, v, raw = native_q[i], native_v[i], native_raw[i]
            else:
                q = np.r_[source_q[i, :7], source_q[i, 7 + KEEP_HW]]
                v = np.r_[source_v[i, :6], source_v[i, 6 + KEEP_HW]]
                raw = source_raw[i]
            if predictor.initialized:
                old_q, old_v = predictor.data.qpos.copy(), predictor.data.qvel.copy()
                _, record = proposal.proposal(old_q, old_v, q, v)
                np.testing.assert_array_equal(predictor.data.qpos, old_q)
                np.testing.assert_array_equal(predictor.data.qvel, old_v)
            else:
                record = dict(
                    exact_identity=True,
                    missing_momentum_jump=np.zeros(6),
                    missing_velocity_correction=np.zeros(6),
                    momentum_residual=np.zeros(6),
                    conditional_correction_energy_j=0.0,
                )
            rows.append(record)
            state = predictor.advance(q, v, raw)
            if name == "native_copy_update" and i < len(expected):
                np.testing.assert_array_equal(state, expected[i])
        values = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        with (output / (name + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, **values)
        bind(output / (name + ".npz"))
        records[name] = dict(
            controls=count,
            old_predictions_unchanged=True,
            missing_velocity_correction_abs_p95_rad_s=np.percentile(
                np.abs(values["missing_velocity_correction"]), 95, axis=0
            ).tolist(),
            missing_velocity_correction_abs_max_rad_s=np.abs(values["missing_velocity_correction"])
            .max(axis=0)
            .tolist(),
            missing_momentum_jump_abs_max=np.abs(values["missing_momentum_jump"]).max(axis=0).tolist(),
            correction_energy_max_j=float(values["conditional_correction_energy_j"].max()),
            last20_waist_roll_velocity_correction=values["missing_velocity_correction"][-20:, 0].tolist(),
            exact_identity_controls=int(values["exact_identity"].sum()),
        )
        print(json.dumps(dict(name=name, **records[name])), flush=True)
    justified = (
        records["native_copy_update"]["missing_velocity_correction_abs_p95_rad_s"][0] > 0.05
        and max(records["matched_original_source"]["missing_velocity_correction_abs_max_rad_s"]) <= 1e-10
    )
    result = dict(
        kind="source_model_measurement_momentum_mismatch_diagnostic_v1",
        inputs=inputs,
        records=records,
        correction_hypothesis_preflight_justified=justified,
        instability_cause_proven=False,
        new_native23_rollouts=0,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(correction_hypothesis_preflight_justified=justified)), flush=True)


if __name__ == "__main__":
    main()
