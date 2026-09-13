"""Complete the declared native velocity and total-torque checks on saved probes."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "probe_v1/velocity_audit.json"


def main():
    if OUT.exists():
        raise FileExistsError("velocity audit refuses overwrite")
    report_path = HERE / "probe_v1/report.json"
    report = json.loads(report_path.read_text())
    for path, digest in report["inputs"].items():
        assert sha256_file(Path(path)) == digest, path
    arrays_path = HERE / "probe_v1/probes.npz"
    assert sha256_file(arrays_path) == report["probes_sha256"]
    p = NativeModelActuationProfile.from_sim_config(
        ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    rows = []
    with np.load(arrays_path, allow_pickle=False) as data:
        for i, item in enumerate(report["rows"]):
            key = f"{item['case']}_{i % 4}"
            velocity = np.abs(data[f"{key}_qvel"][:, 6:]) / p.velocity
            pd, total, applied = (
                data[f"{key}_{field}"] for field in ("pd_requested23", "total_requested23", "applied23")
            )
            expected = pd.copy()
            expected[:, 11] += item["feedforward_right_ankle_roll_nm"]
            np.testing.assert_array_equal(expected, total)
            np.testing.assert_array_equal(np.clip(total, -np.asarray(p.effort), p.effort), applied)
            rows.append(
                dict(
                    case=item["case"],
                    feedforward_nm=item["feedforward_right_ankle_roll_nm"],
                    next20ms_velocity_ratio=float(velocity[:10].max()),
                    hold100ms_velocity_ratio=float(velocity.max()),
                    original_next20ms_range_passed=item["next20ms_checked_range_passed"],
                    next20ms_range_and_velocity_passed=bool(
                        item["next20ms_checked_range_passed"] and velocity[:10].max() <= 1
                    ),
                )
            )
    with OUT.open("x") as stream:
        json.dump(
            dict(
                rows=rows,
                total_torque_math_exact=True,
                report_sha256=sha256_file(report_path),
                auditor_sha256=sha256_file(Path(__file__)),
                deployment_ready=False,
                hardware_authorized=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(dict(rows=rows, deployment_ready=False)), flush=True)


if __name__ == "__main__":
    main()
