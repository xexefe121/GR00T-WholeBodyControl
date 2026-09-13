"""Localize solved ankle forces in fixed windows; no controller execution."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_force_balance import COMPONENTS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "capture_v2"


def main():
    output = CAPTURE / "window_analysis.json"
    if output.exists():
        raise FileExistsError("force window analysis refuses overwrite")
    report = json.loads((CAPTURE / "report.json").read_text())
    pins = {
        str(HERE / "analyze.py"): sha256_file(HERE / "analyze.py"),
        str(CAPTURE / "report.json"): sha256_file(CAPTURE / "report.json"),
    }
    rows = []
    for name in ("parent500", "candidate1000", *(f"rejected_preview_{i}" for i in range(8))):
        path = CAPTURE / name / "forces.npz"
        summary = json.loads((path.parent / "summary.json").read_text())
        assert sha256_file(path) == summary["arrays_sha256"]
        pins[str(path)] = summary["arrays_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        windows = (250, 50, 10) if not name.startswith("rejected") else (10,)
        for count in windows:
            joint, dof = 11, 17
            forces = data["force_components"][-count:, :, dof]
            acceleration = data["acceleration_components"][-count:, :, dof]
            own = data["self_actuator_acceleration"][-count:, dof]
            row = dict(
                name=name,
                window_s=count * 0.002,
                q_start_rad=float(data["pre_qpos"][-count, 7 + joint]),
                q_end_rad=float(data["post_qpos"][-1, 7 + joint]),
                dq_start_rad_s=float(data["pre_qvel"][-count, dof]),
                dq_end_rad_s=float(data["post_qvel"][-1, dof]),
                target_min_max_rad=[
                    float(data["target23"][-count:, joint].min()),
                    float(data["target23"][-count:, joint].max()),
                ],
                requested_torque_min_max_nm=[
                    float(data["requested_torque23"][-count:, joint].min()),
                    float(data["requested_torque23"][-count:, joint].max()),
                ],
                mean_force_components_nm={key: float(forces[:, i].mean()) for i, key in enumerate(COMPONENTS)},
                mean_acceleration_components_rad_s2={
                    key: float(acceleration[:, i].mean()) for i, key in enumerate(COMPONENTS)
                },
                own_actuator_acceleration_rad_s2=float(own.mean()),
                other_actuator_acceleration_rad_s2=float((acceleration[:, 0] - own).mean()),
                mean_forward_acceleration_rad_s2=float(data["forward_acceleration"][-count:, dof].mean()),
                measured_mean_acceleration_rad_s2=float(
                    (data["post_qvel"][-1, dof] - data["pre_qvel"][-count, dof]) / (count * 0.002)
                ),
                foot_load_mean_n=data["foot_normal_load_n"][-count:].mean(0).tolist(),
                foot_load_last_n=data["foot_normal_load_n"][-1].tolist(),
            )
            if name.startswith("rejected"):
                i = int(name.rsplit("_", 1)[1])
                candidate = report["preview_candidates"][i]
                row.update(
                    lower_reserve_excess23=candidate["lower_reserve_excess23"],
                    upper_reserve_excess23=candidate["upper_reserve_excess23"],
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(
        kind="fixed_right_ankle_roll_force_window_analysis_v1",
        rows=rows,
        inputs=pins,
        fixed_solved_contact_decomposition_not_counterfactual_causality=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
