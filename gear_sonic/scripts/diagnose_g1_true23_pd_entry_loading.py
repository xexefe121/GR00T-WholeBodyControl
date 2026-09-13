"""Measure actual entry foot loading/COM on an authenticated offline trajectory."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import assert_bindings, load_arrays
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report
from gear_sonic.utils.g1_true23_reference_floor import motion_qpos
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-report", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    historical = verify_historical_report(
        args.optimizer_report, repository_root=root, source_archive=args.source_archive
    )
    report, reproduction = [
        json.loads(path.read_text()) for path in (args.optimizer_report, args.reproduction_report)
    ]
    assert_bindings(reproduction["inputs"])
    trace_path, reference_path = (
        Path(report["candidate_trace_path"]),
        Path(report["original_requested_motion_path"]),
    )
    parent_path = Path(reproduction["parent_evaluation"])
    parent = json.loads(parent_path.read_text())
    assert_bindings(
        {
            str(trace_path): report["candidate_trace_sha256"],
            str(reference_path): report["original_requested_motion_sha256"],
        }
    )
    paths = [
        args.optimizer_report,
        args.source_archive,
        args.reproduction_report,
        trace_path,
        reference_path,
        parent_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    pelvis = model.body("pelvis").id
    phase = next(row for row in parent["timeline"]["phases"] if row["name"] == "acquisition_ramp")
    start, stop = phase["control_start"], phase["control_stop"]
    loads, control = [], -1

    class ObservedPlant(PdShootingPlant):
        def _tick(self, data, target):
            result = super()._tick(data, target)
            if control >= start:
                normal = np.zeros(2)
                for index, contact in enumerate(data.contact[: data.ncon]):
                    bodies = [int(model.geom_bodyid[int(geom)]) for geom in (contact.geom1, contact.geom2)]
                    if 0 in bodies:
                        body = bodies[1 - bodies.index(0)]
                        if body in feet:
                            force = np.empty(6)
                            mujoco.mj_contactForce(model, data, index, force)
                            normal[feet.index(body)] += max(0.0, float(force[0]))
                loads.append(normal)
            return result

    plant = ObservedPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != report["compiled_physics_sha256"]:
        raise ValueError("entry loading diagnostic changed original physics")
    trace, motion = load_arrays(trace_path), load_arrays(reference_path)
    references = motion_qpos(model, motion)
    data, metric = plant.scratch(trace["integration_state"][0]), mujoco.MjData(model)
    actual_feet, requested_feet, actual_com, requested_com = [], [], [], []

    def geometry(pose):
        metric.qpos[:], metric.qvel[:] = pose, 0
        mujoco.mj_forward(model, metric)
        return metric.xpos[feet].copy(), metric.subtree_com[pelvis].copy()

    for control in range(stop):
        plant.integrate_control(data, trace["target23"][control])
        if plant.state(data).tobytes() != trace["integration_state"][control + 1].tobytes():
            raise ValueError(f"read-only loading observation changed physical replay at control {control}")
        if control >= start:
            foot, com = geometry(data.qpos)
            actual_feet.append(foot)
            actual_com.append(com)
            foot, com = geometry(references[control + 11])
            requested_feet.append(foot)
            requested_com.append(com)
    actual_feet, requested_feet, actual_com, requested_com, loads = map(
        np.asarray, (actual_feet, requested_feet, actual_com, requested_com, loads)
    )
    mean_loads = loads.reshape(stop - start, 10, 2).mean(axis=1)
    summaries = []
    for side, name in enumerate(("left", "right")):
        desired_z, actual_z = requested_feet[:, side, 2], actual_feet[:, side, 2]
        apex = int(np.argmax(desired_z))
        requested_air = desired_z > float(np.min(desired_z)) + 0.02
        selected = mean_loads[requested_air, side]
        summaries.append(
            dict(
                foot=name,
                requested_apex_control=start + apex,
                requested_ankle_apex_z_m=float(desired_z[apex]),
                actual_ankle_z_at_requested_apex_m=float(actual_z[apex]),
                actual_maximum_ankle_z_m=float(actual_z.max()),
                actual_maximum_rise_from_entry_start_m=float(actual_z.max() - actual_z[0]),
                requested_air_controls=int(requested_air.sum()),
                mean_normal_load_during_requested_air_n=None if not len(selected) else float(selected.mean()),
                minimum_control_mean_load_during_requested_air_n=None
                if not len(selected)
                else float(selected.min()),
                fraction_requested_air_controls_below_5n=None
                if not len(selected)
                else float(np.mean(selected < 5.0)),
                entry_end_position_error_xyz_m=(actual_feet[-1, side] - requested_feet[-1, side]).tolist(),
            )
        )
    assert_bindings(bindings)
    plant.assert_unchanged()
    result = dict(
        kind="g1_true23_original_pd_entry_weight_transfer_diagnostic_v1",
        inputs=bindings,
        historical_input_evidence=historical,
        integration_states_reproduced_exactly=stop,
        observed_entry_controls=stop - start,
        summaries=summaries,
        com_xy_tracking_p95_m=float(
            np.percentile(np.linalg.norm(actual_com[:, :2] - requested_com[:, :2], axis=1), 95)
        ),
        com_xy_error_at_each_requested_apex_m=[
            (
                actual_com[row["requested_apex_control"] - start, :2]
                - requested_com[row["requested_apex_control"] - start, :2]
            ).tolist()
            for row in summaries
        ],
        diagnostic_5n_reporting_threshold_not_qualification=True,
        original_29dof_fidelity_not_measured=True,
        prefix_diagnostic_not_complete_motion_evidence=True,
        simulator_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("inputs", "historical_input_evidence")}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
