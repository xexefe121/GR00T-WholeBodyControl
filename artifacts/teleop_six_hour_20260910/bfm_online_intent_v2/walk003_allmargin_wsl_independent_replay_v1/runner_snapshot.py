"""Independent continuous native323 recorded-target physical replay.

No planner/controller imports, no producer state copies, no state resets after
reference-frame10 initialization. This is an offline target replay, not MPC or
live teleoperation qualification. All native engine warnings are saved at 2ms.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(args):
    assert mujoco.__version__ == "3.2.3"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    request = json.loads((args.producer / "request.json").read_text())
    producer_report = json.loads((args.producer / "report.json").read_text())
    assert request["clip"] == "walk003" and request["physical_initialization"] == "optimization_reference_frame10"
    for name in ("request", "trace", "plans"):
        extension = ".npz" if name == "trace" else ".json"
        assert sha(args.producer / (name + extension)) == producer_report[name + "_sha256"]
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    assert manifest == request["model_manifest"]
    for name, field in (("native_prepared.xml", "portable_xml_sha256"),
                        ("prepared_model_arrays.npz", "prepared_arrays_sha256")):
        assert sha(args.bundle / name) == manifest[field]
    for name, digest in manifest["meshes"].items():
        assert sha(args.bundle / "meshes" / name) == digest
    for name in ("manifest.json", "contract.json", "walk003/native_original.npz", "walk003/timeline.json"):
        matches = [value for key, value in request["input_hashes"].items()
                   if key.replace("\\", "/").endswith("/mjbatch_native23_inputs_v1/" + name)]
        assert matches == [sha(args.bundle / name)], name
    native = mujoco.MjModel.from_xml_path(str(args.bundle / "native_prepared.xml"))
    prepared = archive(args.bundle / "prepared_model_arrays.npz")
    for key, value in prepared.items():
        getattr(native, key)[:] = value
    mujoco.mj_setConst(native, mujoco.MjData(native))
    for key, value in prepared.items():
        np.testing.assert_array_equal(getattr(native, key), value, err_msg=key)
    contract = json.loads((args.bundle / "contract.json").read_text())
    assert (native.nq, native.nv, native.nu, native.nbody) == (30, 29, 23, 25)
    assert native.opt.timestep == contract["timestep"] == .002 and contract["decimation"] == 10
    assert native.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert [native.joint(i).name for i in range(1, 24)] == contract["joint_names"]
    np.testing.assert_array_equal(native.jnt_range[1:], contract["joint_limits"])
    kp, kd, effort, speed = [np.asarray(contract[k]) for k in ("kp", "kd", "native_effort", "native_velocity")]
    motion = archive(args.reference)
    override = request["motion_override"]
    assert sha(args.reference) == override["reference_sha256"]
    assert sha(args.reference.parent / "portable_receipt.json") == override["portable_receipt_sha256"]
    assert sha(args.bundle / "walk003/native_original.npz") == override["base_native_reference_sha256"]
    assert sha(args.bundle / "walk003/original29.npz") == override["original29_sha256"]
    timeline = json.loads((args.bundle / "walk003/timeline.json").read_text())
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    assert phase == request["source_phase"]
    source = archive(args.producer / "trace.npz")
    count = len(source["target"])
    assert count == 1569 == request["requested_controls"] == len(motion["joint_pos"]) - 11
    assert phase["requested_controls"] == 819
    np.testing.assert_array_equal(source["source_frame"], np.arange(count) + 11)
    np.testing.assert_array_equal(source["physics_substeps"], np.full(count, 10))
    for name, shape in dict(qpos=(count+1,30),qvel=(count+1,29),target=(count,23),
                            physics_qpos=(count*10+1,30),physics_qvel=(count*10+1,29),
                            physics_torque=(count*10,23)).items():
        assert source[name].shape == shape and np.isfinite(source[name]).all(), name
    np.testing.assert_array_equal(source["qpos"], source["physics_qpos"][::10])
    np.testing.assert_array_equal(source["qvel"], source["physics_qvel"][::10])
    q = motion["body_quat_w"][10, 0]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q)
    initial_qpos = np.r_[motion["body_pos_w"][10, 0], q, motion["joint_pos"][10]]
    initial_qvel = np.r_[motion["body_lin_vel_w"][10, 0],
                         rotation.reshape(3, 3).T @ motion["body_ang_vel_w"][10, 0],
                         motion["joint_vel"][10]]
    np.testing.assert_array_equal(initial_qpos, source["qpos"][0])
    np.testing.assert_array_equal(initial_qvel, source["qvel"][0])
    np.testing.assert_array_equal(initial_qpos, request["initial_qpos"])
    np.testing.assert_array_equal(initial_qvel, request["initial_qvel"])
    relevant = [Path(__file__), args.producer / "request.json", args.producer / "trace.npz",
                args.producer / "report.json", args.producer / "plans.json", args.reference,
                args.reference.parent / "portable_receipt.json"]
    relevant += [args.bundle / name for name in ("native_prepared.xml", "prepared_model_arrays.npz",
                 "manifest.json", "contract.json", "walk003/native_original.npz", "walk003/original29.npz", "walk003/timeline.json")]
    relevant += [args.bundle / "meshes" / name for name in manifest["meshes"]]
    runtime = Path(mujoco.__file__).parent
    relevant += list(runtime.glob("libmujoco.so*"))
    provenance = dict(kind="independent_continuous_recorded_target_native323_replay", clip="walk003",
        mode="executed-target", plan=str(args.producer.resolve()), motion_override=override,
        explicit_local_motion_override=str(args.reference.resolve()), reference_sha256=sha(args.reference),
        original29_path=str((args.bundle / "walk003/original29.npz").resolve()),
        original29_sha256=override["original29_sha256"], mujoco=mujoco.__version__,
        mujoco_module=str(mujoco.__file__), executable=sys.executable, python=sys.version,
        platform=platform.platform(), thread_environment={key:os.environ.get(key) for key in
            ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        hashes={str(path.resolve()):sha(path) for path in relevant},
        compiled_model_fields_restored_and_verified=list(prepared),
        initialization="declared optimization reference frame10; exact equality checked against producer init; qacc_warmstart zero; mj_forward once",
        physical_state_rewrites_after_initialization=0, root_assistance_forces=0,
        target_source="producer actual target, held ten physics steps; no savedK or replanning",
        warning_evidence_scope="new independent replay only; original producer warning ledger missing",
        source_clock_hz=50, physics_hz=500, source_frames_dropped=0,
        conservative_effective_source_preview_seconds=.74,
        maximum_raw_pose_support_seconds=.76,
        timing_qualified=False, hardware_authorized=False)
    write(args.output / "provenance.json", provenance)
    data = mujoco.MjData(native)
    data.qpos[:], data.qvel[:] = initial_qpos, initial_qvel
    data.qacc_warmstart[:] = 0
    mujoco.mj_forward(native, data)
    fields = ("qpos","qvel","target","source_frame","joint_error","root_error","range_excess",
              "velocity_ratio","effort_ratio","physics_qpos","physics_qvel","physics_torque",
              "physics_actuator_torque","physics_substeps","physics_time","physics_warning_counts",
              "physics_warning_lastinfo")
    trace = {key:[] for key in fields}
    for key in ("qpos","physics_qpos"): trace[key].append(data.qpos.copy())
    for key in ("qvel","physics_qvel"): trace[key].append(data.qvel.copy())
    trace["physics_time"].append(float(data.time))
    trace["physics_warning_counts"].append(np.asarray(data.warning.number, dtype=np.int64).copy())
    trace["physics_warning_lastinfo"].append(np.asarray(data.warning.lastinfo, dtype=np.int64).copy())
    limits = native.jnt_range[1:]
    failure = None
    started = time.perf_counter()
    for control, target in enumerate(source["target"]):
        peak_range = peak_speed = peak_effort = 0.
        for sub in range(10):
            assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
            mujoco.mj_step(native, data)
            trace["physics_qpos"].append(data.qpos.copy())
            trace["physics_qvel"].append(data.qvel.copy())
            trace["physics_torque"].append(data.ctrl.copy())
            trace["physics_actuator_torque"].append(data.qfrc_actuator[6:].copy())
            trace["physics_time"].append(float(data.time))
            trace["physics_warning_counts"].append(np.asarray(data.warning.number, dtype=np.int64).copy())
            trace["physics_warning_lastinfo"].append(np.asarray(data.warning.lastinfo, dtype=np.int64).copy())
            peak_range = max(peak_range, float(np.maximum(limits[:,0]-data.qpos[7:], data.qpos[7:]-limits[:,1]).max()))
            peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[6:])/speed)))
            peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:])/effort)))
            tilt = float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1)))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2]<.25 or tilt>1.2:
                failure = dict(kind="fall_or_nonfinite", control=control, substep=sub+1, height=float(data.qpos[2]), tilt=tilt)
            elif peak_range>.01 or peak_speed>1 or peak_effort>1+1e-6:
                failure = dict(kind="native_limit_abort", control=control, substep=sub+1,
                               range_excess=peak_range, velocity_ratio=peak_speed, effort_ratio=peak_effort)
            elif np.any(data.warning.number):
                failure = dict(kind="engine_warning", control=control, substep=sub+1,
                               engine_warning_counts=np.asarray(data.warning.number,dtype=int).tolist())
            if failure: break
        frame = control+11
        # Match producer's metric-only kinematics call; never forwards/reinitializes plant.
        mujoco.mj_kinematics(native,data)
        values = dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),target=target.copy(),source_frame=frame,
            joint_error=data.qpos[7:]-motion["joint_pos"][frame],
            root_error=float(np.linalg.norm(data.qpos[:3]-motion["body_pos_w"][frame,0])),
            range_excess=peak_range,velocity_ratio=peak_speed,effort_ratio=peak_effort,physics_substeps=sub+1)
        for key,value in values.items():trace[key].append(value)
        if (control+1)%250==0 or failure:
            print(json.dumps(dict(controls=control+1,physics_steps=len(trace["physics_torque"]),failure=failure)),flush=True)
        if failure:break
    elapsed = time.perf_counter()-started
    arrays = {key:np.asarray(value) for key,value in trace.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
    n, steps = len(arrays["target"]),len(arrays["physics_torque"])
    differences = {}
    for name in ("qpos","qvel","physics_qpos","physics_qvel","physics_torque","target"):
        a,b = arrays[name],source[name][:len(arrays[name])]
        delta = np.abs(a-b)
        bad = np.flatnonzero(np.any(a!=b,axis=1))
        differences[name] = dict(max_abs=float(delta.max()), bit_exact=bool(np.array_equal(a,b)),
            first_non_bit_exact_row=None if not len(bad) else int(bad[0]))
    physical_q = arrays["physics_qpos"][:,7:]
    excess = np.maximum(0,np.maximum(limits[:,0]-physical_q,physical_q-limits[:,1]))
    tilt = np.arccos(np.clip(1-2*np.sum(arrays["physics_qpos"][:,4:6]**2,axis=1),-1,1))
    warnings = arrays["physics_warning_counts"]
    source_stop = min(n,phase["control_stop"])
    source_count = max(0,source_stop-phase["control_start"])
    source_full = n>=phase["control_stop"] and np.all(arrays["physics_substeps"][:phase["control_stop"]]==10)
    lifecycle_full = n==count and np.all(arrays["physics_substeps"]==10)
    report = dict(kind=provenance["kind"], clip="walk003", mode="executed-target", mujoco=mujoco.__version__,
        motion_override=override, explicit_local_motion_override=str(args.reference.resolve()),
        reference_sha256=sha(args.reference), original29_sha256=override["original29_sha256"],
        completed_controls=n, complete_controls=int(np.sum(arrays["physics_substeps"]==10)),
        requested_controls=count, physics_steps=steps, simulated_seconds=float(data.time),
        expected_simulated_seconds=steps*.002,
        actual_clock_max_abs_error_seconds=float(np.max(np.abs(arrays["physics_time"]-np.arange(steps+1)*.002))),
        source_attempted_controls=source_count, source_completed_controls=int(np.sum(arrays["physics_substeps"][phase["control_start"]:source_stop]==10)),
        source_requested_controls=phase["requested_controls"], full_source_completed=bool(source_full),
        full_lifecycle_completed=bool(lifecycle_full), failure=failure,
        engine_warning_counts=np.max(warnings,axis=0).astype(int).tolist(),
        engine_warning_sampling_seconds=.002, engine_warning_samples=len(warnings),
        warning_evidence_scope=provenance["warning_evidence_scope"],
        range_excess_max=float(excess.max()), raw_joint_limit_physics_samples=int(np.sum(excess.max(axis=1)>0)),
        velocity_ratio_max=float(np.max(np.abs(arrays["physics_qvel"][:,6:])/speed)),
        effort_ratio_max=float(np.max(np.abs(arrays["physics_actuator_torque"])/effort)),
        commanded_effort_ratio_max=float(np.max(np.abs(arrays["physics_torque"])/effort)),
        minimum_root_height_m=float(arrays["physics_qpos"][:,2].min()), maximum_root_tilt_rad=float(tilt.max()),
        producer_comparison=differences,
        all_recorded_physics_bit_exact=bool(lifecycle_full and all(v["bit_exact"] for v in differences.values())),
        final_root_error_m=float(arrays["root_error"][-1]), final_joint_speed_max_radps=float(np.abs(arrays["qvel"][-1,6:]).max()),
        physical_state_rewrites_after_initialization=0, root_assistance_forces=0,
        elapsed_wall_seconds=elapsed, no_replanning=True, received_only_stream_qualified=False,
        full_body_tracking_qualified=False, timing_qualified=False, hardware_authorized=False,
        conservative_effective_source_preview_seconds=.74, maximum_raw_pose_support_seconds=.76,
        original_planning_ms_p50_p95_max=producer_report["planning_ms_p50_p95_max"],
        trace_sha256=sha(args.output / "trace.npz"), provenance_sha256=sha(args.output / "provenance.json"))
    assert all(sha(Path(path))==digest for path,digest in provenance["hashes"].items()), "Input changed during replay"
    write(args.output / "report.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("bundle","reference","producer","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
