"""Four original-lifecycle initial-state seed witnesses; no optimizer or live IO."""
import copy
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).parent
SNAPSHOT = BASE.parent / 'recovery_probe_v1/source_snapshot'
sys.path.insert(0, str(SNAPSHOT))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override, motion_states, Native23Tracker, sha256
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK = Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
ONNX = ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
DEPS = Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
PRODUCERS = {
    'pico': 'pico_v4_native323_freshseed_allmargin_5iter_full_v1',
    'walk002': 'walk002_v4_native323_full_v1',
    'walk003': 'walk003_v4_native323_allmargin_full_v1',
    'walk008': 'walk008_v4_native323_allmargin_relativefoot_full_v1',
}
H = 30


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    started = time.perf_counter()
    assert mujoco.__version__ == '3.2.3' and np.__version__ == '1.26.4'
    results = []
    for clip, folder in PRODUCERS.items():
        dest = BASE / clip
        dest.mkdir(exist_ok=False)
        native, c, original, timeline, manifest = load_native_bundle(BUNDLE, clip)
        reference = TASK / 'mjbatch_intent_floor_inputs_v1' / clip / 'reference.npz'
        motion, override = load_motion_override(reference, BUNDLE, clip, native, c, original, timeline, manifest)
        request_path = TASK / 'mjbatch_full_v1' / folder / 'request.json'
        request = json.loads(request_path.read_text())
        recorded_path = ROOT / request['recorded_target_seed']['path'] / 'trace.npz'
        assert sha256(recorded_path) == request['recorded_target_seed']['trace_sha256']
        with np.load(recorded_path, allow_pickle=False) as a:
            recorded = a['target'][:H].copy()
        kp, kd, effort, speed = [np.asarray(c[k]) for k in ('kp', 'kd', 'native_effort', 'native_velocity')]
        lo, hi = np.asarray(c['joint_limits']).T
        initial = motion_states(motion)[10]
        np.testing.assert_array_equal(initial[:30], request['initial_qpos'])
        np.testing.assert_array_equal(initial[30:], request['initial_qvel'])
        seed = Native23BFMRolloutSeed(native, c, original, ONNX, dependency_directory=DEPS, threads=1)
        before_history = {k: v.copy() for k, v in seed.history.data.items()}
        fresh, fresh_diagnostics = seed.propose(0, initial[:30], initial[30:], horizon=H)
        for k, v in before_history.items():
            np.testing.assert_array_equal(seed.history.data[k], v)
        assert seed.recorded_controls == 0 and not np.any(seed.previous_action)
        servo = position_servo_copy(native, kp, kd, effort)
        planner = Native23Tracker(servo, c, motion, horizon=H, threads=1,
            all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400)
        planner.window(10)
        candidates = dict(reference_initial=planner.target_reference(np.arange(H)), recorded_bfm=recorded, fresh_bfm=fresh)
        cases = []
        for name, targets in candidates.items():
            d = mujoco.MjData(native)
            d.qpos[:] = initial[:30]
            d.qvel[:] = initial[30:]
            mujoco.mj_forward(native, d)
            rows = {k: [] for k in ('state', 'target', 'physics_qpos', 'physics_qvel', 'physics_torque', 'physics_actuator_torque', 'physics_time', 'warning_counts', 'warning_lastinfo', 'range_excess', 'speed_ratio', 'effort_ratio')}
            rows['state'].append(initial.copy())
            rows['physics_qpos'].append(d.qpos.copy())
            rows['physics_qvel'].append(d.qvel.copy())
            rows['physics_time'].append(float(d.time))
            failure = None
            for t, target in enumerate(targets):
                np.testing.assert_array_equal(target, np.clip(target, lo, hi))
                rows['target'].append(target.copy())
                for sub in range(10):
                    d.ctrl[:] = np.clip(kp * (target-d.qpos[7:])-kd*d.qvel[6:], -effort, effort)
                    mujoco.mj_step(native, d)
                    excess = float(np.maximum(0, np.maximum(lo-d.qpos[7:], d.qpos[7:]-hi)).max())
                    velocity = float(np.max(np.abs(d.qvel[6:])/speed))
                    force = float(np.max(np.abs(d.qfrc_actuator[6:])/effort))
                    values = dict(physics_qpos=d.qpos.copy(), physics_qvel=d.qvel.copy(), physics_torque=d.ctrl.copy(),
                        physics_actuator_torque=d.qfrc_actuator[6:].copy(), physics_time=float(d.time),
                        warning_counts=d.warning.number.copy(), warning_lastinfo=d.warning.lastinfo.copy(),
                        range_excess=excess, speed_ratio=velocity, effort_ratio=force)
                    for key, value in values.items():
                        rows[key].append(value)
                    expected = (t*10+sub+1)*.002
                    tilt = float(np.arccos(np.clip(1-2*np.sum(d.qpos[4:6]**2), -1, 1)))
                    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number) or abs(d.time-expected)>1e-10:
                        failure = dict(kind='engine_or_clock', control=t, substep=sub+1)
                    elif excess>1e-6 or velocity>1 or force>1+1e-9 or d.qpos[2]<.25 or tilt>1.2:
                        failure = dict(kind='physical_feasibility', control=t, substep=sub+1, range_excess=excess,
                            speed_ratio=velocity, effort_ratio=force, height=float(d.qpos[2]), tilt=tilt)
                    if failure:
                        break
                mujoco.mj_kinematics(native, d)
                rows['state'].append(np.r_[d.qpos, d.qvel])
                if failure:
                    break
            arrays = {k: np.asarray(v) for k, v in rows.items()}
            steps = len(arrays['physics_torque'])
            full = steps == 300 and failure is None
            cost = None
            if full:
                feat = planner.features(arrays['state'])
                residual = planner.residual(np.arange(H+1), feat)
                cost = float(np.sum(residual**2) + planner.control_weight*np.sum((targets-planner.target_reference(np.arange(H)))**2))
            path = dest / (name+'.npz')
            np.savez_compressed(path, **arrays, complete_proposed_targets=targets)
            result = dict(candidate=name, full_horizon_physical_pass=full, physics_steps=steps, failure=failure,
                same_objective_cost=cost, range_excess_max=float(arrays['range_excess'].max()),
                speed_ratio_max=float(arrays['speed_ratio'].max()), effort_ratio_max=float(arrays['effort_ratio'].max()),
                warning_counts=arrays['warning_counts'].max(axis=0).tolist(), warning_lastinfo=arrays['warning_lastinfo'].max(axis=0).tolist(),
                clock_max_error=float(np.max(np.abs(arrays['physics_time']-np.arange(steps+1)*.002))),
                trace_sha256=sha256(path))
            cases.append(result)
        feasible = [r for r in cases if r['full_horizon_physical_pass']]
        selected = min(feasible, key=lambda r:r['same_objective_cost'])['candidate'] if feasible else None
        result = dict(clip=clip, initial_frame=10, control=0, actor_goal_first_frame=11, planner_window_frame=10,
            initial_state_source='v4 optimization reference frame10, bitexact historical producer initialization',
            original_goal_source='unchanged original native reference for fresh BFM, horizon8, position1/yaw2',
            fresh_history='all zero initial history and previous action; proposal leaves actual history unchanged',
            objective='v4 plus alljoint.05rad/2000 and relativefoot400, identical for all clips and candidates',
            feasible_starting_incumbent_exists=bool(feasible), feasible_minimum_cost_candidate=selected,
            cases=cases, fresh_diagnostics=fresh_diagnostics, fresh_identity=seed.identity(),
            hashes={str(p):sha256(p) for p in (reference, request_path, recorded_path, BUNDLE/clip/'native_original.npz', BUNDLE/'contract.json')})
        write(dest/'report.json', result)
        results.append(result)
        print(json.dumps(dict(clip=clip, selected=selected, candidates=[{k:r[k] for k in ('candidate','full_horizon_physical_pass','same_objective_cost','range_excess_max','speed_ratio_max')} for r in cases])), flush=True)
    report = dict(kind='initial_four_clip_H30_seed_preflight', cases=results, horizon=H, physics_hz=500, control_hz=50,
        strict_range_tolerance_rad=1e-6, optimizer_used=False, original_source_or_lifecycle_qualification=False,
        scope='initial600ms only; source begins after350controls, so no dynamic-source feasibility claim',
        source_packet_preview_seconds=.74, conservative_raw_pose_support_seconds=.76,
        elapsed_seconds=time.perf_counter()-started,
        hashes={str(p):sha256(p) for p in [Path(__file__), *sorted((SNAPSHOT/'gear_sonic/utils').glob('*.py'))]})
    write(BASE/'report.json', report)
    print(json.dumps(dict(done=True, seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
