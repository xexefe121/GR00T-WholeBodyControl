"""Private replay of saved final proposals; no optimization or actual controls."""
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import finite_json
from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_motion_override, load_native_bundle, sha256

BASE = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
RUN = BASE / 'pico_full_hard_restoration_v1'
OUT = BASE / 'pico_final3800_hold_tail_v1'
OUT.mkdir(exist_ok=False)
request = json.loads((RUN / 'request.json').read_text())
for name, digest in request['input_hashes'].items():
    if name.endswith('.py'):
        assert sha256(Path(name)) == digest, name
bundle = Path('artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
native, contract, original, timeline, manifest = load_native_bundle(bundle, 'pico')
motion, _ = load_motion_override(Path(request['motion_override']['path']), bundle, 'pico', native,
                                 contract, original, timeline, manifest)
guided_path = RUN / 'restoration_03800_guided.npz'
zero_path = RUN / 'restoration_03800_zero_feedback.npz'
with np.load(guided_path, allow_pickle=False) as a:
    integration, spec = a['initial_integration'].copy(), int(a['integration_state_spec'])
    cases = {'shifted_warm': a['original_warm_targets'].copy(), 'guided': a['targets'].copy()}
with np.load(zero_path, allow_pickle=False) as a:
    np.testing.assert_array_equal(integration, a['initial_integration'])
    np.testing.assert_array_equal(cases['shifted_warm'], a['original_warm_targets'])
    cases['zero_feedback'] = a['targets'].copy()
w = cases['shifted_warm']
cases = {'hold_tail': np.concatenate((w[:25], np.repeat(w[24:25], 5, axis=0)))}
data = mujoco.MjData(native)
mujoco.mj_setState(native, data, integration, spec)
mujoco.mj_forward(native, data)
mujoco.mj_setState(native, data, integration, spec)
servo = position_servo_copy(native, contract['kp'], contract['kd'], contract['native_effort'])
tracker = Native23Tracker(servo, contract, motion, horizon=30, threads=2, hard_feasibility=True,
                          all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400)
tracker.window(3810)
results = []
for name, targets in cases.items():
    states, _, costs = tracker.rollout(np.r_[data.qpos, data.qvel], targets)
    witness = tracker.last_rollout_feasibility['first_violation'][0]
    if witness is not None:
        q = states[-1, 0, 7:30]
        excess = np.maximum(np.maximum(tracker.lo - q, q - tracker.hi), 0)
        j = int(np.argmax(excess))
        witness = dict(witness, worst_joint_index=j, worst_joint=contract['joint_names'][j],
                       q=float(q[j]), lower=float(tracker.lo[j]), upper=float(tracker.hi[j]),
                       direction='below_lower' if q[j] < tracker.lo[j] else 'above_upper')
    oracle, trace = inspect_native_segment(native, data, targets, contract,
                                           stop_on_failure=False, retain_trace=True)
    q = trace['physics_qpos'][:, 7:]
    excess = np.maximum(np.maximum(tracker.lo - q, q - tracker.hi), 0)
    violation_joints = np.flatnonzero(excess.max(axis=0) > 1e-6)
    details = [dict(index=int(j), name=contract['joint_names'][j],
                    maximum_excess_rad=float(excess[:, j].max()), final_excess_rad=float(excess[-1, j]))
               for j in violation_joints]
    if oracle['first_failure'] is not None:
        oracle['first_failure']['worst_joint_name'] = contract['joint_names'][oracle['first_failure']['worst_joint_index']]
    final = dict(root_height=float(trace['physics_qpos'][-1, 2]),
                 root_vertical_speed=float(trace['physics_qvel'][-1, 2]),
                 root_tilt_rad=float(np.arccos(np.clip(
                     1 - 2 * np.sum(trace['physics_qpos'][-1, 4:6] ** 2), -1, 1))))
    path = OUT / (name + '.npz')
    np.savez_compressed(path, **trace, targets=targets, initial_integration=integration,
                        integration_state_spec=spec, hard_nominal_states=states[:, 0])
    result = dict(case=name, hard_nominal_pass=bool(np.isfinite(costs[0])), hard_nominal_first_failure=witness, full_native_oracle=oracle,
                  violated_joints=details, final=final, trace_sha256=sha256(path))
    results.append(result)
    print(json.dumps(finite_json(result)), flush=True)
report = finite_json(dict(kind='saved_proposal_private_replay_only', actual_control=3800,
                          actual_controls_executed=0, solver_calls=0, results=results,
                          actual_root_position=data.qpos[:3].tolist(), actual_velocity=data.qvel[:6].tolist(),
                          input_hashes={str(p): sha256(p) for p in
                                        (RUN / 'request.json', guided_path, zero_path, Path(__file__))}))
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
(OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
