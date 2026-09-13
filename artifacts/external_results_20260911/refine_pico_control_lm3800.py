"""One K0 solve with declared control-space LM and factored value updates."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np

BASE = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
RUN = BASE / 'pico_full_hard_restoration_v1'
OUT = BASE / 'pico_final3800_control_lm_k0_v1'
OUT.mkdir(exist_ok=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value

request = json.loads((RUN / 'request.json').read_text())
repo = Path.cwd()
frozen = OUT / 'frozen_repo'
source_hashes = {}
for package in ('gear_sonic', 'gear_sonic/utils'):
    (frozen / package).mkdir(parents=True, exist_ok=True)
    source = repo / package / '__init__.py'
    (frozen / package / '__init__.py').write_bytes(source.read_bytes() if source.exists() else b'')
for name in ('g1_true23_mjbatch_ilqr_core', 'g1_true23_mjbatch_mpc', 'g1_true23_mjbatch_model',
             'g1_true23_mjbatch_restoration', 'g1_true23_feasibility_referee',
             'g1_true23_relative_foot_cost'):
    source = repo / 'gear_sonic/utils' / (name + '.py')
    digest = sha(source)
    old_key = 'gear_sonic/utils/' + name + '.py'
    if old_key in request['input_hashes']:
        assert digest == request['input_hashes'][old_key], old_key
    source_hashes[old_key] = digest
    shutil.copy2(source, frozen / old_key)
fixture = RUN / 'restoration_03800_guided.npz'
shutil.copy2(fixture, OUT / 'initial_guided_fixture.npz')
shutil.copy2(RUN / 'request.json', OUT / 'original_request.json')
shutil.copy2(Path(__file__), OUT / 'source.py')
(OUT / 'frozen_receipt.json').write_text(json.dumps(dict(
    source_hashes=source_hashes, fixture_sha256=sha(fixture), request_sha256=sha(RUN / 'request.json'),
    experiment_source_sha256=sha(__file__), frozen_before_import_and_solve=True), indent=2))
sys.path.insert(0, str(frozen))

from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, load_motion_override
from gear_sonic.utils.g1_true23_mjbatch_restoration import Native23RestorationTracker

bundle = Path('artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
native, contract, original, timeline, manifest = load_native_bundle(bundle, 'pico')
motion, _ = load_motion_override(Path(request['motion_override']['path']), bundle, 'pico', native,
                                 contract, original, timeline, manifest)
with np.load(fixture, allow_pickle=False) as a:
    original_anchor = a['original_warm_targets'].copy()
    seed = a['targets'].copy()
    integration, spec = a['initial_integration'].copy(), int(a['integration_state_spec'])
data = mujoco.MjData(native)
mujoco.mj_setState(native, data, integration, spec)
mujoco.mj_forward(native, data)
mujoco.mj_setState(native, data, integration, spec)
initial = np.r_[data.qpos, data.qvel]
servo = position_servo_copy(native, contract['kp'], contract['kd'], contract['native_effort'])
restorer = Native23RestorationTracker(servo, contract, motion, original_anchor, threads=8,
                                      zero_rollout_feedback=True)
restorer.window(3810)
certifier = Native23Tracker(servo, contract, motion, horizon=30, threads=8, hard_feasibility=True,
                            all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400)
certifier.window(3810)
calls, searches, sequences, directions = [], [], [], []
backward, rollout = core.backward, restorer.rollout
current_merit = [None]
matrix_diagnostics = []

def factored_backward(A, B, lx, lxx, lu, luu, lo, hi, mu):
    T, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1], lxx[-1]
    k, K = np.zeros((T + 1, nu)), np.empty((T, nu, nx))
    for t in reversed(range(T)):
        qu = lu[t] + B[t].T @ vx
        raw_quu = luu[t] + B[t].T @ vxx @ B[t]
        symmetric_quu = .5 * (raw_quu + raw_quu.T)
        quu_reg = symmetric_quu + mu * np.eye(nu)
        qux_reg = B[t].T @ vxx @ A[t]
        # Declared control-space LM only. Preserve original strict finite/SPD
        # check, box-QP, mu schedule/cap, and actual unregularized stage cost.
        finite = bool(np.isfinite(quu_reg).all())
        entry = dict(backward_call=len(calls), knot=t, mu=float(mu), finite=finite)
        eigen_min = None
        if finite:
            eigen_min = float(np.linalg.eigvalsh(quu_reg).min())
            entry.update(raw_lower_triangle_eigen_min=float(np.linalg.eigvalsh(raw_quu).min()),
                         symmetric_eigen_min=float(np.linalg.eigvalsh(symmetric_quu).min()),
                         regularized_eigen_min=eigen_min,
                         regularized_eigen_max=float(np.linalg.eigvalsh(quu_reg).max()),
                         raw_to_symmetric_max=float(np.abs(raw_quu-symmetric_quu).max()))
        matrix_diagnostics.append(entry)
        if not finite or eigen_min <= 0:
            np.savez_compressed(OUT / ('failed_matrix_%02d_%02d.npz' % (len(calls), t)),
                                raw_quu=raw_quu, symmetric_quu=symmetric_quu, quu_reg=quu_reg,
                                qux_reg=qux_reg, V=vxx, B=B[t], A=A[t], R=luu[t], mu=mu)
            return None
        k[t], free = core.boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])
        K[t] = 0
        K[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        M = A[t] + B[t] @ K[t]
        vx = lx[t] + K[t].T @ (lu[t] + luu[t] @ k[t]) + M.T @ (vx + vxx @ B[t] @ k[t])
        vxx = lxx[t] + K[t].T @ luu[t] @ K[t] + M.T @ vxx @ M
        vxx = .5 * (vxx + vxx.T)
    return k[:-1], K

def observed_backward(*values):
    result = factored_backward(*values)
    entry = dict(iteration=len(calls), mu=float(values[-1]), returned_sweep=result is not None,
                 all_derivatives_finite=all(np.isfinite(a).all() for a in values[:-1]))
    if result is not None:
        directions.append((result[0].copy(), result[1].copy()))
        entry.update(max_abs_k=float(np.abs(result[0]).max()), max_abs_K=float(np.abs(result[1]).max()))
    calls.append(entry)
    return result

def observed_rollout(*values, **kwargs):
    states, targets, costs = rollout(*values, **kwargs)
    guided = len(values) >= 3 or kwargs.get('gains') is not None
    entry = dict(kind='alpha_search' if guided else 'initial', costs=costs.tolist())
    if guided:
        best = int(np.argmin(costs))
        accepted = bool(costs[best] < current_merit[0])
        entry.update(iteration=len(calls)-1, incumbent_merit_before=current_merit[0],
                     best_lane=best, alpha=float(core.ALPHAS[best]), accepted=accepted)
        if accepted:
            drop = current_merit[0] - float(costs[best])
            current_merit[0] = float(costs[best])
            entry.update(drop=drop, tolerance_stop=bool(drop < core.TOL * current_merit[0]))
        entry['incumbent_merit_after'] = current_merit[0]
    else:
        current_merit[0] = float(costs[0])
    searches.append(entry)
    sequences.append(targets.copy())
    return states, targets, costs

restorer.rollout = observed_rollout
seed_states, _, initial_costs = restorer.rollout(initial, seed)
started = time.perf_counter()
try:
    core.backward = observed_backward
    states, targets, gains, cost = core.ilqr(restorer, initial, seed.copy(), iters=10,
        initial_rollout=(seed_states[:, 0], initial_costs[0]))
finally:
    core.backward = backward
solve_ms = (time.perf_counter() - started) * 1000
np.testing.assert_array_equal(restorer.original_targets, original_anchor)

def certificate(candidate, retain=False):
    checked_states, _, costs = certifier.rollout(initial, candidate)
    witness = certifier.last_rollout_feasibility['first_violation'][0]
    result = dict(hard_nominal_pass=bool(np.isfinite(costs[0])), nominal_first_failure=witness,
                  tracking_cost=float(costs[0]))
    trace = None
    if retain or np.isfinite(costs[0]):
        oracle, trace = inspect_native_segment(native, data, candidate, contract,
                                               stop_on_failure=not retain, retain_trace=retain)
        result['full_native_oracle'] = oracle
        result['dual_certificate_pass'] = result['hard_nominal_pass'] and oracle['feasible']
    else:
        result['dual_certificate_pass'] = False
    return result, trace, checked_states[:, 0]

final_cert, final_trace, checked_states = certificate(targets, retain=True)
np.savez_compressed(OUT / 'ordinary_final.npz', **final_trace, targets=targets,
                    hard_nominal_states=checked_states, merit_states=states, final_K=gains,
                    initial_integration=integration, integration_state_spec=spec,
                    optimization_initial_targets=seed, original_regularization_anchor=original_anchor)
report = dict(kind='one_private_guided_to_K0_control_LM_solve', actual_control=3800, actual_controls_executed=0,
              horizon=30, maximum_iterations=10, batch_threads=8, fd_epsilon=1e-6,
              alphas=core.ALPHAS.tolist(), merit_contract=restorer.merit_contract(),
              initial_merit=float(initial_costs[0]), final_merit=float(cost), solve_ms=solve_ms,
              ordinary_final_certificate=final_cert, ordinary_final_sha256=sha(OUT / 'ordinary_final.npz'),
              optimization_initial_targets='frozen guided final proposal',
              regularization_anchor='original shifted warm targets, unchanged',
              backward_override='factored closed-loop vx/vxx plus declared control-space LM',
              control_LM='Qraw=luu+B.T@V@B; Qsym=.5*(Qraw+Qraw.T); Qreg=Qsym+mu*I; Qux=B.T@V@A unregularized',
              same_Qreg_for_SPD_boxQP_and_feedback_solve=True,
              control_LM_schedule=dict(initial=1., reject_factor=10., accept_factor=.1, floor=1e-6, cap=1e6),
              eigenvalue_clipping=False, extra_diagonal_or_jitter_beyond_declared_LM=False,
              original_mu_schedule_and_cap=True, main_controller_changed=False,
              admission='ordinary final proposal only; generated candidates are diagnostic',
              backward_calls=calls, backward_matrix_diagnostics=matrix_diagnostics,
              searches=searches, source_hashes=source_hashes,
              original_actual_state_and_history_fixture_sha256=sha(fixture))
(OUT / 'report.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False))
np.savez_compressed(OUT / 'all_generated.npz', targets_by_search=np.asarray(sequences),
                    backward_feedforward=np.asarray([x[0] for x in directions]),
                    backward_feedback=np.asarray([x[1] for x in directions]), final_targets=targets,
                    initial_integration=integration, integration_state_spec=spec,
                    optimization_initial_targets=seed, original_regularization_anchor=original_anchor)
print(json.dumps(clean(dict(initial=initial_costs[0], final=cost, solve_ms=solve_ms,
                           ordinary_final_certificate=final_cert))), flush=True)
certificates = []
for call, batch in enumerate(sequences):
    for lane in range(9 if call else 1):
        result, _, _ = certificate(batch[:, lane])
        result.update(search=call, lane=lane, merit=searches[call]['costs'][lane])
        certificates.append(result)
state_after = np.empty_like(integration)
mujoco.mj_getState(native, data, state_after, spec)
np.testing.assert_array_equal(integration, state_after)
assert sha(fixture) == report['original_actual_state_and_history_fixture_sha256']
report.update(all_generated_certificates=certificates,
              feasible_generated_candidates=sum(x['dual_certificate_pass'] for x in certificates),
              original_full_actual_state_unchanged=True, original_history_fixture_unchanged=True)
(OUT / 'report.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False))
print(json.dumps(dict(all_generated=len(certificates), feasible_generated=report['feasible_generated_candidates'],
                      ordinary_final_pass=final_cert['dual_certificate_pass'], actual_state_unchanged=True)), flush=True)
