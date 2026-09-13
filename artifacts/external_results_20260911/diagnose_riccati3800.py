"""Saved initial/final derivative and Riccati diagnostics; no optimizer or execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np

BASE = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
INPUT = BASE / 'pico_final3800_guided_k0_refinement_v1'
OUT = BASE / 'pico_final3800_riccati_diagnostic_v1'
OUT.mkdir(exist_ok=False)
shutil.copytree(INPUT / 'frozen_repo', OUT / 'frozen_repo')
shutil.copy2(__file__, OUT / 'source.py')
sys.path.insert(0, str(OUT / 'frozen_repo'))

from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override, load_native_bundle
from gear_sonic.utils.g1_true23_mjbatch_restoration import Native23RestorationTracker

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(matrix):
    eig = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    return dict(minimum_eigenvalue=float(eig[0]), maximum_eigenvalue=float(eig[-1]),
                max_abs=float(np.max(np.abs(matrix))), symmetry_error=float(np.max(np.abs(matrix-matrix.T))))

def recursion(d, mu, form, label):
    A, B, lx, lxx, lu, luu, lo, hi = d
    T, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1].copy(), lxx[-1].copy()
    ks = np.zeros((T+1, nu)); gains = np.zeros((T, nu, nx))
    entries, updates = [], []
    for t in reversed(range(T)):
        qx, qu = lx[t] + A[t].T @ vx, lu[t] + B[t].T @ vx
        qxx = lxx[t] + A[t].T @ vxx @ A[t]
        quu, qux = luu[t] + B[t].T @ vxx @ B[t], B[t].T @ vxx @ A[t]
        reg = vxx + mu * np.eye(nx)
        quu_reg, qux_reg = luu[t] + B[t].T @ reg @ B[t], B[t].T @ reg @ A[t]
        entry = dict(knot=t, next_value=stats(vxx), regularized_control=stats(quu_reg),
                     next_gradient_max=float(np.abs(vx).max()))
        entries.append(entry)
        if not (np.isfinite(quu_reg).all() and np.linalg.eigvalsh(quu_reg).min() > 0):
            np.savez_compressed(OUT / (label + '_first_failure.npz'), knot=t, mu=mu,
                A=A[t], B=B[t], lx=lx[t], lxx=lxx[t], lu=lu[t], luu=luu[t],
                next_value=vxx, next_gradient=vx, quu=quu, regularized_quu=quu_reg,
                regularized_qux=qux_reg, ks=ks, gains=gains)
            return dict(form=form, mu=mu, completed=False, first_failure_knot=t, entries=entries, updates=updates)
        ks[t], free = core.boxqp(quu_reg, qu, lo[t], hi[t], ks[t+1])
        K = gains[t]
        K[free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        k = ks[t]
        expanded_x = qx + K.T @ quu @ k + K.T @ qu + qux.T @ k
        expanded_xx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
        M = A[t] + B[t] @ K
        grouped_x = lx[t] + K.T @ (lu[t] + luu[t] @ k) + M.T @ (vx + vxx @ B[t] @ k)
        grouped_xx = lxx[t] + K.T @ luu[t] @ K + M.T @ vxx @ M
        # Recompute mathematically identical grouped formula at extended precision
        # from the SAME incoming binary64 matrices/gains, not a different policy.
        L = np.longdouble
        Al, Bl, Kl, Vl = A[t].astype(L), B[t].astype(L), K.astype(L), vxx.astype(L)
        Ml = Al + Bl @ Kl
        reference = lxx[t].astype(L) + Kl.T @ luu[t].astype(L) @ Kl + Ml.T @ Vl @ Ml
        scale = max(float(np.max(np.abs(reference))), 1.0)
        update = dict(knot=t, max_abs_K=float(np.abs(K).max()), max_abs_k=float(np.abs(k).max()),
            expanded=stats(expanded_xx), grouped=stats(grouped_xx),
            extended_precision_grouped=stats(np.asarray(reference, float)),
            expanded_max_abs_error=float(np.max(np.abs(expanded_xx.astype(L)-reference))),
            grouped_max_abs_error=float(np.max(np.abs(grouped_xx.astype(L)-reference))),
            expanded_relative_error=float(np.max(np.abs(expanded_xx.astype(L)-reference))) / scale,
            grouped_relative_error=float(np.max(np.abs(grouped_xx.astype(L)-reference))) / scale,
            gradient_formula_max_difference=float(np.max(np.abs(expanded_x-grouped_x))))
        updates.append(update)
        # Save every numerical update witness for independent arithmetic replay.
        np.savez_compressed(OUT / (label + '_knot%02d.npz' % t),
            A=A[t], B=B[t], lx=lx[t], lxx=lxx[t], lu=lu[t], luu=luu[t],
            next_value=vxx, next_gradient=vx, k=k, K=K, expanded_x=expanded_x,
            grouped_x=grouped_x, expanded_xx=expanded_xx, grouped_xx=grouped_xx,
            extended_grouped_xx=np.asarray(reference, float))
        vx, vxx = (expanded_x, expanded_xx) if form == 'expanded' else (grouped_x, grouped_xx)
        vxx = .5 * (vxx + vxx.T)
    np.savez_compressed(OUT / (label + '_completed.npz'), ks=ks[:-1], gains=gains)
    return dict(form=form, mu=mu, completed=True, entries=entries, updates=updates)

request = json.loads((INPUT / 'original_request.json').read_text())
bundle = Path('artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
native, contract, original, timeline, manifest = load_native_bundle(bundle, 'pico')
motion, _ = load_motion_override(Path(request['motion_override']['path']), bundle, 'pico', native,
                                 contract, original, timeline, manifest)
with np.load(INPUT / 'ordinary_final.npz', allow_pickle=False) as a:
    initial_targets = a['optimization_initial_targets'].copy()
    final_targets = a['targets'].copy()
    anchor = a['original_regularization_anchor'].copy()
    final_states = a['merit_states'].copy()
    integration, spec = a['initial_integration'].copy(), int(a['integration_state_spec'])
data = mujoco.MjData(native)
mujoco.mj_setState(native, data, integration, spec)
mujoco.mj_forward(native, data)
mujoco.mj_setState(native, data, integration, spec)
initial = np.r_[data.qpos, data.qvel]
servo = position_servo_copy(native, contract['kp'], contract['kd'], contract['native_effort'])
planner = Native23RestorationTracker(servo, contract, motion, anchor, threads=8, zero_rollout_feedback=True)
planner.window(3810)
initial_states, _, _ = planner.rollout(initial, initial_targets)
report = dict(kind='derivative_and_backward_arithmetic_only', solver_calls=0, actual_controls_executed=0,
              initial_state_fixture_sha256=sha(INPUT/'initial_guided_fixture.npz'),
              saved_final_proposal_sha256=sha(INPUT/'ordinary_final.npz'), cases=[])
for name, states, targets, mus in (
        ('guided_initial', initial_states[:, 0], initial_targets, (1., 1e6)),
        ('refined_final', final_states, final_targets, (1e5, 1e6))):
    started = time.perf_counter()
    d = (*planner.linearize(states, targets), *planner.expand(states, targets),
         planner.lo-targets, planner.hi-targets)
    path = OUT / (name + '_derivatives.npz')
    np.savez_compressed(path, **dict(zip(('A','B','lx','lxx','lu','luu','lo','hi'), d)),
                        states=states, targets=targets, original_anchor=anchor)
    case = dict(name=name, derivative_seconds=time.perf_counter()-started,
                derivatives_sha256=sha(path), all_finite=all(np.isfinite(x).all() for x in d),
                stage_lxx_min_eigenvalue=float(np.linalg.eigvalsh(d[3]).min()),
                stage_luu_min_eigenvalue=float(np.linalg.eigvalsh(d[5]).min()), recursions=[])
    for mu in mus:
        for form in ('expanded','grouped'):
            result = recursion(d, mu, form, name + '_' + form + '_mu%.0e' % mu)
            case['recursions'].append(result)
            print(json.dumps(dict(case=name, form=form, mu=mu, completed=result['completed'],
                                  first_failure_knot=result.get('first_failure_knot'))), flush=True)
    report['cases'].append(case)
after = np.empty_like(integration)
mujoco.mj_getState(native, data, after, spec)
np.testing.assert_array_equal(after, integration)
report['original_full_actual_state_unchanged'] = True
report['source_hashes'] = {str(p.relative_to(OUT)):sha(p) for p in (OUT/'frozen_repo').rglob('*.py')}
report['source_hashes']['source.py'] = sha(OUT/'source.py')
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
