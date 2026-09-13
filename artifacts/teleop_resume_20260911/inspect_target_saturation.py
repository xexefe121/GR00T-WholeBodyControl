"""Saved-state diagnostic of ten-substep PD target identifiability, no dynamics."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract_path = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'
    c = json.loads(contract_path.read_text())
    kp, kd, effort, limits = [np.asarray(c[k]) for k in ('kp', 'kd', 'native_effort', 'joint_limits')]
    datasets = {
        'old': (BASE / 'fast_controller_nominal_pilot_v1/labels/labels.npz',
                Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1/trace.npz')),
        'new': (BASE / 'fresh_expert_labels_resume_v1/labels/labels.npz',
                BASE / 'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz'),
    }
    reports, saved, hashes = {}, {}, {str(contract_path): sha(contract_path), str(Path(__file__)): sha(Path(__file__))}
    for name, (label_path, trace_path) in datasets.items():
        with np.load(label_path) as a:
            control, target, base = a['control'].copy(), a['expert_target'].copy(), a['base_target'].copy()
        with np.load(trace_path) as a:
            np.testing.assert_array_equal(target, a['target'][control])
            indices = control[:, None] * 10 + np.arange(10)[None, :]
            q = a['physics_qpos'][indices, 7:]
            dq = a['physics_qvel'][indices, 6:]
            torque = a['physics_torque'][indices]
        requested = kp * (target[:, None, :] - q) - kd * dq
        np.testing.assert_array_equal(np.clip(requested, -effort, effort), torque)
        pos, neg = requested >= effort, requested <= -effort
        fully = np.all(pos | neg, axis=1)
        # Only components saturated during all ten substeps have a nontrivial
        # target equivalence interval on this recorded trajectory. A component
        # with any unsaturated sample remains pinned to its original target.
        low = np.maximum(limits[:, 0], np.max(np.where(pos, q + (effort + kd * dq) / kp, -np.inf), axis=1))
        high = np.minimum(limits[:, 1], np.min(np.where(neg, q + (-effort + kd * dq) / kp, np.inf), axis=1))
        low, high = np.where(fully, low, target), np.where(fully, high, target)
        assert np.all(low <= target + 1e-13) and np.all(high >= target - 1e-13)
        proposal = np.where(fully, np.clip(base, low, high), target)
        proposed_torque = np.clip(kp * (proposal[:, None, :] - q) - kd * dq, -effort, effort)
        diff = float(np.max(np.abs(proposed_torque - torque)))
        np.testing.assert_allclose(proposed_torque, torque, atol=2e-12, rtol=0)
        width = np.maximum(high - low, 0.)
        movement = target - proposal
        reports[name] = dict(rows=len(control), substep_joint_saturation_fraction=float(np.mean(pos | neg)),
            fully_saturated_control_joint_components=int(np.sum(fully)),
            fully_saturated_component_fraction=float(np.mean(fully)),
            fully_saturated_counts_by_joint=np.sum(fully, axis=0).tolist(),
            interval_width_p95_rad=float(np.percentile(width[fully], 95)) if np.any(fully) else 0.,
            interval_width_max_rad=float(width.max()),
            target_to_closest_equivalent_base_rms_rad=float(np.sqrt(np.mean(movement ** 2))),
            target_to_closest_equivalent_base_max_rad=float(np.abs(movement).max()),
            changed_components_over_001_rad=int(np.sum(np.abs(movement) > .01)),
            torque_difference_max_on_recorded_states_nm=diff,
            target_rmse_to_base_original_rad=float(np.sqrt(np.mean((target - base) ** 2))),
            target_rmse_to_base_hypothetical_rad=float(np.sqrt(np.mean((proposal - base) ** 2))))
        saved.update({name + '_' + k: v for k, v in dict(control=control, fully_saturated=fully,
            lower=low, upper=high, hypothetical_target=proposal, target_difference=movement).items()})
        hashes.update({str(p): sha(p) for p in (label_path, trace_path)})
    out = BASE / 'expert_target_saturation_diagnostic_v1'
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / 'diagnostic_arrays.npz', **saved)
    result = dict(kind='saved_state_PD_target_identifiability_diagnostic', datasets=reports,
        joint_names=c['joint_names'], physics_steps=0, inference_calls=0, training_or_label_change=False,
        actual_history_unchanged=True, rollout_equivalence_qualified=False,
        limitation='Hypothetical targets are never executed. Different targets change BFM action history; saved-state torque equality is not closed-loop policy equivalence.',
        hashes=hashes, arrays_sha256=sha(out / 'diagnostic_arrays.npz'))
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
