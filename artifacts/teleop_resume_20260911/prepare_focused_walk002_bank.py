"""Add actual successful walk002 quiet/hold examples; no physics or training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from gear_sonic.utils.g1_true23_received_features import MeasuredHistory, features_numpy

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
CAUSAL = BASE / 'causal_dynamics_v1'


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=CAUSAL / 'focused_walk002_bank_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = CAUSAL / 'bank'
    meta = json.loads((source / 'bank.json').read_text())
    clip = next(i for i, row in enumerate(meta['clips']) if row['name'] == 'walk002')
    row = meta['clips'][clip]
    quiet_start = row['source_stop'] + 100 + 11
    reference = archive(source / row['file'])
    reference.pop('states')
    contract = json.loads((ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
    for key in ('default_q', 'kp', 'training_effort'):
        contract[key] = np.asarray(contract[key], dtype=np.float64)
    traces = [BASE / 'walk002_terminal_bfm_hybrid_v1/trace.npz',
              BASE / 'walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s/trace.npz']
    history = MeasuredHistory(contract)
    features, targets, resets, parts, clocks = [], [], [], [], []
    prior_error = history_error = 0.
    previous = None
    counts = []
    for part, path in enumerate(traces):
        trace = archive(path)
        count = len(trace['target'])
        assert trace['qpos'].shape == (count + 1, 30)
        assert trace['qvel'].shape == (count + 1, 29)
        assert trace['source_frame'].shape == (count,)
        if previous is not None:
            np.testing.assert_array_equal(previous['qpos'][-1], trace['qpos'][0])
            np.testing.assert_array_equal(previous['qvel'][-1], trace['qvel'][0])
        selected = 0
        for i, target in enumerate(trace['target']):
            q, v, frame = trace['qpos'][i], trace['qvel'][i], int(trace['source_frame'][i])
            h = history.vector()
            prior_error = max(prior_error, float(np.max(np.abs(history.prior - trace['previous_action'][i]))))
            history_error = max(history_error, float(np.max(np.abs(h - trace['history'][i]))))
            if frame >= quiet_start:
                feature = features_numpy(q, v, reference, frame, contract['default_q'], history.prior, h)
                assert feature.shape == (1323,) and np.isfinite(feature).all()
                features.append(feature)
                targets.append(target)
                resets.append(np.r_[clip, frame, q, v, history.prior, h])
                parts.append(part)
                clocks.append(int(trace['global_control'][i]))
                selected += 1
            history.commit(q, v, target)
        counts.append(selected)
        previous = trace
    assert counts == [300, 250], counts
    # Existing traces store float64 history; the actual new runtime owns float32
    # history. This comparison permits only that storage conversion error.
    assert prior_error < 2e-6 and history_error < 2e-6, (prior_error, history_error)
    extra_features = np.asarray(features, np.float32)
    extra_targets = np.asarray(targets, np.float32)
    extra_resets = np.asarray(resets, np.float64)
    assert extra_resets.shape == (550, 384)
    # Rebuilding against a truncated received prefix gives exactly the same
    # feature vector, including its past reference slots and physical history.
    for i in (0, 25, 299, 300, 549):
        r = extra_resets[i]
        frame = int(r[1])
        received = {k: v[:frame + 1] for k, v in reference.items()}
        rebuilt = features_numpy(r[2:32], r[32:61], received, frame,
                                 contract['default_q'], r[61:84], r[84:])
        np.testing.assert_array_equal(rebuilt, extra_features[i])
    original = archive(source / 'expert.npz')
    offset = len(original['resets'])
    payload = dict(features=np.concatenate((original['features'], extra_features)),
                   targets=np.concatenate((original['targets'], extra_targets)),
                   resets=np.concatenate((original['resets'], extra_resets)))
    for key in original:
        np.testing.assert_array_equal(payload[key][:offset], original[key])
    shutil.copytree(source, args.output)
    unchanged = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*')
                 if p.is_file() and p.name not in ('expert.npz', 'bank.json')}
    for name, sha in unchanged.items():
        assert digest(args.output / name) == sha
    np.savez_compressed(args.output / 'expert.npz', **payload)
    np.savez_compressed(args.output / 'focused_quiet_rows.npz',
                        row_ids=np.arange(offset, offset + len(extra_resets)),
                        trace_part=np.asarray(parts), global_control=np.asarray(clocks))
    meta['expert_rows'] = len(payload['resets'])
    meta['supplemental_walk002_quiet_rows'] = len(extra_resets)
    meta['original_bank'] = str(source)
    (args.output / 'bank.json').write_text(json.dumps(meta, indent=2) + '\n')
    report = dict(kind='actual_walk002_quiet_and_continuous_hold_bank',
                  original_rows=offset, supplemental_rows=550, total_rows=len(payload['resets']),
                  quiet_start_reference_frame=quiet_start, per_trace_rows=counts,
                  quiet_row_start=offset, quiet_row_stop=offset+550,
                  reconstructed_prior_max_error=prior_error, reconstructed_history_max_error=history_error,
                  history='float32 MeasuredHistory rebuilt once across full nominal and continuous hold',
                  original_expert_prefix_unchanged=True, received_prefix_features_exact=True,
                  unchanged_bank_file_sha256=unchanged,
                  inputs={str(p):digest(p) for p in traces},
                  source_script_sha256=digest(Path(__file__)),
                  training_reset_only=True, new_physics_steps=0, training_updates=0,
                  physical_qualification_inherited=False, hardware_authorized=False)
    (args.output / 'focused_bank_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
