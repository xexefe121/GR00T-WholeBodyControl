"""Build AMP demonstrations from consecutive actual MuJoCo states, never targets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from gear_sonic.utils.g1_true23_motion_prior import physical_features

ROOT = Path(__file__).resolve().parents[2]


def build(source, bank, output):
    output.mkdir(parents=True, exist_ok=False)
    bundle = ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    model = mujoco.MjModel.from_xml_path(str(bundle/'native_prepared.xml'))
    with np.load(bundle/'prepared_model_arrays.npz') as z:
        for key in z.files:
            getattr(model, key)[:] = z[key]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    meta = json.loads((bank/'bank.json').read_text())
    feet_ids = [model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
    task_ids = [model.body(t['target_body']).id for t in meta['tasks']]
    task_offsets = np.array([t['target_point'] for t in meta['tasks']])
    files = {
        'walk003': ['fast_feedback_walk003_v1/baseline_v1/nominal/trace.npz',
                    'fast_feedback_walk003_v1/baseline_v1/post_lifecycle_hold_5s/trace.npz'],
        'walk002': ['walk002_terminal_bfm_hybrid_v1/trace.npz',
                    'walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s/trace.npz'],
        'pico': ['pico_walk002_labels_v1/collection/pico/labels.npz'],
    }
    pairs, ids, provenance = [], [], []
    for clip_id, (clip, paths) in enumerate(files.items()):
        assert next(r for r in meta['clips'] if r['name'] == clip)['training']
        for relative in paths:
            path = source/relative
            with np.load(path, allow_pickle=False) as z:
                labels = 'teacher_qpos' in z
                q = z['teacher_qpos' if labels else 'qpos'].copy()
                v = z['teacher_qvel' if labels else 'qvel'].copy()
                controls = z['control' if labels else 'global_control']
                assert q.shape == (len(controls)+1, 30) and v.shape == (len(controls)+1, 29)
                assert np.all(np.diff(controls) == 1), relative
                if labels:
                    assert bool(z['complete'])
                    np.testing.assert_allclose(np.diff(z['control_time_before']), .02, atol=1e-9, rtol=0)
                else:
                    assert np.all(z['physics_substeps'] == 10)
                    np.testing.assert_allclose(np.diff(z['physics_time']), .002, atol=1e-9, rtol=0)
            assert np.isfinite(q).all() and np.isfinite(v).all()
            feet = np.empty((len(q), 2, 3))
            tasks = np.empty((len(q), 3, 3))
            for i in range(len(q)):
                data.qpos[:] = q[i]; data.qvel[:] = v[i]
                mujoco.mj_forward(model, data)
                feet[i] = data.xpos[feet_ids]
                tasks[i] = data.xpos[task_ids]+np.einsum('tij,tj->ti', data.xmat[task_ids].reshape(3, 3, 3), task_offsets)
            features = physical_features(*[torch.tensor(a, dtype=torch.float64) for a in (q, v, feet, tasks)]).numpy()
            transition = np.concatenate((features[:-1], features[1:]), axis=1).astype(np.float32)
            pairs.append(transition)
            ids.append(np.full(len(transition), clip_id, dtype=np.int64))
            provenance.append(dict(clip=clip, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   transitions=len(transition), consecutive_actual_states=True))
            print(json.dumps(provenance[-1]), flush=True)
    path = output/'transitions.npz'
    np.savez_compressed(path, transitions=np.concatenate(pairs), clip_ids=np.concatenate(ids),
                        clip_names=np.array(list(files)), control_dt=np.array(.02))
    report = dict(kind='actual_native23_physical_transition_demonstrations', transitions=sum(map(len, pairs)),
                  state_features=71, transition_features=142, control_dt=.02, heldout_excluded='walk008',
                  kinematic_reference_states_used=False, actor_future_access=False, sources=provenance,
                  dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.bank, args.output)), flush=True)
