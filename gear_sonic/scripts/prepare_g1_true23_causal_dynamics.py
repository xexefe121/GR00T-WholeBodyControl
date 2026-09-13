"""Prepare causal full-body references and existing expert initialization rows."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override, motion_states, sha256
from gear_sonic.utils.g1_true23_received_features import prepare_reference, features_numpy, MeasuredHistory

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts' if sys.platform == 'win32' else '/mnt/e/codex-artifacts')
OLD = BASE / 'sonic23_teleop_six_hour_20260910'
NEW = BASE / 'sonic23_teleop_resume_20260911'
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'


def archive(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows, targets, reset_rows, metadata = [], [], [], []
    for clip in ('walk003', 'walk002', 'pico', 'walk008'):
        model, c, original, timeline, manifest = load_native_bundle(BUNDLE, clip)
        for k in ('default_q', 'kp', 'kd', 'native_effort', 'training_effort', 'native_velocity', 'joint_limits'):
            c[k] = np.asarray(c[k])
        refpath = OLD / 'mjbatch_intent_floor_inputs_v1' / clip / 'reference.npz'
        motion, _ = load_motion_override(refpath, BUNDLE, clip, model, c, original, timeline, manifest)
        original29 = archive(BUNDLE / clip / 'original29.npz')
        reference = prepare_reference(motion, original29, c)
        states = motion_states(motion)
        path = a.output / (clip + '.npz')
        np.savez_compressed(path, **reference, states=states)
        metadata.append(dict(name=clip, file=path.name, training=clip!='walk008', length=len(states),
            total=timeline['total_requested_controls'], initial_frame=10,
            source_start=next(p['control_start'] for p in timeline['phases'] if p['name']=='source_motion'),
            source_stop=next(p['control_stop'] for p in timeline['phases'] if p['name']=='source_motion'),
            reference_sha256=sha256(refpath), original_sha256=sha256(BUNDLE/clip/'original29.npz')))
        if clip == 'walk008': continue
        clip_id = len(metadata)-1
        if clip == 'walk003':
            traces = [archive(NEW/'fast_feedback_walk003_v1/baseline_v1'/part/'trace.npz')
                      for part in ('nominal','post_lifecycle_hold_5s')]
            history = MeasuredHistory(c)
            for trace in traces:
                for i, target in enumerate(trace['target']):
                    q, v, f = trace['qpos'][i], trace['qvel'][i], int(trace['source_frame'][i])
                    h = history.vector()
                    rows.append(features_numpy(q,v,reference,f,c['default_q'],history.prior,h))
                    targets.append(target)
                    reset_rows.append(np.r_[clip_id,f,q,v,history.prior,h])
                    history.commit(q,v,target)
        else:
            labels = archive(NEW/'pico_walk002_labels_v1/collection'/clip/'labels.npz')
            for i, target in enumerate(labels['expert_target']):
                q,v,f = labels['teacher_qpos'][i], labels['teacher_qvel'][i], int(labels['source_frame'][i])
                prior = labels['previous_action'][i]
                # Saved history is the pre-update history returned by BFMHistory.
                h = labels['history'][i]
                rows.append(features_numpy(q,v,reference,f,c['default_q'],prior,h))
                targets.append(target)
                reset_rows.append(np.r_[clip_id,f,q,v,prior,h])
        print(json.dumps(dict(prepared=clip, expert_rows=len(rows))), flush=True)
    np.savez_compressed(a.output/'expert.npz', features=np.asarray(rows,np.float32),
                        targets=np.asarray(targets,np.float32), resets=np.asarray(reset_rows,np.float64))
    # Use the exact task point convention already used by the native referee.
    audit=json.loads((OLD/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/recorded_source_audit_v2.json').read_text())
    tasks=[next(t for t in audit['original_intent']['convention']['tasks'] if t['name']==n)
           for n in ('left_hand','right_hand','head_proxy')]
    result=dict(clips=metadata, expert_rows=len(rows), features=1323, reference_offsets=[0,-1,-2,-4,-8,-16,-24,-37],
        derivatives='backward_only', future_reference_access=False, simulation_privileged_root=True,
        heldout='walk008', tasks=tasks, hardware_authorized=False)
    (a.output/'bank.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
