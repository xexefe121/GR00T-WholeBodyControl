"""Diagnose original-reference derivative fields without changing their bytes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion

OUT = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/original_native_derivative_audit_v2')
OUT.mkdir(exist_ok=False)


def stats(x):
    return dict(abs_max=float(np.max(np.abs(x))), abs_p50=float(np.percentile(np.abs(x), 50)), abs_p95=float(np.percentile(np.abs(x), 95)), rms=float(np.sqrt(np.mean(x*x))))


rows = []
for clip in ('pico', 'walk002'):
    motion, timeline, path = load_motion(clip)
    n, dt = len(motion['joint_pos']), .02
    phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    source = slice(phase['frame_start'], phase['frame_stop'])
    expected = {'joint_vel': np.gradient(motion['joint_pos'], dt, axis=0),
                'body_lin_vel_w': np.gradient(motion['body_pos_w'], dt, axis=0)}
    earlier, later = np.maximum(np.arange(n)-1, 0), np.minimum(np.arange(n)+1, n-1)
    angular = np.empty_like(motion['body_ang_vel_w'])
    for b in range(24):
        r = Rotation.from_quat(motion['body_quat_w'][:, b, [1,2,3,0]])
        angular[:, b] = (r[later] * r[earlier].inv()).as_rotvec() / ((later-earlier)*dt)[:,None]
    expected['body_ang_vel_w'] = angular
    fields = {}
    for key, correct in expected.items():
        error = motion[key]-correct
        flat = np.abs(error).reshape(n, -1)
        worst = np.unravel_index(np.argmax(np.abs(error)), error.shape)
        fields[key] = dict(all=stats(error), source=stats(error[source]), max_error_index=list(map(int,worst)),
                           mismatch_frames_above_1e_6=int(np.sum(flat.max(axis=1)>1e-6)),
                           source_existing=stats(motion[key][source]), source_canonical=stats(correct[source]))
    joint = motion['joint_vel'][source]
    canon = expected['joint_vel'][source]
    lag_rmse = {}
    for lag in range(-3,4):
        a = joint[max(0,lag):len(joint)+min(0,lag)]
        b = canon[max(0,-lag):len(canon)+min(0,-lag)]
        lag_rmse[str(lag)] = float(np.sqrt(np.mean((a-b)**2)))
    comparisons = {}
    for key, position_key in (('joint_vel','joint_pos'), ('body_lin_vel_w','body_pos_w')):
        position=motion[position_key]
        backward=np.concatenate((np.diff(position,axis=0)[:1],np.diff(position,axis=0)),axis=0)/dt
        forward=np.concatenate((np.diff(position,axis=0),np.diff(position,axis=0)[-1:]),axis=0)/dt
        comparisons[key] = {}
        for method, derivative in (('backward',backward),('forward',forward),('central',expected[key])):
            for shift in (-1,0,1):
                indices=np.arange(phase['frame_start'],phase['frame_stop'])
                comparisons[key][f'{method}_candidate_shift_{shift:+d}']=stats(motion[key][indices]-derivative[indices+shift])
    comparisons['body_ang_vel_w']={}
    for frame_mode in ('world','body'):
        variants={k:np.empty_like(angular) for k in ('backward','forward','central','rotvec_gradient')}
        for b in range(24):
            r=Rotation.from_quat(motion['body_quat_w'][:,b,[1,2,3,0]])
            rv=r.as_rotvec()
            variants['rotvec_gradient'][:,b]=np.gradient(rv,dt,axis=0)
            for method,a,c in (('backward',earlier,np.arange(n)),('forward',np.arange(n),later),('central',earlier,later)):
                # Endpoint denominator uses one available interval; source indices are interior.
                duration=np.maximum(c-a,1)*dt
                rel=r[c]*r[a].inv() if frame_mode=='world' else r[a].inv()*r[c]
                variants[method][:,b]=rel.as_rotvec()/duration[:,None]
                if method=='backward': variants[method][0,b]=variants[method][1,b]
                if method=='forward': variants[method][-1,b]=variants[method][-2,b]
        for method,derivative in variants.items():
            if method=='rotvec_gradient' and frame_mode=='body': continue
            for shift in (-1,0,1):
                indices=np.arange(phase['frame_start'],phase['frame_stop'])
                comparisons['body_ang_vel_w'][f'{frame_mode}_{method}_candidate_shift_{shift:+d}']=stats(motion['body_ang_vel_w'][indices]-derivative[indices+shift])
    rows.append(dict(clip=clip, reference_path=str(path), reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     source_phase=phase, fields=fields, joint_source_lag_rmse=lag_rmse, full_source_convention_comparisons=comparisons))
report = dict(kind='original_native_derivative_consistency_audit', dt_seconds=.02, results=rows,
              original_reference_files_changed=False, pose_arrays_changed=False,
              canonical_convention='Root load_case_motion: np.gradient joint/body position; world-axis relative rotation interval for angular velocity.',
              producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
(OUT/'producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps([dict(clip=r['clip'],fields=r['fields'],best_conventions={k:min(v,key=lambda x:v[x]['rms']) for k,v in r['full_source_convention_comparisons'].items()}) for r in rows], indent=2))
