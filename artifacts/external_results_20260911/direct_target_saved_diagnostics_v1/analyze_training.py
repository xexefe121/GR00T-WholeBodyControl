"""Read completed saved predictions/curves only; no model or physics calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
FIT = NEW / 'direct_target_student_v1/fit'
GEN = NEW / 'velocity_chord_student_v1/generation'
used = {}

def pin(path):
    path = Path(path)
    used[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def arr(path):
    return np.load(pin(path), allow_pickle=False)

def stats(prediction, target):
    p, t = np.asarray(prediction, np.float64).ravel(), np.asarray(target, np.float64).ravel()
    p2, t2, dot = float(p @ p), float(t @ t), float(p @ t)
    return dict(components=len(p), prediction_RMS_rad=float(np.sqrt(p2/len(p))),
                target_RMS_rad=float(np.sqrt(t2/len(t))),
                error_RMS_rad=float(np.sqrt(np.mean((p-t)**2))),
                prediction_to_target_norm_ratio=float(np.sqrt(p2/t2)) if t2 else None,
                cosine=dot/np.sqrt(p2*t2) if p2 and t2 else None,
                target_explained_energy=1-float(np.sum((p-t)**2))/t2 if t2 else None)

def main():
    progress = arr(FIT/'training_progress.npy')
    assert progress.shape == (5000,6) and np.isfinite(progress).all()
    names = ['nominal','sampled_velocity','physical','total','learning_rate','preclip_gradient_norm']
    windows = []
    for start in range(0,5000,500):
        a = progress[start:start+500]
        windows.append(dict(updates=[start+1,start+500],
            mean={k:float(v) for k,v in zip(names,a.mean(0))},
            first={k:float(v) for k,v in zip(names,a[0])},
            last={k:float(v) for k,v in zip(names,a[-1])}))
    with arr(FIT/'normalization.npz') as a:
        span = a['joint_span'].astype(np.float64)
        default = a['default_q'].copy()
        limits = a['joint_limits'].copy()
    with arr(FIT/'data_identities.npz') as a:
        center_map = a['center_to_nominal'].copy()
        dataset = a['dataset'].copy()
        phase = a['phase'].copy()
        controls = a['control'].copy()
    with arr(GEN/'centers.npz') as a:
        target = a['expert_target'].copy()
    target_velocity = arr(GEN/'teacher_target.npy')
    assert target_velocity.shape == (3057,23,2,23)
    nominal = arr(FIT/'final_ORT_nominal.npy')
    velocity = arr(FIT/'final_ORT_velocity.npy').astype(np.float64).reshape(3057,23,2,23)
    center_prediction = nominal[center_map].astype(np.float64)
    p_response = (velocity-center_prediction[:,None,None])*span
    t_response = target_velocity-target[:,None,None]
    ds = ['old','query1','query250']
    phases = ['acquisition','source','return']
    responses = []
    for d in range(3):
        for ph in range(3):
            ids = np.flatnonzero((dataset[center_map]==d)&(phase[center_map]==ph))
            responses.append(dict(dataset=ds[d],phase=ph,centers=len(ids),
                all=stats(p_response[ids],t_response[ids]),
                first24=stats(p_response[ids[:24]],t_response[ids[:24]])))
    center250 = int(np.flatnonzero((dataset[center_map]==2)&(controls[center_map]==250))[0])
    assert center250 == 2038
    raw = default+span*center_prediction[center250]
    applied = np.clip(raw,limits[:,0],limits[:,1])
    activation = dict(center=center250,control=250,
        predicted_raw=raw.tolist(),predicted_target=applied.tolist(),
        teacher_target=target[center250].tolist(),
        target_error=(applied-target[center250]).tolist(),
        target_RMSE_rad=float(np.sqrt(np.mean((applied-target[center250])**2))),
        clipped_components=int(np.sum(applied!=raw)),
        velocity_response=stats(p_response[center250],t_response[center250]))
    report = dict(kind='saved direct-target training and response diagnostics',
        model_calls=0,native_steps=0,optimizer_steps=0,
        no_checkpoint_selection=True,
        limitation='Saved ORT batch256 predictions; actual WSL batch1 may differ within export tolerance. Local finite responses do not establish closed-loop stability.',
        update_windows=windows,response_cells=responses,activation250=activation,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_pins=used)
    for p,h in used.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    out=BASE/'training_report.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        last1000_nominal_reduction=1-windows[-1]['mean']['nominal']/windows[-3]['mean']['nominal'],
        activation250=activation,response_cells=responses),allow_nan=False))

if __name__=='__main__':main()
