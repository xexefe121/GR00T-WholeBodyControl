"""Saved-only arithmetic summary; no new predictions, optimization, or dynamics."""
import json,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as x:return {k:x[k] for k in x.files}
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
paths=dict(actual=NEW/'velocity_chord_student_evaluation_v1/nominal/trace.npz',
    teacher=NEW/'bfm_entry250_labels_v1/labels/labels.npz',
    predictions=NEW/'velocity_chord_student_v1/fit/final_predictions.npz',
    report=BASE/'report.json',normalization=NEW/'fast_controller_phase_fit_v1/fit/teacher_fit.npz')
assert not (BASE/'early_departure.json').exists()
pins={key:sha(path) for key,path in paths.items()}
assert pins['actual']=='1e8e44a6c405a86138558ea89681fb225b73b2be969e2e7a608903ebb5f80872'
assert json.loads(paths['report'].read_text())['passed']
a,q,p,n=[load(paths[key]) for key in ('actual','teacher','predictions','normalization')]
std=n['feature_std'].astype(np.float64)
blocks=dict(joint_position=(0,23),joint_velocity=(23,46),previous_target=(52,75),goals=(79,1023),BFM_base=(1023,1046),raw_prior=(1046,1069))
rows=[]
for control in range(250,len(a['target'])):
    j=control-250;idx=2038+j
    base_change=a['base_target'][control]-q['base_target'][j]
    head_change=a['delta'][control].astype(np.float64)-p['predicted_delta'][idx].astype(np.float64)
    feature_change=(a['features'][control].astype(np.float64)-q['features'][j].astype(np.float64))/std
    rows.append(dict(control=control,base_change_rms_rad=rms(base_change),head_change_rms_rad=rms(head_change),
        total_unclipped_target_change_rms_rad=rms(base_change+head_change),
        feature_block_standardized_rms={key:rms(feature_change[lo:hi]) for key,(lo,hi) in blocks.items()},
        left_knee=dict(qpos=float(a['qpos'][control,10]),qvel=float(a['qvel'][control,9]),
            base=float(a['base_target'][control,3]),delta=float(a['delta'][control,3]),
            raw_target=float(a['raw_proposal'][control,3]),applied_target=float(a['target'][control,3]),
            previous_raw_action=float(a['previous_action'][control,3]),next_raw_action=float(a['action'][control,3]),
            applied_normalized_action=float(a['actual_normalized_action'][control,3]),
            same_clock_expert_target=float(q['expert_target'][j,3]))))
assert {key:sha(path) for key,path in paths.items()}==pins
report=dict(input_sha256=pins,source_sha256=sha(__file__),rows=rows,
    interpretation='At251 both actual-state and saved teacher-state controller outputs are compared. Base/head differences are arithmetic decomposition of two recorded trajectories, not independent causal effects or a replanned expert action at the actual state.',
    inference_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0)
(BASE/'early_departure.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows[:4],indent=2))
