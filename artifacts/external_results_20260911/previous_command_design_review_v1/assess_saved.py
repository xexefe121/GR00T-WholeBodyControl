"""Saved-Jacobian linearized comparison; no head evaluation or physics."""
from pathlib import Path
import hashlib
import json
import numpy as np

N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
O=Path(__file__).parent
paths=dict(actual=N/'velocity_chord_student_evaluation_v1/nominal/trace.npz',
    labels=N/'bfm_entry250_labels_v1/labels/labels.npz',
    predictions=N/'velocity_chord_student_v1/fit/final_predictions.npz',
    jacobians=N/'velocity_chord_student_v1/fit/final_jacobians.npz',
    initial_sensitivity=N/'velocity_chord_student_v1/fit/initial_sensitivity.json',
    final_sensitivity=N/'velocity_chord_student_v1/fit/final_sensitivity.json',
    chronology=N/'student_velocity_chord_saved_outcome_v1/report.json',
    decomposition=N/'student_velocity_chord_saved_outcome_v1/early_departure.json',
    contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'),
    runtime=N/'velocity_chord_student_evaluation_v1/source_snapshot_v1/student_linear_runtime.py',
    history=N/'velocity_chord_student_evaluation_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_bfm_seed_observations.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
rms=lambda x:float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
pins={str(p):sha(p) for p in paths.values()}
request=dict(kind='read_only_fixed_observed_departures_times_saved_activation_J',
    controls=[251,252,253],source_sha256=sha(__file__),input_hashes=pins,
    model_evaluations=0,physics_steps=0,optimizer_updates=0)
with (O/'request.json').open('x') as f:json.dump(request,f,indent=2);f.write('\n')
a,t,p,j=[load(paths[k]) for k in ('actual','labels','predictions','jacobians')]
c=read(paths['contract']);J=j['control250_J_raw_features']
assert np.array_equal(a['features'][250],t['features'][0])
span=np.diff(np.asarray(c['joint_limits']),axis=1).ravel();C=.25*np.asarray(c['training_effort'])/np.asarray(c['kp'])
rows=[]
for control in request['controls']:
    local=control-250;delta_x=a['features'][control].astype(np.float64)-t['features'][local].astype(np.float64)
    actual_head=a['delta'][control].astype(np.float64)-p['predicted_delta'][2038+local].astype(np.float64)
    base=a['base_target'][control]-t['base_target'][local]
    parts={name:J[:,lo:hi]@delta_x[lo:hi] for name,(lo,hi) in dict(joint_position=(0,23),joint_velocity=(23,46),
        root_angular_gravity=(46,52),previous_target=(52,75),root_linear_height=(75,79),goals=(79,1023),BFM_base=(1023,1046),raw_prior=(1046,1069)).items()}
    prior=parts['previous_target']+parts['raw_prior'];full=J@delta_x
    dqprior=(a['previous_action'][control].astype(np.float64)-t['previous_action'][local].astype(np.float64))*C
    radius=.01*span
    rows.append(dict(control=control,observed_head_change_rms_rad=rms(actual_head),observed_BFM_base_change_rms_rad=rms(base),
        linearized_head_change_rms_rad=rms(full),linearization_error_rms_rad=rms(full-actual_head),
        direct_prior_combined_rms_rad=rms(prior),velocity_rms_rad=rms(parts['joint_velocity']),
        block_rms_rad={k:rms(v) for k,v in parts.items()},
        block_left_knee_rad={k:float(v[3]) for k,v in parts.items()},
        direct_prior_left_knee_rad=float(prior[3]),actual_head_left_knee_rad=float(actual_head[3]),
        prior_target_equivalent_departure_rms_rad=rms(dqprior),prior_target_equivalent_departure_max_rad=float(np.abs(dqprior).max()),
        per_axis_departure_over_one_percent_span_max=float(np.max(np.abs(dqprior)/radius)),
        axes_beyond_one_percent_span=int(np.sum(np.abs(dqprior)>radius))))
for p,h in pins.items():assert sha(p)==h
result=dict(passed=True,rows=rows,one_percent_span_radius_rad=radius.tolist(),
    conditional_sensitivity_at_exact_shared_activation={label:read(paths[label+'_sensitivity'])['rows'][0] for label in ('initial','final')},
    scope='Fixed saved control250 Jacobian multiplied by observed same-clock actual-minus-teacher feature differences. No Jacobian or head recomputation.',
    limitations=['At controls251 onward, this is a local first-order approximation anchored at the exact shared control250 state; it is not a causal intervention.',
                 'Blocks are coupled and may cancel; their RMS values must not be interpreted as fractions of causal responsibility.',
                 'One-percent-span axis radius is a proposed bounded coverage scale, not a new command safety guarantee.'],
    source_sha256=sha(__file__),request_sha256=sha(O/'request.json'),input_hashes=pins,
    model_evaluations=0,physics_steps=0,optimizer_updates=0)
with (O/'saved_assessment.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(sha256=sha(O/'saved_assessment.json'),rows=rows)))
