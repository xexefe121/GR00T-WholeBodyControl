"""Saved-array-only student diagnosis: no model inference or simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'fast_controller_nominal_pilot_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def archive(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def rms(x,axis=None):return np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2,axis=axis))
def quant(x):return dict(zip(('min','p50','p95','max'),map(float,np.quantile(x,[0,.5,.95,1]))))
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


l=archive(BASE/'labels/labels.npz'); f=archive(BASE/'fit/teacher_fit.npz'); a=archive(BASE/'nominal/trace.npz')
contract=json.loads(Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
names=contract['joint_names']; n=len(a['target']); assert n==24
assert np.array_equal(a['global_control'],l['control'][:n])
assert np.array_equal(a['source_frame'],l['source_frame'][:n])
initial={k:dict(bit_exact=bool(np.array_equal(a[k][0],l[k][0])),max_abs_delta=float(np.max(np.abs(a[k][0]-l[k][0])))) for k in ('features','state','history','previous_action','base_target')}
initial['qpos']=dict(bit_exact=bool(np.array_equal(a['qpos'][0],l['teacher_qpos'][0])))
initial['qvel']=dict(bit_exact=bool(np.array_equal(a['qvel'][0],l['teacher_qvel'][0])))
initial['rollout_vs_saved_teacher_prediction_target_max_delta_rad']=float(np.max(np.abs(a['target'][0]-f['predicted_applied_target'][0])))
initial['note']='Predictions were saved from separate batch/single numerical execution; input/base arrays are exact. No new inference performed.'

std=f['feature_std'].astype(float)
teacher_error=f['applied_error'][:n]
roll_error=a['target']-l['expert_target'][:n]
perjoint=[]
for j,name in enumerate(names):
    perjoint.append(dict(joint=name,index=j,initial_expert_target=float(l['expert_target'][0,j]),
        initial_teacher_input_prediction=float(f['predicted_applied_target'][0,j]),
        initial_teacher_input_error=float(teacher_error[0,j]),initial_rollout_error=float(roll_error[0,j]),
        first24_teacher_input_rmse=float(rms(teacher_error[:,j])),
        first24_teacher_input_max_abs=float(np.max(np.abs(teacher_error[:,j]))),
        first24_rollout_rmse=float(rms(roll_error[:,j])),
        final_teacher_input_error=float(teacher_error[-1,j]),final_rollout_error=float(roll_error[-1,j])))
groups={'proprio':(0,79),'goals_in_measured_heading':(79,1023),'BFM_unclipped_base':(1023,1046),'previous_effective_action':(1046,1069)}
controls=[]
for i in range(n):
    d=a['features'][i].astype(float)-l['features'][i].astype(float)
    controls.append(dict(control=i,time_seconds=.02*i,source_frame=int(a['source_frame'][i]),
        qpos_max_abs_delta=float(np.max(np.abs(a['qpos'][i]-l['teacher_qpos'][i]))),
        joint_q_max_abs_delta=float(np.max(np.abs(a['qpos'][i,7:]-l['teacher_qpos'][i,7:]))),
        qvel_max_abs_delta=float(np.max(np.abs(a['qvel'][i]-l['teacher_qvel'][i]))),
        root_position_delta_m=float(np.linalg.norm(a['qpos'][i,:3]-l['teacher_qpos'][i,:3])),
        normalized_feature_RMS_delta=float(rms(d/std)),
        feature_group_normalized_RMS_delta={k:float(rms(d[s:e]/std[s:e])) for k,(s,e) in groups.items()},
        BFM_base_target_RMS_delta=float(rms(a['base_target'][i]-l['base_target'][i])),
        history_max_abs_delta=float(np.max(np.abs(a['history'][i]-l['history'][i]))),
        previous_action_max_abs_delta=float(np.max(np.abs(a['previous_action'][i]-l['previous_action'][i]))),
        teacher_input_target_RMSE=float(rms(teacher_error[i])),
        actual_rollout_target_RMSE=float(rms(roll_error[i]))))

# Euclidean metric is based on exactly the saved training normalization. RMS
# divides by sqrt(1069); no chosen "incompatibility" threshold is implied.
x=l['features'][:250].astype(float)
xnorm=(x-f['feature_mean'])/std
distance=cdist(xnorm,xnorm)/np.sqrt(x.shape[1])
np.fill_diagonal(distance,np.inf)
_,inv,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
duplicate_groups=[]
for g in np.flatnonzero(counts>1):
    rows=np.flatnonzero(inv==g)
    duplicate_groups.append(dict(rows=rows.tolist(),expert_target_max_spread_rad=float(np.ptp(l['expert_target'][rows],axis=0).max()),residual_max_spread_rad=float(np.ptp(l['residual_rad'][rows],axis=0).max())))


def neighbor_audit(d):
    nearest=np.argmin(d,axis=1); dd=d[np.arange(250),nearest]
    td=l['expert_target'][:250]-l['expert_target'][nearest]
    rd=l['residual_rad'][:250]-l['residual_rad'][nearest]
    rows=[]
    for i,j in enumerate(nearest):
        k=int(np.argmax(np.abs(td[i])))
        rows.append(dict(control=i,neighbor=int(j),separation_controls=abs(i-int(j)),
            normalized_feature_RMS_distance=float(dd[i]),normalized_feature_L2_distance=float(dd[i]*np.sqrt(1069)),
            target_RMS_difference_rad=float(rms(td[i])),target_max_difference_rad=float(np.max(np.abs(td[i]))),
            maximum_target_difference_joint=names[k],signed_maximum_target_difference_rad=float(td[i,k]),
            residual_RMS_difference_rad=float(rms(rd[i]))))
    closest=sorted(rows,key=lambda r:r['normalized_feature_RMS_distance'])
    ratio=np.asarray([r['target_RMS_difference_rad'] for r in rows])/dd
    return dict(distance_quantiles=quant(dd),target_RMS_difference_quantiles=quant(rms(td,axis=1)),
        target_max_difference_quantiles=quant(np.max(np.abs(td),axis=1)),
        target_RMS_per_normalized_feature_RMS_ratio_quantiles=quant(ratio),
        closest10=closest[:10],largest_ratio10=[rows[i] for i in np.argsort(ratio)[-10:][::-1]],all_rows=rows)


nn=neighbor_audit(distance)
local_witness=[]
for p in nn['largest_ratio10'][:3]:
    i,j=p['control'],p['neighbor']
    gi=l['features'][i,79:1023].reshape(8,118)
    gj=l['features'][j,79:1023].reshape(8,118)
    local_witness.append(dict(**p,
        actual_joint_position_max_difference_rad=float(np.max(np.abs(l['teacher_qpos'][i,7:]-l['teacher_qpos'][j,7:]))),
        actual_joint_velocity_max_difference_rad_s=float(np.max(np.abs(l['teacher_qvel'][i,6:]-l['teacher_qvel'][j,6:]))),
        native_joint_goal_knots_bit_exact=bool(np.array_equal(gi[:,:23],gj[:,:23])),
        native_joint_goal_velocity_knots_bit_exact=bool(np.array_equal(gi[:,23:46],gj[:,23:46]))))
nonlocal_distance=distance.copy()
nonlocal_distance[np.abs(np.arange(250)[:,None]-np.arange(250)[None,:])<=4]=np.inf
nn_far=neighbor_audit(nonlocal_distance)
thresholds=[]
tri=np.triu_indices(250,1)
paird=distance[tri]; targetdiff=l['expert_target'][tri[0]]-l['expert_target'][tri[1]]
for cut in [.01,.05,.1,.25]:
    mask=paird<=cut
    thresholds.append(dict(normalized_feature_RMS_distance_at_most=cut,pairs=int(mask.sum()),
        target_RMS_difference_quantiles=quant(rms(targetdiff[mask],axis=1)) if mask.any() else None,
        target_max_difference_quantiles=quant(np.max(np.abs(targetdiff[mask]),axis=1)) if mask.any() else None))
curve=json.loads((BASE/'fit/metrics.json').read_text())
loss=np.array([r['normalized_residual_training_loss'] for r in curve])
lossinfo=dict(records=curve,record_count=len(curve),declines=int(np.sum(np.diff(loss)<0)),increases=int(np.sum(np.diff(loss)>0)),
    loss_reduction_100_to_1000_fraction=float(1-loss[-1]/loss[0]),loss_reduction_800_to_1000_fraction=float(1-loss[-1]/loss[-3]),
    interpretation='Saved sampled-minibatch losses mostly continue descending; not a plateau witness. These are different sampled batches, not a repeated fixed validation loss. No forecast of additional fitting or closed-loop behavior follows.')

# Earliest actual state after any measured divergence, before source and before
# the first bound failure. Retain exact student effective-action history.
qdiff=np.max(np.abs(a['qpos'][:n]-l['teacher_qpos'][:n]),axis=1)
selected=int(np.flatnonzero(qdiff>0)[0]); assert selected==1
snapshot=dict(integration_state_spec=a['integration_state_spec'],integration=a['control_integration_before'][selected],
    qpos=a['qpos'][selected],qvel=a['qvel'][selected],history_flat=a['control_history_before'][selected],
    previous_action=a['control_previous_action_before'][selected],recorded_controls=np.array(selected,np.int64),
    next_source_frame=a['source_frame'][selected],applied_targets_prefix=a['target'][:selected],
    combined_preclip_actions_prefix=a['action'][:selected],teacher_same_time_target=l['expert_target'][selected],
    actual_history_previous_action_semantics=np.array('student combined preclip BFM-plus-residual action; do not silently replace by clipped-target normalization'))
offset=0
for key,size in [('actions',23),('base_ang_vel',3),('dof_pos',23),('dof_vel',23),('projected_gravity',3)]:
    snapshot['history_'+key]=snapshot['history_flat'][offset:offset+4*size].reshape(4,size).copy();offset+=4*size
assert offset==300
np.savez_compressed(OUT/'selected_actual_control1.npz',**snapshot)
selection=dict(control=selected,seconds=.02,reason='First measured actual state divergence after exact control0 input; control0 already has a qualified teacher label. This captures error growth before prolonged drift or physical violation.',
    original_trace_sha256=sha(BASE/'nominal/trace.npz'),snapshot_sha256=sha(OUT/'selected_actual_control1.npz'),
    actual_query_launched=False,selected_state_qualified_as_expert=False,
    history='Named arrays are copied from precontrol buffer; previous action retains the student combined preclip contract. Goal source/timing unchanged.',
    initialization_only='mj_setState(full integration), mj_forward, mj_setState(same full integration); no further state writes. Query and full-remaining-continuation certification not authorized by this artifact.')
np.savez_compressed(OUT/'comparison_arrays.npz',teacher_error=teacher_error,rollout_error=roll_error,
    normalized_feature_delta=(a['features'].astype(float)-l['features'][:n])/std,
    normalized_pair_distance=distance,teacher_initial_targets=l['expert_target'][:250])
summary=dict(kind='saved_arrays_student_diagnosis_v1',no_inference=True,no_physics=True,no_training=True,
    initial_parity=initial,first24_teacher_input_target_RMSE=float(rms(teacher_error)),
    first24_actual_rollout_target_RMSE=float(rms(roll_error)),per_joint=perjoint,controls=controls,
    exact_duplicate_feature_groups=duplicate_groups,nearest_neighbor_all=nn,nearest_neighbor_excluding_four_adjacent_controls=nn_far,
    largest_local_sensitivity_witnesses=local_witness,
    near_pair_threshold_descriptions=thresholds,training_curve=lossinfo,selected_actual_query_snapshot=selection,
    caveat='Different targets at distinct nearby feature vectors show finite local sensitivity, not contradictory labels. Exact duplicate rows with different labels would be a genuine deterministic representation conflict. This audit does not establish impossibility for a smooth MLP.',
    hashes={str(p.relative_to(BASE)):sha(p) for p in [BASE/'labels/labels.npz',BASE/'fit/teacher_fit.npz',BASE/'fit/metrics.json',BASE/'nominal/trace.npz',BASE/'frozen_inputs_v2.json']},
    script_sha256=sha(__file__))
dump('report.json',summary)
print(json.dumps({k:summary[k] for k in ['initial_parity','first24_teacher_input_target_RMSE','first24_actual_rollout_target_RMSE','exact_duplicate_feature_groups','near_pair_threshold_descriptions','selected_actual_query_snapshot']},indent=2))
print('NEAREST',json.dumps({k:v for k,v in nn.items() if k not in ('all_rows','largest_ratio10')},indent=2))
