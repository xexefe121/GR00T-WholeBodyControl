"""Actual saved phase-student failure versus saved expert data, no inference/physics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

BASE=Path(__file__).resolve().parent;NEW=BASE.parent;RUN=NEW/'fast_controller_phase_fit_v1'
SRC=RUN/'source_snapshot_v1';sys.path.insert(0,str(SRC))
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rms(a,axis=None):return np.sqrt(np.mean(np.asarray(a,np.float64)**2,axis=axis))
def first(controls,mask):
    ids=np.flatnonzero(mask);return int(controls[ids[0]]) if len(ids) else None

def main():
    assert not (BASE/'report.json').exists()
    bundle=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
    paths=dict(actual=RUN/'nominal/trace.npz',fit=RUN/'fit/teacher_fit.npz',contract=bundle/'contract.json',
        old=NEW/'fast_controller_nominal_pilot_v1/labels/labels.npz',query1=NEW/'fresh_expert_labels_resume_v1/labels/labels.npz',
        query250=NEW/'bfm_entry250_labels_v1/labels/labels.npz',original29=bundle/'walk003/original29.npz',
        motion=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz'),
        root_qualification=NEW/'bfm250_expert_root_qualification_v1/qualification.json',
        final_report=RUN/'fit/report.json',prefix_parity=RUN/'canonical_prefix250_parity.json',
        query250_parity=RUN/'actual_query250_input_parity.json')
    a=load(paths['actual']);fit=load(paths['fit']);datasets=[load(paths[k]) for k in ('old','query1','query250')]
    q=datasets[2];c=json.loads(paths['contract'].read_text());names=c['joint_names']
    default,kp,effort=[np.asarray(c[k]) for k in ('default_q','kp','training_effort')];limits=np.asarray(c['joint_limits'])
    controls=np.arange(250,279);count=len(controls);assert len(a['target'])==279 and len(a['physics_torque'])==2784
    np.testing.assert_array_equal(a['global_control'],np.arange(279))
    np.testing.assert_array_equal(a['previous_action'][1:],a['action'][:-1])
    goals=GoalFeatures(load(paths['motion']),load(paths['original29']),c)
    history=BFMHistory();previous=np.zeros(23,np.float32);named={k:[] for k in history.data}
    reconstructed_features=[];reconstructed_actions=[]
    for control in range(279):
        np.testing.assert_array_equal(previous,a['previous_action'][control])
        np.testing.assert_array_equal(previous,a['control_previous_action_before'][control])
        for key in named:named[key].append(history.data[key].copy())
        sensed,terms=state_and_terms(a['qpos'][control,7:],a['qvel'][control,6:],a['qpos'][control,3:7],a['qvel'][control,3:6],previous,default)
        before=history.before_update(terms)
        np.testing.assert_array_equal(before,a['control_history_before'][control]);np.testing.assert_array_equal(before,a['history'][control])
        np.testing.assert_array_equal(sensed,a['state'][control])
        previous_target=np.clip(default+previous*.25*effort/kp,limits[:,0],limits[:,1])
        features=np.r_[goals(a['qpos'][control],a['qvel'][control],previous_target,control+11),a['base_target'][control]-default,previous].astype(np.float32)
        np.testing.assert_array_equal(features,a['features'][control]);reconstructed_features.append(features)
        target=np.clip(a['base_target'][control]+a['delta'][control],limits[:,0],limits[:,1])
        np.testing.assert_array_equal(target,a['target'][control])
        rawbase=((a['base_target'][control]-default)*kp/(.25*effort)).astype(np.float32)
        combined=(rawbase+a['delta'][control]*kp/(.25*effort)).astype(np.float32)
        np.testing.assert_array_equal(combined,a['action'][control]);reconstructed_actions.append(combined)
        previous=a['action'][control].copy()
    for key in named:
        named[key]=np.asarray(named[key]);np.testing.assert_array_equal(history.data[key],a['final_history_'+key])
        np.testing.assert_array_equal(named[key][250],q['history_'+key][0])
    np.testing.assert_array_equal(previous,a['final_previous_action'])
    selected=[{key:d[key][(d['control']>=250)&(d['control']<1269)] for key in ('features','expert_target','residual_rad','control','base_target','previous_action')} for d in datasets]
    features=np.concatenate([d['features'] for d in selected]).astype(np.float64)
    expert=np.concatenate([d['expert_target'] for d in selected]);teacherbase=np.concatenate([d['base_target'] for d in selected])
    mean,std=[fit[k].astype(np.float64) for k in ('feature_mean','feature_std')]
    x=a['features'][controls].astype(np.float64);z=(x-mean)/std;train=(features-mean)/std
    distances=np.empty((count,3057),np.float64)
    for i in range(count):distances[i]=rms(z[i]-train,axis=1)
    nearest=np.argmin(distances,axis=1)
    lo,hi=features.min(0),features.max(0);excursion=np.maximum(np.maximum(lo-x,x-hi),0)/std
    raw=a['base_target'][controls]+a['delta'][controls];clip=a['target'][controls]-raw
    normalized_applied=((a['target']-default)*kp/(.25*effort)).astype(np.float32)
    raw_applied_difference=a['action']-normalized_applied
    error=a['target'][controls]-q['expert_target'][:count]
    matching_fit=fit['predicted_applied_target'][2038:2038+count]
    saved_error=matching_fit-q['expert_target'][:count]
    feature_delta=(x-q['features'][:count])/std
    base_change=a['base_target'][controls]-q['base_target'][:count]
    head_change=a['delta'][controls].astype(np.float64)-fit['predicted_delta'][2038:2038+count].astype(np.float64)
    saved_clip=matching_fit-(q['base_target'][:count]+fit['predicted_delta'][2038:2038+count])
    decomposition=base_change+head_change+saved_error+clip-saved_clip
    np.testing.assert_allclose(decomposition,error,rtol=0,atol=1e-14)
    named_delta={key:named[key][controls]-q['history_'+key][:count] for key in named}
    blocks=dict(proprio=slice(0,79),future_goal=slice(79,1023),BFM_base_offset=slice(1023,1046),previous_action=slice(1046,1069))
    rows=[]
    for i,control in enumerate(controls):
        nn=int(nearest[i]);joint=int(np.argmax(np.abs(error[i])))
        closest={}
        for code,label in enumerate(('old','query1','query250')):
            j=code*1019+int(np.argmin(distances[i,code*1019:(code+1)*1019]))
            closest[label]=dict(control=int(j%1019+250),distance=float(distances[i,j]),
                actual_target_difference_rms_rad=float(rms(a['target'][control]-expert[j])),
                saved_teacher_input_fit_error_rms_rad=float(rms(fit['predicted_applied_target'][j]-expert[j])))
        rows.append(dict(control=int(control),physics_substeps=int(a['physics_substeps'][control]),
            actual_vs_matching_expert_target_rmse_rad=float(rms(error[i])),matching_expert_input_fit_rmse_rad=float(rms(saved_error[i])),
            actual_vs_matching_expert_input_feature_distance=float(rms(feature_delta[i])),
            actual_vs_matching_expert_input_block_distance={key:float(rms(feature_delta[i,sl])) for key,sl in blocks.items()},
            actual_base_change_rms_rad=float(rms(base_change[i])),actual_head_output_change_rms_rad=float(rms(head_change[i])),
            qpos_joint_max_difference_rad=float(np.max(np.abs(a['qpos'][control,7:]-q['teacher_qpos'][i,7:]))),
            qvel_joint_max_difference_radps=float(np.max(np.abs(a['qvel'][control,6:]-q['teacher_qvel'][i,6:]))),
            matching_history_difference_rms={key:float(rms(value[i])) for key,value in named_delta.items()},
            nearest=closest,nearest_any=dict(dataset=('old','query1','query250')[nn//1019],control=nn%1019+250,distance=float(distances[i,nn])),
            clipped_joints=[names[j] for j in np.flatnonzero(np.abs(clip[i])>1e-12)],clip_max_abs_rad=float(np.max(np.abs(clip[i]))),
            features_outside_training_minmax=int(np.sum(excursion[i]>0)),max_standardized_feature_excursion=float(excursion[i].max()),
            previous_action_outside_training_joint_ranges=[names[j] for j in np.flatnonzero(excursion[i,-23:]>0)],
            previous_action_max_abs=float(np.abs(a['previous_action'][control]).max()),combined_action_max_abs=float(np.abs(a['action'][control]).max()),
            combined_minus_normalized_applied_action_max_abs=float(np.abs(raw_applied_difference[control]).max()),
            worst_matching_target_joint=names[joint],worst_matching_target_signed_error_rad=float(error[i,joint]),
            right_hip_yaw=dict(qpos=float(a['qpos'][control,15]),qvel=float(a['qvel'][control,14]),
                base=float(a['base_target'][control,8]),delta=float(a['delta'][control,8]),raw_target=float(raw[i,8]),
                applied_target=float(a['target'][control,8]),expert_target=float(q['expert_target'][i,8]),
                previous_action=float(a['previous_action'][control,8]),combined_action=float(a['action'][control,8]))))
    ratios=np.abs(a['physics_qvel'][-1,6:])/np.asarray(c['native_velocity']);failed=int(np.argmax(ratios))
    physical_start=2500
    result=dict(kind='saved_phase_final65000_failure_diagnosis',controls=[250,278],learned_full_controls=28,partial_control_substeps=4,
        all279_actual_features_history_state_targets_actions_reconstructed_exact=True,all_final_named_history_exact=True,
        query250_named_history_exact=True,query250_full_input_exact=True,
        query250_actual_ONNX_vs_saved_Torch_max_delta=float(np.max(np.abs(a['delta'][250]-fit['predicted_delta'][2038]))),
        query250_actual_target_error_rms_rad=float(rms(error[0])),
        first_actual_target_differs_from_expert=first(controls,np.any(np.abs(error)>1e-12,axis=1)),
        first_actual_input_differs_from_matching_expert=first(controls,np.any(x!=q['features'][:count],axis=1)),
        first_named_history_differs_from_matching_expert={key:first(controls,np.any(delta.reshape(count,-1)!=0,axis=1)) for key,delta in named_delta.items()},
        first_any_feature_training_minmax_excursion=first(controls,np.any(excursion>0,axis=1)),
        first_previous_action_training_span_excursion=first(controls,np.any(excursion[:,-23:]>0,axis=1)),
        first_native_target_clipping=first(controls,np.any(np.abs(clip)>1e-12,axis=1)),
        first_preclip_combined_action_differs_from_applied_normalized_action=first(controls,np.any(np.abs(raw_applied_difference[controls])>1e-5,axis=1)),
        failed_joint=names[failed],failed_joint_index=failed,failed_control=278,failed_substep=4,
        failed_velocity_radps=float(a['physics_qvel'][-1,6+failed]),native_velocity_limit_radps=float(c['native_velocity'][failed]),
        failed_velocity_ratio=float(ratios[failed]),observed_previous_action_max_abs=float(np.abs(a['previous_action'][controls]).max()),
        training_previous_action_max_abs=float(np.abs(features[:,-23:]).max()),rows=rows,
        comparison='Matching expert-input predictions are saved values from a different expert trajectory after control250. They are not expert counterfactual actions evaluated at actual later student states.',
        distribution_shift_definition='First mismatch to corresponding expert input and first coordinate outside the three moving-dataset min/max envelope are reported separately; nearest distances have no fitted pass threshold.',
        input_sha256={key:sha(path) for key,path in paths.items()},source_sha256={str(p):sha(p) for p in [Path(__file__),SRC/'gear_sonic/utils/g1_true23_bfm_seed_observations.py',SRC/'gear_sonic/utils/g1_true23_mpc_student.py']},
        inference_calls=0,optimizer_calls=0,new_physics_steps=0,hardware_authorized=False)
    np.savez_compressed(BASE/'arrays.npz',control=controls,cross_distances=distances,nearest=nearest,
        standardized_feature_excursion=excursion,native_clip_adjustment=clip,actual_raw_target=raw,
        raw_combined_minus_applied_normalized_action=raw_applied_difference,
        actual_target_minus_matching_expert=error,matching_teacher_input_fit_error=saved_error,
        matching_feature_delta=feature_delta,actual_base_change=base_change,actual_head_change=head_change,
        actual_precontrol_qpos=a['qpos'][controls],actual_precontrol_qvel=a['qvel'][controls],
        physics_time=a['physics_time'][physical_start:],physics_qpos=a['physics_qpos'][physical_start:],
        physics_qvel=a['physics_qvel'][physical_start:],physics_command=a['physics_torque'][physical_start:],
        physics_actuator_torque=a['physics_actuator_torque'][physical_start:],
        **{'actual_history_'+k:v[controls] for k,v in named.items()},**{'matching_history_delta_'+k:v for k,v in named_delta.items()})
    result['arrays_sha256']=sha(BASE/'arrays.npz')
    (BASE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('first_') or k in ('failed_joint','failed_velocity_radps','query250_actual_target_error_rms_rad','query250_actual_ONNX_vs_saved_Torch_max_delta')}))

if __name__=='__main__':main()
