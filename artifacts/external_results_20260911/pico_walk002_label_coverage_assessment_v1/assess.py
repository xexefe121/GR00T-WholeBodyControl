"""Saved-array coverage/provenance only. No labels, policy, model, or dynamics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

def local(p):
    s=str(p).replace('\\','/')
    return Path('/mnt/'+s[0].lower()+s[2:] if sys.platform!='win32' and len(s)>1 and s[1]==':' else s)
ROOT=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
NEW=local('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OLD=local('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
OUT=Path(__file__).parent
OBS=NEW/'pico_control_lm_integration_v1/repo/gear_sonic/utils/g1_true23_bfm_seed_observations.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise ValueError('Byte mismatch: '+label)
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

spec=importlib.util.spec_from_file_location('frozen_pure_observation_contract',OBS)
obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
assert sha(OBS)=='a3c68a9aefad87c3dbda5b515f34a968923e9d35234a278b01b727d73062f68d'
contract=json_read(BUNDLE/'contract.json')
default,kp,training=[np.asarray(contract[k],np.float64) for k in ('default_q','kp','training_effort')]
limits=np.asarray(contract['joint_limits'])
pins={str(OBS):sha(OBS),str(BUNDLE/'contract.json'):sha(BUNDLE/'contract.json'),str(Path(__file__)):sha(Path(__file__))}
cases={}
for clip,qualification_name,qualification_sha,total,stop,history_key,previous_key,audit_rows in (
    ('pico','pico_full_root_qualification_v1','746502265523f527a7836f8b091ba521a56078629b1943468eff04f8bfa61daa',6530,6230,
        'fresh_seed_measured_history','fresh_seed_previous_action',6530),
    ('walk002','walk002_hybrid_root_qualification_v1','a7eaec31dd572724921e2d2986ab8c6ab5db33008f6b3c1af5d2dbd72bf92e4b',1417,1117,
        'history','previous_action',1118)):
    qualification_path=NEW/qualification_name/'qualification.json'
    assert sha(qualification_path)==qualification_sha
    qualification=json_read(qualification_path)
    assert qualification['lifecycle_controls']==total and qualification['both_quiet_windows_pass']
    assert qualification['source_controls']==stop-450 and qualification['separate_hold_controls']==250
    pins[str(qualification_path)]=qualification_sha
    for entry in [*qualification['traces'].values(),*qualification['independent_reports'].values()]:
        p=local(entry['path']);assert sha(p)==entry['sha256'];pins[str(p)]=entry['sha256']
    intent=json_read(local(qualification['independent_reports']['full_intent']['path']))
    assert intent['full_lifecycle_source_intent_pass'] and intent['requested_segment_quiet_pass']
    assert all(intent['source_metric_gates'].values()) and all(intent['quiet_last_three_seconds']['gates'].values())
    timeline_path=BUNDLE/clip/'timeline.json';timeline=json_read(timeline_path)
    phases={p['name']:p for p in timeline['phases']}
    assert phases['acquisition_ramp']['control_start']==250 and phases['acquisition_ramp']['control_stop']==350
    assert phases['source_motion']['control_start']==350 and phases['source_motion']['control_stop']==stop-100
    assert phases['return_ramp']['control_start']==stop-100 and phases['return_ramp']['control_stop']==stop
    assert phases['returned_standing']['control_start']==stop and timeline['total_requested_controls']==total
    path=local(qualification['traces']['full']['path']);trace=read(path)
    assert trace['qpos'].shape==(total+1,30) and trace['qvel'].shape==(total+1,29)
    assert trace['target'].shape==(total,23) and trace[history_key].shape==(total,300) and trace[previous_key].shape==(total,23)
    exact(trace['source_frame'],np.arange(11,total+11,dtype=trace['source_frame'].dtype),'canonical source mapping '+clip)
    assert np.all(trace['physics_substeps']==10)
    assert np.isfinite(trace['target']).all() and np.all(trace['target']>=limits[:,0]) and np.all(trace['target']<=limits[:,1])
    if 'global_control' in trace:exact(trace['global_control'],np.arange(total,dtype=trace['global_control'].dtype),'canonical global controls')
    history=obs.BFMHistory();previous=np.zeros(23,np.float32)
    flat_digest=hashlib.sha256();prior_digest=hashlib.sha256();named_digest=hashlib.sha256()
    checkpoints={};check_controls={0,249,250,349,350,stop-101,stop-100,stop-1,min(stop,audit_rows-1),audit_rows-1}
    action_max=0.;outside_five=0
    for control in range(audit_rows):
        q,dq=trace['qpos'][control],trace['qvel'][control]
        exact(previous,trace[previous_key][control],clip+' previous at '+str(control))
        state,terms=obs.state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],previous,default)
        named={key:value.copy() for key,value in history.data.items()}
        flat=history.before_update(terms)
        exact(flat,trace[history_key][control],clip+' history at '+str(control))
        if 'state' in trace:exact(state,trace['state'][control],clip+' sensor state at '+str(control))
        flat_digest.update(flat.tobytes());prior_digest.update(previous.tobytes())
        for key,value in sorted(named.items()):named_digest.update(key.encode());named_digest.update(value.tobytes())
        if control in check_controls:
            checkpoints[str(control)]=dict(source_frame=control+11,flat_history_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                previous_action_sha256=hashlib.sha256(previous.tobytes()).hexdigest(),named_shapes={k:list(v.shape) for k,v in named.items()})
        if 250<=control<stop:
            action_max=max(action_max,float(np.abs(previous).max()));outside_five+=int(np.sum(np.abs(previous)>5))
        # Walk002's first BFM precontrol1117 is checked, but its raw actor action
        # cannot be recovered from its clipped target. No terminal row is selected.
        if control+1<audit_rows:
            previous=((trace['target'][control]-default)*kp/(.25*training)).astype(np.float32)
    prefix_comparison=None
    if clip=='walk002':
        original_path=NEW/'walk002_full_control_lm_v1/trace.npz';original=read(original_path)
        assert sha(original_path)=='ca656be5b4af07a05a9af3b9ac51b150e300404feb6274c170baa4d6cd734b51'
        checks={}
        for key in ('qpos','qvel','target','source_frame','physics_qpos','physics_qvel','physics_requested_torque',
                    'physics_torque','physics_actuator_force','physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo'):
            n=stop*10+int(key not in ('physics_requested_torque','physics_torque','physics_actuator_force')) if key.startswith('physics_') else stop+int(key in ('qpos','qvel'))
            exact(trace[key][:n],original[key][:n],'original MPC prefix '+key);checks[key]=True
        exact(trace['history'][:stop+1],original['fresh_seed_measured_history'][:stop+1],'original MPC histories through boundary')
        exact(trace['previous_action'][:stop+1],original['fresh_seed_previous_action'][:stop+1],'original MPC prior actions through boundary')
        pins[str(original_path)]=sha(original_path)
        prefix_comparison=dict(all11170_native_samples_and1118_histories_prior_bitexact=True,checks=checks)
    reference_path=OLD/'mjbatch_intent_floor_inputs_v1'/clip/'reference.npz'
    reference=read(reference_path)
    assert len(reference['joint_pos'])==total+11
    for p in (timeline_path,reference_path,reference_path.parent/'portable_receipt.json',BUNDLE/clip/'native_original.npz',BUNDLE/clip/'original29.npz'):
        pins[str(p)]=sha(p)
    cases[clip]=dict(qualified_trace=str(path),qualified_trace_sha256=sha(path),
        qualification_sha256=qualification_sha,full_lifecycle_controls=total,
        candidate_moving_control_inclusive=[250,stop-1],candidate_moving_controls=stop-250,
        selected_source_frames_inclusive=[261,stop+10],
        phases={name:dict(controls_inclusive=[phases[name]['control_start'],phases[name]['control_stop']-1],
            rows=phases[name]['requested_controls']) for name in ('acquisition_ramp','source_motion','return_ramp')},
        excluded_initial_controls_inclusive=[0,249],excluded_terminal_controls_inclusive=[stop,total-1],excluded_separate_hold_controls=250,
        history_source_fields=dict(previous=previous_key,flat_history=history_key),history_rows_byte_exact=audit_rows,
        moving_previous_action_max_abs=action_max,moving_previous_action_components_outside_five=outside_five,
        rolling_history_sha256=flat_digest.hexdigest(),rolling_prior_sha256=prior_digest.hexdigest(),rolling_named_history_sha256=named_digest.hexdigest(),
        checkpoints=checkpoints,original_MPC_prefix_comparison=prefix_comparison,
        complete291_control_integration_available='control_integration_before' in trace,
        saved_base_target_available='base_target' in trace,saved_base_action_available='base_action' in trace,
        latest_preview_frame=(stop-1)+11+37,latest_preview_raw_support_frame_upper_bound=(stop-1)+11+38,
        preview_inside_original_full_reference=((stop-1)+11+38)<len(reference['joint_pos']),
        no_network_clock_or_clip_identity_input=True)

fit=NEW/'fast_controller_phase_fit_v1/fit'
request=json_read(fit/'request.json')
assert request['total_samples']==3057 and request['samples_each']==1019
assert request['phase_rows']==[[100,819,100]]*3 and request['last_global_step']==65000
label_paths=dict(old=NEW/'fast_controller_nominal_pilot_v1/labels/labels.npz',
    query1=NEW/'fresh_expert_labels_resume_v1/labels/labels.npz',query250=NEW/'bfm_entry250_labels_v1/labels/labels.npz')
for name,p in label_paths.items():
    assert sha(p)==request['label_sha256'][name]
    with np.load(p,allow_pickle=False) as z:
        ids=np.flatnonzero((z['control']>=250)&(z['control']<1269))
        exact(z['control'][ids],np.arange(250,1269,dtype=z['control'].dtype),'existing moving rows '+name)
        assert z['features'].shape[1]==1069
        exact(z['joint_limits'],limits,'existing native limits')
        exact(z['joint_span'],np.diff(limits,axis=1).ravel().astype(np.float32),'existing native span')
    pins[str(p)]=sha(p)
for p in (fit/'request.json',fit/'teacher_fit.npz',fit/'student_head.onnx',
    NEW/'bfm_entry250_labels_v1/source_snapshot_v1/collect_bfm250_labels.py',
    NEW/'bfm_entry250_labels_v1/source_snapshot_v1/collect_nominal_labels.py',
    NEW/'bfm_entry250_labels_v1/source_snapshot_v1/student_linear_runtime.py',
    ROOT/'gear_sonic/utils/g1_true23_mpc_student.py'):
    pins[str(p)]=sha(p)
with np.load(fit/'teacher_fit.npz',allow_pickle=False) as norm:
    normalization={key:dict(shape=list(norm[key].shape),dtype=str(norm[key].dtype),array_sha256=hashlib.sha256(norm[key].tobytes()).hexdigest())
        for key in ('feature_mean','feature_std')}
assert all('walk008' not in p.lower() for p in pins)
assert all(sha(local(p))==h for p,h in pins.items())
result=dict(kind='read_only_PICO_walk002_moving_label_coverage_assessment',cases=cases,
    selected_candidate_clips=['pico','walk002'],walk008_held_out_and_not_read=True,
    candidate_new_moving_rows=6847,existing_walk003_moving_rows=3057,potential_total_if_separately_selected=9904,
    candidate_new_phase_rows=dict(acquisition=200,source=6447,return_ramp=200),
    potential_all_phase_rows=dict(acquisition=500,source=8904,return_ramp=500),
    existing_head=dict(ordinary_final_step=65000,onnx_sha256=sha(fit/'student_head.onnx'),
        training_clips=['walk003'],moving_rows_per_branch=1019,branches=['old','query1','query250'],
        current_objective=request['objective'],normalization=normalization),
    feature_contract=dict(head_inputs=1069,goal_features=1023,proprio_features=79,received_goal_slots=8,features_per_slot=118,
        additional_unclipped_BFM_base_minus_default=23,additional_raw_previous_action=23,
        BFM_measured_history=300,BFM_history_concatenated_into_head=False,
        received_offsets=[0,1,2,4,8,16,24,37],prepared_preview_seconds=.74,raw_pose_support_upper_bound_seconds=.76),
    future_residual_definition='actual qualified native PD target minus unclipped frozen BFM baseline target at the SAME actual state/history',
    future_base_goal='original native per-clip goal, horizon8/position1/yaw2; terminal yaw4 excluded from moving rows',
    missing_for_complete_residual_dataset=['Unclipped BFM baseline target/action at every selected actual precontrol state. These are not saved for all rows;6847 selected baseline backward+actor evaluations would need separate authorization.',
        'Full291 per-control integration snapshots are absent in both source traces. Existing collector exports them; do not synthesize qacc_warmstart/ctrl or pretend qpos/qvel alone is full integration. Either approve a state/history-only label schema with root full-trace provenance, or separately authorize exact native replay to capture per-control291.'],
    required_modifications=['Parameterize clip/reference/original29/root qualification and trace field aliases; derive all phases and counts from each frozen timeline.',
        'Rebuild real prefix history from control0, then select250 up to returned_standing exclusively; use normalized actual MPC target previous actions even during initial0..249 for these two traces, unlike query250 BFM prefix.',
        'Check byte equality for float32 previous action, sorted300 flat history and reconstructed named history at every row. Preserve signed zeros and no action clamp.',
        'Load unchanged1069 GoalFeatures/LinearFeatures and native span; infer the unclipped yaw2 baseline on each actual row only if collection is separately selected. Never substitute actual teacher action, a selected fresh-seed rollout, or terminal yaw4 as baseline.',
        'Keep original normalization initially pinned; report out-of-distribution feature ranges and cross-dataset exact/near-target conflicts after separately selected collection. No silent renormalization, averaging, deletion or target smoothing.',
        'Existing compatibility and fit code fixes1019 per branch and nine branch/phase cells. Generalize row metadata and phase-weighted objective only under a separate training selection; plain concatenation would heavily overweight PICO.',
        'Keep source/local quality annotations: PICO source67..69 and69..71 local tracking failures remain archived despite full aggregate qualification. Do not silently filter these rows.',
        'Exclude all original terminal and separate hold targets, and leave walk008 held out. No clip ID, frame ID, wall clock, solver plan or gain input may enter the head.'],
    labels_created=False,feature_rows_created=False,base_inference_calls=0,policy_or_model_constructed=False,physics_steps=0,
    fitting_selected=False,normalization_refitted=False,shared_sources_edited=False,
    input_hashes=pins,python=sys.executable,numpy=np.__version__)
write(OUT/'assessment.json',result)
print(json.dumps(dict(assessment_sha256=sha(OUT/'assessment.json'),new_moving_rows=6847,
    pico_history_rows=cases['pico']['history_rows_byte_exact'],walk002_history_rows=cases['walk002']['history_rows_byte_exact'],
    input_hashes=len(pins),no_labels_or_inference_or_dynamics=True)))
