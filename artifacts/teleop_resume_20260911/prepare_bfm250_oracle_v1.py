"""Freeze one expert transition from independently qualified original BFM entry."""
from pathlib import Path
import ast, hashlib, json, shutil

NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OLD=NEW/'student_actual_oracle_control1_v1'
BASELINE=NEW/'original_bfm_entry250_v1'
DEST=NEW/'bfm_entry250_actual_oracle_v1'
DEST.mkdir(exist_ok=False)
SNAP=DEST/'source_snapshot_v1';SNAP.mkdir()
INPUT=DEST/'inputs';INPUT.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((OLD/'frozen_inputs.json').read_text())
for name,digest in receipt['source_sha256'].items():
    source=OLD/'source_snapshot_v2'/name;assert sha(source)==digest,name
    target=SNAP/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
shutil.copy2(BASELINE/'entry250/trace.npz',INPUT/'baseline250_trace.npz')
assert sha(INPUT/'baseline250_trace.npz')=='8a2013cee1112aedc9a9e290382a7c7801d1c8c49f0e1bd0d978619873de8b41'
import numpy as np
with np.load(INPUT/'baseline250_trace.npz',allow_pickle=False) as a:
    snap=dict(integration=a['final_integration'],integration_state_spec=a['integration_state_spec'],
        qpos=a['physics_qpos'][-1],qvel=a['physics_qvel'][-1],warning_counts=a['physics_warning_counts'][-1],
        warning_lastinfo=a['physics_warning_lastinfo'][-1],expected_time=a['physics_expected_time'][-1],
        actual_time=a['physics_time'][-1],recorded_controls=a['final_recorded_controls'],previous_action=a['final_previous_action'])
    for key in ('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity'):snap['history_'+key]=a['final_history_'+key]
    snap['history_flat']=np.concatenate([snap['history_'+k].reshape(-1) for k in ('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity')])
    np.savez_compressed(INPUT/'precontrol250.npz',**snap)

head=r'''"""One original-BFM prefix250 followed by actual-state expert transition.

Original14 expert source files are unchanged. Only the branch initialization,
prefix accounting and persistent evidence differ from the qualified driver.
"""
from run_actual_student_oracle import *
import run_actual_student_oracle as original_driver

START=250
PREFIX_SHA='8a2013cee1112aedc9a9e290382a7c7801d1c8c49f0e1bd0d978619873de8b41'
PREFIX_MAP={k:k for k in (*CONTROL_FIELDS,'qpos','qvel','physics_qpos','physics_qvel',
    'physics_torque','physics_time','physics_expected_time','range_excess','velocity_ratio','effort_ratio','clock_error')}
PREFIX_MAP.update(physics_actuator_force='physics_actuator_torque',physics_warning_number='physics_warning_counts',
    physics_warning_lastinfo='physics_warning_lastinfo')
PREFIX_MAP.update({'control_history_'+key:'control_history_before_'+key for key in KEYS})

def setup():
    frozen();assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    physics=json.loads((BASE.parent/'original_bfm_entry250_independent_physics_v1/report.json').read_text())
    intent=json.loads((BASE.parent/'original_bfm_entry250_independent_intent_v1/report.json').read_text())
    assert physics['independent_segment_pass'] and physics['requested_segment_completed']
    assert physics['physics_steps']==2500 and all(physics['original_trace_comparison'].values())
    assert PREFIX_SHA in physics['input_hashes'].values()
    assert intent['requested_segment_quiet_pass'] and intent['independent_physical_pass']
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    assert timeline['phases'][0]['control_start']==0 and timeline['phases'][0]['control_stop']==250
    original29=archive(BUNDLE/'walk003/original29.npz')
    audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    planner=Native23Tracker(position_servo_copy(native,kp,kd,effort),c,motion,horizon=30,threads=2,
        all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=0,
        fd_epsilon=1e-6,hard_feasibility=True)
    fresh=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    prefix=archive(BASE/'inputs/baseline250_trace.npz');snap=archive(BASE/'inputs/precontrol250.npz')
    assert sha256(BASE/'inputs/baseline250_trace.npz')==PREFIX_SHA
    assert int(snap['recorded_controls'])==START and int(snap['integration_state_spec'])==8191
    assert int(prefix['final_recorded_controls'])==START and len(prefix['physics_torque'])==2500
    np.testing.assert_array_equal(prefix['global_control'],np.arange(START))
    np.testing.assert_array_equal(prefix['source_frame'],np.arange(START)+11)
    np.testing.assert_array_equal(prefix['physics_substeps'],np.full(START,10))
    np.testing.assert_array_equal(prefix['controller_mode'],np.zeros(START))
    np.testing.assert_array_equal(prefix['delta'],np.zeros((START,23),np.float32))
    np.testing.assert_array_equal(prefix['physics_time'],prefix['physics_expected_time'])
    assert prefix['physics_expected_time'][0]==0.
    expected=0.
    for i in range(2500):
        expected+=.002
        assert expected==prefix['physics_expected_time'][i+1]
    assert expected==float(snap['expected_time'])==float(snap['actual_time'])
    np.testing.assert_array_equal(prefix['final_integration'],snap['integration'])
    # One initialization restores the complete plant, including native warmstart.
    # Forward initializes derived arrays; reapplying integration before any step
    # preserves the recorded clock, controls, warmstart and external-force arrays.
    data=mujoco.MjData(native)
    mujoco.mj_setState(native,data,snap['integration'],8191);mujoco.mj_forward(native,data)
    mujoco.mj_setState(native,data,snap['integration'],8191)
    data.warning.number[:]=snap['warning_counts'];data.warning.lastinfo[:]=snap['warning_lastinfo']
    for key,actual in [('integration',get_state(native,data)),('qpos',data.qpos),('qvel',data.qvel)]:
        np.testing.assert_array_equal(actual,snap[key])
    np.testing.assert_array_equal(data.qpos,prefix['qpos'][-1]);np.testing.assert_array_equal(data.qvel,prefix['qvel'][-1])
    np.testing.assert_array_equal(data.qpos,prefix['physics_qpos'][-1]);np.testing.assert_array_equal(data.qvel,prefix['physics_qvel'][-1])
    np.testing.assert_array_equal(data.warning.number,prefix['physics_warning_counts'][-1])
    np.testing.assert_array_equal(data.warning.lastinfo,prefix['physics_warning_lastinfo'][-1])
    assert data.time==expected and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    reasons,_=assess(data,c,expected);assert not reasons,reasons
    # Rebuild sensor lags only from every recorded actual BFM entry control.
    # No actor, optimizer or physics call occurs during this reconstruction.
    for control in range(START):
        np.testing.assert_array_equal(flat(fresh),prefix['control_history_before'][control])
        np.testing.assert_array_equal(fresh.previous_action,prefix['control_previous_action_before'][control])
        np.testing.assert_array_equal(fresh.previous_action,prefix['previous_action'][control])
        for key in KEYS:
            np.testing.assert_array_equal(fresh.history.data[key],prefix['control_history_before_'+key][control])
        _,terms=fresh._terms(prefix['qpos'][control],prefix['qvel'][control],fresh.previous_action)
        history=fresh.history.before_update(terms)
        np.testing.assert_array_equal(history,prefix['history'][control])
        fresh.previous_action=prefix['action'][control].copy()
    np.testing.assert_array_equal(flat(fresh),snap['history_flat'])
    for key in KEYS:
        np.testing.assert_array_equal(fresh.history.data[key],snap['history_'+key])
        np.testing.assert_array_equal(fresh.history.data[key],prefix['final_history_'+key])
    np.testing.assert_array_equal(fresh.previous_action,snap['previous_action'])
    np.testing.assert_array_equal(fresh.previous_action,prefix['final_previous_action'])
    fresh.recorded_controls=START
    fresh.actual_action_max_abs=float(np.max(np.abs(prefix['action'])))
    fresh.actual_action_components_outside_five=int(np.sum(np.abs(prefix['action'])>5))
    recorded=archive(RECORDED/'trace.npz')['target']
    assert recorded.shape==(1569,23) and np.isfinite(recorded).all()
    np.testing.assert_array_equal(recorded,np.clip(recorded,planner.lo,planner.hi))
    request=dict(kind='original_BFM_entry250_plus_actual_expert_transition',clip='walk003',
        initial_global_control=START,BFM_prefix_controls=START,requested_controls=LIFECYCLE,
        requested_branch_controls=LIFECYCLE-START,requested_source_controls=819,
        MPC_global_controls=[250,1269],MPC_controls=1019,eligible_future_label_controls=[250,1269],
        terminal_BFM_global_controls=[1269,1569],separate_extension_controls=250,
        horizon=30,commit=5,iterations=10,batch_threads=2,BLAS_threads=1,feedback_correction_clip_rad=.1,
        finite_difference_epsilon=1e-6,all_joint_margin=.05,all_joint_weight=2000,relative_foot_weight=0,
        hard_feasibility=True,guided_then_K0_restoration=True,private_single_lane_optimization=False,
        initial_clock=expected,clock_contract='repeated+.002 from zero through saved2500substeps, then continued without resynchronization',
        initial_history='exact raw originalBFM action and named four-lag actual sensor history from qualified prefix250',
        MPC_history='first real MPC250 pushes preceding BFM249 raw action once; subsequent previous action normalized actually applied MPC target, never clipped to+-5',
        terminal_history='same accumulated buffer; raw BFM actor*5 next action, no record_control normalization',
        original_recorded_seed='bfm_walk003_arms_v3 targets only',fresh_BFM_goals='originalnative position1/yaw2/h8',
        terminal_BFM_goals='originalnative position1/yaw4/h8',optimization_goals='v4floor native reference; original full timing',
        model_manifest=manifest,motion_override=override,fresh_BFM_identity=fresh.identity(),
        received_goal_preview_seconds=.74,conservative_raw_pose_support_seconds=.76,
        initial_snapshot_sha256=sha256(BASE/'inputs/precontrol250.npz'),BFM_prefix_sha256=PREFIX_SHA,
        frozen_sources_sha256=sha256(BASE/'frozen_inputs.json'),controller_mode_map={'0':'recorded actual originalBFM entry','1':'fresh expert MPC','2':'terminal originalBFM yaw4'},
        physically_copied_later_student_or_teacher_states=False,physical_statewrites_after_initialization=0,
        initial_warnings_restored_separately=True,labels_admissible=False,DAgger_fitting_authorized=False,
        learned_head_loaded=False,labels_gate='complete1319 remaining controls/all819source plus both quiet holds independently qualify',hardware_authorized=False)
    return dict(native=native,c=c,original=original,motion=motion,original29=original29,audit=audit,
        planner=planner,fresh=fresh,data=data,snapshot=snap,prefix=prefix,expected=expected,recorded=recorded,request=request)

def initial_trace(s):
    return {target:list(s['prefix'][source].copy()) for target,source in PREFIX_MAP.items()}

def prefix_equality(s,trace):
    checks={}
    for target,source in PREFIX_MAP.items():
        original=s['prefix'][source]
        actual=np.asarray(trace[target])[:len(original)]
        exact=bool(np.array_equal(actual,original));checks[target]=dict(source_field=source,records=len(original),bitexact=exact)
        assert exact,('preserved_BFM250_prefix_changed',target)
    write(BASE/'preserved_baseline250_prefix_equality.json',dict(passed=True,checks=checks,
        full_original_prefix_trace_sha256=PREFIX_SHA,all_original_fields_preserved_in_bound_input=True,
        original_control_features_still_available_in_input=True,initial_BFM_controls=250,expert_labels_exclude_prefix=True))

def save_trace(path,s,trace,**extra):
    if len(trace['global_control']) and trace['global_control'][0]==0:prefix_equality(s,trace)
    return original_driver.save_trace(path,s,trace,**extra)

def assert_initial_actual_unchanged(s):
    np.testing.assert_array_equal(get_state(s['native'],s['data']),s['snapshot']['integration'])
    np.testing.assert_array_equal(flat(s['fresh']),s['snapshot']['history_flat'])
    np.testing.assert_array_equal(s['fresh'].previous_action,s['snapshot']['previous_action'])
    np.testing.assert_array_equal(s['data'].warning.number,s['snapshot']['warning_counts'])
    np.testing.assert_array_equal(s['data'].warning.lastinfo,s['snapshot']['warning_lastinfo'])
    assert s['fresh'].recorded_controls==START and s['expected']==float(s['snapshot']['expected_time'])==s['data'].time

def certify_initial(s):
    out=BASE/'initial_seed';out.mkdir(exist_ok=False);write(out/'request.json',s['request'])
    s['planner'].window(START+10);warm=s['planner'].target_reference(np.arange(30))
    best,log,_=choose_seed(s,START,warm,out,certify_full=True)
    assert_initial_actual_unchanged(s)
    result=dict(initial_H30_feasible_seed_exists=best is not None,selection=log,physical_branch_started=False,
        labels_admissible=False,request_sha256=sha256(out/'request.json'),actual_plant_and_history_unchanged=True,
        first_global_control=250,planner_window_frame=260,first_output_frame=261)
    if best is not None:
        selected=dict(cost=np.asarray(best[0]),candidate=np.asarray(best[1]),nominal_states=best[2],targets=best[3])
        atomic_trace(out/'selected.npz',{},**selected);result['selected_sha256']=sha256(out/'selected.npz')
    else:selected=None
    write(out/'report.json',result);print(json.dumps(plain(result)),flush=True)
    return result,selected

'''

original=(SNAP/'run_actual_student_oracle.py').read_text()
run=original[original.index('def run_branch():'):original.index("\n\nif __name__=='__main__':")]
run=run.replace("def run_branch():\n    initial=json.loads((BASE/'initial_seed/report.json').read_text())", "def run_branch(s,initial,selected):")
run=run.replace(";(out/'plans').mkdir();s=setup();write", ";(out/'plans').mkdir();write")
run=run.replace('failure=None;control=1;warm=None;last_commit=0','failure=None;control=START;warm=None;last_commit=0')
run=run.replace("selected=archive(BASE/'initial_seed/selected.npz');started=time.perf_counter();next_checkpoint=100", "started=time.perf_counter();next_checkpoint=START+100\n    save_trace(out/'trace.partial.npz',s,trace,checkpoint_metadata=np.asarray(json.dumps(dict(completed_controls=START,requested_controls=LIFECYCLE,failure=None,labels_admissible=False))))\n    write(out/'plans.json',plans)")
run=run.replace('if control==1:','if control==START:')
run=run.replace('            count=min(5,SWITCH-control);last_commit=count;warm=targets.copy()', '            if control==START:assert_initial_actual_unchanged(s)\n            count=min(5,SWITCH-control);last_commit=count;warm=targets.copy()')
run=run.replace("'explicit_real_student_control0_prefix_plus_fresh_expert_branch'", "'actual_original_BFM250_prefix_plus_fresh_expert_transition'")
run=run.replace("actual_expert_controls=len(trace['target'])-1,student_prefix_controls=1", "actual_expert_controls=int(np.sum(np.asarray(trace['controller_mode'])==1)),BFM_prefix_controls=START")
run=run.replace("# Branch-only evidence drops just the duplicate prefix boundary. The combined\n    # trace above is explicitly declared counterfactual prefix+branch evidence.", "# Branch-only evidence starts at the exact baseline250 boundary.\n    # Combined evidence retains the original actual BFM entry prefix unchanged.")
run=run.replace("v[10:] if k.startswith('physics_')", "v[START*10:] if k.startswith('physics_')")
run=run.replace("else v[1:]", "else v[START:]")
tail=r'''

def main():
    for name in ('nominal','initial_seed','post_lifecycle_hold_5s','initial_restore_preflight.json'):
        assert not (BASE/name).exists(),'Existing expert branch attempt must remain preserved: '+name
    s=setup()
    assert_initial_actual_unchanged(s)
    prefix_equality(s,initial_trace(s))
    write(BASE/'initial_restore_preflight.json',dict(passed=True,initial_global_control=START,initial_clock=s['expected'],
        full_integration_exact=True,all250_named_and_flat_sensor_history_entries_exact=True,
        raw_previous_action_exact=True,independent_repeated_clock_exact=True,
        actor_inference_calls=0,optimizer_calls=0,physics_steps_executed=0,
        BFM_prefix_sha256=PREFIX_SHA,snapshot_sha256=sha256(BASE/'inputs/precontrol250.npz'),
        labels_admissible=False,hardware_authorized=False))
    initial,selected=certify_initial(s)
    if selected is None:
        write(BASE/'outcome.json',dict(physical_branch_started=False,initial_seed=initial,
            labels_admissible=False,DAgger_fitting_launched=False,hardware_authorized=False))
        return
    run_branch(s,initial,selected)

if __name__=='__main__':main()
'''
adapter=head+run+tail
ast.parse(adapter)
(SNAP/'run_bfm250_actual_oracle.py').write_text(adapter)
# Retain only physical/runtime/reference inputs from the previous expert receipt.
# No previous student's checkpoint, head, query1 fixture or partial run is loaded.
inputs={name:digest for name,digest in receipt['input_sha256'].items()
    if '/sonic23_teleop_resume_20260911/' not in name}
for p in (INPUT/'baseline250_trace.npz',INPUT/'precontrol250.npz',
          BASELINE/'entry250/report.json',BASELINE/'entry250/prefix100_parity.json',
          BASELINE/'frozen_inputs_v2.json',BASELINE/'prelaunch_clearance.json',
          NEW/'original_bfm_entry250_independent_physics_v1/report.json',
          NEW/'original_bfm_entry250_independent_intent_v1/report.json',OLD/'frozen_inputs.json'):
    inputs[p.as_posix()]=sha(p)
frozen=dict(kind='one_actual_original_BFM250_to_expert_transition',source_sha256={p.relative_to(SNAP).as_posix():sha(p) for p in sorted(SNAP.rglob('*')) if p.is_file()},
    input_sha256=inputs,initial_global_control=250,preserved_BFM_controls=250,requested_combined_controls=1569,
    requested_branch_controls=1319,expert_control_interval=[250,1269],future_eligible_expert_labels=1019,
    requested_source_controls=819,terminal_BFM_start=1269,separate_hold250=True,horizon=30,iterations=10,
    commit=5,batch_threads=2,BLAS_threads=1,all_joint_margin=.05,all_joint_weight=2000,relative_foot_weight=0,
    hard_feasibility=True,guided_then_K0_restoration=True,original14_sources_byte_exact=True,
    physical_statewrites_after_initialization=0,no_previous_query_or_head_loaded=True,
    labels_admissible=False,fit_authorized=False,hardware_authorized=False)
(DEST/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
(DEST/'README.md').write_text('One actual expert transition begins at global250 from the independently qualified original BFM entry. The entire250-control baseline trace is copied byte-exactly and bound; all common combined controls/states/torques/clocks/named history are asserted exact on every checkpoint. Extra baseline feature arrays remain preserved in the bound original input.\n\nOne setup restores full native integration/warnings and reconstructs all250 sensor-history entries without actor/optimizer/physics calls. First H30 initial/recorded/fresh seeds are certified from that same actual state, with candidate certificate arrays saved. Actual plant/history equality is checked again after seed certification and first iLQR, before the first real control. No second setup or query occurs.\n\nOriginal14 expert v2 files remain byte-exact. MPC controls250..1268 retain H30/10iter/commit5/2Batchthreads/1BLASthread, margin .05/weight2000, relativefoot0, guided thenK0 restoration, full native500Hz strict gates, original recorded/freshBFM seeds. Lastplan1265 commits4. TerminalBFMyaw4 begins1269; combined1569 and continuous separate250 hold unchanged. No learned head, label collection or fitting is authorized. Eligible future expert labels250..1268 only, pending full independent qualification.\n')
print(json.dumps(dict(directory=str(DEST),receipt_sha256=sha(DEST/'frozen_inputs.json'),adapter_sha256=sha(SNAP/'run_bfm250_actual_oracle.py'),sources=len(frozen['source_sha256']),inputs=len(inputs))))
