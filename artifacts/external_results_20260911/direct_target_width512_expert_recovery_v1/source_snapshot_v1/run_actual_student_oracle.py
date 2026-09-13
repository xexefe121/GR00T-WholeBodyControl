"""One frozen actual-student-state MPC continuation; labels remain pending audit."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import atomic_trace
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker,load_native_bundle,load_motion_override,sha256
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy,preview_native_control
from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr,NoFeasiblePlan
from gear_sonic.utils.g1_true23_mjbatch_restoration import restore_feasible_seed
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed,BFMSeedRolloutError
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory
from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from substep_recorder import record_native_substep
from oracle_metrics import get_state,assess,source_metrics
from quiet_metrics import standing_windows,quiet_diagnostic
from terminal_yaw4_goal import terminal_goal_yaw4

BASE=Path(__file__).resolve().parent.parent
ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
RECORDED=ROOT/'artifacts/teleop_six_hour_20260910/bfm_walk003_arms_v3'
ONNX=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
DEPS=Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
TEACHER=TASK/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
KEYS=('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity')
SWITCH,LIFECYCLE,EXTENSION=1269,1569,250


def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write(path,obj):path.write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+'\n')
def flat(seed):return np.concatenate([seed.history.data[k].reshape(-1) for k in sorted(seed.history.data)])


def frozen():
    r=json.loads((BASE/'frozen_inputs.json').read_text())
    for name,expected in r['source_sha256'].items():assert sha256(Path(__file__).parent/name)==expected,name
    for name,expected in r['input_sha256'].items():
        if len(name)>2 and name[1]==':':name='/mnt/'+name[0].lower()+name[2:]
        assert sha256(Path(name))==expected,name
    return r


def setup():
    frozen();assert mujoco.__version__=='3.2.3'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz')
    audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    planner=Native23Tracker(position_servo_copy(native,kp,kd,effort),c,motion,horizon=30,threads=2,
        all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=0,
        fd_epsilon=1e-6,hard_feasibility=True)
    fresh=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    snap=archive(BASE/'inputs/actual_student_control1.npz');prefix=archive(BASE/'inputs/student_control0_prefix.npz')
    assert int(snap['recorded_controls'])==1 and int(snap['integration_state_spec'])==8191
    assert prefix['physics_substeps'].tolist()==[10]
    data=mujoco.MjData(native);spec=int(snap['integration_state_spec'])
    mujoco.mj_setState(native,data,snap['integration'],spec);mujoco.mj_forward(native,data)
    mujoco.mj_setState(native,data,snap['integration'],spec)
    data.warning.number[:]=snap['warning_counts'];data.warning.lastinfo[:]=snap['warning_lastinfo']
    for name,actual in [('integration',get_state(native,data)),('qpos',data.qpos),('qvel',data.qvel)]:
        np.testing.assert_array_equal(actual,snap[name])
    np.testing.assert_array_equal(prefix['final_integration'],snap['integration'])
    np.testing.assert_array_equal(prefix['physics_qpos'][-1],data.qpos)
    np.testing.assert_array_equal(prefix['physics_qvel'][-1],data.qvel)
    np.testing.assert_array_equal(prefix['physics_warning_counts'][-1],data.warning.number)
    np.testing.assert_array_equal(prefix['physics_warning_lastinfo'][-1],data.warning.lastinfo)
    expected=float(snap['expected_time']);assert expected==float(data.time)==float(prefix['physics_expected_time'][-1])
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    reasons,_=assess(data,c,expected);assert not reasons,reasons
    # Rebuild only the real student control0 sensor history. Its effective action
    # is preserved; record_control would incorrectly replace that convention.
    _,terms=fresh._terms(prefix['qpos'][0],prefix['qvel'][0],prefix['previous_action'][0])
    fresh.history.before_update(terms)
    np.testing.assert_array_equal(flat(fresh),snap['history_flat'])
    for key in KEYS:
        np.testing.assert_array_equal(fresh.history.data[key],snap['history_'+key])
    fresh.previous_action=snap['previous_action'].copy()
    np.testing.assert_array_equal(fresh.previous_action,prefix['action'][0])
    fresh.recorded_controls=1
    fresh.actual_action_max_abs=float(np.max(np.abs(prefix['action'])))
    fresh.actual_action_components_outside_five=int(np.sum(np.abs(prefix['action'])>5))
    recorded=archive(RECORDED/'trace.npz')['target']
    assert recorded.shape==(1569,23) and np.isfinite(recorded).all()
    np.testing.assert_array_equal(recorded,np.clip(recorded,planner.lo,planner.hi))
    request=dict(kind='one_final20000_actual_student_control1_expert_branch',clip='walk003',
        initial_global_control=1,student_prefix_controls=1,requested_controls=LIFECYCLE,
        requested_branch_controls=LIFECYCLE-1,requested_source_controls=819,
        MPC_global_controls=[1,1269],terminal_BFM_global_controls=[1269,1569],separate_extension_controls=250,
        horizon=30,commit=5,iterations=10,batch_threads=2,BLAS_threads=1,feedback_correction_clip_rad=.1,
        finite_difference_epsilon=1e-6,all_joint_margin=.05,all_joint_weight=2000,relative_foot_weight=0,
        hard_feasibility=True,guided_then_K0_restoration=True,private_single_lane_optimization=False,
        initial_clock=expected,clock_contract='saved independently accumulated prefix clock, then repeated+.002; no resynchronization',
        initial_history='final20000 student combined preclip previous action and named four-lag sensor history',
        MPC_history='first real MPC control pushes preceding student action once; subsequent previous action normalized actually applied MPC target, never clipped to±5',
        terminal_history='same accumulated buffer; raw BFM actor*5 next action, no record_control normalization',
        original_recorded_seed='bfm_walk003_arms_v3 targets only',fresh_BFM_goals='originalnative position1/yaw2/h8',
        terminal_BFM_goals='originalnative position1/yaw4/h8',optimization_goals='v4floor native reference; original full timing',
        model_manifest=manifest,motion_override=override,fresh_BFM_identity=fresh.identity(),
        received_goal_preview_seconds=.74,conservative_raw_pose_support_seconds=.76,
        initial_snapshot_sha256=sha256(BASE/'inputs/actual_student_control1.npz'),
        student_prefix_sha256=sha256(BASE/'inputs/student_control0_prefix.npz'),
        selection_receipt_sha256=sha256(BASE/'inputs/selection_receipt.json'),
        frozen_sources_sha256=sha256(BASE/'frozen_inputs.json'),
        physically_copied_later_student_or_teacher_states=False,physical_statewrites_after_initialization=0,
        initial_warnings_restored_separately=True,labels_admissible=False,DAgger_fitting_authorized=False,
        labels_gate='full remaining lifecycle source/return plus both quiet holds independently qualify',hardware_authorized=False)
    return dict(native=native,c=c,original=original,motion=motion,original29=original29,audit=audit,
        planner=planner,fresh=fresh,data=data,snapshot=snap,prefix=prefix,expected=expected,recorded=recorded,request=request)


def choose_seed(s,control,warm,out,*,certify_full=False,restorer=None):
    p=s['planner'];data=s['data'];fresh=s['fresh'];actual=np.r_[data.qpos,data.qvel]
    p.window(control+10);warm=np.asarray(warm).copy()
    sequences=[('shifted_or_initial_reference',warm),('recorded_bfm',s['recorded'][np.minimum(control+np.arange(30),1568)])]
    log=[]
    try:
        proposal,diagnostics=fresh.propose(control,data.qpos,data.qvel,horizon=30)
        sequences.append(('fresh_bfm',proposal));log.append(dict(candidate='fresh_BFM_proposal',diagnostics=diagnostics))
    except BFMSeedRolloutError as exc:
        log.append(dict(candidate='fresh_bfm',rejected=True,error=str(exc),diagnostics=exc.diagnostics))
    best=None
    for name,targets in sequences:
        states,bounded,cost=p.rollout(actual,targets)
        info=dict(candidate=name,cost=float(cost[0]),nominal_feasibility=copy.deepcopy(p.last_rollout_feasibility))
        accepted=bool(np.isfinite(cost[0]))
        if certify_full:
            physical,trace=inspect_native_segment(s['native'],data,targets,s['c'])
            info['actual_full_integration_horizon']=physical
            atomic_trace(out/(name+'.npz'),trace,targets=targets)
            info['actual_trace_sha256']=sha256(out/(name+'.npz'))
            accepted=accepted and physical['feasible']
        info['eligible']=accepted;log.append(info)
        if accepted and (best is None or cost[0]<best[0]):best=(float(cost[0]),name,states[:,0].copy(),bounded[:,0].copy())
    restoration=None
    if best is None:
        def save_proposal(mode,targets):
            path=out/('restoration_%05d_%s.npz'%(control,mode))
            atomic_trace(path,{},targets=targets,initial_integration=get_state(s['native'],data),
                integration_state_spec=np.asarray(8191),expected_time=np.asarray(s['expected']),
                previous_action=fresh.previous_action,**{'history_'+k:v for k,v in fresh.history.data.items()})
            return dict(proposal_file=path.name,proposal_sha256=sha256(path))
        best,restoration,restorer=restore_feasible_seed(p,s['native'],data,s['c'],s['motion'],warm,control+10,
            restorer=restorer,threads=2,retry_zero_feedback=True,save_proposal=save_proposal)
    return best,dict(control=control,candidates=log,restoration=restoration,selected=None if best is None else best[1]),restorer


def certify_initial():
    out=BASE/'initial_seed';out.mkdir(exist_ok=False);s=setup();write(out/'request.json',s['request'])
    s['planner'].window(11);warm=s['planner'].target_reference(np.arange(30))
    best,log,_=choose_seed(s,1,warm,out,certify_full=True)
    result=dict(initial_H30_feasible_seed_exists=best is not None,selection=log,physical_branch_started=False,
        labels_admissible=False,request_sha256=sha256(out/'request.json'))
    if best is not None:
        atomic_trace(out/'selected.npz',{},cost=np.asarray(best[0]),candidate=np.asarray(best[1]),nominal_states=best[2],targets=best[3])
        result['selected_sha256']=sha256(out/'selected.npz')
    write(out/'report.json',result);print(json.dumps(plain(result)),flush=True)


CONTROL_FIELDS=('target','source_frame','global_control','controller_mode','joint_error','root_error','physics_substeps',
    'control_integration_before','control_history_before','control_previous_action_before','previous_action','action')
PHYSICS_FIELDS=('physics_qpos','physics_qvel','physics_torque','physics_actuator_force','physics_time',
    'physics_expected_time','physics_warning_number','physics_warning_lastinfo')


def initial_trace(s):
    a=s['prefix'];trace={k:list(a[k].copy()) for k in CONTROL_FIELDS if k in a}
    trace['controller_mode']=[0]
    for k in ['qpos','qvel','physics_qpos','physics_qvel','physics_torque','physics_time','physics_expected_time']:
        trace[k]=list(a[k].copy())
    for target,source in [('physics_actuator_force','physics_actuator_torque'),('physics_warning_number','physics_warning_counts'),('physics_warning_lastinfo','physics_warning_lastinfo')]:
        trace[target]=list(a[source].copy())
    for k in ['range_excess','velocity_ratio','effort_ratio','clock_error']:trace[k]=list(a[k].copy())
    for key in KEYS:trace['control_history_'+key]=[np.zeros_like(s['snapshot']['history_'+key])]
    return trace


def empty_trace(s):
    t={k:[] for k in (*CONTROL_FIELDS,*PHYSICS_FIELDS,'qpos','qvel','range_excess','velocity_ratio','effort_ratio','clock_error')}
    for key in KEYS:t['control_history_'+key]=[]
    for key in ['qpos','physics_qpos']:t[key]=[s['data'].qpos.copy()]
    for key in ['qvel','physics_qvel']:t[key]=[s['data'].qvel.copy()]
    t['physics_time']=[float(s['data'].time)];t['physics_expected_time']=[s['expected']]
    t['physics_warning_number']=[s['data'].warning.number.copy()];t['physics_warning_lastinfo']=[s['data'].warning.lastinfo.copy()]
    return t


def apply_control(s,trace,control,target,terminal=False):
    native,data,fresh=s['native'],s['data'],s['fresh'];c=s['c']
    witness,predicted,forces=preview_native_control(native,data,target,c,s['planner'].feasibility)
    if not witness['feasible']:return dict(kind='imminent_control_infeasible',control=control,witness=witness,
        rejected_target=target.copy(),executed_rejected_controls=0)
    before=get_state(native,data);hist=flat(fresh);previous=fresh.previous_action.copy()
    names={k:fresh.history.data[k].copy() for k in KEYS}
    if not terminal:fresh.record_control(control,data.qpos,data.qvel,target)
    else:
        before,hist,previous,names=s.pop('terminal_before')
        after_history,after_action=s.pop('terminal_after')
        fresh.history=after_history;fresh.previous_action=after_action;fresh.recorded_controls+=1
    count=0;failure=None
    for sub in range(10):
        s['expected'],failure=record_native_substep(native,data,target,c,s['planner'].feasibility,trace,
            s['expected'],predicted[sub+1],forces[sub])
        lengths=[len(trace[k]) for k in ('physics_time','physics_expected_time','physics_qpos','physics_qvel')]
        assert lengths==[len(trace['physics_torque'])+1]*4,('physics_clock_ledger_lengths',lengths)
        count+=1;_,metrics=assess(data,c,s['expected'])
        for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):trace[key].append(metrics[key])
        if failure:
            failure.update(global_control=control,substep=sub+1)
            break
    mujoco.mj_kinematics(native,data)
    frame=min(control+11,len(s['motion']['joint_pos'])-1)
    values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),target=target.copy(),source_frame=frame,global_control=control,
        controller_mode=2 if terminal else 1,physics_substeps=count,
        joint_error=data.qpos[7:]-s['motion']['joint_pos'][frame],root_error=data.qpos[:3]-s['motion']['body_pos_w'][frame,0],
        control_integration_before=before,control_history_before=hist,control_previous_action_before=previous,
        previous_action=previous,action=fresh.previous_action.copy(),**{'control_history_'+k:v for k,v in names.items()})
    for k,v in values.items():trace[k].append(v)
    return failure


def terminal_proposal(s,control):
    seed=s['fresh'];data=s['data'];assert seed.recorded_controls==control
    before=(get_state(s['native'],data),flat(seed),seed.previous_action.copy(),{k:seed.history.data[k].copy() for k in KEYS})
    sensed,terms=seed._terms(data.qpos,data.qvel,seed.previous_action)
    local_history=copy.deepcopy(seed.history)
    history=local_history.before_update(terms);goal=terminal_goal_yaw4(seed,control+11,data.qpos)
    raw=seed.sessions['actor'].run(None,dict(state=sensed[None],last_action=seed.previous_action[None],history=history[None],z=goal))[0][0]*5
    c=seed.contract;target=np.clip(c['default_q']+raw*.25*c['training_effort']/c['kp'],s['planner'].lo,s['planner'].hi)
    if not np.isfinite(raw).all() or not np.isfinite(target).all():raise ValueError('nonfinite terminal BFM output')
    s['terminal_before']=before;s['terminal_after']=(local_history,raw.copy())
    return target


def arrays_of(trace):return {k:np.asarray(v) for k,v in trace.items()}


def save_trace(path,s,trace,**extra):
    initial_vector=extra.pop('initial_integration_override',s['prefix']['initial_integration'])
    seed=s['fresh'];atomic_trace(path,trace,initial_integration=initial_vector,
        branch_initial_integration=s['snapshot']['integration'],integration_state_spec=np.asarray(8191),
        final_integration=get_state(s['native'],s['data']),final_recorded_controls=np.asarray(seed.recorded_controls),
        final_previous_action=seed.previous_action,**{'final_history_'+k:v for k,v in seed.history.data.items()},**extra)


def report_segment(s,trace,requested,failure,start_time,kind):
    a=arrays_of(trace);counts=np.asarray(a['physics_substeps']);complete=int(np.sum(counts==10))
    full=len(counts)==requested and complete==requested and failure is None
    quiet=None
    if full:quiet=quiet_diagnostic(standing_windows(a,s['motion'],s['original'],s['original29']),True)
    return dict(kind=kind,clip='walk003',failure=failure,requested_controls=requested,completed_controls=len(counts),
        completed_full_controls=complete,partial_substeps=len(a['physics_torque'])%10,physics_steps=len(a['physics_torque']),
        full_segment_completed=full,probe_completed=full,simulated_seconds=float(s['data'].time)-start_time,
        simulation_start_time=start_time,simulation_end_time=float(s['data'].time),
        range_excess_max=float(np.max(a['range_excess'])) if len(a['range_excess']) else 0.,
        velocity_ratio_max=float(np.max(a['velocity_ratio'])) if len(a['velocity_ratio']) else 0.,
        effort_ratio_max=float(np.max(a['effort_ratio'])) if len(a['effort_ratio']) else 0.,
        engine_warning_counts=s['data'].warning.number.tolist(),engine_warning_lastinfo=s['data'].warning.lastinfo.tolist(),
        clock_error_max=float(np.max(a['clock_error'])) if len(a['clock_error']) else 0.,
        source_metrics=source_metrics(s['native'],a,s['motion'],s['original29'],s['audit']),
        quiet_standing_diagnostic=quiet,motion_override=s['request']['motion_override'],
        labels_admissible=False,independent_source_and_quiet_audit_pending=True,hardware_authorized=False)


def run_branch():
    initial=json.loads((BASE/'initial_seed/report.json').read_text())
    assert initial['initial_H30_feasible_seed_exists']
    assert sha256(BASE/'initial_seed/selected.npz')==initial['selected_sha256']
    out=BASE/'nominal';out.mkdir(exist_ok=False);(out/'plans').mkdir();s=setup();write(out/'request.json',s['request'])
    trace=initial_trace(s);plans=[];restorer=None;failure=None;control=1;warm=None;last_commit=0
    selected=archive(BASE/'initial_seed/selected.npz');started=time.perf_counter();next_checkpoint=100
    try:
        while control<SWITCH:
            if (BASE/'STOP_REQUESTED').exists():failure=dict(kind='requested_stop',control=control);break
            p=s['planner'];p.window(control+10);tick=time.perf_counter()
            if control==1:
                warm=selected['targets'].copy();xs,us,cost=p.rollout(np.r_[s['data'].qpos,s['data'].qvel],warm)
                assert np.isfinite(cost[0]);np.testing.assert_array_equal(xs[:,0],selected['nominal_states'])
                assert float(cost[0])==float(selected['cost'])
                best=(float(cost[0]),str(selected['candidate']),xs[:,0],warm);selection=initial['selection']
            else:
                warm=np.concatenate((warm[last_commit:],p.target_reference(np.arange(30-last_commit,30))))
                best,selection,restorer=choose_seed(s,control,warm,out,restorer=restorer)
            if best is None:failure=dict(kind='no_feasible_seed',control=control,selection=selection);break
            states,targets,gains,cost=ilqr(p,np.r_[s['data'].qpos,s['data'].qvel],best[3].copy(),iters=10,
                initial_rollout=(best[2],best[0]))
            if not np.isfinite(cost) or not np.isfinite(targets).all() or not np.isfinite(gains).all():
                failure=dict(kind='invalid_plan',control=control);break
            count=min(5,SWITCH-control);last_commit=count;warm=targets.copy()
            plan=dict(control=control,cost=float(cost),selection=selection,solver_feasibility=copy.deepcopy(p.last_solve_feasibility),
                requested_commit=count,executed_controls=0,solve_seconds=time.perf_counter()-tick)
            plans.append(plan)
            atomic_trace(out/'plans'/('plan_%05d.npz'%control),{},nominal_states=states,targets=targets,gains=gains)
            for local in range(count):
                actual=np.r_[s['data'].qpos,s['data'].qvel]
                correction=np.clip(gains[local]@p.difference(states[local:local+1],actual[None])[0],-.1,.1)
                target=np.clip(targets[local]+correction,p.lo,p.hi)
                before=len(trace['target']);failure=apply_control(s,trace,control,target)
                if len(trace['target'])>before:control+=1;plan['executed_controls']+=1
                if failure:break
            if control>=next_checkpoint or failure:
                metadata=dict(completed_controls=control,requested_controls=LIFECYCLE,failure=failure,labels_admissible=False)
                save_trace(out/'trace.partial.npz',s,trace,checkpoint_metadata=np.asarray(json.dumps(plain(metadata))))
                write(out/'plans.json',plans);next_checkpoint=control+100
            if control%25<5 or failure:
                print(json.dumps(plain(dict(control=control,cost=cost,height=s['data'].qpos[2],solve_seconds=plan['solve_seconds'],failure=failure))),flush=True)
            if failure:break
        if failure is None:
            assert control==SWITCH and s['fresh'].recorded_controls==SWITCH
            while control<LIFECYCLE:
                target=terminal_proposal(s,control);failure=apply_control(s,trace,control,target,terminal=True)
                if failure:break
                control+=1
    except Exception as exc:
        failure=dict(kind='oracle_runtime_fault',control=control,type=type(exc).__name__,message=str(exc))
        if isinstance(exc,NoFeasiblePlan):failure['solver_feasibility_diagnostics']=exc.diagnostics
    save_trace(out/'trace.npz',s,trace);write(out/'plans.json',plans)
    result=report_segment(s,trace,LIFECYCLE,failure,0.,'explicit_real_student_control0_prefix_plus_fresh_expert_branch')
    result.update(trace_sha256=sha256(out/'trace.npz'),request_sha256=sha256(out/'request.json'),elapsed_seconds=time.perf_counter()-started,
        actual_expert_controls=len(trace['target'])-1,student_prefix_controls=1,branch_initial_time=float(s['snapshot']['actual_time']))
    write(out/'report.json',result);print(json.dumps(plain(result)),flush=True)
    # Branch-only evidence drops just the duplicate prefix boundary. The combined
    # trace above is explicitly declared counterfactual prefix+branch evidence.
    a=arrays_of(trace);branch={}
    for k,v in a.items():
        branch[k]=v[10:] if k.startswith('physics_') and k!='physics_substeps' or k in ('range_excess','velocity_ratio','effort_ratio','clock_error') else v[1:]
    atomic_trace(out/'branch_only.npz',branch,initial_integration=s['snapshot']['integration'],integration_state_spec=np.asarray(8191))
    ext=BASE/'post_lifecycle_hold_5s';ext.mkdir(exist_ok=False)
    if result['full_segment_completed']:
        extended=empty_trace(s);start_time=float(s['data'].time);extension_initial=get_state(s['native'],s['data']);failure=None
        write(ext/'request.json',dict(s['request'],segment='separate_continuous5s_hold',initial_global_control=LIFECYCLE,requested_controls=EXTENSION))
        try:
            for control in range(LIFECYCLE,LIFECYCLE+EXTENSION):
                target=terminal_proposal(s,control);failure=apply_control(s,extended,control,target,terminal=True)
                if failure:break
        except Exception as exc:failure=dict(kind='terminal_runtime_fault',control=control,type=type(exc).__name__,message=str(exc))
        save_trace(ext/'trace.npz',s,extended,initial_integration_override=extension_initial)
        extended_result=report_segment(s,extended,EXTENSION,failure,start_time,'separate_continuous5s_hold')
        extended_result.update(trace_sha256=sha256(ext/'trace.npz'),request_sha256=sha256(ext/'request.json'))
    else:extended_result=dict(requested_controls=250,attempted_controls=0,full_segment_completed=False,labels_admissible=False,not_run_reason='original lifecycle did not complete; no reset or skipped source')
    write(ext/'report.json',extended_result)
    write(BASE/'outcome.json',dict(nominal=result,extension=extended_result,labels_admissible=False,DAgger_fitting_launched=False,
        independent_full_source_return_and_both_quiet_audits_required=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['seed','continue'],required=True)
    args=parser.parse_args();certify_initial() if args.stage=='seed' else run_branch()
