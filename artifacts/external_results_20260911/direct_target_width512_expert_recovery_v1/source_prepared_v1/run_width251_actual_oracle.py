"""Actual BFM250 + learned250 prefix, then one fresh expert branch from251.

All14 qualified expert modules are unchanged. This adapter preserves the actual
prefix, restores full291 and incoming causal state, and counts selected work.
"""
from run_actual_student_oracle import *
import run_actual_student_oracle as original_driver

from recovery_contract import START, SOURCE_TRACE_SHA, protocol
from recovery_admission import admit
from counted_work import WorkLedger, Hooks

PREFIX_SHA=SOURCE_TRACE_SHA
WORK=None
HOOKS=None
ACTIVE_STATE=None
PREFIX_MAP={k:k for k in (*CONTROL_FIELDS,'qpos','qvel','physics_qpos','physics_qvel',
    'physics_torque','physics_time','physics_expected_time','range_excess','velocity_ratio','effort_ratio','clock_error')}
PREFIX_MAP.update(physics_actuator_force='physics_actuator_torque',physics_warning_number='physics_warning_counts',
    physics_warning_lastinfo='physics_warning_lastinfo')
PREFIX_MAP.update({'control_history_'+key:'control_history_before_'+key for key in KEYS})

def setup():
    frozen();assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    global ACTIVE_STATE
    selection=json.loads((BASE/'inputs/selection_receipt.json').read_text())
    assert selection['passed'] and selection['source_trace_sha256']==SOURCE_TRACE_SHA
    assert selection['selected_control']==251 and selection['controls_copied']==251
    assert selection['physics_steps_copied']==2510 and selection['learned_prefix_controls']==1
    assert set(selection['outputs'])=={'precontrol251.npz','actual_prefix251.npz'}
    for name,digest in selection['outputs'].items():assert sha256(BASE/'inputs'/name)==digest
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
    prefix=archive(BASE/'inputs/actual_prefix251.npz');snap=archive(BASE/'inputs/precontrol251.npz')
    assert int(snap['recorded_controls'])==START and int(snap['integration_state_spec'])==8191
    assert int(prefix['final_recorded_controls'])==START and len(prefix['physics_torque'])==START*10
    np.testing.assert_array_equal(prefix['global_control'],np.arange(START))
    np.testing.assert_array_equal(prefix['source_frame'],np.arange(START)+11)
    np.testing.assert_array_equal(prefix['physics_substeps'],np.full(START,10))
    np.testing.assert_array_equal(prefix['controller_mode'][:250],np.zeros(250))
    assert int(prefix['controller_mode'][250])==1
    np.testing.assert_array_equal(prefix['physics_time'],prefix['physics_expected_time'])
    assert prefix['physics_expected_time'][0]==0.
    expected=0.
    for i in range(START*10):
        expected+=.002
        assert expected==prefix['physics_expected_time'][i+1]
    assert expected==float(snap['expected_time'])==float(snap['actual_time'])
    np.testing.assert_array_equal(prefix['final_integration'],snap['integration'])
    # One initialization restores the complete plant, including native warmstart.
    # Forward initializes derived arrays; reapplying integration before any step
    # preserves the recorded clock, controls, warmstart and external-force arrays.
    data=mujoco.MjData(native)
    HOOKS.register_live(data)
    HOOKS.instrument_seed(fresh)
    ACTIVE_STATE=dict(native=native,c=c,data=data,fresh=fresh,snapshot=snap,prefix=prefix)
    mujoco.mj_setState(native,data,snap['integration'],8191);mujoco.mj_forward(native,data)
    mujoco.mj_setState(native,data,snap['integration'],8191)
    data.warning.number[:]=snap['warning_counts'];data.warning.lastinfo[:]=snap['warning_lastinfo']
    for key,actual in [('integration',get_state(native,data)),('qpos',data.qpos),('qvel',data.qvel)]:
        np.testing.assert_array_equal(actual,snap[key])
    np.testing.assert_array_equal(data.qpos,prefix['qpos'][-1]);np.testing.assert_array_equal(data.qvel,prefix['qvel'][-1])
    np.testing.assert_array_equal(data.qpos,prefix['physics_qpos'][-1]);np.testing.assert_array_equal(data.qvel,prefix['physics_qvel'][-1])
    np.testing.assert_array_equal(data.warning.number,prefix['physics_warning_counts'][-1])
    np.testing.assert_array_equal(data.warning.lastinfo,prefix['physics_warning_lastinfo'][-1])
    assert data.time==expected  # Full291 already preserves all applied forces; no inferred zeros.
    reasons,_=assess(data,c,expected);assert not reasons,reasons
    # Rebuild sensor lags only from every recorded actual entry control, including learned250.
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
    request=dict(protocol(),kind='width81000_actual_BFM250_and_learned250_prefix_plus_fresh_expert251',clip='walk003',
        requested_controls=LIFECYCLE, BFM_prefix_controls=250,
        MPC_global_controls=[START,SWITCH],eligible_future_label_controls=[START,SWITCH],
        terminal_BFM_global_controls=[SWITCH,LIFECYCLE],separate_extension_controls=EXTENSION,
        feedback_correction_clip_rad=.1, finite_difference_epsilon=1e-6,
        all_joint_margin=.05,all_joint_weight=2000,relative_foot_weight=0,
        hard_feasibility=True,guided_then_K0_restoration=True,private_single_lane_optimization=False,
        initial_clock=expected,
        initial_history='exact incoming pre251: inverse actually applied learned250 target plus four measured lags',
        MPC_history='first real MPC251 pushes measured251 and incoming applied250 prior once; later applied-target inverse, never clipped to+-5',
        terminal_history='same accumulated buffer; raw BFM actor*5 next action, no record_control normalization',
        original_recorded_seed='bfm_walk003_arms_v3 targets only',fresh_BFM_goals='originalnative position1/yaw2/h8',
        terminal_BFM_goals='originalnative position1/yaw4/h8',optimization_goals='v4floor native reference; original full timing',
        model_manifest=manifest,motion_override=override,fresh_BFM_identity=fresh.identity(),
        received_goal_preview_seconds=.74,conservative_raw_pose_support_seconds=.76,
        initial_snapshot_sha256=sha256(BASE/'inputs/precontrol251.npz'),source_trace_sha256=SOURCE_TRACE_SHA,
        preserved_prefix_sha256=sha256(BASE/'inputs/actual_prefix251.npz'),
        frozen_sources_sha256=sha256(BASE/'frozen_inputs.json'),
        controller_mode_map={'0':'actual originalBFM controls0..249','1':'actual learned250; fresh expert only globalcontrol>=251','2':'terminal originalBFM yaw4'},
        physically_copied_later_student_or_teacher_states=False,physical_statewrites_after_initialization=0,
        initial_warnings_restored_separately=True,DAgger_fitting_authorized=False,learned_head_loaded=False,
        labels_gate='complete1318 remaining controls/all819source plus both quiet holds independently qualify',
        direct_head_abort_tolerance_difference='original expert bounds allow1e-6; earlier direct abort used exact bounds; neither changed here')
    return dict(native=native,c=c,original=original,motion=motion,original29=original29,audit=audit,
        planner=planner,fresh=fresh,data=data,snapshot=snap,prefix=prefix,expected=expected,recorded=recorded,request=request)

def initial_trace(s):
    return {target:list(s['prefix'][source].copy()) for target,source in PREFIX_MAP.items()}

def prefix_equality(s,trace):
    checks={}
    for target,source in PREFIX_MAP.items():
        original=s['prefix'][source]
        actual=np.asarray(trace[target])[:len(original)]
        exact=bool(actual.shape==original.shape and actual.dtype==original.dtype and actual.tobytes()==original.tobytes())
        checks[target]=dict(source_field=source,records=len(original),bitexact=exact)
        assert exact,('preserved_actual251_prefix_changed',target)
    write(BASE/'preserved_actual251_prefix_equality.json',dict(passed=True,checks=checks,
        full_original_prefix_trace_sha256=PREFIX_SHA,all_original_fields_preserved_in_bound_input=True,
        original_control_features_still_available_in_input=True,initial_BFM_controls=250,initial_learned_controls=1,initial_controls=251,expert_labels_exclude_prefix=True))

def save_trace(path,s,trace,**extra):
    if len(trace['global_control']) and trace['global_control'][0]==0:prefix_equality(s,trace)
    result=original_driver.save_trace(path,s,trace,**extra)
    save_work('trace_saved:'+str(path.relative_to(BASE)))
    return result

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
        first_global_control=251,planner_window_frame=261,first_output_frame=262)
    if best is not None:
        selected=dict(cost=np.asarray(best[0]),candidate=np.asarray(best[1]),nominal_states=best[2],targets=best[3])
        atomic_trace(out/'selected.npz',{},**selected);result['selected_sha256']=sha256(out/'selected.npz')
    else:selected=None
    write(out/'report.json',result);print(json.dumps(plain(result)),flush=True)
    save_work('first_query_complete')
    return result,selected

def run_branch(s,initial,selected):
    assert initial['initial_H30_feasible_seed_exists']
    assert sha256(BASE/'initial_seed/selected.npz')==initial['selected_sha256']
    out=BASE/'nominal';out.mkdir(exist_ok=False);(out/'plans').mkdir();write(out/'request.json',s['request'])
    trace=initial_trace(s);plans=[];restorer=None;failure=None;control=START;warm=None;last_commit=0
    started=time.perf_counter();next_checkpoint=START+100
    save_trace(out/'trace.partial.npz',s,trace,checkpoint_metadata=np.asarray(json.dumps(dict(completed_controls=START,requested_controls=LIFECYCLE,failure=None,labels_admissible=False))))
    write(out/'plans.json',plans)
    try:
        while control<SWITCH:
            if (BASE/'STOP_REQUESTED').exists():failure=dict(kind='requested_stop',control=control);break
            p=s['planner'];p.window(control+10);tick=time.perf_counter()
            if control==START:
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
            if control==START:assert_initial_actual_unchanged(s)
            count=min(5,SWITCH-control);last_commit=count;warm=targets.copy()
            plan=dict(control=control,cost=float(cost),selection=selection,solver_feasibility=copy.deepcopy(p.last_solve_feasibility),
                requested_commit=count,executed_controls=0,solve_seconds=time.perf_counter()-tick)
            plans.append(plan)
            atomic_trace(out/'plans'/('plan_%05d.npz'%control),{},nominal_states=states,targets=targets,gains=gains)
            for local in range(count):
                actual=np.r_[s['data'].qpos,s['data'].qvel]
                correction=np.clip(gains[local]@p.difference(states[local:local+1],actual[None])[0],-.1,.1)
                target=np.clip(targets[local]+correction,p.lo,p.hi)
                if control==START:
                    atomic_trace(out/'first_expert_target.npz',{},global_control=np.asarray(START),
                        target=target.copy(),nominal_target=targets[local].copy(),correction=correction.copy(),
                        actual_state=actual.copy(),incoming_prior=s['snapshot']['previous_action'].copy(),
                        incoming_history=s['snapshot']['history_flat'].copy(),proposal_before_preview=np.asarray(True))
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
    result=report_segment(s,trace,LIFECYCLE,failure,0.,'actual_BFM250_learned250_prefix_plus_fresh_expert251')
    result.update(trace_sha256=sha256(out/'trace.npz'),request_sha256=sha256(out/'request.json'),elapsed_seconds=time.perf_counter()-started,
        actual_expert_controls=int(np.sum((np.asarray(trace['controller_mode'])==1)&(np.asarray(trace['global_control'])>=START))),
        newly_executed_controls=len(trace['global_control'])-START,
        newly_executed_native_steps=len(trace['physics_torque'])-START*10,
        BFM_prefix_controls=250,learned_prefix_controls=1,preserved_prefix_controls=START,branch_initial_time=float(s['snapshot']['actual_time']))
    write(out/'report.json',result);print(json.dumps(plain(result)),flush=True)
    # Branch-only evidence starts at the exact actual pre251 boundary.
    # Combined evidence retains the original actual BFM250 plus learned250 prefix unchanged.
    a=arrays_of(trace);branch={}
    for k,v in a.items():
        branch[k]=v[START*10:] if k.startswith('physics_') and k!='physics_substeps' or k in ('range_excess','velocity_ratio','effort_ratio','clock_error') else v[START:]
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


def run_selected():
    for name in ('nominal','initial_seed','post_lifecycle_hold_5s','initial_restore_preflight.json'):
        assert not (BASE/name).exists(),'Existing expert branch attempt must remain preserved: '+name
    s=setup()
    save_work('restoration_complete')
    assert_initial_actual_unchanged(s)
    prefix_equality(s,initial_trace(s))
    write(BASE/'initial_restore_preflight.json',dict(passed=True,initial_global_control=START,initial_clock=s['expected'],
        full_integration_exact=True,all251_named_and_flat_sensor_history_entries_exact=True,
        raw_previous_action_exact=True,independent_repeated_clock_exact=True,
        actor_inference_calls=0,optimizer_calls=0,physics_steps_executed=0,
        source_trace_sha256=SOURCE_TRACE_SHA,preserved_prefix_sha256=sha256(BASE/'inputs/actual_prefix251.npz'),snapshot_sha256=sha256(BASE/'inputs/precontrol251.npz'),
        labels_admissible=False,hardware_authorized=False))
    initial,selected=certify_initial(s)
    if selected is None:
        write(BASE/'outcome.json',dict(physical_branch_started=False,initial_seed=initial,
            labels_admissible=False,DAgger_fitting_launched=False,hardware_authorized=False))
        return
    run_branch(s,initial,selected)


def save_work(phase):
    WORK.phase=phase
    record=WORK.snapshot()
    with (BASE/'work_snapshots.jsonl').open('a') as stream:
        stream.write(json.dumps(plain(record),allow_nan=False)+'\n')
    write(BASE/'work_counters.json',record)


def preserve_failure_state():
    if ACTIVE_STATE is None:return
    s=ACTIVE_STATE;fields={};errors={}
    for key,read in {
        'integration':lambda:get_state(s['native'],s['data']),
        'qpos':lambda:s['data'].qpos.copy(),'qvel':lambda:s['data'].qvel.copy(),
        'qfrc_actuator':lambda:s['data'].qfrc_actuator.copy(),
        'qfrc_applied':lambda:s['data'].qfrc_applied.copy(),'xfrc_applied':lambda:s['data'].xfrc_applied.copy(),
        'warning_number':lambda:s['data'].warning.number.copy(),'warning_lastinfo':lambda:s['data'].warning.lastinfo.copy(),
        'history':lambda:flat(s['fresh']),'previous_action':lambda:s['fresh'].previous_action.copy(),
        'recorded_controls':lambda:np.asarray(s['fresh'].recorded_controls),
    }.items():
        try:fields[key]=read()
        except BaseException as exc:errors[key]=dict(type=type(exc).__name__,message=str(exc))
    atomic_trace(BASE/'last_actual_state.npz',{},**fields)
    write(BASE/'last_actual_state_capture.json',dict(errors=errors,second_step_attempted=False,
        derived_forward_attempted=False,physics_state_repair_attempted=False))


def main():
    global WORK,HOOKS,ilqr
    admit(BASE)
    # One selected task attempt. Never replaces an existing lock or output.
    with (BASE/'ATTEMPT_STARTED').open('x') as stream:stream.write('one fixed offline recovery\n')
    WORK=WorkLedger()
    import gear_sonic.utils.g1_true23_mjbatch_ilqr_core as core
    import gear_sonic.utils.g1_true23_mjbatch_restoration as restoration
    HOOKS=Hooks(WORK,mujoco,core,restoration,original_driver)
    ilqr=HOOKS.ilqr
    try:
        save_work('task_started')
        run_selected()
        outcome=json.loads((BASE/'outcome.json').read_text())
        complete=bool(outcome.get('nominal',{}).get('full_segment_completed') and
                      outcome.get('extension',{}).get('full_segment_completed'))
        write(BASE/'driver_completion.json',dict(driver_returned=True,requested_branch_completed=complete,
              independent_physical_and_intent_qualification_pending=True,labels_admissible=False))
        return 0 if complete else 1
    except BaseException as exc:
        write(BASE/'failure.json',dict(type=type(exc).__name__,message=str(exc),labels_admissible=False))
        raise
    finally:
        try:preserve_failure_state()
        finally:
            save_work('task_ended')
            HOOKS.close()


if __name__=='__main__':raise SystemExit(main())
