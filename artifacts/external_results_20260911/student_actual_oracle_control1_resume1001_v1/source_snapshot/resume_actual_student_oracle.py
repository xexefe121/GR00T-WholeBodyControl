"""Resume the interrupted single expert branch at its atomic committed boundary.

The original source and all recorded controls remain immutable. Only process
initialization and output persistence differ from the reviewed original driver.
"""
from run_actual_student_oracle import *
import run_actual_student_oracle as original_driver

RESUME = BASE / 'resume_inputs'
CHECKPOINT_CONTROLS = 1001
TRACE_FIELDS = (*CONTROL_FIELDS, *PHYSICS_FIELDS, 'qpos', 'qvel',
                'range_excess', 'velocity_ratio', 'effort_ratio', 'clock_error',
                *('control_history_' + key for key in KEYS))


def setup():
    frozen()
    assert mujoco.__version__ == '3.2.3'
    native, c, original, timeline, manifest = load_native_bundle(BUNDLE, 'walk003')
    motion, override = load_motion_override(REFERENCE, BUNDLE, 'walk003', native, c,
                                            original, timeline, manifest)
    original29 = archive(BUNDLE / 'walk003/original29.npz')
    audit = json.loads((TEACHER / 'recorded_source_audit_v2.json').read_text())
    kp, kd, effort = [np.asarray(c[k]) for k in ('kp', 'kd', 'native_effort')]
    planner = Native23Tracker(position_servo_copy(native, kp, kd, effort), c, motion,
        horizon=30, threads=2, all_joint_limit_margin=.05, all_joint_limit_weight=2000,
        relative_foot_weight=0, fd_epsilon=1e-6, hard_feasibility=True)
    fresh = Native23BFMRolloutSeed(native, c, original, ONNX,
                                  dependency_directory=DEPS, threads=1)
    snap = archive(BASE / 'inputs/actual_student_control1.npz')
    prefix = archive(BASE / 'inputs/student_control0_prefix.npz')
    checkpoint = archive(RESUME / 'trace.partial.npz')
    metadata = json.loads(str(checkpoint['checkpoint_metadata']))
    assert metadata == dict(completed_controls=1001, requested_controls=1569,
                            failure=None, labels_admissible=False)
    assert int(checkpoint['integration_state_spec']) == 8191
    assert int(checkpoint['final_recorded_controls']) == CHECKPOINT_CONTROLS
    np.testing.assert_array_equal(checkpoint['global_control'], np.arange(1001))
    np.testing.assert_array_equal(checkpoint['source_frame'], np.arange(1001) + 11)
    np.testing.assert_array_equal(checkpoint['physics_substeps'], np.full(1001, 10))
    np.testing.assert_array_equal(checkpoint['controller_mode'], np.r_[0, np.ones(1000)])
    np.testing.assert_array_equal(checkpoint['initial_integration'], prefix['initial_integration'])
    np.testing.assert_array_equal(checkpoint['branch_initial_integration'], snap['integration'])
    np.testing.assert_array_equal(checkpoint['control_integration_before'][1], snap['integration'])
    np.testing.assert_array_equal(checkpoint['target'][0], prefix['target'][0])
    np.testing.assert_array_equal(checkpoint['physics_qpos'][:11], prefix['physics_qpos'])
    np.testing.assert_array_equal(checkpoint['physics_qvel'][:11], prefix['physics_qvel'])
    np.testing.assert_array_equal(checkpoint['physics_time'], checkpoint['physics_expected_time'])
    assert len(checkpoint['physics_time']) == 10011
    expected = float(checkpoint['physics_expected_time'][-1])
    accumulated = float(checkpoint['physics_expected_time'][0])
    for i in range(10010):
        accumulated += .002
        assert accumulated == checkpoint['physics_expected_time'][i + 1]

    # Restore the final committed physical state once at process initialization.
    # Forward initializes derived arrays, then the integration vector is reapplied
    # to preserve the exact native warmstart, clock, controls and external forces.
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, checkpoint['final_integration'], 8191)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, checkpoint['final_integration'], 8191)
    data.warning.number[:] = checkpoint['physics_warning_number'][-1]
    data.warning.lastinfo[:] = checkpoint['physics_warning_lastinfo'][-1]
    np.testing.assert_array_equal(get_state(native, data), checkpoint['final_integration'])
    np.testing.assert_array_equal(data.qpos, checkpoint['physics_qpos'][-1])
    np.testing.assert_array_equal(data.qvel, checkpoint['physics_qvel'][-1])
    np.testing.assert_array_equal(data.qpos, checkpoint['qpos'][-1])
    np.testing.assert_array_equal(data.qvel, checkpoint['qvel'][-1])
    assert float(data.time) == expected
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    reasons, _ = assess(data, c, expected)
    assert not reasons, reasons

    # Sensor-only reconstruction proves that the stored named lag arrays and
    # previous applied actions form one continuous history. No actor call or
    # simulation step is used here. Control0 retains the student's combined raw
    # action; every following MPC action uses the actually applied target.
    for control in range(CHECKPOINT_CONTROLS):
        np.testing.assert_array_equal(flat(fresh), checkpoint['control_history_before'][control])
        np.testing.assert_array_equal(fresh.previous_action,
                                      checkpoint['control_previous_action_before'][control])
        for key in KEYS:
            np.testing.assert_array_equal(fresh.history.data[key],
                                          checkpoint['control_history_' + key][control])
        _, terms = fresh._terms(checkpoint['qpos'][control],
                                checkpoint['qvel'][control], fresh.previous_action)
        fresh.history.before_update(terms)
        if control == 0:
            action = prefix['action'][0]
        else:
            action = ((checkpoint['target'][control] - fresh.contract['default_q'])
                      * fresh.contract['kp'] / (.25 * fresh.contract['training_effort'])).astype(np.float32)
        np.testing.assert_array_equal(action, checkpoint['action'][control])
        fresh.previous_action = action.copy()
    for key in KEYS:
        np.testing.assert_array_equal(fresh.history.data[key], checkpoint['final_history_' + key])
    np.testing.assert_array_equal(fresh.previous_action, checkpoint['final_previous_action'])
    fresh.recorded_controls = CHECKPOINT_CONTROLS
    fresh.actual_action_max_abs = float(np.max(np.abs(checkpoint['action'])))
    fresh.actual_action_components_outside_five = int(np.sum(np.abs(checkpoint['action']) > 5))
    recorded = archive(RECORDED / 'trace.npz')['target']
    assert recorded.shape == (1569, 23) and np.isfinite(recorded).all()
    np.testing.assert_array_equal(recorded, np.clip(recorded, planner.lo, planner.hi))
    plans = json.loads((RESUME / 'plans.json').read_text())
    assert len(plans) == 200
    assert [p['control'] for p in plans] == list(range(1, 1001, 5))
    assert all(p['requested_commit'] == p['executed_controls'] == 5 for p in plans)
    warm = archive(RESUME / 'plans/plan_00996.npz')['targets']
    assert warm.shape == (30, 23)
    np.testing.assert_array_equal(warm, np.clip(warm, planner.lo, planner.hi))
    request = json.loads((RESUME / 'request.json').read_text())
    assert request['requested_controls'] == LIFECYCLE == 1569
    assert request['initial_global_control'] == 1
    receipt = json.loads((BASE / 'resume_receipt.json').read_text())
    assert sha256(RESUME / 'trace.partial.npz') == receipt['original_checkpoint_sha256']
    request['process_interruption_resume'] = dict(receipt,
        receipt_sha256=sha256(BASE / 'resume_receipt.json'),
        resume_frozen_sources_sha256=sha256(BASE / 'frozen_inputs.json'),
        resume_clock=expected, physical_state_restores_after_process_initialization=0)
    return dict(native=native, c=c, original=original, motion=motion, original29=original29,
        audit=audit, planner=planner, fresh=fresh, data=data, snapshot=snap, prefix=prefix,
        expected=expected, recorded=recorded, request=request, checkpoint=checkpoint,
        resume_plans=plans, resume_warm=warm)


def initial_trace(s):
    return {k: list(s['checkpoint'][k].copy()) for k in TRACE_FIELDS}


def verify_first_resumed_plan(control, states, targets, gains):
    if control != CHECKPOINT_CONTROLS:
        return
    saved = archive(RESUME / 'uncommitted_plan_01001_verification_only.npz')
    checks = {}
    for key, actual in [('nominal_states', states), ('targets', targets), ('gains', gains)]:
        checks[key] = dict(exact=bool(np.array_equal(actual, saved[key])),
                           maximum_absolute_difference=float(np.max(np.abs(actual - saved[key]))))
    write(BASE / 'first_resumed_plan_equality.json', dict(control=control, checks=checks,
          cached_targets_executed=False, optimizer_executed_once=True))
    assert all(v['exact'] for v in checks.values()), checks


def preflight():
    s = setup()
    write(BASE / 'resume_preflight.json', dict(
        checkpoint_controls=1001, physics_steps=10010, next_global_control=1001,
        final_integration_exact=True, named_history_and_prior_reconstruction_exact=True,
        clock_exact=True, expected_time=s['expected'], source_controls=651,
        committed_plans=200, warm_plan_control=996, last_commit=5,
        actor_inference_calls=0, optimizer_calls=0, physics_steps_executed=0,
        original_requested_controls=LIFECYCLE, separate_extension_controls=EXTENSION,
        labels_admissible=False, hardware_authorized=False))
    print(json.dumps(dict(preflight_pass=True, next_global_control=1001)), flush=True)



def run_branch():
    initial=json.loads((BASE/'initial_seed/report.json').read_text())
    assert initial['initial_H30_feasible_seed_exists']
    assert sha256(BASE/'initial_seed/selected.npz')==initial['selected_sha256']
    out=BASE/'nominal';out.mkdir(exist_ok=False);(out/'plans').mkdir();s=setup();write(out/'request.json',s['request'])
    trace=initial_trace(s);plans=copy.deepcopy(s['resume_plans']);restorer=None;failure=None
    control=CHECKPOINT_CONTROLS;warm=s['resume_warm'].copy();last_commit=5
    for old_plan in plans:
        name='plan_%05d.npz'%old_plan['control']
        (out/'plans'/name).write_bytes((RESUME/'plans'/name).read_bytes())
    selected=None;started=time.perf_counter();next_checkpoint=CHECKPOINT_CONTROLS+100
    save_trace(out/'trace.partial.npz',s,trace,checkpoint_metadata=np.asarray(json.dumps(dict(
        completed_controls=control,requested_controls=LIFECYCLE,failure=None,labels_admissible=False))))
    write(out/'plans.json',plans)
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
            verify_first_resumed_plan(control,states,targets,gains)
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



if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','continue'],required=True)
    args=parser.parse_args();preflight() if args.stage=='preflight' else run_branch()
