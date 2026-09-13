"""Source preparation only. Copies qualified modules and derives the c251 adapter."""
import json
import hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'bfm_entry250_actual_oracle_v1/source_snapshot_v1'
OUT=BASE/'source_draft_v1'
receipt=json.loads((OLD.parent/'frozen_inputs.json').read_text())
for relative,digest in receipt['source_sha256'].items():
    if relative=='run_bfm250_actual_oracle.py':continue
    source=OLD/relative;target=OUT/relative
    raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)

source=(OLD/'run_bfm250_actual_oracle.py').read_text()
source=source.replace('"""One original-BFM prefix250 followed by actual-state expert transition.\n\nOriginal14 expert source files are unchanged. Only the branch initialization,\nprefix accounting and persistent evidence differ from the qualified driver.\n"""',
'''"""Actual BFM250 + learned250 prefix, then one fresh expert branch from251.

All14 qualified expert modules are unchanged. This adapter preserves the actual
prefix, restores full291 and incoming causal state, and counts selected work.
"""''')
source=source.replace("START=250\nPREFIX_SHA='8a2013cee1112aedc9a9e290382a7c7801d1c8c49f0e1bd0d978619873de8b41'",'''from recovery_contract import START, SOURCE_TRACE_SHA, protocol
from recovery_admission import admit
from counted_work import WorkLedger, Hooks

PREFIX_SHA=SOURCE_TRACE_SHA
WORK=None
HOOKS=None
ACTIVE_STATE=None''')
start=source.index('    physics=json.loads',source.index('def setup():'))
stop=source.index('    native,c,original,timeline,manifest=',start)
source=source[:start]+'''    global ACTIVE_STATE
    selection=json.loads((BASE/'inputs/selection_receipt.json').read_text())
    assert selection['passed'] and selection['source_trace_sha256']==SOURCE_TRACE_SHA
    assert selection['selected_control']==251 and selection['controls_copied']==251
    assert selection['physics_steps_copied']==2510 and selection['learned_prefix_controls']==1
    assert set(selection['outputs'])=={'precontrol251.npz','actual_prefix251.npz'}
    for name,digest in selection['outputs'].items():assert sha256(BASE/'inputs'/name)==digest
'''+source[stop:]
source=source.replace("prefix=archive(BASE/'inputs/baseline250_trace.npz');snap=archive(BASE/'inputs/precontrol250.npz')",
                      "prefix=archive(BASE/'inputs/actual_prefix251.npz');snap=archive(BASE/'inputs/precontrol251.npz')")
source=source.replace("    assert sha256(BASE/'inputs/baseline250_trace.npz')==PREFIX_SHA\n",'')
source=source.replace("len(prefix['physics_torque'])==2500","len(prefix['physics_torque'])==START*10")
source=source.replace("np.testing.assert_array_equal(prefix['controller_mode'],np.zeros(START))\n    np.testing.assert_array_equal(prefix['delta'],np.zeros((START,23),np.float32))",
    "np.testing.assert_array_equal(prefix['controller_mode'][:250],np.zeros(250))\n    assert int(prefix['controller_mode'][250])==1")
source=source.replace('range(2500)','range(START*10)')
source=source.replace('    data=mujoco.MjData(native)','''    data=mujoco.MjData(native)
    HOOKS.register_live(data)
    HOOKS.instrument_seed(fresh)
    ACTIVE_STATE=dict(native=native,c=c,data=data,fresh=fresh,snapshot=snap,prefix=prefix)''')
source=source.replace("    assert data.time==expected and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)",
                      "    assert data.time==expected  # Full291 already preserves all applied forces; no inferred zeros.")
source=source.replace('every recorded actual BFM entry control.','every recorded actual entry control, including learned250.')
start=source.index("    request=dict(kind=",source.index('def setup():'))
stop=source.index('    return dict(native=',start)
source=source[:start]+'''    request=dict(protocol(),kind='width81000_actual_BFM250_and_learned250_prefix_plus_fresh_expert251',clip='walk003',
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
'''+source[stop:]
source=source.replace("'preserved_BFM250_prefix_changed'","'preserved_actual251_prefix_changed'")
source=source.replace("exact=bool(np.array_equal(actual,original));checks[target]=dict(source_field=source,records=len(original),bitexact=exact)",
    "exact=bool(actual.shape==original.shape and actual.dtype==original.dtype and actual.tobytes()==original.tobytes())\n        checks[target]=dict(source_field=source,records=len(original),bitexact=exact)")
source=source.replace("'preserved_baseline250_prefix_equality.json'","'preserved_actual251_prefix_equality.json'")
source=source.replace('initial_BFM_controls=250,expert_labels_exclude_prefix=True',
                      'initial_BFM_controls=250,initial_learned_controls=1,initial_controls=251,expert_labels_exclude_prefix=True')
source=source.replace("    return original_driver.save_trace(path,s,trace,**extra)",
'''    result=original_driver.save_trace(path,s,trace,**extra)
    save_work('trace_saved:'+str(path.relative_to(BASE)))
    return result''')
source=source.replace('first_global_control=250,planner_window_frame=260,first_output_frame=261',
                      'first_global_control=251,planner_window_frame=261,first_output_frame=262')
source=source.replace("    return result,selected","    save_work('first_query_complete')\n    return result,selected")
source=source.replace('actual_original_BFM250_prefix_plus_fresh_expert_transition','actual_BFM250_learned250_prefix_plus_fresh_expert251')
source=source.replace("                before=len(trace['target']);failure=apply_control(s,trace,control,target)",
'''                if control==START:
                    atomic_trace(out/'first_expert_target.npz',{},global_control=np.asarray(START),
                        target=target.copy(),nominal_target=targets[local].copy(),correction=correction.copy(),
                        actual_state=actual.copy(),incoming_prior=s['snapshot']['previous_action'].copy(),
                        incoming_history=s['snapshot']['history_flat'].copy(),proposal_before_preview=np.asarray(True))
                before=len(trace['target']);failure=apply_control(s,trace,control,target)''')
source=source.replace("actual_expert_controls=int(np.sum(np.asarray(trace['controller_mode'])==1)),BFM_prefix_controls=START,",
'''actual_expert_controls=int(np.sum((np.asarray(trace['controller_mode'])==1)&(np.asarray(trace['global_control'])>=START))),
        newly_executed_controls=len(trace['global_control'])-START,
        newly_executed_native_steps=len(trace['physics_torque'])-START*10,
        BFM_prefix_controls=250,learned_prefix_controls=1,preserved_prefix_controls=START,''')
source=source.replace('exact baseline250 boundary.','exact actual pre251 boundary.')
source=source.replace('original actual BFM entry prefix unchanged.','original actual BFM250 plus learned250 prefix unchanged.')
source=source.replace('def main():','def run_selected():')
source=source.replace('    s=setup()','    s=setup()\n    save_work(\'restoration_complete\')')
source=source.replace('all250_named_and_flat_sensor_history_entries_exact=True','all251_named_and_flat_sensor_history_entries_exact=True')
source=source.replace('BFM_prefix_sha256=PREFIX_SHA,snapshot_sha256=sha256(BASE/\'inputs/precontrol250.npz\')',
                      "source_trace_sha256=SOURCE_TRACE_SHA,preserved_prefix_sha256=sha256(BASE/'inputs/actual_prefix251.npz'),snapshot_sha256=sha256(BASE/'inputs/precontrol251.npz')")
source=source.replace("if __name__=='__main__':main()",'''
def save_work(phase):
    WORK.phase=phase
    record=WORK.snapshot()
    with (BASE/'work_snapshots.jsonl').open('a') as stream:
        stream.write(json.dumps(plain(record),allow_nan=False)+'\\n')
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
    with (BASE/'ATTEMPT_STARTED').open('x') as stream:stream.write('one fixed offline recovery\\n')
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


if __name__=='__main__':raise SystemExit(main())''')
target=OUT/'run_width251_actual_oracle.py'
with target.open('x',newline='\n') as stream:stream.write(source)
