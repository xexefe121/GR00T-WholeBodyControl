"""Derive a continuous 50 Hz benchmark with startup warmup and 30 s hold."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE/'run_controller.py').read_text()
s=s.replace("    progress=a.output/'progress.json'",'''    # Warm the policy before the timed epoch; no history commit or plant step.
    before_history={k:v.copy() for k,v in base.seed.history.data.items()}
    before_prior=base.seed.previous_action.copy()
    warm_started=time.perf_counter()
    base.propose(0,data.qpos,data.qvel)
    for k,v in before_history.items():np.testing.assert_array_equal(base.seed.history.data[k],v)
    np.testing.assert_array_equal(base.seed.previous_action,before_prior)
    assert base.seed.recorded_controls==0
    base.pending=None;base.current_mode=None;base.context={}
    warmup_ms=(time.perf_counter()-warm_started)*1000
    epoch=time.perf_counter()+.2
    progress=a.output/'progress.json' ''')
s=s.replace('conditional_hold_controls=250','conditional_hold_controls=1500')
s=s.replace("            inference_ms=(count,),control_loop_ms=(count,),feedback_correction_max=(count,),", "            inference_ms=(count,),control_loop_ms=(count,),deadline_finish_ms=(count,),start_lateness_ms=(count,),feedback_correction_max=(count,),")
s=s.replace('            tick=time.perf_counter()', '''            scheduled=epoch+control*.02
            while True:
                remaining=scheduled-time.perf_counter()
                if remaining<=0:break
                if remaining>.0004:time.sleep(max(0,remaining-.0003))
            tick=time.perf_counter()
            arrays['start_lateness_ms'][local_control]=(tick-scheduled)*1000''')
s=s.replace("            arrays['control_loop_ms'][local_control]=(time.perf_counter()-tick)*1000", "            finished=time.perf_counter()\n            arrays['control_loop_ms'][local_control]=(finished-tick)*1000\n            arrays['deadline_finish_ms'][local_control]=(finished-scheduled)*1000")
s=s.replace('motion_specific=True,real_time_paced=False,live_teleoperation_qualified=False)', '''motion_specific=True,real_time_paced=True,live_teleoperation_qualified=False,
            pacing='50 Hz cycles; ten native substeps per cycle; plant not independently paced',
            startup_uncommitted_policy_warmup_ms=warmup_ms,startup_policy_calls=2,
            finish_after_scheduled_20ms_deadline=int(np.sum(arrays['deadline_finish_ms']>20)),
            finish_ms_p50_p95_max=timing('deadline_finish_ms'),start_lateness_ms_p50_p95_max=timing('start_lateness_ms'))''')
start=s.index("    main_result=segment('nominal',0,1569)")
stop=s.index("if __name__=='__main__':main()",start)
s=s[:start]+'''    continuous=segment('continuous',0,3069)
    result=dict(continuous=continuous,full_motion_and_hold_completed=continuous['full_segment_completed'],
        main_controls=1569,hold_controls=1500,hold_seconds=30,live_teleoperation_qualified=False,
        hardware_authorized=False,offline_motion_specific_controller=True)
    if continuous['full_segment_completed']:
        with np.load(a.output/'continuous/trace.npz',allow_pickle=False) as z:
            states={'physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'}
            steps={'physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error'}
            for name,start,stop in [('main',0,1569),('hold',1569,3069)]:
                arrays={}
                for key in z.files:
                    if key in ('initial_integration','final_integration','integration_state_spec'):continue
                    arrays[key]=z[key][start*10:stop*10+1] if key in states else z[key][start*10:stop*10] if key in steps else z[key][start:stop+1] if key in ('qpos','qvel') else z[key][start:stop]
                result[name]=dict(completed_controls=stop-start,source_metrics=source_metrics(native,arrays,motion,original29,intent_audit,timeline),
                    quiet_standing_diagnostic=quiet_diagnostic(standing_windows(arrays,motion,original,original29),True),
                    policy_ms_p50_p95_max=np.percentile(arrays['inference_ms'],[50,95,100]).tolist(),
                    finish_after_scheduled_20ms_deadline=int(np.sum(arrays['deadline_finish_ms']>20)))
    write(a.output/'report.json',result)
    print(json.dumps(dict(full_motion_and_hold_completed=result['full_motion_and_hold_completed'],output=str(a.output))),flush=True)
''' +s[stop:]
out=BASE/'run_paced_controller.py';assert not out.exists();out.write_text(s,encoding='utf-8')
launcher=(BASE/'run_trial_once.py').read_text().replace('baseline_process_v1','paced_process_v1').replace("BASE/'baseline_v1'","BASE/'paced_v1'").replace('run_controller.py','run_paced_controller.py')
out=BASE/'run_paced_trial_once.py';assert not out.exists();out.write_text(launcher,encoding='utf-8')
print('Prepared full 31.38 s motion plus continuous 30 s hold, one warmed 50 Hz run.')
