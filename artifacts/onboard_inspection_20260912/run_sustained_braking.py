"""One preregistered from-start offline persistent-filter experiment."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from continue_pico_shoulder_recovery import Study,save_report
from gear_sonic.utils.g1_true23_sustained_braking import CONFIG,Predictor,SustainedBrakingFilter


class FilterStudy(Study):
    def __init__(self,run,library):
        super().__init__(run)
        self.braking=SustainedBrakingFilter(Predictor(self.model,self.c,library))
        self.braking.commit_applied(self.previous)
        self.controller.sustained_braking_filter=self.braking

    def step(self,control,intervention=None):
        self.braking.commit_applied(self.previous)
        result=super().step(control)
        result['filter_state']=self.braking.snapshot()
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--delay-schedule',choices=['archived_then_12ms','alternating_0_12ms'],default='archived_then_12ms')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    study=FilterStudy(args.input_run,args.library)
    if args.delay_schedule=='alternating_0_12ms':study.delay={c:6*(c%2) for c in range(study.total)}
    root=Path(__file__).resolve().parents[2]
    source_paths=[Path(__file__),root/'gear_sonic/utils/g1_true23_sustained_braking.py',
        root/'gear_sonic/utils/native23_sustained_prediction.c',root/'gear_sonic/utils/g1_true23_native_targets.py',
        root/'gear_sonic/utils/g1_true23_controller_state.py']
    prereg=dict(configuration=study.braking.configuration(),input_run=str(args.input_run),
        delay_schedule=args.delay_schedule,total_requested_controls=study.total,total_seconds=study.total*.02,
        terminal_hold_seconds=30,source_files=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in source_paths],
        stopping_rule='One fixed design, original-start full-motion attempt; stop on strict rejection or physical failure. No tuning after outcome.',
        current_prediction_initialization='fresh MjData q/v and mj_forward; no plant warmstart used by filter',
        continuation='same captured shoulder goal, bias-compensated acceleration feedback recomputed every 20ms; other 22 current actor targets held fixed',
        terminal_condition='shoulder margin >=.02rad and abs speed <=.25rad/s at both 60ms and 80ms',
        scope='49 current/next delay pairs, next delay held for third/fourth update; not all longer jitter sequences',
        timing_qualified=False)
    save_report(args.output/'preregistered.json',prereg)
    states=[np.r_[study.initial,0.]];calls=[];rows=[];snapshots={};expected={};failure=None;started=time.monotonic()
    with (args.output/'filter_events.jsonl').open('w') as events:
        for control in range(study.total):
            study.admit(control)
            mode=study.braking.mode
            if mode not in snapshots:
                snapshots[mode]=study.snapshot(control);expected[mode]=[]
            result=study.step(control);diagnostic=study.braking.last_diagnostic
            for captured in snapshots:
                if len(expected[captured])<3:expected[captured].append(copy.deepcopy(result))
            states.extend(result['physical']);calls.append(result['inference_ms'])
            record=dict(control=control,mode=diagnostic['mode'],mode_before=diagnostic['mode_before'],
                applied_delay_substeps=study.delay.get(control,6),accepted=diagnostic['accepted'],
                nominal_passed=diagnostic['nominal_passed'],correction_rad=diagnostic['correction_rad'],
                release_counter=diagnostic['release_counter'],prediction_count=len(diagnostic['predictions']))
            rows.append(record)
            if diagnostic['mode']!='TRACK' or not diagnostic['accepted'] or control%100==0:
                from gear_sonic.utils.g1_true23_sim_preview import json_safe
                events.write(json.dumps(json_safe(dict(control=control,**diagnostic)),allow_nan=False)+'\n');events.flush()
            if control%100==0:print(json.dumps(dict(control=control,seconds=control*.02,mode=study.braking.mode,elapsed_seconds=time.monotonic()-started)),flush=True)
            if result['failure']:failure=result['failure'];break
    states=np.asarray(states);report=study.summarize(states,failure,calls)
    n=(len(states)-1)//10;ids=np.arange(n)
    attempted_active=np.array([r['mode']!='TRACK' or abs(r['correction_rad'])>1e-9 for r in rows])
    active=attempted_active[:n]
    streaks=[];streak=0
    for value in active:
        if value:streak+=1
        elif streak:streaks.append(streak);streak=0
    if streak:streaks.append(streak)
    mask=active
    from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import metrics
    frames=np.minimum(ids+11,len(study.motion['joint_pos'])-1)
    interval_metrics=metrics(study.model,states[(ids[mask]+1)*10,:30],frames[mask],ids[mask],
        study.motion,study.original,study.timeline,study.tasks)
    report.update(experiment='persistent_state_triggered_shoulder_braking',delay_schedule=args.delay_schedule,
        configuration=study.braking.configuration(),terminal_filter=study.braking.last_diagnostic,
        intervention_controls=int(active.sum()),intervention_attempts=int(attempted_active.sum()),
        intervention_fraction=float(active.mean()) if n else 0.,
        intervention_seconds=float(active.sum()*.02),active_episode_durations_seconds=[n*.02 for n in streaks],
        correction_abs_p50_p95_max=np.percentile(np.abs([r['correction_rad'] for r in rows[:n]]),[50,95,100]).tolist() if n else [0.,0.,0.],
        tracking_during_intervention=interval_metrics,intervention_tracking_metrics_are_partial=True,
        from_original_start=True,uses_failure_timestamp=False,elapsed_wall_seconds=time.monotonic()-started,
        decision='candidate_for_further_checks' if report['physical_complete'] and report['tracking_passed'] and report['standing_passed'] else 'not_promoted_stop_fixed_design')
    np.savez_compressed(args.output/'trace.npz',states=states)
    save_report(args.output/'control_log.json',rows)
    save_report(args.output/'report.json',report)
    # Owned snapshots include filter mode/counter/correction/continuation and real
    # applied history. Predictions must not mutate this state on restored replay.
    checks=[]
    for mode,snapshot in snapshots.items():
        study.restore(snapshot)
        for offset,original in enumerate(expected[mode]):
            control=snapshot['control']+offset
            if offset:study.admit(control)
            actual=study.step(control)
            for key in ('features','target','physical','integration','torques'):
                np.testing.assert_array_equal(np.asarray(actual[key]),np.asarray(original[key]),err_msg=f'{mode}:{control}:{key}')
            for key in actual['filter_state']:
                left,right=actual['filter_state'][key],original['filter_state'][key]
                if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
                else:assert left==right,(mode,control,key)
            assert actual['failure']==original['failure']
            checks.append(dict(mode=mode,control=control,exact=True,physics_steps=len(actual['physical'])))
    save_report(args.output/'restoration.json',dict(passed=True,cases=checks))
    print(json.dumps({k:report[k] for k in ('physical_seconds','physical_complete','failure','intervention_controls','decision')}),flush=True)


if __name__=='__main__':main()
