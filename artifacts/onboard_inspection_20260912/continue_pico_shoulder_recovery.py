"""Offline counterfactual continuations of a fresh uninterrupted received run.

Logical packet and command-delay schedules are declared exogenous inputs.
No wall-clock timing qualification is claimed. Production strict runner unchanged.
"""
import argparse,copy,json,pickle,time
from pathlib import Path
import mujoco
import numpy as np
from diagnose_pico_preview_failure import integration_state,restore_physics,set_pd,load_case
from search_pico_shoulder_recovery import joint_index,rollout
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import assess,metrics,quiet
from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from gear_sonic.utils.g1_true23_sim_preview import SimulatorPreviewRejected,publish_checked,json_safe,terminal_standing


class CaptureSession:
    def __init__(self,session):self.session=session;self.last=None
    def run(self,names,inputs):
        self.last=inputs['features'].copy()
        return self.session.run(names,inputs)


class Study:
    def __init__(self,run):
        self.run=run;self.report=json.loads((run/'report.json').read_text())
        self.manifest=json.loads((run/'run_manifest.json').read_text())
        args=self.manifest['arguments']
        self.model,self.c,self.motion,self.original,self.timeline=load_case('pico')
        self.tasks=json.loads((Path(args['bank'])/'bank.json').read_text())['tasks']
        with np.load(run/'trace.npz') as z:
            self.initial=z['states'][0,:59].copy();timing=z['timing'];epoch=int(z['epoch_ns'])
        self.delay={0:0}
        for step,row in enumerate(timing):
            ident=int(row[4])
            if ident not in self.delay:self.delay[ident]=step-ident*10
        if any(not 0<=d<=6 for d in self.delay.values()):raise ValueError('Archived delay outside tested budget')
        calls=np.load(run/'controller_calls.npy')
        self.admitted={int(row[0]):int(row[6]) for row in calls}
        self.now={int(row[0]):(int(row[2])-epoch)*1e-9 for row in calls if row[0]}
        self.now[0]=0.
        self.total=self.timeline['total_requested_controls']+1500
        fw=Path(args['factory_config']).parents[3]
        self.controller=NativeTargetController(Path(args['actor']),fw,self.c,model=self.model,tasks=self.tasks,
            standing_qpos=self.timeline['configured_standing_qpos'],now=-.22,
            native_preview_guard=True,native_standing_capture=True,native_preview_delay_substeps=6,
            native_preview_library=Path(args['native_preview_library']),fault_standing_capture=False)
        self.controller.session=CaptureSession(self.controller.session)
        self.joint=joint_index(self.model,self.c)
        self.data=mujoco.MjData(self.model);self.data.qpos[:]=self.initial[:30];self.data.qvel[:]=self.initial[30:]
        mujoco.mj_forward(self.model,self.data)
        self.previous=self.initial[7:30].copy();self.last_admitted=-1
        for sequence in range(12):self.receive(sequence,(sequence-11)*.02)
        self.controller.import_native_history(0)
        self.controller.warmup(self.data.qpos,self.data.qvel,0.,10)

    def logical_time(self,control):
        return self.now.get(control,control*.02+.0001)

    def receive(self,sequence,now):
        fields={k:v[sequence] for k,v in self.motion.items() if k!='fps'}
        final=sequence==len(self.motion['joint_pos'])-1
        accepted=self.controller.receive(Packet(0,sequence,sequence*.02,fields,final),
            self.original['source_task_position_w'][sequence],self.original['source_task_quaternion_wxyz'][sequence],now)
        if not accepted:raise RuntimeError(f'Packet {sequence} rejected in fresh schedule')
        self.last_admitted=sequence

    def admit(self,control):
        last=self.admitted.get(control,min(control+10,len(self.motion['joint_pos'])-1))
        for sequence in range(self.last_admitted+1,last+1):self.receive(sequence,self.logical_time(control))

    def snapshot(self,control):
        return dict(control=control,logical_time=self.logical_time(control),
            controller=self.controller.snapshot(self.logical_time(control)),
            integration_state=integration_state(self.model,self.data),
            previous_actual_target=self.previous.copy(),last_admitted=self.last_admitted,
            application_delay_substeps=self.delay.get(control,6),
            pending_command='previous target held; current command not generated',
            boundary='packet admitted; before current controller observation and history import')

    def restore(self,snapshot):
        self.controller.restore_snapshot(snapshot['controller'])
        self.data=restore_physics(self.model,snapshot['integration_state'])
        self.previous=snapshot['previous_actual_target'].copy();self.last_admitted=snapshot['last_admitted']
        if self.logical_time(snapshot['control'])!=snapshot['logical_time']:raise ValueError('Logical clock mismatch')

    def step(self,control,intervention=None):
        self.controller.import_native_history(control)
        q=self.data.qpos.copy();v=self.data.qvel.copy()
        original_apply=self.controller.preview_guard.apply;replacement={}
        if intervention is not None:
            def replace(qpos,qvel,requested,previous=None):
                changed=requested.copy();changed[self.joint]=intervention
                replacement.update(original_requested=requested.copy(),intervened_requested=changed.copy())
                return original_apply(qpos,qvel,changed,previous)
            self.controller.preview_guard.apply=replace
        started=time.perf_counter()
        try:command=self.controller.command(q,v,self.logical_time(control))
        finally:self.controller.preview_guard.apply=original_apply
        elapsed=(time.perf_counter()-started)*1000
        evidence=copy.deepcopy(self.controller.preview_guard.last_diagnostic)
        evidence.update(control=control,preview_status=command.status['native_preview_guard'],intervention=replacement)
        target=command.targets.copy();physical=[];integration=[];torques=[]
        try:publish_checked(command,evidence,dict(control=control),'strict',[],lambda:[1,1])
        except SimulatorPreviewRejected:
            return dict(control=control,target=target,features=self.controller.session.last.copy(),diagnostic=evidence,
                physical=physical,integration=integration,torques=torques,inference_ms=elapsed,
                failure=dict(kind='preview_rejected',control=control),completed=False)
        expected=self.data.time;failure=None
        delay=self.delay.get(control,6)
        for substep in range(10):
            applied=self.previous if substep<delay else target
            set_pd(self.data,applied,self.c['kp'],self.c['kd'],self.c['native_effort'],self.c['joint_limits'])
            torques.append(self.data.ctrl.copy());mujoco.mj_step(self.model,self.data);expected+=.002
            physical.append(np.r_[self.data.qpos,self.data.qvel,self.data.time])
            integration.append(integration_state(self.model,self.data))
            reasons,_=assess(self.data,self.c,expected)
            if reasons:
                failure=dict(kind='physics',control=control,substep=substep+1,reasons=reasons);break
        if failure is None:
            # Same existing observed-history recurrence, fed actual last applied
            # command and pre-control state. Never commit a rejected proposal.
            self.controller.receiver.history.commit(q,v,applied)
            self.previous=applied.copy()
        return dict(control=control,target=target,features=self.controller.session.last.copy(),diagnostic=evidence,
            physical=physical,integration=integration,torques=torques,inference_ms=elapsed,
            failure=failure,completed=failure is None)

    def summarize(self,states,failure,calls):
        count=len(states)-1;steps=count//10;ids=np.arange(steps)
        frames=np.minimum(ids+11,len(self.motion['joint_pos'])-1)
        source=metrics(self.model,states[(ids+1)*10,:30],frames,ids,self.motion,self.original,self.timeline,self.tasks)
        complete=steps==self.total and failure is None
        trailing=quiet(states[:,:30],states[:,30:59],self.original['source_qpos29'][-1])
        trailing30=quiet(states[:,:30],states[:,30:59],self.original['source_qpos29'][-1],seconds=30)
        reached=bool(complete and self.controller.receiver.gate.epoch_report()['full_source_consumed'])
        final,hold=terminal_standing(trailing,trailing30,reached)
        return dict(physical_complete=complete,physical_seconds=count*.002,completed_controls=steps,
            failure=failure,source=source,tracking_passed=source['passed'],standing_passed=hold['passed'],
            final_quiet=final,continuous_quiet30=hold,terminal_standing_reached=reached,
            timing_passed=False,timing_status='not_evaluated_offline_counterfactual',
            independent_wall_clock=False,physics_step_seconds=.002,control_period_seconds=.02,
            command_delay='archived 0-6 substep exogenous schedule through available recording; 6 substeps afterward',
            packet_schedule='archived admitted sequence indices; new logical admission times at control boundaries; causal 50Hz thereafter',
            inference_ms_p50_p95_max=np.percentile(calls,[50,95,100]).tolist(),
            new_uninterrupted_run_not_exact_claim_about_strict_newer_v1=True,
            packet_report=self.controller.receiver.gate.epoch_report(),
            future_reference_frames=0,simulation_time_frozen_for_offline_computation=True,
            general_live_teleop_qualified=False,hardware_authorized=False)


def save_report(path,report):path.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')


def verify_saved_continuations(args):
    """Recover terminal diagnostics for the same fixed trials; no new candidates.

    Only read the locally generated, trusted snapshot pickle. Verify every
    reproduced physical step against the saved branch before releasing evidence.
    """
    previous=args.verify_existing
    verified=json.loads((previous/'restoration_verification.json').read_text())
    if not verified['passed']:raise ValueError('Verified controller restoration required')
    with (previous/'coherent_snapshots.local.pkl').open('rb') as stream:snapshots=pickle.load(stream)
    study=Study(args.input_run);results=[]
    for directory in sorted(previous.glob('candidate_*')):
        recorded=json.loads((directory/'report.json').read_text())
        if not recorded['continuation_started']:continue
        boundary=recorded['boundary'];study.restore(snapshots[boundary])
        with np.load(directory/'trace.npz') as z:expected=z['states']
        cursor=boundary*10+1;targets=[];controls=[]
        for control in range(boundary,study.total):
            if control!=boundary:study.admit(control)
            result=study.step(control,recorded['target_rad'] if control==boundary else None)
            actual=np.asarray(result['physical']);count=len(actual)
            if count:np.testing.assert_array_equal(actual,expected[cursor:cursor+count])
            cursor+=count;targets.append(result['target']);controls.append(control)
            if result['failure']:break
        assert cursor==len(expected) and result['failure']==recorded['failure']
        np.savez_compressed(args.output/(directory.name+'_commands.npz'),
            control=controls,target=targets,last_command_rejected=bool(result['failure']))
        evidence=dict(candidate=directory.name,target_rad=recorded['target_rad'],
            reproduced_every_physical_step_exactly=True,steps_after_intervention=cursor-boundary*10-1,
            terminal_diagnostic=result['diagnostic'],failure=result['failure'])
        save_report(args.output/(directory.name+'.json'),evidence);results.append(evidence)
    save_report(args.output/'report.json',dict(repeated_same_candidates_only=True,cases=results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True)
    p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-existing',type=Path,help='Reproduce saved local continuations and capture terminal diagnostics')
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    if args.verify_existing:
        verify_saved_continuations(args);return
    study=Study(args.input_run);snapshots={};tail={};states=[np.r_[study.initial,0.]];calls=[];failure=None
    for control in range(study.total):
        study.admit(control)
        snapshots[control]=study.snapshot(control)
        if control>10:snapshots.pop(control-11,None);tail.pop(control-11,None)
        result=study.step(control);tail[control]=result
        calls.append(result['inference_ms']);states.extend(result['physical'])
        if result['failure']:
            failure=result['failure'];break
        if control%500==0:print(json.dumps(dict(stage='fresh_uninterrupted',control=control)),flush=True)
    states=np.asarray(states)
    base_report=study.summarize(states,failure,calls)
    base_report['terminal_diagnostic']=result['diagnostic']
    save_report(args.output/'baseline.json',base_report)
    np.savez_compressed(args.output/'baseline_trace.npz',states=states)
    with (args.output/'coherent_snapshots.local.pkl').open('wb') as stream:pickle.dump(snapshots,stream)
    # Verify unmodified restored continuation across the whole available tail,
    # including features, next target, every integration state and rejection.
    first=min(snapshots);study.restore(snapshots[first]);checks=[]
    for control in range(first,max(snapshots)+1):
        if control!=first:study.admit(control)
        actual=study.step(control);expected=tail[control]
        for key in ('features','target','physical','integration','torques'):
            np.testing.assert_array_equal(np.asarray(actual[key]),np.asarray(expected[key]),err_msg=f'{control}: {key}')
        assert actual['failure']==expected['failure']
        checks.append(dict(control=control,exact=True,physics_steps=len(actual['physical'])))
    save_report(args.output/'restoration_verification.json',dict(passed=True,cases=checks,
        actual_controller_history=True,logical_timestamps=True,admitted_packets=True,
        pending_delay_schedule=True,integration_state_and_warmstart=True))
    if failure is None or failure['kind']!='preview_rejected':
        save_report(args.output/'report.json',dict(baseline=base_report,continuations=[],
            reason='Fresh run did not terminate at a preview rejection; no automatic new search'))
        return
    # Use three distinct archived 20ms-earlier survivors: best worst margin,
    # positive saturation seed, and greatest legal surviving target.
    archived_failure=json.loads((args.input_run/'preview_rejection.json').read_text())['control']
    search=json.loads((args.search/f'control_{archived_failure-1}'/'report.json').read_text())
    survivors=[r for r in search['candidates'] if r['all_delays_passed'] and r['runtime_fixed_candidate_preview_passed']]
    if not survivors:raise RuntimeError('No earlier survivor available for bounded continuation')
    best=max(survivors,key=lambda r:r['worst_reserve_margin_rad'])
    original_diagnostic={r['control']:r for r in map(json.loads,(args.input_run/'preview_diagnostics.jsonl').read_text().splitlines())}[archived_failure-1]
    j=study.joint;q=original_diagnostic['qpos'][7+j];v=original_diagnostic['qvel'][6+j]
    saturation_seed=q+(study.c['native_effort'][j]+study.c['kd'][j]*v)/study.c['kp'][j]
    chosen=[best,min(survivors,key=lambda r:abs(r['target_rad']-saturation_seed)),max(survivors,key=lambda r:r['target_rad'])]
    chosen=list({r['candidate_id']:r for r in chosen}.values())[:3]
    boundary=failure['control']-1;snapshot=snapshots[boundary];results=[]
    for trial,candidate in enumerate(chosen):
        directory=args.output/f'candidate_{trial}';directory.mkdir()
        target=float(candidate['target_rad']);study.restore(snapshot)
        diagnostic=tail[boundary]['diagnostic'];requested=np.asarray(diagnostic['requested_target']).copy();requested[j]=target
        previews=[rollout(study.model,study.c,snapshot['integration_state'],snapshot['previous_actual_target'],requested,d)[0] for d in range(7)]
        if not all(r['passed'] for r in previews):
            report=dict(target_rad=target,boundary=boundary,new_state_short_preview_passed=False,
                previews=previews,continuation_started=False,reason='Selected target failed new-state seven-delay check')
            save_report(directory/'report.json',report);results.append(report);continue
        physical=list(states[:boundary*10+1]);candidate_calls=[];stop=None;intervention=None
        for control in range(boundary,study.total):
            if control!=boundary:study.admit(control)
            result=study.step(control,intervention=target if control==boundary else None)
            if control==boundary:intervention=result['diagnostic']
            physical.extend(result['physical']);candidate_calls.append(result['inference_ms'])
            if result['failure']:stop=result['failure'];break
            if control%1000==0:print(json.dumps(dict(stage='continuation',trial=trial,control=control)),flush=True)
        physical=np.asarray(physical);report=study.summarize(physical,stop,candidate_calls)
        report.update(target_rad=target,boundary=boundary,archived_candidate_id=candidate['candidate_id'],
            new_state_short_preview_passed=True,previews=previews,intervention=intervention,
            terminal_diagnostic=result['diagnostic'],
            timestamp_selected_counterfactual=True,continuation_started=True,
            subsequent_targets_from_controller=True,restored_unmodified_branch_verified=True)
        np.savez_compressed(directory/'trace.npz',states=physical)
        save_report(directory/'report.json',report);results.append(report)
        print(json.dumps(dict(stage='candidate_finished',trial=trial,target=target,failure=stop,
            physical_seconds=report['physical_seconds'],complete=report['physical_complete'])),flush=True)
    save_report(args.output/'report.json',dict(baseline=base_report,restoration_verified=True,
        new_intervention_boundary=boundary,archived_selection_boundary=archived_failure-1,
        continuations=results,timing_qualified=False,general_live_teleop_qualified=False))


if __name__=='__main__':main()
