"""Native3.2.3 complete received-motion rollout with optional short lookahead."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
# Reuse the pinned native interpreter's ancillary dependency setup.
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,assess,metrics,quiet
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--library',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--clip',choices=['walk002','walk003','pico','walk008'],default='walk002')
    ap.add_argument('--steps',type=int,default=30);ap.add_argument('--controls',type=int);ap.add_argument('--initial-vx',type=float,default=0)
    ap.add_argument('--feedback',action='store_true',help='Retain native factory balance throughout candidate predictions')
    ap.add_argument('--response',action='store_true',help='Differentiate all23 targets through native contact dynamics')
    ap.add_argument('--response-preserve-root',action='store_true')
    ap.add_argument('--planner',action='store_true',help='Bounded continuous search through longer factory-feedback predictions')
    ap.add_argument('--planner-checkpoint',type=Path,help='Matching learned checkpoint to retain its complete feedback during prediction')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.response and (a.feedback or not a.library):ap.error('Response servo requires its own --library and excludes --feedback')
    if a.response_preserve_root and not a.response:ap.error('Base preservation requires --response')
    if a.planner and (a.feedback or a.response or not a.library):ap.error('Planner requires its own --library and excludes other prediction modes')
    if a.planner_checkpoint and not a.planner:ap.error('A planner checkpoint requires --planner')
    model,c,motion,original,timeline=load_case(a.clip)
    bank=NEW/'causal_dynamics_v1/bank';meta=json.loads((bank/'bank.json').read_text())
    initial=archive(bank/(a.clip+'.npz'))['states'][10].copy();initial[30]+=a.initial_vx
    d=mujoco.MjData(model);d.qpos[:]=initial[:30];d.qvel[:]=initial[30:];mujoco.mj_forward(model,d)
    controller=LocomotionConditionedController(a.actor,NEW/'onboard_factory_firmware_v1',c,model=model,tasks=meta['tasks'],
        standing_qpos=timeline['configured_standing_qpos'],now=-.22,lookahead_library=None if a.response or a.planner else a.library,
        lookahead_steps=a.steps,lookahead_feedback=a.feedback,response_library=a.library if a.response else None,
        target_filter_alpha=.9 if a.response or a.planner else 1.,response_preserve_root=a.response_preserve_root,
        planner_library=a.library if a.planner else None,planner_checkpoint=a.planner_checkpoint)
    predictor=controller.planner if a.planner else controller.response if a.response else controller.lookahead
    def receive(sequence,now):
        fields={k:v[sequence] for k,v in motion.items() if k!='fps'}
        assert controller.receive(Packet(0,sequence,sequence*.02,fields,sequence==len(motion['joint_pos'])-1),
            original['source_task_position_w'][sequence],original['source_task_quaternion_wxyz'][sequence],now)
    for sequence in range(11):receive(sequence,(sequence-11)*.02)
    total=timeline['total_requested_controls']+1500;count=a.controls or total
    trace={k:[] for k in ('qpos','qvel','target','frame','control','inference_ms','physics_qpos','physics_qvel')}
    reports=[];failure=None;max_speed=max_effort=0.;expected=0.;started=time.monotonic()
    for control in range(count):
        tick=time.perf_counter_ns();sequence=control+11
        if sequence<len(motion['joint_pos']):receive(sequence,control*.02)
        command=controller.command(d.qpos,d.qvel,control*.02);target=command.targets
        controller.commit_applied(d.qpos,d.qvel,target)
        trace['inference_ms'].append((time.perf_counter_ns()-tick)*1e-6)
        if predictor:reports.append(dict(control=control,**predictor.status))
        trace['target'].append(target.copy());trace['control'].append(control);trace['frame'].append(min(sequence,len(motion['joint_pos'])-1))
        for substep in range(10):
            band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2)
            penetration=d.qpos[7:]-np.clip(d.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
            outward=np.where(penetration*d.qvel[6:]>0,d.qvel[6:],0)
            d.ctrl[:]=np.clip(controller.kp*(target-d.qpos[7:])-controller.kd*d.qvel[6:]-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
            mujoco.mj_step(model,d);expected+=.002
            trace['physics_qpos'].append(d.qpos.copy());trace['physics_qvel'].append(d.qvel.copy())
            reasons,values=assess(d,c,expected);max_speed=max(max_speed,values['speed_ratio']);max_effort=max(max_effort,values.get('effort_ratio',0))
            if reasons:failure=dict(time=float(d.time),reasons=reasons,**values);break
        trace['qpos'].append(d.qpos.copy());trace['qvel'].append(d.qvel.copy())
        if control%250==0:print(json.dumps(dict(control=control,count=count,seconds=float(d.time),lookahead=reports[-1] if reports else None)),flush=True)
        if failure:break
    trace={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(a.output/'trace.npz',**trace)
    source=metrics(model,trace['qpos'],trace['frame'],trace['control'],motion,original,timeline,meta['tasks'])
    hold=quiet(trace['physics_qpos'],trace['physics_qvel'],original['source_qpos29'][-1])
    report=dict(policy='received_native23_short_physics_lookahead',actor=str(a.actor),clip=a.clip,
        simulation_seconds=float(d.time),requested_seconds=count*.02,physical_complete=failure is None,
        complete_requested_lifecycle=failure is None and count>=total,failure=failure,source=source,quiet=hold,
        max_speed_ratio=max_speed,max_effort_ratio=max_effort,inference_ms_p50_p95_max=np.percentile(trace['inference_ms'],[50,95,100]).tolist(),
        wall_seconds=time.monotonic()-started,future_reference_frames=0,independent_realtime=False,hardware_commands=False,
        lookahead_steps=a.steps if a.library else 0,lookahead_candidates=8 if a.library else 0,
        reference_prediction='constant velocity from current/past received poses; no recorded suffix',packet_report=controller.receiver.gate.epoch_report(),
        passed=bool(failure is None and count>=total and source['passed'] and hold['passed']),general_live_teleop_qualified=False)
    if a.feedback:report['policy']='received_native23_factory_feedback_prediction'
    if a.response:
        report.update(policy='received_native23_response_servo',differentiated_targets=23,prediction_queries=52,
            lookahead_candidates=5,leg_target_filter_alpha=.9,preserve_factory_base_motion=a.response_preserve_root)
    if a.planner:
        report.update(policy='received_native23_factory_feedback_planner',lookahead_candidates=32,
            generations_per_plan=2,plan_interval_seconds=.1,leg_target_filter_alpha=.9,
            synchronous_planning_diagnostic=True,independent_realtime=False)
        report['planner_learned_checkpoint']=str(a.planner_checkpoint) if a.planner_checkpoint else None
    if failure is None and count>=total:
        report['continuous_quiet30']=quiet(trace['physics_qpos'],trace['physics_qvel'],original['source_qpos29'][-1],seconds=30)
        report['passed']=bool(report['passed'] and report['continuous_quiet30']['passed'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'lookahead.json').write_text(json.dumps(reports))
    if predictor:predictor.close()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
