"""Fresh native23 closed-loop rollout of a motion-specific compiled feedback law.

This is a prepared-motion baseline, not a general live teleoperation policy.
No optimizer or saved actual plant state is used by the feedback controller.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
SOURCE=NEW/'direct_target_width251_evaluation_v2/source_draft_v1'
sys.path.insert(0,str(SOURCE))
from direct_runtime import DirectStudentRuntime
from evaluate_direct_target_student import assess,get_state,source_metrics
from runtime_common import BUNDLE,REFERENCE,TEACHER,archive,sha,finite_json
from quiet_metrics import standing_windows,quiet_diagnostic
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
import mujoco
sys.path.insert(0,str(NEW/'direct_target_width251_collection_v1/source_draft_v1'))
from collection_math import difference_function,committed_target

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,r):
    with p.open('x',encoding='utf-8') as f:json.dump(finite_json(r),f,indent=2,allow_nan=False);f.write('\n')

class FeedbackController:
    def __init__(self,base,package,difference,limits):
        self.base=base;self.package=package;self.difference=difference;self.limits=limits
        self.feedback_calls=0
    def step(self,control,qpos,qvel):
        if not 251<=control<1269:
            proposal=self.base.propose(control,qpos,qvel);self.base.commit(proposal)
            return proposal['target'],0.0
        seed=self.base.seed
        assert seed.recorded_controls==control and self.base.pending is None
        i=control-251
        target,raw,correction,_=committed_target(self.difference,self.package['nominal_states'][i],
            self.package['feedforward_targets'][i],self.package['feedback_gains'][i],qpos,qvel,self.limits)
        seed.record_control(control,qpos,qvel,target)
        self.feedback_calls+=1
        return target,float(np.max(np.abs(correction)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    assert sys.platform!='win32' and np.__version__=='1.26.4' and mujoco.__version__=='3.2.3'
    manifest=read(BASE/'controller_manifest.json');assert manifest['compiled'] is True
    assert sha(BASE/'controller.npz')==manifest['controller_sha256']
    assert sha(NEW/'direct_target_width251_collection_v1/source_draft_v1/collection_math.py')==manifest['collection_math_sha256']
    core=NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
    assert sha(core)==manifest['core_sha256']
    package=archive(BASE/'controller.npz');difference=difference_function(core)
    native,c,original,timeline,model_manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,model_manifest)
    original29=archive(BUNDLE/'walk003/original29.npz');intent_audit=read(TEACHER/'recorded_source_audit_v2.json')
    old_fit=NEW/'direct_target_causal_width512_student_v1/fit'
    head=old_fit/'student_head.onnx';assert sha(head)=='8a4c394b8836dd763d6eb1c5beac73bf49921bcd59fc5f04850b377d0e843044'
    norm=archive(old_fit/'shared/normalization.npz')
    base=DirectStudentRuntime.from_native(native,c,original,motion,original29,head,norm['joint_span'],timeline,'causal',norm['context_mean'])
    runtime=FeedbackController(base,package,difference,np.asarray(c['joint_limits']))
    initial=motion_states(motion)[10];data=mujoco.MjData(native)
    data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
    with np.load(NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz') as fixture:
        np.testing.assert_array_equal(get_state(native,data),fixture['state_vector'])
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    expected=float(data.time);assert not assess(data,c,expected)[0]
    write(a.output/'request.json',dict(controller='BFM startup; original81000 head at250; compiled state feedback251..1268; BFM terminal and hold',
        original_timing=True,requested_main_controls=1569,conditional_hold_controls=250,
        physics_hz=500,policy_hz=50,offline_planning_required=True,motion_specific=True,
        simulation_privileged_root_pose_velocity=True,arbitrary_live_motion_supported=False,
        physical_statewrites_after_initialization=0,root_forces=False,source_frames_removed=0,
        controller_sha256=manifest['controller_sha256'],head_sha256=sha(head),source_sha256=sha(__file__),
        motion_override=override,model_manifest=model_manifest,hardware_authorized=False))
    progress=a.output/'progress.json'
    def segment(name,start,count):
        nonlocal expected
        dest=a.output/name;dest.mkdir(exist_ok=False)
        initial_state=get_state(native,data);segment_time=float(data.time)
        shapes=dict(qpos=(count+1,30),qvel=(count+1,29),target=(count,23),
            physics_qpos=(count*10+1,30),physics_qvel=(count*10+1,29),physics_torque=(count*10,23),
            physics_actuator_torque=(count*10,23),physics_time=(count*10+1,),physics_expected_time=(count*10+1,),
            physics_warning_counts=(count*10+1,8),physics_warning_lastinfo=(count*10+1,8),
            source_frame=(count,),global_control=(count,),physics_substeps=(count,),controller_mode=(count,),
            inference_ms=(count,),control_loop_ms=(count,),feedback_correction_max=(count,),
            range_excess=(count*10,),velocity_ratio=(count*10,),effort_ratio=(count*10,),clock_error=(count*10,))
        integers={'source_frame','global_control','physics_substeps','controller_mode','physics_warning_counts','physics_warning_lastinfo'}
        arrays={k:np.empty(shape,np.int64 if k in integers else np.float64) for k,shape in shapes.items()}
        for key in ('qpos','physics_qpos'):arrays[key][0]=data.qpos
        for key in ('qvel','physics_qvel'):arrays[key][0]=data.qvel
        arrays['physics_time'][0]=data.time;arrays['physics_expected_time'][0]=expected
        arrays['physics_warning_counts'][0]=data.warning.number;arrays['physics_warning_lastinfo'][0]=data.warning.lastinfo
        failure=None;issued=steps=0;started=time.perf_counter()
        for local_control,control in enumerate(range(start,start+count)):
            tick=time.perf_counter()
            try:target,correction=runtime.step(control,data.qpos,data.qvel)
            except Exception as error:
                failure=dict(kind='policy_fault',control=control,error=repr(error));break
            arrays['inference_ms'][local_control]=(time.perf_counter()-tick)*1000
            arrays['target'][local_control]=target;arrays['feedback_correction_max'][local_control]=correction
            arrays['global_control'][local_control]=control
            arrays['source_frame'][local_control]=min(control+11,len(motion['joint_pos'])-1)
            arrays['controller_mode'][local_control]=0 if control<250 else 1 if control==250 else 3 if control<1269 else 2
            actual_substeps=0
            for sub in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(native,data);expected+=.002;actual_substeps+=1
                for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_time',data.time),
                    ('physics_expected_time',expected),('physics_warning_counts',data.warning.number),('physics_warning_lastinfo',data.warning.lastinfo)):
                    arrays[key][steps+1]=value
                arrays['physics_torque'][steps]=data.ctrl;arrays['physics_actuator_torque'][steps]=data.qfrc_actuator[6:]
                reasons,metrics=assess(data,c,expected)
                for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):arrays[key][steps]=metrics[key]
                steps+=1
                if reasons:
                    failure=dict(kind='native_fault',control=control,substep=sub+1,time=float(data.time),reasons=reasons,**metrics);break
            arrays['physics_substeps'][local_control]=actual_substeps
            arrays['qpos'][local_control+1]=data.qpos;arrays['qvel'][local_control+1]=data.qvel
            issued+=1
            if control%100==0 or failure:
                temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(dict(control=control,simulation_time=float(data.time),failure=failure)));temp.replace(progress)
                print(json.dumps(dict(control=control,simulation_time=float(data.time),failure=failure)),flush=True)
            arrays['control_loop_ms'][local_control]=(time.perf_counter()-tick)*1000
            if failure:break
        state_fields={'physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'}
        step_fields={'physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error'}
        for key in arrays:
            size=steps+1 if key in state_fields else steps if key in step_fields else issued+1 if key in ('qpos','qvel') else issued
            arrays[key]=arrays[key][:size]
        complete=failure is None and issued==count and steps==count*10
        np.savez_compressed(dest/'trace.npz',**arrays,initial_integration=initial_state,final_integration=get_state(native,data),integration_state_spec=np.asarray(8191))
        quiet=quiet_diagnostic(standing_windows(arrays,motion,original,original29),True) if complete else None
        metrics=source_metrics(native,arrays,motion,original29,intent_audit,timeline) if issued else {}
        def timing(key):return np.percentile(arrays[key],[50,95,100]).tolist() if issued else []
        result=dict(full_segment_completed=complete,requested_controls=count,completed_controls=int(np.sum(arrays['physics_substeps']==10)),
            physics_steps=steps,failure=failure,source_metrics=metrics,quiet_standing_diagnostic=quiet,
            policy_ms_p50_p95_max=timing('inference_ms'),policy_20ms_deadline_misses=int(np.sum(arrays['inference_ms']>20)),
            control_loop_ms_p50_p95_max=timing('control_loop_ms'),control_loop_20ms_deadline_misses=int(np.sum(arrays['control_loop_ms']>20)),
            inference_counts=dict(base.counts),feedback_calls=runtime.feedback_calls,
            simulation_start_time=segment_time,simulation_end_time=float(data.time),elapsed_seconds=time.perf_counter()-started,
            maximum_speed_ratio=float(np.max(arrays['velocity_ratio'])) if steps else None,
            trace_sha256=sha(dest/'trace.npz'),motion_specific=True,real_time_paced=False,live_teleoperation_qualified=False)
        write(dest/'report.json',result);return result
    main_result=segment('nominal',0,1569)
    hold_result=segment('post_lifecycle_hold_5s',1569,250) if main_result['full_segment_completed'] else dict(full_segment_completed=False,not_run_reason='main motion failed')
    result=dict(main=main_result,hold=hold_result,full_motion_and_hold_completed=main_result['full_segment_completed'] and hold_result['full_segment_completed'],
        live_teleoperation_qualified=False,hardware_authorized=False,offline_motion_specific_controller=True)
    write(a.output/'report.json',result)
    print(json.dumps(dict(full_motion_and_hold_completed=result['full_motion_and_hold_completed'],output=str(a.output))),flush=True)
if __name__=='__main__':main()
