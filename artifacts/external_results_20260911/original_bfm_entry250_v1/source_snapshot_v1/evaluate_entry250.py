"""ONE original h8pos1/yaw2 BFM run through canonical initial_entry250."""
import json
import os
import platform
import time
import mujoco
import numpy as np
from student_linear_runtime import *
from evaluate_nominal_pilot import get_state, assess, new_trace
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
from quiet_metrics import standing_windows,quiet_diagnostic

PILOT=BASE.parent/'fast_controller_nominal_pilot_v1'

def main():
    frozen=assert_frozen()
    assert mujoco.__version__=='3.2.3'
    assert np.__version__=='1.26.4'
    assert not (BASE/'entry250').exists(),'Existing run must remain preserved.'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz')
    phase=next(p for p in timeline['phases'] if p['name']=='initial_standing')
    assert phase['control_start']==0 and phase['control_stop']==250 and phase['requested_controls']==250,phase
    prior=archive(PILOT/'zero_parity/trace.npz')
    assert prior['physics_states'].shape==(1001,59)
    assert prior['physics_torque'].shape==(1000,23)
    runtime=LinearStudentRuntime(native,c,original,motion,original29,PILOT/'fit/zero_head.onnx')
    assert runtime.seed.recorded_controls==0
    initial=motion_states(motion)[10]
    data=mujoco.MjData(native)
    data.qpos[:]=initial[:30];data.qvel[:]=initial[30:]
    mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    expected=float(data.time)
    initial_reasons,_=assess(data,c,expected)
    assert not initial_reasons,initial_reasons
    output=BASE/'entry250';output.mkdir(exist_ok=False)
    request=dict(kind='original_BFM_canonical_initial_entry250',clip='walk003',requested_controls=250,
        controller='original frozen h8pos1 yaw2 BFM; residual disabled for all250 controls',
        initial_frame=10,global_control_start=0,global_control_end_exclusive=250,
        source_frames=list(range(11,261)),terminal_yaw4_used=False,learned_head_inference_calls=0,
        prior_action='BFM actor raw output times5, unclipped history, same disabled-head semantics as original zero_parity100',
        original_zero_head_sha256=sha(PILOT/'fit/zero_head.onnx'),
        original_zero_parity_trace_sha256=sha(PILOT/'zero_parity/trace.npz'),
        frozen_receipt_sha256=sha(FROZEN_RECEIPT),model_manifest=manifest,motion_override=override,
        physics_hz=500,control_hz=50,independent_expected_time_increment_seconds=.002,
        physical_statewrites_after_initialization=0,history_resets_after_initialization=0,
        prefix_parity_required_controls=100,prefix_injected_or_replayed=False,
        simulation_privileged_root_pose_velocity=True,hardware_authorized=False,
        optimizer_calls=0,new_expert_queries=0,source_acquisition_or_switch_tested=False,
        process_id=os.getpid(),python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__)
    (output/'request.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
    trace=new_trace(data)
    named_history={key:[] for key in runtime.seed.history.data}
    initial_vector=get_state(native,data)
    failure=None;prefix=None;started=time.perf_counter()
    for control in range(250):
        trace['control_integration_before'].append(get_state(native,data))
        trace['control_history_before'].append(np.concatenate([runtime.seed.history.data[k].reshape(-1) for k in sorted(runtime.seed.history.data)]).copy())
        for key in named_history:named_history[key].append(runtime.seed.history.data[key].copy())
        trace['control_previous_action_before'].append(runtime.seed.previous_action.copy())
        try:
            proposed=runtime.propose(control,data.qpos,data.qvel,terminal=False,disable_head=True)
            np.testing.assert_array_equal(proposed['delta'],np.zeros(23,np.float32))
        except Exception as exc:
            failure=dict(reasons=['policy_inference_fault'],global_control=control,substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc))
            break
        target=proposed['target'];actual_substeps=0
        for sub in range(10):
            data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
            mujoco.mj_step(native,data);expected+=.002;actual_substeps+=1
            for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl),('physics_actuator_torque',data.qfrc_actuator[6:]),('physics_warning_counts',data.warning.number),('physics_warning_lastinfo',data.warning.lastinfo)):
                trace[key].append(value.copy())
            trace['physics_time'].append(float(data.time));trace['physics_expected_time'].append(expected)
            reasons,metrics=assess(data,c,expected)
            for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):trace[key].append(metrics[key])
            if reasons:
                failure=dict(reasons=reasons,global_control=control,substep=sub+1,time=float(data.time),**metrics)
                break
        frame=control+11
        values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),source_frame=frame,global_control=control,
            controller_mode=0,physics_substeps=actual_substeps,
            joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],**proposed)
        for key,value in values.items():trace[key].append(value)
        if control==99 and actual_substeps==10:
            states=np.concatenate((np.asarray(trace['physics_qpos']),np.asarray(trace['physics_qvel'])),axis=1)
            torques=np.asarray(trace['physics_torque'])
            state_equal=np.array_equal(states,prior['physics_states'])
            torque_equal=np.array_equal(torques,prior['physics_torque'])
            prefix=dict(controls=100,physics_steps=1000,all1001physics_states_bitexact=state_equal,
                all1000command_torques_bitexact=torque_equal,
                all101control_boundary_states_bitexact=np.array_equal(states[::10],prior['physics_states'][::10]),
                state_max_abs_difference=float(np.max(np.abs(states-prior['physics_states']))),
                torque_max_abs_difference=float(np.max(np.abs(torques-prior['physics_torque']))),
                passed=state_equal and torque_equal,comparison_only_no_state_or_target_injection=True,
                reference_trace_sha256=sha(PILOT/'zero_parity/trace.npz'))
            (output/'prefix100_parity.json').write_text(json.dumps(prefix,indent=2,allow_nan=False)+'\n')
            if not prefix['passed']:
                failure=dict(reasons=['original_zero100_prefix_mismatch'],global_control=control,substep=10,time=float(data.time))
        if failure:break
    arrays={k:np.asarray(v) for k,v in trace.items()}
    np.savez_compressed(output/'trace.npz',**arrays,initial_integration=initial_vector,
        final_integration=get_state(native,data),integration_state_spec=np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION),dtype=np.int64),
        final_previous_action=runtime.seed.previous_action,final_recorded_controls=np.asarray(runtime.seed.recorded_controls),
        **{'final_history_'+k:v for k,v in runtime.seed.history.data.items()},
        **{'control_history_before_'+k:np.asarray(v) for k,v in named_history.items()})
    complete=int(np.sum(arrays['physics_substeps']==10))
    strict=failure is None
    prefix_pass=bool(prefix and prefix['passed'])
    maximum=lambda key:float(np.max(arrays[key])) if len(arrays[key]) else 0.
    quiet=None
    if len(arrays['physics_torque']) and np.isfinite(arrays['physics_qpos']).all() and np.isfinite(arrays['physics_qvel']).all():
        quiet=quiet_diagnostic(standing_windows(arrays,motion,original,original29),strict and prefix_pass)
    result=dict(kind=request['kind'],requested_controls=250,completed_full_controls=complete,
        attempted_controls=len(arrays['target']),physics_steps=len(arrays['physics_torque']),failure=failure,
        full_segment_completed=complete==250 and strict and prefix_pass,native2ms_strict_pass=strict,
        original_zero100_prefix_parity=prefix,quiet_standing_diagnostic=quiet,
        range_excess_max=maximum('range_excess'),velocity_ratio_max=maximum('velocity_ratio'),effort_ratio_max=maximum('effort_ratio'),
        independent_accumulated_clock_max_error=maximum('clock_error'),
        engine_warning_counts=data.warning.number.tolist(),engine_warning_lastinfo=data.warning.lastinfo.tolist(),
        policy_ms_p50_p95_max=np.percentile(arrays['inference_ms'],[50,95,100]).tolist() if len(arrays['inference_ms']) else [],
        policy_20ms_deadline_misses=int(np.sum(arrays['inference_ms']>20)),
        simulation_start_time=0.,simulation_end_time=float(data.time),elapsed_seconds=time.perf_counter()-started,
        source_controls=0,source_acquisition_or_switch_tested=False,learned_head_inference_calls=0,
        full_body_teleoperation_qualified=False,hardware_authorized=False,
        trace_sha256=sha(output/'trace.npz'),request_sha256=sha(output/'request.json'))
    (output/'report.json').write_text(json.dumps(finite_json(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(finite_json(result)),flush=True)

if __name__=='__main__':main()
