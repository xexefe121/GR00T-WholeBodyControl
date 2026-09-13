"""Two fixed, bounded H30 tail completions at actual control3770; no optimizer."""
import copy
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).parent
SNAPSHOT = BASE.parent/'recovery_probe_v1/source_snapshot'
sys.path.insert(0,str(SNAPSHOT))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,Native23Tracker,sha256
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed

ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
SOURCE=BASE.parent/'hard_feasibility_continuation_3740_v1'
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
ONNX=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
DEPS=Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
CONTROL=3770


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    started=time.perf_counter()
    with np.load(SOURCE/'trace.npz',allow_pickle=False) as a:
        source={k:a[k].copy() for k in a.files}
    producer=json.loads((SOURCE/'report.json').read_text())
    assert producer['trace_sha256']==sha256(SOURCE/'trace.npz') and producer['final_control']==CONTROL
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'pico')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'pico',native,c,original,timeline,manifest)
    kp,kd,effort,speed=[np.asarray(c[k]) for k in ('kp','kd','native_effort','native_velocity')]
    lo,hi=np.asarray(c['joint_limits']).T
    spec=mujoco.mjtState(int(source['integration_state_spec']))
    initial=mujoco.MjData(native)
    mujoco.mj_setState(native,initial,source['final_integration'],spec)
    mujoco.mj_forward(native,initial)
    mujoco.mj_setState(native,initial,source['final_integration'],spec)
    np.testing.assert_array_equal(initial.qpos,source['qpos'][-1])
    np.testing.assert_array_equal(initial.qvel,source['qvel'][-1])
    prefix=source['final_warm_targets'][:25].copy()
    assert prefix.shape==(25,23)
    seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    with np.load(BASE.parent/'recovery_probe_v1/actual_3740_bfm_history.npz',allow_pickle=False) as a:
        prior_max=float(a['actual_action_max_abs']);prior_outside=int(a['actual_action_components_outside_five'])
    normalized=((source['target']-np.asarray(c['default_q']))*kp/(.25*np.asarray(c['training_effort']))).astype(np.float32)
    max_actual=max(prior_max,float(np.max(np.abs(normalized))))
    count_actual=prior_outside+int(np.sum(np.abs(normalized)>5))
    planner=Native23Tracker(position_servo_copy(native,kp,kd,effort),c,motion,horizon=30,threads=1,
        all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=400)
    planner.window(CONTROL+10)
    cases=[];prefix_golden=None
    for mode in ('hold_last_accepted_target','bfm_original_goal_tail'):
        live=copy.deepcopy(initial)
        for k in seed.history.data:seed.history.data[k][:]=source['final_history_'+k]
        seed.previous_action=source['final_previous_action'].copy();seed.recorded_controls=CONTROL
        seed.actual_action_max_abs=max_actual;seed.actual_action_components_outside_five=count_actual
        rows={k:[] for k in ('state','target','previous_action','history','physics_qpos','physics_qvel','physics_torque','physics_actuator_force','physics_time','warning_counts','warning_lastinfo','range_excess','speed_ratio','effort_ratio')}
        rows['state'].append(np.r_[live.qpos,live.qvel]);rows['physics_qpos'].append(live.qpos.copy());rows['physics_qvel'].append(live.qvel.copy());rows['physics_time'].append(float(live.time))
        failure=None
        for local in range(30):
            action=seed.previous_action.copy()
            rows['previous_action'].append(action)
            if local<25 or mode=='hold_last_accepted_target':
                target=prefix[local] if local<25 else prefix[-1]
                history=np.concatenate([seed.history.data[k].reshape(-1) for k in sorted(seed.history.data)])
                # Known commanded prefix is recorded as applied target normalization.
                seed.record_control(CONTROL+local,live.qpos,live.qvel,target)
            else:
                state,terms=seed._terms(live.qpos,live.qvel,action)
                history=seed.history.before_update(terms)
                z=seed._goal(CONTROL+local+11,live.qpos)
                raw=seed.sessions['actor'].run(None,dict(state=state[None],last_action=action[None],history=history[None],z=z))[0][0]*5
                target=np.clip(np.asarray(c['default_q'])+raw*.25*np.asarray(c['training_effort'])/kp,lo,hi)
                # Local BFM rollout retains its raw actor*5 action convention.
                seed.previous_action=raw.copy();seed.recorded_controls+=1
            rows['history'].append(history.copy());rows['target'].append(target.copy())
            for sub in range(10):
                live.ctrl[:]=np.clip(kp*(target-live.qpos[7:])-kd*live.qvel[6:],-effort,effort)
                mujoco.mj_step(native,live)
                excess=float(np.maximum(0,np.maximum(lo-live.qpos[7:],live.qpos[7:]-hi)).max())
                velocity=float(np.max(np.abs(live.qvel[6:])/speed));force=float(np.max(np.abs(live.qfrc_actuator[6:])/effort))
                values=dict(physics_qpos=live.qpos.copy(),physics_qvel=live.qvel.copy(),physics_torque=live.ctrl.copy(),physics_actuator_force=live.qfrc_actuator[6:].copy(),physics_time=float(live.time),warning_counts=live.warning.number.copy(),warning_lastinfo=live.warning.lastinfo.copy(),range_excess=excess,speed_ratio=velocity,effort_ratio=force)
                for k,v in values.items():rows[k].append(v)
                expected=initial.time+(local*10+sub+1)*.002
                tilt=float(np.arccos(np.clip(1-2*np.sum(live.qpos[4:6]**2),-1,1)))
                if not np.isfinite(live.qpos).all() or not np.isfinite(live.qvel).all() or np.any(live.warning.number) or abs(live.time-expected)>1e-8:
                    failure=dict(kind='engine_or_clock',control=local,substep=sub+1)
                elif excess>1e-6 or velocity>1 or force>1+1e-9 or live.qpos[2]<.25 or tilt>1.2:
                    failed_joints=np.flatnonzero(np.maximum(lo-live.qpos[7:],live.qpos[7:]-hi)>1e-6)
                    failure=dict(kind='physical_feasibility',control=local,substep=sub+1,range_excess=excess,speed_ratio=velocity,effort_ratio=force,height=float(live.qpos[2]),tilt=tilt,
                        joints=[dict(name=c['joint_names'][i],q=float(live.qpos[7+i]),lower=float(lo[i]),upper=float(hi[i]),target=float(target[i])) for i in failed_joints])
                if failure:break
            mujoco.mj_kinematics(native,live)
            rows['state'].append(np.r_[live.qpos,live.qvel])
            if local==24:
                state25=np.r_[live.qpos,live.qvel]
                if prefix_golden is None:prefix_golden=state25.copy()
                else:np.testing.assert_array_equal(prefix_golden,state25)
            if failure:break
        arrays={k:np.asarray(v) for k,v in rows.items()}
        steps=len(arrays['physics_torque']);full=steps==300 and failure is None
        cost=None
        if full:
            features=planner.features(arrays['state']);residual=planner.residual(np.arange(31),features)
            cost=float(np.sum(residual**2)+planner.control_weight*np.sum((arrays['target']-planner.target_reference(np.arange(30)))**2))
        path=BASE/(mode+'.npz');np.savez_compressed(path,**arrays,initial_integration=source['final_integration'],integration_state_spec=source['integration_state_spec'])
        result=dict(mode=mode,initial_control=CONTROL,horizon=30,full_horizon_physical_pass=full,physics_steps=steps,full_controls=steps//10,partial_substeps=steps%10,failure=failure,
            same_v4_allmargin_relativefoot_objective_cost=cost,range_excess_max=float(arrays['range_excess'].max()),speed_ratio_max=float(arrays['speed_ratio'].max()),effort_ratio_max=float(arrays['effort_ratio'].max()),
            warning_counts=arrays['warning_counts'].max(axis=0).tolist(),warning_lastinfo=arrays['warning_lastinfo'].max(axis=0).tolist(),
            clock_max_error=float(np.max(np.abs(arrays['physics_time']-(initial.time+np.arange(steps+1)*.002)))),
            trace_sha256=sha256(path),unchanged_prefix25_completed=steps>=250)
        cases.append(result);print(json.dumps(result),flush=True)
    result=dict(kind='bounded_actual3770_two_tail_completion_witnesses',cases=cases,initial_control=CONTROL,
        preserved_shifted_prefix_controls=25,replaced_direct_reference_tail_controls=5,
        prefix_action_history='preceding applied target normalization, no clipping to ±5; exact final actual3770 history restored',
        bfm_tail_action_history='starts from normalized prefix action/history then rawactor*5 for its own5predictedcontrols',
        original_goal_clock='global control+11, horizon8 position1/yaw2, original source untouched',
        physics_initialization='fullactualfinal_integration, setState-forward-setState beforefirststep; no subsequentphysicalstatewrites',
        cumulative_actual_action_max_abs=max_actual,cumulative_actual_action_components_outside_five=count_actual,
        no_optimizer=True,no_full_source_or_recovery_qualification=True,elapsed_seconds=time.perf_counter()-started,
        seed_identity=seed.identity(),hashes={str(p):sha256(p) for p in (Path(__file__),SOURCE/'trace.npz',SOURCE/'report.json',SOURCE/'request.json',REFERENCE,BUNDLE/'contract.json')})
    write(BASE/'report.json',result)
    print(json.dumps(dict(done=True,seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
