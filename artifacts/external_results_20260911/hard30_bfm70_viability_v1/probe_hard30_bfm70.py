"""Saved hard30 plan followed by BFM70, continuous actual3740 native physics."""
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).parent
SNAPSHOT=BASE.parent/'recovery_probe_v1/source_snapshot'
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
RECOVERY=BASE.parent/'recovery_probe_v1'
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
ONNX=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
DEPS=Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')


def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def main():
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    started=time.perf_counter()
    source=archive(SOURCE/'trace.npz')
    hard_path=BASE.parent/'hard_feasibility_3740_v1/trace.npz'
    hard=archive(hard_path)
    hard_report=json.loads((hard_path.parent/'report.json').read_text())
    actual=archive(RECOVERY/'actual_3740_integration_state.npz')
    memory=archive(RECOVERY/'actual_3740_bfm_history.npz')
    source_report=json.loads((SOURCE/'report.json').read_text())
    assert source_report['trace_sha256']==sha256(SOURCE/'trace.npz')
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'pico')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'pico',native,c,original,timeline,manifest)
    kp,kd,effort,speed=[np.asarray(c[k]) for k in ('kp','kd','native_effort','native_velocity')]
    lo,hi=np.asarray(c['joint_limits']).T
    seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    cases=[]
    for mode,control,horizon,prefix_count in [('3740_hard30_bfm70',3740,100,30)]:
        spec=mujoco.mjtState(int(source['integration_state_spec'] if control==3770 else actual['state_spec']))
        vector=source['final_integration'] if control==3770 else actual['state_vector']
        live=mujoco.MjData(native)
        mujoco.mj_setState(native,live,vector,spec);mujoco.mj_forward(native,live);mujoco.mj_setState(native,live,vector,spec)
        initial_time=float(live.time)
        np.testing.assert_array_equal(live.qpos,source['qpos'][-1] if control==3770 else actual['qpos'])
        np.testing.assert_array_equal(live.qvel,source['qvel'][-1] if control==3770 else actual['qvel'])
        for k in seed.history.data:seed.history.data[k][:]=source['final_history_'+k] if control==3770 else memory['memory_'+k]
        seed.previous_action=(source['final_previous_action'] if control==3770 else memory['previous_action']).copy()
        seed.recorded_controls=control
        seed.actual_action_max_abs=float(memory['actual_action_max_abs']);seed.actual_action_components_outside_five=int(memory['actual_action_components_outside_five'])
        if control==3770:
            normalized=((source['target']-np.asarray(c['default_q']))*kp/(.25*np.asarray(c['training_effort']))).astype(np.float32)
            seed.actual_action_max_abs=max(seed.actual_action_max_abs,float(np.max(np.abs(normalized))))
            seed.actual_action_components_outside_five+=int(np.sum(np.abs(normalized)>5))
        rows={k:[] for k in ('state','target','previous_action','history','physics_qpos','physics_qvel','physics_torque','physics_actuator_force','physics_time','warning_counts','warning_lastinfo','range_excess','speed_ratio','effort_ratio')}
        rows['state'].append(np.r_[live.qpos,live.qvel]);rows['physics_qpos'].append(live.qpos.copy());rows['physics_qvel'].append(live.qvel.copy());rows['physics_time'].append(float(live.time))
        failure=None
        for local in range(horizon):
            action=seed.previous_action.copy();rows['previous_action'].append(action)
            if local<prefix_count:
                target=hard['planned_targets'][local]
                history=np.concatenate([seed.history.data[k].reshape(-1) for k in sorted(seed.history.data)])
                seed.record_control(control+local,live.qpos,live.qvel,target)
            else:
                sensed,terms=seed._terms(live.qpos,live.qvel,action)
                history=seed.history.before_update(terms)
                z=seed._goal(control+local+11,live.qpos)
                raw=seed.sessions['actor'].run(None,dict(state=sensed[None],last_action=action[None],history=history[None],z=z))[0][0]*5
                target=np.clip(np.asarray(c['default_q'])+raw*.25*np.asarray(c['training_effort'])/kp,lo,hi)
                seed.previous_action=raw.copy();seed.recorded_controls+=1
            rows['history'].append(history.copy());rows['target'].append(target.copy())
            for sub in range(10):
                live.ctrl[:]=np.clip(kp*(target-live.qpos[7:])-kd*live.qvel[6:],-effort,effort)
                mujoco.mj_step(native,live)
                excess=float(np.maximum(0,np.maximum(lo-live.qpos[7:],live.qpos[7:]-hi)).max())
                velocity=float(np.max(np.abs(live.qvel[6:])/speed));force=float(np.max(np.abs(live.qfrc_actuator[6:])/effort))
                values=dict(physics_qpos=live.qpos.copy(),physics_qvel=live.qvel.copy(),physics_torque=live.ctrl.copy(),physics_actuator_force=live.qfrc_actuator[6:].copy(),physics_time=float(live.time),warning_counts=live.warning.number.copy(),warning_lastinfo=live.warning.lastinfo.copy(),range_excess=excess,speed_ratio=velocity,effort_ratio=force)
                for k,v in values.items():rows[k].append(v)
                expected=initial_time+(local*10+sub+1)*.002
                tilt=float(np.arccos(np.clip(1-2*np.sum(live.qpos[4:6]**2),-1,1)))
                if not np.isfinite(live.qpos).all() or not np.isfinite(live.qvel).all() or np.any(live.warning.number) or abs(live.time-expected)>1e-8:
                    failure=dict(kind='engine_or_clock',control=local,substep=sub+1)
                elif excess>1e-6 or velocity>1 or force>1+1e-9 or live.qpos[2]<.25 or tilt>1.2:
                    bad=np.flatnonzero(np.maximum(lo-live.qpos[7:],live.qpos[7:]-hi)>1e-6)
                    failure=dict(kind='physical_feasibility',control=local,global_control=control+local,substep=sub+1,range_excess=excess,speed_ratio=velocity,effort_ratio=force,height=float(live.qpos[2]),tilt=tilt,
                        joints=[dict(name=c['joint_names'][i],q=float(live.qpos[7+i]),lower=float(lo[i]),upper=float(hi[i]),target=float(target[i])) for i in bad])
                if failure:break
            mujoco.mj_kinematics(native,live);rows['state'].append(np.r_[live.qpos,live.qvel])
            if failure:break
        arrays={k:np.asarray(v) for k,v in rows.items()};steps=len(arrays['physics_torque']);full=steps==horizon*10 and failure is None
        first30_cost=None
        if steps>=300:
            planner=Native23Tracker(position_servo_copy(native,kp,kd,effort),c,motion,horizon=30,threads=1,all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=400)
            planner.window(control+10);f=planner.features(arrays['state'][:31]);r=planner.residual(np.arange(31),f)
            first30_cost=float(np.sum(r**2)+planner.control_weight*np.sum((arrays['target'][:30]-planner.target_reference(np.arange(30)))**2))
        np.testing.assert_array_equal(arrays['target'][:min(prefix_count,len(arrays['target']))],hard['planned_targets'][:min(prefix_count,len(arrays['target']))])
        path=BASE/(mode+'.npz');np.savez_compressed(path,**arrays,initial_integration=vector,integration_state_spec=np.asarray(int(spec),dtype=np.int64))
        result=dict(mode=mode,initial_control=control,requested_controls=horizon,retained_shifted_prefix=prefix_count,
            full_horizon_physical_pass=full,physics_steps=steps,full_controls=steps//10,partial_substeps=steps%10,failure=failure,
            first30_control_same_v4_allmargin_relativefoot_cost=first30_cost,
            range_excess_max=float(arrays['range_excess'].max()),speed_ratio_max=float(arrays['speed_ratio'].max()),effort_ratio_max=float(arrays['effort_ratio'].max()),
            warning_counts=arrays['warning_counts'].max(axis=0).tolist(),clock_max_error=float(np.max(np.abs(arrays['physics_time']-(initial_time+np.arange(steps+1)*.002)))),
            trace_sha256=sha256(path),source_seconds_end=(control-350)*.02+steps*.002,
            saved_hard_target_prefix_bitexact=True,
            hard_report_first5_physics_max_delta=float(np.max(np.abs(np.c_[arrays['physics_qpos'][:51],arrays['physics_qvel'][:51]]-hard['physics_states'][:min(51,len(arrays['physics_qpos']))]))) if steps>=50 else None)
        cases.append(result);print(json.dumps(result),flush=True)
    result=dict(kind='saved_hard30_then_original_BFM70_terminal_viability_witness',cases=cases,source_goal_unchanged=True,no_optimizer=True,
        no_physics_statewrites_after_initialization=True,no_full_source_qualification=True,
        history='actual normalized prior targets until BFM takeover; subsequent BFM predictions retain own rawactor*5 prioraction',
        first30_cost_scope='unchanged H30 objective only, no comparison of600ms and2s totals',
        elapsed_seconds=time.perf_counter()-started,seed_identity=seed.identity(),
        hashes={str(p):sha256(p) for p in (Path(__file__),hard_path,hard_path.parent/'report.json',SOURCE/'trace.npz',SOURCE/'report.json',RECOVERY/'actual_3740_integration_state.npz',RECOVERY/'actual_3740_bfm_history.npz',REFERENCE,BUNDLE/'contract.json')})
    with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(done=True,seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
