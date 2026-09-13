"""One causal constant-last-actual-target H30 witness at actual control3805."""
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
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms

ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
SOURCE=BASE.parent/'hard_feasibility_restoration_continuation_3740_v1'
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
CONTROL=3805
H=30


def main():
    started=time.perf_counter()
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    with np.load(SOURCE/'trace.npz',allow_pickle=False) as a:source={k:a[k].copy() for k in a.files}
    producer=json.loads((SOURCE/'report.json').read_text())
    assert producer['trace_sha256']==sha256(SOURCE/'trace.npz') and producer['final_control']==CONTROL
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'pico')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'pico',native,c,original,timeline,manifest)
    kp,kd,effort,speed=[np.asarray(c[k]) for k in ('kp','kd','native_effort','native_velocity')]
    lo,hi=np.asarray(c['joint_limits']).T
    spec=mujoco.mjtState(int(source['integration_state_spec']))
    live=mujoco.MjData(native)
    mujoco.mj_setState(native,live,source['final_integration'],spec)
    mujoco.mj_forward(native,live)
    mujoco.mj_setState(native,live,source['final_integration'],spec)
    np.testing.assert_array_equal(live.qpos,source['qpos'][-1]);np.testing.assert_array_equal(live.qvel,source['qvel'][-1])
    initial_time=float(live.time)
    target=source['target'][-1].copy()
    np.testing.assert_array_equal(target,np.clip(target,lo,hi))
    history=BFMHistory()
    for k in history.data:history.data[k][:]=source['final_history_'+k]
    previous_action=source['final_previous_action'].copy()
    normalized=((target-np.asarray(c['default_q']))*kp/(.25*np.asarray(c['training_effort']))).astype(np.float32)
    np.testing.assert_array_equal(previous_action,normalized)
    rows={k:[] for k in ('state','target','previous_action','history','physics_qpos','physics_qvel','physics_torque','physics_actuator_force','physics_time','warning_counts','warning_lastinfo','range_excess','speed_ratio','effort_ratio')}
    rows['state'].append(np.r_[live.qpos,live.qvel]);rows['physics_qpos'].append(live.qpos.copy());rows['physics_qvel'].append(live.qvel.copy());rows['physics_time'].append(initial_time)
    failure=None
    for local in range(H):
        _,terms=state_and_terms(live.qpos[7:],live.qvel[6:],live.qpos[3:7],live.qvel[3:6],previous_action,np.asarray(c['default_q']))
        rows['previous_action'].append(previous_action.copy());rows['history'].append(history.before_update(terms));previous_action=normalized.copy()
        rows['target'].append(target.copy())
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
                failure=dict(kind='physical_feasibility',control=local,global_control=CONTROL+local,substep=sub+1,range_excess=excess,speed_ratio=velocity,effort_ratio=force,height=float(live.qpos[2]),tilt=tilt,
                    joints=[dict(name=c['joint_names'][i],q=float(live.qpos[7+i]),lower=float(lo[i]),upper=float(hi[i]),target=float(target[i])) for i in bad])
            if failure:break
        mujoco.mj_kinematics(native,live);rows['state'].append(np.r_[live.qpos,live.qvel])
        if failure:break
    arrays={k:np.asarray(v) for k,v in rows.items()};steps=len(arrays['physics_torque']);full=steps==H*10 and failure is None
    cost=None
    if full:
        planner=Native23Tracker(position_servo_copy(native,kp,kd,effort),c,motion,horizon=H,threads=1,all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=400)
        planner.window(CONTROL+10);features=planner.features(arrays['state']);residual=planner.residual(np.arange(H+1),features)
        cost=float(np.sum(residual**2)+planner.control_weight*np.sum((arrays['target']-planner.target_reference(np.arange(H)))**2))
    path=BASE/'counterfactual.npz'
    np.savez_compressed(path,**arrays,initial_integration=source['final_integration'],integration_state_spec=np.asarray(int(spec),dtype=np.int64),proposed_targets=np.repeat(target[None],H,axis=0),
        initial_previous_action=source['final_previous_action'],**{'initial_history_'+k:source['final_history_'+k] for k in history.data})
    result=dict(kind='one_causal_actual3805_last_target_hold_seed',initial_control=CONTROL,requested_controls=H,
        actual_controls_executed=0,full_horizon_physical_pass=full,physics_steps=steps,full_controls=steps//10,partial_substeps=steps%10,failure=failure,
        unchanged_v4_allmargin_relativefoot_objective_cost=cost,range_excess_max=float(arrays['range_excess'].max()),speed_ratio_max=float(arrays['speed_ratio'].max()),effort_ratio_max=float(arrays['effort_ratio'].max()),
        warning_counts=arrays['warning_counts'].max(axis=0).tolist(),warning_lastinfo=arrays['warning_lastinfo'].max(axis=0).tolist(),clock_max_error=float(np.max(np.abs(arrays['physics_time']-(initial_time+np.arange(steps+1)*.002)))),
        final_root_height=float(live.qpos[2]),final_root_vertical_velocity=float(live.qvel[2]),
        targets='lastACTUALLYapplied23targets heldconstant; no futureplannedprefix, noBFM, nooptimizer',
        history='exact actual3805 history restored; normalizedheldtarget actions recorded withoutclipping, inferenceunused',
        no_statewrites_after_initialization=True,no_source_goal_or_model_change=True,no_full_source_qualification=True,
        elapsed_seconds=time.perf_counter()-started,trace_sha256=sha256(path),
        hashes={str(p):sha256(p) for p in (Path(__file__),SOURCE/'trace.npz',SOURCE/'report.json',REFERENCE,BUNDLE/'contract.json',BUNDLE/'prepared_model_arrays.npz')})
    with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
