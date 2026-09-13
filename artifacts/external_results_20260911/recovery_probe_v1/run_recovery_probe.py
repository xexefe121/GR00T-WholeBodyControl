"""Bounded native323 recovery witnesses; no optimizer, teacher labels or hardware."""
import copy
import json
from pathlib import Path
import sys
import time
from types import MethodType

BASE=Path(__file__).parent
sys.path.insert(0,str(BASE/"source_snapshot"))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states,Native23Tracker,sha256
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed,BFMSeedRolloutError
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,_quaternion_matrix
from terminal_yaw4_goal import terminal_goal_yaw4

ROOT=Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
TASK=Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910")
BUNDLE=ROOT/"artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
PRODUCER=TASK/"mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1"
REFERENCE=TASK/"mjbatch_intent_floor_inputs_v1/pico/reference.npz"
ONNX=ROOT/"artifacts/teleop_six_hour_20260910/bfm_onnx_v2"
DEPS=Path("/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps")
CONTROLS=(3700,3720,3740,3755)
HORIZON=30

def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def write(path,value):
    with path.open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
def clone_memory(source):
    result=BFMHistory()
    for key in result.data:result.data[key][:]=source.data[key]
    return result
def memory_vector(history):
    return np.concatenate([history.data[k].reshape(-1) for k in sorted(history.data)])

def causal_standing_motion(original,qpos):
    """Known initial standing template placed at current XY/heading, not final source pose."""
    initial=10
    oldrot=_quaternion_matrix(original["body_quat_w"][initial,0])
    newrot=_quaternion_matrix(qpos[3:7])
    yaw0=np.arctan2(oldrot[1,0],oldrot[0,0]);yaw1=np.arctan2(newrot[1,0],newrot[0,0])
    rz=Rotation.from_euler("z",yaw1-yaw0)
    initial_pos=original["body_pos_w"][initial]
    origin=initial_pos[0].copy();destination=origin.copy();destination[:2]=qpos[:2]
    placed=rz.apply(initial_pos-origin)+destination
    quat=original["body_quat_w"][initial]
    rotated=(rz*Rotation.from_quat(quat[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    length=max(CONTROLS)+HORIZON+30
    return dict(fps=np.array([50.]),joint_pos=np.repeat(original["joint_pos"][initial:initial+1],length,axis=0),
        joint_vel=np.zeros((length,23)),body_pos_w=np.repeat(placed[None],length,axis=0),
        body_quat_w=np.repeat(rotated[None],length,axis=0),body_lin_vel_w=np.zeros((length,24,3)),
        body_ang_vel_w=np.zeros((length,24,3)))

def run():
    assert mujoco.__version__=="3.2.3" and np.__version__=="1.26.4"
    started=time.perf_counter()
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,"pico")
    motion,override=load_motion_override(REFERENCE,BUNDLE,"pico",native,c,original,timeline,manifest)
    source=archive(PRODUCER/"trace.npz")
    producer_report=json.loads((PRODUCER/"report.json").read_text())
    assert sha256(PRODUCER/"trace.npz")==producer_report["trace_sha256"]
    kp,kd,effort,speed=[np.asarray(c[k]) for k in ("kp","kd","native_effort","native_velocity")]
    lower,upper=np.asarray(c["joint_limits"]).T
    seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    initial=motion_states(motion)[10]
    np.testing.assert_array_equal(initial[:30],source["qpos"][0]);np.testing.assert_array_equal(initial[30:],source["qvel"][0])
    data=mujoco.MjData(native);data.qpos[:]=initial[:30];data.qvel[:]=initial[30:]
    mujoco.mj_forward(native,data)
    snapshots={};warnings=[];clocks=[];maxima={key:0. for key in ("qpos","qvel","torque","history","previous_action")}
    for control in range(max(CONTROLS)+1):
        np.testing.assert_array_equal(data.qpos,source["qpos"][control]);np.testing.assert_array_equal(data.qvel,source["qvel"][control])
        np.testing.assert_array_equal(seed.previous_action,source["fresh_seed_previous_action"][control])
        np.testing.assert_array_equal(memory_vector(seed.history),source["fresh_seed_measured_history"][control])
        if control in CONTROLS:
            # Full MjData continuation retains warmstart and all engine memory.
            snapshots[control]=(copy.deepcopy(data),clone_memory(seed.history),seed.previous_action.copy(),seed.actual_action_max_abs,seed.actual_action_components_outside_five)
        if control==max(CONTROLS):break
        target=source["target"][control]
        seed.record_control(control,data.qpos,data.qvel,target)
        for sub in range(10):
            data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
            mujoco.mj_step(native,data)
            step=control*10+sub+1
            for key,value,index in (("physics_qpos",data.qpos,step),("physics_qvel",data.qvel,step),("physics_torque",data.ctrl,step-1)):
                np.testing.assert_array_equal(value,source[key][index],err_msg=f"{key} step{step}")
            assert not np.any(data.warning.number) and abs(data.time-step*.002)<1e-8
            warnings.append(data.warning.number.copy());clocks.append(data.time)
        mujoco.mj_kinematics(native,data)
    assert len(snapshots)==4
    np.savez_compressed(BASE/"reconstruction_evidence.npz",physics_time=clocks,warning_counts=warnings,
        controls=np.asarray(CONTROLS),qpos=[snapshots[t][0].qpos.copy() for t in CONTROLS],
        qvel=[snapshots[t][0].qvel.copy() for t in CONTROLS],qacc_warmstart=[snapshots[t][0].qacc_warmstart.copy() for t in CONTROLS],
        history=[memory_vector(snapshots[t][1]) for t in CONTROLS],previous_action=[snapshots[t][2] for t in CONTROLS])
    print(json.dumps(dict(reconstruction_bitexact=True,physics_steps=max(CONTROLS)*10,controls=list(CONTROLS))),flush=True)
    servo=position_servo_copy(native,kp,kd,effort)
    planner=Native23Tracker(servo,c,motion,horizon=HORIZON,threads=1,all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=400)
    def objective(xs,targets,control):
        if len(targets)!=HORIZON:return None
        planner.window(control+10)
        features=planner.features(np.asarray(xs))
        residual=planner.residual(np.arange(HORIZON+1),features)
        return float(np.sum(residual**2)+planner.control_weight*np.sum((targets-planner.target_reference(np.arange(HORIZON)))**2))
    results=[]
    for control in CONTROLS:
        current,history,previous,action_max,action_outside=snapshots[control]
        for mode in ("source_goals_yaw2","causal_standing_yaw4"):
            if mode=="source_goals_yaw2":policy=seed
            else:
                hold=causal_standing_motion(original,current.qpos)
                policy=Native23BFMRolloutSeed(native,c,hold,ONNX,dependency_directory=DEPS,threads=1)
                policy._goal=MethodType(terminal_goal_yaw4,policy)
                np.savez_compressed(BASE/f"standing_goal_{control}.npz",**hold)
            policy.history=clone_memory(history);policy.previous_action=previous.copy();policy.recorded_controls=control
            policy.actual_action_max_abs=action_max;policy.actual_action_components_outside_five=action_outside
            try:
                proposed,proposal_diagnostics=policy.propose(control,current.qpos,current.qvel,horizon=HORIZON)
                proposal_error=None
            except BFMSeedRolloutError as exc:
                proposed=None;proposal_diagnostics=exc.diagnostics;proposal_error=str(exc)
            live=copy.deepcopy(current);local_history=clone_memory(history);action=previous.copy()
            arrays={k:[] for k in ("state","target","action","history","previous_action","physics_qpos","physics_qvel","physics_torque","physics_actuator_torque","physics_time","warning_counts","warning_lastinfo","range_excess","speed_ratio","effort_ratio")}
            arrays["state"].append(np.r_[live.qpos,live.qvel]);arrays["physics_qpos"].append(live.qpos.copy());arrays["physics_qvel"].append(live.qvel.copy())
            arrays["physics_time"].append(live.time);failure=None
            for local in range(HORIZON):
                sensed,terms=policy._terms(live.qpos,live.qvel,action)
                history_value=local_history.before_update(terms)
                latent=policy._goal(control+11+local,live.qpos)
                arrays["previous_action"].append(action.copy());arrays["history"].append(history_value.copy())
                action=policy.sessions["actor"].run(None,dict(state=sensed[None],last_action=action[None],history=history_value[None],z=latent))[0][0]*5
                target=np.clip(np.asarray(c["default_q"])+action*.25*np.asarray(c["training_effort"])/kp,lower,upper)
                arrays["target"].append(target.copy());arrays["action"].append(action.copy())
                for sub in range(10):
                    live.ctrl[:]=np.clip(kp*(target-live.qpos[7:])-kd*live.qvel[6:],-effort,effort)
                    mujoco.mj_step(native,live)
                    excess=float(np.maximum(0,np.maximum(lower-live.qpos[7:],live.qpos[7:]-upper)).max())
                    velocity=float(np.max(np.abs(live.qvel[6:])/speed));force=float(np.max(np.abs(live.qfrc_actuator[6:])/effort))
                    values=dict(physics_qpos=live.qpos.copy(),physics_qvel=live.qvel.copy(),physics_torque=live.ctrl.copy(),physics_actuator_torque=live.qfrc_actuator[6:].copy(),physics_time=live.time,
                        warning_counts=live.warning.number.copy(),warning_lastinfo=live.warning.lastinfo.copy(),range_excess=excess,speed_ratio=velocity,effort_ratio=force)
                    for key,value in values.items():arrays[key].append(value)
                    expected=(control*10+local*10+sub+1)*.002
                    tilt=float(np.arccos(np.clip(1-2*np.sum(live.qpos[4:6]**2),-1,1)))
                    if not np.isfinite(live.qpos).all() or not np.isfinite(live.qvel).all() or np.any(live.warning.number) or abs(live.time-expected)>1e-8:
                        failure=dict(kind="engine_or_clock",local_control=local,substep=sub+1)
                    elif excess>1e-6 or velocity>1 or force>1+1e-9 or live.qpos[2]<.25 or tilt>1.2:
                        failure=dict(kind="physical_feasibility",local_control=local,substep=sub+1,range_excess=excess,speed_ratio=velocity,effort_ratio=force,root_height=float(live.qpos[2]),tilt=tilt)
                    if failure:break
                mujoco.mj_kinematics(native,live)
                arrays["state"].append(np.r_[live.qpos,live.qvel])
                if failure:break
            arrays={k:np.asarray(v) for k,v in arrays.items()}
            if proposed is not None:arrays["helper_proposed_target"]=proposed
            path=BASE/f"{control}_{mode}.npz";np.savez_compressed(path,**arrays)
            complete=len(arrays["physics_torque"])//10
            full=complete==HORIZON and failure is None
            steps=len(arrays["physics_torque"])
            result=dict(control=control,source_start_seconds=(control-350)*.02,mode=mode,horizon=HORIZON,
                completed_controls=complete,partial_substeps=steps%10,physics_steps=steps,full_horizon_physical_pass=full,failure=failure,
                range_excess_max=float(arrays["range_excess"].max()),speed_ratio_max=float(arrays["speed_ratio"].max()),effort_ratio_max=float(arrays["effort_ratio"].max()),
                warning_counts=arrays["warning_counts"].max(axis=0).tolist(),warning_lastinfo=arrays["warning_lastinfo"].max(axis=0).tolist(),
                same_declared_source_objective_cost=objective(arrays["state"],arrays["target"],control) if full else None,
                cost_unavailable_reason=None if full else "incomplete strict-feasibility rollout; no partial/full cost comparison",
                helper_proposal_diagnostics=proposal_diagnostics,helper_proposal_error=proposal_error,
                helper_vs_exact_continuation_target_max_delta=None if proposed is None else float(np.max(np.abs(proposed[:len(arrays["target"])]-arrays["target"]))),
                old_failure_seconds_ahead=producer_report["failure"]["time"]-control*.02,
                horizon_spans_old_failure_time=bool(control*.02+HORIZON*.02>=producer_report["failure"]["time"]),
                goal_raw_pose_support_seconds=.76 if mode=="source_goals_yaw2" else 0.,
                standing_scope=None if mode=="source_goals_yaw2" else "known initial-standing frame10 pose rigidly placed at current measured rootXY/yaw; zero velocities; known template height; fixed goal interrupts original source",
                physical_initialization="deep copy of actual continued MjData including warmstart, no forward/reset",trace_sha256=sha256(path))
            results.append(result);print(json.dumps(result),flush=True)
    report=dict(kind="bounded_native323_actual_state_recovery_probe",reconstruction_bitexact=True,reconstructed_physics_steps=max(CONTROLS)*10,
        controls=list(CONTROLS),cases=results,strict_range_tolerance_rad=1e-6,
        objective="same frozen v4 + alljoint.05/2000 + relativefoot400; no cost retuning",
        teacher_labels_created=False,optimizer_used=False,full_source_or_lifecycle_claim=False,hardware_authorized=False,
        baseline_identity=seed.identity(),elapsed_seconds=time.perf_counter()-started,
        hashes={str(p):sha256(p) for p in [Path(__file__),PRODUCER/"trace.npz",PRODUCER/"request.json",PRODUCER/"report.json",REFERENCE,BUNDLE/"contract.json",BASE/"terminal_yaw4_goal.py",*[BASE/"source_snapshot/gear_sonic/utils"/name for name in ("g1_true23_mjbatch_mpc.py","g1_true23_mjbatch_ilqr_core.py","g1_true23_mjbatch_model.py","g1_true23_mjbatch_bfm_seed.py","g1_true23_bfm_seed_observations.py","g1_true23_relative_foot_cost.py")]]})
    write(BASE/"report.json",report)
    print(json.dumps(dict(done=True,elapsed_seconds=report["elapsed_seconds"],passes=[(r["control"],r["mode"]) for r in results if r["full_horizon_physical_pass"]])),flush=True)

if __name__=="__main__":run()
