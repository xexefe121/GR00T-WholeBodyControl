"""Full physical rollouts of the received-only factory pose adapter. Sim only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_factory_mimic_sim import FIRMWARE, ASSETS, BUNDLE, load_model
from gear_sonic.utils.g1_true23_factory_controller import FactoryPoseController, ReceivedPose, ReceivedReference
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import assess, metrics, quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["dance", "fight"], default="dance")
    ap.add_argument("--clip", choices=["stand", "walk002", "walk003", "pico", "walk008"], default="stand")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--initial-vx", type=float, default=0)
    ap.add_argument("--actor",type=Path,help="ONNX export matching the selected controller; simulation only")
    ap.add_argument('--packet-receiver',action='store_true',help='Use the existing timestamped full-body packet receiver')
    ap.add_argument('--input-loss-control',type=int)
    ap.add_argument('--native-reference',action='store_true',help='Native23 factory controller in received-pose error coordinates')
    ap.add_argument('--no-bias-compensation',action='store_true')
    ap.add_argument('--native-conditioned',action='store_true',help='Preserved native23 balance with trained full-body command head')
    ap.add_argument('--factory-locomotion',action='store_true',help='Native factory leg locomotion with received root and upper-body commands')
    ap.add_argument('--factory-model',type=Path,help='Optional factory MNN candidate; requires --factory-locomotion')
    ap.add_argument('--factory-profile',choices=['blind','sknee0','run1'],default='blind')
    ap.add_argument('--factory-amp',action='store_true',help='Experimental native23 AMP backbone observing actual arms and waist')
    ap.add_argument('--target-filter-alpha',type=float,default=1.)
    ap.add_argument('--boundary-matched-damping',action='store_true',help='Experimental inertia-derived damping with torque-equivalent boundary targets')
    ap.add_argument('--benchmark-gains',action='store_true',help='Experimental existing native benchmark gains with boundary torque conversion')
    ap.add_argument('--root-velocity-feedback',type=float,default=0.,help='Measured root linear/angular velocity error damping; zero preserves existing behavior')
    ap.add_argument('--task-commands',action='store_true')
    ap.add_argument('--native-targets',action='store_true',help='Original native PD and complete target bounds for a native-target actor')
    ap.add_argument('--native-preview-guard',action='store_true',help='Bounded measured-state joint-limit preview under native PD')
    ap.add_argument('--native-standing-capture',action='store_true',help='Stop the gait after received standing goals and measured standing conditions')
    ap.add_argument('--native-preview-delay-substeps',type=int,choices=range(7),default=0)
    ap.add_argument('--task-noise-seed',type=int,help='Explicit seeded Gaussian input for a1599-input sampled-command export')
    ap.add_argument('--locomotion-conditioned',action='store_true',help='Trainable full-body leg corrections around native locomotion')
    ap.add_argument('--received-gait',action='store_true',help='Experimental gait-phase servo from owned received leg/foot motion')
    ap.add_argument('--factory-command-limits',action='store_true',help='Use the recovered native factory velocity-command envelope; reference speed remains unchanged')
    ap.add_argument('--command-delay-substeps',type=int,choices=[0,1,2,3],default=0,
        help='Keep actual previous target for this many2ms steps; initial command is warmed')
    args = ap.parse_args()
    if args.native_preview_guard and not args.native_targets:ap.error('Native preview requires --native-targets')
    if args.native_standing_capture and not args.native_targets:ap.error('Native standing capture requires --native-targets')
    if args.native_preview_delay_substeps and not args.native_preview_guard:ap.error('Native preview delay requires --native-preview-guard')
    if args.native_targets and (not args.task_commands or args.benchmark_gains or args.boundary_matched_damping or args.task_noise_seed is not None):
        ap.error('Native targets require deterministic task commands and their own original actuator interface')
    if (args.factory_model or args.factory_profile!='blind') and not args.factory_locomotion:
        ap.error('Factory candidate selection requires --factory-locomotion')
    if args.boundary_matched_damping and not (args.factory_locomotion or args.locomotion_conditioned):
        ap.error('Boundary damping requires a native locomotion controller')
    if args.benchmark_gains and (args.boundary_matched_damping or not (args.factory_locomotion or args.locomotion_conditioned or args.task_commands)):
        ap.error('Benchmark gains require one native locomotion/task controller and cannot combine with boundary damping')
    if args.root_velocity_feedback and not (args.factory_locomotion or args.locomotion_conditioned or args.task_commands):
        ap.error('Root velocity feedback requires a native locomotion/task controller')
    if not np.isfinite(args.root_velocity_feedback) or args.root_velocity_feedback<0:
        ap.error('Root velocity feedback must be finite and nonnegative')
    if args.task_noise_seed is not None and (not args.task_commands or args.task_noise_seed<0):
        ap.error('Task noise requires --task-commands and a nonnegative seed')
    if sum((args.native_reference,args.native_conditioned,args.factory_locomotion,args.factory_amp,args.locomotion_conditioned,args.task_commands))>1:
        ap.error('Select only one factory controller mode')
    if args.task_commands and (args.received_gait or args.factory_command_limits):
        ap.error('Task-command exports contain their learned phase and velocity mapping')
    if args.target_filter_alpha!=1 and not args.locomotion_conditioned:
        raise ValueError('Target filtering requires --locomotion-conditioned')
    args.output.mkdir(parents=True, exist_ok=True)
    model, c = load_model()
    for k in ("native_velocity", "native_effort", "joint_limits"):
        c[k] = np.asarray(c[k])
    variant = "Feb13_20-31-05_" if args.policy == "dance" else "Mar05_20-20-37_"
    controller = FactoryPoseController(ASSETS/f"policies/cpy_{args.policy}/{variant}/actor.onnx",
        FIRMWARE/f"decoded_configs/policies/cpy_{args.policy}/{args.policy}.yaml", c["joint_limits"])
    if args.actor and not (args.native_reference or args.native_conditioned or args.factory_locomotion or args.locomotion_conditioned or args.task_commands):
        from gear_sonic.utils.g1_true23_factory_policy import FactoryReceivedController
        controller=FactoryReceivedController(args.actor,FIRMWARE/'decoded_configs/policies/cpy_dance/dance.yaml',c['joint_limits'])
    receiver = ReceivedReference()
    data = mujoco.MjData(model)
    motion = original = timeline = tasks = None
    if args.clip == "stand":
        data.qpos[:] = c["initial_qpos"]
        data.qpos[7:] = controller.default
        mujoco.mj_forward(model, data)
        feet = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        data.qpos[2] += .001-min(data.geom_xpos[i,2]-model.geom_size[i,0] for i in feet)
        goal = data.qpos.copy()
        count = 1750
    else:
        old = Path(r"E:\codex-artifacts\sonic23_teleop_six_hour_20260910")
        with np.load(old/f"mjbatch_intent_floor_inputs_v1/{args.clip}/reference.npz", allow_pickle=False) as z:
            motion = {k:z[k].copy() for k in z.files}
        with np.load(BUNDLE/f"{args.clip}/original29.npz", allow_pickle=False) as z:
            original = {k:z[k].copy() for k in z.files}
        timeline = json.loads((BUNDLE/f"{args.clip}/timeline.json").read_text())
        bank = FIRMWARE.parent/"causal_dynamics_v1/bank"
        with np.load(bank/f"{args.clip}.npz", allow_pickle=False) as z:
            data.qpos[:] = z["states"][10,:30]
            data.qvel[:] = z["states"][10,30:]
            received_goals={k:z[k].copy() for k in ('root','feet','tasks')}
        tasks = json.loads((bank/"bank.json").read_text())["tasks"]
        feet_ids=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
        task_ids=[model.body(t['target_body']).id for t in tasks]
        task_offsets=np.asarray([t['target_point'] for t in tasks])
        goal = original["source_qpos29"][-1,:7]
        count = timeline["total_requested_controls"] + 1500
        for frame in range(11):
            receiver.accept(ReceivedPose(frame,frame*.02,motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]))
    data.qvel[0] += args.initial_vx
    mujoco.mj_forward(model, data)
    teleop=None
    if args.packet_receiver or args.native_reference or args.native_conditioned or args.factory_locomotion or args.locomotion_conditioned or args.factory_amp or args.task_commands:
        if (args.actor is None and not (args.factory_locomotion or args.factory_amp)) or motion is None:raise ValueError('packet receiver requires a controller and recorded lifecycle')
        from gear_sonic.utils.g1_true23_factory_teleop import FactoryTeleopController
        from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
        if args.task_commands:
            from gear_sonic.utils.g1_true23_task_commands import TaskCommandController
            controller_class=TaskCommandController
            if args.native_targets:
                from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
                controller_class=NativeTargetController
            native_options={'native_preview_guard':args.native_preview_guard,'native_standing_capture':args.native_standing_capture,
                'native_preview_delay_substeps':args.native_preview_delay_substeps} if args.native_targets else {}
            teleop=controller_class(args.actor,FIRMWARE,c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22,noise_seed=args.task_noise_seed,root_velocity_feedback=args.root_velocity_feedback,**native_options)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        elif args.factory_amp:
            from gear_sonic.utils.g1_true23_factory_amp import FactoryAmpTeleop
            teleop=FactoryAmpTeleop(FIRMWARE,c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        elif args.locomotion_conditioned:
            from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedController
            teleop=LocomotionConditionedController(args.actor,FIRMWARE,c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22,received_gait=args.received_gait,factory_command_limits=args.factory_command_limits,target_filter_alpha=args.target_filter_alpha,root_velocity_feedback=args.root_velocity_feedback)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        elif args.factory_locomotion:
            from gear_sonic.utils.g1_true23_factory_locomotion import FactoryLocomotionTeleop
            teleop=FactoryLocomotionTeleop(FIRMWARE,c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22,onnx_path=args.actor,received_gait=args.received_gait,factory_command_limits=args.factory_command_limits,factory_model=args.factory_model,factory_profile=args.factory_profile,root_velocity_feedback=args.root_velocity_feedback)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        elif args.native_conditioned:
            from gear_sonic.utils.g1_true23_factory_conditioned import NativeFactoryConditionedController
            teleop=NativeFactoryConditionedController(args.actor,FIRMWARE/'decoded_configs/policies/mimic_test/fsm_mimic_test.yaml',
                c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        elif args.native_reference:
            from gear_sonic.utils.g1_true23_factory_reference_tracker import Native23ReferenceTracker
            with np.load(FIRMWARE/'native23_trainable_v1/recorded_start_stand/trace.npz') as z:
                equilibrium={k:z[k][-1].copy() for k in ('qpos','target')}
            teleop=Native23ReferenceTracker(args.actor,FIRMWARE/'decoded_configs/policies/mimic_test/fsm_mimic_test.yaml',
                equilibrium,c,model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22,
                bias_compensation=not args.no_bias_compensation)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        else:
            teleop=FactoryTeleopController(args.actor,FIRMWARE/'decoded_configs/policies/cpy_dance/dance.yaml',c,
                model=model,tasks=tasks,standing_qpos=timeline['configured_standing_qpos'],now=-.22)
        if args.boundary_matched_damping:
            from gear_sonic.utils.g1_true23_damped_actuation import BoundaryMatchedDamping
            teleop=BoundaryMatchedDamping(teleop,model,c)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        if args.benchmark_gains:
            from gear_sonic.utils.g1_true23_benchmark_actuation import NativeBenchmarkGains
            teleop=NativeBenchmarkGains(teleop,c)
            controller.kp=teleop.kp;controller.kd=teleop.kd
        for sequence in range(11):
            fields={k:v[sequence] for k,v in motion.items() if k!='fps'}
            assert teleop.receive(Packet(0,sequence,sequence*.02,fields),original['source_task_position_w'][sequence],
                original['source_task_quaternion_wxyz'][sequence],(sequence-11)*.02)
    trace = {k:[] for k in ["qpos","qvel","target","frame","control","inference_ms","physics_qpos","physics_qvel"]}
    failure = None
    max_speed = max_effort = 0.0
    started = time.perf_counter()
    expected_time=0.
    previous_target=None
    for control in range(count):
        if motion is None:
            frame = control
            pose = ReceivedPose(control, control*.02,goal[:3],goal[3:7],goal[7:])
        else:
            frame = min(control+11,len(motion["joint_pos"])-1)
            pose = ReceivedPose(control+11,(control+11)*.02,motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame])
        reference = receiver.accept(pose)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation,data.qpos[3:7])
        gravity = rotation.reshape(3,3).T @ np.array([0.,0.,-1.])
        begin = time.perf_counter_ns()
        if teleop:
            if control+11<len(motion['joint_pos']) and (args.input_loss_control is None or control<args.input_loss_control):
                sequence=control+11
                fields={k:v[sequence] for k,v in motion.items() if k!='fps'}
                accepted=teleop.receive(Packet(0,sequence,sequence*.02,fields,sequence==len(motion['joint_pos'])-1),
                    original['source_task_position_w'][sequence],original['source_task_quaternion_wxyz'][sequence],control*.02)
                if not accepted:raise ValueError('recorded packet rejected')
            command=teleop.command(data.qpos,data.qvel,control*.02)
            target=command.targets
            teleop.commit_applied(data.qpos,data.qvel,target)
        elif args.actor:
            if motion is None:raise ValueError('Use a complete recorded lifecycle for trained factory evaluation')
            feet=data.xpos[feet_ids].copy()
            task_positions=data.xpos[task_ids]+np.einsum('tij,tj->ti',data.xmat[task_ids].reshape(3,3,3),task_offsets)
            target=controller.step(data.qpos,data.qvel,reference,feet,task_positions,
                received_goals['root'][frame],received_goals['feet'][frame],received_goals['tasks'][frame])
        else:
            target = controller.step(data.qpos[7:],data.qvel[6:],data.qvel[3:6],gravity,data.qpos[3:7],reference)
        if teleop is None:controller.commit_applied(target)
        trace["inference_ms"].append((time.perf_counter_ns()-begin)/1e6)
        trace["target"].append(target.copy())
        trace["frame"].append(frame)
        trace["control"].append(control)
        for substep in range(10):
            applied=previous_target if previous_target is not None and substep<args.command_delay_substeps else target
            torque = controller.kp*(applied-data.qpos[7:])-controller.kd*data.qvel[6:]
            band = np.minimum(.1,np.diff(c["joint_limits"],axis=1).ravel()*.2)
            penetration = data.qpos[7:]-np.clip(data.qpos[7:],c["joint_limits"][:,0]+band,c["joint_limits"][:,1]-band)
            outward = np.where(penetration*data.qvel[6:]>0,data.qvel[6:],0)
            if not args.native_targets:torque=torque-100*penetration-2*outward
            data.ctrl[:] = np.clip(torque,-c["native_effort"],c["native_effort"])
            mujoco.mj_step(model,data)
            expected_time+=.002
            trace["physics_qpos"].append(data.qpos.copy())
            trace["physics_qvel"].append(data.qvel.copy())
            reasons, values = assess(data,c,expected_time)
            max_speed = max(max_speed,values["speed_ratio"])
            max_effort = max(max_effort,values.get("effort_ratio",0))
            if reasons:
                failure = {"time":float(data.time),"reasons":reasons,**values}
                break
        trace["qpos"].append(data.qpos.copy())
        trace["qvel"].append(data.qvel.copy())
        previous_target=target.copy()
        if failure:
            break
    trace = {k:np.asarray(v) for k,v in trace.items()}
    np.savez_compressed(args.output/"trace.npz",**trace)
    report = {"policy":args.policy,"actor":str(args.actor) if args.actor else None,"clip":args.clip,"simulation_seconds":float(data.time),"requested_seconds":count*.02,
        "physical_complete":failure is None,"failure":failure,"wall_seconds":time.perf_counter()-started,
        "max_speed_ratio":max_speed,"max_effort_ratio":max_effort,"inference_ms_p50_p95_max":np.quantile(trace["inference_ms"],[.5,.95,1]).tolist(),
        "future_reference_frames":0,"reference_velocity":"received-pose backward differences, reference body frame",
        "absent_vendor_joints":"fixed zero position and velocity", "independent_realtime":False,"hardware_commands":False}
    report['clock_expected_time']='independent repeated addition of .002; tolerance unchanged at1e-10'
    report['command_delay_substeps']=args.command_delay_substeps
    report['native_target_actuation']=bool(args.native_targets)
    report['native_preview_guard']=bool(args.native_preview_guard)
    report['native_standing_capture']=bool(args.native_standing_capture)
    report['native_preview_delay_substeps']=args.native_preview_delay_substeps
    report['additional_limit_brake']=not args.native_targets
    report['initial_world_velocity_x_delta_mps']=args.initial_vx
    report['command_delay_physics_seconds']=args.command_delay_substeps*.002
    report['received_gait_phase']=args.received_gait
    report['factory_command_limits']=args.factory_command_limits
    if motion is not None:
        report["source"] = metrics(model,trace["qpos"],trace["frame"],trace["control"],motion,original,timeline,tasks)
    if failure is None:
        report["quiet"] = quiet(trace["physics_qpos"],trace["physics_qvel"],goal)
        report['continuous_quiet30']=quiet(trace['physics_qpos'],trace['physics_qvel'],goal,seconds=30)
    if teleop:
        report['packet_report']=teleop.receiver.gate.epoch_report()
        report['controller_status']=command.status
        report['input_loss_control']=args.input_loss_control
        report['automatic_rearm']=False
        if teleop.receiver.stop is not None:
            report['fault_quiet']=quiet(trace['physics_qpos'],trace['physics_qvel'],teleop.receiver.stop.last)
            report['fault_continuous_quiet30']=quiet(trace['physics_qpos'],trace['physics_qvel'],teleop.receiver.stop.last,seconds=30)
        report['fault_scenario_passed']=bool(args.input_loss_control is not None and failure is None
            and report.get('fault_quiet',{}).get('passed',False) and report.get('fault_continuous_quiet30',{}).get('passed',False)
            and command.status['explicit_rearm_required'])
    report["passed"] = failure is None and report.get("quiet",{}).get("passed",False) and report.get('continuous_quiet30',{}).get('passed',False) and (motion is None or report["source"]["passed"])
    if teleop and teleop.receiver.gate.fault is not None:report['passed']=False
    if args.native_reference or args.native_conditioned:
        report['policy']='native23_conditioned' if args.native_conditioned else 'native23_pose_error'
        report['absent_vendor_joints']=None
    if args.factory_locomotion:
        report['policy']='factory_native_locomotion_received'
        import hashlib
        factory_policy=(teleop.base if args.boundary_matched_damping or args.benchmark_gains else teleop).policy
        report['factory_model']=str(factory_policy.path) if args.actor is None else str(args.actor)
        report['factory_model_sha256']=hashlib.sha256((factory_policy.path if args.actor is None else args.actor).read_bytes()).hexdigest()
        report['factory_profile']=factory_policy.profile
        report['factory_frequency_hz']=factory_policy.frequency
        report['factory_duty_fraction']=factory_policy.duty
        report['absent_vendor_joints']=None
        report['reference_velocity']='backward differences; world-frame root velocity converted to measured heading'
    if args.locomotion_conditioned:
        report['policy']='native_locomotion_full_body_conditioned';report['absent_vendor_joints']=None
        report['reference_velocity']='backward differences; current measured heading'
    if args.boundary_matched_damping:report['boundary_matched_damping']=teleop.description()
    if args.benchmark_gains:report['native_benchmark_gains']=teleop.description()
    if args.factory_amp:
        report['policy']='factory_native23_amp_received';report['absent_vendor_joints']=None
        report['reference_velocity']='backward differences; current measured heading'
    if args.task_commands:
        report['policy']='native23_received_task_commands';report['absent_vendor_joints']=None
        report['task_command_noise_seed']=args.task_noise_seed
        report['leg_target_filter_alpha']=1. if args.native_targets else .9
        if args.native_targets:report['factory_prior_filter_alpha']=.9
        report['reference_velocity']='backward differences; measured body errors; learned factory phase/velocity commands'
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
