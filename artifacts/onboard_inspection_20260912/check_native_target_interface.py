"""Exercise native target authority and actual training/runtime history parity."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
if sys.platform!='win32':
    for dependency in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
        '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(dependency)
import torch,onnxruntime as ort
from gear_sonic.envs.mjlab.g1_true23_native_target_dynamics import NativeTargetEnv
from gear_sonic.utils.g1_true23_native_targets import NativeTargetActor,NativeTargetController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet


def run(args):
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4 if args.device=='cpu' else 1);torch.manual_seed(20260913)
    base=Path('E:/codex-artifacts' if sys.platform=='win32' else '/mnt/e/codex-artifacts')
    fw=base/'sonic23_teleop_resume_20260911/onboard_factory_firmware_v1'
    bank=fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
    env=NativeTargetEnv(bundle,bank,cfg,count=args.count,device=args.device,
        canonical_starts=True,canonical_worlds=6,command_delay_substeps=2,physics_backend='mjbatch')
    env.terminate_tracking_errors=False
    actor=NativeTargetActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c,
        fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt').to(args.device).eval()
    x=env.observe()
    with torch.no_grad():
        low=actor.target(actor.from_commands(x,x.new_tensor([-100.]*23+[0.]*5)[None].expand(args.count,-1)))
        high=actor.target(actor.from_commands(x,x.new_tensor([100.]*23+[0.]*5)[None].expand(args.count,-1)))
        assert torch.allclose(low,actor.limits[:,0].expand_as(low),atol=5e-7,rtol=0)
        assert torch.allclose(high,actor.limits[:,1].expand_as(high),atol=5e-7,rtol=0)
    onnx=args.output/'actor_zero.onnx'
    cpu=actor.cpu();torch.onnx.export(cpu,torch.zeros(1,1582),str(onnx),input_names=['features'],
        output_names=['normalized_target'],opset_version=17,dynamo=False,
        dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}})
    actor.to(args.device)
    with torch.no_grad():expected=actor(x).cpu().numpy()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx),sess_options=opts,providers=['CPUExecutionProvider'])
    export_error=float(abs(session.run(None,{'features':x.cpu().numpy()})[0]-expected).max())
    assert export_error<5e-5,export_error
    ci=next(i for i,r in enumerate(env.meta['clips']) if r['name']=='pico')
    with np.load(base/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/pico/reference.npz') as z:motion={k:z[k] for k in z.files}
    with np.load(bundle/'pico/original29.npz') as z:original={k:z[k] for k in z.files}
    timeline=json.loads((bundle/'pico/timeline.json').read_text())
    controller=NativeTargetController(onnx,fw,env.c,model=env.sim.model,tasks=env.meta['tasks'],
        standing_qpos=timeline['configured_standing_qpos'],now=-.22)
    clock_controller=NativeTargetController(onnx,fw,env.c,model=env.sim.model,tasks=env.meta['tasks'],
        standing_qpos=timeline['configured_standing_qpos'],now=-.22)
    sys.path.insert(0,str(ROOT/'artifacts/teleop_resume_20260911'))
    from run_causal_native_clock import import_applied_history
    class Capture:
        def __init__(self,session):self.session=session;self.x=None
        def run(self,names,inputs):self.x=inputs['features'][0].copy();return self.session.run(names,inputs)
    capture=Capture(controller.session);controller.session=capture
    clock_capture=Capture(clock_controller.session);clock_controller.session=clock_capture
    def receive(frame,now):
        fields={k:v[frame] for k,v in motion.items() if k!='fps'}
        for candidate in (controller,clock_controller):
            assert candidate.receive(Packet(0,frame,frame*.02,fields),original['source_task_position_w'][frame],
                original['source_task_quaternion_wxyz'][frame],now)
    for f in range(11):receive(f,(f-11)*.02)
    env.sim.reset(torch.arange(env.count,device=args.device))
    env.clips[:]=ci;env.frames[:]=11;env.age[:]=0;env.canonical_world[:]=True
    env.q[:]=env.states[ci,10,:30];env.v[:]=env.states[ci,10,30:];env.sim.forward()
    env.prior[:]=0
    for h in env.history:h[:]=0
    env.loco_previous[:]=0;env.command_velocity[:]=0;env.loco_phase[:]=0;env.loco_walking[:]=False
    env.loco_history[:]=env.native_prop()[:,None]
    env.last_applied[:]=env.factory_default
    peak=np.zeros(1582);target_error=torque_error=0.;tick=time.monotonic()
    clock_feature_error=clock_target_error=0.
    with torch.no_grad():
        for step in range(64):
            q,v=env.q[0].cpu().numpy().astype(float),env.v[0].cpu().numpy().astype(float)
            receive(step+11,step*.02)
            memory=controller.receiver.history
            values=np.r_[q,v,memory.prior,memory.vector()]
            import_applied_history(clock_controller,values,step)
            clock_command=clock_controller.command(q,v,step*.02)
            command=controller.command(q,v,step*.02)
            clock_feature_error=max(clock_feature_error,float(abs(capture.x-clock_capture.x).max()))
            clock_target_error=max(clock_target_error,float(abs(command.targets-clock_command.targets).max()))
            x=env.observe();peak=np.maximum(peak,abs(x[0].cpu().numpy()-capture.x))
            targets=actor.target(actor(x))
            target_error=max(target_error,float(abs(targets[0].cpu().numpy()-command.targets).max()))
            # Both histories receive the same actual command, including a
            # small changed command; equivalent factory memory is not recorded
            # in the received full-body command history.
            delta=targets.new_tensor(np.sin(np.arange(23)+step)*.005)
            applied=(targets+delta).clamp(env.limits[:,0],env.limits[:,1])
            controller.commit_applied(q,v,applied[0].cpu().numpy())
            env.control_substep=10
            actual=env.control_torque(applied)
            expected=(env.kp*(applied-env.q[:,7:])-env.kd*env.v[:,6:]).clamp(-env.effort,env.effort)
            torque_error=max(torque_error,float(abs(actual-expected).max()))
            _,_,done,info=env.step(applied)
            assert not bool(done[0]),dict(step=step,failed=bool(info['failed'][0]))
    elapsed=time.monotonic()-tick
    sections={name:float(peak[a:b].max()) for name,a,b in (
        ('factory_history',0,210),('command',210,218),('received',218,1541),('closure',1541,1542),('errors',1542,1581),('valid',1581,1582))}
    assert max(sections.values())<1e-4,sections
    assert target_error<5e-5,target_error
    assert torque_error==0,torque_error
    assert clock_feature_error<1e-5,clock_feature_error
    assert clock_target_error<5e-5,clock_target_error
    report=dict(passed=True,device=args.device,count=args.count,full_native_target_range_all23=True,
        extra_target_margin=0.,additional_limit_brake=False,original_native_pd=True,original_physical_limits=True,
        onnx_error=export_error,runtime_training_feature_error=sections,runtime_training_target_error=target_error,
        original_pd_torque_error=torque_error,physical_controls=64,controlled_states_per_second=64*args.count/elapsed,
        imported_native_history_feature_error=clock_feature_error,imported_native_history_target_error=clock_target_error,
        full_motion_passed=False,simulation_ready=False,actor=str(onnx))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=128);p.add_argument('--device',default='cpu');run(p.parse_args())
