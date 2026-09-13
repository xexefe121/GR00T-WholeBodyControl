"""Meaningful conversion and CPU/GPU observation checks before factory PPO."""
import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import yaml
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_factory_policy import FactoryReceivedActor,body_errors_numpy,IDS


def check(args):
    torch.set_num_threads(1)
    cfg=yaml.safe_load((args.firmware/'decoded_configs/policies/cpy_dance/dance.yaml').read_text())
    source=args.firmware/'ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport/policies/cpy_dance/Feb13_20-31-05_/actor.onnx'
    contract=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
    actor=FactoryReceivedActor(source,cfg['default_dof_pos'],contract['joint_limits'])
    torch.manual_seed(42); x=torch.randn(32,655)*.2
    # Test the complete imported ONNX graph, not just a handpicked layer.
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider'])
    error=0.
    for row in x.numpy():
        expected=session.run(['actor_actions'],{'memory':row[:465][None],'proprioception':row[465:558][None],
            'quaternion':row[558:562][None],'target_state':row[562:633][None]})[0][0]
        actual=actor.raw29(torch.from_numpy(row)[None]).detach().numpy()[0]
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-5)
        error=max(error,float(np.max(np.abs(actual-expected))))
    result={'torch_onnx_raw29_max_error':error,'all_factory_layers_trainable':all(p.requires_grad for p in actor.parameters()),
        'parameters':sum(p.numel() for p in actor.parameters())}
    if args.gpu:
        from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
        env=FactoryNative23Env(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,
            args.firmware/'decoded_configs/policies/cpy_dance/dance.yaml',count=64)
        actor=actor.cuda()
        obs=env.observe().cpu().numpy()
        error=0.
        from gear_sonic.utils.g1_true23_factory_controller import ReceivedReference,ReceivedPose
        for i in range(8):
            clip=int(env.clips[i]);frame=min(int(env.frames[i]),int(env.lengths[clip])-1)
            q=env.q[i].cpu().numpy();v=env.v[i].cpu().numpy()
            feet=env.sim.data.xpos[i,env.feet].cpu().numpy()
            mats=env.sim.data.xmat[i,env.task_ids].reshape(3,3,3)
            tasks=(env.sim.data.xpos[i,env.task_ids]+torch.einsum('tij,tj->ti',mats,env.task_offsets)).cpu().numpy()
            r={k:val[clip,frame].cpu().numpy() for k,val in env.references.items()}
            expected=body_errors_numpy(q,v,feet,tasks,r['root'],r['feet'],r['tasks'])
            np.testing.assert_allclose(obs[i,633:],expected,atol=2e-5,rtol=2e-5)
            error=max(error,float(np.abs(obs[i,633:]-expected).max()))
            receiver=ReceivedReference()
            for f in (frame-1,frame):
                qr=env.reference_quat[clip,f].cpu().numpy()[[3,0,1,2]]
                received=receiver.accept(ReceivedPose(f,f*.02,env.references['root'][clip,f].cpu().numpy(),qr,
                    env.references['joint'][clip,f].cpu().numpy()))
            np.testing.assert_allclose(obs[i,562:633],received,atol=3e-4,rtol=3e-4)
        # Unseen future changes cannot change this observation.
        before=env.observe().clone()
        saved=[]
        for i in range(env.count):
            clip=int(env.clips[i]);frame=int(env.frames[i]);saved.append((clip,frame))
        for clip in set(c for c,f in saved):
            last=max(f for c,f in saved if c==clip)
            for val in env.references.values():val[clip,last+1:]+=1.2345
        torch.testing.assert_close(before,env.observe(),rtol=0,atol=0)
        begin=time.perf_counter()
        for _ in range(8):
            with torch.no_grad():target=actor.target(actor(env.observe()))
            _,reward,done,info=env.step(target)
            assert torch.isfinite(reward).all()
        torch.cuda.synchronize()
        result.update(cpu_gpu_body_error_max=error,future_mutation_unchanged=True,gpu=str(torch.cuda.get_device_name()),
            eight_controls_seconds=time.perf_counter()-begin,batch=64)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--firmware',type=Path,required=True);ap.add_argument('--bank',type=Path)
    ap.add_argument('--gpu',action='store_true');check(ap.parse_args())
