"""Check received features and factory-initialized full-body dynamics."""
from pathlib import Path
import argparse
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedActor,LOCOMOTION_CONDITIONED_WIDTH

ap=argparse.ArgumentParser();ap.add_argument('--firmware',type=Path,required=True);ap.add_argument('--bank',type=Path)
ap.add_argument('--gpu',action='store_true');ap.add_argument('--standing-controls',type=int,default=1650)
args=ap.parse_args();torch.set_num_threads(1)
folder=args.firmware/'locomotion_conditioned_v1';folder.mkdir(exist_ok=True)
bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
c=json.loads((bundle/'contract.json').read_text());cfg=args.firmware/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
actor=LocomotionConditionedActor(args.firmware/'human_loco_trainable_v1/factory_weights.npz',cfg,c['joint_limits']).eval()
if not args.gpu:
    target=folder/'actor_initial.onnx'
    torch.onnx.export(actor,torch.zeros(1,LOCOMOTION_CONDITIONED_WIDTH),str(target),input_names=['features'],output_names=['normalized_target'],
        dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
    print(target)
else:
    from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
    from gear_sonic.utils.g1_true23_received_features import features_numpy
    env=LocomotionConditionedEnv(bundle,args.bank,cfg,count=64)
    x=env.observe();peak=0.
    for i in range(8):
        clip=int(env.clips[i]);frame=int(env.frames[i]);ref={k:v[clip,:env.lengths[clip]].cpu().numpy() for k,v in env.references.items()}
        wanted=features_numpy(env.q[i].cpu().numpy(),env.v[i].cpu().numpy(),ref,frame,env.default.cpu().numpy(),env.prior[i].cpu().numpy(),env.history_vector()[i].cpu().numpy())
        actual=x[i,218:1541].cpu().numpy();np.testing.assert_allclose(actual,wanted,atol=2e-5,rtol=2e-5)
        peak=max(peak,float(np.abs(actual-wanted).max()))
    # Stand from the actual common recorded start. Resetting a failed world is
    # never counted as surviving: stop this check on its first done signal.
    with np.load(args.bank/'walk003.npz') as z:initial=z['states'][10].copy()
    env.sim.reset();env.q[:]=torch.tensor(initial[:30],device='cuda');env.v[:]=torch.tensor(initial[30:],device='cuda')
    env.clips[:]=0;env.frames[:]=11;env.prior[:]=0
    for h in env.history:h[:]=0
    env.loco_previous[:]=0;env.command_velocity[:]=0;env.loco_phase[:]=0;env.loco_walking[:]=False
    env.loco_history[:]=env.native_prop()[:,None];env.age[:]=0
    env.episode_control_limits[:]=10000;env.episode_end_frames[:]=10000;env.focused_tracking=False;env.sim.forward()
    actor=actor.cuda();failure=None;max_speed=0.;poses=[];velocities=[]
    for step in range(args.standing_controls):
        env.frames[:]=11
        with torch.no_grad():target=actor.target(actor(env.observe()))
        _,reward,done,info=env.step(target)
        max_speed=max(max_speed,float(info['speed_ratio'].max()))
        poses.append(env.q[0].cpu().numpy().copy());velocities.append(env.v[0].cpu().numpy().copy())
        if done.any():failure={'control':step,'failed_worlds':done.nonzero().flatten().cpu().tolist(),'physical_failures':int(info['failed'].sum())};break
        if step%250==0:print(json.dumps({'standing_control':step,'limit':args.standing_controls,'feature_error':peak}),flush=True)
    report={'received_feature_max_error':peak,'standing_controls':len(poses),'requested_controls':args.standing_controls,
        'physical_complete':failure is None,'failure':failure,'maximum_speed_ratio':max_speed,'native323_still_required':True}
    np.savez_compressed(folder/'gpu_standing_trace.npz',qpos=poses,qvel=velocities)
    (folder/'gpu_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
