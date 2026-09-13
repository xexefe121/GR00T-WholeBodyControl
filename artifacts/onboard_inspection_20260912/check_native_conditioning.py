"""Check native23 conditioning preserves balance and matches received features."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_factory_conditioned import NativeFactoryConditionedActor,NATIVE_CONDITIONED_WIDTH


def main(args):
    torch.set_num_threads(1)
    folder=args.firmware/'native23_trainable_v1'
    cfg=args.firmware/'decoded_configs/policies/mimic_test/fsm_mimic_test.yaml'
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    c=json.loads((bundle/'contract.json').read_text())
    actor=NativeFactoryConditionedActor(folder/'factory_weights.npz',cfg,c['joint_limits'])
    torch.manual_seed(123);x=torch.randn(64,NATIVE_CONDITIONED_WIDTH)*.3
    x[:,:380].reshape(64,5,76)[:,:,-1]=0
    with torch.no_grad():
        base=(actor.default+actor.factory_scale*actor.backbone(x[:,:380].reshape(-1,5,76))).clamp(actor.limits[:,0]+.06,actor.limits[:,1]-.06)
        actual=actor.target(actor(x));error=float((actual-base).abs().max())
        assert error<1e-6,error
    result={'zero_command_head_preserves_native_balance_target_max_error_rad':error,
        'native_backbone_frozen':all(not p.requires_grad for p in actor.backbone.parameters()),
        'command_head_parameters':sum(p.numel() for p in actor.goal_head.parameters())}
    if args.gpu:
        from gear_sonic.envs.mjlab.g1_true23_factory_conditioned_dynamics import NativeFactoryConditionedEnv
        from gear_sonic.utils.g1_true23_received_features import features_numpy
        env=NativeFactoryConditionedEnv(bundle,args.bank,cfg,count=64)
        x=env.observe();assert x.shape==(64,NATIVE_CONDITIONED_WIDTH)
        max_error=0.
        for i in range(8):
            clip=int(env.clips[i]);frame=int(env.frames[i])
            ref={k:v[clip,:env.lengths[clip]].cpu().numpy() for k,v in env.references.items()}
            expected=features_numpy(env.q[i].cpu().numpy(),env.v[i].cpu().numpy(),ref,frame,
                env.default.cpu().numpy(),env.prior[i].cpu().numpy(),env.history_vector()[i].cpu().numpy())
            actual=x[i,380:1703].cpu().numpy();np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
            max_error=max(max_error,float(np.abs(actual-expected).max()))
        actor=actor.cuda()
        for _ in range(4):
            with torch.no_grad():target=actor.target(actor(env.observe()))
            _,reward,done,info=env.step(target)
            assert torch.isfinite(reward).all()
        result['cpu_gpu_received_feature_max_error']=max_error
    else:
        target=folder/'native_conditioned_initial.onnx'
        torch.onnx.export(actor.eval(),torch.zeros(1,NATIVE_CONDITIONED_WIDTH),str(target),input_names=['features'],output_names=['normalized_target'],
            dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
        result['onnx']=str(target)
    (folder/('conditioning_gpu_check.json' if args.gpu else 'conditioning_check.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--firmware',type=Path,required=True);ap.add_argument('--bank',type=Path)
    ap.add_argument('--gpu',action='store_true');main(ap.parse_args())
