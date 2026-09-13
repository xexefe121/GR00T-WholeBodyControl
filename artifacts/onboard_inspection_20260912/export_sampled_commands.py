"""Export one fixed command-space policy with explicit Gaussian sampling input."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np,torch,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor


class SampledCommands(torch.nn.Module):
    def __init__(self,actor):
        super().__init__();self.actor=actor
        sigma=torch.cat((.03/((actor.limits[:12,1]-actor.limits[:12,0])*.5),
            torch.full((2,),.3/np.pi),torch.tensor([.05/.6,.05/.5,.2/3.])))
        self.register_buffer('sigma',sigma)

    def forward(self,x):
        features=x[:,:1582]
        command=self.actor.command_mean(features)+self.sigma*x[:,1582:1599]
        return self.actor.from_commands(features,command)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(1)
    fw=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    c=json.loads((ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['request']['kind']!='native23_received_command_space_ppo':raise ValueError('Requires a command-space-trained checkpoint')
    actor=TaskCommandActor(fw/'human_loco_trainable_v1/factory_weights.npz',
        fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',c['joint_limits'],
        fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt').eval()
    actor.load_state_dict(saved['actor'],strict=True);model=SampledCommands(actor).eval()
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'actor_sampled.onnx'
    torch.onnx.export(model,torch.zeros(1,1599),str(path),input_names=['features'],output_names=['normalized_target'],
        dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    exported=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    retained=ort.InferenceSession(str(args.checkpoint.with_suffix('.onnx')),sess_options=opts,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20260913);x=rng.normal(0,.1,(32,1599)).astype(np.float32);x[:,1581]=1
    x[:,1582:]=0
    zero_error=float(np.max(np.abs(exported.run(None,{'features':x})[0]-retained.run(None,{'features':x[:,:1582]})[0])))
    x[:,1582:]=rng.normal(size=(32,17))
    with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
    error=float(np.max(np.abs(exported.run(None,{'features':x})[0]-expected)))
    if max(zero_error,error)>5e-5:raise ValueError((zero_error,error))
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        actor=str(path),features=1599,noise_input='17 independent standard-normal samples per control',
        latent_sigma=model.sigma.tolist(),quiet_noise_gating=False,
        zero_noise_agrees_with_deterministic_export_max_error=zero_error,torch_onnx_max_error=error,
        weights_fixed=True,training=False,physical_acceptance=False,hardware_commands=False)
    (args.output/'export.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
