"""Export the deterministic heading/position-relative factory candidate."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from run_factory_mimic_sim import FIRMWARE,ASSETS,BUNDLE
from gear_sonic.utils.g1_true23_factory_policy import FactoryReceivedActor,WIDTH

if __name__=='__main__':
    torch.set_num_threads(1)
    cfg=yaml.safe_load((FIRMWARE/'decoded_configs/policies/cpy_dance/dance.yaml').read_text())
    contract=json.loads((BUNDLE/'contract.json').read_text())
    actor=FactoryReceivedActor(ASSETS/'policies/cpy_dance/Feb13_20-31-05_/actor.onnx',
        cfg['default_dof_pos'],contract['joint_limits'],heading_relative=True).eval()
    folder=FIRMWARE/'factory_relative_v1';folder.mkdir(exist_ok=True)
    # Inputs related by one common world XY translation/yaw rotation must
    # produce the same controller command. Body-relative errors stay identical.
    sample=torch.randn(8,WIDTH)*.1
    yaw=torch.linspace(-2.,2.,8)
    sample[:,558:562]=torch.stack((torch.zeros(8),torch.zeros(8),torch.sin(yaw/2),torch.cos(yaw/2)),-1)
    sample[:,565:569]=sample[:,558:562]
    altered=sample.clone();turn=.7
    altered[:,558:562]=torch.stack((torch.zeros(8),torch.zeros(8),torch.sin((yaw+turn)/2),torch.cos((yaw+turn)/2)),-1)
    altered[:,565:569]=altered[:,558:562]
    altered[:,562:564]+=17.
    with torch.no_grad():
        difference=float((actor(sample)-actor(altered)).abs().max())
        assert difference<1e-4,difference
    path=folder/'actor_relative.onnx'
    torch.onnx.export(actor,torch.zeros(1,WIDTH),str(path),input_names=['features'],output_names=['normalized_target'],
        dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
    (folder/'request.json').write_text(json.dumps({'trained':False,'factory_weights_unchanged':True,
        'world_xy_yaw_invariance_max_action_error':difference,'future_reference_frames':0,
        'relative_reference':'horizontal displacement and orientation in measured current heading; reference height retained',
        'hardware_commands':False,'simulation_ready':False},indent=2))
    print(path)
