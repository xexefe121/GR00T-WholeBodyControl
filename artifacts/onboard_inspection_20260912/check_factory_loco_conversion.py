"""Verify portable factory locomotion reconstruction against actual MNN."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from run_factory_mimic_sim import FIRMWARE,ASSETS,MNN
from gear_sonic.utils.g1_true23_factory_native import NativeLegFactoryNetwork
torch.set_num_threads(1)
folder=FIRMWARE/'human_loco_trainable_v1'
actor=NativeLegFactoryNetwork(folder/'factory_weights.npz').eval()
mnn=MNN.Interpreter(str(ASSETS/'policies/human_loco/g1_b_l_ankle_track.mnn'))
session=mnn.createSession({'numThread':1,'backend':'CPU'})
rng=np.random.default_rng(20260913);peak=0.
for _ in range(128):
    obs=rng.normal(0,.4,(1,5,42)).astype(np.float32)
    command=rng.normal(0,.5,(1,8)).astype(np.float32)
    for key,x in (('p_obs',obs),('cmd',command)):
        mnn.getSessionInput(session,key).copyFrom(MNN.Tensor(tuple(x.shape),MNN.Halide_Type_Float,x.ravel(),MNN.Tensor_DimensionType_Caffe))
    mnn.runSession(session);expected=np.asarray(mnn.getSessionOutput(session,'act').getData())
    with torch.no_grad():actual=actor(torch.from_numpy(obs),torch.from_numpy(command)).numpy()[0]
    np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
    peak=max(peak,float(abs(actual-expected).max()))
path=folder/'factory_loco12.onnx'
torch.onnx.export(actor,(torch.zeros(1,5,42),torch.zeros(1,8)),str(path),input_names=['p_obs','cmd'],output_names=['act'],
    dynamic_axes={'p_obs':{0:'batch'},'cmd':{0:'batch'},'act':{0:'batch'}},opset_version=17,dynamo=False)
torch.save(actor.state_dict(),folder/'factory_loco12_state.pt')
report={'samples':128,'maximum_raw_action_error':peak,'parameters':sum(p.numel() for p in actor.parameters()),'onnx':str(path)}
(folder/'conversion_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
