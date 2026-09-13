"""Compare exact native23 factory reconstruction against the original MNN."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from run_factory_mimic_sim import FIRMWARE,FactoryMimic23,MNN
from gear_sonic.utils.g1_true23_factory_native import Native23FactoryNetwork


if __name__=='__main__':
    torch.set_num_threads(1)
    folder=FIRMWARE/'native23_trainable_v1'
    net=Native23FactoryNetwork(folder/'factory_weights.npz').eval()
    factory=FactoryMimic23()
    rng=np.random.default_rng(20260913)
    error=0.
    for i in range(128):
        obs=rng.normal(0,.4,(1,5,76)).astype(np.float32)
        obs[:,:,-1]=rng.uniform(0,1)
        value=MNN.Tensor((1,5,76),MNN.Halide_Type_Float,obs.ravel(),MNN.Tensor_DimensionType_Caffe)
        factory.tensor.copyFrom(value);factory.net.runSession(factory.session)
        expected=np.asarray(factory.net.getSessionOutput(factory.session,'act').getData(),np.float32)
        with torch.no_grad():actual=net(torch.from_numpy(obs)).numpy()[0]
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
        error=max(error,float(np.abs(actual-expected).max()))
    path=folder/'factory23_raw.onnx'
    torch.onnx.export(net,torch.zeros(1,5,76),str(path),input_names=['obs'],output_names=['act'],
        dynamic_axes={'obs':{0:'batch'},'act':{0:'batch'}},opset_version=17,dynamo=False)
    torch.save(net.state_dict(),folder/'factory23_state.pt')
    result={'samples':128,'max_raw_action_error':error,'all_layers_trainable':all(p.requires_grad for p in net.parameters()),
        'native_action_count':23,'parameters':sum(p.numel() for p in net.parameters()),'onnx':str(path)}
    (folder/'conversion_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
