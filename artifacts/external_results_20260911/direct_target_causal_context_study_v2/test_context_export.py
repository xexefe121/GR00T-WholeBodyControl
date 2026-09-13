"""Synthetic1323 export contract tests; no task checkpoint/model is loaded."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'source_draft_v1'))
import numpy as np
import torch
torch.set_num_threads(1)
from context_model import ContextTarget
from context_promoted import from_checkpoint,export_onnx

def synthetic():
    torch.manual_seed(71)
    model=ContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32))
    saved=dict(kind='direct_absolute_native23_target',ordinary_final_step=68000,
        actor_state=model.actor.state_dict(),feature_mean=model.feature_mean,feature_std=model.feature_std)
    return model,from_checkpoint(saved)

def test_promoted_public_float32_and_synthetic_preclamp_parity():
    original,promoted=synthetic();x=torch.linspace(-1,1,1323).reshape(1,1323).to(torch.float32)
    with torch.no_grad():a=original(x);b=promoted(x)
    assert b.shape==(1,23) and b.dtype==torch.float32
    assert torch.max(torch.abs(a-b))<1e-5
    assert all(value.dtype==torch.float64 for value in promoted.state_dict().values())

def test_manual_graph_uses_actual1323_shape_and_exact_promoted_weights(tmp_path):
    import onnx
    from onnx import numpy_helper,TensorProto
    _,promoted=synthetic();path=tmp_path/'synthetic_context.onnx';export_onnx(promoted,path)
    graph=onnx.load(path).graph;arrays={a.name:numpy_helper.to_array(a) for a in graph.initializer}
    assert graph.input[0].type.tensor_type.elem_type==TensorProto.FLOAT
    assert graph.output[0].type.tensor_type.elem_type==TensorProto.FLOAT
    assert graph.input[0].type.tensor_type.shape.dim[1].dim_value==1323
    assert graph.output[0].type.tensor_type.shape.dim[1].dim_value==23
    assert arrays['w0'].shape==(1323,256) and arrays['mean'].shape==(1323,)
    assert arrays['w0'].tobytes()==promoted.w0.T.numpy().copy().tobytes()
    assert all(value.dtype==np.float64 for value in arrays.values())
    assert len(graph.node)==20 and sum(node.op_type=='Where' for node in graph.node)==2
