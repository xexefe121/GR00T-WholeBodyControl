"""Reject semantically altered exports of a synthetic model; zero forward calls."""
import importlib.util
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
from audit_graph import audit_graph

@pytest.fixture
def synthetic(tmp_path):
    path = Path(__file__).resolve().parent.parent / 'direct_target_fp64_export_v1/source_draft_v1/promoted_model.py'
    spec = importlib.util.spec_from_file_location('synthetic_export_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mean, std = np.zeros(1000,np.float32), np.ones(1000,np.float32)
    weights = [np.zeros(shape,np.float32) for shape in [(256,1000),(256,256),(23,256)]]
    biases = [np.zeros(size,np.float32) for size in (256,256,23)]
    weights[0][0,0] = np.float32(.1)
    model = module.PromotedDirect(mean,std,weights,biases)
    out = tmp_path / 'synthetic.onnx'
    module.export_onnx(model,out)
    actor = {}
    for i,w,b in zip((0,2,4),weights,biases):
        actor[str(i)+'.weight'] = torch.from_numpy(w)
        actor[str(i)+'.bias'] = torch.from_numpy(b)
    checkpoint = dict(actor_state=actor,feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std))
    return onnx.load(out), checkpoint, dict(feature_mean=mean,feature_std=std)

def test_exact_same_weight_graph_passes(synthetic):
    assert audit_graph(*synthetic)['same_source_weights'] is True

def test_unbounded_unused_exponential_rejected(synthetic):
    graph, checkpoint, norm = synthetic
    exp = next(n for n in graph.graph.node if n.op_type == 'Exp')
    exp.input[0] = 'a0'
    with pytest.raises(AssertionError): audit_graph(graph,checkpoint,norm)

def test_wrong_final_output_dtype_rejected(synthetic):
    graph, checkpoint, norm = synthetic
    graph.graph.node[-1].attribute[0].i = onnx.TensorProto.DOUBLE
    with pytest.raises(AssertionError): audit_graph(graph,checkpoint,norm)

def test_tiny_weight_change_is_not_exact_promotion(synthetic):
    graph, checkpoint, norm = synthetic
    index = next(i for i,n in enumerate(graph.graph.initializer) if n.name == 'w0')
    values = onnx.numpy_helper.to_array(graph.graph.initializer[index]).copy()
    values[0,0] += 1e-12
    graph.graph.initializer[index].CopyFrom(onnx.numpy_helper.from_array(values,'w0'))
    with pytest.raises(AssertionError): audit_graph(graph,checkpoint,norm)

def test_float32_hidden_weight_rejected(synthetic):
    graph, checkpoint, norm = synthetic
    index = next(i for i,n in enumerate(graph.graph.initializer) if n.name == 'w1')
    values = onnx.numpy_helper.to_array(graph.graph.initializer[index]).astype(np.float32)
    graph.graph.initializer[index].CopyFrom(onnx.numpy_helper.from_array(values,'w1'))
    with pytest.raises(AssertionError): audit_graph(graph,checkpoint,norm)
