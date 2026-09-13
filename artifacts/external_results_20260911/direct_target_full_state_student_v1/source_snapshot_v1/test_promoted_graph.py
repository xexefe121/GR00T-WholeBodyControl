"""Tiny synthetic CPU graph tests only; no task checkpoint or CUDA call."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import onnx
from promoted_model import PromotedDirect,export_onnx

def tiny():
    return PromotedDirect(np.array([.1,-.3],np.float32),np.array([.05,2.],np.float32),
        [np.array([[.2,-.3],[.7,.1],[-.3,.2]],np.float32),np.array([[.2,.7,-.4],[-.5,.3,.6],[.1,-.3,.8]],np.float32),np.array([[.2,-.1,.9]],np.float32)],
        [np.array([.1,-.2,.3],np.float32),np.array([-.1,.3,.2],np.float32),np.array([.01],np.float32)])

class PromotedGraph(unittest.TestCase):
    def test_source_promotions_and_no_trainable_parameters(self):
        model=tiny();self.assertEqual(list(model.parameters()),[])
        self.assertTrue(all(value.dtype==torch.float64 for value in model.buffers()))
        np.testing.assert_array_equal(model.feature_mean.numpy(),np.array([.1,-.3],np.float32).astype(np.float64))

    def test_cast_precedes_normalization(self):
        model=tiny();features=torch.tensor([[.1000001,-.2999999]],dtype=torch.float32)
        value=(features.double()-model.feature_mean)/model.feature_std
        for i in range(3):
            value=torch.nn.functional.linear(value,getattr(model,'w'+str(i)),getattr(model,'b'+str(i)))
            if i<2:value=torch.nn.functional.elu(value)
        torch.testing.assert_close(model(features),value.float(),rtol=0,atol=0)
        with self.assertRaises(ValueError):model(features.double())

    def test_graph_twenty_nodes_double_internals_final_float(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tiny.onnx';export_onnx(tiny(),path);graph=onnx.load(path).graph
            self.assertEqual(len(graph.node),20)
            self.assertTrue(all(value.data_type==onnx.TensorProto.DOUBLE for value in graph.initializer))
            self.assertEqual(graph.input[0].type.tensor_type.elem_type,onnx.TensorProto.FLOAT)
            self.assertEqual(graph.output[0].type.tensor_type.elem_type,onnx.TensorProto.FLOAT)
            self.assertEqual([node.op_type for node in graph.node].count('Exp'),2)
            self.assertEqual([node.op_type for node in graph.node].count('Min'),2)
            self.assertEqual([node.op_type for node in graph.node].count('Where'),2)
            self.assertNotIn('Elu',[node.op_type for node in graph.node])

    def test_nonfinite_original_source_rejected(self):
        with self.assertRaises(ValueError):PromotedDirect(np.array([np.nan],np.float32),np.ones(1,np.float32),[np.ones((1,1),np.float32)]*3,[np.zeros(1,np.float32)]*3)

if __name__=='__main__':unittest.main(verbosity=2)
