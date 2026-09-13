"""Synthetic-only kernel/formula/immutable-weight tests; never load a task checkpoint."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from promoted_model import PromotedDirect,export_onnx

def tiny():
    return PromotedDirect(np.array([.1,-.3],np.float32),np.array([.05,2.],np.float32),
        [np.array([[.2,-.3],[.7,.1],[-.3,.2]],np.float32),np.array([[.2,.7,-.4],[-.5,.3,.6],[.1,-.3,.8]],np.float32),np.array([[.2,-.1,.9]],np.float32)],
        [np.array([.1,-.2,.3],np.float32),np.array([-.1,.3,.2],np.float32),np.array([.01],np.float32)])
class PromotedTests(unittest.TestCase):
    def test_source_values_exact_and_no_parameters(self):
        m=tiny();self.assertEqual(len(list(m.parameters())),0)
        self.assertTrue(all(v.dtype==torch.float64 and not v.requires_grad for v in m.buffers()))
        np.testing.assert_array_equal(m.feature_mean.numpy(),np.array([.1,-.3],np.float32).astype(np.float64))
    def test_cast_before_normalization(self):
        m=tiny();x=torch.tensor([[.1000001,-.2999999]],dtype=torch.float32)
        value=(x.double()-m.feature_mean)/m.feature_std
        for i in range(3):
            value=torch.nn.functional.linear(value,getattr(m,'w'+str(i)),getattr(m,'b'+str(i)))
            if i<2:value=torch.nn.functional.elu(value)
        torch.testing.assert_close(m(x),value.float(),rtol=0,atol=0)
        with self.assertRaises(ValueError):m(x.double())
    def test_graph_all_double_until_final_cast(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.onnx';m=tiny();export_onnx(m,p);g=onnx.load(p)
            self.assertEqual(g.graph.node[0].op_type,'Cast');self.assertEqual(g.graph.node[-1].op_type,'Cast')
            self.assertTrue(all(x.data_type==onnx.TensorProto.DOUBLE for x in g.graph.initializer))
            self.assertEqual(sum(n.op_type=='Elu' for n in g.graph.node),0);self.assertEqual(sum(n.op_type=='Exp' for n in g.graph.node),2)
            self.assertEqual(sum(n.op_type=='Min' for n in g.graph.node),2);self.assertEqual(g.graph.output[0].type.tensor_type.elem_type,onnx.TensorProto.FLOAT)
    def test_synthetic_cpu_gpu_ort(self):
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x=np.array([[-100.,100.],[0.,0.],[.1,-.3],[1e-7,-1e-7],[10.,-10.]],np.float32);m=tiny()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.onnx';export_onnx(m,p)
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider'])
            with torch.inference_mode():cpu=m(torch.from_numpy(x)).numpy();gpu=m.cuda()(torch.from_numpy(x).cuda()).cpu().numpy()
            actual=session.run(None,{'features':x})[0]
            self.assertEqual(actual.dtype,np.float32);self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual,cpu,rtol=0,atol=1e-7);np.testing.assert_allclose(actual,gpu,rtol=0,atol=1e-7)
            del session
    def test_reject_nonfinite_or_promoted_source(self):
        for dtype,value in [(np.float64,0.),(np.float32,float('nan'))]:
            with self.assertRaises(ValueError):PromotedDirect(np.array([value],dtype),np.ones(1,np.float32),[np.ones((1,1),np.float32)]*3,[np.zeros(1,np.float32)]*3)
if __name__=='__main__':unittest.main()
