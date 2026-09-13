"""Synthetic-only contracts, objective gradients, export and preservation tests."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from direct_contract import *
from direct_data import mapping_to_nominal,array_path
from direct_model import DirectTarget,export_onnx
from direct_objective import nominal_loss,velocity_loss,physical_loss
import direct_diagnostics as diagnostics

class ContractTests(unittest.TestCase):
    def test_kept_indices_exclude_prior_and_base(self):
        x=np.arange(1069,dtype=np.float32)[None]
        y=reduce_features(x)
        self.assertEqual(y.shape,(1,1000));self.assertTrue(np.array_equal(y[0],np.r_[x[0,:52],x[0,75:1023]]))
        self.assertFalse(np.isin(np.r_[np.arange(52,75),np.arange(1023,1069)],y).any())
    def test_weighted_normalization_includes_between_cell_variance(self):
        x=np.array([[0.,5.],[2.,5.],[10.,5.],[12.,5.],[14.,5.],[16.,5.]],np.float32)
        cells=[np.array([0,1]),np.array([2,3,4,5])]
        mean,std,mean64,var64=weighted_normalization(x,cells)
        weights=np.array([.25,.25,.125,.125,.125,.125])
        wanted=(x.astype(np.float64)*weights[:,None]).sum(0)
        variance=(((x.astype(np.float64)-wanted)**2)*weights[:,None]).sum(0)
        np.testing.assert_array_equal(mean64,wanted);np.testing.assert_array_equal(var64,variance)
        self.assertGreater(var64[0],4.);self.assertEqual(std[1],np.float32(.05))
        self.assertEqual(mean.dtype,np.float32);self.assertEqual(std.dtype,np.float32)
    def test_equal_cells_not_global_population(self):
        x=np.array([[0.],[8.],[8.],[8.]],np.float32)
        mean,_,_,_=weighted_normalization(x,[np.array([0]),np.array([1,2,3])])
        self.assertEqual(float(mean[0]),4.);self.assertNotEqual(float(mean[0]),float(x.mean()))
    def test_nominal_cell_counts_and_coverage(self):
        counts=((1,2,3),(4,5,6));d=np.repeat(np.arange(2),[6,15]);p=np.r_[np.repeat(np.arange(3),counts[0]),np.repeat(np.arange(3),counts[1])]
        self.assertEqual(list(map(len,nominal_cells(d,p,counts))),[1,2,3,4,5,6])
        with self.assertRaises(ValueError):nominal_cells(d,p,((2,1,3),(4,5,6)))
    def test_promoted_existing_span_and_label_cast(self):
        span=np.array([2.967067],np.float32);default=np.array([.3],np.float64);target=np.array([[.123456789012]],np.float64)
        label=normalized_labels(target,default,span)
        exact(label,((target-default)/span.astype(np.float64)).astype(np.float32))
        self.assertNotEqual(float(span[0]),2.967067)
        with self.assertRaises(ValueError):normalized_labels(target.astype(np.float32),default,span)
    def test_runtime_reconstruction_float64_then_clamp(self):
        span=np.full(23,np.float32(2.967067));default=np.linspace(-.2,.2,23,dtype=np.float64);limits=np.tile([-.4,.5],(23,1)).astype(np.float64)
        h=np.linspace(-1,1,23,dtype=np.float32)[None]
        raw,applied=absolute_targets(h,default,span,limits)
        exact(raw,default+span.astype(np.float64)*h.astype(np.float64));exact(applied,np.clip(raw,-.4,.5))
        self.assertEqual(raw.dtype,np.float64)
    def test_explicit_identity_map_with_reordered_nominal(self):
        d=np.array([1,0,1,0]);c=np.array([251,250,250,251]);wanted=mapping_to_nominal(np.array([0,0,1,1]),np.array([250,251,250,251]),d,c)
        np.testing.assert_array_equal(wanted,[1,3,2,0])
        with self.assertRaises(ValueError):mapping_to_nominal(np.array([0]),np.array([250]),np.array([0,0]),np.array([250,250]))
    def test_manifest_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):array_path(Path(folder)/'manifest.json',dict(path='../escape.npy'))
    def test_fixed_cosine_endpoints_and_monotonicity(self):
        self.assertEqual(cosine_rate(0),3e-4);self.assertEqual(cosine_rate(4999),3e-5)
        values=[cosine_rate(i) for i in range(5000)];self.assertTrue(np.all(np.diff(values)<0))
        with self.assertRaises(ValueError):cosine_rate(5000)

class ObjectiveTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'Isolated CUDA runtime required for this synthetic gradient test.')
    def test_cuda_duplicate_center_gradients_are_deterministic(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        gradients=[]
        for _ in range(2):
            center=torch.linspace(0,1,23*17,dtype=torch.float32,device='cuda').reshape(17,23).requires_grad_()
            probe=torch.linspace(-.5,.5,1152*23,dtype=torch.float32,device='cuda').reshape(1152,23).requires_grad_()
            ids=(torch.arange(576,device='cuda')%17).long()
            loss,_=velocity_loss(center,probe,ids,torch.zeros((576,2,23),dtype=torch.float64,device='cuda'))
            loss.backward();torch.cuda.synchronize();gradients.append((center.grad.cpu().clone(),probe.grad.cpu().clone()))
        self.assertTrue(torch.equal(gradients[0][0],gradients[1][0]));self.assertTrue(torch.equal(gradients[0][1],gradients[1][1]))
    def test_nominal_equal15_cell_loss(self):
        cells=[];values=[];start=0
        for k in range(15):
            n=k+1;cells.append(torch.arange(start,start+n));values.append(torch.full((n,23),float(k+1)));start+=n
        prediction=torch.cat(values).requires_grad_();loss,per=nominal_loss(prediction,torch.zeros_like(prediction),cells)
        np.testing.assert_allclose(per.detach().numpy(),np.arange(1,16)**2)
        self.assertAlmostEqual(float(loss.detach()),float(np.mean(np.arange(1,16)**2)),places=5)
        loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
    def test_velocity_sign_axis_cell_reduction(self):
        center=torch.zeros((9,23),dtype=torch.float32,requires_grad=True)
        values=np.empty((576,2,23),np.float32)
        for cell in range(9):
            values[cell*64:(cell+1)*64,0]=np.arange(1,24,dtype=np.float32)*(cell+1)
            values[cell*64:(cell+1)*64,1]=3*np.arange(1,24,dtype=np.float32)*(cell+1)
        probes=torch.from_numpy(values.reshape(1152,23)).requires_grad_();ids=torch.repeat_interleave(torch.arange(9),64)
        loss,per=velocity_loss(center,probes,ids,torch.zeros((576,2,23),dtype=torch.float64))
        wanted=np.array([np.mean(values[k*64:(k+1)*64].astype(np.float64)**2) for k in range(9)])
        np.testing.assert_array_equal(per.detach().numpy(),wanted);self.assertEqual(float(loss.detach()),float(wanted.mean()))
        loss.backward();self.assertTrue((center.grad!=0).any());self.assertTrue((probes.grad!=0).any())
    def test_output_promotion_precedes_subtraction(self):
        center=torch.ones((1,23),dtype=torch.float32);probe=torch.full((1152,23),1e-7,dtype=torch.float32)
        exact_change=(probe[0].double()-center[0].double()).expand(576,2,23).clone()
        loss,_=velocity_loss(center,probe,torch.zeros(576,dtype=torch.long),exact_change)
        self.assertEqual(float(loss),0.)
        self.assertNotEqual(float(((probe[0]-center[0]).double()-exact_change[0,0]).abs().max()),0.)
    def test_physical_fixed_requested_denominators_and_live_centers(self):
        counts=(99,819,100)*3;cells=[];values=[];start=0
        for k,n in enumerate(counts):
            valid=n if k%3==0 else (2 if k%3==1 else 0)
            cells.append(torch.arange(start,start+valid));values.extend([k+1]*valid);start+=valid
        centers=torch.zeros((1,23),requires_grad=True);prediction=torch.tensor(values,dtype=torch.float32)[:,None].repeat(1,23).requires_grad_()
        loss,per=physical_loss(centers,prediction,torch.zeros(len(values),dtype=torch.long),torch.zeros((len(values),23),dtype=torch.float64),cells,counts)
        wanted=np.array([len(ids)*(k+1)**2/n for k,(ids,n) in enumerate(zip(cells,counts))])
        np.testing.assert_allclose(per.detach().numpy(),wanted,rtol=0,atol=1e-15)
        self.assertAlmostEqual(float(loss.detach()),float(wanted.mean()),places=14);loss.backward();self.assertTrue((centers.grad!=0).any())
    def test_absolute_teacher_changes_not_residuals(self):
        centers=torch.zeros((1,23),dtype=torch.float32);end=torch.full((1,23),.25,dtype=torch.float32)
        teacher=torch.full((1,23),.25,dtype=torch.float64);cells=[torch.tensor([0])]*9
        loss,_=physical_loss(centers,end,torch.tensor([0]),teacher,cells,(1,)*9)
        self.assertEqual(float(loss),0.)

class ModelTests(unittest.TestCase):
    def test_fresh_default_seed_initialization(self):
        torch.manual_seed(SEED);one=DirectTarget(np.zeros(1000,np.float32),np.ones(1000,np.float32))
        torch.manual_seed(SEED);two=DirectTarget(np.zeros(1000,np.float32),np.ones(1000,np.float32))
        for a,b in zip(one.parameters(),two.parameters()):self.assertTrue(torch.equal(a,b));self.assertEqual(a.dtype,torch.float32)
        self.assertTrue((one.actor[-1].weight!=0).any());self.assertTrue((one.actor[-1].bias!=0).any())
        self.assertEqual([type(v).__name__ for v in one.actor],['Linear','ELU','Linear','ELU','Linear'])
    def test_synthetic_ONNX_contains_only_normalization_and_MLP(self):
        import onnx,onnxruntime as ort
        torch.manual_seed(91);model=DirectTarget(np.linspace(-.1,.1,1000,dtype=np.float32),np.linspace(.5,2,1000,dtype=np.float32)).eval()
        x=np.linspace(-1,1,3000,dtype=np.float32).reshape(3,1000)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.onnx';export_onnx(model,path);graph=onnx.load(path).graph
            self.assertEqual(graph.output[0].name,'normalized_target');self.assertEqual(graph.input[0].type.tensor_type.shape.dim[1].dim_value,1000)
            self.assertEqual([n.op_type for n in graph.node],['Sub','Div','MatMul','Add','Elu','MatMul','Add','Elu','MatMul','Add'])
            self.assertFalse(any(a.name in ('span','base','default_q') for a in graph.initializer))
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            returned=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'features':x})[0]
            with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
            self.assertLess(float(np.max(np.abs(returned.astype(np.float64)-expected.astype(np.float64)))*6),1e-5)

class DiagnosticTests(unittest.TestCase):
    def data(self):
        return dict(features=np.zeros((3,1000),np.float32),velocity_features=np.zeros((5,1000),np.float32),physical_features=np.zeros((2,1000),np.float32),
            default=np.zeros(23,np.float64),span=np.ones(23,np.float32),limits=np.tile([-1.,1.],(23,1)))
    def test_partition_counts_are_per_corpus(self):
        self.assertEqual(sum((n+255)//256 for n in (9904,140622,3054)),601)
        self.assertEqual((sum((9904,140622,3054))+255)//256,600)
    def test_stub_ORT_batches_and_outputs(self):
        calls=[]
        class Session:
            def run(self,_,feed):calls.append(len(feed['features']));return [np.ones((len(feed['features']),23),np.float32)]
        with tempfile.TemporaryDirectory() as folder,patch.object(diagnostics,'SIZES',(3,5,2)),patch.object(diagnostics,'BATCH',2):
            counters=dict(ORT_calls_attempted=0,ORT_calls_returned=0);active={}
            out=diagnostics.run_ort_pass('unused',self.data(),folder,counters,active,lambda _:Session())
            self.assertEqual(calls,[2,1,2,2,1,2]);self.assertEqual(counters,dict(ORT_calls_attempted=6,ORT_calls_returned=6))
            self.assertTrue(np.isfinite(out['velocity']).all())
            for value in out.values():value._mmap.close()
    def test_stub_failure_preserves_returned_nonfinite(self):
        class Session:
            def run(self,_,feed):return [np.full((len(feed['features']),23),np.nan,np.float32)]
        with tempfile.TemporaryDirectory() as folder,patch.object(diagnostics,'SIZES',(3,5,2)),patch.object(diagnostics,'BATCH',2):
            counters=dict(ORT_calls_attempted=0,ORT_calls_returned=0);active={'diagnostic_output':'stale'}
            with self.assertRaises(ValueError):diagnostics.run_ort_pass('unused',self.data(),folder,counters,active,lambda _:Session())
            self.assertEqual(counters,dict(ORT_calls_attempted=1,ORT_calls_returned=1));self.assertTrue(active['diagnostic_current_returned'])
            self.assertTrue(np.isnan(active['diagnostic_output']).all());self.assertEqual(active['diagnostic_valid_rows'],2)
    def test_constructor_failure_clears_stale_call_identity(self):
        def fail(_):raise RuntimeError('synthetic constructor failure')
        active={'diagnostic_output':'stale','diagnostic_current_returned':True}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):diagnostics.run_ort_pass('unused',self.data(),folder,{},active,fail)
        self.assertNotIn('diagnostic_output',active);self.assertFalse(active['diagnostic_current_returned'])
    def test_nonfinite_parity_cannot_pass(self):
        good={k:np.zeros((2,23),np.float32) for k in diagnostics.CORPORA};bad={k:a.copy() for k,a in good.items()};bad['velocity'][0,0]=np.nan
        result=diagnostics.parity(good,good,bad,self.data())
        self.assertFalse(result['passed']);self.assertIsNone(result['maximum_preclamp_rad']);self.assertTrue(result['nonfinite_comparisons'])
    def test_preclamp_parity_detects_hidden_saturation_difference(self):
        one={k:np.full((2,23),2.,np.float32) for k in diagnostics.CORPORA};two={k:np.full((2,23),3.,np.float32) for k in diagnostics.CORPORA}
        result=diagnostics.parity(one,one,two,self.data());self.assertFalse(result['passed']);self.assertEqual(result['maximum_preclamp_rad'],1.)

if __name__=='__main__':unittest.main(verbosity=2)
