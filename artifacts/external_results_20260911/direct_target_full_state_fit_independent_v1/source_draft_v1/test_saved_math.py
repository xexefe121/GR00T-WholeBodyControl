"""Synthetic arrays and source AST only; no task fit/model/native calls."""
import ast
import unittest
from pathlib import Path
import numpy as np
from audit_full_state_math import (AXES,schedule,probe_indices,rate,gradient_geometry,initial_losses,metrics,numerical_comparisons,drift_summary)
from audit_math import phase_indices

def fixture():
    return dict(default=np.zeros(23,np.float64),span=np.ones(23,np.float32),limits=np.tile([-10.,10.],(23,1)),
        nominal_teacher=np.zeros((9904,23),np.float64),full_state_teacher=np.zeros((354612,23),np.float64),
        physical_teacher=np.zeros((3054,23),np.float64),
        physical_successor=np.repeat(np.arange(3,dtype=np.int64),1018)*1019+np.tile(np.arange(1,1019,dtype=np.int64),3),
        full_state_flags={name:np.zeros((3057,58,2,23),bool) for name in ('feedback_clipped','native_clipped')})

class SavedMathTests(unittest.TestCase):
    def test_private_sampler_reproducible_global_rng_unchanged(self):
        before=np.random.get_state();c,a,r0,r1=schedule(2);d,b,s0,s1=schedule(2);after=np.random.get_state()
        np.testing.assert_array_equal(c,d);np.testing.assert_array_equal(a,b);self.assertEqual(r0,s0);self.assertEqual(r1,s1)
        np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
        self.assertEqual(c.dtype,np.int32);self.assertEqual(a.dtype,np.int8);self.assertEqual(c.shape,(2,864))

    def test_schedule54cells16pairs_each(self):
        c,a,_,_=schedule(1)
        for cell,eligible in enumerate(phase_indices()[:9]):
            for group,axes in enumerate(AXES):
                part=slice((cell*6+group)*16,(cell*6+group+1)*16)
                self.assertTrue(np.isin(c[0,part],eligible).all());self.assertTrue(np.isin(a[0,part],axes).all())

    def test_endpoint_widening_and_sign_order(self):
        got=probe_indices(np.array([0,3056],np.int32),np.array([0,57],np.int8))
        np.testing.assert_array_equal(got,[0,1,354610,354611]);self.assertEqual(got.dtype,np.int64)
        with self.assertRaises(ValueError):probe_indices(np.array([0.]),np.array([0]))
        with self.assertRaises(ValueError):probe_indices(np.array([3057]),np.array([0]))

    def test_inclusive_learning_rate(self):
        self.assertEqual(rate(0),1e-4);self.assertEqual(rate(9999),1e-5)
        values=[rate(i) for i in range(10000)];self.assertTrue(np.all(np.diff(values)<=0))

    def test_saved_gradient_geometry_fixed_weight(self):
        shapes=((256,1000),(256,),(256,256),(256,),(23,256),(23,))
        arrays={name+'_'+str(i):np.full(shape,scale,np.float32) for name,scale in [('nominal',1),('full_state',2),('physical',0)] for i,shape in enumerate(shapes)}
        result=gradient_geometry(arrays);self.assertEqual(result['coefficient'],.5)
        self.assertAlmostEqual(result['cosine_matrix'][0][1],1.);self.assertIsNone(result['cosine_matrix'][2][2])
        self.assertEqual(result['first_order_descent_dots']['physical'],0.)
        arrays['full_state_0'][0,0]=np.nan
        with self.assertRaises(AssertionError):gradient_geometry(arrays)

    def test_zero_feedback_gradient_not_calibratable(self):
        shapes=((256,1000),(256,),(256,256),(256,),(23,256),(23,))
        arrays={name+'_'+str(i):np.full(shape,1 if name=='nominal' else 0,np.float32) for name in ('nominal','full_state','physical') for i,shape in enumerate(shapes)}
        with self.assertRaises(AssertionError):gradient_geometry(arrays)

    def test_actual_calibration_forward_group_loss(self):
        data=fixture();c,a,_,_=schedule(1)
        pred=dict(nominal=np.ones((9904,23),np.float32),physical=np.ones((3054,23),np.float32),
            full_state=np.repeat(np.arange(1,55,dtype=np.float32),32)[:,None]*np.ones((1,23),np.float32))
        losses,cells=initial_losses(pred,data,c[0],a[0]);self.assertEqual(losses['nominal'],1.);self.assertEqual(losses['physical'],0.)
        np.testing.assert_array_equal(cells['full_state'],np.arange(54,dtype=np.float64)**2)

    def test_full_metrics_equal_groups_not_axis_weighted(self):
        data=fixture();full=np.zeros((3057,58,2,23),np.float32)
        for group,axes in enumerate(AXES):full[:,axes]=group+1
        data['full_state_flags']['feedback_clipped'][0,0,:,0]=True
        pred=dict(nominal=np.zeros((9904,23),np.float32),full_state=full.reshape(354612,23),physical=np.zeros((3054,23),np.float32))
        got=metrics(pred,data);self.assertEqual(len(got['full_state_cells']),54)
        self.assertEqual(got['full_state_objective'],float(np.mean(np.arange(1,7)**2)))
        self.assertNotEqual(got['full_state_objective'],float(np.average(np.arange(1,7)**2,weights=[3,3,23,3,3,23])))
        self.assertEqual(got['full_state_flag_diagnostics']['feedback_clipped']['endpoint_rows'],2)
        self.assertEqual(got['full_state_flag_diagnostics']['feedback_clipped']['raw_row_weighted_response_MSE'],1.)
        self.assertIsNone(got['full_state_flag_diagnostics']['native_clipped']['raw_row_weighted_response_MSE'])

    def test_cast_before_large_prediction_subtraction(self):
        data=fixture();c,a,_,_=schedule(1)
        pred=dict(nominal=np.full((9904,23),2**24,np.float32),full_state=np.full((1728,23),-(2**24)+1,np.float32),physical=np.full((3054,23),2**24,np.float32))
        loss,_=initial_losses(pred,data,c[0],a[0]);correct=float((-float(2**25)+1)**2)
        self.assertAlmostEqual(loss['full_state'],correct,delta=.5)
        self.assertNotEqual(correct,float(np.float32(np.float32(-(2**24)+1)-np.float32(2**24)))**2)

    def test_parity_preclamp_even_equal_applied_targets(self):
        data=fixture();outputs={label:{corpus:np.zeros((2,23),np.float32) for corpus in ('nominal','full_state','physical')} for label in ('CPU64','GPU64','ORT64')}
        for label in outputs:outputs[label]['nominal'][0,0]=11
        outputs['ORT64']['nominal'][0,0]=12
        got=numerical_comparisons(outputs,data);self.assertEqual(len(got),9);self.assertEqual(got['nominal_CPU64_ORT64'],1.)
        outputs['GPU64']['physical'][0,0]=np.nan
        with self.assertRaises(ValueError):numerical_comparisons(outputs,data)

    def test_drift_clipping_mask_accounting(self):
        data=fixture();old=np.zeros((2,23),np.float32);new=old.copy();new[0,0]=11
        delta,got=drift_summary(old,new,data);self.assertEqual(delta.dtype,np.float64)
        self.assertEqual(got['old_clipped_components'],0);self.assertEqual(got['new_clipped_components'],1);self.assertEqual(got['clipping_changed_rows'],1)

    def test_no_producer_or_model_execution_imports(self):
        source=Path(__file__).with_name('audit_saved_fit.py').read_text();tree=ast.parse(source)
        forbidden={'onnxruntime','mujoco','direct_model','promoted_model','train_full_state','full_state_objective','full_state_contract'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertTrue(all(alias.name.split('.')[0] not in forbidden for alias in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,('backward','step','autograd','InferenceSession','mj_step'))

if __name__=='__main__':unittest.main(verbosity=2)
