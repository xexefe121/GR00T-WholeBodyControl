"""Synthetic tensors/metadata only. No task model, optimizer or native calls."""
import unittest
import numpy as np
import torch
from full_state_contract import (GROUP_SIZES, build_schedule, verify_schedule, endpoint_rows,
                                 center_cells, cosine_rate, calibrate)
from full_state_objective import full_state_loss

class FullStateHelpers(unittest.TestCase):
    def metadata(self):
        return (np.repeat(np.arange(3,dtype=np.int64),1019),
                np.tile(np.arange(250,1269,dtype=np.int64),3),
                np.repeat(np.arange(6,dtype=np.int8),GROUP_SIZES))

    def test_schedule_exact_repeat_and_all54_cells(self):
        args=self.metadata()
        a=build_schedule(*args,updates=3)
        b=build_schedule(*args,updates=3)
        self.assertTrue(np.array_equal(a[0],b[0]));self.assertTrue(np.array_equal(a[1],b[1]))
        self.assertEqual(a[2],b[2]);self.assertEqual(a[3],b[3])
        verify_schedule(a[0],a[1],*args,updates=3)
        self.assertFalse(np.array_equal(a[0][0],a[0][1]))
        for cell in range(54): self.assertEqual(a[0][:,cell*16:(cell+1)*16].shape,(3,16))

    def test_schedule_wrong_cell_or_group_rejected(self):
        args=self.metadata();a,b,_,_=build_schedule(*args,updates=1)
        a[0,0]=3056
        with self.assertRaises(ValueError): verify_schedule(a,b,*args,updates=1)
        a,b,_,_=build_schedule(*args,updates=1);b[0,0]=57
        with self.assertRaises(ValueError): verify_schedule(a,b,*args,updates=1)

    def test_metadata_control_shift_rejected(self):
        d,c,g=self.metadata();c[100]+=1
        with self.assertRaises(ValueError): center_cells(d,c,g)

    def test_endpoint_order_and_widening(self):
        centers=np.full(864,3056,dtype=np.int32);axes=np.full(864,57,dtype=np.int8)
        actual=endpoint_rows(centers,axes)
        self.assertEqual(actual.dtype,np.int64)
        np.testing.assert_array_equal(actual.reshape(864,2),np.tile([354610,354611],(864,1)))
        centers[0]=-1
        with self.assertRaises(ValueError): endpoint_rows(centers,axes)

    def test_inclusive_learning_rate(self):
        self.assertAlmostEqual(cosine_rate(0),1e-4,places=18)
        self.assertEqual(cosine_rate(9999),1e-5)
        self.assertGreater(cosine_rate(123),cosine_rate(124))
        with self.assertRaises(ValueError): cosine_rate(10000)

    def test_unequal_cells_signs_and_outputs_equal_cell_mean(self):
        # Distinct group/cell/sign/output patterns would expose accidental sum or axis-count weights.
        nominal=torch.zeros((1,23),dtype=torch.float32)
        ids=torch.zeros(864,dtype=torch.int64)
        target=torch.zeros((864,2,23),dtype=torch.float64)
        values=np.empty((54,16,2,23),dtype=np.float32)
        for cell in range(54):
            for pair in range(16):
                for sign in range(2): values[cell,pair,sign]=cell+1+pair*.125+sign*.5+np.arange(23)*.0625
        endpoints=torch.from_numpy(values.reshape(1728,23))
        loss,cells=full_state_loss(nominal,endpoints,ids,target)
        expected=np.mean(values.astype(np.float64)**2,axis=(1,2,3))
        np.testing.assert_allclose(cells.numpy(),expected,rtol=0,atol=1e-12)
        self.assertAlmostEqual(float(loss),float(expected.mean()),places=11)
        self.assertGreater(float(cells[-1]),float(cells[0]))

    def test_cast_before_subtraction_and_center_graph_retained(self):
        nominal=torch.full((1,23),-1.,dtype=torch.float32,requires_grad=True)
        endpoint=torch.full((1728,23),float(2**24),dtype=torch.float32)
        target=torch.full((864,2,23),float(2**24+1),dtype=torch.float64)
        loss,_=full_state_loss(nominal,endpoint,torch.zeros(864,dtype=torch.int64),target)
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(loss.requires_grad)

    def test_loss_rejects_wrong_dtype_or_batch(self):
        a=torch.zeros((1,23),dtype=torch.float32);b=torch.zeros((1728,23),dtype=torch.float32)
        ids=torch.zeros(864,dtype=torch.int64);target=torch.zeros((864,2,23),dtype=torch.float64)
        with self.assertRaises(ValueError): full_state_loss(a,b.double(),ids,target)
        with self.assertRaises(ValueError): full_state_loss(a,b,ids,target.float())
        with self.assertRaises(ValueError): full_state_loss(a,b[:-1],ids,target)

    def gradients(self):
        return dict(nominal=[np.array([3.,4.])]+[np.zeros(1)]*5,
                    full_state=[np.array([0.,2.])]+[np.zeros(1)]*5,
                    physical=[np.array([1.,0.])]+[np.zeros(1)]*5)

    def test_one_fixed_norm_calibration(self):
        report=calibrate(self.gradients())
        self.assertEqual(report['coefficient'],2.5)
        self.assertEqual(report['norms']['nominal'],5.)
        self.assertEqual(report['norms']['full_state'],2.)
        self.assertAlmostEqual(report['total_gradient_norm'],np.sqrt(97),places=14)
        self.assertFalse(report['dynamic_recalibration'])
        self.assertFalse(report['AdamW_or_finite_step_guarantee'])

    def test_calibration_zero_nonfinite_and_shape_fail(self):
        for value in (0.,np.nan,np.inf):
            grads=self.gradients();grads['full_state']=[np.array([value,0.])]+[np.zeros(1)]*5
            with self.assertRaises(ValueError): calibrate(grads)
        grads=self.gradients();grads['nominal']=[np.zeros(2)]*6
        with self.assertRaises(ValueError): calibrate(grads)

    def test_zero_physical_gradient_is_recorded_not_rejected(self):
        grads=self.gradients();grads['physical']=[np.zeros(2)]+[np.zeros(1)]*5
        result=calibrate(grads)
        self.assertEqual(result['coefficient'],2.5)
        self.assertEqual(result['norms']['physical'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
