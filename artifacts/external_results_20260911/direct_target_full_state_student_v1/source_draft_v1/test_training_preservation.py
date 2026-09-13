"""Injected CPU tensors and temporary synthetic files only; no task models or CUDA calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from training_support import counter,counted_forward,save_loss_prefix,flatten_active
from full_state_diagnostics import parity,save_drift

class ForwardPreservation(unittest.TestCase):
    def test_return_saved_before_nonfinite_check(self):
        active={'nominal_prediction':torch.ones((1,23))};counts=counter()
        with patch('torch.cuda.synchronize') as sync:
            with self.assertRaises(ValueError):counted_forward(lambda x:torch.full((1,23),float('nan')),torch.zeros((1,1000)),'nominal',counts,active,3,10)
        self.assertEqual(sync.call_count,1);self.assertTrue(torch.isnan(active['nominal_prediction']).all())
        self.assertEqual(counts['calls_returned'],1);self.assertEqual(counts['calls_verified'],0)
        self.assertTrue(active['current_forward']['returned']);self.assertFalse(active['current_forward']['verified'])

    def test_call_failure_clears_only_current_output(self):
        active={'nominal_prediction':torch.ones((1,23)),'full_state_prediction':torch.ones((2,23))};counts=counter()
        def fail(x):raise RuntimeError('synthetic graph failure')
        with self.assertRaises(RuntimeError):counted_forward(fail,torch.zeros((2,1000)),'full_state',counts,active,3,10)
        self.assertNotIn('full_state_prediction',active);self.assertIn('nominal_prediction',active)
        self.assertEqual(counts['calls_attempted'],1);self.assertEqual(counts['calls_returned'],0)
        self.assertFalse(active['current_forward']['returned'])

    def test_sync_failure_retains_return_and_not_verified(self):
        active={};counts=counter()
        with patch('torch.cuda.synchronize',side_effect=RuntimeError('synthetic sync failure')):
            with self.assertRaises(RuntimeError):counted_forward(lambda x:torch.ones((2,23)),torch.zeros((2,1000)),'nominal',counts,active,3,10)
        self.assertEqual(counts['rows_returned'],2);self.assertEqual(counts['rows_verified'],0)
        self.assertTrue(active['current_forward']['returned']);self.assertFalse(active['current_forward']['synchronized'])

    def test_budget_exhaustion_does_not_call_model(self):
        counts=counter();counts['calls_attempted']=3
        called=[]
        with self.assertRaises(ValueError):counted_forward(lambda x:called.append(True),torch.zeros((1,1000)),'nominal',counts,{},3,10)
        self.assertEqual(called,[]);self.assertEqual(counts['calls_attempted'],3)

    def test_loss_prefix_preserves_empty_shapes_and_committed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);save_loss_prefix(path,[],[],[],[])
            for name,width in [('training_progress',6),('nominal_cell_losses',15),('full_state_cell_losses',54),('physical_cell_losses',9)]:
                actual=np.load(path/(name+'.npy'));self.assertEqual(actual.shape,(0,width));self.assertEqual(actual.dtype,np.float64)
            save_loss_prefix(path,[[1,2,3,4,5,6]],[[1]*15],[[2]*54],[[3]*9])
            np.testing.assert_array_equal(np.load(path/'full_state_cell_losses.npy'),np.full((1,54),2.))

    def test_active_nested_returns_keep_identity_and_dtype(self):
        evidence={};flatten_active({'stage':'calibration','nominal_prediction':torch.ones((2,23)),
            'current_gradient':dict(name='physical',returned=False)},'active',evidence)
        self.assertEqual(evidence['active_nominal_prediction'].dtype,np.float32)
        self.assertEqual(str(evidence['active_current_gradient_name']),'physical')
        self.assertFalse(bool(evidence['active_current_gradient_returned']))

class ExportArithmetic(unittest.TestCase):
    def data(self):
        return dict(default=np.zeros(23,np.float64),span=np.full(23,2.,np.float32),limits=np.tile([-1.,1.],(23,1)))

    def outputs(self):
        return {label:{corpus:np.zeros((2,23),np.float32) for corpus in ('nominal','full_state','physical')}
                for label in ('CPU64','GPU64','ORT64','final_GPU32')}

    def test_preclamp_gate_rejects_difference_even_both_clip(self):
        outputs=self.outputs()
        for label in outputs:outputs[label]['nominal'][0,0]=2.
        outputs['ORT64']['nominal'][0,0]=3.
        report=parity(outputs,self.data());self.assertFalse(report['passed']);self.assertEqual(report['maximum_preclamp_rad'],2.)

    def test_nonfinite_parity_never_ignored_by_max(self):
        outputs=self.outputs();outputs['GPU64']['physical'][0,0]=np.nan
        report=parity(outputs,self.data())
        self.assertFalse(report['passed']);self.assertIsNone(report['maximum_preclamp_rad'])
        self.assertEqual(len(report['nonfinite_comparisons']),2)

    def test_nine_drift_arrays_and_exact_mask_change_counts(self):
        outputs=self.outputs();outputs['CPU64']['nominal'][0,0]=.75
        with tempfile.TemporaryDirectory() as folder:
            result=save_drift(outputs,self.data(),Path(folder));self.assertEqual(len(result),9)
            row=result['CPU64_vs_final_GPU32_nominal'];self.assertEqual(row['old_clipped_components'],0)
            self.assertEqual(row['new_clipped_components'],1);self.assertEqual(row['clipping_changed_rows'],1)
            array=np.load(Path(folder)/'drift_CPU64_vs_final_GPU32_nominal.npy')
            self.assertEqual(array.dtype,np.float64);self.assertEqual(array[0,0],1.5)

if __name__=='__main__':unittest.main(verbosity=2)
