"""Synthetic saved arrays only; no actual native, clock or task input calls."""
import copy
import unittest
import numpy as np
from clock_saved_math import physical_arrays,timing_arrays,compare_expert_prefix,repeated_time,first_difference


def fixture(n=2):
    initial=np.zeros(291,np.float64)
    initial[3]=.75
    initial[4]=1.
    packed=np.tile(np.r_[initial,initial[1:31],initial[31:60],np.zeros(23)],(n,1))
    packed[:,0]=repeated_time(0.,n)
    a=dict(packed_capture=packed,step_integration=packed[:,:291].copy(),step_qpos=packed[:,291:321].copy(),
        step_qvel=packed[:,321:350].copy(),step_actuator_force=packed[:,350:].copy(),
        step_command=np.zeros((n,23)),step_target=np.zeros((n,23)),step_raw_action=np.zeros((n,23),np.float32),
        step_warning_counts=np.zeros((n,8),np.int32),step_warning_lastinfo=np.zeros((n,8),np.int32),
        step_expected_simulation_time=packed[:,0].copy(),step_verified=np.ones(n,bool),step_index=np.arange(n,dtype=np.int64))
    epoch=100_000_000
    a['step_nominal_start']=epoch+np.arange(n,dtype=np.int64)*2_000_000
    a['step_nominal_end']=a['step_nominal_start']+2_000_000
    a['step_actual_start']=a['step_nominal_start']+100
    a['step_actual_end']=a['step_actual_start']+100_000
    a['step_wake_lateness_ns']=np.full(n,100,np.int64)
    a['step_finish_lateness_ns']=np.zeros(n,np.int64)
    a['step_wake_debt']=np.zeros(n,np.int64)
    outer=[dict(index=i,returned=i+1,cycle_return_ns=int(a['step_actual_end'][i])+100,
        fixed_nominal_end_ns=int(a['step_nominal_end'][i])) for i in range(n)]
    contract=dict(kp=np.ones(23),kd=np.ones(23),native_effort=np.full(23,100.),native_velocity=np.full(23,20.),
        joint_limits=np.tile([-1.,1.],(23,1)))
    return a,initial,contract,outer,epoch


class SavedClockTests(unittest.TestCase):
    def test_partial_valid_arrays_cannot_qualify_full_scope(self):
        a,initial,c,outer,epoch=fixture()
        r=physical_arrays(a,initial,c)
        self.assertTrue(r['all_actual_saved_states_strict'])
        self.assertFalse(r['full_scope'])
        t=timing_arrays(a,outer,epoch)
        self.assertTrue(t['complete_outer_coverage'])
        self.assertFalse(t['fixed_epoch_timing_pass'])

    def test_manual_pd_corruption_rejected(self):
        a,initial,c,_,_=fixture()
        a['step_target'][0,0]=.2
        with self.assertRaisesRegex(AssertionError,'manual PD'):physical_arrays(a,initial,c)

    def test_duplicate_capture_slice_corruption_rejected(self):
        a,initial,c,_,_=fixture()
        a['step_qpos'][0,7]=.1
        with self.assertRaisesRegex(AssertionError,'packed slices'):physical_arrays(a,initial,c)

    def test_strict_failed_capture_preserved(self):
        a,initial,c,_,_=fixture(1)
        a['packed_capture'][0,37]=21.
        a['packed_capture'][0,327]=21.
        a['step_integration']=a['packed_capture'][:,:291].copy()
        a['step_qvel']=a['packed_capture'][:,321:350].copy()
        a['step_verified'][0]=False
        r=physical_arrays(a,initial,c)
        self.assertEqual(r['strict_failures']['joint_speed'],[0])
        self.assertFalse(r['all_actual_saved_states_strict'])

    def test_false_strict_success_rejected(self):
        a,initial,c,_,_=fixture(1)
        a['step_warning_counts'][0,0]=1
        with self.assertRaisesRegex(AssertionError,'strict predicates'):physical_arrays(a,initial,c)

    def test_outer_deadline_failure_separate_from_native(self):
        a,_,_,outer,epoch=fixture()
        outer[-1]['cycle_return_ns']=outer[-1]['fixed_nominal_end_ns']+1
        r=timing_arrays(a,outer,epoch)
        self.assertEqual(r['captured_step_deadline_misses'],[])
        self.assertEqual(r['outer_deadline_misses'],[1])

    def test_rebased_deadline_rejected(self):
        a,_,_,outer,epoch=fixture()
        a['step_nominal_start'][1]+=1
        with self.assertRaisesRegex(AssertionError,'Fixed nominal starts'):timing_arrays(a,outer,epoch)

    def test_nonmonotonic_actual_clock_rejected(self):
        a,_,_,outer,epoch=fixture()
        a['step_actual_end'][1]=a['step_actual_start'][1]-1
        with self.assertRaisesRegex(AssertionError,'timing order'):timing_arrays(a,outer,epoch)

    def test_signed_zero_and_empty_difference(self):
        a=np.zeros((2,23),np.float32);b=a.copy();b[1,3]=-0.
        self.assertEqual(first_difference(a,b),1)
        self.assertIsNone(first_difference(a[:0],b[:0]))

    def test_zero_step_failure_preserves_shapes(self):
        a,initial,c,outer,epoch=fixture(0)
        self.assertEqual(physical_arrays(a,initial,c)['captured_steps'],0)
        self.assertFalse(timing_arrays(a,outer,epoch)['fixed_epoch_timing_pass'])

if __name__=='__main__':unittest.main()
