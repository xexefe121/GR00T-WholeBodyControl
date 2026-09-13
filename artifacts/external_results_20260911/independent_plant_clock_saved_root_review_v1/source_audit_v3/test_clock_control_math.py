"""Synthetic control/evidence mutation checks; no actual task arrays or native calls."""
import base64,copy,hashlib,importlib.util
from pathlib import Path
import unittest
import numpy as np
from clock_control_math import SIZES,decode,terms,controls,named_bytes

class ControlMathTests(unittest.TestCase):
    def test_owned_bytes_digest_and_float_bits(self):
        raw=b'\x00\xffnative'
        packed=dict(type='bytes',base64=base64.b64encode(raw).decode(),length=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(decode(packed),raw)
        packed['sha256']='0'*64
        with self.assertRaisesRegex(AssertionError,'hash'):decode(packed)
        self.assertTrue(np.signbit(decode({'type':'float','hex':'-0x0.0p+0'})))

    def test_independent_measured_terms_match_original_on_synthetic_rotations(self):
        path=Path(__file__).resolve().parents[2]/'independent_native_stepper_v1/source_draft_v3/bfm_observations.py'
        spec=importlib.util.spec_from_file_location('original_observation_reference',path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        rng=np.random.default_rng(4)
        for _ in range(15):
            state=rng.normal(size=291);quat=state[4:8];quat/=np.linalg.norm(quat)
            incoming=rng.normal(size=23).astype(np.float32);default=rng.normal(size=23)
            q,v=state[1:31],state[31:60]
            expected=original.state_and_terms(q[7:],v[6:],q[3:7],v[3:6],incoming,default)[1]
            actual=terms(state,incoming,default)
            for key in SIZES:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())

    def test_named_history_rejects_wrong_order_and_length(self):
        good=[(k,np.zeros((4,size),np.float32).tobytes()) for k,size in sorted(SIZES.items())]
        self.assertEqual(set(named_bytes(good,SIZES,4)),set(SIZES))
        with self.assertRaisesRegex(AssertionError,'order'):named_bytes(list(reversed(good)),SIZES,4)
        altered=copy.deepcopy(good);altered[0]=(altered[0][0],b'')
        with self.assertRaisesRegex(AssertionError,'schema'):named_bytes(altered,SIZES,4)

    def test_zero_control_partial_failure_never_qualifies(self):
        a=dict(control_control=np.empty(0,np.int64),step_index=np.empty(0,np.int64),control_physics=np.empty(0,np.int64),
            control_integration=np.empty((0,291)),control_target=np.empty((0,23)),
            control_raw_action=np.empty((0,23),np.float32),control_incoming_raw=np.empty((0,23),np.float32),
            control_flat_history_before=np.empty((0,300),np.float32),control_held=np.empty(0,bool),
            control_nominal_source_frame=np.empty(0,np.int64),control_nominal_activation_ns=np.empty(0,np.int64),
            control_actual_activation_ns=np.empty(0,np.int64),control_admitted_ns=np.empty(0,np.int64))
        metadata={k:[] for k in ('control_history_before','control_history_after','control_measured_terms',
            'control_command_ids','control_nominal_windows','control_active_windows')}
        full=dict(target=np.zeros((1569,23)),action=np.zeros((1569,23),np.float32))
        hold=dict(target=np.zeros((250,23)),action=np.zeros((250,23),np.float32))
        result=controls(a,metadata,np.zeros(291),dict(default_q=np.zeros(23)),full,hold,100)
        self.assertFalse(result['exact_recorded_commands'])
        self.assertEqual(result['control_count'],0)

if __name__=='__main__':unittest.main()
