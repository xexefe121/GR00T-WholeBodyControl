"""Synthetic accounting/serialization only. No native, worker or clock execution."""
import ast
import base64
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from packet_io import lossless,worker_verdict,require_roles,ready,write,sha,pin_check,save_owned
from run_clock import require_epoch


class RunnerTests(unittest.TestCase):
    def test_raw_exit_zero_does_not_override_worker_failure(self):
        cleanup={'normal_exit':True,'pid':7}
        report={'pid':7,'failure':None,'overflow':None,'native_steps':0,'model_calls':0}
        self.assertTrue(worker_verdict(cleanup,report,7))
        for name,value in [('failure',{'type':'ValueError'}),('overflow','record'),('pid',8),('native_steps',1)]:
            bad=dict(report);bad[name]=value;self.assertFalse(worker_verdict(cleanup,bad,7))
        self.assertFalse(worker_verdict({'start_side_effects_uncertain':True,'normal_exit':False},report,7))

    def test_terminated_worker_cannot_pass(self):
        self.assertFalse(worker_verdict({'normal_exit':False,'pid':7},{'pid':7,'failure':None},7))
        self.assertFalse(worker_verdict(None,None,None))

    def test_epoch_never_rebased(self):
        require_epoch(10,200000010,200000009)
        for triple in [(10,200000010,200000010),(10,200000010,9),(10,200000011,11),(True,200000001,2)]:
            with self.assertRaises(RuntimeError):require_epoch(*triple)

    def test_lossless_signed_zero_nan_and_bytes(self):
        raw=np.asarray([-0.,np.nan,np.inf],np.float64).tobytes()
        @dataclasses.dataclass(frozen=True)
        class Capture:state:bytes;time:float
        value=lossless(Capture(raw,-0.))
        self.assertEqual(base64.b64decode(value['fields']['state']['base64']),raw)
        self.assertEqual(value['fields']['time']['hex'],'-0x0.0p+0')
        json.dumps(value,allow_nan=False)
        with self.assertRaises(TypeError):lossless(object())

    def test_owned_nested_evidence_has_no_mutable_alias(self):
        original={'items':[{'value':b'abc'}]};copied=lossless(original)
        original['items'][0]['value']=b'def'
        self.assertEqual(base64.b64decode(copied['items'][0]['value']['base64']),'abc'.encode())

    def test_unpinned_role_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state';path.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'unpinned role'):
                require_roles({'input_files':[],'roles':{'fixture':str(path)},'native_bundle':folder})

    def test_missing_selection_and_wrong_literal_subject_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);request=base/'request.json';clearance=base/'clearance.json';review=base/'review.json'
            write(request,{});write(clearance,{'root_selected_single_run':False})
            with self.assertRaisesRegex(ValueError,'not selected'):ready(request,clearance)
            write(review,{'passed':True,'request_subject':{'path':str(request),'sha256':'0'*64}})
            clearance.unlink();write(clearance,{'root_selected_single_run':True,'request_sha256':sha(request),
                'review':{'path':str(review),'sha256':sha(review),'pass_field':'passed'}})
            with self.assertRaisesRegex(ValueError,'literal reviewed request'):ready(request,clearance)

    def test_streaming_pin_check_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input';path.write_bytes(b'a');entry={'path':str(path),'sha256':sha(path)}
            self.assertTrue(pin_check([entry])['all_exact']);path.write_bytes(b'b')
            self.assertFalse(pin_check([entry])['all_exact'])

    def test_zero_command_and_partial_capture_writer_keeps_schema(self):
        arrays={'packed_capture':np.empty((0,373),np.float64),'control_raw_action':np.empty((0,23),np.float32)}
        with tempfile.TemporaryDirectory() as folder,patch('evidence.export_owned',return_value=(arrays,{'outer_cycles':(b'{"index":0}',)}, {'initial_capture_state':bytes(373*8)})):
            result=save_owned(folder,object())
            with np.load(Path(folder)/'trace.npz') as z:self.assertEqual(z['control_raw_action'].shape,(0,23))
            self.assertEqual((Path(folder)/'raw_capsules/initial_capture_state.bin').read_bytes(),bytes(373*8))
            self.assertEqual(result['arrays']['packed_capture']['dtype'],'<f8')

    def test_sources_do_not_call_real_apis_at_import(self):
        module=ast.parse(Path(__file__).with_name('run_clock.py').read_text())
        top_calls=[n for n in module.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
        self.assertEqual(top_calls,[])
        text=Path(__file__).with_name('run_clock.py').read_text()
        self.assertNotIn('mj_step(',text)
        self.assertNotIn('InferenceSession(',text)


if __name__=='__main__':unittest.main()
