"""Injected call/capture tests; no task weights, GPU or ORT sessions."""
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import run_fp64 as driver
class RunnerTests(unittest.TestCase):
    def test_consumed_role_membership_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'subject.bin';path.write_bytes(b'fixed subject');key=path.as_posix();digest=driver.sha(path)
            request=dict(subjects={'checkpoint':dict(path=key,sha256=digest)},old_outputs={'GPU':{'nominal':key}})
            receipt=dict(input_sha256={key:digest});driver.check_consumed_roles(request,receipt)
            with self.assertRaises(ValueError):driver.check_consumed_roles(request,dict(input_sha256={}))
            request['subjects']['checkpoint']['sha256']='0'*64
            with self.assertRaises(ValueError):driver.check_consumed_roles(request,receipt)
            request['subjects']['checkpoint']['sha256']=digest;path.write_bytes(b'changed')
            with self.assertRaises(ValueError):driver.check_consumed_roles(request,receipt)
    def run_fixture(self,model=None,session=None,backend='CPU64'):
        temporary=tempfile.TemporaryDirectory();path=Path(temporary.name);active={};outputs={backend:{}};ledger=io.StringIO()
        # StringIO has no file descriptor; an injected in-memory ledger supplies one.
        ledger=open(path/'ledger.jsonl','w+',encoding='utf-8')
        counters={backend:dict(calls_attempted=0,calls_returned=0,calls_synchronized=0,calls_verified=0,rows_attempted=0,rows_returned=0,rows_verified=0)}
        error=None
        with patch.multiple(driver,CORPORA=('nominal',),SIZES=(3,),SOURCE_KEYS=('features',),BATCH=2):
            try:driver.run_backend(backend,model,session,{'features':np.zeros((3,1000),np.float32)},path,counters,active,ledger,outputs)
            except Exception as e:error=e
        ledger.seek(0);records=[json.loads(line) for line in ledger];ledger.close()
        arrays={k:np.asarray(v).copy() for k,v in outputs[backend].items()}
        for v in outputs[backend].values():v._mmap.close()
        temporary.cleanup();return counters[backend],active,records,arrays,error
    def test_success_exact_counts_and_prefix(self):
        c,a,log,out,error=self.run_fixture(model=lambda x:torch.ones((len(x),23),dtype=torch.float32))
        self.assertIsNone(error);self.assertEqual(c['calls_verified'],2);self.assertEqual(c['rows_verified'],3)
        self.assertEqual([(x['start'],x['stop']) for x in log],[(0,2),(2,3)]);self.assertTrue(np.all(out['nominal']==1))
    def test_second_call_failure_clears_old_output(self):
        calls=[]
        def model(x):
            calls.append(len(x))
            if len(calls)==2:raise RuntimeError('injected failure')
            return torch.ones((len(x),23),dtype=torch.float32)
        c,a,log,out,error=self.run_fixture(model=model)
        self.assertIsInstance(error,RuntimeError);self.assertEqual(c['calls_attempted'],2);self.assertEqual(c['calls_returned'],1)
        self.assertEqual(c['rows_attempted'],3);self.assertEqual(c['rows_verified'],2);self.assertNotIn('output',a);self.assertFalse(a['returned'])
        self.assertTrue(np.all(out['nominal'][:2]==1));self.assertTrue(np.isnan(out['nominal'][2]).all());self.assertFalse(log[-1]['verified'])
    def test_wrong_ORT_shape_retains_raw_return(self):
        class Stub:
            def run(self,*args):return [np.ones((2,22),np.float32)]
        c,a,log,out,error=self.run_fixture(session=Stub(),backend='ORT64')
        self.assertIsInstance(error,ValueError);self.assertEqual(c['calls_returned'],1);self.assertEqual(c['calls_verified'],0)
        self.assertEqual(a['all_outputs'][0].shape,(2,22));self.assertTrue(np.isnan(out['nominal']).all())
    def test_nonfinite_output_is_saved_before_rejection(self):
        c,a,log,out,error=self.run_fixture(model=lambda x:torch.full((len(x),23),float('nan')))
        self.assertIsInstance(error,ValueError);self.assertEqual(c['rows_verified'],0);self.assertEqual(c['calls_returned'],1)
        self.assertTrue(np.isnan(a['output']).all());self.assertTrue(np.isnan(out['nominal']).all())
if __name__=='__main__':unittest.main()
