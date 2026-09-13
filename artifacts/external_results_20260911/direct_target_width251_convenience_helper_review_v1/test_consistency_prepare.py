"""Synthetic JSON/hash assembly only; no task archives or actual request."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
SOURCE=NEW/'direct_target_width251_consistency_v1/prepare_actual_request.py'
ROOT_SHA='1100d985008254ed4fb92363c0c0b0cb930ebf64ed34937d866fbca98d4404fa'
def module():
    spec=importlib.util.spec_from_file_location('review_consistency_prepare',SOURCE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class ConsistencyPrepare(unittest.TestCase):
    def case(self,rejected=False):
        m=module();reads=[];gate=[]
        source_map={p.name:'a'*64 for p in m.SOURCE.glob('*.py')}
        def read(path):
            path=Path(path);reads.append(path.name)
            if path.suffix!='.json':raise AssertionError('Preparation tried to JSON-read numerical/source file')
            if path.name=='training_request.json':return {'paths':{k:'E:/synthetic/'+k+('.json' if k in ('physical_manifest','contract') else '.npz') for k in ('centers','pico','walk002','physical_manifest','contract')}}
            if path.name=='physical_manifest.json':return {'arrays':{k:{'path':k+'.npy'} for k in m.PHYSICAL_KEYS}}
            if path.name=='root_review.json':return {'source_sha256':source_map}
            return {}
        def report_gate(subjects,reports,physical):
            gate.append((subjects,reports,physical))
            if rejected:raise ValueError('synthetic unqualified collection')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);error=None
            with (patch.object(m,'BASE',base),patch.object(m,'read',side_effect=read),
                  patch.object(m,'sha',side_effect=lambda p:ROOT_SHA if Path(p).name=='root_review.json' else 'a'*64),
                  patch.object(m,'report_gate',side_effect=report_gate),
                  patch.object(sys,'argv',['fake','--collection-report-sha256','a'*64]),patch('builtins.print')):
                try:m.main()
                except BaseException as exc:error=exc
            path=base/'actual_v1/request.json';record=json.loads(path.read_text()) if path.exists() else None
            return m,record,error,reads,gate
    def test_exact_saved_scope_and_report_only_reads(self):
        m,record,error,reads,gate=self.case();self.assertIsNone(error);self.assertEqual(len(gate),1)
        self.assertEqual(set(record['subjects']),set(m.ROLES));self.assertEqual(set(record['physical_arrays']),set(m.PHYSICAL_KEYS))
        self.assertEqual(record['scope'],dict(old_nominal=9904,old_physical=3054,new_rows=1018,alias_modes=6,proximity_queries=72,candidate_block=256))
        self.assertEqual((record['model_calls'],record['native_steps'],record['optimizer_updates']),(0,0,0))
        self.assertFalse(record['actual_numeric_arrays_read']);self.assertTrue(all(p.endswith('.json') for p in reads))
    def test_failed_collection_gate_precedes_request_creation(self):
        _,record,error,_,gate=self.case(True);self.assertIsInstance(error,ValueError);self.assertIsNone(record);self.assertEqual(len(gate),1)

if __name__=='__main__':unittest.main()
