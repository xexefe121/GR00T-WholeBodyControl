"""Tiny saved-file metadata fixtures; no probe clocks, native state or workers."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from sidecar_owner_evidence import check_sidecars,FILES

def fixture(folder):
    hooks={'failed':False};entries={}
    meta={'instrumentation_complete':True,'hook_status':hooks}
    for name in FILES:
        raw=json.dumps(meta).encode() if name=='timing_metadata.json' else b'{}' if name.endswith('.json') else b''
        (folder/name).write_bytes(raw);entries[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    info=dict(enabled=True,complete=True,files=copy.deepcopy(entries),errors=[],hook_status=hooks,new_native_reads=0)
    return {'timing_instrumentation':info,'component_preliminary_pass':True},{'files':entries}

class SidecarOwner(unittest.TestCase):
    def test_complete_identity_without_runtime_claim(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);report,manifest=fixture(folder);result=check_sidecars(folder,report,manifest)
            self.assertTrue(result['instrumentation_complete']);self.assertTrue(result['independent_sidecar_math_pending'])
            self.assertFalse(result['verification_return_or_native_credit_inferred'])
    def test_disabled_pre_setup_failure_remains_accountable(self):
        with tempfile.TemporaryDirectory() as name:
            report={'component_preliminary_pass':False,'timing_instrumentation':dict(enabled=False,complete=False,files={},errors=[])}
            result=check_sidecars(Path(name),report,{'files':{}})
            self.assertTrue(result['evidence_accounted']);self.assertFalse(result['instrumentation_complete'])
    def test_writer_missing_sidecar_preserved_not_qualified(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);report,manifest=fixture(folder);target='timing_gc.bin'
            (folder/target).unlink();del report['timing_instrumentation']['files'][target];del manifest['files'][target]
            report['component_preliminary_pass']=False;report['timing_instrumentation'].update(complete=False,errors=[{'phase':'gc','type':'OSError','detail':'synthetic'}])
            result=check_sidecars(folder,report,manifest);self.assertEqual(result['missing_sidecar_files'],[target])
    def test_unclaimed_partial_bytes_preserved_with_writer_error(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);report,manifest=fixture(folder);target='timing_gc.bin'
            del report['timing_instrumentation']['files'][target]
            report['component_preliminary_pass']=False;report['timing_instrumentation'].update(complete=False,errors=[{'phase':'gc'}])
            result=check_sidecars(folder,report,manifest);self.assertIn(target,result['unclaimed_partial_files'])
    def test_fault_or_pending_probe_does_not_qualify(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);report,manifest=fixture(folder)
            meta={'instrumentation_complete':False,'hook_status':{'failed':True}}
            raw=json.dumps(meta).encode();(folder/'timing_metadata.json').write_bytes(raw)
            entry={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            manifest['files']['timing_metadata.json']=entry;report['timing_instrumentation']['files']['timing_metadata.json']=entry
            report['timing_instrumentation'].update(complete=False,hook_status=meta['hook_status']);report['component_preliminary_pass']=False
            self.assertFalse(check_sidecars(folder,report,manifest)['instrumentation_complete'])
    def test_identity_and_completion_mutations_reject(self):
        mutations=[lambda r,m:r['timing_instrumentation']['files']['timing_gc.bin'].update(sha256='a'*64),
            lambda r,m:m['files']['timing_gc.bin'].update(bytes=1),
            lambda r,m:r['timing_instrumentation'].update(new_native_reads=1),
            lambda r,m:r['timing_instrumentation'].update(complete=1),
            lambda r,m:r['timing_instrumentation'].update(hook_status={'failed':True}),
            lambda r,m:r['timing_instrumentation']['files'].update({'../escape':{}})]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as name:
                folder=Path(name);report,manifest=fixture(folder);mutate(report,manifest)
                with self.assertRaises(ValueError):check_sidecars(folder,report,manifest)
    def test_unclaimed_without_error_rejects(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);report,manifest=fixture(folder);del report['timing_instrumentation']['files']['timing_gc.bin']
            with self.assertRaises(ValueError):check_sidecars(folder,report,manifest)

if __name__=='__main__':unittest.main()
