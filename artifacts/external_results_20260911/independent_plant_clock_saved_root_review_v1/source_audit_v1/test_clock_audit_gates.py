"""Temporary synthetic JSON and source hashes only; no runtime task outputs."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_clock
from audit_clock import validate_request,sha,write
from prepare_request import prepare
from clock_accounting import saved_timing_counters
from clock_saved_math import timing_arrays
from test_clock_accounting import count_case


class GateTests(unittest.TestCase):
    def request(self,folder):
        source=Path(audit_clock.__file__).parent;source_hashes={p.name:sha(p) for p in source.glob('*.py')}
        review=folder/'review.json';write(review,dict(passed=True,source_sha256=source_hashes))
        input_file=folder/'input.json';write(input_file,dict(synthetic=True))
        pins={str(source/k):v for k,v in source_hashes.items()};pins[str(review)]=sha(review);pins[str(input_file)]=sha(input_file)
        request=dict(kind='independent_saved_clock_audit',root_selected_saved_audit=True,numpy_version='1.26.4',input_sha256=pins,
            source_sha256=source_hashes,source_review=dict(path=str(review),sha256=sha(review),pass_field='passed'),roles=dict(synthetic=str(input_file)))
        return request
    def test_frozen_source_review_and_role_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);r=self.request(folder);path=folder/'request.json';write(path,r)
            with patch.object(np,'__version__','1.26.4'):self.assertEqual(validate_request(path)[0]['roles'],r['roles'])
    def test_unpinned_role_missing_source_and_changed_review_rejected(self):
        for mutation in ('role','source','review'):
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);r=self.request(folder)
                if mutation=='role':r['roles']['synthetic']=str(folder/'unbound')
                if mutation=='source':r['input_sha256'].pop(str(Path(audit_clock.__file__).parent/'audit_clock.py'))
                if mutation=='review':r['source_review']['sha256']='0'*64
                path=folder/'request.json';write(path,r)
                with patch.object(np,'__version__','1.26.4'),self.assertRaises(ValueError):validate_request(path)
    def test_selection_and_actual_numpy_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);r=self.request(folder);r['root_selected_saved_audit']=False;path=folder/'request.json';write(path,r)
            with self.assertRaisesRegex(ValueError,'selection'):validate_request(path)
        with self.assertRaisesRegex(ValueError,'selection'):prepare('missing','missing','passed',False)
    def test_request_preparer_will_not_accept_placeholder_source_review(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);review=folder/'review.json';write(review,dict(passed=True,source_sha256={}))
            with self.assertRaisesRegex(ValueError,'Literal current source'):prepare(folder,review,'passed',True)
    def test_debt_and_outer_summary_corruption_rejected(self):
        a,m,r,c,initial,contract,outer,epoch=count_case()
        timing=timing_arrays(a,outer,epoch);r['session']['outer_deadline_miss_indices']=[]
        self.assertTrue(saved_timing_counters(a,r,timing)['complete_counter_reconstruction'])
        r['session']['foundation']['cumulative_wake_debt']=1
        with self.assertRaisesRegex(AssertionError,'debt/deadline'):saved_timing_counters(a,r,timing)
    def test_native_calls_forbidden_in_auditor_source(self):
        import ast
        source=Path(audit_clock.__file__).parent
        forbidden={'mujoco','onnxruntime','torch','subprocess','multiprocessing'}
        for path in source.glob('*.py'):
            if path.name.startswith('test_'):continue
            tree=ast.parse(path.read_text())
            modules=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):modules.extend(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node,ast.ImportFrom):modules.append((node.module or '').split('.')[0])
            self.assertFalse(set(modules)&forbidden,path.name)


if __name__=='__main__':unittest.main()
