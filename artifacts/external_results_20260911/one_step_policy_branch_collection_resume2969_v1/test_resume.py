"""Bounded metadata-retry and copied-evidence tests only."""
import ast,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'draft'))
import collection_arrays as arrays
from resume_prefix import PREFIX_ROWS,PREFIX_STEPS
class ResumeTests(unittest.TestCase):
    def test_sharing_retry_uses_same_temp_and_never_repeats_work(self):
        class Temporary:
            def __init__(self):self.calls=[]
            def replace(self,target):
                self.calls.append(target)
                if len(self.calls)<3:raise PermissionError('simulated transient reader')
        temporary=Temporary()
        with patch.object(arrays.time,'sleep') as sleep:arrays.replace_with_sharing_retry(temporary,'target')
        self.assertEqual(temporary.calls,['target']*3);self.assertEqual(sleep.call_count,2)
    def test_retry_exhaustion_preserved(self):
        class Temporary:
            calls=0
            def replace(self,target):self.calls+=1;raise PermissionError('still locked')
        temporary=Temporary()
        with patch.object(arrays.time,'sleep') as sleep:
            with self.assertRaises(PermissionError):arrays.replace_with_sharing_retry(temporary,'target')
        self.assertEqual(temporary.calls,41);self.assertEqual(sleep.call_count,40)
    def test_other_filesystem_failure_not_retried(self):
        class Temporary:
            calls=0
            def replace(self,target):self.calls+=1;raise OSError('other filesystem failure')
        temporary=Temporary()
        with patch.object(arrays.time,'sleep') as sleep:
            with self.assertRaises(OSError):arrays.replace_with_sharing_retry(temporary,'target')
        self.assertEqual(temporary.calls,1);self.assertEqual(sleep.call_count,0)
    def test_copied_store_is_byteexact_and_original_unchanged(self):
        old=arrays.ROWS;arrays.ROWS=2
        try:
            with tempfile.TemporaryDirectory() as d:
                original=arrays.Store(Path(d)/'original');original.row(0,nominal_verified=True,nominal_status=1,nominal_valid_steps=10)
                manifest=original.manifest(dict(complete=False));copied=arrays.Store(Path(d)/'copied',resume_directory=original.folder)
                for key,spec in manifest['arrays'].items():self.assertEqual(arrays.sha(copied.folder/spec['path']),spec['sha256'])
                copied.row(1,nominal_verified=True);copied.flush();self.assertFalse(original.arrays['nominal_verified'][1])
                for key,spec in manifest['arrays'].items():self.assertEqual(arrays.sha(original.folder/spec['path']),spec['sha256'])
                for store in (original,copied):
                    for value in store.arrays.values():value._mmap.close()
        finally:arrays.ROWS=old
    def test_budget_and_no_repeat_guard(self):
        self.assertEqual(PREFIX_ROWS,2969);self.assertEqual(PREFIX_STEPS,29690)
        self.assertEqual((3054-PREFIX_ROWS+3054)*10,61080-PREFIX_STEPS)
        source=(BASE/'draft/collect_branches.py').read_text()
        self.assertIn('if row<PREFIX_ROWS:continue',source)
        self.assertIn('ENGINE.total_attempted=ENGINE.total_returned=PREFIX_STEPS',source)
        self.assertIn('if self.total_attempted>=61080', (BASE/'draft/native_capture.py').read_text())
        for p in (BASE/'draft').rglob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':
    with (BASE/'resume_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests))
    assert result.wasSuccessful()
    arrays.atomic(BASE/'resume_tests.json',dict(passed=True,tests=result.testsRun,physical_native_calls=0,graph_calls=0,
        source_sha256={p.relative_to(BASE/'draft').as_posix():arrays.sha(p) for p in (BASE/'draft').rglob('*.py')},log_sha256=arrays.sha(BASE/'resume_tests.log')))
    print(json.dumps(dict(passed=True,tests=result.testsRun,physical_native_calls=0,graph_calls=0)))
