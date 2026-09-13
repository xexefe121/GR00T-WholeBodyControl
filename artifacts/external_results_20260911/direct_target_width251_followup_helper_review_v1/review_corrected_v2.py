"""Focused corrected hold-routing metadata proof; no actual arrays or audits."""
import copy,difflib,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=NEW/'direct_target_width251_followup_design_v1'
NAMES=('run_saved_comparison.py','prepare_audit_commands.py','prepare_qualified_collection.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module():
    spec=importlib.util.spec_from_file_location('corrected_audit_commands',SOURCE/'prepare_audit_commands.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class Corrected(unittest.TestCase):
    def run_case(self,main_complete=True,hold_complete=False,trace=True,report=True,skipped=False):
        mod=module()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'recovery';output=Path(directory)/'synthetic_commands.json'
            owned={str(base/'nominal'/name):'a'*64 for name in ('trace.npz','report.json')}
            if trace:owned[str(base/'post_lifecycle_hold_5s/trace.npz')]='a'*64
            if report:owned[str(base/'post_lifecycle_hold_5s/report.json')]='a'*64
            owner=dict(completion_accounting_passed=True,raw_exit_known=True,processes_absent=True,
                all_postrun_pins_exact=True,accounting_uncertainty=[],requested_recovery_completed=main_complete and hold_complete,output_sha256=owned)
            def read(path):
                path=Path(path)
                if path.name=='owner_completion.json':return copy.deepcopy(owner)
                if path.parent.name=='nominal':return dict(requested_controls=1569,full_segment_completed=main_complete)
                if path.parent.name=='post_lifecycle_hold_5s':
                    result=dict(requested_controls=250,full_segment_completed=hold_complete)
                    if skipped:result.update(attempted_controls=0,not_run_reason='main incomplete')
                    return result
                raise AssertionError('Unexpected source metadata read')
            with patch.object(mod,'read',side_effect=read),patch.object(mod,'sha',return_value='a'*64),patch.object(sys,'argv',['helper','--base',str(base),'--owner-sha','a'*64,'--output',str(output)]),patch('sys.stdout',new=io.StringIO()):mod.main()
            return json.loads(output.read_text())
    def test_partial_hold_gets_all_original_scope_audits(self):
        result=self.run_case();self.assertEqual([x['stage'] for x in result['commands']],['main_physics','main_intent','hold_fixture','hold_physics','hold_intent'])
        self.assertEqual([x['native_step_max'] for x in result['commands']],[15690,0,0,2500,0])
        self.assertIn('250',result['commands'][3]['arguments']);self.assertIn('--preceding-trace',result['commands'][4]['arguments'])
    def test_completed_hold_still_identical_scope(self):self.assertEqual(len(self.run_case(hold_complete=True)['commands']),5)
    def test_hold_without_full_main_rejected(self):
        with self.assertRaisesRegex(AssertionError,'full original main'):self.run_case(main_complete=False)
    def test_recorded_hold_without_report_rejected(self):
        with self.assertRaisesRegex(AssertionError,'owner-bound report'):self.run_case(report=False)
    def test_skipped_hold_report_retained_without_native_audit(self):
        result=self.run_case(main_complete=False,trace=False,skipped=True)
        self.assertEqual(len(result['commands']),2);self.assertTrue(any('/post_lifecycle_hold_5s/report.json' in p for p in result['input_sha256']))
    def test_missing_trace_with_non_skipped_report_rejected(self):
        with self.assertRaisesRegex(AssertionError,'not a skipped'):self.run_case(trace=False)
    def test_completed_owner_without_trace_rejected(self):
        with self.assertRaisesRegex(AssertionError,'both hold'):self.run_case(hold_complete=True,trace=False)

def main():
    original=json.loads((BASE/'review.json').read_text())['source_sha256']
    current={name:sha(SOURCE/name) for name in NAMES}
    assert [k for k in NAMES if current[k]!=original[k]]==['prepare_audit_commands.py']
    diff=''.join(difflib.unified_diff((BASE/'reviewed_sources_v1/prepare_audit_commands.py').read_text().splitlines(True),
        (SOURCE/'prepare_audit_commands.py').read_text().splitlines(True),fromfile='preserved_v1/prepare_audit_commands.py',tofile='corrected_v2/prepare_audit_commands.py'))
    with (BASE/'correction_v2.diff').open('x',encoding='utf-8') as f:f.write(diff)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Corrected))
    with (BASE/'corrected_synthetic_v2.log').open('x') as f:f.write(stream.getvalue())
    assert result.wasSuccessful() and result.testsRun==7
    assert all(sha(SOURCE/name)==digest for name,digest in current.items())
    report=dict(passed=True,source_review_pass=True,source_sha256=current,previous_findings_resolved=True,
        original_review_sha256=sha(BASE/'review.json'),diff_sha256=sha(BASE/'correction_v2.diff'),
        tests=7,failures=0,unchanged_comparison_and_collection_helpers=True,collection_full_qualification_unchanged=True,
        actual_arrays_loaded=0,actual_audits_executed=0,actual_requests_created=0,native_steps=0,model_calls=0,
        source_only=True,writer_sha256=sha(__file__),synthetic_receipt_sha256=sha(BASE/'corrected_synthetic_v2.log'))
    with (BASE/'review_v2.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(review_sha256=sha(BASE/'review_v2.json'),helper_sha256=current['prepare_audit_commands.py'],tests=7)))

if __name__=='__main__':main()
