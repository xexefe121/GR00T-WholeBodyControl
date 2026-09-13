"""Independent source and synthetic metadata review; no actual recovery arrays."""
import copy,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=NEW/'direct_target_width251_followup_design_v1'
NAMES=('run_saved_comparison.py','prepare_audit_commands.py','prepare_qualified_collection.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def module():
    spec=importlib.util.spec_from_file_location('reviewed_audit_commands',SOURCE/'prepare_audit_commands.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class ReviewTests(unittest.TestCase):
    def metadata(self,hold_complete):
        mod=module()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'recovery';output=Path(directory)/'synthetic_commands.json'
            owner=dict(completion_accounting_passed=True,raw_exit_known=True,processes_absent=True,
                all_postrun_pins_exact=True,accounting_uncertainty=[],requested_recovery_completed=hold_complete,
                output_sha256={str(base/part/name):'a'*64 for part in ('nominal','post_lifecycle_hold_5s') for name in ('trace.npz','report.json')})
            reports={'owner_completion.json':owner}
            def read(path):
                path=Path(path)
                if path.name=='owner_completion.json':return copy.deepcopy(owner)
                if path.parent.name=='nominal':return dict(requested_controls=1569,full_segment_completed=True)
                if path.parent.name=='post_lifecycle_hold_5s':return dict(requested_controls=250,full_segment_completed=hold_complete)
                raise AssertionError('Unexpected metadata read: '+str(path))
            with patch.object(mod,'read',side_effect=read),patch.object(mod,'sha',return_value='a'*64),patch.object(sys,'argv',['helper','--base',str(base),'--owner-sha','a'*64,'--output',str(output)]),patch('sys.stdout',new=io.StringIO()):
                mod.main()
            return json.loads(output.read_text())
    def test_complete_control_budgets_and_dependency_order(self):
        value=self.metadata(True)
        self.assertEqual([x['stage'] for x in value['commands']],['main_physics','main_intent','hold_fixture','hold_physics','hold_intent'])
        self.assertEqual([x['native_step_max'] for x in value['commands']],[15690,0,0,2500,0])
        self.assertIn('1569',value['commands'][0]['arguments']);self.assertIn('250',value['commands'][3]['arguments'])
        hold=value['commands'][-1]['arguments'];self.assertIn('--global-start',hold);self.assertIn('1569',hold);self.assertIn('--preceding-trace',hold)
    def test_partial_hold_omission_reproduced(self):
        value=self.metadata(False)
        # Reproduction: main completed and a partial hold was recorded/owned,
        # but no hold audit is emitted because combined completion is false.
        self.assertEqual([x['stage'] for x in value['commands']],['main_physics','main_intent'])
    def test_collection_exact_rows_and_plan_budget(self):
        controls=list(range(251,1269));plans=list(range(251,1269,5))
        self.assertEqual(len(controls),1018);self.assertEqual(len(plans),204)
        self.assertEqual((plans[0],plans[-1],1269-plans[-1]),(251,1266,3))
        self.assertEqual([sum(lo<=c<hi for c in controls) for lo,hi in [(251,350),(350,1169),(1169,1269)]],[99,819,100])

def main():
    original={name:sha(SOURCE/name) for name in NAMES};preserved=BASE/'reviewed_sources_v1';preserved.mkdir()
    for name in NAMES:
        with (preserved/name).open('xb') as f:f.write((SOURCE/name).read_bytes())
    sys.path.insert(0,str(SOURCE))
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests),unittest.defaultTestLoader.discover(str(SOURCE),pattern='test_first_target.py')])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_review.log').open('x') as f:f.write(stream.getvalue())
    assert result.wasSuccessful() and result.testsRun==6
    assert all(sha(SOURCE/name)==digest for name,digest in original.items())
    findings=[dict(priority=2,file='prepare_audit_commands.py',line=53,
        problem='Partial conditional hold is omitted from independent audit commands.',
        trigger='Main1569 completed; hold250 was attempted and failed partway; owner.requested_recovery_completed=false.',
        observed='Only main_physics and main_intent commands are emitted despite owned hold trace/report.',
        requested_fix='Route any saved nonempty conditional hold trace to original250 fixture/physics/intent audits; do not require full hold completion to audit its failure.')]
    evidence=[SOURCE/'compare_first_target.py',SOURCE/'test_first_target.py',
        NEW/'direct_target_width512_expert_recovery_v2/verify_recovery_completed.py',
        NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/recovery_contract.py',
        NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/run_width251_actual_oracle.py',
        NEW/'direct_target_width512_expert_recovery_v2/source_snapshot_v1/run_actual_student_oracle.py',
        NEW/'direct_target_width251_collection_v1/source_prepared_v1/qualification_gate.py']
    report=dict(passed=False,source_review_pass=False,review_complete=True,source_sha256=original,
        input_sha256={p.as_posix():sha(p) for p in evidence},findings=findings,synthetic_tests=6,synthetic_failures=0,
        no_other_blocker_found=True,v2_owner_schema_matches=True,collection_scope=dict(rows=1018,plans=204,cell_counts=[99,819,100],fresh_query_rows=1,connected_rows=1017),
        comparison_requires_exact_actual_boundary=True,comparison_does_not_qualify_labels=True,
        actual_task_arrays_loaded=0,actual_audits_executed=0,actual_requests_created=0,native_steps=0,model_calls=0,
        review_writer_sha256=sha(__file__),test_receipt_sha256=sha(BASE/'synthetic_review.log'))
    write(BASE/'review.json',report);print(json.dumps(dict(review_sha256=sha(BASE/'review.json'),findings=len(findings),tests=result.testsRun)))

if __name__=='__main__':main()
