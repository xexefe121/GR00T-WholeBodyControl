"""Pure preparation tests; generated launchers are parsed, never executed."""
import ast,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_snapshot_v1';sys.path.insert(0,str(SOURCE))
from evaluation_gate import require_ready,sha
from head_activation_witness import require_witness_ready
def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
freeze=load('freeze_final_package');launch=load('prepare_bound_launcher')
class BindingTests(unittest.TestCase):
    def config(self,selected=False):
        return dict(root_selected=selected,ordinary_final_step=75000,stage='witness',requested_main_controls=0,conditional_hold_controls=0,separate_head_calls=1,hardware_authorized=False,selected_subject_sha256={})
    def test_unbound_source_rejects_before_any_model(self):
        with self.assertRaises(ValueError):require_ready(BASE)
        with self.assertRaises(ValueError):require_witness_ready(BASE)
    def test_missing_root_selection_stops_before_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);config=p/'reviews.json';config.write_text(json.dumps(self.config(False)))
            with patch.object(freeze,'BASE',p),patch.object(sys,'argv',['freeze','--mode','witness','--reviews',str(config)]):
                with self.assertRaises(AssertionError):freeze.main()
            self.assertFalse((p/'witness_binding.json').exists())
    def test_selected_without_actual_final_artifacts_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);config=p/'reviews.json';config.write_text(json.dumps(self.config(True)))
            with patch.object(freeze,'BASE',p),patch.object(freeze,'FIT',p/'missing_actual_fit'),patch.object(sys,'argv',['freeze','--mode','witness','--reviews',str(config)]):
                with self.assertRaises(AssertionError):freeze.main()
            self.assertFalse((p/'witness_binding.json').exists());self.assertFalse((p/'head_witness').exists())
    def test_wrong_stage_and_step_do_not_bind(self):
        for key,value in [('stage','evaluation'),('ordinary_final_step',70000),('separate_head_calls',2),('hardware_authorized',True)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                p=Path(d);config=p/'reviews.json';c=self.config(True);c[key]=value;config.write_text(json.dumps(c))
                with patch.object(freeze,'BASE',p),patch.object(sys,'argv',['freeze','--mode','witness','--reviews',str(config)]):
                    with self.assertRaises(AssertionError):freeze.main()
                self.assertFalse((p/'witness_binding.json').exists())
    def test_actual_all_dataset_subjects_required(self):
        source=(BASE/'freeze_final_package.py').read_text()
        self.assertIn("for subject in ('centers','physical_manifest','physical_report','physical_collection_request')",source)
        request=freeze.read(BASE.parent/'one_step_physical_student_v1/training_request.json')
        review=freeze.read(BASE.parent/'one_step_branch_combined_data_review_v1/review.json')
        for path in request['physical_paths'].values():self.assertTrue(freeze.has_hash(review,sha(path)))
    def test_generate_guarded_templates_only(self):
        output=BASE/'launcher_template_checks';output.mkdir(exist_ok=False)
        for mode in ('witness','evaluation'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)
                def read_stub(path):
                    if Path(path).name=='exit_capture_test.json':return dict(actual_exit=7,expected_exit=7,handle_acquired_before_wait=True)
                    return dict(input_files=[])
                # Pure renderer stubs only; no fake head/checkpoint artifact is created.
                with patch.object(launch,'BASE',p),patch.object(launch,'read',read_stub),patch.object(launch,'sha',lambda p:'a'*64),patch.object(launch,'require_ready',lambda p:None),patch.object(launch,'require_witness_ready',lambda p:None),patch.object(sys,'argv',['prepare','--mode',mode]):launch.main()
                for name in ('run.ps1','run_durable.ps1'):
                    text=(p/(mode+'_process')/name).read_text();(output/(mode+'_'+name)).write_text(text)
                    if name=='run.ps1':
                        self.assertIn('Existing attempt output must be preserved',text);self.assertIn('execution.lock',text)
                        self.assertIn('CreateNew',text);self.assertIn('if ($null -eq $taskExit)',text)
                        self.assertIn('head_activation_witness.py' if mode=='witness' else 'evaluate_physical_response_student.py',text)
                        if mode=='evaluation':self.assertIn('completed_controls -ne 1569',text);self.assertIn('completed_controls -ne 250',text)
                    else:
                        self.assertIn('-WindowStyle Hidden',text);self.assertIn('started.lock',text)
                        self.assertLess(text.index('$nativeHandle = $taskChild.Handle'),text.index('$taskChild.WaitForExit()'))
                        self.assertIn('if ($null -eq $exitCode)',text);self.assertIn('postrun_hashes.json',text)
    def test_current_runtime_source_hashes_remain_reviewed(self):
        receipt=freeze.read(BASE/'source_freeze.json')
        for path,digest in receipt['source_sha256'].items():self.assertEqual(sha(SOURCE/path),digest)
        self.assertEqual(len(receipt['source_sha256']),25)
        self.assertEqual((SOURCE/'head_activation_witness.py').read_text().count('head.run('),1)
        for p in SOURCE.rglob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':
    with (BASE/'binding_workflow_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingTests))
    assert result.wasSuccessful()
    receipt=dict(passed=True,tests=result.testsRun,real_model_calls=0,native_steps=0,launchers_executed=0,
        source_sha256={p.name:sha(p) for p in [BASE/'freeze_final_package.py',BASE/'prepare_bound_launcher.py',Path(__file__)]},log_sha256=sha(BASE/'binding_workflow_tests.log'),
        template_check_directory='launcher_template_checks',templates_not_bound_to_actual_artifacts=True)
    (BASE/'binding_workflow_tests.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}))
