"""Metadata and generated-text tests only; no native, model or worker calls."""
import copy,unittest
from prepare_concrete_packet import RETRY_CONTRACT,RETRY_DETAILS,validate_retry_contract,check_review
from prepare_clock_stage import arguments,texts,BASE

class HelperTests(unittest.TestCase):
    def test_exact_retry_contract(self):
        validate_retry_contract(dict(job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS)))
    def test_retry_contract_missing_or_changed(self):
        for value in (None,'one_publication','pending_BUSY_same_job_max11_before_original_activation'):
            with self.assertRaises(ValueError):validate_retry_contract(dict(job_publication_retry_contract=value,job_publication_retry_details=dict(RETRY_DETAILS)))
    def test_every_retry_detail_is_literal(self):
        for key,value in RETRY_DETAILS.items():
            q=dict(job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS))
            q['job_publication_retry_details'][key]=not value if type(value) is bool else (value+1 if type(value) is int else 'PUBLISHED')
            with self.assertRaises(ValueError):validate_retry_contract(q)
    def test_boolean_integer_alias_is_rejected(self):
        q=dict(job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS))
        q['job_publication_retry_details']['max_attempts_per_physics_tick']=True
        with self.assertRaises(ValueError):validate_retry_contract(q)
    def test_extra_retry_detail_rejected(self):
        q=dict(job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS,relax_deadline=True))
        with self.assertRaises(ValueError):validate_retry_contract(q)
    def test_review_requires_full_map_and_subject(self):
        prep={'source_sha256':{'a.py':'a'*64}}
        good=dict(passed=True,source_review_pass=True,source_preparation_sha256='b'*64,source_sha256=prep['source_sha256'])
        check_review(prep,good,'b'*64,1)
        for key,value in [('passed',False),('source_review_pass',False),('source_preparation_sha256','c'*64),('source_sha256',{})]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(ValueError):check_review(prep,bad,'b'*64,1)
        with self.assertRaises(ValueError):check_review(prep,good,'b'*64,2)
    def test_absolute_bootstrap_then_same_budget(self):
        args=arguments('clock')
        self.assertEqual(args[:6],['-d','Ubuntu-22.04','--cd','/','--','bash'])
        self.assertEqual(args[8:12],['--signal=TERM','--kill-after=5s','555s','env'])
        self.assertIn('PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_pending_publication_v1/source_draft_v1',args)
        self.assertEqual(args[-4],'--request');self.assertEqual(args[-2],'--clearance')
        with self.assertRaises(ValueError):arguments('witness')
    def test_hidden_handle_lock_and_known_exit(self):
        run,durable=texts()
        self.assertIn('-WindowStyle Hidden -PassThru',durable)
        self.assertLess(durable.index('$nativeHandle = $taskChild.Handle'),durable.index('$taskChild.WaitForExit()'))
        self.assertIn('[System.IO.FileMode]::CreateNew',durable)
        self.assertIn('Child exit status unavailable; no automatic retry.',durable)
        self.assertIn('raw_python_exit_code=$childExit',durable)
    def test_review_precedes_lock(self):
        _,text=texts()
        self.assertLess(text.index('Final review does not name this exact request and launcher.'),text.index("'started.lock'"))
        self.assertIn("$receiptPath.Replace('\\','/')",text)
    def test_streamed_shared_delete_hashing(self):
        run,durable=texts()
        for text in (run,durable):
            self.assertIn('[System.IO.FileShare]::Delete',text)
            self.assertIn('$algorithm.ComputeHash($stream)',text)
            self.assertNotIn('ReadAllBytes',text)
    def test_saved_verdict_and_no_implicit_selection(self):
        _,text=texts()
        self.assertNotIn(" --stage 'clock'",text)
        self.assertIn('stage_verdict.py',text)
        self.assertIn('automatic_retry=$false',text)
        self.assertIn('root_selected_single_run',text)
    def test_render_is_repeatable(self):
        self.assertEqual(texts(),texts())

if __name__=='__main__':unittest.main()
