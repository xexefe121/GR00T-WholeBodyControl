"""Synthetic metadata and text only: no requests, workers, clocks or models."""
import copy,unittest
from pathlib import Path
from prepare_concrete_packet import (RETRY_CONTRACT,RETRY_DETAILS,RESULT_CONTRACT,RESULT_DETAILS,
    DEADLINE_CONTRACT,validate_retry_contract,check_review,check_subject)
from prepare_clock_stage import arguments,texts,BASE

def request():
    return dict(job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS),
        worker_result_retry_contract=RESULT_CONTRACT,worker_result_retry_details=dict(RESULT_DETAILS),
        job_deadline_contract=DEADLINE_CONTRACT)

class HelperTests(unittest.TestCase):
    def test_exact_retry_contract(self):validate_retry_contract(request())
    def test_every_contract_literal(self):
        for key in ('job_publication_retry_contract','worker_result_retry_contract','job_deadline_contract'):
            for value in (None,'changed',True):
                q=request();q[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_retry_contract(q)
    def test_every_detail_literal(self):
        for role in ('job_publication_retry_details','worker_result_retry_details'):
            for key,value in request()[role].items():
                q=request();q[role][key]=not value if type(value) is bool else (value+1 if type(value) in (int,float) else 'PUBLISHED')
                with self.subTest(role=role,key=key),self.assertRaises(ValueError):validate_retry_contract(q)
    def test_boolean_integer_alias_rejected(self):
        for role,key in [('job_publication_retry_details','max_attempts_per_physics_tick'),('worker_result_retry_details','max_attempts_per_worker_iteration')]:
            q=request();q[role][key]=True
            with self.assertRaises(ValueError):validate_retry_contract(q)
    def test_extra_missing_and_nondict_details_rejected(self):
        for role in ('job_publication_retry_details','worker_result_retry_details'):
            for value in (None,{},dict(request()[role],relax_deadline=True)):
                q=request();q[role]=value
                with self.assertRaises(ValueError):validate_retry_contract(q)
    def test_expiry_and_ambiguity_remain_fail_closed(self):
        q=request()
        self.assertFalse(q['worker_result_retry_details']['ambiguous_publication_retry'])
        self.assertFalse(q['worker_result_retry_details']['recompute_reply'])
        self.assertFalse(q['worker_result_retry_details']['deadline_rebase'])
        self.assertTrue(q['worker_result_retry_details']['late_publication_does_not_override_admission'])
        self.assertEqual(q['worker_result_retry_details']['max_attempts_per_result'],20)
        self.assertEqual(q['job_publication_retry_details']['max_attempts_per_job'],10)
    def test_review_requires_exact_named_subject(self):
        prep={'source_sha256':{'a.py':'a'*64}};path=BASE/'synthetic_preparation.json'
        for field in ('source_preparation_subject','preparation_subject'):
            good=dict(passed=True,source_review_pass=True,source_sha256=prep['source_sha256'])
            good[field]={'path':path.as_posix(),'sha256':'b'*64}
            check_review(prep,good,path,'b'*64,1,field)
            for key,value in [('passed',False),('source_review_pass',False),('source_sha256',{})]:
                bad=copy.deepcopy(good);bad[key]=value
                with self.assertRaises(ValueError):check_review(prep,bad,path,'b'*64,1,field)
            with self.assertRaises(ValueError):check_review(prep,good,path,'b'*64,2,field)
            with self.assertRaises(ValueError):check_review(prep,good,path,'b'*64,1,'unrelated')
    def test_wrong_subject_path_hash_or_recursive_only_rejected(self):
        path=BASE/'subject.json'
        for subject in (None,{'elsewhere':{'path':str(path),'sha256':'b'*64}},
            {'path':str(path),'sha256':'c'*64},{'path':str(path)+'x','sha256':'b'*64},
            {'path':str(path),'sha256':'b'*64,'extra':True}):
            with self.assertRaises(ValueError):check_subject(subject,path,'b'*64)
    def test_explicit_windows_path_equivalence(self):
        path=BASE/'subject.json';check_subject({'path':str(path),'sha256':'b'*64},path.as_posix(),'b'*64)
    def test_absolute_bootstrap_and_fixed_budget(self):
        args=arguments('clock')
        self.assertEqual(args[:6],['-d','Ubuntu-22.04','--cd','/','--','bash'])
        self.assertEqual(args[8:12],['--signal=TERM','--kill-after=5s','555s','env'])
        self.assertIn('PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_pending_result_v1/source_draft_v1',args)
        self.assertEqual(args[-4],'--request');self.assertEqual(args[-2],'--clearance')
        with self.assertRaises(ValueError):arguments('witness')
    def test_hidden_handle_lock_known_exit(self):
        _,durable=texts()
        self.assertIn('-WindowStyle Hidden -PassThru',durable)
        self.assertLess(durable.index('$nativeHandle = $taskChild.Handle'),durable.index('$taskChild.WaitForExit()'))
        self.assertIn('[System.IO.FileMode]::CreateNew',durable)
        self.assertIn('Child exit status unavailable; no automatic retry.',durable)
        self.assertIn('raw_python_exit_code=$childExit',durable)
    def test_review_before_lock_and_exact_path_normalization(self):
        _,text=texts()
        self.assertLess(text.index('Final review does not name this exact request and launcher.'),text.index("'started.lock'"))
        self.assertIn("$receiptPath.Replace('\\','/')",text)
        self.assertNotIn("$receiptPath.Replace('\\\\','/')",text)
    def test_streamed_shared_delete_hash(self):
        for text in texts():
            self.assertIn('[System.IO.FileShare]::Delete',text)
            self.assertIn('$algorithm.ComputeHash($stream)',text)
            self.assertNotIn('ReadAllBytes',text)
    def test_saved_verdict_and_no_implicit_selection(self):
        _,text=texts();self.assertNotIn(" --stage 'clock'",text)
        for literal in ('stage_verdict.py','automatic_retry=$false','root_selected_single_run'):
            self.assertIn(literal,text)
    def test_repeatable_render(self):self.assertEqual(texts(),texts())

if __name__=='__main__':unittest.main()
