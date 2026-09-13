"""Synthetic launch text and saved owner outcome checks; no child/native/model calls."""
import unittest
from prepare_audit_stage import render,runtime_entries,VENV
from verify_completion import outcome


def positive():
    end=dict(raw_python_exit_code=0,raw_exit_known=True,exit_code=0,error=None)
    report=dict(request_sha256='a'*64,evidence_audit_passed=True,context_condition='causal',features=1323,
        original_requested_controls=1569,model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,
        behavioral_qualification=False,all_inputs_unchanged=True)
    return end,report


class StageChecks(unittest.TestCase):
    def test_exact_namespace_and_request(self):
        text=render('a'*64)
        self.assertNotIn('direct_target_full_state_saved_semantics_review_v1',text)
        self.assertEqual(text.count('direct_target_response_saved_semantics_review_v1'),2)
        self.assertIn("$receipt.request_sha256 -ne '"+'a'*64+"'",text)

    def test_no_nonliteral_request_hash(self):
        for value in ('','a'*63,'A'*64,'a'*64+"';boom"):
            with self.assertRaises(AssertionError):render(value)

    def test_launch_requires_concrete_review_before_lock(self):
        text=render('a'*64)
        self.assertLess(text.index('Concrete review subject mismatch'),text.index("'started.lock'"))
        self.assertIn('root_selected_single_saved_audit',text)
        self.assertIn('$review.request_subject.path',text)
        self.assertIn('$review.launch_receipt_subject.path',text)

    def test_qualified_command_and_hidden_handle(self):
        text=render('a'*64)
        self.assertIn('--cd / -- bash',text)
        self.assertIn('PYTHONPATH=/mnt/e/codex-artifacts/',text)
        self.assertIn('-WindowStyle Hidden -PassThru',text)
        self.assertLess(text.index('$handle=$child.Handle'),text.index('$child.WaitForExit()'))
        self.assertIn('Unknown Python audit exit.',text)

    def test_streamed_shared_delete_hash_and_create_new(self):
        text=render('a'*64)
        self.assertIn('[IO.FileShare]::Delete',text);self.assertIn('[IO.FileMode]::CreateNew',text)
        self.assertNotIn('Get-FileHash',text);self.assertNotIn('ReadAllBytes',text)
        self.assertIn('Postrun audit pins changed.',text)
        self.assertIn('Concrete clearance/review changed.',text)

    def test_runtime_filter_excludes_models_and_native(self):
        tails=['lib/python3.11/site-packages/numpy/a.py','lib/python3.11/site-packages/scipy/a.so',
               'lib/python3.11/site-packages/numpy.libs/a.so','lib/python3.11/site-packages/scipy.libs/a.so','pyvenv.cfg',
               'lib/python3.11/site-packages/onnxruntime/a.so','lib/python3.11/site-packages/mujoco/a.so']
        files=[dict(path=VENV+tail,sha256='a'*64) for tail in tails]
        files.append(dict(path='E:/training/labels.npz',sha256='b'*64))
        self.assertEqual(runtime_entries({'files':files}),files[:5])

    def test_success_owner_outcome(self):
        end,report=positive();self.assertTrue(outcome(end,report,None,'a'*64))

    def test_wrong_condition_width_or_request_fails(self):
        for key,value in [('context_condition','blinded'),('features',1000),('request_sha256','b'*64)]:
            end,report=positive();report[key]=value
            with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)

    def test_model_native_or_behavioral_credit_fails(self):
        for key,value in [('model_calls',1),('native_steps',1),('behavioral_qualification',True)]:
            end,report=positive();report[key]=value
            with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)

    def test_preserved_failure_is_not_positive(self):
        end=dict(raw_python_exit_code=1,raw_exit_known=True,exit_code=1,error='saved check')
        fail=dict(passed=False,request_sha256='a'*64,model_calls=0,native_steps=0)
        self.assertFalse(outcome(end,None,fail,'a'*64))

    def test_unknown_exit_stays_unknown_without_pass(self):
        end=dict(raw_python_exit_code=None,raw_exit_known=False,exit_code=1,error='unknown')
        fail=dict(passed=False,request_sha256='a'*64,model_calls=0,native_steps=0)
        self.assertFalse(outcome(end,None,fail,'a'*64))
        _,report=positive()
        with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)

    def test_forged_known_flag_or_boolean_exit_fails(self):
        end,report=positive();end['raw_exit_known']=False
        with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)
        end['raw_exit_known']=True;end['raw_python_exit_code']=False
        with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)

    def test_conflicting_or_missing_terminal_evidence_fails(self):
        end,report=positive()
        with self.assertRaises(AssertionError):outcome(end,report,{'passed':False},'a'*64)
        with self.assertRaises(AssertionError):outcome(end,None,None,'a'*64)


if __name__=='__main__':unittest.main()
