"""Synthetic wrapper/owner regressions; no source-audit math or task arrays."""
import unittest
from prepare_launch import render,validate_source_review,SOURCE_PREPARATION_SHA,BASE
import copy
from verify_completion import outcome,normalized_pins


def good():
    end=dict(raw_exit_known=True,raw_python_exit_code=0,exit_code=0,error=None)
    report=dict(request_sha256='a'*64,evidence_integrity_passed=True,model_calls=0,native_steps_executed=0,
        optimizer_updates=0,component_qualified=False,online_policy_qualified=False,realtime_teleoperation_qualified=False)
    return end,report


class LaunchTests(unittest.TestCase):
    def test_actual_source_review_and_full_map_required(self):
        path=BASE/'synthetic_review.json';source_map={str(i)+'.py':'a'*64 for i in range(25)}
        review=dict(passed=True,source_review_pass=True,source_preparation_sha256=SOURCE_PREPARATION_SHA,source_sha256=source_map)
        q=dict(source_review={'path':str(path),'sha256':'b'*64},source_sha256=source_map,input_sha256={str(path):'b'*64})
        validate_source_review(q,path,'b'*64,review,source_map)
        for key,value in [('passed',False),('source_review_pass',False),('source_preparation_sha256','c'*64),('source_sha256',{})]:
            bad=copy.deepcopy(review);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_source_review(q,path,'b'*64,bad,source_map)
        for key,value in [('input_sha256',{}),('source_review',{'path':str(path)+'wrong','sha256':'b'*64})]:
            bad=copy.deepcopy(q);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_source_review(bad,path,'b'*64,review,source_map)

    def test_fixed_retry_v2_source_command(self):
        text=render('a'*64)
        self.assertIn('source_draft_v2/audit_clock.py --request ',text)
        self.assertIn('/independent_plant_timing_saved_actual_v1/request.json --output ',text)
        self.assertIn('PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_timing_saved_audit_v1/source_draft_v2',text)
        self.assertNotIn('direct_target_context_saved_semantics_review_v1',text)

    def test_mandatory_dispatch_argument_preserved(self):
        text=render('a'*64)
        self.assertIn('param([Parameter(Mandatory=$true)][string]$LaunchReceiptSha256)',text)

    def test_literal_hash_required(self):
        for value in ('z'*64,'a'*63,'a'*64+"'"):
            with self.assertRaises(AssertionError):render(value)

    def test_hidden_handle_and_no_retry(self):
        text=render('a'*64)
        self.assertIn('-WindowStyle Hidden -PassThru',text)
        self.assertLess(text.index('$handle=$child.Handle'),text.index('$child.WaitForExit()'))
        self.assertIn('Unknown Python audit exit.',text);self.assertIn('automatic_retry=$false',text)

    def test_review_before_lock_and_integrity_verdict(self):
        text=render('a'*64)
        self.assertLess(text.index('Concrete review subject mismatch'),text.index("'started.lock'"))
        self.assertIn('$result.evidence_integrity_passed',text)
        self.assertNotIn('$result.component_qualified',text)
        self.assertIn('$result.native_steps_executed -ne 0',text)

    def test_shared_delete_stream_hashes(self):
        text=render('a'*64)
        self.assertIn('[IO.FileShare]::Delete',text);self.assertIn('$h.ComputeHash($s)',text)
        self.assertNotIn('Get-FileHash',text);self.assertNotIn('ReadAllBytes',text)

    def test_evidence_pass_can_preserve_component_failure(self):
        end,report=good();self.assertTrue(outcome(end,report,None,'a'*64));self.assertFalse(report['component_qualified'])

    def test_audit_failure_is_false_and_preserved(self):
        end,report=good();end.update(raw_python_exit_code=1,exit_code=1,error='audit failed')
        report.update(evidence_integrity_passed=False,error='sample mismatch')
        self.assertFalse(outcome(end,report,None,'a'*64))

    def test_unknown_not_positive(self):
        end,report=good();end.update(raw_python_exit_code=None,raw_exit_known=False,exit_code=1,error='unknown')
        with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)
        report.update(evidence_integrity_passed=False,error='incomplete')
        self.assertFalse(outcome(end,report,None,'a'*64))

    def test_wrong_request_or_extra_calls_rejected(self):
        for key,value in [('request_sha256','b'*64),('native_steps_executed',1),('model_calls',1)]:
            end,report=good();report[key]=value
            with self.assertRaises(AssertionError):outcome(end,report,None,'a'*64)

    def test_normalized_pin_membership_and_conflicts(self):
        self.assertEqual(normalized_pins({'E:/a/b':'x'}),normalized_pins({'/mnt/e/a/b':'x'}))
        with self.assertRaises(AssertionError):normalized_pins({'E:/a/b':'x','/mnt/e/a/b':'y'})


if __name__=='__main__':unittest.main()
