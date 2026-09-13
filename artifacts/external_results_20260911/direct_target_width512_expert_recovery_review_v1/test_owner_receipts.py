"""Synthetic complete-process receipts and corruption checks; no processes run."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/verify_recovery_completed.py')
spec=importlib.util.spec_from_file_location('recovery_owner_under_review',SOURCE)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


class OwnerReceipts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.process=self.base/'recovery_process_v1';self.process.mkdir()
        self.write('run_recovery_durable.ps1','synthetic launcher only',raw=True)
        names=['native_live_step','native_private_step','native_BFM_step','native_initial_certificate_step',
               'native_restoration_certificate_step','native_preview_step','ordinary_ilqr','BFM_propose','batch_fd_step','batch_line_step']
        budgets={name:dict(attempted_calls_max=20000,attempted_units_max=20000) for name in names}
        counts={name:dict(attempted_calls=0,returned_calls=0,attempted_units=0,returned_units=0) for name in names}
        counts['native_live_step']={name:15680 for name in counts['native_live_step']}
        counts['ordinary_ilqr']={name:204 for name in counts['ordinary_ilqr']}
        self.write('execution_request.json',dict(protocol={'budgets':budgets}))
        self.write('frozen_inputs.json',{'synthetic':True})
        args=['-d','synthetic','--','bash','/synthetic.sh']
        receipt=dict(request_sha256=self.sha('execution_request.json'),frozen_receipt_sha256=self.sha('frozen_inputs.json'),
                     launcher_sha256=self.sha('run_recovery_durable.ps1'),wsl_arguments=args,input_sha256={})
        self.write('launch_receipt.json',receipt)
        review=dict(passed=True,request_sha256=receipt['request_sha256'],frozen_receipt_sha256=receipt['frozen_receipt_sha256'],
                    launch_receipt_sha256=self.sha('launch_receipt.json'))
        self.write('root_review.json',review)
        clearance=dict(review,approved=True,review=dict(path=(self.base/'root_review.json').as_posix(),sha256=self.sha('root_review.json'),pass_field='passed'))
        self.write('execution_clearance.json',clearance)
        start=dict(wrapper_pid=111,request_sha256=receipt['request_sha256'],frozen_receipt_sha256=receipt['frozen_receipt_sha256'],
                   launch_receipt_sha256=self.sha('launch_receipt.json'),clearance_sha256=self.sha('execution_clearance.json'),arguments=args)
        self.write('recovery_process_v1/start.json',start)
        self.write('recovery_process_v1/child.json',dict(wrapper_pid=111,child_pid=222,handle_acquired=True,
            command_arguments=' '.join('"'+x+'"' for x in args)))
        self.write('recovery_process_v1/linux_process.json',dict(linux_pid=333,process_identity='bash exec replaced by selected Python'))
        self.write('recovery_process_v1/dispatch.json',dict(wrapper_pid=111,clearance_sha256=self.sha('execution_clearance.json'),
            launcher_sha256=receipt['launcher_sha256'],arguments=['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
                (self.base/'run_recovery_durable.ps1').as_posix(),'-ClearanceSha256',self.sha('execution_clearance.json')]))
        self.write('recovery_process_v1/process_absence.json',dict(windows_expected_pids=[111,222],linux_expected_pids=[333],
                                                                 windows_all_absent=True,linux_all_absent=True))
        self.write('recovery_process_v1/exit.json',dict(wrapper_pid=111,child_pid=222,raw_exit_known=True,raw_python_exit_code=0,
            exit_code=0,error=None,all_postrun_pins_exact=True,final_work_counters_present=True,requested_branch_completed=True))
        pins={str(self.base/name):self.sha(name) for name in ('launch_receipt.json','execution_clearance.json','root_review.json')}
        pin_report=dict(all_exact=True,files={p:dict(expected=h,actual=h,matched=True) for p,h in pins.items()})
        self.write('recovery_process_v1/prerun_pins.json',pin_report);self.write('recovery_process_v1/postrun_pins.json',pin_report)
        self.write('work_counters.json',dict(phase='task_ended',active=[],limits=budgets,counts=counts,first_failure=None))
        self.write('outcome.json',dict(nominal={'full_segment_completed':True},extension={'full_segment_completed':True}))
        self.write('driver_completion.json',dict(driver_returned=True,requested_branch_completed=True))

    def write(self,name,value,raw=False):
        (self.base/name).write_text(value if raw else json.dumps(value))
    def sha(self,name):return owner.sha(self.base/name)
    def mutate(self,name,key,value):
        data=owner.read(self.base/name);data[key]=value;self.write(name,data)

    def test_complete_fixture_and_absence_output_subject(self):
        result=owner.verify(self.base)
        self.assertTrue(result['completion_accounting_passed'])
        self.assertTrue(result['requested_recovery_completed'])
        for name in ('dispatch.json','start.json','child.json','linux_process.json','process_absence.json'):
            path=self.process/name
            self.assertEqual(result['output_sha256'][str(path)],owner.sha(path))

    def test_wrong_child_parent_rejected(self):
        self.mutate('recovery_process_v1/child.json','wrapper_pid',444)
        with self.assertRaises(AssertionError):owner.verify(self.base)

    def test_missing_linux_identity_rejected_for_completed_child(self):
        (self.process/'linux_process.json').unlink()
        self.mutate('recovery_process_v1/process_absence.json','linux_expected_pids',[])
        with self.assertRaises(AssertionError):owner.verify(self.base)

    def test_changed_actual_start_arguments_rejected(self):
        self.mutate('recovery_process_v1/start.json','arguments',['unselected'])
        with self.assertRaises(AssertionError):owner.verify(self.base)

    def test_raw_zero_wrapper_failure_not_complete_accounting(self):
        self.mutate('recovery_process_v1/exit.json','exit_code',1)
        try:result=owner.verify(self.base)
        except AssertionError:return
        self.assertFalse(result['completion_accounting_passed'])

    def test_self_declared_counter_budget_rejected(self):
        data=owner.read(self.base/'work_counters.json');data['limits']['ordinary_ilqr']['attempted_calls_max']+=1
        self.write('work_counters.json',data)
        with self.assertRaises(AssertionError):owner.verify(self.base)

    def test_boolean_counter_rejected(self):
        data=owner.read(self.base/'work_counters.json');data['counts']['BFM_propose']['returned_calls']=False
        self.write('work_counters.json',data)
        with self.assertRaises(AssertionError):owner.verify(self.base)

    def test_expected_seed_rejection_keeps_complete_accounting(self):
        data=owner.read(self.base/'work_counters.json')
        data['counts']['BFM_propose'].update(attempted_calls=1,attempted_units=1)
        self.write('work_counters.json',data)
        self.assertTrue(owner.verify(self.base)['completion_accounting_passed'])


if __name__=='__main__':unittest.main()
