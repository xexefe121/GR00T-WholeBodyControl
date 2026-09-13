"""Fake-Popen/source tests only. Never starts WSL, native or audit processes."""
import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
CURRENT=NEW/'direct_target_width251_followup_design_v1/run_prepared_audit.py'
BOOT_SHA='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'

def module(path):
    spec=importlib.util.spec_from_file_location('review_fake_launcher',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def flow(path=CURRENT,changed=False,post_error=False,report_error=False,interrupted=False,boot_bad=False):
    m=module(path);calls=[];returned=[False]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);target=root/'synthetic_output';command_file=root/'synthetic_commands.json'
        fixture='E:/synthetic_only_fixture.npz';dependency='E:/synthetic_only_physics.json'
        command=['/synthetic/audit.py','--fixture',fixture,'--physical-audit',dependency,'--output',str(target)]
        q=dict(preparation_pass=True,input_sha256={fixture:'a'*64},python='/synthetic/python',environment={'OMP_NUM_THREADS':'1'},
            commands=[dict(stage='main_physics',arguments=command,native_step_max=15690)])
        def fake_sha(p):
            text=str(p).replace('\\','/')
            if text.endswith('synthetic_only_fixture.npz'):
                if post_error and returned[0]:raise OSError('synthetic postread unavailable')
                return ('b' if changed else 'a')*64
            if text.endswith('RUN_PASSING_WALK_WSL.sh'):return 'c'*64 if boot_bad else BOOT_SHA
            if text==str(target/'report.json').replace('\\','/') and report_error:raise OSError('synthetic report unreadable')
            return 'a'*64
        class FakePopen:
            def __init__(self,argv,**kwargs):calls.append(argv);self.pid=999999
            def wait(self):
                if interrupted:raise InterruptedError('synthetic interrupted wait')
                returned[0]=True
                if report_error:target.mkdir();(target/'report.json').write_text('{}')
                return 0
        exception=None
        with (patch.object(m,'HERE',root/'harness'),patch.object(m,'sha',side_effect=fake_sha),patch.object(m,'read',return_value=q),
             patch.object(m.subprocess,'Popen',FakePopen),patch.object(sys,'argv',['fake','--commands',str(command_file),'--commands-sha256','a'*64,'--stage','main_physics']),patch('builtins.print')):
            try:m.main()
            except BaseException as exc:exception=exc
        folder=root/'harness/actual_audit_processes/main_physics'
        exit_record=json.loads((folder/'exit.json').read_text()) if (folder/'exit.json').exists() else None
        return calls,exception,exit_record

class LauncherReview(unittest.TestCase):
    def test_preserved_v1_fixture_repin_bug_reproduced(self):
        calls,error,end=flow(BASE/'run_prepared_audit_preserved_v1.py',changed=True)
        self.assertEqual(len(calls),1);self.assertIsNone(error);self.assertTrue(end['all_postrun_pins_exact'])
    def test_corrected_changed_fixture_rejects_before_popen(self):
        calls,error,end=flow(changed=True)
        self.assertEqual(calls,[]);self.assertIsInstance(error,ValueError);self.assertIsNone(end)
    def test_new_dependency_and_exact_boot_permit_one_fake_call(self):
        calls,error,end=flow();self.assertEqual(len(calls),1);self.assertIsNone(error)
        self.assertTrue(end['accounting_passed']);self.assertTrue(end['raw_exit_known']);self.assertEqual(end['raw_child_exit_code'],0)
        self.assertNotIn('raw_python_exit_code',end)
        self.assertEqual(calls[0][:7],['C:/Windows/System32/wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash'])
    def test_wrong_boot_rejects_before_popen(self):
        calls,error,end=flow(boot_bad=True);self.assertFalse(calls);self.assertIsInstance(error,ValueError);self.assertIsNone(end)
    def test_normalized_alias_deduplicates_equal_expected_pin(self):
        m=module(CURRENT);pins={}
        with patch.object(m,'sha',return_value='a'*64):
            m.add_pin(pins,'E:/synthetic/file','a'*64);m.add_pin(pins,'/mnt/e/synthetic/file','a'*64)
        self.assertEqual(len(pins),1)
    def test_conflicting_alias_rejects(self):
        m=module(CURRENT);pins={'/mnt/e/synthetic/file':'b'*64}
        with patch.object(m,'sha',return_value='a'*64),self.assertRaises(ValueError):m.add_pin(pins,'E:/synthetic/file')
    def test_preserved_pin_fix_v2_postread_loses_exit_reproduced(self):
        calls,error,end=flow(BASE/'run_prepared_audit_pins_fixed_v2.py',post_error=True)
        self.assertEqual(len(calls),1);self.assertIsInstance(error,OSError);self.assertIsNone(end)
    def test_corrected_postread_error_preserves_known_exit(self):
        calls,error,end=flow(post_error=True)
        self.assertEqual(len(calls),1);self.assertIsInstance(error,RuntimeError)
        self.assertTrue(end['raw_exit_known']);self.assertTrue(end['child_reaped']);self.assertEqual(end['raw_child_exit_code'],0)
        self.assertFalse(end['all_postrun_pins_exact']);self.assertFalse(end['accounting_passed'])
        self.assertTrue(any(v['error'] for v in end['input_checks'].values()))
    def test_report_hash_error_preserves_known_exit(self):
        calls,error,end=flow(report_error=True);self.assertEqual(len(calls),1);self.assertIsInstance(error,RuntimeError)
        self.assertEqual(end['raw_child_exit_code'],0);self.assertTrue(end['raw_exit_known']);self.assertFalse(end['accounting_passed'])
        self.assertEqual(end['output_report_hash_error']['type'],'OSError')
    def test_interrupted_wait_never_claims_return(self):
        calls,error,end=flow(interrupted=True);self.assertEqual(len(calls),1);self.assertIsInstance(error,InterruptedError)
        self.assertFalse(end['raw_exit_known']);self.assertFalse(end['child_reaped']);self.assertIsNone(end['raw_child_exit_code'])
        self.assertEqual(end['process_error']['type'],'InterruptedError');self.assertFalse(end['accounting_passed'])

if __name__=='__main__':unittest.main()
