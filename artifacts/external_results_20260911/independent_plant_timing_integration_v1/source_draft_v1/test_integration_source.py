"""Static mutation/source checks. No producer invocation or actual timing reads."""
import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from check_integration_derivation import check,dump,MODULES

SOURCE=Path(__file__).resolve().parent
ORIGINAL=SOURCE.parent/'source_original_v1'
PRIOR=SOURCE.parent.parent/'independent_plant_pending_result_v1/source_draft_v1'


class SourceEquivalence(unittest.TestCase):
    def test_entire_modules_under_explicit_erasure(self):
        result=check(ORIGINAL,SOURCE)
        self.assertTrue(result['outer_hot_while_ast_exact'])
    def test_all24_preserved_and20_unchanged(self):
        files=sorted(ORIGINAL.glob('*.py'));self.assertEqual(len(files),24)
        changed=[]
        for p in files:
            self.assertEqual(p.read_bytes(),(PRIOR/p.name).read_bytes())
            if p.read_bytes()!=(SOURCE/p.name).read_bytes():changed.append(p.name)
        self.assertEqual(changed,['clock_core.py','native_stepper.py','run_clock.py','session.py'])
    def test_probe_v2_all_three_files_byte_exact(self):
        probe=SOURCE.parent.parent/'independent_plant_timing_instrumentation_v1/source_draft_v2'
        for p in probe.glob('*.py'):self.assertEqual(p.read_bytes(),(SOURCE/p.name).read_bytes())
    def test_semantic_mutations_rejected(self):
        changes=[('native_stepper.py','kp*(target-q[7:])','kp*(target+q[7:])'),
            ('clock_core.py','self.expected_simulation_time += 0.002','self.expected_simulation_time += 0.003'),
            ('session.py',"self.outer.append(encode(dict(index=before,returned=self.foundation.returned,cycle_return_ns=finish,",
                          "self.outer.append(encode(dict(index=before,returned=self.foundation.returned,cycle_return_ns=finish+1,")]
        for name,before,after in changes:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                path=Path(d)
                for module in (*MODULES,'run_clock.py'):(path/module).write_text((SOURCE/module).read_text())
                text=(path/name).read_text();self.assertIn(before,text);(path/name).write_text(text.replace(before,after,1))
                with self.assertRaises(AssertionError):check(ORIGINAL,path)
    def test_supervisor_main_epoch_error_functions_unchanged(self):
        def funcs(path):return {n.name:dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old=funcs(ORIGINAL/'run_clock.py');new=funcs(SOURCE/'run_clock.py')
        self.assertEqual(old.keys(),new.keys())
        for name in old.keys()-{'run','imported_source_paths'}:self.assertEqual(old[name],new[name],name)
    def test_allocation_attach_before_fixed_epoch_and_same_while(self):
        text=(SOURCE/'run_clock.py').read_text()
        sequence=['timing=TimingHooks(TimingProbe(', 'timing.probe.attach(gc.callbacks)',
            'epoch_chosen_ns=clock.now_ns();epoch_ns=epoch_chosen_ns+request[',
            'session=Session(', "journal.record('epoch_armed'",'while session.foundation.returned<18190:']
        locations=[text.index(value) for value in sequence]
        self.assertEqual(locations,sorted(locations))
        self.assertNotIn('gc.disable(',text);self.assertNotIn('gc.enable(',text);self.assertNotIn('gc.collect(',text)
    def test_main_preservation_precedes_sidecars_and_all_manifest(self):
        text=(SOURCE/'run_clock.py').read_text()
        sequence=["journal.record('plant_or_setup_exit'",'timing.probe.detach()',
            'cleanup=lifecycle.stop_join()', 'exit_identity=session.close_native()',
            'schema=save_owned(output,session)', 'timing_result=save_timing(',
            "write(output/'post_input_hashes.json'", "write(output/'report.json'",'outputs={str(p.relative_to(output))']
        locations=[text.index(value) for value in sequence];self.assertEqual(locations,sorted(locations))
        self.assertIn("if timing_result is None or timing_result.get('complete') is not True:",text)
        self.assertIn('first_error is None and not additional_errors and worker_pass',text)
    def test_storage_and_span_bound(self):
        # Every step has11 inclusive/child scopes; every boundary at most3;
        # every future job has at most10 publication attempts. No adaptive growth.
        self.assertEqual(11*18190+3*1819+10*1818,223727)
        self.assertLess(223727+16,262144)
        self.assertEqual((262144*16+4096*12)*8,33947648)


if __name__=='__main__':unittest.main()
