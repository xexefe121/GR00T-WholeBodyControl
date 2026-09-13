"""Synthetic metadata and source checks only; no task array or native imports."""
import ast
import copy
from pathlib import Path
import unittest

import qualification as q


def fixture():
    roles = ('owner', 'recovery_request', 'main_trace', 'main_report', 'hold_trace', 'hold_report',
             'main_physics', 'main_intent', 'hold_physics', 'hold_intent', 'reference',
             'reference_receipt', 'bundle_manifest', 'contract', 'native_xml', 'native_arrays')
    subjects = {r: {'path': 'E:/fixture/' + r, 'sha256': f'{i:064x}'} for i, r in enumerate(roles)}
    pins = {s['path']: s['sha256'] for s in subjects.values()}
    reports = {'owner': dict(completion_accounting_passed=True, requested_recovery_completed=True,
        raw_exit_known=True, all_postrun_pins_exact=True, processes_absent=True,
        raw_python_exit_code=0, exit_code=0, accounting_uncertainty=[],
        input_sha256=copy.deepcopy(pins), output_sha256=copy.deepcopy(pins))}
    for segment, count, start in (('main', 1569, 0), ('hold', 250, 1569)):
        reports[segment + '_report'] = dict(full_segment_completed=True, failure=None,
            requested_controls=count, completed_controls=count, completed_full_controls=count,
            physics_steps=count * 10, partial_substeps=0, trace_sha256=subjects[segment + '_trace']['sha256'])
        reports[segment + '_physics'] = dict(independent_segment_pass=True,
            recorded_trace_reproduced_through_last_sample=True, requested_segment_completed=True,
            intended_segment_controls=count, physics_steps=count * 10, compared_physics_steps=count * 10,
            recorded_partial_substeps=0, private_replay_steps_beyond_recorded_prefix=0, first_failure=None,
            original_trace_comparison={k: True for k in q.EXACT_FIELDS}, input_hashes=copy.deepcopy(pins))
        reports[segment + '_intent'] = dict(independent_physical_pass=True,
            intended_segment_completed=True, requested_segment_quiet_pass=True, requested_controls=count,
            recorded_controls=count, global_start=start, hashes=copy.deepcopy(pins),
            full_lifecycle_source_intent_pass=segment == 'main', source_metrics={'source_controls': 819})
    return reports, subjects


class QualificationTests(unittest.TestCase):
    def test_full_scope_pass_is_not_student_qualification(self):
        self.assertEqual(q.validate_reports(*fixture()), dict(offline_recovery_full_scope_qualified=True,
            fast_student_qualified=False, requested_main_controls=1569, requested_continuous_hold_controls=250))

    def test_owner_gates(self):
        for key, value in [('completion_accounting_passed', False), ('requested_recovery_completed', False),
                           ('raw_exit_known', False), ('all_postrun_pins_exact', False), ('processes_absent', False),
                           ('raw_python_exit_code', 1), ('exit_code', None), ('accounting_uncertainty', ['unknown'])]:
            with self.subTest(key=key):
                reports, subjects = fixture(); reports['owner'][key] = value
                with self.assertRaises(ValueError): q.validate_reports(reports, subjects)

    def test_each_report_must_be_full_pass(self):
        for role, key in [('main_report', 'full_segment_completed'), ('hold_report', 'full_segment_completed'),
                          ('main_physics', 'independent_segment_pass'), ('hold_physics', 'independent_segment_pass'),
                          ('main_intent', 'full_lifecycle_source_intent_pass'), ('main_intent', 'requested_segment_quiet_pass'),
                          ('hold_intent', 'requested_segment_quiet_pass')]:
            with self.subTest(role=role, key=key):
                reports, subjects = fixture(); reports[role][key] = False
                with self.assertRaises(ValueError): q.validate_reports(reports, subjects)

    def test_partial_and_wrong_scope(self):
        for role, key, value in [('main_report', 'physics_steps', 15689), ('hold_report', 'completed_controls', 249),
            ('hold_report', 'partial_substeps', 3), ('hold_physics', 'compared_physics_steps', 2499),
            ('main_physics', 'private_replay_steps_beyond_recorded_prefix', 1),
            ('hold_intent', 'global_start', 0), ('main_intent', 'recorded_controls', True)]:
            with self.subTest(role=role, key=key):
                reports, subjects = fixture(); reports[role][key] = value
                with self.assertRaises(ValueError): q.validate_reports(reports, subjects)

    def test_seven_physics_comparisons_required_both_segments(self):
        for segment in ('main', 'hold'):
            for key in q.EXACT_FIELDS:
                with self.subTest(segment=segment, key=key):
                    reports, subjects = fixture(); reports[segment + '_physics']['original_trace_comparison'][key] = False
                    with self.assertRaises(ValueError): q.validate_reports(reports, subjects)

    def test_direct_subjects_not_transitive_or_different_trace(self):
        for role, mapping, subject in [('owner', 'output_sha256', 'main_trace'),
            ('owner', 'input_sha256', 'native_arrays'), ('main_physics', 'input_hashes', 'main_report'),
            ('hold_physics', 'input_hashes', 'hold_trace'), ('main_intent', 'hashes', 'main_physics'),
            ('hold_intent', 'hashes', 'main_trace'), ('hold_intent', 'hashes', 'reference')]:
            with self.subTest(role=role, subject=subject):
                reports, subjects = fixture(); reports[role][mapping][subjects[subject]['path']] = 'f' * 64
                with self.assertRaises(ValueError): q.validate_reports(reports, subjects)

    def test_alias_normalization_and_conflict(self):
        subject = {'path': 'E:/fixture/trace', 'sha256': 'a' * 64}
        q.bound({'/mnt/e/fixture/trace': 'a' * 64}, subject)
        q.bound({'E:\\fixture\\trace': 'a' * 64}, subject)
        with self.assertRaises(ValueError):
            q.bound({'E:/fixture/trace': 'a' * 64, '/mnt/e/fixture/trace': 'b' * 64}, subject)

    def test_fixed_owner_and_all_four_audits(self):
        self.assertEqual(set(q.FIXED_RECEIPTS), {'owner', 'main_physics', 'main_intent', 'hold_physics', 'hold_intent'})
        self.assertTrue(all(len(v) == 64 for v in q.FIXED_RECEIPTS.values()))


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).parent / 'render_contact_sheet.py'
        cls.text = cls.path.read_text()
        cls.tree = ast.parse(cls.text)
        cls.original = (cls.path.parent.parent / 'render_qualified_expert_original.py').read_text()

    def test_camera_and_geometry_block_byte_identical(self):
        start, end = '    width, height = 480, 400\n', "    out = BASE / "
        def block(text): return text[text.index(start):text.index(end)]
        self.assertEqual(block(self.text), block(self.original))

    def test_fixed_nine_steps_reference_clock_identical(self):
        for line in ("    steps = np.array([0, 3500, 5500, 7500, 9500, 11690, 12690, 15690, 18190])",
                     "    frames = np.minimum(10 + (steps + 9) // 10, len(desired) - 1)",
                     "    actual, target = q[steps], desired[frames]"):
            self.assertIn(line, self.text); self.assertIn(line, self.original)

    def test_no_dynamics_or_inference_calls(self):
        called = {n.func.attr for n in ast.walk(self.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(called & {'mj_step', 'mj_forward', 'InferenceSession', 'run', 'Popen'})
        self.assertIn('mj_kinematics', called)

    def test_native_import_after_explicit_selection_and_gate(self):
        self.assertLess(self.text.index('if not args.root_selected:'), self.text.index('paths_and_pins()'))
        self.assertLess(self.text.index('paths_and_pins()'), self.text.index('    import mujoco'))
        imports = [n for n in self.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('mujoco' in ast.unparse(n) or 'numpy' in ast.unparse(n) or 'PIL' in ast.unparse(n) for n in imports))

    def test_explicit_continuity_final_boundary_and_unchanged_pose_assignment(self):
        for snippet in ("life['final_integration'], hold['initial_integration']",
                        "life['physics_qvel'][-1], hold['physics_qvel'][0]",
                        "life['physics_time'][-1] == hold['physics_time'][0]",
                        'q.shape == (18191, 30)', 'data.qpos[:] = pose'):
            self.assertIn(snippet, self.text)
        self.assertIn('offline expert recovery | fast student unqualified', self.text)

    def test_fresh_outputs_and_final_hashes(self):
        self.assertIn('out.mkdir(exist_ok=False)', self.text)
        self.assertLess(self.text.index('for path, digest in pins.items():'), self.text.index("(out / 'report.json').write_text"))


if __name__ == '__main__':
    unittest.main()
