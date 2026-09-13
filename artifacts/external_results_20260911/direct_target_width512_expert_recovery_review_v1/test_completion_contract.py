"""Exercise actual parsed completion expressions with synthetic traces only."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SOURCE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/source_prepared_v2')


class CompletionContract(unittest.TestCase):
    def test_exit_success_requires_main_and_hold(self):
        tree=ast.parse((SOURCE/'run_width251_actual_oracle.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        value=next(n.value for n in ast.walk(main) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='complete' for t in n.targets))
        expression=compile(ast.Expression(value),'<actual completion expression>','eval')
        cases=[({},False),({'nominal':{'full_segment_completed':True}},False)]
        for nominal in (False,True):
            for hold in (False,True):
                cases.append(({'nominal':{'full_segment_completed':nominal},
                               'extension':{'full_segment_completed':hold}},nominal and hold))
        for outcome,expected in cases:
            self.assertIs(eval(expression,{'outcome':outcome}),expected)

    def test_partial_last_control_and_fault_cannot_count_as_full_segment(self):
        tree=ast.parse((SOURCE/'run_actual_student_oracle.py').read_text())
        report=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='report_segment')
        env=dict(np=np,arrays_of=lambda t:t,quiet_diagnostic=lambda *a:{},standing_windows=lambda *a:[],
                 source_metrics=lambda *a:{})
        exec(compile(ast.Module(body=[report],type_ignores=[]),'<qualified report_segment>','exec'),env)
        s=dict(data=SimpleNamespace(time=.06,warning=SimpleNamespace(number=np.zeros(8,int),lastinfo=np.zeros(8,int))),
               motion={},original={},original29={},native=None,audit={},request={'motion_override':{}})
        cases=[([10,10,10],None,True),([10,10,9],None,False),([10,10,10],{'kind':'last_step_fault'},False),([10,10],None,False)]
        for counts,failure,expected in cases:
            trace=dict(physics_substeps=np.asarray(counts),physics_torque=np.zeros((sum(counts),23)),
                       range_excess=[],velocity_ratio=[],effort_ratio=[],clock_error=[])
            result=env['report_segment'](s,trace,3,failure,0.,'synthetic')
            self.assertIs(result['full_segment_completed'],expected)
            self.assertFalse(result['labels_admissible'])

    def test_hold_is_guarded_by_full_main_completion_and_preserves_range(self):
        tree=ast.parse((SOURCE/'run_width251_actual_oracle.py').read_text())
        branch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_branch')
        guard=next(n for n in branch.body if isinstance(n,ast.If)
                   and ast.unparse(n.test)=="result['full_segment_completed']")
        loops=[n for n in ast.walk(ast.Module(body=guard.body,type_ignores=[])) if isinstance(n,ast.For)]
        self.assertEqual(len(loops),1)
        controls=eval(compile(ast.Expression(loops[0].iter),'<hold range>','eval'),{'LIFECYCLE':1569,'EXTENSION':250})
        self.assertEqual(list(controls),list(range(1569,1819)))
        false_assignment=next(n for n in guard.orelse if isinstance(n,ast.Assign))
        false_result=eval(compile(ast.Expression(false_assignment.value),'<skipped hold>','eval'))
        self.assertEqual(false_result['attempted_controls'],0)
        self.assertFalse(false_result['full_segment_completed'])


if __name__=='__main__':unittest.main()
