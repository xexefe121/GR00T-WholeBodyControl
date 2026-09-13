"""Independent pure-fake tests of the four newly separated private budgets.

No native/model imports or task-array loads. The earlier five counter tests remain
preserved; this file supplies the updated hook surface and new boundary cases.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/source_prepared_v2')
sys.path.insert(0,str(SOURCE))
from counted_work import WorkLedger, BudgetExceeded, Hooks


class Fixture:
    def __init__(self):
        self.calls=[]; self.failure=False
        def native_step(model,data):
            self.calls.append(data)
            if self.failure: raise RuntimeError('partial native failure')
            return data
        self.native=SimpleNamespace(mj_step=native_step)
        class Planner:
            def rollout(self,*a,**k):return ('rollout',a,k)
            def linearize(self,*a,**k):return ('linearize',a,k)
        def ilqr(planner,*a,**k):return planner.rollout(*a,**k)
        self.core=SimpleNamespace(Planner=Planner,Batch=lambda *a,**k:None,ilqr=ilqr)
        def inspect(model,data):return self.native.mj_step(model,data)
        self.driver=SimpleNamespace(ilqr=ilqr,inspect_native_segment=inspect,preview_native_control=inspect)
        self.restoration=SimpleNamespace(ilqr=ilqr,inspect_native_segment=inspect)
        self.original=(self.native.mj_step,Planner.rollout,Planner.linearize,self.core.Batch,
                       self.driver.ilqr,self.driver.inspect_native_segment,self.driver.preview_native_control,
                       self.restoration.ilqr,self.restoration.inspect_native_segment)
        self.ledger=WorkLedger()
        self.hooks=Hooks(self.ledger,self.native,self.core,self.restoration,self.driver)
        self.live=object(); self.private=object(); self.hooks.register_live(self.live)
        self.seed=SimpleNamespace(sessions={'actor':SimpleNamespace(),'backward':SimpleNamespace()},
                                 propose=lambda:self.native.mj_step(None,self.private))
        self.hooks.instrument_seed(self.seed)

    def invoke(self,scope):
        if scope=='BFM':return self.seed.propose()
        if scope=='initial_certificate':return self.driver.inspect_native_segment(None,self.private)
        if scope=='restoration_certificate':return self.restoration.inspect_native_segment(None,self.private)
        if scope=='preview':return self.driver.preview_native_control(None,self.private)
        raise AssertionError(scope)

    def totals(self,key):
        scopes=('BFM','initial_certificate','restoration_certificate','preview')
        self_count=self.ledger.counts['native_private_step'][key]
        category_count=sum(self.ledger.counts['native_'+scope+'_step'][key] for scope in scopes)
        return self_count,category_count


class PrivateCategories(unittest.TestCase):
    def setUp(self):
        self.f=Fixture();self.addCleanup(self.f.hooks.close)

    def assertConservation(self):
        for key in ('attempted_calls','returned_calls','attempted_units','returned_units'):
            self.assertEqual(*self.f.totals(key))

    def test_disjoint_categories_and_live_prefix_exclusion(self):
        f=self.f
        self.assertIs(f.native.mj_step(None,f.live),f.live)
        for scope in ('BFM','initial_certificate','restoration_certificate','preview'):
            self.assertIs(f.invoke(scope),f.private)
            self.assertEqual(f.ledger.counts['native_'+scope+'_step']['returned_units'],1)
        self.assertEqual(len(f.calls),5)
        self.assertEqual(f.ledger.counts['native_live_step']['returned_units'],1)
        self.assertEqual(f.ledger.counts['native_private_step']['returned_units'],4)
        self.assertConservation()

    def test_category_refusal_consumes_no_native_attempt(self):
        f=self.f;f.ledger.limits['native_preview_step']=(0,0)
        with self.assertRaises(BudgetExceeded):f.invoke('preview')
        self.assertEqual(f.calls,[])
        self.assertEqual(f.ledger.counts['imminent_native_preview']['attempted_calls'],1)
        self.assertEqual(f.ledger.counts['imminent_native_preview']['returned_calls'],0)
        self.assertEqual(f.ledger.counts['native_private_step']['attempted_calls'],0)
        self.assertEqual(f.ledger.refused['category'],'native_preview_step')
        self.assertIsNone(f.hooks.private_context)
        self.assertEqual(f.ledger.active,[])
        self.assertConservation()

    def test_aggregate_refusal_consumes_no_category_attempt(self):
        f=self.f;f.ledger.limits['native_private_step']=(0,0)
        with self.assertRaises(BudgetExceeded):f.invoke('BFM')
        self.assertEqual(f.calls,[])
        self.assertEqual(f.ledger.counts['native_BFM_step']['attempted_calls'],0)
        self.assertEqual(f.ledger.counts['BFM_propose']['attempted_calls'],1)
        self.assertEqual(f.ledger.counts['BFM_propose']['returned_calls'],0)
        self.assertEqual(f.ledger.refused['category'],'native_private_step')
        self.assertIsNone(f.hooks.private_context)
        self.assertConservation()

    def test_partial_failure_keeps_attempt_without_return_and_clears_scope(self):
        f=self.f;f.failure=True
        with self.assertRaisesRegex(RuntimeError,'partial native failure'):f.invoke('restoration_certificate')
        self.assertEqual(len(f.calls),1)
        for key in ('native_private_step','native_restoration_certificate_step'):
            self.assertEqual(f.ledger.counts[key],dict(attempted_calls=1,returned_calls=0,attempted_units=1,returned_units=0))
        self.assertEqual([v['category'] for v in f.ledger.first_failure['stack']],
            ['restoration_native_certificate','native_private_step','native_restoration_certificate_step'])
        self.assertIsNone(f.hooks.private_context)
        self.assertEqual(f.ledger.active,[])
        f.failure=False;f.invoke('preview')
        self.assertEqual(f.ledger.first_failure['category'],'native_restoration_certificate_step')
        self.assertConservation()

    def test_unclassified_and_nested_private_calls_do_not_mutate_scope(self):
        f=self.f
        with self.assertRaisesRegex(ValueError,'Unclassified'):f.native.mj_step(None,f.private)
        self.assertEqual(f.calls,[])
        with f.hooks.private('preview'):
            with self.assertRaisesRegex(ValueError,'nested'):f.invoke('BFM')
            self.assertEqual(f.hooks.private_context,'preview')
            f.native.mj_step(None,f.private)
        self.assertIsNone(f.hooks.private_context)
        self.assertEqual(f.ledger.counts['BFM_propose']['attempted_calls'],0)
        self.assertConservation()

    def test_alias_restoration_and_hard_soft_routing(self):
        f=self.f;hard=f.core.Planner();hard.feasibility=object()
        soft=f.core.Planner();soft.feasibility=None
        self.assertEqual(f.driver.ilqr(hard,'x')[1],('x',))
        self.assertEqual(f.restoration.ilqr(soft,'y')[1],('y',))
        self.assertEqual(f.ledger.counts['ordinary_ilqr']['returned_calls'],1)
        self.assertEqual(f.ledger.counts['restoration_ilqr']['returned_calls'],1)
        self.assertEqual(f.ledger.counts['ordinary_rollout']['returned_calls'],1)
        self.assertEqual(f.ledger.counts['restoration_rollout']['returned_calls'],1)
        f.hooks.close()
        current=(f.native.mj_step,f.core.Planner.rollout,f.core.Planner.linearize,f.core.Batch,
                 f.driver.ilqr,f.driver.inspect_native_segment,f.driver.preview_native_control,
                 f.restoration.ilqr,f.restoration.inspect_native_segment)
        for actual,expected in zip(current,f.original):self.assertIs(actual,expected)


if __name__=='__main__':unittest.main()
