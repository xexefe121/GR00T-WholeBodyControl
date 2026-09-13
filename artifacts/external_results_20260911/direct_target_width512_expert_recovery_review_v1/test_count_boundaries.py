"""Synthetic wrapped calls only. No native/model imports or task files."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/source_draft_v1')
sys.path.insert(0,str(SOURCE))
from counted_work import WorkLedger,BudgetExceeded,BatchProxy,SessionProxy,Hooks


class CountBoundaries(unittest.TestCase):
    def test_failure_retains_attempted_multistep_units(self):
        ledger=WorkLedger({'batch_fd_step':(2,50000)})
        progressed=[]
        def step(**kwargs):
            progressed.append(kwargs['nstep'])
            raise RuntimeError('partial internal progress unknown')
        proxy=BatchProxy(SimpleNamespace(step=step),2460,ledger)
        with self.assertRaisesRegex(RuntimeError,'partial internal'):
            proxy.step(nstep=9)
        self.assertEqual(progressed,[9])
        self.assertEqual(ledger.counts['batch_fd_step'],dict(attempted_calls=1,returned_calls=0,attempted_units=22140,returned_units=0))
        self.assertEqual(ledger.snapshot()['active'],[])
        self.assertEqual(ledger.first_failure['units'],22140)

    def test_cap_refusal_happens_before_wrapped_work(self):
        ledger=WorkLedger({'BFM_actor':(1,1)})
        calls=[]
        ledger.invoke('BFM_actor',1,lambda:calls.append('first'))
        with self.assertRaises(BudgetExceeded):
            ledger.invoke('BFM_actor',1,lambda:calls.append('forbidden'))
        self.assertEqual(calls,['first'])
        self.assertEqual(ledger.counts['BFM_actor']['attempted_calls'],1)
        self.assertEqual(ledger.refused['kind'],'budget_refusal')

    def test_nested_failure_keeps_first_actual_operation(self):
        ledger=WorkLedger({'outer':(1,1),'inner':(1,7)})
        def fail():raise OSError('injected')
        with self.assertRaises(OSError):
            ledger.invoke('outer',1,lambda:ledger.invoke('inner',7,fail))
        self.assertEqual(ledger.first_failure['category'],'inner')
        self.assertEqual([x['category'] for x in ledger.first_failure['stack']],['outer','inner'])
        self.assertEqual(ledger.snapshot()['active'],[])

    def test_backward_rows_count_separately_and_output_untouched(self):
        ledger=WorkLedger({'BFM_backward':(1,8)})
        result=object(); inputs={'state':SimpleNamespace(shape=(8,52))}
        seen=[]
        actual=SimpleNamespace(run=lambda names,value:seen.append(value) or result)
        proxy=SessionProxy(actual,'BFM_backward',ledger)
        self.assertIs(proxy.run(None,inputs),result)
        self.assertIs(seen[0],inputs)
        self.assertEqual(ledger.counts['BFM_backward'],dict(attempted_calls=1,returned_calls=1,attempted_units=8,returned_units=8))

    def test_hooks_distinguish_actual_from_private_and_restore_aliases(self):
        ledger=WorkLedger()
        calls=[]
        native_step=lambda model,data:calls.append(data)
        native=SimpleNamespace(mj_step=native_step)
        class Planner:
            def rollout(self,*args,**kwargs):return 'rollout'
            def linearize(self,*args,**kwargs):return 'linearize'
        original_rollout=Planner.rollout
        original_ilqr=lambda planner,*args,**kwargs:planner.rollout()
        core=SimpleNamespace(Batch=lambda *a,**k:None,Planner=Planner,ilqr=original_ilqr)
        restoration=SimpleNamespace(ilqr=original_ilqr)
        driver=SimpleNamespace(ilqr=original_ilqr)
        hooks=Hooks(ledger,native,core,restoration,driver)
        actual=object();private=object()
        try:
            hooks.register_live(actual)
            native.mj_step(None,actual);native.mj_step(None,private)
            hard=Planner();hard.feasibility=object()
            soft=Planner();soft.feasibility=None
            driver.ilqr(hard);restoration.ilqr(soft)
            self.assertEqual(ledger.counts['ordinary_ilqr']['returned_calls'],1)
            self.assertEqual(ledger.counts['restoration_ilqr']['returned_calls'],1)
            self.assertEqual(ledger.counts['ordinary_rollout']['returned_calls'],1)
            self.assertEqual(ledger.counts['restoration_rollout']['returned_calls'],1)
            self.assertEqual(ledger.counts['native_live_step']['returned_units'],1)
            self.assertEqual(ledger.counts['native_private_step']['returned_units'],1)
        finally:hooks.close()
        self.assertIs(native.mj_step,native_step)
        self.assertIs(Planner.rollout,original_rollout)
        self.assertIs(driver.ilqr,original_ilqr)


if __name__=='__main__':unittest.main()
