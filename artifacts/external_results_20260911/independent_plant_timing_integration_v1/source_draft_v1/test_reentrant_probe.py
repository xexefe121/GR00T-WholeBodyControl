"""Fake readers return an earlier captured value after an intervening GC callback."""
import unittest
from timing_probe import SPAN_FIELDS,GC_FIELDS
from test_timing_probe import Fake,values

class ReentrantTests(unittest.TestCase):
    def interrupted(self,clock_name,foreign=False):
        c=Fake();p=c.probe();fired=False;original=getattr(c,clock_name)
        def reader():
            nonlocal fired
            captured=original()
            if not fired:
                fired=True;ident=c.ident
                if foreign:c.ident=99
                p.gc_event('start',{'generation':0});c.wall+=1000;c.thread+=100;c.process+=100
                p.gc_event('stop',{'generation':0,'collected':2,'uncollectable':0});c.ident=ident
            return captured
        clocks=[c.w,c.t,c.p,c.w];clocks[{'w':0,'t':1,'p':2}[clock_name]]=reader;p.readers=tuple(clocks)
        i=p.begin(6,10,1);p.mark_returned(i);p.end(i)
        return p.snapshot()
    def test_captured_wall_before_nested_gc_is_not_regression(self):
        s=self.interrupted('w');self.assertTrue(s['instrumentation_complete']);self.assertEqual(s['fault'],'NONE')
        rows=values(s,'gc',GC_FIELDS);self.assertEqual([r['inside_probe'] for r in rows],[1,1])
    def test_captured_process_cpu_before_nested_gc_is_not_regression(self):
        s=self.interrupted('p');self.assertTrue(s['instrumentation_complete']);self.assertEqual(s['fault'],'NONE')
    def test_captured_thread_cpu_before_nested_gc_is_not_regression(self):
        self.assertTrue(self.interrupted('t')['instrumentation_complete'])
    def test_foreign_thread_gc_interval_can_overlap_inflight_sample(self):
        s=self.interrupted('w',foreign=True);self.assertTrue(s['instrumentation_complete'])
        self.assertEqual([r['active_span'] for r in values(s,'gc',GC_FIELDS)],[-1,-1])
    def test_local_wall_bracket_still_rejects_backward_time(self):
        c=Fake();p=c.probe();p.readers=(lambda:100,c.t,c.p,lambda:99);p.begin(1,0,0)
        self.assertEqual(p.snapshot()['fault'],'CLOCK_REGRESSION')
    def test_paired_span_process_cpu_regression_rejected(self):
        c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);c.process=0;p.end(i)
        self.assertEqual(p.snapshot()['fault'],'CLOCK_REGRESSION')
    def test_completed_root_span_chronology_rejected(self):
        for clock in ('wall','thread','process'):
            with self.subTest(clock=clock):
                c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);p.end(i);setattr(c,clock,0);p.begin(1,1,0)
                self.assertEqual(p.snapshot()['fault'],'CLOCK_REGRESSION')
    def test_gc_pair_process_cpu_regression_rejected(self):
        c=Fake();p=c.probe();p.gc_event('start',{'generation':0});c.process=0;p.gc_event('stop',{'generation':0})
        self.assertEqual(p.snapshot()['fault'],'GC_PAIR')
    def test_closed_span_does_not_allow_new_foreign_span(self):
        c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);p.end(i);c.ident=9
        self.assertEqual(p.begin(1,1,0),-1);self.assertEqual(p.snapshot()['fault'],'FOREIGN_SPAN_THREAD')

if __name__=='__main__':unittest.main()
