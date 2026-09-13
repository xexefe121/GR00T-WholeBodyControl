"""Deterministic clocks and fake callback registry only; no actual GC/native calls."""
from array import array
import unittest
from timing_probe import TimingProbe,SPAN_FIELDS,GC_FIELDS,PHASES,MAX_SPANS,MAX_GC_EVENTS

class Fake:
    def __init__(self):self.wall=100;self.thread=10;self.process=20;self.ident=7
    def w(self):self.wall+=1;return self.wall
    def t(self):self.thread+=1;return self.thread
    def p(self):self.process+=1;return self.process
    def tid(self):return self.ident
    def probe(self,**kw):return TimingProbe(wall_ns=self.w,thread_cpu_ns=self.t,process_cpu_ns=self.p,thread_id=self.tid,span_capacity=8,gc_capacity=8,**kw)

def values(snapshot,name,fields):
    a=array('q');a.frombytes(snapshot[name]);return [dict(zip(fields,a[i:i+len(fields)])) for i in range(0,len(a),len(fields))]

class ProbeTests(unittest.TestCase):
    def test_normal_return_and_owned_snapshot(self):
        c=Fake();p=c.probe();token=p.begin(1,0,0);self.assertTrue(p.mark_returned(token));self.assertTrue(p.end(token))
        snap=p.snapshot();self.assertTrue(snap['instrumentation_complete']);self.assertEqual(snap['integer_itemsize'],8)
        raw=snap['spans'];p.spans[0]=15;self.assertEqual(snap['spans'],raw)
    def test_nested_parent_and_original_call_order(self):
        c=Fake();p=c.probe();calls=[]
        outer=p.begin(1,0,0);inner=p.begin(9,0,0);calls.append('native_stub_once');p.mark_returned(inner);p.end(inner);p.mark_returned(outer);p.end(outer)
        rows=values(p.snapshot(),'spans',SPAN_FIELDS)
        self.assertEqual(calls,['native_stub_once']);self.assertEqual([r['parent'] for r in rows],[-1,0]);self.assertTrue(p.snapshot()['instrumentation_complete'])
    def test_wall_stall_separate_cpu_evidence(self):
        c=Fake();p=c.probe();i=p.begin(9,12099,1209);c.wall+=23000000;p.mark_returned(i);p.end(i)
        r=values(p.snapshot(),'spans',SPAN_FIELDS)[0]
        self.assertGreater(r['end_wall_before']-r['start_wall_after'],23000000)
        self.assertEqual(r['end_thread_cpu']-r['start_thread_cpu'],1)
        self.assertEqual(r['end_process_cpu']-r['start_process_cpu'],1)
    def test_cpu_work_distinguished_and_process_may_exceed_wall(self):
        c=Fake();p=c.probe();i=p.begin(9,1,0);c.wall+=100;c.thread+=100;c.process+=200;p.mark_returned(i);p.end(i)
        r=values(p.snapshot(),'spans',SPAN_FIELDS)[0]
        self.assertGreater(r['end_process_cpu']-r['start_process_cpu'],r['end_wall_after']-r['start_wall_before'])
        self.assertTrue(p.snapshot()['instrumentation_complete'])
    def test_returned_mark_survives_end_clock_failure(self):
        c=Fake();p=c.probe();i=p.begin(9,0,0);p.mark_returned(i)
        def broken():raise RuntimeError('fake')
        p.readers=(broken,c.t,c.p,broken)
        self.assertFalse(p.end(i));r=values(p.snapshot(),'spans',SPAN_FIELDS)[0]
        self.assertEqual(r['return_kind'],1);self.assertEqual(r['end_complete'],0);self.assertEqual(p.snapshot()['fault'],'CLOCK_EXCEPTION')
    def test_original_exception_classification_preserved(self):
        c=Fake();p=c.probe();i=p.begin(9,0,0);caught=None
        try:raise ValueError('original native stub failure')
        except ValueError as e:caught=e;p.mark_returned(i,raised=True);p.end(i)
        self.assertEqual(str(caught),'original native stub failure');self.assertEqual(values(p.snapshot(),'spans',SPAN_FIELDS)[0]['return_kind'],2)
    def test_unknown_call_remains_open(self):
        c=Fake();p=c.probe();p.begin(9,0,0);s=p.snapshot()
        self.assertEqual(s['active_depth'],1);self.assertFalse(s['instrumentation_complete']);self.assertEqual(values(s,'spans',SPAN_FIELDS)[0]['return_kind'],0)
    def test_capacity_never_overwrites(self):
        c=Fake();p=TimingProbe(wall_ns=c.w,thread_cpu_ns=c.t,process_cpu_ns=c.p,thread_id=c.tid,span_capacity=1,gc_capacity=1)
        i=p.begin(1,0,0);p.mark_returned(i);p.end(i);before=p.spans.tobytes();self.assertEqual(p.begin(1,1,0),-1)
        self.assertEqual(p.spans.tobytes(),before);self.assertEqual(p.spans_attempted,2);self.assertEqual(p.spans_denied,1);self.assertFalse(p.snapshot()['instrumentation_complete'])
    def test_typed_argument_rejection(self):
        for phase,tick,control in [(True,0,0),(0,0,0),(16,0,0),(1,-1,0),(1,0,1.0)]:
            p=Fake().probe();self.assertEqual(p.begin(phase,tick,control),-1);self.assertEqual(p.spans_started,0)
    def test_clock_value_and_backward_wall(self):
        for value in (True,-1,1.0,float('inf'),1<<63):
            c=Fake();p=c.probe();p.readers=(lambda:value,c.t,c.p,c.w);p.begin(1,0,0)
            self.assertEqual(p.snapshot()['fault'],'CLOCK_VALUE')
        c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);c.wall=0;p.end(i);self.assertEqual(p.snapshot()['fault'],'CLOCK_REGRESSION')
    def test_thread_cpu_backward(self):
        c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);c.thread=0;p.end(i);self.assertEqual(p.snapshot()['fault'],'CLOCK_REGRESSION')
    def test_stack_and_double_return_errors(self):
        c=Fake();p=c.probe();a=p.begin(1,0,0);p.begin(9,0,0);self.assertFalse(p.mark_returned(a));self.assertEqual(p.snapshot()['fault'],'STACK_ORDER')
        p=Fake().probe();a=p.begin(1,0,0);p.mark_returned(a);self.assertFalse(p.mark_returned(a));self.assertEqual(p.snapshot()['fault'],'RETURN_MARK')
    def test_foreign_thread_cannot_close_span(self):
        c=Fake();p=c.probe();i=p.begin(1,0,0);c.ident=8;self.assertFalse(p.mark_returned(i));self.assertEqual(p.snapshot()['fault'],'FOREIGN_SPAN_THREAD')
    def test_gc_pair_records_cpu_and_active_phase(self):
        c=Fake();p=c.probe();i=p.begin(6,10,1)
        p.gc_event('start',{'generation':0});c.wall+=200;c.thread+=90;c.process+=90
        p.gc_event('stop',{'generation':0,'collected':4,'uncollectable':0});p.mark_returned(i);p.end(i)
        r=values(p.snapshot(),'gc',GC_FIELDS);self.assertEqual([v['active_span'] for v in r],[0,0]);self.assertEqual([v['phase'] for v in r],[1,2]);self.assertTrue(p.snapshot()['instrumentation_complete'])
    def test_gc_foreign_thread_unattributed(self):
        c=Fake();p=c.probe();i=p.begin(6,10,1);c.ident=8;p.gc_event('start',{'generation':1});p.gc_event('stop',{'generation':1});c.ident=7;p.mark_returned(i);p.end(i)
        self.assertEqual([r['active_span'] for r in values(p.snapshot(),'gc',GC_FIELDS)],[-1,-1])
    def test_gc_inside_probe_flag(self):
        c=Fake();p=c.probe();p.in_probe=1;p.gc_event('start',{'generation':0});p.gc_event('stop',{'generation':0})
        self.assertEqual([r['inside_probe'] for r in values(p.snapshot(),'gc',GC_FIELDS)],[1,1])
    def test_gc_overflow_counts_without_throw(self):
        c=Fake();p=TimingProbe(wall_ns=c.w,thread_cpu_ns=c.t,process_cpu_ns=c.p,thread_id=c.tid,span_capacity=1,gc_capacity=1)
        p.gc_event('start',{'generation':0});before=p.gc.tobytes();p.gc_event('stop',{'generation':0})
        self.assertEqual(p.gc.tobytes(),before);self.assertEqual((p.gc_attempted,p.gc_recorded,p.gc_denied),(2,1,1));self.assertFalse(p.snapshot()['instrumentation_complete'])
    def test_gc_invalid_and_unmatched_events(self):
        for info in (None,{'generation':True},{'generation':3},{'generation':0,'collected':-1}):
            p=Fake().probe();p.gc_event('start',info);self.assertEqual(p.snapshot()['fault'],'GC_SCHEMA')
        p=Fake().probe();p.gc_event('stop',{'generation':0});self.assertEqual(p.snapshot()['fault'],'GC_PAIR')
    def test_registry_preserves_existing_callbacks(self):
        p=Fake().probe();a=object();b=object();registry=[a,b];p.attach(registry);self.assertIs(registry[-1],p.callback)
        with self.assertRaises(ValueError):p.attach(registry)
        self.assertTrue(p.detach());self.assertEqual(registry,[a,b]);self.assertTrue(p.snapshot()['instrumentation_complete'])
    def test_missing_registry_callback_never_erases_other_callbacks(self):
        p=Fake().probe();other=object();registry=[other];p.attach(registry);registry.pop();self.assertFalse(p.detach());self.assertEqual(registry,[other])
    def test_scope_storage_bound(self):
        bound=11*18190+3*1819+10*1818
        self.assertEqual(bound,223727);self.assertLess(bound,MAX_SPANS)
        self.assertEqual(MAX_SPANS*len(SPAN_FIELDS)*8+MAX_GC_EVENTS*len(GC_FIELDS)*8,33947648)
    def test_packed_format_and_sampling_order_are_explicit(self):
        c=Fake();p=c.probe();i=p.begin(1,0,0);p.mark_returned(i);p.end(i);s=p.snapshot()
        self.assertIn(s['integer_byteorder'],('little','big'))
        self.assertEqual(s['integer_itemsize'],8)
        r=values(s,'spans',SPAN_FIELDS)[0]
        self.assertLess(r['start_wall_before'],r['start_wall_after'])
        self.assertLess(r['start_wall_after'],r['end_wall_before'])
        self.assertLess(r['end_wall_before'],r['end_wall_after'])

if __name__=='__main__':unittest.main()
