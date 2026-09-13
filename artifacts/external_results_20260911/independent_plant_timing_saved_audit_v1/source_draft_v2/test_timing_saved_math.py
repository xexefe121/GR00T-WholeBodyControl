"""Adversarial saved sidecars built independently, without producer imports."""
import copy
import struct
import unittest
import numpy as np
import timing_saved_math as m

def fixture():
    rows=[];now=1000
    def sample():
        nonlocal now
        value=(now,now,now,now+1);now+=10;return value
    def opening(phase,parent):
        row=dict(zip(m.SPAN_FIELDS,[phase,0,0,1,parent,0,*sample(),-1,-1,-1,-1,1,-1]));rows.append(row);return len(rows)-1
    def closing(index):
        row=rows[index];row.update(zip(m.SPAN_FIELDS[10:14],sample()));row['end_complete']=1;row['return_kind']=1
    root=opening(1,-1)
    for phase in (2,3,4,5,6,7,8,9,10,12,14,15):
        token=opening(phase,root)
        if phase in (10,12):child=opening(phase+1,token);closing(child)
        closing(token)
    closing(root)
    hooks={'enabled':True,'failed':False,'first_hook_fault':None,'hook_faults':0,'active_scopes':0,
        'probe_fault_code':0,'spans_started':len(rows),'spans_closed':len(rows),'gc_recorded':0}
    metadata=dict(schema_version=1,integer_dtype='q',integer_itemsize=8,integer_byteorder='little',
        span_fields=m.SPAN_FIELDS,gc_fields=m.GC_FIELDS,phases=m.PHASES,spans_attempted=len(rows),spans_started=len(rows),
        spans_closed=len(rows),spans_denied=0,gc_attempted=0,gc_recorded=0,gc_denied=0,active_depth=0,
        open_gc_generations=(-1,-1,-1),fault='NONE',fault_record=-1,callback_detached=True,
        instrumentation_complete=True,hook_status=hooks.copy(),no_native_or_timing_qualification=True)
    byphase={r['phase']:r for r in rows}
    arrays={'step_index':np.arange(1,dtype=np.int64),'step_actual_start':np.asarray([byphase[3]['end_wall_after']+1],np.int64),
        'step_actual_end':np.asarray([byphase[12]['end_wall_after']+1],np.int64),
        'step_nominal_start':np.asarray([1000],np.int64),'step_nominal_end':np.asarray([2001000],np.int64)}
    core=dict(attempted=1,returned=1,captured=1,verified=1,committed=1)
    native=dict(attempted=1,returned=1,capture_attempts=1,captured=1,verification_attempts=1,verified=1)
    report=dict(session={'foundation':core,'stepper':native},api_counters={'step_attempted':1,'step_returned':1},
        epoch_ns=1000,setup_finished_ns=20,first_error=None,gc_enabled_before=True,gc_enabled_after=True,
        timing_instrumentation={'enabled':True,'complete':True,'errors':[],'hook_status':hooks.copy()})
    entry=dict(hooks,spans_started=0,spans_closed=0)
    stages=[{'stage':'setup_started','record_created_ns':0},{'stage':'native_restored','record_created_ns':5},
        {'stage':'epoch_armed','record_created_ns':10,'timing_probe':entry,'gc_enabled_unchanged':True},
        {'stage':'plant_or_setup_exit','record_created_ns':now+100,'timing_probe':hooks.copy()},
        {'stage':'preservation_complete','record_created_ns':now+1000}]
    return dict(metadata=metadata,rows=rows,gc=[],report=report,stages=stages,arrays=arrays,owned={})

def raw(rows,fields,order='little'):
    return b''.join(struct.pack(('<' if order=='little' else '>')+'q'*len(fields),*(r[f] for f in fields)) for r in rows)

def check(f):
    return m.audit(f['metadata'],raw(f['rows'],m.SPAN_FIELDS,f['metadata']['integer_byteorder']),
        raw(f['gc'],m.GC_FIELDS,f['metadata']['integer_byteorder']),f['report'],f['stages'],f['arrays'],f['owned'])

def gc_pair(f,foreign=False):
    row=lambda phase,t:dict(zip(m.GC_FIELDS,(phase,0,2 if foreign else 1,-1 if foreign else 0,1,0,0,t,t,t,t+1,1)))
    f['gc']=[row(1,1042),row(2,1045)];meta=f['metadata']
    meta['gc_recorded']=meta['gc_attempted']=2;meta['hook_status']['gc_recorded']=2
    f['report']['timing_instrumentation']['hook_status']['gc_recorded']=2

def partial_capture(f):
    # Fatal child-begin can occur before its row is reserved, after actual capture.
    f['rows']=[r for r in f['rows'] if r['phase']<=10]
    for row in f['rows']:
        if row['phase'] in (1,10):row['return_kind']=2
    n=len(f['rows']);meta=f['metadata'];meta['spans_started']=meta['spans_closed']=meta['spans_attempted']=n
    h=meta['hook_status'];h.update(spans_started=n,spans_closed=n,failed=True,first_hook_fault=['fatal_begin',-1],hook_faults=1)
    f['report']['timing_instrumentation'].update(complete=False,hook_status=h.copy())
    f['stages'][3]['timing_probe']=h.copy()
    f['report']['first_error']={'type':'StageTimeout','detail':'injected'}
    f['report']['session']['foundation'].update(captured=0,verified=0,committed=0)
    f['report']['session']['stepper'].update(verification_attempts=0,verified=0)
    f['arrays']={key:value[:0] for key,value in f['arrays'].items()}
    f['owned']={'adapter':{'last_capture':{'dataclass':'CapturedStep','fields':{}}}}


class TimingSaved(unittest.TestCase):
    def test_complete_independent_packet(self):
        result=check(fixture());self.assertTrue(result['instrumentation_complete']);self.assertEqual(result['spans'],15)
        self.assertEqual(result['phase_timing']['MJ_STEP']['complete_spans'],1)
        span=result['phase_timing']['TICK_ENVELOPE']['largest_complete_spans'][0]
        self.assertLessEqual(span['wall_lower_bound_ns'],span['wall_upper_bound_ns'])
    def test_explicit_big_endian_signed64(self):
        f=fixture();f['metadata']['integer_byteorder']='big';self.assertTrue(check(f)['instrumentation_complete'])
    def test_nested_gc_brackets_not_global_regression(self):
        f=fixture();gc_pair(f);self.assertTrue(check(f)['instrumentation_complete'])
    def test_foreign_inside_probe_is_overlap_only(self):
        f=fixture();gc_pair(f,True);result=check(f);self.assertEqual(result['foreign_thread_gc_overlap_rows'],[0,1])
        self.assertEqual(len(result['complete_gc_pairs']),1)
    def test_foreign_callback_cannot_claim_parent_thread(self):
        f=fixture();gc_pair(f,True);f['gc'][0]['active_span']=0
        with self.assertRaises(AssertionError):check(f)
    def test_partial_capture_retains_counts_no_credit(self):
        f=fixture();partial_capture(f);r=check(f)
        self.assertFalse(r['instrumentation_complete']);self.assertTrue(r['counter_handoff']['partial_counter_proof'])
        self.assertEqual(r['counter_handoff']['native_credit_inferred'],0)
    def test_partial_missing_owned_capture_rejected(self):
        f=fixture();partial_capture(f);f['owned']={}
        with self.assertRaises(AssertionError):check(f)
    def test_normal_tick_cannot_explain_counter_gap(self):
        f=fixture();partial_capture(f);f['rows'][0]['return_kind']=1
        with self.assertRaises(AssertionError):check(f)
    def test_completed_capture_scope_cannot_explain_missing_credit(self):
        f=fixture();partial_capture(f);f['rows'][-1]['return_kind']=1
        with self.assertRaises(AssertionError):check(f)
    def test_fake_failure_needed_for_counter_gap(self):
        f=fixture();partial_capture(f);f['report']['first_error']=None
        with self.assertRaises(AssertionError):check(f)
    def test_gc_capacity_denials_are_retained_failed_evidence(self):
        f=fixture();meta=f['metadata'];meta.update(gc_attempted=1,gc_denied=1,fault='CAPACITY',instrumentation_complete=False)
        meta['hook_status'].update(failed=True,probe_fault_code=1)
        f['report']['timing_instrumentation'].update(complete=False,hook_status=meta['hook_status'].copy())
        f['stages'][3]['timing_probe']=meta['hook_status'].copy()
        self.assertFalse(check(f)['instrumentation_complete'])
    def test_active_callback_is_incomplete_without_false_fault(self):
        f=fixture();gc_pair(f);f['gc'].pop();meta=f['metadata'];meta.update(gc_attempted=1,gc_recorded=1,open_gc_generations=[0,-1,-1],instrumentation_complete=False)
        meta['hook_status']['gc_recorded']=1;f['report']['timing_instrumentation'].update(complete=False,hook_status=meta['hook_status'].copy())
        self.assertFalse(check(f)['instrumentation_complete'])
    def test_adversarial_schema_clock_order_counter_stage_mutations(self):
        mutations={
            'dtype':lambda f:f['metadata'].update(integer_dtype='l'),
            'width':lambda f:f['metadata'].update(integer_itemsize=4),
            'rows':lambda f:f['metadata'].update(spans_started=16),
            'closed':lambda f:f['metadata'].update(spans_closed=14),
            'return':lambda f:f['rows'][8].update(return_kind=0),
            'root':lambda f:f['rows'][0].update(tick=1),
            'control':lambda f:f['rows'][1].update(control=1),
            'parent':lambda f:f['rows'][2].update(parent=1),
            'thread':lambda f:f['rows'][1].update(thread=2),
            'wall':lambda f:f['rows'][0].update(end_wall_before=0),
            'cpu':lambda f:f['rows'][0].update(end_thread_cpu=0),
            'process':lambda f:f['rows'][0].update(end_process_cpu=0),
            'step_epoch':lambda f:f['arrays']['step_nominal_end'].__setitem__(0,2001001),
            'wake':lambda f:f['arrays']['step_actual_start'].__setitem__(0,0),
            'finish':lambda f:f['arrays']['step_actual_end'].__setitem__(0,0),
            'attempt_count':lambda f:f['report']['session']['stepper'].update(attempted=2),
            'exit_counts':lambda f:f['stages'][3]['timing_probe'].update(spans_closed=14),
            'epoch_probe':lambda f:f['stages'][2]['timing_probe'].update(spans_started=1),
            'setup':lambda f:f['report'].update(setup_finished_ns=2000),
            'exit_time':lambda f:f['stages'][3].update(record_created_ns=1100),
            'completion':lambda f:f['report']['timing_instrumentation'].update(complete=False),
            'GC_policy':lambda f:f['report'].update(gc_enabled_after=False)}
        for name,mutation in mutations.items():
            with self.subTest(name=name):
                f=fixture();mutation(f)
                with self.assertRaises(AssertionError):check(f)
    def test_truncated_bytes_rejected(self):
        f=fixture()
        with self.assertRaises(AssertionError):m.audit(f['metadata'],raw(f['rows'],m.SPAN_FIELDS)[:-1],b'',f['report'],f['stages'],f['arrays'],{})
    def test_sequential_sibling_wall_overlap_rejected(self):
        f=fixture();by={row['phase']:row for row in f['rows']}
        by[2]['end_wall_after']=by[3]['start_wall_before']+1
        with self.assertRaisesRegex(AssertionError,'sibling wall'):check(f)
    def test_child_thread_cpu_beyond_parent_rejected(self):
        f=fixture();by={row['phase']:row for row in f['rows']}
        by[11]['end_thread_cpu']=by[10]['end_thread_cpu']+1
        with self.assertRaisesRegex(AssertionError,'Child CPU ends'):check(f)
    def test_child_process_cpu_beyond_parent_rejected(self):
        f=fixture();by={row['phase']:row for row in f['rows']}
        by[11]['end_process_cpu']=by[10]['end_process_cpu']+1
        with self.assertRaisesRegex(AssertionError,'Child CPU ends'):check(f)
    def test_sibling_cpu_regression_rejected(self):
        for clock in ('thread_cpu','process_cpu'):
            with self.subTest(clock=clock):
                f=fixture();by={row['phase']:row for row in f['rows']}
                by[2]['end_'+clock]=by[3]['start_'+clock]+1
                with self.assertRaisesRegex(AssertionError,'sibling CPU'):check(f)


if __name__=='__main__':unittest.main()
