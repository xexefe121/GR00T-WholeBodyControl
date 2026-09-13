"""Deterministic fake clocks, bytes and handles only. No native/model/process calls."""
import base64
import json
import sys
import unittest
import numpy as np
from clock_core import Binding,CapturedStep,WarningLedger,FixedLedger,Result,encode,digest,CapacityFailure
from mailbox import Mailbox
from history import SIZES
from recorded_protocol import CommandTable,decode_job,reply_to_job
from session import Session,DeadlineClock,ObservedEndpoint
from dummy_worker import worker_once,WorkerLifecycle
from pending_result import PendingResultWorker,WorkerResultFailure
from evidence import export_owned

BINDING=Binding('synthetic-process-scaffold','0'*64,'1'*64,3)
LIMITS=np.tile(np.array([-2.,2.]),(23,1))

class Lock:
    def __init__(self):self.held=False
    def acquire(self,blocking):
        if blocking is not False:raise AssertionError('blocking lock forbidden')
        return not self.held
    def release(self):pass

class FakeClock:
    def __init__(self):self.ns=0;self.after_wait=None
    def now_ns(self):return self.ns
    def wait_until_ns(self,deadline):
        if self.after_wait is not None:self.after_wait(deadline)
        self.ns=max(self.ns,deadline)

class FakeStepper:
    def __init__(self,clock,strict_at=None,raise_at=None):
        self.clock=clock;self.initial_time=0.;self.time=0.;self.attempted=self.returned=self.captured=self.verified=0
        self.strict_at,self.raise_at=strict_at,raise_at;self.targets=[];self.closed=False
    def boundary_snapshot(self):
        state=np.zeros(291,np.float64);state[0]=self.time
        terms={k:np.full(v,self.returned,np.float32) for k,v in SIZES.items() if k!='actions'}
        return state.tobytes(),terms
    def step(self,target):
        self.attempted+=1
        if self.attempted==self.raise_at:raise RuntimeError('injected native API attempt failure')
        self.targets.append(target.copy());self.returned+=1;self.time+=.002;self.clock.ns+=100_000
    def capture_step(self):
        self.captured+=1
        # The foundation state is opaque; real adapter packed373 bytes are unchanged.
        state=np.zeros(373,np.float64);state[0]=self.time
        return CapturedStep(self.time,state.tobytes(),np.zeros(23,np.float64).tobytes(),WarningLedger((0,)*8,(0,)*8))
    def verify_step(self,snapshot):
        if self.returned==self.strict_at:return 'synthetic strict failure'
        self.verified+=1
    def counters(self):
        return dict(attempted=self.attempted,returned=self.returned,captured=self.captured,verified=self.verified,
                    model_identity=dict(exit_verified=self.closed))
    def verify_model_exit(self):
        if self.closed:raise ValueError('duplicate close')
        self.closed=True;return dict(passed=True)

def table(n=4,same_targets=False):
    target=np.zeros((n,23),np.float64) if same_targets else np.repeat(np.linspace(0,.3,n)[:,None],23,axis=1)
    actions=np.repeat(np.arange(n,dtype=np.float32)[:,None],23,axis=1)
    return CommandTable(target,actions,np.arange(11,11+n,dtype=np.int64),['2'*64]*n,LIMITS)

def setup(n=4,**stepper_args):
    clock=FakeClock();stepper=FakeStepper(clock,**stepper_args);commands=table(n)
    jobs=Mailbox(2,32768,Lock);results=Mailbox(2,8192,Lock)
    session=Session(clock=clock,stepper=stepper,epoch_ns=100_000_000,binding=BINDING,table=commands,
                    initial_previous_raw=np.zeros(23,np.float32).tobytes(),limits=LIMITS,jobs=jobs,results=results,main_controls=n-1)
    return clock,stepper,commands,jobs,results,session

def run_fake(clock,stepper,commands,jobs,results,session,pump=True):
    ledger=FixedLedger(1000);worker=PendingResultWorker(BINDING,commands,clock,ledger);worker_live=True
    while session.foundation.returned<session.foundation.requested_steps:
        okay=session.tick_once()
        if pump and worker_live:
            try:worker_once(jobs,results,worker)
            except WorkerResultFailure:worker_live=False
        if not okay:break
    session.close_native()
    return session.summary()

class ScaffoldTests(unittest.TestCase):
    def test_complete_dummy_transport_and_continuous_hold(self):
        parts=setup();r=run_fake(*parts);plant=parts[-1].foundation
        self.assertTrue(r['component_preliminary_pass']);self.assertFalse(r['component_qualified'])
        self.assertEqual([r.command.command_id for r in plant.control_records.records()],[c.command_id for c in parts[2].commands])
        self.assertEqual(plant.history.entries,4);self.assertEqual(plant.returned,40)
        self.assertEqual(plant.control_records.records()[3].incoming_raw,parts[2].commands[2].raw_action)
        self.assertEqual(plant.control_records.records()[3].history_before,plant.control_records.records()[2].history_after)
        self.assertEqual([r.nominal_start for r in plant.step_records.records()],[100_000_000+2_000_000*i for i in range(40)])
    def test_identical_targets_preserve_distinct_raw_feedback(self):
        clock,stepper,_,jobs,results,_=setup();commands=table(same_targets=True)
        session=Session(clock=clock,stepper=stepper,epoch_ns=100_000_000,binding=BINDING,table=commands,
            initial_previous_raw=np.zeros(23,np.float32).tobytes(),limits=LIMITS,jobs=jobs,results=results,main_controls=3)
        r=run_fake(clock,stepper,commands,jobs,results,session)
        self.assertTrue(r['component_preliminary_pass'])
        a,b=session.foundation.control_records.records()[1:3]
        self.assertEqual(a.command.target,b.command.target);self.assertNotEqual(a.command.raw_action,b.command.raw_action)
        self.assertEqual(b.incoming_raw,a.command.raw_action)
    def test_hung_worker_holds_and_fails_even_if_foundation_timing_passes(self):
        parts=setup();r=run_fake(*parts,pump=False)
        self.assertTrue(r['foundation']['timing_passed'])
        self.assertFalse(r['component_preliminary_pass']);self.assertEqual(r['commands']['held_controls'],[1,2,3])
        self.assertEqual(r['foundation']['returned'],40);self.assertEqual(r['foundation']['input_fault']['control'],1)
        self.assertEqual([x.command.command_id for x in parts[-1].foundation.control_records.records()],[parts[2].commands[0].command_id]*4)
    def test_late_result_not_silently_admitted(self):
        clock,stepper,commands,jobs,results,session=setup();session.tick_once()
        for _ in range(9):session.tick_once()
        clock.ns=120_000_000
        publication=next(p for _,status,p in jobs.poll_once() if status=='TAKEN')
        results.try_publish(1,reply_to_job(publication.payload,BINDING,commands,clock.ns));session.tick_once()
        events=[json.loads(x) for x in session.foundation.events.records()]
        self.assertTrue(any(x.get('rejection')=='LATE' for x in events));self.assertIsNotNone(session.foundation.input_fault)
    def test_result_during_wait_exposes_conservative_poll_gap(self):
        clock,stepper,commands,jobs,results,session=setup();ledger=FixedLedger(100)
        for _ in range(10):session.tick_once()
        # This callback is test instrumentation only, not an integration feature.
        def after_wait(deadline):
            if deadline==120_000_000:
                clock.ns=119_000_000;worker_once(jobs,results,PendingResultWorker(BINDING,commands,clock,ledger));clock.ns=120_000_000
        clock.after_wait=after_wait;session.tick_once()
        self.assertIsNotNone(session.foundation.input_fault)
        self.assertEqual(session.foundation.control_records.records()[-1].held,True)
    def test_wrong_job_binding_and_digest_rejected(self):
        parts=setup();parts[-1].tick_once();publication=next(p for _,s,p in parts[3].poll_once() if s=='TAKEN')
        job=decode_job(publication.payload,BINDING,parts[2]);self.assertEqual(job.activation,1)
        for field,value in [('activation',2),('input_digest','f'*64),('schedule_digest','f'*64),('created_ns',True)]:
            altered=json.loads(publication.payload);altered[field]=value
            with self.assertRaises(ValueError):decode_job(encode(altered),BINDING,parts[2])
        with self.assertRaises(ValueError):decode_job(publication.payload,Binding('other','0'*64,'1'*64,3),parts[2])
    def test_duplicate_admission_preserved(self):
        clock,stepper,commands,jobs,results,session=setup();session.tick_once()
        p=next(p for _,s,p in jobs.poll_once() if s=='TAKEN');reply=reply_to_job(p.payload,BINDING,commands,clock.ns)
        results.try_publish(1,reply);session.foundation.poll_results();results.try_publish(1,reply);session.foundation.poll_results()
        self.assertEqual(json.loads(session.foundation.events.records()[-1])['rejection'],'DUPLICATE')
    def test_transport_specific_failure_logged_without_remapping(self):
        class Corrupt:
            def poll_once(self):return ((0,'CORRUPT',None),)
        ledger=FixedLedger(1);observed=ObservedEndpoint(Corrupt(),FakeClock(),ledger,'test')
        self.assertEqual(observed.poll_once(),((0,'CORRUPT',None),))
        self.assertEqual(json.loads(ledger.records()[0])['status'],'CORRUPT')
    def test_logger_overflow_retains_returned_capture_and_no_extra_step(self):
        parts=setup();session=parts[-1];session.outer=FixedLedger(0)
        self.assertFalse(session.tick_once());self.assertEqual(session.foundation.returned,1);self.assertEqual(session.stepper.captured,1)
        self.assertIsNotNone(session.outer.overflow_record);self.assertFalse(session.tick_once());self.assertEqual(session.stepper.returned,1)
    def test_strict_failure_counts_returned_and_captured(self):
        parts=setup(strict_at=7);r=run_fake(*parts)
        self.assertEqual([r['foundation'][k] for k in ['attempted','returned','captured','verified']],[7,7,7,6])
        self.assertFalse(r['component_preliminary_pass']);self.assertTrue(r['postrun_model_identity_pass'])
    def test_api_exception_preserves_attempt_return_gap(self):
        parts=setup(raise_at=7);r=run_fake(*parts)
        self.assertEqual([r['foundation'][k] for k in ['attempted','returned','captured']],[7,6,6]);self.assertFalse(r['component_preliminary_pass'])
    def test_fixed_deadline_clock_does_not_rebase_or_skip(self):
        clock=FakeClock();sleeps=[]
        def sleep(seconds):sleeps.append(seconds);clock.ns+=int(round(seconds*1e9))+5
        real=DeadlineClock(clock.now_ns,sleep);real.wait_until_ns(2_000_000);real.wait_until_ns(4_000_000)
        self.assertEqual(clock.ns,4_000_005);self.assertEqual(len(sleeps),2)
        clock.ns=1
        with self.assertRaises(ValueError):real.now_ns()
    def test_native_bounds_and_dtypes_not_changed(self):
        t=np.zeros((2,23),np.float64);a=np.zeros((2,23),np.float32);f=np.arange(2,dtype=np.int64)
        t[0,0]=2.00000001
        with self.assertRaises(ValueError):CommandTable(t,a,f,['2'*64]*2,LIMITS)
        with self.assertRaises(ValueError):CommandTable(t.astype(np.float32),a,f,['2'*64]*2,LIMITS)
    def test_stub_worker_lifecycle_no_restart_and_join(self):
        class Event:
            def __init__(self):self.value=False
            def set(self):self.value=True
            def wait(self,timeout):return self.value
        class Process:
            pid=123;exitcode=0
            def __init__(self):self.live=False;self.closed=False
            def start(self):self.live=True
            def is_alive(self):return self.live
            def join(self,t):self.live=False
            def close(self):self.closed=True
            def terminate(self):self.live=False;self.exitcode=-15
        ready=Event();ready.set();stop=Event();p=Process();life=WorkerLifecycle(p,ready,stop,FakeClock())
        life.start_ready();self.assertTrue(life.stop_join()['normal_exit']);self.assertTrue(stop.value and p.closed)
        with self.assertRaises(ValueError):life.start_ready()
        self.assertIn('observed_ns',life.records[1]);self.assertIn('cleanup_finished_ns',life.records[-1])
    def test_owned_export_keeps_native373_and_hold_boundary(self):
        parts=setup();run_fake(*parts);arrays,metadata,capsules=export_owned(parts[-1])
        self.assertEqual(arrays['packed_capture'].shape,(40,373));self.assertEqual(arrays['control_integration'].shape,(4,291))
        self.assertTrue(all(metadata['hold_continuity'][k] for k in ['present','full291_matches_previous_step','history_matches_previous_control','incoming_matches_previous_actual_action']))
        arrays['packed_capture'][:]=123
        self.assertNotEqual(parts[-1].foundation.step_records.records()[0].captured.state,np.full(373,123,np.float64).tobytes())
        self.assertEqual(len(capsules['last_validated_capture_state']),373*8)
    def test_owned_export_preserves_partial_failed_step(self):
        parts=setup(strict_at=7);run_fake(*parts);arrays,metadata,capsules=export_owned(parts[-1])
        self.assertEqual(arrays['packed_capture'].shape,(7,373));self.assertEqual(arrays['step_verified'].tolist(),[True]*6+[False])
        self.assertEqual(metadata['summary']['foundation']['unexecuted'],33);self.assertFalse(metadata['hold_continuity']['present'])
    def test_held_result_locks_fail_command_coverage_without_waiting(self):
        parts=setup()
        for slot in parts[4].slots:slot.lock.held=True
        r=run_fake(*parts)
        self.assertEqual(r['foundation']['returned'],40);self.assertEqual(r['commands']['held_controls'],[1,2,3])
        self.assertFalse(r['component_preliminary_pass'])
        self.assertTrue(any(json.loads(e).get('status')=='BUSY' for e in parts[-1].transport.records()))
    def test_no_task_engines_imported(self):
        self.assertFalse(any(name.split('.')[0] in {'mujoco','torch','onnxruntime','mjbatch'} for name in sys.modules))
    def test_failed_start_does_not_join_or_mask_first_error(self):
        class FailedStart:
            def __init__(self):self.joins=0;self.closed=False
            def start(self):raise RuntimeError('original selected start failure')
            def join(self,timeout):self.joins+=1;raise AssertionError('unstarted process joined')
            def close(self):self.closed=True
        process=FailedStart();life=WorkerLifecycle(process,None,None,FakeClock())
        with self.assertRaisesRegex(RuntimeError,'original selected start failure'):life.start_ready()
        result=life.stop_join()
        self.assertEqual(process.joins,0);self.assertTrue(process.closed);self.assertFalse(result['process_absence_proven'])
        self.assertTrue(life.start_attempted);self.assertFalse(life.start_returned)
        self.assertIn('original selected start failure',result['first_error'])
        with self.assertRaises(ValueError):life.start_ready()
    def test_readiness_failure_after_returned_start_still_joins(self):
        class Event:
            def wait(self,timeout):return False
            def set(self):pass
        class Process:
            pid=321;exitcode=0
            def __init__(self):self.live=False;self.joined=False
            def start(self):self.live=True
            def is_alive(self):return self.live
            def join(self,timeout):self.joined=True;self.live=False
            def close(self):pass
        process=Process();life=WorkerLifecycle(process,Event(),Event(),FakeClock())
        with self.assertRaisesRegex(RuntimeError,'readiness'):life.start_ready()
        life.stop_join();self.assertTrue(process.joined and life.start_returned);self.assertFalse(life.ready_returned)

if __name__=='__main__':unittest.main()
