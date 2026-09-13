"""Active integration fakes; no task arrays, native/worker APIs or real clocks."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from clock_core import FixedLedger,owned_evidence
from evidence import export_owned
from native_stepper import NativeStepper,NativeAdapterError
from stage_watchdog import StageTimeout
from timing_hooks import TimingHooks,InstrumentationFault
from timing_probe import TimingProbe,SPAN_FIELDS
from timing_preservation import save_timing
from test_scaffold import setup,run_fake,table,Session,BINDING,LIMITS


class Readers:
    def __init__(self):self.value=0
    def read(self):self.value+=1;return self.value


def hooks(capacity=4096,cls=TimingProbe):
    reader=Readers()
    return TimingHooks(cls(wall_ns=reader.read,thread_cpu_ns=reader.read,
        process_cpu_ns=reader.read,thread_id=lambda:1,span_capacity=capacity,gc_capacity=16))


def attach(parts,h):
    parts[-1].timing=parts[-1].foundation.timing=parts[1].timing=h


def lossless(value):
    return json.loads(owned_evidence(value))


class ActiveEquivalence(unittest.TestCase):
    def compare(self,*,pump=True,same_targets=False,locks=False,**kw):
        outputs=[]
        for active in (False,True):
            parts=list(setup(**kw))
            if same_targets:
                parts[2]=table(same_targets=True)
                parts[-1]=Session(clock=parts[0],stepper=parts[1],epoch_ns=100_000_000,
                    binding=BINDING,table=parts[2],initial_previous_raw=np.zeros(23,np.float32).tobytes(),
                    limits=LIMITS,jobs=parts[3],results=parts[4],main_controls=3)
            if locks:
                for slot in parts[4].slots:slot.lock.held=True
            h=hooks()
            if active:attach(parts,h)
            result=run_fake(*parts,pump=pump)
            arrays,metadata,capsules=export_owned(parts[-1])
            outputs.append((result,arrays,metadata,capsules))
            if active:
                self.assertFalse(h.failed);self.assertTrue(h.probe.snapshot()['instrumentation_complete'])
                self.assertGreater(h.probe.spans_started,0)
        left,right=outputs
        self.assertEqual(left[0],right[0]);self.assertEqual(left[2:],right[2:])
        self.assertEqual(left[1].keys(),right[1].keys())
        for key,a in left[1].items():
            b=right[1][key]
            self.assertEqual((a.dtype,a.shape,a.tobytes()),(b.dtype,b.shape,b.tobytes()),key)
    def test_complete_main_and_continuous_hold(self):self.compare()
    def test_strict_failure(self):self.compare(strict_at=7)
    def test_native_api_exception(self):self.compare(raise_at=7)
    def test_hung_worker_hold(self):self.compare(pump=False)
    def test_equal_target_distinct_raw_history(self):self.compare(same_targets=True)
    def test_held_result_lock(self):self.compare(locks=True)
    def test_logger_failure_preserves_capture(self):
        parts=setup();h=hooks();attach(parts,h);parts[-1].outer=FixedLedger(0)
        self.assertFalse(parts[-1].tick_once())
        self.assertEqual((parts[1].returned,parts[-1].foundation.captured),(1,1))
        self.assertIsNotNone(parts[-1].outer.overflow_record)
        self.assertFalse(parts[-1].tick_once());self.assertEqual(parts[1].returned,1)


class HookFailures(unittest.TestCase):
    def test_overflow_before_native_has_no_native_attempt(self):
        parts=setup();h=hooks(1);attach(parts,h)
        self.assertFalse(parts[-1].tick_once());self.assertTrue(h.failed)
        self.assertEqual((parts[1].attempted,parts[-1].foundation.attempted),(0,0))
    def test_capture_end_interrupt_retains_owned_capture(self):
        class Interrupt(TimingProbe):
            def end(self,token):
                if self.spans[token*len(SPAN_FIELDS)]==10:raise StageTimeout('capture end injected')
                return super().end(token)
        parts=setup();h=hooks(cls=Interrupt);attach(parts,h)
        with self.assertRaises(StageTimeout):parts[-1].tick_once()
        self.assertEqual((parts[1].returned,parts[1].captured,parts[-1].foundation.captured),(1,1,1))
        self.assertIsNotNone(parts[-1].foundation.last_capture)
        self.assertFalse(h.probe.snapshot()['instrumentation_complete'])
    def test_verifier_end_interrupt_retains_verified_credit(self):
        class Interrupt(TimingProbe):
            def end(self,token):
                if self.spans[token*len(SPAN_FIELDS)]==12:raise StageTimeout('verify end injected')
                return super().end(token)
        parts=setup();h=hooks(cls=Interrupt);attach(parts,h)
        with self.assertRaises(StageTimeout):parts[-1].tick_once()
        self.assertEqual((parts[1].verified,parts[-1].foundation.verified),(1,1))
        self.assertIsNotNone(parts[-1].foundation.last_verifier_return)
    def test_normal_end_clock_exception_latches_without_masking_body(self):
        class Broken(TimingProbe):
            def end(self,token):raise RuntimeError('injected clock error')
        h=hooks(cls=Broken)
        with self.assertRaisesRegex(ValueError,'measured first'):
            with h.span(1,0,0):raise ValueError('measured first')
        self.assertTrue(h.failed);self.assertEqual(h.depth,0)
    def test_fatal_end_does_not_replace_original_measured_exception(self):
        class Broken(TimingProbe):
            def end(self,token):raise StageTimeout('secondary deadline')
        h=hooks(cls=Broken)
        with self.assertRaisesRegex(ValueError,'measured first'):
            with h.span(1,0,0):raise ValueError('measured first')
        self.assertTrue(h.failed);self.assertEqual(h.depth,0)
    def test_fatal_begin_propagates_without_body(self):
        class Broken(TimingProbe):
            def begin(self,*args):raise StageTimeout('begin deadline')
        h=hooks(cls=Broken);body=[]
        with self.assertRaises(StageTimeout):
            with h.span(1,0,0):body.append(1)
        self.assertEqual(body,[]);self.assertEqual(h.depth,0)


def adapter(h,raises=False):
    a=NativeStepper.__new__(NativeStepper)
    a.timing=h;a.initialized=True;a.failure=None;a.closed=False
    a.returned=a.attempted=a.verified=a.verification_attempts=0
    a.expected_time=0.;a.stage='READY';a.command=a.target=a.last_capture=a.last_capture_return=None
    a.model=object();a.limits=LIMITS.copy()
    a.contract={'kp':np.full(23,30.),'kd':np.full(23,2.),'native_effort':np.full(23,3.)}
    a.data=SimpleNamespace(qpos=np.arange(30,dtype=np.float64)/100,
        qvel=np.arange(29,dtype=np.float64)/200,ctrl=np.zeros(23,np.float64),
        qfrc_applied=np.zeros(29),xfrc_applied=np.zeros((25,6)))
    calls=[]
    def step(model,data):
        calls.append(data.ctrl.copy())
        if raises:raise RuntimeError('fake native raised')
    a.api=SimpleNamespace(mj_step=step)
    a._fail=lambda reason,exc=None,evidence=None:setattr(a,'failure',(reason,str(exc)))
    return a,calls


class NativeMethodFakes(unittest.TestCase):
    def test_exact_pd_and_return_counter(self):
        output=[]
        for h in (TimingHooks(),hooks()):
            a,calls=adapter(h);target=np.linspace(-1.,1.,23,dtype=np.float64)
            expected=np.minimum(np.maximum(30*(target-a.data.qpos[7:])-2*a.data.qvel[6:],-3),3)
            a.step(target)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0].tobytes(),expected.tobytes())
            output.append((a.target,a.command,a.data.ctrl.tobytes(),a.attempted,a.returned,a.expected_time,a.stage))
        self.assertEqual(*output)
    def test_native_exception_keeps_attempt_return_gap(self):
        a,calls=adapter(hooks(),True)
        with self.assertRaisesRegex(NativeAdapterError,'fake native raised'):a.step(np.zeros(23))
        self.assertEqual((a.attempted,a.returned,len(calls)),(1,0,1))
    def test_native_end_interrupt_keeps_returned_counter(self):
        class Interrupt(TimingProbe):
            def end(self,token):
                if self.spans[token*len(SPAN_FIELDS)]==9:raise StageTimeout('native end deadline')
                return super().end(token)
        a,calls=adapter(hooks(cls=Interrupt))
        with self.assertRaises(StageTimeout):a.step(np.zeros(23))
        self.assertEqual((a.attempted,a.returned,len(calls),a.expected_time),(1,1,1,.002))
        self.assertEqual(a.stage,'RETURNED_UNCAPTURED')
    def test_failed_pd_probe_prevents_native(self):
        a,calls=adapter(hooks(1))
        with self.assertRaises(NativeAdapterError):a.step(np.zeros(23))
        self.assertEqual((a.attempted,a.returned,len(calls)),(0,0,0))


class SidecarPreservation(unittest.TestCase):
    def test_complete_owned_bytes_and_no_extra_native_read(self):
        parts=setup();h=hooks();attach(parts,h);run_fake(*parts)
        before=parts[1].counters().copy()
        with tempfile.TemporaryDirectory() as d:
            result=save_timing(d,h,parts[1],parts[-1].foundation,lossless)
            self.assertTrue(result['complete']);self.assertEqual(len(result['files']),4)
            self.assertEqual((Path(d)/'timing_spans.bin').read_bytes(),h.probe.snapshot()['spans'])
        self.assertEqual(before,parts[1].counters())
    def test_incomplete_span_and_overflow_preserved(self):
        h=hooks(1);h.probe.begin(1,0,0);h.probe.begin(2,0,0)
        with tempfile.TemporaryDirectory() as d:
            result=save_timing(d,h,None,None,lossless)
            self.assertFalse(result['complete']);self.assertEqual(len(result['files']),4)
            meta=json.loads((Path(d)/'timing_metadata.json').read_text())
            self.assertEqual((meta['active_depth'],meta['spans_denied']),(1,1))
    def test_sidecar_collision_preserves_main_and_other_sidecars(self):
        h=hooks()
        with tempfile.TemporaryDirectory() as d:
            main=Path(d)/'captured_trace.npz';main.write_bytes(b'existing main evidence')
            existing=Path(d)/'timing_spans.bin';existing.write_bytes(b'prior partial')
            result=save_timing(d,h,None,None,lossless)
            self.assertFalse(result['complete']);self.assertEqual(len(result['errors']),1)
            self.assertEqual(main.read_bytes(),b'existing main evidence')
            self.assertEqual(existing.read_bytes(),b'prior partial')
            self.assertTrue((Path(d)/'timing_owned_partial.json').exists())
    def test_one_bad_owned_field_does_not_drop_other_fields(self):
        obj=SimpleNamespace(last_capture=b'captured',last_assessment=b'assessed',failure=b'bad')
        def convert(v):
            if v==b'bad':raise ValueError('injected conversion failure')
            return lossless(v)
        with tempfile.TemporaryDirectory() as d:
            result=save_timing(d,hooks(),obj,None,convert)
            saved=json.loads((Path(d)/'timing_owned_partial.json').read_text())
            self.assertFalse(result['complete']);self.assertEqual(saved['adapter']['last_capture'],lossless(b'captured'))
            self.assertEqual(saved['adapter']['last_assessment'],lossless(b'assessed'))


if __name__=='__main__':unittest.main()
