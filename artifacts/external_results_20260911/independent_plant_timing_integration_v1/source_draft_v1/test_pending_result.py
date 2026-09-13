"""Deterministic tiny fake inputs/locks only; no process, model or native calls."""
import dataclasses
import json
import unittest
from unittest.mock import patch
from clock_core import FixedLedger,Result,encode,CapacityFailure
from mailbox import Publication
from pending_result import PendingResultWorker,WorkerResultFailure,MAX_RESULT_ATTEMPTS
from recorded_protocol import decode_job
from test_scaffold import setup,BINDING


class Jobs:
    def __init__(self,actual):self.actual=actual;self.calls=0
    def poll_once(self):self.calls+=1;return self.actual.poll_once()


class Results:
    def __init__(self,actual,clock,statuses):
        self.actual,self.clock,self.statuses=actual,clock,list(statuses)
        self.calls=[];self.after=None
    def try_publish(self,key,payload):
        self.calls.append((key,payload,self.clock.ns))
        value=self.statuses.pop(0) if self.statuses else 'PUBLISHED'
        if value in ('PUBLISHED','WRITE_THEN_RAISE'):
            assert self.actual.try_publish(key,payload)=='PUBLISHED'
        if self.after is not None:self.after(value)
        if value=='WRITE_THEN_RAISE':raise RuntimeError('after committed write')
        return value


def fixture(statuses=('BUSY','PUBLISHED'),ledger=None):
    clock,stepper,table,jobs,results,session=setup(n=4)
    assert session.tick_once()
    jobs=Jobs(jobs);published=Results(results,clock,statuses)
    ledger=FixedLedger(1000) if ledger is None else ledger
    state=PendingResultWorker(BINDING,table,clock,ledger)
    return clock,stepper,table,jobs,published,session,state,ledger


def events(ledger,reason):return [json.loads(x) for x in ledger.records() if json.loads(x).get('reason')==reason]


class ResultTests(unittest.TestCase):
    def test_deadline_literal_is_plant_formula_not_created_plus_period(self):
        p=fixture();f=p[5].foundation;job=f.issued[1]
        self.assertEqual(job.deadline_ns,f.epoch_ns+20_000_000)
        changed=dataclasses.replace(job,created_ns=job.created_ns+5_000_000)
        parsed=decode_job(changed.to_bytes(),BINDING,p[2])
        self.assertNotEqual(parsed.deadline_ns,parsed.created_ns+20_000_000)
        self.assertEqual(parsed.deadline_ns,job.deadline_ns)

    def test_deadline_part_of_canonical_identity_and_native_admission(self):
        p=fixture();job=p[5].foundation.issued[1]
        for value in (True,-1,1.5):
            altered=json.loads(job.to_bytes());altered['deadline_ns']=value
            with self.assertRaises(ValueError):decode_job(encode(altered),BINDING,p[2])
        altered=json.loads(job.to_bytes());del altered['deadline_ns']
        with self.assertRaises(ValueError):decode_job(encode(altered),BINDING,p[2])
        forged=dataclasses.replace(job,deadline_ns=job.deadline_ns+1)
        p[4].actual.try_publish(1,Result.for_job(forged,p[2].commands[1],p[0].ns).to_bytes())
        p[5].foundation.poll_results()
        self.assertEqual(events(p[5].foundation.events,'RESULT_REJECTED')[-1]['rejection'],'WRONG_ORIGINAL_DEADLINE')
        self.assertNotIn(1,p[5].foundation.sealed)

    def test_busy_then_success_one_reply_identical_bytes_no_new_poll(self):
        clock,stepper,table,jobs,out,session,state,ledger=fixture()
        before=session.foundation.history.entries;state.tick(jobs,out);saved=state.pending
        clock.ns+=1_000_000;state.tick(jobs,out)
        self.assertEqual(jobs.calls,1);self.assertEqual(out.calls[0][1],out.calls[1][1])
        self.assertEqual(state.counts['replies_attempted'],1);self.assertEqual(state.counts['replies_completed'],1)
        self.assertEqual(state.counts['publications_attempted'],2);self.assertEqual(state.counts['publications_returned'],2)
        self.assertEqual(state.counts['publications_published'],1);self.assertEqual(state.counts['publications_BUSY'],1)
        self.assertEqual(state.counts['terminals'],1);self.assertIsNone(state.pending)
        self.assertEqual(saved.payload,state.last_result.payload);self.assertEqual(saved.completed_ns,state.last_result.completed_ns)
        self.assertEqual(session.foundation.history.entries,before);self.assertEqual(stepper.returned,1)
        session.foundation.poll_results();self.assertIn(1,session.foundation.sealed)
        self.assertEqual(len(events(ledger,'WORKER_RESULT_READY')),1)
        self.assertEqual(len(events(ledger,'WORKER_RESULT_PUBLICATION')),2)

    def test_twenty_busy_attempts_terminal_once_before_deadline(self):
        p=fixture(['BUSY']*30);state=p[6]
        for _ in range(MAX_RESULT_ATTEMPTS-1):state.tick(p[3],p[4]);p[0].ns+=100
        with self.assertRaisesRegex(WorkerResultFailure,'ATTEMPT_LIMIT'):state.tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),20);self.assertEqual(p[3].calls,1)
        self.assertEqual(len(set(x[1] for x in p[4].calls)),1)
        self.assertEqual(state.counts['terminals'],1);self.assertEqual(state.counts['terminal_events'],1)
        state.finish();state.finish();self.assertEqual(state.counts['terminals'],1)
        with self.assertRaises(WorkerResultFailure):state.tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),20)

    def test_expired_before_reply_no_compute_or_publish(self):
        p=fixture();p[0].ns=p[5].foundation.issued[1].deadline_ns
        with self.assertRaisesRegex(WorkerResultFailure,'EXPIRED_BEFORE_REPLY'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].counts['replies_attempted'],0);self.assertEqual(p[4].calls,[])
        self.assertEqual(p[6].pending.job.deadline_ns,p[0].ns)

    def test_expired_pending_no_second_publish(self):
        p=fixture();p[6].tick(p[3],p[4]);p[0].ns=p[6].pending.job.deadline_ns
        with self.assertRaisesRegex(WorkerResultFailure,'ORIGINAL_DEADLINE'):p[6].tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),1);self.assertEqual(p[6].counts['replies_completed'],1)

    def test_expiry_during_serialization_no_publish(self):
        import pending_result as module
        p=fixture();original=module.reply_to_job
        def delayed(*args):
            payload=original(*args);p[0].ns=p[5].foundation.issued[1].deadline_ns;return payload
        with patch.object(module,'reply_to_job',delayed):
            with self.assertRaisesRegex(WorkerResultFailure,'ORIGINAL_DEADLINE'):p[6].tick(p[3],p[4])
        self.assertIsNotNone(p[6].pending.payload);self.assertEqual(p[6].counts['replies_completed'],1)
        self.assertEqual(len(p[4].calls),0)

    def test_published_return_after_deadline_remains_late_at_plant(self):
        p=fixture(['PUBLISHED']);deadline=p[5].foundation.issued[1].deadline_ns
        p[4].after=lambda status:setattr(p[0],'ns',deadline+1)
        with self.assertRaisesRegex(WorkerResultFailure,'PUBLISHED_LATE'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].counts['publications_published'],1);self.assertEqual(p[6].pending.last_status,'PUBLISHED')
        p[5].foundation.poll_results();self.assertNotIn(1,p[5].foundation.sealed)
        self.assertEqual(events(p[5].foundation.events,'RESULT_REJECTED')[-1]['rejection'],'LATE')
        with self.assertRaises(WorkerResultFailure):p[6].tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),1)

    def test_busy_return_after_deadline_no_retry(self):
        p=fixture();p[4].after=lambda status:setattr(p[0],'ns',p[5].foundation.issued[1].deadline_ns)
        with self.assertRaisesRegex(WorkerResultFailure,'ORIGINAL_DEADLINE'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].pending.last_status,'BUSY');self.assertEqual(len(p[4].calls),1)

    def test_write_then_raise_keeps_unknown_return_and_bytes(self):
        p=fixture(['WRITE_THEN_RAISE'])
        with self.assertRaisesRegex(RuntimeError,'after committed write'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].counts['publications_attempted'],1);self.assertEqual(p[6].counts['publications_returned'],0)
        self.assertIsNone(p[6].pending.last_status);self.assertFalse(p[6].pending.call_returned)
        self.assertEqual(p[6].pending.payload,p[4].calls[0][1])
        with self.assertRaises(WorkerResultFailure):p[6].tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),1)

    def test_observer_log_failure_after_write_still_unknown_to_worker(self):
        from session import ObservedEndpoint
        p=fixture(['PUBLISHED']);transport=FixedLedger(0)
        observed=ObservedEndpoint(p[4],p[0],transport,'worker-results')
        with self.assertRaises(CapacityFailure):p[6].tick(p[3],observed)
        self.assertEqual(p[6].counts['publications_attempted'],1)
        self.assertEqual(p[6].counts['publications_returned'],0)
        self.assertIsNone(p[6].pending.last_status)
        self.assertEqual(json.loads(transport.overflow_record)['status'],'PUBLISHED')
        with self.assertRaises(WorkerResultFailure):p[6].tick(p[3],observed)
        self.assertEqual(len(p[4].calls),1)

    def test_reply_exception_retains_inputs_and_attempt_count(self):
        p=fixture()
        with patch('pending_result.reply_to_job',side_effect=RuntimeError('reply failure')):
            with self.assertRaisesRegex(RuntimeError,'reply failure'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].counts['replies_attempted'],1);self.assertEqual(p[6].counts['replies_completed'],0)
        self.assertEqual(p[6].pending.source.payload,p[5].foundation.issued[1].to_bytes())
        self.assertIsNotNone(p[6].pending.job);self.assertIsNotNone(p[6].pending.completed_ns)
        self.assertIsNone(p[6].pending.payload);self.assertEqual(p[4].calls,[])

    def test_full_corrupt_and_other_nonbusy_statuses_terminal(self):
        for status in ('FULL','CORRUPT','UNCOMMITTED','CLOSED','WRONG_EPOCH','BAD_KEY'):
            with self.subTest(status=status):
                p=fixture([status])
                with self.assertRaisesRegex(WorkerResultFailure,'PUBLICATION_'+status):p[6].tick(p[3],p[4])
                self.assertEqual(p[6].counts['publications_returned'],1)
                with self.assertRaises(WorkerResultFailure):p[6].tick(p[3],p[4])
                self.assertEqual(len(p[4].calls),1)

    def test_attempt_event_overflow_retains_known_return_no_retry(self):
        class FailAttempt(FixedLedger):
            def append(self,record):
                if json.loads(record)['reason']=='WORKER_RESULT_PUBLICATION':raise CapacityFailure('injected attempt log full')
                super().append(record)
        p=fixture(['PUBLISHED'],FailAttempt(100))
        with self.assertRaisesRegex(CapacityFailure,'attempt log'):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].counts['publications_returned'],1);self.assertEqual(p[6].counts['publications_published'],1)
        self.assertEqual(p[6].counts['publication_events'],0);self.assertFalse(p[6].pending.attempt_event_recorded)
        self.assertTrue(p[6].pending.call_returned);self.assertIsNotNone(p[6].failure)

    def test_ready_event_overflow_keeps_full_result_without_attempt(self):
        p=fixture(ledger=FixedLedger(0))
        with self.assertRaises(CapacityFailure):p[6].tick(p[3],p[4])
        self.assertIsNotNone(p[6].pending.payload);self.assertFalse(p[6].pending.ready_event_recorded)
        self.assertEqual(p[6].counts['replies_completed'],1);self.assertEqual(p[4].calls,[])
        self.assertEqual(p[6].counts['terminals'],1)
        self.assertFalse(p[6].pending.terminal_event_recorded)

    def test_terminal_event_overflow_keeps_terminal_and_published_credit(self):
        p=fixture(['PUBLISHED'],FixedLedger(2))
        with self.assertRaises(CapacityFailure):p[6].tick(p[3],p[4])
        self.assertEqual(p[6].pending.terminal_reason,'PUBLISHED');self.assertFalse(p[6].pending.terminal_event_recorded)
        self.assertEqual(p[6].counts['publications_published'],1);self.assertEqual(p[6].counts['terminals'],1)
        with self.assertRaises(WorkerResultFailure):p[6].tick(p[3],p[4])
        self.assertEqual(len(p[4].calls),1)

    def test_clock_failure_after_return_preserves_status(self):
        p=fixture(['PUBLISHED']);p[4].after=lambda status:setattr(p[0],'ns',1)
        with self.assertRaisesRegex(ValueError,'monotonic'):p[6].tick(p[3],p[4])
        self.assertTrue(p[6].pending.call_returned);self.assertEqual(p[6].pending.last_status,'PUBLISHED')
        self.assertIsNone(p[6].pending.attempt_end_ns);self.assertEqual(p[6].counts['publications_returned'],1)

    def test_two_taken_batch_retained_and_no_new_computation_while_busy(self):
        p=fixture();source=p[5].foundation.issued[1].to_bytes()
        class Two:
            calls=0
            def poll_once(self):
                self.calls+=1
                return ((0,'TAKEN',Publication(1,1,source)),(1,'TAKEN',Publication(1,2,source)))
        jobs=Two();p[6].tick(jobs,p[4])
        self.assertEqual(len(p[6].buffered),1);self.assertEqual(p[6].counts['jobs_taken'],2)
        self.assertEqual(p[6].counts['replies_completed'],1)
        p[0].ns+=1_000_000;p[6].tick(jobs,p[4])
        self.assertEqual(jobs.calls,1);self.assertEqual(p[6].counts['replies_completed'],1)
        with self.assertRaisesRegex(WorkerResultFailure,'DUPLICATE'):p[6].tick(jobs,p[4])
        self.assertEqual(jobs.calls,1);self.assertEqual(p[6].counts['replies_completed'],1)

    def test_poll_corrupt_preserves_other_taken_job_without_compute(self):
        p=fixture();source=p[5].foundation.issued[1].to_bytes()
        class Mixed:
            def poll_once(self):return ((0,'TAKEN',Publication(1,1,source)),(1,'CORRUPT',None))
        with self.assertRaisesRegex(WorkerResultFailure,'JOB_POLL_TERMINAL'):p[6].tick(Mixed(),p[4])
        self.assertEqual(p[6].buffered[0].payload,source);self.assertEqual(p[6].counts['replies_attempted'],0)

    def test_stop_pending_preserves_one_terminal_no_extra_call(self):
        p=fixture();p[6].tick(p[3],p[4]);saved=p[6].pending.payload;p[6].finish()
        self.assertEqual(p[6].pending.payload,saved);self.assertEqual(p[6].pending.terminal_reason,'STOP_WITH_OWNED_WORK')
        self.assertEqual(p[6].counts['terminals'],1);self.assertEqual(len(p[4].calls),1)
        p[6].finish();self.assertEqual(p[6].counts['terminals'],1)

    def test_evidence_is_owned_and_returned_status_is_preserved(self):
        p=fixture([{'unexpected':[]}])
        with self.assertRaises(ValueError):p[6].tick(p[3],p[4])
        first=p[6].evidence();first['counts']['publications_attempted']=999
        self.assertEqual(p[6].counts['publications_attempted'],1)
        self.assertTrue(p[6].pending.call_returned);self.assertIn('unexpected',p[6].pending.returned_status_evidence.decode())

    def test_invalid_clock_before_attempt_stops_without_publish(self):
        for value in (-1,True,float('nan')):
            with self.subTest(value=value):
                p=fixture();p[0].ns=value
                with self.assertRaises(ValueError):p[6].tick(p[3],p[4])
                self.assertEqual(p[4].calls,[]);self.assertEqual(p[6].counts['publications_attempted'],0)
                self.assertIsNotNone(p[6].clock_failure)


if __name__=='__main__':unittest.main()
