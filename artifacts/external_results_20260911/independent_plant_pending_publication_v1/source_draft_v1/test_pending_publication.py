"""Deterministic fake transport, clocks and steppers only; no native/process calls."""
import json
import unittest
import numpy as np
from clock_core import FixedLedger, Result, encode
from test_scaffold import setup, worker_once, BINDING


class ScriptedPublisher:
    def __init__(self, actual, clock, outcomes):
        self.actual, self.clock, self.outcomes = actual, clock, list(outcomes)
        self.calls = []
        self.before_return = None

    def try_publish(self, key, payload):
        self.calls.append((key, payload, self.clock.ns))
        outcome = self.outcomes.pop(0) if self.outcomes else 'PUBLISHED'
        if outcome in ('PUBLISHED', 'WRITE_THEN_RAISE'):
            assert self.actual.try_publish(key, payload) == 'PUBLISHED'
        if self.before_return is not None:
            self.before_return(outcome)
        if outcome == 'WRITE_THEN_RAISE':
            raise RuntimeError('transport raised after shared write')
        return outcome


def configured(outcomes, **stepper_args):
    clock, stepper, commands, jobs, results, session = setup(n=3, **stepper_args)
    scripted = ScriptedPublisher(jobs, clock, outcomes)
    session.foundation.jobs.endpoint = scripted
    return clock, stepper, commands, jobs, results, session, scripted


def pump(parts):
    clock, _, commands, jobs, results, _, _ = parts
    return worker_once(jobs, results, BINDING, commands, clock, FixedLedger(100))


class PendingTests(unittest.TestCase):
    def test_busy_then_published_same_bytes_and_history_once(self):
        p = configured(['BUSY', 'PUBLISHED'])
        clock, stepper, _, _, _, session, publisher = p
        f = session.foundation
        self.assertTrue(session.tick_once())
        original = f.issued[1].to_bytes()
        incoming = f.previous_raw
        history = f.history.entries
        self.assertTrue(session.tick_once())
        self.assertEqual([c[1] for c in publisher.calls], [original, original])
        self.assertEqual(f.history.entries, history)
        self.assertEqual(f.previous_raw, incoming)
        self.assertEqual(len(f.control_records.records()), 1)
        self.assertEqual(f.pending_publication, None)
        pump(p)
        self.assertTrue(session.tick_once())
        self.assertIn(1, f.sealed)
        self.assertEqual(f.job_publication_attempted, 2)
        self.assertEqual(f.job_publication_returned, 2)
        self.assertEqual(f.job_publication_events, 2)
        self.assertEqual(f.epoch_ns, 100_000_000)
        self.assertEqual(stepper.returned, 3)

    def test_persistent_busy_ten_attempts_then_original_hold(self):
        p = configured(['BUSY'] * 20)
        f = p[5].foundation
        for _ in range(11):
            self.assertTrue(p[5].tick_once())
        self.assertEqual(len(p[6].calls), 10)
        self.assertEqual(len(set(c[1] for c in p[6].calls)), 1)
        events = [json.loads(v) for v in f.events.records()]
        attempts = [v for v in events if v['reason'] == 'JOB_PUBLICATION']
        self.assertEqual([v['physics'] for v in attempts], list(range(10)))
        self.assertEqual([v['attempt'] for v in attempts], list(range(1, 11)))
        self.assertEqual(f.input_fault['control'], 1)
        self.assertTrue(f.control_records.records()[1].held)
        self.assertEqual(f.control_records.records()[1].command, p[2].commands[0])
        self.assertEqual(f.pending_publication, None)
        self.assertEqual(f.pending_expirations, 1)
        self.assertEqual(len(f.issued), 1)

    def test_at_most_one_attempt_in_same_physics_tick(self):
        p = configured(['BUSY'] * 5)
        p[5].tick_once()
        f = p[5].foundation
        f._retry_pending_publication()
        f._retry_pending_publication()
        self.assertEqual(len(p[6].calls), 2)
        self.assertEqual(f.pending_publication.last_attempt_index, 1)

    def test_published_never_republished(self):
        p = configured(['BUSY', 'PUBLISHED'])
        for _ in range(9):
            p[5].tick_once()
        self.assertEqual(len(p[6].calls), 2)

    def test_deadline_equality_no_retry(self):
        p = configured(['BUSY'] * 4)
        p[5].tick_once()
        f = p[5].foundation
        p[0].ns = f.deadline(10)
        p[5].tick_once()
        self.assertEqual(len(p[6].calls), 1)
        self.assertEqual(f.pending_expirations, 1)
        self.assertIsNone(f.pending_publication)

    def test_published_after_deadline_never_sealed_or_activated(self):
        p = configured(['BUSY', 'PUBLISHED'])
        f = p[5].foundation
        p[5].tick_once()
        p[6].before_return = lambda status: setattr(p[0], 'ns', f.deadline(10) + 1)
        p[5].tick_once()
        self.assertIn(1, f.published_jobs)
        pump(p)
        p[5].tick_once()
        self.assertNotIn(1, f.sealed)
        for _ in range(8):
            p[5].tick_once()
        self.assertTrue(f.control_records.records()[1].held)
        self.assertEqual(f.active, p[2].commands[0])
        self.assertTrue(any(json.loads(v).get('rejection') == 'LATE' for v in f.events.records()))
        self.assertIsNotNone(f.first_deadline_failure)

    def test_scheduler_stall_after_retry_check_retains_both_times(self):
        p = configured(['BUSY', 'PUBLISHED'])
        f = p[5].foundation
        p[5].tick_once()
        original = f._attempt_job_publication
        def stalled(pending):
            p[0].ns = f.deadline(10) + 1
            original(pending)
        f._attempt_job_publication = stalled
        p[5].tick_once()
        event = [json.loads(v) for v in f.events.records() if json.loads(v)['reason']=='JOB_PUBLICATION'][-1]
        self.assertLess(event['retry_checked_ns'], f.deadline(10))
        self.assertGreater(p[6].calls[-1][2], f.deadline(10))
        pump(p)
        p[5].tick_once()
        self.assertNotIn(1, f.sealed)
        self.assertTrue(any(json.loads(v).get('rejection')=='LATE' for v in f.events.records()))

    def test_nonbusy_outcomes_end_retry_eligibility(self):
        for status in ('FULL', 'CORRUPT', 'WRONG_EPOCH', 'UNCOMMITTED', 'EMPTY'):
            with self.subTest(status=status):
                p = configured(['BUSY', status])
                for _ in range(11):
                    p[5].tick_once()
                f = p[5].foundation
                self.assertEqual(len(p[6].calls), 2)
                self.assertEqual(f.last_job_publication.last_status, status)
                self.assertIsNone(f.pending_publication)
                self.assertEqual(f.input_fault['control'], 1)

    def test_transport_exception_after_write_no_retry(self):
        p = configured(['BUSY', 'WRITE_THEN_RAISE'])
        p[5].tick_once()
        self.assertFalse(p[5].tick_once())
        f = p[5].foundation
        self.assertFalse(p[5].tick_once())
        self.assertEqual(len(p[6].calls), 2)
        self.assertEqual(f.job_publication_attempted, 2)
        self.assertEqual(f.job_publication_returned, 1)
        self.assertEqual(f.job_publication_events, 1)
        self.assertIsNone(f.pending_publication.last_status)
        self.assertNotIn(1, f.published_jobs)
        self.assertTrue(any(status == 'TAKEN' for _, status, _ in p[3].poll_once()))
        self.assertEqual(f.returned, 1)

    def test_event_logger_failure_preserves_published_vs_recorded(self):
        p = configured(['BUSY', 'PUBLISHED'])
        f = p[5].foundation
        f.events = FixedLedger(1)
        p[5].tick_once()
        self.assertFalse(p[5].tick_once())
        self.assertEqual(f.job_publication_attempted, 2)
        self.assertEqual(f.job_publication_returned, 2)
        self.assertEqual(f.job_publication_events, 1)
        self.assertEqual(f.last_job_publication.last_status, 'PUBLISHED')
        self.assertFalse(f.last_job_publication.event_recorded)
        self.assertIn(1, f.published_jobs)
        self.assertIsNone(f.pending_publication)
        self.assertEqual(f.returned, 1)
        self.assertEqual(json.loads(f.events.overflow_record)['attempt'], 2)

    def test_transport_logger_failure_preserves_ambiguous_attempt(self):
        p = configured(['BUSY', 'PUBLISHED'])
        f = p[5].foundation
        f.jobs.ledger = FixedLedger(1)
        p[5].tick_once()
        self.assertFalse(p[5].tick_once())
        self.assertEqual(f.job_publication_attempted, 2)
        self.assertEqual(f.job_publication_returned, 1)
        self.assertIsNone(f.last_job_publication.last_status)
        self.assertEqual(json.loads(f.jobs.ledger.overflow_record)['status'], 'PUBLISHED')
        self.assertFalse(p[5].tick_once())
        self.assertEqual(len(p[6].calls), 2)

    def test_native_failure_prevents_retry(self):
        p = configured(['BUSY'] * 4, strict_at=1)
        self.assertFalse(p[5].tick_once())
        self.assertFalse(p[5].tick_once())
        self.assertEqual(len(p[6].calls), 1)
        self.assertEqual(p[5].foundation.pending_publication.last_status, 'BUSY')
        self.assertEqual(p[5].foundation.returned, 1)

    def test_snapshot_signed_zero_and_history_are_literal(self):
        p = configured(['BUSY', 'PUBLISHED'])
        raw = np.zeros(23, np.float32)
        raw[::2] = -0.0
        f = p[5].foundation
        f.previous_raw = raw.tobytes()
        p[5].tick_once()
        original = f.issued[1]
        payload = original.to_bytes()
        before = f.history.entries
        p[5].tick_once()
        self.assertIs(f.issued[1], original)
        self.assertEqual(p[6].calls[0][1], p[6].calls[1][1])
        self.assertEqual(p[6].calls[1][1], payload)
        self.assertEqual(f.control_records.records()[0].incoming_raw, raw.tobytes())
        self.assertEqual(f.history.entries, before)

    def test_owned_summary_has_no_alias(self):
        p = configured(['BUSY'])
        p[5].tick_once()
        f = p[5].foundation
        summary = f.summary()
        summary['pending_publication']['identity']['binding']['run'] = 'mutated'
        summary['last_job_publication']['last_status'] = 'PUBLISHED'
        self.assertEqual(f.pending_publication.job.binding.run, BINDING.run)
        self.assertEqual(f.last_job_publication.last_status, 'BUSY')

    def test_fixed_capacity_bound(self):
        steps, controls = 18190, 1819
        jobs = controls - 1
        # Every result slot may generate one event; publications/expiries/holds
        # each have their separately bounded event count.
        transport = 2 * steps + 10 * jobs
        events = 2 * steps + 10 * jobs + jobs + (controls - 1)
        self.assertEqual(transport, 54560)
        self.assertEqual(events, 58196)
        self.assertLess(transport, 80000)
        self.assertLess(events, 80000)


if __name__ == '__main__':
    unittest.main()
