"""Create a fresh source-only fork; never import or execute runtime sources."""
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'independent_plant_clock_timeout_correction_v1/source_draft_v1'
SOURCE = BASE / 'source_draft_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


def main():
    SOURCE.mkdir(exist_ok=False)
    originals = {}
    for path in sorted(OLD.glob('*.py')):
        (SOURCE / path.name).write_bytes(path.read_bytes())
        originals[path.name] = sha(path)
    path = SOURCE / 'clock_core.py'
    old = path.read_text()
    text = change(old, 'from dataclasses import dataclass, asdict',
                  'from dataclasses import dataclass, asdict, replace')
    text = change(text, '@dataclass(frozen=True)\nclass Result:', '''@dataclass(frozen=True)
class PendingPublication:
    job: Job
    payload: bytes
    deadline_ns: int
    attempts: int = 0
    last_attempt_index: int = -1
    last_status: str | None = None
    event_recorded: bool = False

    def evidence(self):
        # A fresh JSON-compatible copy, never an alias to mutable plant storage.
        return dict(identity=self.job.identity(), payload=b64(self.payload),
                    deadline_ns=self.deadline_ns, attempts=self.attempts,
                    last_attempt_index=self.last_attempt_index,
                    last_status=self.last_status, event_recorded=self.event_recorded)


@dataclass(frozen=True)
class Result:''')
    text = change(text, '        self.published_jobs = set()', '''        self.published_jobs = set()
        self.pending_publication = None
        self.last_job_publication = None
        self.job_publication_attempted = self.job_publication_returned = 0
        self.job_publication_events = self.pending_expirations = 0''')
    text = change(text, '    def _boundary(self):', '''    def _attempt_job_publication(self, pending):
        # Called once at the original boundary, or once on a later eligible tick.
        index = self.returned
        if pending.last_attempt_index == index:
            return
        if pending.attempts >= CONTROL_STEPS:
            raise ValueError('fixed publication attempt bound exceeded')
        attempted = replace(pending, attempts=pending.attempts + 1,
                            last_attempt_index=index, last_status=None,
                            event_recorded=False)
        self.pending_publication = self.last_job_publication = attempted
        self.job_publication_attempted += 1
        # An exception may follow a transport write: retain an unresolved attempt,
        # let the existing failure path stop execution, and never retry it.
        status = self.jobs.try_publish(pending.job.activation, pending.payload)
        self.job_publication_returned += 1
        returned = replace(attempted, last_status=status)
        self.last_job_publication = returned
        self.pending_publication = returned if status == 'BUSY' else None
        if status == 'PUBLISHED':
            self.published_jobs.add(pending.job.activation)
        self._event('JOB_PUBLICATION', activation=pending.job.activation, status=status,
                    source_window=pending.job.window_id, input_digest=pending.job.input_digest,
                    attempt=returned.attempts, deadline_ns=pending.deadline_ns,
                    payload_sha256=digest(pending.payload))
        committed = replace(returned, event_recorded=True)
        self.last_job_publication = committed
        if status == 'BUSY':
            self.pending_publication = committed
        self.job_publication_events += 1

    def _expire_pending_publication(self, reason):
        pending = self.pending_publication
        if pending is None:
            return
        # Clear retry eligibility before logging; a full logger cannot cause reuse.
        self.pending_publication = None
        self.pending_expirations += 1
        self._event('PENDING_JOB_EXPIRED', activation=pending.job.activation,
                    expiration=reason, attempts=pending.attempts,
                    last_status=pending.last_status, deadline_ns=pending.deadline_ns,
                    last_attempt_index=pending.last_attempt_index,
                    payload_sha256=digest(pending.payload))

    def _retry_pending_publication(self):
        pending = self.pending_publication
        if pending is None or self.failure is not None or self.input_fault is not None:
            return
        index = self.returned
        if index % CONTROL_STEPS == 0 or pending.last_attempt_index == index:
            return
        if pending.last_status != 'BUSY':
            raise ValueError('only an unambiguous BUSY publication can be retried')
        if pending.job.activation in self.published_jobs or pending.job.activation in self.sealed:
            raise ValueError('published or sealed job cannot be retried')
        now = self._now_ns('PENDING_PUBLICATION_DEADLINE')
        if index >= pending.job.activation * CONTROL_STEPS or now >= pending.deadline_ns:
            self._expire_pending_publication('ORIGINAL_DEADLINE')
            return
        if index <= pending.job.snapshot_physics or index >= pending.job.snapshot_physics + CONTROL_STEPS:
            raise ValueError('retry outside original predecessor control')
        self.stage = 'PENDING_JOB_PUBLICATION'
        self._attempt_job_publication(pending)

    def _boundary(self):''')
    text = change(text, '        control = self.returned // CONTROL_STEPS\n        incoming = self.previous_raw', '''        control = self.returned // CONTROL_STEPS
        if self.pending_publication is not None:
            self._expire_pending_publication('ACTIVATION_BOUNDARY')
        incoming = self.previous_raw''')
    text = change(text, '''            publication = self.jobs.try_publish(control + 1, job.to_bytes())
            if publication == 'PUBLISHED':
                self.published_jobs.add(control + 1)
            self._event('JOB_PUBLICATION', activation=control + 1, status=publication,
                        source_window=job.window_id, input_digest=job.input_digest)''', '''            pending = PendingPublication(job, job.to_bytes(),
                                         self.deadline((control + 1) * CONTROL_STEPS))
            self._attempt_job_publication(pending)''')
    text = change(text, '''                self._boundary()
            self.stage = 'STEP_ATTEMPT'
''', '''                self._boundary()
            else:
                self._retry_pending_publication()
            self.stage = 'STEP_ATTEMPT'
''')
    text = change(text, '            history_entries=self.history.entries,', '''            history_entries=self.history.entries,
            pending_publication=None if self.pending_publication is None else self.pending_publication.evidence(),
            last_job_publication=None if self.last_job_publication is None else self.last_job_publication.evidence(),
            job_publication_attempted=self.job_publication_attempted,
            job_publication_returned=self.job_publication_returned,
            job_publication_events=self.job_publication_events,
            pending_expirations=self.pending_expirations,''')
    path.write_text(text, encoding='utf-8', newline='\n')
    (BASE / 'clock_core.diff').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), text.splitlines(True), fromfile='preserved/clock_core.py',
        tofile='source_draft_v1/clock_core.py')), encoding='utf-8')
    result = dict(preparation_only=True, actual_run_selected=False,
                  originals_sha256=originals,
                  source_sha256={p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
                  changed_original_files=['clock_core.py'],
                  unchanged_original_files=[name for name in originals if name != 'clock_core.py'],
                  diagnosis_receipt_sha256='bbbb8a9d38554ead77a32b9d224647775c517dac518125ebbad68837b6f18975',
                  native_steps=0, model_calls=0, spawned_workers=0, optimizer_updates=0)
    with (BASE / 'derivation.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
