"""Bounded result-publication state machine. Pure injected clocks/endpoints only.

One nonblocking publish per worker iteration, at most twenty per result. Only an
unambiguous BUSY return permits another attempt. No model, plant, sleep, or I/O.
"""
from dataclasses import dataclass,replace
import json
from clock_core import Job,b64,digest,encode,owned_evidence
from mailbox import Publication
from recorded_protocol import decode_job,reply_to_job

MAX_RESULT_ATTEMPTS=20
MAX_BUFFERED_JOBS=2


class WorkerResultFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class OwnedJob:
    slot:int
    key:int
    version:int
    payload:bytes

    def evidence(self):
        return dict(slot=self.slot,key=self.key,version=self.version,
                    payload=b64(self.payload),payload_sha256=digest(self.payload))


@dataclass(frozen=True)
class PendingResult:
    source:OwnedJob
    received_ns:int
    job:Job|None=None
    completed_ns:int|None=None  # Preserves the old pre-reply-construction stamp.
    payload:bytes|None=None
    payload_ready_ns:int|None=None
    ready_event_recorded:bool=False
    attempts:int=0
    last_iteration:int|None=None
    attempt_start_ns:int|None=None
    attempt_end_ns:int|None=None
    last_status:str|None=None
    returned_status_evidence:bytes|None=None
    call_returned:bool=False
    attempt_event_recorded:bool=False
    terminal_reason:str|None=None
    terminal_ns:int|None=None
    terminal_event_recorded:bool=False

    def evidence(self):
        return dict(source=self.source.evidence(),received_ns=self.received_ns,
            identity=None if self.job is None else self.job.identity(),completed_ns=self.completed_ns,
            payload=None if self.payload is None else b64(self.payload),
            payload_sha256=None if self.payload is None else digest(self.payload),
            payload_ready_ns=self.payload_ready_ns,ready_event_recorded=self.ready_event_recorded,
            attempts=self.attempts,last_iteration=self.last_iteration,attempt_start_ns=self.attempt_start_ns,
            attempt_end_ns=self.attempt_end_ns,last_status=self.last_status,call_returned=self.call_returned,
            returned_status_evidence=None if self.returned_status_evidence is None else self.returned_status_evidence.decode(),
            attempt_event_recorded=self.attempt_event_recorded,terminal_reason=self.terminal_reason,
            terminal_ns=self.terminal_ns,terminal_event_recorded=self.terminal_event_recorded)


class PendingResultWorker:
    def __init__(self,binding,table,clock,ledger):
        self.binding,self.table,self.clock,self.ledger=binding,table,clock,ledger
        self.pending=None;self.last_result=None;self.buffered=();self.seen=set()
        self.last_poll_return=None;self.failure=None;self.clock_failure=None;self.last_ns=None;self.closed=False
        self.phase='READY'
        self.counts=dict(iterations_started=0,iterations_completed=0,polls_attempted=0,polls_returned=0,
            jobs_taken=0,jobs_decoded=0,replies_attempted=0,replies_completed=0,ready_events=0,
            publications_attempted=0,publications_returned=0,publications_published=0,
            publications_BUSY=0,publication_events=0,terminals=0,terminal_events=0)

    def _now(self):
        now=self.clock.now_ns()
        if type(now) is not int or now<0 or (self.last_ns is not None and now<self.last_ns):
            if self.clock_failure is None:
                self.clock_failure=encode(dict(previous_ns=self.last_ns,observed=owned_evidence(now).decode()))
            raise ValueError('worker monotonic nonnegative integer clock required')
        self.last_ns=now
        return now

    def _set(self,value):
        self.pending=value;self.last_result=value

    def _event(self,reason,**fields):
        self.ledger.append(encode(dict(reason=reason,**fields)))

    def _terminal(self,reason):
        p=self.pending
        if p is None or p.terminal_reason is not None:
            raise ValueError('exactly one terminal per owned job')
        # State/counters precede logging so an overflow never restores eligibility.
        p=replace(p,terminal_reason=reason,terminal_ns=self.last_ns)
        self._set(p);self.counts['terminals']+=1
        self._event('WORKER_RESULT_TERMINAL',activation=p.source.key,termination=reason,
            observed_ns=self.last_ns,deadline_ns=None if p.job is None else p.job.deadline_ns,
            attempts=p.attempts,last_status=p.last_status,call_returned=p.call_returned,
            job_sha256=digest(p.source.payload),result_sha256=None if p.payload is None else digest(p.payload))
        p=replace(p,terminal_event_recorded=True)
        self._set(p);self.counts['terminal_events']+=1
        if reason=='PUBLISHED':self.pending=None
        else:raise WorkerResultFailure(reason)

    def abort(self,reason,detail):
        """Best-effort in-memory preservation; never masks the first exception."""
        if self.failure is None:
            self.failure=encode(dict(reason=reason,phase=self.phase,detail=detail))
        if self.pending is not None and self.pending.terminal_reason is None:
            try:self._terminal(reason)
            except Exception:pass  # State is already terminal even if logger failed.

    def _poll(self,jobs):
        if self.pending is not None or self.buffered:raise ValueError('pending/buffered work blocks new-job poll')
        self.phase='JOB_POLL';self.counts['polls_attempted']+=1
        items=jobs.poll_once();self.counts['polls_returned']+=1
        self.last_poll_return=owned_evidence(items)
        if type(items) is not tuple or len(items)>MAX_BUFFERED_JOBS:
            raise ValueError('fixed two-slot poll return required')
        statuses=[];slots=[]
        for item in items:
            if type(item) is not tuple or len(item)!=3:raise ValueError('typed mailbox poll item')
            slot,status,publication=item
            if type(slot) is not int or not 0<=slot<MAX_BUFFERED_JOBS or type(status) is not str:
                raise ValueError('typed mailbox slot/status')
            if status=='TAKEN':
                if type(publication) is not Publication or type(publication.key) is not int or publication.key<0 or type(publication.version) is not int or publication.version<=0 or type(publication.payload) is not bytes or not 0<len(publication.payload)<=32768:
                    raise ValueError('bounded owned publication required')
                owned=OwnedJob(slot,publication.key,publication.version,publication.payload)
                self.buffered=(*self.buffered,owned);self.counts['jobs_taken']+=1
            elif publication is not None:raise ValueError('non-TAKEN item cannot own publication')
            statuses.append(status)
            slots.append(slot)
        # Retain every already-taken immutable item before rejecting other slots.
        if len(set(slots))!=len(slots) or any(s not in ('EMPTY','BUSY','TAKEN') for s in statuses):
            raise WorkerResultFailure('JOB_POLL_TERMINAL_STATUS')

    def _prepare(self):
        self.phase='JOB_DECODE'
        source=self.buffered[0];received=self._now()
        self._set(PendingResult(source,received));self.buffered=self.buffered[1:]
        job=decode_job(source.payload,self.binding,self.table)
        self._set(replace(self.pending,job=job));self.counts['jobs_decoded']+=1
        if source.key!=job.activation or source.key in self.seen:
            raise WorkerResultFailure('WRONG_OR_DUPLICATE_JOB_KEY')
        self.seen.add(source.key)
        now=self._now()
        if now>=job.deadline_ns:self._terminal('EXPIRED_BEFORE_REPLY')
        # Keep the original completion stamp semantics; a second stamp makes the
        # end of validation/serialization explicit without changing result bytes.
        self.phase='REPLY_CONSTRUCTION'
        self._set(replace(self.pending,completed_ns=now));self.counts['replies_attempted']+=1
        payload=reply_to_job(source.payload,self.binding,self.table,now)
        self._set(replace(self.pending,payload=payload));self.counts['replies_completed']+=1
        ready=self._now();self._set(replace(self.pending,payload_ready_ns=ready))
        self._event('WORKER_RESULT_READY',activation=job.activation,identity=job.identity(),
            received_ns=received,completed_ns=now,payload_ready_ns=ready,
            job_sha256=digest(source.payload),result_sha256=digest(payload),payload=b64(payload))
        self._set(replace(self.pending,ready_event_recorded=True));self.counts['ready_events']+=1

    def _publish(self,results,iteration):
        p=self.pending
        if p.terminal_reason is not None:raise WorkerResultFailure('terminal work cannot publish')
        if p.attempts and (not p.call_returned or p.last_status!='BUSY' or not p.attempt_event_recorded):
            raise WorkerResultFailure('only fully recorded unambiguous BUSY may retry')
        if p.last_iteration==iteration:raise WorkerResultFailure('one publication per worker iteration')
        start=self._now()
        if start>=p.job.deadline_ns:self._terminal('ORIGINAL_DEADLINE')
        if p.attempts>=MAX_RESULT_ATTEMPTS:self._terminal('ATTEMPT_LIMIT')
        self.phase='RESULT_PUBLICATION'
        p=replace(p,attempts=p.attempts+1,last_iteration=iteration,attempt_start_ns=start,
            attempt_end_ns=None,last_status=None,returned_status_evidence=None,call_returned=False,attempt_event_recorded=False)
        self._set(p);self.counts['publications_attempted']+=1
        status=results.try_publish(p.job.activation,p.payload)
        # Capture the returned value before asking for a timestamp or logging.
        # If the observed endpoint itself throws after writing, no returned credit
        # is claimed and abort retains the ambiguous attempted bytes.
        self.counts['publications_returned']+=1
        p=replace(p,call_returned=True,returned_status_evidence=owned_evidence(status));self._set(p)
        if type(status) is not str:
            raise ValueError('literal publication status required')
        p=replace(p,last_status=status,call_returned=True);self._set(p)
        if status=='PUBLISHED':self.counts['publications_published']+=1
        if status=='BUSY':self.counts['publications_BUSY']+=1
        end=self._now();p=replace(p,attempt_end_ns=end);self._set(p)
        self._event('WORKER_RESULT_PUBLICATION',activation=p.job.activation,iteration=iteration,
            attempt=p.attempts,start_ns=start,end_ns=end,status=status,deadline_ns=p.job.deadline_ns,
            job_sha256=digest(p.source.payload),result_sha256=digest(p.payload))
        p=replace(p,attempt_event_recorded=True);self._set(p);self.counts['publication_events']+=1
        if status=='PUBLISHED':self._terminal('PUBLISHED' if end<p.job.deadline_ns else 'PUBLISHED_LATE')
        elif status!='BUSY':self._terminal('PUBLICATION_'+status)
        elif end>=p.job.deadline_ns:self._terminal('ORIGINAL_DEADLINE')
        elif p.attempts==MAX_RESULT_ATTEMPTS:self._terminal('ATTEMPT_LIMIT')

    def tick(self,jobs,results):
        if self.closed or self.failure is not None:raise WorkerResultFailure('worker cannot restart')
        iteration=self.counts['iterations_started'];self.counts['iterations_started']+=1
        try:
            if self.pending is None:
                if not self.buffered:self._poll(jobs)
                if self.buffered:self._prepare()
            if self.pending is not None:self._publish(results,iteration)
            self.counts['iterations_completed']+=1
        except Exception as exc:
            self.abort('WORKER_ITERATION_EXCEPTION',dict(type=type(exc).__name__,detail=str(exc)))
            raise

    def finish(self):
        if self.closed:return
        if self.pending is not None or self.buffered:
            self.abort('STOP_WITH_OWNED_WORK',dict(buffered_jobs=len(self.buffered)))
        self.closed=True

    def evidence(self):
        return dict(contract='immutable_result_BUSY_max20_original_deadline',max_attempts=MAX_RESULT_ATTEMPTS,
            counts=dict(self.counts),phase=self.phase,closed=self.closed,
            pending=None if self.pending is None else self.pending.evidence(),
            last_result=None if self.last_result is None else self.last_result.evidence(),
            buffered_jobs=[x.evidence() for x in self.buffered],
            last_poll_return=None if self.last_poll_return is None else self.last_poll_return.decode(),
            clock_failure=None if self.clock_failure is None else json.loads(self.clock_failure),
            failure=None if self.failure is None else json.loads(self.failure))
