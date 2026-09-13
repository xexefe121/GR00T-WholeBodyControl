"""Future spawned worker entry. No policy/model/physics; commands are recorded.

Source-only tests call worker_once with fake endpoints and clocks, never process_main.
"""
import json
import os
import time
from clock_core import FixedLedger,encode,digest
from recorded_protocol import decode_job,reply_to_job
from session import DeadlineClock,ObservedEndpoint


def worker_once(jobs,results,binding,table,clock,ledger):
    for slot,status,publication in jobs.poll_once():
        if status!='TAKEN':continue
        received=clock.now_ns()
        job=decode_job(publication.payload,binding,table)
        completed=clock.now_ns()
        payload=reply_to_job(publication.payload,binding,table,completed)
        publication_status=results.try_publish(job.activation,payload)  # Exactly one attempt; no retry.
        ledger.append(encode(dict(reason='RECORDED_REPLY',activation=job.activation,received_ns=received,
            completed_ns=completed,published_return_ns=clock.now_ns(),status=publication_status,
            job_sha256=digest(publication.payload),result_sha256=digest(payload),command_id=table.commands[job.activation].command_id)))


def process_main(jobs,results,binding,table,ready,stop,output_path):
    """Dormant future entry; selected benchmark launcher and review still missing."""
    clock=DeadlineClock(time.monotonic_ns,time.sleep);ledger=FixedLedger(125000)
    jobs=ObservedEndpoint(jobs,clock,ledger,'worker-jobs');results=ObservedEndpoint(results,clock,ledger,'worker-results')
    polls=0;failure=None;start=clock.now_ns()
    try:
        ready.set()  # Table and process imports are complete before parent admits epoch.
        while not stop.is_set() and polls<60000 and clock.now_ns()-start<55_000_000_000:
            worker_once(jobs,results,binding,table,clock,ledger);polls+=1
            time.sleep(.001)  # Worker only; no plant wait, spin or queue insertion.
        if not stop.is_set():raise RuntimeError('fixed worker watchdog exhausted')
    except Exception as exc:failure=dict(type=type(exc).__name__,detail=str(exc))
    finally:
        # Evidence flush is after the worker stops; no disk I/O in mailbox operations.
        report=dict(pid=os.getpid(),polls=polls,start_ns=start,end_ns=clock.now_ns(),failure=failure,
                    events=[json.loads(v) for v in ledger.records()],overflow=None if ledger.overflow_record is None else ledger.overflow_record.decode(),
                    recorded_command_table_sha256=table.sha256,model_calls=0,native_steps=0)
        with open(output_path,'x') as f:json.dump(report,f,indent=2)


class WorkerLifecycle:
    """Before-epoch startup and after-epoch teardown only. Stub-testable handles."""
    def __init__(self,process,ready,stop,clock):
        self.process,self.ready,self.stop,self.clock=process,ready,stop,clock;self.records=[]
        self.start_attempted=False;self.start_returned=False;self.ready_returned=False;self.closed=False;self.start_failure=None
    def start_ready(self):
        if self.start_attempted or self.closed:raise ValueError('no worker restart')
        self.start_attempted=True;start=self.clock.now_ns()
        try:self.process.start()
        except Exception as exc:
            self.start_failure=encode(dict(type=type(exc).__name__,detail=str(exc),start_ns=start))
            self.records.append(dict(event='START_FAILED',first_error=self.start_failure.decode(),start_returned=False))
            raise
        self.start_returned=True
        self.records.append(dict(event='STARTED',pid=self.process.pid,start_ns=start,returned_ns=self.clock.now_ns()))
        if not self.ready.wait(10) or not self.process.is_alive():raise RuntimeError('worker failed pre-epoch readiness')
        self.ready_returned=True
        self.records.append(dict(event='READY',pid=self.process.pid,observed_ns=self.clock.now_ns()))
    def stop_join(self):
        if not self.start_attempted or self.closed:raise ValueError('worker start never attempted or already closed')
        if not self.start_returned:
            # A failed start may not own a joinable process. Do not mask its
            # original exception or infer that no descendant could have existed.
            close_error=None
            try:self.process.close()
            except Exception as exc:close_error=dict(type=type(exc).__name__,detail=str(exc))
            self.closed=True
            result=dict(event='START_FAILURE_CLEANUP',start_attempted=True,start_returned=False,ready_returned=False,
                        join_attempted=False,process_absence_proven=False,start_side_effects_uncertain=True,
                        normal_exit=False,close_error=close_error,
                        first_error=None if self.start_failure is None else self.start_failure.decode())
            self.records.append(result)
            return dict(result)
        start=self.clock.now_ns();self.stop.set();self.process.join(5);terminated=False
        if self.process.is_alive():self.process.terminate();self.process.join(5);terminated=True
        live=self.process.is_alive();exitcode=self.process.exitcode;pid=self.process.pid
        self.records.append(dict(event='STOPPED',pid=pid,alive=live,exitcode=exitcode,terminated=terminated,
                                 stop_requested_ns=start,cleanup_finished_ns=self.clock.now_ns()))
        if not live:self.process.close();self.closed=True
        if live:raise RuntimeError('selected worker process survived cleanup')
        return dict(pid=pid,exitcode=exitcode,terminated=terminated,normal_exit=exitcode==0 and not terminated)
