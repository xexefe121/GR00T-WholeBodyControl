"""Fresh source-only worker result retry preparation; never execute task runtimes."""
import difflib,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
OLD=B.parent/'independent_plant_pending_publication_v1/source_draft_v1'
S=B/'source_draft_v1'
S.mkdir(exist_ok=False)
original={}
for p in sorted(OLD.glob('*.py')):
    data=p.read_bytes();(S/p.name).write_bytes(data);original[p.name]=hashlib.sha256(data).hexdigest()

def change(name,old,new):
    p=S/name;s=p.read_text();assert s.count(old)==1,(name,old[:80]);p.write_text(s.replace(old,new),newline='\n')

change('clock_core.py','    created_ns: int\n\n    @property',
 '''    created_ns: int
    deadline_ns: int

    def __post_init__(self):
        if type(self.deadline_ns) is not int or self.deadline_ns < 0:
            raise ValueError('literal nonnegative original deadline required')

    @property''')
change('clock_core.py','                    created_ns=self.created_ns)',
       '                    created_ns=self.created_ns, deadline_ns=self.deadline_ns)')
change('clock_core.py',"            elif encode(identity) != encode(self.issued[activation].identity()):",
 '''            elif type(identity.get('deadline_ns')) is not int or identity['deadline_ns'] != self.deadline(activation * CONTROL_STEPS):
                reason = 'WRONG_ORIGINAL_DEADLINE'
            elif encode(identity) != encode(self.issued[activation].identity()):''')
change('clock_core.py',"                      self._now_ns('JOB_CREATED'))",
 "                      self._now_ns('JOB_CREATED'), self.deadline((control + 1) * CONTROL_STEPS))")
change('clock_core.py','                                         self.deadline((control + 1) * CONTROL_STEPS))',
       '                                         job.deadline_ns)')
change('recorded_protocol.py',"'history_digest','schedule_digest','created_ns','snapshot_payload'}",
       "'history_digest','schedule_digest','created_ns','deadline_ns','snapshot_payload'}")
change('recorded_protocol.py',"['sequence','snapshot_control','snapshot_physics','activation','created_ns']:",
       "['sequence','snapshot_control','snapshot_physics','activation','created_ns','deadline_ns']:")
change('recorded_protocol.py',"value['schedule_digest'],value['created_ns'])",
       "value['schedule_digest'],value['created_ns'],value['deadline_ns'])")
change('run_clock.py',"'recorded_protocol','evidence','capture_schema'", "'recorded_protocol','pending_result','evidence','capture_schema'")

worker=(S/'dummy_worker.py').read_text()
start=worker.index('def worker_once(');end=worker.index('\n\nclass WorkerLifecycle:')
worker=worker[:start]+'''def worker_once(jobs,results,state):
    """One bounded worker-loop iteration; caller retains the explicit state."""
    return state.tick(jobs,results)


def process_main(jobs,results,binding,table,ready,stop,output_path):
    """Dormant source preparation; actual dispatch needs a new reviewed request."""
    from pending_result import PendingResultWorker
    clock=DeadlineClock(time.monotonic_ns,time.sleep);ledger=FixedLedger(125000)
    jobs=ObservedEndpoint(jobs,clock,ledger,'worker-jobs');results=ObservedEndpoint(results,clock,ledger,'worker-results')
    state=PendingResultWorker(binding,table,clock,ledger)
    polls=0;failure=None;start=clock.now_ns()
    try:
        ready.set()
        while not stop.is_set() and polls<60000 and clock.now_ns()-start<55_000_000_000:
            worker_once(jobs,results,state);polls+=1
            time.sleep(.001)  # Existing worker-loop sleep; never a plant wait.
        if not stop.is_set():raise RuntimeError('fixed worker watchdog exhausted')
    except Exception as exc:
        failure=dict(type=type(exc).__name__,detail=str(exc))
        state.abort('WORKER_EXCEPTION',failure)
    finally:
        state.finish()
        if failure is None and state.failure is not None:failure=json.loads(state.failure)
        report=dict(pid=os.getpid(),polls=polls,start_ns=start,end_ns=clock.now_ns(),failure=failure,
                    events=[json.loads(v) for v in ledger.records()],overflow=None if ledger.overflow_record is None else ledger.overflow_record.decode(),
                    result_retry=state.evidence(),recorded_command_table_sha256=table.sha256,model_calls=0,native_steps=0)
        with open(output_path,'x') as f:json.dump(report,f,indent=2)
''' + worker[end:]
(S/'dummy_worker.py').write_text(worker,newline='\n')

with (B/'original_source_hashes.json').open('x') as f:json.dump(original,f,indent=2)
print(json.dumps({'original_sources':len(original),'source_only':True}))
