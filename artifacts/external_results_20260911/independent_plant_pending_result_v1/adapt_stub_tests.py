"""Keep previous fake semantics; give the new worker explicit retained state."""
from pathlib import Path
B=Path(__file__).resolve().parent/'source_draft_v1'
p=B/'test_scaffold.py';s=p.read_text()
s=s.replace('from dummy_worker import worker_once,WorkerLifecycle','from dummy_worker import worker_once,WorkerLifecycle\nfrom pending_result import PendingResultWorker,WorkerResultFailure')
s=s.replace('    ledger=FixedLedger(100)\n    while session.foundation.returned',
 '''    ledger=FixedLedger(1000);worker=PendingResultWorker(BINDING,commands,clock,ledger);worker_live=True
    while session.foundation.returned''')
s=s.replace('        if pump:worker_once(jobs,results,BINDING,commands,clock,ledger)',
 '''        if pump and worker_live:
            try:worker_once(jobs,results,worker)
            except WorkerResultFailure:worker_live=False''')
s=s.replace('        worker_once(jobs,results,BINDING,commands,clock,FixedLedger(100));session.tick_once()',
 '''        publication=next(p for _,status,p in jobs.poll_once() if status=='TAKEN')
        results.try_publish(1,reply_to_job(publication.payload,BINDING,commands,clock.ns));session.tick_once()''')
s=s.replace('clock.ns=119_000_000;worker_once(jobs,results,BINDING,commands,clock,ledger);clock.ns=120_000_000',
 'clock.ns=119_000_000;worker_once(jobs,results,PendingResultWorker(BINDING,commands,clock,ledger));clock.ns=120_000_000')
p.write_text(s,newline='\n')
p=B/'test_pending_publication.py';s=p.read_text()
s=s.replace('from test_scaffold import setup, worker_once, BINDING','from test_scaffold import setup, worker_once, BINDING\nfrom pending_result import PendingResultWorker\nfrom recorded_protocol import reply_to_job')
s=s.replace('return worker_once(jobs, results, BINDING, commands, clock, FixedLedger(100))',
 'return worker_once(jobs, results, PendingResultWorker(BINDING, commands, clock, FixedLedger(100)))')
s=s.replace('        pump(p)\n        p[5].tick_once()\n        self.assertNotIn(1, f.sealed)',
 '''        publication=next(v for _,status,v in p[3].poll_once() if status=='TAKEN')
        p[4].try_publish(1,reply_to_job(publication.payload,BINDING,p[2],p[0].ns))
        p[5].tick_once()
        self.assertNotIn(1, f.sealed)''')
p.write_text(s,newline='\n')
