"""One synthetic subprocess; preserve exit, test evidence and Linux child cleanup."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (BASE/name).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
def status(pid):
    try:
        text=Path('/proc/'+str(pid)+'/stat').read_text();state=text.rsplit(')',1)[1].split()[0]
        return dict(pid=pid,present=True,state=state,live=state not in ['Z','X'])
    except FileNotFoundError:return dict(pid=pid,present=False,state=None,live=False)

if __name__=='__main__':
    with (BASE/'run.lock').open('x') as f:f.write(str(os.getpid()))
    started=time.monotonic();child=None;code=None;failure=None
    args=[sys.executable,'-B',str(BASE/'run_tests.py')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        with (BASE/'stdout.log').open('xb') as stdout,(BASE/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(args,cwd=str(BASE),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            write('start.json',dict(parent_pid=os.getpid(),child_pid=child.pid,argv=args,request_sha256=sha(BASE/'request.json')))
            code=child.wait(timeout=120)
    except BaseException:
        failure=traceback.format_exc()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    report=json.loads((BASE/'test_report.json').read_text()) if (BASE/'test_report.json').exists() else None
    pids=[] if child is None else [child.pid]
    if report is not None:
        pids += [item['pid'] for item in report['spawned_processes']]
        if report['resource_tracker_pid'] is not None:pids.append(report['resource_tracker_pid'])
    states=[status(pid) for pid in sorted(set(pids))]
    cleanup_deadline=time.monotonic()+2
    while any(s['live'] for s in states) and time.monotonic()<cleanup_deadline:
        time.sleep(.05)
        states=[status(pid) for pid in sorted(set(pids))]
    exit_report=dict(raw_exit_code=code,exception=failure,elapsed_seconds=time.monotonic()-started,process_cleanup=states,
        no_live_selected_children=all(not s['live'] for s in states),test_report_sha256=None if report is None else sha(BASE/'test_report.json'),
        request_sha256=sha(BASE/'request.json'),passed=code==0 and failure is None and report is not None and report['passed'] and all(not s['live'] for s in states),
        stdout_sha256=sha(BASE/'stdout.log'),stderr_sha256=sha(BASE/'stderr.log'),
        limitations=['Linux spawn synthetic transport/admission only; no plant ticks or native/model calls.',
                     'Nonblocking semaphore acquisition does not bound copy/hash/allocation/OS scheduling latency.',
                     'No recovery from a dead lock owner within the same epoch; permanent BUSY remains expected.',
                     'No asynchronous controller integration, 500 Hz timing or balance qualification.'])
    write('completion.json',exit_report)
    print(json.dumps(exit_report,indent=2))
    sys.exit(0 if exit_report['passed'] else 1)
