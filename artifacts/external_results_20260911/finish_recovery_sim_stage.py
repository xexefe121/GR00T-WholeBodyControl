"""Observe actual exited processes, then use the existing completion verifier."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'direct_target_width251_student_v1'))
from execution_common import absent
p=argparse.ArgumentParser();p.add_argument('mode',choices=('witness','evaluation'));p.add_argument('--run',default='direct_target_width251_evaluation_v1');a=p.parse_args()
run=BASE/a.run;folder=run/(a.mode+'_process')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
end=read(folder/'exit.json');start=read(folder/'start.json');child=read(folder/'child.json')
assert end['error'] is None and end['all_postrun_hashes_exact'] is True
record=dict(wrapper_pid=start['wrapper_pid'],child_pid=child['child_pid'],
    wrapper_absent=absent(start['wrapper_pid']),child_absent=absent(child['child_pid']),
    observed_utc=datetime.now(timezone.utc).isoformat())
assert record['wrapper_absent'] is record['child_absent'] is True
path=folder/'process_absence.json'
with path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
subprocess.run([sys.executable,'-B',str(run/'verify_completed_stage.py'),'--mode',a.mode,'--process-check',str(path)],check=True)
