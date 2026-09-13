"""Preserve unexecutedv1, align only actual producer process/owner and audit source paths."""
from pathlib import Path
import difflib,json,hashlib
BASE=Path(__file__).resolve().parent
OLD=BASE/'source_draft_v1';NEW=BASE/'source_draft_v2';NEW.mkdir(exist_ok=False)
changes={}
for p in sorted(OLD.glob('*.py')):
    text=p.read_text();updated=text
    if p.name=='audit_saved_pair.py':updated=text.replace('fit_process/','fit_process_v2/').replace("base/'fit_process'","base/'fit_process_v2'").replace('owner_completion_verification_v2.json','owner_completion_verification_v3.json')
    with (NEW/p.name).open('x',encoding='utf-8',newline='\n') as f:f.write(updated)
    if updated!=text:changes[p.name]=''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile=p.name+' v1',tofile=p.name+' v2'))
specs={
 'prepare_audit_request.py':('prepare_audit_request_v2.py', [('source_draft_v1','source_draft_v2'),('owner_completion_verification_v2.json','owner_completion_verification_v3.json')]),
 'freeze_launch.py':('freeze_launch_v2.py',[('run_audit_durable.ps1','run_audit_durable_v2.ps1'),('prepare_audit_request.py','prepare_audit_request_v2.py'),('source_preparation.json','source_preparation_v2.json'),('verify_completion.py','verify_completion_v2.py')]),
 'run_audit_durable.ps1':('run_audit_durable_v2.ps1',[('source_draft_v1','source_draft_v2')]),
 'verify_completion.py':('verify_completion_v2.py',[('source_preparation.json','source_preparation_v2.json')])}
for old,(new,replacements) in specs.items():
    text=(BASE/old).read_text();updated=text
    for before,after in replacements:
        assert before in updated;updated=updated.replace(before,after)
    with (BASE/new).open('x',encoding='utf-8',newline='\n') as f:f.write(updated)
    changes[new]=''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile=old,tofile=new))
with (BASE/'source_delta_v2.patch').open('x',encoding='utf-8') as f:f.write('\n'.join(changes.values()))
print(json.dumps({'changed':list(changes),'actual_audit_run':False}))
