"""Fresh namespace and path spelling repair only; no task imports or dispatch."""
import hashlib
import json
import shutil
from pathlib import Path

BASE=Path(__file__).resolve().parent
OLD=BASE.with_name('direct_target_width512_expert_recovery_v1')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def norm(p):return Path(p).resolve().as_posix()
def subject(p):return dict(path=norm(p),sha256=sha(p))
def write(p,value):
    with Path(p).open('x',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def copy(p,q):
    q.parent.mkdir(parents=True,exist_ok=True)
    assert not q.exists()
    shutil.copyfile(p,q)
    assert sha(p)==sha(q)
def main():
    original=read(OLD/'frozen_inputs.json')
    owner=read(OLD/'owner_completion_v2.json')
    assert owner['completion_accounting_passed'] and not owner['requested_recovery_completed']
    assert owner['raw_python_exit_code']==1 and owner['processes_absent']
    assert all(v==0 for c in owner['actual_work_counters'].values() for v in c.values())
    for name,digest in original['source_sha256'].items():
        assert sha(OLD/'source_snapshot_v1'/name)==digest
        copy(OLD/'source_snapshot_v1'/name,BASE/'source_snapshot_v1'/name)
    for name in ('precontrol251.npz','actual_prefix251.npz','selection_receipt.json'):
        copy(OLD/'inputs'/name,BASE/'inputs'/name)
    # Runtime/owner names return to one canonical gate in the fresh namespace.
    for old,new in [('run_recovery_durable_v2.ps1','run_recovery_durable.ps1'),
                    ('verify_recovery_completed_v2.py','verify_recovery_completed.py'),
                    ('run_recovery_v2.sh','run_recovery.sh')]:
        value=(OLD/old).read_text()
        for a,b in [('direct_target_width512_expert_recovery_v1','direct_target_width512_expert_recovery_v2'),
                    ('recovery_process_v2','recovery_process_v1'),
                    ('execution_clearance_v2.json','execution_clearance.json'),
                    ('launch_receipt_v2.json','launch_receipt.json'),
                    ('run_recovery_durable_v2.ps1','run_recovery_durable.ps1'),
                    ('owner_completion_v2.json','owner_completion.json')]:value=value.replace(a,b)
        with (BASE/new).open('x',newline='\n') as f:f.write(value)
    request=read(OLD/'execution_request.json')
    request['environment']['PYTHONPATH']='/mnt/e'+BASE.as_posix()[2:]+'/source_snapshot_v1'
    roles=request['subjects']
    for key,value in roles.items():
        value['path']=norm(value['path'])
        if key in ('selected_snapshot','selected_prefix','input_selection'):
            value['path']=norm(BASE/'inputs'/Path(value['path']).name)
    for key,p in [('prior_failed_request',OLD/'execution_request.json'),
                  ('prior_failed_frozen',OLD/'frozen_inputs.json'),
                  ('prior_failed_owner',OLD/'owner_completion_v2.json'),
                  ('prior_failed_exit',OLD/'recovery_process_v2/exit.json'),
                  ('prior_failed_work',OLD/'work_counters.json'),
                  ('prior_failed_failure',OLD/'failure.json')]:roles[key]=subject(p)
    request['metadata_repair']=dict(input_keys_forward_slash=True,task_source_changes=0,
        previous_task_work_all_zero=True,previous_attempt_preserved=True,
        input_arrays_copied_byte_exact=True,input_extraction_repeated=False)
    write(BASE/'execution_request.json',request)
    pins={}
    changed=[]
    for old,digest in original['input_sha256'].items():
        p=norm(old)
        if Path(p).parent==OLD/'inputs':p=norm(BASE/'inputs'/Path(p).name)
        if p in pins:assert pins[p]==digest
        pins[p]=digest
        if p!=old:changed.append(dict(original=old,normalized=p,sha256=digest))
    for role in roles.values():
        if role['path'] in pins:assert pins[role['path']]==role['sha256']
        pins[role['path']]=role['sha256']
    for p,digest in pins.items():
        assert '\\' not in p and p[1:3]==':/' and sha(p)==digest,p
    frozen=dict(original,input_sha256=pins,request_sha256=sha(BASE/'execution_request.json'))
    write(BASE/'frozen_inputs.json',frozen)
    write(BASE/'metadata_derivation.json',dict(passed=True,original_namespace=norm(OLD),
        new_namespace=norm(BASE),source_sha256=original['source_sha256'],source_byte_exact_count=20,
        original_protocol_unchanged=request['protocol']==read(OLD/'execution_request.json')['protocol'],
        copied_inputs={name:sha(BASE/'inputs'/name) for name in ('precontrol251.npz','actual_prefix251.npz','selection_receipt.json')},
        normalized_paths=changed,all_frozen_keys_forward_slash=True,input_pin_count=len(pins),
        request=subject(BASE/'execution_request.json'),frozen=subject(BASE/'frozen_inputs.json'),
        prior_owner=subject(OLD/'owner_completion_v2.json'),task_model_calls=0,native_steps=0,replans=0))
    print(json.dumps(dict(request=subject(BASE/'execution_request.json'),frozen=subject(BASE/'frozen_inputs.json'),pins=len(pins),sources=20)))
if __name__=='__main__':main()
