"""One-line carried-request repair; no task runtime imports or execution."""
from pathlib import Path
import ast,hashlib,json,shutil
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_response_balanced_student_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
prior=read(OLD/'source_preparation.json')
assert read(OLD/'owner_completion_verification.json')['accounting_passed'] is True
assert read(OLD/'owner_completion_verification.json')['completion_passed'] is False
assert read(OLD/'zero_forward_failure_verification.json')['passed'] is True
source=BASE/'source_prepared_v1';source.mkdir(exist_ok=False)
new_map={}
for name,digest in prior['source_sha256'].items():
    original=Path(prior['source_directory'])/name
    assert sha(original)==digest
    shutil.copyfile(original,source/name)
    if name=='train_response_balanced.py':
        text=(source/name).read_text(encoding='utf-8')
        old="dict(**request,condition=condition,initialization_sha256="
        new="dict(request,condition=condition,initialization_sha256="
        assert text.count(old)==1
        (source/name).write_text(text.replace(old,new),encoding='utf-8',newline='\n')
    ast.parse((source/name).read_text(encoding='utf-8'))
    new_map[name]=sha(source/name)
changed=[name for name,digest in new_map.items() if digest!=prior['source_sha256'][name]]
assert changed==['train_response_balanced.py']
for name in ('DESIGN.md','OUTPUT_SCHEMA.md','EXECUTION_SCHEMA.md'):
    shutil.copyfile(OLD/name,BASE/name)
request=read(OLD/'training_request.json');request['root_selected']=False
request.pop('source_preparation_sha256',None)
for key in ('source_preparation','warm_source_review','root_source_data_export_review'):
    request['subjects'].pop(key,None)
for key,path,field in [('previous_failure_owner',OLD/'owner_completion_verification.json','accounting_passed'),
                       ('previous_failure_verification',OLD/'zero_forward_failure_verification.json','passed')]:
    request['subjects'][key]=dict(path=path.as_posix(),sha256=sha(path),pass_field=field)
request.update(pending_before_actual_fit=['root one-line repair and concrete namespace request/launcher clearance'],
    previous_zero_forward_attempt=OLD.as_posix(),previous_forward_calls=0,previous_optimizer_updates=0,
    metadata_repair='dict(request,condition=condition,...) accepts already present condition without duplicate keyword error')
write(BASE/'training_request_proposal.json',request)
write(BASE/'source_derivation.json',dict(source_sha256=new_map,prior_source_sha256=prior['source_sha256'],
    changed_sources=changed,unchanged_sources=[name for name in new_map if name not in changed],
    exact_delta="dict(**request,condition=condition,initialization_sha256= -> dict(request,condition=condition,initialization_sha256=",
    parent_source_preparation_sha256=sha(OLD/'source_preparation.json'),previous_owner_sha256=sha(OLD/'owner_completion_verification.json'),
    previous_failure_verification_sha256=sha(OLD/'zero_forward_failure_verification.json'),
    task_model_calls=0,optimizer_updates=0,native_steps=0))
print(json.dumps(dict(source_count=len(new_map),changed=changed,driver_sha256=new_map['train_response_balanced.py'])))
