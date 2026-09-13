import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_causal_response_balanced_student_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
rp,fp,lp=BASE/'training_request.json',BASE/'training_frozen_inputs.json',BASE/'run_fit_durable_v1.ps1'
assert sha(rp)=='4c332bb060e48cceb1dac7010b330fd868a5c1d517133fb7506979e6a5c2d717'
assert sha(fp)=='470130562e4af377ab7b3fcf389488d38d1e372f92f6eb1e7a9221f91d8009c1'
assert sha(lp)=='f5dc300610b242c6026163b652cd0304c585fc71a0e42502311510b49b5c744a'
assert sha(OUT/'review.json')=='ff54ed8d7e9b96d2c254b9e27dba436cdbc9169d52ab985a3248dcb0a4fd35f7'
r,f=read(rp),read(fp);prep=read(BASE/'source_preparation.json');s=Path(f['source_directory'])
assert f['training_request_sha256']==sha(rp) and r['source_preparation_sha256']==sha(BASE/'source_preparation.json')
assert f['source_sha256']==prep['source_sha256'] and len(f['source_sha256'])==25
assert s.resolve()==(BASE/'source_snapshot_v1').resolve()
for name,digest in f['source_sha256'].items():assert sha(s/name)==sha(Path(prep['source_directory'])/name)==digest
sys.path.insert(0,str(s))
from response_contract import check_protocol
from balance_contract import GROUP_WEIGHTS
check_protocol(r)
assert r['root_selected'] is True and r['group_weights']==list(GROUP_WEIGHTS)
assert len(f['input_sha256'])==328
for path,digest in f['input_sha256'].items():assert sha(path)==digest,path
def pin(path,digest):assert f['input_sha256'][Path(path).as_posix()]==digest==sha(path),path
for name,sub in r['subjects'].items():
    pin(sub['path'],sub['sha256'])
    if 'pass_field' in sub:
        record=read(sub['path']);value=record
        for key in sub['pass_field'].split('.'):value=value[key]
        assert value is True,name
        for key,value in sub.get('required_fields',{}).items():assert record.get(key)==value,(name,key)
for group in ('paths','full_state_paths','restoration_predictions','context_paths'):
    for path in r[group].values():pin(path,sha(path))
assert r['subjects']['checkpoint']['sha256']=='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'
assert r['subjects']['energy_source']['sha256']=='a9b7a2065d8f2d45a1c1a06dc4d5678c68df910f2ebfc6d098989274266d6cc8'
assert r['subjects']['semantics_review']['sha256']=='e633e27d39cde9b24026519c2aeae68f44d22450cc70e33cb55b1c292ed492ae'
old=read(NEW/'direct_target_causal_context_study_v2/training_request.json')
assert r['paths']==old['paths'] and r['full_state_paths']==old['full_state_paths']
for name,path in r['context_paths'].items():assert Path(path).parent.resolve()==(NEW/'direct_target_causal_context_study_v2/fit/shared').resolve()
for name,path in r['restoration_predictions'].items():assert Path(path).resolve()==(NEW/'direct_target_causal_context_study_v2/fit/causal'/('final_GPU32_'+name+'.npy')).resolve()
tests=ET.parse(BASE/'execution_helper_tests.xml').getroot()
suites=[tests] if tests.tag=='testsuite' else list(tests)
assert sum(int(v.attrib['tests']) for v in suites)==8
assert all(int(v.attrib.get(k,0))==0 for v in suites for k in ('errors','failures','skipped'))
assert read(BASE/'launcher_tests.json')['passed'] is True
for path in (lp,BASE/'verify_completed.py',BASE/'launcher_tests.json',BASE/'execution_helper_tests.xml'):pin(path,sha(path))
assert not any((BASE/n).exists() for n in ('fit','fit_process_v1','training_clearance.json','owner_completion_verification.json'))
result=dict(prelaunch_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 training_request_sha256=sha(rp),frozen_receipt_sha256=sha(fp),launcher_sha256=sha(lp),
 source_review_sha256=sha(OUT/'review.json'),source_sha256=f['source_sha256'],checked_input_pins=328,
 source_files=25,root_selected_single_continuation=True,condition='causal',updates=3000,
 ordinary_start_step=68000,ordinary_final_step=71000,optimizer_start_step=3000,optimizer_final_step=6000,
 actual_task_arrays_loaded=0,task_checkpoint_loads=0,task_model_calls=0,task_gradient_calls=0,native_steps=0,
 reviewed_semantics=['all concrete inputs and selected source snapshot exact; data paths equal completed study_v2',
 'actual complete causal checkpoint, same saved shared context/schedule and restoration outputs selected',
 'fixed warm3000 updates/9000 training calls/44058000 rows; four Torch plus one ORT diagnostic pass',
 'original fixed 1e-5rad gates and PS5.1 explicit-double coefficient guard retained',
 'single hidden child with acquired handle, exact clearance/review binding, preserved pre/post pins/raw exit',
 'owner verifies complete or truthful failed counter prefix, actual process absence and direct output identities'],
 writer_sha256=sha(__file__),automatic_retry=False,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
clear=read(BASE/'training_clearance_draft.json');assert clear['approved'] is False
clear.update(approved=True,review_path=path.as_posix(),review_sha256=sha(path))
assert clear['request_sha256']==sha(rp) and clear['frozen_receipt_sha256']==sha(fp) and clear['launcher_sha256']==sha(lp)
cp=BASE/'training_clearance.json'
with cp.open('x',encoding='utf-8') as stream:json.dump(clear,stream,indent=2);stream.write('\n')
print(json.dumps(dict(prelaunch_review_pass=True,review_path=path.as_posix(),review_sha256=sha(path),clearance_sha256=sha(cp))))
