"""Concrete root review of the one selected width512 warm continuation."""
import ast,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_causal_width512_student_v1'
OLD=NEW/'direct_target_causal_response_balanced_student_v2'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
rp,fp,lp=BASE/'training_request.json',BASE/'training_frozen_inputs.json',BASE/'run_fit_durable_v1.ps1'
assert sha(rp)=='f78bc57dc24af30fca2f7d20a05bd82ddf78e51158ebe1f41f22b85784481248'
assert sha(fp)=='4cb56ebc741ac40f6de52e83cba8f718a7db78f07e1d02a1c25c02902d78e939'
assert sha(lp)=='5e32d94a29b2234d2b9e7effc42173405e0d6b4e4c8c14db3cdaaaa886065938'
prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='09871661a31891b28d83daf7810d2dfdd1141b4d1c3d6164193ffdfd42abf675'
review_path=NEW/'direct_target_width512_fit_independent_review_v1/review.json'
assert sha(review_path)=='3ec9fe527f2159b5a3575eae9d7f6e6fd575e205075d2cdaf31fcf3f8a3efaf0'
r,f,prep,review=read(rp),read(fp),read(prep_path),read(review_path)
assert review['source_review_pass'] is True and prep['source_preparation_passed'] is True
assert f['training_request_sha256']==sha(rp) and r['source_preparation_sha256']==sha(prep_path)
assert f['source_sha256']==prep['source_sha256']==review['source_sha256'] and len(f['source_sha256'])==28
source=Path(f['source_directory']);assert source.resolve()==(BASE/'source_snapshot_v1').resolve()
for name,h in f['source_sha256'].items():assert sha(source/name)==sha(Path(prep['source_directory'])/name)==h,name
assert prep['original_byte_exact_count']==21
for name in prep['unchanged_source_files']:assert sha(source/name)==sha(OLD/'source_snapshot_v1'/name),name
sys.path.insert(0,str(source))
from response_contract import check_protocol
from balance_contract import GROUP_WEIGHTS
import torch
check_protocol(r)
assert r['root_selected'] is True and r['group_weights']==list(GROUP_WEIGHTS)
assert not torch.cuda.is_initialized()
assert len(f['input_sha256'])==398
for p,h in f['input_sha256'].items():assert sha(p)==h,p
def pin(p,h):assert f['input_sha256'][Path(p).as_posix()]==h==sha(p),p
for name,sub in r['subjects'].items():
    pin(sub['path'],sub['sha256'])
    if 'pass_field' in sub:
        record=read(sub['path']);value=record
        for k in sub['pass_field'].split('.'):value=value[k]
        assert value is True,name
        for k,v in sub.get('required_fields',{}).items():assert record.get(k)==v,(name,k)
for group in ('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'):
    for p in r[group].values():pin(p,sha(p))
assert r['subjects']['checkpoint']['sha256']=='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'
assert r['subjects']['energy_source']['sha256']=='a9b7a2065d8f2d45a1c1a06dc4d5678c68df910f2ebfc6d098989274266d6cc8'
assert r['subjects']['semantics_review']['sha256']=='d6b8cf79bac101f00614d5ba94d2dd3d490a58fdf8a404a87414e16137c3d43f'
old=read(OLD/'training_request.json')
assert r['paths']==old['paths'] and r['full_state_paths']==old['full_state_paths']
for name,p in r['context_paths'].items():assert Path(p).parent.resolve()==(OLD/'fit/shared').resolve(),name
for name,p in r['restoration_predictions'].items():assert Path(p).resolve()==(OLD/'fit'/('final_GPU32_'+name+'.npy')).resolve()
proof_path=BASE/'saved_schedule_proof.json'
assert sha(proof_path)=='b9abf1256144a569b965bf25982a349003d8f4531927408e4b886cda85a2f677'
proof=read(proof_path)
assert proof['passed'] is True and proof['first3000_centers_byte_exact'] is True and proof['first3000_axes_byte_exact'] is True
assert proof['full_updates']==10000 and proof['prior_updates']==3000 and proof['all54_cells_and_groups_valid'] is True and proof['schedule_generated'] is False
for name,p in r['schedule_paths'].items():assert Path(p).resolve()==(NEW/'direct_target_full_state_student_v1/fit'/(name+'.npy')).resolve()
xml=ET.parse(BASE/'execution_helper_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==8
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
assert read(BASE/'launcher_tests.json')['passed'] is True
for p in (lp,BASE/'verify_completed.py',BASE/'launcher_tests.json',BASE/'execution_helper_tests.xml'):pin(p,sha(p))
assert '1.8188207859141674d' in lp.read_text()
assert '-WindowStyle Hidden' in lp.read_text() or 'CreateNoWindow=$true' in lp.read_text() or 'CreateNoWindow = $true' in lp.read_text()
assert not any((BASE/n).exists() for n in ('fit','fit_process_v1','training_clearance.json','owner_completion_verification.json'))
clear=read(BASE/'training_clearance_draft.json');assert clear['approved'] is False
assert clear['request_sha256']==sha(rp) and clear['frozen_receipt_sha256']==sha(fp) and clear['launcher_sha256']==sha(lp)
result=dict(prelaunch_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 training_request_sha256=sha(rp),frozen_receipt_sha256=sha(fp),launcher_sha256=sha(lp),
 source_review_sha256=sha(review_path),source_preparation_sha256=sha(prep_path),source_sha256=f['source_sha256'],
 checked_input_pins=398,source_files=28,root_selected_single_continuation=True,condition='causal',updates=10000,
 ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000,
 actual_task_arrays_loaded=0,task_checkpoint_loads=0,task_model_calls=0,task_gradient_calls=0,native_steps=0,
 reviewed_semantics=['Frozen28 sources match independently tested source,21 inherited modules byteexact; all398 actual inputs/positive subject bindings checked.',
 'Width512 preserves original256 blocks/normalization/RNG, six warm steps6000/new zero moments; exact actual restoration checked before any forward.',
 'Fixed10000 saved schedule, first3000 proven byteexact;250 inclusive ramp then9750 inclusive cosine;30000 forwards/146860000rows.',
 'Original N/P/balanced54 coefficient and data preserved; initial1e-5 source gate/byte diagnostic and final64 backend1e-5 export gate unchanged.',
 'Original PS5.1 explicit-double guard, hidden acquired-handle child, CreateNew lock, raw exit/prefix preservation and process/input/output owner checks.',
 'No controller witness or physical qualification implied; one actual fit only, no automatic retry/checkpoint selection.'],
 writer_sha256=sha(__file__),automatic_retry=False,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
clear.update(approved=True,review_path=path.as_posix(),review_sha256=sha(path))
cp=BASE/'training_clearance.json'
with cp.open('x',encoding='utf-8') as stream:json.dump(clear,stream,indent=2);stream.write('\n')
print(json.dumps({'prelaunch_review_pass':True,'review_path':path.as_posix(),'review_sha256':sha(path),'clearance_sha256':sha(cp),'pins':398}))
