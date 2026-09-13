"""Root selection of one fixed saved-only diagnostic after completed collection."""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_prepared_v1';PACKET=BASE/'actual_v1'
sys.path.insert(0,str(SOURCE))
from admission import admit,read,sha,report_gate,ROLES,PHYSICAL_KEYS
request=PACKET/'request.json';expected='07b621ffd80f07cf64410e30797b960751348de78b4f2690d1316820d53466fa'
assert sha(request)==expected
q=read(request)
assert q['root_selected_saved_diagnosis'] is True
assert q['scope']==dict(old_nominal=9904,old_physical=3054,new_rows=1018,alias_modes=6,proximity_queries=72,candidate_block=256)
assert q['model_calls']==q['native_steps']==q['optimizer_updates']==0
assert Path(q['output']).resolve()==(PACKET/'results_v1').resolve() and not Path(q['output']).exists()
assert set(q['subjects'])==set(ROLES) and set(q['physical_arrays'])==set(PHYSICAL_KEYS)
for item in [*q['subjects'].values(),*q['physical_arrays'].values()]:assert sha(item['path'])==item['sha256']
for name,h in q['source_sha256'].items():assert sha(SOURCE/name)==h
assert q['source_sha256']==read(BASE/'root_review.json')['source_sha256']
assert q['subjects']['source_review']['sha256']=='1100d985008254ed4fb92363c0c0b0cb930ebf64ed34937d866fbca98d4404fa'
assert q['subjects']['collection_report']['sha256']=='7e650844a98a3e9d5462a1a8ff45e0cbdc6847323e26b82f5b2a5685c07cf316'
nonreports={'old_centers','old_pico','old_walk002','old_nominal_context','old_physical_context','normalization','new_rows','export_source','contract'}
reports={role:read(item['path']) for role,item in q['subjects'].items() if role not in nonreports}
report_gate(q['subjects'],reports,q['physical_arrays'])
review=dict(passed=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),request_sha256=expected,
 source_sha256=q['source_sha256'],scope=q['scope'],all_subjects_exact=True,
 actual_task_arrays_read=0,model_calls=0,native_steps=0,optimizer_updates=0,writer_sha256=sha(__file__))
rp=PACKET/'root_review.json'
with rp.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
cp=PACKET/'clearance.json'
with cp.open('x',encoding='utf-8') as f:json.dump(dict(approved=True,request_sha256=expected,
 review=dict(path=rp.as_posix(),sha256=sha(rp)),automatic_retry=False),f,indent=2);f.write('\n')
admit(request,cp,sha(cp),SOURCE)
print(json.dumps(dict(selected_single_saved_diagnosis=True,review_sha256=sha(rp),clearance_sha256=sha(cp),actual_diagnosis_executed=False)))
