"""Root review of metadata-only path repair; no task imports or execution."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
OLD=NEW/'direct_target_width512_expert_recovery_v1'
BASE=NEW/'direct_target_width512_expert_recovery_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def norm(p):return str(p).replace('\\','/')
parser=argparse.ArgumentParser();parser.add_argument('--independent-review',type=Path,required=True);parser.add_argument('--independent-sha',required=True)
args=parser.parse_args()
expected={'request':'68ee9617562fa8b73ee4abd55d1703ca2f279887c09a5a04b851b814cd34be44',
 'frozen':'c6531b52a65e057602ed560c00448d7ebc2573a501ab9204d651b3af8955367e',
 'launch':'20c30bbfc99454c434006492e393bdfea7d0cc408b191c0868a38ef328593fd1'}
for key,name in [('request','execution_request.json'),('frozen','frozen_inputs.json'),('launch','launch_receipt.json')]:assert sha(BASE/name)==expected[key]
r,f,l=[read(BASE/name) for name in ('execution_request.json','frozen_inputs.json','launch_receipt.json')]
old_r,old_f=[read(OLD/name) for name in ('execution_request.json','frozen_inputs.json')]
assert sha(OLD/'execution_request.json')=='c3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8'
assert sha(OLD/'frozen_inputs.json')=='954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1'
assert f['request_sha256']==l['request_sha256']==expected['request'] and l['frozen_receipt_sha256']==expected['frozen']
assert f['source_sha256']==old_f['source_sha256'] and len(f['source_sha256'])==20
for name,digest in f['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==sha(OLD/'source_snapshot_v1'/name)==digest
assert r['protocol']==f['protocol']==old_r['protocol']==old_f['protocol']
for key in set(old_r)-{'subjects','environment'}:assert r[key]==old_r[key],key
assert set(r)-set(old_r)=={'metadata_repair'}
env=dict(old_r['environment']);env['PYTHONPATH']=env['PYTHONPATH'].replace('expert_recovery_v1/','expert_recovery_v2/')
assert r['environment']==env
copied={'selected_snapshot','selected_prefix','input_selection'}
for key,subject in old_r['subjects'].items():
 want=dict(subject);want['path']=norm(want['path'])
 if key in copied:want['path']=want['path'].replace('expert_recovery_v1/inputs/','expert_recovery_v2/inputs/')
 assert r['subjects'][key]==want,key
extras={'prior_failed_request','prior_failed_frozen','prior_failed_owner','prior_failed_exit','prior_failed_work','prior_failed_failure'}
assert set(r['subjects'])-set(old_r['subjects'])==extras
want={norm(k).replace('expert_recovery_v1/inputs/','expert_recovery_v2/inputs/'):v for k,v in old_f['input_sha256'].items()}
for key in extras:
 sub=r['subjects'][key];want[sub['path']]=sub['sha256']
assert f['input_sha256']==want and len(want)==44 and all('\\' not in k for k in want)
for key,sub in r['subjects'].items():assert f['input_sha256'][sub['path']]==sha(sub['path'])==sub['sha256'],key
pins=l['input_sha256'];assert len(pins)==85
normalized={norm(k).casefold():v for k,v in pins.items()};assert len(normalized)==85
for path,digest in pins.items():assert sha(path)==digest,path
for path,digest in want.items():assert normalized[norm(path).casefold()]==digest
assert sha(BASE/'run_recovery_durable.ps1')==l['launcher_sha256']=='63c139fde9ece01db4ebbd4d419847620c2aaf099535711558a95538afc6363b'
assert sha(BASE/'verify_recovery_completed.py')=='f06fb5e730cdb3416c058719f91d27ad17ccc07784538371cd6f72c500e263b8'
launch_old=(OLD/'run_recovery_durable_v2.ps1').read_text(encoding='utf-8-sig')
for old,new in [('recovery_process_v2','recovery_process_v1'),('execution_clearance_v2.json','execution_clearance.json'),('launch_receipt_v2.json','launch_receipt.json')]:launch_old=launch_old.replace(old,new)
assert (BASE/'run_recovery_durable.ps1').read_text(encoding='utf-8-sig')==launch_old
old_argv=read(OLD/'launch_receipt_v2.json')['wsl_arguments']
assert l['wsl_arguments']==[v.replace('expert_recovery_v1/','expert_recovery_v2/').replace('run_recovery_v2.sh','run_recovery.sh') for v in old_argv]
for name in ('launcher_parse.json','bash_parse.json'):assert read(BASE/name)['passed'] is True
proof_path=BASE/'path_preflight/report.json'
assert sha(proof_path)=='9ce1667c5998215b5db53b503dfbfb05c430a73054509da2e2112d9855934345'
proof=read(proof_path)
assert proof['passed'] and proof['actual_inherited_frozen_function_ast'] and proof['task_modules_imported'] is False
assert (proof['checked_sources'],proof['checked_inputs'],proof['model_calls'],proof['native_calls'],proof['replans'])==(20,44,0,0,0)
assert proof['request_sha256']==expected['request'] and proof['frozen_receipt_sha256']==expected['frozen']
for path,digest in want.items():
 mapped='/mnt/'+path[0].lower()+path[2:] if len(path)>2 and path[1]==':' else path
 assert proof['checked_sha256'][mapped]==digest
assert read(BASE/'path_preflight/exit.json')['raw_exit_code']==0
owner_path=OLD/'owner_completion_v2.json';assert sha(owner_path)=='a0eaedf23819e9187c79ea0640905f16cc89a250071c3bfc3c25192799d6902d'
owner=read(owner_path);assert owner['completion_accounting_passed'] and owner['processes_absent'] and owner['all_postrun_pins_exact']
assert not owner['requested_recovery_completed'] and owner['raw_python_exit_code']==1
assert len(owner['actual_work_counters'])==21 and all(v==0 for count in owner['actual_work_counters'].values() for v in count.values())
assert sha(args.independent_review)==args.independent_sha
ind=read(args.independent_review);assert ind['passed'] and ind['concrete_review_pass']
encoded=json.dumps(ind)
for digest in expected.values():assert digest in encoded
for name in ('execution_clearance.json','recovery_process_v1','owner_completion.json','ATTEMPT_STARTED','work_counters.json','nominal','initial_seed','outcome.json','failure.json','post_lifecycle_hold_5s'):
 assert not (BASE/name).exists(),'Preserve previous selected attempt: '+name
result=dict(passed=True,concrete_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_sha256=expected['request'],frozen_receipt_sha256=expected['frozen'],launch_receipt_sha256=expected['launch'],
 independent_review=dict(path=args.independent_review.as_posix(),sha256=args.independent_sha),
 checked_launch_pins=85,frozen_inputs=44,source_files=20,all_task_source_bytes_unchanged=True,
 copied_actual_boundary_unchanged=True,original_protocol_unchanged=True,prior_failed_attempt_work_all_zero=True,
 inherited_path_check=dict(path=proof_path.as_posix(),sha256=sha(proof_path)),
 actual_native_step_max=15680,private_work_budgets=r['protocol']['budgets'],
 corrected_actual_invocation_selected=True,actual_recovery_dispatched=False,
 labels_admissible=False,model_training_authorized=False,hardware_authorized=False,
 review_task_model_calls=0,review_native_steps=0,writer_sha256=sha(__file__))
out=OUT/'concrete_review.json'
with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,path=out.as_posix(),sha256=sha(out),pins=85)))
