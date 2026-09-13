"""Root metadata/hash review. Never initializes a model or executes recovery."""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_width512_expert_recovery_v1'
SOURCE=BASE/'source_snapshot_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def normalized(p):return Path(p).resolve().as_posix().casefold()

parser=argparse.ArgumentParser()
parser.add_argument('--independent-review',type=Path,required=True)
parser.add_argument('--independent-sha',required=True)
args=parser.parse_args()
rp,fp,lp=[BASE/name for name in ('execution_request.json','frozen_inputs.json','launch_receipt.json')]
expected={'request':'c3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8',
          'frozen':'954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1',
          'launch':'7012ee94d666085f2c48256949a94f34392ccac6388e2ee928248c383ad536c0'}
for key,p in [('request',rp),('frozen',fp),('launch',lp)]:assert sha(p)==expected[key],key
r,f,l=read(rp),read(fp),read(lp)
assert r['root_selected'] is True and r['actual_recovery_run'] is False
assert l['root_concrete_clearance_required'] is True and l['dispatch_authorized'] is False
assert f['request_sha256']==l['request_sha256']==expected['request']
assert l['frozen_receipt_sha256']==expected['frozen']
prep_path=BASE/'source_preparation_v2.json'
source_review_path=NEW/'direct_target_width512_expert_recovery_review_v1/source_review.json'
boundary_path=NEW/'direct_target_width512_expert_recovery_review_v1/boundary_review.json'
assert sha(prep_path)=='1f7211ba3c08e985daf191b2c106682b97e6336db67b5a805e98d6b604f4b573'
assert sha(source_review_path)=='771a5ec79b84039d805533cd09446f1eb8c32370ec0b148e0c52919af05be28f'
assert sha(boundary_path)=='20bb1e718966c902c121e0812b1eb4f0ea28dea8601ae5cf172425c32247d06e'
prep,source_review,boundary=read(prep_path),read(source_review_path),read(boundary_path)
assert source_review['source_review_pass'] is True and boundary['boundary_review_pass'] is True and boundary['checks']==367
assert f['source_sha256']==prep['source_sha256']==source_review['source_sha256']
assert len(f['source_sha256'])==20 and len(prep['unchanged_qualified_source_sha256'])==14
actual_sources={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
assert actual_sources==f['source_sha256']
for name,digest in prep['unchanged_qualified_source_sha256'].items():assert actual_sources[name]==digest
sys.path.insert(0,str(SOURCE))
from recovery_contract import protocol
assert r['protocol']==f['protocol']==prep['protocol']==source_review['protocol']==protocol()
p=r['protocol']
assert (p['initial_global_control'],p['prefix_controls'],p['MPC_controls'],p['terminal_BFM_controls'],p['conditional_hold_controls'])==(251,251,1018,300,250)
assert p['actual_native_step_max']==15680 and p['prefix_native_steps']==2510 and p['combined_native_steps']==18190
assert len(p['plan_controls'])==204 and p['plan_controls'][0]==251 and p['plan_controls'][-1]==1266 and p['final_MPC_commit']==3
assert p['budgets']['native_private_step']['attempted_units_max']==200180
assert p['budgets']['batch_fd_step']['attempted_units_max']==150552000
assert p['budgets']['batch_line_step']['attempted_units_max']==20933100
assert p['labels_admissible'] is False and p['model_training_authorized'] is False and p['hardware_authorized'] is False
assert sha(BASE/'inputs/precontrol251.npz')=='aa8cd94cdefc77b96624c75b4825a1133d805b8fd8be0717dc765e4c8f00c3c3'
assert sha(BASE/'inputs/actual_prefix251.npz')=='8a755a3bf1acd751c456f04805017f8dd70a025988e61c11e4690a07d7f48649'
assert sha(BASE/'inputs/selection_receipt.json')==boundary['selection_subject']['sha256']=='1c71c7c8d69b491a56e27b3e2ca8854e6287c160473c41a5f33fc28c8d0d2596'
pins=l['input_sha256'];assert len(pins)==74 and len(f['input_sha256'])==38
norm={normalized(path):digest for path,digest in pins.items()}
assert len(norm)==len(pins),'Duplicate normalized launch path'
for path,digest in pins.items():assert sha(path)==digest,path
for path,digest in f['input_sha256'].items():assert norm[normalized(path)]==digest
for role,sub in r['subjects'].items():assert norm[normalized(sub['path'])]==sub['sha256'],role
for name,digest in f['source_sha256'].items():assert norm[normalized(SOURCE/name)]==digest
assert sha(BASE/'run_recovery_durable.ps1')==l['launcher_sha256']=='f89670e1bd4749ae7b8f85a3d124235b5c2259a49d6e7e6f05d8f39945bcb836'
assert sha(BASE/'verify_recovery_completed.py')=='f06fb5e730cdb3416c058719f91d27ad17ccc07784538371cd6f72c500e263b8'
lb='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1'
shell='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
assert l['wsl_arguments']==['-d','Ubuntu-22.04','--cd','/','--','bash',shell,'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+lb+'/source_snapshot_v1','bash',lb+'/run_recovery.sh']
assert r['environment']['PYTHONPATH']==lb+'/source_snapshot_v1'
assert b'\r' not in (BASE/'run_recovery.sh').read_bytes()
assert read(BASE/'launcher_parse.json')['passed'] is True
assert sha(args.independent_review)==args.independent_sha
independent=read(args.independent_review)
assert independent.get('passed') is True
# The exact independent record is read by root before this check; require it to
# include all current concrete subjects, rather than accepting a source-only pass.
serialized=json.dumps(independent,sort_keys=True)
for value in expected.values():assert value in serialized,'Independent review missing concrete subject'
for name in ('execution_clearance.json','ATTEMPT_STARTED','initial_seed','nominal','post_lifecycle_hold_5s','outcome.json','failure.json','work_counters.json','recovery_process_v1','owner_completion.json'):
    assert not (BASE/name).exists(),'Preserve prior attempt: '+name
result=dict(passed=True,concrete_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_sha256=expected['request'],frozen_receipt_sha256=expected['frozen'],launch_receipt_sha256=expected['launch'],
 source_preparation_sha256=sha(prep_path),source_review_sha256=sha(source_review_path),boundary_review_sha256=sha(boundary_path),
 independent_concrete_review=dict(path=args.independent_review.resolve().as_posix(),sha256=args.independent_sha),
 checked_launch_pins=74,frozen_inputs=38,source_files=20,unchanged_expert_modules=14,
 selected_scope='One prespecified actual width81000 pre251 expert recovery, exact full remaining1318 controls plus conditional250hold',
 actual_native_step_max=15680,preserved_prefix_native_steps=2510,private_work_budgets=p['budgets'],
 source_and_hold_qualification_pending=True,labels_admissible=False,model_training_authorized=False,hardware_authorized=False,
 actual_recovery_dispatched=False,review_task_model_calls=0,review_native_steps=0,writer_sha256=sha(__file__))
with (OUT/'concrete_review.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,path=(OUT/'concrete_review.json').as_posix(),sha256=sha(OUT/'concrete_review.json'),pins=74)))
