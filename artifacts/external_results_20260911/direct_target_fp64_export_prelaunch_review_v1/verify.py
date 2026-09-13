"""Read-only final source/identity review; no task graph or native calls."""
import ast,hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'direct_target_fp64_export_v1';O=Path(__file__).parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
request=B/'export_request.json';frozen=B/'export_frozen_inputs.json';launcher=B/'run_export_durable_v1.ps1'
assert sha(request)=='f6d4bd17cf2231f010126052b9fa02428fd3e40225707e8df6354038b0e5bc68'
assert sha(frozen)=='64ce9b8d9fa4cb8e654aef84ddd19dd9089d088fe146a28a5167d4da7ab0e346'
r=read(request);f=read(frozen);pins={}
assert f['export_request_sha256']==sha(request) and len(f['source_sha256'])==7 and len(f['input_sha256'])==191
assert Path(f['source_directory']).resolve()==(B/'source_snapshot_v1').resolve()
for path,digest in f['input_sha256'].items():assert sha(path)==digest;pins[path]=digest
for name,digest in f['source_sha256'].items():
 p=B/'source_snapshot_v1'/name;assert sha(p)==sha(B/'source_draft_v1'/name)==digest;ast.parse(p.read_text());pins[str(p)]=digest
for name in ('direct_contract.py','direct_data.py','direct_diagnostics.py'):
 assert sha(B/'source_snapshot_v1'/name)==sha(N/'direct_target_continuation_v1/source_snapshot_v1'/name)
assert r['root_selected'] is True and r['kind']=='one_same_weight_fp64_export_validation'
assert r['ordinary_final_step']==55000 and r['parity_tolerance_rad']==1e-5
assert r['execution_dtype']=='float64' and r['public_input_dtype']==r['public_output_dtype']=='float32'
assert r['backend_order']==['CPU64','GPU64','ORT64'] and r['corpus_rows']==[9904,140622,3054]
assert r['batch_size']==256 and r['per_backend_batches']==[39,550,12]
budget=dict(Torch_calls=1202,Torch_rows=307160,ORT_calls=601,ORT_rows=153580,trace_forward_calls=0,optimizer_updates=0,BFM_calls=0,native_steps=0)
assert r['budgets']==budget and r['full_drift_arrays']==27 and r['canonical_evaluation_authorized'] is False
training_review=read(r['subjects']['training_review']['path']);assert training_review['training_review_pass'] is True and training_review['dataset_review_pass'] is True
assert training_review['export_review_pass'] is False
for key,subject in r['subjects'].items():
 assert f['input_sha256'][subject['path']]==subject['sha256']
 logical={'training_frozen_inputs':'training_manifest','training_audit':'root_training_audit'}.get(key,key)
 if logical in training_review['direct_subject_sha256']:assert training_review['direct_subject_sha256'][logical]==subject['sha256']
original=read(N/'direct_target_continuation_v1/training_request.json');assert r['paths']==original['paths'] and r['runtime']==original['runtime']
root_audit=read(r['subjects']['training_audit']['path'])
for backend,corpora in r['old_outputs'].items():
 for corpus,path in corpora.items():
  expected=N/'direct_target_continuation_v1/fit'/('final_'+backend+'_'+corpus+'.npy');assert Path(path)==expected
  assert sha(path)==root_audit['input_sha256'][str(expected)]
test=read(B/'frozen_tests/report.json');assert test['passed'] is True and test['exit_code']==0 and test['source_sha256']==f['source_sha256']
assert 'Ran 10 tests' in (B/'frozen_tests/stderr.log').read_text() and (B/'frozen_tests/stderr.log').read_text().strip().endswith('OK')
assert read(B/'saved_data_preflight.json')['passed'] is True
assert not (B/'export').exists() and not (B/'export_clearance.json').exists()
for name in ('start.json','child.json','running.lock','stdout.log','stderr.log','exit.json'):
 assert not (B/'export_process'/name).exists()
subjects={'export_request':{'path':str(request),'sha256':sha(request)},'frozen_inputs':{'path':str(frozen),'sha256':sha(frozen)},'launcher':{'path':str(launcher),'sha256':sha(launcher)}}
for key,value in r['subjects'].items():subjects[key]=value
report={'kind':'one_selected_same55000_fp64_validation_prelaunch_review','passed':True,'prelaunch_review_pass':True,
 'export_request_sha256':sha(request),'frozen_receipt_sha256':sha(frozen),'subjects':subjects,
 'direct_subject_sha256':{k:v['sha256'] for k,v in subjects.items()},'source_sha256':f['source_sha256'],'input_sha256':pins,
 'all198_source_input_pins_verified':True,'ordinary_final_step':55000,'budgets':budget,'findings':[],
 'checks':['Fixed three original corpora and saved source inputs, weights, normalization, failed FP32 outputs and actual positive training-only/root evidence all bind exact subjects.',
 'Source retains exact f64 promotions and public f32 contract; independent CPU/GPU Torch native double ELU and ORT bounded Exp/Where have one fixed pass each, no tracing, optimizer or native calls.',
 'Per-call state clears stale returns; attempted/returned/synchronized/verified prefixes and raw output evidence survive injected wrong shape, nonfinite and second-call failures; all memmaps and ledgers flushed.',
 'All nine new preclamp comparisons use unchanged1e-5 tolerance. All27 full drift arrays and six clipping-mask counts retained; unchanged nominal/velocity/physical metrics computed without added graph calls.',
 'Final request/receipt/clearance and source/input rehash precedes completed report. Numerical failure preserves outputs and reports no release; metadata failure produces nonzero wrapper accounting.',
 'Durable launcher uses exact source/runtime, CreateNew locks, hidden child, retained native process handle, known exit0 plus report/count/hash checks and final rehash; no automatic retry.',
 'Ten owner synthetic/stub checks and saved-array preflight pass against final source. Reviewed without additional task calls.'],
 'selected_validation_launch_cleared':True,'export_review_pass':False,'witness_launch_cleared':False,'canonical_launch_cleared':False,
 'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0}
with (O/'review.json').open('x') as file:json.dump(report,file,indent=2);file.write('\n')
print(json.dumps({'prelaunch_review_pass':True,'review_sha256':sha(O/'review.json'),'launcher_sha256':sha(launcher)}))
