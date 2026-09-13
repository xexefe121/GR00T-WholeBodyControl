"""Inspect saved failed initialization only; never create or execute a model."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'direct_target_causal_response_balanced_student_v1';FIT=BASE/'fit'
HELPER=NEW/'direct_target_context_pair_fit_independent_v1/source_draft_v3'
sys.path.insert(0,str(HELPER))
from audit_restoration import differences
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
request=read(BASE/'training_request.json');failure=read(FIT/'failure.json')
source_path=Path(request['subjects']['checkpoint']['path'])
assert sha(source_path)=='10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'
assert sha(BASE/'training_request.json')=='4c332bb060e48cceb1dac7010b330fd868a5c1d517133fb7506979e6a5c2d717'
assert failure['state']['stage']=='initializing' and failure['state']['completed_updates']==failure['state']['committed_loss_rows']==0
assert failure['state']['optimizer_step_counters']==[3000]*6
assert all(failure['state'][k] is False for k in ('optimizer_call_started','optimizer_call_returned','optimizer_synchronized'))
assert failure['preservation_errors']==[] and failure['all_frozen_inputs_unchanged'] is True
assert failure['error']=='''TypeError("dict() got multiple values for keyword argument 'condition'")'''
counts=failure['counts']
assert all(v==0 for v in counts['training'].values())
assert all(v==0 for c in counts['diagnostics'].values() for v in c.values())
assert all(counts[k]==0 for k in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'))
source=torch.load(source_path,map_location='cpu',weights_only=True)
initial=torch.load(FIT/'initialization.pt',map_location='cpu',weights_only=True)
failed=torch.load(FIT/'failed_state.pt',map_location='cpu',weights_only=True)
checks={}
for key in ('actor_state','optimizer_state'):
    checks['initial_'+key]=not differences(initial[key],source[key])
    checks['failed_'+key]=not differences(failed[key],source[key])
for key in ('feature_mean','feature_std'):checks['initial_'+key]=not differences(initial[key],source[key])
checks['initial_RNG']=not differences(initial['rng_after_restoration'],source['rng'])
checks['failed_RNG']=not differences(failed['rng'],source['rng'])
checks['failed_counters']=failed['counters']==counts
assert all(checks.values()),checks
assert not any((FIT/n).exists() for n in ('request.json','diagnostic_calls.jsonl','student_head.pt','student_head.onnx','optimization_completed.json'))
paths=[BASE/'training_request.json',BASE/'source_snapshot_v1/train_response_balanced.py',FIT/'failure.json',FIT/'initialization.pt',FIT/'failed_state.pt',source_path,HELPER/'audit_restoration.py']
report=dict(passed=True,saved_initialization_audit_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 failure='duplicate condition keyword in saved carried-request writer',checks=checks,
 source_actor_optimizer_normalization_RNG_exact=True,actual_fit_forward_calls=0,actual_fit_gradient_calls=0,
 actual_fit_updates=0,actual_fit_native_steps=0,actual_fit_optimizer_steps=[3000]*6,
 saved_audit_checkpoint_loads=3,model_instances_created=0,model_forward_calls=0,gradient_calls=0,native_steps=0,
 input_sha256={p.as_posix():sha(p) for p in paths},writer_sha256=sha(__file__),actual_fit_repeated=False)
path=OUT/'report.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,report_sha256=sha(path),fit_forward_calls=0,fit_updates=0,saved_checkpoint_comparisons=checks)))
