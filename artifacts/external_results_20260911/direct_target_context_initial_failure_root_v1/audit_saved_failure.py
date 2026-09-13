"""Recompute failed initial drift and exact zero-update state using saved files only."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'direct_target_causal_context_study_v1'
FIT=BASE/'fit/blinded'
OLD=HERE.parent/'direct_target_full_state_student_v1/fit'
pins={}
def bind(path):
    path=Path(path);h=hashlib.sha256(path.read_bytes()).hexdigest();pins[path.as_posix()]=h;return path
def read(path):return json.loads(bind(path).read_text(encoding='utf-8-sig'))
def load(path):return torch.load(bind(path),map_location='cpu',weights_only=True)
def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    return type(a)==type(b) and a==b
torch.set_num_threads(1)
failure=read(FIT/'failure.json');restore=read(FIT/'restoration.json')
owner=read(BASE/'owner_completion_verification_v3.json')
assert owner['accounting_passed'] is True and owner['paired_completion_passed'] is False
assert owner['processes_absent'] is True and owner['raw_python_exit_code']==owner['exit_code']==1
assert failure['state']['completed_updates']==0 and failure['state']['optimizer_step_counters']==[0]*6
assert all(v==0 for v in failure['counts']['training'].values())
initial=load(FIT/'initialization.pt');failed=load(FIT/'failed_state.pt');source=load(OLD/'student_head.pt')
assert exact(initial['actor_state'],failed['actor_state']) and exact(initial['optimizer_state'],failed['optimizer_state'])
assert initial['optimizer_state']['state']=={} and exact(initial['rng_after_restoration'],failed['rng'])
for name,value in source['actor_state'].items():
    actual=initial['actor_state'][name]
    if name=='0.weight':
        assert torch.count_nonzero(actual[:,1000:])==0
        actual=actual[:,:1000]
    assert exact(actual,value),name
assert exact(initial['feature_mean'][:1000],source['feature_mean'])
assert exact(initial['feature_std'][:1000],source['feature_std'])
with np.load(bind(BASE/'fit/shared/normalization.npz'),allow_pickle=False) as z:span=z['joint_span'].astype(np.float64)
results={};expected_ledger=[]
for name,n in [('nominal',9904),('full_state',354612),('physical',3054)]:
    a=np.load(bind(FIT/('initial_GPU32_'+name+'.npy')),allow_pickle=False)
    b=np.load(bind(OLD/('final_GPU32_'+name+'.npy')),allow_pickle=False)
    assert a.dtype==b.dtype==np.float32 and a.shape==b.shape==(n,23)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta=(a.astype(np.float64)-b.astype(np.float64))*span
    result=dict(max_preclip_error_rad=float(np.max(np.abs(delta))),RMS_preclip_error_rad=float(np.sqrt(np.mean(delta*delta))),byte_equal=a.tobytes()==b.tobytes())
    recorded=restore['initial_drift']['corpora'][name]
    assert result['max_preclip_error_rad']==recorded['max_preclip_error_rad']
    assert np.isclose(result['RMS_preclip_error_rad'],recorded['RMS_preclip_error_rad'],rtol=1e-12,atol=1e-18)
    assert result['byte_equal']==recorded['byte_equal']
    results[name]=result
    for start in range(0,n,256):expected_ledger.append(dict(backend='initial_GPU32',corpus=name,start=start,stop=min(n,start+256),returned=True,synchronized=True,verified=True))
ledger=[json.loads(line) for line in bind(FIT/'diagnostic_calls.jsonl').read_text().splitlines()]
assert ledger==expected_ledger and len(ledger)==1437
assert sum(x['stop']-x['start'] for x in ledger)==367570
assert max(v['max_preclip_error_rad'] for v in results.values())>1e-5
assert not (BASE/'fit/causal').exists() and not (BASE/'fit/paired_report.json').exists()
for path,digest in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
report=dict(saved_failure_accounting_passed=True,initial_gate_passed=False,
    exact_unchanged_initial_actor_optimizer_RNG=True,original_weights_and_normalization_exact=True,
    added_columns_zero=True,producer_initial_GPU32_calls=1437,producer_initial_GPU32_rows=367570,
    producer_training_updates=0,producer_training_forward_calls=0,producer_native_steps=0,
    causal_condition_started=False,corpora=results,input_sha256=pins,
    auditor_model_calls=0,auditor_gradient_calls=0,auditor_native_steps=0,
    limitation='Saved evidence establishes numerical drift and zero updates; split-kernel correction still requires a new unchanged-threshold actual initial gate.')
out=HERE/'report.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'saved_failure_accounting_passed':True,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'corpora':results}))
