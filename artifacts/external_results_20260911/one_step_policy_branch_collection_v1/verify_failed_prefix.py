"""Saved-only integrity check after progress-file rename failure; no runtime calls."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;OUT=BASE/'collection';sys.path.insert(0,str(BASE/'source_snapshot_v1'))
from branch_inputs import Inputs,read,exact
from collection_arrays import sha,atomic
request=read(BASE/'request.json');failure=read(OUT/'failure.json');manifest=read(OUT/'data/manifest.json');status=read(BASE/'process_status.json')
assert status['state']=='EXITED' and status['process_exit_code']==1
assert failure['error']=="PermissionError(13, 'Permission denied')" and "tmp.replace(path)" in failure['traceback']
p=failure['progress'];assert p['nominal_verified']==2969 and p['native_attempted']==p['native_returned']==29690
assert p['active_row']==2969 and p['policy_attempted']==p['labels_completed']==0
assert all(v==dict(attempted=0,returned=0) for v in p['calls'].values())
for key,digest in request['source_sha256'].items():assert sha(Path(request['source_directory'])/key)==digest,key
for key,digest in request['input_sha256'].items():assert sha(key)==digest,key
data={}
for key,spec in manifest['arrays'].items():
    path=OUT/'data'/spec['path'];assert sha(path)==spec['sha256'],key
    value=np.load(path,mmap_mode='r');assert list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'];data[key]=value
expected=np.arange(3054)<2969;exact(data['nominal_verified'],expected,'verified row mask')
exact(data['nominal_status'],expected.astype(np.int32),'native row status')
for key in ('valid_steps','attempted_steps','returned_steps'):exact(data['nominal_'+key],expected.astype(np.int32)*10,key)
assert not data['policy_status'].any() and not data['label_valid'].any()
for key in ('qpos','qvel','time','command_torque','actuator_force','start_integration','end_integration'):
    assert np.isnan(data['nominal_'+key][2969:]).all(),key
    assert np.isnan(data['policy_'+key]).all(),key
inputs=Inputs({k:Path(v) for k,v in request['paths'].items()})
for row,(dataset,control,index) in enumerate(inputs.rows[:2969]):
    e=inputs.expected(inputs.traces[dataset],control)
    for key,value in e.items():exact(data['nominal_'+key][row],value,f'prefix {row} {key}')
    exact(data['nominal_start_integration'][row],inputs.integration(dataset,control),f'start291 {row}')
    exact(data['nominal_end_integration'][row],inputs.integration(dataset,control+1),f'end291 {row}')
ledger=[json.loads(line) for line in (OUT/'native_rows.jsonl').read_text().splitlines()]
assert len(ledger)==2969 and [r['row'] for r in ledger]==list(range(2969))
assert all(r['phase']=='nominal' and r['report']['feasible'] and r['report']['first_failure'] is None and r['captured']==r['attempted']==r['returned']==10 for r in ledger)
assert not (OUT/'graph_calls.jsonl').exists() and not (OUT/'native_exceptions.jsonl').exists()
result=dict(passed=True,failed_before_next_native_call=True,verified_nominal_prefix=2969,recorded_native_steps=29690,remaining_nominal_rows=85,
    all_policy_rows_remaining=3054,remaining_native_step_ceiling=31390,remaining_graph_call_ceiling=12216,graph_calls=0,
    exact_saved_reference_samples_and_full291=True,all_source_and_input_hashes_unchanged=True,all_array_hashes_exact=True,
    no_uncommitted_physics_in_next_row=True,no_physics_or_graph_calls_in_audit=True,
    failure_sha256=sha(OUT/'failure.json'),manifest_sha256=sha(OUT/'data/manifest.json'),native_rows_sha256=sha(OUT/'native_rows.jsonl'),
    request_sha256=sha(BASE/'request.json'),source_sha256=sha(Path(__file__)),limitation='Owner saved-evidence verification; parent independent replay remains separate.')
assert not (BASE/'failed_prefix_verification.json').exists();atomic(BASE/'failed_prefix_verification.json',result);print(json.dumps(result))
