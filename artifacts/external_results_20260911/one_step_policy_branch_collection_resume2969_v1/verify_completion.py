"""Read-only owner completion/hash/count inspection, zero models or dynamics."""
import json,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;OUT=BASE/'collection';DATA=OUT/'data'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
request=read(BASE/'request.json');report=read(OUT/'report.json');manifest=read(DATA/'manifest.json');status=read(BASE/'process_status.json')
assert status['state']=='EXITED' and status['raw_wsl_exit_code']==status['process_exit_code']==0 and status['launcher_error'] is None
assert report['completed'] and report['passed'] and manifest['complete']
assert report['rows']==report['nominal_verified']==report['policy_branches']==3054
assert report['continuation_attempt']==2 and report['prefix_nominal_rows']==2969 and report['prefix_native_steps']==29690
assert report['current_attempt_native_steps']==report['native_returned']-29690<=31390
restoration=report['restoration'];assert restoration['passed']
with (OUT/'native_rows.jsonl').open('rb') as f:prefix=f.read(restoration['native_ledger_prefix_bytes'])
assert hashlib.sha256(prefix).hexdigest()==restoration['native_ledger_prefix_sha256']
assert report['request_sha256']==manifest['request_sha256']==sha(BASE/'request.json')
assert report['manifest_sha256']==sha(DATA/'manifest.json')
for key,digest in request['source_sha256'].items():assert sha(Path(request['source_directory'])/key)==digest,key
for key,digest in request['input_sha256'].items():assert sha(key)==digest,key
arrays={}
for key,spec in manifest['arrays'].items():
    p=DATA/spec['path'];assert p.resolve().parent==DATA.resolve()
    assert sha(p)==spec['sha256'],key
    value=np.load(p,mmap_mode='r',allow_pickle=False);assert list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'];arrays[key]=value
for key,spec in manifest['evidence_files'].items():assert sha(DATA/spec['path'])==spec['sha256'],key
for key in ('native_rows','graph_calls'):
    assert report[key+'_sha256']==manifest[key+'_sha256']==sha(OUT/(key+'.jsonl'))
native=[json.loads(line) for line in (OUT/'native_rows.jsonl').read_text().splitlines()]
graphs=[json.loads(line) for line in (OUT/'graph_calls.jsonl').read_text().splitlines()]
assert len(native)==6108 and len({(row['phase'],row['row']) for row in native})==6108
assert len(graphs)==report['total_graph_calls']
for name,count in report['graph_calls'].items():
    selected=[row for row in graphs if row['graph']==name]
    assert len(selected)==count['attempted']==count['returned']
    assert [row['attempt'] for row in selected]==list(range(1,len(selected)+1))
assert arrays['nominal_verified'].all() and np.all(arrays['nominal_valid_steps']==10)
assert np.all(arrays['nominal_status']==1) and np.all(np.isin(arrays['policy_status'],[1,2]))
valid=arrays['label_valid'];assert int(valid.sum())==report['labels_valid']==int((arrays['policy_status']==1).sum())
assert int((~valid).sum())==report['strict_failed']
for key in ('endpoint_features','endpoint_base_target','label_fixed_map_target','label_residual_rad','endpoint_head_output'):
    assert np.isfinite(arrays[key][valid]).all() and np.isnan(arrays[key][~valid]).all()
assert int(arrays['nominal_attempted_steps'].sum()+arrays['policy_attempted_steps'].sum())==report['native_attempted']
assert int(arrays['nominal_returned_steps'].sum()+arrays['policy_returned_steps'].sum())==report['native_returned']
assert read(OUT/'query250_actual251_calibration.json')['passed']
by_dataset=[]
for d in range(3):
    mask=arrays['dataset']==d;ok=mask&valid;error=arrays['endpoint_raw_proposal'][ok]-arrays['label_fixed_map_target'][ok]
    by_dataset.append(dict(dataset=d,requested=int(mask.sum()),valid=int(ok.sum()),strict_failed=int((mask&~valid).sum()),
        endpoint_raw_vs_fixed_map_rmse_rad=float(np.sqrt(np.mean(error**2))) if len(error) else None))
failed=[]
for row in native:
    if row['phase']=='policy' and not row['report']['feasible']:
        i=row['row'];failed.append(dict(row=i,dataset=int(arrays['dataset'][i]),control=int(arrays['start_control'][i]),valid_steps=int(arrays['policy_valid_steps'][i]),failure=row['report']['first_failure']))
result=dict(passed=True,report_sha256=sha(OUT/'report.json'),manifest_sha256=sha(DATA/'manifest.json'),request_sha256=sha(BASE/'request.json'),
    process_exit_code=0,source_hashes=len(request['source_sha256']),input_hashes=len(request['input_sha256']),array_hashes=len(arrays),
    continuation_attempt=2,prefix_native_steps=29690,current_attempt_native_steps=report['current_attempt_native_steps'],original_ledger_prefix_byteexact=True,
    all_pins_unchanged=True,native_rows=len(native),graph_rows=len(graphs),native_steps=report['native_returned'],labels=report['labels_valid'],
    by_dataset=by_dataset,failed_rows=failed,zero_new_model_calls=True,zero_new_physics=True,independent_root_physics_and_label_audits_still_required=True)
path=BASE/'completion_verification.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='failed_rows'}))
