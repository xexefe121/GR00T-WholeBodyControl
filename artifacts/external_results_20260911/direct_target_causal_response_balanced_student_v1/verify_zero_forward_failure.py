"""Saved-only verification of zero-forward metadata failure; no model construction."""
from pathlib import Path
import hashlib,json,sys
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_snapshot_v1'))
from restoration_support import exact_saved
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
request=read(BASE/'training_request.json')
paths=dict(source=Path(request['subjects']['checkpoint']['path']),initial=BASE/'fit/initialization.pt',
    failed=BASE/'fit/failed_state.pt',failure=BASE/'fit/failure.json',exit=BASE/'fit_process_v1/exit.json')
pins={key:sha(path) for key,path in paths.items()}
source=torch.load(paths['source'],map_location='cpu',weights_only=True)
initial=torch.load(paths['initial'],map_location='cpu',weights_only=True)
failed=torch.load(paths['failed'],map_location='cpu',weights_only=True)
failure=read(paths['failure']);exit=read(paths['exit'])
for key in ('actor_state','optimizer_state'):
    exact_saved(initial[key],source[key],'initial '+key);exact_saved(failed[key],source[key],'failed '+key)
for key in ('feature_mean','feature_std'):exact_saved(initial[key],source[key],key)
exact_saved(initial['rng_after_restoration'],source['rng'],'initial RNG')
exact_saved(failed['rng'],source['rng'],'failed RNG')
assert initial['optimizer_start_step']==3000 and initial['ordinary_start_step']==68000
assert failure['state']['completed_updates']==failure['state']['committed_loss_rows']==0
assert failure['state']['optimizer_step_counters']==[3000]*6
assert not failure['state']['optimizer_call_started'] and not failure['state']['optimizer_call_returned']
assert failure['preservation_errors']==[] and failure['all_frozen_inputs_unchanged'] is True
counts=failure['counts']
assert all(value==0 for value in counts['training'].values())
assert all(value==0 for record in counts['diagnostics'].values() for value in record.values())
assert all(counts[key]==0 for key in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls'))
assert exit['raw_python_exit_code']==exit['exit_code']==1 and exit['all_postrun_pins_exact'] is True
assert not (BASE/'fit/student_head.pt').exists()
for key,path in paths.items():assert sha(path)==pins[key]
result=dict(passed=True,source_checkpoint_sha256=pins['source'],subject_sha256=pins,
    all_six_actor_and_AdamW_states_exact=True,all_six_optimizer_steps=3000,initial_and_failed_RNG_exact=True,
    initial_normalization_exact=True,task_forward_calls=0,optimizer_updates=0,native_steps=0,
    audit_model_constructions=0,audit_forward_calls=0,source_loads_saved_only=True,
    failure="dict() got multiple values for keyword argument 'condition'",automatic_retry=False,
    source_sha256=sha(__file__))
with (BASE/'zero_forward_failure_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(BASE/'zero_forward_failure_verification.json'))))
