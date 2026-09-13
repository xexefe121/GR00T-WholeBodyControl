"""Bind the reviewed split calculation and unchanged fixed pair to one fresh launch."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
TASK=HERE.parent/'direct_target_causal_context_study_v2'
OLD=HERE.parent/'direct_target_causal_context_study_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
request=TASK/'training_request.json';receipt=TASK/'training_frozen_inputs.json';launcher=TASK/'run_fit_durable_v2.ps1'
assert sha(request)=='5a09d441f47e58f7cde8301aee72c23d6d0432c09cbdd37f49e4e6c3ac079cfe'
assert sha(receipt)=='de1f083af0bbcda9dead1e3714e3c586a9e564bee26879072006c27c7ad2c050'
assert sha(launcher)=='6f114a44e4958598271379ce57be5f553e28ff2c078bfef3da8d3c7759627c5d'
assert launcher.read_bytes()==(OLD/'run_fit_durable_v2.ps1').read_bytes()
assert (TASK/'verify_completed_v3.py').read_bytes()==(OLD/'verify_completed_v3.py').read_bytes()
assert not any((TASK/p).exists() for p in ('fit','fit_process_v2','training_clearance.json'))
prior_review=HERE.parent/'direct_target_causal_context_study_root_review_v2/review.json'
assert sha(prior_review)=='822395eb3f20588e7299fea0a4e8deaac4a82aa86f4f7cbd539ac5a21807d3a8'
source_review=HERE.parent/'direct_target_context_split_source_review_v1/review.json'
assert sha(source_review)=='f08211f05632207c27d43fda8d485b0c295f0d3349b156ddf5b82317e2f940be'
sr=read(source_review);assert sr['source_review_pass'] is True and sr['context_data_review_pass'] is True
req=read(request);oldreq=read(OLD/'training_request.json');frozen=read(receipt)
changed={k for k in set(req)|set(oldreq) if req.get(k)!=oldreq.get(k)}
assert changed=={'export_first_layer_execution','first_layer_execution','pending_before_actual_fit','previous_attempt_counts','previous_attempt_owner_sha256','previous_attempt_root','source_preparation_sha256','subjects'}
assert req['first_layer_execution']=='split_contiguous_1000_plus_323' and req['export_first_layer_execution']=='monolithic_float64_1323'
assert req['conditions']==['blinded','causal'] and req['updates_per_condition']==3000 and req['coefficient']==1.8188207859141674
for name,subject in oldreq['subjects'].items():
    if name!='context_preparation':assert req['subjects'][name]==subject,name
assert len(frozen['input_sha256'])==285 and len(frozen['source_sha256'])==20
assert sr['source_sha256']==frozen['source_sha256']
for path,digest in frozen['input_sha256'].items():assert sha(path)==digest,path
for name,digest in frozen['source_sha256'].items():assert sha(Path(frozen['source_directory'])/name)==digest,name
test=read(TASK/'launcher_v2_test.json')
assert test['passed'] is True and test['PSVersion'].startswith('5.1.')
assert test['actual_corrected_guard_passed'] is True and test['changed_coefficient_rejected'] is True
value=dict(read(prior_review))
value.update(prelaunch_review_pass=True,root_selected_single_pair=True,
    training_request_sha256=sha(request),frozen_receipt_sha256=sha(receipt),launcher_path=str(launcher),launcher_sha256=sha(launcher),
    subjects={role:dict(path=str(path),sha256=sha(path)) for role,path in [('training_request',request),('frozen_inputs',receipt),('launcher',launcher)]},
    source_sha256=frozen['source_sha256'],input_sha256=frozen['input_sha256'],
    source_preparation_sha256=sha(TASK/'source_preparation.json'),
    independent_source_review=dict(path=str(source_review),sha256=sha(source_review)),
    inherited_full_review=dict(path=str(prior_review),sha256=sha(prior_review)),
    reviewed=['Root read exact split first-layer forward: contiguous1000 and323 operands, original bias once, sum before unchanged ELU; six trainable tensors retain gradients.',
        'Root read complete driver diff: execution disclosures only, original initial gate and fixed objectives/updates/diagnostics retained.',
        'All data/schedule/normalization/hyperparameters/budgets unchanged; fresh output namespace preserves failed attempt.',
        'Actual split source independent review binds25 producer and17 independent synthetic CPU/CUDA checks.',
        'Corrected durable launcher and owner byte-identical to reviewed versions; actual Windows5.1 exact request guard passes.',
        'All285 inputs and20 actual snapshot sources rehashed; completed context proof reused without rerun.'],
    historical_producer_initial_GPU32_calls=1437,historical_producer_initial_GPU32_rows=367570,historical_producer_updates=0,
    split_execution_selected=True,initial_and_final_tolerance_rad=1e-5,controller_selected=False)
# Remove the old no-child repair description; this fresh study follows a saved
# initial diagnostic failure, not another metadata-only retry.
value.pop('repair_review',None)
review=HERE/'review.json';write(review,value)
clear=read(TASK/'training_clearance_draft.json');assert clear['approved'] is False
assert clear['request_sha256']==sha(request) and clear['frozen_receipt_sha256']==sha(receipt)
clear.update(approved=True,review_path=review.as_posix(),review_sha256=sha(review),
    selected_execution='one fresh split-first-layer matched pair; blinded then causal; no automatic retry')
write(TASK/'training_clearance.json',clear)
print(json.dumps({'review_sha256':sha(review),'clearance_sha256':sha(TASK/'training_clearance.json'),'approved':True,'dispatched':False}))
