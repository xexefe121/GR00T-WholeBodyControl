import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_causal_response_balanced_student_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='1d9635f256508c34e4a8ffc7ba893da6c8fff755216fc878a94d8162b2135d2c'
prep=read(prep_path);source=Path(prep['source_directory'])
assert prep['source_preparation_passed'] is True and prep['synthetic_CPU_tests_passed']==35
assert len(prep['source_sha256'])==25
for name,digest in prep['source_sha256'].items():assert sha(source/name)==digest,name
prior=NEW/'direct_target_causal_context_study_v2/source_snapshot_v1'
for name,digest in prep['original_modules_byte_exact'].items():assert sha(prior/name)==digest==sha(source/name)
for name,digest in prep['reviewed_math_modules_byte_exact'].items():assert sha(NEW/'direct_target_response_balance_math_v1/source_prepared_v1'/name)==digest==sha(source/name)
for sub in prep['subjects'].values():assert sha(sub['path'])==sub['sha256']
warm=NEW/'direct_target_response_warm_source_review_v1/review.json'
assert sha(warm)=='b9970d61ddcca81d8072a5b3b925d9e59fb049ceaf27f0f02cfb4a1a99aec15d'
assert read(warm)['warm_training_source_review_pass'] is True
old=(source/'context_promoted.py').read_text()
expected=old.replace('68000','71000').replace('Exact paired ordinary71000 subject required','Exact warm ordinary71000 subject required').replace('Paired context architecture changed','Causal context architecture changed')
assert expected==(source/'response_promoted.py').read_text()
result=dict(source_review_pass=True,data_export_source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(prep_path),source_sha256=prep['source_sha256'],warm_review_sha256=sha(warm),writer_sha256=sha(__file__),
 original_modules_byte_exact=14,reviewed_math_modules_byte_exact=6,promoted_export_math_unchanged=True,
 reviewed_semantics=['all input paths and completed qualification subjects frozen before task arrays/model calls',
 'saved context and normalization arrays reused with manifest and data-identity checks; no new context moments or rows',
 'saved fixed 3000 center/axis schedule validated; original data loader and causal feature assembly unchanged',
 'initial GPU32 preclamp 1e-5 rad gate retained; byte equality diagnostic only',
 'ordinary71000 checkpoint saved before exact same-weight FP64 export; unchanged monolithic export math',
 'original and balanced full-state objective and 54-cell arrays separately named',
 'complete five-backend diagnostics, parity and drift retained with fixed row/call budgets',
 'failure retains endpoint, active evidence, optimizer counters and committed loss rows; no automatic retry'],
 actual_task_checkpoint_loads=0,actual_task_arrays_loaded=0,task_model_calls=0,task_gradient_calls=0,native_steps=0,
 actual_fit_selected=False,concrete_request_launcher_review_pending=True)
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(source_review_pass=True,path=path.as_posix(),sha256=sha(path))))
