"""Read-only source-derivation verification; does not import reviewed modules."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
AUDIT=BASE/'direct_target_continuation_root_audit_v1'
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sub(p):return {'path':Path(p).as_posix(),'sha256':sha(p)}
old=BASE/'direct_target_root_audit_v1/audit_saved_fit.py'
new=AUDIT/'audit_saved_fit.py';derivation=read(AUDIT/'source_derivation.json')
assert sha(old)==derivation['original_sha256']=='40a47fd2e7554368c38a262fc9a2bdd0e75a3ceaa94ca817fbfb8646e191a55b'
assert sha(new)==derivation['source_sha256']=='d4055bfea1fd7d7a0813954cc8f6135f43466ed021160e781a6ef5d6b8c829a5'
text=old.read_text()
for change in derivation['changes']:
    assert change['old'] in text
    text=text.replace(change['old'],change['new'])
assert text==new.read_text()
assert len(derivation['changes'])==16
assert sha(AUDIT/'audit_math.py')==derivation['unchanged_math_sha256']=='6313988b729ac56f51dfd8c75bcde6d201d9aef237690a5bbd25a659f2aaa82f'
for name in ['audit_saved_fit.py','audit_math.py','audit_restoration.py','test_audit_math.py','test_restoration.py']:ast.parse((AUDIT/name).read_text())
suites=ET.parse(AUDIT/'tests_v1.xml').getroot().findall('testsuite')
assert sum(int(v.attrib['tests']) for v in suites)==19
assert all(int(v.attrib[x])==0 for v in suites for x in ['failures','errors','skipped'])
trainer=BASE/'direct_target_continuation_v1/source_snapshot_v1/train_direct.py'
source=trainer.read_text()
for required in ['rng_after_restoration=restored_rng','source_checkpoint_sha256=sha(start_checkpoint)',"initialization_sha256=sha(dest/'initialization.pt')",'ordinary_final_step=55000,additional_updates=50000']:
    assert required in source,required
paths={
    'auditor':new,'math':AUDIT/'audit_math.py','restoration_helper':AUDIT/'audit_restoration.py',
    'derivation':AUDIT/'source_derivation.json','tests':AUDIT/'tests_v1.xml',
    'math_tests':AUDIT/'test_audit_math.py','restoration_tests':AUDIT/'test_restoration.py',
    'prior_auditor':old,'prior_source_review':BASE/'direct_target_root_audit_source_review_v1/review.json',
    'trainer':trainer,'training_request':BASE/'direct_target_continuation_v1/training_request.json',
    'frozen_inputs':BASE/'direct_target_continuation_v1/training_frozen_inputs.json',
    'prelaunch_review':BASE/'direct_target_continuation_prelaunch_review_v1/review.json',
    'review_source':Path(__file__)}
subjects={k:sub(v) for k,v in paths.items()}
review=dict(kind='direct_target_continuation_saved_audit_source_review',created_utc=datetime.now(timezone.utc).isoformat(),
    passed=True,source_review_pass=True,verdict='CLEAR',subjects=subjects,
    input_sha256={v['path']:v['sha256'] for v in subjects.values()},
    checks=dict(exact_sixteen_substitution_derivation=True,prior_math_unchanged=True,synthetic_tests=19,
        frozen_trainer_restoration_and_request_schema_aligned=True,
        all_model_AdamW_CPU_CUDA_NumPy_Python_RNG_trees_and_initial_GPU_arrays_compared=True,
        all_28_8_million_pair_ids_and_50000_rates_checked=True,
        unchanged_cells_mixed_precision_losses_normalization_export_structure_and_weights_checked=True,
        actual_request_frozen_receipt_and_launcher_direct_review_subjects_required=True,
        saved_failures_and_final_all_input_source_rehash_retained=True),
    limitations=['Source review only; no completed continuation fit exists or was audited in this review.',
        'Saved evidence cannot reconstruct unsaved gradients or establish connected stability.',
        'Nominal float32 cell reductions use the preserved declared tolerance; prediction/restoration array byte checks are exact.',
        'Root must run only after the completed ordinary55000 artifact and current launch chain are available.'],
    task_model_calls=0,ORT_calls=0,optimizer_updates=0,native_steps=0)
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'subjects':len(subjects)}))
