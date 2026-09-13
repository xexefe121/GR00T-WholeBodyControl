"""Source-only review of unified65000 release gates; no evaluation execution."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_evaluation_v1')
OLD=BASE.parent/'direct_target_fp64_export_evaluation_v2/source_draft_v1'
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep=BASE/'source_preparation.json'
assert sha(prep)=='13c24f8987a1cd9e1c62bc33e7b9cafb62a8c728db4c4fa28f3b58c9ee716326'
record=json.loads(prep.read_text())
assert record['source_preparation_passed'] is True and record['synthetic_tests_passed']==55 and record['observed_exit_code']==0
assert record['changed_files']==['evaluation_gate.py','export_release_gate.py'] and record['added_files']==['test_release_gate.py']
assert len(record['unchanged_files'])==29 and len(record['source_sha256'])==32
for name,digest in record['source_sha256'].items():
    p=BASE/'source_draft_v1'/name;assert sha(p)==digest;ast.parse(p.read_text(encoding='utf-8'))
for name in record['unchanged_files']:assert (BASE/'source_draft_v1'/name).read_bytes()==(OLD/name).read_bytes()
for name in ('witness_binding.json','evaluation_binding.json','head_witness','nominal','post_lifecycle_hold_5s'):assert not (BASE/name).exists()
result=dict(source_review_pass=True,source_preparation_passed=True,passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
    preparation_sha256=sha(prep),source_sha256=record['source_sha256'],unchanged_runtime_files=29,changed_release_gate_files=2,added_test_files=1,
    findings=[],scope='Source-only unified ordinary65000 full58 same-weight FP64 release validation.',
    reviewed=['Exact29-module preservation includes native strict oracle, evaluator, direct1000 features, applied inverse prior/history, and separate WSL head witness.',
        'New release requires completed65000/fresh AdamW10000, original1e-5 preclamp FP64 gate, positive fixed calibration and exact output subjects.',
        'Full58 data audit and corrected owner, independent fit evidence/export pass, owner exit/pins/absence, final release review and actual source hashes use13 literal subjects.',
        'Observed55 existing/native stub and receipt mutation tests pass; no actual model/native calls.'],
    direct_subject_sha256=dict(evaluator=record['source_sha256']['evaluate_direct_target_student.py'],witness=record['source_sha256']['head_activation_witness.py'],
        evaluation_gate=record['source_sha256']['evaluation_gate.py'],export_release_gate=record['source_sha256']['export_release_gate.py']),
    actual_head_bound=False,actual_head_calls=0,native_steps=0,optimizer_updates=0,launch_cleared=False,
    canonical_evaluation_cleared=False,hardware_authorized=False,
    remaining='Actual ordinary-final fit/root audit/owner/release subjects and concrete witness/canonical binding/launcher reviews remain required.',
    review_source_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print((OUT/'review.json').as_posix(),sha(OUT/'review.json'))
