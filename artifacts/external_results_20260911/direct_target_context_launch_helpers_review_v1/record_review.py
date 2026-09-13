"""Record bounded helper source review, no actual binding or invocation."""
from pathlib import Path
import hashlib,json,ast,xml.etree.ElementTree as ET
ROOT=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');BASE=ROOT/'direct_target_causal_context_evaluation_v1';DEST=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=read(BASE/'launch_helper_preparation_v1.json')
assert sha(BASE/'launch_helper_preparation_v1.json')=='8b1e27e51cb073234d20cafb8a0199ba46b115b30fd2786b392db619bfcc175f'
assert prep['source_preparation_passed'] is True and prep['preparation_only'] is True
for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest;ast.parse((BASE/name).read_text(encoding='utf-8-sig'))
for name in prep['unchanged_helper_files']:assert sha(ROOT/'direct_target_full_state_evaluation_v1'/name)==prep['helper_sha256'][name]
for name,digest in prep['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
for role in ('runtime_source_review','corrected_baseline_helper_review'):
    s=prep[role];assert sha(s['path'])==s['sha256'] and read(s['path'])['passed'] is True
assert read(prep['runtime_source_review']['path'])['source_sha256']==prep['source_sha256']
for name,digest in prep['evidence_sha256'].items():assert sha(BASE/name)==digest
inventory=read(BASE/'runtime_inventory.json')
assert inventory['source_sha256']==prep['source_sha256'] and inventory['recursive_training_hashes'] is False
assert len(inventory['files'])==5192 and inventory['total_files']==5192
assert any(e['path'].endswith('/original_bfm_entry250_v1/entry250/trace.npz') for e in inventory['files'])
assert prep['tests']==dict(tests=42,failures=0,errors=0,skipped=0)
suites=ET.parse(BASE/'launch_helper_tests_v2.xml').getroot().findall('testsuite')
assert sum(int(s.get('tests','0')) for s in suites)==42 and all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
for name in ('release_reviews.json','witness_binding.json','evaluation_binding.json','head_witness','nominal'):assert not (BASE/name).exists()
result=dict(passed=True,source_review_pass=True,preparation_only=True,helper_sha256=prep['helper_sha256'],source_sha256=prep['source_sha256'],
 preparation_sha256=sha(BASE/'launch_helper_preparation_v1.json'),runtime_source_review=prep['runtime_source_review'],
 corrected_baseline_helper_review=prep['corrected_baseline_helper_review'],evidence_sha256=prep['evidence_sha256'],
 scope=['Eight additive helper files; three inherited helper/test files byte-identical. Runtime arithmetic relies on the separately qualified36-source review.',
  'Mandatory explicit blinded/causal selection, both completed condition reports,18 exact subject roles, actual independent audit and authoritative condition-specific owner/final release.',
  'Shared normalization/context alignment and selected endpoint paths, fresh one-call WSL witness, same-head/same-condition witness-to-canonical binding, original1569 plus conditional250.',
  'Original hidden durable process, full input hashes, captured handle before wait, known raw/diagnostic exits, explicit slash normalization and CreateNew run lock. No automatic retry.',
  'Runtime inventory metadata/provenance checked, without repeating5192 runtime-file hashes. Actual final binding/launcher review must verify real released artifacts and generated command.'],
 tests=dict(owner_stub_tests=42,tests_repeated_by_this_review=0),no_condition_selected=True,actual_bindings_created=False,
 excluded_unexecuted_helpers=prep['excluded_unexecuted_owner_drafts'],model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,controller_cleared=False)
with (DEST/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=(DEST/'review.json').as_posix(),sha256=sha(DEST/'review.json'),passed=True)))
