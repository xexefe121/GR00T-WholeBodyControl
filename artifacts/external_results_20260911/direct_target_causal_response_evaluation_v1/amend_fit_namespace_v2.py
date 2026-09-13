"""Preserve unbound helper v1 and point only the release bindings to repaired fitv2."""
from pathlib import Path
import json,hashlib,shutil
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
for name in ('release_reviews.json','witness_binding.json','evaluation_binding.json','head_witness','nominal'):
    if (BASE/name).exists():raise ValueError('Cannot amend a bound or executed evaluator.')
prior=read(BASE/'launch_helper_preparation_v1.json')
for name,digest in prior['helper_sha256'].items():assert sha(BASE/name)==digest
source_map={p.relative_to(BASE/'source_draft_v1').as_posix():sha(p) for p in (BASE/'source_draft_v1').rglob('*.py')}
assert source_map==prior['source_sha256']
preserve=BASE/'launch_helpers_preserved_v1';preserve.mkdir(exist_ok=False)
for name in list(prior['helper_sha256'])+['launch_helper_preparation_v1.json','launch_helper_derivation.json','launch_helper_tests_v2.xml','powershell_path_normalization_v1.json']:
    shutil.copyfile(BASE/name,preserve/name)
path=BASE/'freeze_final_package.py';text=path.read_text(encoding='utf-8')
assert text.count("FIT=NEW/'direct_target_causal_response_balanced_student_v1'")==1
text=text.replace("FIT=NEW/'direct_target_causal_response_balanced_student_v1'","FIT=NEW/'direct_target_causal_response_balanced_student_v2'")
text=text.replace('launch_helper_preparation_v1.json','launch_helper_preparation_v2.json').replace('launch_helper_tests_v2.xml','launch_helper_tests_v3.xml')
path.write_text(text,encoding='utf-8',newline='\n')
path=BASE/'test_release_helpers.py';text=path.read_text(encoding='utf-8').replace('launch_helper_preparation_v1.json','launch_helper_preparation_v2.json')
path.write_text(text,encoding='utf-8',newline='\n')
new_map={name:sha(BASE/name) for name in prior['helper_sha256']}
assert {p.relative_to(BASE/'source_draft_v1').as_posix():sha(p) for p in (BASE/'source_draft_v1').rglob('*.py')}==source_map
record=dict(preparation_only=True,helper_sha256=new_map,prior_helper_sha256=prior['helper_sha256'],source_sha256=source_map,
    changed_helpers=[name for name in new_map if new_map[name]!=prior['helper_sha256'][name]],all37_runtime_sources_unchanged=True,
    exact_scope='FIT v1 to repaired v2; versioned helper receipt/test filenames only. No controller, rollout scope or acceptance change.',
    prior_preparation_sha256=sha(preserve/'launch_helper_preparation_v1.json'),source_sha256_of_amendment=sha(__file__),
    model_calls=0,native_steps=0,actual_bindings_created=False)
with (BASE/'launch_helper_derivation_v2.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(dict(changed=record['changed_helpers'],runtime_unchanged=True)))
