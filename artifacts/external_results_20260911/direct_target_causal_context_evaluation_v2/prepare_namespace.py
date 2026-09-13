"""Prepare unchanged simulator runtime against the separate split-initialization study."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'direct_target_causal_context_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
prep=read(OLD/'source_preparation.json');helpers=read(OLD/'launch_helper_preparation_v1.json')
assert sha(OLD/'source_preparation.json')=='f74d261be338d5fe071ecb92c3c22f958003316d08de8edde15d8fc8804c0698'
assert sha(OLD/'launch_helper_preparation_v1.json')=='8b1e27e51cb073234d20cafb8a0199ba46b115b30fd2786b392db619bfcc175f'
source=HERE/'source_draft_v1';source.mkdir(exist_ok=False)
for name,digest in prep['source_sha256'].items():
    old=OLD/'source_draft_v1'/name;assert sha(old)==digest
    dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,dest)
for name,digest in helpers['helper_sha256'].items():
    old=OLD/name;assert sha(old)==digest
    data=old.read_bytes()
    if name=='freeze_final_package.py':
        before=b"FIT=NEW/'direct_target_causal_context_study_v1'"
        assert data.count(before)==1
        data=data.replace(before,b"FIT=NEW/'direct_target_causal_context_study_v2'")
    with (HERE/name).open('xb') as f:f.write(data)
for name in ['source_synthetic_tests_v1.xml','original_runtime_source_sha256.json',
             'launch_helper_tests_v2.xml','powershell_path_normalization_v1.json']:
    shutil.copyfile(OLD/name,HERE/name)
prep.update(source_directory=str(source),
    original_evaluation_preparation={'path':str(OLD/'source_preparation.json'),'sha256':sha(OLD/'source_preparation.json')},
    all_36_runtime_sources_byte_identical=True,
    synthetic_tests=dict(prep['synthetic_tests'],path=str(HERE/'source_synthetic_tests_v1.xml'),reused_unchanged_source_evidence=True),
    limitations=['Runtime is byte-identical to independently reviewed context evaluation v1.',
        'Only helper FIT namespace points to separately prepared study_v2; no actual fit or controller selected.',
        'Prepared preview and ground-truth root remain simulation inputs.'])
write(HERE/'source_preparation.json',prep)
result=dict(preparation_only=True,all_runtime_sources_byte_identical=True,
    old_source_preparation_sha256=sha(OLD/'source_preparation.json'),new_source_preparation_sha256=sha(HERE/'source_preparation.json'),
    old_helper_preparation_sha256=sha(OLD/'launch_helper_preparation_v1.json'),
    source_sha256=prep['source_sha256'],helper_sha256={name:sha(HERE/name) for name in helpers['helper_sha256']},
    changed_helpers=['freeze_final_package.py'],
    helper_delta="FIT=NEW/'direct_target_causal_context_study_v1' -> FIT=NEW/'direct_target_causal_context_study_v2'",
    actual_endpoint_selected=False,actual_bindings_created=False,actual_model_calls=0,native_steps=0)
write(HERE/'namespace_derivation.json',result)
print(json.dumps({'source_files':len(prep['source_sha256']),'helper_files':len(helpers['helper_sha256']),'namespace_derivation_sha256':sha(HERE/'namespace_derivation.json')}))
