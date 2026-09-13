"""Record independent eight-helper review, exact sources and synthetic-only evidence."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
SUBJECT=BASE.parent/'direct_target_full_state_evaluation_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    prep_path=SUBJECT/'launch_helper_preparation_v2.json'
    assert sha(prep_path)=='02f40a1a0178bf8c13accff001b9f0859c5dfd3fc99b961b8217a6ec97b35400'
    prep=read(prep_path);pins={prep_path.as_posix():sha(prep_path)}
    expected=prep['helper_sha256'];assert len(expected)==8
    for name,digest in expected.items():
        path=SUBJECT/name;assert sha(path)==digest,name;pins[path.as_posix()]=digest
    for name,digest in prep['source_sha256'].items():
        path=SUBJECT/'source_draft_v1'/name;assert sha(path)==digest,name;pins[path.as_posix()]=digest
    assert len(prep['source_sha256'])==32
    original=read(SUBJECT/'launch_helpers_preserved_v1/launch_helper_preparation.json')
    assert original['source_sha256']==prep['source_sha256']
    inventory_path=SUBJECT/'runtime_inventory.json';assert sha(inventory_path)==prep['runtime_inventory_sha256']
    inventory=read(inventory_path);pins[inventory_path.as_posix()]=sha(inventory_path)
    assert inventory['source_sha256']==prep['source_sha256'] and inventory['recursive_training_hashes'] is False
    assert inventory['baseline_full291_fixture_byteexact'] is True
    paths=[entry['path'].replace('\\','/').casefold() for entry in inventory['files']]
    assert len(paths)==len(set(paths))==5187
    assert all(re.fullmatch('[0-9a-f]{64}',entry['sha256']) for entry in inventory['files'])
    assert all(not any(part in path for part in ('/direct_target_gpu_20260911/','/direct_target_full_state_secants_v1/generation/','/one_step_policy_branch_collection_resume2969_v1/collection/data/')) for path in paths)
    parsings=read(BASE/'powershell_parse.json');assert len(parsings)==4 and all(row['error_count']==0 for row in parsings)
    for row in parsings:assert row['sha256']==prep['synthetic_template_sha256'][row['name']]
    tree=ET.parse(BASE/'tests_v2.xml');suites=list(tree.getroot().iter('testsuite'))
    assert sum(int(suite.attrib['tests']) for suite in suites)==39
    assert all(int(suite.attrib.get('failures',0))==int(suite.attrib.get('errors',0))==0 for suite in suites)
    for path in (BASE/'tests_v2.xml',BASE/'tests.xml',BASE/'powershell_parse.json',BASE/'test_independent_helper_boundaries.py',Path(__file__),
                 SUBJECT/'powershell_path_normalization_v2.json',SUBJECT/'source_preparation.json',SUBJECT/'launch_helper_tests_v2.xml'):
        pins[path.as_posix()]=sha(path)
    for name in ('release_reviews.json','witness_binding.json','evaluation_binding.json','head_witness','nominal','witness_process','evaluation_process'):
        assert not (SUBJECT/name).exists(),name
    result=dict(passed=True,helper_sha256=expected,preparation_sha256=sha(prep_path),source_sha256=prep['source_sha256'],
        input_sha256=pins,helper_count=8,runtime_sources_unchanged=32,synthetic_tests_passed=39,
        owner_synthetic_checks=12,existing_helper_synthetic_checks=27,powershell_templates_parsed=4,
        fixed_finding='Generated reviewer receipt path now uses Replace([char]92,[char]47); actual PowerShell assertion passes.',
        reviewed_contracts=['all13 actual fit/export/source/data subjects directly bound through owner/root/release receipts',
            'helper review must match exact eight-file map; final stage review must identify actual binding and launch receipt paths and hashes',
            'one separate batch1 WSL witness, original1569 controls plus conditional continuous250 hold; no extra fit or calls',
            'fixed runtime inventory and all32 modules; source prep cannot create task sessions; no recursive training/GPU inventory',
            'CreateNew stage and runtime guards, hidden captured-handle child, known raw/diagnostic exit distinction and postrun hashes',
            'saved completion rejects wrong/live process proofs, changed pins/reviews, inconsistent raw/final exits and repeat receipt writes'],
        inventory_entries_checked=5187,inventory_file_sha256=sha(inventory_path),runtime_binary_contents_rehashed_by_this_source_review=False,
        inventory_limitations=inventory['limitations'],concrete_stage_review_still_required=True,
        initial_test_collection_error='First pytest invocation inferred drive-root collection and hit E:/WpSystem security metadata before collecting tests; XML preserved. Scoped root/confcutdir run passed all39, no task calls in either.',
        actual_bindings_created=False,task_model_calls=0,native_steps=0,optimizer_updates=0,controller_launches=0)
    with (BASE/'review.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(BASE/'review.json'),pins=len(pins),helper_count=8,tests=39)))
if __name__=='__main__':main()
