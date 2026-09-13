"""Verify preparation pins and saved test receipts without executing a model."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_student_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    assert not (BASE/'source_preparation.json').exists()
    for name in ('witness_binding.json','evaluation_binding.json','head_witness','nominal','evaluation_process','witness_process'):
        assert not (BASE/name).exists(),name
    derivation=read(BASE/'source_derivation.json');inventory=read(BASE/'runtime_inventory.json')
    source={}
    for name,e in derivation['sources'].items():
        p=BASE/'source_draft_v1'/name;assert sha(p)==e['sha256'] and sha(e['original'])==e['original_sha256']
        source[name]=e['sha256']
    helper_copies={}
    for name in ('prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py'):
        assert sha(BASE/name)==sha(OLD/name);helper_copies[name]=sha(BASE/name)
    for entry in inventory['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
    tests={}
    for name,count in (('source_stub_tests.xml',30),('launch_helper_tests.xml',21)):
        tree=ET.parse(BASE/name);cases=list(tree.getroot().iter('testcase'))
        assert len(cases)==count and all(not list(c) or all(x.tag not in ('failure','error','skipped') for x in c) for c in cases)
        tests[name]=dict(tests=count,failures=0,errors=0,skips=0,sha256=sha(BASE/name))
    evidence=[BASE/name for name in ('prepare_sources.py','source_derivation.json','runtime_inventory.py','runtime_inventory.json',
        'freeze_final_package.py','prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py',
        'test_launch_helpers.py','source_stub_tests.xml','launch_helper_tests.xml','README.md','record_preparation.py')]
    result=dict(kind='direct55000_minimal_runtime_evaluation_source_preparation',passed=True,preparation_only=True,
        source_sha256=source,unchanged_helper_sha256=helper_copies,evidence_sha256={p.as_posix():sha(p) for p in evidence},
        tests=tests,runtime_inventory_files=inventory['total_files'],runtime_inventory_bytes=inventory['total_bytes'],
        recursive_training_hashes=False,ordinary_final_step=55000,additional_updates=50000,
        requested_main_controls=1569,conditional_hold_controls=250,expected_separate_head_calls=1,
        actual_final_head_bound=False,model_calls=0,native_steps=0,optimizer_updates=0,hardware_authorized=False)
    (BASE/'source_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation.json'),runtime_source_files=len(source),
        stub_tests=51,model_calls=0,native_steps=0)))

if __name__=='__main__':main()
