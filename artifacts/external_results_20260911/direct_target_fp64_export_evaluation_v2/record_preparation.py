"""Version2 preparation and real audit-schema checks, zero model/native calls."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_fp64_export_evaluation_v1'
FIT=BASE.parent/'direct_target_continuation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    for name in ('source_preparation.json','witness_binding.json','evaluation_binding.json','head_witness','nominal','witness_process','evaluation_process'):
        assert not (BASE/name).exists(),name
    prior=read(OLD/'source_preparation.json')
    for path,digest in prior['evidence_sha256'].items():assert sha(path)==digest,path
    for name,digest in prior['source_sha256'].items():assert sha(OLD/'source_draft_v1'/name)==digest
    derivation=read(BASE/'source_derivation.json');inventory=read(BASE/'runtime_inventory.json');sources={}
    for name,entry in derivation['sources'].items():
        p=BASE/'source_draft_v1'/name
        assert sha(p)==entry['sha256'] and sha(entry['original'])==entry['original_sha256']
        sources[name]=entry['sha256']
    assert sum(e['byte_identical'] for e in derivation['sources'].values())==30
    unchanged={}
    for name in ('freeze_final_package.py','prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','runtime_inventory.py','test_launch_helpers.py'):
        assert sha(BASE/name)==sha(OLD/name);unchanged[name]=sha(BASE/name)
    for entry in inventory['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
    tests={}
    for name,count in (('source_stub_tests.xml',30),('launch_helper_tests_v2.xml',21),('release_gate_tests.xml',33)):
        cases=list(ET.parse(BASE/name).getroot().iter('testcase'))
        assert len(cases)==count and not any(x.tag in ('failure','error','skipped') for c in cases for x in c)
        tests[name]=dict(tests=count,failures=0,errors=0,skips=0,sha256=sha(BASE/name))
    root_path=BASE.parent/'direct_target_continuation_failure_audit_v1/results_v1/report.json';root=read(root_path)
    assert root['evidence_audit_passed'] is True and root['export_qualified'] is False and root['canonical_cleared'] is False
    roles={k:FIT/v for k,v in dict(fit_report='fit/report.json',checkpoint='fit/student_head.pt',source_head='fit/student_head.onnx',
        normalization='fit/normalization.npz',training_manifest='training_frozen_inputs.json',training_request='training_request.json').items()}
    consumed={str(p).replace('\\','/').casefold():d for p,d in root['input_sha256'].items()}
    memberships={}
    for role,path in roles.items():
        digest=sha(path);assert consumed[path.as_posix().casefold()]==digest
        memberships[role]=dict(path=path.as_posix(),sha256=digest,explicit_root_input_membership=True)
    evidence=[BASE/name for name in ('prepare_sources.py','export_release_gate.py','source_derivation.json','runtime_inventory.py','runtime_inventory.json',
        'freeze_final_package.py','prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py',
        'test_export_release_gate.py','source_stub_tests.xml','launch_helper_tests_v2.xml','release_gate_tests.xml','README.md','record_preparation.py')]
    evidence.extend([OLD/'source_preparation.json',root_path])
    result=dict(kind='same55000_fp64_export_evaluation_source_preparation_v2',passed=True,preparation_only=True,
        source_sha256=sources,unchanged_helper_sha256=unchanged,evidence_sha256={p.as_posix():sha(p) for p in evidence},
        prior_v1_preserved_sha256=sha(OLD/'source_preparation.json'),root_audit_schema_membership=memberships,tests=tests,
        runtime_inventory_files=inventory['total_files'],runtime_inventory_bytes=inventory['total_bytes'],recursive_training_hashes=False,
        ordinary_final_step=55000,additional_training_updates=0,requested_main_controls=1569,conditional_hold_controls=250,
        expected_separate_head_calls=1,original_failed_export_preserved=True,actual_final_head_bound=False,
        model_calls=0,native_steps=0,optimizer_updates=0,hardware_authorized=False)
    out=BASE/'source_preparation.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(preparation_sha256=sha(out),sources=len(sources),stub_tests=84,model_calls=0,native_steps=0)))
if __name__=='__main__':main()
