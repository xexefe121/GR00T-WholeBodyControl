"""Record the existing simulator's reviewed endpoint adapter and helper checks."""
import ast,hashlib,json
import xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_width512_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def tests(path,count):
    root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==count
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
tests(BASE/'root_release_tests_v1.xml',82);tests(BASE/'launch_helper_tests_v4.xml',41)
source=BASE/'source_draft_v1';actual={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')}
derivation=read(BASE/'runtime_derivation.json');assert actual==derivation['source_sha256'] and len(actual)==40
for p in source.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for name in ('evaluate_direct_target_student.py','head_activation_witness.py','direct_runtime.py','causal_features.py','runtime_common.py'):
    assert sha(source/name)==sha(OLD/'source_draft_v1'/name)
helpers=read(BASE/'launch_helper_derivation.json')['helper_sha256'];helpers={n:sha(BASE/n) for n in helpers}
for name in ('prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py'):
    assert sha(BASE/name)==sha(OLD/name)
prep=dict(source_preparation_passed=True,source_directory=source.as_posix(),source_sha256=actual,
    source_files=40,unchanged_source_files=34,synthetic_tests=82,ordinary_final_step=91000,architecture=[1323,512,512,23],
    controller='direct_absolute_target_1323_causal_width512_recovery',requested_main_controls=1569,conditional_continuous_hold_controls=250,
    physical_hz=500,control_hz=50,model_calls=0,native_steps=0,actual_binding_created=False,
    original_physical_and_intent_acceptance_unchanged=True,hardware_authorized=False)
write(BASE/'source_preparation.json',prep)
write(BASE/'launch_helper_preparation_v2.json',dict(passed=True,helper_sha256=helpers,tests=41,
    tests_sha256=sha(BASE/'launch_helper_tests_v4.xml'),original_execution_helpers_byte_exact=True))
write(BASE/'powershell_path_normalization_v1.json',dict(passed=True,unchanged_reviewed_helper=True,
    helper_sha256=sha(BASE/'prepare_bound_launcher.py'),prior_evidence_sha256=sha(OLD/'powershell_path_normalization_v1.json'),
    slash_normalization='Replace([char]92,[char]47)',model_calls=0,native_steps=0))
write(BASE/'source_root_review.json',dict(passed=True,source_review_pass=True,source_sha256=actual,helper_sha256=helpers,
    source_preparation_sha256=sha(BASE/'source_preparation.json'),tests=123,
    source_tests_sha256=sha(BASE/'root_release_tests_v1.xml'),helper_tests_sha256=sha(BASE/'launch_helper_tests_v4.xml'),
    native_control_code_unchanged=True,model_calls=0,native_steps=0,physical_qualification=False))
print(json.dumps({'source_preparation_sha256':sha(BASE/'source_preparation.json'),'review_sha256':sha(BASE/'source_root_review.json')}))
