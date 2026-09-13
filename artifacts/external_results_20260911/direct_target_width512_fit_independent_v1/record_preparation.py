"""Freeze source/tests/helpers only; no task array, checkpoint or model reads."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'direct_target_response_balanced_fit_independent_v3'
PRODUCER=NEW/'direct_target_causal_width512_student_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def tests(name,count):
    root=ET.parse(BASE/name).getroot()
    assert len(root.findall('.//testcase'))==count
    assert not any(root.findall('.//'+k) for k in ('failure','error','skipped'))
tests('tests_v1.xml',73);tests('metadata_tests_v2.xml',4)
source=BASE/'source_prepared_v1'
prior=read(OLD/'source_preparation.json');unchanged={};changed={}
for name,digest in prior['source_sha256'].items():
    assert sha(OLD/'source_prepared_v1'/name)==digest
    actual=sha(source/name)
    (unchanged if actual==digest else changed)[name]=actual
assert set(changed)=={'audit_saved_warm.py','audit_graph.py','test_prior_math.py','test_metadata.py'}
assert len(unchanged)==9
new_names={'audit_width_math.py','test_width_audit.py','test_producer_metadata.py'}
assert {p.name for p in source.glob('*.py')}==set(prior['source_sha256'])|new_names
for p in source.glob('*.py'):ast.parse(p.read_text())
helpers={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')}
assert helpers['verify_completion.py']==sha(OLD/'verify_completion.py')
assert 'nominal_f32_mean' not in (source/'audit_saved_warm.py').read_text()
producer_prep=read(PRODUCER/'source_preparation.json')
assert sha(PRODUCER/'source_preparation.json')=='09871661a31891b28d83daf7810d2dfdd1141b4d1c3d6164193ffdfd42abf675'
for name,digest in producer_prep['source_sha256'].items():assert sha(PRODUCER/'source_prepared_v1'/name)==digest
assert not any((BASE/n).exists() for n in ('audit_request.json','launch_receipt.json','process_v1','results_v1'))
references=[OLD/'source_preparation.json',OLD/'results_v1/report.json',
    NEW/'direct_target_response_fit_audit_root_review_v3/review.json',
    PRODUCER/'source_preparation.json',PRODUCER/'OUTPUT_SCHEMA.md',PRODUCER/'verify_completed.py',
    NEW/'direct_target_width512_fit_independent_review_v1/review.json']
evidence=['tests_v1.xml','metadata_tests_v1.xml','metadata_tests_v2.xml','source_delta.patch','derivation.json',
    'prepare_source.py','record_preparation.py','ROOT_REVIEW_GUIDE.md','launcher_parse.json']
parse=read(BASE/'launcher_parse.json');assert parse['parse_errors']==0
result=dict(source_preparation_passed=True,preparation_only=True,actual_audit_executed=False,
    source_directory=source.as_posix(),experiment=PRODUCER.as_posix(),
    source_sha256={p.name:sha(p) for p in sorted(source.glob('*.py'))},helper_sha256=helpers,
    unchanged_source_sha256=unchanged,changed_source_sha256=changed,new_source_files=sorted(new_names),
    synthetic_tests_passed=77,new_width_tests=23,new_producer_metadata_tests=4,
    producer_source_sha256=producer_prep['source_sha256'],reference_sha256={p.as_posix():sha(p) for p in references},
    evidence_sha256={n:sha(BASE/n) for n in evidence},
    preserved_test_failure='metadata_tests_v1.xml: one test-only field-count expectation corrected to exact27/13; no production change',
    original_nominal_vs_float64_relative_tolerance=3e-7,initial_parity_tolerance_rad=1e-5,initial_byte_gate_required=False,
    model_parity_tolerance_rad=1e-5,release_subject_count=16,
    expected_updates=10000,expected_training_calls=30000,expected_training_rows=146860000,
    expected_optimizer_start=6000,expected_optimizer_final=16000,
    actual_request_prepared=False,actual_launch_prepared=False,automatic_retry=False,
    task_array_loads=0,task_checkpoint_loads=0,task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
    synthetic_model_forwards=0,synthetic_optimizer_updates=0,
    limitations=['Saved-only verification reconstructs declared initialization and algebra, not task-model outputs or optimizer gradients.',
        'Local CPU tensor RNG is used solely to reconstruct declared added incoming weights; no task model is constructed.'])
with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(result['source_sha256']),helpers=len(helpers),tests=77)))
