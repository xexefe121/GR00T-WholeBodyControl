from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_response_balanced_student_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
tree=ET.parse(BASE/'metadata_regression_tests.xml').getroot()
suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==3 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
derivation=read(BASE/'source_derivation.json');prior=read(OLD/'source_preparation.json')
for name,digest in derivation['source_sha256'].items():assert sha(BASE/'source_prepared_v1'/name)==digest
subjects={name:dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name)) for name in
    ('source_derivation.json','training_request_proposal.json','DESIGN.md','OUTPUT_SCHEMA.md','EXECUTION_SCHEMA.md',
     'prepare_repair.py','test_carried_request.py','metadata_regression_tests.xml','record_repair_preparation.py')}
for name,path in [('prior_preparation',OLD/'source_preparation.json'),('prior_owner',OLD/'owner_completion_verification.json'),
    ('prior_failure_verification',OLD/'zero_forward_failure_verification.json'),('prior_warm_source_review',BASE.parent/'direct_target_response_warm_source_review_v1/review.json'),
    ('prior_root_source_review',BASE.parent/'direct_target_response_root_source_review_v1/review.json')]:
    subjects[name]=dict(path=path.as_posix(),sha256=sha(path))
receipt=dict(source_preparation_passed=True,preparation_only=True,root_selected=False,
    source_directory=(BASE/'source_prepared_v1').as_posix(),source_sha256=derivation['source_sha256'],
    original_modules_byte_exact=prior['original_modules_byte_exact'],reviewed_math_modules_byte_exact=prior['reviewed_math_modules_byte_exact'],
    unchanged_prior_sources=derivation['unchanged_sources'],changed_sources=derivation['changed_sources'],
    metadata_regression_tests_passed=3,inherited_owner_synthetic_tests=35,inherited_independent_synthetic_tests=15,
    subjects=subjects,task_model_calls=0,task_gradient_calls=0,optimizer_updates=0,native_steps=0,
    actual_request_created=False,execution_dispatched=False)
with (BASE/'source_preparation.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),proposal_sha256=sha(BASE/'training_request_proposal.json'),
    driver_sha256=derivation['source_sha256']['train_response_balanced.py'])))
