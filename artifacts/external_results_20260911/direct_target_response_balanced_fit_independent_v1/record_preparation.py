"""Freeze already-tested auditor sources and small schema references only."""
from pathlib import Path
import ast,hashlib,json,shutil,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    target=BASE/'source_prepared_v1';target.mkdir(exist_ok=False)
    for p in sorted((BASE/'source_draft_v1').glob('*.py')):
        ast.parse(p.read_text());shutil.copyfile(p,target/p.name)
    tests=ET.parse(BASE/'tests_final_v2.xml').getroot();cases=tests.findall('.//testcase')
    assert len(cases)==39 and not tests.findall('.//failure') and not tests.findall('.//error') and not tests.findall('.//skipped')
    old=NEW/'direct_target_context_pair_fit_independent_v1/source_draft_v3'
    copied=['audit_math.py','audit_restoration.py','audit_graph.py','audit_context_math.py','audit_full_state_math.py']
    for name in copied:assert sha(target/name)==sha(old/name)
    schema=NEW/'direct_target_causal_response_balanced_student_v2'
    prep=read(schema/'source_preparation.json')
    for name,d in prep['source_sha256'].items():assert sha(Path(prep['source_directory'])/name)==d
    owner=schema/'verify_completed.py'
    assert sha(owner)==sha(NEW/'direct_target_causal_response_balanced_student_v1/verify_completed.py')
    references=[NEW/'direct_target_context_pair_audit_root_review_v3/review.json',schema/'source_preparation.json',
        schema/'OUTPUT_SCHEMA.md',schema/'EXECUTION_SCHEMA.md',owner,
        NEW/'direct_target_response_root_source_review_v1/concrete_review.json',
        NEW/'direct_target_causal_response_balanced_student_v1/training_clearance.json']
    references += [Path(prep['source_directory'])/name for name in prep['source_sha256']]
    concrete=read(references[5])
    assert 'subjects' not in concrete and all(k in concrete for k in ('training_request_sha256','frozen_receipt_sha256','launcher_sha256','source_sha256'))
    result=dict(source_preparation_passed=True,preparation_only=True,actual_audit_executed=False,
        experiment=(NEW/'direct_target_causal_response_balanced_student_v2').as_posix(),source_directory=target.as_posix(),
        source_sha256={p.name:sha(p) for p in sorted(target.glob('*.py'))},
        helper_sha256={'prepare_audit_request.py':sha(BASE/'prepare_audit_request.py')},
        copied_math_sha256={name:sha(target/name) for name in copied},
        schema_reference_sha256={p.as_posix():sha(p) for p in references},
        evidence_sha256={name:sha(BASE/name) for name in ('tests_final_v2.xml','README.md','source_delta.patch','derivation.json','derive_auditor.py','record_preparation.py')},
        synthetic_tests_passed=39,task_array_loads=0,task_checkpoint_loads=0,task_model_calls=0,
        ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,automatic_retry=False,
        future_eligibility='Completed optimization and diagnostics in corrected v2 only; original numerical failure is separately retained.')
    with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(result['source_sha256']),tests=39)))
if __name__=='__main__':main()
