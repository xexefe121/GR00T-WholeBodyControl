"""Freeze reviewed source and compact existing test/proof evidence; no model import."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_balanced_student_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as stream:stream.write(v if isinstance(v,str) else json.dumps(v,indent=2,allow_nan=False)+'\n')
def subject(p,field=None):
    s=dict(path=Path(p).as_posix(),sha256=sha(p))
    if field:s['pass_field']=field
    return s
def tests(path,count):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==count
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))

def main():
    tests(BASE/'execution_helper_tests.xml',8)
    tests(NEW/'direct_target_width512_fit_independent_review_v1/integration_tests_v1.xml',18)
    assert read(BASE/'saved_schedule_proof.json')['passed'] is True
    assert read(BASE/'launcher_tests.json')['passed'] is True
    source=BASE/'source_draft_v1';frozen=BASE/'source_prepared_v1';frozen.mkdir(exist_ok=False)
    before=OLD/'source_snapshot_v1';mapping={};unchanged=[];changed=[];added=[];patch=[]
    for p in sorted(source.glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8-sig'))
        shutil.copyfile(p,frozen/p.name);mapping[p.name]=sha(frozen/p.name)
        assert mapping[p.name]==sha(p)
        original=before/p.name
        if original.exists():
            if sha(original)==mapping[p.name]:unchanged.append(p.name)
            else:
                changed.append(p.name)
                patch.extend(difflib.unified_diff(original.read_text().splitlines(True),p.read_text().splitlines(True),fromfile=original.as_posix(),tofile=(frozen/p.name).as_posix()))
        else:added.append(p.name)
    assert len(mapping)==28 and len(unchanged)==21 and len(changed)==4 and len(added)==3
    assert sha(frozen/'width512.py')==sha(NEW/'direct_target_causal_width512_preparation_v1/source_prepared_v1/width512.py')
    assert sha(frozen/'width512_promoted.py')==sha(NEW/'direct_target_width512_export_preparation_v1/source_draft_v1/width512_promoted.py')
    write(BASE/'source_derivation_final.patch',''.join(patch))
    write(BASE/'source_derivation_final.json',dict(source_sha256=mapping,unchanged=unchanged,changed=changed,added=added,
        original_directory=before.as_posix(),patch_sha256=sha(BASE/'source_derivation_final.patch')))
    subjects={name:subject(path,field) for name,path,field in [
        ('initializer',NEW/'direct_target_causal_width512_preparation_v1/source_preparation.json','source_preparation_passed'),
        ('initializer_review',NEW/'direct_target_width512_independent_source_review_v1/review.json','source_review_pass'),
        ('exporter',NEW/'direct_target_width512_export_preparation_v1/source_preparation.json','source_preparation_passed'),
        ('exporter_root_review',NEW/'direct_target_width512_export_root_review_v1/review.json','source_review_pass'),
        ('warm_source',OLD/'source_preparation.json','source_preparation_passed'),
        ('source_fit_owner',OLD/'owner_completion_verification.json','owner_verification_passed'),
        ('source_fit_audit',NEW/'direct_target_response_balanced_fit_independent_v3/results_v1/report.json','evidence_audit_passed'),
        ('latest_semantics',NEW/'direct_target_response_saved_semantics_review_v1/results_v1/report.json','passed'),
        ('latest_semantics_owner',NEW/'direct_target_response_saved_semantics_review_v1/owner_completion.json','completion_accounting_passed'),
        ('schedule_proof',BASE/'saved_schedule_proof.json','passed'),
        ('static_checks',BASE/'source_static_checks.json','passed'),
        ('launcher_checks',BASE/'launcher_tests.json','passed')]}
    for key,item in subjects.items():
        assert read(item['path'])[item['pass_field']] is True,key
    for name in ('OUTPUT_SCHEMA.md','prepare_source.py','prepare_execution.py','prepare_checks.py','record_preparation.py',
        'source_derivation_final.json','source_derivation_final.patch','verify_completed.py','test_execution_helpers.py',
        'execution_helper_tests.xml','run_fit_durable_v1.ps1','read_fit_progress_v1.ps1','test_launcher.ps1','launcher_test_wrong_host.json'):
        subjects[name]=subject(BASE/name)
    tests(NEW/'direct_target_width512_fit_independent_review_v1/export_mutation_tests_v1.xml',1)
    for name in ('integration_tests_v1.xml','export_mutation_tests_v1.xml','test_integration.py','reviewed_source_before_freeze.json'):
        subjects['independent_'+name]=subject(NEW/'direct_target_width512_fit_independent_review_v1'/name)
    prep=dict(source_preparation_passed=True,preparation_only=True,source_directory=frozen.as_posix(),source_sha256=mapping,
        subjects=subjects,unchanged_source_files=unchanged,changed_source_files=changed,added_source_files=added,
        original_byte_exact_count=21,source_files=28,owner_metadata_tests=8,independent_integration_tests=19,
        inherited_initializer_tests=27,inherited_exporter_tests=20,initial_GPU32_gate_rad=1e-5,
        initial_byte_gate_required=False,final_admission_backends=['CPU64','GPU64','ORT64'],
        final_GPU32_drift_is_diagnostic=True,training_updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,
        optimizer_start_step=6000,optimizer_final_step=16000,training_forward_calls=30000,training_forward_rows=146860000,
        actual_checkpoint_loads=0,task_model_calls=0,task_optimizer_updates=0,native_steps=0,actual_fit_selected=False,
        inherited_test_response_continuation_not_executed=True,
        all_source_and_subject_hashes_verified=True)
    write(BASE/'source_preparation.json',prep)
    print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(mapping))))

if __name__=='__main__':main()
