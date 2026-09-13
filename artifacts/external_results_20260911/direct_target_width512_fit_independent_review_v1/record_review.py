"""Record exact source review and completed synthetic evidence; no numerical work."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
PRODUCER=NEW/'direct_target_causal_width512_student_v1'
PREP=PRODUCER/'source_preparation.json'
EXPECTED='09871661a31891b28d83daf7810d2dfdd1141b4d1c3d6164193ffdfd42abf675'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    assert sha(PREP)==EXPECTED
    prep=read(PREP);assert prep['source_preparation_passed'] is True and prep['preparation_only'] is True
    source=Path(prep['source_directory']);expected=prep['source_sha256'];assert len(expected)==28
    reviewed=read(BASE/'reviewed_source_before_freeze.json');assert reviewed['source_sha256']==expected
    inputs={PREP.as_posix():EXPECTED}
    for name,digest in expected.items():
        assert sha(source/name)==sha(PRODUCER/'source_draft_v1'/name)==digest,name
        ast.parse((source/name).read_text());inputs[(source/name).as_posix()]=digest
    old=NEW/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1'
    unchanged=[]
    for name in expected:
        if (old/name).exists() and sha(old/name)==expected[name]:
            unchanged.append(name);inputs[(old/name).as_posix()]=expected[name]
    assert sorted(unchanged)==sorted(prep['unchanged_source_files']) and len(unchanged)==21
    for name,entry in prep['subjects'].items():
        p=Path(entry['path']);assert sha(p)==entry['sha256'],name;inputs[p.as_posix()]=entry['sha256']
        if 'pass_field' in entry:
            value=read(p)
            for part in entry['pass_field'].split('.'):value=value[part]
            assert value is True,name
    tests=[]
    for name,count in [('integration_tests_v1.xml',18),('export_mutation_tests_v1.xml',1)]:
        suite=ET.parse(BASE/name).getroot().find('testsuite')
        assert int(suite.attrib['tests'])==count
        assert all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
        tests.append(dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name),passed=count))
    schedule=read(PRODUCER/'saved_schedule_proof.json')
    assert schedule['passed'] and schedule['full_updates']==10000 and schedule['prior_updates']==3000
    assert schedule['first3000_centers_byte_exact'] and schedule['first3000_axes_byte_exact'] and schedule['all54_cells_and_groups_valid']
    assert schedule['schedule_generated'] is False
    for name in ('NOTE.md','record_review.py','test_integration.py','reviewed_source_before_freeze.json'):
        inputs[(BASE/name).as_posix()]=sha(BASE/name)
    result=dict(passed=True,source_review_pass=True,reviewer='/root/review_continuation',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Exact width512 trainer source and synthetic CPU integration review only',source_sha256=expected,
        source_preparation_subject=dict(path=PREP.as_posix(),sha256=EXPECTED),source_preparation_sha256=EXPECTED,
        source_directory=source.as_posix(),source_files=28,old_files_byte_identical=unchanged,changed_or_added_files=sorted(set(expected)-set(unchanged)),
        tested_draft_equals_frozen_source=True,independent_synthetic_tests=tests,independent_tests_passed=19,
        schedule_proof_subject=prep['subjects']['schedule_proof'],initializer_review_subject=prep['subjects']['initializer_review'],
        exporter_review_subject=prep['subjects']['exporter_root_review'],input_sha256=inputs,
        ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000,
        updates=10000,training_forward_calls=30000,training_forward_rows=146860000,
        initial_GPU32_gate_rad=1e-5,initial_byte_gate_required=False,final_FP64_gate_rad=1e-5,
        final_GPU32_drift_is_diagnostic=True,all_actual_subjects_unchanged=True,
        actual_checkpoint_loads=0,actual_dataset_loads=0,task_model_calls=0,task_gradient_calls=0,task_optimizer_updates=0,
        CUDA_calls=0,ORT_calls=0,native_steps=0,synthetic_CPU_model_forward_calls=3,synthetic_optimizer_steps=0,
        actual_fit_selected=False,actual_launch_authorized=False,controller_qualified=False,unresolved_source_findings=[])
    with (BASE/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,source_review_pass=True,review_sha256=sha(BASE/'review.json'),sources=28,independent_tests=19)))

if __name__=='__main__':main()
