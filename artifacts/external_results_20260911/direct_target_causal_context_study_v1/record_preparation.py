"""Freeze declared Python sources and completed saved-only proof; no execution gate."""
from pathlib import Path
import ast
import json
import hashlib
import shutil
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    draft=BASE/'source_draft_v1';source={p.name:sha(p) for p in sorted(draft.glob('*.py'))}
    proof=read(BASE/'context_preflight/report.json')
    assert proof['passed'] and proof['all_inputs_unchanged'] and proof['source_sha256']==source
    assert proof['proof']['nominal_chronological_history_and_prior_pairs']==9899
    assert proof['input_sha256']==read(BASE/'preparation_inputs.json')['input_sha256']
    for name,digest in proof['output_sha256'].items():assert sha(BASE/'context_preflight'/name)==digest,name
    for path in draft.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    tests=ET.parse(BASE/'synthetic_tests_v3.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in tests)==15
    assert all(int(s.attrib.get(k,0))==0 for s in tests for k in ('failures','errors','skipped'))
    snapshot=BASE/'source_prepared_v1';snapshot.mkdir(exist_ok=False)
    for name,digest in source.items():
        shutil.copyfile(draft/name,snapshot/name);assert sha(snapshot/name)==digest
    copied=['direct_contract.py','direct_data.py','direct_objective.py','full_state_contract.py','full_state_data.py',
        'full_state_objective.py','restoration_support.py','training_support.py','full_state_diagnostics.py','promoted_model.py']
    original=BASE.parent/'direct_target_full_state_student_v1/source_snapshot_v1'
    assert all(sha(original/name)==source[name] for name in copied)
    subjects={name:dict(path=(BASE/name).as_posix(),sha256=sha(BASE/name)) for name in
        ('DESIGN.md','OUTPUT_SCHEMA.md','training_request_proposal.json','preparation_inputs.json','context_preflight/report.json',
         'synthetic_tests_v3.xml','test_context_export.py','prepare_proposal.py','record_preparation.py')}
    report=dict(source_preparation_passed=True,preparation_only=True,source_directory=snapshot.as_posix(),
        source_sha256=source,unchanged_original_modules=copied,subjects=subjects,synthetic_tests_passed=15,
        completed_context_proof_sha256=sha(BASE/'context_preflight/report.json'),input_pins=len(proof['input_sha256']),
        context_proof=proof['proof'],actual_task_model_calls=0,actual_gradient_calls=0,actual_optimizer_updates=0,native_steps=0,
        actual_training_request_present=(BASE/'training_request.json').exists(),actual_training_clearance_present=(BASE/'training_clearance.json').exists())
    assert not report['actual_training_request_present'] and not report['actual_training_clearance_present']
    with (BASE/'source_preparation.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation.json'),sources=len(source),tests=15,
        proof_sha256=report['completed_context_proof_sha256'],training_launched=False)))

if __name__=='__main__':main()
