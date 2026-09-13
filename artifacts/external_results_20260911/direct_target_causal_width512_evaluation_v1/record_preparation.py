"""Record completed source tests; never create a model or execution binding."""
import ast,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=BASE/'source_draft_v1'
OLD=NEW/'direct_target_causal_response_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
original=read(OLD/'source_preparation.json')['source_sha256']
actual={p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*.py'))}
assert len(actual)==38
changed=[n for n,h in original.items() if actual[n]!=h]
assert set(changed)=={'context_release.py','evaluation_gate.py','test_context_release.py'}
assert set(actual)-set(original)=={'test_width_release.py'}
for p in SOURCE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
xml=ET.parse(BASE/'synthetic_tests.xml').getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==121
assert all(int(s.attrib.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
assert actual['evaluate_direct_target_student.py']==original['evaluate_direct_target_student.py']=='061e2a3145c6ed273bf7ac915efec0f1866ba712ccb8ba571e8c08417243d970'
paths=[OLD/'source_preparation.json',BASE/'runtime_derivation.json',BASE/'source_changes.diff',BASE/'synthetic_tests.xml',
       BASE/'prepare_runtime_sources.py',Path(__file__),NEW/'direct_target_causal_width512_student_v1/source_preparation.json',
       NEW/'direct_target_causal_width512_student_v1/training_request.json',NEW/'direct_target_width512_fit_independent_review_v1/review.json',
       NEW/'direct_target_width512_export_root_review_v1/review.json']
assert not any((BASE/n).exists() for n in ('witness_binding.json','evaluation_binding.json','nominal','head_witness'))
result=dict(source_preparation_passed=True,source_directory=SOURCE.as_posix(),source_sha256=actual,
 input_sha256={p.as_posix():sha(p) for p in paths},original_source_sha256=original,unchanged_source_files=34,
 changed_source_files=changed,added_source_files=['test_width_release.py'],source_files=38,
 synthetic_tests=121,failures=0,errors=0,skips=0,ordinary_final_step=81000,architecture=[1323,512,512,23],
 controller='direct_absolute_target_1323_causal_width512',release_kind='ordinary81000_width512_warm_balanced_context_same_weight_fp64_export',
 context_condition='causal',direct_release_subjects=16,physical_hz=500,control_hz=50,
 requested_main_controls=1569,conditional_continuous_hold_controls=250,
 preserved=['Native evaluator/physics/original strict gates and fixture initialization byteexact.',
 'Current1000 plus incoming prior23/H300 features, original BFMs startup/terminal, query250 witness and prior feedback byteexact.',
 'Existing target span/default/clipping and no filters/root forces/reference changes.',
 'All task-data/fit/export/owner and runtime subject gates retained; new width metadata/rates/counts checked explicitly.'],
 model_calls=0,ORT_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,task_arrays_loaded=0,
 actual_binding_created=False,actual_evaluation_selected=False,preparation_only=True,
 original_physical_and_intent_acceptance_unchanged=True,hardware_authorized=False)
path=BASE/'source_preparation.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'source_preparation_sha256':sha(path),'source_files':38,'tests':121,'model_calls':0,'native_steps':0}))
