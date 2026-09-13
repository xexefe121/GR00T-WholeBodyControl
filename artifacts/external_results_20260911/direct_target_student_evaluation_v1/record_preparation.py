"""Freeze a source-only preparation receipt; never create an execution binding."""
from pathlib import Path
import ast
import hashlib
import json
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

for name in ('witness_binding.json','evaluation_binding.json','head_witness','nominal','post_lifecycle_hold_5s','evaluation_process','witness_process'):
    assert not (BASE/name).exists(),name
originals=read(BASE/'original_sources.json')
for name,item in originals.items():
    assert sha(SOURCE/name)==sha(Path(item['original']))==item['sha256'],name
test=ET.parse(BASE/'stub_tests_final_v3.xml').getroot().find('testsuite')
assert int(test.attrib['tests'])==30 and all(int(test.attrib[k])==0 for k in ('failures','errors','skipped'))
feature=read(BASE/'saved_feature_checks_v1/report.json')
assert feature['passed'] and feature['rows']==9904 and feature['first_query250']['byte_exact']
for path,expected in feature['input_sha256'].items():assert sha(path)==expected,path
sources={p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*.py'))}
for path in SOURCE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
derivation=read(BASE/'evaluator_derivation_v2.json')
assert sha(SOURCE/'evaluate_direct_target_student.py')==derivation['derived_sha256']
assert all(derivation['unchanged_AST'].values())
evidence={str(path.resolve()):sha(path) for path in [Path(__file__),BASE/'README.md',BASE/'prepare_sources.py',
    BASE/'derive_evaluator.py',BASE/'check_saved_features.py',BASE/'original_sources.json',BASE/'feature_derivation.json',
    BASE/'evaluator_derivation.json',BASE/'evaluator_derivation_v2.json',BASE/'stub_tests_v1.xml',
    BASE/'stub_tests_final_v1.xml',BASE/'stub_tests_final_v2.xml',BASE/'stub_tests_final_v3.xml',
    BASE/'saved_feature_checks_v1/request.json',BASE/'saved_feature_checks_v1/report.json']}
report=dict(kind='direct_target_evaluation_source_preparation',passed=True,source_preparation_only=True,
    source_sha256=sources,evidence_sha256=evidence,original_dependency_count=len(originals),all_originals_byte_exact=True,
    tests=30,failures=0,skips=0,saved_feature_rows=9904,first_query250=feature['first_query250'],
    architecture=[1000,256,256,23],hidden_activation='ELU',head_output='normalized_target',
    intended_fresh_final_step=5000,feature_indices='0:52 +75:1023',
    output_formula='default_f64 + existing_span_f32.astype(f64) * normalized_head_f32.astype(f64); native_f64_clip',
    learned_BFM_calls_per_control=0,learned_head_calls_per_control=1,
    learned_prior='inverse actually clipped native target, float32, no +/-5 clamp',
    original_BFM_startup_terminal_math_unchanged=True,history_before_update_source_unchanged=True,
    proposed_main_controls=1569,conditional_continuous_hold_controls=250,proposed_separate_head_witness_calls=1,
    actual_head_calls=0,actual_BFM_calls=0,native_steps=0,optimizer_updates=0,models_constructed=0,
    actual_execution_binding_created=False,actual_launcher_created=False,clock_foundation_connected=False,
    runtime_installed=False,behavioral_qualification=False,hardware_authorized=False,
    next_gate='actual final model/data/fit/source/export receipts and separate execution selection; then final freeze/launcher review')
with (BASE/'source_preparation.json').open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(dict(report_sha256=sha(BASE/'source_preparation.json'),source_count=len(sources),tests=30,saved_feature_rows=9904)))
