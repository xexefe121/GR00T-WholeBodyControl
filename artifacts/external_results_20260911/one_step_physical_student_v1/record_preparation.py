"""Read source/tests and record preparation; no module/model execution."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (BASE/'preparation_report.json').exists()
source=BASE/'source_draft_v1'
paths=sorted(source.rglob('*.py'))
for p in paths:compile(p.read_text(encoding='utf-8'),str(p),'exec')
root=ET.parse(BASE/'preparation_tests_final.xml').getroot()
suites=list(root.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==30
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
derivation=json.loads((BASE/'source_derivation.json').read_text())
assert all(sha(source/name)==digest for name,digest in derivation['original_sources_sha256'].items())
absent=['training_request.json','training_frozen_inputs.json','training_clearance.json','fit']
assert all(not (BASE/name).exists() for name in absent)
report=dict(kind='physical_continuation_trainer_preparation_only',passed=True,
    tests=30,failures=0,errors=0,skips=0,source_compiled_without_execution=True,
    original_helpers=32,exact_trainer_substitutions=len(derivation['exact_substitutions']),
    actual_model_evaluations=0,actual_head_onnx_calls=0,optimizer_instances=0,optimizer_updates=0,
    physics_steps=0,synthetic_tensor_backward_tests_only=True,stub_session_calls_are_not_inference=True,
    proposed_updates=[70001,75000],ordinary_final_only=True,
    max_training_rows=36315000,max_head_onnx_calls=1150,max_diagnostic_torch_rows=293480,
    requested_cell_denominators=[99,819,100]*3,unit_loss_coefficients=[1,1,1],
    source_sha256={p.relative_to(source).as_posix():sha(p) for p in paths},
    evidence_sha256={name:sha(BASE/name) for name in ('PREPARATION.md','source_derivation.json','prepare_source.py',
        'test_preparation.py','preparation_tests.xml','preparation_tests_final.xml','record_preparation.py')},
    absent_launch_artifacts=absent,
    final_collection_root_audit_and_fit_selection_required=True)
(BASE/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,report_sha256=sha(BASE/'preparation_report.json'),sources=len(paths),tests=30)))
