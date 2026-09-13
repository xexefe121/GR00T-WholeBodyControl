import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
derivation=json.loads((BASE/'runtime_derivation.json').read_text())
source=BASE/'source_draft_v1'
current={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')}
assert current==derivation['source_sha256']
test=BASE/'runtime_tests_v1.xml';xml=ET.parse(test).getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml)
assert sum(int(s.attrib['tests']) for s in suites)==83
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
prep=dict(source_preparation_passed=True,source_directory=source.as_posix(),source_sha256=current,
 original_evaluation_preparation=dict(path=(BASE.parent/'direct_target_causal_context_evaluation_v2/source_preparation.json').as_posix(),sha256=derivation['original_preparation_sha256']),
 unchanged_original_files=sorted(derivation['unchanged_source_sha256']),changed_original_files=derivation['changed_sources'],new_files=['balance_contract.py'],
 synthetic_tests=dict(passed=83,failed=0,skipped=0,path=test.as_posix(),sha256=sha(test),reused_unchanged_source_evidence=False),
 architecture=[1323,256,256,23],conditions_supported=['causal'],ordinary_endpoint_step=71000,
 context_order='original1000_then_previous_action23_then_incoming_history300',incoming_history_before_update=True,
 applied_target_inverse_after_learned_commit=True,original_BFM_startup_and_terminal_preserved=True,
 original_PD_and_strict_native_predicates_preserved=True,original_full291_and_query250_gates_retained=True,
 extended_first_witness_required=True,original_main_controls=1569,conditional_continuous_hold_controls=250,
 unchanged_preclamp_export_tolerance_rad=1e-5,clock_foundation_connected=False,
 actual_controller_selected=False,actual_binding_created=False,actual_task_model_calls=0,native_steps=0,
 derivation_sha256=sha(BASE/'runtime_derivation.json'),runtime_changes_sha256=sha(BASE/'runtime_changes.diff'),
 limitations=['Single warm71000 endpoint and fixed response weights must pass complete saved-fit audit before any witness.',
 '33 source files including native control/evaluator/features byte-identical to causal68000 evaluation.',
 'Prepared preview and ground-truth root remain simulation inputs; no real Pico or robot proof.'])
path=BASE/'source_preparation.json'
with path.open('x',encoding='utf-8') as f:json.dump(prep,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),source_files=37,tests=83)))
