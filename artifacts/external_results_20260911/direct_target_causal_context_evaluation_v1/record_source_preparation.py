"""Freeze additive evaluation source and completed synthetic test evidence."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((BASE/'original_runtime_source_sha256.json').read_text())
current={p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*.py'))}
tests=BASE/'source_synthetic_tests_v1.xml'
suites=ET.parse(tests).getroot().findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites)==76
assert all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped'))
assert len(old)==32 and len(current)==36
changed=[name for name in old if old[name]!=current[name]]
unchanged=[name for name in old if old[name]==current[name]]
new=[name for name in current if name not in old]
assert set(new)=={'causal_features.py','test_causal_features.py','context_release.py','test_context_release.py'}
result=dict(source_preparation_passed=True,source_directory=str(SOURCE),source_sha256=current,
    original_source_sha256=old,changed_original_files=changed,unchanged_original_files=unchanged,new_files=new,
    synthetic_tests=dict(passed=76,failed=0,skipped=0,path=str(tests),sha256=sha(tests)),
    architecture=[1323,256,256,23],conditions_supported=['blinded','causal'],ordinary_endpoint_step=68000,
    context_order='original1000_then_previous_action23_then_incoming_history300',
    incoming_history_before_update=True,applied_target_inverse_after_learned_commit=True,
    original_BFM_startup_and_terminal_preserved=True,original_PD_and_strict_native_predicates_preserved=True,
    original_full291_and_query250_gates_retained=True,extended_first_witness_required=True,
    original_main_controls=1569,conditional_continuous_hold_controls=250,
    unchanged_preclamp_export_tolerance_rad=1e-5,clock_foundation_connected=False,
    actual_controller_selected=False,actual_binding_created=False,actual_task_model_calls=0,native_steps=0,
    limitations=['Source and synthetic checks only; actual fit artifacts, independent fit audit and endpoint selection remain pending.',
                'Prepared reference preview and ground-truth root remain simulation inputs.',
                'Durable launch helpers and concrete endpoint bindings are not prepared yet.'])
destination=BASE/'source_preparation.json'
with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=str(destination),sha256=sha(destination),changed=changed,new=new,unchanged=len(unchanged))))
