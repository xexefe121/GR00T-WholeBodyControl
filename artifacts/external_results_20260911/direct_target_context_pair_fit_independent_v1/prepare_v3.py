"""Preserve v2; align the saved auditor to the corrected split study only."""
from pathlib import Path
import difflib, json, hashlib

BASE = Path(__file__).resolve().parent
OLD = BASE / 'source_draft_v2'
NEW = BASE / 'source_draft_v3'
NEW.mkdir(exist_ok=False)
changes = {}

for path in sorted(OLD.glob('*.py')):
    previous = path.read_text()
    updated = previous
    if path.name == 'audit_saved_pair.py':
        before = 'changed_MatMul_dimension=True,byte_gate_required=False'
        after = "changed_MatMul_dimension=False,stored_first_layer_width=1323,first_layer_execution='split_contiguous_1000_plus_323',byte_gate_required=False"
        assert updated.count(before) == 1
        updated = updated.replace(before, after)
        before = "public_output_dtype='float32',parity_tolerance_rad=1e-5"
        after = "public_output_dtype='float32',training_first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323',parity_tolerance_rad=1e-5"
        assert updated.count(before) == 1
        updated = updated.replace(before, after)
    with (NEW / path.name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(updated)
    if updated != previous:
        changes[path.name] = ''.join(difflib.unified_diff(previous.splitlines(True), updated.splitlines(True), fromfile=path.name + ' v2', tofile=path.name + ' v3'))

specifications = {
    'prepare_audit_request_v2.py': ('prepare_audit_request_v3.py', [('source_draft_v2', 'source_draft_v3'), ('direct_target_causal_context_study_v1', 'direct_target_causal_context_study_v2')]),
    'freeze_launch_v2.py': ('freeze_launch_v3.py', [('run_audit_durable_v2.ps1', 'run_audit_durable_v3.ps1'), ('prepare_audit_request_v2.py', 'prepare_audit_request_v3.py'), ('source_preparation_v2.json', 'source_preparation_v3.json'), ('verify_completion_v2.py', 'verify_completion_v3.py')]),
    'run_audit_durable_v2.ps1': ('run_audit_durable_v3.ps1', [('source_draft_v2', 'source_draft_v3')]),
    'verify_completion_v2.py': ('verify_completion_v3.py', [('source_preparation_v2.json', 'source_preparation_v3.json')]),
}
for old_name, (new_name, substitutions) in specifications.items():
    previous = (BASE / old_name).read_text()
    updated = previous
    for before, after in substitutions:
        assert before in updated
        updated = updated.replace(before, after)
    with (BASE / new_name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(updated)
    changes[new_name] = ''.join(difflib.unified_diff(previous.splitlines(True), updated.splitlines(True), fromfile=old_name, tofile=new_name))

with (BASE / 'source_delta_v3.patch').open('x', encoding='utf-8') as stream:
    stream.write('\n'.join(changes.values()))
print(json.dumps({'changed': list(changes), 'actual_audit_run': False}))
