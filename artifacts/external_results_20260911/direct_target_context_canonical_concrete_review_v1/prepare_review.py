"""Derive the next-stage reviewer without executing any runtime or review."""
from pathlib import Path
import difflib

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'direct_target_context_witness_concrete_review_v1/review_concrete.py'
old = OLD.read_text()
new = old.replace('single causal68000 WSL witness package', 'single causal68000 canonical package after completed witness')
new = new.replace('import hashlib, json, re', 'import hashlib, json, re, argparse\nimport numpy as np')
old_constants = "BINDING_SHA = '1502f0365a29c5ec8b21b2547f158a37d38999fb90260b34371e89a319e4006d'\nLAUNCH_SHA = 'f3772bc07cf1a46b4f212cafd8f13ec4ad9f1c81388b35b95e3753027982b1d8'"
new_constants = """parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binding-sha256', required=True)
parser.add_argument('--launch-sha256', required=True)
parser.add_argument('--witness-owner-sha256', required=True)
parser.add_argument('--witness-sha256', required=True)
parser.add_argument('--binding-pins', required=True, type=int)
parser.add_argument('--launch-pins', required=True, type=int)
args = parser.parse_args()
BINDING_SHA, LAUNCH_SHA = args.binding_sha256, args.launch_sha256
assert args.binding_pins > 5225 and args.launch_pins > 5230"""
assert old_constants in new
new = new.replace(old_constants, new_constants)
substitutions = [
    ("BASE / 'witness_process'", "BASE / 'evaluation_process'"),
    ("BASE / 'witness_binding.json'", "BASE / 'evaluation_binding.json'"),
    ("'one_selected_context_target_witness'", "'one_selected_context_target_evaluation'"),
    ("launch['requested_main_controls'] == launch['conditional_hold_controls'] == 0", "launch['requested_main_controls'] == 1569 and launch['conditional_hold_controls'] == 250"),
    ("launch['expected_separate_head_calls'] == b['expected_head_calls'] == 1", "launch['expected_separate_head_calls'] == 0 and b['expected_head_calls'] == 1"),
    ("b['physics_authorized'] is False and b['root_authorized_witness'] is True", "b['physics_authorized'] is True and b['root_authorized_evaluation'] is True"),
    ("linux + '/source_draft_v1/head_activation_witness.py'", "linux + '/source_draft_v1/evaluate_direct_target_student.py'"),
    ("'--mode \\'witness\\''", "'--mode \\'evaluation\\''"),
    ("len(b['input_files']) == 5225 and len(pins) == 5230", "len(b['input_files']) == args.binding_pins and len(pins) == args.launch_pins"),
    ("[BASE / 'head_witness', BASE / 'nominal', BASE / 'post_lifecycle_hold_5s']", "[BASE / 'nominal', BASE / 'post_lifecycle_hold_5s', BASE / 'pilot_outcome.json', BASE / 'canonical_initial_snapshot.npz']"),
    ("checked_binding_pins=5225, checked_input_pins=5230", "checked_binding_pins=args.binding_pins, checked_input_pins=args.launch_pins"),
    ("mode='witness'", "mode='evaluation'"),
    ("permitted_witness_head_calls=1, permitted_BFM_calls=0, permitted_native_steps=0", "permitted_additional_witness_calls=0, requested_main_controls=1569, conditional_hold_controls=250, maximum_native_steps=18190, learned_BFM_calls=0"),
    ("canonical_launch_cleared=False", "canonical_concrete_review_passed=True"),
]
for before, after in substitutions:
    assert before in new, before
    new = new.replace(before, after)
insert = """
# Completed witness is evidence only: never call its head again during review.
wfolder = BASE / 'witness_process'
wbp = BASE / 'witness_binding.json'
wb = read(wbp)
assert sha(wbp) == '1502f0365a29c5ec8b21b2547f158a37d38999fb90260b34371e89a319e4006d'
assert wb['head'] == b['head'] and wb['context_condition'] == 'causal'
wp, wrp = BASE / 'head_witness/witness.npz', BASE / 'head_witness/report.json'
assert b['first_export_witness'] == dict(path=wp.as_posix(), sha256=args.witness_sha256)
assert b['first_export_receipt'] == dict(path=wrp.as_posix(), sha256=sha(wrp))
assert sha(wp) == args.witness_sha256
wreport = read(wrp)
assert wreport['pass_all'] is True and wreport['expected_head_calls'] == wreport['attempted_head_calls'] == wreport['returned_head_calls'] == 1
assert wreport['BFM_inference_calls'] == wreport['physics_steps'] == 0
assert wreport['head_sha256'] == b['head']['sha256'] and wreport['witness_sha256'] == args.witness_sha256
assert wreport['binding_sha256'] == sha(wbp) and wreport['features'] == 1323 and wreport['context_condition'] == 'causal'
assert wreport['exact_extended_query250_features'] is True and wreport['centers_row'] == 2038 and wreport['control'] == 250 and wreport['source_frame'] == 261
assert wreport['pinned_WSL_runtime'] is True and wreport['onnxruntime_version'] == '1.23.2'
assert wreport['intra_op_threads'] == wreport['inter_op_threads'] == 1
assert wreport['execution_mode'] == 'ORT_SEQUENTIAL' and wreport['execution_provider'] == 'CPUExecutionProvider'
assert wreport['source_sha256'] == sha(BASE / 'source_draft_v1/head_activation_witness.py')
wo_path = BASE / 'witness_completion_verification.json'
assert sha(wo_path) == args.witness_owner_sha256
wo = read(wo_path)
assert wo['owner_completion_accounting_passed'] is True and wo['diagnostic_passed'] is True and wo['mode'] == 'witness'
assert wo['raw_python_exit_code'] == wo['diagnostic_exit_code'] == 0 and wo['diagnostic_reasons'] == [] and wo['pins_exact'] == 5230
for path, digest in wo['output_hashes'].items(): assert sha(path) == digest, path
start, child, end = [read(wfolder / name) for name in ('start.json', 'child.json', 'exit.json')]
raw = read(wfolder / 'raw_exit.json')
assert raw['known'] is True and raw['raw_python_exit_code'] == 0 and raw['raw_error'] is None
assert end['raw_child_exit_code'] == end['exit_code'] == 0 and end['error'] is None and end['all_postrun_hashes_exact'] is True
absence = read(wfolder / 'process_absence.json')
assert wo['process_absence'] == absence and absence['wrapper_absent'] is True and absence['child_absent'] is True
assert absence['wrapper_pid'] == start['wrapper_pid'] == child['wrapper_pid'] and absence['child_pid'] == child['child_pid'] and child['handle_acquired'] is True
wlaunch = read(wfolder / 'launch_receipt.json')
assert sha(wfolder / 'launch_receipt.json') == start['receipt_sha256'] == 'f3772bc07cf1a46b4f212cafd8f13ec4ad9f1c81388b35b95e3753027982b1d8'
assert read(wfolder / 'postrun_hashes.json') == wlaunch['input_hashes']
clearance = read(wfolder / 'launch_clearance.json')
assert sha(wfolder / 'launch_clearance.json') == start['clearance_sha256']
assert clearance['launch_receipt_sha256'] == start['receipt_sha256']
assert sha(clearance['review']['path']) == clearance['review']['sha256'] == start['review_sha256'] == 'e7e1798917a748f91395eedb8d2f26ffe0e38b33ac88194a2386ca8229d887e9'
with np.load(wp, allow_pickle=False) as archive: saved = {key: archive[key].copy() for key in archive.files}
assert saved['features'].shape == (1323,) and saved['features'].dtype == np.float32 and np.isfinite(saved['features']).all()
assert saved['actual_context'].shape == (323,) and saved['actual_context'].dtype == np.float32 and saved['actual_context'].tobytes() == saved['features'][1000:].tobytes()
assert saved['normalized_target'].shape == (23,) and saved['normalized_target'].dtype == np.float32 and np.isfinite(saved['normalized_target']).all()
assert saved['default'].dtype == saved['limits'].dtype == np.float64 and saved['span'].dtype == np.float32
delta = saved['span'].astype(np.float64) * saved['normalized_target'].astype(np.float64)
raw_target = saved['default'] + delta
assert saved['delta'].tobytes() == delta.tobytes() and saved['raw_proposal'].tobytes() == raw_target.tobytes()
assert saved['target'].tobytes() == np.clip(raw_target, saved['limits'][:,0], saved['limits'][:,1]).tobytes()
assert str(saved['head_sha256'].item()) == b['head']['sha256'] and str(saved['context_condition'].item()) == 'causal'
assert str(saved['runtime_binary_sha256'].item()) == wreport['runtime_binary_sha256']
assert int(saved['control']) == 250 and int(saved['source_frame']) == 261
assert any(digest == wreport['runtime_binary_sha256'] for digest in pins.values())
for path in (wp, wrp, wo_path, wbp): assert pins[path.as_posix()] == sha(path), path
driver = (BASE / 'source_draft_v1/evaluate_direct_target_student.py').read_text()
assert "record_parity('canonical_initial_full291_parity.json'" in driver
assert "initial_integration.shape==(291,)" in driver
assert "verify_generated_prefix250(trace)" in driver and "verify_actual_query250(proposed)" in driver
for name in ('canonical_initial_full291_parity', 'canonical_prefix250_parity', 'actual_query250_input_parity', 'actual_query250_ownexport_output_parity'):
    assert name in (BASE / 'diagnostic_verdict.py').read_text(), name
baseline = BASE.parent / 'original_bfm_entry250_v1/entry250/trace.npz'
assert pins[baseline.as_posix()] == read(baseline.with_name('report.json'))['trace_sha256']
"""
marker = "for path in [BASE / 'nominal'"
assert marker in new
new = new.replace(marker, insert + '\n' + marker)
new = new.replace("powershell_parse_subject=dict(", "witness_subject=dict(path=wp.as_posix(),sha256=args.witness_sha256), witness_owner_subject=dict(path=wo_path.as_posix(),sha256=args.witness_owner_sha256), completed_witness_verified=True,\n    powershell_parse_subject=dict(")
with (HERE / 'review_concrete.py').open('x', encoding='utf-8', newline='\n') as stream: stream.write(new)
with (HERE / 'source_delta.patch').open('x', encoding='utf-8') as stream:
    stream.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='qualified_witness_review.py', tofile='canonical_review.py')))
compile(new, 'review_concrete.py', 'exec')
print('Canonical reviewer source prepared; no actual review or task execution.')
