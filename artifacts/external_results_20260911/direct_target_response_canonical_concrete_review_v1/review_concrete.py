"""Independent read-only review of the single causal71000 canonical package after completed witness."""
from pathlib import Path
import hashlib, json, re, argparse
import numpy as np
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'direct_target_causal_response_evaluation_v1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binding-sha256', required=True)
parser.add_argument('--launch-sha256', required=True)
parser.add_argument('--witness-owner-sha256', required=True)
parser.add_argument('--witness-sha256', required=True)
parser.add_argument('--binding-pins', required=True, type=int)
parser.add_argument('--launch-pins', required=True, type=int)
args = parser.parse_args()
BINDING_SHA, LAUNCH_SHA = args.binding_sha256, args.launch_sha256
assert args.binding_pins > 5226 and args.launch_pins == args.binding_pins + 5
RELEASE_SHA = '0ae9699fcb2b70ff33948223e9630a0f8692751baf2fdb27bbb191e7d76d5480'
SOURCE_SHA = 'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'
HELPER_SHA='b95fa1160498af88cf3fe38c80aaf12b8707ac3d18f25ed69fdfe54c9920d16f'
AUDIT_SHA = '427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff'
OWNER_SHA = 'a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(path): return Path(path).resolve().as_posix().lower()
def entry(value):
    assert sha(value['path']) == value['sha256'], value['path']
    return read(value['path'])

folder = BASE / 'evaluation_process'
bp, lp = BASE / 'evaluation_binding.json', folder / 'launch_receipt.json'
assert sha(bp) == BINDING_SHA and sha(lp) == LAUNCH_SHA
b, launch = read(bp), read(lp)
assert launch['kind'] == 'one_selected_context_target_evaluation'
assert launch['context_condition'] == b['context_condition'] == 'causal'
assert launch['binding_path'] == bp.as_posix() and launch['binding_sha256'] == BINDING_SHA
assert launch['requested_main_controls'] == 1569 and launch['conditional_hold_controls'] == 250
assert launch['expected_separate_head_calls'] == 0 and b['expected_head_calls'] == 1
assert launch['raw_python_exit_preserved'] is True and launch['failed_or_incomplete_diagnostic_exit'] == 2
assert launch['automatic_retry'] is False and launch['hardware_authorized'] is False
assert b['physics_authorized'] is True and b['root_authorized_evaluation'] is True
assert b['ordinary_final_step'] == 71000 and b['architecture'] == [1323, 256, 256, 23]
assert b['hidden_activation'] == 'ELU' and b['head_output'] == 'normalized_target'
assert b['export_input_dtype'] == b['export_output_dtype'] == 'float32' and b['export_internal_dtype'] == 'float64'
assert b['span_contract'] == 'existing_float32_joint_span_promoted_float64'
assert b['requested_main_controls'] == 1569 and b['conditional_hold_controls'] == 250
for key in ('filters_enabled', 'compiled_preview_enabled', 'hardware_authorized', 'clock_foundation_connected', 'recursive_training_hashes'):
    assert b[key] is False, key
assert b['learned_BFM_calls'] == 0
assert b['head']['sha256'] == 'c70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a'
assert b['checkpoint']['sha256'] == '395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d'

linux = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/' + BASE.name
arguments = ['-d', 'Ubuntu-22.04', '--cd', '/', '--', 'bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=' + linux + '/source_draft_v1', '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python', '-B',
    linux + '/source_draft_v1/evaluate_direct_target_student.py']
assert launch['exact_wsl_arguments'] == arguments
run = (folder / 'run.ps1').read_text()
literal_arguments = re.search(r'\$arguments = @\((.*?)\n\)', run, re.S).group(1)
assert re.findall(r"'([^']*)'", literal_arguments) == arguments
assert BINDING_SHA in run and '--mode \'evaluation\'' in run

release = entry(b['reviews']['release']); source = entry(b['reviews']['source'])
audit = entry(b['root_training_audit']); owner = entry(b['fit_owner_completion'])
assert b['reviews']['release']['sha256'] == RELEASE_SHA and release['release_review_pass'] is True
assert b['reviews']['source']['sha256'] == SOURCE_SHA and source['passed'] is True
assert b['root_training_audit']['sha256'] == AUDIT_SHA and audit['evidence_audit_passed'] is True and audit['export_qualified'] is True
assert b['fit_owner_completion']['sha256'] == OWNER_SHA
for key in ('owner_verification_passed', 'accounting_passed', 'completion_passed', 'numerical_completion_passed', 'raw_exit_known', 'all_postrun_pins_exact', 'processes_absent'):
    assert owner[key] is True, key
assert owner['raw_python_exit_code'] == owner['exit_code'] == 0
assert owner['condition'] == release['context_condition'] == 'causal'
assert release['ordinary_final_step'] == 71000 and release['selected_witness_calls'] == 1
assert release['physical_qualification'] is False and release['per_stage_concrete_review_required'] is True
roles=release['subjects']
assert len(roles)==16 and set(roles)==set(audit['direct_subject_sha256'])
assert release['selected_main_controls']==1569 and release['conditional_hold_controls']==250
config=read(BASE/'release_reviews.json');helper_entry=config['launch_helper_review'];helper=entry(helper_entry)
assert helper_entry['sha256']==HELPER_SHA and helper['passed'] is True
assert len(source['source_sha256'])==37
audit_pins = {canonical(p): h for p, h in audit['input_sha256'].items()}
for role, subject in roles.items():
    assert b[role] == subject and release['subjects'][role] == subject, role
    assert owner['direct_subject_sha256'][role] == release['direct_subject_sha256'][role] == audit['direct_subject_sha256'][role] == subject['sha256'], role
    assert audit_pins[canonical(subject['path'])] == subject['sha256'], role
    assert sha(subject['path']) == subject['sha256'], role
assert release['direct_subject_sha256']['root_training_audit'] == AUDIT_SHA
assert release['direct_subject_sha256']['fit_owner_completion'] == OWNER_SHA
assert release['source_review_sha256']==SOURCE_SHA and release['helper_review_sha256']==HELPER_SHA
audit_owner_path = HERE.parent / 'direct_target_response_balanced_fit_independent_v3/owner_completion.json'
assert sha(audit_owner_path) == release['audit_owner_sha256'] == '0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3'
assert read(audit_owner_path)['evidence_audit_passed'] is True

pins = launch['input_hashes']
assert pins[(BASE/'release_reviews.json').as_posix()]==sha(BASE/'release_reviews.json')
assert pins[helper_entry['path']]==HELPER_SHA
assert len(b['input_files']) == args.binding_pins and len(pins) == args.launch_pins
assert len({canonical(p) for p in pins}) == len(pins)
assert len({canonical(v['path']) for v in b['input_files']}) == len(b['input_files'])
for value in b['input_files']: assert pins[value['path']] == value['sha256']
for path, digest in pins.items(): assert sha(path) == digest, path
for name, digest in source['source_sha256'].items():
    path = BASE / 'source_draft_v1' / name
    assert sha(path) == digest and pins[path.as_posix()] == digest, name
for name, digest in helper['helper_sha256'].items():
    path = BASE / name
    assert sha(path) == digest and pins[path.as_posix()] == digest, name
for name in ('run.ps1', 'run_durable.ps1'):
    assert pins[(folder / name).as_posix()] == sha(folder / name)
ps = read(HERE / 'powershell_parse.json')
assert ps['passed'] is True and ps['files'] == {name: sha(folder / name) for name in ('run.ps1', 'run_durable.ps1')}

# Completed witness is evidence only: never call its head again during review.
wfolder = BASE / 'witness_process'
wbp = BASE / 'witness_binding.json'
wb = read(wbp)
assert sha(wbp) == '2a56cf0ab008f866625a336c8d6445fec2cebcfb4d55f26e8cf3995c0fa33f2c'
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
assert wo['raw_python_exit_code'] == wo['diagnostic_exit_code'] == 0 and wo['diagnostic_reasons'] == [] and wo['pins_exact'] == 5231
for path, digest in wo['output_hashes'].items(): assert sha(path) == digest, path
start, child, end = [read(wfolder / name) for name in ('start.json', 'child.json', 'exit.json')]
raw = read(wfolder / 'raw_exit.json')
assert raw['known'] is True and raw['raw_python_exit_code'] == 0 and raw['raw_error'] is None
assert end['raw_child_exit_code'] == end['exit_code'] == 0 and end['error'] is None and end['all_postrun_hashes_exact'] is True
absence = read(wfolder / 'process_absence.json')
assert wo['process_absence'] == absence and absence['wrapper_absent'] is True and absence['child_absent'] is True
assert absence['wrapper_pid'] == start['wrapper_pid'] == child['wrapper_pid'] and absence['child_pid'] == child['child_pid'] and child['handle_acquired'] is True
wlaunch = read(wfolder / 'launch_receipt.json')
assert sha(wfolder / 'launch_receipt.json') == start['receipt_sha256'] == '75ce87dbc5ffcbf83ffdaff719be5194bf1c4fe2b76a7dfc5b9ebd2eeb07255c'
assert read(wfolder / 'postrun_hashes.json') == wlaunch['input_hashes']
clearance = read(wfolder / 'launch_clearance.json')
assert sha(wfolder / 'launch_clearance.json') == start['clearance_sha256']
assert clearance['launch_receipt_sha256'] == start['receipt_sha256']
assert sha(clearance['review']['path']) == clearance['review']['sha256'] == start['review_sha256'] == '20268e368ad0dc9453a355f28288b85d8f495b3ee00e3f156161aa7dd3d00b46'
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

for path in [BASE / 'nominal', BASE / 'post_lifecycle_hold_5s', BASE / 'pilot_outcome.json', BASE / 'canonical_initial_snapshot.npz'] + [folder / name for name in ('started.lock', 'start.json', 'child.json', 'launch_clearance.json', 'raw_exit.json', 'exit.json')]:
    assert not path.exists(), path
assert sha(bp) == BINDING_SHA and sha(lp) == LAUNCH_SHA
result = dict(passed=True, reviewer='review_continuation', reviewed_utc=datetime.now(timezone.utc).isoformat(),
    binding_subject=dict(path=bp.as_posix(), sha256=BINDING_SHA), launch_receipt_subject=dict(path=lp.as_posix(), sha256=LAUNCH_SHA),
    all_current_input_pins_exact=True, checked_binding_pins=args.binding_pins, checked_input_pins=args.launch_pins,
    exact_launch_arguments=True, release_subjects_checked=16, source_modules_checked=len(source['source_sha256']),
    helper_files_checked=len(helper['helper_sha256']), helper_review_sha256=HELPER_SHA, source_review_sha256=SOURCE_SHA, release_review_sha256=RELEASE_SHA,
    root_training_audit_sha256=AUDIT_SHA, fit_owner_completion_sha256=OWNER_SHA,
    context_condition='causal', ordinary_final_step=71000, features=1323, selected_single_run=True, mode='evaluation',
    permitted_additional_witness_calls=0, requested_main_controls=1569, conditional_hold_controls=250, maximum_native_steps=18190, learned_BFM_calls=0,
    task_model_calls=0, ORT_calls=0, native_steps=0, dispatch_performed=False,
    original_acceptance_unchanged=True, canonical_concrete_review_passed=True, writer_sha256=sha(__file__),
    witness_subject=dict(path=wp.as_posix(),sha256=args.witness_sha256), witness_owner_subject=dict(path=wo_path.as_posix(),sha256=args.witness_owner_sha256), completed_witness_verified=True,
    powershell_parse_subject=dict(path=(HERE/'powershell_parse.json').as_posix(),sha256=sha(HERE/'powershell_parse.json')),
    reviewed_launch_semantics=['CreateNew execution lock and output preservation','exact final review binding and launch subjects required before dispatch','raw Python exit retained separately from diagnostic child and wrapper exits','captured child handle before waiting; unknown exit fails','all launch inputs rehashed after run and clearance identity checked','owner receipt separately requires process absence, output identities and diagnostic meaning'])
with (HERE / 'review.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(path=(HERE/'review.json').as_posix(),sha256=sha(HERE/'review.json'),pins=len(pins))))
