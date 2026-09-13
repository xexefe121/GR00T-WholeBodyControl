"""Independent read-only review of the single causal68000 WSL witness package."""
from pathlib import Path
import hashlib, json, re
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'direct_target_causal_context_evaluation_v2'
BINDING_SHA = '1502f0365a29c5ec8b21b2547f158a37d38999fb90260b34371e89a319e4006d'
LAUNCH_SHA = 'f3772bc07cf1a46b4f212cafd8f13ec4ad9f1c81388b35b95e3753027982b1d8'
RELEASE_SHA = 'c8002f3667de0627695262338bb5bd7a0ffeadfe85c9f003b6baa5b8d53e5dc4'
SOURCE_SHA = '949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29'
AUDIT_SHA = 'ab65dd600ac6ad8e1ff7093cc8e0a56f40285d09eca55d212124f08bd789c2af'
OWNER_SHA = '8e6f1bfde3249bf8b8ada99a1c395c6cc3df57ce05f59ab0d91192aef466c979'

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

folder = BASE / 'witness_process'
bp, lp = BASE / 'witness_binding.json', folder / 'launch_receipt.json'
assert sha(bp) == BINDING_SHA and sha(lp) == LAUNCH_SHA
b, launch = read(bp), read(lp)
assert launch['kind'] == 'one_selected_context_target_witness'
assert launch['context_condition'] == b['context_condition'] == 'causal'
assert launch['binding_path'] == bp.as_posix() and launch['binding_sha256'] == BINDING_SHA
assert launch['requested_main_controls'] == launch['conditional_hold_controls'] == 0
assert launch['expected_separate_head_calls'] == b['expected_head_calls'] == 1
assert launch['raw_python_exit_preserved'] is True and launch['failed_or_incomplete_diagnostic_exit'] == 2
assert launch['automatic_retry'] is False and launch['hardware_authorized'] is False
assert b['physics_authorized'] is False and b['root_authorized_witness'] is True
assert b['ordinary_final_step'] == 68000 and b['architecture'] == [1323, 256, 256, 23]
assert b['hidden_activation'] == 'ELU' and b['head_output'] == 'normalized_target'
assert b['export_input_dtype'] == b['export_output_dtype'] == 'float32' and b['export_internal_dtype'] == 'float64'
assert b['span_contract'] == 'existing_float32_joint_span_promoted_float64'
assert b['requested_main_controls'] == 1569 and b['conditional_hold_controls'] == 250
for key in ('filters_enabled', 'compiled_preview_enabled', 'hardware_authorized', 'clock_foundation_connected', 'recursive_training_hashes'):
    assert b[key] is False, key
assert b['learned_BFM_calls'] == 0
assert b['head']['sha256'] == 'd61915c1bf30b16431660134be6057855bbc7befeb620630a5b36dd506738f5c'
assert b['checkpoint']['sha256'] == '10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd'

linux = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/' + BASE.name
arguments = ['-d', 'Ubuntu-22.04', '--cd', '/', '--', 'bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=' + linux + '/source_draft_v1', '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python', '-B',
    linux + '/source_draft_v1/head_activation_witness.py']
assert launch['exact_wsl_arguments'] == arguments
run = (folder / 'run.ps1').read_text()
literal_arguments = re.search(r'\$arguments = @\((.*?)\n\)', run, re.S).group(1)
assert re.findall(r"'([^']*)'", literal_arguments) == arguments
assert BINDING_SHA in run and '--mode \'witness\'' in run

release = entry(b['reviews']['release']); source = entry(b['reviews']['source'])
audit = entry(b['root_training_audit']); owner = entry(b['fit_owner_completion'])
assert b['reviews']['release']['sha256'] == RELEASE_SHA and release['release_review_pass'] is True
assert b['reviews']['source']['sha256'] == SOURCE_SHA and source['passed'] is True
assert b['root_training_audit']['sha256'] == AUDIT_SHA and audit['evidence_audit_passed'] is True and audit['export_qualified'] is True
assert b['fit_owner_completion']['sha256'] == OWNER_SHA
for key in ('owner_verification_passed', 'accounting_passed', 'paired_completion_passed', 'numerical_completion_passed', 'raw_exit_known', 'all_postrun_pins_exact', 'processes_absent'):
    assert owner[key] is True, key
assert owner['raw_python_exit_code'] == owner['exit_code'] == 0
assert owner['condition'] == release['context_condition'] == 'causal'
assert release['ordinary_final_step'] == 68000 and release['selected_witness_calls'] == 1
assert release['physical_qualification'] is False and release['per_stage_concrete_review_required'] is True
roles = audit['condition_subjects']['causal']
assert len(roles) == 18
audit_pins = {canonical(p): h for p, h in audit['input_sha256'].items()}
for role, subject in roles.items():
    assert b[role] == subject and release['subjects'][role] == subject, role
    assert owner['direct_subject_sha256'][role] == release['direct_subject_sha256'][role] == subject['sha256'], role
    assert audit_pins[canonical(subject['path'])] == subject['sha256'], role
    assert sha(subject['path']) == subject['sha256'], role
assert release['direct_subject_sha256']['root_training_audit'] == AUDIT_SHA
assert release['direct_subject_sha256']['fit_owner_completion'] == OWNER_SHA
assert release['source_and_helper_review_sha256'] == SOURCE_SHA
audit_owner_path = HERE.parent / 'direct_target_context_pair_fit_independent_v1/owner_completion.json'
assert sha(audit_owner_path) == release['audit_owner_sha256'] == '7f1806a2132ea3879457a3bb2705105b3075835417151cc9003143974bc99dd9'
assert read(audit_owner_path)['evidence_audit_passed'] is True

pins = launch['input_hashes']
assert len(b['input_files']) == 5225 and len(pins) == 5230
assert len({canonical(p) for p in pins}) == len(pins)
assert len({canonical(v['path']) for v in b['input_files']}) == len(b['input_files'])
for value in b['input_files']: assert pins[value['path']] == value['sha256']
for path, digest in pins.items(): assert sha(path) == digest, path
for name, digest in source['source_sha256'].items():
    path = BASE / 'source_draft_v1' / name
    assert sha(path) == digest and pins[path.as_posix()] == digest, name
for name, digest in source['helper_sha256'].items():
    path = BASE / name
    assert sha(path) == digest and pins[path.as_posix()] == digest, name
for name in ('run.ps1', 'run_durable.ps1'):
    assert pins[(folder / name).as_posix()] == sha(folder / name)
ps = read(HERE / 'powershell_parse.json')
assert ps['passed'] is True and ps['files'] == {name: sha(folder / name) for name in ('run.ps1', 'run_durable.ps1')}
for path in [BASE / 'head_witness', BASE / 'nominal', BASE / 'post_lifecycle_hold_5s'] + [folder / name for name in ('started.lock', 'start.json', 'child.json', 'launch_clearance.json', 'raw_exit.json', 'exit.json')]:
    assert not path.exists(), path
assert sha(bp) == BINDING_SHA and sha(lp) == LAUNCH_SHA
result = dict(passed=True, reviewer='review_continuation', reviewed_utc=datetime.now(timezone.utc).isoformat(),
    binding_subject=dict(path=bp.as_posix(), sha256=BINDING_SHA), launch_receipt_subject=dict(path=lp.as_posix(), sha256=LAUNCH_SHA),
    all_current_input_pins_exact=True, checked_binding_pins=5225, checked_input_pins=5230,
    exact_launch_arguments=True, release_subjects_checked=18, source_modules_checked=len(source['source_sha256']),
    helper_files_checked=len(source['helper_sha256']), source_review_sha256=SOURCE_SHA, release_review_sha256=RELEASE_SHA,
    root_training_audit_sha256=AUDIT_SHA, fit_owner_completion_sha256=OWNER_SHA,
    context_condition='causal', ordinary_final_step=68000, features=1323, selected_single_run=True, mode='witness',
    permitted_witness_head_calls=1, permitted_BFM_calls=0, permitted_native_steps=0,
    task_model_calls=0, ORT_calls=0, native_steps=0, dispatch_performed=False,
    original_acceptance_unchanged=True, canonical_launch_cleared=False, writer_sha256=sha(__file__),
    powershell_parse_subject=dict(path=(HERE/'powershell_parse.json').as_posix(),sha256=sha(HERE/'powershell_parse.json')),
    reviewed_launch_semantics=['CreateNew execution lock and output preservation','exact final review binding and launch subjects required before dispatch','raw Python exit retained separately from diagnostic child and wrapper exits','captured child handle before waiting; unknown exit fails','all launch inputs rehashed after run and clearance identity checked','owner receipt separately requires process absence, output identities and diagnostic meaning'])
with (HERE / 'review.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(path=(HERE/'review.json').as_posix(),sha256=sha(HERE/'review.json'),pins=len(pins))))
