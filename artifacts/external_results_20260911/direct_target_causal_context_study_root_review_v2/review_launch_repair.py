"""Review only the preserved pre-Python launch failure and exact launcher repair."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / 'direct_target_causal_context_study_v1'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

prior = HERE.parent / 'direct_target_causal_context_study_root_review_v1/review.json'
assert sha(prior) == '3a15c0b9b896e4bbc1c2a661205ee4e595af007f57eb5392c6f211167747f819'
baseline = read(prior)
assert baseline['prelaunch_review_pass'] is True
request = TASK / 'training_request.json'
receipt = TASK / 'training_frozen_inputs.json'
launcher = TASK / 'run_fit_durable_v2.ps1'
assert sha(request) == '284e1e6a69617b6f3f39ab877b92de4e1d426f96bf6abd983fe8d6a163a70c6c'
assert sha(receipt) == '4e292ff3005a330318f99c08aa47241d7b18a752a807befa72e41daae2a30bad'
assert sha(launcher) == '6f114a44e4958598271379ce57be5f553e28ff2c078bfef3da8d3c7759627c5d'
old = (TASK / 'run_fit_durable_v1.ps1').read_text(encoding='utf-8')
expected = old.replace("$processRoot=Join-Path $runRoot 'fit_process'", "$processRoot=Join-Path $runRoot 'fit_process_v2'")
expected = expected.replace('$request.coefficient -ne 1.8188207859141674)', '$request.coefficient -ne 1.8188207859141674d)')
assert expected.encode('utf-8') == launcher.read_bytes()
old_owner = (TASK / 'verify_completed_v2.py').read_text(encoding='utf-8')
expected_owner = old_owner.replace("process=BASE/'fit_process'", "process=BASE/'fit_process_v2'").replace('owner_completion_verification_v2.json', 'owner_completion_verification_v3.json')
assert expected_owner.encode('utf-8') == (TASK / 'verify_completed_v3.py').read_bytes()
failure = read(TASK / 'fit_process/exit.json')
assert failure['child_started'] is False and failure['child_pid'] is None
assert failure['raw_python_exit_code'] is None and failure['error'] == 'Unexpected request.'
assert not (TASK / 'fit').exists() and not (TASK / 'fit_process_v2').exists()
test = read(TASK / 'launcher_v2_test.json')
assert test['passed'] is True and test['PSVersion'].startswith('5.1.')
assert test['actual_corrected_guard_passed'] is True and test['changed_coefficient_rejected'] is True
assert test['json_coefficient_type'] == 'System.Decimal'
frozen = read(receipt)
assert len(frozen['input_sha256']) == 266
assert frozen['source_sha256'] == baseline['source_sha256']
for path, digest in frozen['input_sha256'].items():
    assert sha(path) == digest, path
for name, digest in frozen['source_sha256'].items():
    assert sha(Path(frozen['source_directory']) / name) == digest, name
subjects = {role: {'path': str(path), 'sha256': sha(path)} for role, path in
            [('training_request', request), ('frozen_inputs', receipt), ('launcher', launcher)]}
result = dict(baseline)
result.update(prelaunch_review_pass=True, training_request_sha256=sha(request),
    frozen_receipt_sha256=sha(receipt), launcher_path=str(launcher), launcher_sha256=sha(launcher),
    subjects=subjects, input_sha256=frozen['input_sha256'], source_sha256=frozen['source_sha256'],
    preceding_root_review={'path': str(prior), 'sha256': sha(prior)},
    repair_review={'exact_launcher_delta': True, 'exact_owner_delta': True,
        'failed_attempt_child_never_started': True, 'failed_attempt_model_calls': 0,
        'failed_attempt_optimizer_updates': 0, 'actual_PS51_guard_passed': True,
        'changed_coefficient_rejected': True, 'test_sha256': sha(TASK / 'launcher_v2_test.json'),
        'request_and_all_19_training_sources_unchanged': True,
        'all_266_input_pins_exact': True, 'separate_process_directory': 'fit_process_v2'},
    reviewed=baseline['reviewed'] + ['Root read complete repaired launcher, exact decimal guard, failure record and preserving derivation; checked unchanged trainer/request and exact two launcher substitutions plus owner process/receipt paths.'])
write(HERE / 'review.json', result)
print(json.dumps({'review_path': str(HERE / 'review.json'), 'sha256': sha(HERE / 'review.json'), 'prelaunch_review_pass': True}))
