"""Saved metadata/hash checks only; never imports the native runner or dispatches."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
NEW = BASE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def subject(path):
    path = Path(path).resolve()
    return {'path': path.as_posix(), 'sha256': sha(path)}


def main():
    request_path = BASE / 'clock_request.json'
    launch_path = BASE / 'clock_process/launch_receipt.json'
    request, launch = read(request_path), read(launch_path)
    assert sha(request_path) == '0c22640dfb617a20c31aff7356434a6253096b0fc4cbed38aae64faeb14e75eb'
    assert sha(launch_path) == 'f3eb4aca76e45b56761a95017090359a4de21d6edd1f3d7cacc71754158f1713'
    assert launch['request_sha256'] == sha(request_path)
    assert request['execution_selected'] is False and launch['execution_selected'] is False
    request_pins = {entry['path']: entry['sha256'] for entry in request['input_files']}
    assert len(request_pins) == len(request['input_files']) == 3742
    pins = launch['input_hashes']
    assert len(pins) == 3745
    assert all(pins.get(path) == digest for path, digest in request_pins.items())
    actual = {path: sha(path) for path in pins}
    assert actual == pins, 'Launch input changed'
    for path in request['roles'].values():
        assert request_pins[path] == actual[path]
    audit_review = Path(request['roles']['saved_stage_audit_source_review'])
    assert sha(audit_review) == '278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7'
    audit = read(audit_review)
    audit_preparation = read(request['roles']['saved_stage_audit_preparation'])
    assert audit['passed'] is True
    assert audit['source_sha256'] == audit_preparation['source_sha256']
    old = read(NEW / 'independent_plant_process_clock_v1/clock_request.json')
    unchanged = ['requested_controls', 'main_controls', 'hold_controls', 'native_step_budget',
                 'serialization_budget', 'model_inference_calls', 'optimizer_updates',
                 'other_oracle_native_steps', 'epoch_lead_ns', 'epoch_rebase_allowed',
                 'debt_abort_steps', 'elapsed_abort_ns', 'native_bundle',
                 'expected_model_sha256', 'command_table_sha256']
    assert all(request[key] == old[key] for key in unchanged)
    assert all(request['roles'][key] == value for key, value in old['roles'].items())
    assert request['watchdog_budgets'] == {'setup_ns': 240000000000, 'plant_ns': 120000000000,
                                         'preservation_ns': 180000000000}
    assert request['outer_process_timeout_seconds'] == 555
    args = launch['exact_wsl_arguments']
    assert args[args.index('--cd') + 1] == '/'
    assert args[args.index('timeout') + 1:args.index('timeout') + 4] == [
        '--signal=TERM', '--kill-after=5s', '555s']
    assert args[args.index('--request') + 1].endswith('/independent_plant_clock_timeout_correction_v1/clock_request.json')
    parse_path = BASE / 'concrete_launcher_parse_v1.json'
    parse = read(parse_path)
    assert parse['normalization_passed'] is True
    assert parse['literal_separator_length'] == 1 and parse['literal_separator_codepoint'] == 92
    assert all(item['passed'] is True and item['errors'] == [] for item in parse['launchers'])
    absent_paths = [BASE / name for name in ['clock_process/launch_clearance.json',
        'clock_process/started.lock', 'clock_process/start.json', 'clock_process/child.json',
        'clock_process/exit.json', 'clock_process/stdout.log', 'clock_process/stderr.log',
        'run', 'stage_receipts']]
    assert all(not path.exists() for path in absent_paths), 'Unexpected launch or run artifact'
    subjects = {name: subject(BASE / name) for name in ['clock_request.json',
        'clock_process/launch_receipt.json', 'clock_process/run.ps1', 'clock_process/run_durable.ps1',
        'prepare_concrete_packet.py', 'prepare_clock_stage.py', 'input_scope_derivation.json',
        'concrete_launcher_parse_v1.json', 'verify_concrete_metadata_v1.py']}
    report = {'passed': True, 'preparation_only': True, 'execution_selected': False,
        'clearance_created': False, 'dispatch_performed': False, 'subjects': subjects,
        'request_pin_count': len(request_pins), 'launch_pin_count': len(pins),
        'all_launch_pins_exact': True, 'input_sha256': actual,
        'saved_auditor_review': subject(audit_review), 'saved_auditor_source_map_exact': True,
        'unchanged_original_scope_fields': unchanged, 'original_roles_unchanged': True,
        'watchdog_budgets': request['watchdog_budgets'], 'outer_timeout_seconds': 555,
        'exact_wsl_arguments': args, 'absent': [path.as_posix() for path in absent_paths],
        'native_steps': 0, 'MJB_serializations': 0, 'model_calls': 0, 'worker_processes': 0}
    output = BASE / 'concrete_metadata_preparation_v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': True, 'report_sha256': sha(output), 'subjects': subjects,
                      'launch_pin_count': len(pins), 'dispatched': False}))


if __name__ == '__main__':
    main()
