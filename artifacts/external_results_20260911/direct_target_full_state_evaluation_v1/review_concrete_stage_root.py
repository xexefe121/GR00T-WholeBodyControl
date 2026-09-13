"""Root verification of one concrete, unlaunched simulation stage; no execution."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main(args):
    binding_path = BASE / (args.mode + '_binding.json')
    folder = BASE / (args.mode + '_process')
    launch_path = folder / 'launch_receipt.json'
    assert sha(binding_path) == args.binding_sha256
    assert sha(launch_path) == args.launch_sha256
    assert not (folder / 'started.lock').exists()
    assert not (folder / 'launch_clearance.json').exists()
    binding, launch = read(binding_path), read(launch_path)
    assert launch['binding_path'] == binding_path.as_posix()
    assert launch['binding_sha256'] == args.binding_sha256
    assert launch['automatic_retry'] is False and launch['hardware_authorized'] is False
    assert launch['raw_python_exit_preserved'] is True
    assert launch['requested_main_controls'] == (1569 if args.mode == 'evaluation' else 0)
    assert launch['conditional_hold_controls'] == (250 if args.mode == 'evaluation' else 0)
    assert launch['expected_separate_head_calls'] == (1 if args.mode == 'witness' else 0)
    assert binding['physics_authorized'] is (args.mode == 'evaluation')
    assert binding['ordinary_final_step'] == 65000
    assert binding['requested_main_controls'] == 1569 and binding['conditional_hold_controls'] == 250
    for field in ('filters_enabled', 'compiled_preview_enabled', 'hardware_authorized', 'clock_foundation_connected'):
        assert binding[field] is False
    assert binding['learned_BFM_calls'] == 0
    assert binding['head']['sha256'] == '045f04138610a06a0171a899d002cfa4f03e30e316dc9442199c833c16e43902'
    assert binding['reviews']['release']['sha256'] == 'b378683c10f7de20243f6bbdd734895e9fb4a802b4382cdcc78d6c11e03b1ad5'
    linux = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/' + BASE.name
    script = 'head_activation_witness.py' if args.mode == 'witness' else 'evaluate_direct_target_student.py'
    expected_arguments = ['-d', 'Ubuntu-22.04', '--cd', '/', '--', 'bash',
        '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
        'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=' + linux + '/source_draft_v1',
        '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python', '-B', linux + '/source_draft_v1/' + script]
    assert launch['exact_wsl_arguments'] == expected_arguments
    pins = launch['input_hashes']
    for entry in binding['input_files']:
        assert pins[entry['path']] == entry['sha256']
    for path, digest in pins.items():
        assert sha(path) == digest, path
    for name in ('run.ps1', 'run_durable.ps1'):
        assert pins[(folder / name).as_posix()] == sha(folder / name)
    result = dict(passed=True, reviewer='root', reviewed_utc=datetime.now(timezone.utc).isoformat(),
        binding_subject=dict(path=binding_path.as_posix(), sha256=args.binding_sha256),
        launch_receipt_subject=dict(path=launch_path.as_posix(), sha256=args.launch_sha256),
        all_current_input_pins_exact=True, checked_input_pins=len(pins), exact_launch_arguments=True,
        selected_single_run=True, mode=args.mode, model_calls=0, native_steps=0,
        writer_sha256=sha(__file__), original_acceptance_unchanged=True)
    destination = BASE / (args.mode + '_root_concrete_review.json')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=destination.as_posix(), sha256=sha(destination), pins=len(pins))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['witness', 'evaluation'], required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--launch-sha256', required=True)
    main(parser.parse_args())
