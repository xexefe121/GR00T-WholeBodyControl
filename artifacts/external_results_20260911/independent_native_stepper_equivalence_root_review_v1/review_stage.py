"""Concrete request/launcher review; does not launch or construct native models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'independent_native_stepper_equivalence_v1'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def subject(p):
    return dict(path=Path(p).resolve().as_posix(), sha256=sha(p))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['witness','replay'], required=True)
    stage = parser.parse_args().stage
    request_path = BASE / (stage + '_request.json')
    folder = BASE / (stage + '_process')
    receipt_path = folder / 'launch_receipt.json'
    request, receipt = read(request_path), read(receipt_path)
    source_review = read(ROOT / 'review.json')
    assert source_review['source_review_passed'] is True
    for p, digest in source_review['source_and_input_pins'].items():
        assert sha(p) == digest, p
    preparation_path = BASE / 'source_preparation.json'
    preparation = read(preparation_path)
    for p, digest in preparation['input_hashes'].items():
        assert sha(p) == digest, p
    steps, saves = (0, 2) if stage == 'witness' else (21348, 8)
    assert request['stage'] == receipt['stage'] == stage
    assert request['source_preparation_only'] is False
    assert request['native_step_budget'] == receipt['requested_native_steps'] == steps
    assert request['serialization_budget'] == receipt['requested_serializations'] == saves
    assert request['model_inference_calls'] == request['optimizer_updates'] == 0
    assert request['plant_foundation_connected'] is False
    assert receipt['automatic_retry'] is False and receipt['hardware_authorized'] is False
    assert receipt['final_clearance_required'] is True and receipt['execution_selected'] is False
    assert receipt['request_path'] == request_path.as_posix()
    assert receipt['request_sha256'] == sha(request_path)
    assert request['source_preparation_sha256'] == sha(preparation_path)
    assert request['proposal_sha256'] == sha(BASE / 'proposal.json')
    pins = receipt['input_hashes']
    for p, digest in pins.items():
        assert sha(p) == digest, p
    for item in request['input_files']:
        assert pins[item['path']] == item['sha256']
    for p in (request_path, folder/'run.ps1', folder/'run_durable.ps1'):
        assert pins[p.as_posix()] == sha(p)
    spec = importlib.util.spec_from_file_location('reviewed_stage_generator', BASE / 'prepare_stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert receipt['exact_wsl_arguments'] == module.arguments(stage)
    assert (folder / 'run.ps1').read_text() == module.run_text(stage)
    assert (folder / 'run_durable.ps1').read_text() == module.durable_text(stage)
    assert not (folder / 'started.lock').exists()
    assert not (folder / 'launch_clearance.json').exists()
    assert not (BASE / stage).exists()
    if stage == 'replay':
        owner_path = BASE / 'witness_completion_verification.json'
        owner = read(owner_path)
        assert owner['passed'] is True
        assert pins[owner_path.as_posix()] == sha(owner_path)
        for p, digest in owner['output_hashes'].items():
            assert pins[p] == digest and sha(p) == digest
        witness = read(BASE / 'witness/report.json')
        assert witness['passed'] is True and witness['error'] is None
        assert witness['api_counters']['step_attempted'] == witness['api_counters']['step_returned'] == 0
        assert witness['api_counters']['serialization_attempted'] == witness['api_counters']['serialization_returned'] == 2
        assert request['expected_model_mjb'] == (BASE / 'witness/expected_model.mjb').as_posix()
        assert sha(BASE / 'witness/expected_model.mjb') == witness['mjb_sha256']
    result = dict(passed=True, root_selected_single_run=True, stage=stage,
                  request_subject=subject(request_path), launch_receipt_subject=subject(receipt_path),
                  source_review_subject=subject(ROOT / 'review.json'), source_preparation_subject=subject(preparation_path),
                  exact_argv_and_generated_scripts=True, reviewed_input_hashes=pins,
                  selected_native_steps=steps, selected_serializations=saves,
                  model_calls=0, optimizer_updates=0, actual_native_steps=0,
                  actual_serializations=0, launched=False, hardware_authorized=False,
                  root_reviewed_helpers=[subject(BASE / name) for name in
                      ('prepare_stage.py','stage_verdict.py','verify_completed_stage.py','test_stage_preparation.py')],
                  root_helper_tests=subject(BASE / 'root_stage_tests.xml'),
                  scope='One requested adapter verification stage; no policy balance or real-time qualification.')
    output = ROOT / (stage + '_review.json')
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(passed=True, stage=stage, sha256=sha(output), inputs=len(pins))))

if __name__ == '__main__':
    main()
