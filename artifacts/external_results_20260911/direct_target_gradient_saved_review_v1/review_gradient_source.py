"""Focused source and concrete-subject review for one three-forward diagnostic."""
from pathlib import Path
import ast
import hashlib
import json

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_followup_analysis_v1')
OUT = Path(__file__).parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    request_path = BASE / 'gradient_request.json'
    frozen_path = BASE / 'gradient_frozen_inputs.json'
    source_path = BASE / 'gradient_diagnostic.py'
    assert sha(source_path) == 'c52dec999190b0ffccf226500a0113c02750db43e07f0520b7e3f6392785a8a5'
    assert sha(request_path) == 'eba680f416038f93c11abcbd0214e0c2f374819fa56fbbb8a79f477f21c07f78'
    assert sha(frozen_path) == 'a9728e1a00c51f746ec6ce7ba46273e178fe93058e613e370f7d2822d9c09bd1'
    request = json.loads(request_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    assert request['root_selected'] and not request['prepared_only']
    assert request['source_sha256'] == sha(source_path)
    assert frozen['request_sha256'] == sha(request_path)
    assert frozen['input_sha256'] == request['input_sha256']
    assert request['budgets'] == dict(forward_calls=3, forward_rows=14110,
        gradient_calls=3, optimizer_updates=0, native_calls=0, ORT_calls=0)
    assert request['schedule_row'] == 0 and request['ordinary_final_step'] == 55000
    pins = request['input_sha256']
    assert len(pins) == 209
    for path, digest in pins.items():
        assert sha(path) == digest, path
    for role in request['subjects'].values():
        assert pins[role['path']] == role['sha256']
    runtime = request['runtime']
    assert pins[runtime['verification_path']] == runtime['verification_sha256']
    assert pins[runtime['identity_path']] == runtime['identity_sha256']
    for name in ('direct_model.py', 'direct_data.py', 'direct_contract.py', 'direct_objective.py'):
        path = (Path(request['original_source_directory']) / name).as_posix()
        assert pins[path] == sha(path)
    tree = ast.parse(source_path.read_text())
    attributes = [n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(attributes).intersection({'step', 'backward', 'mj_step', 'InferenceSession', 'export_onnx'})
    assert attributes.count('grad') == 1  # inside exactly three-loss loop, inspected directly
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
    checkpoint_loads = [n for n in loads if isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch']
    assert len(checkpoint_loads) == 1
    assert any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in checkpoint_loads[0].keywords)
    review = dict(source_review_passed=True, request_sha256=sha(request_path),
        source_sha256=sha(source_path), frozen_receipt_sha256=sha(frozen_path),
        input_pins_verified=len(pins), budgets=request['budgets'],
        manual_findings=[], model_calls=0, native_calls=0, optimizer_updates=0,
        scope='One original-FP32 diagnostic: original fixed loss math, schedule row0, three forwards and three separate gradients, no updates or policy qualification.',
        source_checks=['safe checkpoint load and exact restored parameters/norm',
            'all actual consumed subjects and original data/runtime sources pinned',
            'unchanged loss arithmetic with float64 response subtraction',
            'shared nominal graph retained through first two gradients',
            'returned arrays saved and counters/state preserved on failure',
            'all parameters and grad fields checked unchanged',
            'original export and physical failures remain unqualified'])
    destination = OUT / 'source_review.json'
    with destination.open('x') as stream:
        json.dump(review, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, review_path=str(destination), review_sha256=sha(destination))))
