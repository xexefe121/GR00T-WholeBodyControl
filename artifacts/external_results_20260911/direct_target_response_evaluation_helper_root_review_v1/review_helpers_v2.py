"""Independent source-only correction of a CRLF-sensitive provenance claim.

The original checker and evaluator files are never changed. Tests construct only
synthetic JSON and PowerShell text; no real binding, head, or native call occurs.
"""
import ast
import datetime
import difflib
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
NEW = OUT.parent
BASE = NEW / 'direct_target_causal_response_evaluation_v1'
OLD = NEW / 'direct_target_causal_context_evaluation_v2'
SAME = ['prepare_bound_launcher.py', 'diagnostic_verdict.py',
        'verify_completed_stage.py', 'test_launch_helpers.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def main():
    prep_path = BASE / 'launch_helper_preparation_v2.json'
    source_path = BASE / 'source_preparation.json'
    assert sha(prep_path) == 'd20b5537223f4740df79aa540e0e9dff85d864c82cf830dda5d9730b757e056e'
    prep, source = read(prep_path), read(source_path)
    assert prep['source_sha256'] == source['source_sha256']
    assert len(prep['helper_sha256']) == 8 and len(source['source_sha256']) == 37
    protected = {BASE / name: digest for name, digest in prep['helper_sha256'].items()}
    protected.update({BASE / 'source_draft_v1' / name: digest
                      for name, digest in source['source_sha256'].items()})
    protected.update({prep_path: sha(prep_path), source_path: sha(source_path)})
    for path, digest in protected.items():
        assert sha(path) == digest, str(path)
    original = OUT / 'review_helpers.py'
    preserved = OUT / 'review_helpers_preserved_v1.py'
    with preserved.open('xb') as stream:
        stream.write(original.read_bytes())
    assert sha(preserved) == sha(original)
    protected[original] = sha(original)
    # Reproduce the already reported checker-only failure and preserve its exact
    # stderr separately. This is not a second producer or controller attempt.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    reproduction = subprocess.run([sys.executable, '-B', str(original)],
                                  capture_output=True, text=True, env=env)
    write(OUT / 'original_checker_reproduction.log', reproduction.stdout + '\n' + reproduction.stderr)
    assert reproduction.returncode != 0
    assert 'AssertionError: prepare_bound_launcher.py' in reproduction.stderr
    assert not (OUT / 'review.json').exists() and not (OUT / 'root_tests.xml').exists()
    write(OUT / 'original_checker_reproduction.json', {
        'kind': 'independent_reproduction_of_reported_checker_failure',
        'original_first_attempt_console_not_recovered': True,
        'original_checker_sha256': sha(original),
        'preserved_checker_sha256': sha(preserved),
        'exit_code': reproduction.returncode,
        'log_sha256': sha(OUT / 'original_checker_reproduction.log'),
        'failure': 'AssertionError: prepare_bound_launcher.py',
        'tests_reached': False, 'model_calls': 0, 'native_steps': 0})

    evidence, patches = {}, []
    for name in prep['helper_sha256']:
        old_path, new_path = OLD / name, BASE / name
        old, new = old_path.read_bytes(), new_path.read_bytes()
        old_norm, new_norm = old.replace(b'\r\n', b'\n'), new.replace(b'\r\n', b'\n')
        old_text, new_text = old_norm.decode('utf-8'), new_norm.decode('utf-8')
        entry = {
            'old_path': old_path.as_posix(), 'new_path': new_path.as_posix(),
            'old_sha256': sha(old_path), 'new_sha256': sha(new_path),
            'old_bytes': len(old), 'new_bytes': len(new),
            'old_crlf': old.count(b'\r\n'), 'new_crlf': new.count(b'\r\n'),
            'old_bare_cr': old.count(b'\r') - old.count(b'\r\n'),
            'new_bare_cr': new.count(b'\r') - new.count(b'\r\n'),
            'old_utf8_bom': old.startswith(b'\xef\xbb\xbf'),
            'new_utf8_bom': new.startswith(b'\xef\xbb\xbf'),
            'byte_equal': old == new, 'normalized_bytes_equal': old_norm == new_norm,
            'new_is_exact_old_crlf_to_lf': new == old_norm,
            'normalized_ast_equal': ast.dump(ast.parse(old_text), include_attributes=False)
                == ast.dump(ast.parse(new_text), include_attributes=False),
        }
        if name in SAME:
            assert entry['new_is_exact_old_crlf_to_lf']
            assert entry['normalized_ast_equal']
            assert not any(entry[k] for k in ('old_bare_cr', 'new_bare_cr', 'old_utf8_bom', 'new_utf8_bom'))
        evidence[name] = entry
        protected[old_path] = sha(old_path)
        patches.extend(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                           fromfile='old/' + name, tofile='new/' + name))
    assert [name for name in SAME if not evidence[name]['byte_equal']] == ['prepare_bound_launcher.py']
    launcher = evidence['prepare_bound_launcher.py']
    assert launcher['old_crlf'] == 139 and launcher['new_crlf'] == 0
    assert launcher['old_bytes'] - launcher['new_bytes'] == 139
    write(OUT / 'helper_byte_evidence.json', evidence)
    write(OUT / 'normalized_helper_derivation.patch', ''.join(patches))

    independent_path = NEW / 'direct_target_causal_response_evaluation_independent_review_v1/review.json'
    assert sha(independent_path) == 'e40aca20c419219a8e6d1e4630ef5e4442ac7c1f492d85b009bc94b1460a0121'
    independent = read(independent_path)
    assert independent['source_sha256'] == source['source_sha256']
    assert independent.get('passed', independent.get('source_review_pass')) is True
    protected[independent_path] = sha(independent_path)
    command = [sys.executable, '-B', '-m', 'pytest', '--rootdir=.', '--confcutdir=.', '-q',
               'test_launch_helpers.py', 'test_release_helpers.py', '--junitxml=' + str(OUT / 'root_tests_v2.xml')]
    run = subprocess.run(command, cwd=BASE, capture_output=True, text=True, env=env)
    write(OUT / 'root_tests_v2.log', run.stdout + '\n' + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    xml = ET.parse(OUT / 'root_tests_v2.xml').getroot()
    suites = [xml] if xml.tag == 'testsuite' else list(xml)
    assert sum(int(s.attrib['tests']) for s in suites) == 41
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    for path, digest in protected.items():
        assert sha(path) == digest, str(path)
    absent = ['release_reviews.json', 'witness_binding.json', 'evaluation_binding.json',
              'witness_process', 'evaluation_process', 'head_witness', 'nominal']
    assert all(not (BASE / name).exists() for name in absent)
    receipts = ['review_helpers_preserved_v1.py', 'original_checker_reproduction.log',
                'original_checker_reproduction.json', 'helper_byte_evidence.json',
                'normalized_helper_derivation.patch', 'root_tests_v2.xml', 'root_tests_v2.log']
    review = dict(
        passed=True, helper_review_pass=True, source_review_pass=True,
        reviewer='/root/expert_resume', scope='independent helper source and synthetic tests only',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_sha256=prep['helper_sha256'], source_sha256=source['source_sha256'],
        helper_preparation_sha256=sha(prep_path), source_preparation_sha256=sha(source_path),
        independent_runtime_source_review_sha256=sha(independent_path),
        requested_root_helper_tests=41, actual_independent_helper_tests=41,
        test_exit_code=run.returncode, test_command=command,
        byte_identical_helpers=[name for name in SAME if evidence[name]['byte_equal']],
        newline_only_helpers=['prepare_bound_launcher.py'],
        normalized_source_and_ast_identical_helpers=SAME,
        corrected_provenance_claim='Launcher differs only by 139 CRLF-to-LF conversions. Other three helpers are byte-identical. The earlier four-byte-identical assertion was false; no evaluator fix was needed.',
        reviewed_semantics=[
            'Causal71000 actual subjects use corrected fitv2 and16 explicit release/data/energy roles.',
            'Source, owner, independent fit audit, release and exact eight-helper review bind before model-ready configuration.',
            'Original1569 plus conditional250, one WSL witness and full291/query250 gates retained.',
            'Hidden captured-handle launch, exact receipt/clearance subjects, CreateNew locks, known raw exit, separate diagnostic verdict and complete postrun map retained.',
            'Inventory retains native/reference/runtime closure and excludes recursive training corpora.',
            '41 synthetic tests include actual PowerShell path normalization and four parser-only template checks.'],
        preserved_original_checker_sha256=sha(original),
        evidence_sha256={name: sha(OUT / name) for name in receipts},
        input_sha256={path.as_posix(): digest for path, digest in protected.items()},
        all_rechecked_inputs_exact=True, evaluator_files_changed=False,
        actual_endpoint_selected=False, actual_binding_created=False,
        model_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0,
        actual_controller_processes_launched=0, writer_sha256=sha(__file__))
    write(OUT / 'review.json', review)
    print(json.dumps({'passed': True, 'path': (OUT / 'review.json').as_posix(),
                      'sha256': sha(OUT / 'review.json'), 'tests': 41,
                      'evidence_sha256': sha(OUT / 'helper_byte_evidence.json')}))


if __name__ == '__main__':
    main()
