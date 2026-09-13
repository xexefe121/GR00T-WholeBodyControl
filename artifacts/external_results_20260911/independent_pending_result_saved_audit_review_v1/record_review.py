"""Freeze completed independent source and synthetic evidence; no runtime calls."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PACKAGE = BASE / 'independent_plant_pending_result_saved_audit_v1'
SOURCE = PACKAGE / 'source_draft_v2'
PREP = PACKAGE / 'source_preparation_v2.json'
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    assert expected is None or value == expected, path
    assert str(path) not in INPUTS or INPUTS[str(path)] == value
    INPUTS[str(path)] = value
    return value


assert pin(PREP) == '09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'
prep = json.loads(PREP.read_text())
checks = json.loads((HERE/'pretest_checks.json').read_text())
for path, value in checks['input_sha256'].items():
    pin(path, value)
actual = {p.name: pin(p) for p in SOURCE.glob('*.py')}
assert actual == prep['source_sha256'] == checks['source_sha256'] and len(actual) == 20
for name, value in prep['evidence_sha256'].items():
    pin(PACKAGE/name, value)
for entry in prep['inherited_evidence'].values():
    pin(entry['path'], entry['sha256'])
producer = BASE/'independent_plant_pending_result_v1/source_draft_v1'
for name, value in prep['producer_source_sha256'].items():
    pin(producer/name, value)
for name in prep['unchanged_from_v1']:
    assert (SOURCE/name).read_bytes() == (PACKAGE/'source_draft_v1'/name).read_bytes()
old = (PACKAGE/'source_draft_v1/worker_result_math.py').read_text()
new = (SOURCE/'worker_result_math.py').read_text()
delta = "    chronological_iterations=[row['iteration'] for k in ordered for _,row,_ in pubs.get(k,[])]\n    if chronological_iterations!=sorted(chronological_iterations):raise AssertionError('Publication iterations moved backwards across jobs')\n"
assert new.count(delta) == 1 and new.replace(delta, '') == old
for path in SOURCE.glob('*.py'):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or '']
            assert not any(name.split('.')[0] in {'torch','onnxruntime','mujoco','subprocess','multiprocessing'} for name in names), path
test_log = (HERE/'combined_v2_tests.log').read_text()
assert '134 passed, 26 subtests passed' in test_log
xml = ET.parse(HERE/'combined_v2_tests.xml').getroot()
suite = xml.find('testsuite')
assert suite.attrib['tests'] == '160' and len(xml.findall('.//testcase')) == 134
assert all(suite.attrib[k] == '0' for k in ['errors','failures','skipped'])
before = (HERE/'old_v1_regression.log').read_text()
after = (HERE/'corrected_v2_regression.log').read_text()
assert 'AssertionError: AssertionError not raised' in before and 'FAILED (failures=1)' in before
assert 'Ran 2 tests' in after and after.rstrip().endswith('OK')
for name in ['test_cross_job_iteration.py','pretest_checks.json','combined_v2_tests.log','combined_v2_tests.xml',
             'old_v1_regression.log','corrected_v2_regression.log']:
    pin(HERE/name)
result = dict(passed=True, source_review_pass=True, reviewer='expert_resume',
    source_directory=SOURCE.as_posix(), source_sha256=actual,
    preparation_subject=dict(path=PREP.as_posix(), sha256=pin(PREP)),
    producer_preparation=prep['inherited_evidence']['producer_preparation'],
    producer_source_review=prep['inherited_evidence']['producer_source_review'],
    exact_twenty_source_hashes=True, exact_eighteen_inherited_v1_files=True,
    v2_production_delta='Two lines enforce chronological publication iterations across taken jobs; existing global uniqueness makes ordering strict.',
    findings=[dict(status='fixed_and_verified', source='worker_result_math.py',
        issue='Unique publication iteration numbers could regress across successive jobs while event timestamps remained chronological.',
        old_result='Independent valid two-job control passes; reversed unique1,0 regression was accepted (expected test failure).',
        corrected_result='Both independent tests pass; chronological0,1 remains accepted and reversed1,0 rejects.')],
    remaining_blockers=[], tests=dict(combined_passed=134, subtests_passed=26, junit_count=160,
        independent_regression_passed=2, independent_old_regression_expected_failures=1,
        python='C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
    reviewed_contracts=[
        'Original fixed1819controls/18190steps and full291/native/PD/history/source/hold reconstruction remain inherited.',
        'Job deadline reconstructs plant epoch plus activation times20ms; no rebase or late extension.',
        'One immutable reply, at most20 attempts, one per worker iteration; only recorded unambiguous BUSY retries.',
        'Ready/completed/attempted/returned/terminal events and final owned descriptors reconstruct separate counters.',
        'No polling/recomputation while result pending; taken second-slot work preserved in bounded buffer.',
        'Unknown writes, clock errors and logger overflow remain failures with possible plant receipt accounting.',
        'Actual source/request/review/process/PID/input/output hashes remain mandatory and separate from qualification.'],
    limitations=[
        'EMPTY polling spans are not recorded by producer; exact individual empty-poll timing cannot be reconstructed. Counts use frozen source-loop accounting.',
        'Source and synthetic evidence only; no actual pending-result clock run or completed saved audit qualified here.',
        'Missing/incomplete worker evidence cannot qualify full command coverage; recorded-command qualification does not qualify an online policy.'],
    input_sha256=INPUTS, writer_sha256=sha(__file__),
    actual_audit_requests=0, actual_clock_calls=0, worker_processes=0, model_calls=0,
    ORT_calls=0, optimizer_updates=0, native_steps=0)
for path, value in INPUTS.items():
    assert sha(path) == value, path
with (HERE/'review.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(dict(review=(HERE/'review.json').as_posix(), sha256=sha(HERE/'review.json'), input_count=len(INPUTS))))
