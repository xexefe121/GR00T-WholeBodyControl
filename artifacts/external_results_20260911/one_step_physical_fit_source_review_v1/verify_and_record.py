import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

N = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
B = N / 'one_step_physical_student_v1'
O = Path(__file__).parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(name, value):
    with (O / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

prep_path = B / 'launcher_preparation_v1.json'
assert sha(prep_path) == '0657947add79d3705a226928740ba54013107e0e87c879c23fbfa2dc1d409c08'
prep = read(prep_path)
assert prep['passed'] is True and prep['tests'] == 23
assert prep['failures'] == prep['errors'] == prep['skips'] == 0
assert prep['powershell_parse_pass'] is True
assert prep['shared_delete_atomic_replace_pass'] is True
assert prep['child_handle_exit7_pass'] is True
assert len(prep['source_sha256']) == 39
checked = {}
for path, expected in {**prep['source_sha256'], **prep['evidence_sha256']}.items():
    assert sha(path) == expected, path
    checked[path] = expected
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(encoding='utf-8-sig'), filename=path)
prior = N / 'one_step_physical_trainer_preparation_review_v2/review.json'
assert sha(prior) == '2901f771d194dda2fa273248782f4db9fcd12fcc9473bdf1612ad9958cfa69d7'
data = N / 'one_step_branch_combined_data_review_v1/review.json'
assert sha(data) == '6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed'
assert read(data)['branch_data_review_pass'] is True
for p in (prep_path, prior, data):
    checked[p.as_posix()] = sha(p)
xml = ET.parse(B / 'launcher_tests_v2/pytest_final.xml')
suites = list(xml.getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites) == 23
assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
for name in prep['actual_artifacts_absent']:
    assert not (B / name).exists(), name
verification = dict(passed=True, source_files_checked=39, checked_sha256=checked,
                    preparation_tests=23, source_syntax_checked=True,
                    actual_fit_artifacts_absent=prep['actual_artifacts_absent'],
                    model_evaluations=0, optimizer_updates=0, ORT_calls=0, native_steps=0)
save('verification.json', verification)
review = dict(
    kind='independent_physical_fit_source_only_review', passed=True,
    source_review_pass=True, preparation_review_pass=True,
    final_frozen_request_review_pass=False, launch_authorized_by_this_review=False,
    source_sha256=prep['source_sha256'],
    direct_subject_sha256=prep['source_sha256'],
    launcher_preparation=dict(path=prep_path.as_posix(), sha256=sha(prep_path)),
    prior_trainer_review=dict(path=prior.as_posix(), sha256=sha(prior)),
    branch_data_review=dict(path=data.as_posix(), sha256=sha(data)),
    verification=dict(path=(O/'verification.json').as_posix(), sha256=sha(O/'verification.json')),
    reviewed_scope=[
        'All 36 prepared trainer source files unchanged from the prior source review; all 3 freeze/runner/reader files directly pinned.',
        'Original nominal and velocity objectives preserved; physical response uses live nominal successor gradients and fixed 99/819/100 requested-cell denominators.',
        'One fixed 5000-update continuation from ordinary 70000 to ordinary 75000, exact AdamW/RNG/normalization restoration and declared declining learning-rate reset.',
        'Actual branch manifest, producer request/report, root physics/labels and zero-conflict reports must be literal review subjects and verified input pins.',
        'Source-only freeze creates a fresh byte-identical snapshot, concrete training request and launch plan without model, native or optimizer calls.',
        'Finalization requires a separate review directly binding the real request, frozen receipt, launch plan, root selection and combined data review.',
        'Durable runner uses a persistent CreateNew launch lock, exact absolute Windows one-thread command, hidden child, captured handle, nonzero/unknown exit rejection and all pre/post hashes.',
        'Progress reader permits shared deletion; no controller launch, physics or resumed fit is selected by these helpers.',
    ],
    budget_at_all_3054_valid_rows=dict(updates=5000, final_step=75000,
        training_head_rows=36315000, head_ONNX_calls=1150, native_steps=0, BFM_calls=0),
    findings=[], limitations=['This is source preparation clearance. The actual frozen request and launch plan still require review before the selected fit.'],
    reviewer_execution=dict(model_evaluations=0, optimizer_updates=0, ORT_calls=0, native_steps=0))
save('review.json', review)
print(json.dumps(dict(review_path=(O/'review.json').as_posix(), review_sha256=sha(O/'review.json'), verification_sha256=sha(O/'verification.json'), checked_files=len(checked))))
