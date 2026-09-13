"""Final source/input identity review; no task feature, model or physics calls."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

r = read(B/'request.json')
receipt = read(B/'launch_receipt.json')
assert sha(B/'request.json') == receipt['request_sha256'] == '474d5e4a04f84702b8ada696560df4baf6a3ebfa3b0ffebb0c343d4da11ebde1'
assert sha(B/'run_generation_durable.ps1') == receipt['launcher_sha256'] == '724150b85e53fce2dca07c458b60026ddff50788b1d9678fb808fa5cc49b8230'
assert r['generation_selected'] is True
assert [r[k] for k in ('rows','axes','signed_rows','old_overlap_rows','added_rows','pure_feature_map_calls')] == [3057,58,354612,140622,213990,357669]
assert all(r[k] == 0 for k in ('model_calls','BFM_calls','native_steps','optimizer_updates'))
verified = {}
for p, expected in r['windows_launch_sha256'].items():
    actual = sha(p)
    assert actual == expected, p
    verified[p] = actual
assert len(verified) == 31
assert len(r['runtime_sha256']) == 1645
assert r['runtime_identity']['python_version'] == '3.11.15'
assert r['scipy_version'] == '1.15.3'
assert {Path(p).resolve() for p in r['paths'].values()} <= {Path(p).resolve() for p in r['input_sha256']}
pre = read(B/'input_preflight.json')
assert pre['passed'] and pre['archive_schemas'] == r['archive_schemas']
assert pre['task_feature_calls'] == pre['task_map_calls'] == pre['task_probe_rows'] == 0
assert pre['versions'] == {'python':'3.11.15','numpy':'1.26.4','scipy':'1.15.3'}
assert pre['joint_margin_min'] > .01
tests = read(B/'tests_final.json')
assert tests['passed'] and tests['tests_passed'] == 14 and tests['failures'] == tests['skips'] == 0
assert tests['source_sha256'] == r['source_sha256']
sources = B/'source_snapshot_v1'
for name, digest in {
    'direct_features.py':'26b816a2059edbb83daba386196c21b70002d1c83e2eda6e5cbf6a550ee4a251',
    'secant_math.py':'d92782fee29bfe45b3e53f110a27ded22227933421e4b9702bf6899734784128',
    'test_secants.py':'3c84c16ffdd098be18c20071093237d35ba385eff7e57b79591e4873455a2c45',
}.items():
    assert sha(sources/name) == digest
for source in sources.glob('*.py'):
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        assert not any(n.split('.')[0] in {'torch','onnxruntime','mujoco','mjbatch'} for n in names)
assert not (B/'generation').exists()
assert not (B/'process'/'started.lock').exists()
out = B/'root_final_review.json'
assert not out.exists()
result = {
    'passed':True, 'reviewed_utc':datetime.now(timezone.utc).isoformat(),
    'request_sha256':sha(B/'request.json'), 'launcher_sha256':sha(B/'run_generation_durable.ps1'),
    'review_source_sha256':sha(__file__), 'windows_launch_pins_verified':verified,
    'runtime_pins_preflighted_and_enforced_by_producer':1645,
    'source_review':[
        'Prior reviewed fixed radii, quaternion chart, full K and nested clips unchanged.',
        'Final changes add selective numeric schema gate and saved 54-cell summaries/manifest.',
        'All 3057 centers and 140622 old overlaps must pass before 213990 added rows.',
        'Hidden .NET SHA supervisor preserves one attempt, exact exit/PIDs and pre/post pins.',
        'Producer source performs no model/native/replanning/optimizer calls.'
    ],
    'synthetic_tests_observed_by_owner':14,
    'selected_runs':1, 'task_feature_calls_this_review':0, 'model_calls':0, 'physics_steps':0,
    'limitation':'Generation qualifies saved static finite feedback data only. Independent full-data audit required before fit.'
}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'review_path':str(out),'review_sha256':sha(out),'pins_verified':len(verified)}))
