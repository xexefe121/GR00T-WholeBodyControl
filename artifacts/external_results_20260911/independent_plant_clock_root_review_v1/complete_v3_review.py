"""Root receipt for source review and the single scoped fake-only suite."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

root = Path(__file__).parent
foundation = root.parent / 'independent_plant_clock_foundation_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
preparation = foundation / 'preparation_report_v3.json'
assert sha(preparation) == '362a5c577197e071342f9dd9da8f8b6e4dcde04c6f8642ec79802aa4abc720d4'
report = json.loads(preparation.read_text())
checks = []
for section in ('source_sha256', 'evidence_sha256', 'example_sha256'):
    for path, expected in report[section].items():
        p = Path(path) if Path(path).is_absolute() else foundation / path
        actual = sha(p)
        checks.append(dict(path=str(p), expected=expected, actual=actual))
        assert actual == expected, p
for section in ('preserved_v2_checks', 'original_runtime_source_checks'):
    for path, record in report[section].items():
        actual = sha(path)
        checks.append(dict(path=path, expected=record['expected'], actual=actual))
        assert actual == record['expected'], path
suite_path = root / 'tests_v3_root_scoped.xml'
suites = list(ET.parse(suite_path).getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites) == 70
assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
output = dict(
    kind='root_source_and_fake_clock_v3_review', passed=True,
    source_review='v2-to-v3 core diff, typed immutable capture/verifier contracts, all clock paths, and regression test source inspected',
    findings_resolved=['mutable verifier issue alias', 'nested mutable warning payload', 'backward finish clock false pass'],
    preparation_sha256=sha(preparation), pinned_inputs=checks,
    scoped_test_count=70, scoped_tests_sha256=sha(suite_path),
    earlier_unscoped_collection_failure={'path':str(root/'tests_v3_root.xml'), 'sha256':sha(root/'tests_v3_root.xml'),
        'reason':'pytest traversed E:/WpSystem and failed collection with WinError1337; no tests executed; rerun scoped to source directory'},
    source_sha256=sha(__file__), task_model_calls=0, optimizer_updates=0, native_steps=0,
    real_process_timing_qualified=False, balance_qualified=False,
    limitations=['opaque structural state payload still needs native full-state adapter',
                 'in-process mailbox and Python allocation/copy/hash paths unqualified for real deadlines',
                 'input-fault holding is not verified balancing or recovery'],
)
destination = root / 'review_v3.json'
assert not destination.exists()
destination.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({'passed':True,'pins':len(checks),'tests':70,'report_sha256':sha(destination)}))
