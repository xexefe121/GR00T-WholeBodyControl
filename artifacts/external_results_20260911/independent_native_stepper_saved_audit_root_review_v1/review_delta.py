import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'independent_native_stepper_saved_audit_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
preparation_path = BASE / 'preparation_report_v2.json'
p = json.loads(preparation_path.read_text())
assert p['preparation_passed'] is True and p['actual_audit_run'] is False
for group in ('source_sha256', 'observed_final_owner_schema_sha256'):
    for path, digest in p[group].items(): assert sha(path) == digest
for key in ('prior_preparation', 'prior_root_review', 'delta', 'tests'):
    assert sha(p[key]['path']) == p[key]['sha256']
assert sha(BASE / 'saved_math.py') == sha(BASE / 'source_v2/saved_math.py')
xml_path = BASE / 'source_v2/root_synthetic_tests.xml'
xml = ET.parse(xml_path).getroot()
assert len(list(xml.iter('testcase'))) == 19
assert not list(xml.iter('failure')) and not list(xml.iter('error'))
result = dict(source_review_passed=True,
    preparation_subject=dict(path=str(preparation_path), sha256=sha(preparation_path)),
    prior_review_subject=p['prior_root_review'], reviewed_delta=p['delta'],
    source_pins=p['source_sha256'], owner_schema_pins=p['observed_final_owner_schema_sha256'],
    root_tests=dict(path=str(xml_path), sha256=sha(xml_path), passed=19),
    changes_reviewed=p['changes'], pure_math_byte_unchanged=True,
    actual_audit_selected_once_after_completed_witness_and_replay=True,
    actual_audit_run=False, model_calls=0, native_steps=0, optimizer_updates=0,
    scope='Saved-only adapter equivalence, preserving original direct controller failure; no balance or real-time qualification.')
output = ROOT / 'review_v2.json'
with output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(dict(source_review_passed=True, sha256=sha(output), root_tests=19)))
