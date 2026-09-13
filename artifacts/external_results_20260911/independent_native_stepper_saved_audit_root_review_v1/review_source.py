"""Record root review of the independently implemented saved-evidence auditor."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'independent_native_stepper_saved_audit_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    preparation_path = BASE / 'preparation_report.json'
    preparation = json.loads(preparation_path.read_text())
    assert preparation['preparation_passed'] is True and preparation['source_only'] is True
    assert preparation['actual_audit_run'] is False and preparation['actual_audit_request_created'] is False
    pins = {}
    for name, digest in preparation['source_sha256'].items():
        p = BASE / name
        assert sha(p) == digest
        pins[str(p)] = digest
    for path, digest in preparation['observed_producer_schema_sha256'].items():
        assert sha(Path(path)) == digest
        pins[path] = digest
    forbidden = {'mujoco','torch','onnxruntime','native_stepper','capture_schema','clock_core','replay_core','model_identity'}
    for name in ('saved_math.py', 'audit_saved.py'):
        for node in ast.walk(ast.parse((BASE / name).read_text())):
            if isinstance(node, ast.Import):
                assert all(x.name.split('.')[0] not in forbidden for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in forbidden
    xml_path = BASE / 'root_synthetic_tests.xml'
    xml = ET.parse(xml_path).getroot()
    assert len(list(xml.iter('testcase'))) == 18
    assert not list(xml.iter('error')) and not list(xml.iter('failure'))
    result = dict(source_review_passed=True,
        preparation_subject=dict(path=str(preparation_path), sha256=sha(preparation_path)),
        source_and_producer_pins=pins,
        root_tests=dict(path=str(xml_path), sha256=sha(xml_path), passed=18),
        reviewed_behavior=[
            'Independent 373-float capture and typed fault decoding; no adapter decode import.',
            'All21348 saved samples, original291-state boundaries and continuous hold checked exactly.',
            'Independent PD arithmetic, repeated2ms clock and strict predicates reconstructed.',
            'Expected direct bound failure and full returned/fault capture retained.',
            'All10 actual model-serialization buffers compared byte-exact with witness.',
            'Explicit owner request/report/clearance/exit/launch/absence fields and pre/post ledgers bound.',
            'Canonical fixture, contract, witness and trace roles must be producer-pinned.',
            'Original trace hashes/control ranges/step counts and independent qualification checked.'],
        actual_run_selected_after_completed_stage_gates=True,
        actual_request_created=False, actual_audit_run=False,
        pending=['Final producer-owner schema alignment and completed witness/replay success receipts.'],
        native_steps=0, model_calls=0, optimizer_updates=0)
    output = ROOT / 'review.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(source_review_passed=True, sha256=sha(output), root_tests=18)))

if __name__ == '__main__':
    main()
