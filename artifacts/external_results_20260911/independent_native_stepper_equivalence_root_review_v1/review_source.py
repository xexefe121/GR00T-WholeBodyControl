"""Pin reviewed native equivalence source and independently verify its inventory."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent.parent / 'independent_native_stepper_equivalence_v1'
OUT = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    proposal_path = BASE / 'proposal.json'
    proposal = json.loads(proposal_path.read_text())
    prior_path = OUT / 'source_checks.json'
    prior = json.loads(prior_path.read_text())
    assert prior['saved_source_checks_passed'] is True
    for p, digest in prior['pins'].items():
        assert sha(p) == digest, p
    assert proposal['source_preparation_only'] is True and proposal['execution_selected'] is False
    assert proposal['model_inference_calls'] == proposal['optimizer_updates'] == proposal['other_oracle_native_steps'] == 0
    assert proposal['plant_foundation_connected'] is False and proposal['real_wallclock_experiment'] is False
    assert proposal['planned_stages'] == [
        dict(stage='witness', native_step_budget=0, serialization_budget=2, output='witness'),
        dict(stage='replay', native_step_budget=21348, serialization_budget=8, output='replay')]
    pins = {}
    size = 0
    for subject in proposal['input_files']:
        p = Path(subject['path']).resolve()
        assert p.is_absolute() and p.is_file()
        assert sha(p) == subject['sha256'] and p.stat().st_size == subject['bytes'], str(p)
        assert str(p) not in pins, 'duplicate inventory subject'
        pins[str(p)] = subject['sha256']
        size += subject['bytes']
    assert size == proposal['input_bytes']
    for p in (BASE / 'source_draft_v1').glob('*.py'):
        assert pins[str(p.resolve())] == sha(p)
    xml_path = BASE / 'source_draft_v1/root_stub_tests_v2.xml'
    xml = ET.parse(xml_path).getroot()
    assert len(list(xml.iter('testcase'))) == 26
    assert not list(xml.iter('failure')) and not list(xml.iter('error'))
    result = dict(source_review_passed=True, actual_execution_cleared=False,
                  proposal_subject=dict(path=str(proposal_path), sha256=sha(proposal_path)),
                  source_checks_subject=dict(path=str(prior_path), sha256=sha(prior_path)),
                  root_tests_subject=dict(path=str(xml_path), sha256=sha(xml_path), passed=26),
                  reviewed_input_count=len(pins), reviewed_input_bytes=size,
                  source_and_input_pins=pins,
                  findings_resolved=[
                      'Review requires dedicated exact request_subject path and SHA.',
                      'All consumed model, mesh, fixture, trace and witness roles require input pins.',
                      'Every native serialization buffer is retained without extra calls.'],
                  review_scope=[
                      'Eight reviewed v3 files exact; original model loader prefix AST unchanged.',
                      'Every returned step captured and strictly verified once before sample comparisons.',
                      'Continuous expert main and hold use one initialized adapter.',
                      'Direct failure must match original control315/substep8 joint bound, with full evidence.',
                      'Entry/exit complete model identity required even on constructor or replay failure.',
                      'No retries, extra steps, model inference or wallclock qualification.'],
                  limitations=[
                      'Concrete durable launcher and request require their own review before actual execution.',
                      'Source tests use fake APIs; no real MuJoCo execution has occurred in this review.'],
                  model_constructions=0, native_steps=0, serializations=0, inference_calls=0)
    output = OUT / 'review.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(source_review_passed=True, sha256=sha(output), inputs=len(pins), bytes=size)))

if __name__ == '__main__':
    main()
