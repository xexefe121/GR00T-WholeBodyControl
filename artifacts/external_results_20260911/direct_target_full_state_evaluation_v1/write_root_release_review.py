"""Bind completed independently audited fit to one simulation pipeline. No execution."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
NEW = BASE.parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main(args):
    assert sha(args.audit) == args.audit_sha256
    audit = read(args.audit)
    assert audit['evidence_audit_passed'] is True and audit['export_qualified'] is True
    assert audit['ordinary_final_step'] == 65000 and audit['optimizer_step'] == 10000
    assert 0 <= audit['maximum_preclamp_rad'] <= 1e-5
    for key in ('model_calls', 'ORT_calls', 'gradient_calls', 'optimizer_updates', 'native_steps'):
        assert audit[key] == 0
    owner_path = NEW / 'direct_target_full_state_student_v1/owner_completion_verification.json'
    assert sha(owner_path) == '8bacc308af12b0c888313848a8b20fdce04e11e252cb62cc850a3102efb5fce5'
    owner = read(owner_path)
    for key in ('owner_verification_passed', 'raw_exit_known', 'all_postrun_pins_exact', 'processes_absent'):
        assert owner[key] is True
    assert owner['raw_python_exit_code'] == owner['exit_code'] == 0
    subjects = dict(audit['direct_subject_sha256'])
    assert set(subjects) == set(('fit_report', 'checkpoint', 'head', 'normalization', 'training_manifest',
        'training_request', 'export_manifest', 'coefficient', 'source_checkpoint', 'full_state_generation_request',
        'full_state_generation_report', 'full_state_data_audit', 'full_state_data_owner'))
    request_path = args.audit.parent.parent / 'audit_request.json'
    assert sha(request_path) == audit['audit_request_sha256']
    request = read(request_path)
    for role, digest in subjects.items():
        entry = request['subjects'][role]
        assert entry['sha256'] == digest == owner['direct_subject_sha256'][role]
        assert sha(entry['path']) == digest
        canonical = Path(entry['path']).resolve().as_posix()
        assert audit['input_sha256'][canonical] == digest
    assert subjects['head'] == '045f04138610a06a0171a899d002cfa4f03e30e316dc9442199c833c16e43902'
    assert subjects['checkpoint'] == '8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a'
    subjects.update(root_training_audit=args.audit_sha256, fit_owner_completion=sha(owner_path))
    result = dict(release_review_pass=True, reviewer='root', reviewed_utc=datetime.now(timezone.utc).isoformat(),
        direct_subject_sha256=subjects, audit_path=args.audit.resolve().as_posix(),
        same_source_weights_FP64_export=True, physical_qualification=False,
        selected_witness_calls=1, selected_main_controls=1569, conditional_hold_controls=250,
        per_stage_concrete_review_required=True, native_steps=0, model_calls=0,
        scope='One original canonical simulation pipeline. Full simulation acceptance remains pending.',
        writer_sha256=sha(__file__))
    destination = BASE / 'root_final_release_review.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=destination.as_posix(), sha256=sha(destination))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    main(parser.parse_args())
