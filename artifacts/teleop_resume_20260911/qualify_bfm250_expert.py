"""Bind completed independent expert audits to a collection-only decision."""

import hashlib
import json
from pathlib import Path


BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = BASE / 'bfm_entry250_actual_oracle_v1'
    traces = {'nominal': run / 'nominal/trace.npz',
              'extension': run / 'post_lifecycle_hold_5s/trace.npz'}
    names = {'nominal_physics': 'bfm250_expert_full_independent_physics_v1',
             'nominal_intent': 'bfm250_expert_full_independent_intent_v1',
             'extension_physics': 'bfm250_expert_hold_independent_physics_v1',
             'extension_intent': 'bfm250_expert_hold_independent_intent_v1'}
    reports = {}
    for key, name in names.items():
        path = BASE / name / 'report.json'
        report = json.loads(path.read_text())
        part = key.split('_')[0]
        assert sha(traces[part]) in report.get('input_hashes', report.get('hashes', {})).values()
        requested = 1569 if part == 'nominal' else 250
        if key.endswith('physics'):
            assert report['independent_segment_pass']
            assert report['recorded_trace_reproduced_through_last_sample']
            assert all(report['original_trace_comparison'].values())
            assert report['intended_segment_controls'] == requested
        else:
            assert report['independent_physical_pass'] and report['intended_segment_completed']
            assert report['requested_segment_quiet_pass']
            assert report['requested_controls'] == requested
            assert report['global_start'] == (0 if part == 'nominal' else 1569)
            if part == 'nominal':
                assert report['full_lifecycle_source_intent_pass']
                assert report['source_metrics']['source_controls'] == 819
        reports[key] = {'path': path.as_posix(), 'sha256': sha(path)}
    output = BASE / 'bfm250_expert_root_qualification_v1'
    output.mkdir(exist_ok=False)
    decision = dict(root_authorized_extraction=True, model_fitting_authorized=False,
                    nominal_trace_sha256=sha(traces['nominal']),
                    extension_trace_sha256=sha(traces['extension']),
                    independent_reports=reports,
                    scope='Collect and inspect expert labels controls250..1268 only; no fitting authorization',
                    no_hardware_authorization=True,
                    source_sha256=sha(Path(__file__)))
    path = output / 'qualification.json'
    path.write_text(json.dumps(decision, indent=2) + '\n')
    (output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'qualification': str(path), 'sha256': sha(path),
                      'full_lifecycle_and_separate_hold_independently_qualified': True}))


if __name__ == '__main__':
    main()
