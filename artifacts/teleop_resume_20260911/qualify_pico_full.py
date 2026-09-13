"""Bind the complete offline PICO lifecycle and continuous hold root audits."""

import hashlib
import json
from pathlib import Path

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports = {}
    traces = {}
    for part, producer, prefix, count, start in (
        ('full', 'pico_full_control_lm_v1', 'pico_full_control_lm_independent', 6530, 0),
        ('hold', 'pico_terminal_hold_v1', 'pico_hold_independent', 250, 6530),
    ):
        trace = BASE / producer / 'trace.npz'
        traces[part] = {'path': trace.as_posix(), 'sha256': sha(trace)}
        physical_path = BASE / (prefix + '_physics_v1') / 'report.json'
        intent_path = BASE / (prefix + '_intent_v1') / 'report.json'
        physical, intent = [json.loads(p.read_text()) for p in (physical_path, intent_path)]
        assert physical['independent_segment_pass']
        assert physical['recorded_trace_reproduced_through_last_sample']
        assert all(physical['original_trace_comparison'].values())
        assert physical['intended_segment_controls'] == count
        assert physical['compared_physics_steps'] == 10 * count
        assert sha(trace) in physical['input_hashes'].values()
        assert sha(trace) in intent['hashes'].values()
        assert sha(physical_path) in intent['hashes'].values()
        assert intent['independent_physical_pass'] and intent['intended_segment_completed']
        assert intent['requested_controls'] == count and intent['global_start'] == start
        assert intent['requested_segment_quiet_pass']
        assert all(intent['quiet_last_three_seconds']['gates'].values())
        if part == 'full':
            assert intent['full_lifecycle_source_intent_pass']
            assert intent['source_metrics']['source_controls'] == 5780
            assert all(intent['source_metric_gates'].values())
        else:
            assert sha(BASE / 'pico_full_endpoint_independent_v1/endpoint.npz') in intent['hashes'].values()
            assert traces['full']['sha256'] in intent['hashes'].values()
            assert intent['initialization_checked'] == 'separate hold continuous with preceding full integration state/pose/velocity/clock'
        for kind, path in (('physics', physical_path), ('intent', intent_path)):
            reports[part + '_' + kind] = {'path': path.as_posix(), 'sha256': sha(path)}
    out = BASE / 'pico_full_root_qualification_v1'
    out.mkdir(exist_ok=False)
    result = dict(kind='root_independent_complete_offline_pico_and_continuous_hold',
                  complete_offline_pico_pass=True, source_controls=5780,
                  lifecycle_controls=6530, separate_hold_controls=250, physics_steps=67800,
                  both_quiet_windows_pass=True, native_constraints_unchanged=True,
                  source_samples_removed_or_retimed=False,
                  traces=traces, independent_reports=reports,
                  timing_qualified=False, live_teleoperation_qualified=False,
                  hardware_authorized=False, source_sha256=sha(Path(__file__)))
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'qualification': str(out / 'qualification.json'), 'sha256': sha(out / 'qualification.json')}))


if __name__ == '__main__':
    main()
