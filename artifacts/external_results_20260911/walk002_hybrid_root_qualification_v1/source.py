"""Bind independent native replay and original intent checks for the walk002 hybrid."""

import hashlib
import json
from pathlib import Path

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports, traces = {}, {}
    for part, producer, prefix, count, start in (
        ('full', 'walk002_terminal_bfm_hybrid_v1', 'walk002_hybrid_independent', 1417, 0),
        ('hold', 'walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s',
         'walk002_hybrid_hold_independent', 250, 1417),
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
            assert intent['source_metrics']['source_controls'] == 667
            assert all(intent['source_metric_gates'].values())
        else:
            assert sha(BASE / 'walk002_hybrid_endpoint_independent_v1/endpoint.npz') in intent['hashes'].values()
            assert traces['full']['sha256'] in intent['hashes'].values()
            assert intent['initialization_checked'] == 'separate hold continuous with preceding full integration state/pose/velocity/clock'
        for kind, path in (('physics', physical_path), ('intent', intent_path)):
            reports[part + '_' + kind] = {'path': path.as_posix(), 'sha256': sha(path)}
    producer_path = BASE / 'walk002_terminal_bfm_hybrid_v1/report.json'
    producer = json.loads(producer_path.read_text())
    assert producer['complete_prefix_physics_bitexact'] and producer['prefix_physics_steps'] == 11170
    assert producer['measured_history_comparisons'] == 1118 and producer['switch_verified']
    assert producer['original_main_quiet_failure_preserved']
    out = BASE / 'walk002_hybrid_root_qualification_v1'
    out.mkdir(exist_ok=False)
    result = dict(kind='root_independent_complete_offline_walk002_hybrid_and_continuous_hold',
                  complete_offline_walk002_hybrid_pass=True, source_controls=667,
                  lifecycle_controls=1417, separate_hold_controls=250, physics_steps=16670,
                  recorded_MPC_prefix_controls=1117, actual_terminal_BFM_controls=550,
                  controller='recorded MPC command prefix followed by actual original BFM yaw4 pos1 h8',
                  fresh_MPC_replanning_reproduced=False,
                  original_same_MPC_quiet_failure_preserved=True,
                  both_quiet_windows_pass=True, native_constraints_unchanged=True,
                  source_samples_removed_or_retimed=False,
                  traces=traces, independent_reports=reports,
                  producer_switch_receipt={'path': producer_path.as_posix(), 'sha256': sha(producer_path)},
                  timing_qualified=False, live_teleoperation_qualified=False,
                  hardware_authorized=False, source_sha256=sha(Path(__file__)))
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'qualification': str(out / 'qualification.json'), 'sha256': sha(out / 'qualification.json')}))


if __name__ == '__main__':
    main()
