"""Prepared-packet feature parity on recorded expert inputs; no policy or dynamics."""

import json
from pathlib import Path

import numpy as np

from .inspect_recorded_intent import ROOT, sha
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures
from gear_sonic.utils.g1_true23_student_packets import DT, NATIVE_SHAPES, StudentPacket, StudentPacketGate


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def main():
    base = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    original_path = bundle / 'walk003/native_original.npz'
    task_path = bundle / 'walk003/original29.npz'
    reference_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
    contract_path = bundle / 'contract.json'
    manifest_path = bundle / 'manifest.json'
    frozen_features = base / 'fast_controller_continued_fit_v1/source_snapshot_v3/gear_sonic/utils/g1_true23_mpc_student.py'
    shared_features = ROOT / 'gear_sonic/utils/g1_true23_mpc_student.py'
    gate_path = ROOT / 'gear_sonic/utils/g1_true23_student_packets.py'
    assert sha(shared_features) == sha(frozen_features)
    labels_paths = [base / 'fast_controller_nominal_pilot_v1/labels/labels.npz',
                   base / 'fresh_expert_labels_resume_v1/labels/labels.npz']
    assert [sha(p) for p in labels_paths] == [
        '8003407282b01f2c66fe1720d7eb038f43377c1d6eab80dce47f1f75911e8083',
        'e93753597450881c08a46fd520113abd21470653083dba0226c6450469d16ad1']
    manifest = json.loads(manifest_path.read_text())
    assert sha(original_path) == manifest['cases']['walk003']['native_original.npz']
    assert sha(task_path) == manifest['cases']['walk003']['original29.npz']
    receipt_path = reference_path.parent / 'portable_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['reference_sha256'] == sha(reference_path)
    assert receipt['original29_reference_sha256'] == sha(task_path)
    original, motion, tasks = [archive(p) for p in (original_path, reference_path, task_path)]
    contract = json.loads(contract_path.read_text())
    labels = [archive(p) for p in labels_paths]
    count = len(motion['joint_pos'])
    assert count == len(original['joint_pos']) == len(tasks['source_task_position_w']) == 1580
    rows = [{int(frame): i for i, frame in enumerate(data['source_frame'])} for data in labels]
    checked = [0, 0]
    gate = StudentPacketGate()
    native_samples_checked = 0
    default, limits = np.asarray(contract['default_q']), np.asarray(contract['joint_limits'])

    def inspect(window):
        nonlocal native_samples_checked
        first = window.samples[0].packet.sequence
        bfm, retargeted, packet_tasks = window.arrays()
        offset = window.feature_frame
        indices = np.arange(first - offset, first + len(window.samples))
        for name in NATIVE_SHAPES:
            np.testing.assert_array_equal(bfm[name], original[name][indices])
            np.testing.assert_array_equal(retargeted[name], motion[name][indices])
        for name in ('source_task_position_w', 'source_task_quaternion_wxyz'):
            np.testing.assert_array_equal(packet_tasks[name], tasks[name][indices])
        native_samples_checked += len(window.samples)
        encoder = None
        for k, (data, lookup) in enumerate(zip(labels, rows)):
            if first not in lookup:
                continue
            i = lookup[first]
            assert first == data['control'][i] + 11
            if encoder is None:
                encoder = GoalFeatures(retargeted, packet_tasks, contract)
            previous = data['previous_action'][i]
            previous_target = np.clip(default + previous * .25 * np.asarray(contract['training_effort']) /
                np.asarray(contract['kp']), limits[:, 0], limits[:, 1])
            goals = encoder(data['teacher_qpos'][i], data['teacher_qvel'][i], previous_target, offset)
            features = np.r_[goals, data['base_target'][i] - default, previous].astype(np.float32)
            np.testing.assert_array_equal(features, data['features'][i])
            checked[k] += 1

    for i in range(count):
        packet = StudentPacket(0, i, i * DT,
            {name: original[name][i] for name in NATIVE_SHAPES},
            {name: motion[name][i] for name in NATIVE_SHAPES},
            tasks['source_task_position_w'][i], tasks['source_task_quaternion_wxyz'][i], i == count - 1)
        assert gate.receive(packet, i * DT)
        window = gate.consume(i * DT)
        if window is not None:
            inspect(window)
    # Explicit final receipt allows draining the remaining short windows.
    while (window := gate.consume((count - 1) * DT)) is not None:
        inspect(window)
    assert checked == [1269, 1268] and gate.consumed == count and gate.final_consumed and gate.fault is None
    report = dict(kind='prepared_received_packet_recorded_feature_parity_no_inference_or_physics',
        received_packets=count, consumed_windows=gate.consumed, checked_reference_samples=native_samples_checked,
        feature_rows_by_dataset=checked, feature_components_per_row=1069,
        all_features_bitexact=True, both_native_banks_and_original_tasks_bitexact=True,
        receipt_schedule='one complete prepared sample per20ms;38-frame delayed consumption; explicit EOF drain',
        raw_pose_derivative_causality_qualified=False, clock_pacing_qualified=False,
        policy_inference_executed=False, dynamics_executed=False,
        standing_or_rearm_behavior_qualified=False, full_teleoperation_qualified=False,
        hashes={str(p): sha(p) for p in (original_path, task_path, reference_path, receipt_path,
            manifest_path, contract_path, frozen_features, shared_features, gate_path, *labels_paths, Path(__file__))})
    out = base / 'received_student_recorded_feature_parity_v1'
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key != 'hashes'}))


if __name__ == '__main__':
    main()
