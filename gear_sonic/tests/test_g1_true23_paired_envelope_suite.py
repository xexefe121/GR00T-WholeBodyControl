from argparse import Namespace
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.evaluate_g1_true23_paired_envelope_suite import (
    case_command,
    file_identity,
    main,
    manifest_cases,
    verify_case,
)


def _manifest(tmp_path, *, kind="g1_true23_pico_fullbody_multimotion_manifest_v1"):
    root, assets = tmp_path / "repo", tmp_path / "assets"
    root.mkdir()
    assets.mkdir()
    motions = []
    for name, frames in (("dance", 546), ("standing", 1024)):
        np.savez_compressed(assets / f"{name}.npz", joint_pos=np.zeros((frames, 23)))
        motions.append({"name": name, "path": f"{name}.npz", "weight": 1})
    value = {"kind": kind, "motions": motions}
    path = root / "manifest.json"
    path.write_text(json.dumps(value))
    return root, assets, path, value


def test_all_manifest_clips_and_frames_are_preserved(tmp_path):
    root, assets, path, value = _manifest(tmp_path)
    manifest, cases = manifest_cases(path, assets, root)
    assert manifest == value
    assert [case["name"] for case in cases] == ["dance", "standing"]
    assert [case["frames"] for case in cases] == [546, 1024]
    assert cases[0]["motion"] == file_identity(assets / "dance.npz")


@pytest.mark.parametrize("defect", ["duplicate_name", "duplicate_path", "unsafe_name", "outside_root", "empty"])
def test_manifest_does_not_silently_drop_or_redirect_cases(tmp_path, defect):
    root, assets, path, value = _manifest(tmp_path)
    if defect == "duplicate_name":
        value["motions"][1]["name"] = "dance"
    elif defect == "duplicate_path":
        value["motions"][1]["path"] = "dance.npz"
    elif defect == "unsafe_name":
        value["motions"][0]["name"] = "../dance"
    elif defect == "outside_root":
        outside = tmp_path / "outside.npz"
        np.savez(outside, joint_pos=np.zeros((546, 23)))
        value["motions"][0]["path"] = str(outside)
    else:
        value["motions"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        manifest_cases(path, assets, root)


def test_stance_references_remain_unaccepted_candidates(tmp_path):
    root, assets, path, value = _manifest(tmp_path, kind="g1_true23_stance_candidate_manifest_v1")
    with pytest.raises(ValueError, match="unaccepted"):
        manifest_cases(path, assets, root)
    value.update(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)
    path.write_text(json.dumps(value))
    assert len(manifest_cases(path, assets, root)[1]) == 2
    value["teacher_accepted"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unaccepted"):
        manifest_cases(path, assets, root)


def _pair():
    return {part: {"path": f"/{part}.onnx", "sha256": part[0] * 64} for part in ("encoder", "decoder")}


def test_fixed_command_preserves_limits_pair_and_full_duration(tmp_path):
    args = Namespace(
        repository_root=tmp_path,
        asset_root=tmp_path,
        encoder_report=Path("encoder.json"),
        decoder_report=Path("decoder.json"),
        motor_health_snapshot=Path("health.json"),
        transition_balance_model=Path("standing.onnx"),
    )
    case = {"motion": {"path": "/dance.npz"}}
    reference = case_command(args, _pair(), case, tmp_path / "reference", measured=False)
    measured = case_command(args, _pair(), case, tmp_path / "measured", measured=True)
    for command in (reference, measured):
        assert "--maximum-steps" not in command
        assert "--allow-unpaired-diagnostic" not in command
        assert "--predictive-active-effort" not in command
        assert "--stateful-native-controller" in command
        assert command[command.index("--slew-rates") + 1] == "5"
        assert command[command.index("--ankle-efforts") + 1] == "35"
        assert command[command.index("--gain-profiles") + 1] == "configured_sim"
        assert command[command.index("--expected-encoder-sha256") + 1] == "e" * 64
    assert "--startup-hold-s" not in reference
    assert measured[measured.index("--startup-hold-s") + 1] == "5"
    assert measured[measured.index("--return-hold-s") + 1] == "5"
    assert "--project-transition-effort" in measured


def _summary():
    case = {"frames": 546, "motion": {"path": "/dance.npz", "sha256": "a" * 64}}
    summary = {
        "diagnostic_pair": _pair(),
        "unpaired_diagnostic_only": False,
        "sources": {"/dance.npz": "a" * 64},
        "authorization": {"deployment_ready": False},
        "cases": [
            {
                "requested_transitions": 535,
                "gain_profile": "configured_sim",
                "action_fraction": 1.0,
                "ankle_effort_nm": 35.0,
                "target_slew_rad_s": 5.0,
                "stateful_native_controller": True,
                "encoder_decoder_pair_validated": True,
                "active_effort_target_projection": True,
                "actuation_trace_recorded": True,
                "observation_timing": "current_post_integration_pose_and_velocity_v2",
            }
        ],
    }
    return summary, case


def test_case_validator_accepts_full_requested_clip_even_if_tracking_fails():
    summary, case = _summary()
    summary["cases"][0].update(completed_transitions=0, failure="guard")
    assert verify_case(summary, _pair(), case)["completed_transitions"] == 0


@pytest.mark.parametrize("defect", ["pair", "clip_length", "higher_effort", "source", "authorization"])
def test_case_validator_rejects_unmatched_or_weakened_comparison(defect):
    summary, case = _summary()
    if defect == "pair":
        summary["diagnostic_pair"] = copy.deepcopy(_pair())
        summary["diagnostic_pair"]["encoder"]["sha256"] = "f" * 64
    elif defect == "clip_length":
        summary["cases"][0]["requested_transitions"] = 50
    elif defect == "higher_effort":
        summary["cases"][0]["ankle_effort_nm"] = 50
    elif defect == "source":
        summary["sources"]["/dance.npz"] = "b" * 64
    else:
        summary["authorization"]["deployment_ready"] = True
    with pytest.raises(ValueError):
        verify_case(summary, _pair(), case)


def test_main_rejects_partial_measured_inputs_before_opening_policy(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--asset-root",
                str(tmp_path),
                "--manifest",
                "absent.json",
                "--encoder-report",
                "encoder.json",
                "--decoder-report",
                "decoder.json",
                "--output-dir",
                str(tmp_path / "new"),
                "--motor-health-snapshot",
                "health.json",
            ]
        )
    assert not (tmp_path / "new").exists()
