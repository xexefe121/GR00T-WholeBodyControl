import hashlib
import json

import pytest

from gear_sonic.scripts.audit_g1_true23_motion_fidelity import audit
from gear_sonic.utils.g1_true23_motion_fidelity import MAXIMUM_ERRORS


def test_rescore_preserves_and_binds_historical_report(tmp_path):
    source = {
        "kind": "g1_true23_genuine_sonic_library_motion_mujoco_replay",
        "frame_count": 546, "completed_transitions": 535,
        "requested_transitions": 535, "passed": True, "failure": None,
        "metrics": dict(MAXIMUM_ERRORS, maximum_pelvis_position_error_m=5.01),
    }
    path = tmp_path / "source.json"
    raw = json.dumps(source).encode()
    path.write_bytes(raw)
    result = audit(path, 535)
    assert path.read_bytes() == raw
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["cases"][0]["legacy_completion_passed"] is True
    assert result["cases"][0]["motion_fidelity"]["passed"] is False
    with pytest.raises(ValueError, match="full clip count"):
        audit(path, 534)


def test_envelope_prefix_is_not_full_clip(tmp_path):
    source = {
        "kind": "g1_true23_deployment_envelope_diagnostic_v1",
        "cases": [{
            "label": "prefix", "completed_transitions": 10, "requested_transitions": 10,
            "library_completion_passed": True, "failure": None,
            "maximums": {
                "pelvis_position_error_m": 0., "pelvis_orientation_error_rad": 0.,
                "relative_body_error_m": 0., "joint_rmse_rad": 0.,
            },
        }],
    }
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(source))
    result = audit(path, 535)["cases"][0]["motion_fidelity"]
    assert result["failed_checks"] == ["full_clip_completion"]


def test_unrecognized_report_rejected(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="unsupported"):
        audit(path, 535)
