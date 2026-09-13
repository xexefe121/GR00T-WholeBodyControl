"""Check actual sampled training inputs against original saved reference bytes."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.teleop.buffered_source_horizon import _relative_orientation_6d
from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_EXCLUDED_INDICES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
RUN = HERE / "train500_v1"


def main():
    output = RUN / "sampled_reference_audit.json"
    if output.exists():
        raise FileExistsError("sampled reference audit refuses overwrite")
    pins = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, path
        pins[str(path)] = digest
        return path

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as bundle:
            return {key: bundle[key].copy() for key in bundle.files}

    bind(__file__)
    request = json.loads(bind(RUN / "request.json").read_text())
    args = request["arguments"]
    spec = json.loads(bind(args["spec"]).read_text())
    spans = json.loads(bind(args["spans"]).read_text())["spans"]
    native = arrays(spec["files"]["native_motion"]["path"], spec["files"]["native_motion"]["sha256"])
    original = arrays(spec["files"]["original_reference"]["path"], spec["files"]["original_reference"]["sha256"])
    captured = arrays(RUN / "sampled_actual_inputs.npz")
    with np.load(bind(RUN / "reward_capture.npz"), allow_pickle=False) as bundle:
        done = bundle["stored_done"].copy()
    controls, anchors = captured["control_index"], captured["reference_q0"]
    tokenizer = captured["tokenizer"]
    assert anchors.shape == (156, 128) and tokenizer.shape == (156, 128, 268)
    np.testing.assert_array_equal(tokenizer[..., 0], np.ones((156, 128), np.float32))
    covered = np.zeros(anchors.shape, bool)
    exposure = []
    for span in spans:
        start, end = span["start"], span["start"] + span["length"]
        mask = (anchors >= start) & (anchors < end)
        assert not np.any(mask & covered)
        covered |= mask
        q0 = anchors[mask]
        assert (q0 + 1 < end).all()
        # Each lifecycle declares a constant standing tail. Clamp only the
        # admitted terminal padding, never wrap to a different recording.
        sample = np.minimum(q0[:, None] + 5 * np.arange(10)[None], end - 1)
        following = np.minimum(q0[:, None] + 5 * np.arange(10)[None] + 1, end - 1)
        position = native["joint_pos"][sample, :12].astype(np.float32)
        velocity = (native["joint_pos"][following, :12].astype(np.float32) - position) / np.float32(0.02)
        expected = np.concatenate((position.reshape(-1, 120), velocity.reshape(-1, 120)), axis=1)
        np.testing.assert_array_equal(tokenizer[mask, 1:241], expected)
        np.testing.assert_array_equal(tokenizer[mask, 241:262], original["virtual_vr21"][q0].astype(np.float32))
        exposure.append(dict(name=span["name"], sampled_env_controls=int(mask.sum())))
    assert covered.all()
    measured_quat = captured["measured_root_quaternion_wxyz"]
    orientation = np.stack(
        [
            _relative_orientation_6d(measured, reference)
            for measured, reference in zip(
                measured_quat.reshape(-1, 4), native["body_quat_w"][anchors, 0].reshape(-1, 4), strict=True
            )
        ]
    ).reshape(156, 128, 6)
    orientation_error = float(np.max(np.abs(tokenizer[..., 262:] - orientation)))
    assert orientation_error <= 2e-6
    w, x, y, z = np.moveaxis(measured_quat, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    expected_velocity = (native["body_pos_w"][anchors + 1, 0] - native["body_pos_w"][anchors, 0]).astype(
        np.float32
    ) / np.float32(0.02)
    expected_velocity = np.stack(
        (
            c * expected_velocity[..., 0] + s * expected_velocity[..., 1],
            -s * expected_velocity[..., 0] + c * expected_velocity[..., 1],
            expected_velocity[..., 2],
        ),
        axis=-1,
    )
    velocity_error = float(np.max(np.abs(captured["root_feedback"][..., 3:6] - expected_velocity)))
    assert velocity_error <= 2e-6
    history = captured["policy"]
    gravity = history[..., 900:].reshape(156, 128, 10, 3)[..., -1, :]
    expected_gravity = np.stack((2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1), axis=-1)
    gravity_error = float(np.max(np.abs(gravity - expected_gravity)))
    assert gravity_error <= 2e-6
    blocks = ((0, 30, 3), (30, 320, 29), (320, 610, 29), (610, 900, 29), (900, 930, 3))
    continuations = 0
    for begin, end, width in blocks:
        block = history[..., begin:end].reshape(156, 128, 10, width)
        if width == 29:
            assert np.count_nonzero(block[..., list(SOURCE_IL29_EXCLUDED_INDICES)]) == 0
        for row in range(1, len(controls)):
            if controls[row] != controls[row - 1] + 1:
                continue
            valid = ~done[controls[row - 1]]
            np.testing.assert_array_equal(block[row, valid, :-1], block[row - 1, valid, 1:])
            if begin == 0:
                continuations += int(valid.sum())
    report = dict(
        kind="actual_sampled_pico_reference_and_history_consistency_v1",
        passed=True,
        sampled_env_controls=156 * 128,
        exposure=exposure,
        sampled_lower240_and_original_vr21_bit_exact=True,
        absent_proprioception_slots_zero=True,
        continuous_non_reset_history_shifts_bit_exact=continuations,
        orientation_max_abs_error=orientation_error,
        desired_root_velocity_max_abs_error=velocity_error,
        newest_gravity_max_abs_error=gravity_error,
        full_unsampled_stream_checked=False,
        measured_root_velocity_independently_verified=False,
        dynamics_or_tracking_success_claimed=False,
        hardware_authorized=False,
        deployment_ready=False,
        inputs=pins,
    )
    for path, expected in pins.items():
        assert sha256_file(Path(path)) == expected, path
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
