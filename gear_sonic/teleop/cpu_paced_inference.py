"""Explicit non-spinning CPU sessions for the localhost SIM diagnostic.

Disables worker busy-waiting between inference calls, including the inactive
balance session. Model bytes, graph options, numerical precision, thread count,
physical controller and deadlines are unchanged. Finite measured trials, not a
hard-real-time or hardware guarantee, must establish whether pacing improves.
"""

import copy
from pathlib import Path

import onnxruntime as ort

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import FALLBACK_RELATIVE_PATH
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy


def nonspinning_options():
    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def prepare_nonspinning_sessions(controller, identity, repository_root):
    if controller.completed != 0 or float(controller.data.time) != 0 or controller.fallback_active:
        raise ValueError("prepare CPU sessions before streaming or physics")
    pair = identity.get("diagnostic_pair")
    if not isinstance(pair, dict):
        raise ValueError("non-spinning SIM profile requires a validated encoder/decoder pair")
    policy = ExactHashSonicPolicy(
        encoder_path=Path(pair["encoder"]["path"]),
        decoder_path=Path(pair["decoder"]["path"]),
        expected_encoder_sha256=identity["encoder_sha256"],
        expected_decoder_sha256=identity["decoder_sha256"],
        session_options=nonspinning_options(),
    )
    fallback = UnitreeZeroVelocityFallbackPolicy(
        Path(repository_root) / FALLBACK_RELATIVE_PATH, session_options=nonspinning_options()
    )
    for session in (policy.encoder, policy.decoder, fallback.session):
        options = session.get_session_options()
        if any(
            options.get_session_config_entry(name) != "0"
            for name in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning")
        ):
            raise ValueError("CPU worker spinning setting did not apply")
    # Install only after both hash/ABI-validated policies construct successfully.
    controller.policy, controller.fallback_policy = policy, fallback
    result = copy.deepcopy(identity)
    result["cpu_pacing"] = dict(
        worker_spinning=False,
        all_three_sessions_configured=True,
        thread_count_and_graph_options="unchanged_ORT_defaults",
        inference_warmup_added=False,
        source_or_physics_changed=False,
        hard_realtime_proven=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return result

