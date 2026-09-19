"""Lossless development mixes of already saved original29/native23 lifecycles.

Each member remains an independent episode source. Concatenation is storage,
never a transition between recordings. These are research inputs, not a
qualified motion corpus or a dynamically feasible reference certificate.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, derive_curriculum
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)

KIND = "g1_true23_original_intent_development_mix_v1"


def assemble_mix(members, evaluation_recording_ids, *, source_model, native_model, simulation_config):
    """Reconstruct each saved lifecycle, then join its unchanged source arrays.

    Members contain name, recording_id, motion, timeline and original_reference.
    Evaluation IDs name known development recordings excluded from this optimizer;
    this is deliberately not labelled an untouched held-out test.
    """
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("original-intent mix requires at least two complete recordings")
    if (
        not isinstance(evaluation_recording_ids, list)
        or any(not isinstance(value, str) or not value for value in evaluation_recording_ids)
        or len(set(evaluation_recording_ids)) != len(evaluation_recording_ids)
    ):
        raise ValueError("evaluation recording IDs must be unique nonempty strings")
    names, recordings, content = set(), set(), set()
    sections, originals, old_lifecycles, old_timelines, rows = [], [], [], [], []
    cursor = 0
    for member in members:
        if set(member) != {"name", "recording_id", "motion", "timeline", "original_reference"}:
            raise ValueError("unexpected original-intent member fields")
        name, recording = member["name"], member["recording_id"]
        if not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None or name in names:
            raise ValueError("member names must be unique plain identifiers")
        if not isinstance(recording, str) or not recording or recording in recordings:
            raise ValueError("training recording IDs must be unique nonempty strings")
        if recording in evaluation_recording_ids:
            raise ValueError("evaluation recording cannot enter training mix")
        names.add(name)
        recordings.add(recording)
        motion, timeline, saved = member["motion"], member["timeline"], member["original_reference"]
        reference = build_original29_reference(source_model, saved["source_qpos29"])
        for key, value in reference.arrays().items():
            if key not in saved or not np.array_equal(saved[key], value):
                raise ValueError(f"saved original29 task reconstruction differs: {name}/{key}")
        verify_unmodified_native_pair(reference, motion)
        count = len(reference.source_qpos29)
        for key in MOTION_KEYS:
            if len(motion[key]) != count or not np.isfinite(motion[key]).all():
                raise ValueError(f"invalid native lifecycle channel: {name}/{key}")
        start, stop = timeline["source_start_frame"], timeline["source_stop_frame_exclusive"]
        if any(type(value) is not int for value in (start, stop)) or not 11 <= start < stop <= count:
            raise ValueError("source range must be an interior complete lifecycle section")
        length = stop - start
        if (
            timeline["source_frames"] != length
            or timeline["total_frames"] != count
            or timeline["source_frame_indices"] != list(range(length))
            or timeline["source_timing_scale"] != 1.0
            or timeline["all_source_channels_samples_preserved"] is not True
            or timeline["return_target"] != "fixed_planned_terminal_xy_and_heading"
        ):
            raise ValueError("requires complete original-speed planned-endpoint lifecycle")
        source = {key: motion[key][start:stop].copy() for key in MOTION_KEYS}
        digest = hashlib.sha256(reference.source_qpos29[start:stop].tobytes()).hexdigest()
        if digest in content:
            raise ValueError("duplicate source content cannot gain sampling weight under another name")
        content.add(digest)
        sections.append(source)
        originals.append(reference.arrays())
        old_lifecycles.append(motion)
        old_timelines.append(timeline)
        rows.append(
            dict(
                name=name,
                recording_id=recording,
                start=cursor,
                length=length,
                split="local_regression",
                source_qpos29_sha256=digest,
            )
        )
        cursor += length
    combined = {key: np.concatenate([part[key] for part in sections]) for key in MOTION_KEYS}
    combined["fps"] = np.array([50.0])
    spans = dict(
        kind="g1_true23_motion_corpus_spans_v1", fps=50, clip_count=len(rows), total_frames=cursor, spans=rows
    )
    derived, lifecycle_spans, _ = derive_curriculum(
        combined,
        spans,
        dict(smoke_only=True),
        stage="lifecycle",
        model=native_model,
        simulation_config=simulation_config,
        return_target="planned_endpoint",
    )
    for row, prior, timeline in zip(lifecycle_spans["spans"], old_lifecycles, old_timelines, strict=True):
        section = slice(row["start"], row["start"] + row["length"])
        for key in MOTION_KEYS:
            if not np.array_equal(derived[key][section], prior[key]):
                raise ValueError(f"reconstructed lifecycle changes saved channel: {row['name']}/{key}")
        for field in ("phases", "source_frame_indices", "configured_standing_qpos", "returned_standing_qpos"):
            if row["timeline"][field] != timeline[field]:
                raise ValueError(f"reconstructed lifecycle changes timing or endpoints: {field}")
    original = {key: np.concatenate([part[key] for part in originals]) for key in originals[0]}
    reference = build_original29_reference(source_model, original["source_qpos29"])
    for key, value in reference.arrays().items():
        np.testing.assert_array_equal(value, original[key])
    verify_unmodified_native_pair(reference, derived)
    contract = dict(
        kind=KIND,
        training_recording_ids=[m["recording_id"] for m in members],
        evaluation_only_recording_ids=evaluation_recording_ids,
        evaluation_data_is_previously_seen_development=True,
        untouched_test_generalization_claimed=False,
        clip_sampling="uniform_recording_choice_only_at_environment_reset",
        exact_realized_training_exposure_balancing_claimed=False,
        physical_state_transition_between_recordings=False,
        saved_native_lifecycles_bit_exact=True,
        all_original29_tasks_bit_exact=True,
        lifecycle_spans=lifecycle_spans,
        source_frames=cursor,
        lifecycle_frames=len(derived["joint_pos"]),
        physical_qualification_or_reference_feasibility_claimed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return combined, spans, original, derived, contract
