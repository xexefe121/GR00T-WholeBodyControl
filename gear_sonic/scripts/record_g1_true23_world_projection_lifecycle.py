"""Strict new-checkpoint adapter for the unchanged four-world SIM recorder.

The existing recorder's report kind describes its measurement engine, not the
training recipe. Its actual policy identity and this wrapper receipt identify
the new recipe explicitly; neither old reports nor checkpoint headers change.
"""

import json
from pathlib import Path
import sys

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.scripts import record_g1_true23_world_quality_lifecycle as recorder
from gear_sonic.trl.mjlab.native23_world_projection_runner import training_contract
from gear_sonic.utils import (
    g1_true23_world_projection_checkpoint as projection,
    g1_true23_world_quality_checkpoint as quality,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def verified_update_checkpoint(run, update):
    if (
        update.get("kind") != "native23_world_projection_actual_update_audit_v1"
        or update.get("passed") is not True
        or update.get("update_verification_passed") is not True
        or update.get("completed_updates") != 100
        or update.get("actual_transitions") != 204800
        or update.get("initial_actor_and_critic_equal_matched_quality_predecessor") is not True
        or update.get("world_projection") != training_contract()
        or update.get("projection_loss", {}).get("actual_ppo_minibatches") != 800
        or update.get("hardware_authorized") is not False
        or update.get("deployment_ready") is not False
    ):
        raise ValueError("requires the independently audited100-update projection experiment")
    path = (Path(run) / "train/checkpoints/world_projection_model_100.pt").resolve()
    digest = update.get("inputs", {}).get(str(path))
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("audit does not bind this exact projection checkpoint")
    return path, digest


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    old_verify, old_load = recorder.verified_update_checkpoint, quality.load_cpu_actor

    def load_actor(*args, **kwargs):
        actor, identity, semantics = projection.load_cpu_actor(*args, **kwargs)
        path = Path(__file__).resolve(strict=True)
        semantics["reverified_repository_sources"][str(path)] = sha256_file(path)
        return actor, identity, semantics

    recorder.verified_update_checkpoint = verified_update_checkpoint
    quality.load_cpu_actor = load_actor
    try:
        from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

        with ieee_training_precision() as (precision, precision_guard):
            recorder.main(precision, precision_guard)
    finally:
        recorder.verified_update_checkpoint = old_verify
        quality.load_cpu_actor = old_load
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if report["identity"]["kind"] != "native23_world_projection_pytorch_cpu_research_reader_v1":
        raise ValueError("measurement did not use the new strict reader")
    if report["additional_training_failure"] != termination_contract():
        raise ValueError("measurement changed the training failure predicates")
    receipt = dict(
        kind="native23_world_projection_unchanged_lifecycle_engine_receipt_v1",
        report_path=str(report_path),
        report_sha256=sha256_file(report_path),
        identity=report["identity"],
        training_contract=training_contract(),
        recorded_engine_kind_preserved=report["kind"],
        engine_source_sha256=sha256_file(Path(recorder.__file__)),
        wrapper_source_sha256=sha256_file(Path(__file__)),
        checkpoint_header_or_measurement_report_relabelled=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "projection_reader_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
