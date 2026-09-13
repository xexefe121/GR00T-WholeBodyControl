"""Make a hash-bound INCOMPLETE replay package from an atomic checkpoint.

No physical states, targets, source frames, solver settings or timings change.
This does not resume planning, finish a trajectory or create a final result.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def materialize(checkpoint, request_path, output):
    payload, request_bytes = checkpoint.read_bytes(), request_path.read_bytes()
    request = json.loads(request_bytes)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["checkpoint_metadata"]))
        count = metadata["completed_controls"]
        if metadata["is_final_result"] is not False or metadata["request_sha256"] != digest(request_bytes):
            raise ValueError("not an incomplete checkpoint bound to this request")
        if not 0 < count < request["requested_controls"]:
            raise ValueError("checkpoint must contain a strict nonempty lifecycle prefix")
        phase = request["source_phase"]
        source_count = max(0, min(count, phase["control_stop"]) - phase["control_start"])
        if metadata["completed_source_full_controls"] != source_count:
            raise ValueError("checkpoint source completion count differs")
        if not np.array_equal(archive["source_frame"], np.arange(count) + 11):
            raise ValueError("checkpoint source clock differs")
        steps = archive["physics_substeps"]
        if steps.shape != (count,) or np.any(steps != 10):
            raise ValueError("only checkpoints at complete control boundaries supported")
        if (archive["qpos"].shape != (count + 1, 30)
                or archive["qvel"].shape != (count + 1, 29)
                or archive["physics_qpos"].shape != (count * 10 + 1, 30)
                or archive["physics_qvel"].shape != (count * 10 + 1, 29)
                or archive["physics_torque"].shape != (count * 10, 23)):
            raise ValueError("checkpoint physical dimensions differ")
        np.testing.assert_array_equal(archive["qpos"], archive["physics_qpos"][::10])
        np.testing.assert_array_equal(archive["qvel"], archive["physics_qvel"][::10])
    plans = metadata["plans"]
    if not plans or sum(plan["controls_committed"] for plan in plans) != count:
        raise ValueError("checkpoint planning ledger differs")
    planning_ms = np.asarray([plan["solve_ms"] for plan in plans])
    if not np.isfinite(planning_ms).all() or np.any(planning_ms < 0):
        raise ValueError("invalid measured planner durations")
    plans_bytes = json.dumps(plans, indent=2).encode()
    report = dict(kind="derived_incomplete_checkpoint_replay_package", is_final_result=False,
                  failure=metadata["failure"], completed_controls=count,
                  requested_controls=request["requested_controls"],
                  checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=digest(payload),
                  trace_sha256=digest(payload), request_sha256=digest(request_bytes),
                  plans_sha256=digest(plans_bytes),
                  planning_ms_p50_p95_max=np.percentile(planning_ms, [50, 95, 100]).tolist(),
                  metadata=metadata, materializer_sha256=digest(Path(__file__).read_bytes()),
                  full_source_completed=source_count == request["source_requested_controls"] and metadata["failure"] is None,
                  full_lifecycle_completed=False,
                  full_body_tracking_qualified=False, timing_qualified=False, hardware_authorized=False)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (("trace.npz", payload), ("request.json", request_bytes), ("plans.json", plans_bytes)):
        (output / name).write_bytes(content)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "materializer_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key != "metadata"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.checkpoint, args.request, args.output)
