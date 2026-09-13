"""Supplement immutable MPC requests with separate packet and raw-pose preview bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    receipts = []
    derivative_audit = json.loads(args.derivative_audit.read_text())
    audited = {item["clip"]: item["reference_sha256"] for item in derivative_audit["results"]}
    common_evidence = [
        dict(path=str(path), sha256=digest(path)) for path in (args.derivative_audit, args.generator)
    ]
    for root in args.roots:
        for request_path in sorted(root.rglob("request.json")):
            request = json.loads(request_path.read_text())
            if request.get("kind") != "offline_native23_mjbatch_ilqr_probe":
                continue
            horizon = int(request["horizon"])
            seed = request.get("recorded_target_seed")
            seed_seconds = seed["original_generator_preview_seconds"] if seed else 0.0
            seed_frames = round(seed_seconds / 0.02)
            if abs(seed_seconds - seed_frames * 0.02) > 1e-10:
                raise ValueError("seed preview does not lie on the original 50Hz clock")
            packet_frames = horizon + seed_frames
            declared = request.get("conservative_effective_source_preview_seconds", packet_frames * 0.02)
            if abs(declared - packet_frames * 0.02) > 1e-10:
                raise ValueError("legacy preview field disagrees with actual horizon and seed window")
            override = request.get("motion_override") or {}
            base_hash = override.get("base_native_reference_sha256") or next(
                (value for path, value in request.get("input_hashes", {}).items()
                 if path.endswith("native_original.npz")), None
            )
            if base_hash is not None and base_hash != audited[request["clip"]]:
                raise ValueError("request original reference differs from derivative audit")
            evidence = list(common_evidence)
            floor = request.get("reference_floor_transform") or {}
            if floor:
                path = Path(floor["transform_receipt_path"])
                if digest(path) != floor["transform_receipt_sha256"]:
                    raise ValueError("v4 floor transform receipt hash mismatch")
                evidence.append(dict(path=str(path), sha256=digest(path)))
            # r is the measured state's source frame. Last regularized target
            # and terminal pose use r+H; last seed goal extends through r+H+7.
            # Central joint/linear derivatives add one raw-pose support sample.
            receipt = dict(
                schema_version=1,
                kind="supplementary_packet_vs_raw_pose_preview_disclosure",
                request_file="request.json",
                request_sha256=digest(request_path),
                audit_source_sha256=digest(Path(__file__)),
                source_evidence=evidence,
                original_reference_hash_checked_against_derivative_audit=base_hash is not None,
                clip=request["clip"],
                source_clock_hz=50,
                measured_state_source_frame="r = completed_control_count + 10",
                horizon_frames=horizon,
                last_planner_reference_packet_frame=f"r+{horizon}",
                last_recorded_seed_reference_packet_frame=f"r+{packet_frames}" if seed else None,
                central_derivative_additional_raw_pose_frames=1,
                central_derivative_channels=["joint_vel", "body_lin_vel_w"],
                original_angular_derivative_convention="backward world intervals",
                derivative_scope="central joint/linear fields except two preserved splice samples; upper bound",
                planner_reference_packet_preview_seconds=horizon * 0.02,
                planner_raw_pose_support_seconds=(horizon + 1) * 0.02,
                effective_reference_packet_preview_seconds=packet_frames * 0.02,
                conservative_effective_raw_pose_support_seconds=(packet_frames + 1) * 0.02,
                legacy_effective_preview_field_meaning=(
                    "reference packet fields; excludes their upstream derivative support"
                ),
                legacy_field_value_seconds=declared,
                raw_pose_support_scope=(
                    "Includes one-frame central joint/linear derivative support upstream of the supplied packets. "
                    "Recorded BFM action history is causal; it adds no further future samples. "
                    "End clamping can reduce these conservative bounds. The v4 common-Z pose filter is causal; "
                    "its central velocity support is this same extra sample, not a second extra sample."
                ),
                request_and_report_bytes_rewritten=False,
                controller_behavior_changed=False,
                received_only_140ms_stream_qualification=False,
                timing_scope="offline MPC experiment",
            )
            output = request_path.parent / "preview_support_supplement_v1.json"
            encoded = json.dumps(receipt, indent=2)
            if output.exists() and output.read_text() != encoded:
                raise ValueError(f"refusing to overwrite a different preview receipt: {output}")
            if not output.exists():
                output.write_text(encoded)
            receipts.append(dict(
                path=str(output), receipt_sha256=digest(output), request_sha256=receipt["request_sha256"]
            ))
    index = dict(kind="mpc_preview_disclosure_receipt_index", receipts=receipts)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(index, indent=2))
    print(json.dumps(dict(receipts=len(receipts), index=str(args.output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derivative-audit", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    run(parser.parse_args())
