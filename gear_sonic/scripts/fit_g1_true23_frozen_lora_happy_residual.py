"""Fit tiny happy-dance residual heads on a frozen-LoRA diagnostic decoder.

The original SONIC compatibility rollout supplies offline labels.  Candidates
remain true23, simulator-only diagnostics and require closed-loop screening.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.fit_g1_true23_sonic_crawl_head import _fit_delta
from gear_sonic.scripts.fit_g1_true23_sonic_multimotion_head import (
    _dataset,
    _export_candidate,
    select_ridge,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair

ALPHAS = (-0.02, -0.01, -0.005, -0.002, -0.001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
MOTION = Path("artifacts/g1_true23/sonic_library_true23_happy_physical_reference_v1/happy_dance.true23.npz")
ROLLOUT = Path(
    "artifacts/g1_true23/sonic_library_true23_released_adapter_v10_happy_dataset/happy_dance.true23.physical.npz"
)


def _load_base_decoder(report_path: Path) -> tuple[Path, str, dict[str, Any]]:
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("kind") != "g1_true23_frozen_lora_diagnostic_decoder_onnx"
        or value.get("diagnostic_only") is not True
        or value.get("deployment_ready") is not False
        or value.get("hardware_authorized") is not False
        or value.get("active_motor_control_authorized") is not False
    ):
        raise ValueError("base decoder report safety contract mismatch")
    decoder = value.get("decoder")
    if not isinstance(decoder, dict):
        raise ValueError("base decoder identity is missing")
    path = report_path.with_name(str(decoder.get("filename"))).resolve()
    digest = decoder.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("base decoder hash is invalid")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError("base decoder identity mismatch")
    return path, digest, value


def _alpha_name(alpha: float) -> str:
    if not math.isfinite(alpha) or alpha == 0.0 or abs(alpha) > 0.25:
        raise ValueError("residual alpha must be finite, nonzero, and at most 0.25")
    sign = "minus" if alpha < 0 else "plus"
    magnitude = f"{abs(alpha):.3f}".replace(".", "p")
    return f"{sign}_{magnitude}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-decoder-report", type=Path, required=True)
    parser.add_argument(
        "--encoder-report",
        type=Path,
        required=True,
        help="Matching diagnostic encoder export from the same checkpoint; no legacy default",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    args = parser.parse_args(argv)

    root = args.repository_root.expanduser().resolve(strict=True)
    report_path = args.base_decoder_report.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"residual output exists: {output_dir}")
    alphas = tuple(float(alpha) for alpha in args.alphas)
    names = [_alpha_name(alpha) for alpha in alphas]
    if len(set(names)) != len(names):
        raise ValueError("residual alpha names collide")
    base_decoder, base_digest, base_report = _load_base_decoder(report_path)
    pair = load_diagnostic_pair(args.encoder_report, report_path)
    encoder = Path(pair["encoder"]["path"])
    if Path(pair["decoder"]["path"]) != base_decoder or pair["decoder"]["sha256"] != base_digest:
        raise ValueError("residual base decoder pair mismatch")
    motion = (root / MOTION).resolve(strict=True)
    rollout = (root / ROLLOUT).resolve(strict=True)

    decoder994, hidden, teacher = _dataset(
        root=root,
        motion_path=motion,
        rollout_path=rollout,
        encoder_path=encoder,
        decoder_path=base_decoder,
        decoder_sha256=base_digest,
        encoder_sha256=pair["encoder"]["sha256"],
    )
    session = ort.InferenceSession(str(base_decoder), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    base_output = np.concatenate(
        [np.asarray(session.run(None, {input_name: row[None]})[0]) for row in decoder994],
        axis=0,
    ).astype(np.float32)
    residual = teacher - base_output
    selected_ridge, ridge_grid = select_ridge(hidden, residual, [len(hidden)])
    delta_w, delta_b = _fit_delta(hidden, residual, selected_ridge)

    output_dir.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        name = _alpha_name(alpha)
        output = output_dir / f"candidate.{name}.decoder.onnx"
        _export_candidate(
            base_decoder=base_decoder,
            output_path=output,
            delta_w=delta_w,
            delta_b=delta_b,
            alpha=alpha,
        )
        prediction = base_output + alpha * (hidden @ delta_w.T + delta_b)
        candidates.append(
            {
                "name": name,
                "alpha": alpha,
                "decoder_filename": output.name,
                "decoder_sha256": sha256_file(output),
                "teacher_state_rmse": float(np.sqrt(np.mean(np.square(prediction - teacher)))),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_happy_residual_diagnostic_v1",
        "source": {
            "diagnostic_pair": pair,
            "encoder_sha256": pair["encoder"]["sha256"],
            "base_decoder_report": report_path.name,
            "base_decoder_report_sha256": sha256_file(report_path),
            "base_decoder_sha256": base_digest,
            "base_adapter_state_sha256": base_report["source"]["adapter_state_sha256"],
            "motion_sha256": sha256_file(motion),
            "teacher_rollout_sha256": sha256_file(rollout),
        },
        "fit": {
            "rows": len(hidden),
            "selected_ridge": selected_ridge,
            "ridge_grid": ridge_grid,
            "base_teacher_state_rmse": float(np.sqrt(np.mean(np.square(base_output - teacher)))),
            "delta_weight_frobenius": float(np.linalg.norm(delta_w)),
            "delta_bias_l2": float(np.linalg.norm(delta_b)),
        },
        "candidates": candidates,
        "closed_loop_screening_required": True,
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }
    destination = output_dir / "manifest.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(destination)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
