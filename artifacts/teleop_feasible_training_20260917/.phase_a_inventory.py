from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOTS = [
    Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/g1_true23/internet_pico_twist2_v1"),
    Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/g1_true23/pico_internet_fullbody_curriculum_v1"),
    Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic_deploy/reference/example"),
    Path("/mnt/e/codex-artifacts/twist2_import_20260915"),
    Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910"),
    Path("/mnt/e/codex_sonic_runtime/reference_floor_audit_20260910"),
]


def frame_count(path: Path):
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ("joint_pos", "qpos", "body_pos_w", "dof_pos"):
                if key in data:
                    return int(len(data[key])), sorted(data.files)
            return None, sorted(data.files)
    except Exception as exc:
        return None, [f"ERROR: {type(exc).__name__}: {exc}"]


def main():
    result = {}
    for root in ROOTS:
        rows = []
        if root.exists():
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in {".npz", ".json", ".pt", ".pkl"}:
                    continue
                relative = str(path.relative_to(root))
                row = {"path": relative, "bytes": path.stat().st_size, "suffix": path.suffix}
                if path.suffix == ".npz":
                    row["frames"], row["keys"] = frame_count(path)
                elif path.suffix == ".json":
                    try:
                        payload = json.loads(path.read_text())
                        row["json_kind"] = payload.get("kind") if isinstance(payload, dict) else type(payload).__name__
                        for key in ("source_frames", "resampled_frames", "packet_count", "frames", "length"):
                            if isinstance(payload, dict) and key in payload:
                                row[key] = payload[key]
                    except Exception as exc:
                        row["json_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
        result[str(root)] = rows
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
