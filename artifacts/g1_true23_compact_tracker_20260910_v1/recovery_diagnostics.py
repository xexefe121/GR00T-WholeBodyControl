"""Read-only audit of persisted compact learner and recovered CPU traces."""

import json
from pathlib import Path

import numpy as np
import torch


BASE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(line) for line in (BASE / "train2000_v2/metrics.jsonl").read_text().splitlines()]
    result = {"last_metric": rows[-1], "checkpoint": {}, "replays": {}}
    for count in (0, 200, 500, 1000):
        saved = torch.load(BASE / f"train2000_v2/compact_{count:04d}.pt", weights_only=True, map_location="cpu")
        result["checkpoint"][str(count)] = {
            "completed_updates": saved["completed_updates"],
            "optimizer_steps": sorted(set(int(v["step"]) for v in saved["optimizer_state"]["state"].values())),
            "all_actor_tensors_finite": all(torch.isfinite(v).all().item() for v in saved["actor_state"].values()),
            "learning_rate": saved["learning_rate"],
            "normalization": {k: {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max())} for k, v in saved["actor_state"].items() if "normal" in k},
        }
    for name in ("pico", "walk002", "walk003", "walk008"):
        folder = BASE / "eval1000_recovered_v1" / name
        with np.load(folder / "trace.npz", allow_pickle=False) as z:
            trace = dict(z)
        with np.load(folder / "attempts.npz", allow_pickle=False) as z:
            attempts = dict(z)
        count = len(attempts["accepted23"])
        preview_delta = np.max(np.abs(attempts["accepted23"] - attempts["inverse23"][:count]), axis=1)
        changed = np.flatnonzero(preview_delta > 1e-6)
        qpos = trace["qpos"]
        report = json.loads((folder / "report.json").read_text())
        result["replays"][name] = {
            "failure": report["physics"]["failure"],
            "source_metrics": report["source_metrics"],
            "preview_first_changed_control": int(changed[0]) if len(changed) else None,
            "preview_changed_controls": int(len(changed)),
            "preview_max_raw_delta": float(preview_delta.max()),
            "source_target_projection_max_rad": float(np.max(np.abs(attempts["projection23"]))),
            "initial_root": qpos[0, :7].tolist(),
            "final_root": qpos[-1, :7].tolist(),
            "root_error_m_at_controls": {str(i): float(trace["pelvis_error_m"][i]) for i in (0, 100, 200, 300, 349, 400, 450, 500) if i < len(trace["pelvis_error_m"])},
            "peak_velocity_ratio": float(trace["physics_velocity_ratio23"].max()),
            "peak_hard_limit_excess_rad": float(trace["physics_hard_limit_excess23"].max()),
            "torque_saturation_fraction": float(trace["torque_saturated23"].mean()),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
