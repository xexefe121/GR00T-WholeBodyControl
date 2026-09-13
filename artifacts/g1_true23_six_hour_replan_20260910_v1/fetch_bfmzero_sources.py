"""Cache pinned public source for review; no downloaded Python is executed."""

import hashlib
import json
import urllib.request
from pathlib import Path

OUT = Path("artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_sources_v1")
REVISION = "318cf44a3262e5bdec5944f82f1a5f509b95d09b"
FILES = [
    "LICENSE", "humanoidverse/agents/base.py", "humanoidverse/agents/nn_models.py",
    "humanoidverse/agents/nn_filters.py", "humanoidverse/agents/normalizers.py",
    "humanoidverse/agents/nn_filter_models.py", "humanoidverse/agents/envs/humanoidverse_isaac.py",
    "humanoidverse/agents/fb/model.py", "humanoidverse/envs/legged_base_task/legged_robot_base.py",
    "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py",
    "humanoidverse/utils/helpers.py", "humanoidverse/utils/torch_utils.py",
    "humanoidverse/utils/history_handler.py",
]


def main():
    OUT.mkdir(exist_ok=False)
    report = {"revision": REVISION, "files": {}}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/LeCAR-Lab/BFM-Zero/{REVISION}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(500_001)
            if len(raw) > 500_000:
                raise ValueError("source exceeds review bound")
            target = OUT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            report["files"][name] = {"url": url, "bytes": len(raw),
                                       "sha256": hashlib.sha256(raw).hexdigest()}
        except Exception as exc:
            report["files"][name] = {"url": url, "error": str(exc)}
    with (OUT / "sources.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
