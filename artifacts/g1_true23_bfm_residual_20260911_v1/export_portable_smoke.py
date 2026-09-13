"""Migrate this session's locally generated smoke metadata; keep originals."""

import json
from pathlib import Path
import torch

base = Path(__file__).resolve().parent
for number in (0, 2):
    path = base / "smoke64_v2" / f"residual_{number:05d}.pt"
    # These two files were generated locally by the current trainer, not fetched
    # checkpoints. The external BFM artifact remains safetensors-only.
    value = torch.load(path, map_location="cpu", weights_only=False)
    value["manifest"] = json.loads(json.dumps(value["manifest"], allow_nan=False))
    output = path.with_name(path.stem + "_portable.pt")
    with output.open("xb") as stream:
        torch.save(value, stream)
    portable = torch.load(output, map_location="cpu", weights_only=True)
    print(json.dumps(dict(path=str(output), updates=portable["completed_updates"], weights_only_passed=True)))
