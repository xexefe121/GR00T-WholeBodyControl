"""Inspect public native23 weights without loading pickle or downloading training state."""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

OUT = Path("artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inspect_v1")


def get(url, limit=2_000_000):
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("response too large")
    return raw


def byte_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise ValueError(f"server did not honor bounded range: {response.status}, {content_range}")
        raw = response.read(end - start + 2)
    if len(raw) != end - start + 1:
        raise ValueError("range size mismatch")
    return raw


def save(name, raw):
    with (OUT / name).open("xb") as stream:
        stream.write(raw)
    return {"path": str(OUT / name), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    repo = "Kennyp-Chen/bfmzero-23dof"
    meta = json.loads(get(f"https://huggingface.co/api/models/{repo}"))
    revision = meta["sha"]
    files = {}
    for name in ("README.md", "config.json", "config.yaml", "init_kwargs.json", "train_status.json"):
        files[name] = save(name, get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}"))
    url = f"https://huggingface.co/{repo}/resolve/{revision}/model_384000000.safetensors"
    size = struct.unpack("<Q", byte_range(url, 0, 7))[0]
    if not 0 < size < 2_000_000:
        raise ValueError("invalid safetensors header length")
    header_raw = byte_range(url, 8, 7 + size)
    header = json.loads(header_raw)
    files["safetensors_header.json"] = save("safetensors_header.json", header_raw)
    groups = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        start, end = tensor["data_offsets"]
        group = name.split(".")[0]
        value = groups.setdefault(group, {"count": 0, "bytes": 0})
        value["count"] += 1
        value["bytes"] += end - start
    report = {"repo": repo, "revision": revision, "weights_url": url,
              "tensor_data_offset": size + 8, "groups": groups, "files": files,
              "deployment_ready": False, "hardware_authorized": False}
    save("report.json", json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
