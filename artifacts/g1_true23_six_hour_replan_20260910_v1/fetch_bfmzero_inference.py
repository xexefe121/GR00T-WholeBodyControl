"""Fetch only public actor/backward-map/normalizers, preserving tensor bytes."""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

BASE = Path("artifacts/g1_true23_six_hour_replan_20260910_v1")
INSPECT = BASE / "bfmzero_inspect_v1"
OUT = BASE / "bfmzero_inference_v1"


def main():
    report = json.loads((INSPECT / "report.json").read_text())
    header = json.loads((INSPECT / "safetensors_header.json").read_text())
    names = sorted(name for name in header if name.startswith(
        ("_actor.", "_backward_map.", "_obs_normalizer.")))
    selected = sorted(names, key=lambda name: header[name]["data_offsets"][0])
    ranges = []
    for name in selected:
        start, end = header[name]["data_offsets"]
        if ranges and ranges[-1][1] == start:
            ranges[-1][1] = end
            ranges[-1][2].append(name)
        else:
            ranges.append([start, end, [name]])
    total = sum(end - start for start, end, _ in ranges)
    if total > 150_000_000 or len(names) != 54:
        raise ValueError("unexpected inference subset")
    OUT.mkdir(exist_ok=False)
    tensors = {}
    receipts = []
    for start, end, group in ranges:
        absolute_start = start + report["tensor_data_offset"]
        absolute_end = end + report["tensor_data_offset"] - 1
        request = urllib.request.Request(report["weights_url"],
            headers={"Range": f"bytes={absolute_start}-{absolute_end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            crange = response.headers.get("Content-Range", "")
            if response.status != 206 or not crange.startswith(
                f"bytes {absolute_start}-{absolute_end}/"):
                raise ValueError("server did not honor inference-only byte range")
            raw = response.read(end - start + 1)
        if len(raw) != end - start:
            raise ValueError("truncated inference range")
        receipts.append({"range": [absolute_start, absolute_end], "bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest()})
        print(json.dumps(receipts[-1]), flush=True)
        for name in group:
            lo, hi = header[name]["data_offsets"]
            tensors[name] = raw[lo - start:hi - start]
    packed = {}
    offset = 0
    per_tensor = {}
    for name in names:
        packed[name] = {**header[name], "data_offsets": [offset, offset + len(tensors[name])]}
        offset += len(tensors[name])
        per_tensor[name] = hashlib.sha256(tensors[name]).hexdigest()
    packed["__metadata__"] = {"source_repo": report["repo"], "revision": report["revision"],
                              "subset": "inference-only; training state excluded"}
    packed_raw = json.dumps(packed, separators=(",", ":")).encode()
    packed_raw += b" " * (-len(packed_raw) % 8)
    path = OUT / "inference.safetensors"
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        for raw in [struct.pack("<Q", len(packed_raw)), packed_raw]:
            stream.write(raw)
            digest.update(raw)
        for name in names:
            stream.write(tensors[name])
            digest.update(tensors[name])
    result = {"source": report["weights_url"], "source_revision": report["revision"],
              "path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
              "range_receipts": receipts, "tensor_sha256": per_tensor,
              "full_source_file_downloaded": False, "hardware_authorized": False}
    with (OUT / "download.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "tensor_sha256"}), flush=True)


if __name__ == "__main__":
    main()
