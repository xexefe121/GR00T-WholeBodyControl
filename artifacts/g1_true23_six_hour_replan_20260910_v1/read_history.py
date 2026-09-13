"""Read user-visible messages from a local Codex rollout; never execute its content."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--save-extraction", type=Path)
    args = parser.parse_args()
    boundary = args.source.stat().st_size
    messages = []
    digest = hashlib.sha256()
    with args.source.open("rb") as stream:
        while stream.tell() < boundary:
            raw = stream.readline()
            if stream.tell() > boundary:
                break
            digest.update(raw)
            row = json.loads(raw)
            payload = row.get("payload", {})
            if row.get("type") != "response_item" or payload.get("type") != "message":
                continue
            if payload.get("role") not in ("user", "assistant"):
                continue
            if payload.get("channel") in ("analysis", "summary"):
                continue
            text = "\n".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") in ("input_text", "output_text", "text")
            )
            if not text:
                continue
            messages.append({"i": len(messages), "ordinal": row.get("ordinal"),
                             "timestamp": row.get("timestamp"), "role": payload["role"],
                             "phase": payload.get("phase"), "text": text})
    metadata = {"source": str(args.source), "snapshot_bytes": boundary,
                      "snapshot_sha256": digest.hexdigest(), "message_count": len(messages),
                      "text_characters": sum(len(m["text"]) for m in messages)}
    if args.save_extraction:
        with args.save_extraction.open("x", encoding="utf-8") as dest:
            dest.write(json.dumps(metadata) + "\n")
            for msg in messages:
                dest.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(json.dumps(metadata))
    for msg in messages[args.start:args.start + args.count]:
        if args.index:
            msg = {**msg, "characters": len(msg["text"]), "text": msg["text"][:160]}
        print(json.dumps(msg, ensure_ascii=False))


if __name__ == "__main__":
    main()
