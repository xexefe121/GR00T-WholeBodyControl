"""Decode an existing incident recording offline; never initialize DDS.

CRC/status interpretation happens only after recording. CDR bytes are SDK
reserializations, not the original wire bytes. Observed changes are not proof
of cause, firmware compatibility, applied commands or standing restoration.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

from gear_sonic.scripts.record_g1_true23_incident_readonly import decode_packet, load_readonly_api
from gear_sonic.utils.g1_true23_incident_capture import StatusEdges, TOPICS, json_safe, sha256


def decode_recording(source: Path, output: Path, *, decode=None):
    """Preserve original capture separately and verify its packet/hash accounting."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    summary_path = source / "summary.json"
    summary_hash = sha256(summary_path)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    packet_path = source / "packets.jsonl.gz"
    packet_hash = sha256(packet_path)
    if packet_hash != source_summary["packets_sha256"]:
        raise ValueError("recorded compressed-packet SHA256 mismatch")
    codec_paths = []
    if decode is None:
        _, _, types, crc = load_readonly_api()

        def decode(topic, raw):
            return decode_packet(topic, raw, types, crc)

        codec_paths = [Path(inspect.getfile(kind)).resolve() for kind in types.values()]
        codec_paths.append(Path(inspect.getfile(type(crc))).resolve())
        native_crc = getattr(getattr(crc, "crc_lib", None), "_name", None)
        if native_crc is not None and Path(native_crc).is_file():
            codec_paths.append(Path(native_crc).resolve())
        codec_paths += [
            Path(module.__file__).resolve()
            for name, module in tuple(sys.modules.items())
            if name.startswith(("unitree_sdk2py.idl.unitree_hg.", "unitree_sdk2py.idl.unitree_api."))
            and getattr(module, "__file__", None)
        ]
    decoder_paths = [
        Path(__file__).resolve(),
        Path(inspect.getfile(decode_packet)).resolve(),
        Path(inspect.getfile(StatusEdges)).resolve(),
    ]
    decoder_paths += codec_paths
    decoder_hashes = {str(path): sha256(path) for path in decoder_paths}
    output.mkdir(parents=True, exist_ok=False)
    observer = StatusEdges()
    errors = Counter()
    counts = Counter()
    last_index = -1
    saw_end = False
    result_path = output / "decoded.jsonl.gz"
    with (
        gzip.open(packet_path, "rt", encoding="utf-8") as source_stream,
        gzip.open(result_path, "xt", encoding="utf-8", compresslevel=1) as destination,
    ):
        for ordinal, line in enumerate(source_stream):
            record = json.loads(line)
            if ordinal == 0:
                if record.get("event") != "capture_start" or record.get("schema_version") != 1:
                    raise ValueError("unsupported incident capture header")
            elif record.get("event") == "packet" and not saw_end:
                topic = record["topic"]
                index = record["capture_index"]
                if topic not in TOPICS or not isinstance(index, int) or index <= last_index:
                    raise ValueError("invalid packet topic or receipt order")
                last_index = index
                counts[topic] += 1
                # Never trust an earlier interpretation over the saved bytes.
                record.pop("decoded", None)
                record.pop("decode_error", None)
                if "reserialized_cdr_b64" in record:
                    raw = base64.b64decode(record["reserialized_cdr_b64"], validate=True)
                    if hashlib.sha256(raw).hexdigest() != record["cdr_sha256"]:
                        raise ValueError("recorded per-packet CDR SHA256 mismatch")
                    try:
                        record["decoded"] = decode(topic, raw)
                    except Exception as error:
                        record["decode_error"] = f"{type(error).__name__}: {error}"
                        errors["decode_error"] += 1
                elif "snapshot_error" in record:
                    errors["snapshot_error"] += 1
                else:
                    raise ValueError("packet missing both payload and snapshot-error evidence")
                record["observations"] = observer.observe(record)
            elif record.get("event") == "capture_end" and not saw_end:
                saw_end = True
            else:
                raise ValueError("unexpected record order")
            destination.write(json.dumps(json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")
    if not saw_end or dict(counts) != source_summary["written_packets"]:
        raise ValueError("capture footer or packet accounting mismatch")
    if sha256(packet_path) != packet_hash or sha256(summary_path) != summary_hash:
        raise ValueError("capture source changed while decoding")
    report = {
        "kind": "g1_true23_incident_offline_decode_v1",
        "source_directory": str(source),
        "source_packets_sha256": packet_hash,
        "source_summary_sha256": summary_hash,
        "decoder_source_sha256": decoder_hashes,
        "decoder_source_unchanged": all(sha256(path) == decoder_hashes[str(path)] for path in decoder_paths),
        "capture_and_decoder_schema_equivalence_proven": False,
        "decoded_file_sha256": sha256(result_path),
        "packet_counts": dict(counts),
        "errors": dict(errors),
        "observations": dict(observer.counts),
        "offline_decode_completed": True,
        "source_capture_completed": source_summary["capture_completed"],
        "source_all_observed_callbacks_preserved": source_summary["all_observed_callbacks_preserved"],
        "collector_dropped_packets": source_summary["collector_dropped_packets"],
        "upstream_DDS_or_network_loss_excluded": False,
        "dds_subscriptions_opened": False,
        "robot_commands_published": False,
        "physical_cause_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "decode_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    report = decode_recording(args.capture_directory, args.output_directory)
    print(json.dumps(report, indent=2))
    return (
        0
        if (
            report["source_capture_completed"]
            and report["source_all_observed_callbacks_preserved"]
            and not report["errors"]
            and report["decoder_source_unchanged"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
