"""Compare actual ONNX-state streams with CPU and both CUDA math policies."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_torch
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_standing_initialization import read_standing_initialization
from gear_sonic.utils.g1_true23_standing_retention import StandingOutputAnchor, load_training_states
from gear_sonic.utils.g1_true23_training_precision import backend_state, ieee_training_precision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PREVIOUS = HERE.parent / "standing_retention_ppo_20260906_v1"
inputs, streams, checkpoints, records, output_arrays = {}, {}, {}, [], {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = file_sha256(path)
    if actual != inputs.get(str(path), actual) or (expected is not None and actual != expected):
        raise ValueError(f"recorded-stream precision input changed: {path}")
    inputs[str(path)] = actual
    return path


def read(path):
    value = json.loads(bind(path).read_text())
    for source, digest in value.get("inputs", {}).items():
        bind(source, digest)
    return value


def errors(actual, reference):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return dict(max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta**2))))


bind(Path(__file__))
receipt = read(HERE / "breadth_serial/standing_initialization.json")
for label, evaluation, training in (
    ("tf32_trained_100", HERE / "tf32_recorded_evaluation/report.json", PREVIOUS / "breadth_serial"),
    ("ieee_trained_100", HERE / "model_100/recorded_evaluation/report.json", HERE / "breadth_serial"),
):
    report = read(evaluation)
    assert len(report["records"]) == 11
    arrays, cases, cursor = [], [], 0
    for row in report["records"]:
        if row.get("not_executed"):
            assert row["name"] == "elbow_crawling"
            continue
        with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
            values = {key[7:]: archive[key].copy() for key in archive.files if key.startswith("policy_")}
        assert values["inference_returned"].all()
        assert len(values["encoder267"]) == row["result"]["policy_input_trace"]["inference_calls"]
        arrays.append(values)
        cases.append(dict(case=row["label"], start=cursor, stop=cursor + len(values["encoder267"])))
        cursor += len(values["encoder267"])
    data = {key: np.concatenate([row[key] for row in arrays]) for key in arrays[0]}
    assert np.isfinite(data["decoder994"]).all() and np.isfinite(data["raw23"]).all()
    np.testing.assert_array_equal(data["history930"], data["decoder994"][:, 64:])
    streams[label] = dict(data=data, cases=cases, rows=cursor)
    checkpoint = load_frozen_platform_lora_checkpoint(
        bind(training / "checkpoints/frozen_lora_model_100.pt"),
        expected_contract=receipt["frozen_platform_contract"],
    )
    assert checkpoint["update_count"] == 100
    assert report["pair"]["source"]["adapter_state_sha256"] == checkpoint["adapter_state_sha256"]
    checkpoints[label] = checkpoint
standing = HERE.parent / "interior_effort_20260906_v1"
payload, descriptor = read_standing_initialization(bind(standing / "standing_lora500/report.json"))
training_states, _ = load_training_states(bind(standing / "representable_standing/report.json"), descriptor)
core = FrozenPlatformTrue23Core(
    warm_start_path=bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
    source_checkpoint_path=bind(ASSETS / "low_latency/last.pt"),
    lora_rank=8,
    lora_alpha=8,
).eval()
anchor_reconstruction = None
for mode, device in (("cpu_ieee", "cpu"), ("cuda_ieee", "cuda:0"), ("cuda_tf32", "cuda:0")):
    if mode == "cuda_tf32":
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends(allow_tf32=True, deterministic=False)
    context = nullcontext() if mode == "cuda_tf32" else ieee_training_precision()
    with context:
        core.to(device)
        if mode == "cuda_ieee":
            anchor = StandingOutputAnchor(core, training_states, payload, batch_size=128, seed=20260906)
            runtime = json.loads(bind(HERE / "breadth_serial/standing_retention.json").read_text())
            assert anchor.cache_sha256 == runtime["anchor_cache_sha256"]
            anchor_reconstruction = dict(cache_sha256=anchor.cache_sha256, matches_actual_training=True)
            del anchor
        for label, stream in streams.items():
            print(json.dumps(dict(comparing=label, backend=mode, rows=stream["rows"])), flush=True)
            core.load_lora_state_dict(checkpoints[label]["adapter_state_dict"], strict=True)
            data = stream["data"]
            semantic = torch.from_numpy(data["encoder267"]).to(device)
            history = torch.from_numpy(data["history930"]).to(device)
            recorded_decoder = torch.from_numpy(data["decoder994"]).to(device)
            encoder_records = []
            with torch.no_grad():
                proprio = core.codec.encode_proprioception(history)
                for size in (1, 32, 64, 128):
                    tokens = torch.cat([core.encode(chunk) for chunk in semantic.split(size)])
                    tokens_cpu = tokens.cpu().numpy()
                    differences = tokens_cpu != data["decoder994"][:, :64]
                    encoder_records.append(
                        dict(
                            batch_size=size,
                            differing_elements=int(differences.sum()),
                            differing_rows=int(differences.any(axis=1).sum()),
                            error=errors(tokens_cpu, data["decoder994"][:, :64]),
                            cases=[
                                dict(
                                    **case,
                                    differing_rows=int(
                                        differences[case["start"] : case["stop"]].any(axis=1).sum()
                                    ),
                                )
                                for case in stream["cases"]
                            ],
                        )
                    )
                    output_arrays[f"{label}.{mode}.tokens_batch{size}"] = tokens_cpu
                    if size == 32:
                        tokens32 = tokens
                actual_input = torch.cat((tokens32, proprio), dim=-1)
                raw = torch.cat(
                    [core.codec.decode_action(core.decoder(chunk)) for chunk in actual_input.split(32)]
                )
                isolated = torch.cat(
                    [core.codec.decode_action(core.decoder(chunk)) for chunk in recorded_decoder.split(32)]
                )
                reference = torch.from_numpy(data["raw23"]).to(device)
                target = safe_target_transform_torch(raw)[1].cpu().numpy()
                reference_target = safe_target_transform_torch(reference)[1].cpu().numpy()
                raw_cpu, isolated_cpu = raw.cpu().numpy(), isolated.cpu().numpy()
            output_arrays[f"{label}.{mode}.raw_batch32"] = raw_cpu
            output_arrays[f"{label}.{mode}.recorded_token_raw_batch32"] = isolated_cpu
            records.append(
                dict(
                    checkpoint=label,
                    backend=mode,
                    actual_backend_state=backend_state(),
                    rows=stream["rows"],
                    encoder=encoder_records,
                    proprio_error=errors(proprio.cpu().numpy(), data["history930"]),
                    raw_action_error=errors(raw_cpu, data["raw23"]),
                    decoder_with_recorded_tokens_error=errors(isolated_cpu, data["raw23"]),
                    safe_target_error_rad=errors(target, reference_target),
                    cases=[
                        dict(
                            **case,
                            raw_error=errors(
                                raw_cpu[case["start"] : case["stop"]], data["raw23"][case["start"] : case["stop"]]
                            ),
                            target_error_rad=errors(
                                target[case["start"] : case["stop"]],
                                reference_target[case["start"] : case["stop"]],
                            ),
                        )
                        for case in stream["cases"]
                    ],
                )
            )
core.assert_frozen_platform_unchanged()
output = HERE / "recorded_stream_precision.npz"
with output.open("xb") as stream:
    np.savez_compressed(stream, **output_arrays)
bind(output)
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
        bind(path)
for path in list(inputs):
    bind(path)
dump(
    HERE / "recorded_stream_precision.json",
    dict(
        kind="g1_true23_actual_motion_stream_precision_comparison_v1",
        inputs=inputs,
        records=records,
        ieee_anchor_reconstruction=anchor_reconstruction,
        held_out_arrays_loaded=False,
        additional_optimization_performed=False,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
