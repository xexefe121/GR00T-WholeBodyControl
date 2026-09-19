"""Exact CPU ONNX acceleration for the frozen native23 BFM inference graph.

Export is from the hash-verified local safetensors graph. This is a simulation
runtime optimization, never a new policy or evidence of robot qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, WEIGHTS_SHA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, last_action, history, z):
        return self.policy.actor(state, last_action, history, z)


class _Backward(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, privileged):
        return self.policy.backward(state, privileged)


class _ExportPolicy(BFMZeroInference):
    def layernorm(self, key, x):
        # Width comes from immutable weights; the graph batch stays dynamic.
        weight = self.weights[key + ".weight"]
        return F.layer_norm(x, (int(weight.numel()),), weight,
                            self.weights[key + ".bias"], eps=1e-5)


def export_graphs(weights, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy = _ExportPolicy(weights)
    actor_inputs = tuple(torch.zeros(1, n) for n in (52, 23, 300, 256))
    backward_inputs = (torch.zeros(8, 52), torch.zeros(8, 373))
    for name, module, inputs, names, output in (
        ("actor", _Actor(policy), actor_inputs, ["state", "last_action", "history", "z"], "action"),
        ("backward", _Backward(policy), backward_inputs, ["state", "privileged"], "z"),
    ):
        torch.onnx.export(module.eval(), inputs, str(directory / f"{name}.onnx"),
                          input_names=names, output_names=[output], opset_version=17,
                          dynamic_axes={key: {0: "batch"} for key in names + [output]},
                          dynamo=False)
    manifest = dict(weights_sha256=WEIGHTS_SHA, exporter_sha256=digest(__file__),
                    actor_sha256=digest(directory / "actor.onnx"),
                    backward_sha256=digest(directory / "backward.onnx"),
                    backend="onnxruntime_CPUExecutionProvider", simulator_qualified=False,
                    hardware_authorized=False)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class BFMZeroONNX:
    """Drop-in tensor interface; shape and finite checks retained at runtime."""
    def __init__(self, directory, threads=1):
        import onnxruntime as ort
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["weights_sha256"] != WEIGHTS_SHA:
            raise ValueError("unexpected original BFM weights")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessions = {}
        for name in ("actor", "backward"):
            path = directory / f"{name}.onnx"
            if digest(path) != manifest[f"{name}_sha256"]:
                raise ValueError(f"changed {name} ONNX artifact")
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=opts,
                                                       providers=["CPUExecutionProvider"])

    def _run(self, name, values, sizes):
        inputs = {}
        batch = None
        for key, value in values.items():
            value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if value.ndim != 2 or value.shape[1] != sizes[key] or not np.isfinite(value).all():
                raise ValueError(f"invalid BFM {name} input {key}")
            if batch is not None and value.shape[0] != batch:
                raise ValueError("BFM batch dimensions differ")
            batch = value.shape[0]
            inputs[key] = np.ascontiguousarray(value, dtype=np.float32)
        result = self.sessions[name].run(None, inputs)[0]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite BFM ONNX output")
        return torch.from_numpy(result)

    def actor(self, state, last_action, history, z):
        return self._run("actor", dict(state=state, last_action=last_action, history=history, z=z),
                         dict(state=52, last_action=23, history=300, z=256))

    def backward(self, state, privileged):
        return self._run("backward", dict(state=state, privileged=privileged),
                         dict(state=52, privileged=373))
