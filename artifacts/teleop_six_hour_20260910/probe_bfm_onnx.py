"""Export, compare real-input outputs, and time both frozen inference backends."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, PACKAGE, load_motion
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, load_contract, reference_features
from gear_sonic.utils.g1_true23_bfmzero_onnx import BFMZeroONNX, export_graphs


def main():
    torch.set_num_threads(1)
    folder = ROOT / "artifacts/teleop_six_hour_20260910/bfm_onnx_v2"
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    if not folder.exists():
        export_graphs(weights, folder)
    eager = BFMZeroInference(weights)
    ort = BFMZeroONNX(folder)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    motion, _, _ = load_motion("pico")
    ref, priv = reference_features(motion, contract)
    traces = ROOT / "artifacts/teleop_six_hour_20260910/bfm_pico_feedback_v2/trace.npz"
    with np.load(traces, allow_pickle=False) as saved:
        indices = np.unique(np.linspace(0, len(saved["action"])-1, 256, dtype=int))
        state, history = saved["state"][indices], saved["history"][indices]
        all_actions = np.vstack((np.zeros((1,23), np.float32), saved["action"][:-1]))
        action = all_actions[indices]
    ref_inputs = (torch.from_numpy(ref[indices+11]), torch.from_numpy(priv[indices+11]))
    ze, zo = eager.backward(*ref_inputs), ort.backward(*ref_inputs)
    inputs = tuple(torch.from_numpy(x.astype(np.float32)) for x in (state, action, history)) + (ze,)
    ae, ao = eager.actor(*inputs), ort.actor(*inputs)
    result = dict(samples=len(indices), backward_max_abs=float((ze-zo).abs().max()),
                  actor_max_abs=float((ae-ao).abs().max()),
                  actor_rms=float(torch.sqrt(torch.mean((ae-ao)**2))), timings={})
    for name, policy in (("pytorch", eager), ("onnxruntime", ort)):
        for operation, args in (("actor", tuple(x[:1] for x in inputs)),
                                ("backward", tuple(x[:8] for x in ref_inputs))):
            function = getattr(policy, operation)
            for _ in range(30):
                function(*args)
            samples=[]
            for _ in range(500):
                tick=time.perf_counter()
                function(*args)
                samples.append(1000*(time.perf_counter()-tick))
            result["timings"][f"{name}_{operation}"] = dict(
                p50_ms=float(np.percentile(samples,50)), p95_ms=float(np.percentile(samples,95)),
                max_ms=max(samples), over_20ms=sum(x>20 for x in samples))
    result["parity_passed"] = result["backward_max_abs"] < 1e-4 and result["actor_max_abs"] < 1e-4
    result["training_active_during_timing"] = True
    result["closed_loop_qualified"] = False
    (folder / "probe.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
