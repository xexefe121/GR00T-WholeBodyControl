"""Initial continued-policy parity with actual captured parent500 decisions."""

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_pico_foot_precision_replay import PicoFootPrecisionPolicy
from gear_sonic.utils.g1_true23_normal_lora_replay import NormalLoraPolicy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PRIOR = ROOT / "artifacts/g1_true23_pico_training_20260910_v1"


def main():
    output = HERE / "initial_policy_audit.json"
    if output.exists():
        raise FileExistsError("initial policy audit refuses overwrite")
    pins = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None:
            assert actual == expected, path
        pins[str(path)] = actual
        return path

    bind(__file__)
    torch.set_num_threads(1)
    path = bind(HERE / "smoke_v1/checkpoints/foot_precision_model_500.pt")
    kwargs = dict(
        warm_start_path=ASSETS / "sonic_release/g1_23dof_rev_1_0_init.pt",
        source_checkpoint_path=ASSETS / "sonic_release/last.pt",
    )
    policy = PicoFootPrecisionPolicy(path, **kwargs)
    identity = policy.identity()
    assert policy.completed_updates == 500
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        directory = PRIOR / "eval500_v1" / name
        report = json.loads(bind(directory / "report.json").read_text())
        with np.load(bind(directory / "attempts.npz", report["attempts_sha256"]), allow_pickle=False) as z:
            attempts = {
                k: z[k].copy()
                for k in ("released_raw23", "encoder267", "history930", "root_feedback9", "decoder994")
            }
        indices = np.unique(np.linspace(0, len(attempts["released_raw23"]) - 1, 192).astype(np.int64))
        for i in indices:
            action, decoder = policy.infer(
                attempts["encoder267"][i], attempts["history930"][i], attempts["root_feedback9"][i]
            )
            np.testing.assert_array_equal(action, attempts["released_raw23"][i])
            np.testing.assert_array_equal(decoder, attempts["decoder994"][i])
        rows.append(dict(name=name, decisions_bit_exact=len(indices)))
        print(json.dumps(rows[-1]), flush=True)
    assert policy.identity() == identity
    parent = bind(PRIOR / "train500_v1/checkpoints/normal_lora_model_500.pt")
    rejected = []
    for loader, artifact in ((NormalLoraPolicy, path), (PicoFootPrecisionPolicy, parent)):
        try:
            loader(artifact, **kwargs)
        except ValueError as error:
            assert "header" in str(error)
            rejected.append(dict(loader=loader.__name__, error=str(error)))
        else:
            raise AssertionError("wrong checkpoint family accepted")
    with output.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                rejected_wrong_families=rejected,
                identity=identity,
                inputs=pins,
                exact_initial_inference=True,
                new_dynamics_execution=False,
                deployment_ready=False,
                hardware_authorized=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
