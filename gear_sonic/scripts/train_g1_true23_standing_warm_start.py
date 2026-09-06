"""Motion-conditioned PPO from a checked standing-only decoder LoRA adapter.

Additive launcher: legacy breadth/polish, checkpoint and deployment contracts
remain unchanged. A distinct checked initialization imports only actor LoRA;
subsequent --resume restores the exact newly created PPO training state.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from gear_sonic.scripts import train_g1_23dof_mjlab_frozen_lora as frozen
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_standing_initialization import (
    apply_standing_initialization,
    read_standing_initialization,
)


def option(values, name, default=None):
    indices = [i for i, value in enumerate(values) if value == name]
    if len(indices) > 1 or (indices and indices[0] + 1 >= len(values)):
        raise SystemExit(f"{name} requires one value and may appear once")
    return values[indices[0] + 1] if indices else default


def install_standing_hooks(payload, descriptor, *, initialization_mode, run_directory):
    parent = frozen.FrozenPlatformLoraRunner

    class StandingInitializationRunner(parent):
        def load(self, path, *args, **kwargs):
            if not initialization_mode:
                return super().load(path, *args, **kwargs)
            requested = Path(path).expanduser().resolve()
            if (
                str(requested) != descriptor["adapter_path"]
                or file_sha256(requested) != descriptor["adapter_file_sha256"]
            ):
                raise ValueError("standing initialization path or bytes changed")
            receipt = apply_standing_initialization(self, payload, descriptor)
            with (run_directory / "standing_initialization.json").open("x") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True)
            return receipt

    frozen.FrozenPlatformLoraRunner = StandingInitializationRunner
    original_resolved = frozen.base._resolved_training_config

    def resolved(**kwargs):
        return {**original_resolved(**kwargs), "standing_training_initialization": descriptor}

    frozen.base._resolved_training_config = resolved
    frozen.base.CAUSAL_SOURCE_FILES += (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "utils/g1_true23_standing_initialization.py",
    )
    original_install = frozen._install_frozen_lora_hooks

    def checked_install(**kwargs):
        original_install(**kwargs)
        original_preflight = frozen.base.preflight

        def preflight(args):
            report = original_preflight(args)
            actual = report.get("frozen_platform_lora", {}).get("adapter_contract")
            if actual is not None and actual != descriptor["frozen_platform_contract"]:
                report["problems"].append("standing initialization differs from the actual frozen actor contract")
            report["standing_training_initialization"] = descriptor
            report["ready"] = not report["problems"]
            return report

        frozen.base.preflight = preflight

    frozen._install_frozen_lora_hooks = checked_install


def main(argv=None):
    values = []
    for value in sys.argv[1:] if argv is None else argv:
        values.extend(value.split("=", 1) if value.startswith("--") and "=" in value else [value])
    report = frozen._pop_option(values, "--standing-bootstrap-report")
    if report is None:
        raise SystemExit("requires --standing-bootstrap-report; this is not ordinary PPO resume")
    if (
        option(values, "--phase", "breadth") != "breadth"
        or option(values, "--adapter-init") is not None
        or option(values, "--behavior-bank") is not None
    ):
        raise SystemExit(
            "standing bootstrap is breadth initialization, not gate-selected polish or a behavior bank"
        )
    if option(values, "--actuation-profile") != "native_support_stateful_v2":
        raise SystemExit("standing bootstrap requires unchanged --actuation-profile native_support_stateful_v2")
    initialization_mode = option(values, "--resume") is None
    run_dir = option(values, "--run-dir")
    if run_dir is None:
        raise SystemExit("standing bootstrap requires an explicit separate --run-dir")
    destination = Path(run_dir).expanduser().resolve()
    if initialization_mode and values[0] != "preflight" and destination.exists() and any(destination.iterdir()):
        raise SystemExit("standing initialization requires a new empty run directory")
    payload, descriptor = read_standing_initialization(Path(report))
    contract = descriptor["frozen_platform_contract"]
    if (
        int(option(values, "--lora-rank", str(frozen.DEFAULT_RANK))) != contract["lora_rank"]
        or float(option(values, "--lora-alpha", str(frozen.DEFAULT_ALPHA))) != contract["lora_alpha"]
    ):
        raise SystemExit("standing initialization rank/alpha differs from requested training")
    source = Path(option(values, "--source-checkpoint", str(frozen.DEFAULT_SOURCE))).expanduser().resolve()
    if file_sha256(source) != contract["source_checkpoint_sha256"]:
        raise SystemExit("standing initialization source differs from requested frozen SONIC release")
    if initialization_mode:
        values.extend(("--resume", descriptor["adapter_path"]))
    install_standing_hooks(payload, descriptor, initialization_mode=initialization_mode, run_directory=destination)
    return frozen.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
