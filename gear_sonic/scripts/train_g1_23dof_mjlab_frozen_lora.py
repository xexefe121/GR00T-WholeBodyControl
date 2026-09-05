"""Train true23 G1 by adapting only frozen SONIC decoder LoRA tensors.

This launcher reuses the causal, clip-contained v14 task and replaces only the
actor/runner/sampling hooks in this process.  Breadth uses adaptive sampling.
Polish uses near-uniform sampling plus a fixed 10% feasibility-screened target
behavior bank.  No path in this module authorizes robot execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_history as base,
    train_g1_23dof_mjlab_teleop_v14 as v14,
)
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import (
    FrozenPlatformTrue23Core,
)
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import (
    FrozenPlatformLoraRunner,
)
from gear_sonic.utils.g1_true23_frozen_lora_gates import (
    frozen_lora_sampling_contract,
)
from gear_sonic.utils.g1_true23_actuation_profile import HEADER, StageOneActuationProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "low_latency" / "last.pt"
DEFAULT_RANK = 8
DEFAULT_ALPHA = 8.0
DEFAULT_LEARNING_RATE = 5.0e-6
_THIS_FILE = Path(__file__).resolve()
_SOURCE_FILES = (
    _THIS_FILE,
    _THIS_FILE.parents[1] / "trl" / "mjlab" / "frozen_platform_lora_actor.py",
    _THIS_FILE.parents[1] / "trl" / "mjlab" / "frozen_platform_lora_runner.py",
    _THIS_FILE.parents[1] / "utils" / "g1_true23_frozen_lora_gates.py",
)


def _pop_option(
    values: list[str],
    name: str,
    *,
    default: str | None = None,
) -> str | None:
    found: str | None = None
    index = 0
    while index < len(values):
        if values[index] != name:
            index += 1
            continue
        if found is not None:
            raise SystemExit(f"{name} may be supplied only once")
        if index + 1 >= len(values):
            raise SystemExit(f"{name} requires a value")
        found = values[index + 1]
        del values[index : index + 2]
    return default if found is None else found


def _span_sidecar(values: Sequence[str]) -> Path:
    for index, token in enumerate(values):
        if token == "--spans":
            if index + 1 >= len(values):
                raise SystemExit("--spans requires a value")
            return Path(values[index + 1]).expanduser().resolve()
    for index, token in enumerate(values):
        if token == "--motion-file":
            if index + 1 >= len(values):
                raise SystemExit("--motion-file requires a value")
            return Path(values[index + 1]).expanduser().with_suffix(".spans.json").resolve()
    raise SystemExit("frozen LoRA training requires --motion-file and corpus --spans")


def _validate_span_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"corpus span sidecar is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    spans = value.get("spans")
    clip_count = value.get("clip_count")
    total_frames = value.get("total_frames")
    if (
        value.get("kind") != "g1_true23_motion_corpus_spans_v1"
        or isinstance(clip_count, bool)
        or not isinstance(clip_count, int)
        or clip_count <= 0
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 12
        or not isinstance(spans, list)
        or len(spans) != clip_count
        or sum(int(span["length"]) for span in spans) != total_frames
    ):
        raise SystemExit("frozen LoRA corpus span identity mismatch")
    return value


def _behavior_bank_indices(
    path: Path | None,
    *,
    span_sidecar: Path,
    clip_count: int,
) -> tuple[int, ...]:
    if path is None:
        return ()
    if not path.is_file():
        raise SystemExit(f"behavior bank is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_sha = base._sha256(span_sidecar)  # noqa: SLF001
    raw = value.get("clip_indices")
    if (
        value.get("kind") != "g1_true23_frozen_lora_behavior_bank_v1"
        or value.get("span_sidecar_sha256") != expected_sha
        or value.get("feasibility_screen_passed") is not True
        or value.get("override_and_dedup_complete") is not True
        or not isinstance(raw, list)
        or not raw
        or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < clip_count for index in raw
        )
        or len(raw) != len(set(raw))
    ):
        raise SystemExit("behavior bank contract or corpus binding mismatch")
    return tuple(raw)


def _install_phase_sampler(
    *,
    phase: str,
    bank_indices: tuple[int, ...],
) -> None:
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    command_cls = causal_task.CausalHistoryMotionCommand
    parent_cls = command_cls.__mro__[1]
    original_uniform = command_cls._uniform_sampling  # noqa: SLF001

    def _sample_within_clip_indices(
        command: Any,
        env_ids: torch.Tensor,
        clip_indices: torch.Tensor,
    ) -> None:
        lead_in = 12
        trail = 2
        lo = command._clip_starts[clip_indices] + lead_in  # noqa: SLF001
        hi = command._clip_stops[clip_indices] - trail  # noqa: SLF001
        room = (hi - lo - 1).clamp(min=0)
        offset = (torch.rand(len(env_ids), device=command.device) * (room + 1)).long()
        command.time_steps[env_ids] = lo + torch.minimum(offset, room)
        command._env_clip_stop[env_ids] = hi  # noqa: SLF001

    def adaptive_sampling(command: Any, env_ids: torch.Tensor) -> None:
        # Preserve the stock failure-driven frame distribution, then enforce
        # clip-contained causal margins on the sampled frame.
        parent_cls._adaptive_sampling(command, env_ids)  # noqa: SLF001
        sampled = command.time_steps[env_ids].clamp(0, command.motion.time_step_total - 1)
        owners = command._frame_owner[sampled]  # noqa: SLF001
        lengths = command._clip_stops - command._clip_starts  # noqa: SLF001
        eligible = torch.where(lengths > 15)[0]
        if eligible.numel() == 0:
            raise ValueError("no causal-safe clip remains for adaptive sampling")
        invalid = lengths[owners] <= 15
        if torch.any(invalid):
            invalid_count = int(torch.count_nonzero(invalid).item())
            owners[invalid] = eligible[
                torch.randint(
                    eligible.numel(),
                    (invalid_count,),
                    device=command.device,
                )
            ]
        lo = command._clip_starts[owners] + 12  # noqa: SLF001
        hi = command._clip_stops[owners] - 2  # noqa: SLF001
        command.time_steps[env_ids] = torch.minimum(torch.maximum(sampled, lo), hi - 1)
        command._env_clip_stop[env_ids] = hi  # noqa: SLF001

    def polish_uniform(command: Any, env_ids: torch.Tensor) -> None:
        original_uniform(command, env_ids)
        if not bank_indices or len(env_ids) == 0:
            return
        bank_mask = torch.rand(len(env_ids), device=command.device) < 0.10
        selected_envs = env_ids[bank_mask]
        if len(selected_envs) == 0:
            return
        bank = torch.tensor(bank_indices, dtype=torch.long, device=command.device)
        selected_clips = bank[torch.randint(bank.numel(), (len(selected_envs),), device=command.device)]
        _sample_within_clip_indices(command, selected_envs, selected_clips)

    command_cls._adaptive_sampling = adaptive_sampling
    if phase == "polish":
        command_cls._uniform_sampling = polish_uniform

    original_builder = causal_task.make_causal_history_recovery_env_cfg

    def phase_builder(*args: Any, **kwargs: Any) -> Any:
        cfg = original_builder(*args, **kwargs)
        cfg.commands["motion"].sampling_mode = "adaptive" if phase == "breadth" else "uniform"
        return cfg

    causal_task.make_causal_history_recovery_env_cfg = phase_builder


def _install_frozen_lora_hooks(
    *,
    source_checkpoint: Path,
    lora_rank: int,
    lora_alpha: float,
    phase: str,
    span_sidecar: Path,
    behavior_bank: Path | None,
    bank_indices: tuple[int, ...],
    adapter_initialization: Path | None,
    adapter_initialization_mode: bool,
    actuation_profile: StageOneActuationProfile | None = None,
) -> None:
    v14._install_isolated_v14_hooks(span_sidecar)  # noqa: SLF001
    _install_phase_sampler(phase=phase, bank_indices=bank_indices)
    if actuation_profile is not None:
        from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
        from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import apply_stage_one_actuation_profile

        original_env_builder = causal_task.make_causal_history_recovery_env_cfg

        def profiled_env_builder(**kwargs: Any) -> Any:
            return apply_stage_one_actuation_profile(original_env_builder(**kwargs), actuation_profile)

        causal_task.make_causal_history_recovery_env_cfg = profiled_env_builder

    if not adapter_initialization_mode:
        launcher_runner = FrozenPlatformLoraRunner
    else:
        if adapter_initialization is None:
            raise AssertionError("adapter initialization mode lacks source")

        class AdapterInitializationRunner(FrozenPlatformLoraRunner):
            def load(self, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                requested = Path(path).expanduser().resolve()
                if requested != adapter_initialization:
                    raise ValueError("launcher adapter initialization path changed")
                return self.load_adapter_initialization(str(requested))

        launcher_runner = AdapterInitializationRunner

    runner_module.CausalHistoryMjlabOnPolicyRunner = launcher_runner
    base.CausalHistoryMjlabOnPolicyRunner = launcher_runner

    def frozen_cfg() -> Any:
        from gear_sonic.trl.mjlab.config import (
            frozen_platform_lora_runner_cfg,
        )

        cfg = frozen_platform_lora_runner_cfg(
            source_checkpoint_path=str(source_checkpoint),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        cfg.run_name = phase
        return cfg

    original_run_training = base.run_training

    def run_training_with_frozen_config(args: Any) -> Path:
        # Keep preflight importable without MJLab. Smoke/train enter here only
        # after the pinned simulator runtime is available.
        from gear_sonic.trl.mjlab import config as config_module

        config_module.true23_mjlab_ppo_runner_cfg = frozen_cfg
        return original_run_training(args)

    base.run_training = run_training_with_frozen_config

    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in _SOURCE_FILES:
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    if actuation_profile is not None:
        base.CAUSAL_SOURCE_FILES += (
            REPO_ROOT / HEADER,
            REPO_ROOT / "gear_sonic/utils/g1_true23_actuation_profile.py",
            REPO_ROOT / "gear_sonic/envs/mjlab/sonic_true23_stage_one_actuation.py",
        )

    original_resolved = base._resolved_training_config

    def resolved_with_frozen_lora(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original_resolved(*args, **kwargs)
        if actuation_profile is not None:
            resolved["stage_one_actuation"] = actuation_profile.contract()
        resolved["frozen_platform_lora"] = {
            "schema_version": 1,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": base._sha256(source_checkpoint),  # noqa: SLF001
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "phase": phase,
            "span_sidecar": {
                "path": str(span_sidecar),
                "sha256": base._sha256(span_sidecar),  # noqa: SLF001
            },
            "behavior_bank": (
                {
                    "path": str(behavior_bank),
                    "sha256": base._sha256(behavior_bank),  # noqa: SLF001
                    "feasibility_screen_passed": True,
                    "override_and_dedup_complete": True,
                }
                if behavior_bank is not None
                else None
            ),
            "behavior_bank_clip_indices": list(bank_indices),
            "adapter_initialization": (
                {
                    "path": str(adapter_initialization),
                    "sha256": base._sha256(adapter_initialization),  # noqa: SLF001
                    "critic_reused": False,
                    "optimizer_reused": False,
                    "counters_reused": False,
                }
                if adapter_initialization is not None
                else None
            ),
            "sampling": frozen_lora_sampling_contract(),
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        return resolved

    base._resolved_training_config = resolved_with_frozen_lora

    original_preflight = base.preflight

    def preflight_with_frozen_lora(args: Any) -> dict[str, Any]:
        report = original_preflight(args)
        if actuation_profile is not None:
            report["stage_one_actuation"] = actuation_profile.contract()
        problems = list(report["problems"])
        contract: dict[str, Any] | None = None
        if not source_checkpoint.is_file():
            problems.append(f"frozen source checkpoint missing: {source_checkpoint}")
        elif not problems:
            try:
                core = FrozenPlatformTrue23Core(
                    warm_start_path=args.warm_start,
                    source_checkpoint_path=source_checkpoint,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                contract = core.adapter_contract()
                core.assert_frozen_platform_unchanged()
            except Exception as exc:  # noqa: BLE001 - audit must report all failures.
                problems.append(f"frozen platform parity failed: {exc}")
        report["problems"] = problems
        report["ready"] = not problems
        report["frozen_platform_lora"] = {
            "adapter_contract": contract,
            "phase": phase,
            "span_sidecar": {
                "path": str(span_sidecar),
                "sha256": base._sha256(span_sidecar),  # noqa: SLF001
            },
            "behavior_bank": (
                {
                    "path": str(behavior_bank),
                    "sha256": base._sha256(behavior_bank),  # noqa: SLF001
                }
                if behavior_bank is not None
                else None
            ),
            "behavior_bank_clip_indices": list(bank_indices),
            "adapter_initialization": (
                str(adapter_initialization) if adapter_initialization is not None else None
            ),
            "sampling": frozen_lora_sampling_contract(),
            "simulator_only": True,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        return report

    base.preflight = preflight_with_frozen_lora


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    values = list(sys.argv[1:] if argv is None else argv)
    source = (
        Path(_pop_option(values, "--source-checkpoint", default=str(DEFAULT_SOURCE)) or DEFAULT_SOURCE)
        .expanduser()
        .resolve()
    )
    rank = int(_pop_option(values, "--lora-rank", default=str(DEFAULT_RANK)) or 0)
    alpha = float(_pop_option(values, "--lora-alpha", default=str(DEFAULT_ALPHA)) or 0.0)
    phase = _pop_option(values, "--phase", default="breadth")
    bank_text = _pop_option(values, "--behavior-bank")
    adapter_text = _pop_option(values, "--adapter-init")
    actuation_name = _pop_option(values, "--actuation-profile")
    if actuation_name not in {None, "stage_one_cpp"}:
        raise SystemExit("--actuation-profile must be stage_one_cpp (simulator only)")
    actuation_profile = (
        StageOneActuationProfile.from_cpp(REPO_ROOT / HEADER) if actuation_name is not None else None
    )
    resume_supplied = "--resume" in values
    if rank <= 0 or alpha <= 0.0:
        raise SystemExit("LoRA rank and alpha must be positive")
    if phase not in {"breadth", "polish"}:
        raise SystemExit("--phase must be breadth or polish")
    if phase == "polish" and bank_text is None:
        raise SystemExit("polish phase requires --behavior-bank")
    if phase == "polish" and adapter_text is None:
        raise SystemExit("polish phase requires gate-selected --adapter-init")
    if phase != "polish" and adapter_text is not None:
        raise SystemExit("--adapter-init is reserved for polish phase")
    span_sidecar = _span_sidecar(values)
    span_payload = _validate_span_sidecar(span_sidecar)
    behavior_bank = Path(bank_text).expanduser().resolve() if bank_text is not None else None
    bank_indices = _behavior_bank_indices(
        behavior_bank,
        span_sidecar=span_sidecar,
        clip_count=span_payload["clip_count"],
    )
    if actuation_profile is not None and behavior_bank is not None:
        bank_profile = json.loads(behavior_bank.read_text(encoding="utf-8")).get("actuation_profile_source_sha256")
        if bank_profile != actuation_profile.source_sha256:
            raise SystemExit("behavior bank was not qualified against this actuation profile")
    adapter_initialization = Path(adapter_text).expanduser().resolve() if adapter_text is not None else None
    if adapter_initialization is not None:
        if not adapter_initialization.is_file():
            raise SystemExit(f"adapter initialization is unavailable: {adapter_initialization}")
        if not resume_supplied:
            values.extend(("--resume", str(adapter_initialization)))
    if "--learning-rate" not in values:
        values.extend(("--learning-rate", str(DEFAULT_LEARNING_RATE)))
    _install_frozen_lora_hooks(
        source_checkpoint=source,
        lora_rank=rank,
        lora_alpha=alpha,
        phase=phase,
        span_sidecar=span_sidecar,
        behavior_bank=behavior_bank,
        bank_indices=bank_indices,
        adapter_initialization=adapter_initialization,
        adapter_initialization_mode=(adapter_initialization is not None and not resume_supplied),
        actuation_profile=actuation_profile,
    )
    # v14 consumes --spans itself; the shared parser does not know this option.
    if "--spans" in values:
        span_index = values.index("--spans")
        del values[span_index : span_index + 2]
    return base.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
