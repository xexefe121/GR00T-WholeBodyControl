"""Exact simulator-only PPO runner for frozen SONIC decoder LoRA."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import (
    CausalHistoryMjlabOnPolicyRunner,
)
from gear_sonic.trl.mjlab.runner import (
    _environment_common_step_counter,
    _require_learning_rate,
    _require_nonnegative_integer,
    _set_environment_common_step_counter,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_mjlab_training import (
    validate_mjlab_training_lineage,
)

CHECKPOINT_HEADER = "g1_true23_frozen_sonic_lora_training_checkpoint"
CHECKPOINT_KIND = "g1_true23_frozen_sonic_lora_training_resume"
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ROLE = "simulator_training_resume_only"
RUNTIME_FILENAME = "frozen_platform_lora_runtime.json"


def _state_sha256(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("tensor state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("tensor state must map strings to tensors")
        contiguous = value.detach().cpu().contiguous()
        if contiguous.is_floating_point() and not torch.isfinite(contiguous).all():
            raise ValueError(f"tensor state contains NaN or Inf: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(contract))).hexdigest()


def _checkpoint_header() -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "role": CHECKPOINT_ROLE,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
    }


def _validate_checkpoint(
    value: Any,
    *,
    expected_contract: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    minimum_update_count: int = 0,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("frozen LoRA checkpoint must be a mapping")
    if value.get(CHECKPOINT_HEADER) != _checkpoint_header():
        raise ValueError("frozen LoRA checkpoint header mismatch")
    contract = value.get("adapter_contract")
    if not isinstance(contract, Mapping) or dict(contract) != dict(expected_contract):
        raise ValueError("frozen LoRA adapter contract mismatch")
    if value.get("adapter_contract_sha256") != _contract_sha256(expected_contract):
        raise ValueError("frozen LoRA adapter contract hash mismatch")
    lineage = validate_mjlab_training_lineage(value.get("lineage"))
    if value.get("lineage_sha256") != lineage["lineage_sha256"]:
        raise ValueError("frozen LoRA checkpoint lineage_sha256 mismatch")
    expected = validate_mjlab_training_lineage(expected_lineage)
    if lineage["lineage_sha256"] != expected["lineage_sha256"]:
        raise ValueError("frozen LoRA checkpoint lineage mismatch")
    update_count = _require_nonnegative_integer(
        value.get("update_count"),
        "frozen LoRA checkpoint update_count",
    )
    if update_count < minimum_update_count:
        raise ValueError("frozen LoRA checkpoint predates current runner")
    adapter_state = value.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("frozen LoRA checkpoint lacks adapter state")
    if value.get("adapter_state_sha256") != _state_sha256(adapter_state):
        raise ValueError("frozen LoRA adapter state hash mismatch")
    critic_state = value.get("critic_state_dict")
    if not isinstance(critic_state, Mapping):
        raise ValueError("frozen LoRA checkpoint lacks critic state")
    if value.get("critic_state_sha256") != _state_sha256(critic_state):
        raise ValueError("frozen LoRA critic state hash mismatch")
    if not isinstance(value.get("optimizer_state_dict"), Mapping):
        raise ValueError("frozen LoRA checkpoint lacks optimizer state")
    trainer_state = value.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("frozen LoRA checkpoint lacks trainer state")
    if (
        trainer_state.get("completed_update_count") != update_count
        or trainer_state.get("current_learning_iteration") != update_count
    ):
        raise ValueError("frozen LoRA trainer counters mismatch")
    merged_hash = value.get("merged_true23_policy_sha256")
    if (
        not isinstance(merged_hash, str)
        or len(merged_hash) != 64
        or any(character not in "0123456789abcdef" for character in merged_hash)
    ):
        raise ValueError("frozen LoRA merged policy hash is invalid")
    return value


def load_frozen_platform_lora_checkpoint(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
    expected_lineage: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Weights-only load one adapter resume under exact contract/lineage."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("frozen LoRA checkpoint may not be a symlink")
    requested = requested.resolve()
    if not requested.is_file():
        raise FileNotFoundError(f"frozen LoRA checkpoint missing: {requested}")
    loaded = torch.load(requested, map_location="cpu", weights_only=True)
    if expected_lineage is None:
        if not isinstance(loaded, Mapping):
            raise ValueError("frozen LoRA checkpoint must be a mapping")
        raw_lineage = loaded.get("lineage")
        if not isinstance(raw_lineage, Mapping):
            raise ValueError("frozen LoRA checkpoint lacks lineage")
        expected_lineage = raw_lineage
    return _validate_checkpoint(
        loaded,
        expected_contract=expected_contract,
        expected_lineage=expected_lineage,
    )


def _save_checkpoint_atomic(
    output: Path,
    checkpoint: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen LoRA checkpoint: {output}")
    if output.is_symlink():
        raise ValueError("frozen LoRA checkpoint path may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(checkpoint), temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_checkpoint(
            loaded,
            expected_contract=expected_contract,
            expected_lineage=expected_lineage,
        )
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen LoRA checkpoint: {output}"
            )
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class FrozenPlatformLoraRunner(CausalHistoryMjlabOnPolicyRunner):
    """Fresh critic/optimizer; frozen platform; decoder LoRA only."""

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        count = _require_nonnegative_integer(
            update_count,
            "frozen LoRA checkpoint update_count",
        )
        return self.checkpoint_dir / f"frozen_lora_model_{count}.pt"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        actor = self.alg.get_policy()
        if not hasattr(actor, "core") or not hasattr(actor, "export_lora_sidecar"):
            raise TypeError("frozen LoRA runner requires frozen-platform actor")
        self._adapter_contract = actor.core.adapter_contract()
        trainable_actor = [
            (name, parameter)
            for name, parameter in actor.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_actor or any(
            not name.endswith(("lora_a", "lora_b"))
            for name, _ in trainable_actor
        ):
            raise ValueError(
                "only decoder LoRA tensors may be trainable actor parameters"
            )
        count = sum(parameter.numel() for _, parameter in trainable_actor)
        if count != self._adapter_contract["trainable_actor_parameter_count"]:
            raise ValueError("frozen LoRA trainable parameter count mismatch")
        critic_parameters = [
            parameter
            for parameter in self.alg.critic.parameters()
            if parameter.requires_grad
        ]
        if not critic_parameters:
            raise ValueError("frozen LoRA runner requires a trainable fresh critic")
        learning_rate = _require_learning_rate(
            self.cfg["algorithm"]["learning_rate"],
            "frozen LoRA learning rate",
        )
        old_optimizer = self.alg.optimizer
        defaults = dict(old_optimizer.defaults)
        defaults["lr"] = learning_rate
        self.alg.optimizer = type(old_optimizer)(
            [
                *(parameter for _, parameter in trainable_actor),
                *critic_parameters,
            ],
            **defaults,
        )
        self.alg.learning_rate = learning_rate
        self._trainable_actor_names = tuple(name for name, _ in trainable_actor)
        self._optimizer_parameter_ids = {
            id(parameter)
            for group in self.alg.optimizer.param_groups
            for parameter in group["params"]
        }
        actor.core.assert_frozen_platform_unchanged()
        runtime = {
            "schema_version": 1,
            "kind": "g1_true23_frozen_sonic_lora_runtime",
            "adapter_contract": self._adapter_contract,
            "adapter_contract_sha256": _contract_sha256(self._adapter_contract),
            "trainable_actor_parameters": list(self._trainable_actor_names),
            "learning_rate": learning_rate,
            "critic_fresh": True,
            "optimizer_fresh": True,
            "resume_checkpoint_loaded": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        try:
            log_dir = Path(args[2]).expanduser().resolve()
        except (IndexError, TypeError):
            log_dir = Path(getattr(self, "log_dir", ".")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        destination = log_dir / RUNTIME_FILENAME
        encoded = json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise ValueError("frozen LoRA runtime contract differs on resume")
        else:
            destination.write_text(encoded, encoding="utf-8")
        self._frozen_lora_runtime = runtime

    def _assert_boundary(self) -> None:
        actor = self.alg.get_policy()
        actor.core.assert_frozen_platform_unchanged()
        actual_names = tuple(
            name for name, parameter in actor.named_parameters() if parameter.requires_grad
        )
        if actual_names != self._trainable_actor_names:
            raise RuntimeError("frozen LoRA trainable parameter set changed")
        optimizer_ids = {
            id(parameter)
            for group in self.alg.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != self._optimizer_parameter_ids:
            raise RuntimeError("frozen LoRA optimizer parameter set changed")

    def _checkpoint_payload(self) -> dict[str, Any]:
        self._assert_boundary()
        actor = self.alg.get_policy()
        adapter_state = actor.core.lora_state_dict()
        critic_state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.alg.critic.state_dict().items()
        }
        merged_hash = actor.core.merged_true23_policy_sha256(
            actor.distribution.std_param
        )
        update_count = self._require_counter_coherence()
        return {
            CHECKPOINT_HEADER: _checkpoint_header(),
            "adapter_contract": copy.deepcopy(self._adapter_contract),
            "adapter_contract_sha256": _contract_sha256(self._adapter_contract),
            "adapter_state_dict": adapter_state,
            "adapter_state_sha256": _state_sha256(adapter_state),
            "critic_state_dict": critic_state,
            "critic_state_sha256": _state_sha256(critic_state),
            "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
            "update_count": update_count,
            "trainer_state": self._trainer_state(),
            "lineage": copy.deepcopy(self._training_lineage),
            "lineage_sha256": self._lineage_sha256,
            "merged_true23_policy_sha256": merged_hash,
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        if infos is not None:
            raise ValueError("stock RSL infos are forbidden in frozen LoRA checkpoints")
        self._require_checkpointable()
        output = Path(path).expanduser().resolve()
        payload = self._checkpoint_payload()
        _save_checkpoint_atomic(
            output,
            payload,
            expected_contract=self._adapter_contract,
            expected_lineage=self._training_lineage,
        )
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = payload["update_count"]

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        if load_cfg is not None or strict is not True:
            raise ValueError("frozen LoRA resume must be complete and strict")
        requested = Path(path).expanduser().resolve()
        current_count = self._require_counter_coherence()
        actor = self.alg.get_policy()
        adapter_before = actor.core.lora_state_dict()
        critic_before = copy.deepcopy(self.alg.critic.state_dict())
        optimizer_before = copy.deepcopy(self.alg.optimizer.state_dict())
        env_counter_before = _environment_common_step_counter(self.env)
        learning_rate_before = self.alg.learning_rate
        self._training_state_poisoned = True
        try:
            loaded = torch.load(
                requested,
                map_location=map_location or "cpu",
                weights_only=True,
            )
            checkpoint = _validate_checkpoint(
                loaded,
                expected_contract=self._adapter_contract,
                expected_lineage=self._training_lineage,
                minimum_update_count=current_count,
            )
            actor.core.load_lora_state_dict(
                checkpoint["adapter_state_dict"],
                strict=True,
            )
            self.alg.critic.load_state_dict(
                checkpoint["critic_state_dict"],
                strict=True,
            )
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer_state = checkpoint["trainer_state"]
            update_count = _require_nonnegative_integer(
                checkpoint["update_count"],
                "restored frozen LoRA update_count",
            )
            learning_rate = _require_learning_rate(
                trainer_state["algorithm_learning_rate"],
                "restored frozen LoRA learning rate",
            )
            self.completed_update_count = update_count
            self.current_learning_iteration = update_count
            self.alg.learning_rate = learning_rate
            _set_environment_common_step_counter(
                self.env,
                trainer_state["env_common_step_counter"],
            )
            merged_hash = actor.core.merged_true23_policy_sha256(
                actor.distribution.std_param
            )
            if merged_hash != checkpoint["merged_true23_policy_sha256"]:
                raise ValueError("restored frozen LoRA merged policy hash mismatch")
            self._assert_boundary()
            self._require_counter_coherence()
        except BaseException:
            try:
                actor.core.load_lora_state_dict(adapter_before, strict=True)
                self.alg.critic.load_state_dict(critic_before, strict=True)
                self.alg.optimizer.load_state_dict(optimizer_before)
                self.completed_update_count = current_count
                self.current_learning_iteration = current_count
                self.alg.learning_rate = learning_rate_before
                _set_environment_common_step_counter(self.env, env_counter_before)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "frozen LoRA restore rollback failed; discard runner"
                ) from rollback_error
            self._training_state_poisoned = False
            raise
        self._training_state_poisoned = False
        self._last_checkpoint_path = requested
        self._last_checkpoint_update_count = self.completed_update_count
        self._frozen_lora_runtime["resume_checkpoint_loaded"] = True
        return {
            "trainer_state": dict(checkpoint["trainer_state"]),
            "lineage_sha256": self._lineage_sha256,
            "adapter_state_sha256": checkpoint["adapter_state_sha256"],
            "merged_true23_policy_sha256": checkpoint[
                "merged_true23_policy_sha256"
            ],
        }

    def load_adapter_initialization(self, path: str) -> dict[str, Any]:
        """Start a new phase from an adapter while keeping fresh PPO state.

        This is intentionally distinct from exact resume.  It imports only the
        actor adapter selected by independent gates; critic, optimizer,
        environment curriculum, and update counters remain fresh and become
        part of the new phase lineage.
        """

        if self._require_counter_coherence() != 0:
            raise RuntimeError("adapter initialization requires a fresh runner")
        requested = Path(path).expanduser().resolve()
        actor = self.alg.get_policy()
        before = actor.core.lora_state_dict()
        try:
            loaded = torch.load(requested, map_location="cpu", weights_only=True)
            if not isinstance(loaded, Mapping):
                raise ValueError("adapter initialization checkpoint must be a mapping")
            if loaded.get(CHECKPOINT_HEADER) != _checkpoint_header():
                raise ValueError("adapter initialization checkpoint header mismatch")
            contract = loaded.get("adapter_contract")
            if not isinstance(contract, Mapping) or dict(contract) != dict(
                self._adapter_contract
            ):
                raise ValueError("adapter initialization contract mismatch")
            if loaded.get("adapter_contract_sha256") != _contract_sha256(contract):
                raise ValueError("adapter initialization contract hash mismatch")
            adapter = loaded.get("adapter_state_dict")
            if not isinstance(adapter, Mapping):
                raise ValueError("adapter initialization state is missing")
            if loaded.get("adapter_state_sha256") != _state_sha256(adapter):
                raise ValueError("adapter initialization state hash mismatch")
            prior_lineage = validate_mjlab_training_lineage(loaded.get("lineage"))
            if loaded.get("lineage_sha256") != prior_lineage["lineage_sha256"]:
                raise ValueError("adapter initialization lineage hash mismatch")
            actor.core.load_lora_state_dict(adapter, strict=True)
            merged_hash = actor.core.merged_true23_policy_sha256(
                actor.distribution.std_param
            )
            if merged_hash != loaded.get("merged_true23_policy_sha256"):
                raise ValueError("adapter initialization merged hash mismatch")
            self._assert_boundary()
        except BaseException:
            actor.core.load_lora_state_dict(before, strict=True)
            raise
        self._frozen_lora_runtime["adapter_initialization"] = {
            "path": str(requested),
            "adapter_state_sha256": loaded["adapter_state_sha256"],
            "prior_lineage_sha256": prior_lineage["lineage_sha256"],
            "critic_reused": False,
            "optimizer_reused": False,
            "counters_reused": False,
        }
        return copy.deepcopy(self._frozen_lora_runtime["adapter_initialization"])


__all__ = [
    "CHECKPOINT_HEADER",
    "FrozenPlatformLoraRunner",
    "RUNTIME_FILENAME",
    "load_frozen_platform_lora_checkpoint",
]
