"""Standing-only output anchors for motion PPO; never a full-motion teacher.

Use the original three validated training episodes as input states. The
anchor labels are the checked standing adapter's own outputs, not failed
motion actions. The fourth episode is hash-bound but never loaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.scripts.fit_g1_true23_standing_lora import standing_loss, validate_standing_labels
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_torch
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256


def training_records(report):
    if (
        report.get("kind") != "g1_true23_representable_standing_teacher_comparison_v1"
        or report.get("held_out_episode_separate_from_training") is not True
        or any(
            report.get(key) is not False
            for key in ("hardware_authorized", "deployment_ready", "full_motion_teacher_accepted")
        )
    ):
        raise ValueError("retention requires unqualified, explicitly standing-only teacher evidence")
    rows = report["records"]
    train = [row for row in rows if row.get("role") == "standing_train"]
    held = [row for row in rows if row.get("role") == "held_out_episode"]
    if [row["name"] for row in train] != ["bounded_nominal", "bounded_plus", "bounded_minus"] or [
        row["name"] for row in held
    ] != ["bounded_holdout"]:
        raise ValueError("retention training/held-out episode partition changed")
    paths = [str(Path(row["arrays"]).resolve()) for row in [*train, *held]]
    if len(set(paths)) != 4:
        raise ValueError("retention training and held-out paths must be disjoint")
    for row in train:
        if any(
            row["result"].get(key) is not True
            for key in ("standing_only_labels_usable", "original_request_projected_before_actuation")
        ):
            raise ValueError("retention input episode did not pass bounded standing")
    return train


def load_training_states(teacher_path, initialization):
    teacher_path = Path(teacher_path).expanduser()
    if teacher_path.is_symlink():
        raise ValueError("retention teacher report may not be a symlink")
    teacher_path = teacher_path.resolve(strict=True)
    fit_path = Path(initialization["report_path"])
    if file_sha256(fit_path) != initialization["report_sha256"]:
        raise ValueError("retention standing initialization report changed")
    fit = json.loads(fit_path.read_text())
    digest = file_sha256(teacher_path)
    if fit["inputs"].get(str(teacher_path)) != digest:
        raise ValueError("retention teacher was not bound by the checked standing fit")
    report = json.loads(teacher_path.read_text())
    selected = training_records(report)
    for path, expected in report["inputs"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"retention teacher input changed: {path}")
    arrays, episodes = [], []
    for row in selected:
        path = Path(row["arrays"]).resolve(strict=True)
        actual = file_sha256(path)
        if report["inputs"].get(str(path)) != actual:
            raise ValueError("retention training trace must be directly hash-bound")
        with np.load(path, allow_pickle=False) as archive:
            loaded = {key: archive[key] for key in archive.files}
        validate_standing_labels(loaded)
        arrays.append({key: loaded[key].astype(np.float32) for key in ("encoder267", "history930")})
        episodes.append(dict(name=row["name"], path=str(path), sha256=actual, rows=500))
    data = {key: np.concatenate([row[key] for row in arrays]) for key in ("encoder267", "history930")}
    descriptor = dict(
        kind="g1_true23_standing_only_retention_inputs_v1",
        teacher_report_path=str(teacher_path),
        teacher_report_sha256=digest,
        episodes=episodes,
        train_rows=1500,
        held_out_rows_loaded=0,
        full_motion_teacher_accepted=False,
        failed_motion_actions_used=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return data, descriptor


def batch_indices(step, batch_size, count, seed):
    if any(type(value) is not int for value in (step, batch_size, count, seed)):
        raise ValueError("retention sampling requires integer counters")
    if step < 0 or not 1 <= batch_size <= count or not 0 <= seed < 2**63:
        raise ValueError("retention sampling counters out of range")
    generator = torch.Generator(device="cpu").manual_seed((seed + step) % (2**63))
    return torch.randint(count, (batch_size,), generator=generator)


class StandingOutputAnchor:
    def __init__(self, core, data, payload, *, batch_size, seed):
        if core.adapter_contract() != payload["adapter_contract"]:
            raise ValueError("standing retention frozen platform differs from initialization")
        device = next(core.parameters()).device
        self.batch_size, self.seed = batch_size, seed
        self.count = len(data["encoder267"])
        batch_indices(0, batch_size, self.count, seed)
        before = core.lora_state_dict()
        try:
            core.load_lora_state_dict(payload["adapter_state_dict"], strict=True)
            with torch.no_grad():
                semantic = torch.from_numpy(data["encoder267"]).to(device)
                history = torch.from_numpy(data["history930"]).to(device)
                core.codec.validate_padded_proprioception(history)
                tokens = torch.cat([core.encode(chunk) for chunk in semantic.split(128)])
                self.inputs = torch.cat((tokens, core.codec.encode_proprioception(history)), dim=-1).detach()
                self.raw = torch.cat(
                    [core.codec.decode_action(core.decoder(chunk)) for chunk in self.inputs.split(128)]
                ).detach()
                self.target = safe_target_transform_torch(self.raw)[1].detach()
            if not all(torch.isfinite(value).all() for value in (self.inputs, self.raw, self.target)):
                raise ValueError("standing retention anchor contains nonfinite values")
            core.assert_frozen_platform_unchanged()
        finally:
            core.load_lora_state_dict(before, strict=True)
        self.core = core
        self.cache_sha256 = _state_sha256(dict(input=self.inputs, raw=self.raw, target=self.target))

    def loss(self, step):
        indices = batch_indices(step, self.batch_size, self.count, self.seed).to(self.inputs.device)
        raw = self.core.codec.decode_action(self.core.decoder(self.inputs[indices]))
        return standing_loss(raw, self.raw[indices], self.target[indices])

    def metrics(self):
        with torch.no_grad():
            raw = torch.cat(
                [self.core.codec.decode_action(self.core.decoder(chunk)) for chunk in self.inputs.split(128)]
            )
            target = safe_target_transform_torch(raw)[1]
            return dict(
                target_rmse_rad=float((target - self.target).square().mean().sqrt()),
                raw_rmse=float((raw - self.raw).square().mean().sqrt()),
                maximum_target_drift_rad=float((target - self.target).abs().max()),
            )
