import hashlib
import json
from pathlib import Path

import pytest

from gear_sonic.scripts import train_g1_23dof_mjlab_frozen_lora as launcher


def _materials(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    motion = tmp_path / "corpus.npz"
    metadata = tmp_path / "corpus.json"
    spans = tmp_path / "corpus.spans.json"
    adapter = tmp_path / "frozen_lora_model_100.pt"
    resume = tmp_path / "frozen_lora_model_200.pt"
    for path in (motion, metadata, adapter, resume):
        path.write_bytes(b"fixture")
    spans.write_text(
        json.dumps(
            {
                "kind": "g1_true23_motion_corpus_spans_v1",
                "clip_count": 1,
                "total_frames": 20,
                "spans": [{"name": "clip", "start": 0, "length": 20}],
            }
        ),
        encoding="utf-8",
    )
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "kind": "g1_true23_frozen_lora_behavior_bank_v1",
                "span_sidecar_sha256": hashlib.sha256(spans.read_bytes()).hexdigest(),
                "feasibility_screen_passed": True,
                "override_and_dedup_complete": True,
                "clip_indices": [0],
            }
        ),
        encoding="utf-8",
    )
    return motion, metadata, spans, bank, adapter, resume


def _arguments(
    materials: tuple[Path, Path, Path, Path, Path, Path],
) -> list[str]:
    motion, metadata, spans, bank, adapter, _resume = materials
    return [
        "train",
        "--phase",
        "polish",
        "--adapter-init",
        str(adapter),
        "--behavior-bank",
        str(bank),
        "--motion-file",
        str(motion),
        "--motion-metadata",
        str(metadata),
        "--spans",
        str(spans),
    ]


def test_first_polish_run_imports_selected_adapter(monkeypatch, tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    captured: dict = {}
    delegated: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_install_frozen_lora_hooks",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        launcher.base,
        "main",
        lambda values: delegated.extend(values) or 0,
    )

    assert launcher.main(_arguments(materials)) == 0
    assert captured["adapter_initialization_mode"] is True
    assert "--resume" in delegated
    assert delegated[delegated.index("--resume") + 1] == str(materials[4].resolve())
    assert "--spans" not in delegated


def test_polish_resume_keeps_adapter_provenance_but_restores_full_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    materials = _materials(tmp_path)
    captured: dict = {}
    delegated: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_install_frozen_lora_hooks",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        launcher.base,
        "main",
        lambda values: delegated.extend(values) or 0,
    )
    arguments = _arguments(materials) + ["--resume", str(materials[5])]

    assert launcher.main(arguments) == 0
    assert captured["adapter_initialization_mode"] is False
    assert captured["adapter_initialization"] == materials[4].resolve()
    assert delegated[delegated.index("--resume") + 1] == str(materials[5])


@pytest.mark.parametrize("name", ["stage_one_cpp", "native_support_projected", "native_support_stateful_v2"])
def test_optional_actuation_profile_is_consumed_and_bound(monkeypatch, tmp_path, name):
    captured = {}
    delegated = []
    monkeypatch.setattr(launcher, "_install_frozen_lora_hooks", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(launcher.base, "main", lambda values: delegated.extend(values) or 0)
    motion, metadata, spans, _, _, _ = _materials(tmp_path)
    args = [
        "smoke",
        "--motion-file",
        str(motion),
        "--motion-metadata",
        str(metadata),
        "--spans",
        str(spans),
        "--actuation-profile",
        name,
    ]
    assert launcher.main(args) == 0
    expected = (
        launcher.StageOneActuationProfile.from_cpp(launcher.REPO_ROOT / launcher.HEADER)
        if name == "stage_one_cpp"
        else launcher.NativeSupportActuationProfile.from_sim_config(launcher.REPO_ROOT / launcher.SIM_CONFIG)
    )
    if name == "native_support_stateful_v2":
        from dataclasses import replace

        expected = replace(expected, consistent_controller_state=True)
    assert captured["actuation_profile"] == expected
    assert "--actuation-profile" not in delegated


def test_old_behavior_bank_cannot_claim_feasibility_under_new_actuation(tmp_path):
    args = _arguments(_materials(tmp_path)) + ["--actuation-profile", "stage_one_cpp"]
    with pytest.raises(SystemExit, match="not qualified against this actuation profile"):
        launcher.main(args)


def test_native_source_hash_alone_does_not_bind_controller_state(tmp_path):
    materials = _materials(tmp_path)
    bank = materials[3]
    payload = json.loads(bank.read_text())
    profile = launcher.NativeSupportActuationProfile.from_sim_config(launcher.REPO_ROOT / launcher.SIM_CONFIG)
    payload["actuation_profile_source_sha256"] = profile.source_sha256
    payload["actuation_profile_contract"] = profile.contract()
    bank.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="exact native-support controller-state contract"):
        launcher.main(_arguments(materials) + ["--actuation-profile", "native_support_stateful_v2"])
