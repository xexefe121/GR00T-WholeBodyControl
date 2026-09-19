"""Recorder metadata adapter; leaves the trained, hash-bound reader unchanged."""

from copy import deepcopy
from pathlib import Path

from gear_sonic.utils import g1_true23_failure_sampling_checkpoint as checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def recorder_identity(identity, semantics):
    """Expose only geometry compatibility already verified by the strict reader."""
    if (
        identity.get("kind") != "native23_failure_sampling_cpu_research_reader_v1"
        or identity.get("deployment_ready") is not False
        or identity.get("hardware_authorized") is not False
    ):
        raise ValueError("requires the SIM-only failure-sampling reader")
    compatibility = semantics["compatibility"]
    for key in ("source_geometry_sha256", "native_geometry_sha256"):
        value = compatibility.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("verified geometry identity missing: " + key)
    if "release_compatibility" in identity and identity["release_compatibility"] != compatibility:
        raise ValueError("conflicting existing geometry identity")
    return dict(identity, release_compatibility=deepcopy(compatibility))


def load_cpu_actor(*args, **kwargs):
    actor, identity, semantics = checkpoint.load_cpu_actor(*args, **kwargs)
    identity = recorder_identity(identity, semantics)
    path = Path(__file__).resolve(strict=True)
    semantics["reverified_repository_sources"][str(path)] = sha256_file(path)
    return actor, identity, semantics
