"""The recorder's required geometry fields come from validated semantics only."""

from copy import deepcopy

import pytest

from gear_sonic.utils.g1_true23_failure_sampling_evaluation import recorder_identity


def values():
    return (
        dict(
            kind="native23_failure_sampling_cpu_research_reader_v1",
            checkpoint_sha256="c" * 64,
            deployment_ready=False,
            hardware_authorized=False,
        ),
        dict(compatibility=dict(source_geometry_sha256="a" * 64, native_geometry_sha256="b" * 64)),
    )


def test_recorder_fields_derived_without_mutating_verified_inputs():
    identity, semantics = values()
    before = deepcopy((identity, semantics))
    result = recorder_identity(identity, semantics)
    assert (identity, semantics) == before
    assert result["release_compatibility"] == semantics["compatibility"]
    assert result["checkpoint_sha256"] == identity["checkpoint_sha256"]
    result["release_compatibility"]["source_geometry_sha256"] = "d" * 64
    assert semantics == before[1]


@pytest.mark.parametrize("field", ["kind", "deployment_ready", "hardware_authorized"])
def test_no_wrong_reader_or_authorization_promotion(field):
    identity, semantics = values()
    identity[field] = True if field != "kind" else "old_reader"
    with pytest.raises(ValueError):
        recorder_identity(identity, semantics)


@pytest.mark.parametrize("field", ["source_geometry_sha256", "native_geometry_sha256"])
def test_no_unbound_geometry(field):
    identity, semantics = values()
    semantics["compatibility"][field] = "not-a-hash"
    with pytest.raises(ValueError):
        recorder_identity(identity, semantics)


def test_conflicting_metadata_rejected():
    identity, semantics = values()
    identity["release_compatibility"] = {}
    with pytest.raises(ValueError):
        recorder_identity(identity, semantics)
