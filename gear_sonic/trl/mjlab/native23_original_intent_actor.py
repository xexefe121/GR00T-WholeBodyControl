"""Same native23 network, explicitly different original-source input contract."""

import copy

from gear_sonic.trl.mjlab.native23_root_feedback_actor import True23RootFeedbackActorModel
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest
from gear_sonic.utils.g1_true23_hand_frame_tasks import HAND_FRAME_CONVENTION
from gear_sonic.utils.g1_true23_original29_reference import REFERENCE_KIND
from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION

ACTOR_KIND = "g1_native23_root_feedback_original_source_intent_actor_v1"
COMPATIBILITY_KIND = "native23_original_source_intent_release_compatibility_v4"


def original_intent_compatibility(source_geometry_sha256, native_geometry_sha256):
    if (
        not isinstance(native_geometry_sha256, str)
        or len(native_geometry_sha256) != 64
        or any(c not in "0123456789abcdef" for c in native_geometry_sha256)
    ):
        raise ValueError("original-intent actor requires pinned native23 geometry")
    result = release_compatibility_contract(source_geometry_sha256, SOURCE_ACTION_CONVENTION, BUFFERED_TIMING)
    result.pop("contract_sha256")
    result.update(
        kind=COMPATIBILITY_KIND,
        reference_geometry=REFERENCE_KIND,
        native_geometry_sha256=native_geometry_sha256,
        native_hand_frame_convention=HAND_FRAME_CONVENTION,
        evaluation_target_geometry="native23_lower_body_and_original29_hand_head_tasks",
        source_missing_axes_preserved_for_VR_not_physical_feedback=True,
        old_zero_absent_checkpoint_relabelling_allowed=False,
    )
    return {**result, "contract_sha256": canonical_digest(result)}


def validate_original_intent_compatibility(value):
    if not isinstance(value, dict) or value != original_intent_compatibility(
        value.get("source_geometry_sha256"), value.get("native_geometry_sha256")
    ):
        raise ValueError("original-intent actor compatibility mismatch")
    return value


class True23OriginalIntentActorModel(True23RootFeedbackActorModel):
    def __init__(self, *args, release_compatibility, **kwargs):
        self._original_intent_compatibility = copy.deepcopy(
            validate_original_intent_compatibility(release_compatibility)
        )
        # Reuse identical construction/action/history machinery. Its legacy VR
        # label is never exported as the new actor's actual reference contract.
        construction = release_compatibility_contract(
            release_compatibility["source_geometry_sha256"], SOURCE_ACTION_CONVENTION, BUFFERED_TIMING
        )
        super().__init__(*args, release_compatibility=construction, **kwargs)
        self.release_compatibility = self._original_intent_compatibility

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": ACTOR_KIND,
            "release_compatibility": copy.deepcopy(self._original_intent_compatibility),
            "legacy_root_feedback_export_acceptance_claimed": False,
        }
