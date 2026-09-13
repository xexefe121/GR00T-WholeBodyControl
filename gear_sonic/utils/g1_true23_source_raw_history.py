"""SIM-only raw requested-action history counterfactual, not a live profile.

The current training interface encodes bounded target equivalents in history.
Original SONIC's CreatePolicyCommand instead stores the raw network output.
Changing this observation does NOT change targets, gains, effort or joint bounds.
It is deliberately a distribution change for already-trained native23 actors;
no checkpoint, exporter or hardware profile may be relabelled as compatible.
"""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29

HISTORY_START = 610
HISTORY_STOP = 900
RETAINED = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)


def source_raw_history_contract():
    return dict(
        kind="native23_source_raw_previous_action_counterfactual_v1",
        previous_action="raw23 network requests before target projection and range preview",
        ordering="ten chronological frames, zero pre-inference history, retained canonical IL29 slots",
        absent_six_history_slots="unchanged zero placeholders, not fabricated measurements",
        upstream_source="CreatePolicyCommand assigns last_action[i] from floatarr[i]",
        current_native23_training_history="projected physical target in source units",
        training_runtime_distribution_changed=True,
        checkpoint_compatibility_or_full_upstream_equivalence_claimed=False,
        target_transform_gains_effort_physical_limits_changed=False,
        live_export_or_transport_supported=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def replace_previous_action_history(history930, previous_raw23):
    """Replace only retained action slots; no input mutation or hidden state.

    ``previous_raw23`` contains at most the last ten chronological requests,
    excluding the action about to be inferred. Earlier missing rows are zero.
    All other measured/absent slots remain byte-identical.
    """
    history = np.asarray(history930)
    raw = np.asarray(previous_raw23)
    if history.shape != (930,) or history.dtype != np.float32 or not np.isfinite(history).all():
        raise ValueError("raw-history diagnostic requires finite float32 history930")
    if raw.ndim != 2 or raw.shape[1] != 23 or len(raw) > 10 or raw.dtype != np.float32:
        raise ValueError("raw-history diagnostic requires float32 [0..10,23] previous requests")
    if not np.isfinite(raw).all() or (len(raw) and np.max(np.abs(raw)) >= 10):
        raise ValueError("raw-history diagnostic requires finite requests within existing raw bound")
    result = history.copy()
    block = result[HISTORY_START:HISTORY_STOP].reshape(10, 29)
    block[:, RETAINED] = 0
    if len(raw):
        block[10 - len(raw) :, RETAINED] = raw
    return result
