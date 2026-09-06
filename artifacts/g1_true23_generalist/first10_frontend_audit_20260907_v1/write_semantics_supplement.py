"""Append explanatory metadata without overwriting completed experiment receipts."""

import hashlib
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_generalist_benchmark import observation_phase_contract


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


folder = Path(__file__).resolve().parent
repo = folder.parents[2]
relative_reports = [
    "artifacts/g1_true23_generalist/historical_walk_reproduction_20260907_v1/report.json",
    "artifacts/g1_true23_generalist/lifecycle_nominal_20260907_v2/report.json",
]
reports = {}
for relative in relative_reports:
    path = repo / relative
    reports[relative] = dict(sha256=sha(path))
v2 = json.loads((repo / relative_reports[1]).read_text())
candidate = next(row for row in v2["records"] if row["policy"] == "generalist_candidate_000")
receipt = dict(
    kind="historical_reset_and_observation_phase_clarification_v1",
    original_receipts_unchanged=True,
    no_simulation_or_policy_execution_in_supplement=True,
    bound_original_reports=reports,
    observation_phase_contract=observation_phase_contract(),
    original_comparison_reproduced_scope="reference_reset_boundary_only",
    historical_released_gains_scope="gain_arrays_only_not_original_stale_cvel_frontend",
    current_nominal_frontend_changed_to_historical_stale_cvel=False,
    exact_original_first10_capture="original actual library frontend matches archived first11 qpos exactly; current shared frontend matches current actual library first10 controls exactly",
    diagnostic_receipt=dict(path=str(folder / "comparison.json"), sha256=sha(folder / "comparison.json")),
    fresh_v2_policy_identity=candidate["policy_identity"],
    fresh_v2_observed_results=dict(
        completed_controls=candidate["result"]["completed_controls"],
        source_motion_tracking=candidate["lifecycle"]["source_motion_tracking"],
        final_proof_root_speed_max_m_s=candidate["lifecycle"]["final_proof_root_speed_max_m_s"],
        final_proof_standing_joint_error_max_rad=candidate["lifecycle"]["final_proof_standing_joint_error_max_rad"],
    ),
    reference_scope_note="Lifecycle source is existing native23 fit to recorded original29 policy motion, not newly accepted planned-choreography retarget. Completion is duration only; fidelity and standing/contact qualification remain false.",
    inputs={str(Path(__file__)): sha(Path(__file__))},
    hardware_authorized=False,
    deployment_ready=False,
    simulator_qualified=False,
)
output = folder / "report_semantics_supplement.json"
with output.open("x") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps(dict(path=str(output), sha256=sha(output))))
