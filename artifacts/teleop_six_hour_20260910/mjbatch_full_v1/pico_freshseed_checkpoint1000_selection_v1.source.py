"""Read-only audit of declared online candidate selection and accepted cost."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import sha256


def run(args):
    with np.load(args.trace, allow_pickle=False) as trace:
        count = len(trace["target"])
        if "checkpoint_metadata" in trace:
            metadata = json.loads(str(trace["checkpoint_metadata"]))
            plans = metadata["plans"]
        else:
            plans = json.loads((args.run / "plans.json").read_text())
    request = json.loads((args.run / "request.json").read_text())
    names = {"shifted_mpc": "warm_cost", "recorded_bfm": "recorded_cost", "fresh_bfm": "fresh_cost"}
    counts = dict.fromkeys(names, 0)
    witnesses = []
    rejected = 0
    maximum_selection_excess = maximum_accepted_excess = 0.0
    for plan in plans:
        selection = plan["seed_selection"]
        available = {key: selection.get(field) for key, field in names.items()}
        available = {key: value for key, value in available.items() if value is not None and np.isfinite(value)}
        selected = selection["selected"]
        assert selected in available, "selected seed has no finite declared actual-rollout cost"
        selected_cost = available[selected]
        tolerance = 1e-10 * max(1.0, abs(selected_cost))
        selection_excess = selected_cost - min(available.values())
        accepted_excess = plan["cost"] - selected_cost
        assert selection_excess <= tolerance, (plan["control"], available, selected)
        assert accepted_excess <= tolerance, (plan["control"], plan["cost"], selected_cost)
        counts[selected] += 1
        maximum_selection_excess = max(maximum_selection_excess, selection_excess)
        maximum_accepted_excess = max(maximum_accepted_excess, accepted_excess)
        if selection.get("fresh_rejected"):
            rejected += 1
            assert selected != "fresh_bfm" and selection.get("fresh_cost") is None
        if selected == "fresh_bfm":
            diagnostics = selection["fresh_diagnostics"]
            assert not diagnostics["actual_history_mutated_by_proposal"]
            assert not diagnostics["real_physics_state_mutated"]
            assert not any(diagnostics["private_engine_warning_counts"])
            witnesses.append(dict(control=plan["control"], costs=available, accepted_cost=plan["cost"]))
    report = dict(
        kind="declared_candidate_cost_selection_audit", controls=count, plans=len(plans),
        source_controls=max(0, min(count, request["source_phase"]["control_stop"])
                            - request["source_phase"]["control_start"]),
        selected_counts=counts, private_candidate_rejections=rejected,
        maximum_selection_cost_excess=maximum_selection_excess,
        maximum_accepted_cost_excess=maximum_accepted_excess,
        all_selections_are_lowest_finite_declared_cost=True,
        all_accepted_costs_do_not_increase=True, fresh_selection_witnesses=witnesses,
        request_sha256=sha256(args.run / "request.json"), trace_sha256=sha256(args.trace),
        source_sha256=sha256(__file__),
        scope="Recorded online candidate costs; no reconstruction of unstored full counterfactual states",
        full_body_tracking_qualified=False,
    )
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key != "fresh_selection_witnesses"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
