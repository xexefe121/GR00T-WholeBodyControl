# One cost-ranked native forecast experiment

Selected by root on 2026-09-11 after the fixed 132-forecast study. Simulation only.

Keep the frozen update-65,000 learner, original BFM, native model, prepared v4 reference, original 250-control BFM entry, full lifecycle and separate hold request. No new labels, fitting, solver optimization, reference changes or shortened source.

Where the prior adapter used first-feasible admission, evaluate all four original candidates: primary, original BFM, previous applied target, current joint position. Every candidate uses the same constant target for five controls / 100 ms. Preserve unchanged hard native feasibility predicates and all candidate outcomes. Infeasible or interrupted forecasts are ineligible; their partial costs cannot compete with complete horizons.

Among feasible candidates, choose the smallest exact original qualified query250 MPC state-plus-input cost over the five-control prefix. Window starts at control c+10; initial state knot is c+10, post-control knots c+11 through c+15, and input penalties c+11 through c+15. No terminal multiplier, horizon rescaling or new objective weights. Exact ties retain original candidate order. This truncated objective is not the full H30 MPC cost. The history records the actual applied command using the existing filtered adapter convention.

Save every candidate cost, complete forecast or first failure, selected index and actual substeps. Retain the complete current state/history on rejection or exception. Freeze source, inputs and request before one durable canonical run; source review and saved-case cost parity must pass first. No new connected trial before that run. Separate hold runs only after complete lifecycle success under the unchanged gates.

Evidence: `filtered_all_candidates132_v1/results/report.json`, SHA256 `58d2ea9a9e5fc569667a540ec9667cfcf9a6d393a5847a34bc56da1d1ec0cf0c`. All four candidates were feasible at saved states 250 through 267, but primary cost exceeded BFM at every state. The first-feasible controller did not intervene until 268. This motivates earlier tracking-aware selection; it does not prove connected recovery. Constant-target forecasts may prefer standing, and evaluating four candidates may exceed the 20 ms deadline. This experiment isolates controller behavior before runtime optimization.
