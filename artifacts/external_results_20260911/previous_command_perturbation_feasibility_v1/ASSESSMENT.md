# Previous-command perturbation feasibility

Feasible, with one missing prerequisite: the old expert dataset has no complete integration snapshots. The query1 and query250 traces already contain all 1,020 required boundary states, controls 249–1268, with full 291-value integration state, precontrol history/prior, repeated time and warning ledgers. Query250's control249 full state/history/prior matches the qualified BFM250 prefix exactly, and its control250 full state matches that prefix's endpoint. Its first moving label therefore does not need to be dropped.

The old expert has every needed actual target, q/dq, history/prior and native sample, but q/dq cannot recover solver warm-start state. A separately authorized recorded-command reconstruction through control1267 costs 12,680 native steps and captures old precontrols249–1268. Capturing through control1268 costs 12,690 and also preserves boundary1269. No reconstruction ran in this assessment.

`report.json` and `comparisons.json` retain 39,226 passing saved-byte checks. These rebuild all 1,269 precontrol histories/priors for each dataset, check all 3,057 selected zero-perturbation priors and full committed teacher products, decode the available full-state q/dq/time/previous-command/force channels, and check warning ledgers. The old trace serializes warning counters as int64; query traces use int32. All selected ledgers are zero. A future native recorder must explicitly preserve the original serialization convention when comparing them.

The coherent branch at moving control c is:

1. Copy the exact full integration state and warning ledger at c−1 into a private native instance. Copy its precontrol history, previous action and independent accumulated time. Preserve all original native model/solver parameters.
2. Form a selected bounded perturbation of the actual native joint-position target at c−1, with native target clipping. For ten 2 ms steps, recompute the original PD command and native torque clipping. Apply the unchanged strict oracle after every step. This is target perturbation, not direct torque or measured-state substitution.
3. Shift history once using the original q/dq and incoming prior at c−1. The outgoing perturbed target becomes the separate prior at c through the original float32 inverse actual-target formula, without a ±5 clamp.
4. At the physically resulting c state, evaluate original yaw2 `_goal` at frame c+11, then original actor and 1,069-feature builder. Recompute the backward map for each probe: changed root position/yaw generally invalidates the original cached latent.
5. Evaluate the original committed teacher at c: complete tangent difference, full K matrix product, correction clipped to ±0.1 rad, original planned target addition, and native target clipping. The residual label is that native teacher target minus the unclipped BFM base.

The history timing matters: **H_c stays exactly nominal for this single outgoing-command perturbation.** Its newest action entry is the prior entering c−1, representing command c−2. The altered command c−1 appears separately as `last_action` at c and enters history only on the next shift. Replacing the newest history action with the perturbed command at c would introduce a one-control timing error. The zero-perturbation inverse target formula reproduces every saved center prior byte for byte, including query250's raw-BFM control249 boundary.

Before any probe at a row, one nominal ten-step replay must recover every saved q/dq/command/actual-force/time/warning sample and the complete full291 state at c. The nominal BFM/base/features and teacher target must also match the selected center. Keep first mismatches and stop before that row's probes. Each probe restarts from its own copy of the predecessor state; probes are never chained. Strict failures remain explicit requested probes with partial traces, not silently removed examples or opportunities for adaptive retries.

The following counts assume one nominal replay and BFM check per center. Direction sets and amplitudes remain unselected.

| Directions per center | Perturbed rows | Native steps including nominal checks | Actor calls | Backward calls |
|---:|---:|---:|---:|---:|
| 2 | 6,114 | 91,710 | 9,171 | 9,171 |
| 23 | 70,311 | 733,680 | 73,368 | 73,368 |
| 46 (23 signed axis pairs) | 140,622 | 1,436,790 | 143,679 | 143,679 |

The missing old-prefix reconstruction is additional. The earlier velocity dataset used 143,679 actor calls in 696.5 seconds and 3,057 backward calls in 0.458 seconds, with 820.2 seconds total generation time. A full signed-axis grid here has the same actor count, many more goal/backward evaluations, 1.44 million native steps and larger trace storage. The observed actor time alone is about 11.6 minutes; no native-throughput benchmark or wall-time guarantee was performed. Retaining every branch's full native samples, features, history/prior and final integration state would require roughly 2–3 GB before compression, plus metadata and source snapshots.

There are unavoidable limits. The three datasets contain 612 center controls at original replan boundaries; using their saved K/plan supplies a frozen local teacher map, not a newly optimized response to the changed state. A full signed-axis grid would contain at least 2,411 zero-displacement outward probes because the predecessor target is already exactly at a native bound. Preserve actual displacement, clipping masks and duplicates. A small target displacement does not guarantee safe contact or velocity behavior, so strict per-step rejection remains necessary. One 20 ms branch does not establish multi-step rollout, source tracking or quiet standing.

The labels would follow the existing teacher's actual-target prior convention. The unchanged student's raw combined-action prior can still disagree after deployment target clipping; adding these labels alone does not repair that controller-history mismatch. No amplitude, probe direction set, new generation, inference, physics, fit or canonical trial is selected by this report. Held-out walk008 and the broader PICO/walk002 rows are excluded from this assessment's 3×1,019 centers.
