# PICO and walk002 moving-state label coverage

The qualified saved trajectories provide **6,847 candidate moving controls** beyond the current 3,057 walk003 training rows. The existing 300-value BFM history and raw previous-action inputs are reconstructible byte for byte from the actual recorded states and targets. They do **not** yet provide complete residual-training datasets: per-row unclipped BFM baseline inference is still missing.

This assessment used saved arrays and the frozen pure NumPy observation module. It created no feature dataset or residual labels, constructed no model, performed no inference or physics, changed no shared source, and selected no fitting run. walk008 remained held out and was not read.

| Qualified stream | Acquisition controls | Source controls | Return controls | Candidate rows | Received state frames |
| --- | --- | --- | --- | ---: | --- |
| PICO | 250–349 | 350–6129 | 6130–6229 | 5,980 | 261–6240 |
| walk002 hybrid, recorded MPC portion | 250–349 | 350–1016 | 1017–1116 | 867 | 261–1127 |

Ranges are inclusive and derived from the original timelines. Initial controls 0–249, all original terminal controls, and both separate 250-control holds are excluded. The walk002 terminal BFM yaw4 portion starts at 1117 and supplies no candidate labels here. Its preceding MPC portion exactly matches the original whole-MPC run.

The new candidates contain 200 acquisition, 6,447 source, and 200 return rows. Combining them with the existing three walk003 branches would yield 9,904 rows: 500 acquisition, 8,904 source, and 500 return. This is coverage accounting, not a training selection.

## Verified state and history provenance

The check reconstructed every PICO precontrol history/prior pair (6,530 rows) and every walk002 pair through the first BFM boundary (1,118 rows). All float32 bytes match the saved observations. The walk002 prefix also matches the original MPC trajectory for all 11,170 native samples, actual commanded targets, torques, actuator forces, accumulated times, and warning ledgers. No dynamics were rerun for this comparison.

History reconstruction starts at control 0, before selecting moving rows. Both trajectories use MPC during their initial 250 controls. The prior action therefore follows the exact frozen expression `(actual_target - default_q) * kp / (0.25 * training_effort)`, cast to float32, without a ±5 clamp. Candidate PICO prior actions reach 10.2740898, with 810 components outside ±5; walk002 reaches 6.5243621, with 28 components outside ±5. Applying the BFM raw-action clamp would change these observations.

The history is the pre-update, newest-first four-lag structure, flattened in sorted field order: actions, angular velocity, joint position, joint velocity, and projected gravity. The frozen quaternion/gravity expression and signed zeros must be preserved. Named histories are reconstructed and hashed; the stored flat histories provide the direct byte-equality evidence.

Received goal state frames remain `control + 11`. The final moving-row prepared preview reaches PICO frame 6277 and walk002 frame 1164; their raw support bounds are 6278 and 1165. All lie within the original full references. No candidate moving-row goal requires EOF padding.

## Existing head and collector contract

The current ordinary final-65000 head is trained only on three walk003 branches, each with controls 250–1268: old nominal, first expert continuation, and BFM-entry250 expert. Their 1,019 rows each contain 100 acquisition, 819 source, and 100 return controls. The head ONNX SHA256 is `861b4c39349276851e23edab43978a74bab4cc613915da87805471c6164bc365`.

The 1,069 inputs remain 1,023 GoalFeatures plus 23 unclipped BFM-base-minus-default values and 23 raw previous-action values. GoalFeatures contain 79 proprioceptive values and eight 118-value received-goal slots at offsets `[0, 1, 2, 4, 8, 16, 24, 37]`. The 300-value BFM history influences baseline inference; it is not concatenated into the head. Native v4 references and immutable original29 goals must follow each actual clip.

A residual label is the **actual qualified native target minus the unclipped original BFM baseline target at the same actual precontrol state and history**, using the original horizon8/position1/yaw2 moving-goal configuration. Neither selected fresh-seed proposals nor normalized teacher actions are substitutes for this baseline. Terminal yaw4 is excluded.

Current normalization, native joint span, feature math, and the nine dataset-by-phase MSE means are pinned. The existing collector and compatibility code hardcode walk003 reference paths, a BFM initial prefix, terminal boundary 1269, 1,019 rows per branch, and specific root authorization receipts. They cannot be reused unchanged for these two clips.

## Required changes before any selected collection

1. Parameterize qualified clip, native/reference/original29 paths, timeline phases, source-frame mapping, and saved history/prior field aliases. Bind both root qualifications, their four independent reports, and the actual traces. Keep walk008 excluded.
2. Reconstruct the actual all-MPC prefix history from control 0 with byte-equality checks; then select the timeline-derived moving ranges above. Preserve no action clamp and the frozen float32 observation arithmetic.
3. Separately select and authorize 6,847 actual-state backward-plus-actor baseline evaluations. Neither trace saves the required unclipped baseline target/action for every candidate row. Inference alone needs no new physics.
4. Decide the integration-state schema explicitly. Both traces lack a complete 291-value precontrol integration snapshot for every candidate row, while the current collector exports that field. Either use a reviewed state/history-only label schema bound to the complete qualified source trace, or authorize an exact native replay to capture per-control 291-value states. Do not synthesize `qacc_warmstart` or `ctrl`, or label qpos/qvel alone as a complete integration state.
5. Generalize compatibility row counts and dataset/phase metadata. Check normalization coverage and cross-dataset exact/near-target conflicts after selected collection. Keep normalization and targets unchanged unless a later training decision explicitly changes them. Plain concatenation would heavily overweight PICO.
6. Preserve the existing local PICO tracking-failure annotations at source 67–69 s and 69–71 s despite the full-source aggregate pass. No filtering, relabeling, smoothing, or row deletion is selected.

The bounded next useful action, if selected, is an artifact-local generalized collector with byte-exact observation preflight and a reviewed integration-state schema. This report does not authorize that action or promise generalization or real-time performance.

## Evidence

- [Machine-readable assessment](assessment.json), SHA256 `bc701011d012a573a5df63e977124db928b8ee64c7920b5c01bdc65f4554b89a`.
- [Reproducible saved-array check](assess.py); 38 input hashes are recorded and were rechecked at completion. Runtime: pinned WSL Python, NumPy 1.26.4.
- PICO root qualification SHA256 `746502265523f527a7836f8b091ba521a56078629b1943468eff04f8bfa61daa`; main trace SHA256 `438e6420685666b104d9f944620bc9df6745763727848d8709fb223eb47f36cb`.
- walk002 hybrid root qualification SHA256 `a7eaec31dd572724921e2d2986ab8c6ab5db33008f6b3c1af5d2dbd72bf92e4b`; main trace SHA256 `a7e548b4a684d6c9755c590f1a7002844c4cd1e13450ed56181ee38126992284`.
- The first preparation attempt stopped at a source-count assertion before history processing. The arithmetic omitted the 100-control return ramp. Correcting it from `returned_standing_start - 350` to `returned_standing_start - 450` made the assertion match both immutable timelines; the complete read-only check then passed.
