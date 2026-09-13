# Actual-state expert label extraction plan

Status: preparation only. No extraction, inference, fitting, new query, or simulation launched by this plan. The single actual-state oracle remains in progress. All new labels remain inadmissible until parent independently qualifies its entire remaining source, return, original terminal hold, and separate five-second hold.

## Bound inputs and row ownership

Use final `student_actual_oracle_control1_v1/nominal/trace.npz`, its frozen v2 request/source receipt and selection receipt, plus final independent physical, original-intent, and quiet reports. Freeze their final hashes in a new extraction receipt; never read a changing partial trace. Original immutable dataset is `fast_controller_nominal_pilot_v1/labels/labels.npz` (1269 rows, SHA256 `8003407282b01f2c66fe1720d7eb038f43377c1d6eab80dce47f1f75911e8083`).

- Global control 0 is the actual final-20000 STUDENT prefix. Retain it as provenance/history only; never add its command as an expert label. Existing original-teacher control-0 label remains unchanged.
- New MPC expert labels are exactly controls 1 through 1268 inclusive: 1268 rows, including entry, acquisition, all 819 source controls, and return. Require controller mode 1, ten executed physical substeps, and frame `control + 11` for every row.
- Controls 1269 through 1568 and separate 1569 through 1818 are terminal yaw-4 BFM. Preserve their qualification evidence separately; do not train the residual head on them. Existing runtime continues to bypass the head there.

No old teacher state, saved feedback-gain extrapolation, optimized nominal state, or unused candidate target is a new label. Labels are only ACTUALLY APPLIED MPC joint targets on the certified branch's measured states.

## Minimal new collector

Create one artifact-local `collect_actual_branch_labels.py`, adapting the frozen `collect_nominal_labels.py`; reuse byte-identical `LinearFeatures`, `infer_base`, `GoalFeatures`, BFM observation/history helpers and the frozen ONNX models. Native model loading and FK are permitted; no `mj_step`, optimizer, or physical rollout occurs during collection. Use ORT/BLAS one thread.

For each row, copy saved precontrol qpos/qvel, named four-lag arrays, flat history and preceding action. Independently rebuild the complete buffer from the real control-0 prefix and verify every saved lag/action exactly. `BFMHistory.before_update` returns the PRE-UPDATE buffer; assert it equals `control_history_before` before shifting in the current sensed terms.

At control 1, preserve the final student's combined **preclip** prior action exactly. Do not replace it with normalized clipped target. At controls 2 onward, preceding action is the recorded normalization of the prior actually applied MPC target:

`((target - default_q) * kp / (.25 * training_effort)).astype(float32)`.

No silent clipping to +/-5. Report maxima/exceedance counts. The future student runtime remains unchanged: its own preceding action is the combined preclip BFM-plus-residual action, with raw actor history when residual is zero. This teacher/student history distinction remains explicit.

Evaluate frozen original-native BFM goals (horizon 8, position gain 1, yaw gain 2) on that EXACT measured state/history. Preserve arithmetic `default_q + raw * .25 * training_effort / kp`. Features remain 1069 values: existing 1023 goal/proprio features, 23 unclipped BFM target offsets, 23 preceding actions. Goals use immutable v4 native + original29 task inputs; offsets `[0,1,2,4,8,16,24,37]`, declared .74-second goal buffer and .76-second raw pose support. No clip ID, control index, phase ID, or time input enters the network.

Save `expert_target = actual target`, `residual_rad = expert_target - base_target`, measured states, base action/target, all histories and control/frame metadata in a SEPARATE new NPZ matching the existing dataset schema. Existing normalization and native joint spans are recorded for comparison, not refitted during extraction. Require finite arrays, exact row/counter/history checks, unchanged native bounds, and reconstruction of each applied target from base plus residual (report floating-point delta; no false bit-exact promise after subtraction/addition).

## Cross-dataset compatibility audit before any fit decision

Use the original saved feature mean/std (floor .05) for all distance calculations. Compare the 1268 new rows against all 1269 original rows, and report:

1. Exact float32 feature duplicates and target/residual spread. Different targets at identical features are actual deterministic label conflicts; distinct nearby inputs are sensitivity evidence only.
2. Cross-dataset nearest neighbors in normalized feature RMS/L2; also exclude same-time +/-4 controls to distinguish local phase matches from repeated similar states. Report distance and target/residual RMS/max differences, named worst joints, closest pairs and largest sensitivity ratios. Never divide a zero distance silently.
3. Same-global-control comparisons, grouped by entry/acquisition/source/return: measured q/dq, history, preceding action, goal/proprio, BFM base, target and residual differences. Highlight control 1 versus the original teacher and the final student's saved precontrol-1 features. That student's feature/base parity at the query input must hold under the same frozen inference/runtime arithmetic, within explicitly measured numerical tolerance.
4. Target slew, normalized action range and per-joint residual/native-span distributions for old versus new data. No arbitrary command canonicalization, averaging, conflict deletion, or target smoothing.

Save complete pair indices/distances and raw discrepancies, not only summaries. Estimated work: 1268 frozen BFM forward calls plus about 1.61 million feature-distance comparisons, bounded CPU/ORT one thread; minutes, no MPC load.

Deliver hashes, exact row provenance, and compatibility findings to parent. Parent chooses whether a single DAgger fit is justified. This plan does not authorize fitting, model selection, a second query, or changed controller semantics.
