# Conditional collector plan: actual expert transition after BFM standing

This is a source/array inspection plan only. No collector, inference, optimizer or new physics has run for these labels. The single expert branch remains in progress at `bfm_entry250_actual_oracle_v1`.

Collect exactly **1,019 actual MPC commands**, global controls **250 through1268 inclusive**, after root explicitly authorizes extraction. Controls0–249 are the original BFM standing prefix and supply history provenance only. Controls1269–1818 are terminal BFM and remain excluded. No prefix duplication, nominal plan target substitution or feedback-gain supervision is allowed.

Before creating an ONNX session or label directory, require the final root qualification receipt. Bind the exact final nominal and separate-hold trace hashes plus all four independent reports. Nominal must contain all1569 controls, all15,690 native steps, all819 original source controls and passing source/return/quiet gates. Hold must contain all250 controls/all2,500 steps, passing physical and last3s quiet gates, with exact final-to-initial integration/history/prior continuity. Bind the expert frozen receipt, approved launch/review receipts, original BFM prefix trace and original BFM250 independent reports. Process exit or producer completion alone is insufficient.

The frozen source/input receipt must verify every bound file before collection. Use the previously reviewed, byte-identical feature/base helpers:1069 input features, original native h8/position1/yaw2 BFM, original joint mapping/native span and original feature normalization. No clock/clip identifiers, target cap, normalization refit or architecture change is part of extraction.

## History and action reconstruction

Reconstruct controls0..1268 sequentially from recorded actual precontrol qpos/qvel. At each control, compare the accumulated previous action with both `control_previous_action_before` and `previous_action`. Compare every named history tensor and the flattened300-vector before updating history with the current measured terms. `before_update` returns the prior history and then pushes current measurements; preserve this order.

For controls0..249, compare combined trace named tensors `control_history_<key>` with the bound baseline's `control_history_before_<key>`. Use exact saved `action[control]` as the next previous action: this is original BFM raw actor output times5, not a normalized clipped target. Produce no row and invoke no BFM inference for these controls. This recreates the complete250-control measured prefix without simulating it.

For controls250..1268, compute exactly one original BFM base prediction at the recorded actual state/history and frame `control+11`. Build features with the unchanged helper. Label `expert_target` from the actual applied `trace.target[control]`; set `residual_rad = expert_target - base_target`. Preserve the base prediction before native target clipping. Then advance previous action using the original MPC convention, normalized from the actually applied target and cast to float32. Do not clip this previous action to±5. Assert exact equality with recorded `action[control]`.

The five named histories remain `actions`, `base_ang_vel`, `dof_pos`, `dof_vel`, and `projected_gravity`, with four lags each. At control250, integration, named/flat history and previous raw BFM249 action must equal the bound precontrol250 snapshot. The first authorized base prediction can additionally assert that its native-clipped target equals `initial_seed/fresh_bfm.npz` target0 exactly. That existing first candidate used the same state, original goal261 and history; this is an array comparison, not another query or rollout. There is no legacy student-control1 parity check.

After processing control1268, assert the reconstructed history and normalized previous action equal the recorded precontrol1269 values. Do not run a terminal actor call to perform this boundary check.

## Exact row and endpoint indexing

| Field | Required selection or shape |
|---|---|
| Row indices | 0..1018 |
| `control` | `arange(250,1269)` |
| `source_frame` | `arange(261,1280)` |
| Controller mode | All1 (actual MPC) |
| `expert_target` | `trace.target[250:1269]`, shape(1019,23) |
| `features` | float32(1019,1069) |
| `base_target`, `residual_rad` | float64(1019,23) |
| `base_action`, `previous_action` | float32(1019,23) |
| `state`, `history` | float32(1019,52), float32(1019,300) |
| `control_integration_before` | `trace.control_integration_before[250:1269]`, float64(1019,291) |
| `teacher_qpos` | `trace.qpos[250:1270]`, float64(1020,30) |
| `teacher_qvel` | `trace.qvel[250:1270]`, float64(1020,29) |
| Named action/position/velocity histories | float32(1019,4,23) |
| Named angular-velocity/gravity histories | float32(1019,4,3) |

Endpoint arrays include both the state before first selected control and the state after last selected control; they are one row longer than labels. The three phase counts are acquisition100 (250..349), source819 (350..1168), and return100 (1169..1268). All control and source-frame values must be asserted explicitly rather than inferred from dataset length.

The collector report must bind qualification, frozen collector sources/inputs, new expert branch provenance, original BFM prefix and first-query parity. Record exact1019 actor calls, zero optimizer calls, zero new physics steps, zero fitting, all finite values/native target bounds, residual reconstruction error and previous-action maxima/outside-five counts. Preserve original normalization in a separate output archive and prove array equality with the current ordinary-final60000 normalization and earlier original normalization.

After collection, a separately authorized saved-array audit can compare phase-only old/new/query250 cohorts. Use float64 standardized feature distances, explicit control values and unchanged normalization. Preserve exact duplicates and report both target conflicts and residual conflicts separately; no averaging, removal, tolerance-based merging or checkpoint selection. Final model fitting is a separate root decision.
