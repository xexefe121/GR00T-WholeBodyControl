# One fresh expert branch from an actual student state

Interface plan and read-only review only. No expert query, fit, inference, simulation, or shared edit performed by this reviewer. Parent has now authorized compact_learner to implement one branch from the **final 20,000-step student's precontrol1**. The older 1,000-step snapshot below is superseded and must not be queried.

## Final selected input — supersedes provisional snapshot

Use `E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_continued_fit_v1/nominal/trace.npz`, SHA256 `c419119a3945a4255930d3ec423e04a5653d103e35ae916f8ba57149018a2ac0`. The final fit's ordinary nominal failed at global15/substep6, .312s, native speed ratio1.062584448; no source frames were reached. Selection remains the same earliest-divergence rule, precontrol1, not the later failure state.

The new checkpoint must copy `control_integration_before[1]` (291 float64), `qpos[1]`, `qvel[1]`, `control_history_before[1]` (300 float32), `control_previous_action_before[1]` (23 float32), and named history arrays decoded with the existing fixed term ordering. Read-only verification confirms previous action equals the final student's combined preclip `action[0]`; first control has ten substeps, next source frame12, actual and expected checkpoint time `0.020000000000000004`, and eight zero warning counts/lastinfo. Bind this trace and a new checkpoint hash before launch.

Parent's final authorized compute setting is **two Batch threads and BLAS1**, replacing the initial plan's historical four-thread suggestion. Objective and controller settings remain H30/10 iterations/commit5/.05 all-joint margin/weight2000/no added relative-foot cost/±.1 feedback, current hard guided/K0 restoration, original BFM goals, terminal global1269, and separate250 hold. No label admission until independent full-branch and quiet qualification.

## Superseded provisional input — retained review history only

`E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_nominal_diagnosis_v1/selected_actual_control1.npz` SHA256 `e1b46b8cfd85f75d29fe0bca5f5e3abdce57686b5b26f3e05d4be6888d6d8f8b` contains global precontrol **1**, next reference frame **12**, a 291-value float64 integration state with specification **8191**, qpos30/qvel29, previous-action23, and five named history arrays totaling 300 float32 values. The previous action is the student's combined **preclip** BFM-plus-residual action.

Read-only equality checks matched the checkpoint to the original student's precontrol integration/history/action buffers and qpos/qvel. The original trace SHA is `62b05301f5ab8f7f7d53c5a7d9177bf1bbb5f14cf7187da4d30f33401a5fbff8`. At its tenth physics step, actual time and accumulated expected time both equal `0.020000000000000004`; all eight warning counts and lastinfo entries are zero. The preceding control has exactly ten substeps.

## Minimal reusable pieces

Workspace paths below are relative to `Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof`; executable copies must be frozen in a new artifact before any authorized run.

| Existing source | Reuse |
|---|---|
| `gear_sonic/scripts/continue_g1_true23_mpc_hard_feasibility.py` | Full-integration restore pattern and `record_native_substep`, including raw evidence before any physical/parity failure. Its CLI/run is hardcoded to PICO/control3740 and cannot run this checkpoint unchanged. |
| `gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py` | `atomic_trace`, `finite_json`, generic seed selection, accepted plan/correction execution, restoration receipts. Its current run always initializes reference frame10/control0; do not invoke it as if it supports checkpoint continuation. |
| `gear_sonic/utils/g1_true23_mjbatch_mpc.py` | `load_native_bundle`, `load_motion_override`, `Native23Tracker`, reference window/target mapping. |
| `gear_sonic/utils/g1_true23_mjbatch_model.py` | Exact native servo copy, strict `Native23Feasibility`, full-MjData `preview_native_control`. |
| `gear_sonic/utils/g1_true23_mjbatch_bfm_seed.py` | Stateful actual-history holder, stateless fresh BFM proposal, `record_control` transition to actual MPC target normalization. |
| `gear_sonic/utils/g1_true23_mjbatch_restoration.py` | Existing guided-first/one-K0 restoration only if all ordinary seeds are infeasible; ordinary final proposal needs both existing certificates. |
| `E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_nominal_pilot_v1/source_snapshot_v2/terminal_yaw4_goal.py` | Exact qualified original-native terminal BFM goal, position1/yaw4/horizon8. |
| Same snapshot's `quiet_metrics.py` | Unchanged lifecycle-tail and separate-extension quiet diagnostics. |
| `artifacts/teleop_six_hour_20260910/qualify_recorded_candidate.py` | Fifteen original-source gates on an explicitly assembled full-clock trace. |

Use a **new standalone adapter**, not edits to the active main job or a disguised call to the PICO-only CLI. The older `evaluate_walk003_terminal_bfm_yaw4_hybrid.py` provides the history handoff pattern, but its old `.01 rad` online abort threshold is unsuitable: use the current strict substep recorder throughout.

## Exact branch initialization and clock

Load the pinned WSL MuJoCo3.2.3 native23 bundle for walk003 and the declared v4 reference at `E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz`; keep the original native motion for BFM goals and original29 for intent metrics. Validate their existing receipts and hashes.

Initialize with `mj_setState(integration,8191)`, `mj_forward`, then `mj_setState(the same integration,8191)`. Do not zero warmstart or replace qpos/qvel with reference values. **Warnings are not included in mjSTATE_INTEGRATION**: restore and bind the checkpoint-time warning counts/lastinfo from the original student physics ledger. Check full integration/qpos/qvel equality, zero external forces, and strict initial predicates. After initialization, no state writes except normal control/physics.

Copy each named `history_*` array into `fresh.history.data`, verify alphabetically flattened history equals `history_flat`, copy previous action without renormalizing it, and set `recorded_controls=1`. Any cumulative action diagnostics must be derived from the saved combined-action prefix and labeled as mixed student/MPC semantics. They are bookkeeping, not inference inputs.

Keep separate `initial_control`, `global_control`, attempted/full/partial counts, and actual/expected clocks. `planner.window(global_control+10)`; the applied control's post-state is scored against frame `global_control+11`. Preserve the absolute accumulated clock starting at the saved expected time; add .002 after every native step. Do not restart source time or pad a missing control.

For this checkpoint: fresh MPC covers controls **1–1268** (1,268 controls), terminal BFM covers **1269–1568** (300), then a separate hold covers **1569–1818** (250). Source remains **350–1168**, all 819 controls. Return remains **1169–1268**. Limit the final MPC commit to `min(commit,1269-global_control)` so no plan crosses the terminal boundary. The remaining lifecycle lasts 31.36s and ends at original absolute 31.38s; the separate hold ends at 36.38s.

## Proposed expert configuration and seed policy

Start from the **qualified walk003** objective, not the PICO-specific cost: H30, commit5, ten iLQR iterations, finite difference1e−6, all-joint margin .05/weight2000, relative-foot extra weight0, actual feedback correction clipped per joint to ±.1rad. These objective/control values come from `E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1/request.json`. Use the now-authorized **two Batch threads/BLAS1** and disclose the change from historical four Batch threads. Current hard-feasibility/accepted-K/restoration logic is also a declared new expert variant, not a claim of reproducing the historical teacher controller.

At the selected state no previous expert plan exists: initial warm seed is the existing reference feedforward target window, explicitly labeled as such. Compare it with the existing recorded BFM target window used by the qualified teacher (`bfm_walk003_arms_v3`) and one fresh BFM proposal from the selected actual state/history. The recorded targets are seeds only; never copy recorded physical states or call them labels. Subsequent windows shift the newly solved plan. Reroll every candidate from current actual qpos/qvel, retain the matching state/target/cost tuple, and use the current hard feasibility checks. If all seeds fail, use only the existing opt-in guided/K0 restoration and its dual certification; otherwise stop with raw failure evidence.

Before each committed target, run the current full-native-copy imminent-control guard. Apply the same manual PD and check actual every-2ms qpos/qvel/actuator force/warnings/clock plus forecast parity. The private one-lane optimization is excluded.

## History and label transition

At first expert control, infer/record the BFM base and student features from the **saved student previous action and precontrol history**. Do not update the real history while scoring seeds; `fresh.propose` works on a copy. Once an expert target is selected and certified, call `fresh.record_control(global_control,pre_qpos,pre_qvel,actual_applied_target)` exactly once before stepping. It stores the current measured observation with its preceding student action, then sets the next previous action to the actual MPC target normalization. This changes the action convention at the explicit controller switch; it does not rewrite historical student actions. Save named precontrol history and full integration before every proposal, not only flattened diagnostics.

At global1269, copy the accumulated expert history and previous action into a separate BFM controller history. Every terminal control updates history once, uses `terminal_goal_yaw4`, and retains raw actor×5 as next previous action. Do **not** call `record_control` for BFM terminal actions, because that would replace raw action with clipped-target normalization. Continue the same terminal history/physical state into the separate hold.

## Qualification and admission

Keep the fresh branch trace separately. Existing source auditor assumes control0 and source frames `arange(N)+11`, so it cannot directly audit an offset branch. Assemble an additional **explicitly declared prefix-plus-branch evidence trace**: immutable original student controls before selection, followed by the fresh continuation, dropping only the duplicate boundary state. Verify complete integration/history/action/warning/clock equality at the splice and bind both parent hashes. This is a counterfactual branch from saved actual state, not a claim that the entire record was rerun in one process. For control1, the prefix is exactly the original ten physics steps and one control. Never copy any later failed student or old teacher state into the branch.

The combined lifecycle must contain exactly1569 controls/15690 physics steps, all819 source controls and complete return. Recompute every original-root/heading/hand/head gate, declared-reference feet/legs gate, strict physical gate and warning/clock gate. Score the unchanged quiet criteria on the **original lifecycle's final three seconds**, then separately on the five-second hold. A partial control, no-feasible-seed stop, source failure, or quiet failure prevents admission; a feasible H30 plan alone is insufficient.

Only after full continuation qualification may the selected actual-student prestate and its **newly executed expert target** become an admitted query label. Store the base, features, residual target−unclippedBFMbase, original preclip previous action/history, source frame, model/reference/solver/request hashes and qualification hash. Additional expert-trajectory states must be labeled as expert continuation states, not as separately observed student states. No additional query, changed snapshot, aggregation or retraining is authorized by this plan.
