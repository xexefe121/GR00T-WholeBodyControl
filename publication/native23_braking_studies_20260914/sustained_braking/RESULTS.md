# Sustained state-triggered braking experiment

**Completed implementation and bounded tests. This fixed design failed at startup and remains disabled.** It is a regression relative to the unmodified experimental controller, not a recovery candidate.

| Original-start run | Physics completed | Applied filter interventions | Stop |
|---|---:|---:|---|
| Archived delays, then 12 ms | 0.02 s / 10 steps | 1 | Second command rejected |
| Alternating 0/12 ms | 0.02 s / 10 steps | 1 | Second command rejected |

Both runs entered BRAKE at their first command. The applied shoulder target differed from the actor request by **0.35189124 rad**, and acted for one 20 ms control interval. At the next update, the best sampled candidate passed **42/49** delay pairs. No candidate passed all 49, so strict publication stopped. There were two attempted interventions, only one actually applied; duration metrics count only the applied command.

The rejection came from the **new terminal settling criterion**: shoulder margin >=0.02 rad and absolute speed <=0.25 rad/s at both 60 and 80 ms. All tested trajectories at the rejected boundary passed the original physical prediction checks; their failures were terminal-condition failures. This does not establish that a physical violation was imminent or that other braking strategies are impossible. The chosen backup/terminal definition is too restrictive or ineffective at this start state to be useful as configured.

Full motion: **not completed**. Tracking: **not qualified**. Terminal 30-second standing: **not reached**. Timing: **not evaluated**. Both schedules stopped before the second target was applied, so successful changing-delay closed-loop operation was not demonstrated. The 49 current/next-delay combinations were exercised inside predictions.

## Delivered implementation

- Persistent TRACK/BRAKE/RELEASE states integrated into the existing `NativeTargetController.command()` path; no second publisher.
- Fixed 80 ms coupled-model prediction with explicit feedback braking after the first 20 ms control interval. Whole-body position/reserve, velocity, applied effort, fall, nonfinite, warning, quaternion and clock checks remain active.
- All 49 current/next 0–12 ms delay combinations. Subsequent two controls repeat the next delay; longer arbitrary jitter is not covered.
- Legal corrections restricted to the same shoulder coordinate; current actor values for the other 22 joints remain intact. The final guard-adjusted target is checked again before acceptance.
- Actual applied-command history, three-update release hysteresis, checked release interpolation, persistent captured goal, and snapshot fields for the complete filter state.
- Opt-in simulator launcher, native predictor, independent Python parity checker, focused tests and reproduction instructions. Default controller does not attach the filter.

## Evidence

- **147/147 native/Python delay-pair predictions matched exactly**, across three archived states. This validates implementation agreement, not controller effectiveness.
- **32 tests passed; 3 skipped** because MJLab was unavailable in the Windows test environment.
- Both runs reproduced saved TRACK and BRAKE controller/filter state, next commands and physical integration exactly. RELEASE was tested at unit level but never reached during these runs.
- Label/absolute-offset independence passed the focused filter test: those metadata never enter its decision API.
- With filter disabled, **30 baseline physics steps matched exactly** and the legacy wrapper remained unchanged. The completed shoulder replay/search study was not repeated.

## Decision

**Reject this fixed design. Do not promote or enable it.** No subsequent gain, horizon, threshold, joint-scope or actor tuning was performed. The original full-body teleoperation objective remains unresolved; this experiment did not move it closer to completion.

Configuration was recorded before each attempt in `preregistered.json`. Exact executed sources are archived under `executed_sources`; a later reporting-only correction excludes rejected commands from intervention duration and leaves all physical traces/decisions unchanged.

- [Combined report](report.json)
- [Frozen design and input/source hashes](archived_delays_v1/preregistered.json)
- [Archived-delay result](archived_delays_v1/report.json)
- [Alternating-delay result](alternating_delays_v1/report.json)
- [Prediction parity](prediction_parity.json)
- [Filter/controller restoration](archived_delays_v1/restoration.json)
- [Disabled baseline compatibility](baseline_compatibility.json)

Source instructions: `Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/onboard_inspection_20260912/SUSTAINED_BRAKING.md`.
