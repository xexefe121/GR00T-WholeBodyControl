# Native23 hard-feasibility work — 2026-09-11

The optional `--hard-feasibility` path rejects physically invalid plans before execution. The full canonical PICO experiment stopped at 69 source seconds because no next seed was feasible. All 38,000 executed physics steps passed and reproduced independently bit-for-bit. It does **not** yet provide a complete recovery controller or a complete PICO replay.

The prior PICO failure was visible at the predicted 20ms endpoint: left ankle pitch exceeded its native lower bound by 0.02404418 rad. Privately finishing the actual control produced 0.02377251 rad excess. The original prefix reproduced bit-for-bit. This was not an excursion hidden between planning knots.

## Implemented behavior

- Every rollout's initial state and every 2ms substep must satisfy all 23 native joint ranges (1e-6 rad tolerance), native speed caps, native effort caps (ratio tolerance 1e-9), finite state/force/clock, unit root quaternion, root height at least 0.25m, tilt at most 1.2rad, and zero MuJoCo warnings.
- Invalid line-search candidates remain invalid. NaN and all-infinite costs cannot win selection. A feasible initial incumbent is required; accepted states, targets, and their generating feedback gains survive rejected updates.
- Before applying each actual target, the runner checks ten native manual-PD substeps from a complete private MjData copy. Rejection does not advance the actual state, clock, or BFM action history. No recovery is claimed for an abort.
- Actual physics uses the same strict predicate. Raw actuator force, observed/expected time, warnings, state, and commands remain in NPZ even when a nonfinite failure requires JSON summaries to use null.
- The actual expected clock advances independently by repeated addition of 0.002. Ideal `step_count * 0.002` roundoff is reported separately; otherwise a normal 130.6-second PICO run would falsely fail a 1e-10-second tolerance.
- The default-off controller path remains bit-for-bit equivalent to the prior frozen solver. Quiet feasible rollouts also match the old 20ms dynamics bit-for-bit. Derived fields are forwarded once per control, preserving the old numerical contract.

## Verification

`hard_feasibility_tests_v4.xml`: 37 tests passed in pinned MuJoCo 3.2.3. Tests cover default parity, feasible rollout parity, intermediate lane faults, all-invalid seeds, NaN candidate costs, accepted gain retention, full private-state preservation, nonfinite faults after prediction, empty trace shapes, raw failure evidence, and the full PICO clock recurrence. Ruff passed.

Root's independent native manual-PD oracle reproduced the first five committed controls bit-for-bit: `hard_feasibility_3740_v1_independent/report.json`.

## Bounded continuation result

`hard_feasibility_continuation_3740_v1` started from the archived actual state and complete actual BFM history at control 3740 / source 67.8 seconds. It used H30, five iterations, five-control commits, eight threads, feedback correction bound 0.1rad, native all-joint soft margin 0.05rad/weight2000, relative-foot cost400, and unchanged original-goal BFM seeds. Source files were frozen and remained unchanged during execution.

The requested two-second continuation stopped after **30 of 100 controls / 300 physics steps / 0.6 seconds**. Every executed step passed the strict physical checks and matched its private forecast bit-for-bit. At control3770 / source68.4 seconds, all available 30-control seeds were infeasible:

| Candidate | First predicted failure | Joint excess |
|---|---:|---:|
| Shifted previous plan + reference tail | 534ms, horizon control26/substep7 | 0.00480872rad |
| Recorded BFM commands | 20ms | 0.00251453rad |
| Fresh BFM from actual history | 74ms | 0.00065949rad |

The shifted plan's first25 controls remained feasible; its newly appended reference tail starts at control25. Tail completion is the next bounded experiment. The failed requested continuation is preserved, and no full restart is qualified by this result.

The terminal integration state, complete named BFM history, previous action, and shifted targets are saved in `hard_feasibility_continuation_3740_v1/trace.npz`. Restore uses `mj_setState → mj_forward → mj_setState` only during initialization to preserve the archived warm-start state. No state rewriting occurs during execution.

## One bounded restoration solve at control3770

Both fixed-prefix25 tail variants failed, so neither was integrated. Holding the last target failed at540ms; a five-control BFM completion failed at516ms. The retained prefix ended with root height0.32675m and vertical velocity−0.92975m/s, explaining why changing only the last five targets was insufficient.

The separately authorized restoration experiment allowed all30 controls to change, using only the fixed physical-margin merit: 0.02rad joint interior margin, 90% native speed margin, 0.30m minimum root height, 1.1rad maximum tilt, and target regularization1e-4 toward the original shifted targets. These are seed-search costs, not relaxed acceptance thresholds. The main tracking objective remains unchanged.

`restoration_3770_v1` completed one H30 solve with at most10 iterations in4.318s. Merit decreased52.3580→0.979045. The proposed seed passed the strict nominal rollout and the independent full-state native manual-PD oracle over all300 physics steps: joint excess0, speed ratio0.52073, effort ratio1, maximum tilt0.36943rad, minimum height0.28343m, warnings0, and clock error0. Source files remained unchanged during the experiment. No actual trajectory control was executed.

The exported `restoration_3770_v1/feasible_seed.npz` has SHA256 `f114fb8062a8a02d1e24d427056d8a36e2644e928656231a860888cb6ddcb20b`. This is a feasible 0.6-second seed, not a qualified recovery policy or completed source recording.

All work remains simulation-only. No robot, DDS, or live Pico interface was used.

## Repeated restoration and remaining stall

`hard_feasibility_restoration_continuation_3740_v1` repeats the original actual3740 initialization and requests100 controls. It completed65 controls/650 physics steps/1.3 seconds, then rejected all seeds at3805. The first30 controls and complete saved BFM history are bit-for-bit identical to the prior continuation. Four restoration seeds were independently certified and used, at3770/3775/3780/3785. A fifth attempt at3805 produced no accepted update. Every executed substep remained physically valid, but the intended two-second continuation is incomplete and overall tracking over the segment also misses the yaw/leg gates.

The final actual state has root height0.323273m, vertical velocity+0.032867m/s and tilt0.36198rad. It is not undergoing the earlier downward collapse. Holding its last actual target nevertheless fails a private H30 probe after410ms, so holding is not a certified fallback.

`restoration_stall_3805_v1` instruments one identical fixed-merit solve. All63 generated alpha candidates fail the strict native nominal predicate. There is no higher-merit feasible candidate accidentally rejected. The original seed fails at570ms on right ankle roll upper bound by0.0000427055rad. Initial fixed-state merit is zero; all nonzero merit arises at knots20–30. All derivatives and backward sweeps are finite/successful, but all seven line searches reject; regularization grows beyond1e6 and ends the solve. Initial/final merit remain4.2928157553. Very large dynamics derivatives and feedback gains produce target departures exceeding4rad even at alpha1/256. The next numerical experiment requires separately evaluating whether restoration feedback amplification can be avoided without changing the strict predicate.

After these frozen trials, the bounded runner gained raw-before-failure evidence retention, correct partial-control/time labels, independent expected-clock accumulation, and per-restoration proposal/full-integration/history artifacts. Three isolated fault tests passed (`bounded_substep_fault_tests_v1.xml`); no new trajectory has yet run with these reporting fixes.

## Certified retry, completed physical segment, and full replay in progress

`restoration_zero_feedback_3805_v1` changed only private restoration rollout feedback to zero. The ordinary final proposal lowered merit4.29281576→0.000251825 and passed all300 nominal and actual native substeps. The shared fallback now tries guided restoration first and retries once with this zero-feedback rollout from the unchanged original shifted seed only if certification fails.

`hard_feasibility_zero_feedback_retry_3740_v1` completed the requested100 controls/1000 physics steps/2 seconds. Six restoration triggers were accepted; exactly one needed the zero-feedback retry, at3805. Prior65 controls and all16 shared state/action/history fields reproduced bit-for-bit. Every executed substep passed strict physical checks; the independent oracle reproduced all1000 state, velocity, torque, actuator-force, time and warning samples bit-for-bit. Trace SHA256: `6dfa397b10731f295c8c6b5009e09fa9b2e141ff492643629d355add8f28514c`.

This is a physical segment pass only. Original-intent tracking over the two seconds fails multiple gates: root p95 0.1853m, yaw19.27deg, hands0.295/0.218m, feet0.172/0.142m and leg RMSE0.188rad. These results do not establish a qualified recovery or full-body tracking controller.

The bounded and full-source runners now share exactly one restoration helper. Four focused checks passed: default solver parity, zero-feedback scope, exact3770 guided-seed equivalence and exact3805 guided-to-zero-feedback equivalence, including corresponding returned states/cost and preserved actual integration state. Read-only review found no new acceptance or policy mismatch.

The full PICO replay in `pico_full_hard_restoration_v1` started from canonical v4 reference frame10 with all6530 lifecycle controls and5780 original-source controls requested. Configuration: H30, five iterations, five-control commits, eight threads, EPS1e-6, feedback correction0.1rad, all-joint soft margin0.05rad/weight2000, relative-foot cost400, original recorded/fresh BFM goals, strict hard feasibility, and the shared restoration fallback. No late physical-state fixture was used. The command is preserved in `launch_pico_full_hard_restoration_v1.ps1`; output log is its sibling `pico_full_hard_restoration_v1.stdout.log`.

It stopped at control3800: 3800/6530 lifecycle controls, 3450/5780 source controls, 38,000 physics steps and 69 source seconds. The reason was `no_feasible_seed`; no rejected control executed. Fourteen restoration triggers produced thirteen accepted seeds and six zero-feedback retries. Final trace SHA256: `419ceb38d95ae1a6314089b56274cea2e576153b72bfa78e6d2f276886299612`.

The independent replay in `pico_hard_final3800_independent_v1` reproduced every recorded position, velocity, command torque, actuator force, time and warning sample bit-for-bit. Maximum native range excess0, speed ratio0.864598, effort ratio1, tilt0.694047rad and repeated-clock error0. The intended full lifecycle remains incomplete and fails qualification.

Original-intent metrics over the completed source prefix remain below their aggregate thresholds: root p95 0.153459m, yaw4.97297deg, relative hands0.070768/0.100380m, head0.034226m, same-world-axis relative feet0.077778/0.066450m and leg RMSE0.082602rad. These prefix metrics do not establish full-source success.

Saved-plan inspection found that every shifted-seed rejection over controls3700–3800 starts in the five newly appended reference commands, after the retained25 controls. At3800 the guided restoration proposal first violates left ankle roll at282ms and also later violates right knee and speed limits; the zero-feedback proposal first violates right ankle pitch at512ms. Both rejected full proposals and all300 private physics samples are retained in `pico_final3800_rejection_v1`. A single fixed last-target tail certificate is being tested before any further optimization or full restart. No shared controller changes have followed this full run.

## Control-space damping: bounded physical pass, tracking still incomplete

The ordinary final control-space LM restoration seed at actual3800 passes all300 native2ms steps. It uses a symmetric control Hessian plus fixed muI, unchanged cross derivative, factored value update and private zero-feedback rollout. Original merit, target bounds, native physical checks and the main MPC objective remain unchanged. All10 iterations improve merit7.27870567 to.0125824653. The root oracle reproduces every saved state, torque, clock and warning sample exactly; ordinary-final SHA55ffd9805cfe58936c9633b05e14f3485f22d6d4462a155628bde159337b36db.

The single continuous actual3800-to3900 experiment, control_lm_continuation_3800_v1, completed100/100 controls and1000 physics steps (source69–71 seconds), with no failure. It regenerated the original guided, original K0 and control-LM proposals exactly before execution. Third fallback was used at3800, ordinary guided restoration at3805, then normal MPC through3900. Independent root replay matches all1000 positions, velocities, command and actual torques, times and warning arrays bit-for-bit. Range excess0, speed ratio.510965, effort ratio1, tiltmax.418345rad, clock error0. Trace SHA0ce95ddf66876efa78a00b3c7148004e95c52d8989d9693656a13be218a6525a.

Local tracking over these two seconds still misses yaw, feet and legs: original root p95.163754m; yaw22.459deg; relative hands.125364/.118772m and head.033928m; same-world-axis relative feet.141928/.159131m; leg RMSE.159989rad. Thus this is a bounded physical pass, not a full-source or tracking qualification.

Next authorized work is shared opt-in third-fallback integration with default equivalence tests and read-only source review, then one fresh canonical full6530-control PICO trial. No old/new trajectory composition, relaxed gates or hardware actions. Durable hidden process wrappers now record launch, PID, logs and exit status so a new chat message should not terminate a running simulation.
