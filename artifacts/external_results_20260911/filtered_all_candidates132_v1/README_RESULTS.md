# Fixed candidate diagnostic: tracking cost warns before feasibility fails

The original tracking cost components distinguish the learned proposal's deteriorating motion before the first admission override at control 268. At every recorded state from controls 250 through 267, all four constant-target forecasts pass the 100 ms physical gates, but the primary proposal has greater truncated tracking cost than the original BFM target. BFM has the lowest score at 14 of those 18 states; the current-pose target has the lowest score at the other four. Primary is never the lowest-cost feasible candidate in this diagnostic.

This supports investigating tracking-based selection. It does **not** demonstrate that such selection would complete acquisition, walk the source motion, or recover balance. No new controller has been selected or run.

## What was evaluated

Exactly 132 forecasts: the same four fixed candidate targets at each saved actual filtered-controller state from controls 250–282, including candidates that first-feasible selection previously skipped. Each forecast starts with that state's complete 291-value integration state, warning counts and warning information, and holds its candidate target for at most five controls / 100 ms. The unchanged native forecaster stops on its first strict 2 ms failure.

All 132 outcomes are retained. There were 6,154 private physics steps, zero connected-plant steps, zero policy inferences, zero optimizer calls and zero model updates. Of 132 forecasts, 113 complete and pass: primary 22/33, BFM 28/33, previous-applied 32/33 and current-pose 31/33. A failed forecast's shorter available state sequence is censored; its finite partial sum is never ranked against a complete feasible 100 ms sequence.

Request was frozen before execution: `request.json`, SHA256 `485aaa3b9df22c7c606574e7eddd0c64b24c36693047dc16c22c88d00b3ded93`. Eight source files and 160 inputs are bound. The single hidden durable process exited 0. `results/report.json` retains all case reports and per-state diagnostic orderings; `results/forecasts.npz` retains every checked physics sample; `results/cost_knots.npz` retains all available knot features, residuals, components, references and tracking measurements.

## Cost conventions and checks

The cost implementation comes directly from the qualified BFM250→expert branch's frozen `Native23Tracker`. Its original H30 constructor, all-joint interior margin 0.05 rad / weight 2000, relative-foot weight 0, and all other weights remain unchanged. Only feature evaluation and exact residual/target-reference functions are called. Planner rollout, advance, step, linearization and expansion are disabled in the diagnostic.

For recorded precontrol **c**, planner window is **c+10**. State knot **t=0** uses reference frame **c+10**. Predicted post-control knots **t=1..5** use frames **c+11..c+15**. Five target penalties **t=0..4** use the clipped reference `joint_pos + kd/kp * joint_vel` at frames **c+11..c+15**.

The reported prefix score sums the exact original state residual squared at knots 0–5 and five exact original input penalties at knots 0–4. It is a **truncated prefix**, not the complete original H30 objective, which contains 31 state knots and 30 input penalties. No half factor, timestep multiplier, horizon normalization, new terminal multiplier or weight was introduced. Initial state cost is common to the four cases from one recorded state and is reported separately. Splitting residual sums into named components introduces at most 4.55e-13 of floating-point summation roundoff.

Position and quaternion-log costs use the native body origins in order pelvis, torso, left ankle, right ankle, left rubber hand, right rubber hand. These are not the separate original29 hand/head acceptance measurements. The archive additionally records unweighted root position/height/yaw, signed vertical velocity, full23 joint RMSE and world/root-relative foot position errors.

All 50 previously saved candidate forecasts reproduced bit-for-bit across all seven trace fields. All 132 source integration and warning states remained unchanged. Native-model forward kinematics matched servo-copy cost body positions and quaternions bit-for-bit at 739 available knots. At 625 available running knots, direct calls to the original cost method exactly matched the reconstructed state-plus-input cost. Physics forecasts use the unchanged native PD component with the saved integration state; no claim is made that a separate planner rollout with its own warm-start conventions would produce identical trajectories.

## Evidence before control 268

Every row below compares two separate fixed-target forecasts from the **same recorded actual state**. Costs include the common initial state knot. Endpoint metrics refer to 100 ms.

| Actual control | Primary / BFM prefix cost | Primary / BFM vertical velocity, m/s | Primary / BFM root-height error, m | Primary / BFM joint RMSE, rad |
|---|---:|---:|---:|---:|
| 250 | 89.661 / 39.402 | +0.300 / +0.00182 | −0.00463 / −0.04586 | 0.10693 / 0.08866 |
| 251 | 44.985 / 29.976 | +0.109 / −0.01569 | −0.01813 / −0.03174 | 0.08913 / 0.05728 |
| 255 | 359.058 / 60.546 | −0.504 / −0.257 | −0.01547 / −0.01646 | 0.19332 / 0.09115 |
| 260 | 686.207 / 215.257 | −0.674 / +0.117 | −0.07052 / −0.02897 | 0.29457 / 0.20408 |
| 267 | 3910.850 / 1073.075 | −2.418 / +0.03301 | −0.16970 / −0.05430 | 0.65168 / 0.29691 |

At control 250 the primary actually predicts less root-height error than BFM; the early cost warning is principally excessive motion, not a predicted fall. Its generalized-velocity component is 53.936 versus BFM's 0.00389. Primary's velocity subtotal comprises root translation 5.776, root rotation 22.970 and joints 25.190. All native-speed and joint-margin hinge penalties are zero for both forecasts at this state.

By control 267, primary predicts root descent at −2.418 m/s, yaw error 41.07°, and foot position errors 0.298/0.881 m; BFM predicts +0.033 m/s, yaw error 9.88°, and foot errors 0.043/0.379 m. Both still satisfy the 100 ms hard physical gates. Primary's weighted position/rotation/velocity subtotals are 1243.306 / 1772.235 / 794.895, versus BFM's 507.892 / 376.075 / 160.058. Tracking degradation is therefore visible substantially before the first hard rejection.

At control 268, primary first fails native speed at 20 ms, and BFM fails an ankle joint bound at 66 ms; previous-applied and current-pose remain feasible. The lowest prefix cost among these two is current-pose (1978.800 versus 5196.388), while the existing fixed-order controller chose previous-applied. This is a diagnostic discrepancy, not a newly executed decision.

## Limits of the conclusion

The 33 source states come from the already failed first-feasible filtered trajectory. Choosing a different target at 250 would change every later state, action history and BFM/head proposal, so later case rankings cannot be concatenated into a recovered trajectory. Also, the deployed proposal would ordinarily be recomputed every 20 ms; holding it constant for 100 ms is the deliberately fixed forecasting assumption. A low short-prefix score does not ensure future feasibility or full source tracking and may favor staying near the standing pose during acquisition. The data justify testing a declared tracking-based controller only if separately selected, with the complete lifecycle/source/quiet/timing gates retained.

All 132 cases remain evidence. No labels were generated, no head was changed, and no connected controller was run by this diagnostic.
