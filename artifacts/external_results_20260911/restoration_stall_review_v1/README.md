**Saved candidates prove that guided feedback overwhelms the small feedforward step. No new causal implementation defect or state-index mismatch was established by this read-only review.** The audit uses only archived arrays, source and the pinned mjbatch implementation; it does not run a solver, derivatives, inference or dynamics. `report.json` contains independent algebraic bounds and input SHA256 hashes.

Restoration inherits the core line-search law `clip(u + alpha*k + K*dx, native_lower, native_upper)`. There is **no ±0.1 feedback correction clip in restoration**, or in the inherited nominal/guided planning rollout. The continuation runner applies that clip separately when executing the main controller. Scaling only `k` while leaving `K*dx` unscaled is the existing iLQR search law; this fact alone is not a coding error or permission to change the physical controller.

For alpha 1/256, boxQP's feedforward constraints imply `abs(alpha*k[j]) <= native_joint_width[j]/256`, at most **0.0224984 rad**. Since the original seed lies within native target bounds and clipping is non-expansive, every saved candidate gives the lower bound `abs(K*dx) >= max(0, abs(candidate_target-seed_target)-alpha*joint_width)` for some knot/joint. This bound is independent of an assumed contact model and requires no additional rollout.

| Regularization | Largest saved target departure at alpha 1/256, rad | Proven minimum feedback magnitude, rad | First-knot target departure, rad | Candidate merit |
|---:|---:|---:|---:|---:|
| 1 | 4.80963 | 4.78850 | 0.000211181 | 286.858 |
| 10 | 4.78292 | 4.76178 | 0.000178667 | 387.032 |
| 100 | 4.78292 | 4.76178 | 0.000028205 | 389.490 |
| 1,000 | 0.10302 | 0.08052 | 0.000002568 | 4.33408 |
| 10,000 | 4.45390 | 4.43277 | 0.000004769 | 331.840 |
| 100,000 | 4.78292 | 4.76178 | 0.000003982 | 292.178 |
| 1,000,000 | 4.52349 | 4.50235 | 0.000001963 | 328.701 |

The initial merit is 4.29281576. All 63 generated candidates fail the separate strict feasibility predicate; the initial seed does too. Therefore the rejection rule did not discard a feasible higher-merit candidate. The original nine unguided target sequences and final returned targets are bit-identical to the seed. Each small-alpha first-knot change respects its feedforward bound; large departure develops later, where feedback is active.

Increasing the current state regularization does not necessarily damp feedback. The backward pass forms `K = -(Quu + mu*B.T*B)^-1 * (Qux + mu*B.T*A)` on free controls. The numerator also grows with `mu`; where the relevant inverse exists, its large-mu limit is approximately `-(B.T*B)^-1*B.T*A`, not zero. Thus the observed finite but large dynamics derivatives (max |A| 89471.3, |B| 85961.6) and returned |K| max 5120.63 are compatible with severe guided coupling even after repeated regularization increases. This is an algebraic explanation of the recorded search behavior, not a proposed parameter sweep.

Derivative/state convention inspection:

- Both derivative `step` and line-search `advance` reset `qacc_warmstart` to zero once per 20 ms control and apply the same affine clipped-PD servo model for ten 2 ms Euler steps. The derivative batch's default `forward=False` was verified in the installed pinned mjbatch Python/C++ source. It performs one step for the input-pose feature probe, then nine more. The ungated line batch performs ten steps, then one `mj_forward`; the hard line path likewise forwards once after all ten steps. That final forward does not change the generalized state used for the discrete A/B map.
- `Native23Tracker.probe` deliberately pairs pre-step body features with a saved copy of the pre-step generalized state. Restoration's residual discards all 42 body-feature entries and uses only the following 59 generalized state values. Its joint positions are indices 7–29 and joint velocities 36–58; the quaternion is 3–6. Those dimensions and signs match the native23 contract. Stale body-position features cannot explain this restoration merit.
- Perturbation/retraction and output differences use matching free-joint tangent coordinates; one-sided action perturbations reverse sign near upper target limits. The cost Hessian scale 5000 is consistent with the declared 0.02-rad joint-margin residual normalization. No off-by-one frame or accidental source-reference tracking term was found in this state-only restoration merit.
- Nominal planner rollouts intentionally differ from a continuation of the full actual native `MjData` because they reset warmstart and use the servo representation. The independent full-state manual-PD certificate exists to test that difference. This distinction is disclosed; it is not silently treated as actual-state physics identity.

Finite A/B values do not prove a useful local linear model around contact transitions. The archived stall artifact does not retain full A/B matrices, per-iteration k/K or candidate state trajectories, so it cannot support a component-by-component directional-derivative or zero-alpha replay test. No such new test was run here. The evidence establishes feedback amplification; it does not establish that all large derivatives are mathematically incorrect.

One existing code-level return-value defect is visible but **does not cause this restoration stall**: the legacy soft `ilqr` assigns `K` from a backward sweep before checking whether its rollout is accepted. With all searches rejected, returned targets/states remain the original incumbent while returned `K` comes from the last rejected sweep. The restoration caller discards that gain, and the hard solver has separate accepted-incumbent gain handling. The diagnostic's 5120.63 figure is therefore a rejected search gain, not a gain attached to a successfully accepted restored controller. Do not execute or present it as such.

Parent subsequently reported an independently certified K=0 candidate at the same control, with merit 0.000251825 and all 300 native steps passing. That is separate new physical evidence supplied by the parent; this artifact's calculations remain tied to the original failed guided search. A narrowly gated K=0 retry can now be assessed without claiming recursive recovery or changing the main tracking objective.
