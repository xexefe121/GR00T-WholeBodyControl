# One proposed learner change: bounded velocity chords

**Design only. No model inference, perturbed feature/target generation, optimization, physics, or controller trial has run for this proposal.** Root must select the experiment before preparation becomes execution. The existing ordinary-final 65000 failure and both filter failures remain unchanged.

Choose **C as a soft finite-response regularizer**, using only joint velocity perturbations of the 3057 qualified walk003 moving rows. Keep the residual architecture, BFM, original feature normalization, native spans, all nominal labels, and original phase runtime unchanged. Do not add prior-action perturbations, PICO/walk002 labels, or held-out walk008.

This choice tests whether supervising the *local response* of the existing learner reduces the observed early velocity amplification. It does not claim that the committed MPC map is the replanned expert or that its nearby commands are physically feasible.

## Evidence fixing the choice

- Control 250 has exactly the qualified query250 feature/history/prior input, yet its target error is 0.03195782 rad RMS. The first input departure is control 251, before the first native clipping at 252. At 251, actual target error is 0.15084772 rad RMS versus 0.01765025 on the corresponding teacher input. The two later states belong to different trajectories.
- The saved analytical attribution assigns 0.16084 rad RMS of the teacher-to-actual head change at 251 to joint velocity. Correlated blocks cancel; this is not an independent causal intervention. Direct previous-target/raw-prior contributions are smaller, 0.01708/0.01531 rad RMS. BFM base itself changes 0.00669774 rad RMS.
- The first velocity departure is 0.30828113 rad/s RMS, 0.72267470 maximum; divided jointwise by native speed caps, RMS is 0.00945588 and maximum 0.02258358. This fixes a modest **one-percent-of-native-cap axial radius**, not a radius selected by a new rollout.
- Committed velocity gain spectral medians are 0.914/0.915/0.923 for old/query1/query250, but maxima are 11.55/489.86/559.82. One row contributes **99.05%** of query1's sum of squared velocity-gain Frobenius norms and **97.88%** of query250's. Raw squared Jacobian matching would be dominated by these isolated plans.
- There are 54/35/21 velocity clipping-kink rows and 0/5/45 zero-gain rows. Zero gain can mean no accepted solver update; it does not establish that the desired replanned feedback is zero. Every such nominal row stays in the loss.
- The prior-action recurrence norm/radius at control250, 7.36/1.54, holds plant and BFM base fixed. These are not closed-loop stability eigenvalues. The earliest observed growth precedes clipping and the later applied-action/history mismatch; a prior-only change would miss the strongest present evidence.

`scales.json` binds the saved input hashes and scalar arithmetic. Every recorded moving precontrol state has at least 17.35% native speed margin. The proposed axial perturbations therefore remain inside static joint-speed bounds, but that says nothing about subsequent torque, contact, height, tilt, or speed feasibility.

## Exact observation and teacher construction, if selected

For each dataset/control row i, preserve its physical qpos, root linear/angular velocity, actual prior action, returned pre-update history300, reference clock `control+11`, original motion, and already committed plan/local index. For every native joint j form two physical qvel copies:

`v_i± = v_i ± 0.01 * native_velocity[j] * e_j`.

The step is 0.20–0.37 rad/s. There are 70,311 axis pairs and 140,622 perturbed inputs. These are independent instantaneous observation probes, not integrated trajectories. An axis probe has cap-normalized RMS 0.002085; the full simultaneous control251 displacement is outside this set. Do not describe these probes as complete coverage of that displacement.

The preparation budget is **143,679 actor ONNX calls**: one center plus46 probes for each of3057 anchors. Compute and cache one exact backward-encoder goal latent per center, because qpos and reference clock remain identical across its probes: **3057 backward calls**, **146,736 BFM ONNX calls total**. Cache reuse must be source-reviewed against the frozen `_goal(frame,qpos)` dependency; it changes no actor input or arithmetic. There are no extra actor queries for label selection. Feature float32 storage alone is614,371,404 bytes, before targets/provenance/compression.

Recompute the frozen BFM actor at each probe. Its state52 order is joint offset23, joint velocity23, gravity3, scaled body gyro3. The actor also explicitly receives `last_action`, history300 and the goal latent. Hold these latter inputs exactly fixed for this velocity-only probe. Do not call `history.before_update` or mutate a history buffer while constructing probes. Preserve all five named history arrays in each source row's provenance.

Recompute features with the original feature function and unchanged float32 conversions. Only joint-velocity features23:46 and BFM-base features1023:1046 may change. Previous-target52:75, raw prior1046:1069, and every other feature must remain bit-exact. Reusing the nominal BFM base would be an inconsistent observation and is prohibited.

For the teacher, use the exact saved plan, its already-signed K, and original tangent convention:

`T_i(v) = native_clip(u_plan + clip(K * difference(x_plan, x_i(v)), -0.1, +0.1))`.

Joint velocity occupies tangent columns35:58. No sign changes, replanning, line search, candidate selection, physical rollout or new expert query. Reconstruct all 3057 nominal actual targets bit-exact before accepting any probes. Preserve each plan-control/local-index hash binding. Count every feedback saturation, native saturation, zero K and crossed clipping branch in the generated audit; retain them all, including difficult originals. A finite chord crossing a clipping branch is a valid value of this fixed map, not a smooth derivative or a trustworthy feasibility label.

The exact teacher center-to-probe command difference is bounded by 0.2 rad per joint: both feedback corrections lie in [-0.1,+0.1], and native clipping is nonexpansive. This bound does not depend on K. Native-span normalization gives a largest bound of 0.38197 for the narrow ankle-roll span. No division by perturbation size, gain norm, or tiny branch radius is used.

## Objective and one fixed fit

Let `Uθ(x)=b(x)+s*fθ((features(x)-mean)/std)` be the total *unclipped* target, with frozen BFM base b and native span s. Keep the existing nominal objective exactly: equal weight for all nine dataset×phase cell means of `((Uθ(x_i)-T_i)/s)^2`, over all 3057 rows and 23 outputs.

Add one equally weighted finite-response term, coefficient **1** fixed before generation. For each sampled axis pair and each sign, compare the centered changes:

`[((Uθ(x_i±)-Uθ(x_i)) - (T_i±-T_i))/s]^2`.

Average over both signs, joints, sampled pairs and the nine cells. Both center and perturbed model evaluations receive gradients; there is no stop-gradient center. This targets response while the original anchor term fixes absolute command values. The teacher uses both exact clips. Student unclipped differences retain learning gradients when a poor proposal would saturate; applied-target chord errors and clipping counts are separate mandatory diagnostics. The runtime native clip is unchanged. This is not raw derivative matching, and it does not divide by a small velocity step.

Proposed single continuation: restore ordinary-final65000 model, all six AdamW states, Torch/NumPy RNG, original mean/std/span; **5000 additional updates, final70000 only**. Keep AdamW weight decay1e-5, gradient clip10, CPU one thread. Fixed cosine learning rate3e-6 to3e-7, starting at the previous final learning rate. At every update use all3057 nominal rows plus **64 uniformly sampled row-axis pairs per each of the nine cells**, 576 pairs/1152 perturbed rows. Sample with replacement from each cell's row×23 Cartesian product using the restored Torch RNG, in dataset then phase order. This is an unbiased nine-cell chord objective; full nominal weighting is exact at every update. No difficult rows are removed or downweighted by K.

This consumes 21.045M training head row evaluations, 1.377× the prior5000-update full-batch fit, excluding fixed diagnostics. It is a fixed diagnostic budget, not a claim of convergence. Save ordinary final checkpoint, optimizer/RNG and fit arrays before export. No early best checkpoint, intermediate physical evaluation, adaptive radius, coefficient, learning rate, extra budget or sweep.

## Required preparation and fixed test

Before fitting, freeze source/input receipts and review all probes: source/control/local-plan alignment; bit-exact nominal targets, BFM outputs, state/history/prior/features; finite perturbed BFM/features; exact unchanged-feature masks; static speed bounds; exact two teacher clips; native target bounds; clipping/zero-K counts; and no hidden history/clock mutation. Generate no physical feasibility certificate for a probe without simulation. No simulation is proposed for preparation.

Record initial and ordinary-final full nominal loss, full chord loss on all70,311 pairs, each cell, first24 controls per phase, query250 signed errors, and the original fixed nearest pairs. Report unclipped and applied errors separately. Re-evaluate the existing seven saved actual states250,251,252,255,260,270,278 only as fixed diagnostic inputs: no new labels and no claim that their matching teacher states are counterfactual truth. Report joint-velocity sensitivity and conditional prior recurrence with the original limitations. Diagnostics never select a checkpoint. Require numerical finiteness and final Torch/ONNX parity before a physical trial; any numerical failure preserves artifacts and stops.

For a concrete export budget, run restored65000 and final70000 head parity on the143,679 fixed inputs in562 batches of at most256, and the seven saved actual inputs as one additional batch per head: **1126 head ONNX calls total**. Reuse those saved predictions for all initial/final numerical diagnostics. Analytical head Jacobians use the two frozen Torch models at the same seven saved inputs; they add no BFM calls, optimizer steps, physical states or teacher labels. Initial/final full Torch predictions add287,358 row evaluations plus14 fixed actual-input rows, outside the21.045M training row budget. No intermediate full chord diagnostics are required.

Then, only after root selection and source/export clearance, run **one fresh canonical1569-control trial plus a separate continuous250 hold if reached**. Use the original unfiltered phase runtime: original BFM controls0..249, learned controls250..1268, terminal yaw4 controls1269 onward. Keep its raw combined-action prior semantics, native clipping, goals, plant and strict every2ms gates unchanged. Verify the fresh BFM250 prefix and first query250 feature/base/prior/history exactly against the qualified fixture before applying the new head. The new head output must match its own frozen export; it is expected to differ from the65000 head.

Preserve both proposed raw action and applied normalized action to diagnose any later clipping-history difference, but do not change that runtime convention in this learner experiment. Parent independently replays every actual native sample, checks all819 source controls and both quiet windows. Incomplete execution fails the task. Timing remains measured; there is no real-time or hardware claim.

## Why A and B are not selected

**A:** Exact smooth teacher target derivative is `M_native*M_feedback*K_v`. For the total learner it would be `B_v + J_residual*(E_v + E_base*B_v)`, where `B_v` is the frozen BFM response and the residual Jacobian includes original normalization/span. BFM derivatives are not currently measured. Raw matching is dominated by extreme K and undefined at kinks; normalized robust derivative matching would still require a normalization/branch convention while labeling a committed derivative as only that. Finite chords retain the exact clipping map and implicitly include BFM's nonlinear change without differentiating it.

**B:** Limiting only direct velocity or prior-input weights leaves an indirect path through the BFM-base feature block. Removing that block or forcing a contraction changes architecture and may remove useful expert damping; no measured full BFM chain or justified contraction bound currently selects such a constraint. The three datasets have no exact duplicate conflicts, but nearest-input target differences are typically about0.25–0.26rad RMS. That does not establish that a more restrictive architecture is identifiable or can fit the same evidence.

For prior action specifically, the actor explicitly uses `last_action`; it also determines previous-target features and staged action history. Perturbing only the final23 features is inconsistent. At fixed physical state/history/committed plan, teacher target has no explicit prior-action dependence, but this does not establish invariance of a newly selected/replanned expert or a physically reachable causal history. Defer prior-action augmentation. Full history remains absent from the residual's direct inputs, and clock, committed-plan identity, contacts and longer temporal dependence can still make this finite-response experiment fail.
