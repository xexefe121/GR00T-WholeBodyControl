Source preparation only. Actual fitting waits for complete full-state data qualification, frozen inputs and concrete source review. This is one changed-objective continuation from the exact direct 55,000 checkpoint; the prior failed policy stays preserved.

Keep the 1000→256 ELU→256 ELU→23 linear architecture and existing normalization/default/native-span/output contract. Restore all model and normalization bytes from `direct_target_continuation_v1/fit/student_head.pt` (9ceef509...), and restore its RNG state after model construction. Create a **fresh AdamW** for the changed objective; no old moments are reused. Seed the independent precomputed sampler with 20260911, leaving training RNG separate. Global lineage steps are 55,001…65,000; new optimizer steps are 1…10,000.

All 9,904 nominal absolute labels stay in the existing 15 dataset/phase cells. All 3,054 physical-response labels stay in the existing nine cells with requested counts 99/819/100 per dataset. Their arithmetic is unchanged: nominal targets normalize in float64 then cast to float32; response predictions cast individually from float32 to float64 before subtraction. Retain every difficult/clipped/zero-gain label.

The new corpus has 3,057 centers×58 tangent axes×2 signs =354,612 endpoints. The generator owns exact quaternion charts, radii, source clocks, static checks and old 23-axis overlap. No BFM/base/prior input is introduced. Match centers to nominal anchors by dataset/control identity, never an unexplained concat offset. Verify every center target/input and the full completed generation/independent qualification before loading training tensors.

Use 54 cells in fixed dataset→phase→tangent-group order: 3 datasets×3 phases×6 groups. Group widths are 3 root-position, 3 root-rotation, 23 joint-position, 3 root-linear-velocity, 3 root-angular-velocity and23 joint-velocity axes. Each update samples 16 center/axis pairs per cell, with both signs: 864 pairs / 1,728 endpoint rows. Draw centers uniformly within the cell and axes uniformly within its group, with replacement, from a standalone NumPy PCG64 generator seeded 20260911. Generate all 10,000 schedule rows before optimization; save IDs and RNG states/hash. Consume these saved indices directly with no training-time sampling, rejection or resampling.

Let `u` be float32 normalized-absolute head output and `s` the original float32 native span promoted to float64. For each pair, compute

    e± = u_probe.astype(float64) − u_center.astype(float64)
         − (T_probe64 − T_center64) / s64.

The full-state loss averages `e±²` over outputs, signs and 16 pairs, then equally across 54 cells. It does **not** divide by physical radius. The fixed group radii define the neighborhoods and equal group weights define their relative emphasis. This is a bounded finite-feedback objective, not a common-unit raw Jacobian loss. Full-state centers gather differentiable outputs already computed in the 9,904-row nominal pass.

At the initial fixed schedule row 0, run one nominal/endpoint/physical forward each, then three separate `autograd.grad` calls. Compute parameter-gradient norms in CPU float64 over the same six ordered parameter tensors. Freeze `lambda=||g_nominal||/||g_full_state||`; reject nonfinite or zero norms. Save gradients, norms, Gram/cosines, coefficient and first-order geometry. The objective for every update is `L_nominal + lambda*L_full_state + L_physical`. No weight sweep, clipping the coefficient, dynamic recalibration or projection into a descent interval. Euclidean gradient balance does not guarantee an AdamW step or later trajectory stability; geometry is reported, not misrepresented as a certificate.

Perform exactly 10,000 updates with inclusive cosine LR 1e-4→1e-5, AdamW weight decay 1e-5, `foreach=False`, `fused=False`, gradient clip 10, deterministic CUDA and no TF32/AMP. Record the original global step and new optimizer counter separately. Save per-cell losses, learning rate, gradient norm, sample IDs and all partial-call state on failure. Save ordinary-final PT/optimizer/RNG before export. No early stopping, intermediate physical evaluations or checkpoint choice.

Export the same final float32 parameter values promoted exactly to float64, using the already qualified normalization/linear/ELU execution and final float32 normalized-output cast. Use the bounded Exp/Min/Where ONNX ELU expression and manual graph export; create no FP32 ONNX release. Public input/output stay float32. Runtime reconstruction remains `default64 + promoted_existing_span64*head32`, then native float64 clamp. Final CPU64/GPU64/ORT64 comparisons must pass the original 1e-5 rad **preclamp** tolerance; failed parity preserves the final checkpoint and stops release.

Root-confirmed complete accounting:

| Stage | Rows / calls |
|---|---:|
| Calibration |14,686 forward rows /3 forwards /3 gradients|
|10,000 updates |146,860,000 forward rows /30,000 forwards|
| One full diagnostic corpus |367,570 rows:9,904 nominal +354,612 full-state +3,054 physical|
| InitialGPU32, finalGPU32, finalCPU64, finalGPU64 |1,470,280 rows /5,748 batched forwards|
| FinalORT64, per-corpus batch256 |367,570 rows /1,437 calls (39+1,386+12)|

The GPU32 passes document the training objective; the separate CPU64/GPU64/ORT64 passes qualify the execution graph. Save all backend predictions, preclamp parity and clipping drift from finalGPU32 without more inference. Only the ordinary-final model may later receive one WSL batch-one activation witness and the original fresh canonical 1,569-control lifecycle plus conditional continuous 250 hold, after separate selection/review. That future test retains original BFM startup, direct learned-phase behavior, strict every 2 ms native gates, exact prefix/query input and full failure capture; no hardware or connected real-clock claim.

Meaningful source tests should challenge unequal 54-cell/group weights, different sign/axis errors, explicit center-to-anchor mapping, float32-before-float64 subtraction, sampler boundaries/order, zero/nonfinite calibration norms, fresh optimizer versus lineage counters, immutable schedules, failure-prefix accounting and the double-internal/public-float32 graph contract. Existing qualified export helpers should be copied unchanged where possible; no new runtime framework is needed.
