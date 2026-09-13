# Recovery witnesses before the late PICO ankle failure

The unchanged original-goal BFM controller supplies one useful late recovery candidate: from actual control **3740**, source **67.8 s**, its 30-control rollout completes all **300 native 2 ms steps**, with zero joint-range excess, speed ratio 0.19905, effort ratio 1 and zero warnings. It reaches source 68.4 s, beyond the old trajectory's 68.17 s ankle failure. This is a 0.6-second feasibility witness, not a complete recovery or full-source pass.

The actual producer trajectory was reconstructed from the declared v4 reference frame10 through 37,550 physical steps. Every qpos, qvel, applied torque, measured BFM history and previous action matched the stored producer exactly. Branches deep-copy the actual `MjData`, retaining time and solver warm-start state. No branch resets the physical state, applies root forces, changes the floor or clips actual joint positions.

| Actual source start | Original-goal BFM | Explicit standing fallback |
|---|---|---|
| control3700 / 67.0 s | H30 passes, ends67.6 s before the old failure | Joint bound fails at140 ms |
| control3720 / 67.4 s | Joint bound fails at38 ms | Joint bound fails at18 ms |
| control3740 / 67.8 s | H30 passes, ends68.4 s beyond the old failure | Joint bound fails at32 ms |
| control3755 / 68.1 s | Joint bound fails at82 ms | Joint bound fails at72 ms |

The standing alternative uses the known initial-standing frame10 template, rigidly placed at the current measured root XY and heading, with its known standing height and zero goal velocities. Its only policy gain change is yaw2 to yaw4. It requires no unknown final source pose or future sensor packet, but it explicitly interrupts source tracking. All four standing trials fail strict joint-bound feasibility; it is not a qualified fallback.

The original-goal trials preserve horizon8, position1/yaw2 feedback and pre-clipping future BFM action history. They require the existing 0.74 seconds of goal packets, conservatively 0.76 seconds of raw pose support. That is sufficient for the observed control3740 H30 witness; no minimum preview or longer-term recovery guarantee follows. At control3700 the 0.6-second horizon does not even span the original failure 1.17 seconds later.

All complete branch trajectories are scored under the unchanged v4 reference, all-joint state margin0.05/weight2000, relative-foot weight400 objective. The control3740 fresh-seed cost is **3890.6218804002515**, matching its original producer log within 1.4e-12. The old cost-only selection chose a shifted-plan cost of **867.824696065224**. This supports checking candidate physical feasibility before cost selection; the saved records do not contain that older shifted plan's complete H30 trajectory, so its specific feasibility is not inferred here.

The existing helper's private fresh proposal and the branch retaining the full actual simulator state generate bit-exact targets in all eight cases over the compared interval. Thus a private warm-start reset does not explain these particular failures. Historical normalized MPC actions remain explicitly unclipped and exceed the BFM actor's nominal +/-5 output range; that observation alone is not a causal diagnosis.

`report.json` binds all sources and lists each branch, abort point, warning/physical checks and comparable full-horizon cost. Failed branches retain their actual partial traces and do not receive a misleading partial-vs-full cost. Raw strict range tolerance is 1e-6 rad. No optimizer, new training, synthetic expert labels, full-source trial or real device was used.

The control3740 artifact contains `state[31,59]`, `target[30,23]`, and every physical qpos/qvel/torque/time sample. The separately exported `actual_3740_integration_state.npz` and restoration receipt provide a portable native MuJoCo integration-state fixture for the next bounded candidate-feasibility check.

The integration-state restoration receipt is `actual_3740_state_restoration_v2.json`. Its 291-value `mjSTATE_INTEGRATION` vector restores all 300 continuation qpos/qvel/torque samples bit-exactly. Initialize with `mj_setState`, run `mj_forward` to build derived fields, then reapply the same integration state once before the first physical step. This last initialization operation is necessary because `mj_forward` changes `qacc_warmstart`; there are no state writes after physics starts. The initial failed one-forward restoration script is preserved as `export_3740_state.py` and is not the qualified restoration recipe.
