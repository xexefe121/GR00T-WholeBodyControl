# BFM-conditioned MPC residual: initial result

This experiment is rejected as a full-body tracking controller. It preserves the frozen BFM feedback-v2 base and adds a bounded learned joint-target correction, but its first closed-loop tests worsen leg tracking and violate actual joint limits. No hardware qualification is claimed.

The accepted training data consist of 2,000 actual physical controls from four successful MuJoCo 3.2.3 trajectories: the nominal 3-second-source MPC replay and three initial perturbations. Thirteen other physical trajectories remain preserved and excluded. No synthetic state-plus-K labels were used. The teacher's unclipped command was reconstructed exactly from its saved plan, actual observed state and correction clipped to ±0.1 rad. The BFM actor was then evaluated on that same observed state and the preceding expert **preclip combined action history**, using position gain 1, yaw gain 2 and horizon 8.

Network inputs contain measured proprioception, preceding combined action, current BFM target, and native/full-original29 goals at offsets `[0,1,2,4,8,16,24,37]`. This requires an explicit 38-frame, 0.74-second received-goal buffer. Clip identity, source frame index, time and MPC states/gains are absent from the input. The source timeline is unchanged. The actor is a zero-initialized 1069→256→256→23 ELU MLP with a ±0.25 rad tanh output. Native PD runs at 500 Hz with original effort and physical model settings.

The successful teacher commands differ substantially from frozen BFM commands at those states: absolute residual p95 0.3408 rad, p99 0.6793 rad and maximum 1.8100 rad. A 0.25 rad envelope cannot reproduce 8.18% of joint labels. This approximation was declared before training. A one-thread CPU fit of 1,000 steps took 6.7 seconds and reached held-out target RMSE 0.08658 rad. That fitting score did not establish control quality.

| Closed-loop 500-control probe | Completion | Actual range excess | Source leg RMSE | Root p95 |
|---|---:|---:|---:|---:|
| Zero residual, nominal | 500/500 | 0 rad | 0.1533 rad | 0.3501 m |
| Learned residual, nominal | 500/500 | 0.006366 rad | 0.2458 rad | 0.3357 m |
| Learned residual, held-out initial perturbation 8 | 418 controls + 4 substeps | 0.011764 rad | 0.4181 rad | 0.3637 m |

The held-out perturbation was a successful physical teacher rollout, but the learned controller failed after only 68 complete source controls. Nominal hand/arm tracking improved somewhat while leg tracking worsened. Full policy inference, including BFM goal/actor and residual features/head, had p95 approximately 8–9 ms in these short local tests. Timing does not rescue the behavioral failure.

Detailed immutable reports, traces and checkpoint snapshots are under `E:\codex_sonic_runtime\mpc_student_20260910\bfm_residual_labels_3s_v1` and `E:\codex_sonic_runtime\mpc_student_20260910\bfm_residual_fit_3s_r025_v1`. The zero checkpoint exposed for independent referee parity is `E:\codex_sonic_runtime\mpc_student_20260910\bfm_residual_zero_v1.pt`.

The independent root referee later confirmed every zero-head residual is exactly zero. It also found a floating-point arithmetic-order difference between this prototype's precomputed action scale and the root referee's sequential multiply/divide expression: initial target difference 1.39e-17 rad, growing over 500 controls to maximum qpos difference 1.07e-6 and qvel difference 3.34e-5. Therefore bit-exact trajectory parity with the independent root referee is not claimed. The paired zero/learned results above use the same local evaluator. This rejected prototype was not retrained.

The subsequent full native-engine saved-K audit produced **zero eligible trajectories out of 17**. The nominal trajectory completed 1,417 controls including all 667 source controls, but its left ankle roll exceeded its actual range by 0.001824873 rad. All eight initial perturbations and all eight observation-noise cases failed before the full source completed. No full-teacher expert dataset was created. These results concern replay of a fixed saved plan and gain sequence; they do not establish how an MPC controller that replans on every perturbed measured state would perform.

Full physical traces and failures are under `E:\codex_sonic_runtime\mpc_student_20260910\teacher_native323_full_perturbed_v1`. Its original collector report included return/standing samples in the aggregate tracking slice. The immutable correction sidecar `source_phase_metrics_v2.json` uses source controls 350:1017, reproducing nominal leg RMSE 0.11215 rad and root p95 0.13865 m. Eligibility and all short three-second student results were unaffected. Future collector/evaluator code now stops source metrics at the declared phase boundary.

Merely increasing student fit duration or correction radius is unsupported by these results. A robust controller would need closed-loop expert correction on states the student actually visits, with successful physical execution rather than unverified local gain labels.
