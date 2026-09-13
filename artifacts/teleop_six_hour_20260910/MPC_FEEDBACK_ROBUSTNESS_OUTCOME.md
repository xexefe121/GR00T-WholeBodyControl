# Saved-plan MPC feedback robustness

The bounded feedback regularization experiment did not produce a robust native23 controller. It preserved the nominal walk when used alone, but failed both tested physical perturbations. Adding the existing measured ankle repulsion removed nominal range excess but caused a fall. No wider parameter sweep or new student fit followed.

The audited teacher is the complete MuJoCo 3.2.3 walk002 H30 plan, trace SHA256 `53e31c21ed6eceae4203221993166fa0bf7f07f28fa13e20feb8cacffccfcc6d`. The native model, original effort caps and 500 Hz physical dynamics remained unchanged. Feedback used actual measured state relative to the saved nominal state; the plant was never overwritten after initialization.

The saved final-backward gains contain large isolated sensitivities: maximum absolute gain 526,145 and 99th-percentile successive gain-matrix change approximately 37,509 in Frobenius norm. Under the declared independent measurement noise, the linear map predicts per-joint raw target-noise RMS p95 0.0606 rad and maximum 1,176 rad, before the existing ±0.1 rad correction clip. These values describe the local linear map; actual commands remain clipped.

Forward native-MuJoCo probes at nine lifecycle points compared microscopic state perturbations with larger normalized physical perturbations, executing all ten 2 ms PD steps. At acquisition control 300 and source control 500, all twelve finite probes changed the contact sequence; median normalized finite-scale derivative changes were approximately 1.03 and 0.90. At source controls 350, 500, 873 and 950, the saved K increased median one-control state error relative to the same perturbed state under the nominal command. These measurements include actual contact activation/deactivation through simulation. No ideal rigid-impact saltation matrix was asserted or substituted.

The single candidate applies an SVD cap to `K * diag(noise_std)`, capping each singular value at 0.04 rad, then transforms back to physical state coordinates. This bounds each joint's linear predicted RMS sensitivity under the declared independent noise. The final correction remains clipped to ±0.1 rad. The rule has no temporal smoothing and makes no stability guarantee. It still requires the saved plan and is not a general live teleoperation policy.

| Candidate | Nominal | Initial perturbation 1 | Measurement-noise case 9 |
|---|---:|---:|---:|
| Original K, no extra torque | 1417 controls, 0.001825 rad ankle excess | 628 controls + 1 substep, range failure | Failed in original 17-case audit |
| Noise-capped K | 1417 controls, same ankle excess and tracking | 592 controls + 8 substeps, range failure | 518 controls + 7 substeps, fall |
| Noise-capped K plus ankle repulsion | 715 controls + 3 substeps, fall; zero range excess | 625 controls, range failure | 518 controls + 7 substeps, fall |

The added ankle torque was the already declared inward spring/outward damper: margin 0.05 rad, stiffness 150 Nm/rad and damping 2 Nm/(rad/s), applied before native effort clipping. Its disturbance of the saved contact trajectory matters: the cap-only nominal replay reproduced source root p95 0.13865 m and leg RMSE 0.11215 rad, whereas adding ankle repulsion caused nominal failure. No simulator physics or effort limit was changed to obtain these results.

Evidence is preserved under `E:\codex_sonic_runtime\mpc_student_20260910`: `feedback_audit_v2.json`, `robust_feedback_r004_barrier_v1`, `robust_feedback_r004_no_barrier_nominal_v1`, and `robust_feedback_r004_no_barrier_perturbed_v1`. Version 1 of the numerical audit is retained; version 2 uses microscopic component sizes closer to the planner's finite-difference scale and reaches the same conclusion.

Useful next work would require actual replanning on perturbed states, or feedback fitted and verified against finite physical forward rollouts with changing contacts. Smoothing or capping the existing saved K did not establish a sufficient region of stable behavior.
