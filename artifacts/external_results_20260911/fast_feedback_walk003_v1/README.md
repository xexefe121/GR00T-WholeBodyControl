# Fast native23 full-motion feedback baseline

The controller completed all 1,569 controls of walk003 followed by 250 continuous standing controls in fresh MuJoCo 3.2.3 dynamics. All 819 source frames and both quiet-standing windows passed the existing tracking thresholds. Original timing, 23 joints, native speed/effort limits, and physical floor were retained.

This is a **motion-specific controller using plans prepared offline**, not a general learned policy or live Pico teleoperation. It uses current simulated position and velocity in the original expert feedback law. It does not overwrite the simulated robot with recorded states. The latest learned controller failed at control 272 and remains unqualified.

`compile_controller.py` packs the qualified expert's 204 plans into 1,018 feedforward targets and feedback matrices. `run_controller.py` performs a fresh rollout: original BFM standing controller, one original learned transition command, measured-state feedback through acquisition/motion/return, then BFM standing. No planning runs inside the control loop.

`baseline_v1/report.json` contains the measured full-motion result. Combined policy latency p50/p95/max was 0.322/7.401/13.659 ms, with zero policy calls over 20 ms. One full control tick at cold startup took 22.568 ms. This unpaced run is not an independent real-time plant qualification.

`run_paced_controller.py` tests 50 Hz cycles after an uncommitted policy warmup, followed by a 30-second continuous hold. It executes ten native 2 ms substeps per cycle. The plant is not independently paced; that remains a separate test.

Run another fresh simulation from PowerShell:

```powershell
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_FAST_FULL_MOTION_SIM.ps1'
# 50 Hz cycles and a longer continuous hold:
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_FAST_FULL_MOTION_SIM.ps1' -Paced
```

`video_v1/full_walk003_and_continuous_hold.fixed_world.mp4` shows the entire first successful rollout. Left: declared native reference. Right: fresh recorded physics. No time warp or state alignment.

`follow_video_v1/full_walk003_and_continuous_hold.shared_follow_camera.mp4` gives a closer view, using the same moving camera for both panels. Both videos preserve all original elapsed time.

The 50 Hz `paced_v1` run completed the full motion plus 30 seconds of continuous standing and passed tracking/quiet-standing checks, but missed 10 cycle deadlines following a 132 ms policy stall at terminal handoff. `paced_hot_v1` kept the inactive standing model warm using 72 extra noncontrol inference calls; it also completed and stood, but still missed seven deadlines. Neither is a real-time qualification.

An initial root x velocity offset of +0.03 m/s also completed physically, but ended with about 12.5 cm position error and failed the 5 cm standing-position criterion. Terminal position gains 3 and 4 were tested without changing reference timing or native limits. Both passed the nominal case; neither passed every motion-end quiet criterion in the perturbed case. These remain preserved experimental results, not robustness passes.

Remaining: independent 500 Hz plant/50 Hz control with deadlines and fault handling, perturbation tests, the other motions, and a controller accepting new motion input. No hardware authorization or readiness is implied.
