# Native23 fast-motion result

Updated 2026-09-12T02:17:15.165992+00:00.

Fast motion-specific state feedback completed all31.38 seconds of walk003 and an uninterrupted5-second standing hold in fresh MuJoCo physics. A50Hz cycle run also completed the same motion plus30 seconds of standing. Original full-source and nominal quiet-standing thresholds passed.

The first run's policy p95 was7.40ms, maximum13.66ms, with0 policy calls exceeding20ms. End-to-end timing is not qualified: cold startup exceeded20ms, and paced trials had deadline misses. A small initial-velocity perturbation completed physically but failed motion-end standing criteria. Stronger return gains are preserved experiments, not robustness passes.

This controller uses offline prepared motion-specific plans and current simulated state. It is not the general learned student or a live Pico controller. The newest learned student failed at5.456 seconds. Hardware remains unqualified.

Watch [full closer video](E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/follow_video_v1/full_walk003_and_continuous_hold.shared_follow_camera.mp4) or [fixed-world video](E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/video_v1/full_walk003_and_continuous_hold.fixed_world.mp4).

Run [fresh simulation](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/RUN_FAST_FULL_MOTION_SIM.ps1). See [implementation and limitations](E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/README.md), [measured result](E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/baseline_v1/report.json), and [current work](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/CURRENT.md).
