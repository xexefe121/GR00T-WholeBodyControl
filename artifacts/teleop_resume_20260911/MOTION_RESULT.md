# Native23 motion push — final result, September12

**Real-time execution and continuous standing now work for one full recorded
lifecycle. Accurate, reusable full-body teleoperation is still not ready.**

The retained learned23-joint controller completed58.34seconds of walk002
acquisition, source motion, braking and a continuous30-second standing hold.
Physics advanced independently at500Hz, with received packets and control at
50Hz. All29170 physics steps completed, all source packets were consumed, and
both quiet windows passed. There were zero missed controller deadlines, zero
missed observation publications and zero physics steps finishing more than2ms
late. Controller runtime was10.68ms at p95 and15.05ms maximum.

[Watch the complete real-time comparison video](E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_final_clock_v1/video/walk002_learned_independent_clock.mp4).
[Read its measured report](E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_final_clock_v1/report.json).
The video uses actual saved states, a shared reference/robot camera and original
timing, with no pose alignment or time warp. Its tracking failure stays visible.

## What still prevents teleoperation

The real-time run's source position error is .992m at p95, versus the .20m
limit; feet, hands, heading and leg tracking also miss their limits. The same
actor's unpaced walk002 result is better, at .526m position error and .1635rad
leg RMSE, but still fails tracking. Compared with the initial standing-repaired
actor, that unpaced leg error improved31% and position error improved15%.

The remaining recordings were evaluated once with that same retained actor:

| Recording | Physical duration reached | Result |
|---|---:|---|
| walk002 |58.34s, complete|Standing passes; movement tracking fails. Real-time execution also completes.|
| walk003 |6.264s|Native joint-limit guard stops acquisition before source motion.|
| Recorded Pico |22.724s|Native joint-limit guard stops the recording during source motion.|
| Held-out walk008 |52.28s, complete|Standing passes; movement follows the wrong path.|

[Walk002 unpaced report](E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_run_v1/eval_actor_00226/report.json).
[Other three recordings](E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_final_other_clips_v1/report.json).
Partial Pico tracking metrics are not a completed-recording pass.

## Implemented and tested

- Received-only task-closure standing behavior, matching training/runtime over
  all10674 input frames. No future reference samples or per-motion gains.
- Fixed75% source,12.5% transition/recovery and12.5% standing environment roles;
  free-running2/4/8second/full-source curriculum with physical segment tests.
- Correct truncation bootstrap from the actual final state before resetting.
- Acquisition and braking episodes that cross the movement boundaries. The
  initial sampler mistakenly cut those handoffs; physical regression checks
  now exercise both crossings without reset.
- Bounded local GPU training, exported learned23-target policies and automatic
  complete native evaluations every100updates.
- Independent500Hz physics/50Hz control, persistent warmed inference, actual
  applied-command history, full-duration rendering and a simulation launcher.

The first run stopped at226updates to repair transition coverage. The corrected
run resumed its best actor and critic, with a fresh optimizer; both scheduled
checkpoints regressed, so it stopped at206additional updates. Its final export
also regressed. The original three-hour deadline was never extended.
Training is finished; the cancelled scheduled task remains cancelled.

Retained actor:
[actor_00226.onnx](E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_run_v1/training/actor_00226.onnx).
It is a development candidate, not a qualified teleop controller. The curriculum
never advanced beyond2seconds because none of the64 fixed starts met every
short-segment criterion at the measured checkpoints.

## Run the simulation

```powershell
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_CURRENT_CAUSAL_SIM.ps1' -Clip walk002 -Independent -Render
```

This runs actual independent physics, then renders the saved trajectory.
Without `-Independent`, it uses the unpaced native evaluator. Other clip names
are walk003, pico and walk008. Reports and videos are printed before exit;
exit code2 means behavioral criteria remain unmet. No physical robot connection
is opened. Backend execution and rendering were exercised; launcher syntax
was checked separately.

Current scope remains simulation with privileged robot state. The retained
candidate has not qualified initial-velocity perturbations, input loss/rearm,
the sensor estimator or calibrated real Pico input. One remaining modeling
mismatch is command delay: training applies each target immediately, while the
measured independent runner first applies new targets10–16ms after the control
boundary. Further controller work must address accurate motion, generalization
and that deployment delay before the Pico phase.
