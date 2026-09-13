# Received-only native23 motion training

The retained learned controller completes walk002 and continuous30-second
standing, including independent real-time execution with zero measured deadline
misses. Motion tracking remains inadequate. Walk003 and Pico hit native joint
bounds; walk008 completes but follows the wrong path. Full-body teleoperation
is not ready. See MOTION_RESULT.md. These commands open no robot connection.

## Run the current development controller

From PowerShell:

```powershell
& 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_CURRENT_CAUSAL_SIM.ps1' -Clip walk002 -Render
```

This uses the campaign's best completely evaluated movement candidate when one
exists, otherwise the tested standing-repaired initial actor. It executes the
recording and 30-second hold with MuJoCo3.2.3, then renders the actual saved
trajectory. The comparison camera is shared; there is no pose alignment or time
warp. Exit code2 means the evaluated controller missed a behavioral criterion;
the report and video remain available. Other clips are walk003, pico and walk008.
The unpaced evaluation does not qualify independent 500Hz/50Hz timing.

## Completed bounded campaign

`E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_run_v2`

V2 repairs transition coverage: acquisition episodes cross into source motion,
and normal braking episodes start before source motion ends. Both boundaries
were physically exercised without reset. V1 stopped at226updates; its final
evaluated actor is the best initialization, with its critic retained and a
fresh optimizer. The original deadline remains12:29:10UTC on September12.
Both corrected scheduled checkpoints regressed; training stopped early under
the accepted rule. No training process or scheduled follow-up remains active.

- 128 environments on the local RTX3070; three-hour wall cap.
- Fixed75% source motion,12.5% acquisition/braking/interruption,12.5% standing.
- Free-running source segments2→4→8seconds→full source. Advance only when at
  least90% of64 fixed actual-state starts meet training tracking/physical limits.
- Current and past received poses, robot state and actual applied history only.
  The actor receives neither reference frame index nor future samples.
- Separate terminal failure and time-limit truncation. Truncation bootstraps
  the critic on its actual final observation before the next episode resets.
- Source exploration .06rad, standing/transition .03rad, PPO KL protection
  retained. A paired64-start physical test supported the larger source noise.
- Every100updates: complete native walk002 at original speed plus30-second
  standing. Two consecutive regressions stop this configuration. A full pass
  stops this phase for the next three-clip training step.

`running.json` records the live process state. `latest_evaluation.json` links the
latest complete-motion report. `best_motion.json` records the inherited best
result or an improvement over it, and does not imply teleop readiness.
`outcome.json` exists only after completion. Write `training/STOP` to finish the
current update and stop; no automation restarts it.

The fresh-initialization training recipe can be run through
`RUN_MOTION_CURRICULUM_SIM.ps1 -WallHours 3`; the exact corrected resume command
is recorded in the campaign's launch.json. This recipe was stopped for
regression and is not being restarted automatically.

After accurate walk002 with standing: train the same controller equally on the
three training recordings, keep walk008 held out, then evaluate initial-velocity
perturbations, movement input loss/rearm and independent timing. Estimator and
real Pico input driving simulation come after those behaviors pass.
