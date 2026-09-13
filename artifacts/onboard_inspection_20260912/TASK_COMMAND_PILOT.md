# Received task-command controller pilot

**Finished and rejected.** Both scheduled checkpoints regressed from the
preserved Pico controller. The200-update run stopped after994.55 seconds;
no extension or promotion. The preserved Pico demo and factory dance remain
available. Full-body teleoperation acceptance is still unmet.

| Controller | walk002 | walk003 | Pico | held-out walk008 | Complete motions |
|---|---:|---:|---:|---:|---:|
| Initial preserved function | 11.006s | 15.038s | 160.600s | 10.108s | 1/4 |
| Checkpoint100 | 12.074s | 15.272s | 44.142s | 11.002s | 0/4 |
| Checkpoint200 | 11.588s | 15.084s | 40.488s | 9.928s | 0/4 |

These are complete-rollout requests evaluated without pacing; time stops at
physical failure. Initial Pico also passes its final continuous30s quiet hold.
Neither learned checkpoint reaches that hold. All body-tracking verdicts fail.
Partial-motion tracking numbers must not be treated as improved full-motion
tracking. Walk002/003 fall; learned Pico/008 exceed native joint bounds.

An input replay checked2350 saved states across all four recordings. Training
and runtime features agree within1.36e-5, with repeated runtime target error
below1.23e-6. This checks interface consistency only. Training episodes with
changing weights/noise were not reliable evidence of fixed-controller success.

The next controller retains the native-trained Pico policy and its alpha0.9
leg-target filter. A new zero-initialized head sees measured leg position and
velocity errors, pelvis-relative errors for both feet, both hands and the head,
and the existing current/past received poses and robot observations. It learns
two foot phase offsets, three factory velocity corrections and twelve full-range
leg corrections. All eleven upper-body outputs still follow received q/dq.
No future packet, recording identifier or prepared per-motion gain is an input.

The value estimator now trains independently of the actor KL guard. Twenty
initial updates fit value without changing the actor; subsequent rollouts fit
value for two full epochs before bounded actor updates. All existing body-task
rewards and native physical limits remain. Held-out walk008 never enters training.

Run: `E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/task_commands_ppo_v1`.
Native MuJoCo3.2.3 through mjbatch, 128 CPU worlds, 64 controls per rollout,
200-update cap, decisions at updates100 and200, 45-minute wall cap. Actual-start
episodes occupy75% of worlds; transition/input-loss and quiet states occupy12.5%
each. Command application is delayed2–4ms in training. Checkpoint evaluation
uses each entire original-speed recording and an additional30s standing hold.
No automatic extension or live promotion.

Initial checks passed: NumPy/native and Torch body-error features agree within
6.15e-8; initial targets preserve the retained controller within1.49e-8;
changing a learned phase changes physical joint targets; corrupting unseen
reference suffixes does not change inputs; ONNX/Torch outputs agree within2.69e-7.
The initial full Pico rollout physically completes160.6s; body tracking still
fails. Walk002 and walk003 fall after11.006s and15.038s respectively.

A three-second independent-clock integration run completed1500 physical steps,
150 controls and zero deadline misses. It ran while training was active and
does not establish full-duration standing or timing acceptance. Observed first
application latency reached6ms, beyond the training delay range; this must be
checked on a retained candidate under isolated timing before qualification.

Run an explicit experimental export with:

```powershell
& Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\onboard_inspection_20260912\RUN_RECEIVED_NATIVE23_SIM.ps1 -TaskCommands -Actor E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1\task_commands_ppo_v1\actor_00100.onnx -Clip walk002 -IndependentClock
```

Both experimental checkpoints are saved for reproduction, not recommended
controllers. The previous Pico demo remains the runnable default.
