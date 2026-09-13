# Native23 command-space exploration pilot

This bounded pilot changes where reinforcement learning explores. The previous
task-command actor had trainable foot-phase and velocity outputs, but PPO
sampled twelve independent final joint targets. It did not directly sample
those coordinated commands. That pilot finished and is not being extended.

The new sampler explores 17 latent commands before the frozen factory network:
twelve full-range joint corrections, two foot-phase offsets and three velocity
corrections. Gaussian likelihoods and the KL limit are computed in that same
17-dimensional space. Near zero, the declared standard deviations correspond
to 0.03 rad joint targets, 0.3 rad foot phase, and velocity changes of
0.05 m/s, 0.05 m/s and 0.2 rad/s. The existing nonlinear command bounds, native
joint targets and alpha 0.9 filtering remain. Upper-body targets remain
deterministic received q/dq commands. All original body tracking objectives
and native physical acceptance limits remain mandatory.

Initialization is the pinned native-trained Pico controller with a zero new
head. Deterministic exports retain the existing 1582-input / 23-target runtime
interface. Current/past received body poses and measured robot feedback supply
all inputs. Runtime inference does not sample exploratory commands.

Local run: `onboard_factory_firmware_v1/command_space_ppo_v1`, using native
MuJoCo 3.2.3 and mjbatch, 128 worlds, 64 control steps per rollout, CPU policy
training, 2–4 ms training command delay, 75% complete actual-start episodes,
12.5% transitions/input loss, and 12.5% standing states. walk008 stays held out.
Twenty value-only warmup updates precede policy updates. The cap is 200 total
updates or 45 minutes, with complete four-recording decisions at 100 and 200.
No supervised bootstrap, paid compute, recurring automation or hardware
commands are involved. Regressing checkpoints do not replace the Pico demo.

Preflight agrees with the previous zero-head ONNX export within 2.69e-7 on
32 actual native observations. All five phase/velocity latent coordinates
change coordinated leg targets while preserving upper-body outputs. The first
sampled physical step has zero canonical-world failures. This check proves
implementation behavior only, not complete-motion success.

The fresh deterministic baseline evaluation reproduced the previous result:
one physical completion, Pico at 160.6 seconds including its 30-second quiet
hold. The three walks still fail. Combined physical duration is 196.752 seconds.
This confirms the new sampling path begins with the retained behavior; it is
not a new readiness result. The pilot subsequently finished both scheduled
fixed-checkpoint evaluations; their results are below.

Implementation: `gear_sonic/utils/g1_true23_task_commands.py`,
`gear_sonic/scripts/train_g1_true23_factory_dynamics.py` with
`--task-commands --command-space-exploration`, and
`artifacts/onboard_inspection_20260912/check_command_space.py`.


## Finished result

Stopped after 200 updates in 934.44 seconds. Neither scheduled checkpoint
improved the complete-motion result. No model promotion or extension.

| Fixed checkpoint | walk002 | walk003 | Pico | held-out walk008 | Physical completions |
|---|---:|---:|---:|---:|---:|
| Initialization | 11.006s | 15.038s | 160.600s | 10.108s | 1/4 |
| 100, deterministic | 14.846s | 11.066s | 40.050s | 9.340s | 0/4 |
| 200, deterministic | 12.494s | 10.130s | 160.600s | 9.906s | 1/4 |
| 100, sampled seed 20260913 | 20.002s | 13.418s | 40.646s | 10.102s | 0/4 |

All four full-body tracking verdicts fail at every listed checkpoint. Only
Pico completes physically at initialization and update 200, including the
30-second quiet hold. Update 200 worsens Pico root p95 to 0.347m.
The 24 completions logged during training occurred with changing weights and
exploratory commands; they are not fixed-controller qualifications.

The optional sampled export appends 17 explicit standard-normal values to the
1582 received/state features. Zero-noise output matches the deterministic
export exactly; Torch/ONNX agreement is within 2.98e-7. A complete four-clip
fixed-weight test using training noise failed every recording. No seed sweep.
These unpaced tests do not qualify independent control deadlines.

Authoritative files: `command_space_ppo_v1/running.json`,
`checkpoint_results.json`, and the four recording reports in each evaluation
directory. The retained `received_pico_demo_v1/controller.onnx` is unchanged.
