# Physical motion-prior pilot

Status: finished at200 updates,1694.50seconds (28.24minutes). Both scheduled
native checkpoint decisions are complete. No controller promoted or extension
started. Native23 full-body teleop remains
unfinished. The received-only Pico demonstration remains pinned and unchanged.

This pilot adds an adversarial training reward over actual consecutive native23
states. It uses the state-transition discriminator, least-squares objective and
gradient penalty described in [AMP](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf).
The hypothesis is that successful physical transitions provide balance feedback
that the existing tracking terms alone have not taught. It is an experiment,
not evidence of a working controller.

## Implemented

- 9,466 consecutive physical transitions from successful prepared walk002,
  walk003 and Pico trajectories. Source files and hashes are recorded in
  `onboard_factory_firmware_v1/motion_prior_data_v1/report.json` on the artifact
  drive. No kinematic reference states and no held-out walk008 demonstrations.
- 71 features per physical state: root height, projected gravity, heading-frame
  root velocity, body-local angular velocity, all23 joint positions/velocities,
  and relative foot, hand and head positions. The discriminator sees two states.
  No desired pose, absolute XY/yaw, clip identity or motion phase is included.
- Successor states are captured after physics and before reset. Failed or
  nonfinite transitions receive zero style reward; the original failure penalty
  remains. All existing full-body, settling and native-limit terms remain.
- Discriminator256/128, Adam1e-4, eight512-sample updates per rollout, equal
  demonstration-recording weights, 50,000-transition policy replay. Loss is
  expert/policy least squares plus5 times squared feature-gradient norm.
  Bounded0..1 style reward adds weight0.5 to the original task reward.
- The actor still receives1582 causal features and emits23 targets. The
  discriminator is training-only. No new runtime requirement or future access.

## Local checks

Native MuJoCo3.2.3 and GPU MuJoCo Warp3.5 checks passed. Both retain the falling
physical successor before reset, suppress its style reward, preserve features
under world-yaw rotation/XY translation, and learn to score actual expert
transitions above synthetic perturbed states. These checks validate plumbing,
not complete-motion behavior.

The512-world GPU rollout measured7,080 controlled states/second over64 controls,
with about6.68GB GPU memory free afterward. Native16-world check measured696;
this is not a matched-size performance comparison. Training throughput is
reported separately in the pilot's `running.json`.

The first native check incorrectly edited its observation mirror without an
explicit reset. The test was corrected to use the simulator's reset/write
contract, then passed. No runtime state-projection behavior was changed.

## Bounded decision

Completed run: `onboard_factory_firmware_v1/motion_prior_ppo_v1` on the artifact drive.
Local RTX3070,512 worlds,64 controls/update,200 updates maximum,40-minute wall
cap. Fixed checkpoints100 and200 each run all four complete native MuJoCo3.2.3
recordings plus30seconds standing. No paid compute or recurring automation.

Start from the preserved Pico controller with a zero-initialized all23 command
head. Twenty critic-only warmup updates. Full-range23-joint corrections plus
two foot-phase and three velocity commands; existing actual-target filter and
factory limits remain. Resets:50% complete actual-start episodes,25% expert
motion,12.5% acquisition/braking/input loss,12.5% actual standing. Initial XY
velocity varies within±0.03m/s; training command delay is2–4ms.

The GPU physics version differs from the native acceptance simulator. Fixed
native full rollouts decide whether the policy improves. No promotion on
training reward, discriminator loss or changing-policy episode completions.
Independent timing, perturbations and input loss follow only for a candidate
whose full-motion behavior warrants those tests. No automatic extension.

## First fixed checkpoint

Native checkpoint100: walk002 ends14.778s, walk00310.828s, Pico completes160.6s
with30s continuous quiet, walk008 ends8.518s. Only Pico physically completes;
all full-body tracking verdicts fail. Its full-duration root p95 improves
from0.311m to0.263m versus this pilot's unpaced baseline, but right-foot error
worsens from0.341m to0.366m. It does not meet the promotion rule.

The second fixed checkpoint was evaluated at200. Initial GPU training
throughput was5,424 controlled states/s during value warmup; subsequent full
updates typically measured about4,000–4,700 states/s, including learning.

The full checkpoint100 Pico video is rendered and visually checked:
[160.6seconds, including30seconds standing](E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/motion_prior_ppo_v1/eval_00100/pico/video_fast/pico_learned_received_unpaced.mp4).
All1607 frame timestamps match the saved physical clock. The shared camera
preserves root displacement. Captions identify failed tracking and the absence
of independent timing qualification. No new physics was run for rendering.

To replay this experimental controller, use the existing
`RUN_RECEIVED_NATIVE23_SIM.ps1` with `-TaskCommands -Clip pico -Actor` pointing to
`onboard_factory_firmware_v1/motion_prior_ppo_v1/actor_00100.onnx` on the artifact
drive. This does not change the pinned demonstration. The separate
`RUN_MOTION_PRIOR_PILOT.ps1` reproduces the bounded training recipe into a new
output directory; it does not resume or launch itself automatically.

## Final decision

| Native test | Baseline | Checkpoint100 | Checkpoint200 |
| --- | --- | --- | --- |
| walk002 | 11.006s | 14.778s | 12.982s; fall |
| walk003 | 15.038s | 10.828s | 16.034s; fall |
| Pico | 160.600s; quiet pass | 160.600s; quiet pass | 160.600s; quiet fail |
| held-out walk008 | 10.108s | 8.518s | 10.090s; joint bound |

All full-body tracking verdicts fail. Checkpoint200 Pico root p95 is0.289m,
feet0.289/0.343m, leg RMSE0.315rad, hands0.172/0.206m, head0.125m. Its physical
completion does not count as successful continuous standing. Checkpoint100
preserves quiet standing but worsens right-foot tracking and fails the walks;
it also fails the promotion rule.

The pilot processed6,553,600 control-state transitions. Its101 changing-policy
canonical training completions do not establish a fixed-controller pass and
did not transfer into the requested native full-body behavior. GPU/native
solver differences were declared before training; this run does not isolate
them from learning failure. No extra simulation, disturbance or independent
timing tests were added after these failed behavior decisions.

Recipe closed. The original pinned Pico demo and factory23 dance remain
unchanged. No training or rendering job remains running from this pilot.
