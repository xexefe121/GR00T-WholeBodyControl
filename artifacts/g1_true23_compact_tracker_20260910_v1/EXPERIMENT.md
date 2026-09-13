# Compact native23 full-body tracker: one bounded learner change

User target: working full-body PICO teleop within twelve hours of 21:35 Sydney,
10 September 2026 (09:35, 11 September). This is a target, not a promised or
already achieved hardware qualification. Goal remains active.

## Decision and prior evidence

Frozen SONIC decoder variants, raw-joint dropping, whole-decoder fine-tuning,
reward changes, failure-weighted resets, native124 substitution and external
ankle corrections have not passed the complete saved-motion tests. Latest
continuous learner lineage has 2.048 million transitions. Failure-phase
coverage is present; more unseen-data explanations are not supported.

Change architecture deliberately: retain the exact existing PICO/SONIC motion
references, but learn a compact native23 tracker directly. This is a distinct
motion-compatible controller, not an unchanged SONIC checkpoint, not a trained
teacher distilled from failed trajectories, and not claimed general parity.
The existing native124 comparison used a different actor/input/action setup;
it was not training this controller on the current PICO bank.

## Fixed first trial

- Actor: three 256-unit ELU layers; latest measured 75-value proprioception
  plus explicit 90-value goal (165 total). Goal includes all23 requested joint
  positions/velocities, root feedback, root orientation, original29 hand/head
  intent, both foot errors and desired/measured pelvis heights.
- Zero-initialized residual around the bounded reference target. Gaussian
  actions remain in the existing released-source raw units. Actual targets,
  motor gains, effort/velocity/range limits and physics use unchanged code.
- Policy normalization is saved with weights. Original frozen SONIC weights
  are not loaded as this tracker or altered. No 29-DoF dynamics or phantom axes.
- Same existing-PICO bank: walk002, walk003 and full115.6s PICO. walk008 remains
  optimizer-excluded, previously inspected development data, not pristine test.
- Same bounded-progress + world-quality + foot-precision objective and reset
  schedule. No changed source, timing, success threshold, terminal condition,
  physical limit, body target or per-motion recovery patch.
- PPO: gamma.99, lambda.95, clip.2, separate actor/critic grad limit.5,
  unclipped critic; 4 epochs,4 minibatches, initial LR3e-4 with KL adaptation.
  Exploration starts.20, bounded.03..50. This is a new recipe, not a one-variable
  causal ablation. No claim attributes improvement uniquely to architecture.
- Smoke:2 updates,8 envs,24 controls. Fixed main:256 envs,24 controls, at most
  2000 updates (12.288m transitions), checkpoints0/200/500/1000/2000. Main starts
  only after smoke tests and actual input/action/reward checks pass.
- Preserve learner and optimizer continuously. No automatic further extension.
  Sparse input/reward checks and per-update aggregate metrics replace storing
  every large tensor; no omission of actual full replay failures.
- Main has a four-hour wall cap and stops on nonfinite state/gradients,
  unexpected action contract or insufficient disk. Main and checkpoints stay
  in this separate folder; originals/rejected artifacts remain untouched.

## Decision gates

Evaluate checkpoint200 on complete PICO and all three walking lifecycles while
main proceeds. Further milestones must show full-source tracking progress,
not merely larger return or longer survival. Compare completion and identical
prefix/root/leg/foot/hand errors against saved parent500 and foot1000.
If a complete matrix fails, retain the failure. Do not deploy or silently relax
guards. If intermediate evidence is clearly regressive, stop the bounded run
and record its real completed optimizer count rather than declare convergence.

After full-body tracking passes: measure CPU runtime, repeated standing
entry/return and continuous-time pause/disconnect/explicit recovery. Live PICO
capture and supervised physical checks remain separate evidence requirements.
No physical robot, DDS, SSH, arming, mode transition or motor commands.

Method context: native reference-conditioned tracking follows the general
BeyondMimic/MJLab pattern, not a claim to reproduce its published results:
https://github.com/HybridRobotics/whole_body_tracking
