# Teleop diagnosis after schedule cancellation

2026-09-12. User requested cancellation and a direct explanation of the fastest
route to working native23 full-body teleoperation. The app confirmed deletion
of `six-hour-g1-simulation-work`. All former training and evaluation jobs have
finished. No replacement schedule or training run was started.

## Actual state

Prepared motion-specific feedback can execute walk003 and a 30-second hold.
The reusable received-only controller does not track the requested movements.
Its final direct-body checkpoint fails all four source gates. It can stay
upright while taking a substantially different path: walk003 reaches 2.386 m
XY error around 14.12 seconds. Standing alone is not teleoperation.

Full actual-trajectory comparison video (no new dynamics, shared camera,
original timestamps, includes entire 30-second hold):
E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/direct_body_global200_eval_v1/walk003/video/walk003_learned_tracking_failed.mp4

## What the investigation established

1. The BFM checkpoint really has the native23 output order, but its training
   actuation differs from the enforced native contract. Hip-pitch effort was
   139 Nm versus 88 Nm; ankle effort 50 Nm versus 35 Nm. Exact joint count does
   not establish a compatible motion controller. Native limits remain fixed.
2. The previous single-policy goal initially encoded neutral pose; received
   movement entered a new adapter. Direct current-body encoding corrected that
   limitation, but its initial and trained physical rollouts still fail.
3. The output mapping hard-clipped some ankle targets before converting them
   to logits. This made the gradient through pretrained outputs zero and the
   output adapter gradient very small at those states. On 384 fixed moving
   states, 100 states contain 111 affected ankle targets.
4. The final 196-update run stayed at learning rate 1e-8 throughout. Its expert
   action error barely changed and leg tracking worsened. These observations
   do not justify another unchanged training extension. Actual motion/standing
   occupancy was not logged; claims that training was mostly standing are not
   established. Shared actor/critic gradient clipping is an additional concern,
   but critic gradient dominance was not measured.

## Concrete repair and result

SinglePolicy now has an opt-in `smooth_action_tau` parameter. It places the
residual before smooth native saturation, with ordinary nonzero gradients
outside the old clipping region. The default branch preserves previous
behavior exactly. A fresh initial direct-body candidate with tau .02 and
original pretrained weights was exported; no trained checkpoint was silently
reinterpreted and no optimizer steps were run.

All 111 affected examples have a correct inward gradient. Maximum change on
15 measured standing states is .00212 rad. ONNX target agreement is within
1.15e-5 rad. These are implementation checks, not controller success.

Native MuJoCo3.2.3 results for the repaired initialization:
- 30-second standing at 0, +.03 and -.03 m/s: all pass.
- Complete walk002 plus hold attempt: fails at 24.692 s on a native ankle
  bound. All 667 source samples had executed, but source tracking fails:
  root p95 .780 m, leg RMSE .246 rad. No full lifecycle or hold pass.

Reports live under `causal_dynamics_v1/smooth_action_initial_standing_v1` and
`smooth_action_initial_walk002_v1`. The gradient check and initial ONNX live
under `smooth_action_initial_v1`.

The trainer exposes `--smooth-action-tau .02` and optional
`--separate-gradient-clipping`, records both, and rejects checkpoint resume
when action-mapping semantics differ. These options have not yet produced a
new trained controller. Existing raw checkpoints and default exports are intact.

## Fastest next decision

No verified swap-in controller exists in the inspected local alternatives.
Prepared MPC is too slow online; its compiled gains require per-motion plans.
Sonic variants, native124, arm IK, odometry and small residuals already have
failed physical trials. Repeating those under new names does not close the gap.

Use the repaired native action path for a deliberately motion-focused dynamics
experiment, first demonstrating that it learns corrective leg/foot actions
from actual moving states and improves a complete received-only walk002
rollout plus 30-second hold. Keep all body objectives and original speed.
Require measured full-motion improvement before enlarging the run or adding
more recipes. Then the same checkpoint must pass the four-clip, disturbance,
input-loss/rearm and independent timing suite. Only then connect calibrated
Pico input to simulation using the existing receiver.

This remains controller development. No defensible teleop-ready date is
available from the current failures. No hardware commands were issued.
