# Saved PICO ankle dynamics: fixed diagnostic, no training or deployment

Previous goal turn completed the fixed foot-objective continuation and both
full evaluations. No job remains live. Final1000 fails at30.60s source while
its torso stays within30cm; parent500 completes69.34s with worse matched-prefix
tracking. That contrast changes the next action from another reward-only run
to measuring solved ankle mechanics. Earlier range-only training diagnostics
are retained and are not rerun unchanged.

Reintegrate stored torques through control1880 for parent500 and final1000 on
the unchanged native23 MuJoCo model/physics. Require exact original pre/post
qpos/qvel at every2ms step. Record only final100controls/1000substeps (source
28.60–30.60s) for detailed force analysis, with full prefix identity verified.
These diagnostic prefixes are not new complete-motion qualification trials.

Read the just-solved, preintegration force solution after each existing
mj_step. No added forward solve on actual data. Verify observer preserves the
integration state. Partition generalized forces into actuation, passive, bias,
external and constraint groups; split solved floor contacts by physical foot.
Use complete inertia to expose coupled acceleration, not scalar torque/mass.
Force components hold the existing contact solution fixed and are not claims
about what contacts would do after a control change. Retain actual velocity
increments separately from continuous-time forward acceleration.

Reproduce final1000's existing eight-candidate range rejection from its saved
last state/action, using the existing zero-warmstart probe, original target
codec,20ms horizon and0.0019rad reserve. Capture candidate forces and predicted
violations; do not apply a rejected candidate to any rollout or robot.

All sources, parents and failures remain unchanged. No new rewards, gains,
limits, policy weights, state resets in a running replay, or hardware transport.
No automatic controller/training extension. Observational data and tests are
diagnostic only; source fidelity, complete motion, timing, standing return and
live headset/hardware requirements still apply to the full goal.
