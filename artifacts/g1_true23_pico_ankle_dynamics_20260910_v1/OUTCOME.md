# Solved ankle mechanics: diagnostic complete, no deployment qualification

capture_v2 completed both fixed parent500/final1000 prefixes. All37,600 actual
2ms states match the original traces exactly. Force captures cover2,000 actual
steps and the80 steps of all eight original rejected candidate probes.
Seven force-observer tests pass. The initial accumulator setup failure is
preserved in SETUP_FAILURE.md; it ran no replay physics. No actual state change
from the observer, no source/model/physics/policy/limit/gain changes.

Force, acceleration and constraint partition maximum residuals are respectively
4.39e-12,9.61e-12 and2.27e-13. The complete inertia is used, not a scalar
ankle inertia approximation. Components hold the solved contact forces fixed;
they are not counterfactual contact forces under different actions.

Final1000's last accepted20ms right ankle roll ends q0.248943rad,dq+1.347234rad/s
despite inward target-0.065971rad. Mean actuator torque-7.840278Nm is opposed by
right-floor-contact generalized torque+8.326149Nm. In the original rejected
next20ms nominal probe, target-0.223619774rad is already the strongest bounded
ankle command. Requested torque magnitude is14.38–15.91Nm, below35Nm motor
capacity; target authority at the unchanged PD gains is the narrower limit.
The right ankle reaches0.263795298rad, exceeding its hard limit by0.001995298rad
and the existing reserve by0.003895298rad. Left foot unloads to0N while right
normal load ends345.63N. Parent500 remains better braked at the matched time.

All eight rejected targets differ by less than9.8e-6rad. The old numerical
endpoint inset weakens, rather than strengthens, the already saturated nominal
braking target. Fixing this redundant search cannot create missing ankle-only
authority at that state. No motor saturation or missing encoder field was found.

Next distinct controller test: anticipate ankle excursions using100ms constant-
target simulation, while preserving every-joint20ms checks and all existing
physical bounds. This is earlier intervention, not the prior slow coordinated
leg search with2rad jumps. It remains an unqualified offline hypothesis. No
additional reward-only training, physical robot commands or checkpoint export.

Detailed values: capture_v2/report.json and capture_v2/window_analysis.json.
ExistingPICO, both legs and original29 intent remain the full goal. All earlier
tracking, timing and live transition failures remain unresolved. Goal ACTIVE.
