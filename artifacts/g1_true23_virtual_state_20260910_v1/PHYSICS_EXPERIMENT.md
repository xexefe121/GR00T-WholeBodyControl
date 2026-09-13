# Separate physics-prediction hypothesis after the rejected affine predictor

Affine calibration_v1 is REJECTED: all three fit-excluded walking checks fail;
velocity error exceeds zero-fill and retained leg commands move further from
the unchanged source decoder. Its parameters will not be deployed or used for
a native23 trial. Its position/velocity consistency is comparable to source
sampling, so imposing a new finite-difference rule is not a demonstrated fix.

A different, parameter-free-in-identification test uses the already pinned
original29 MuJoCo model and source PD/effort settings as an internal predictor.
Its root and retained23 state are assimilated from current measurements; the
six absent axes evolve only inside this separate model. No native23 state,
force, limit or reference is written by that predictor. Outputs are explicitly
hypothetical source states, never fabricated physical sensors. This differs
from the earlier six-command-only memory and from affine model fitting.

Before any native23 test, replay all four complete existing source29 traces.
The predictor sees only current root/retained23 q/dq and current full source29
command. No true missing state is provided, including initialization. Require
missing q/dq predictions to reproduce original source states within1e-5; check
the source plant and gains/effort pins per recording. Walking and PICO traces
use different external effort caps, so each source-side identity must be explicit.
Any predictor mismatch or virtual missing-joint range failure blocks the
candidate. Other source29 tracking/range failures remain documented; accurate
prediction of a failed source controller is NOT successful target tracking.

Only if that causal prediction check passes, one subsequent guarded full native23
PICO trial may test whether internally consistent missing axes help transfer.
That trial must preserve actual23 dynamics, encoder references, target/effort/
range guards and the whole115.60s source plus lifecycle. Fixed internal source
effort profile: the full source29 PICO benchmark's existing YAML-derived caps.
Do not increase any native physical rating, use future source frames beyond the
existing declared920ms buffer, or substitute shadow states for measured native
states/acceptance metrics. Keep rejected outcomes. No robot commands.
