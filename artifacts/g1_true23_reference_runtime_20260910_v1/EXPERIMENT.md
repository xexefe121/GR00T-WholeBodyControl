# Remove dynamics work from reference-only forward kinematics

SIM-only runtime correction, not a new tracking policy. The prior paced walking
trial missed its unchanged20ms deadline once (25.336ms control393). The current
reference path allocates a fresh MjData and calls mj_forward twice per packet,
although it reads only body positions/quaternions. This is unnecessary dynamics
work; it is not yet a demonstrated explanation of the historical outlier.

Prepare two separate native23 scratch states before subscription. Use only
mj_kinematics for reference body transforms, retaining exact arithmetic, height,
all validation, source-orientation calibration and task offsets. Actual native
model/integration, policy,23 joints, gains/efforts and fallback remain unchanged.
Preserve executed old sources and evidence; add a separate opt-in SIM path.

Before clocked trials, compare all fields of every existing002/003/008 source
packet against legacy reference FK. Require bit-exact outputs, independent
scratch states, zero actual plant writes/steps and unchanged invalid-input
rejection. Run one complete virtual walk with stage timing and require exact
qpos/qvel/time equality to the existing measured baseline. Profile legacy and
prepared FK on identical inputs; never infer total schedulability from it alone.

If preflight passes, run a fixed four-case actual50Hz localhost matrix: full
walk then EOF balance, pause, gap, malformed payload. Keep100ms freshness,
40ms startup buffer,20ms deadline and250-control balance tail. Then two additional
full-walk repetitions, regardless of first result: report ALL attempts, not a
chosen green rerun. Failures stop tracking and latch balance exactly as before.
Independently check delivery/timing/physical traces and baseline SONIC equality.

Measured pacing can qualify only these finite software runs. It cannot qualify
hard-real-time operation, ordinary-standing entry, explicit recovery, headset
sensing, foot tracking or hardware handback. Existing poor walking fidelity
remains a deployment blocker. No robot interfaces or newly downloaded clips.
