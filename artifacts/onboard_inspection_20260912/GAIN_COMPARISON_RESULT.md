# Native gain and root-velocity comparisons

Both fixed-controller comparisons finished. No new training, model promotion,
parameter sweep, hardware commands or default change. Both preserved all
original tracking and physical limits and ran complete requested durations.

The benchmark-gain adapter uses the same gains as the successful prepared
expert. It converts targets to equal unclipped torque at each measured
boundary; this changes substep feedback. Preflight on128 recorded states
preserves torque within1.07e-14Nm and preserves actual received-command history.
21 of2944 mapped target components clip to the existing native target margin.

The root-velocity variant adds one fixed gain0.5 against measured minus requested
linear velocity and world vertical angular velocity. Existing position gains,
command limits, slew, factory joint gains and alpha0.9 stay unchanged. A native
quaternion-integration check confirms the angular frame conversion within
1.73e-11rad/s. Zero gain preserves the previous command; the new term opposes
measured drift in64 checks. No independent-clock qualification is claimed.

| Variant | Fixed controller | walk002 | walk003 | Pico | walk008 | Physical completions | 30s standing passes |
|---|---|---:|---:|---:|---:|---:|---:|
| Native benchmark gains | retained_pico | 11.902s | 15.146s | 160.600s | 9.320s | 1/4 | 1/4 |
| Native benchmark gains | full_body200 | 13.186s | 12.936s | 51.396s | 13.506s | 0/4 | 0/4 |
| Root velocity gain0.5 | retained_pico | 10.012s | 14.948s | 160.600s | 10.120s | 1/4 | 0/4 |
| Root velocity gain0.5 | full_body200 | 11.232s | 14.620s | 77.558s | 15.818s | 0/4 | 0/4 |

Every full-body tracking verdict fails. Native gains preserve the retained
Pico lifecycle and hold but do not rescue the other walks or the all23 model.
Velocity damping extends the all23 Pico run from51.884s to77.558s and walk008
from8.538s to15.818s, but neither completes its lifecycle. It also loses the
retained Pico quiet hold. Longer partial survival does not qualify either
controller or make partial tracking statistics directly comparable.

The earlier experimental BoundaryMatchedDamping adapter now commits actual
native targets to received history, then converts only the separate factory
policy action memory. Its old rejected experiment was not rerun or promoted.

A separate training-label bug was fixed: quiet standing now requires stationary
root rotation, feet, hands and head as well as root translation and joints.
It changes only1–3 early acquisition controls per recording and zero source
controls. It does not explain the current failures; no retraining was launched
on that basis. Six isolated motion cases and genuine standing pass the label
check in `quiet_reference_check_v1/report.json`.

Recorded-physics comparison images were rendered and inspected at
`full_body_failure_snapshots_v1/walk002.png` and `pico.png`, with the same fixed
world camera for reference, retained baseline and all23 checkpoint200.
Root alignment and new simulation steps were not used. The images show the
loss of balance while tracking, rather than a purely numerical gate failure.

Raw comparison reports are under the two named variant folders in
`onboard_factory_firmware_v1`; each has a complete `results.json` and per-clip
reports/traces. Normal-walk firmware inspection is recorded separately in
`NORMAL_WALK_FINDING.md`.
