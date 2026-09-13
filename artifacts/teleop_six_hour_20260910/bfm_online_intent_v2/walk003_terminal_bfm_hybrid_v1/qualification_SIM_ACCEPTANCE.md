# Native23 full-body simulation acceptance

These are engineering acceptance thresholds for this six-hour effort, declared
before assessing the first main residual-training milestone. They are not
manufacturer certification or authorization to command a physical robot.

Tracking must complete every original timed sample of PICO, walk002, walk003,
and held-out walk008, including supported entry and return. No resets, root
forces, hidden reference scaling, path fitting, cropped success windows or
post-hoc temporal alignment. A declared initial coordinate calibration is fine.

For every complete source phase:

- Root world-position error p95 <=0.20m; heading error p95 <=15deg.
- Each native foot error relative to pelvis, in the same world axes, p95 <=0.12m.
- All twelve leg joints together RMSE <=0.15rad.
- Original29 requested left/right hand position relative to pelvis p95 <=0.15m
  each; original head proxy <=0.10m. Report world errors and orientations too.
  Missing wrist/waist axes do not justify silently changing those source tasks.
- Actual joint positions stay within native hard bounds (numerical tolerance
  1e-6rad); actual effort and speed stay within native configured limits.

For an explicitly retargeted native reference, report root position and heading
against the original source as well as the adapted goal; original-source errors
govern the root/heading acceptance above. Any floor-clearance adaptation must
declare its per-frame transform, source support/latency and derivative changes.
It cannot change the physical floor, rewrite the simulated plant, remove source
frames or substitute adapted hand/head intent for the original tasks. Kinematic
clearance alone does not establish contact feasibility or controller stability.

Recorded-source tracking is separate from an online simulation pass. The latter
also requires a received-only input buffer with explicit latency, independently
advancing500Hz physics and50Hz control, packet fault latch, measured balanced
standing after loss, explicit rearm, and measured full-loop timing. Timing must
report p50/p95/max and every missed20ms deadline; a fast unpaced average is not
a real-time pass. Test stops at different dynamic phases, not only idle.

Ground-truth root pose may establish a controller simulation baseline, but must
be disclosed. A sensor-only simulation candidate needs an actual simulated
IMU/proprioception estimator and closed-loop tests with declared perturbations.
Recorded optical full-body data does not establish that a live Pico pipeline
can observe or reconstruct all leg motion. That is a later, explicit stage.

No candidate is ready for hardware merely because it survives or one gate
passes. Retain failed and incomplete evidence, and identify remaining gates.
