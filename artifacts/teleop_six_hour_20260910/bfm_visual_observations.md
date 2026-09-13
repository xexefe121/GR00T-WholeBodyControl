# BFM native23 visual comparison

Rendered requested native23 poses beside **recorded actual MuJoCo qpos** for PICO and walk002. Both panels share one fixed camera, original world coordinates, and a 0.5 m grid. No root alignment, pose adjustment, time warp, policy inference, or new physics occurs in rendering. Left panel is explicitly kinematic; right panel is the saved physical simulation response.

The source/actual indexing is checked numerically: saved qpos boundary `i` corresponds to source frame `10+i`. Differences reproduce the evaluator's root and joint errors to 1e-12. Report, trace, and source hashes are checked before and after rendering.

## What follows and what fails

- Both recorded rollouts remain upright through their full requested lifecycles. Arms and both legs change with the reference; early frames show useful pose response. This is stronger than an arm-only or static-standing demonstration.
- Accurate global tracking still fails. Walk002 root error is 0.576 m at source 4 s and 1.057 m in final standing. Its source-phase heading error p95 is 63.65 degrees; early source heading is about 62 degrees while simulation is about 2 degrees.
- PICO source-phase heading error p95 is 143.55 degrees; maximum is 179.73 degrees. Its unwrapped simulated heading accumulates to approximately 500 degrees over the 115.6 s source while the reference stays near its initial heading with transient turns. Actual root travels far outside the intended local path.
- At source 4 s, PICO has root error 0.227 m and leg-joint RMSE 0.154 rad. The broad pose resemblance at that instant does not establish later full-body tracking.
- Existing range reports show measured joint-range excess of 6.65 mrad for PICO and 3.96 mrad for walk002. Full-lifecycle completion is not a strict joint-limit or teleoperation qualification pass.

Checked whether walk002 entry rotation was absent from supplied angular velocity: acquisition quaternion yaw changes 61.83 degrees; integral of supplied world angular velocity Z is 61.78 degrees. The reference includes that turn command. PICO entry yaw is approximately zero, so its later drift is a separate failure.

## Outputs

Each `bfm_*_v1/visual_comparison_v1` directory contains:

- Side-by-side MP4 at sampled 25 fps and original playback speed. Walk002 includes full lifecycle. PICO uses explicitly named early, middle, and late/return windows with original timestamps displayed.
- Six-sample contact sheet and an early-motion PNG.
- Root-path and absolute-heading plot, with no alignment or resets.
- Render receipt containing original input hashes and camera settings.

Numerical heading measurements: `bfm_visual_observations.json`.

The native23 desired geometry is shown, rather than absent-joint original29 hand/head intent. These artifacts visualize saved simulation; they are not new evaluation results or hardware evidence.
