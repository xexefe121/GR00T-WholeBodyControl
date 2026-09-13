# Native23 BFM simulation with observable root feedback

All eight full physical simulation runs completed: walk002, walk003, walk008 and the full 115.6 s recorded PICO source, each with ideal sensors and a fixed bias/noise case. Actual MuJoCo pelvis IMU acceleration and gyro were sampled at 500 Hz. The estimator and outer controller received no actual root position or actual root linear velocity.

This establishes a working **simulation candidate** for observable feedback. It does not establish full tracking fidelity, physical hardware readiness, or real-time performance. Every source/lifecycle was run unchanged, including acquisition and return. There was no learned residual in these tests.

| Clip | Sensors | Completed controls | Desired-root error p95 | Leg joint RMSE | Arm joint RMSE | Estimator XY error p95 |
|---|---|---:|---:|---:|---:|---:|
| walk002 | ideal | 1417 / 1417 | 0.367 m | 0.182 rad | 0.043 rad | 0.018 m |
| walk002 | fixed bias/noise | 1417 / 1417 | 0.399 m | 0.189 rad | 0.042 rad | 0.029 m |
| walk003 | ideal | 1569 / 1569 | 0.597 m | 0.202 rad | 0.051 rad | 0.021 m |
| walk003 | fixed bias/noise | 1569 / 1569 | 0.590 m | 0.216 rad | 0.051 rad | 0.120 m |
| walk008 | ideal | 1114 / 1114 | 0.818 m | 0.289 rad | 0.127 rad | 0.031 m |
| walk008 | fixed bias/noise | 1114 / 1114 | 0.671 m | 0.289 rad | 0.127 rad | 0.080 m |
| PICO | ideal | 6530 / 6530 | 0.415 m | 0.159 rad | 0.070 rad | 0.188 m |
| PICO | fixed bias/noise | 6530 / 6530 | 0.510 m | 0.163 rad | 0.071 rad | 0.278 m |

Desired-root and joint errors cover the source-motion phase. Estimator error covers the full lifecycle, scored at matched 500 Hz sensor timestamps. These two root errors measure different things: controller tracking against its request, and estimator position against the actual physical trajectory. The relatively small walking odometry errors show that estimator accuracy alone does not fix the BFM reference-following error. Long PICO still accumulates approximately 0.19 m ideal or 0.28 m perturbed odometry error.

## Implemented files

- `gear_sonic/utils/g1_true23_bfm_imu_odometry.py`: `Native23IMUOdometry`, a reusable sensor-only class with private FK scratch state. Inputs: monotonic timestamp, native23 measured q/dq, pelvis IMU WXYZ quaternion, body gyro, and body specific force. It contains no robot transport, motion-reference input, GT position input, or GT velocity input.
- `gear_sonic/scripts/evaluate_g1_true23_bfm_observable.py`: independent simulation evaluator. It reuses source loading, BFM inference and goal transformation helpers, while supplying estimated root XY and measured heading to the goal transformation. No root/training/inference file was edited for this subtask.

The estimator assumes zero local XY and zero initial velocity while the robot starts supported and stationary. One initial IMU yaw registration defines a persistent start frame. This condition was explicitly checked against each test's prescribed initial state, rather than injecting initial truth into the estimator. Z starts at a lower-sole kinematic height and is integrated, but **only XY is validated and consumed by outer feedback**. Z is not qualified for a residual root-height feature.

The filter propagates base velocity and accelerometer bias, compensates the pelvis IMU lever arm, and rejects kinematic support candidates whose point velocity disagrees with the inertial prediction. It integrates the corrected base velocity into position. If all support candidates are inconsistent, it continues inertial propagation; it does not force a contact label or reset to the reference.

## Sensor and timing evidence

The evaluator reads the actual `imu-pelvis-linear-acceleration` and `imu-pelvis-angular-velocity` sensors at `imu_in_pelvis`, using the physical model's 2 ms timestep. It does not synthesize acceleration from truth-velocity differences. This supersedes the earlier **offline synthetic accelerometer experiment** for closed-loop evidence.

MuJoCo's sensor values returned by mj_step correspond to the pre-integration state. The evaluator pairs them with joint/IMU orientation copies from that same timestamp, then updates the estimator. At a 50 Hz policy boundary the most recent packet is 2 ms old. The actor and estimator both consume that packet; no current root XY is substituted. Joint servo torques use the simulated motor controller's ideal internal joint feedback.

The fixed synthetic noise is identical across clips: joint offset standard deviation 0.003 rad and position noise 0.0005 rad; joint velocity noise 0.01 rad/s; gyro bias `[0.001,-0.001,0.002]` rad/s and noise 0.003 rad/s; accelerometer bias `[0.03,-0.02,0.02]` m/s² and noise 0.1 m/s²; orientation noise 0.1° and yaw drift 0.05°/s; seed 260911. These are declared test perturbations, not measured G1 U2 calibration results. Encoder/IMU values are sensor-equivalent simulated measurements; there is no actual hardware connection.

Validation in `bfm_runtime_odometry_checks_v1.json`:

- Replaying 1,000 saved sensor-only samples reproduced every estimator output exactly.
- Adding a constant 1.2 rad initial world-yaw offset changed the local position estimate by at most 5.21e-18 m.
- A stale sample was rejected.
- A separate MuJoCo gyro timing test matched the pre-integration body angular velocity to 1.23e-15 rad/s.
- Earlier scratch-FK Jacobian finite differences agreed within 3.53e-7 m/s; no plant state enters that scratch FK.

## Remaining limitations

Completion is not a strict physical-joint-range pass: ideal walk008 exceeded a raw joint range by **0.008744 rad**, below the inherited 0.01 rad stop threshold. The other seven runs had zero recorded positive raw range excess. Native effort caps remained in force and all measured joint velocity ratios stayed below one. There were no fall or limit-stop events under the existing referee.

The CPU prototype is **not yet real-time ready**. Goal/actor inference p95 ranged from 20.26 to 26.37 ms, at or above the 20 ms policy period. Full PICO took approximately 146 and 153 s wall time for 130.6 s simulated lifecycle. Those measurements include concurrent-machine conditions and are not a dedicated deployment benchmark, but they do not support a real-time claim. End-to-end timing needs profiling and optimization before a live Pico stream.

A separate 5,000-sample estimator-only replay, while rendering ran concurrently, measured 0.684 ms median, 0.935 ms mean, 2.906 ms p95 and 4.162 ms p99 per update. Thus the 2 ms estimator scheduling budget also needs work. This diagnostic is saved in `bfm_runtime_odometry_timing_v1.json`.

This test does not validate physical sensor calibration, different starting velocity, IMU resets/dropout, synchronization faults, long-term drift, terrain changes, foot compliance/slip, or hardware motor handoff. The desired motions remain the recorded simulation reference corpus; a live Pico transport was not exercised. Further controller work is needed for leg/root fidelity, especially walk008. A valid estimator is a prerequisite for portable root feedback, not proof that the policy tracks every requested motion.

## Reproduction and evidence

Run from the repository root:

```powershell
python -m gear_sonic.scripts.evaluate_g1_true23_bfm_observable --output artifacts/teleop_six_hour_20260910/bfm_observable_closed_loop_recheck
```

The output defaults to all four clips with ideal and fixed bias/noise. Existing output directories are rejected. The evaluator uses position gain 1, yaw gain 2, an eight-frame reference goal window, and direct reference arm targets through physical PD; there is no residual or post-initialization physical pose overwrite.

Primary result: `bfm_observable_closed_loop_v1/summary.json`. Every case includes `report.json`, control-rate physical `trace.npz`, `sensor_only_500hz.npz`, `estimator_500hz.npz`, and separate `privileged_odometry_score_only.npz`. Root output includes source snapshots and hashes. The PICO and walk008 perturbed cases also have `visual_comparison_v1/` renders of immutable actual qpos beside desired native23 geometry using the same fixed world camera; no path recentering or time warp is applied.

Five finished videos decode cleanly: walk008's full lifecycle, PICO early/middle/late segments, and an additional four-second PICO worst-heading window. Both contact sheets, the PICO world/yaw plot, and the worst-heading snapshot were opened and inspected. At source32.08s the noisy PICO robot faces 126.4 degrees away from the desired heading, with 0.555 m root error; it remains upright but plainly does not match the pose. At walk008 source4s the root error is 0.680 m and leg RMSE 0.402 rad. These worst cases are included rather than omitted by the representative windows. Source yaw error p95 is 26.3/27.2 degrees for ideal/noisy PICO, and maxima are 111.4/126.4 degrees. Final heading errors are 7.6/15.8 degrees. See `yaw_score_only.json` and `visual_path_receipt.json`.

Renderer provenance correction: while the first rendering processes ran, the shared helper received two later import/call edits for motion overrides. The exact pre-edit source was reconstructed in a new snapshot by reverting only those two lines; its SHA256 `b5bd0cd49001b0e6e32ffb3544305160f9c746aa66716e47c11e23a63973234b` matches the earlier baseline receipt. The PICO receipt initially recorded the later on-disk source hash, so it was corrected to the actually loaded snapshot, with the original receipt retained as `provenance_erratum_original.render_receipt.json`. Walk008 already recorded the correct pre-edit hash. Both now explicitly bind the snapshot. The separate worst-heading renderer started after the edit and binds the newer helper snapshot. Trace/reference/video hashes and source alignment checks remain unchanged.
