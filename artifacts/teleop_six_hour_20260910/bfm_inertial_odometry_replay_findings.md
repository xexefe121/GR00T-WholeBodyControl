# Accelerometer-aided offline replay findings

This follows `bfm_observable_pose_audit.md`. The earlier joints/gyro/orientation-only estimator failed on PICO and walk008. This experiment adds a declared synthetic accelerometer and a causal velocity/bias Kalman filter with inertial rejection of incompatible support hypotheses. It remains unqualified.

| Saved physical trajectory | Earlier foot-only XY p95 | Accelerometer-aided ideal XY p95 | Fixed bias/noise XY p95 | Fixed bias/noise final XY |
|---|---:|---:|---:|---:|
| walk002_v1 | 0.074 m | 0.026 m | 0.068 m | 0.057 m |
| walk002_arms_v2 | 0.144 m | 0.029 m | 0.050 m | 0.052 m |
| walk003_arms_v3 | 0.059 m | 0.018 m | 0.120 m | 0.120 m |
| walk008_arms_v3 | 0.840 m | 0.047 m | 0.098 m | 0.093 m |
| pico_v1 | 1.046 m | 0.107 m | 0.107 m | 0.068 m |
| pico_arms_v3 | 0.419 m | 0.123 m | 0.199 m | 0.202 m |
| pico_yaw_only_arms_v1 | Not run | 0.143 m | 0.250 m | 0.252 m |

The settings are identical across every case; there is no per-clip tuning. These are estimator errors against the actual MuJoCo trajectory, not desired-motion tracking errors. The new probe removes the previous persistent anchor correction and integrates corrected base velocity. That avoids accumulated errors from treating rolling/sliding sole points as indefinitely fixed anchors.

The emulator computes central finite differences of **recorded world root velocity at 50 Hz**, then rotates acceleration minus gravity into the IMU frame and adds the pelvis-site lever-arm terms from angular velocity/acceleration. It writes a separate synthetic sensor file. It does not read root position. The estimator consumes only this sensor file, joint geometry, and a zero-initial-velocity assumption. All seven source traces had zero initial world velocity according to the separate scorer.

This synthetic accelerometer is not a recorded physical sensor. Its central difference uses a future velocity neighbor inside the offline emulator. The estimator itself is causal: rerunning the first 1,101 sensor samples reproduces every saved estimator array exactly, for both ideal and perturbed input. Specific-force/lever-arm reconstruction matches the declared central-difference acceleration to 7.11e-15 m/s². The separate estimator does not receive either velocity neighbor or actual root position/velocity.

Noise-free finite-difference acceleration alone can be an overly favorable test. We therefore include one fixed synthetic perturbation: joint offset standard deviation 0.003 rad, joint position noise 0.0005 rad, joint velocity noise 0.01 rad/s, gyro bias `[0.001,-0.001,0.002]` rad/s and noise 0.003 rad/s, accelerometer bias `[0.03,-0.02,0.02]` m/s² and noise 0.1 m/s², orientation noise 0.1°, and yaw drift 0.05°/s. These values are a stated stress case, not measurements of this G1's sensors. The fixed RNG seed is 260911.

The pure-inertial ablation drifts substantially because of sampling, lever-arm approximation, initialization and bias. Contact-aided corrections materially improve it. Long PICO still accumulates 10–25 cm errors despite good local velocity estimates. The ideal and perturbed walk008 plots show the large earlier incorrect-support jumps are reduced; the PICO yaw-only plots still show growing offsets late in the recording. Both plots were opened and inspected.

The replay result justifies a separate simulation experiment using the native MuJoCo pelvis accelerometer at the physical timestep, rather than promoting the offline emulator to a deployable controller. Closed-loop dynamics, actual sensor timing, calibration, unknown initial motion, terrain/slip, dropout, and hardware noise remain to be established. No controller or hardware was driven by this offline replay.

Evidence: `inertial_odometry_probe_summary_v1.json`, `inertial_odometry_probe_checks_v1.json`, `probe_bfm_inertial_odometry.py`, `check_bfm_inertial_odometry.py`, and each listed case's `inertial_odometry_probe_v1/` directory.
