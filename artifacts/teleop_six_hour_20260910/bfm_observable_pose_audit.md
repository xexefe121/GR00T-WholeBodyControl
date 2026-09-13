# BFM goal feedback: observable-pose audit and offline foot-odometry probe

Audit performed September 10–11, 2026. No robot connection, DDS subscription, hardware probe, controller integration, or training was performed by this audit. The estimator remains unqualified.

## Finding

BFM's actor can run from joints and IMU. The new outer feedback currently uses MuJoCo root XY and yaw. IMU yaw offers a direct portable replacement for heading feedback after one initial heading registration. Root XY requires an additional estimator or robot-mounted external localization.

A bounded contact-sensor-free leg-odometry probe was implemented and replayed on six saved physical MuJoCo trajectories. It estimated some walking paths closely, but failed on PICO and walk008, even with ideal simulated sensors. The simple lower-foot/slow-point heuristic is insufficient to establish portable full-body XY tracking.

| Recorded physical trajectory | Duration | XY error p95 | XY error final | XY velocity RMSE |
|---|---:|---:|---:|---:|
| bfm_walk002_v1 | 28.34 s | 0.074 m | 0.029 m | 0.021 m/s |
| bfm_walk002_arms_v2 | 28.34 s | 0.144 m | 0.138 m | 0.024 m/s |
| bfm_walk003_arms_v3 | 31.38 s | 0.059 m | 0.048 m | 0.028 m/s |
| bfm_pico_v1 | 130.60 s | 1.046 m | 1.046 m | 0.103 m/s |
| bfm_pico_arms_v3 | 130.60 s | 0.419 m | 0.395 m | 0.124 m/s |
| bfm_walk008_arms_v3 | 22.28 s | 0.840 m | 0.840 m | 0.242 m/s |

These are estimator errors against actual simulated motion, not controller errors against the desired motion. No source-to-actual best-fit alignment was applied. Only the initial IMU heading and initial positional gauge were registered; all subsequent drift remains visible. Parameters were fixed across all six cases before scoring.

## Exact local sensor evidence

All paths below are relative to the repository root.

| Source | Local evidence | Implication |
|---|---|---|
| `gear_sonic_deploy/thirdparty/unitree_sdk2/include/unitree/idl/hg/LowState_.hpp:28` | `tick`, IMUState, 35 MotorState slots, remote/mode/status fields | No base position, base linear velocity, foot force, or contact field in this LowState schema. |
| `.../hg/IMUState_.hpp:24` | Quaternion, gyroscope, accelerometer, RPY, temperature | Orientation/gyro/accelerometer fields exist; this audit did not validate physical calibration, bias, update rate, or meaning of acceleration. |
| `.../hg/MotorState_.hpp:24` | q, dq, ddq, tau_est plus motor status/temperature/voltage | q/dq support FK and kinematic velocity. tau_est is a motor torque estimate, not a measured foot contact force. |
| `.../hg/SportModeState_.hpp:24` | fsm_id, fsm_mode, task_id, task_time | This G1 hg schema does not provide odometry. |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:3036` | Base quaternion WXYZ from LowState IMU; body gyro and accel copied alongside it | Existing deployment conventions treat this as the base/pelvis IMU. |
| `.../include/robot_parameters.hpp:28` | `rt/lowstate`; secondary torso IMU at `rt/secondary_imu` | Torso orientation is not the pelvis orientation when waist yaw moves. |
| `gear_sonic/scripts/pico_g1_preflight.py:30` | Explicit `TRUE23_HARDWARE_JOINT_IDS` | Reuse exact motor mapping; the first 23 slots of a 35-slot message are not the native23 joint vector. |
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:181` | Comment says ground truth, then `rt/odostate` position/velocity copied from floating_base_pose/vel | Simulated DDS odometry is privileged truth, not an implemented robot estimator. |

Upstream RoboJuDo offers UNITREE and ZED odometry interfaces. Its G1 real config defaults UNITREE; however, its Python implementation subscribes to **unitree_go** SportModeState and reads position/velocity, while this checkout's **unitree_hg** SportModeState is different. This is a reason to verify the exact service/message/firmware later, not proof that all G1 firmware lacks odometry. The current local code does not establish an available, accurate G1 U2 XY stream that continues after relinquishing the stock sport controller. [RoboJuDo environment documentation](https://github.com/HansZ8/RoboJuDo/blob/release/docs/environment.md), [implementation](https://github.com/HansZ8/RoboJuDo/blob/release/robojudo/environment/unitree_env.py), [G1 configuration](https://github.com/HansZ8/RoboJuDo/blob/release/robojudo/config/g1/env/g1_real_env_cfg.py).

## Reusable pieces, and what is missing

- `gear_sonic/utils/g1_true23_bfmzero_inference.py:128`: actor state52 already takes q, dq, quaternion, body gyro; BFMHistory adds previous observable samples/actions. No actual root translation or actual root linear velocity is required by the actor.
- `gear_sonic/scripts/evaluate_g1_true23_bfmzero.py:48`: corrected_goal needs actual XY and heading to modify the backward goal's desired velocity fields. Height remains the source's explicit target. This is an outer controller dependency, not a new actor input dependency.
- `gear_sonic/utils/g1_true23_step1b_mujoco.py:987`: hash-bound native23 model preparation. A separate MjData can run FK/Jacobians from measured joints and orientation with root translation zero. This uses a known geometric model and does not require access to the physical simulation state.
- `gear_sonic/teleop/kinematic_reference.py:23`: existing scratch-FK pattern and named ankle bodies. The class itself is reference generation, not robot odometry.
- `gear_sonic/utils/g1_23dof_task_space_retarget.py:2149`: infer_foot_contacts consumes an entire world-foot trajectory, a trajectory quantile and finite differences. It is an offline clip heuristic, not a causal sensor-only estimator; it must not be reused with simulation world-foot coordinates as an implicit measurement.
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/utils.hpp:59`: immutable timestamped DataBuffer is reusable for sample freshness and sensor synchronization.
- `gear_sonic/utils/g1_true23_root_feedback_benchmark.py:117` already lists a real state-estimator deployment dependency. Its current adapter explicitly copies privileged MuJoCo pose/velocity.

No implemented live contact-aided base estimator was found in the searched `gear_sonic`/`gear_sonic_deploy` code. The pose_estimation_server_onboard_test is a human webcam pose estimator, not robot base localization.

## Implemented offline probe

`probe_bfm_foot_odometry.py` exports exactly four arrays into a separate `sensor_only_input.npz`: native23 q/dq, WXYZ IMU quaternion and body gyro. Its FK and estimator functions receive only these sensor copies plus the fixed robot model. Free-joint position and translational velocity are opened afterward by the separate scoring function. No reference motion, simulator contacts, contact forces, or torque is consumed.

The model has four sole spheres per foot. Their fixed body points are known geometry. Scratch FK at zero root translation computes each point's start-frame offset r_i and relative velocity dr_i/dt. The initial IMU yaw defines the local frame, and initial XY is zero. Initial height comes from the lower modeled sole, a flat-ground assumption.

For a stationary contact point, v_base = -dr_i/dt. Each sample scores candidate points by relative sole height and residual speed using the previous estimated base velocity. Enter/exit hysteresis selects support, and double support receives continuous weights. Newly selected points receive touchdown anchors from the predicted pose. Persistent anchors produce p_base = anchor_i - r_i, blended with the kinematic-velocity integration. If no point is eligible, the probe explicitly logs a forced lowest-point support assumption. It cannot establish flight or simultaneous slip from these signals alone.

The optional velocity integral in each plot uses the same inferred support velocities, with no anchor correction. It is an ablation, not a second independently qualified estimator. PICO_arms_v3 final error is 0.132 m by integration versus 0.395 m using anchors: fixed anchors can themselves accumulate error as soles roll, slide, and switch contact points.

`check_bfm_foot_odometry.py` independently perturbs measured configurations with MuJoCo's position integrator and checks the point-velocity Jacobians. Maximum finite-difference discrepancy was **3.52e-7 m/s**, supporting correct gyro and free-joint conventions. A score-only identity also confirms estimated velocity error equals violation of the weighted stationary-contact assumption, to less than 4e-16 m/s.

The failures are substantive. Around 10 s in walk008, the selected support points actually move about 1.9 m/s while at least one sole is near the floor. The previous estimated velocity makes a wrong support choice self-consistent. PICO errors similarly coincide with selected-point motion above 1.5 m/s. These events often have no forced-support flag, so a low fallback count is not reliable confidence. All six traces have at least one sole within 2 cm of the ground at every sampled frame; these failures cannot be dismissed as both-feet flight. That geometry check is privileged scoring only, not an input to the estimator.

Plots for walk003_arms_v3, pico_arms_v3, and walk008_arms_v3 were opened and visually inspected. Walk003 follows the actual world path closely. PICO shows accumulating offsets; walk008 shows sharp errors around 10 and 12 s that persist after the robot settles. These pictures do not show desired-motion tracking quality.

## Practical recommendation

1. Keep a sensor-only heading-feedback comparison available: derive yaw from the pelvis IMU quaternion, subtract one initial heading registration, and transform the reference into the same frame. Preserve acquisition and return phases. Do not repeatedly reset heading to hide drift. Yaw changes are observable over this local run; absolute world yaw remains dependent on IMU convention and drift.
2. Treat current GT-XY feedback as a simulation diagnostic. This offline heuristic does not justify replacing truth and declaring a hardware-portable controller. Closed-loop behavior with the estimator has not been measured.
3. For a more credible estimator, add causal pelvis accelerometer samples at the true sensor site, timestamp them with q/dq/gyro, and use inertial propagation with bias states plus contact-aided kinematic updates. The model's pelvis site is `[0.04525, 0, -0.08339]` relative to pelvis, so raw sensor acceleration is not automatically pelvis-origin acceleration during rotation. Preserve gravity/specific-force and body/world conventions explicitly. Existing BFM traces do not contain accelerometer samples; do not fill them using hidden root velocity differences and call that a hardware measurement.
4. Use contact probability and innovations to reject inconsistent stance, with covariance inflation when contacts slide or switch. Motor tau_est and a dynamics-based contact residual may help, but require model/bias checks and are not dedicated force sensors. A reference support schedule can be a prior, never a measurement that the robot actually followed the reference.
5. Test the estimator on fresh closed-loop rollouts after sensor-only replay succeeds: full acquisition/source/return, PICO, all walking cases, heading changes, sensor bias/noise, latency and dropout. Score against truth only. Record confidence and drift without resets, and compare controller tracking with and without estimated feedback.

The contact-aided InEKF literature provides a concrete architecture and explains that absolute position and global yaw are unobservable from IMU/contact information alone. The original C++ library supports IMU propagation and kinematic/contact corrections; it is a reusable implementation candidate, not something already integrated here. [Hartley et al. paper](https://arxiv.org/abs/1805.10410), [invariant-ekf library](https://github.com/RossHartley/invariant-ekf). Learned proprioceptive contact detection is another established approach, but would require a suitable native23 dataset and validation rather than assuming its contact labels transfer. [Lin et al.](https://arxiv.org/abs/2106.15713).

If persistent room-frame XY accuracy is required beyond local leg odometry, robot-mounted visual-inertial localization or another externally referenced robot pose source is the practical path. Human Pico/headset motion is the requested reference, not a measurement of the robot's actual position. No particular camera purchase or new hardware setup is assumed by this audit.

## Evidence files

- `foot_odometry_probe_summary_v1.json`: all six metrics and source hashes.
- `foot_odometry_probe_checks_v1.json`: independent derivative test and privileged failure diagnostics.
- Each case's `foot_odometry_probe_v1/`: sensor-only input, estimated trace, separate privileged score, report, and `odometry_vs_hidden_truth.png`.
- `probe_bfm_foot_odometry.py`: estimator and replay tool, no controller integration.
- `check_bfm_foot_odometry.py`: derivative verification and scorer-only failure analysis.

No existing runner, BFM inference module, source trace, or source report was changed by this subtask. All estimator evidence assumes perfect simulator joint and IMU orientation/gyro samples; physical sensor noise, calibration, timing, and contact uncertainty remain unvalidated.
