# Native response-servo result

**Rejected as a teleoperation controller.** This experiment added continuous
full-range target correction around the pinned Pico controller. It did not
complete any of the four motions. No trained model, runtime default, factory
dance baseline or Pico demo was replaced. No hardware command was sent.

The native predictor uses independent MuJoCo3.2.3 data instances and the actual
joint PD, limit brake, motor limits and collision response. Each command measures
the effect of positive/negative perturbations on all23 targets, solves a bounded
correction, and physically predicts five fractions of that correction before
choosing one. All goals come from the latest received pose and backward-derived
velocity. The20ms prediction uses constant velocity; no future packets or
prepared motion-specific gains are available.

The first variant minimized root, leg, foot, hand and head tracking errors plus
target-change cost. It fell on walk002 after7.478s. A second variant constrains
the linearized six-component base velocity to the factory prediction and
checks the nonlinear predicted difference against0.01m/s and0.03rad/s limits.
Those are controller-internal prediction checks; they do not establish actual
physical safety or alter the native physical acceptance criteria.

| Motion | Requested duration | Actual physical duration | Failure |
|---|---:|---:|---|
| walk002 | 58.34s | 10.782s | fall |
| walk003 | 61.38s | 9.820s | fall |
| Pico | 160.60s | 17.812s | joint speed |
| held-out walk008 | 52.28s | 9.300s | fall |

All four used the same second-variant controller and settings. All tracking
verdicts failed. No run reached its continuous30s terminal standing hold.
Partial-motion errors cannot establish improved full-motion tracking. The
unpaced controller's p95 computation time was18.36–21.03ms, with maxima up to
40.46ms. It was not taken into independent-clock qualification or hardware.

The native prediction implementation was checked against ordinary MuJoCo for
the same held target, including end state and velocities at both feet, hands
and head. Maximum discrepancy was1.09e-19 in that case. A52-query warmed check
took a median12.02ms. Correct local prediction was insufficient for complete
closed-loop behavior.

Artifacts are under
`E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/response_servo_v1`.
`walk002` contains the first variant; `protected_walk002`, `protected_walk003`,
`protected_pico` and `protected_walk008` contain the base-preserving variant.
Every requested full rollout saved physical states, targets, per-control
prediction records and the unchanged tracking referee's report.

Implementation: `gear_sonic/native/true23_response.cpp`,
`gear_sonic/utils/g1_true23_response_servo.py`, and the optional response mode in
`artifacts/onboard_inspection_20260912/run_native_shoot_sim.py`. The mode is
experimental and is not enabled in the user-facing Pico launcher.

This result rejects the20ms local correction strategy tested here. Further work
must address the longer-term balance behavior that these local improvements
failed to preserve. No training or simulation process from this experiment
remains running.
