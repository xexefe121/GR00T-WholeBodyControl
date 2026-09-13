Read-only assessment of the saved restoration candidate at global control 3770.
The assessed states come from `restoration_3770_v1/counterfactual.npz`, specifically
its independent native manual-PD private oracle. They are counterfactual states;
this assessment does not claim any additional actual controls were executed.

All 30 candidate controls are measured at original source frames 3781 through
3810, corresponding to goal samples 68.40 through 68.98 seconds and completion
time 69.00 seconds. Geometric metrics use the original29 root, yaw, hand and head
tasks, the declared v4 native leg reference, and root-relative feet in common
world axes. No heading alignment is used for acceptance metrics.

Observed p95 errors: original root 0.10611 m, yaw 17.5107 degrees, relative hands
0.08095 / 0.08470 m, relative head 0.05904 m, and relative feet 0.08488 / 0.08875 m.
Native leg RMSE is 0.18498 rad. These bounded-window metrics do not qualify full
source tracking.

Terminal kinematics remain concerning despite the producer's strict H30 pass:
root height 0.28343 m, downward velocity 0.34258 m/s, tilt 0.36943 rad, and left /
right ankle pitch velocities 14.841 / 7.098 rad/s. At 500 ms, downward velocity
was 0.73140 m/s. These observations do not prove that continuation must fail,
but the terminal state is not a settled hold or evidence of recoverability.

`metric_adapter` preserves the saved oracle arrays and adds explicit control /
frame metadata for the common read-only inspector. No optimizer or physical
integration ran during this assessment. `original_metrics/report.json` and
`terminal_kinematics.json` bind the original artifact hashes.
