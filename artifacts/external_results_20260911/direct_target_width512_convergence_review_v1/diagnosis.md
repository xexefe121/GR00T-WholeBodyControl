Width512 trial substantially improves saved fit; response mismatch remains. This compares71000 and81000 ordinary endpoints using the same ORT64 backend. GPU32 comparisons are retained separately. No new model or simulation calls.

Nominal MSE 0.0003139006634 → 9.525679058e-05 (-69.65%); physical response -43.79%; original full-state response -11.66%; balanced response -19.26%; selected total -39.92%. Improved cells: nominal 15/15, physical 9/9, response 54/54.

Original response/zero falls 1.049253× → 0.926889×. Balanced response/zero falls 1.330734× → 1.074390× and remains worse than zero. Balanced odd error contributes 97.77%; this establishes directional mismatch, not its sign or cause.

| Tangent group | Error change | Final error / zero |
|---|---:|---:|
| root_position | -27.57% | 1.1647× |
| root_rotation | -6.15% | 0.9772× |
| joint_position | -25.33% | 1.1529× |
| root_linear_velocity | -7.65% | 0.9025× |
| root_angular_velocity | -14.58% | 0.8277× |
| joint_velocity | -22.95% | 1.4213× |

Exact query250/control250 saved entry improves 0.0479951952 → 0.0183145673 rad RMSE (-61.84%). No first-action clipping: 0 components; largest error 0.046309 rad at left_hip_pitch_joint. First24 acquisition rows improve 0.05353460 → 0.02957155 rad. Whole100-row acquisition improves -61.33%. This uses saved teacher-state inputs; actual departed states remain untested here.

Raw native-target-clipped rows: nominal 4946 → 4707 / 9904; full_state 173614 → 164802 / 354612; physical 1500 → 1470 / 3054. These counts do not measure actual physics violations.

Nominal training loss peaks at update 239, 1.053× first-batch loss. Total preclip gradient norm maximum 0.0669163, median 0.00360097; 0 updates exceed clip10. Ramp and later window values are retained in report.json. Full-state sampled-window fluctuations do not establish a fixed-corpus plateau.

The engineering change improves representational fit while retaining the old function at initialization. It does not uniquely prove a capacity bottleneck: width,10000 more updates, ramped LR and new moments all changed together. Weakest residual sensitivity remains joint velocity, followed by root/joint position. No evidence here supports raising gradient clipping, replacing strict physics gates, or declaring a stable full-body controller.

No further fit selected. Use the already selected canonical result to decide next work. If it fails, distinguish first-action nominal error from early feedback mismatch using actual saved departures and all58 tangent groups. A concrete follow-up should target whichever residual that trace supports; these supervised improvements alone cannot choose more width, changed loss weights or broader coupled-state labels. Require measured feature+inference+target latency below20ms and original1569+conditional250 acceptance separately.

Exact objectives, all cells/phases/groups, query250 errors, clipping and training windows: [report.json](report.json).
