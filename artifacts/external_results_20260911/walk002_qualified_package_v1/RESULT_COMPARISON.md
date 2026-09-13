# Qualified walk002 offline hybrid

All667 source metrics exactly match the qualified original MPC run. That original run failed terminal quiet speed gates. Switching to the known BFM terminal controller at original control1117 passes both the original300-control terminal window and separate250-control hold. Root independently reproduced all16,670 native samples exactly.

| Quiet metric | Original MPC terminal | Hybrid terminal | Separate hold |
|---|---:|---:|---:|
| Root XY p95 (m) | 0.005336 | 0.009697 | 0.010423 |
| Original yaw p95 (deg) | 0.147604 | 2.656799 | 2.805199 |
| Root speed p95 (m/s) | 0.094038 | 0.000427 | 0.000248 |
| Joint speed p95 (rad/s) | 1.095768 | 0.008891 | 0.001841 |
| Joint speed max (rad/s) | 5.577074 | 0.009584 | 0.001895 |
| Tilt max (rad) | 0.009266 | 0.067964 | 0.067921 |

Video uses recorded physical states only: declared native23 reference on the left, recorded physics on the right, one fixed world camera. The full33.34 seconds includes original28.34-second lifecycle and separate5 seconds. Exact source-end20.34s, controller-handoff22.34s, lifecycle-end28.34s and final33.34s frames are included. Visual sampling is normally100ms with shorter boundary intervals; timestamps stay on original2ms grid. Final displayed endpoint lasts an explicit2ms.

This is an offline saved-MPC-prefix plus actual-terminal-BFM hybrid. Recorded-command replay recomputes native physics from saved targets without model inference. Neither artifact establishes live received-Pico control, realtime MPC or hardware readiness. Original MPC quiet failure remains archived.
