All four recordings have a feasible starting incumbent under the same settings.

Each recording starts once from its unchanged v4 reference frame 10, with the
exact native MuJoCo 3.2.3 model and 500 Hz manually capped PD. The three candidate
sequences are the runner's initial reference targets, its original archived BFM
targets, and fresh BFM targets generated from this actual initial state. Fresh
BFM uses the original native goals, position gain 1, yaw gain 2, horizon 8, and
zero initial history. The optimization goal remains v4 plus the common
0.05 rad / 2000 predicted-state margin and relative-foot weight 400.

All 12 candidate rollouts completed all 30 controls / 300 physics steps:

| Candidate | PICO / walk002 cost | walk003 / walk008 cost |
|---|---:|---:|
| Initial reference targets | 123.171407 | 123.171407 |
| Recorded BFM targets | 137.723248 | 135.434159 |
| Fresh BFM targets | 132.215065 | 132.215065 |

Every rollout had zero actual joint-range excess and zero engine warnings.
Across all cases, maximum native speed ratio was 0.1684371 and effort ratio
0.8415987. Maximum cumulative clock error was 4.44e-16 seconds. Feasibility-first
minimum-cost selection chooses the initial reference sequence in every case.
Repeated costs reflect the common initial standing prefix, not evidence that
the four source motions have equivalent difficulty.

This checks only the first 600 ms. Source motion starts at control 350. No
optimizer was run, no full-source or lifecycle success is claimed, and no
recursive-feasibility guarantee follows from these initial witnesses.

`report.json` binds script, source snapshot, goals, archived target traces and
runtime identities. Each clip directory contains all 2 ms states, applied
torques, targets, warning counters/lastinfo and clocks for all three candidates.
Inputs and historical producer artifacts were not changed.
