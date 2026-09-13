# Bounded original29-to-native23 references

Use the four `reference.npz` files in this directory for the next declared-retarget simulation experiment. They contain all 10,674 original lifecycle frames at unchanged 50 Hz timing. The final coupled leg solver uses previous-pose warm starts and a second current-source initialization, both inside the same physical and adjacent-step bounds. No source frame is dropped or slowed.

**Verified scope:** export integrity, native joint ranges, adjacent joint speeds, original torso orientation/XY, consistent body FK, and causal pose generation. **Not verified:** dynamic tracking, contact feasibility, real-time execution, or hardware operation.

Each leg step is bounded by `0.8 * native_velocity_limit * 0.02`. The joint-specific bounds are `[0.512, 0.32, 0.512, 0.32, 0.48, 0.48]` rad for each leg. Arm bounds remain 0.15 rad per step. Torso orientation and XY remain those of the original29 source; pelvis height relief remains limited to 6 cm. Leg posture weight is 0.002 toward the previous exported pose. Both initialization candidates use the same objective and bounds; the lower-cost result is retained.

| Clip | Frames | Maximum adjacent speed / native limit | Foot p95, left / right | Foot maximum, left / right |
| --- | ---: | ---: | --- | --- |
| walk002 | 1428 | 0.800 | 1.180 / 1.159 mm | 6.884 / 5.717 mm |
| walk003 | 1580 | 0.800 | 3.299 / 3.819 mm | 17.528 / 20.562 mm |
| walk008 | 1125 | 0.800 | 1.053 / 1.095 mm | 3.625 / 2.803 mm |
| PICO | 6541 | 0.800 | 0.023 / 0.501 mm | 11.110 / 28.121 mm |

All four have zero adjacent speed-limit violations, zero violations of the tighter 0.8 leg bound, and zero joint-range excess. The original walk003 frame 1106 to 1107 knee jump fell from 24.1517 rad/s to 16 rad/s; the source sample times are unchanged.

Source-phase original left hand / right hand / head p95 errors, in metres:

| Clip | Left hand | Right hand | Head |
| --- | ---: | ---: | ---: |
| walk002 | 0.003766 | 0.001436 | 0.013434 |
| walk003 | 0.006005 | 0.003301 | 0.060000 |
| walk008 | 0.001131 | 0.014573 | 0.012749 |
| PICO | 0.001750 | 0.069087 | 0.007416 |

Foot pose objectives are soft. These nonzero residuals must remain visible in evaluation. Walk003 head error reaches the declared height-relief bound. PICO right-hand error remains significant; native23 geometry and the causal arm solution do not reproduce every original29 hand pose exactly.

The selected leg solution hit its 40-evaluation budget at 3 walk002, 5 walk003, and 3 walk008 frames; no PICO leg solution hit the budget. Across both attempted initializations there were 38 / 20 / 7 / 0 failed leg attempts. All attempt status, cost, selection, and counts are recorded. PICO right-arm IK hits its 60-evaluation budget at frame 3193, with 0.11963 m hand position error. No failed sample is omitted, replaced, or marked converged.

A single previous-pose initialization was insufficient. Preserved earlier `intent_retarget_v3/pico` evidence develops a straight-knee local minimum, then large hand errors. This final version keeps the same step bounds and compares a second source-seeded solve; it restores PICO task errors to approximately the v2 geometric baseline. The failed single-initialization evidence remains separate and should not be selected for evaluation.

Validation completed:

- 25 leg, torso, and arm tests passed, including independent analytic-Jacobian checks for the coupled 13-variable leg/height solve, physical/step bound intersection, candidate selection, and visible solver-budget failure.
- `intent_retarget_v3_multistart_independent_audit.json` checks every frame. Exported body FK is exact; original torso orientation error is below 7e-16 rad; torso XY error is below 5e-16 m; world-angular interval reconstruction error is below 6e-16 rad. Reported errors, adjacent speeds, source timelines, and all recorded hashes reconstruct.
- An independent 800-frame PICO prefix, including the earlier failure onset, reproduces every pose and diagnostic array exactly. Derivatives before its endpoint are exact. The final prefix derivative uses the documented one-sided endpoint convention; the full export uses a central derivative there.
- Original models, source/native reference files, physics contract, runner, arm IK, and hand-geometry helper hashes are recorded. Runner, arm IK, hand geometry, and physics snapshots accompany each clip. Task-point convention is recorded explicitly.

Each clip contains `reference.npz`, `kinematic_diagnostics.npz`, `report.json`, `original_timeline.json`, `solver_failures.json`, and the code/contract snapshots. The audit and prefix receipts live one directory above this file. Existing v2 and failed single-initialization v3 outputs were preserved.
