# Independent review: original29 intent retarget v2

Read-only review completed September 11, 2026. No controller, retarget, training, or existing score files were edited. New audit scripts and JSON are under this artifact directory. No physics rollout or hardware connection was made during this review.

The geometric construction and frame conventions check out. The exported walk003 trajectory is nevertheless infeasible at the configured knee speed limit: adjacent samples demand 24.1517 rad/s where the native contract allows 20 rad/s. Its reported central-difference velocity hides that violation. These references remain unqualified for dynamics, as their reports already state.

## Actionable findings

1. **P1 — enforce adjacent-sample leg feasibility.** `gear_sonic/utils/g1_true23_intent_retarget.py:85` solves each frame's legs independently; line 204 reports central differences. Walk003 right knee, frames 1106 to 1107, moves -0.4830349585 rad in 0.02 seconds. Required average speed is -24.1517479 rad/s, 1.2075874 times the configured 20 rad/s limit. The maximum exported central-difference ratio is only 0.8771123. Use previous-pose bounds intersected with physical bounds, or explicitly reject a generated reference when adjacent-sample displacement exceeds velocity_limit * dt. Preserve frame timing and expose resulting task error. A central-difference-only feasibility check cannot certify these references.

2. **P2 — validate the declared override's kinematic contract.** `gear_sonic/scripts/evaluate_g1_true23_bfmzero.py:55` checks fields, shapes, finite values, FPS, and quaternion normalization. Mocked reads with every joint velocity, body linear velocity, or body angular velocity set to 1e6 are accepted; joint positions set to 1e6 with unchanged body poses are also accepted. No bad file was created and no counterexample entered physics. Validate joint ranges, adjacent-step feasibility, body FK consistency, and declared derivative construction, or require and verify a corresponding independent audit receipt. Actual four v2 exports do have exact FK and derivative consistency; the walk003 adjacent-step violation is the concrete present failure.

3. **P2 — complete evidence binding.** `g1_true23_intent_retarget.py:157` hashes the runner and arm IK but omits `g1_true23_hand_frame_tasks.py`, even though it determines task-point geometry. The task convention returned at line 145 is discarded. `inspect_bfm_tracking.py:22` reloads reference bytes from the reported path without checking their hash against rollout provenance. Include the geometry helper and task convention in the receipt, and verify input hashes before re-scoring/rendering. All existing v2 hashes and snapshots checked here match; this finding is a reproducibility gap, not a claim that these inputs changed.

4. **P3 — identify the relative metric's root convention.** `inspect_bfm_tracking.py:64` subtracts the retarget root from the original29 task point. Therefore `original_hand_head_relative_p95_m` is original task error compensated by *retarget-root* displacement, not original29 pelvis-relative articulation error. Retargeting deliberately shifts the root. Rename this field or add a separate metric using `original['source_qpos29'][frame, :3]` for the desired root. The original task world metric applies no root/heading alignment and remains an honest score.

## Verified evidence

`pytest gear_sonic/tests/test_g1_true23_intent_retarget.py gear_sonic/tests/test_g1_true23_intent_arm_ik.py -q` passed all 16 tests. Existing tests verify analytic leg and arm Jacobians against numerical derivatives, reachable pose recovery, unchanged non-target joints, torso pose reconstruction, bounded height relief, and causal arm bounds.

`audit_intent_retarget_v2.py` independently inspected **all 10,674 frames** across all four exports. It reconstructed native FK from root and joints, compared original29 source FK and task points, checked exported scalar/vector derivatives, reconstructed rotation intervals from exported world angular velocities, and verified timelines and recorded hashes.

| Clip | Frames | Central velocity max / limit | Adjacent-interval max / limit | Worst adjacent speed |
| --- | ---: | ---: | ---: | ---: |
| walk002 | 1428 | 0.504006 | 0.985549 | left knee 19.710980 rad/s |
| walk003 | 1580 | 0.877112 | **1.207587** | right knee 24.151748 rad/s |
| walk008 | 1125 | 0.545456 | 0.895800 | right knee 17.915995 rad/s |
| PICO | 6541 | 0.737779 | 0.972743 | right knee 19.454865 rad/s |

Every exported body position and orientation reproduces native FK exactly. Original29 torso orientation matches within 6.82e-16 rad; torso XY matches within 4.44e-16 m. Height changes reach the declared 6 cm bound for walk003 and PICO. There is no native joint-range excess. Arm step bounds remain 0.15 rad or less.

Exported joint and body linear velocities exactly match the declared 50 Hz central/endpoint derivative. World angular velocity uses `R_later * R_earlier.inv()` and reconstructs the measured rotation interval within 5.65e-16 rad; this is the correct world-frame convention. These are finite-interval derivatives rather than independently measured instantaneous dynamics. Central derivative export uses a following sample; the arm pose solver itself remains causal. The current evaluator already declares a 140 ms reference lookahead.

Original29 task positions reconstruct from the source model and neutral-hand task convention within 8.89e-16 m. Stored foot/task errors reproduce exactly. Source-phase original task p95, ordered left hand / right hand / head:

| Clip | Original task p95 metres |
| --- | --- |
| walk002 | 0.003800 / 0.001428 / 0.013395 |
| walk003 | 0.006066 / 0.003284 / 0.060000 |
| walk008 | 0.001124 / 0.014766 / 0.012751 |
| PICO | 0.001766 / 0.069160 / 0.007418 |

The foot SE3 objective is a weighted soft objective, not an exact constraint. For example PICO's maximum foot position errors are 1.11 cm left and 2.81 cm right; those residuals are correctly retained in its report. Root height relief also shifts torso/hand/head height; no exact torso-position claim should survive after relief.

The evaluator initializes once at reference frame 10 and records post-control pose `qpos[control + 1]` against source frame `control + 11`. The inspector removes the initial pose with `qpos[1:]` and uses `frame = control + 11`; indexing is consistent. The unchanged original phase timeline and FPS preserve all requested samples. Own-heading foot error explicitly removes headings for diagnosis, is labeled as such, and accompanies separate yaw and world task errors. No world metric resets, translation alignment, time warp, or source cropping was found.

All four original timeline dictionaries, reference output hashes, runner/arm snapshot hashes, and recorded source-file hashes match their current bytes. The independent JSON records the currently observed omitted geometry-helper hash without claiming it was present in the original receipt.

## Artifacts

- `audit_intent_retarget_v2.py`: full export audit, independent FK and angular reconstruction.
- `intent_retarget_v2_independent_audit.json`: per-clip complete numeric results and provenance checks.
- `audit_intent_motion_validation.py`: mocked-file validator counterexamples, never run in physics.
- `intent_motion_validation_counterexamples.json`: acceptance/rejection evidence.

No full-body tracking or hardware qualification follows from this review. Existing retarget dynamics reports already fail independently: walk002 root p95 is 1.792 m with 0.006715 rad maximum joint-range excess; PICO stops at control 2017 with 0.012747 rad range excess and completes only 1668 of 5780 source controls. The speed finding is additional to those measured failures.
