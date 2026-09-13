# Qualified-trajectory moving-label extraction

One selected collection completed **6,847 rows** using the frozen original BFM baseline on actual qualified states. No dynamics, optimizer, model fitting, normalization changes, target smoothing, or held-out walk008 data were used. Independent root label verification remains the next gate.

| Dataset | Actual controls, inclusive | Rows | Acquisition / source / return | Label SHA256 |
| --- | --- | ---: | --- | --- |
| [PICO](collection/pico/labels.npz) | 250–6229 | 5,980 | 100 / 5,780 / 100 | `28097c143fa1c8ac86c660a8945c83a59cf58bb9586778e5041fbeb98245a2df` |
| [walk002 hybrid MPC prefix](collection/walk002/labels.npz) | 250–1116 | 867 | 100 / 667 / 100 | `f19c08d17e2e801e169bdd8ad4018ae56efe7c2e23dc7a7e330921fff0379207` |

The source frame is `control + 11`. Initial controls before 250, terminal standing, the terminal BFM yaw4 portion, and separate holds supply no label rows. History reconstruction nevertheless begins at control 0, preserving the actual preceding MPC actions.

Each NPZ includes unchanged actual native23 targets, unclipped original horizon8/position1/yaw2 BFM baseline targets/actions, residuals, 1,069 input features, raw previous action23, BFM state52, flat history300 and named four-lag histories. `teacher_qpos` and `teacher_qvel` contain N+1 boundary states. The N full291 integration snapshots, repeated clock, and warning ledgers come from the independent root saved-command replay; this collector never reconstructs or advances simulator state.

The residual remains `actual_native_target - unclipped_BFM_baseline`. Previous MPC actions are the normalized actual targets, with no ±5 clamp. Feature construction and all nine reused helper files are byte-identical to the existing collector. The old normalization is copied unchanged to [existing_normalization.npz](collection/existing_normalization.npz).

[Producer report](collection/report.json) SHA256 `38d91bf7b0dca625bb24d4fa2ddff5adacc44cafa8ff3b1d7649d907b88da658` records exactly 6,847 backward and actor evaluations, zero dynamics/fitting, and 65.60 seconds after the saved-array preflight. [Bounded compatibility report](collection/compatibility/report.json) covers all 9,904 old and new moving rows, with zero exact duplicate or target/residual conflict groups. Exhaustive nearest-neighbor distances and training qualification are not included.

PICO substantially broadens the old feature coverage: all 5,980 rows have at least one component outside the prior 3,057-row min/max envelope; 5,178 contain an absolute standardized component above 6, with a maximum of 36.593. walk002 has 779 such envelope rows and 335 rows above 6, maximum 14.108. These are diagnostics, not rejection or filtering rules. No normalization was recomputed.

Twenty focused checks and the final all-row saved-array preflight passed. [Final prelaunch review](../broader_label_collection_prelaunch_review_v1/review.json) SHA256 `39fb8330207315b14073d0bfd6909ed8bf04aed26d13755abe0ff7ee8d6a12c6` binds 130 collector and 136 launch inputs. All 136 remained unchanged after the run.

The hidden wrapper's original [exit receipt](process/exit.json) has an unavailable (`null`) child exit code, which is preserved as unknown. The independent [completion verification](completion_verification.json), SHA256 `d00b7735a0cb98b25baaa02671f099ee98167e5f900aa422f6cdb4277ccd9973`, instead verifies all expected output counts, shapes and hashes, both per-clip reports, the completed-row receipt, compatibility, all input hashes, no failure artifact, and absence of the wrapper/child/WSL processes. No inference was rerun to repair this reporting issue. The completed frozen source and single-run lock remain unchanged.

The [original coverage assessment](../pico_walk002_label_coverage_assessment_v1/ASSESSMENT.md) preserves source-clock and local PICO tracking limitations. These labels expand offline training evidence; they do not qualify a fast controller, live Pico operation, or hardware deployment.
