# Saved-array diagnosis of the failed nominal student

**Initial fitting error precedes closed-loop divergence.** At control 0, student and corresponding teacher-label qpos, qvel, state, history, previous action, all 1069 features, and BFM base target are bit exact. This includes goal, proprioception, arithmetic, and history inputs. The saved single-row rollout prediction differs from the saved batch prediction by at most 3.13e-7 rad, far below the observed fitting error. No new inference, training, or physics was performed.

| Joint | Target error at identical initial input | First 24 teacher-input target RMSE | First 24 actual-rollout target RMSE |
|---|---:|---:|---:|
| Left knee | -0.11326 rad | 0.13821 rad | 0.42874 rad |
| Right knee | -0.21701 rad | 0.13392 rad | 0.36200 rad |
| Left ankle pitch | -0.12167 rad | 0.22864 rad | 0.62695 rad |
| Right ankle pitch | -0.15803 rad | 0.24697 rad | 0.74244 rad |
| Left ankle roll | +0.03613 rad | 0.03837 rad | 0.12646 rad |
| Right ankle roll | +0.00158 rad | 0.02825 rad | 0.14233 rad |

The initial all-joint target RMSE is 0.10864 rad. After one 20 ms interval, actual joint position differs from the teacher by up to 0.03919 rad and qvel by 2.82361 rad/s. At this first divergent state, target RMSE is 0.15450 rad; the saved prediction on the corresponding undisturbed teacher input has RMSE 0.07127 rad. The lag buffer itself is still exact at control 1 because it contains control 0's identical sensed state, while previous action differs by up to 0.61887. Later, both state and lag histories diverge.

Across all first 24 controls, teacher-input target RMSE is 0.09391 rad and actual-rollout target RMSE is 0.31762 rad. At control 23, their target RMSE values are 0.09612 and 0.38406 rad; maximum same-time joint-position difference is 0.63754 rad. All per-control feature groups and all 23 per-joint errors are retained in report.json. Goal clocks/source-frame indices match; measured-heading goal features naturally change as the physical state changes.

The ten saved minibatch-loss samples descend from 0.006039 at step 100 to 0.002124 at step 1000, a 64.84% reduction. Eight of nine intervals descend; step 800 to 1000 drops 18.14%, with a small rise at step 900. This is continuing sampled-loss descent, not evidence of a plateau. Each observation used a different sampled minibatch, so it does not predict the result of further optimization or establish closed-loop improvement.

**No exact conflicting feature rows found** among the initial 250 demonstration controls. Distances use the saved training mean/std and exclude each row itself. The nearest-neighbor normalized RMS feature distance has median 0.02840 and p95 0.06936; corresponding target RMS differences have median 0.05592 rad and p95 0.21729 rad. Excluding all neighbors within four controls still gives target RMS differences of 0.07752 rad median and 0.23669 rad p95. Both RMS and L2 feature distances are reported; RMS averages over all 1069 dimensions and should not be mistaken for a maximum per-feature bound.

A concrete sensitivity witness: teacher controls 84 and 85 have unchanged native joint-position/velocity goal knots, maximum actual joint-position difference 0.004596 rad and joint-velocity difference 0.16237 rad/s. Their normalized feature L2 distance is 0.54797 (RMS 0.01676), but the right ankle-pitch target changes 0.91261 rad over 20 ms. Another pair, 164/165, has unchanged native joint goals, only 0.003151 rad maximum joint-position difference, and a 0.59362 rad ankle-pitch target change. The demonstrated command map therefore includes sharp local changes. These distinct inputs do not prove contradictory labels or that a smooth MLP cannot represent the map. They also do not establish that the command jitter is necessary for a successful physical controller.

**Potential oracle snapshot selected: actual precontrol 1, at 0.02 seconds.** This is the first measured physical divergence after an exact initial input. Control 0 already has a qualified teacher label; choosing control 1 captures the earliest state departure before large drift or any limit failure. selected_actual_control1.npz retains the complete integration state, actual q/dq, named four-lag history arrays, preceding student combined preclip action, completed prefix targets, and source frame 12. Its history must not silently be replaced by normalized clipped-target history. No expert query has run, and the snapshot is not an expert label. Any later oracle output still requires a qualified full remaining continuation, return, and standing hold.

The report binds all source arrays and the diagnostic script by SHA-256. Original labels, fit outputs, and failed trajectory remain untouched.
