# Matched walk003: positive control passes; frozen SONIC fails

Canonical run: **pair_v2**. This is the requested matched-plant comparison, not a new controller or training run.

| Result | Prepared applied-control sequence | Frozen paired SONIC breadth25 |
|---|---:|---:|
| Physical duration | 61.380 s, complete | 20.218 s, stopped on tilt/fall condition |
| Complete source controls | 819 / 819 | 660 / 819 |
| Original root position p95 | 0.104578 m | 3.484070 m, partial source |
| Original heading p95 | 9.579782 degrees | 17.151150 degrees, partial source |
| Original relative hands/head p95 | 0.055483 / 0.061956 / 0.038916 m | 0.253669 / 0.218253 / 0.216280 m, partial source |
| Relative feet p95 | 0.099404 / 0.088051 m | 0.388285 / 0.366685 m, partial source |
| Leg RMSE | 0.087155 rad | 0.292288 rad, partial source |
| Original tracking criteria | Pass | Fail; incomplete |
| Additional 30 s quiet standing | Pass, all 15,000 samples | Not reached |

**Supported conclusion:** the preserved task is achievable in this particular prepared simulator configuration. The selected frozen SONIC pipeline fails on that same plant. This does not establish that every SONIC checkpoint is incapable, that a knee sign is wrong, or that another training run is justified.

## Positive control and task

Replayed the exact applied PD-target sequence from `native_clock_v1/braking_v5`. Every one of 30,690 freshly integrated physics steps, every applied target and every applied torque matches the historical trace exactly. This is applied-control replay, not measured-state playback or a new feedback-policy qualification. No successful-controller history is fed into the SONIC trajectory.

The historical lifecycle is unchanged: 5 s standing, 2 s acquisition, 16.38 s source movement, 2 s return, 6 s standing/proof margin, then the existing extra 30 s hold. No standing-start phase was added. Initial physical state is the historical reference state at frame 10, with one identical `mj_forward` initialization.

Original29 hand/head goals and original root/heading remain the scoring authority. Feet and joint posture use the same existing prepared native reference in both arms. The positive control passes these preserved goals; feasibility is not inherited from a substituted hand/head task. World hand/head errors are also retained in the report; the original acceptance uses their pelvis-relative errors.

The archive's historical name `braking_v5` refers to its existing prepared terminal controller. Neither the experimental shoulder filter nor sustained braking filter runs in this comparison.

## What is identical, and what differs from walk002

Both arms use the **same compiled MuJoCo model object**, all numerical overrides, PD implementation, stop conditions, effort/velocity limits, physical initial state, task and scoring. Every exposed numeric model array is saved and checked unchanged before/after the controller swap. The compiled model and all solver options are archived. The positive trace's exact reproduction separately establishes numerical agreement with the historical run.

Against the previous walk002 plant, the only differing exposed model array is `jnt_actfrcrange`: four ankles use +/-50 Nm instead of +/-35 Nm. Solver options match. The separate physical PD contract also differs: waist Kp is 300 instead of 40.17923847367; the full exact Kp/Kd/effort arrays, including rounding differences, are in `walk002_numeric_model_difference.json`. This comparison does not infer numerical equivalence from XML filenames or line endings.

The archived per-2-ms command-ID schedule is reused in both arms. Each new controller's own current or previous command is selected by that common schedule. This is an offline behavior test with recorded application timing, **not a fresh independently clocked timing qualification**. Inference times remain in the SONIC trace. Historical timing success is separate evidence.

## Frozen SONIC interface

Weights remain the hash-bound paired breadth25 encoder/decoder. The asymmetric target transform, native joint ordering, neutral offsets and action scales stay fixed; none are recomputed from prepared gains. For both knees the frozen neutral offset is 0.669 rad and scale is 0.35 rad. The complete decoding formula/constants are in `comparison_contract.json`.

SONIC history begins with ten repeated measured initial observations and the declared zero previous-action startup sentinel. Thereafter, it evolves on its own physical states and actual applied commands in the fixed decoder units. It never receives later expert states or expert history. Encoder reference positions and original hand/head goals, all decoded targets, and the actual-command feedback conversion were checked against the synchronized records.

The positive sequence is a prepared, motion-specific feasibility witness with privileged offline planning. It has no normalized decoder output or online observation history in this replay. CSV entries mark that distinction explicitly instead of inventing matching raw actions. This demonstrates task achievability, not equal causal information available to the two controller families.

## First sustained divergence

First difference meeting the localization rule is **control 292, post-control time 5.86 s, during the existing acquisition ramp**. Rule: five consecutive samples above an original tracking threshold while the positive control remains below that same threshold. This locates a problem; it does not replace full-motion acceptance.

At that boundary SONIC leg RMSE is **0.156936 rad**, versus **0.033060 rad** for the positive control. Root error is still only 0.046773 m.

| Knee quantity at control 292 | Left | Right |
|---|---:|---:|
| Reference anchor presented to policy | 0.239256 | 0.224082 |
| Reference scored at interval end | 0.192399 | 0.170960 |
| SONIC measured angle before command | 0.456745 | 0.331748 |
| Raw decoder output | -1.357654 | -0.977112 |
| Final SONIC PD target | 0.274161 | 0.360191 |
| Prepared PD target | 0.077484 | 0.272904 |

PD targets are actuator commands, not reference poses. The differing anchor and scored sample reflect the existing causal observation/control timing, with both timestamps recorded. Raw outputs pass through the frozen transform exactly; this test does not find an accidental scale recomputation or a knee sign swap.

From the **same SONIC pre-command simulator state**, a one-control probe gives leg RMSE 0.156936 with its own command and 0.142012 with the prepared command at that timestamp. Both preserve the same prior SONIC command until the common application point. The nominal fork reproduces the SONIC continuation exactly. This isolates a local command effect; it does not turn the archived command into a reusable recovery controller.

## Reference-frame repair during this comparison

The provisional `pair_v1` paired original pelvis-local hand/head goals with an adapted pelvis orientation. Those frames differ, so that run is excluded from the preserved-task conclusion. `pair_v2` pairs the original goals with their original pelvis orientation and checks reconstruction of original world-axis relative goals for every packet. Positive-control state/target/torque reproduction remains exact. The frame correction changes SONIC duration from 18.888 to 20.218 s while tracking still fails. It is a task-adapter repair, not a knee correction or policy change.

## Standing verification and scope

The positive control passes the unchanged 30-second limits on all 15,000 samples: root XY p95 0.039178 m, heading p95 2.324222 degrees, root speed p95 0.000391 m/s, joint speed p95 0.002204 rad/s, maximum joint speed 0.002663 rad/s and maximum tilt 0.074660 rad. All 1,501 overlapping three-second windows covering the extra hold pass too.

No actuator-contract crossover, training, input-loss test, sensor-estimator substitution, live Pico session or robot execution was performed. Success under this prepared plant does not transfer automatically to intended native actuation. The next diagnosis belongs in SONIC conditioning/action behavior on this demonstrated task; any crossover must keep these reference and decoding contracts fixed.

## Evidence

- `report.json`: paired outcome and separate physical/tracking/standing results.
- `comparison_contract.json`: pinned inputs, model, gains, solver, history and decoding contracts.
- `synchronized_knee_boundaries.csv`: one synchronized record per control update, including raw output, applied targets, all ten requested/applied knee torques and contact counts.
- `first_divergence.json`: first errors, complete relevant values, exact identical-state probes and full 30-second standing metrics.
- `positive/trace.npz`, `sonic/trace.npz`: complete measured states, per-step targets, torques and controller arrays.
- `*/contacts_at_first_substep.json`: contact geometry pairs/distances at each control's first physics evaluation.
- `walk002_numeric_model_difference.json`: exact numerical model and actuator differences.

Runner: `gear_sonic/scripts/compare_g1_true23_matched_walk003.py`. Analysis: `gear_sonic/scripts/analyze_g1_true23_matched_walk003.py`. Shared PD/stop implementation: `gear_sonic/native/true23_matched_step.cpp`. Runtime: MuJoCo 3.2.3, NumPy 1.26.4, ONNX Runtime 1.23.2. This code has no hardware publication path.

[Download complete evidence and pinned inputs](https://github.com/xexefe121/GR00T-WholeBodyControl/releases/tag/native23-matched-walk003-2026-09-14). [Earlier walk002 diagnosis](previous_walk002/README.md).
