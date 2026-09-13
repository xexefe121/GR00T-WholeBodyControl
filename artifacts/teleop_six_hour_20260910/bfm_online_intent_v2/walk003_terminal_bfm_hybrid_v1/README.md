# Walk003: recorded walking followed by BFM standing control

This separate hybrid simulation completes all 1,569 original lifecycle controls, including all 819 source controls, and passes the unchanged 15 recorded-source gates. It also passes the separately declared quiet-standing diagnostic over the final 3 seconds of the original lifecycle. It does not establish live MPC, received-only teleoperation, broad robustness or hardware readiness.

The robot starts at the declared v4 reference frame 10. It executes every preceding recorded MPC target continuously under pinned WSL MuJoCo 3.2.3 manual PD at 500 Hz. Only at control 1,269 (25.38 s), after all original source and return-ramp controls, does the controller switch to the native BFM policy with original native standing goals, position gain 1, yaw gain 2 and horizon 8. The remaining 300 original lifecycle controls use BFM. There are no state resets, root forces or private physical rollouts.

All preceding 12,690 physics steps reproduce the passing original producer bit for bit. The BFM observation state, previous action and four-sample history were independently checked at every preceding control and the switch boundary: 1,270 exact comparisons. Applied MPC targets are converted to their effective normalized actions without clipping history. At the switch, history is copied into a separate BFMHistory object. Thereafter, history advances once per control using actual pre-control state and the preceding BFM raw actor output times 5; the actual native-clipped target is recorded separately. Switch inputs, independent expected arrays, helper/observation snapshots, ONNX/runtime identities and receipts are included.

The original lifecycle has zero native joint-bound excess, zero engine warnings, maximum hardware speed ratio 0.807809 and effort ratio 1.0. All source tracking metrics remain exactly those of the original passing run because the complete source and return-ramp physics are unchanged. `recorded_source_audit_v2.json` was generated with the explicit v4 reference and `--require-pass`, which exited successfully.

Final 3 seconds of the original lifecycle:

| Diagnostic | Original recorded-target standing | Hybrid BFM standing | Declared threshold |
|---|---:|---:|---:|
| Root XY error p95 | 0.00961 m | 0.01589 m | ≤0.05 m |
| Original heading error p95 | 0.104° | 4.870° | ≤5° |
| Root linear speed p95 | 0.12952 m/s | 0.000958 m/s | ≤0.05 m/s |
| Maximum-over-joints absolute speed p95 | 2.27686 rad/s | 0.006624 rad/s | ≤0.5 rad/s |
| Maximum joint speed | 7.83417 rad/s | 0.007084 rad/s | ≤2 rad/s |
| Maximum tilt | 0.01161 rad | 0.06931 rad | ≤0.15 rad |

The hybrid's final maximum joint speed is 0.002895 rad/s, versus 0.492944 rad/s previously. Final-3-second v4 root 3D error p95 increases from 0.01532 m to 0.04629 m; original-native BFM root error p95 is 0.02084 m. This reflects a different standing pose, including the original BFM goal height. Reduced motion does not mean every pose metric improved.

Only after the complete original lifecycle passed strict physical checks, the same plant continued for a separate 5-second hold of the last original BFM standing goal. `post_lifecycle_hold_5s` contains that extension; it does not pad or replace any lifecycle frames. It retains zero bounds/warnings and very low joint speeds (last-3-second p95 0.001987 rad/s), but its original heading error p95 reaches 5.325°, exceeding the predeclared 5° limit. Therefore the extension **fails the quiet-heading diagnostic**. The threshold was not changed after seeing the result.

`comparison.json` includes all raw last-1-second/last-3-second distributions and every quiet-standing predicate for baseline, original hybrid lifecycle and separate extension. Quiet-standing thresholds were written into `provenance.json` before simulation, independently of the unchanged source acceptance gates. The prior full walk003 video shows the original recorded-target controller and its standing jitter; it does not show this new hybrid.
