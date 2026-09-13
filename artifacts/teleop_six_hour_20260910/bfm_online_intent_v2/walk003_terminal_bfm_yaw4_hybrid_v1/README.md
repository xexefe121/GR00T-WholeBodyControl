# Walk003 hybrid: full recorded source and tested terminal standing pass

This separate simulation completes all 1,569 original lifecycle controls, including every one of the 819 source controls, and passes all unchanged 15 recorded-source gates. It passes the separately declared final-3-second quiet-standing diagnostic both within the original lifecycle and during a distinct subsequent 5-second hold. This is one nominal recorded-motion result, not live teleoperation or hardware qualification.

The physical robot starts at the declared v4 reference frame 10 and continuously executes the original MPC targets through all source and return controls. At control 1,269 (25.38 s), only terminal standing changes to native BFM control. The original native BFM standing goal, position gain 1 and horizon 8 remain fixed. This single follow-up variant changes terminal yaw correction gain from 2 to 4; the angular-velocity correction cap remains 0.8 rad/s. A saved derivation proves exactly one numeric AST node changed in the goal method, with all other goal operations unchanged. Shared controller/helper files were not edited.

All preceding 12,690 physics steps and switch state are bit exact against the passing original producer and yaw2 trial. Independent review verified every state, history, previous action, target conversion, source clock and extension boundary across all 1,819 controls. Before switching, applied MPC targets supply effective normalized previous actions. After switching, the separate BFM history advances once per actual control using raw actor output times 5, as in the published policy contract. There are no resets, state copies after initialization, root assistance forces or private physical rollouts.

Both original lifecycle and separate extension retain zero native joint-bound excess and zero engine warnings. Original source intent metrics remain exactly unchanged; `recorded_source_audit_v2.json` passes with `--require-pass`. The original lifecycle maximum hardware speed ratio is 0.807809 and effort ratio is 1.0.

Final 3 seconds, scored with the same thresholds declared before either hybrid trial:

| Diagnostic | Original lifecycle tail | Separate 5-second hold tail | Threshold |
|---|---:|---:|---:|
| Original root XY error p95 | 0.01421 m | 0.01498 m | ≤0.05 m |
| Original heading error p95 | 3.094° | 3.231° | ≤5° |
| Root linear speed p95 | 0.000435 m/s | 0.000289 m/s | ≤0.05 m/s |
| Maximum-over-joints speed p95 | 0.007530 rad/s | 0.001664 rad/s | ≤0.5 rad/s |
| Maximum joint speed | 0.008153 rad/s | 0.001696 rad/s | ≤2 rad/s |
| Maximum tilt | 0.06859 rad | 0.06809 rad | ≤0.15 rad |

Final maximum joint speed is 0.002517 rad/s at the original lifecycle end and 0.001469 rad/s after the extension. Original recorded-target standing had final speed 0.492944 rad/s and a last-3-second peak of 7.83417 rad/s. This hybrid reduces motion while changing the standing pose: v4 root 3D error p95 is 0.04480 m in the original tail, versus 0.01532 m previously. The original native BFM root error p95 is 0.01891 m. Yaw2 passed the original lifecycle's quiet diagnostic but failed the extension heading limit at 5.325°; yaw4's extension passes without relaxing that threshold.

`post_lifecycle_hold_5s` is a separate continuous extension after the original 31.38 s lifecycle. Its controls do not replace, pad, slow or crop original source frames. Only this additional 5-second duration was tested; indefinite standing is not claimed.

Evidence includes `report.json`, `trace.npz`, `comparison.json`, pretrial `provenance.json`, switch input/receipt archives, source snapshots, goal-derivation receipt, unchanged qualification audit, and the independent history/quiet-standing review. The associated new video preserves all 36.38 s and explicitly marks the controller switch and lifecycle/extension boundary. The older nominal walk003 video remains unchanged and shows its original standing jitter.

MPC walking remains offline, using at least 740 ms reference packets and up to 760 ms raw-pose support, with original planning p95 10.095 s per 100 ms commitment. This terminal controller improvement does not establish received-only operation, sensor-only feedback, robustness at arbitrary stops, real-time planning or robot deployment readiness.
