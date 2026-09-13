# Private native bulk forecast prototype

The fixed 364-case diagnostic passed. Every one of 13,560 computed native 2 ms steps matches the independent manual-PD referee exactly: qpos, qvel, requested torque, actuator force, clock, warning count and warning last-info. The first-failure locations and reasons match all 232 original and 132 subsequent saved cases.

This component is not connected to a controller. It deliberately computes the fixed requested private horizon before running the unchanged scalar NumPy acceptance predicates, with an earlier native stop only for critical nonfinite state/force/time or engine warnings. The comparison referee uses `stop_on_failure=False`. The saved result retains every computed suffix: 12,634 steps lie in first-failure prefixes, and 926 additional private steps were computed after those first failures. `physics_steps` means actual computed steps. Separate `checked_prefix_steps`, suffix count and suffix warning arrays prevent an early-stop claim.

The C ABI uses the already loaded, pinned MuJoCo 3.2.3 library and prepared native model with a separate private MjData. It keeps manual PD operation order and saturation, with no fast math, FMA contraction or compiler vectorization. Full 291-value integration restoration and warning copies occur before computation. A function-address check confirms that the compiled loop and Python resolve the same native `mj_step`. Input/source state and warnings remain unchanged. The library never receives the actual plant data pointer.

| Horizon | Bulk p50 / p95 / max | Independent full-horizon referee p50 / p95 / max |
| --- | --- | --- |
| 20 ms | 1.41 / 1.90 / 3.17 ms | 38.81 / 42.66 / 76.71 ms |
| 100 ms | 5.79 / 7.01 / 10.94 ms | 45.45 / 49.29 / 69.04 ms |

Timing includes restoration, native rollout, strict assessment, result metadata and owned trace-storage copy. Cases ran once each, with alternating component/referee order; system contention was uncontrolled. At 100 ms, native rollout median was 3.33 ms and scalar assessment median was 2.05 ms. Four candidates plus policy/cost computation are not qualified against the 20 ms control deadline.

The separate input-boundary suite passed 23 checks without executing dynamics. It checks invalid horizons, nonfinite/out-of-range or incorrectly represented targets, external force rejection, initial position/speed/height/tilt/quaternion boundaries, existing engine warnings, caller state preservation and cleanup after an exception at native entry. It replaces the compiled entry with a spy; six valid initial boundaries reached that spy and zero native steps occurred.

Sources, compiler flags, headers, shared-library hash and all fixed-case input hashes are recorded in `build.json` and `request.json`. Output is in `results/report.json`, `bulk_traces.npz` and `oracle_traces.npz`. `input_boundary_report.json` binds the separate test source. The original early-stop component, controller trial and referee remain unchanged. Independent source review is pending. This proof covers the recorded cases and listed input checks; it is not universal numerical or live teleoperation qualification.
