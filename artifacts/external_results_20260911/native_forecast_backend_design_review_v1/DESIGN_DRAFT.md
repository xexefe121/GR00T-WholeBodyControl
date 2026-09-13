# Native manual-PD forecast backend: read-only design review

A small C++ shared object calling the already installed MuJoCo 3.2.3 library is a viable next backend experiment. The probe establishes the compiler, matching headers, native version 323, model/data pointer access and library identity. It does not establish performance or numerical parity. It loaded raw XML rather than the prepared model arrays; the future backend must receive the already prepared native model used by the qualified Python component.

No compilation, dynamics, model edits or connected controller execution occurred in this review. The selected cost-ranked controller experiment must remain independent of this backend work.

## Smallest interface and ownership

Use one `extern "C"` ABI with explicit argument/restype declarations in ctypes. Compile against the installed `include/mujoco` headers, link the exact existing `libmujoco.so.3.2.3` (SHA256 78f5455cbcbc2b4c1452f09a096aaa1d79893d63bbeb0bad93f610ddde19d264), and verify the resolved runtime library/header version. Do not rebuild MuJoCo, use another wheel, reconstruct the model from XML, replace actuators, or implement a different integrator.

Keep the prepared MjModel and one private MjData alive through strong Python references. Pass their `_address` only through typed void pointers; C++ accesses fields using the matching installed headers, never hand-coded ctypes structure offsets. Construct state, trace, PD and verdict workspaces once. Validate native dimensions/address order/gain/gear/timestep, `sizeof(mjtNum)==8`, `mj_version()==323`, integration spec 8191 and state size 291. Reject model/private-data aliasing and wrong array shape/dtype/contiguity/capacity before entering C.

Prefer passing an owned immutable 291-element snapshot and eight warning-number/lastinfo entries to C, rather than giving C the actual plant data pointer. The Python wrapper captures the actual state while holding the controller's exclusive state lock, and verifies the actual state and warnings unchanged afterward. C receives the prepared const model, private MjData, snapshot, bounded target(s) and preallocated output buffers. Thus it has no reason or API to write the actual plant.

A single-target function with maximum 50 steps is the smallest first implementation; four calls add little foreign-call overhead compared with 200 Python per-step loops. A fixed up-to-four-target C loop can reuse exactly the same function/workspaces later without changing candidate ordering or scoring. Do not combine backend qualification with candidate-selection changes.

ctypes `CDLL` releases the GIL. Strong references do not prevent another thread from mutating the model, snapshot or actual plant: require explicit nonconcurrent use and the existing synchronous controller lock. MuJoCo physics callbacks are global, not model-local; assert the expected unchanged callback configuration and do not install Python control/sensor callbacks or alter error/warning handlers as an optimization. See [Python ctypes](https://docs.python.org/3.11/library/ctypes.html) and [MuJoCo 3.2.3 globals](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIglobals.html). Direct C calls do not automatically inherit the Python binding's exception interception. A fatal MuJoCo error must fail the isolated benchmark/worker; never install an error callback that returns and continues an invalid simulation.

## Restore exactly, then execute the same operations

For every candidate, reproduce the qualified Python component's sequence:

1. `mj_setState(private, snapshot, 8191)`; copy all eight warning `number` and `lastinfo` fields.
2. `mj_forward` on private data only. Reject changed warnings just as the current component does.
3. `mj_setState` again and restore warnings again, recovering ctrl and qacc_warmstart after forward. Read back and compare all 291 elements. Do not replace this with an assumed equivalent `mj_copyData` path in the first experiment.
4. Record initial qpos/qvel/time/warnings and apply the initial predicates before a physics step.
5. For each 2 ms step, execute the operation order below, call the same `mj_step`, accumulate an independent `expected += 0.002`, record the post-step state/command/actual force/warnings, then inspect and stop at the first violation.

For each native joint j, use separate double operations:

```text
position_error = target[j] - private.qpos[7+j]
proportional = position_error * kp[j]
damping = private.qvel[6+j] * kd[j]
requested = proportional - damping
command = maximum(requested, -effort[j])
command = minimum(command, effort[j])
private.ctrl[j] = command
```

Do not fuse multiply/subtract, replace division with reciprocal multiplication, use single precision or fast-math, reorder state updates, or substitute a servo model for manual native PD. Start with a fixed `-O2 -fPIC -shared -ffp-contract=off -fno-fast-math` build; preserve signed zeros, infinities and NaNs. No LTO or CPU/compiler option sweep. GCC documents that [contraction can create fused operations and fast-math changes IEEE assumptions](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html). These flags preserve ordinary separate arithmetic but do not prove that C libm and NumPy reductions are identical.

The target has already been checked finite and bounded; native PD effort saturation is the existing operation, not candidate repair. For finite values the comparison clamps are straightforward. Do not blindly substitute `fmin/fmax` for NumPy minimum/maximum when NaNs are possible: their NaN rules can differ. Match the existing qualified component's propagation and predicate order.

## Predicate and report semantics that must remain exact

Preserve the existing reason order: nonfinite qpos/qvel; invalid quaternion when state is finite; measured joint bounds >1e-6; joint speed ratio >1; actual generalized actuator force ratio >1+1e-9 after a step; root height <.25 or tilt >1.2; nonzero warning counts; nonfinite time or accumulated-clock error >1e-10. Warning lastinfo is copied/compared and retained but is not independently a failure in this component. Do not accidentally import the different hybrid helper's tighter lastinfo/bitwise-clock policy.

The initial step does not reject initial actual-force magnitude/nonfiniteness; the current `_assess` applies that gate only after a step. A nonfinite state sets bound/speed/tilt metrics to infinity and appends their associated reasons in the existing order. The first-failure `worst_joint_index` is the first argmax of position excess, even if the failure is speed: do not improve its meaning in a backend performance change.

Repeated expected time must use one double addition per executed step. The separate ideal-clock diagnostic is `abs(time - (start + step*0.002))`, with its original multiply and add order. Maximum/minimum accumulators need the current Python ordering and NaN behavior; `std::max`/`fmax` are not interchangeable for this purpose. Record every warning number and lastinfo at initial and post-step samples. MuJoCo's own instability reset/warning must remain visible, be rejected at the same sample, and never be treated as a successful recovered forecast.

The nontrivial bit-exact boundary is math-library behavior. Installed NumPy `linalg.py` uses `x.dot(x)` then `sqrt` for the current four-component quaternion norm. A scalar C accumulation, pairwise vector reduction or `mju_norm4` is not established equivalent. Tilt uses the existing NumPy scalar square/add/clamp/arccos expression. `std::acos`, an algebraic cosine-threshold comparison, or another square/reduction path can differ by an ULP. Even when verdicts agree, full metric bytes can differ. Preserve the exact supported operations or demonstrate parity; do not loosen the gates or substitute a numerical tolerance to make tests pass.

Strict stop after the same sample is implementable inside C without per-step Python callbacks in principle. The availability probe alone does not establish that a hand-written C predicate has that property. Gate deployment on exact predicate and trajectory parity below. If near-boundary quaternion/tilt tests disagree, stop this implementation attempt: reproduce the pinned NumPy math path, or explicitly retain an authoritative ambiguity check while the private state is paused. Such a fallback is a distinct, measured implementation choice; claiming zero callbacks or universal bit-exactness before resolving it is unwarranted.

## Full-horizon generation is a different accounting contract

A kernel may compute all 50 private steps and let Python locate the first failing saved sample afterward. This is useful for a bounded backend experiment, but is not strict early-stop. It must return both `computed_steps=50` and `checked_prefix_steps=k`, plus `first_failure_step=k`, `computed_beyond_first_failure=50-k`, the full raw arrays, and final computed warning/state information. Initial rejection needs `checked_prefix_steps=0`, with the actual computed count still disclosed.

Do not slice a full computation at k and report `physics_steps=k` as though no later native steps happened. Full-horizon timing includes those later steps. An unstable extra suffix may generate additional warnings or a fatal error; it remains private but cannot be hidden. Accepted cases still require all 50 predicates to pass. This mode can reduce crossing overhead, yet it does not remove Python predicate overhead if `_assess` is still called 50 times. Vectorizing those predicates also needs its own exact reduction-parity proof.

## One bounded proof before any controller connection

Freeze one C++ source, one build command, compiler/header/library identities, shared-object hash, wrapper, all inputs and the exact result schema before executing. There is no policy/horizon/controller sweep.

Use the existing 232 cases and the newly frozen 132 filtered-branch cases, preserving their fixed targets and horizons. No fresh policy or expert query is needed. Compare all native qpos/qvel/command/actual-force/time/warning-number/lastinfo samples and first-failure indices/reasons against their saved Python reference traces. Compare full metrics, initial and final integration, computed and checked counts, and actual caller state/warnings unchanged. Reuse the same private MjData across alternating successful/failed cases to expose stale ctrl/warmstart/warning state. This adds only the candidate backend dynamics; the existing saved reference need not be rerun unless a discrepancy requires it.

Add pure-array predicate tests without physics: exact and adjacent representable values around bound1e-6, speed1, effort1+1e-9, quaternion1e-10, tilt1.2, root height.25 and clock1e-10; NaN/Inf in each relevant field; nonzero warning counts versus lastinfo alone; signed-zero/tie argmax behavior; initial-force exclusion; malformed/aliased buffers and invalid targets. Match the Python predicate's reasons, index and nonfinite metric semantics. An initial failure must execute zero steps in strict-stop mode; a mid-horizon failure must execute exactly k.

Retain one fixed-order, warmed-but-no-best-of benchmark including snapshot capture, warnings/state restoration, checks, output copies and ctypes overhead. Report per-candidate and four-candidate total p50/p95/max, plus computed counts under current contention. The cost-ranked controller additionally needs policy inference, candidate scoring and actual control work timed: faster prediction alone cannot establish a 20 ms deadline, recursive feasibility or tracking.
