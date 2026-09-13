# Bounded timing sidecar proposal

The selected diagnostic question is why a recorded plant tick took23.076358ms even though it woke only37,441ns late. Saved data already proves that job1211 was created after its original deadline. It does not distinguish native work, capture/verification, allocation/GC or scheduling. This proposal collects those distinctions before choosing an optimization. No runtime integration or timed run is selected here.

## Exact preservation contract

Keep the existing producer24-file source frozen. A future reviewed derived integration may add sidecar hook calls only. Original native target/PD/mj_step, full291/packed373 capture, strict verifier, named history, previous-action ownership, mailbox/job/result bytes and counters remain unchanged. Keep one fixed epoch, every2ms deadline,20ms activation and late-rejection rule,1,819 controls/18,190 steps, all four MJB checks and setup240/plant120/preservation180/outer555s watchdogs. No GC disabling, call retries, skipped capture/checks, deadline rebasing, reset/catch-up or held-command repair.

Allocate all sidecar storage and attach its one exact GC callback before choosing the same future epoch. Record attachment/readiness in the existing outside-loop setup receipt. The callback registry's existing entries remain in their original order. Remove only this callback during preservation; keep incomplete spans/events if interrupted. No sidecar input enters native data, observation/history, commands, policy features, deadlines or admission decisions.

## Proposed sites

The probe has15 numeric phase codes. A later derived integration will mark:

1. `TICK_ENVELOPE`: Session.tick_once entry through the return after outer logging. It includes wait/poll and is a parent envelope; never add its duration to child durations.
2. `RESULT_POLL_AND_ADMIT`: PlantFoundation.poll_results, including transport polling, receipt validation, admission and existing event serialization.
3. `FIXED_WAIT`: existing wait_until call. No change to sleep or deadline calculation.
4. `BOUNDARY_SNAPSHOT`: only NativeStepper.boundary_snapshot;5 `BOUNDARY_HISTORY`: the existing history.advance and control record ownership;6 `BOUNDARY_SERIALIZATION`: payload JSON/base64, identity/hash and Job.to_bytes work. Preserve the original JOB_CREATED timestamp's exact position within serialization.
5. `JOB_PUBLICATION`: each original or BUSY-retry publication call, including its existing observer record. Preserve exactly the existing maximum10 attempts, one per eligible physics tick.
6. `PD_AND_INPUT`: NativeStepper.step validation, force checks, manual PD, target/torque ownership and ctrl assignment. `MJ_STEP` surrounds only the original `self.api.mj_step(...)` call, so PD is not mislabeled native compute.
7. `CAPTURE`: original capture_step call. `CAPTURE_OWNERSHIP`: foundation owned_evidence/validated_capture/last-capture assignment. No additional native read.
8. `VERIFY`: original verify_step call. `VERIFY_OWNERSHIP`: returned issue ownership/type check/repeated simulation-clock comparison. Strict outcome and counter ordering remain unchanged.
9. `STEP_LEDGER`: original StepRecord creation/append after the existing STEP_FINISH timestamp. `OUTER_LOG`: existing outer JSON encode/append after its original cycle_return timestamp.

This separation matters because existing STEP_FINISH and cycle_return timestamps precede some logging. The new envelope reports that remaining wall/CPU time separately; it never rewrites old timestamps or silently substitutes a new timing acceptance definition. Root must review the derived integration and saved-audit accounting before any actual run. Hook sites and source hashes are frozen in `hook_plan.json`; no hook has been inserted into the producer.

## Measurements and fixed bounds

`TimingProbe` uses injected wall, thread-CPU and process-CPU nanosecond readers. Each start/end sample reads **wall-before, thread CPU, process CPU, wall-after** in that fixed order. Preserve all four numbers; they are not simultaneous. Wall brackets expose sampling skew. Process CPU may exceed elapsed wall when other threads run; do not reject that valid observation. Only same-thread span starts/ends may pair.

The preallocated span table holds262,144 rows of16 signed64-bit fields, with a fixed16-entry nesting stack. Proposed worst case is223,727 spans:11 per attempted tick, three extra boundary phases per control, and at most10 publications per future job. No hook is planned on every individual generic event. GC table holds4,096 rows of12 signed64-bit fields. Combined table payload33,947,648bytes is allocated during setup. An overflow latches integrity failure and increments denied counters; no overwrite, silent drop or positive completeness claim. Python integer clock results and interpreter operations still allocate and add cost: preallocated storage does not imply allocation-free or real-time-safe execution.

GC callback records start/stop, generation, callback thread, currently active same-thread span, whether the callback interrupted the probe, wall/CPU brackets and stop collected/uncollectable counts. GC in another thread gets active_span=-1. Preserve unmatched starts/stops, invalid metadata, lost callback registry membership, clock faults and capacity exhaustion as incomplete instrumentation. A GC interval overlapping a stall is evidence of overlap, not automatic attribution of the entire stall.

## Failure and interpretation

For every measured call, reserve/start its record before entry. Mark returned or raised immediately after the original call, **before** reading end clocks. If end timing fails, known return status remains. An unreturned native call stays unknown/open; no capture or extra step is allowed to resolve it. Hook clock/schema/capacity faults latch sidecar evidence without masking the original operation's exception. The eventual supervisor must preserve unexpected hook failures independently, complete all still-possible original capture/verification for an already-returned native step, and stop before another native step if instrumentation has failed. Never turn hook failure into another controller/step retry.

Timing intervals include Python work and preemption. Long wall with little owning-thread CPU supports a wait/preemption hypothesis; it does not identify a scheduler or GC cause. Long thread CPU localizes CPU work only to the recorded phase. GC callbacks and CPU stamps add overhead and can perturb scheduling. A later bounded component run must report this overhead and retain all original native/command/timing failures. No actual overhead measurement, timing improvement or balance qualification is claimed by these fake tests.

The implementation is a small sidecar prototype, not a complete runtime adapter. Remaining source work before an actual run: exact hook insertion with original-exception preservation, fixed-scope counter reconciliation, outside-loop packed-file writer/manifest binding, callback lifecycle/failure accounting in the supervisor, and an independent saved sidecar auditor. All require review and a separately selected run; no optimization or run is selected by this preparation.
