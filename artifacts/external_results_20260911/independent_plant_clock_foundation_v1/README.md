# Independent plant clock foundation — source preparation only

This artifact implements a deterministic scheduling and ownership core. Its tests use an injected virtual clock, scripted stepper, and try-lock stubs. It does not start a process, load a policy, install a runtime, call MuJoCo, or claim robot balance or real-time performance.

The selected design is `independent_plant_clock_design_v1/NOTE.md`. Existing frozen controllers and runtime files are not changed.

## Core contract

`PlantFoundation.tick()` executes at most one 2 ms step. The epoch has no public setter, and every nominal deadline is `epoch + index * 2,000,000 ns`. Control boundaries occur at indices divisible by ten. Waiting never rebases the next deadline. An early return from the clock stops before a step; overdue steps retain their original indices and deadlines.

At a boundary, the plant switches an already sealed local command or repeats the last actually activated command. Each control record contains nominal and actual activation timestamps, the command's original admission timestamp, target/raw-action bytes, incoming raw action, current measured terms, complete named/flat incoming history, outgoing named history, and nominal versus active window IDs. A held interval therefore exposes stale source coverage explicitly.

The pure history update matches the frozen `BFMHistory.before_update` operation: return four prior samples, then push measured terms with the incoming previous raw action. It occurs once per actual control boundary, including held intervals. Proposal attempts do not update it. The selected float32 raw action is retained even when another raw action produces an identical clipped native target. Targets are checked against supplied immutable limits; the core does not introduce another clamp or reconstruct raw action from the clipped target.

A copied job requests only the next control. It binds run/model/reference/input-epoch identity, job sequence, snapshot indices, intended window ID, snapshot/history/schedule digests and job creation time. Results must match a job that was successfully published. The plant validates and copies the result before recording authoritative receipt time. Receipt at or after the activation deadline is late. Worker timestamps cannot backdate publication. Wrong, malformed, late, duplicate, old/future-epoch and unpublished-job results are retained as rejected events.

Missing the next command latches an input fault and records held-command intervals until the fixed run end or an explicit failure. Later ordinary results cannot clear this fault. No synchronous BFM fallback, rearm, prediction injection, source-clock shift, or immediate late-command activation exists.

## Accounting and storage

Attempted, returned, captured, verified and committed step counts are separate. Expected simulation time uses repeated floating addition by `0.002`; it is checked independently of the integer wall schedule. Step/capture/verification exceptions retain their stage and known counters. A captured payload remains available if subsequent verification throws. An unreturned attempted step has explicitly uncertain mutation; it receives no returned-step credit.

Each step records nominal start/end, actual transaction start/end, active target/raw-action bytes, wake/finish lateness and wake debt. Transaction start includes boundary activation/history work; these fake-clock observations are not measured native PD start times. Finish strictly greater than the nominal end is a deadline failure; equality passes. The first deadline failure is retained after catch-up. If waking already after the next completion deadline, it is recorded before a debt/run bound can stop the run.

Current debt is the number of nominal completion deadlines already elapsed, capped at the requested run length, minus returned steps. Maximum debt includes wake and finish observations. Cumulative wake debt is the sum of outstanding step debt at successive wakes; it is a sampled backlog total, **not** integrated wall-time delay. Bounds stop with the exact known returned prefix and remaining scheduled count. A completed virtual run may pass timing while input fault/holding remains latched; that says nothing about balance.

Trace and control slot counts are reserved for the full finite run before the epoch. Event capacity is separately declared. Append publishes its index after the immutable payload is complete. Readers get committed immutable records and cannot reclaim slots. Capacity exhaustion stops explicitly, retaining the first overflow payload in a separately reserved failure slot; the plant never waits for a logger to drain storage.

`Mailbox` provides a fixed ring of bounded copied-byte slots. Every read/write calls the supplied lock's `acquire(False)` exactly once. Occupied slots cannot be replaced by newer or older publications; full/contended publication returns a named outcome. Reads copy bytes before releasing the lock. Slot versions, publication IDs, rejected payloads and job-publication failures remain visible.

## What remains unimplemented or unqualified

- Slot backing and metadata are currently in-process Python storage. A real spawned-process/shared-memory backend, process death, OS scheduling, GIL behavior and actual process-shared locks require separate implementation and adversarial timing tests. The held-lock stubs prove the nonblocking call contract only.
- Slot counts and payload sizes are bounded, but Python records, serialization, byte copies and snapshots still allocate. This is not an allocation-free plant loop or a benchmark of full-loop wall latency. A fixed native trace layout remains an integration task.
- Prepared window IDs are fixed for the requested control clocks; their packet admission, receipt/dependency age, original29/native goal banks, preceding task sample, 38-frame completeness and explicit EOF checks are outside this foundation. No source-motion tracking credit is implied.
- The stepper supplies measured terms, copied state and strict checks. Only a fake stepper is implemented. There is no complete291 native state adapter, MuJoCo replay, actual-force/quaternion oracle, controller inference, forecast/preview comparison or hardware timing evidence here.
- One-period latency changes controller behavior. A later worker must use its own state/history copies and a separately qualified delay-aware forecast. Target holding is mechanical fallback, not a balance controller.
- Rearm after measured standing, epoch transitions during a run, recovery, quiet standing, held-out motion and hardware remain separate gates. This core deliberately has no rearm API.

## Verification

Run existing Python and pytest from `source_draft`:

```powershell
python -m pytest test_foundation.py -q
```

The final preparation receipt binds 40 passing tests, all core/test sources, the design note, preserved earlier test reports, and two fake-only example ledgers. No actual controller, MuJoCo, ORT or optimizer calls are made.
