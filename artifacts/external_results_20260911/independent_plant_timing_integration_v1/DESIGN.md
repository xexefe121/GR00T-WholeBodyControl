# Timing integration: source preparation only

This copy adds diagnostic timing to the recorded-command pending-result producer. No clock run, worker process, native call, model call, real clock reader, or actual GC callback was executed for this preparation. Synthetic readers and arrays verify behavior; they do not establish WSL performance or balance. The observed previous 23.076358 ms body stall remains unattributed.

All 24 prior producer sources are preserved in `source_original_v1`. Twenty remain byte-identical in `source_draft_v1`. Only `clock_core.py`, `native_stepper.py`, `session.py`, and `run_clock.py` change. The root-reviewed v2 probe and its two test modules are copied byte-exact. Prior standalone and integrated initial derivation files remain historical evidence; `integration_final.diff` and `source_preparation.json` describe the final draft.

The static checker erases only enumerated timing scopes, imports, wiring, and pre-native instrumentation-fault guards. The resulting complete AST of each core, native stepper, and session module must equal its original. Separate mutation tests reject altered PD arithmetic, simulation time, and recorded finish times. The outer fixed while loop and supervisor gate/main functions remain AST-identical.

## Hooks

The phase IDs retain the reviewed probe schema:

| ID | Scope | Meaning |
|---|---|---|
| 1 | TICK_ENVELOPE | Entire session tick, including fixed wait and outer logging; inclusive. |
| 2 | RESULT_POLL_AND_ADMIT | Original nonblocking poll and full plant validation/admission. |
| 3 | FIXED_WAIT | Original wait for the immutable epoch deadline. |
| 4 | BOUNDARY_SNAPSHOT | Original measured full291 boundary and measured terms. |
| 5 | BOUNDARY_HISTORY | Original one history update, control record, outgoing raw action assignment. |
| 6 | BOUNDARY_SERIALIZATION | JSON/base64, exact Job creation time and original deadline, owned pending bytes. |
| 7 | JOB_PUBLICATION | Original immutable-job publication attempt and its bookkeeping/event. |
| 8 | PD_AND_INPUT | Original target checks, force prohibition, PD arithmetic, owned target/torque and ctrl write. |
| 9 | MJ_STEP | The single API call plus essential attempt/return/time/stage bookkeeping. Not a claim of pure API compute. |
| 10 | CAPTURE | Original capture call and child ownership scope, inclusive. |
| 11 | CAPTURE_OWNERSHIP | Foundation raw/validated capture ownership and captured count. |
| 12 | VERIFY | Original strict verification and child ownership scope, inclusive. |
| 13 | VERIFY_OWNERSHIP | Verifier return ownership, type/expected-time checks and verified count. |
| 14 | STEP_LEDGER | Original completed/failed StepRecord append. |
| 15 | OUTER_LOG | Original outer JSON serialization and append. |

Existing authoritative wake/finish/publication/receive/admission times are neither replaced nor moved. Phase1 includes wait; phases10/12 include phases11/13. Nested intervals must not be summed as disjoint durations. The worker remains byte-identical and retains its original timestamps and retry ledger. This probe measures the plant thread/process; worker CPU is not measured.

Full lifecycle remains 1,569 main plus 250 continuous hold controls, 18,190 native steps and four identity serializations, subject to the original first failure. Native2ms, control20ms, strict oracle, actual raw feedback/history, mailbox retries, admission deadline, debt100 and elapsed60s constraints remain unchanged. Setup240s, plant120s, preservation180s and outer555s guards remain unchanged. No epoch rebase, catch-up reset, skipped tick, relaxed check, changed target or extra native diagnostic read is added.

## Allocation, clocks, and GC

After native/channel/worker readiness, before choosing the single epoch, allocate 262,144 span rows, 4,096 GC rows, stacks and 16 reusable scope objects. Integer arrays occupy 33,947,648 bytes, excluding Python objects. A conservative normal full-scope bound is 223,727 span starts: 11 per native step, at most3 per boundary, and at most10 publications per future job. Capacity failure is explicit; storage never grows or overwrites old rows. This is not an allocation-free claim: Python integers, calls and existing JSON/history work still allocate and cost time.

Each bracket records monotonic wall, thread CPU, process CPU, monotonic wall. Only local brackets, same-thread paired spans, and successive completed root spans enforce clock order. GC callbacks may interrupt sampling without creating a false global regression. Wall minus thread/process CPU is evidence about execution versus elapsed time, not sufficient scheduler/GC attribution on its own.

The exact callback is attached to the existing registry before epoch choice and removed during preservation. GC enablement is observed before/after; it is never enabled, disabled or forced. The process-shared `inside_probe` field means overlap with probe execution. For a foreign-thread callback, `active_span=-1` and `inside_probe=1` can mean another thread was probing; it does not prove that callback interrupted the sampled thread. GC generation pairs and all incomplete/overflow rows remain visible.

## Failure and preservation

Ordinary hook errors latch first fault and preserve the measured operation's original exception. A fatal deadline/interrupt during a hook propagates if no measured exception already exists. A fatal begin is separately recorded even if it happened before the probe reserved a row. Before any next native call, a latched instrumentation fault stops progress. The end of native/capture/verify scopes follows essential original ownership counters, so an end-clock fault cannot erase that completed work.

A fatal interruption between adapter return and foundation ownership can legitimately leave different adapter/foundation counts. No false foundation credit is added. Already-owned adapter capture/assessment and foundation return/capture fields are saved separately; no future mutable native data is read to fill the gap. A native end-hook interruption may leave adapter returned1/foundation returned0; actual API counters remain authoritative and no extra step or capture is attempted to repair evidence.

Supervisor preservation retains its original order: first stage counters, worker cleanup, required model exit, channel close, main owned trace/capsules. Only then write timing sidecars, followed by original posthashes, report, output manifest, and final stage receipt. Sidecar errors cannot prevent an earlier main-evidence write. Each sidecar uses exclusive creation; failed writes and field conversion failures are retained individually and other sidecars/owned fields are still attempted. Any missing, incomplete, overflowed or faulted timing evidence forces the overall preliminary result false. Physical, command, timing and instrumentation verdicts remain distinct.

An OS hard kill, preservation deadline, journal failure, memory exhaustion, or partially written filesystem file can still prevent final records. Existing incomplete-stage/process evidence remains required; absence of a final report does not imply zero native steps. No live owner/launcher or saved-audit source is adapted in this preparation. Future actual packet review must bind this new namespace and sidecar-aware audit before a separate run selection.

## Sidecar output schema

`timing_spans.bin`: owned used-prefix signed64 values, 16 fields per row from `timing_metadata.json`; native `array('q')` byte order and item size are explicit.

`timing_gc.bin`: owned used-prefix signed64 values, 12 fields per row. Attempted/recorded/denied counts and incomplete generation pairs are explicit.

`timing_metadata.json`: field names, phase names, storage representation, attempted/started/closed/denied counters, active depth, open generations, first probe fault, callback-detached state, hook first fault/count, and `instrumentation_complete`. Return-kind1/2 is the measured scope's normal/raised exit, not a substitute for native attempt/returned/captured/verified counters. Composite scopes are as declared above.

`timing_owned_partial.json`: lossless existing adapter/foundation failure, capture, capture-return, verifier-return and assessment fields when present. No native reads.

`report.json.timing_instrumentation`: enabled/complete, per-file SHA256/byte length, individual writer errors, hook status and `new_native_reads=0`. Report also records `gc_enabled_before/after`. Output manifest hashes the sidecars with all original files. The existing epoch and exit-stage receipts additionally carry compact probe counters, outside the 2ms loop.

## Validation evidence

Original67 fake tests and reviewed probe31 tests are reused unchanged. Added30 tests cover active enabled/disabled main-evidence byte equality across success/continuous hold, strict failure, API exception, hung worker, held result lock, equal targets with distinct raw feedback, logger exhaustion; exact manual PD; partial return/ownership/watchdog failure; sidecar collisions and partial conversion; source/ordering/capacity mutation checks. All clocks injected; GC tests use plain fake registries. The initial test draft incorrectly constructed StageTimeout with one argument; its four failures and original test source are preserved. Final tests use its actual three-argument API; no producer runtime correction resulted from those test failures.
