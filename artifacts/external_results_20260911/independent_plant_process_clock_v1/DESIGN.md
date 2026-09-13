# Recorded-command process-clock component scaffold

Preparation only. No actual native, worker process or process-clock execution is selected here. The only local checks use deterministic clocks, fake native steppers, copied bytes and fake process handles. The previous WSL mailbox experiment separately verified its unchanged 11 spawned-process synthetic tests.

The proposed component experiment is one continuous instance with the qualified query250 expert's recorded 1569 original controls followed by its actual continuous 250-control hold: 1819 activations and 18190 native 2 ms steps. It is a recorded-command transport and scheduling benchmark. Its worker does not infer an action from measured state, and success cannot qualify an online policy or full-body teleoperation controller.

## Preserved components and ownership

Eight NativeStepper v3 files are copied byte-for-byte, including foundation v3, four-lag history, original observation and strict-oracle sources. The reviewed shared mailbox is also copied byte-for-byte. No production or frozen original is edited. `session.py` composes these components through their original methods. The real MuJoCo loader remains a later, separately frozen runner dependency; this scaffold never imports it.

Only the plant owns native model/data and measured history. Native initialization remains a single full291 restore/forward/restore with exact warnings and initial strict check. Complete MJB identity uses the existing qualified expected bytes and two entry plus two exit serializations, outside the fixed epoch. No new MJB witness is proposed. A constructor failure cannot expose an unreturned adapter; the later runner must retain its external API counter/buffer evidence. Restoration failures retain owned first-failure evidence and attempt exit verification without replacing the original exception.

Each native step uses unchanged manual PD and one original `mj_step`. Captures keep the unchanged 373-float64 payload: full291 integration, qpos30, qvel29, actual actuator force23; commanded torque and typed int32 warning ledger remain sidecars. The full291 restore is used only at canonical initialization. Main-to-hold boundary1569 is an ordinary activation inside the same instance, with no state/history/prior/clock reset or forward call.

Foundation history advances exactly once per actual control boundary from measured terms and the incoming action of the last activated command. The selected source `action` bytes are authoritative, including BFM raw actions and MPC normalized applied-target feedback. Identical clipped targets may carry distinct raw-action bytes; the benchmark preserves these differences. The initial canonical history/prior must be all zero. Noncanonical history initialization is unsupported by unchanged foundation and is outside this proposal.

## Fixed clock and command protocol

Before admitting an epoch: validate all actual input hashes/runtime, load and compare initial state, allocate fixed ledgers/mailboxes, load the immutable command table, start the spawned worker and await readiness. Only then choose one future monotonic integer epoch, proposed lead200ms. Failure to initialize before that epoch aborts; it must never rebase it. `DeadlineClock` waits against the same absolute integer deadline using an injected monotonic source and relative sleep; it never changes epoch or native index. Sleep jitter, GIL and scheduler latency are measured limitations, not hidden by a catch-up clock.

The worker receives job c at actual boundary c, carrying the measured full291 snapshot, exact incoming prior, before/after named history, current active command and literal binding. It returns recorded command c+1, with the exact original Job identity. `recorded_protocol.py` checks canonical fields/integers, binding, sequence, snapshot clock, future activation, source window, snapshot/history/active-command hashes and byte schema, including four-lag shift order. A unique window binds global control, actual original source frame and the entire immutable recorded command table digest. Full frames are11..1579; all250 hold frames are1579. The source table hashes identify the actual full and hold traces.

Only one publication attempt per job/result; no transport retry. Two fixed slots per direction, job32768 bytes and result8192 bytes. Worker polling and1ms sleeps run only on the worker, with fixed60000-poll/55s watchdogs and125000 event slots. Worker readiness waits and stop/join/termination occur outside the plant epoch. The future supervisor must retain worker PID, start/ready/stop/error and known exit records, plus a fixed last-resort cleanup path for pre-epoch readiness failure.

The plant endpoint wrapper logs nonempty statuses, copy start/end timestamps, key/version and payload digest in a separate fixed ledger. It never translates a failure status into a valid publication or retries a lock. Foundation still owns authoritative admission time after decoding, validation and owned copy; worker completion time never substitutes for plant admission. Added status logging itself consumes real time and is included before authoritative admission.

## Acceptance and honest failure semantics

Success requires all18190 attempted/returned/captured/verified native steps; all1819 actual command IDs, targets, raw actions and source windows; no held intervals/input fault; unchanged full291 initial/main-to-hold/final continuity; final MJB identity; no fixed-end deadline misses; known normal worker exit; and root's independent comparison of every saved native sample against the already-qualified expert traces. No duplicate old native rollout is needed. Root owns the later pure saved-array audit.

Foundation timestamps step wake and finish, while the scaffold additionally records `cycle_return_ns` after `tick()` and the immutable step ledger append. Timing acceptance checks that outer return too. Nominal deadlines remain epoch+i*2ms; independent simulation time is repeated+0.002, never resynchronized to wall time. Proposed debt bound100 native steps and elapsed bound60s are finite abort limits, not acceptable timing slack. Any missed2ms end fails the timing result even if it catches up. Actual step/capture/verification and outer-ledger counts remain separate on an exception.

On missing results, original foundation latches input fault and holds the last active command at every remaining boundary. Nominal windows continue in the ledger; active windows and actual commands explicitly show gaps. No late result is used to repair the same epoch. The current foundation's `timing_passed` ignores held commands/input fault, so the scaffold's preliminary verdict additionally requires no input fault, exact command coverage and no held interval. It never relabels that foundation field as component success. Native strict failures retain the returned/captured failing step and unexecuted suffix; a new tick is forbidden afterward. Logger overflow is a failure with the reserved overflow item and actual counters preserved.

## Remaining explicit integration gaps

1. This is an API scaffold, not a selected launcher. Actual frozen input request, WSL runtime/import inventory, durable supervisor, full output writer and root audit request must be reviewed before any native/process-clock run.
2. The existing `poll_results()` occurs before `wait_until_ns()`. A reply arriving after the last pre-deadline poll but before activation can still miss the switch. The scaffold preserves this conservative behavior, tests it explicitly and does not add a polling callback or weaken the strict received-before-deadline gate.
3. Native attempted count can differ from foundation step-method attempted count if input validation fails before `mj_step`; preserve both. Equality is required for the intended successful benchmark, not asserted after every exceptional path.
4. Constructor-level MJB failures need the later runner's external counted API wrapper to retain buffers/counters; no adapter object may exist yet. Required exit identity must still be attempted whenever an adapter was constructed.
5. Future real execution must enforce private top-level module import paths (`mailbox` shadows a standard-library name), bind all actual runtime/native/model/trace sources, and reserve memory before admitting epoch. No runtime installation is proposed.
6. Fixed Python ledgers bound item counts, not allocation time or bytes per entry. Copying, JSON/base64, hashing, garbage collection, page faults and OS scheduling remain measured costs. No lock-free, hard-real-time or allocation-free claim follows.
7. Final acceptance still needs independent every-step saved comparison, model-exit evidence and process cleanup. `component_preliminary_pass` is explicitly not a qualification field.

No task inference, optimizer updates, Pico or hardware actions belong to this component experiment. It does not address the failed direct head's control stability.
