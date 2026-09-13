# Saved schema delta

Job canonical JSON and Result.identity add `deadline_ns`, a nonnegative integer. The plant's issued job, pending-job descriptor, job bytes, worker result identity, and sealed/rejected result all contain the same original deadline. This changes canonical job/result digests; no old digests may be reused.

Existing plant step/control captures, outer-cycle records, job-publication retry ledger, six stage receipts, MJB buffers, and native counters remain unchanged. The plant's result admission gains `WRONG_ORIGINAL_DEADLINE` before full identity comparison; its timing cutoff remains unchanged.

Worker JSON retains pid, polls, start_ns, end_ns, failure, events, overflow, recorded_command_table_sha256, model_calls=0, and native_steps=0. `polls` still counts completed worker-loop iterations; actual mailbox poll counts are now explicit under result_retry.counts because pending iterations do not poll jobs.

New worker events:

- WORKER_RESULT_READY: activation, exact Job identity, received_ns, original pre-construction completed_ns, payload_ready_ns, job SHA, result SHA, full immutable result payload in base64. One event per computed result, before its first attempted publish.
- WORKER_RESULT_PUBLICATION: activation, worker iteration, 1-based attempt, attempt start/end, returned status, original deadline, unchanged job/result digests. One event per call that returned and was successfully logged. Underlying ObservedEndpoint transport events remain separate and retain the narrower transport span.
- WORKER_RESULT_TERMINAL: activation, termination reason, observed time, deadline, attempts, last status, whether call returned, job/result digests. State and terminal count commit before logging; a logger failure can therefore leave terminal_event_recorded=false.

`result_retry` includes contract `immutable_result_BUSY_max20_original_deadline`, max_attempts=20, counts, phase, closed, pending, last_result, buffered_jobs, last_poll_return, clock_failure, and failure. Counts separate iterations started/completed; polls attempted/returned; jobs taken/decoded; replies attempted/completed; ready events; publications attempted/returned/published/BUSY; publication events; terminal states/events. Each pending/last descriptor retains original owned job bytes, identity, received/completed/ready timestamps, exact result bytes, attempt details, recorded flags, and terminal state. At most two buffered jobs retain exact payload bytes and original slot/key/version.

The independent audit must reconstruct every ready/result from the bound issued job and recorded target/raw action; verify exact deadline from the one epoch; correlate transport and worker attempts without granting returned credit to unknown calls; allow only BUSY continuation; enforce 20 attempts and one per worker iteration; prohibit new-job polling or next-job computation while pending; explain all buffered and terminal states; and keep worker evidence integrity, physical outcome, command coverage, and timing qualification separate. A missing final worker report or hard timeout cannot qualify the run.
