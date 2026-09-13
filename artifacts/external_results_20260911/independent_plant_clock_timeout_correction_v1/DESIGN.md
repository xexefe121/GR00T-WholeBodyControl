# Versioned timeout correction: source preparation only

The failed original attempt and its incomplete evidence remain untouched. Its 120-second outer timeout covered input hashing, model setup, worker startup, all 36.38 seconds of the intended plant schedule, cleanup and large evidence writes. Only about 2.34 seconds remained after the saved worker-ready landmark. That attempt provides no native-step-count or plant timing qualification.

This draft changes outer watchdog placement and durable reporting. It preserves the original foundation, session, shared mailbox, worker, protocol, native loader, stepper, model identity, capture schema, limits and commanded targets byte-for-byte. The stepping `while` AST is identical. There are no new reads of live native state, captures, steps or model calls in the reporting helper.

| Proposed stage | Bound and semantics |
| --- | --- |
| Setup | 240 seconds from the setup arm, before expensive pin checks and native setup |
| Plant | One absolute deadline at the single fixed epoch plus 120 seconds |
| Preservation | 180 seconds from loop/setup exit; includes cleanup, required model-exit identity, owned trace output and postrun hashes |
| Outer guard | GNU timeout 555 seconds, followed by the existing 5-second kill grace |

These are proposed request values, not an actual run selection. The outer bound exceeds 240 + 0.2 + 120 + 180 seconds by 14.8 seconds for entry and bounded final bookkeeping. Bootstrap and Windows wrapper hashing remain outside the GNU child timeout, as before. The existing 2 ms step and 20 ms control deadlines, 60-second elapsed abort and 100-step debt abort are unchanged. The plant deadline is never extended, skipped or rebased.

The Python POSIX timer requests a graceful stop through a `BaseException`, bypassing ordinary controller/native exception catches. Native C calls can delay Python signal delivery. Interrupted return/capture bookkeeping remains uncertain; the code records separate attempted, returned, captured and verified counters and does not infer unobserved completion. The independent outer GNU timeout remains the hard fallback. A cleanup or native call that does not return can still require that fallback, so this draft does not guarantee complete output on every hard kill.

Small CreateNew JSON records are flushed and fsynced in `stage_receipts/`, outside the 2 ms hot loop:

1. Setup start before input hashing, then native-setup and restored landmarks.
2. The chosen epoch, allocation completion, current counters and worker identity before stepping. After writing this record, the original same-epoch test runs again. A slow write aborts without rebasing or stepping.
3. Immediate loop/setup-exit counters before worker cleanup, exit MJB serialization, trace writing or postrun hashing.
4. Preservation completion with final counters and actual report/manifest hashes, or an explicit incomplete-process-exit record.

Stage records retain their creation timestamp; they do not pretend that timestamp is fsync completion. The final `setup_finished_ns` is sampled after the epoch record's fsync and must be strictly before the unchanged epoch. The run report carries it, and loop-exit evidence preserves it even when later output fails.

The unchanged full trace/schema and four-serialization contract still apply. A final diagnostic pass additionally requires the last stage record to bind the actual completed report and output manifest. Owner accounting binds all stage records. If preservation ends early or only counters survive, the old complete-output audit cannot run; that incomplete evidence must be assessed separately. Neither the stage receipts nor counters establish native physical equivalence by themselves.

No actual request, clearance, native model, serialization, worker, plant clock or controller is executed by this preparation. Future use requires new original-input/source freezing, a concrete stage-budget and launcher review, separate parent selection and independent saved-output audit adaptation for these additional stage records.
