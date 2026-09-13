# Corrected timing sidecar auditor v2

Source and synthetic evidence only. The original 25-file v1, preparation, tests and root's independent four-gap reproduction remain unchanged. No actual task arrays, clocks, worker processes, native/model calls or GC callbacks were used.

`reserved_capture` now reports whether the native verification counter records an attempt. Its return field is explicitly `null`: an attempted call may raise or be interrupted before returning. Neither equal attempt/return counts nor an existing assessment creates verification-return credit. The independent saved-state strict check still reports its own computed result and creates no new verification credit.

The span auditor now verifies wall, thread-CPU and process-CPU chronology between sequential scopes sharing a parent, plus thread/process CPU containment of each child in its parent. Checks use explicit same-thread call ordering and completed local samples. They do not compare callback clocks against global mutable thresholds; valid nested and foreign-thread GC overlap fixtures still pass.

Only `timing_saved_math.py`, `timing_sidecar_io.py` and their two tests change from v1. All other 21 v1 files, all 20 preserved original sources, and the 17 unchanged original modules remain exact. Original physics, deadlines, job/result retries, owner accounting and stage checks are untouched.

Validation: 163 pytest tests plus 56 subtests, 219 JUnit cases, zero failures/errors/skips. Five new tests cover the four independently reproduced findings plus sibling CPU order for both clock kinds. V1's passing test evidence is preserved as historical evidence; it does not qualify the corrected source.

Actual instrumented clock and actual saved audit remain unselected. A new root source review and later concrete input/launch reviews are still required.
