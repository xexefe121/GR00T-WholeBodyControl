# Saved audit v2: strict failure preservation

Use `source_audit_v2` for final source review. The 12-file v1 source snapshot, source receipt and 40-test evidence remain unchanged. V2 changes only `clock_saved_math.py`, `audit_clock.py` and `test_clock_saved_math.py`; the other nine files are byte-identical.

The correction permits nonfinite bytes in actual captured state/force fields so the independent original strict predicates can classify those records as physical failures. It preserves the original initial step-zero force exception. Actual commands, targets and prior/history inputs still require finite values, and captured overlap/repeated-clock/PD comparisons remain byte-exact. A failed capture cannot be followed by another native step. Three synthetic regressions cover initial nonfinite force, returned nonfinite force and returned nonfinite state; all 43 tests pass.

All scope, failure-accounting and future input-freeze rules in `AUDIT_PREPARATION.md` continue to apply, with `source_audit_v2` replacing the v1 path. This remains source preparation only: no actual task arrays, native/model calls, spawned worker or clock benchmark were evaluated. No actual saved-audit request exists. Parent review and a separately selected completed-run audit remain required.
