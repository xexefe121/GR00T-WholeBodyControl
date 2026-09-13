# Preserved v3 lifecycle correction

Source_draft and source_draft_v2 remain unchanged with their original test receipts. Source_draft_v3 changes only the dummy-worker lifecycle and its synthetic tests. All nine copied foundation/native/mailbox sources remain byte-identical.

`WorkerLifecycle` distinguishes start_attempted, start_returned and ready_returned. A failed `Process.start()` preserves the original exception and forbids retry. Cleanup avoids joining an unstarted handle. It attempts handle close without replacing the first error, returns normal_exit=false, and explicitly leaves process absence unproven because a partially failed spawn could have side effects. The later concrete supervisor must use its process-group/PID evidence for that exceptional cleanup; no success or no-descendant claim is fabricated.

A successfully returned start followed by a readiness timeout still uses normal stop/join cleanup outside the epoch, while readiness remains false. Two fake-process regression tests cover these cases. There are no worker processes, native calls, models or timed plant loops in this preparation.
