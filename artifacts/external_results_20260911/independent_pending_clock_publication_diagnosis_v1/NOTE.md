# Saved publication diagnosis

The recorded benchmark stopped after 6,824 of 18,190 intended native steps. Its saved audit passed 12,393 evidence checks; physical, command, timing, and component qualification remain false. This diagnosis runs no native steps, model calls, or optimizer updates.

The plant's pending-job repair recovered four BUSY publications: activations 136, 192, 213, and 614. Each used two attempts with identical payload bytes and the original deadline; the resulting commands were admitted before that deadline.

Activation 656 failed on the opposite mailbox direction. The job was published and consumed, and the worker computed its recorded reply. The worker's single result publication returned BUSY after a 2,842 ns call, with 17,413,713 ns still remaining before the original activation deadline. That result was never admitted. The plant retained command 655 through controls 656–682, then recorded a native joint-bound failure at step 6,824. The observed sequence supports retaining that same result for bounded BUSY retries; it does not establish the exact lock holder or lock duration because empty poll spans were not recorded.

Timing remains a separate failure: 15 captured-step deadline misses and 16 outer-cycle misses. Outer index 5314 accounts for the extra miss; this is not a missing final sample. Runtime maximum debt was five steps. Saved spans include scheduler preemption and cannot identify garbage collection or pure compute time as the cause. No watchdog expired.

The worker currently stamps `completed_ns` before reply validation and serialization. That stamp is not the end of full result construction. Any source change should retain exact job/deadline identity and explicit actual publication timestamps rather than reinterpret this old field.

Selected next source preparation: add the plant's literal original `deadline_ns` to each immutable Job identity; retain one computed result and its bytes after unambiguous BUSY; retry with fixed bounds before that deadline while consuming no new job. FULL, CORRUPT, ambiguous publication exceptions, and expired work remain terminal. Plant admission, native limits, 2 ms/20 ms deadlines, epoch, and held-command behavior stay unchanged. A matching saved-ledger auditor will require its own version before any future run.
