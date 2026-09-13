# Bounded pending job publication — draft only

This document proposes a narrow dispatch change. No implementation is frozen, no new clock run is selected, and no native, model, optimizer, or worker calls were made for this assessment. The corrected clock attempt and every failure remain immutable.

## Observed cause and limits

The independent reviewer decoded job 420's plant publication as BUSY from 132991903623 to 132991908848 ns. The call took 5,225 ns and returned 19,464,140 ns before its original activation deadline, 133011372988 ns. Jobs/results 1–419 correlate; the worker has no job 420. The current foundation issues a job once at its predecessor boundary and does not retry a BUSY publication. BUSY means the nonblocking lock was not acquired and the publisher wrote nothing.

The saved logs do not identify the exact holder of that lock: EMPTY worker polls are omitted. They do not establish how a retry would have turned out. Worker-result publication BUSY is a separate possible failure, outside this proposal. The existing six plant deadline misses, including index 610, independently disqualify the recorded run; dispatch repair cannot erase them. Native failure at step 4694 and the held controls 420–469 also remain failures.

## One pending job, unchanged identity

Retain one owned pending descriptor only when the initial publication returns BUSY. Store the exact immutable Job and its already serialized payload, activation key, original deadline, attempt ordinal, and last attempted physics index. The issued job retains its original sequence, creation time, measured snapshot, incoming raw action, named history, source window and digests. Never resample state, advance history, regenerate an identity, or substitute a newer source clock to retry.

Keep the initial publication at the existing boundary location. On later non-boundary physics ticks, after fixed wake/debt/elapsed checks and before the one native step, attempt publication at most once. For job 420, those indices are 4191 through 4199: at most nine later attempts, ten including the original. Check the clock again immediately before a retry and require it to be strictly before the original activation deadline. At deadline equality, do not retry. No polling loop, sleep, extra result poll, auxiliary dispatch thread, or clock rebasing is introduced.

The foundation owns both pending state and published_jobs. A narrow core method and one non-boundary tick hook avoid moving publication authority into an external wrapper. SharedMailbox, NativeStepper, the worker, and result-admission arithmetic remain unchanged. The tick hook is an explicit source change requiring separate review; the old hot-loop-identity claim does not carry over to this variant.

## State transitions and failures

* BUSY retains the same payload for the next eligible tick.
* PUBLISHED records that fact once and clears pending state. A result still needs the existing plant-side decode, identity, bounds, ownership-copy and strictly-before-deadline admission checks. Publication itself is never admission.
* If publication starts before the deadline but returns after it, record its actual result and timestamps. A late result cannot be sealed. Do not falsify publication history or grant deadline credit.
* FULL, CORRUPT, WRONG_EPOCH, UNCOMMITTED, and other non-BUSY statuses end this job's retry eligibility. Preserve their exact outcome. Do not add a new early input fault: the original activation-boundary miss behavior remains authoritative.
* An exception may follow a partial publication. Preserve its evidence and stop through the existing failure path; never retry an ambiguous result.
* Expiry clears pending state and records its last status, attempts and immutable identity. At the original activation boundary, the unchanged command-miss latch and held-command interval logic apply. No rearm or later job stream is added after that latch.
* Native/clock/logger failure prevents further retries. Preserve pending evidence on exit. If logging fails after a successful transport write, retain returned status and distinguish publication from recorded-event completion; no retry may be inferred safe.

A per-descriptor last-attempt index makes the once-per-tick limit explicit. Pending payload storage is fixed at one payload, within the existing 32,768-byte job capacity. No queue growth is allowed.

## Evidence and capacity

Each actual attempt needs activation, original identity/digest, attempt ordinal, physics index, immutable deadline and actual status. Existing transport records already contain start/end timestamps and payload digest. Add an explicit expiry/final-pending record so an unattempted or unresolved job is not mistaken for success. Keep admission timestamps separate from worker completion and publication timestamps.

With 1,819 controls there are at most 1,818 jobs and 18,180 publication attempts. At most two result slots are polled per each of 18,190 ticks, yielding at most 36,380 result status records. Their sum is 54,560 transport records, below the existing 80,000 capacity. Foundation events need a separately checked exact bound including attempts, result outcomes, expiry and held intervals before implementation freeze; no capacity increase is presumed. Worker ledger behavior stays unchanged. Logs remain in fixed memory during the hot loop, with durable serialization outside it.

The pure saved auditor must understand repeated BUSY attempts while requiring one successful publication at most and exact matching issued bytes. It must independently enforce attempt indices, immutable deadlines, no retry after terminal status, and unchanged result admission. It must continue to separate evidence integrity, actual command coverage, native safety and deadline qualification.

## Focused fake-only verification proposed

1. BUSY then success on a later eligible tick; exact same payload, key and identity.
2. Persistent BUSY: original plus at most nine attempts, then original miss/held behavior.
3. No two attempts on one tick; no republish after PUBLISHED.
4. Deadline equality and overdue wake cause no retry; publication crossing the deadline cannot admit a late result.
5. FULL/CORRUPT/epoch errors and ambiguous exceptions receive no automatic retry.
6. Signed-zero raw actions and different raw actions with identical clipped targets retain literal action/history ownership.
7. No extra boundary snapshot/history update, result poll, native step or changed simulation clock.
8. Native/clock/logger failure preserves pending/status/counters without false capture or successful-event credit.
9. Wrong/duplicate/late result checks and first-fault latch remain unchanged.
10. Maximum request size and fixed ledger capacities are proved with synthetic records.

No claim of 500 Hz operation, recovery, balance, online policy correctness or counterfactual success follows from this outline. A future source version, fake tests, independent review, and explicitly selected actual run would be separate steps.
