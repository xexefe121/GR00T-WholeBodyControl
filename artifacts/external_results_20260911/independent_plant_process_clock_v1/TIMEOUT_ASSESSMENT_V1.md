# Clock attempt: preserved timeout, no repeat

The single selected original request `d6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a` and launch `4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80` ran once. GNU timeout returned 124; the separate diagnostic and wrapper verdicts are 2. All 3,718 input pins remain exact. The Windows wrapper 26068, child 16024 and WSL launcher 8448, plus Linux native process 390, resource tracker 465 and worker 466, are absent. No blanket process termination or benchmark retry occurred.

Saved preservation report: `timeout_preservation_v2/report.json`, SHA `103208d22c2d00f91d241eac22c17d45f5411e1dbf4811adc0f2d9417bd45569`. It binds all available setup files, process absence evidence, input checks and the two actual entry MJB buffers. Both buffers are 95,115,288 bytes with expected SHA `717cc6f01a61be2fd1e79bff25710b30de4d01d5f14511f0f76ae45168194ee4`. There are no exit MJB buffers, epoch receipt, final report, trace, worker report or final actual-step counters. The actual native step count is unknown, not zero. Full timing, native physics and command qualification are false/unavailable, and the completed-trace saved audit must not run.

| Saved timing landmarks | UTC / elapsed |
| --- | --- |
| Windows wrapper start | 16:43:09.329388 |
| `run/process_start.json` modified | 16:44:35.966843; +86.637 s |
| `run/worker_ready.json` modified | 16:45:15.017366; +39.051 s |
| Timeout raw exit recorded | 16:45:17.360884; +2.344 s |
| Required fixed full clock duration | 18,190 × 2 ms = 36.38 s |

These are filesystem/receipt landmarks, not a profiler. Worker readiness precedes the fixed epoch in the frozen source. Only about 2.3 seconds remained after readiness, which is insufficient for the original 36.38-second fixed schedule. The source does not support claiming that this attempt reached final capture/export. The missing epoch/step ledger prevents determining whether it failed allocation, entered plant stepping, or how many steps returned.

The static cause of this budget placement is explicit: `clock_process/run.ps1` wraps the entire Python invocation in a 120-second timeout. `packet_io.ready()` resolves and hashes all original input files before `run()` creates its first output. `run()` subsequently reads all trace arrays, verifies/copies the 95 MB MJB, loads the native model, performs two model serializations/restoration and spawns the worker. Postrun cleanup, two more serializations, full owned evidence output and another complete input hash pass also lie inside that same timeout. No individual file or function duration was measured; the timestamps establish only the broad stage totals above.

## Bounded next source proposal; not selected

Keep the original 120-second bound for the plant execution stage, the fixed 60-second elapsed/debt aborts, 2 ms epoch, 18,190-step budget, recorded targets, native model identity and all strict predicates unchanged. Prepare a versioned supervisor that distinguishes bounded preflight/setup, armed plant execution, and bounded postrun preservation. The plant-stage watchdog should arm exactly once after readiness and before the single epoch; it must not rebase deadlines or permit a retry. Any separate setup/preservation wall-time bounds require an explicit reviewed request; none are selected or changed here.

Add small owned stage receipts outside the timed stepping loop: actual epoch/setup completion before first step, and attempted/returned/captured/verified counters immediately when the loop exits, before slow cleanup/model-exit/hash work. This improves failure localization without disk I/O on every native step. If the process dies during stepping, preserve that uncertainty rather than inventing returned samples. A new source review and explicit one-run selection would still be needed before any further benchmark.

An earlier apparent PowerShell backslash defect was a display-escaping false alarm. The actual original literal contains one char 92 and passed direct normalization; no launch file changed. The abandoned v2 supervisor generator failed its exact-token assertion before producing any replacement/clearance. The first timeout preservation checker completed hashes/absence but rejected a seven-digit .NET UTC timestamp on Python 3.10; it is preserved. The versioned checker only normalizes fractional precision and records the successful preservation accounting separately from benchmark qualification.
