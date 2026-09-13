# Pico preview rejection repair — 2026-09-13

Implemented simulator rejection at publication, preserving the shared controller interface and numerical preview search. This fixes execution of rejected targets; it does not supply a qualified teleop controller or SONIC deployment.

## Measured result

| Fresh run | Physical motion + hold | Tracking | Terminal 30-second standing | Timing |
|---|---|---|---|---|
| Pinned older Pico | Completed 160.600 s | Failed | Passed | Passed this run |
| Strict newer Pico | Stopped at 63.392 s | Failed on available prefix | Not reached | No controller misses; one late physics finish |

Newer rejection occurred at control 3169, right shoulder roll, lower boundary. Seven preview calls exhausted the unchanged search budget; predicted reserve excess was 0.004404485 rad. The candidate was never published or applied. Native stop completed with zero joint-range excess; one already in-flight physics step finished after the stop request.

Older full-motion root p95 was 0.2794 m, foot p95 0.2879/0.3488 m and leg RMSE 0.3144 rad. Those remain outside existing limits. On the identical first 3169 complete controls, older/newer root p95 was 0.2178/0.2195 m and leg RMSE 0.2670/0.2726 rad. Partial scores are not complete-motion acceptance.

The runs used local compute, without concurrent training, rendering or archive compression. Existing background GitHub uploads continued. Timing is measured evidence from one run per configuration, not repeated timing qualification.

## What changed

- `NativePreviewGuard.apply()` retains its tuple API, target search, 7-call maximum, gains, limits, 32 ms configured horizon and 0.0001 rad reserve. It now retains owned request/candidate and per-joint prediction arrays; nonfinite preview arrays fail visibly.
- Independent simulator enforces `SimulatorPreviewRejected` before either native result channel receives an unacceptable candidate. It invokes native `clock_stop` before signaling workers or writing logs. Startup rejection produces a normal zero-step trace and report. Other worker exceptions also stop physics.
- Guard-enabled simulator qualification defaults to `strict`. Explicit `diagnostic-only` allows historical execution of rejected candidates and cannot earn the overall acceptance flag. The PowerShell guard launcher selects the independent simulation path.
- Command evidence stays buffered until physics stops. `preview_diagnostics.jsonl` records measured q/v, previous applied command and validity, raw request, selected candidate, per-joint low/high excess, limiting joint, timestamps and publication outcome. Rejected application time is null. Current serializer distinguishes actual native PD start, scheduled step deadline and old-target substep count.
- Reports separate completion, tracking, timing and terminal standing. Unreached terminal hold reports `passed: false, status: not_reached`; trailing quiet values remain separately labeled diagnostics.
- Each run saves exact arguments and SHA-256 hashes for actor, native libraries, configuration, motion, initialization bank and relevant source. Older v3 clock and newer v4 clock/v2 preview are explicitly pinned; no default-library substitution.

No robot publisher, robot stop behavior, controller gains, joint limits, reward, training recipe or checkpoint was changed.

## Archived failure diagnosis

Replayed all 31,830 archived 2 ms steps from fresh `MjData`, saved initial FP64 q/v and the recorded applied targets. Position, velocity and commanded-torque errors were exactly zero at every step. This validates simulator integration continuation for this trace; it does not reconstruct missing actor requests or packet-admission history.

The archive published two rejected commands:

- Control 3169: predicted reserve excess 0.001086269 rad with six old-target substeps. Its actual five-substep delay happened to remain inside the envelope.
- Control 3182: predicted excess 0.005178670 rad at six substeps, and 0.001091548 rad even at immediate application. Actual four-substep delay remained inside the configured delay budget. The recorded run then crossed the shoulder joint bound at 63.660 s.

The preceding accepted control 3181 remained inside its own full 32 ms preview when held through that horizon with reconstructed solver memory. The evidence identifies execution of rejected candidates; it does not establish failure of that preceding accepted prediction.

## Run

From repository root in PowerShell:

```powershell
.\artifacts\onboard_inspection_20260912\RUN_PICO_PREVIEW_REPAIR.ps1
.\artifacts\onboard_inspection_20260912\RUN_PICO_PREVIEW_REPAIR.ps1 -Case strict-newer
```

Default is the completed older benchmark. Both commands require the existing local artifacts and pinned WSL MuJoCo 3.2.3 runtime. They run simulation only and write a fresh output directory on E:.

## Verification and evidence

- 21 focused regression tests passed, including all-unsafe constant and improving seven-call searches, safe/corrected targets, lower/upper/reserve-only violations, nonfinite output, publication rejection, owned arrays and report labels.
- Native integration tests passed at controls 0 and 4: normal reports, no rejected command in applied trace, zero setup physics steps, at most one physics step finishing after stop request.
- Native/Python targets matched exactly across 12 sampled states; per-joint predicted excess arrays matched within 1e-8.
- PowerShell launchers parsed and changed Python files compiled.

Evidence directory: `E:\codex-artifacts\native23_preview_repair_20260913`.

- `older_v1/report.json`, `strict_newer_v1/report.json`: fresh results.
- `strict_newer_v1/preview_rejection.json`, `preview_diagnostics.jsonl`: failed command and complete captured command evidence.
- `archive_replay_v1/report.json`, `every_2ms_replay_error.npy`: archived failure replay and continuation comparison.
- `matched_prefix.json`: four-run comparison on identical control IDs, without pose realignment.
- `native_integration_v2/verification.json`, `native_parity_v1/report.json`: native checks.

Existing archived reports and checkpoints remain intact. The baseline can execute a whole recording and stand; arbitrary full-body tracking, live Pico qualification and hardware execution remain unfinished.

Full saved-trajectory videos (original speed, reference beside actual):

- `older_v1/video_fast/pico_factory_received_independent_clock.mp4`: complete 160.600 s, including terminal hold.
- `strict_newer_v1/video_fast/pico_learned_independent_clock.mp4`: complete available 63.392 s; stops at rejection, with no padded continuation.

Both renderers verified every encoded frame timestamp against the recorded physics timeline.
