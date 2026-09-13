# Concrete process-clock runner preparation

Source and saved-input preparation only. No worker, native call or clock benchmark has run. A separate literal root-selected final request/launcher review is required before the dormant command can run. Scope is one recorded query250 expert main1569 + continuous250 hold, 1819 controls and18190 native steps, four existing-witness MJB comparisons, zero inference or optimizer updates. A successful component result would not qualify an online policy.

The `source_runner_v1` directory preserves every `source_draft_v3/*.py` file byte-for-byte, plus the already-qualified `native_loader.py` and `counted_api.py`. Added modules provide only request gates, orchestration and owned postrun evidence. Native manual PD, strict returned-step oracle, full291 restoration, measured history, action feedback, nonblocking mailbox and deadline arithmetic remain in those preserved files.

## Fixed timing and lifecycle

Native setup, expected MJB comparison, saved-input validation, shared channels and spawned worker readiness precede selection of one future epoch. The epoch is chosen once as `epoch_chosen_ns + 200000000`. Unchanged Session allocates its internal ledgers during construction. `setup_finished_ns` must be strictly less than this same epoch, or execution aborts without a first step or epoch change. No step skipping/rebasing, no main/hold reset, and no private native copy is introduced.

The unchanged dummy worker uses recorded commands only. Its raw exit may be0 even when `worker.json.failure` is populated: `worker_pass` requires normal unforced cleanup, matching PID, no failure/overflow and zero worker native/model calls. Failed process starts retain `start_side_effects_uncertain`, never claim unobserved descendants absent, and never join an unstarted process. No retry is available. Parent foundation input faults remain explicit; held-command intervals cannot pass command acceptance even if core timing finishes.

The outer GNU timeout is120seconds with TERM and a5second KILL fallback. It bounds a stuck process/native call; such termination fails accounting/diagnostics and may leave only already-written setup/MJB/raw stderr evidence. In-memory step records cannot be recovered after an OS kill. This limitation is disclosed rather than inventing captured credit. Normal strict/timing/worker failures flush owned partial records and preserve requested18190.

## Files written after a future selected run

`run/trace.npz`: no pickled objects. Every row is an actual committed record. Failed captured native steps remain present and `step_verified` is false. If a step returned but its StepRecord could not commit, separate counters and raw capsule/overflow evidence preserve the distinction. Empty arrays retain trailing dimensions.

| Array | Shape / dtype |
|---|---|
| packed_capture | N×373 float64 |
| step_integration, step_qpos, step_qvel, step_actuator_force | N×291/30/29/23 float64 |
| step_command, step_target, step_raw_action | N×23 float64/float64/float32 |
| step_warning_counts, step_warning_lastinfo | N×8 int32 |
| step_index, step_nominal_start/end, step_actual_start/end, step_wake_lateness_ns, step_finish_lateness_ns, step_wake_debt | N int64 |
| step_expected_simulation_time, step_verified | N float64/bool |
| control_integration, control_incoming_raw, control_flat_history_before | C×291 float64, C×23 float32, C×300 float32 |
| control_control, control_physics, control_nominal_activation_ns, control_actual_activation_ns, control_admitted_ns | C int64 |
| control_held, control_nominal_source_frame, control_active_source_frame | C bool/int64/int64 |
| control_target, control_raw_action | C×23 float64/float32 |

`packed_capture` columns remain [0:291] full integration, [291:321] qpos, [321:350] qvel, [350:373] actual actuator force. Full291 contains time at0, qpos1:31, qvel31:60, warmstart60:89, ctrl89:112, external forces112:291. Initial step0 force handling remains the original oracle distinction.

`run/evidence.json` is the exact metadata from reviewed `evidence.export_owned`, encoded by the separate lossless writer. It includes:

- Every outer cycle byte record: index, returned count, cycle_return_ns and fixed_nominal_end_ns.
- Every foundation event. RESULT_SEALED/REJECTED contains original full result payload, authoritative plant received_ns, worker_completed_ns and rejection reason; JOB_PUBLICATION records status and input digest.
- Every nonempty transport status with copy start/end timestamps, slot/version and payload SHA.
- Named measured terms and before/after histories for every control; nominal versus actually active command IDs/source windows; all issues and reserved overflow records.
- Full291/history/prior main-to-hold continuity checks when that boundary was actually reached.

Lossless JSON rules: ordinary null/bool/int/string remain ordinary; every Python float is `{"type":"float","hex":...}` including signed zero/NaN/Inf; bytes are `{"type":"bytes","base64":...,"sha256":...,"length":...}`; frozen dataclasses are `{"type":"dataclass","class":...,"fields":...}`; mappings and sequences recursively preserve all contents. No truncation, repr substitution or object pickle. This outer encoding does not alter original typed native fault capsule bytes.

`run/raw_capsules/index.json` binds each `.bin` file. Initial capture state is exactly373×8bytes, initial torque23×8bytes; warning ledger is in evidence metadata. Last validated capture state/torque, raw foundation capture/verifier returns, original typed native fault/capture-return evidence and logger overflow bytes are kept whenever available. No new native read occurs in this writer.

`run/issued_jobs.json` maps activation to lossless original canonical Job bytes, including exact measured full291/history/prior, source window, snapshot and schedule digests and job creation timestamp. `published_job_activations.json` separately records which publication calls returned PUBLISHED. Original result bytes remain in foundation events. `run/worker.json` is the unchanged worker's full125000-capacity event ledger, with received/completed/publication-return times, input/result hashes, command identity, transport status, polling count, failure and overflow. Worker replies are compared to the plant's exact issued identities at admission.

`run/mjb/00.mjb` through `03.mjb` are the four actual entry/exit differently-filled serialization buffers, not new witnesses. CountedAPI records each attempted/returned serialization and captures partially filled failed buffers too. Constructor failure can additionally write named fallback copies of the same two exit outputs; these are not additional API calls. Expected bytes are pinned to the previously qualified `717cc6f0...` MJB. No other model serialization is permitted.

`run/report.json` keeps requested/actual scope, all foundation/native/API counts, first error and separate later cleanup/preservation errors, worker lifecycle/cleanup and worker failure, fixed epoch timestamps, source/runtime identities, four-MJB verdict, preliminary component verdict, and permanent `component_qualified:false` / `root_saved_audit_pending:true`. Wrong commands, held/input-fault intervals and timing failure remain separate from physical feasibility. Root's pure saved audit decides exact expert equality only until actual command/state divergence; it must not demand expert equality after a different command or call that qualification.

`run/process_start.json` / `worker_ready.json` record Linux parent/worker/resource-tracker IDs and `/proc` start identity before epoch. `pre_input_hashes.json` / `post_input_hashes.json` bind actual inputs. `output_manifest.json` hashes all completed output files, excluding itself.

`clock_process/` retains hidden Windows wrapper/child PID and start/handle receipts, exact WSL argv, CreateNew started lock, stdout/stderr, raw known exit and separate diagnostic verdict, pre/post launch pins and final clearance. Owner verification later records Windows/Linux observed PID absence, validates every output hash, and keeps diagnostic pass separate from accounting pass. Unknown exit, missing output or incomplete preservation cannot become success. Root's independent trace audit remains a separate gate.
