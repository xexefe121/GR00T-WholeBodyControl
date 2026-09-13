# Interrupted expert branch continuation

This resumes the same single expert branch from the final atomic checkpoint after its foreground process disappeared. No new learner state, query selection, training, or controller policy is introduced.

The original checkpoint preserves 1001 complete controls, 10010 physics steps, 651 of 819 source controls, and simulation time 20.02 seconds. Global control0 remains the student's actual command and is excluded from any new expert label set. New expert controls are1 through1268. Original lifecycle remains1569 controls, followed by the separate250-control quiet hold.

`resume_inputs/trace.partial.npz` is byte-identical to the original saved checkpoint. All200 committed plans through996 are preserved. The resumed controller starts at1001, shifts plan996's targets by its five committed controls, and replans with the unchanged frozen H30/10-iteration/two-thread hard-feasible MPC and existing guided/K0 restoration. The preserved but uncommitted plan1001 is used only as an exact deterministic equality assertion before the first resumed actual command. It is never injected as a target source.

The adapter restores full integration state once at process initialization, separately restores warnings, proves the complete named BFM history/prior-action chain by sensor-only reconstruction, and continues the original independently accumulated clock. All recorded prefix fields remain intact. No physical state copies occur after initialization.

The14 original source files are unchanged. `source_snapshot/resume_actual_student_oracle.py` adds checkpoint initialization, exact first-plan verification and new output persistence. Source and input hashes are in `frozen_inputs.json`; parent and resumption boundaries are in `resume_receipt.json`.

Run `--stage preflight` only for load/history/clock checks; it performs zero actor inference, optimization or physics steps. After independent source review and successful preflight, `run_durable.ps1` is launched through hidden `Start-Process`. It records its PID, stdout, stderr and final process exit status independently of a foreground tool lifetime. Process exit is not a physical or full-source pass; those verdicts require the complete reports and independent audit.

No hardware, Pico, DDS or robot actions are authorized by these artifacts. Labels remain inadmissible until the full remaining source, return and both quiet windows qualify independently.

Prelaunch source review passed17 read-only checks in sibling `student_actual_oracle_resume1001_prelaunch_review_v1`. Preflight passed full state/history/clock reconstruction with zero actor inference, optimization or physics. Root separately reproduced all10010 original checkpoint physics samples exactly.

The first durable launcher exited before starting WSL because its Windows PowerShell child could not resolve `Get-FileHash`; the failed `process_status.json` and original launcher are preserved. `run_durable_v2.ps1` changes only this hash operation to equivalent .NET SHA256 and uses separate `process_status_v2.json`, `stdout_v2.log`, `stderr_v2.log`. It was launched hidden at2026-09-11T03:05:59UTC, WindowsPID13904. See `launch_started_v2.json`. The controller and input hashes remain unchanged.

The durable resumed branch completed at2026-09-11T03:18:41UTC with process exit0. Original lifecycle1569/1569 controls and15690 physics steps, all819 source controls, plus separate250/250-control hold and2500 physics steps completed without producer physical failure. Producer source metrics and both quiet diagnostics pass. Final nominal trace SHA25685de329f57b4582c61f5b7055e7e567f3b4a971d71fbcde1673998197eef149a; extension22e18d5423aff51217e03d38c0dca57343e032dec445d214df78f7f64ac0bf4d. All34 recorded prefix fields through1001 remain bit-identical, checked in `preserved1001_prefix_equality_final.json`. Independent full lifecycle/source/quiet audits remain required; producer success does not yet make labels admissible.
