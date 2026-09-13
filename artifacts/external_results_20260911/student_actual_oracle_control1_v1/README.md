# One actual-student-state expert branch

This experiment queries only the final 20,000-step student's actual precontrol 1 state, at saved time 0.020000000000000004 s. The original student control 0 is the real prefix. The branch restores its complete integration state, warning counters, independently accumulated clock, four-lag history, and exact combined preclip previous action. No later failed student state or historical teacher state is used as a physical reset.

The copied native MuJoCo3.2.3 controller uses H30, ten iLQR iterations, five-control commits, all-joint predicted-state margin .05 rad/weight2000, no extra relative-foot cost, and existing hard-feasibility/guided-then-K0 restoration. Two native batch threads and one BLAS/ONNX thread are explicit. Native physics, torque/speed/joint limits, full timeline, and original goals remain fixed. Recorded BFM targets are seeds only.

The first stage certifies an initial 30-control seed in both the nominal batch model and a complete private copy of actual native MjData. The second stage attempts one uninterrupted branch through remaining entry, source and return (global controls 1–1268), then uses the existing original-goal yaw-4 BFM for controls1269–1568 and a separately stored continuous250-control hold. The first actual MPC control preserves the preceding student action in its measured lag update, then sets the next action from the actually applied MPC target. The terminal controller preserves raw BFM actions thereafter.

All accepted controls receive a private full-native imminent check; every real2ms state, actual actuator force, warning counter and accumulated clock is recorded and compared against that forecast. The combined trace explicitly concatenates the immutable one-control real student prefix and its new counterfactual branch; branch_only.npz separately removes exactly that prefix. No claim is made that both portions originally ran in one process.

All potential labels remain inadmissible until independent checks certify the full remaining source/return and both unchanged quiet-standing windows. A feasible initial horizon or partial recovery is insufficient. A failed branch is retained and stops this single query; no second state query or DAgger fit is included.

source_snapshot and frozen_inputs_v1.json preserve the unexecuted first adapter. Review caught a duplicate expected-clock log append before any run. source_snapshot_v2 removes it, asserts ledger lengths, and retains fuller failure evidence. Its controller dependencies and physical laws are unchanged. frozen_inputs.json and frozen_inputs_v2.json bind the final source, selected inputs, original goals, models, native runtime and ONNX binary.

Pinned WSL commands (after mounting E: at /mnt/e if absent):

```text
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python \
 /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/student_actual_oracle_control1_v1/source_snapshot_v2/run_actual_student_oracle.py --stage seed

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python \
 /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/student_actual_oracle_control1_v1/source_snapshot_v2/run_actual_student_oracle.py --stage continue
```

The second command requires an existing successful initial-seed certificate and refuses to overwrite an existing branch directory. These commands are an offline simulation experiment, not a real-device teleoperation launcher.
