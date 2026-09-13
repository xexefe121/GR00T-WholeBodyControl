# Frozen oracle adapter review

Final reviewed adapter: `E:/codex-artifacts/sonic23_teleop_resume_20260911/student_actual_oracle_control1_v1/source_snapshot_v2/run_actual_student_oracle.py`, SHA256 `c96045c577e9047b58128c47cc098ae2f9f53d4e25eba08787c413cffe3a9086`.

Receipt `frozen_inputs_v2.json` SHA256 `675ef4844aa632bf2eb2df1e22f9852c25c986710a95b21e44c1de3531a7d33d`. Independent read-only check verified all 14 frozen source hashes and all 26 input hashes. No policy inference, model construction, physics, solver or fit executed by reviewer.

One prelaunch blocker found and fixed in preserved v2: `apply_control` appended expected physics time after calling a recorder that already appended it, doubling that ledger. The redundant append is removed; per-substep lengths are asserted. V1 was never run. Rejected imminent targets and `NoFeasiblePlan` diagnostics are now retained too.

The selected snapshot's integration, qpos/qvel, combined previous action and history exactly match final-20000 student precontrol1. Every saved prefix control array, ten actuator/metric rows, and eleven physical state/time/warning rows exactly match the original final-student trace. Named history arrays flatten exactly to the saved 300-value history. Earlier 1000-step student state is excluded.

Reviewed contracts: initialization restores full integration and separate warnings; rebuilds control0 history without normalizing the student's prior action; global counters/window/frame remain 1/11/12. Initial seeds are certified by both hard nominal rollout and full-native-copy H30, then the selected cached nominal state/cost is checked again from the same actual input. Later seeds are rerolled from the actual state, with existing guided/K0 restoration on failure. First committed MPC control advances history once and switches only the next action to applied-target normalization.

Actual every-2ms PD uses the frozen strict recorder/forecast check. The final MPC commit stops at global1269. Terminal BFM computes history on a private copy and commits raw actor history only after imminent-control certification. The separate 250-control hold retains continuous physical state/history and cannot replace lifecycle source/return/standing. Full/partial counts and failure predicates prevent incomplete label admission. Combined prefix/branch evidence is explicitly declared; branch-only slicing removes exactly one control and ten physics steps while retaining the boundary state.

No remaining code or input blocker found for the parent's authorized initial-seed stage. This is a prelaunch interface review, not a behavioral qualification. Labels remain inadmissible until independent full-source/return, strict physical/clock/warning, and both quiet-hold audits pass.
