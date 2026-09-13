# Walk002: qualified offline hybrid video and replay

Full native23 source tracking and both quiet windows pass independent audits. Original 1,417-control lifecycle and continuous 250-control hold total 16,670 native physics steps. Source tracking exactly matches the qualified original MPC prefix; the known BFM controller replaces terminal standing from control1117. The original whole-MPC quiet failure remains archived.

- [Full 33.34-second video](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_qualified_hybrid_video_v1/full_walk002_hybrid_and_continuous_hold.fixed_world.mp4)
- [Reference/actual contact sheet](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_qualified_hybrid_video_v1/contact_sheet.png)
- [Result comparison](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_qualified_package_v1/RESULT_COMPARISON.md)
- [Independent qualification](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_hybrid_root_qualification_v1/qualification.json)
- [Encoded timestamps and decoded boundaries](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_video_verification_v1/report.json)

Run the saved commands through native physics again from PowerShell:

```powershell
& 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\RUN_WALK002_RECORDED_PHYSICS_REPLAY.ps1'
```

The launcher creates a new output directory, checks frozen sources/models/qualification, replays all1,417 lifecycle commands and250 hold commands, and compares each native state, torque, actual force, clock and warning sample. It performs no controller inference. Its one packaging smoke check passed every16,670 sample exactly; [smoke receipt](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_one_command_replay_smoke_v1/replay_result.json).

The video uses recorded states, fixed world camera, and original source timing. Left: declared native23 reference. Right: recorded native physics. All338 encoded timestamps match the native2ms grid, including source-end20.34s, controller-switch22.34s, lifecycle-end28.34s and final33.34s. The final endpoint has an explicit2ms display duration, giving encoded duration33.342s. No physics or controller inference ran to render it.

These artifacts demonstrate a qualified offline saved-MPC-prefix/actual-terminal-BFM hybrid. They do not establish fresh realtime MPC, received-Pico operation, or hardware readiness.
