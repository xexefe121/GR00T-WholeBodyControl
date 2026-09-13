# Native 23-DOF G1 U2 simulation status

Updated 2026-09-11 14:33 UTC. Work continues in simulation. No fast full-body controller is qualified yet.

Latest step-55,000 controller failed at 5.832 simulated seconds, during acquisition: right hip-roll speed reached 22.398976 rad/s against the original 20 rad/s limit. Independent replay reproduced all 2,916 physics steps and seven recorded state/force/clock/warning fields exactly. The original request remains 1,569 controls; 291 completed, none of 819 source controls ran, and the continuous hold was skipped. All 5,227 launch inputs remain unchanged and both processes exited. [Physics evidence](E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_independent_physics_v1/report.json).

The same controller's higher-precision export passed numerical verification: all 153,580 outputs match byte-exact across CPU/GPU/ONNX after the final float32 cast. Independent audit passed 8,638 checks. It retains the same trained weights, feature contract, native limits and 1e-5 rad export tolerance. The original float32 export failure remains preserved. Numerical consistency did not establish balance.

The 50,000 additional training updates reduced nominal loss 69.45% and physical-response loss 50.56%, while velocity-response loss worsened 22.39%. The latest simulation also failed earlier than the step-5,000 controller, whose ankle-bound failure occurred at 6.316 seconds. Saved feedback and trajectory diagnostics are being checked to select a concrete correction; more unchanged training is not selected.

The independent native simulator adapter passed 21,348 exact replay steps and a separate saved-evidence audit, including 2,141 full simulator states and ten matching full-model serializations. Cross-process mailbox preparation passed 15 independent synthetic tests covering complete publication, corruption, stale epochs, contention, interrupted writers and version exhaustion. These verify components; an actual independent 500 Hz plant / 50 Hz controller run remains unqualified.

## Qualified offline simulation

Recorded PICO motion passes the full offline planner test: 6,530 controls plus a continuous 250-control hold, all 67,800 native physics steps independently exact. All 5,780 source controls and both standing checks pass. Planning took 5,869 wall seconds for 130.6 simulated seconds, so this does not establish live teleoperation.

Watch the [complete 135.6-second physical simulation](E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4). It includes the full motion and separate hold with original timing and a fixed world camera.

[RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1) reproduces all saved commands through native physics and compares all recorded samples. Its smoke run passed. This is recorded-command replay, not live planning.

Walk002 and Walk003 also have qualified offline planner-plus-terminal-standing runs. Walk008 passed under a different offline planner configuration. The same fast controller still needs full PICO, Walk002, Walk003 and held-out Walk008 qualification.

Remaining live-simulation checks include received-input handling, independent plant/inference clocks, full 20 ms deadlines, input-fault stopping, measured standing, rearm and perturbations. Ground-truth root state and prepared motion preview remain disclosed assumptions. Real Pico, DDS and robot commands are outside this stage.

[Complete history and exact artifact identities](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/SESSION.md). [Current implementation state](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/CURRENT.md).
