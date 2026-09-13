# Native23 simulation result

Updated 2026-09-11T19:59:33.575360+00:00. Fast full-body teleoperation is not yet qualified in simulation.

The latest71000 model failed before source motion at5.758s, control287/substep9. Left hip yaw reached-32.226rad/s against32rad/s native limit. Independent replay reproduced all2879 physics steps exactly, including the failure. Full source motion and quiet hold remain incomplete. The earlier68000 model lasted3021steps; lower training loss did not improve this rollout.

Training and export verification passed: all3000updates, allfive numerical backends, worst same-weight export error1.03e-8rad and66447 independent saved-data checks. Those results establish numerical integrity, not stable behavior.

One independent recorded-command clock trial with bounded BUSY retry is active, started19:57UTC. Training and numerical tests are paused during it. Previous clock failed a dropped command and six2ms deadlines. Original timing and physical gates remain unchanged.

Next candidate is a width512 model initialized to preserve previous predictions and optimizer state, followed by fixed10000updates with a gentle initial learning-rate ramp. Source preparation underway; no new training yet. The candidate must pass actual initialization/export checks and full physics evaluation.

Slow expert controllers already passed full recorded PICO, walk002, walk003 and walk008 offline. Full PICO video: E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. These results do not prove the fast controller or live Pico path.

Acceptance requires the same fast controller across all full clips, strict native limits, original source tracking, final3s quiet plus continuous5s hold, then independently clocked500Hz simulation/50Hz policy with received-only inputs and fault/rearm tests. Current prepared preview and ground-truth root inputs remain disclosed. Real Pico and robot work wait for simulation confidence.
