# Native23 simulation result

Updated 2026-09-11T20:30:40.879263+00:00. Fast full-body teleoperation remains unqualified.

New width512 model is training:6200/10000 updates last reported, no failure. All367570 initialization predictions exactly matched the previous model. Existing data, objective and acceptance limits remain fixed. Final numerical checks and physical trial still pending. The future evaluator passed121 independent tests; physics code unchanged.

Previous71000 model failed at5.758s: left hip yaw32.226rad/s against32rad/s limit. Independent2879-step replay was exact; saved input/history audit passed6658checks. Targets diverged from expert after activation; source motion and quiet hold never completed.

Recorded-command clock trial failed at6824steps after command656was lost. Four BUSY job publications recovered; worker result656 then hit BUSY with17.4ms remaining and was not retried. Saved audit confirms failure. A bounded result retry with explicit original deadline is being prepared;15plant timing misses remain a separate issue.

Slow expert controllers already passed full recordedPICO,walk002,walk003,walk008 offline. PICO video: E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. Fast-controller full-suite, independent500Hz/50Hz timing, received-only input and fault/rearm tests must pass before real Pico or robot work. Current preview and ground-truth root assumptions remain disclosed.
