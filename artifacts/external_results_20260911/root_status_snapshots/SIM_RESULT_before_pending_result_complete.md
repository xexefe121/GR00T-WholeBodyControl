# Native23 simulation result

Updated 2026-09-11T21:01:12.497382+00:00. Fast full-body teleoperation remains unqualified.

Width512 model completed10,000updates; saved-fit audit passed66,929checks. All367570initial predictions matched71000 exactly; export worst difference5.20e-8rad passed1e-5gate. Nominal error fell70%, balanced response19%, physical examples44%; first takeover targetRMSE improved.048→.018rad.

Fullwalk003 trial still failed6.192s: right hip pitch32.1125rad/s against32limit. Exactly3096physicssteps/310commands/60learned actions; source motion and quiet hold not reached. Independent replay reproduced every saved step exactly. Full control-cycle timing p95=8.68ms,max13.83ms,zero20msmisses. This is a controller-stability failure; timing alone does not qualify independent500Hzplant operation. Saved history/feedback audit next, then evidence-based assessment of actual failed-state expert relabeling.

Clock result retry source passed67independenttests. Matching saved auditor is being strengthened to enforce iteration order across successive results; earlier source preserved. No new clock run. Original plant deadline misses remain unresolved.

Slow offline experts already passed full recordedPICO,walk002,walk003,walk008. Video E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. Same fast-controller fullsuite, independent500Hz/50Hz timing, received-only input, fault/rearm tests remain required before real Pico/robot. Prepared preview and ground-truth root assumptions remain disclosed.
