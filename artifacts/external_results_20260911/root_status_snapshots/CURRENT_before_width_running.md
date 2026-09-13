# Current native23 simulation work

Updated 2026-09-11T19:59:33.575360+00:00. User says continue; continue autonomously. Fast full-body controller remains unqualified. Real Pico/DDS/robot waits for simulation qualification. NEW means E:/codex-artifacts/sonic23_teleop_resume_20260911.

## Latest actual simulation: 71000 failed, independent physics confirmed

ONE canonical run NEW/direct_target_causal_response_evaluation_v1 completed after one passing WSL activation witness. Canonical start19:44:49 UTC, wrapper24380/child11172 both absent; rawPython0, diagnostic/wrapper2. Owner evaluation_completion_verification.json653fe9a1f06aa7138dad468891f823b71efa42fc9ed0ce11a83851b9c9c85fcc verifies all5242 pins. Main report4e751a4feb6f6897d0f51bd1afd4376b1954baa2d31a9a584c3c9eba65a1f665; trace78b2f9692b6de5d5d8b5310f9ac4e4803cc43a45532bb4ae0e79bea94f604801. No retry.

Strict joint-speed failure control287/substep9, t5.758s: left_hip_yaw_joint -32.225616684066146rad/s versus32.0, ratio1.007050521377067. Exactly2879 native steps,287 completed controls/288 issued,38 learned head calls,250 original BFM backward+250actor startup calls. Learned BFM0, source controls0, hold0. Full291 fixture/prefix250 and own query250 head parity passed. Same500Hz/50Hz evaluator and original limits. Physical completion, source intent and quiet remain false.

Root independently replayed all2879 recorded native steps ONCE: all seven qpos/qvel/command torque/actuator torque/time/warning fields bitexact; same failure, no steps beyond trace. NEW/direct_target_response_independent_physics_v1/report.json d1ad17ef99dd0294b5ccc402c6073536b6e49ecafeea23a86dd1d004b7ea51eb. Root saved intent report NEW/direct_target_response_independent_intent_v1/report.json acbcad444f1a8a6ba58c2aac22aeb82bfa6615afa7bcd0317abb13fbd8d66a95 completed; source intent/quiet false. Both functions causal71000PhysicsAuditStarted and causal71000IntentAuditStarted are true and completed; never rerun their commands.

Saved semantics auditor NEW/direct_target_response_saved_semantics_review_v1 prepared15 sources/56 producer synthetic tests; source f92270c7e6a61eeec05c496860ce0f40ddf36b79ee9a4cb53644c3df2e72e148. Root review source drafted but tests/request/actual audit wait until clock finishes. Unchanged trace/context/map arithmetic; actual71000 metadata and same-trace root physics/intent binding only.

## Active clock benchmark

Exactly ONE reviewed pending-BUSY clock selected and dispatched19:57:10 UTC, hidden wrapper1124, Pico sole owner. NEW/independent_plant_pending_publication_v1 request2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c, launchc7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e/3768pins, clearancedf60020ce5a53becfd5b6cafcdeba84699fe5b1569b723087e8a533ee8adde1b. Root concrete60bde676ea9e8c44d6c3abf769a9beb5bd852532d65201bff87a3754a87d3ce0. Do not duplicate. All peers source-only; no training, numerical tests or physics alongside timed plant.

Same immutable job/payload, BUSY-only maximum10 nonblocking attempts, once per eligible tick before original activation. Original admission/deadline/watchdogs/native limits unchanged. Requested1819 controls/18190steps/four MJB. Retry-aware saved audit v2 root review81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0/106tests; completed-result helper c2e17f4e4eb28c8cdb4aa95e3deb9fb2cf8beb20cedff4b0cf382caee53b44b0/10tests. Actual saved audit request waits for complete clock owner.

Previous clock failed at4694steps after dropped BUSY job420; six2ms timing misses. Owner3bf9f7ee97c7feb74434f1f928a33bb4827e62420a8f68f12f49ce5b73dd816e; saved audit48e1a6275a70998da3a2dfdda6f017b5a488c8ff4bd422d3329b4f98267ec8b1/8605checks. Completed, do not repeat. Saved timing analysis71d6ea21df432918d0c2706583dd7322e19bc6ae007456d0186491cfa7887dca shows all six misses involved body time; sleep-only fix insufficient. No new timing result yet.

## Next model preparation

Expert source-only preparation NEW/direct_target_causal_width512_student_v1, no fit selected or launched. Reviewed initializer NEW/direct_target_causal_width512_preparation_v1/source_preparation.json33c12592a71a709b78e0843219541e6cfc9124ba25027ff103820c77c7617ac2; independent27CPU tests reviewf70d2f83a841d5ffeffe8d4f3244cd846dfc2a73c22f0ad2618e7c651050b5ee. Expand1323→512→512→23 preserving old256 contractions/weights/moments and global RNG; new outgoing blocks zero, independent local incoming initialization. Six optimizer steps remain6000; new neurons share warm bias-correction age. Source proof is synthetic only, actual all367570 CUDA initialization parity must pass before updates.

Proposed fixed10000 updates to81000/optimizer16000; original10000x864 schedule must verify first3000 against previous;250 inclusive linear1e-6→1e-5 then9750 cosine1e-5→1e-6. Same N/P/balanced54 data, normalization and coefficient1.8188207859141674. Same five backends/original1e-5rad preclamp gates. Pico FP64 exporter draft preserves arithmetic with512 shapes/81000 label,20 synthetic tests unexecuted. Training/test execution waits for clock completion and concrete source checks. Capacity and learning-rate changes cannot isolate one cause.

## Completed training evidence

Corrected balanced71000 fit NEW/direct_target_causal_response_balanced_student_v2 completed3000updates/9000forwards/44,058,000rows. Reportf918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded, ownera6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3. PT395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d, ONNXc70e90adab31efd2abcebd9817bf3c3c1ecc84cde546e080f6367c710633423a. Initial all367570 predictions byteexact68000; worst export error1.0272860784255045e-8rad. N-3.520%, originalF-2.341%, balancedF-8.951%, P-4.636%. Balanced response remains1.330734x zero baseline. Numerical improvement did not improve this rollout.

Independent saved-fit audit v3 report427a91b32475e7f59212954d109865227f6b523c3e43ea9037c14462300a50ff/66447checks; owner0899cdf7b9d56481ab71ea4d9c61f86d1104c4d5238d8df4d7e1c5e4e5f7d7a3. Exact16 release roles and original gates. Failed initial fit metadata and two audit-checker attempts remain preserved; no duplicate fit/audit. Previous68000 canonical failed3021steps/right hip roll; independent physics and6967 saved semantics checks already complete.

## Qualified offline evidence and remaining acceptance

Slow expert full PICO6530+250 controls, walk0021417+250, walk0031569+250, walk0081114 already passed offline source/physical/quiet checks. This is not same fast-controller qualification. Recorded PICO video: NEW/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. Replays perform zero inference and do not establish live teleoperation.

Original artifacts/teleop_six_hour_20260910/SIM_ACCEPTANCE.md unchanged: same fast controller must pass all four full clips, source tracking, every2ms native joint/speed/effort limits, final3s quiet and continuous5s hold. Then independent500Hz plant/50Hz policy, received-only input, fault stopping/rearm and perturbations. Prepared preview and ground-truth root inputs remain disclosed. Never crop, retime, edit reference, apply root forces or relax limits to qualify.

All earlier detailed lineage retained in NEW/root_status_snapshots/CURRENT_before_71000_failure.md and earlier immutable artifacts. No real robot authorization yet.
