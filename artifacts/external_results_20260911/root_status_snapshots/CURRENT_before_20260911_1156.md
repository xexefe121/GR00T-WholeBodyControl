# Current native23 simulation work

Updated 2026-09-11 11:23 UTC. User repeatedly instructed continue; original six-hour cutoff is superseded. Continue simulation work autonomously. No real Pico, DDS or robot commands in this stage. No fast full-body controller is qualified yet.

Repo: Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof.
NEW: E:/codex-artifacts/sonic23_teleop_resume_20260911.
OLD: E:/codex-artifacts/sonic23_teleop_six_hour_20260910.
Exact history/contracts/failures: SESSION.md. Previous CURRENT saved at NEW/root_status_snapshots/CURRENT_20260911_1123.md.

## Active work

Root selected ONE full canonical experiment in NEW/physical_student_clipped_feedback_v1. Expert owns source derivation, tests, actual selection/freeze and hidden launch after prior-outcome, source and concrete-launch reviews CLEAR, without further root permission. At this entry it is preparation only; no new run launched.

Use the ordinary final75000 head unchanged. Only native-clipped learned action components use inverse-normalized APPLIED target for feedback into seed.previous_action. All unclipped raw float32 components remain bit-exact, with original BFM-only startup/terminal behavior unchanged. Record original raw combined action, raw proposal, applied normalization, actual feedback action and clip masks separately. This tests feedback amplification from unsent target components; initial drift already preceded clipping, so no root-cause or success claim.

Reuse passed75000 WSL activation witness; no new witness call. Full original1569 controls plus conditional250 hold, all original native/source/quiet gates. Require prior-run controls0..264/all2650 steps plus actual state265 byte-exact. First feedback264 changes prior265; lag-action effect begins266. Preserve failed gates and never auto-rerun. No new fit, filter, gains, limits, source edits or checkpoint selection.

Root independent audit commands are prepared in functions stores clippedFeedbackPhysicsAuditCommand and clippedFeedbackIntentAuditCommand. Fresh outputs: NEW/student_clipped_feedback_independent_physics_v1 and student_clipped_feedback_independent_intent_v1. Neither executed. They use unchanged original referee, walk003 fixture/reference and requested1569; never shorten a failed request.

Reviewer currently owns saved-outcome review for the unmodified75000 failure plus pure fixed-map diagnostics for ALL70 actual controls250..319, matching the saved query250 committed maps. Fresh NEW/student_physical_response_fixed_map_v1; no model/native/replan calls. This is frozen-map comparison, not replanned expert truth. Afterward reviewer assesses clipped-feedback source/launch.

Pico owns separate source-only independent-clock foundation NEW/independent_plant_clock_foundation_v1. Pure fake-stepper core/tests only: fixed2ms epoch, ten-step boundaries, nonblocking command mailboxes, actual raw-action provenance/history and retained deadline debt. Forty tests passed; v1 preserved and small v2 hardens nested fault records against logger mutation. No real MuJoCo/ORT/process timing or connected controller run. Design note NEW/independent_plant_clock_design_v1/NOTE.md is pinned; STREAM_NEXT_GATE.md retains later received-input requirements.

## Latest actual outcome: unmodified75000 FAILED

NEW/one_step_physical_student_evaluation_v1. ONE hidden wrapper3168/child2900 started11:13:25 UTC; exit1/errornull11:14:35, processes absent. Failure control319/substep3 at6.385999999999519s: left_ankle_roll_joint position0.26654404676883836 exceeds upper0.2618 by0.004744046768838384rad. Velocity at failure2.504358rad/s, cap30. Requested1569, completed319, attempted320, actual3193 steps, source0/819 and no hold.

Trace38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3; report4a81e6c4d352116efe20673b36e7634fd8d3ac710ca3f00e4046bdd9b1b968f1. All1569 launch pins unchanged; prefix/activation/full291 failure/history checks pass. First learned target clip264;47/70 learned commands clipped. Earlier state drift persists: joint-position deviation RMS .005 at251, .015 at260, .0505 at270, .1588 at290, .3918 at319. Root height remains roughly.74-.79 before failure. No simple ankle-margin-only explanation.

Root independently reproduced all3193 actual steps and seven trace fields byte-exact: NEW/student_physical_response_independent_physics_v1/report.json ede8ff939da69885f509785e62e6c6cc8d0ecb190b2cb8fb15eb1e5ec4394133. Intent audit finished with source/quiet/full lifecycle false. Original request retained. Policy-only timing p50/p95/max6.82/8.71/14.95ms, zero policy-only20ms misses; this failed synchronous run does not establish independent plant or full-loop timing.

WSL witness passed exactly one head call: head_witness/witness.npz fa2dad8a24f43f577ec1120a86e41d7e76f6d09805cc0e637b14907c0a56e088; report9bc2c1f112ae321b9ef5a4a6661afe3c556381b34ee7857ad9cca52fce58e96d. Exact query250/center2038/sourceframe261 input. ORT1.23.2CPUSequential1, binaryf22a6994fbadefe3888cd3a5300a63021884372dafecaea9875987dce697c8c1. No repeated witness selected.

## Final75000 fit and verified training data

NEW/one_step_physical_student_v1/fit. One fixed5000-update continuation70001..75000 completed exit0at10:59:34. Original70000 model/allAdamW/RNG/norm/span restored exactly. All3057 nominal anchors, original velocity sampler and all3054 physical branches retained. Objective nominal+velocity+physical, coefficients1; exact fixed requested9-cell weighting, no survivor renormalization. LR3e-6→3e-7, CPU1.36.315M training head rows,293480 diagnostic rows,1150ORT calls,14 analytical heads,0BFM/native.

PT9f31d74855c28a57231c51e0a652a5f2eeeaac87a3c5aff032d59ba5ca981e34; ONNXfb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81; report5153ef3e193059062cf071b2b82c1961f7465683e8d8c9e5471dd89a27b25a1b; trainingreceiptac61a9cfdcb32f4af210664ec20562e78bf3826f85bac4e7cf5e08f300167045. Export max2.74181366e-6rad. Root saved-fit audit physical_fit_evidence_independent_v1/report.json d52a859e78f4c1bb6e84ef24cc110fcb548777ff58e854fc2a88d51071f8685a passed191851 checks, all2.88M draws/RNG/5000losses/LR/optimizer/normalization/metrics/ONNX weights+graph exact,0model/native calls. Final fit/export review11516b421e4d7555a6d2f068d0f5a6a0c4f00a604946b2cb2a15756ad7fd94c4 CLEAR.

Physical-response loss improved78.29%, velocity35.35%, but nominal loss worsened119.58%; original targetRMSE .06112→.08090rad. Preserve this tradeoff. No further fit selected.

Physical data NEW/one_step_policy_branch_collection_resume2969_v1: all3054 endpoints valid,0strictfail,61080 aggregate native steps/12216graphs. Producer reportcaf8342428008bf1b4551a0c3ee84cadc6d3b401de479a9ec7a2c7a273802fce; manifest92fa1575670b33c361d909d76cf8a892f1cdb9c420949e63b2040c6e1dd01260. Root all61080steps/full291 endpoints exact (report93ae0835...); labels167977checks exact (206ebc0c...),6111unique inputs/0conflicts. Combined review6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed. Original prefix metadata PermissionError and2969-row continuation remain preserved; no repeated generation.

## Established results and remaining scope

Recorded PICO offline planner passes full6530+250 lifecycle/hold, all67800 independently exact steps and original tracking/quiet gates. Video NEW/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4,135.6s. Repo launcher RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1 reproduces saved commands; not live planning. Planner cost5869wall seconds for130.6sim seconds.

Walk002 recorded-MPC prefix plus actual terminalBFMyaw4 passes1417+250/all16670 steps/source667/both quiet gates. Video NEW/walk002_qualified_hybrid_video_v1; one-command replay in walk002_qualified_package_v1. Original same-MPC quiet failure remains preserved. Old walk003 hybrid also passes; walk008 had a different offline configuration. None establishes the same fast controller on all four clips.

Broader PICO5980/walk002867 labels independently verified, including every6847actor/backward repeat output; not in75000 fit. Received prepared-packet gate has28 tests and2537 exact saved-feature comparisons, unconnected. Needs received-only740/760ms preview provenance, independent500Hz plant/50Hz inference, actual activation-state timing, input-fault stopping, measured standing/rearm and perturbation/full-loop deadline tests.

## Runtime and reading rules

Windows Python C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe:3.10.11,NumPy1.23.5,Torch2.10.0+cpu,ORT1.23.2. WSL Ubuntu-22.04 Python /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python:MuJoCo3.2.3,NumPy1.26.4. Native nq30/nv29/nu23,dt.002,all original strict limits unchanged. Large artifacts stay E; Z nearly full.

Mount E and run WSL in the SAME invocation through artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh (SHA392de6ec...). Direct WSL calls after idle can fail before Python because E unmounted. Safe progress readers must open FileShare.ReadWrite|Delete; do not Get-Content active atomically replaced JSON/NPZ. Existing evaluation read_status.ps1 accepts -Path. No hardware or external messaging. Shared dirty repo: preserve others' files; only own files edited.
