# Current native23 simulation work

Updated 2026-09-12T02:17:15.165992+00:00. Root sole owner; no active task processes or agents at this checkpoint.

## Main result: fast full motion and continuous hold now completed

User redirected work from audits/bookkeeping to a fast controller completing an entire motion and continuous standing hold. That nominal motion-specific milestone is now demonstrated in fresh simulation.

Implementation: E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1. Read README.md and the scripts there. It is a motion-specific compiled state-feedback controller built from the already qualified expert's saved plans. It uses current measured simulated qpos/qvel with original quaternion tangent arithmetic, all 23 target outputs, original ±0.1 rad feedback clipping and native bounds. No online planning, physical state playback, resets, reference retiming, or root forces. It requires prepared plans and privileged simulated root state. It is NOT a general learned policy, arbitrary live teleoperation, independent real-time plant qualification, or hardware readiness.

Controller phases: original BFM controls0..249; original width81000 head only at250 to reproduce the established transition; compiled feedback251..1268; original BFM yaw4 terminal and continuous hold. Actual feedback advances measured prior/history once. 204 saved plans compile into1018 rows. controller.npz SHA 7574c6efc03f3c087bf9865783dc8ab97187bb23ef2f0557437bb645e07cb46b. Compiler reproduced all1018 actual expert commands exactly;60 synthetic measured-state perturbations changed commands. This is feedback, with offline motion-specific nominal states/gains.

baseline_v1 completed1569 original controls +250 continuous hold, all18190 native2ms steps. All819 source samples and both original quiet windows pass. Root p95 .106874m, heading9.47179deg, relative hands .052616/.061867m, head .038811m, feet .101347/.088550m,12-leg RMSE .087070rad. Maximum native speed ratio .758498; no physical fault. Combined policy p50/p95/max .3224/7.400799/13.658899ms;0 policy calls over20ms. One full control tick at cold start0 took22.568299ms. Unpaced simulated36.38s completed in about9.59s excluding initialization.

Report E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/baseline_v1/report.json SHA 68ed9c12884ba779dcb241a433b199c52df2f4fb004abdf2509407742eea22ba. Full291 main-to-hold continuity checked. Every native step was executed fresh through unchanged strict assessor.

## Full-duration videos and runnable entry point

Closer full video: E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1/follow_video_v1/full_walk003_and_continuous_hold.shared_follow_camera.mp4 (SHA1d2d2e06aa157e2a3416d39227ecfa708be8a1508c4e78b1f835ee96e8af6067). Root visually inspected full contact sheet; robot unobscured, same moving camera for reference and actual. Fixed-world companion: video_v1/full_walk003_and_continuous_hold.fixed_world.mp4 (SHA97233c1a6beeee5cf0c87e96c56d05a57357e4d05a51304e2c461684a7fe691b). Both367 frames, all36.38s, timestamps checked, no new dynamics/render-time alignment. Codex open requests queued, not confirmed visibly opened.

Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/RUN_FAST_FULL_MOTION_SIM.ps1 runs a fresh original-feedback baseline; -Paced selects the warmed50Hz run with30s hold. New timestamped output each call. Launcher verifies full motion, original source thresholds and quiet windows; it does not claim timing qualification. Syntax checked.

## Timing and robustness results: still unresolved

paced_v1: full3069 controls/30690 native steps =31.38s motion +30s uninterrupted standing; all source/quiet criteria pass. 50Hz cycle scheduling with ten2ms native substeps per cycle, NOT independently paced500Hz physics.10 missed20ms cycle deadlines caused by132.49045ms policy stall at terminal handoff1269, followed by catch-up cycles1270..1278. Hold1500 controls had0 deadline misses. No altered physical outputs to conceal delay.

paced_hot_v1: same full motion +30s hold pass. Background standing-policy keepalive uses original dry startup inputs,72 attempted/returned extra noncontrol inferences, none affecting commanded targets.7 deadline misses remain; policy max21.31830ms; scheduled finish max57.272164ms; p95 finish9.7345ms. Thus keepalive is not a timing pass. Timing runs were separate from video rendering.

initial_perturbation_v1: declared initial root world-x velocity +.03m/s before any native step. Full motion+5s hold physically completed, all source metrics pass, but both quiet windows fail XY (~.1285m main/.1252m hold vs.05m).

position3_baseline_v1 and position4_baseline_v1: terminal position gain3 or4, references/native limits unchanged; both nominal full motion/hold and quiet pass. position3_perturbed_v1: main XY p95 .065544m fails; hold .006923m passes. position4_perturbed_v1: main XY .028217m passes but root/joint speed quiet gates fail (root speed p95 .2637m/s, joint speed p95 1.4358rad/s/max7.0482rad/s); hold XY .022055m and quiet pass. Neither altered-gain perturbed trial is an overall robustness pass. Do not weaken standing criteria or extend original motion timing to hide this.

No run is currently active; all raw process exits0 for these feedback rollouts, except physical/acceptance failures are explicit in report fields. Do not equate full_motion_and_hold_completed with every tracking/quiet/timing criterion; it reports physical completion only.

## Learned fit and release completed, learned physical trial failed

ONE D3 fit direct_target_width251_student_v1 completed01:29:27UTC, wrapper24844/child5864, raw0, all483pins. 10000 updates81000->91000, Adam16000->26000,lambda.2,no expansion/reset. PT3dc9f3086e7f28f12ddaa5d637165d7d654953636ac75103d9ebfe776813c584; ONNXb5a14810d98551159cde005168309ddb3e64ff51cfdda23dd161a007a3d8f5bf. Owner1b93c1d06a62b76b0fd39d5c0ae92195761007175071b1451302f4012ebb6d2d. Never repeat fit.

Saved fit validator v1 failed pre-numerical KeyError(source_sha256): concrete review binds frozen receipt, source map lives in separate source review. Preserved owner26961815a800d8d9f331e02c2e8009a571e277b87a652f80920e6f287f212f5f. Corrected one lookup in direct_target_width251_fit_independent_v2;101298 checks PASS; report99b518eec465b57addacd12b5744d34a7aeb6ce2a1e6b92914f54a18eee26361, owner001767d4ddc119f27af49229fab44e992361c9899caed924f359bca4dedfe398. No model/native calls in auditor.

Evaluation v1 witness failed before inference due Windows/Linux cosine last-bit differences in recomputed schedule. v2 uses exact selected10000 little-endian float64 schedule SHA47c79db375e6c47234be296b635521ec6528224b165559d118ae7381fef3fa76, no tolerance weakening. Linux actual passes and one-ULP mutation rejects;145 old differences max1.694e-21. Controller/physics sources unchanged. v2 witness passed1call; owner1f98ecf0e31a834117ffcf5e46fdfb43206cb75b273be1a949fa993df43e02df.

direct_target_width251_evaluation_v2 actual full attempt failed control272/sub8/t5.456s due native joint-speed ratio1.01390425. Completed272 controls/2728 native steps; source0/hold0. Policy p50/p95/max6.465/8.448/15.843ms,0 policy deadline misses. Owner e3b165ba9097681cb1c24cdcad58ad7e928c32f1357ed5c8e7370d67acdd7b09; all5251pins. Latest learned controller is less stable than prior81000, which failed309. No new learned fitting queued.

## Next work

Keep work directed at controllers completing whole motions and standing hold, not expanding audit infrastructure. Preserve nominal fast feedback success and videos. Next meaningful improvements: robust return-to-standing after changed initial state/disturbance, independent500Hz plant/50Hz control with genuine deadlines/fault handling, other complete motions, then ability to accept new motion input. Compiled motion-specific plans cannot be silently presented as live Pico/full teleoperation. Real Pico/DDS/robot commands remain unauthorized. Original timing/hand-head/12leg/native limits remain unchanged.

Previous detailed fit/data/expert/timing lineage preserved in E:/codex-artifacts/sonic23_teleop_resume_20260911/root_status_snapshots/before_fast_motion_milestone_20260912T021715Z/CURRENT.md. Original143turn conversation and mjbatch already reviewed; do not restart that work. Existing heartbeat six-hour-g1-simulation-work remains authorized after repeated continues; old six-hour cutoff expired and no longer applies. No new subagents authorized under current developer mode. All previous agents hit usage limits; do not assume they are active.
