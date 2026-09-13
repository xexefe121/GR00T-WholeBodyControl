# Native 23-DOF G1 U2 simulation status

Updated 2026-09-11 14:15 UTC. Work continues. Simulation only. No fast full-body controller is qualified.

Higher-precision inference from the same step55,000 weights passed: all153,580 outputs matched exactly across CPU/GPU/ONNX after the final float32 cast. Independent audit passed8,638 checks, including exact weight promotion and every saved numerical comparison. Full walk plus conditional hold trial is selected; actual simulation qualification remains pending.

The original float32 export from step55,000 failed verification: maximum CPU/GPU/ONNX disagreement0.0000155602rad exceeds0.00001rad. That failure remains preserved. The repaired export uses float64 internal arithmetic and the same original tolerance, weights and native limits. No further training selected.

Native simulator adapter verification passed21,348 exact steps: the qualified expert walk with continuous hold, plus exact reproduction of the earlier controller's ankle-bound failure. Independent saved-evidence audit confirmed every sample,2,141 full-state boundaries and ten matching model serializations. This verifies the adapter, not a new controller or real-time teleoperation.

Fresh direct-target model FAILED acquisition at6.316 simulated seconds. Left ankle pitch reached0.525288rad, exceeding unchanged0.5236rad bound. All3,158 native steps independently reproduce exactly. None of819 source controls or separate hold ran. Saved semantic checks passed5,691 comparisons;46of66 learned commands clipped, with first input drift at251 and first clip at270. Plot: [acquisition failure](E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_saved_diagnostics_v1/acquisition_failure.png).

The original5,000 GPU updates and export had passed numerical checks: maximum preclamp disagreement6.71e−6rad,16,812 independent saved-fit checks. Its continuation reduced nominal objective69.45% and physical-response objective50.56%, while velocity-response objective worsened22.39%. These saved-data improvements do not establish closed-loop stability; the latest numerical failure must be resolved first.

Both step-75,000 controllers failed during acquisition before source motion. The original failed at6.386seconds when left ankle roll exceeded its native upper bound. The variant feeding back applied values only for clipped components failed at6.292seconds when left knee reached−0.094225rad below its unchanged−0.087267rad lower bound. Root independently reproduced all3193 and3146 actual native steps exactly. The variant's original2650-step prefix and intended prior/history change passed independent saved-array verification. No fast full-body controller is qualified yet.

Saved-state analysis shows errors beginning at control251, before target clipping at264. Correcting clipped feedback did not solve stability. The next selected experiment learns absolute joint targets from1000 measured-state and motion-goal features, excluding previous action, previous target and BFM output. All153580 verified examples remain distinct with those inputs; this does not prove state sufficiency. A fixed5000-update fit will include all9904 qualified nominal examples across five datasets plus velocity and physical-response supervision. Trainer/runtime source preparation and isolated CUDA setup are underway; no new fit or controller result exists yet. Learned-phase runtime will require zero BFM actor/backward calls and retain original native limits and full-lifecycle tests.

One-control physical training collection completed: all 3,054 policy branches remain within native limits, all labels retained. Nominal checks and policy branches used 61,080 native steps and 12,216 graph calls in total. The first attempt's file replacement failure remains preserved; continuation reused its exact 2,969-row prefix and executed only remaining generation work. Root independently reproduced all 61,080 steps and full simulator endpoints exactly, then passed 167,977 label/state/history/feature comparisons. All 6,111 combined inputs are unique. Isolated 20 ms branches do not establish sustained controller stability.

One fixed fit from update 70,000 to 75,000 completed, exit0. It adds verified physical-response examples while retaining nominal and velocity-response objectives. Root independent audit passed 191,851 comparisons, including sampling, all updates' saved losses, optimizer state, unchanged normalization, complete metrics and exported parameters. Export maximum error is 0.00000274 rad. Physical-response loss fell 78.29% and velocity-response loss fell 35.35%, while nominal loss rose 119.58%; original applied-target RMSE rose from 0.06112 to 0.08090 rad. The WSL activation witness passed; the subsequent physical failure is described above. Numerical improvements did not establish full controller stability.

The current fixed experiment trains the fast controller's response to small joint-velocity changes. All 3,057 original walking examples passed exact target, BFM output, state, history and feature reconstruction. Generation of 140,622 signed velocity probes completed. Root independently verified every saved teacher target, feature, state and clipping flag, plus 423 actor and nine backward repeat calls. All matched exactly; the complete 143,679-input dataset has no duplicate-input conflicts.

The fixed 5,000-update fit completed at 08:30:42 UTC. Ordinary final update 70,000 passed export parity (maximum difference 0.00000370 rad). Root independently checked all 2.88 million sampled pairs, final RNG, every learning rate/loss composition, unchanged normalization, optimizer counters, final ONNX parameters and graph, and core metrics over all saved initial/final predictions. Nominal loss fell 16.95%; full velocity-response loss fell 45.39%. Applied-target RMSE fell from 0.06745 to 0.06112 rad; first-control error fell from 0.03196 to 0.02861 rad.

The WSL activation witness passed, but the one fresh native simulation failed earlier: control 261, substep 6, at 5.232 seconds. Left knee speed reached 20.43808 rad/s against its native 20 rad/s cap. All 2,616 actual steps reproduce exactly in root's independent physics audit. The original request remains 1,569 controls; only 261 completed and none of the 819 source controls ran. Neither final standing nor the separate hold was reached. All 1,096 frozen launch inputs remain unchanged. The numerical improvements did not establish physical stability.

Saved evidence shows input divergence at control 251, before the first target clip at 253. Comparing against the same frozen expert feedback map evaluated on the actual state gives target RMSE 0.02861, 0.11901 and 0.29678 rad at controls 250, 251 and 252. This fixed map is not a replanned expert. The selected 12-state memory intervention completed exactly 60 graph calls and passed all baseline comparisons. Substituting the teacher's current previous action while rebuilding the matching BFM base/features reduced error by only 3.57% at251 and 21.79% at252; control254 worsened by16.97%. Independent saved review passes1,000 checks. Current-prior feedback contributes but does not explain most of the first error.

The completed collection contains 3,054 independent physical branches, each applying the frozen head's command for one 20 ms control from a recorded expert state. Exact nominal-transition checks preceded all policy branches. Resulting state, raw action history and local frozen expert feedback labels are retained. Missing old simulator snapshots were reconstructed: all 12,680 original steps and 1,269 full-state boundaries match. Collector, independent physics and complete label checks passed. The selected physical-response fit uses all 3,054 verified endpoints paired with the corresponding nominal successor. See [the selected branch design](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/PROPOSED_ONE_STEP_DISTILLATION.md).

Separate PICO and Walk002 extraction completed: 5,980 and 867 moving examples, each with its actual command, full simulator state and history. All 136 frozen inputs remain unchanged; the combined 9,904 nominal examples have no exact-input target conflicts. Independent reconstruction passed, and all 6,847 actor plus 6,847 backward repeat calls reproduced the saved outputs exactly. These broader examples are ready for later training and are not included in the selected velocity-response fit.

## Full PICO recorded motion passes the offline planner checks

Watch the [complete 135.6-second simulation video](E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4). Reference is on the left; independently audited physical motion is on the right, with a fixed world camera and original timing. The video includes the full lifecycle and separate hold. All 1,357 displayed timestamps match the recorded 2 ms physics grid.

To reproduce the recorded commands through native physics, run [RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1](Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/RUN_PICO_RECORDED_PHYSICS_REPLAY.ps1) in PowerShell. The launcher verifies pinned inputs, replays all 67,800 steps and compares every recorded sample. Its complete smoke run passed in `pico_one_command_replay_smoke_v1`. This replays the qualified commands; it does not perform live planning.

The fresh canonical run completed all6,530 control intervals and65,300 native physics steps: initial standing, acquisition, all115.6 seconds of source motion, return and final standing. Independent manual-PD replay reproduced every recorded state, command torque, actual actuator force, clock and warning counter exactly.

| Complete-source measurement | Result | Limit |
| --- | ---: | ---: |
| Original root position p95 | 14.10 cm | 20 cm |
| Original heading p95 | 4.44 degrees | 15 degrees |
| Original left/right hand position relative to pelvis p95 | 6.52 /12.20 cm | 15 cm each |
| Original head position relative to pelvis p95 | 3.19 cm | 10 cm |
| Native left/right foot position relative to pelvis p95, same world axes | 7.30 /6.37 cm | 12 cm each |
| Twelve leg joints, aggregate RMSE | 0.07972 rad | 0.15 rad |

No actual joint-range, speed, effort, fall, engine-warning or repeated-clock violation. Maximum speed ratio0.86460; effort ratio1.0. Native hard position tolerance remains1e-6 rad.

The final three seconds pass the declared quiet-standing checks: root XY p95 1.01 mm, heading0.14 degrees, root speed p95 0.01244 m/s, joint speed p95/max0.11468/1.85045 rad/s, maximum tilt0.00554 rad.

The separate continuous five-second hold also passes. Independent replay reproduces all 2,500 hold steps exactly, beginning at the independently reconstructed complete 291-value lifecycle endpoint. Its final three seconds have root XY p95 0.744 mm, heading 0.103 degrees, root speed p95 0.00796 m/s, joint speed p95/max 0.08625/0.23032 rad/s and tilt 0.00393 rad. No state reset occurs between lifecycle and hold. Total independently matched physics steps: 67,800.

Local tracking errors around source67–71 seconds remain recorded. Tracking recovered by71–73 seconds. The complete-source aggregate thresholds above pass; no source samples were removed or retimed.

Evidence:

- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_control_lm_v1/`: completed frozen canonical run.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_control_lm_independent_physics_v1/report.json`: all65,300-step independent replay.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_control_lm_independent_intent_v1/report.json`: full original-source and quiet-standing checks.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_post_recovery_tracking_windows_v1/report.json`: unchanged local diagnostic windows.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_hold_independent_physics_v1/report.json`: all 2,500 separate hold steps.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_hold_independent_intent_v1/report.json`: complete state continuity and separate quiet-standing checks.
- `E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_root_qualification_v1/qualification.json`: binds both completed root audits and original traces.

## Remaining work

This planner is too slow for live teleoperation: median plan computation4.01 seconds for a0.10-second command batch. All1,306 planning deadlines were missed. The passing recorded-motion result does not qualify a live Pico or robot session.

Four fast learned controller trials failed physical checks. Their failures remain preserved. Original BFM standing independently passes the full 250-control entry. The expert continuation from that actual standing endpoint now passes all 819 walking source controls, acquisition, return, final standing and a separate five-second hold. Every one of its 18,190 physics steps independently matches exactly.

All 1,019 moving-phase expert examples were independently reconstructed from actual commands, states and histories, with no duplicate-input conflicts across the three datasets. The fixed 5,000-update fit and ordinary final checkpoint/export at update 65,000 passed their numerical review. Its one fresh canonical test passed the exact original BFM standing prefix and first learned-input comparison, then failed acquisition at control 278/substep 4: right hip yaw reached 32.14138 rad/s against its native 32 rad/s cap. Root replay reproduces all 2,784 steps exactly. Source motion and the hold were not reached.

Saved-input diagnosis shows command error amplification before target clipping or action-history conventions diverge. A reusable private dynamics preview matches all 232 fixed independent-oracle forecasts exactly, including every one of 6,480 private physics steps and all failures. Typical 100 ms preview time fell from about 53 ms to 9 ms under concurrent load.

The one fixed first-feasible controller trial completed 282 controls, then stopped before applying control 282 because all four 100 ms candidates failed their forecasts. Root replay reproduces all 2,820 actual steps exactly, with no actual hard-limit violation. Acquisition still failed: the robot had descended to 0.404 m with vertical speed -1.633 m/s. No source sample or hold was reached. Nine of 33 moving decisions exceeded 20 ms; this trial does not qualify live timing.

A fixed diagnostic evaluated all four candidates at each saved actual state 250 through 282: 132 outcomes retained, 6,154 private steps, no connected controller. All 50 previously recorded forecasts repeated exactly. Using the exact original MPC cost terms over the five-control prefix, primary scored worse than BFM at every state 250 through 267, even though all four were still physically feasible.

The subsequent single cost-ranked trial also failed. It completed 430 controls / 4,300 actual native steps, then rejected all four candidates before control 430. Root replay reproduces every applied sample exactly; no rejected target was applied. It reached only 80 of 819 source controls, with root p95 error 58.87 cm, yaw 23.98 degrees, foot errors 55.86/73.33 cm and leg RMSE 0.24985 rad. All original source tracking gates fail and no hold ran. Among 180 applied moving decisions, primary was selected once, BFM 66 times, previous target 99 and current q 14. The standing preference did not produce walking progress or eventual recoverability. All 181 attempted moving cycles missed 20 ms (median 46.97 ms). Same frozen 65,000 head, model, references, 100 ms horizon and hard limits were retained. These results shift work back to learner feedback behavior and broader full-body data; no further candidate-filter trial is selected.

The fresh walk002 run with the exact passing PICO planner configuration completes all 1,417 controls. Root replay reproduces every one of 14,170 physics steps exactly: no range violation, maximum speed ratio 0.80877, effort ratio 1.0 and no warnings. All 667 original-source samples pass: root p95 15.91 cm, yaw 2.84 degrees, relative hands 5.03/5.06 cm, head 3.73 cm, feet 7.80/7.02 cm and leg RMSE 0.11028 rad.

The original same-MPC walk002 run fails its quiet-standing window: root speed p95 0.09404 m/s versus 0.05, joint speed p95 1.09577 rad/s versus 0.5, and maximum 5.57707 rad/s versus 2.0. That failure and the unexecuted same-planner hold draft remain preserved.

The subsequent walk002 hybrid passes the complete lifecycle and continuous five-second hold. It physically replays the first 1,117 recorded MPC commands from the canonical state, then runs the original BFM yaw4 standing controller for 300 terminal controls and 250 additional hold controls. All 11,170 prefix physics samples, 1,118 measured histories, and the independently reconstructed complete 291-value switch state match exactly before the first BFM action; no state is injected at the switch. Root independently reproduces all 16,670 physics steps and verifies all 667 original-source samples against the unchanged thresholds. Both original and separate quiet windows pass. Main/hold root speed p95 is 0.000427/0.000248 m/s; joint speed p95 is 0.00889/0.00184 rad/s. This is recorded MPC command replay plus actual BFM terminal inference, not fresh MPC planning or live teleoperation. Evidence: [complete root qualification](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_hybrid_root_qualification_v1/qualification.json).

Watch the [complete 33.34-second walk002 hybrid and hold](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_qualified_hybrid_video_v1/full_walk002_hybrid_and_continuous_hold.fixed_world.mp4). All 338 encoded timestamps match the native physics grid, including exact controller/hold boundaries; the final frame has an explicitly recorded 2 ms display duration. [RUN_WALK002_RECORDED_PHYSICS_REPLAY.ps1](E:/codex-artifacts/sonic23_teleop_resume_20260911/walk002_qualified_package_v1/RUN_WALK002_RECORDED_PHYSICS_REPLAY.ps1) reproduces all 16,670 recorded-command steps and passed its one complete smoke run. Root viewed the decoded handoff frame. Playback does not rerun MPC or BFM inference.

A separate compiled private-preview prototype v2 matches all 364 fixed forecast cases and all 13,560 computed native steps against the independent referee. It explicitly retains 926 computed steps beyond first failures. An immediate call lock and owned target buffer address issues found in v1 review; 23 input-boundary checks and two target-alias checks pass, with 20 additional private steps. Independent v2 review is CLEAR. Median 100 ms preview time is 5.63 ms; four candidates plus policy inference are not qualified within 20 ms. Neither completed controller trial used this compiled backend.

The standalone prepared-packet buffer passes28 tests and exactly reconstructs all2,537 stored1069-component policy inputs. It is not yet connected to a controller. Upstream raw-pose provenance, independent500/50Hz scheduling, physical stopping/rearm, full-loop timing and perturbation tests remain.

The same final fast controller must still pass PICO, walk002, walk003 and held-out walk008. Live Pico reconstruction and real hardware are later stages.

Full handoff details: `CURRENT.md`, `SESSION.md`, `STREAM_NEXT_GATE.md` and the original `artifacts/teleop_six_hour_20260910/SIM_ACCEPTANCE.md`.
