# Native23 simulation result

Updated 2026-09-11T18:36:28.337698+00:00. Work continues. Native 23-DOF G1 U2 simulation runs, but the fast full-body controller is not yet qualified. Latest trained controller fails a native joint-speed limit before source motion begins. Real Pico and robot operation remain the next stage after simulation passes.

## Latest actual controller result

The history-enabled ordinary68000 model ran the original walk003 test from the complete canonical initial state. It stopped at control302, substep1, simulated time6.042s: right_hip_roll_joint velocity -21.0314182208rad/s exceeds the20rad/s native limit. It issued303 controls, including53 learned commands, and returned3021 native2ms steps. Source coverage was0; the conditional5s hold was not run. No input retiming, joint-limit changes or reset to a later point was used.

Independent native replay reproduced all3021 steps exactly across position, velocity, commanded torque, actuator force, time and both warning fields. It found the same failure with no extra replay steps. Saved input/history audit passed6967 checks: original startup250 controls, all1323 features, incoming prior23/pre-update history300, applied-target feedback, activation output and failure state agree. All53 existing same-clock feedback maps reproduce their nominal teacher commands. First input departure is251; first clipping is253;46/53 learned commands clip. At the identical query250 input, target RMSE against the expert is0.04737772244rad. After departure, same-clock teacher rows describe different states and are not fresh expert replans.

Observed policy-call p50/p95/max was6.254/7.768/14.074ms, with0 measured20ms misses in this short failed run. This timing does not establish an independent500Hz plant, complete motion coverage or stability.

Evidence under E:/codex-artifacts/sonic23_teleop_resume_20260911:

- direct_target_causal_context_evaluation_v2/nominal/trace.npz â€”589cbab8051ab9c1257049d62098248e7866b10e6c0eb085e73cbaf32d79082d.
- direct_target_causal_context_evaluation_v2/evaluation_completion_verification.json â€”c1e578569bd9f7e1bb0c86342f196440573ecbfc2f574e6fa3dce7fafc32f97e. RawPython0; diagnostic/wrapper2; all5241 inputs unchanged; processes stopped.
- direct_target_causal_context_independent_physics_v1/report.json â€”fccc2ec215c8536529161eb98982038232ac9116bcfdbec685233807b60f0753.
- direct_target_causal_context_independent_intent_v1/report.json â€”source and quiet gates false. First invocation failed before main because PYTHONPATH was missing; corrected invocation completed without any repeated task calculation. Launch failure preserved separately.
- direct_target_context_saved_semantics_review_v1/results_v1/report.json â€”e633e27d39cde9b24026519c2aeae68f44d22450cc70e33cb55b1c292ed492ae; owner512bed04e808a1a3fd1659ca6838c852e3a80a85ead7afe56f2d614b12a453ee. Saved-only audit; no new native/model calls.

## What the training comparison established

Matched blinded and causal conditions each completed exactly3000 updates from ordinary65000. Initial outputs matched exactly; all complete CPU/GPU/ORT export checks passed. Independent saved-pair audit passed118097 checks. Compared with blinded finalORT64, causal context reduced nominal MSE26.00%, full-state response error4.77%, physical one-step response error10.24% and weighted objective15.37%. This establishes utility on the saved corpus, not a stable controller. Causal full-state response error remains1.0744 times a zero-response baseline.

Joint-position and joint-velocity response errors remain about1.78 and2.07 times their zero-response energies. The current equal54-cell mean does not equalize task scale: six teacher-response energies differ13.24-fold. Root linear/angular velocity account for about62% of response loss despite being slightly better than their baselines. Convergence, hidden-state sufficiency and capacity limits remain unresolved.

## Work now underway

Corrected fixed3000-update continuation toordinary71000 launched at18:55UTC. It restores causal68000 actor, full AdamW moments/step3000, RNG, normalization and saved context/schedule. Original nominal/physical objectives and coefficient stay fixed; each response group is weighted by mean teacher energy divided by that group's energy. All original and balanced metrics remain reported. The first launch failed before any forward pass or optimizer update because the saved-request writer supplied condition twice. Actor, optimizer and RNG remained exact; failure and process evidence are preserved. A fresh directory fixes only that writer line, with three regression tests. No result from the corrected fit is claimed yet.

The unchanged native evaluator is prepared with71000 release checks and passed83 independent synthetic tests. It requires complete saved-fit and export checks before any controller test. Independent saved-fit audit preparation runs alongside training.

Corrected recorded-command clock benchmark returned4694 native steps and four exact model snapshots, then failed a native joint bound. Command420 hit a BUSY mailbox19.464ms before its deadline and was never retried or delivered. The plant held419 through control469. Six2ms timing misses also occurred, first at step610 before the command failure. The completed saved-only audit passed8605 integrity checks; physical, command and timing qualification remain false.

A bounded nonblocking publication retry now passes67 independent tests plus five subtests. It preserves command bytes, original deadline and late-result rejection; scheduler stalls remain measured. A separate saved-protocol auditor is being adapted for retry evidence. No new clock benchmark is selected, and correcting BUSY alone cannot establish2ms timing.

## Existing qualified offline evidence

Slow expert controllers already completed native physical and intent checks for full PICO6530+250 controls, walk0021417+250, and walk0031569+250, with independent replay. Full recorded PICO video is available at E:/codex-artifacts/sonic23_teleop_resume_20260911/pico_qualified_full_video_v1/full_pico_and_continuous_hold.fixed_world.mp4. These are offline expert results, not proof that the fast trained controller works.

Acceptance remains the same fast controller across full PICO, walk002, walk003 and held-out walk008; original root/yaw/feet/legs/hands/head tracking; every2ms native limits; final3s quiet and continuous5s hold. Then independently clocked500Hz plant/50Hz policy, received-only inputs, fault stopping, rearming and perturbations must pass. Current evaluation uses prepared preview and ground-truth root state, which remain disclosed.
