# One native23 policy through standing and movement

Started2026-09-12 05:17UTC, within original08:46:40UTC decision deadline.
Previous two learned families are finished and rejected. This is a changed
controller architecture, not an extension of either fit.

`g1_true23_single_policy.py` starts from the native23 BFM balance actor and
trains all36 actor parameter tensors. Its observation normalizers and backward
reference encoder remain fixed. A zero-initialized goal adapter receives all
1323 existing causal features (legs, feet, hands, head, current/past poses,
robot state, previous applied targets and measured history). A zero-initialized
output adapter changes coordinates over the full native joint interval.
There is no separate motion policy, target handoff, phase selector, future
access, motion-specific gain table, or small0.15rad residual cap.

Initial goal is the already tested neutral pose with received-root XY/yaw and
measured-velocity balance feedback. Full-body pose conditioning is learned by
the goal adapter. Therefore the initial policy is NOT a demonstrated motion
tracker; it is deliberately initialized to preserve balance. All original
full-body objectives remain in the dynamics reward and final referee.

The shared received-only CausalController loads its ONNX output unchanged.
Training reuses actual expert and interruption-state reset pools, native PD
and MJLab simulation. It applies this one actor throughout; no frozen balance
target blend is executed. Training uses expert targets as a regularizer while
PPO drives adaptation through physical rollouts. No supervised bootstrap.
All evaluation uses original native3.2.3 physics and30s continuous hold.

Initial integration check:12 actual expert states, max target difference from
the existing neutral balance function3.46e-6rad. All36 backbone tensors receive
nonzero gradients. Output adapter can reach both native endpoints for all23
joints to1e-5rad. These checks do not establish physical motion success.

`single_policy_smoke_v1` stopped before PPO: ONNX exporter required a static
LayerNorm width. Fixed by using parameter shape; no behavior change intended.
`single_policy_smoke_v2` is a4-update laptop throughput test,32 environments,
32 steps,8 minibatches, no extra training hold. It pauses at initial export
pending native30s standing and complete-motion baseline. Do not release it
until initial standing passes. Artifacts under E:/codex-artifacts/
sonic23_teleop_resume_20260911/causal_dynamics_v1.

Update05:29UTC: initial shared ONNX policy passes30s native standing at0,+.03,
-.03m/s. Full attempts physically execute every source packet and30s hold on
all four clips, but all fail full-body tracking. Initial goal adapter is zero;
this result demonstrates retained balance, not matching the body motion.
Root p95 errors: walk0031.531m,walk002.402m,Pico.186m,walk0082.085m.
Leg RMSE:.377,.282,.319,.337rad, all above.15. All30s hold quiet windows pass;
walk008 main quiet XY.050363m fails. No independent timing qualification.
Reports: single_policy_initial_standing_v1 and single_policy_initial_full_v1.
CONTINUE released only the4-update throughput smoke after these results.

Smoke completed4 updates. Measured iteration times2.28..3.29s for1024
control transitions; roughly300..400 transitions/s including iteration
overhead/export. Full backbone KL constraint settled actor learning rate to
1e-8. Main starts at that measured rate, retains existing adaptive KL limit.
Smoke4 export passes all three native30s standing cases and completes the
full61.38s walk003 attempt physically with both quiet windows. Motion tracking
still fails: rootp95 1.509m, yaw82.34deg, legRMSE.3745rad. Runtime timing remains
unqualified; this unpaced test had a31.18ms maximum policy call.

Main pilot_single_policy_v1 started05:31UTC, released05:33UTC after verifying
its resumed ONNX is byte-identical to tested smoke4.64 environments x64 steps,
32 minibatches,396 additional updates, global total400. Two scheduled exports:
actor_00198.onnx (global202), actor_00396.onnx (global400). Evaluate each with
the existing evaluate_g1_true23_causal_dynamics script, all four clips and
--hold30, WITHOUT --balanced. Also recheck standing and input loss if physical
tracking improves. Actor and critic resume, optimizer fresh. No bootstrap.
Training wall cap1.8h including initialization, still before08:46:40UTC.
Do not extend after the scheduled final checkpoint. Source snapshot lives in
the main pilot directory; the previous families and failed export are intact.
First3 main iterations measure534..556 control transitions/s (7.05..7.66s per
4096-transition update). Unified exec session55455 owns this sole ML process.
At that initial throughput, scheduled checks are roughly06:00 and06:25UTC;
these estimates can move with recovery-state physics and laptop load.

Global202 scheduled full evaluation finished: all four physical durations
and all main/hold quiet windows pass, but all full-body tracking gates fail.
Rootp95:1.495,.406,.187,1.966m; headingp95:80.5,103.1,20.5,95.5deg;
legRMSE:.381,.280,.316,.325rad (walk003,walk002,Pico,walk008 order).
Hand errors reach.83m; smaller errors on a few measures do not offset large
heading/hand regressions. Candidate rejected. Main continued to its already
scheduled global400 endpoint; at06:16UTC global295 completed. Final all-four
evaluation is queued on actor_00396's normalization marker. No duplicate.

Do not assume success from loss, training tails, exit0 or standing-only
results. Keep original all-four motion, held-out walk008, limits, tracking,
timing, disturbance and input-loss/rearm gates. No hardware or paid compute.

Final global400 completed and evaluated06:35UTC. All four source gates fail;
physical durations complete, but Pico main and hold quiet windows regress.
Rootp95:1.450,.407,.188,1.854m; heading:82.3,88.6,19.4,96.0deg;
legRMSE:.374,.250,.330,.307rad. Hands reach.817m error. No candidate selected.
Main training completed396 additional updates in3306.7s,1,622,016 transitions,
about501 control transitions/s. Family stopped at400 including4 smoke updates.
No active ML/evaluation jobs. Do not extend the completed recipe unchanged.

Remaining input-routing issue: pretrained backward encoder receives neutral
pose plus feedback; full received body poses must be learned by the newly
initialized adapter. Next concrete architecture change, if attempted within
the remaining original time, is direct causal body-pose encoding using the
existing BFM reference_features and backward map. Keep a single trainable
actor and full native range, and retain actual hand/head task inputs and
rewards. Recompute all velocities from received past; no original centrally
differentiated velocities or future windows. First demonstrate initialization
standing and original-speed full baseline, then decide whether a bounded
pilot fits remaining time. No new code or training for this route exists yet.
