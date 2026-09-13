# Six-hour native23 simulation effort

Started 2026-09-10 13:22:46 UTC. Deadline 19:22:46 UTC / 2026-09-11 05:22:46 Sydney.
User wants actual Unitree G1 U2 23-DOF full-body teleoperation, including legs.
Simulation first; live Pico and physical robot later. SONIC is optional per the
fully recovered earlier task. Work remains in this separate repository.

App heartbeat `six-hour-g1-simulation-work` continues this task every 10 minutes
when appropriate and must be paused at completion/deadline. Never operate DDS,
physical robot modes, motors, or Pico during this simulation effort.

## Recovered history

Read all 15 pages / 143 turns / 2,356 visible messages from archived task
`01a07113-5766-7993-982a-4a69214b143e`. See `history_digest.md` and complete
`history_visible_transcript.md`. None of its candidates passes full tracking.
Prior readiness claims confused packet flow, upright duration and robot modes
with actual motion tracking. Many small LoRA/guard/reward recipes already failed.

## Current findings at 14:10 UTC

- WSL was stopped and GPU idle when this task began. Old compact training did
  not remain active: last metrics update1015; complete checkpoint1000 recovered.
- Compact1000 improves standing but fails all four source motions. Independent
  clean GPU rollout agrees with CPU until instability; no gross ABI/PD mismatch.
- BFM native23 inference weights now load on Windows Python after installing
  safetensors. Independent actor/backward/geometry/state/history checks pass.
- `gear_sonic/scripts/evaluate_g1_true23_bfmzero.py` is the new CPU referee.
  BFM target uses published gains, published action scale and default pose,
  native effort caps and native23 physics. Policy50Hz, physics500Hz; current+7
  reference goal buffer is140ms. No root forces, pose rewrites or fallback.
- Baseline BFM completes all6530 PICO lifecycle controls including115.6s source,
  all1417walk002 and1569walk003 controls. It is NOT a tracking pass: PICO root
  p951.212m, yaw p95143.55deg, own-heading foot p95about19cm. Mild actual joint
  bound overshoot also recorded (max.00665rad); do not call zero-limit error.
- Bounded source-goal velocity feedback (position gain1,yaw gain2) improves
  drift while preserving full replay. V2 correctly expresses desired global
  root velocity in measured heading. It consumes simulation ground-truth pose;
  a physical pose estimator remains unqualified.
- Direct arm reference target with derivative feedforward reduced walk002 arm
  RMSE.166→.0435rad and preserved full replay. It is a separate optional branch,
  not the base for new residual training. Native23 reference metrics alone do
  not measure original29 hand/head intent; retain both metric sets.

## Updates after baseline

- Pure heading feedback without root XY feedback still completes PICO but root
  error p95 is2.469m; walk002 .780m. Original29 hand/head metrics are now
  independently scored; baseline PICO direct-arm relative hand p95 .285/.297m.
- All four source joint references stay inside hard model bounds. The small
  PICO actual overshoot is left ankle roll under contact load, despite targets
  well inside bounds. Maximum .00605rad in direct-arm branch. Simple target
  margins cannot resolve this particular dynamics failure.
- Residual feature/normalizer and policy CPU/GPU parity verified, including
  yaw-error cases. Portable zero checkpoint produces EXACT full CPU walk002
  trajectory of feedback v2. Update2 gives only a tiny measured improvement.
- Main residual training ACTIVE in
  `artifacts/g1_true23_bfm_residual_20260911_v1/train3h_v1`; 256env x24steps,
  3h cap, checkpoint every200. Agent owns exec session68204. Update87 at14:08.
  Learning rate reached adaptive floor1e-5; agent auditing KL read-only.
- Hidden Windows watcher PID27056 evaluates every>=200 checkpoint on all four
  full CPU lifecycle clips and scores original intent. Script
  `watch_residual_evaluations.py`; results `residual_milestones_v1`.
- Native state/history/geometry contract tests:12passed. Stream/fault harness
  and paced50Hz loop currently under independent testing.
- Sensor-only kinematic foot odometry works on short walk002/003 but drifts
  .419m PICO and .840m walk008 p95 even with ideal sensors. No hardware pose
  estimator qualified. Agent testing explicit accelerometer emulation next.

## Active work ownership

- Root: BFM CPU referee, full replay, original-intent metrics, milestone watcher,
  bounded path/heading feedback ablations and final evidence.
- `compact_learner`: new small residual PPO head on frozen BFM base, dynamic
  goal feedback v2, .15rad tanh residual. Own new files; no unchanged scratch
  extension. Zero-residual parity + GPU throughput smoke before bounded3h run.
  Preserve preclip combined BFM action history, native effort/velocity/range
  limits, original29 hand/head and native leg/foot rewards.
- `sim_inventory`: sim-only received-packet stream/fault harness, live clock,
  pause/dropout/reorder/nonfinite/rearm and paced50Hz verification.
- `history_audit`: offline sensor-only odometry probe; no controller integration.
  Earlier fixed-world requested-vs-actual videos and full history audit done.

## Next decisions

Verify new residual baseline reproduces stable BFM and run full replay at useful
checkpoints; reject regressions. Complete easy source motions plus long PICO,
standing entry/return, current-time packet loss/pause/resume, and paced50Hz
execution. Source recordings remain unmodified. Render actual simulated state
beside requested motion. No claims about arbitrary unseen motions or hardware.

## 15:09 UTC update — mjbatch steering

Deadline remains 2026-09-10 19:22:46 UTC. No full-body tracking candidate passes.

- User supplied https://github.com/kevinzakka/mjbatch. Pinned clone commit
  77966f85bcd8f7ef4351cb4a1a6f42e133d19725, isolated WSL Python3.11 environment
  `/root/.venvs/g1_true23_mjbatch`, MuJoCo3.11.0. Original Windows referee remains
  MuJoCo3.2.3. Batch vs single3.11 native physics is exact in the short test;
  64sim throughput ~5.15x serial. Cross-version equivalence remains unproven.
- Portable original native23 model, full precision model arrays, source motions,
  original29 intent and contracts in `mjbatch_native23_inputs_v1`. Upstream
  29DOF flip model, lifted reference and cropped timeline are NOT our robot.
- sim_inventory owns native23 MPC/iLQR. First0.2s standing probe has rootp954.8mm,
  legRMSE.0082rad, zero physical range excess. Planning ~2s per100ms commit:
  OFFLINE only, with declared300ms preview. Testing standing1s and complete
  initialization plus first3s walk002 source. Root owns independent Windows3.2.3
  replay of saved targets and local feedback; no postinitialization state writes.
- Residual200/400/600 do not qualify; heldoutwalk008 still stops early, PICO
  physical range excess persists. compact_learner stops only own trainer after
  complete checkpoint800. Watcher PID27056 finishes full four-clip evaluation.
  No v2 RL launch; MPC expert distillation is design only until real teacher data.
- Received-packet stream fault scenarios pass protocol and standing return.
  Full physical traces independently reproduce. Pacing still fails on loaded
  host despite Windows high-resolution sleep:151/900 missed20ms deadlines.
- Ankle torque barrier resolves small ankle range excess across six full
  reference cases, but tracking remains inaccurate. It does not solve upper-body
  violations on changed references. See `bfm_ankle_barrier_v1/OUTCOME.md`.
- Sensor-only IMU/foot odometry integrated and replay-audited. All eight ideal/
  noisy full cases complete, but root fidelity, PICO yaw and timing fail. No
  physical pose estimator qualified. See `bfm_observable_closed_loop_v1`.
- Native torso/leg/arm IK improves original29 pose intent sharply, but v2 BFM
  dynamics regress. Independent review caught walk003 adjacent knee speed24.15
  rad/s >20 despite central derivative17.54. Root loader now verifies interval
  speed, FK, derivatives and full-source receipt. V2walk003 is rejected.
- history_audit owns retarget v3: causal leg speed bounds fix all three walks;
  PICO suffers branch trap and large hand error. Bounded multistart rescue being
  tested in separate output. Do not promote v3 PICO from pose summaries alone.
- Original-intent scoring revision3 uses original root for original-relative
  hand/head metrics, separately labels retarget-root compensation, binds hashes.
  Acceptance stays predeclared in SIM_ACCEPTANCE.md; upright != tracking pass.

Current ownership: root independent MPC replay/referee, metrics and final
evidence; sim_inventory MPC and stream timing; compact_learner trainer stop800,
mjbatch runtime parity and conditional distillation design; history_audit
bounded retarget rescue and independent visual/geometry review.

## 15:29 UTC update — independent MPC replay and disk recovery

- Residual800 complete and all four CPU evaluations finished; watcher exited.
  PICO rootp95 .281m, legRMSE .176rad, rangeexcess .00198rad. Heldoutwalk008
  fails417/1114controls. v1 rejected; GPU free. Preserve complete800 separately
  from interrupted next-rollout snapshot.
- Independent3.2.3 MPC referee implemented in
  `gear_sonic/scripts/evaluate_g1_true23_mjbatch_plan_replay.py`. Standing1s
  replays with qposmaxdifference1.4e-7 using localK. Unseeded walking fails
  at same409th attemptedcontrol,7thsubstep as3.11. Openloop targets diverge
  much more and fall earlier; localfeedback cannot be omitted. Partial final
  controls no longer count as complete source/lifecycle. Original first standing
  intent metric was wrong; correctedv2 independently rerun, erratum retained.
- Independent cost/tangent/actuator witnesses found no convention bug. Short
  H15 MPC with BFM target seed still fails428controls. H30/10iteration seeded
  probe active; costpreview600ms, conservative sourcepreview740ms including
  seed's140ms. All these are OFFLINE; no expert/student yet qualified.
- compact_learner now owns native inverse-dynamics QP prototype. Standing1s
  works with native torque/contact constraints; first walking probe infeasible
  after10sourcecontrols. Correcting MuJoCo moving-point acceleration convention
  from independent Jacobian audit before bounded repeat. No artificial support.
- Root added explicit measured-body arm IK ablation in BFM referee: original29
  relative hand positions translated to measured root, current measured root/
  waist/legs, causal .12rad target steps, six solver evaluations, same BFM leg
  policy. Walk002 handp95 .141/.091m; velocityfeedforward improves .113/.086m
  but ankleovershoot .00229rad. PICO withoutmargin stops3902/6530controls.
  New optional .03rad IK interior margin and declared ankle barrier being tested
  with full2ms physical traces, separate outputs. Optional waist yaw ablation.
- Goal gyro convention ablation explicitly replaces published unscaled world
  target gyro with actor-scaled body gyro. First walk002 complete; PICO ran to
  end but ENOSPC prevented trace save, so its evidence is INVALID/incomplete and
  requires fresh replay. Canonical published convention remains default.
- Z: filled during render and saving experiments. Four COMPLETED artifact
  folders moved reversibly to `E:\codex-artifacts\sonic23_teleop_six_hour_20260910`
  with NTFS junctions at old paths: bfm_ankle_barrier_v1,
  bfm_observable_closed_loop_v1, stream_phase_grid_v1, bfm_stream_v1. All235files
  verified SHA256 unchanged; receipt on E:. ~790MB Zfree now. Current MPC/code
  paths untouched. New `bfm_online_intent_v2` is junction to roomy E: storage.
  New root runs that failed beforephysics dueENOSPC stay preserved/incomplete.
- Quiet-host full stream timing is next after activeMPC/root/WBC jobs finish;
  agents coordinated to pause heavy jobs briefly for this test.

## 16:32 UTC update — native323 full walking and PICO probes

- Native MuJoCo3.2.3 mjbatch compiled successfully in isolated E: runtime:
  `/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python`.
  Native batch/manual parity and affine-PD saturation tests pass; 22 upstream
  tests pass. Binary/source/compatibility receipt in MJBATCH323_ENVIRONMENT.md
  and E:/codex_sonic_runtime/mjbatch323_20260910/build_receipt.json. Engine and
  physical model unchanged; old3.5 GPU and3.11 experiment runtimes preserved.
- Full native323 walk002 MPC completed1417controls/667source/14170physics,
  including all return and standing. Output E:/codex-artifacts/
  sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk002_native323_h30_clip1_v1.
  H30/10iterations/commit5/8threads, local feedback capped .1rad; only BFM
  target seed, no copied physical states. Conservative preview740ms.
  Independent Windows3.2.3 feedback replay matches qpos to2.46e-11,dq8.05e-10.
  Source rootp95 .13865m,yaw1.443deg,legRMSE .11215, original hands .11067/.12078,
  head .07411. Required same-world root-relative feet .13685/.10665m; LEFTFAIL.
  Strict raw ankle range excess .00182487rad for25physics samples; LIMITFAIL.
  Producer absolute world feet .07846/.12106 are a different metric.
  Offline planning:1207.8s wall/28.34s simulated,all284planningdeadlinesmissed.
  This is progress, not full tracking or online qualification.
- Quiet-host BFM baseline received-only full stream test passes actual paced
  20ms loop:1824controls,0misses,p50/p95/max8.240/9.616/16.451ms. Independent
  every2ms state/torque/sensor/history/goal/action audit exact. Source tracking
  remains unqualified. E:/.../bfm_online_intent_v2/stream_quiet_normal_v1.
- Final v3 multistart references all4clips/10674frames pass full native timing,
  FK,bounds,adjacent speeds,derivative/quaternion audits. Portable pack at
  E:/.../mjbatch_intent_inputs_v1 includes original receipts and frozen sources.
  Original geometry cannot satisfy original29 intent for PICO/walk003/walk008;
  all later probes must declare portable v3 override. Seed validation uses
  original native reference; physical initialization uses retargetframe10.
- PICO v3 first3s after full7s entry completed500controls with zero physical
  range excess, but root/foot tracking fails. Independent Windows replay
  confirms original relative hands .0916/.0857/head.0742,yaw3.97deg; root.2537,
  feet .1825/.2404. Tiny platform roundoff grows aroundcontrol456 under huge
  saved K; physicalqmaxdelta .1689 despite same3.2.3 and identicalinitialstate.
  Both survive; this is feedback robustness failure, not qualified teacher.
  Referee now reconstructs source targets with declared producer correction
  clip; omission previously hidden because fullwalk clipping stayed inactive.
- Solver reuse optimization bitexact xs/us/K/cost versus frozen source at four
  representative knots. Oneiteration .76->.42s,10iterations4.09->3.81s;
  still offline. Cached selected seed rollout and skipped unused final FD.
- Direct goal-conditioned fast student failed balance; BFM-conditioned .25rad
  residual student also worsens legs .153->.246rad and fails heldout initial
  perturbation. Full walk saved-K expansion0/17eligible: nominal ankleexcess,
  all16initial/noise perturbations fail early. No failed traces accepted as
  teacher labels. See BFM_MPC_STUDENT_INITIAL_OUTCOME.md. No active training.
- WBC QP standing works, walking infeasible/contact acceleration mismatch;
  rejected after bounded second probe. Gyro/body-scaled BFM PICO rerun after
  disk recovery completes withrange0 but yaw49deg and originalhand errors
  .298/.324m; no qualification. Measured-arm/waist IK helps hands onwalk002,
  still root/leg tracking fails. Keep all variants separate.

Current agents: sim_inventory explicit ankle predicted-cost margin .05rad/
weight2000 fixed ablation for fullwalk002 (default parity first), then further
PICO cost diagnosis; compact_learner one bounded noise-scaled K regularization
with native ankle repulsion experiment; history_audit independent fullwalk
metric/physics review and full lifecycle comparison video. Root independent
replays, original intent qualification and robustness investigations.
Remaining deadline19:22:46UTC. No physical device actions authorized.

## 16:45 UTC update — floor geometry and limit regression

- PICO v3 requested native foot collision spheres penetrate the unchanged
  z=0 floor by29-48mm in first3s; fullrecording maximum58.176mm. Sphere-plane
  distances independently match actual MuJoCo contact distances exactly.
  This was absent from earlier FK/joint/speed validation. Root drift is mostly
  XY, not Z, so clearance correction is a hypothesis requiring fresh dynamics.
- History is preparing v4 explicit common-body upward Z transform, preserving
  every joint, quaternion, sourceframe and relative hand/head/foot task. Fixed
  causal critically damped filter15/s plus20mm buffer: no guard activations in
  all10674frames, maximum lift75.39mm, added speed.1245m/s,accel1.893m/s2.
  Posefilter haszero preview; central-difference exported velocity requires20ms
  future rawpose support, separately declared. No physicalfloor/modelchanges.
  Portablepack pending. Simagent will run fullPICO withcheckpoint500 gate;
  root will run frozenBFM onv4 as a separate fast candidate.
- Native selfcollisions also exist in v3: PICO1199frames,max114mm shoulder/torso;
  hand/hip penetration105mm. CommonZ cannotfix them. Compact probing bounded
  contact-aware arm corrections atworstframes before consideringfullretarget.
- Fullwalk ankle-cost variant .05/2000 removesankle crossings but causes new
  leftknee overshoot .004805rad for5physics samples, source5.782s. Sourcefirst3s
  passed alltracking/limit thresholds; deepertrial failed. Continuingreturntail
  unchanged to retainfulllifecycle, notpromoting. No broad margin sweep.
- Noise-scaled K cap .04rad withoutputcap.1 preservesnominalwalk but init/noise
  perturbations stillfail. Addingoldanklebarrier causesnominalfall. Rejected.
  RootPICOcap replay survives3s butroot/feetstillfail; fixedplan replay remains
  distinctfromactualclosedloopMPCreplanning.
- Root independentlychecked studentzero: residualexactzero, qmaxdifference
  1.07e-6 over500controls versusrootBFM duefloat64targetscale reassociation.
  Firsttargetdiff1.39e-17 atcontrol0;firstactionfloat32difference313. No bitexact
  rootreferee claim; its matched ownzero baseline remains valid. SidecaronE.
- Independentfullwalkvideo now complete, all28.34s fixedworldcoordinates,
  originalclockplus2msankle-eventframes, explicitstrictfailure/offline labels.
  E:/.../mjbatch_full_v1/walk002_native323_h30_clip1_v1/visual_full_lifecycle_v1/
  full_lifecycle.fixed_world.mp4. Rootinspectedcontactsheet/openedvideopanel.
- Root BFM referee now rejects MuJoCo warning/clockreset and records engine
  warningcounts/simulatedtime;12existingtests pass. Checkpointinspector now
  includesoriginalsource rootp95 foradaptedreferences. Acceptance clarified
  originalroot/heading gates remainunchanged by floor/reference adaptation.

## 17:34 UTC update — no full simulation qualification

- Deadline remains 19:22:46 UTC. Simulation only, no device work. No GPU
  training is running. Keep failed trials; no candidate promoted.
- Final v4 floor pack is E:/codex-artifacts/sonic23_teleop_six_hour_20260910/
  mjbatch_intent_floor_inputs_v1. All10674 frames independently checked.
  Common Z lift, fixed15/s filter plus20mm buffer, zero guard activations.
  Joint poses/quaternions/XY preserved from v3. This only clears geometry:
  almost all PICO source feet are now above floor; contact feasibility is
  NOT established. Source self collisions remain.
- Original-native derivative differences were legitimate WORLD BACKWARD
  angular differences versus the validator's CENTRAL convention. Do NOT
  describe original velocities as corrupt. Joint/linear discrepancies occur
  only at two splice frames. Canonical A and floor-corrected B packs are
  explicit convention ablations, independently verified15938 variant frames.
- Four baseline BFM v4 trials rejected: PICO full6530, rootoriginal.3325,
  yaw98deg, legs.1939, hands.490/.577, feet.298/.334; walk002 root.807,
  walk0031.773; walk008 stops644/1114 withrange.01127. All full raw physics
  retained, engine warnings zero. Canonical A/B PICO+walk002 also fail.
  History agent auditing these four with original-intent gates now.
- At identical actual baseline states/history, original BFM commands exactly
  reproduced200 sampled source controls. A->v3 goal retarget changes target
  RMS arms .235rad PICO/.313 walk002; derivative-only .00352/.00125 and
  floor-only .0110/.00358. Policy sensitivity demonstrated, OOD not proved.
  Evidence E:/.../bfm_goal_counterfactual_v1, independently checked.
- PICO v4 MPC prefix500: originalroot.19364 PASSES .20, legs.09073,
  originalhands/head.05095/.07830/.08238; feet.10683/.15820 right FAIL.
  Agent mistakenly stopped using assumed .10root threshold. Marker consumed
  at535; actual ankle crossing .002953 atsource3.414s independently rejects
  it anyway. Never present500root as failure. Fullsource not completed.
- walk003 v4 producer stops924, source574 attempted, physical knee excess
  .013385. Adapted root.08623, legs.08890, relative feet.08978/.09060 until
  failure. Earlier actual ankle crossings source7.456s. Promising tracking,
  strict physical limits failed. All-joint margin-only next hypothesis.
- walk002 v4 stops471/source121 attempted withrange.011697, root.6309.
  Absolute feet .0718/.1163 conceal relative feet .553/.663/root drift.
- Independent Windows3.2.3 saved-K replay strongly diverges on these cases:
  walk003 stops568 (source218 attempted), qdelta1.025/dq17.60, originalroot
  .7115; walk002 replays471 partial prefix but qdelta1.418/dq20.61, root.7514.
  No engine warnings. This tests stored feedback, not fresh replanning.
- Isolated PICO v4 FD epsilon1e-3 FAILED396 controls, only46 source attempted,
  tilt1.203/root.8887/range.002518. No feedback clips. Larger FD not promoted.
- Active sim agent: walk008 v4 baseline session8943 and walk003 v4 all-joint
  margin .05rad/weight2000 session68747. H30/10iterations/commit5/4threads,
  epsilon1e-6, feedbackclip.1. Exact default parity and13 focused tests pass.
  Paths E:/.../mjbatch_full_v1/walk008_v4_native323_full_v1 and
  walk003_v4_native323_allmargin_full_v1. Agent wiring optional relative-foot
  cost400, default0 exact parity, then full PICO v4 foot-only when slot frees.
- Root relative-foot helper g1_true23_relative_foot_cost.py independently
  matches native point-Jacobian atfour knots (max5.97e-9), common translation
  invariant, exact400*error2. Evidence E:/.../bfm_online_intent_v2/
  relative_foot_objective_witness_v1. No controller success claimed yet.
- Compact agent arm repair accepted86 frames3754-3839 then strict hand gate
  failed. Return splice violates speed, fixed-body collisions up15mm and
  accepted acceleration1192rad/s2 remain. No complete usable v5 reference.
  Now inspecting fresh Riccati feedback along accepted trajectories atfour
  knots in a separate E helper, no shared solver edits or long runs.
- Root recorded-source auditor qualify_recorded_candidate.py independently
  checks every2ms physics and original gates, always online/hardware false.
  History found torque shape and warning gate omissions; root will fix after
  their frozen A/B audit finishes. No accepted candidate can change status.
- Old native323 fullwalk remains best complete MPC: root.13865/legs.11215,
  hands.11067/.12078/head.07411, but leftrelativefoot.13685 and ankle .001825
  fail. Ankle-only .05/2000 variant trades that for knee excess.004805 and
  foot.14865. Both offline, all planning deadlines missed.

## 17:58 UTC update — walking improvement, PICO later fall

- Walk003 all-joint .05/2000 cost trial still RUNNING session68747. At1000
  controls/650 source(13s), every predeclared tracking and physical threshold
  passes: originalroot .08962m,yaw8.051deg,legs .08599rad,hands .05204/.05625,
  head .03745,relativefeet .09898/.09158,rawrange0,speed.80183,effort1. This
  exceeds prior924-control failure. Still incomplete:819source,1569lifecycle.
  Immutable1000 checkpoint SHA cc25506ebeea79891f13c0fcf8f08623a8c03c518daf77768f1498b31087cd00.
  Root sidecar E:/.../bfm_online_intent_v2/walk003_allmargin_checkpoint1000_review_v1.json.
- Root authorized full walk008 v4 identical all-joint .05/2000 only,4threads,
  H30/10iterations/commit5/defaultEPS1e-6/feedbackclip.1, no relative-foot term.
  Sim agent launching; verify session/path from their messages. This checks
  same limit correction on prior ankle/shoulder crossing. No source success yet.
- PICO v4 +relative-foot400/defaultmargin trial FAILED873 controls/8725physics,
  after522 complete+1partialsource(10.45s). It passed all first150source gates:
  rootoriginal.09077,feet.07613/.05262,hands.03876/.04335/head.04462,range0.
  Passed through700 controls without range violations, but later pitched
  forward to68.76deg/2.84rad/s and tilt1.20008, rootheight.39486 atfall.
  Final originalrootp95.20483,feet.1920/.1801,head.1223; legs.07916 and hands
  .12676/.14836. Actualrange0/speed.5021/effort1 throughout. Reject fall.
  Root geometry: pico_relativefoot_failure_geometry_v1.json. Reference actual
  rootpitch mismatch grows6.5deg at9.5s to19.3deg at10s then68.8deg; no actual
  selfcontact at finalfall. Earlier hand/hip contacts fewmm. No exclusivecause.
- Sim late-window analysis shows physical plant follows committed plans very
  closely (rootmax.57um after810); optimized costs escalate5930->73666 and
  often cannot improve shifted/recorded seeds. Local recovery optimization
  fails; no evidence of a material one-step native plant mismatch there.
- Root new materialize_mpc_checkpoint.py converts atomic incomplete checkpoint
  into clearly labelled hash-bound replay package without changing trace or
  request bytes; does not resume planning or manufacture full completion.
  PICO500 independent Windowsfeedback replay completes all500 with all3s
  tracking/physical gates passing, but qdelta.0795/dq7.09 shows different path.
  bfm_online_intent_v2/pico_relativefoot_checkpoint500_{plan,replay}_v1.
  Replayer now reports source_plan_is_final_result and source_plan_kind.
- Fresh zero-feedforward Riccati K audit rejected: four30-control windows have
  no universal winner across epsilon1e-6/1e-3. Full535-control refit nominal
  WSL replay is BITEXACT to original failed teacher; fixedmeasurementnoise
  falls at366+5substeps (only16 complete sourcecontrols). No integration.
  E:/codex_sonic_runtime/fresh_k_20260910/README.md and continuous535_v1.
- Compact now prototypes fresh BFM rollout seeds from actual CURRENT MPC
  state/history/previousappliedtargets (instead of recorded targets generated
  on different states). BoundedoneH30 atPICOcontrol800, originalnativeBFM goals,
  scored against samev4+foot400 objective. ONNXRuntime1.23.2 installed only E:/
  codex_sonic_runtime/bfm_seed_20260910/onnx_deps, pinned native323 unchanged.
  PyTorch originalBFM goldenstep800 bitexact; currentMPC last4normalizedhistory
  actions max3.778 withinactor5, oneoldpriorhistory reached6.132, disclosed.
  No full fresh-seed trial or shared MPC integration authorized yet; evaluate
  prototype evidence first. Targetsetup/prototype<=10min, no training.
- Matched full PICO selectedv3 BFM test completes6530/5780, rawlimits/clock/
  warnings pass; originalroot.32825,yaw105.91deg,legs.19511,hands.5424/.5791,
  head.1018,feet.2847/.3312. v4 samecontroller stillfails similarly; no further
  BFMphysical sweeps. Comparison E:/.../canonical_qualification_comparison_v2.
- Recorded-source auditor revision2 now enforces torque shape(N,23), requires
  all8integer zero enginewarnings; missingledger cannot pass. Negativewitness
  v2 rejects broadcastablebadtorque and nonzero/missingwarnings. Earlierwitness
  v1 retained incomplete because its oldreport lacks warningledger, correctly.
  Other gates unchanged, explicit adaptednativefoot/joint reference label.
- Preview disclosure correction CONFIRMED: H30 packet .60s/rawpose .62s;
  BFMseed packet .74s/rawpose up to.76s because stored central joint/linear
  velocity needs one additionalrawpose. Neverrewrite runningrequest/report
  hashes. Simagent preparing bound sidecars; old .74 remains packet support.
  Corrected derivative evidence original_native_derivative_audit_v2/report.json.
- PICO500 visual rendered and root inspected/opened: visual_pico_relativefoot_
  checkpoint500_v3/full_initialization_and_partial_source.fixed_world.mp4.
  Entire10s, fixedworld/unaligned original29 hand/head cyanmarkers; clearly
  incomplete150/5780 andoffline. History now rendering FULL available17.45s
  through laterfall in separate visual_pico_relativefoot_failed_full_v1,
  adding failure note to prefixREADME. No hidden cropping or physicsrerun.
- Deadline19:22:46UTC (~85min remain). Active main jobs tracked bysimagent;
  root no longrunningprocess. No GPUtraining, nohardware/deviceauthorization.

## 18:26 UTC update — one complete recorded-source pass; final trials active

- Walk003 allmargin complete1569/819/15690. Producer exact original metrics:
  root.0904586m,yaw9.032deg,legs.087376rad,hands.0526965/.0580233m,
  head.0391301m,relativefeet.100971/.0969195m,range0,speed.807809,effort1.
  Producer lacked warningledger (never inferzero); root direct audit correctly
  passed all15 gates except missing enginewarning evidence.
- History independently replayed ALL actualtargets from referenceframe10 in
  SAME pinned WSLnative323/manual500Hz. ALLcontrol and every2msq/dq/torque are
  BITEXACT producer; actual15691x8 warningcounts ANDlastinfo allzero. New
  genuine physical replay: bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1.
  Root updated auditor --require-pass exits0: ALL15 recorded-source gatespass.
  auditSHA94f4ca376828a13db2b836304ad625d3862f2be169242277e546d192048869b0.
  This is one nominal offline source/lifecycle result, notlive/fullteleop.
- Standing tail is NOTsettled: last1s rooterrorp95.01536m, final.01177;
  jointspeedp95(maxoverjoints)2.653rad/s, peak7.834rad/s rightanklepitch at
 31.126s, finalmax.493rad/s. Allnativebounds stillpass, but residualmotion
  remains. Native Windows savedKfeedback replay fails570controls/range.011676,
  qdelta.296/dq16.47, so fixedfeedback portability/robustness unresolved.
- History now rendering entire31.38s nominalwalk003, explicitrecordedpass /
  offline / unsteadyhold, fixedworld/original29markers, exact31.126event and
  final31.38. Output visual_walk003_allmargin_nominal_full_v1, noextra physics.
- Walk008 allmargin completes1114/364/11140 withrange0/speed.84649/effort1.
  Originalroot.122906,yaw3.781deg,legs.132203,hands.07169/.08178/head.04486.
  Feet.11084/.16837: RIGHTFOOTfails. Oldwarningledger absent. Noqualification.
  Combined allmargin.05/2000 PLUSrelativefoot400 now RUNNING session33831,
  path mjbatch_full_v1/walk008_v4_native323_allmargin_relativefoot_full_v1.
  Started18:22:34, H30/10iterations/commit5/4threads/defaultEPS/.1, NO freshseed.
  Onlycostchange frompriorwalk008; newwarning/clockguards on, allframeskept.
- Fresh BFMonactualstate seed works as third minimum-cost seed. Atcontrol800
  private H30cost705vsrecorded1401; at830 fresh2457vsrecorded3741 and
  retrospectivestitchedMPC3721 (truewarmplanunavailable; nofalsecomparison).
  Private physical nativeparity <9e-14; range0. Fixed5-vs10iteration control800
  comparison: costs587.17/564.19 (+4.07%), times2.69/4.59s, bothnativephysical
  valid. Root authorized FIVE iterations for fulltrial, explicitlyadditional
  change, noisolatedfreshseed causalclaim.
- FULL PICO latest RUNNING session81878, started18:17UTC, output
  mjbatch_full_v1/pico_v4_native323_freshseed_5iter_full_v1. Native323 H30,
  commit5, FIVEiterations,8threads,feedbackclip.1,EPS1e-6,DEFAULTjointmargin,
  relativefoot400, fresh/recorded/shifted minactualrollout cost. All6530/5780
  requested, packetpreview.74/rawpose.76 explicit. Forecastbeforewalk008load
 43-50min; recentloadedsolve3.3-3.6s/5controls. KeepPICO priority.
- Freshhelper g1_true23_mjbatch_bfm_seed.py +g1_true23_bfm_seed_observations.py
  validatedgolden, repeat, deep-copiedhistory/nonmutation, actual800/830,
  terminalhold andfaultinjection. Constructorunitfixedgain native TORQUE
  actuator guard; privateevery2mswarning/nonfinite/clockreset typedrejection
  preservesoldtwo-seedselection. FullhelperSHA2f3d808b6a95fd80a3c1f9a0fabbe80264cf9cc93fbf01e2b2be65f6a5ec75de.
  ORT1.23.2 only E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps.
  Coredefaultoff20arrays BITEXACT;16focusedtests includingphysicalfaultguards.
  Physicalactualhistory updatedeverycontrol exactlyonce, actualtarget normalized
  withoutclipping; localcounterfactualBFM usesownrawaction*5 history. Independent
  first400actual300-feature histories/frameindices/previousactions BITEXACT.
  SomeactualMPCnormalizedactions exceed5, disclosed; nohiddenclamp.
- FreshPICO immutable600checkpoint SHA9ad4add62b366953172f33d6d5e7f550f24a8b6e15a33edea3286829c9850a8a.
  Original5s/250source metrics ALLthresholdspass: root.150434,yaw5.299deg,
  legs.067767,hands.033584/.046479/head.037916,feet.076880/.065108,
  range0/speed.572381/effort1, allwarningcounts0. Rootsidecar
  bfm_online_intent_v2/pico_freshseed_checkpoint600_review_v1.json.
  Freshseed selected4/120plans at435/465/520/545, shifted116, recorded0,
  rejectedprivate0. Current~755sourceapproachesoldfall873, keepfulltrial.
- Existingfastresidualstudent cannotrepresentnewqualifiedwalk003 teacher:
  fixed.25radcap exceededby40.26%sourcejointcommandcomponents, everycontrol
  needs >=1larger correction. Perfectcappedhead appliedtargetRMSerror.24881rad
  (legs.22346,arms.28716), p95absresidual.8846,max2.8137. Exactteacher target
  reconstruction, zerohead/history/features passall1569states. No fit or new
  training launched; E:/codex_sonic_runtime/mpc_student_20260910/
  walk003_fixedcap_capacity_v1. Capacityproblem, not justtrainingtime.
- Rootauditor/intentinspector now accept explicitlocal --motion-override for
  immutablecross-platform producerreceipts; hashes reference andoriginal29.
  --require-pass exits1forfailed/incomplete (positivePICO3sprefix correctly
  failsonlysource/lifecycle), exits0forfullWSLwalkingpass. Frozenolderoutputs
  retained. SIM_RUNBOOK.md explains actualtestedcommands andscope.
- CompletefailedPICOfoot-only video now visual_pico_relativefoot_failed_full_v1/
  full_available_failed_source.fixed_world.mp4, all17.45s,439timestampsdecoded,
  exact17.448/17.450failurestates, noalignment/cropping; clearlyfallreject.
- DEADLINE19:22:46UTC (~57min remain). No GPUtraining orrootlongrunningjob.
  Onlysimagent twoMPCjobs above; history renderoneCPU. Do nottouchotheruserjobs.

## 18:37 UTC update — fresh PICO reaches 28 seconds; stricter margin trial

- Fresh PICO checkpoint1000 passes all thresholds over 650 source controls
  (13 seconds): original root .126746 m, yaw 2.5873 deg, legs .073204 rad,
  original hands .034182/.039920 m, head .029624 m, feet .057321/.063145 m.
  Range excess zero; speed .572381; all engine warning counts zero.
- By checkpoint1400, 1050 source controls (21 seconds), tracking still passes:
  root .119244 m, yaw 2.2863 deg, legs .072303 rad, hands .042147/.043706 m,
  head .028944 m, feet .064671/.066017 m. But LEFT ankle roll exceeds its
  lower native bound by .003564494 rad. First crossing source15.330 s,
  maximum15.876 s, 25 physics samples; target equals the native lower bound.
  This fails the physical acceptance gate despite remaining upright.
  Root independent sidecars: bfm_online_intent_v2/
  pico_freshseed_checkpoint{1000,1400}_review_v1.json.
- Preserved old run and gracefully stopped ONLY session81878 at1765 controls,
  1415 source controls (28.3 source seconds),17650 physics steps.
  Final trace SHA1520bdd45648a3e7e9d745433915cf3f2c44d82324320c67b1c9605bdd4c2fcf.
  This is a rejected, incomplete candidate, not a completed PICO trial.
- New final PICO trial started18:36 UTC, session35780:
  mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1.
  Same fresh BFM/H30/5 iterations/commit5/8 threads/EPS1e-6/.1 feedback and
  relative-foot400; adds existing all-joint interior margin .05 rad/weight2000.
  Full6530 controls requested. Deadline19:22:46; incomplete if it cannot finish.
- Walk008 combined margin + relative-foot session33831 remains running.
  No further broad sweeps. Compact agent doing read-only margin mechanism
  review; history finishing full31.38 s walk003 video and command provenance.

## 18:44 UTC update — held-out source passes; final PICO repair starts well

- Walk008 combined margin + relative-foot checkpoint800 contains ALL364
  source controls. Root independent original-intent audit passes every tracking
  threshold: root .0661637 m, yaw3.43376 deg, legs .0896106 rad,
  original hands .0624076/.0794288 m, head .0398423 m,
  relative feet .0563561/.0698077 m. All8000 physics steps have zero range
  excess, speed ratio .812339, effort ratio1, zero simulator warnings.
  Lifecycle return/standing still running; no complete qualification yet.
  Sidecar bfm_online_intent_v2/walk008_allmargin_relativefoot_checkpoint800_review_v1.json.
- New PICO stronger-margin checkpoint500 passes first150 source controls:
  original root .0983504 m, yaw3.28374 deg, legs .0591891 rad,
  original hands .0214056/.0243165 m, head .0238614 m,
  feet .0664549/.0488047 m; range0/speed.449905/effort1/warnings0.
  Main trial session35780 remains running; next key point is past source15.876 s.
- Old stopped fresh PICO final1765/1415-source audit completed:
  all tracking thresholds pass through28.3 source seconds (root .119040,
  yaw2.1932 deg, legs .070257 rad, hands .042886/.044460, head .028640,
  feet .066029/.063856). Strict audit fails source/lifecycle, explicit
  graceful-stop failure, and known ankle .00356449 rad. Evidence retained.
- Full walk003 video done, all787 frames/31.38 s independently decoded.
  Root inspected contact sheet and opened MP4; exact31.126 s ankle event and
  final31.380 s included. Full path visual_walk003_allmargin_nominal_full_v1/
  full_source_and_lifecycle.fixed_world.mp4. SIM_RUNBOOK now contains exact
  frozen same-WSL runner command and strict auditor with fresh output path.
- One NEW bounded hybrid trial assigned to history: physical replay from
  original initial state, every source + return target unchanged and checked
  bit exact, then switch to native BFM at control1269/25.38 s returned_standing.
  Actual pre-control state/applied-target histories reconstructed and checked;
  subsequent BFM rawactor*5 history convention. Original native BFM goals
  disclosed. Full300 standing controls; optional separate5 s hold only if
  strict lifecycle physics pass. One CPU thread, no new MPC solve.
  Output bfm_online_intent_v2/walk003_terminal_bfm_hybrid_v1.
- RESULTS.md working report created; must replace pending outcomes at deadline.
  No root running tool session now. Deadline19:22:46 UTC.

## 18:59 UTC update — two full walking passes; quiet standing repaired

- Walk008 combined margin+relativefoot FULL1114/364-source/11140physics
  complete. Root audit exits0/all15 gates pass, original metrics unchanged
  from800 checkpoint. Actual range0/speed.8123386/effort1/warnings0.
  Producer trace SHA509a578672fc558fa52a2ddb498e6ec1844d43034c783634b34d18d094cc581b.
  Compact independent same-WSL native physical replay reproduces every state,
  target and torque BIT EXACT;11141 warning counts ANDlastinfo snapshots0,
  maxclockerror2.01e-12. Independent output
  bfm_online_intent_v2/walk008_allmargin_relativefoot_wsl_independent_replay_v1,
  traceSHAfef64f767a29e77571d307a3a8366703caa1f57a15ff37784a5c6763d6ddf168.
- Walk008 final3s quiet diagnostic also PASS: rootXYp95 .001707m,
  originalheading .06541deg, rootspeed .01617m/s, jointspeedp95 .12127 and
  max1.26826rad/s, tiltmax .003074rad, strict physical/warnings pass.
  quiet_standing_independent.json saved beside independent replay.
- Main PICO repair checkpoint2000/1650-source (33s) passes all measured
  source thresholds: originalroot .146638m, yaw2.7581deg, legs .0769575rad,
  hands .049216/.053464, head .034457, feet .083864/.077862m.
  All20000physics range0/speed.860315/effort1/warnings0. Recent highcost/root
  excursion recovered, not a reason to declare failure. Prefix incomplete.
  Rootsidecar bfm_online_intent_v2/pico_freshseed_allmargin_checkpoint2000_review_v1.json.
  TraceSHA50a65511535900bbd398b2eab7b289f186447618b80f6823083b6f22fabee376.
  Main session35780 sole heavy MPC; estimated complete may exceed19:22:46
  deadline. Preserve honest incomplete result if necessary; no new heavy jobs.
- Compact read-only exact old ankle mechanism confirms newmargin only config
  change, sources identical. Oldplanned target atnativebound, feedback~0;
  .05/2000 adds stronger early predicted-state cost, not target interior
  constraints or hard2ms guarantees. E:/codex_sonic_runtime/bfm_seed_20260910/
  fresh_allmargin_mechanism_v1/report.json.
- Terminal hybrid yaw2 preserved source+return bitexact, switched BFM at
  control1269, full1569 sourceaudit PASS and original last3s quietPASS.
  Separate250-control5s hold physicallysafe but quietheading5.3253degFAIL.
  Thisv1 retained unchanged at walk003_terminal_bfm_hybrid_v1.
- Exactly one justified terminal-only yaw gain2->4 variant now complete:
  bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1.
  Full1569 sourceaudit PASS; original final3s ANDseparate5s-extension final3s
  quietPASS under unchanged STANDING_DIAGNOSTIC.md (declared before outcome).
  Originaltail: XYp95 .014210m, heading3.093796deg, rootspeed .000435m/s,
  jointspeedp95 .007530/max.008153rad/s, final .002517, tilt .068594rad.
  Extension: heading3.230571deg, jointspeedp95 .001664/max.001696,
  final .001469rad/s. Allnativebounds/warnings pass, no physical model change.
  TraceSHA93e108071973fb69746d22987bcd1918a20e75c14c17ddcbef8f7aa21f091488.
- Compact independently validates yaw4 AST exactly one intended2->4 literal,
  original1269-control/12690-step prefix BITEXACT to producer/yaw2, all1819
  actual state/history/action reconstructions exact, firstBFM switchhistory
  exact, extension continuous. E:/codex_sonic_runtime/bfm_seed_20260910/
  hybrid_walk003_yaw4_history_quiet_audit_v1/report.json.
- History now rendering ONE full36.38s yaw4 hybrid video at15fps, fixedworld
  original29markers, exact25.38switch/31.38lifecycle/36.38final timestamps,
  explicit separate5s extension, offline and MPC/BFM switch labels.
  No further physical variants. Rootno activeexecsession. Compactidle.
- RESULTS.md updated; SIM_RUNBOOK has fullwalk003 replay commands.
  Exact fresh PICO launch including ORT dependency path now recorded at
  mjbatch_native23_v1/REPRODUCE_CURRENT_PICO.md. No reproduction launched.
- Deadline19:22:46UTC; ~24min remain. Atdeadline preserve ownrun, pause
  heartbeat and deliver measured incomplete/failed/pass outcomes honestly.

## Final six-hour result — 19:22 UTC

- Main PICO trial ended naturally at a physical joint-limit failure, not the
  deadline. 3758 complete controls plus one half control;37585 physics steps;
  source3408 complete plus one half control of5780,68.17 source seconds.
  LEFT ankle pitch lower-bound excess .010637401 rad, first crossing at
  source68.166 s, peak68.170 s (3 samples). Actual q-.8833074 versus-.87267;
  target-.8202694 was already .0524 rad inside the bound. No fall/warnings,
  maxspeedratio.860315, effort1. Simple target interior clipping is not a
  demonstrated cure. Last-control mechanism diagnosis is separate evidence.
- Final root independent source audit exits1. Aggregate measured source
  tracking still passes: root .1628083m, yaw3.19287deg, legs .0723196rad,
  hands .0494509/.0840368m, head .0340824m, feet .0713930/.0702177m.
  Includes terminal partial sample, no full-source/lifecycle qualification.
  Failing gates are source/lifecycle completion, reported failure, joint bound.
  TraceSHA6d39520555c54a72e59e9975bef3de383f1849f207d65089c3fbff4e36bf2777.
  FinalreportSHA0f1d0b35c09ecce6e9bc76378ce20d6700987966af8fbb2506322d6f26a9b38e.
  Directory mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1.
- Final PICO planning median2.338s, p953.512s, maximum4.084s per100ms block;
  all752/752 deadlines missed. No real-time controller qualification.
- Completed positive evidence: full walk003 and held-out walk008 recorded
  source/lifecycle pass, independently reproduced bitexact in native WSL323.
  Walk008 final3s quiet pass. Separate walk003 terminalBFM yaw4 hybrid passes
  fullsource and originaltail quiet plus separate5s hold. No one unchanged
  controller has passed all4sources, perturbations and received-only timing.
- Final hybrid video completed and root inspected/opened:
  visual_walk003_terminal_bfm_yaw4_full_v1/
  full_lifecycle_and_separate_hold.fixed_world.mp4, all36.38s/549frames.
  Independent full decode/timestamps exact, source+return+switch+extension
  boundaries explicit, fixedworld original29markers, offline labels.
- RUN_PASSING_WALK.ps1 created and TESTED end to end. Pinned input hashes,
  newoutput-only guard, original fullwalking physical replay, terminal BFM,
  strict15-source audit and both quiet gates. Actual verified output
  bfm_online_intent_v2/walk003_quiet_launcher_verification_v1. No live transport.
  SIM_RUNBOOK.md includes one-command invocation. RESULTS.md is final user report.
- Compact independently reviewed RESULTS.md, actual finalPICO counts/metrics,
  positive walk/quiet claims and tested launcher. No material corrections.
- All main MPC/training/renderer jobs have ended. No robot/Pico/DDS commands.
  Heartbeat six-hour-g1-simulation-work PAUSED via official automation tool
  at19:21:57UTC; no automatic continuation beyond window. Goal remains unmet:
  native23 sim demonstrations work, full-body live teleoperation is unqualified.

- Final bounded diagnosis finished at19:22:47: actual saved prefix bitexact;
  reconstructed accepted nominal20ms endpoint violates ankle .02404418rad,
  privately completed actual endpoint .02377251rad. Prior3 nominal endpoints
  reproduce <=9.44e-16. NOT an inter-knot-only blind spot: optimizer accepted
  a predicted violation under softcost. Actualfailedtrial unchanged. Evidence
  mjbatch_full_v1/pico_repair_final_control_diagnosis_v1/{report.json,arrays.npz,source.py}.
  Final PICO artifacts +7 snapshots frozen under producer/final_frozen/receipt.json.
  All agents/jobs ended; no additional experiments after six-hour window.
