# Paired true23 frozen-LoRA live PICO simulator test

> **2026-09-10 priority change:** Full-body VR is now the active target; dance
> qualification is paused. See `VR_READINESS.md` at the repository root.
> Receiver startup CRLF bug is fixed; current input status distinguishes
> `receiver_reachable_tracking_unavailable` from a stopped receiver. Current PC
> Wi-Fi IPv4 is192.168.1.7. The paired simulator commands below remain diagnostic,
> not physical readiness. Historical dance launch/qualification sections do not
> define the new VR acceptance path or authorize hardware operation.

> **2026-09-09,23:04 Sydney:** Joint world-task quality experiment COMPLETE,
> REJECTED.100 updates/204800 transitions independently audit. CPU completion
> improves2/4 to4/4; GPU4/4, but all world tracking screens FAIL. Mean GPU root
> error worsens3.09%; walk008 worsens8.58%.5396 CPU outputs/53960 physics steps
> reproduce exactly; actual range/effort excess zero. Target jumps up to1.268rad/
>20ms remain unqualified.47 quality tests pass. No recipe extension, active run,
> hardware/default change, export or damping fix. Full-body teleop NOT READY.
> See PROGRESS.md and world_quality_bonus_v1/OUTCOME.md in the artifact root.

> **2026-09-09,21:39 Sydney:** Same-checkpoint full-pose encoder test COMPLETE,
> REJECTED. Original29 source model, not native23:3/3 lifecycles finish,0/3 world
> tracking screens pass. Root p950.590/1.236/2.476m. Mean root improves10.71%,
> but two leg regressions exceed predeclared5% limit.4100 outputs/41000 physics
> substeps independently reproduce exactly.003 actual range excess0.00767rad;
> target jumps up to2.15rad/20ms are not hardware-qualified. No native23 retry,
> training, live/default change or hardware use. Full-body teleop NOT READY.
> See PROGRESS.md and full_pose_encoder_v1/OUTCOME.md in the artifact root.

> **2026-09-09,21:09 Sydney:** Raw-request history counterfactual COMPLETE,
> not a deployment fix.3/4 CPU lifecycles finish,0/4 tracking screens pass;
>008 now guard-stops.4835 policy outputs/48350 physics substeps independently
> reproduce exactly. Same weights, targets, limits and source speed; only retained
> previous-action inputs changed. Deliberate runtime-distribution change, not a
> compatible checkpoint relabel.27 tests pass; default profile unchanged.
> No hardware use or new video. Full-body teleop NOT READY. See PROGRESS.md.

> **2026-09-09,20:36 Sydney:** World-tracking100 COMPLETE and REJECTED. Training
> audit verifies204800 transitions. CPU002/003 stop at range guard;008/dance finish
> but fail tracking. GPU finishes all four; all four tracking screens fail, root
> p950.770/2.215/3.174/1.835m.36590 CPU substeps independently reproduce exactly;
>53960 saved GPU substeps audit cleanly. Turn root error improves22%, but walk002
> and every leg-RMSE comparison worsen; final standing unqualified. Video verified
>1570 frames/31.40s.47 new+28 existing tests pass. All processes terminal, no
> automatic training extension, promotion, live export or hardware commands.
> Full-body teleop NOT READY. See PROGRESS.md and world_tracking_termination_v1/OUTCOME.md.

> **2026-09-09,19:47 Sydney:** First world-tracking regression was interrupted
> before any trained snapshot was saved;83 logged updates are not a usable policy.
> Existing files preserved. Identical capped recipe now running in separate
> regression_recovery_v1 with durable logs (Windows PID19612, Linux PID371).
>47 new tests pass; full-motion candidate tests still pending. No hardware use,
> source-speed/limit relaxation or live export. Full-body teleop NOT READY.

> **2026-09-09,18:28 Sydney:** Added world-root training failure>0.30m without
> changing old failures, rewards, physical limits or tracking acceptance gates.
>24 tests and64-transition GPU smoke audit pass; actual new terminal events not
> yet exercised by smoke. Fresh capped100 regression39486 running. No tracking
> gain claimed; all four complete CPU/GPU motion tests remain required. Distinct
> SIM-only snapshots; no live launcher or robot commands. Full-body teleop NOT READY.
> See PROGRESS.md and world_tracking_termination_v1/EXPERIMENT.md.

> **2026-09-09,17:55 Sydney:** Full training-engine comparison COMPLETE:
> all four timelines integrate, all four tracking screens FAIL. Public002/003/008
> root p95 is0.684/2.827/3.208m; SONIC dance1.956m.53960 measured substeps audit
> cleanly; this is not successful motion or hardware return. Poor world tracking
> exists in both CPU and GPU loops. Candidate remains REJECTED; no live export,
> physical commands or policy changes.28 tests pass. Public003 video verified:
>1570 frames,31.40s, fixed world camera visibly preserves final2.818m root error;
> full-body teleop NOT READY. See PROGRESS.md and training_engine_lifecycle_v1/OUTCOME.md.

> **2026-09-09,17:16 Sydney:** New measured input and one-control physics probes
> COMPLETE.3028 training/replay inputs have identical FSQ tokens and raw-action
> differences below3.8e-6.32 independent20ms GPU/CPU probes have joint-position
> differences below5.1e-6rad. No gross mismatch found in these tested boundaries;
> NOT full closed-loop parity or improved tracking. Bounded-progress100 remains
> REJECTED. No policy update, hardware command or active probe. Next: matched
> full-lifecycle training-engine control test; limits unchanged. Full-body teleop
> NOT READY. See PROGRESS.md and train_replay_boundary_v1/OUTCOME.md.

> **2026-09-09,16:39 Sydney:** Bounded-progress100 COMPLETE and REJECTED.
>2/4 full lifecycles finish but miss tracking;003/dance stop at unchanged range
>guard. Dance already has1.870m root p95 before failing return-to-stand.41720
>physics substeps and4172 singleton inference rows independently reproduce
>exactly. First batch32 audit mismatch is preserved and diagnosed; no thresholds
>relaxed. Full public008 comparison video verified. No training/evaluation session
>left running, no live export or robot commands. Full-body teleop NOT READY.
>See PROGRESS.md and bounded_progress_v1/OUTCOME.md.

> **2026-09-09,15:55 Sydney:** Bounded-cost/progress SIM reward implementation
> and2-update GPU smoke audited; fresh capped100 regression33953 ACTIVE.
>43 new tests and20 existing reader checks pass. No full-motion improvement
> established; previous decoder-wide100 remains rejected. Reward-scale evidence
> does not establish physical damping causality. No hardware/launcher/interlock
> change. Full-body live teleop remains NOT READY. See PROGRESS.md and
> bounded_progress_v1/EXPERIMENT.md.

> **2026-09-09,14:55 Sydney:** Decoder-wide100 is COMPLETE and REJECTED for
> deployment.45 tests and saved-update audit pass; all four full CPU replays
> complete. Public002/003/008 root errors worsen31/48/26%; dance improves26%
> but leg and relative upper-body errors worsen. All world screens fail and
> standing/contact remain unqualified. Independent53960-substep physics audit
> reproduces all states exactly. Zero range excess does not qualify target jumps
> up to1.17rad/20ms or live timing. No automatic training extension, hardware
> launcher/export change, robot commands or damping fix. See PROGRESS.md and
> decoder_wide_lora_v1/OUTCOME.md. Live full-body teleop remains NOT READY.

> **2026-09-09,14:31 Sydney:** Decoder-wide original-intent LoRA is implemented;
>45 tests and the2-update GPU smoke audit pass. All28 released tensors remain
>frozen; all nine affines now have adapters. Fresh capped100 run ACTIVE in90621;
>initial actor/critic/data/reset and5396 same-input means match the first-layer
>baseline. Four full guarded CPU motion tests and final audit remain pending.
>No live launcher/export accepts this research candidate; no robot commands or
>motion-quality gain claimed. See PROGRESS.md and decoder_wide_lora_v1/EXPERIMENT.md.

> **2026-09-09,13:57 Sydney:** Root-position gain0/gain2 closed-loop test COMPLETE,
> both changes REJECTED. Eight full-duration attempts: six finish but fail world
> tracking; both changed gains stop walk002 at the unchanged range guard. No
> unchecked target is applied.30 tests and90060 independently replayed substeps
> verify outcomes, not live readiness. Baseline weights/gain and all hardware
> launchers remain unchanged. No automatic gain/training extension or robot use.
> See PROGRESS.md and root_gain_closed_loop_v1/OUTCOME.md. Full-body control,
> standing/handoff, live timing/estimation and physical damping remain unresolved.

> **2026-09-09,13:26 Sydney:** Offline predictive braking removes ankle-range
> excess in four full SIM trials; all 53960 new substeps independently replay
> bit-exactly. Gains, limits, source and actor unchanged. Four controls intervene.
> Path/standing still fail; corrected walks have slightly worse leg tracking.
> Abrupt target changes and full 50 Hz timing remain unqualified. This is NOT a
> hardware-safe controller or a fix for physical motor damping. No live launcher
> changed, no robot commands. See PROGRESS.md and range_preview_v1/OUTCOME.md.

> **2026-09-09,12:40 Sydney:** Unclipped-critic100 COMPLETE, REJECTED. All four
> full CPU lifecycles complete. Walk/turn path errors improve23/34%, but excluded
> development/dance worsen0.8/19.3%. All world/standing screens fail; walk002 ankle
> overshoot increases to0.017rad.18 tests and independent107920-substep audit
> verify implementation, not readiness. No extension, promotion, physical
> commands or interlock changes. See PROGRESS.md for full comparison and video.
> Live estimation, headset/handoff and prior physical damping remain unresolved.

> **2026-09-09,12:08 Sydney:** Value-learning diagnostic verified8192 actual
> simulator transitions and bit-exact replay of eight disposable critic fits.
> Unclipped critic fits lower error22–30%; controller improvement NOT proven.
> A fresh100-update one-boolean critic-loss experiment is now being tested.
> Same physical limits and four full-motion screens remain mandatory. Do not
> deploy the prior rejected pose100 or infer live readiness from critic loss.

> **2026-09-09,11:21 Sydney:** Pose-conditioned100 experiment COMPLETE, REJECTED.
> Full walk002/turn003/walk008/original dance finish, but root-path p95 remains
> 0.864/2.965/3.385/1.808m, all worse than initial. Some leg errors improve;
> all world/standing screens fail and two walks exceed actual joint ranges.
>40 tests, strict saved-weight checks and107920-substep physics audit pass;
> motion quality does not. This distinct first-affine LoRA research checkpoint
> is not accepted by legacy live/export tooling. No further updates, physical
> commands, promotion or interlock changes. Estimation, headset/dropout/handoff
> and prior motor damping remain unresolved. Do not deploy this candidate.

> **2026-09-09,10:26 Sydney:** Three-recording100 experiment COMPLETE and REJECTED.
> All four full CPU replays finish upright, but source-path root p95 worsens:
> walk002 0.733→0.843m, turn003 2.074→3.139m, excluded-development walk008
> 3.232→3.650m, original SONIC dance 1.709→1.746m. All world/standing screens
> fail; two clips retain physical joint-range excess.57 focused tests and
> independent update/107920-substep audits pass; motion quality does not.
> Training, six new CPU attempts and full dance rendering have finished.
> No extension, promotion, live export, physical commands or interlock changes.
> Live estimation, headset/dropout/handoff and prior motor damping remain
> unresolved. Do not deploy this candidate. See PROGRESS.md and
> original_intent_multimotion_v1/regression_v1/OUTCOME.md in the artifact root.

> **2026-09-09,09:17 Sydney:** Frozen-decoder100 experiment finished and REJECTED.
> Full walk/turn complete, but turning root error worsens2.074→2.762m, walking
> range excess grows0.0082→0.0136rad and turning gains0.0074rad excess. All world/
> standing screens fail.130 tests and independent saved-weight/physics audits
> pass; base weights truly stay frozen, but motion quality does not qualify.
> No extension, live export, physical control or interlock change. Runtime
> estimation, headset/dropout/handoff and prior motor damping are still unresolved.
> Do not deploy this candidate. See `PROGRESS.md` and the frozen-decoder outcome.

> **2026-09-09,08:50 Sydney:** Component isolation confirms decoder updates worsen
> turning. Root correction with the original decoder lowers error2.074→1.983m,
> still a tracking/standing FAIL, not a deployment candidate. A distinct frozen-
> decoder actor now trains only root correction/exploration/critic;130 tests and
> a two-update GPU smoke pass, with all28 base tensors unchanged. One fresh capped
> 100-update regression is initializing. No full-motion improvement for it is
> claimed yet. Old live/export tools reject its research snapshot. Hardware root
> estimation, transport/dropout, handoff and previous robot damping stay unresolved;
> commands and interlocks below are unchanged. No physical deployment authorized.

> **2026-09-09, 08:06 Sydney:** Source-aware100-update regression finished;
> candidate REJECTED. Full walk and turn complete upright, but walking joint
> range excess grows0.008217→0.020089rad and turning root error2.074→2.644m.
> All tracking/standing screens still fail.93 tests and saved-state update/physics
> audits pass; controller quality does not. No further updates on this run.
> Original-intent actor remains a distinct offline research contract, not an
> accepted live export. Commands below, physical limits, estimator status and
> the prior motor-damping diagnosis are unchanged. Do not deploy this candidate.

> **2026-09-09, 07:43 Sydney:** New original-source-task training recipe now runs
> with23 physical actions and preserved29-source hand/head intent. Two-update
> GPU smoke is verified, including frozen encoder and finite critic/optimizer.
> Both full public walking CPU attempts complete, but tracking still fails:
> root p95 worsens0.733→0.796m and joint-range excess0.008217→0.017910rad.
> Candidate is rejected for deployment. A separate capped100-update regression
> is running; its outcome is not known. This new actor uses a distinct contract;
> old export/live tools must reject it. No commands below, physical limits,
> estimator qualification or previous motor-damping diagnosis change. Original
> hand/head targets are now trained, but live full-body teleop is not qualified.

> **2026-09-09, 06:45 Sydney:** Full original SONIC dance now completes all546
> source frames plus entry/return in both native23 reference variants and the
> original29 comparison. No fall or hard joint-range excess in these three runs.
> Tracking still FAILS: preserved-source native23 has1.683m root-position p95,
> 0.341/0.360/0.249m pelvis-centered original hand/head position p95 and poor final
> standing accuracy. New lossless original29 source bundles retain missing-axis
> intent without adding physical axes;49 tests pass. Existing training rewards
> can still maximize native-pose tracking despite11–23cm original hand error.
> No new reward, trained policy, live receiver, physical damping fix, limit or
> command below is installed. Source convention checks pass, but that is not a
> controller qualification. The old runbook and newer research ABI remain distinct.

> **2026-09-09, 05:43 Sydney:** Restoring original29 hand/head source VR21 while
> retaining native23 physics, weights, limits and timing lets the zero-update
> actor finish all three public lifecycles. The previously falling turn now
> completes1,569/1,569 controls. This is not deployment qualification: source
> root-position p95 is0.576/1.832/2.915m, all tracking/standing screens fail, and
>002 has0.008696rad hard joint-range excess. Policy1300 still falls during return
> with this source convention. Untouched all29 SONIC also drifts on these clips.
>47 recorder/buffer tests and9 input-rejection tests pass; all substeps and source
> comparisons are independently checked. No live manifest, commands below,
> physical limits, estimator status or damping diagnosis changes. The experiment
> deliberately differs from the checkpoints' absent-source-axes-zero training
> reference and is not a drop-in qualified live receiver update.

> **2026-09-09, 04:55 Sydney:** Complete public walking contact fitting now passes
> existing sampled geometry checks: zero self-collisions on1,333 samples, unchanged
> limits and timing;175 tests pass. This is not a dynamic controller fix. Fresh
> initialization falls on the repaired reference; policy1300 finishes but has
> 1.421 m source root-position p95 error. A bounded fresh100-update public walking
> regression also fails both CPU motion tests: falls at812/1417 controls on the
> repaired walk and1096/1569 on a separate raw turn, with worse matched-source
> world drift than checkpoint0. Weight updates/export parity pass; motion does
> not. Candidate rejected, not promoted. No commands below, physical limits,
> hardware estimator status, or prior damping diagnosis change.

> **2026-09-09, 04:04 Sydney:** Offline contact fitting now handles an explicitly
> moving source boundary and normalized numerical objective. Full public walking
> reference self-collisions drop480 ->242 ->1 out of1,333 samples, but the last
> overlap and poor conditional support remain. Reference is rejected.169 tests
> pass, including preserving geometry when a later support solver is uncertain.
> No controller tracking improvement, standing acquisition, new live policy,
> physical damping fix or deployment qualification is established. Commands
> below and all physical limits remain unchanged.

> **2026-09-09, 03:11 Sydney:** An absolute-pelvis-tilt bug in offline reference
> fitting is fixed and 202 focused tests pass. All three public clips can now
> be geometrically fitted within existing reference constraints while retaining
> sub-5-mm original foot-path error. The same trained actor still fails all
> tracking screens and falls on two fitted clips. This is not a controller fix,
> causal live retargeter or deployment promotion. No policy, limits or commands
> below change; contact-consistent motion and dynamic tracking remain unresolved.

> **2026-09-09, 02:33 Sydney:** The newer policy1300 was separately tested with
> its correct buffered/root-feedback CPU runtime on all three public motions.
> All tracking screens still fail; the turn also falls when initialized at its
> source pose. Its unchanged zero-update baseline likewise fails. This is not
> fixed by swapping exports or entry alone. The research referee's absolute
> stop is not this live receiver's nominal fallback. No new weights or runtime
> conventions are promoted into the commands below.

> **2026-09-09, 02:08 Sydney:** The further offline
> `--source-action-units-diagnostic` ablation reduces leg-joint error modestly,
> but public walk003 still falls back and foot p95 remains 21-35 cm. This switch
> is not available in the live receiver or physical launcher. Existing weights
> are not relabeled as trained for the changed runtime convention. A separate
> original29 FK audit also finds source torso poses outside the present native23
> tilt envelope; neither deleting joints nor raising the fallback threshold
> establishes full-body parity. Physical teleop remains unqualified.

> **2026-09-09, 01:28 Sydney:** New offline public-motion comparisons confirm
> the original V14 walk policy is not a ready replacement. An explicitly labeled
> source-reference geometry correction reduces arm error, but foot errors and
> turning fallback remain. The geometry ablation is available only through
> `record_g1_true23_saved_teleop_diagnostic --virtual-source-reference-diagnostic`;
> it is not silently enabled in this live receiver or the physical launcher.
> Changing encoder conventions does not qualify or relabel existing weights.

> **2026-09-09, 01:00 Sydney:** The SIM receiver now requires matching encoder
> and decoder reports and preserves source pelvis orientation after one initial
> yaw calibration. Historical unpaired runs require the explicit
> `--legacy-unpaired-diagnostic` flag. New v2 reports retain controller failures
> and failed states instead of exiting without JSON. This is not a hardware
> promotion or full-body fidelity pass: public walk002 completes 656 controls,
> but measured foot/hand errors remain excessive; walk003 still uses fallback.
> The physical launcher and its old pins are unchanged and remain unqualified.
> New v2 reports must not be relabeled v1 to satisfy a historical readiness audit.

> **2026-09-08 current-policy compatibility warning:** This runbook is for the
> older, explicitly hash-pinned q9-anchored causal decoder below, not the newer
> root-feedback/contact-step research policies. The newer runtime uses the
> received-source 200 ms horizon (q0 body/orientation anchor, q1 root setpoint)
> and a separate nine-value root-feedback decoder input. The older PICO producer
> sends q9 body/orientation terms, and the C++ true23 signature gate requires one
> decoder input. Do not swap the newer ONNX into these commands, relabel its
> timing contract, or supply zeros for missing root feedback. A matching
> received-source transport, deployment inference path and validated physical
> world-position/velocity estimator are still required. Older saved-walk or
> transport passes below do not qualify the newer policy. The current learned
> policy still fails complete motion tracking in simulation; no hardware run
> is authorized by this page. See [current progress](../../../PROGRESS.md).

> **2026-09-05 status:** Physical full-body teleop is not ready. Current
> stage-one gains/fraction/slew fail dance simulation; restoration alone is
> insufficient. Fresh PICO health reports zero trackers and no calibration.
> The original-checkout XRT binding path below exists and matches its pin.
> Read [updated progress](../../../PROGRESS.md) before any hardware work.
> Commands below target simulator testing, not a qualified physical robot.

This runbook starts a paired 23-DoF policy in **CPU MuJoCo only**. The
consumer is restricted to localhost, opens no DDS or Unitree channel, and
publishes no robot commands. A timeout, stale packet, 50 Hz gap, malformed
payload, physical velocity jump, or excessive tilt latches the reviewed
zero-velocity balance fallback.

The current simulator example uses paired breadth25 encoder SHA-256
`3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`
and decoder SHA-256
`f4416889023eb629656fa189649d8cd071cdc3ae61fc1bfd888d07815d21bdc8`.
Their reports must identify the same training checkpoint/encoder and match
model bytes and validated ABI. Failure is checked before the subscriber opens.
The legacy residual decoder `44d1fb...` below is historical, not this model.

## Fresh simulator checks: 2026-09-08, 23:55 Sydney

The pinned older candidate completed saved full-body PICO `walk001` (684/684)
and `walk010` (499/499) through real localhost ZMQ in native23 MuJoCo, with
fallback enabled and no nominal fallback. Maximum packet ages were 35.994 ms
and 9.171 ms; minimum base heights 0.723594 m and 0.701787 m; maximum tilts
0.312957 rad and 0.423733 rad. These are transport/stability screens, **not
whole-body tracking-fidelity, standing-acquisition or hardware qualification**.
Initialization still resets the simulator to the first source pose.

Timeout, stale and isolated missing-frame injections each triggered the
correct fallback after 120 controls and held simulated balance for 100 more.
The first missing-frame test, run concurrently with two other consumers,
instead hit an early 113 ms stale-input fault at control 3. Its stable fallback
does not make that test pass; the failed report remains preserved. This
workstation has not passed a loaded real-time execution qualification.

Evidence directory:
`C:/Users/camer/sonic23_sim_artifacts/teleop_readiness_20260908_2345/`.
Fresh health probe found no connected headset client, zero trackers and no
calibration. PC Wi-Fi address at the probe was `192.168.1.182`; recheck if the
network changes. No real-headset session or physical robot test was performed.

## Historical transport qualification (not current deployment readiness)

The authentic saved PICO `walk001` clip was published through the exact live
localhost ABI at 50 Hz. The selected candidate completed all 684 transitions,
frames 10 through 693, with 30 ms maximum reference age, 0.7126 m minimum base
height, 0.3399 rad maximum tilt, and no fallback. The original true23 live run
also completed 684/684, with 24.77 ms maximum age, 0.6706 m minimum height,
0.3648 rad maximum tilt, and no fallback.

Timeout, missing-frame, and stale-frame injections were each applied after
120 live transitions. Every case latched the expected transport trigger and
held the balance fallback for 100 transitions. The fallback stayed above
0.748 m and below 0.160 rad tilt. The machine-readable audit reports
`software_live_teleop_ready: true` and original transport parity true.

## Before putting on the headset

Start XRoboToolKit on PICO, pair and calibrate the body trackers, and use a
clear area. Then run this read-only WSL health probe from the isolated
worktree:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  /usr/bin/python3 -m gear_sonic.scripts.probe_g1_true23_pico_tracking_health \
  --duration-seconds 5 \
  --output artifacts/g1_true23_frozen_lora/live_teleop_v1/pico_health_live.json
```

Do not continue unless the report says `passed: true`. The latest probe on
this workstation verified the service and binding hashes but saw no connected
headset or trackers, so that external gate is currently closed.

## Live PICO to MuJoCo

Use two WSL terminals in the same Ubuntu instance, so producer and consumer
share the monotonic clock used by the freshness check. Start the consumer
first. Do not run a Windows consumer against unconverted WSL timestamps.
Choose new evidence filenames for every attempt; existing files are refused.

**Terminal 1 — WSL, paired diagnostic policy consumer:**

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab \
  /root/.venvs/g1_true23_mjlab/bin/python -m gear_sonic.scripts.run_g1_true23_frozen_lora_live_teleop \
  --repository-root /mnt/z/codex/GR00T-WholeBodyControl \
  --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json \
  --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json \
  --endpoint tcp://127.0.0.1:5603 \
  --steps 1500 \
  --startup-timeout-ms 120000 \
  --receive-timeout-ms 500 \
  --maximum-age-ms 100 \
  --fallback-hold-steps 100 \
  --trace-output /mnt/c/Users/camer/sonic23_sim_artifacts/paired_live_20260909/pico_live_30s.trace.npz \
  --output /mnt/c/Users/camer/sonic23_sim_artifacts/paired_live_20260909/pico_live_30s.json
```

This first live-input check is headless. Add `--viewer` for a subsequent
visualization run; viewer startup and real-input latency still need testing.

**Terminal 2 — WSL, real PICO producer:**

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
: "${PICO_CLIENT_APK_SHA256:?Set this to the verified SHA-256 of the installed PICO client APK}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /root/.venvs/g1_true23_soma/bin/python -m gear_sonic.scripts.stream_g1_23dof_pico_causal_zmq \
  --workspace /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --xrt-module-dir /mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  --soma-source-root /root/.cache/g1_true23_soma/source \
  --capture-python /usr/bin/python3 \
  --bind tcp://127.0.0.1:5603 \
  --packets 1500 \
  --timeout-seconds 120 \
  --subscriber-warmup-s 2 \
  --frame-timeout-s 2 \
  --pico-client-apk-sha256 "$PICO_CLIENT_APK_SHA256" \
  --evidence /mnt/c/Users/camer/sonic23_sim_artifacts/paired_live_20260909/pico_live_30s.producer.jsonl
```

The nominal transport/stability screen succeeds only after all 1500 fresh,
contiguous transitions without fallback. Freshness is rechecked at actual use,
including the two startup packets. If the producer stops, MuJoCo requests the
latched fallback. A failed fallback is reported as failure, not claimed safe.
Optional traces contain initial and post-received-control states; transport-
fault fallback hold is excluded explicitly. Physical errors preserve the last
attempted source index, completed/successful counts and terminal state. No retry
or additional recovery motion follows a controller failure. Passing transport
alone does not qualify tracking, standing acquisition, headset or hardware.

## Physical G1 boundary

**Current physical status: not ready for live teleop.** A dance session that
returns the G1 in damped mode fails the same ownership/shutdown path used by
teleop. No further physical dance or teleop run is a valid readiness test until
the robot-free lifecycle gate below passes and a later bounded hardware handoff
test proves the robot remains in the captured standing mode.

The simulator consumer above remains read-only. Do not connect it to a robot
publisher. Physical control uses the separate promoted C++ controller plus a
fresh hardware shadow and gantry-only active sidecar.

Two managed Windows launchers now own the complete producer/controller
lifecycle:

- `run_g1_true23_frozen_lora_dance_gantry` replays the hash-bound original
  SONIC dance clip. It repeats source values with new contiguous source indices
  and exact 20 ms timestamps, so the READY window does not expire.
- `run_g1_true23_frozen_lora_live_gantry` starts the real PICO/SOMA causal
  producer first. It does not start the robot controller until publisher
  evidence contains a fresh reference packet. Publisher loss starts a
  positive-gain posture return and exact Unitree mode handoff.

Both launchers keep the controller-attached console visible. Saved-clip dance
uses a separately bound exact `DANCE` command and starts automatically only
after `[READY]`; it does not depend on L2/A state. Direct mode is restricted to
the hash-bound frozen-LoRA dance, gantry-only, and at most five seconds.
Reviewed duration, wireless B/R2 or L2 release, and app/process cancellation
stop policy motion without dumping the robot: the controller snapshots the
current 23-joint pose, writes 250 positive-gain zero-feedforward hold packets,
quiesces the writer, closes LowCmd, re-selects the captured Unitree service,
calls `SwitchToInternalCtrl(WALKRUN)`, and then requires 100 consecutive
non-zero-torque standing-mode samples over ten seconds. Stale policy/state/source and inference
faults use the same return. This shutdown contract still requires a new
physical qualification before any live-teleop readiness claim.
Final-boundary validation rejects every post-release command whose controlled
joints lack positive `kp`/`kd`; the controller never publishes a synthesized
`kp=0` damping tail. Physical e-stop remains the independent hard-stop path.

Real PICO live teleop retains the wireless deadman contract. After `[READY]`,
hold L2 and press A once. `[REMOTE]` lines show every decoded L2, A, and STOP
transition for that live path.

Controller startup is hold-first. ONNX and the ten-frame real-proprio window
warm while Unitree motion mode still owns posture, with zero LowCmd writes. The
LowCmd publisher is initialized without writing, the current 23-joint posture
is sampled, and only then is motion mode released. The first post-release
packet has positive position gains and zero feedforward torque. At least 25
successful posture-hold packets must be written before direct or wireless
arming becomes possible. `kp=0` damping is fault-only; successful dance or
teleop completion must prove positive-gain return hold and motion-mode restore.
Execution evidence rejects any pre-release write, any startup damping packet,
or a first hold packet delayed by more than 20 ms after release.

The 2026-09-01 physical records
`live_dance.direct.execution.b0180a162573.jsonl` and
`live_dance.direct.verify.execution.b0180a162573.jsonl` are invalidated for
readiness. Their controller released motion mode and emitted `kp=0` damping for
about 22.75 seconds while policy history warmed, matching the observed dumped
mode. Their later policy-packet counts do not prove a safe startup. Do not use
either record as physical qualification.

Later physical dance attempts that ended with the robot damped are also failed
evidence. They exposed two remaining lifecycle bugs: a simultaneous
watchdog/end-of-clip transition was misread as failed return and latched
`OperatorStop`, and the writer exception handler intentionally emitted 250
damping packets. Both paths are now removed, but this code change alone does
not make live teleop physically ready.

The bounded one-second `lifecyclefix_1s_20260902_030328` run used the rebuilt
no-damping binary. It wrote 467 policy packets, then 305 positive-gain return
packets, with zero damping packets and zero rejected non-positive-gain packets.
It still failed qualification because `SelectMode("ai")` ran while the LowCmd
writer remained active, and Unitree rejected the ownership-transfer RPC. The
follow-up interlock now requests handoff, waits for the writer to quiesce,
closes the LowCmd publisher, retries `SelectMode`, and requires exact
`ai / 801 / 0` before success.

The later bounded run
`live_dance.rerun_1s_20260902_041120.execution.jsonl` invalidated that check.
The robot physically entered damp after exit even though the motion service
reported `ai / 801 / 0`; a later read-only probe still returned those same
values while the robot was visibly damped. `SelectMode("ai")` therefore proves
service selection, not transfer from external/user control back to Unitree's
internal controller. Current Unitree SDK adds a separate
`SwitchToInternalCtrl` API for this handback.

The next one-second physical run
`live_dance.handofffix_1s_retry1_20260902_052042.execution.jsonl` proved that
the `LAST` selector is unsafe on this firmware: after the RPC, the service was
`ai` but its FSM settled at 0 (zero torque). The controller now requests
explicit `WALKRUN`, rejects pre-release state unless it is standing FSM 500 or
801, and accepts post-handoff standing only at captured FSM 801 or Unitree
stand FSM 500. Evidence records the actually observed post-handoff FSM instead
of copying the captured value. This correction has robot-free coverage only;
do not use that failed run as physical qualification.

Run the no-robot lifecycle qualification from WSL after rebuilding
`true23_active_gantry_core_harness` and `g1_true23_active_gantry`:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
python3 -m gear_sonic.scripts.qualify_g1_true23_active_lifecycle_no_robot \
  --repository-root . \
  --output /tmp/g1_true23_active_lifecycle_no_robot.json
```

This command launches neither controller nor publisher. It opens no DDS or
LowCmd channel. Pass requires nine lifecycle scenarios, 4,000 positive-gain
recovery frames, injected 101/290/500 ms stalls, zero published damping frames,
explicit WALKRUN handback with standing FSM 500/801, and a compiled/source surface audit that forbids the
old damping-tail code.

Future physical readiness order is strict:

1. Robot-free lifecycle report passes.
2. Read-only PICO health and hardware shadow pass; no LowCmd.
3. Separate bounded ownership-handoff smoke proves captured standing mode
   returns after positive-gain hold, `SwitchToInternalCtrl(WALKRUN)`, and a
   ten-second no-write stability window.
4. One-second then five-second gantry dance pass with zero damping packets and
   exact mode/FSM restore in execution evidence.
5. Only then run one-second live PICO teleop with deadman, followed by longer
   sessions.

The original `GR00T-WholeBodyControl` true23 implementation has the same
startup defect: it releases motion mode and starts the writer before policy
readiness, and its unarmed `BuildCommand` path returns `BuildDampingCommand`.
This separate sonic-transfer repository intentionally diverges there. It adds
policy-before-release prewarm, positive-gain sampled hold, delayed operator
arming, hold-first evidence gates, and distinct direct-versus-wireless frozen
SONIC active sidecars. Policy/action mapping remains the same true 23-DoF
mapping and safe-target transform.

Example saved-dance launch after creating a fresh sidecar:

```powershell
python -m gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry `
  --repository-root Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof `
  --encoder Z:\codex\GR00T-WholeBodyControl\artifacts\g1_true23\causal_model_250_20260803\causal_model_250.encoder.onnx `
  --decoder-report artifacts\g1_true23_frozen_lora\original_sonic_happy_residual_v1\candidate.plus_0p002.decoder.json `
  --promotion artifacts\g1_true23_frozen_lora\physical_dance_v1\candidate.plus_0p002.dance_shadow_promotion.v2.json `
  --active-promotion <fresh-active-sidecar.json> `
  --live-shadow-evidence <fresh-shadow.jsonl> `
  --packet-bundle artifacts\g1_true23_frozen_lora\physical_dance_v1\original_sonic_happy.true23.causal_packets.json `
  --authorization-id <matching-id> `
  --evidence <new-controller-evidence.jsonl> `
  --publisher-evidence <new-publisher-evidence.json> `
  --duration-seconds 5 `
  --repeat-count 100 `
  --gantry-authorize I_CONFIRM_G1_TRUE23_STAGE1_GANTRY `
  --direct-dance-command DANCE
```

For live PICO, use `run_g1_true23_frozen_lora_live_gantry` with the same
artifact arguments, but create the active sidecar with
`authorize_g1_true23_frozen_lora_live_gantry`. The live sidecar is distinct from
the direct-dance sidecar: it binds wireless L2/A and B/R2, a 1--10 second
reviewed window, and no direct `DANCE` command. Add `--xrt-module-dir` and the
installed hardened PICO APK SHA-256. Its pinned publisher Python defaults to the SOMA venv containing
`pyzmq`; its raw capture worker remains `/usr/bin/python3` for the XRT binding
ABI. A live-health probe on 2026-09-01 still reported no connected headset,
trackers, calibration, or body stream. Saved-clip physical shadow passed, but a
hold-first armed dance and live-headset physical teleoperation remain unproven
until new terminal execution evidence passes.
