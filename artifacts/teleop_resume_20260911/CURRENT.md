# Current native23 simulation work

**2026-09-13: corrected controller-memory A/B pilot finished; continuation rejected.**
Both seeds tie at1/192 fixed movement tracking passes. Restored memory improves
physical completions from281 to284 and282 to287, but quiet-standing passes remain
zero. Training has stopped; no controller is promoted. Three isolated current
replays last13.592s,58.340s and19.244s; the long run loses input after0.442s and
34 packets. Trained seed0 stops at12.170s with zero deadline misses. All tracking
and continuous-standing verdicts fail. The complete58.34-second comparison video
shows all retained runs, the candidate, input faults and physical stops.
State snapshots, exact native restoration, shared
preview and adapters are implemented. The invalid initial pilot is preserved
and excluded from this decision.
See `../onboard_inspection_20260912/CONTROLLER_MEMORY_EXPERIMENT.md` for current
launcher, progress and remaining stages. Entries below describe earlier work.

**Walk002 live-loop milestone:** the native preview controller completes all
58.34 seconds and a 30-second quiet hold on independent clocks, consuming every
packet with zero control deadlines or observation publications missed. One
physics finish is 2.9163 ms late. Full-body tracking still fails substantially.
The same-controller suite finished: walk003 stopped at14.912s, Pico at63.660s,
and walk008 at10.972s, all on native joint bounds. This is not simulation readiness. See
`../onboard_inspection_20260912/NATIVE_PREVIEW_CONTROL.md`.

**Native target actuation implemented:** restores the original expert PD law
and full legal target range for a new controller mode; original physical and
tracking limits remain. Training/runtime history and target checks passed.
Independent500/50Hz support is now connected and its6-second physical plumbing
replay completed with exact native PD. Under concurrent training, two physics
finishes were late; no timing qualification is claimed.
The bounded native MuJoCo3.2.3 dynamics pilot finished at512 worlds and200
total updates. Both fixed checkpoints complete zero recordings. Final physical
durations: walk00212.806s, walk00314.524s, Pico41.312s, walk0088.386s; all
tracking verdicts fail. No active training job, model promotion or automatic
extension. See `../onboard_inspection_20260912/NATIVE_TARGET_ACTUATION.md`.

**Physical motion-prior pilot finished:**200 updates/1694.50s, both scheduled
native tests complete. Checkpoint100 retains full160.6s Pico plus30s quiet, but
fails the walks and full-body tracking. Checkpoint200: walk00212.982s fall,
walk00316.034s fall, walk00810.090s joint bound; Pico160.6s physical completion
but quiet hold fails. All tracking verdicts fail. No model promotion, extension
or active pilot job. The full checkpoint100 Pico video is rendered and checked.
The new training code uses9,466 actual physical expert transitions; no kinematic
targets or walk008 demonstrations. Existing full-body rewards and causal actor
inputs remain. The original pinned demonstration is unchanged.
See `../onboard_inspection_20260912/MOTION_PRIOR_PILOT.md`.

**Fixed gain comparisons finished:** existing native benchmark gains and
measured-root velocity damping were each tested on all four recordings with
both retained Pico and all23 checkpoint200. Neither produced a reusable
controller. Velocity damping extends all23 Pico to77.558s but still falls and
loses the retained Pico quiet hold. Defaults remain unchanged. A minor training
quiet-label bug is fixed; it affects only1–3 acquisition controls per recording
and does not explain the failures. See
`../onboard_inspection_20260912/GAIN_COMPARISON_RESULT.md`.

**All23-joint pilot finished:** both fixed checkpoints fail all four recordings.
The run stopped at200 updates/855.76s. Learning waist and arm
corrections improved some partial tracking but did not preserve complete-motion
balance; Pico fell at51.822s and51.884s. No model promotion or extension.
The retained Pico demo remains unchanged. See
`../onboard_inspection_20260912/FULL_BODY_COMMAND_PILOT.md`.

**Command-space pilot finished:** 200 updates, 934.44 seconds, no extension.
Checkpoint 100 completed zero recordings; checkpoint 200 completed only Pico
with its 30-second quiet hold. All full-body tracking verdicts still fail.
A fixed checkpoint-100 test retaining exploratory commands also failed all four.
The retained Pico demo remains unchanged. See
`../onboard_inspection_20260912/COMMAND_SPACE_PILOT.md`.

**Two additional controller variants stopped:** the firmware SMPL-foot model
and boundary-matched damping each failed all four recordings. The two named
SMPL-foot files are byte-identical and were tested once. No model or default
was promoted. See `../onboard_inspection_20260912/FACTORY_VARIANTS_RESULT.md`.

**Latest longer-horizon planner stopped:** 400 ms native contact predictions
with 32 candidates and two search generations failed walk002 at 8.498 s.
Retaining the complete learned controller inside those predictions also failed,
at 7.594 s. Its control p95 was 198.08 ms; neither version qualifies timing or
complete-motion behavior. No runtime default or pinned model was changed.
See `../onboard_inspection_20260912/FEEDBACK_PLANNER_RESULT.md`.

**Previous controller experiment rejected:** a native response servo differentiated
all23 targets through20ms of actual MuJoCo contact dynamics. Unconstrained
tracking fell on walk002 at7.478s. Preserving predicted factory base motion
still failed all four:00210.782s,0039.820s,Pico17.812s,0089.300s. Computation
also sometimes exceeded20ms. The controller is experimental only; no default
was changed. See `../onboard_inspection_20260912/RESPONSE_SERVO_RESULT.md`.

**Latest continuation finished:** the bounded task-command pilot added explicit
measured foot/body errors, learned factory foot-phase/velocity commands and
separate value learning. Both scheduled checkpoints regressed: zero complete
motions; Pico ended44.142s at update100 and40.488s at200. The run stopped after
200 updates/994.55s. No model promotion or automatic extension. The original
Pico demo remains pinned. Training/runtime input replay agrees within1.36e-5
on identical states. See `../onboard_inspection_20260912/TASK_COMMAND_PILOT.md`.

Latest implementation update, September 13 (Sydney): **full-body teleop remains
unqualified.** Orin SSH worked earlier without changing robot services or the
other laptop's connection. The latest check still reports both physical Windows
Ethernet adapters Disconnected; WSL exposes only the Wi-Fi interface. SSH cannot
currently reach the robot through the Ethernet link. See the timestamp in
`../onboard_inspection_20260912/latest_connection.json`.
Models came from official public firmware; PC1's
filesystem and installed policy version remain unverified.

**New decisive milestone:** the received-only native-trained controller with
leg-target filter alpha0.9 completes the full160.6s Pico lifecycle, including
30 continuous seconds quiet standing, with independent500Hz physics/50Hz
control and zero missed deadlines. No future references or per-motion plans.
Median control1.023ms; maximum2.688ms. Full-body tracking still fails: feet
p95.280/.350m, leg RMSE.312rad, root.247m, hands.167/.192m, head.137m.
The model is pinned as an experimental simulation demo, not promoted to
general teleop. Run `../onboard_inspection_20260912/RUN_PICO_NATIVE23_DEMO.ps1`
or add `-PicoDemo` to the existing received simulation launcher. Add `-Render`
to make the full-speed comparison video. Both+0.03 and-0.03m/s initial-velocity
runs also complete160.6s,30s quiet hold and every deadline. At40s input loss,
the original controller stayed upright but kept stepping. A fault-only standing
capture now stops the gait inside the quiet region while preserving neural
balance feedback and explicit rearm. Its zero-spin run settles but has one late
physics tick. A100us spin trial passes input-loss behavior and timing, but its
nominal repeat has three late ticks, so zero spin remains default. Timing is
not yet repeatable across all scenarios. Same checkpoint/filter fails walk002
at11.038s, walk00312.586s and held-out walk00810.124s under independent timing.
See `../onboard_inspection_20260912/PICO_DEMO_RESULT.md` and the pinned
`onboard_factory_firmware_v1/received_pico_demo_v1/result.json`.
The complete independent-clock Pico video is finished and visually checked:
[watch160.6s including30s standing](<E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/received_sim_runs/20260913_044020_384_pico/video_fast/pico_factory_received_independent_clock.mp4>).
It shows reference and actual poses side by side, original speed and the failed
tracking verdict. All1607 presentation timestamps match the saved physical
trajectory. Rendering completed about four minutes after the six-hour decision
window; no training recipe was extended.

- Preserved native23 factory dance plus 30s standing: 42.36s total, native limits,
  including +/-0.03m/s initial velocity. Exact Torch/ONNX conversion works.
- Recovered native twelve-leg factory locomotion network and its real history,
  gait and velocity inputs. Portable ONNX agrees with MNN within 1.91e-6.
  Received root/upper-body baseline physically completes full Pico (160.6s) and
  walk002 (58.34s), including the added hold. Foot/leg tracking and quiet criteria
  fail; walk003 and held-out walk008 still hit physical limits. This is not a
  claim that velocity following meets full-body teleoperation requirements.
- New dynamics candidate: frozen factory locomotion feedback plus full-range
  twelve-leg corrections; upper eleven targets follow received q/dq directly.
  All original feet, legs, hands, head and root objectives remain. Only current
  and past body packets are inputs. Corrections remain available for braking
  and standing. GPU zero-head baseline physically holds 64 worlds for 33s.
- `onboard_factory_firmware_v1/locomotion_conditioned_ppo_v2` finished both
  checkpoints and stopped after 200 updates. Checkpoint 100 completes all
  160.6s of unpaced Pico and passes the final quiet hold, but tracking still
  fails. It also passes 33s independent-clock standing with zero missed
  deadlines. Checkpoint 200 regresses; neither model is promoted. V1 was
  preflight only, with zero learning updates.
- Completed expert-initialized dynamics pilot with measured command latency:
  previous target remains applied for 2–4ms while physics advances, then the new
  target applies. Earlier dynamics training used immediate target application.
  Upper-body targets remain received q/dq, and quiet neutral balance has a
  smooth recovery gate. All original full-body criteria remain unchanged.
  `locomotion_expert_latency_ppo_v1` stopped at update 66 after checkpoint 50
  still failed acquisition on all four recordings (27.502s combined).
- Completed `onboard_factory_firmware_v1/locomotion_canonical_ppo_v2` reused that
  evaluated checkpoint and adds 25% complete episodes from the actual start;
  50% expert motion, 12.5% transitions/input loss and 12.5% quiet states remain.
  Command latency remains 2–4ms. V1 stopped after 58 updates to fix stale expert
  targets on overwritten actual-start states; those states now contribute
  dynamics rollouts only, with no unrelated imitation label. V2 resumes its
  learned weights for the remaining 142 updates. Full native-rollout decisions
  remain at cumulative updates 100 and 200 (V2 updates 42 and 142), within the
  original 200-update/60-minute cap. Ordinary
  cumulative KL-budget exhaustion no longer permanently halves the learning
  rate. Both decisions failed all four recordings: 27.146s combined at
  cumulative update100 and28.556s at200. The recipe is stopped.
- Completed the400-update lifecycle pilot. It uses75% complete actual-start
  episodes,12.5% motion interruption/transitions and12.5% quiet states, with
  received command delay2–4ms and no imitation fitting. V1 stopped at127 to fix
  uneven clip allocation and allow physically upright training episodes to
  continue through tracking errors. V2 ran the remaining273 updates; all dense
  body penalties remain, with positive survival reward and physical termination.
  Full native decisions at cumulative100 and400 both fail all four recordings.
  Final times: walk00210.302s, walk00313.296s, Pico49.574s, walk00812.992s.
  The recipe is stopped; no automatic extension or promotion.
- Implemented an exact MuJoCo3.2.3 training backend using installed `mjbatch`.
  It preserves native double-precision integration and only writes states on
  explicit episode reset. Ten-step integration matches ordinary MuJoCo exactly;
  resetting another world leaves existing state bitwise unchanged. CPU training
  runs alongside the GPU pilot without consuming its GPU memory.
  `native_mjbatch_lifecycle_ppo_v1` finished200 updates and stopped. Both100/200
  checkpoints complete160.6s Pico and the entire30s quiet hold; neither completes
  the three walking recordings or passes full-body tracking. Checkpoint200
  worsens Pico tracking relative to100. Neither replaces the retained candidate.
- The hold check now covers every sample of the final30 seconds, using the
  existing quiet limits. The retained full Pico rollout and33s independent
  standing test both pass this stronger check. A moving early hold followed by
  only3 quiet seconds correctly fails. Evidence: `continuous_hold30_check.json`.
- Earlier `native23_conditioned_ppo_v1` finished both checkpoints: learned wrist
  commands destabilized it around 0.49s. That recipe is stopped, not extended.
- Independent 500Hz plant uses actual applied-command history, native gains and
  a futex control wakeup. Cached received derivatives match the previous owned
  window exactly. A separate 402ms replay-producer stall led to replacing its
  timed Python wait with absolute CLOCK_MONOTONIC sleep. With clock v3, Pico
  reaches 40.116s with zero missed control/physics deadlines and no packet fault,
  then hits a joint bound. Median controller time is 1.156ms. Full-duration
  motion/timing qualification remains open.
- Eight-candidate, 60ms native dynamics lookahead was implemented and checked
  against ordinary MuJoCo integration (identical held-target states). It falls
  during acquisition at 6.528s and is rejected as a controller candidate; it is
  not used by the default launcher or the retained factory baseline.
- A400ms feedback predictor retains the exact native factory balance network
  throughout all eight candidate rollouts. Its network agrees with ONNX within
  3.58e-6 and its closed-loop state agrees with ordinary Python/MuJoCo within
  1.99e-6. Full walk002 still fails at7.562s; median computation37.9ms under
  concurrent training. Rejected as a controller, retained only experimentally.
  Received gait-phase alignment also regressed the factory baseline and stays
  disabled. No alternative was selected merely for passing a short prefix.
- Native23 AMP factory adapter now uses the May21 21-action networks, whose
  configuration explicitly names the23-joint robot and excludes only wrist
  rolls. It observes actual waist/arm state and reuses the existing body packet
  receiver. Full walk00361.38s plus30s quiet hold completes physically, but
  source root error2.55m and heading116.6deg fail tracking. Walk002 fails at
  18.226s and Pico during acquisition at6.670s. This optional experiment is
  not a working full-body controller and does not replace the default adapter.
- Independent-clock native-trained checkpoint100 reaches61.412s on Pico with
  zero control or physics deadline misses and no packet fault, then an ankle
  exceeds its speed limit. Median inference1.124ms, maximum2.900ms. Run:
  `received_sim_runs/20260913_043348_615_pico` under the firmware artifact root.
- Full walk002 video is complete and visually checked, with actual and reference
  side by side at original speed. It explicitly shows failed tracking:
  `onboard_factory_firmware_v1/human_loco_received_v3/walk002/video/walk002_factory_received_unpaced.mp4`.
- Full 160.6s Pico video from checkpoint 100 is also complete and visually
  checked, including the continuous 30s hold. It is unpaced and explicitly
  shows failed tracking:
  `onboard_factory_firmware_v1/locomotion_conditioned_ppo_v2/eval_00100/pico/video/pico_factory_received_unpaced.mp4`.

Run `../onboard_inspection_20260912/RUN_RECEIVED_NATIVE23_SIM.ps1` for the current
experimental received-motion baseline. Add `-Actor <1542-input ONNX>` for the
new conditioned actor, `-IndependentClock` for independent timing, `-Standing`
for a 33s standing diagnostic, `-InitialVx 0.03` or `-InputLossControl 500` for
declared disturbances, and `-Render` for full-duration video. Nothing publishes
motor commands. No model is automatically promoted from training.
The shortcut `-Learned` selects the retained learned checkpoint and defaults to
Pico; `-Learned -Render` produces the full unpaced160.6s comparison video.

The recovered 29-joint pose policy and its two completed adaptation pilots
failed native23 full motions. They are not active candidates. Prepared factory
demo remains runnable through `../onboard_inspection_20260912/RUN_FACTORY_NATIVE23_SIM.ps1`.
No recurring automation, paid compute, or hardware motion publisher is active.

Historical results below are retained; their earlier "no training running"
statements describe earlier runs, not the latest pilot.

Factory-controller update, September 12: PC1 SDK services are reachable; PC1
SSH is closed. Public Unitree firmware yielded actual ai_sport weights and
configs without changing robot services or the other laptop's connection.
A recovered native23 factory dance policy now completes 9.35 seconds of motion
plus 30 seconds continuous standing, with a generic joint-limit brake. Nominal
and +/-0.03 m/s initial-velocity runs stay within native joint/speed/effort
limits. Median inference is 0.105 ms. This is a phase-driven factory baseline;
arbitrary full-body teleop and independent deadlines are not proven for it.
See ../onboard_inspection_20260912/RESULT.md and
../onboard_inspection_20260912/RUN_FACTORY_NATIVE23_SIM.ps1.
Next candidate is the recovered pose-conditioned dance/fight interface, not
another unchanged supervised fit. No new training or automation is running.

Previous learned-controller result follows:

FINAL RESULT, September12: see MOTION_RESULT.md and RUN_CURRENT_CAUSAL_SIM.ps1.
All training, native evaluations and video rendering are finished. Schedule
remains cancelled. No hardware or new training job is active.

The retained learned23-joint actor now completes the entire58.34s walk002
run and continuous30s standing with independent500Hz physics/50Hz control:
zero controller deadline misses, zero missed observations, zero physics ticks
finishing more than2ms late. Both quiet windows pass; all source packets are
consumed. This proves real-time execution/standing for that nominal attempt.
Tracking still fails: real-time rootp95 .992m. Other recordings do not qualify:
walk003 hits a joint bound6.264s, Pico22.724s; walk008 completes52.28s and
stands but follows the wrong path. General full-body teleop is not ready.

Final video: causal_dynamics_v1/motion_curriculum_final_clock_v1/video/walk002_learned_independent_clock.mp4.
Final reports: motion_curriculum_final_clock_v1/report.json and
motion_curriculum_final_other_clips_v1/report.json under the same artifact root.
Best actor remains motion_curriculum_run_v1/training/actor_00226.onnx.
Corrected training stopped under the two-regression rule. No further extension.

Update10:52UTC: both corrected scheduled checkpoints regressed; training is
finished. V2 stopped at206additional updates under the agreed rule. Its
local200 source rootp95 is .6351m and both quiet windows miss position limits;
local206 also regresses. No further recipe or training extension is running.
Best remains v1/actor_00226. Its full58.34s video is complete under
motion_curriculum_run_v1/eval_actor_00226/walk002/video/.
Final checks now use that same retained controller: the remaining three
recordings and one complete independent-clock walk002/30s-hold attempt.

Update10:34UTC: corrected v2 local100/global326 completes the physical
walk/hold but regresses: source rootp95 .9163m, yaw29.48degrees,
legs .1654rad, feet .250/.259m. It settles almost motionless but ends
5.2cm/5.53degrees off the reference, outside the unchanged standing limits.
First consecutive regression. V1/actor_00226 remains best; v2 local200 is
the next decision. No additional sampler or controller change is planned.

Update10:15UTC: corrected transition training is active in
motion_curriculum_run_v2, resuming v1/actor_00226 with its critic and a fresh
optimizer. Original deadline is preserved:2026-09-12T12:29:10Z. No schedule.

The first sampler implementation cut acquisition at source_start and sampled
normal braking only after source_stop. It therefore missed those handoffs.
Both are corrected: acquisition continues100controls into source; normal
braking starts in the final100source controls and continues100controls beyond.
Physical regression checks cross both boundaries with no resets; fixed96/16/16
environment roles and >=75% source occupancy remain intact. This is a repair
of the accepted transition coverage, not an extension of the training budget.

V1 stopped at226updates for that repair. Its final native evaluation recovered
and is now the inherited best: full58.34s and both quiet windows pass;
rootp95 .5258m, yaw16.59degrees, legs .1635rad, hands .203/.276m,
feet .235/.248m, head .048m. Movement tracking still does not pass.
V2 automatically evaluates every100additional updates, with the same two
consecutive regression stop rule and unchanged physical/tracking thresholds.
Its running.json and latest_evaluation.json report current state.

Motion-curriculum checkpoint200 regressed: full physical duration and final
standing complete, but rootp95 grows to1.742m and main braking/quiet fails.
LegRMSE .1929rad, yawp95 15.21degrees. Checkpoint100 remains best. This is
the first consecutive complete-motion regression; checkpoint300 is the next
decision. Another regression stops this recipe under the accepted rule.

Motion-curriculum checkpoint100: full58.34s walk002 lifecycle and30s hold
complete; both quiet windows pass. LegRMSE improves .2373→.1836rad,
yawp95 40.55→20.95degrees, hands .284/.372→.244/.313m. Rootp95 changes
.6166→.6062m and remains the largest normalized tracking error. This is a
physical improvement, not a readiness pass. Segment curriculum remains2s
(0/64 complete tracking passes). Training continues under the existing cap;
checkpoint200 is the next automatic full-motion evaluation.
Report: motion_curriculum_run_v1/eval_actor_00100/report.json under the
causal_dynamics_v1 artifact root. Full58.34s comparison video is complete at
motion_curriculum_run_v1/eval_actor_00100/walk002/video/walk002_learned_evaluation.mp4.

Latest update 09:29 UTC, September12: accepted motion-curriculum plan is running.
The user approved implementation and switched this task to Default mode.
Schedule remains deleted. No robot or paid compute is involved.

Task-closure standing repair is complete. One unchanged learned actor now
completes walk002 (58.34s) and walk003 (61.38s), including continuous30s
standing holds and both quiet windows. Movement tracking remains inadequate:
walk002 rootp95 .617m, legRMSE .237rad; walk003 rootp95 1.807m.
Pico and held-out walk008 still hit native joint bounds. Teleop is not ready.
Report: E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/task_closure_v2_full_eval/report.json.

New task-closure training bank matches received-only runtime across10674
frames from all four recordings. Fixed environment quotas are75% source,
12.5% transition/recovery,12.5% actual quiet states. Source episodes truncate
at2,4,8seconds, then full source;90% of64 fixed physical segment evaluations
is required to advance. Time-limit value bootstrap uses the actual final
observation, with reset boundaries kept separate. Exploration .06rad on source
states produced no physical failures in the paired64-start check, as at .03rad.

Two-update integrated smoke completed at128 environments: zero physical
failures, no rejected actor updates,76.2% source occupancy, approximately
10seconds/update. Both short tracking evaluations remain0/64; this is a
starting measurement, not a pass. Main training starts again from the tested
initial actor, with a fresh critic and a three-hour wall cap.

Active campaign: E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/motion_curriculum_run_v1.
Its launcher automatically evaluates full native walk002 plus30s standing
every100updates, and stops after two consecutive complete-motion regressions
or a complete tracking/hold pass. No unchanged extension after the cap.
Runnable launcher: RUN_MOTION_CURRICULUM_SIM.ps1 in this directory.
Current running state, latest native evaluation and final outcome live in
running.json, latest_evaluation.json and outcome.json under the campaign.

Historical entries below are superseded by this update.

Latest update 08:39 UTC: user directly resumed work with "Okay fix it".
Schedule remains deleted. Live teleop is still unqualified.
The normalized range adapter now learns: focused_walk002_pilot_v2 finished40
updates. Its final controller completes58.34s physically, but source tracking
and both quiet windows fail. Rootp95 .765m, legRMSE .228rad.
Both planned checkpoints were evaluated; neither was selected.
The next concrete defect is a terminal body-goal conflict: a redundant arm IK
branch keeps the received body goal active after the actual hand/head tasks
return to standing. On550 actual successful quiet states, replacing only that
body goal cuts final actor target error .191rad to .0108rad. A received-only
task-closure repair is being implemented and will be physically evaluated.
The stronger settling-reward pilot was stopped after one update when this more
direct cause was isolated; it is not being extended unchanged.
Independent learned standing also passed a5s native clock smoke test; that
does not establish complete-motion timing or teleop readiness.
Full learned-versus-reference comparison video is available under
causal_dynamics_v1/direct_body_global200_eval_v1/walk003/video/.

Update07:40UTC: user requested cancellation of scheduled work. Heartbeat
six-hour-g1-simulation-work was deleted through the app. No automatic follow-up
or training extension remains authorized by that schedule. Current work is a
direct diagnosis requested by the user, not another scheduled training pilot.
The final direct-body pilot and evaluator are finished (former30207/98864).
Global200 fails all four tracking gates. Walk003 completes61.38s physically,
but rootp95 2.159m, legRMSE .367rad and main quiet fails; extra hold quiet passes.
Pico completes160.60s physically and both quiet windows, but tracking fails.
Walk002 hits a native joint bound35.770s; walk008 hits one12.070s.
Final report: direct_body_global200_eval_v1/report.json. No candidate selected.
All historical running/pending statements below are superseded by this update.

Update07:28UTC: direct-body global102 native evaluation finished and is rejected.
All four source tracking gates fail. Walk003 completes61.38s with both quiet
windows, but rootp95 is1.962m and legRMSE .334rad. Walk002 completes physically
but fails both quiet windows. Pico hits a native joint bound at75.766s;
held-out walk008 hits a native joint bound at9.198s. This is no readiness pass.
Report: direct_body_global102_eval_v1/report.json. Evaluator63960 is finished.
Pilot30207 remains active; latest logged global124/200. Final evaluator98864
is still waiting for the scheduled final checkpoint. No extension after200.

## 2026-09-12 05:05 UTC: scheduled learned pilots finished; live gate fails

Update07:11UTC: pilot_direct_body_v1 is the only active ML job (session30207).
It runs196 updates after4 smoke updates, global200 maximum,64 environments
x64 steps,16 minibatches. Scheduled local98/global102 and local196/global200
exports; wall cap.9h including initialization. No extension past this budget.
See DIRECT_BODY_PUSH.md. Smoke and initial evaluations are finished.
Initial30s standing passes0,+.03,-.03m/s. Initial all-four baseline fails all
tracking gates: walk003/Pico complete physically with quiet holds; walk002
joint bound24.592s, held-out walk008 bound12.948s. Smoke4 also passes all three
30s standing cases and full walk003 duration/quiet, but tracking fails.
Main initial ONNX is byte-identical to tested smoke4; CONTINUE released.
Future access remains0. No hardware or general teleop qualification.
Both native all-four evaluations are already queued: global102 worker63960
waits for actor_00098.normalization.npz and writes direct_body_global102_eval_v1;
global200 worker98864 waits for actor_00196.normalization.npz and writes
direct_body_global200_eval_v1. Do not duplicate these workers. First3 main
iterations measure678..704 control transitions/s,5.62..6.04s per4096 steps.

Completed earlier family: pilot_single_policy_v1 and both scheduled evaluations are
finished. Global400 also fails all
four source tracking gates. All four durations complete physically, but Pico
now fails both quiet windows too. Recipe rejected and stopped at global400;
no unchanged extension. Final report: single_policy_global400_eval_v1/report.json.
Rootp95 walk0031.450m,walk002.407m,Pico.188m,walk0081.854m;
legRMSE .374,.250,.330,.307rad; heading82.3,88.6,19.4,96.0deg.
Hand errors reach.817m. Some measures improved, but large tracking errors and
Pico standing regression rule out selection. Hardware remains unqualified.

Remaining concrete controller issue to address within the original deadline:
SinglePolicy.goal supplies a neutral pose to the pretrained BFM goal encoder;
received full-body poses currently enter only a new learned adapter. That
adapter did not learn adequate motion conditioning in this bounded pilot.
Assess routing current/past received body poses through the existing full-body
encoder directly, preserving one actor and standing initialization. This is
now implemented in the separate direct-body route above. Do not repeat old
frozen-BFM small-residual training. See SINGLE_POLICY_PUSH.md for completed work.

Historical main pilot protocol (now completed): pilot_single_policy_v1
trains the BFM actor itself with received-only full-body conditioning, one
policy throughout; no frozen-balance target handoff. It resumes4 smoke
updates and runs396 more (global400 maximum), checkpoints local198/global202
and local396/global400. Wall cap1.8h from initialization; original08:46:40UTC
decision deadline remains. Never extend this family automatically.

Initial single-policy baseline passes30s native standing at0,+.03,-.03m/s.
It stays upright through all four complete packet replays and30s hold, but
all fail full-body tracking; this is balance, not motion replication.
Smoke4 checkpoint retains all three standing passes and physically completes
walk00361.38s with both quiet windows, but tracking still fails. Main initial
ONNX is byte-identical to that tested smoke4 checkpoint. CONTINUE released.
See SINGLE_POLICY_PUSH.md. The older scheduled families remain finished.

Update06:16UTC: scheduled global202 evaluation completed. All four replay
durations and both quiet windows pass physically, but every source tracking
gate fails. Rootp95 walk0031.495m,walk002.406m,Pico.187m,walk0081.966m;
legRMSE .381,.280,.316,.325rad. Heading errors80.5,103.1,20.5,95.5deg.
Hand errors reach.83m. This checkpoint is rejected; no overall full-body
tracking improvement. Report: single_policy_global202_eval_v1/report.json.
Training reached global400 and stopped. Its final evaluator finished
all four clips plus30s hold in single_policy_global400_eval_v1. Do not
launch a duplicate. Former training owner55455 and evaluator14648 are finished.
Global202 evaluator13469 also finished; both complete reports are rejected.

Original causal pilot completed800 updates;
changed balanced family completed400 total updates (201 original +199 projected).
Final global400 native rollouts fail before source motion on all four clips:
walk0036.000s, walk0026.122s, Pico6.054s, held-out walk0085.876s.
Native joint-bound or speed limits terminate each run. Source samples completed:0.
Do not extend either completed training recipe unchanged.

Useful completed behavior: prepared walk003 executes31.38s plus30s continuous
standing with independent500Hz physics and50Hz control in braking_v5; the
same prepared benchmark also passes at+.03m/s initial X velocity. The-.03m/s
trial falls at10.07s. Later nominal repeats have occasional physics-finish
deadline misses, so repeated timing and disturbance acceptance remain open.
Shared received-only balanced API separately passes30s standing at0,+.03,-.03m/s;
that is an unpaced standing-only result, not full-body motion qualification.

Full video: E:/codex-artifacts/sonic23_teleop_resume_20260911/native_clock_v1/
braking_v5/video/walk003_full_motion_30s_hold.mp4.
The missing behavior is stable standing-to-motion takeover followed by complete
received motion. All23 joints, legs/feet/hands/head tracking, original speed,
30s hold and native physical limits remain mandatory. No real Pico or robot
validation follows until the reusable simulation gate passes.

## Accepted-plan implementation record (superseded job status below)

User explicitly approved the causal dynamics plan; laptop only. New six-hour
decision window starts02:46:40UTC, ends08:46:40UTC. Read CAUSAL_DYNAMICS_PUSH.md
for new code and process protocol. Original pilot_v1 completed all800 updates.
Checkpoint400 fails at .758s;800 improves to1.224s but still fails standing,
all four clips, no source completion. Do not extend that completed recipe.

Changed pilot family: E:/codex-artifacts/sonic23_teleop_resume_20260911/
causal_dynamics_v1. pilot_balanced_v1 stopped at201 after scheduled200 evaluation.
Checkpoint200 still fails before source: walk0036.094s,walk0026.036s,Pico6.154s,
walk0086.018s. No full-motion improvement. Completed continuation is
pilot_balanced_projected_v1:199 remaining updates, global total400, resumes201.
projected_initial_eval_v1 completed all four attempts (failure5.95..6.09s,
before source); CONTINUE released PPO. This continuation has now finished.
Final evaluation actor_00199.onnx with --balanced (global400) also fails all four.
Do not extend this family beyond400 total updates.
It retains frozen native23 neutral balance during standing, blends full-range
motion takeover over .5s using received pose changes, and seeds actual expert
states near six interruption points. No future goals, per-motion gains or
small residual cap. Actor/critic start from pilot_v1/actor_00800.pt, no new
supervised bootstrap; optimizer fresh. Inactive motion actor gets no updates.
Old pilot code snapshot preserved in pilot_v1/source_snapshot before changes.
Balanced pre-correction snapshot is pilot_balanced_v1/source_snapshot.
Correction: explore around projected legal means, with explicitly experimental
straight-through projection gradients. Eight failed-takeover predictions were
over5 noise standard deviations outside limits; raw hip-yaw3.844rad versus
native upper2.758 meant small noise kept producing identical clipped targets.
Training now omits extra30s frozen-balance hold (existing lifecycle standing
remains); all native evaluation still requires30s continuous hold.
Baseline through same balanced runtime fails during takeover: walk0036.098s,
walk0026.026s, Pico5.918s, walk0085.992s. Source motion still not reached.
These implemented controller changes did not pass readiness.
Check active jobs before launching anything; no duplicate training or old fits.

Shared API and balanced implementation reside in gear_sonic/utils/
g1_true23_causal_controller.py, g1_true23_causal_balance.py, and
g1_true23_neutral_balance.py. Training uses g1_true23_balanced_causal_dynamics.py.
RUN_CAUSAL_FULL_BODY_SIM.ps1 selects balance automatically from training request
metadata, or accepts -Balanced explicitly. Receiver tests:5 passed, including
generated full-body stop and explicit rearm calibration;12 moving expert states
with actual histories agree between tensor/native feature implementations.
balanced_standing30_v1 passes30s standing at0 and +/- .03m/s initial X velocity
through this same received-only balanced API. This is standing-only, unpaced,
and does not establish movement or timing readiness.

### Decisive new timing result, 2026-09-12 03:49 UTC

`native_clock_v1/braking_v5` in the same E: artifact root completed all 31.38s
of walk003 lifecycle plus 30s continuous hold with independent native 500Hz
physics and 50Hz workers. Zero controller deadline misses, zero physics ticks
over2ms late, zero missed observations, all native physical and full-body
tracking limits pass. Both last-three-second quiet windows pass; final XY
p95 .03903m. Terminal controller now uses measured-velocity damping and XY
gain4. This remains a prepared motion-specific benchmark, not live teleop.

`braking_plus003_v1` repeats those passes with +.03m/s initial root X velocity.
`braking_minus003_v1` fails by tilt at10.07s with -.03m/s despite zero deadline
misses. Thus disturbance gate fails. Prior `isolated_v3` full motion succeeded
but quiet XY failed (.13994m) and four physics ticks were late. RT scheduling
trial `braking_v4` fell before terminal; retained as failure.

Causal pilot checkpoint400 regresses: all four motions fail during initial
standing at .758s, native joint bound, before source motion. Bootstrap failed
at .796s. Scheduled800 checkpoint subsequently improved failure time to1.224s
but still failed standing on all four clips. Recipe stopped without extension.

Runnable entry points here: RUN_NATIVE_CLOCK_SIM.ps1 (prepared benchmark),
RUN_CAUSAL_FULL_BODY_SIM.ps1 -Actor <onnx> (received-only candidate evaluation).
Full new 61.38s video rendering in braking_v5/video. No hardware authorized.

All following sections describe the preserved earlier checkpoint.

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
